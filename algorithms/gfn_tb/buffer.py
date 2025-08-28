from typing import NamedTuple, Protocol, Tuple

import chex
import jax
import jax.numpy as jnp


### Core Data Structures ###


class TerminalStateData(NamedTuple):
    """
    Holds the core arrays for the terminal state buffer.

    Attributes:
        states: The terminal states. Shape is `(max_length, dim)`.
        log_w: The log-weights used for prioritized sampling. Shape is `(max_length,)`.
    """

    states: chex.Array
    log_w: chex.Array


class TerminalStateBufferState(NamedTuple):
    """
    Represents the complete state of the buffer at any point in time.

    Attributes:
        data: An instance of TerminalStateData holding the arrays.
        current_index: The index for the next insertion in the circular buffer.
        is_full: A boolean flag, True if the buffer has been filled at least once.
    """

    data: TerminalStateData
    current_index: jnp.int32  # type: ignore
    is_full: jnp.bool_  # type: ignore


### Protocol Definitions for Buffer API ###


class InitFn(Protocol):
    def __call__(self, states: chex.Array, log_w: chex.Array) -> TerminalStateBufferState:
        """Initialises the buffer state with a starting batch of data."""
        ...


class AddFn(Protocol):
    def __call__(
        self, states: chex.Array, log_w: chex.Array, buffer_state: TerminalStateBufferState
    ) -> TerminalStateBufferState:
        """Adds a new batch of data to the buffer."""
        ...


class SampleFn(Protocol):
    def __call__(
        self, key: chex.PRNGKey, buffer_state: TerminalStateBufferState, batch_size: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Samples a batch from the buffer.

        Returns:
            A tuple of (sampled_states, sampled_log_w, sampled_indices).
        """
        ...


class UpdateWeightsFn(Protocol):
    def __call__(
        self, new_log_w: chex.Array, indices: chex.Array, buffer_state: TerminalStateBufferState
    ) -> TerminalStateBufferState:
        """
        Updates the log-weights for a given set of indices.
        This can be used to update priorities or to discard items by setting log_w to -inf.
        """
        ...


class TerminalBuffer(NamedTuple):
    """
    A container for the buffer API functions.

    Attributes:
        init: Function to initialize the buffer.
        add: Function to add new data.
        sample: Function to sample data.
        update_weights: Function to adjust log-weights.
        max_length: The maximum capacity of the buffer.
    """

    init: InitFn
    add: AddFn
    sample: SampleFn
    update_weights: UpdateWeightsFn
    max_length: int


### Build Buffer ###


def build_terminal_state_buffer(
    dim: int,
    max_length: int,
    sample_with_replacement: bool = False,
) -> TerminalBuffer:
    """
    Creates a prioritized replay buffer for terminal states using a circular buffer.

    Args:
        dim: The dimension of the states to be stored.
        max_length: The maximum capacity of the buffer.
        sample_with_replacement: Whether to sample with or without replacement.
    """
    assert max_length > 0, "max_length must be greater than 0."

    def init(states: chex.Array, log_w: chex.Array) -> TerminalStateBufferState:
        """Initialises the buffer state with a starting batch of data."""
        chex.assert_rank(states, 2)  # (batch_size, dim)
        chex.assert_rank(log_w, 1)  # (batch_size,)
        chex.assert_equal_shape_prefix((states, log_w), 1)

        # Pre-allocate memory for the buffer
        buffer_states = jnp.zeros((max_length, dim), dtype=states.dtype)
        buffer_log_w = -jnp.inf * jnp.ones((max_length,), dtype=log_w.dtype)

        data = TerminalStateData(states=buffer_states, log_w=buffer_log_w)

        # Create an initial empty state
        buffer_state = TerminalStateBufferState(
            data=data,
            current_index=jnp.int32(0),
            is_full=jnp.bool_(False),
        )

        # Add the initial data
        return add(states, log_w, buffer_state)

    def add(
        states: chex.Array, log_w: chex.Array, buffer_state: TerminalStateBufferState
    ) -> TerminalStateBufferState:
        """Adds a new batch of data to the buffer."""
        chex.assert_rank(states, 2)
        chex.assert_rank(log_w, 1)
        batch_size = states.shape[0]  # type: ignore

        # Calculate insertion indices for the circular buffer
        indices = (jnp.arange(batch_size) + buffer_state.current_index) % max_length

        # Update data arrays immutably
        new_states_array = buffer_state.data.states.at[indices].set(states)  # type: ignore
        new_log_w_array = buffer_state.data.log_w.at[indices].set(log_w)  # type: ignore
        data = TerminalStateData(states=new_states_array, log_w=new_log_w_array)

        # Update metadata
        new_index = buffer_state.current_index + batch_size
        is_full = jax.lax.select(
            buffer_state.is_full, buffer_state.is_full, new_index >= max_length
        )
        current_index = new_index % max_length

        return TerminalStateBufferState(
            data=data,
            current_index=current_index,
            is_full=is_full,
        )

    def sample(
        key: chex.PRNGKey, buffer_state: TerminalStateBufferState, batch_size: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Samples a batch from the buffer in proportion to log weights."""
        # Determine the number of valid items currently in the buffer
        buffer_size = jax.lax.select(buffer_state.is_full, max_length, buffer_state.current_index)

        # Get log-weights of valid items
        valid_log_w = buffer_state.data.log_w[:buffer_size]  # type: ignore

        # Sample indices based on the calculated logits
        indices = jax.random.choice(
            key,
            buffer_size,
            shape=(batch_size,),
            p=jax.nn.softmax(valid_log_w),
            replace=sample_with_replacement,
        )

        # Gather the data using the sampled indices
        sampled_states = buffer_state.data.states[indices]  # type: ignore
        sampled_log_w = buffer_state.data.log_w[indices]  # type: ignore

        return sampled_states, sampled_log_w, indices

    def update_weights(
        new_log_w: chex.Array, indices: chex.Array, buffer_state: TerminalStateBufferState
    ) -> TerminalStateBufferState:
        """Updates the log-weights for a given set of indices."""
        chex.assert_equal_shape((new_log_w, indices))

        # Update the log_w array immutably and stop gradient flow
        updated_log_w = buffer_state.data.log_w.at[indices].set(  # type: ignore
            jax.lax.stop_gradient(new_log_w)
        )

        # Create new data and state objects
        data = buffer_state.data._replace(log_w=updated_log_w)
        return buffer_state._replace(data=data)

    return TerminalBuffer(
        init=init,
        add=add,
        sample=sample,
        update_weights=update_weights,
        max_length=max_length,
    )
