import gc

import jax
import jax.numpy as jnp
from flax import traverse_util


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key for the current level of the dictionary.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reset_device_memory():
    """Free all live device buffers to release memory."""
    from jax.extend import backend as jax_backend
    backend = jax_backend.get_backend()

    for buf in backend.live_buffers():
        buf.delete()



def stable_mean(x):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, 0)

    # Compute the sum of non-NaN values
    total_sum = jnp.sum(x)

    # Compute the number of non-NaN values
    count = jnp.sum(mask)

    # Compute the mean of non-NaN values
    mean_value = total_sum / count

    return mean_value


def replace_invalid(x, replacement=0.0):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, replacement)

    # Compute the number of non-NaN values
    invalid_count = jnp.sum(mask)

    return x, invalid_count


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(data)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


if __name__ == "__main__":
    init_std = 10
    a = inverse_softplus(10)
    print(jax.nn.softplus(a))
