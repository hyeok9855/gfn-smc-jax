import math
import chex
import distrax
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import wandb

from targets.base_target import Target


# Taken from FAB code
class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1.0):
        dim = 2
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1


class ManyWellEnergy(Target):
    def __init__(
        self,
        a: float = -0.5,
        b: float = -6.0,
        c: float = 1.0,
        dim=32,
        can_sample=True,
        sample_bounds=None,
    ) -> None:
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(a, b, c)

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        self._plot_bound = 3.0

        # Define the proposal distribution for the rejection sampling
        # Mixture of two Gaussians: weights [0.2, 0.8], means [-1.7, 1.7], scales [0.5, 0.5]
        self.means = jnp.array([-1.7, 1.7])
        self.scales = jnp.array([0.5, 0.5])
        self.log_weights = jnp.log(jnp.array([0.2, 0.8]))

        self.proposal_dist = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=self.log_weights),
            components_distribution=distrax.Independent(
                distrax.Normal(loc=self.means, scale=self.scales)
            ),
        )

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        x_reshaped = x.reshape((-1, self.n_wells, 2)).reshape((-1, 2))
        double_well_log_probs = self.double_well_energy.log_prob(x_reshaped)
        log_probs = jnp.sum(double_well_log_probs.reshape((-1, self.n_wells)), axis=-1)

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)
        return log_probs

    def visualise(self, samples: chex.Array = None, show=False, prefix="") -> dict:
        """Visualise samples from the model."""
        plotting_bounds = (-3, 3)
        grid_width_n_points = 100
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex="row", sharey="row")
        samples = jnp.clip(samples, plotting_bounds[0], plotting_bounds[1])
        for i in range(2):
            for j in range(2):
                # plot contours
                def _log_prob_marginal_pair(x_2d, i, j):
                    x = jnp.zeros((x_2d.shape[0], self.dim))
                    x = x.at[:, i].set(x_2d[:, 0])
                    x = x.at[:, j].set(x_2d[:, 1])
                    return self.log_prob(x)

                xx, yy = jnp.meshgrid(
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                )
                x_points = jnp.column_stack([xx.ravel(), yy.ravel()])
                log_probs = _log_prob_marginal_pair(x_points, i, j + 2)
                log_probs = jnp.clip(log_probs, -1000, None).reshape(
                    (grid_width_n_points, grid_width_n_points)
                )
                axs[i, j].contour(xx, yy, log_probs, levels=20)

                # plot samples
                axs[i, j].plot(samples[:, i], samples[:, j + 2], "o", alpha=0.5)

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        # Non-jittable rejection sampling, called once at the beginning.
        if len(sample_shape) == 1:
            n_samples = int(sample_shape[0])
        else:
            raise ValueError(f"Unsupported sample_shape: {sample_shape}")

        key = seed
        pairs = []
        for _ in range(self.n_wells):
            key, k1, k2 = jax.random.split(key, 3)
            x1 = self._rejection_sampling_x1(k1, n_samples)
            x2 = jax.random.normal(k2, shape=(n_samples,))
            pairs.append(jnp.stack([x1, x2], axis=1))  # (n, 2)

        return jnp.concatenate(pairs, axis=1)  # (n, dim)

    # ----- Helpers for rejection sampling (x1 dimension) ----- #
    @staticmethod
    def _target_unnormed_logp_x1(x: chex.Array) -> chex.Array:
        # log p(x1) up to a constant: -(x^4) + 6 x^2 + 0.5 x
        return -(x**4) + 6.0 * (x**2) + 0.5 * x

    def _rejection_sampling_x1(self, key: chex.PRNGKey, n_samples: int) -> chex.Array:
        # Rejection sampling with envelope k
        Z_x1 = 11784.50927
        k = Z_x1 * 3.0

        accepted = []
        remaining = n_samples
        while remaining > 0:
            batch = remaining * 10
            key, k_prop, k_u = jax.random.split(key, 3)
            z0 = self.proposal_dist.sample(seed=k_prop, sample_shape=(batch,))
            prop_lp = self.proposal_dist.log_prob(z0)
            targ_ul = self._target_unnormed_logp_x1(z0)

            # Accept with probability exp(targ_ul - prop_lp) / k
            accept_prob = jnp.exp(targ_ul - prop_lp) / k
            u = jax.random.uniform(k_u, shape=(batch,))
            mask = u < accept_prob
            acc = z0[mask]
            if acc.size > 0:
                accepted.append(acc)
                remaining -= int(acc.size)

        all_acc = jnp.concatenate(accepted, axis=0)
        return all_acc[:n_samples]


class ManyWell2(Target):
    def __init__(
        self,
        dim: float = 5,
        m: float = 5,
        delta: float = 4,
        can_sample: bool = True,
        sample_bounds=None,
    ):
        self.d = dim
        self.m = m
        self.delta = jnp.array(delta)

        self._plot_bound = 3.0

        super().__init__(dim=dim, log_Z=self.log_Z, can_sample=can_sample)

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]
        assert x.shape[1] == self.d, "Dimension mismatch"
        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :m]
        k = ((prefix**2 - delta) ** 2).sum(axis=1)

        suffix = x[:, m:]
        k2 = 0.5 * (suffix**2).sum(axis=1)

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def log_prob_2D(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :2]
        k = ((prefix**2 - delta) ** 2).sum(axis=1)

        # suffix = x[:, m:]
        # k2 = 0.5 * (suffix**2).sum(axis=1)
        k2 = 0

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix="") -> dict:
        """Visualise samples from the model."""
        plotting_bounds = (-self._plot_bound, self._plot_bound)
        grid_width_n_points = 100
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex="row", sharey="row")
        samples = jnp.clip(samples, plotting_bounds[0], plotting_bounds[1])
        for i in range(2):
            for j in range(2):
                # plot contours
                def _log_prob_marginal_pair(x_2d, i, j):
                    x = jnp.zeros((x_2d.shape[0], self.dim))
                    x = x.at[:, i].set(x_2d[:, 0])
                    x = x.at[:, j].set(x_2d[:, 1])
                    return self.log_prob(x)

                xx, yy = jnp.meshgrid(
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                )
                x_points = jnp.column_stack([xx.ravel(), yy.ravel()])
                log_probs = _log_prob_marginal_pair(x_points, i, j + 2)
                log_probs = jnp.clip(log_probs, -1000, None).reshape(
                    (grid_width_n_points, grid_width_n_points)
                )
                axs[i, j].contour(xx, yy, log_probs, levels=20)

                # plot samples
                axs[i, j].plot(samples[:, i], samples[:, j + 2], "o", alpha=0.5)

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()
        return wb

    @property
    def log_Z(self):
        # numerical integration
        l, r = -100, 100
        s = 100000000
        key = jax.random.PRNGKey(0)

        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        fst = jnp.log(jnp.sum(jnp.exp(-((pt**2 - self.delta) ** 2)) * ((r - l) / s)))

        self.logZ_1d = fst

        # well the below works but there's analytic solution this is Gaussian lmao - junhua
        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        snd = jnp.log(jnp.sum(jnp.exp(-0.5 * pt**2) * ((r - l) / s)))

        return fst * self.m + snd * (self.d - self.m)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:

        REJECTION_SCALE = 6

        def doubleWell1dLogDensity(xs, shift, separation):
            return -(((xs - shift) ** 2 - separation) ** 2) - self.logZ_1d

        def rejection_sampling(seed, shape, proposal, target_pdf, scaling):
            new_key, subkey1, subkey2 = jax.random.split(seed, num=3)
            n_samples = math.prod(shape)
            samples = proposal.sample(
                seed=subkey1, sample_shape=(n_samples * math.ceil(scaling) * 10,)
            )
            unif = jax.random.uniform(subkey2, (samples.shape[0],))
            unif *= scaling * jnp.exp(proposal.log_prob(samples))
            accept = unif < target_pdf(samples).squeeze(1)
            samples = samples[accept]
            if samples.shape[0] >= n_samples:
                return jnp.reshape(samples[:n_samples], shape)
            else:
                new_shape = (n_samples - samples.shape[0],)
                new_samples = rejection_sampling(new_key, new_shape, proposal, target_pdf, scaling)
                return jnp.concat([samples.reshape(*shape, -1), new_samples])

        def GetProposal(shift, separation):
            # proposal distribution for 1D doubleWell rejection sampling
            loc = shift + jnp.sqrt(separation) * jnp.array([[-1.0], [1.0]])
            scale = 1 / jnp.sqrt(separation) * jnp.array([[1.0], [1.0]])
            ps = jnp.array([0.5, 0.5])
            components = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            gmm = distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(probs=ps),
                components_distribution=components,
            )
            return gmm

        def Sample1DDoubleWell(seed, shape, shift, separation):
            proposal = GetProposal(shift, separation)
            target_pdf = lambda xs: jnp.exp(doubleWell1dLogDensity(xs, shift, separation))
            return rejection_sampling(seed, shape, proposal, target_pdf, REJECTION_SCALE)

        new_key, subkey1, subkey2 = jax.random.split(seed, num=3)

        n_dw, n_gauss = self.m, self.d - self.m
        dw_samples = Sample1DDoubleWell(subkey1, sample_shape + (n_dw,), 0, self.delta)

        gauss_samples = jax.random.normal(subkey2, sample_shape + (n_gauss,))

        return jnp.concat([dw_samples, gauss_samples], axis=-1)


if __name__ == "__main__":
    # mw = ManyWellEnergy()
    # mw.visualise(samples=mw.sample(jax.random.PRNGKey(0), (1,)))

    # key = jax.random.PRNGKey(42)
    # well = ManyWellEnergy()

    # samples = jax.random.normal(key, shape=(10, 32))
    # print(samples.shape)
    # print((well.log_prob(samples)))
    # print((jax.vmap(well.log_prob)(samples)))
    from eval import discrepancies

    key = jax.random.PRNGKey(0)
    target = ManyWellEnergy(dim=64)
    sample1 = target.sample(key, (2000,))

    min_sd = jnp.inf
    max_sd = 0.0
    sd_list = []
    mmd_list = []
    n_trial = 5

    sd_self = discrepancies.compute_sd(sample1, sample1, None)
    print(f"Self sd: {sd_self:.4f}")
    mmd_self = discrepancies.compute_mmd(sample1, sample1, None)
    print(f"Self mmd: {mmd_self:.4f}")

    key = jax.random.PRNGKey(0)
    _, keygen = jax.random.split(key)
    for i in range(1, n_trial + 1):
        key2, keygen = jax.random.split(keygen)

        sample2 = target.sample(key2, (2000,))
        sd = discrepancies.compute_sd(sample1, sample2, None)
        sd_list.append(sd)
        mmd = discrepancies.compute_mmd(sample1, sample2, None)
        mmd_list.append(mmd)
        if sd < min_sd:
            min_sd = sd
            best_key2 = key2
        if sd > max_sd:
            max_sd = sd
            worst_key2 = key2
        running_mean_sd = sum(sd_list) / i
        running_mean_mmd = sum(mmd_list) / i
        print(
            f"Iteration {i} - Best sd: {min_sd:.2f}, Worst sd: {max_sd:.2f}, Running mean sd: {running_mean_sd:.2f}, Running mean mmd: {running_mean_mmd:.3f}"
        )

    sd_list = jnp.array(sd_list)
    mmd_list = jnp.array(mmd_list)
    mean_sd = sum(sd_list) / n_trial
    std_sd = jnp.std(sd_list)
    mean_mmd = sum(mmd_list) / n_trial
    std_mmd = jnp.std(mmd_list)
    print(
        f"Final (n_trial = {n_trial}) - Best sd: {min_sd:.2f}, Worst sd: {max_sd:.2f}, Mean sd: {mean_sd:.2f}, Std sd: {std_sd:.2f}, Mean mmd: {mean_mmd:.3f}, Std mmd: {std_mmd:.3f}"
    )
