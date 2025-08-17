import jax
import jax.numpy as jnp
from jax._src.scipy.special import logsumexp

from eval import discrepancies
from eval.utils import (
    avg_stddiv_across_marginals,
    compute_reverse_ess,
    moving_averages,
    save_samples,
)


def get_eval_fn(cfg, target, target_samples):
    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "other/target_log_prob": [],
        "other/delta_mean_marginal_std": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    n_eval_samples = cfg.eval_samples

    def eval_fn(samples, log_ratio, target_log_prob, fwd_log_ratio):
        ln_z = logsumexp(log_ratio) - jnp.log(n_eval_samples)
        elbo = jnp.mean(log_ratio)

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(ln_z - target.log_Z))

        logger["logZ/reverse"].append(ln_z)
        logger["KL/elbo"].append(elbo)
        logger["ESS/reverse"].append(compute_reverse_ess(log_ratio, cfg.eval_samples))
        logger["other/delta_mean_marginal_std"].append(
            jnp.abs(avg_stddiv_across_marginals(samples) - target.marginal_std)
        )
        logger["other/target_log_prob"].append(jnp.mean(target_log_prob))

        if cfg.compute_forward_metrics and (target_samples is not None):
            eubo = jnp.mean(fwd_log_ratio)
            fwd_ln_z = -(jax.scipy.special.logsumexp(-fwd_log_ratio) - jnp.log(cfg.eval_samples))
            fwd_ess = jnp.exp(
                fwd_ln_z - (jax.scipy.special.logsumexp(fwd_log_ratio) - jnp.log(cfg.eval_samples))
            )

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)
            logger["ESS/forward"].append(fwd_ess)

        logger.update(target.visualise(samples=samples, show=cfg.visualize_samples))

        if cfg.compute_emc and cfg.target.has_entropy:
            logger["other/EMC"].append(target.entropy(samples))

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"].append(
                getattr(discrepancies, f"compute_{d}")(target_samples, samples, cfg)
                if target_samples is not None
                else jnp.inf
            )

        if cfg.moving_average.use_ma:
            for key, value in moving_averages(
                logger, window_size=cfg.moving_average.window_size
            ).items():
                if isinstance(value, list):
                    value = value[0]
                if key in logger.keys():
                    logger[key].append(value)
                    logger[f"model_selection/{key}_MAX"].append(max(logger[key]))
                    logger[f"model_selection/{key}_MIN"].append(min(logger[key]))
                else:
                    logger[key] = [value]
                    logger[f"model_selection/{key}_MAX"] = [max(logger[key])]
                    logger[f"model_selection/{key}_MIN"] = [min(logger[key])]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return eval_fn, logger
