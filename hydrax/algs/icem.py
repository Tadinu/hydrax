from typing import Any, Literal, Tuple, Optional
import functools
import colorednoise
import numpy as np

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx
import mujoco_warp as mjw

from hydrax.algs.cem import CEM, CEMParams
from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


# Helper: Squeeze dimensions
def squeeze_n(x, n):
    for _ in range(n):
        x = jnp.squeeze(x, axis=0)
    return x


NOISE_BETA = 3  # Fetch pick-place


@dataclass
class ICEMParams(CEMParams):
    """Policy parameters for the cross-entropy method.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        cov: The (diagonal) covariance of the control distribution.
        last_elites: The indices of the last best controls
        last_rollout_knots: The last elite knots of the control.
    """

    last_elites: Optional[jax.Array] = None
    last_rollout_knots: Optional[jax.Array] = None


class ICEM(CEM):
    """Improved Cross-entropy method
    https://github.com/martius-lab/iCEM
    """

    def __init__(
            self,
            task: Task,
            num_samples: int,
            num_elites: int,
            sigma_start: float,
            sigma_min: float,
            num_randomizations: int = 1,
            explore_fraction: float = 0.0,
            risk_strategy: RiskStrategy = None,
            seed: int = 0,
            plan_horizon: float = 1.0,
            spline_type: Literal["zero", "linear", "cubic"] = "zero",
            num_knots: int = 4,
            iterations: int = 1,
            alpha: float = 7.0,
    ) -> None:
        """
        Args:
            alpha: Temperature/sharpness of weighting (higher = more focus on best).
        """
        super().__init__(
            task=task,
            num_samples=num_samples,
            num_elites=num_elites,
            sigma_start=sigma_start,
            sigma_min=sigma_min,
            num_randomizations=num_randomizations,
            explore_fraction=explore_fraction,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.alpha = alpha
        self.factor_decrease_num = 1.25
        self.sampling_count = 0
        self.keep_previous_elites = True
        self.fraction_elites_reused = 0.3
        self.main_shape = (
            self.num_samples - self.num_explore,
            self.num_knots,
            self.task.mjx_model.nu
        )
        self.explore_shape = (
            self.num_explore,
            self.num_knots,
            self.task.mjx_model.nu
        )

    def init_params(
            self, initial_knots: jax.Array = None, seed: int = 0
    ) -> ICEMParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return ICEMParams(
            tk=_params.tk, mean=_params.mean, cov=_params.cov, rng=_params.rng,
            last_elites=jnp.zeros_like(jnp.arange(self.num_elites)),
            last_rollout_knots=jnp.tile(initial_knots[jnp.newaxis, ...],
                                        (self.num_samples, 1, 1)) if initial_knots is not None and jnp.any(
                initial_knots.size > 0) else
            jnp.zeros((self.num_samples, self.num_knots, self.task.mjx_model.nu))
        )

    def sample_knots(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample a control sequence."""

        # Decay of sample size
        # self.num_samples = jnp.int32(self.num_samples / self.factor_decrease_num)

        # Start sampling
        rng, sample_rng, explore_rng = jax.random.split(params.rng, 3)

        # Pre-compute shapes for both main and exploration samples
        # Sample main knots with current covariance
        main_colored_noise = colorednoise.powerlaw_psd_gaussian(NOISE_BETA, size=self.main_shape)
        main_controls = params.mean + params.cov * jnp.array(main_colored_noise) \
            if (self.main_shape[0] > 0) else jnp.empty(self.main_shape)

        # Sample exploration knots with initial covariance
        explore_colored_noise = colorednoise.powerlaw_psd_gaussian(NOISE_BETA, size=self.explore_shape)
        explore_controls = params.mean + self.sigma_start * jnp.array(explore_colored_noise) \
            if self.explore_shape[0] > 0 else jnp.empty(self.explore_shape)

        # Combine both sets of controls
        controls = jnp.concatenate([main_controls, explore_controls])
        return controls, params.replace(rng=rng)

    def update_params(self, params: ICEMParams, rollouts: Trajectory) -> CEMParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        # Sort the costs and get the indices of the elites.
        indices = jnp.argsort(costs)
        elites = indices[: self.num_elites]

        # The new proposal distribution is a Gaussian fit to the elites.
        # Combine elites from this iteration & a subset of prev one
        elite_knots = jnp.concatenate([rollouts.knots[elites],
                                       params.last_rollout_knots[
                                           params.last_elites[
                                               :int(len(params.last_elites) * self.fraction_elites_reused)]]])
        mean = jnp.mean(elite_knots, axis=0)
        cov = jnp.maximum(
            jnp.std(elite_knots, axis=0), self.sigma_min
        )
        return params.replace(mean=mean, cov=cov,
                              last_rollout_knots=rollouts.knots,
                              last_elites=elites) if self.keep_previous_elites else (
            params.replace(mean=mean, cov=cov))
