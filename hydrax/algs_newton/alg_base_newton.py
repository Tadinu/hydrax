from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Literal, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import newton
import warp as wp

from hydrax.risk_newton import AverageCostNewton, RiskStrategyNewton
from hydrax.task_base_newton import TaskNewton
from hydrax.utils.spline import get_interp_func


@dataclass
class TrajectoryNewton:
    """Data class for storing rollout data.

    Throughout, H denotes the number of control steps (given by the times at
    which the control spline is interpolated).

    Attributes:
        controls: Control actions of shape (num_rollouts, H, nu).
        knots: Control spline knots of shape (num_rollouts, num_knots, nu).
        costs: Costs of shape (num_rollouts, H+1).
        trace_sites: Poses of trace sites of shape (num_rollouts, H+1, 3).
        base_poses: Poses of agent base (num_rollouts, H+1, 7).
    """

    controls: wp.Array
    knots: wp.Array
    costs: wp.Array
    trace_sites: wp.Array
    base_poses: wp.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1

    def print(self):
        print("Control", self.controls.shape)  # num_samples, horizon (ctrl_steps), num_ctrls
        print("Knots", self.knots.shape)
        print("Costs", self.costs.shape)
        print("Base poses", self.base_poses.shape)


@dataclass
class SamplingParams:
    """Parameters for sampling-based control algorithms.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

    tk: wp.Array
    mean: wp.Array
    rng: wp.Array


class SamplingBasedControllerNewton(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
            self,
            task: TaskNewton,
            num_randomizations: int,
            risk_strategy: RiskStrategyNewton,
            seed: int,
            plan_horizon: float,
            spline_type: Literal["zero", "linear", "cubic"] = "zero",
            num_knots: int = 4,
            iterations: int = 1,
            ctrl_callback: Optional[Callable[[newton.State, wp.Array], wp.Array]] = None
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        self.task = task
        self.num_ctrls = task.num_ctrls
        self.num_randomizations = max(num_randomizations, 1)
        self.ctrl_callback = ctrl_callback

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCostNewton()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.nt_model as the controller's
        # internal model, not the model used for simulation. dt is the
        # time between spline queries.
        self.plan_horizon = plan_horizon
        self.dt = self.task.dt
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))

        # Spline setup for control interpolation
        self.spline_type = spline_type
        self.num_knots = num_knots
        self.interp_func = get_interp_func(spline_type)

        # Use a single model (no domain randomization) by default
        self.nt_model = task.nt_model
        self.randomized_axes = None

        # Number of optimization iterations
        if iterations < 1:
            raise ValueError("iterations must be greater than 0!")

        self.iterations = iterations

        if self.num_randomizations > 1:
            # Make domain randomized models
            subrngs = [wp.rand_init(seed)] * num_randomizations
            random_models = [self.task.domain_randomize_model(subrng) for subrng in subrngs]
            self.nt_model = random_models.pop()

    def step_callback(self, state: newton.State):
        self.task.step_callback(state)

    def optimize(self, state: newton.State, params: Any) -> Tuple[Any, TrajectoryNewton]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = wp.array(
            np.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        @wp.kernel
        def _optimize_scan_body(params: Any, iteration: Any):
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = wp.clamp(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, self.num_ctrls)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = wp.rand_init(params.rng)
            rollouts = self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return params, rollouts

        params = []
        rollouts = []
        for i in range(self.iterations):
            # launch kernel that does one step: new_state, out_i = f(state, i)
            param, rollout_i = _optimize_scan_body(params[i], i)
            params.append(param)
            rollouts.append(rollout_i)

        rollouts_final = rollouts[-1]

        return params, rollouts_final

    def rollout_with_randomizations(
            self,
            state: newton.State,
            tk: wp.Array,
            knots: wp.Array,
            rng: wp.Array,
    ) -> TrajectoryNewton:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = wp.map(lambda _, x: x, in_axes=(0, None))(
            wp.array(np.arange(self.num_randomizations)), state
        )

        if self.num_randomizations > 1:
            # Make domain randomized models
            subrngs = [wp.rand_init(self.seed)] * self.num_randomizations
            states = [self.task.domain_randomize_data(subrng) for subrng in subrngs]

        # Compute the control sequence from the knots
        tq = wp.array(np.linspace(tk[0], tk[-1], self.ctrl_steps))
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, self.ctrl_steps, self.num_ctrls)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = wp.map(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None)
        )(self.nt_model, states, controls, knots)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        # (self.num_randomizations, self.num_samples, self.ctrl_steps) -> (self.num_samples, self.ctrl_steps)
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st

        # (self.num_randomizations, self.num_samples, self.ctrl_steps, 7) -> (self.num_samples, self.ctrl_steps, 7)
        base_poses = wp.array(np.mean(rollouts.base_poses, axis=0))
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites, base_poses=base_poses
        )

    # @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    @wp.kernel
    def eval_rollouts(
            self,
            model: newton.Model,
            state: newton.State,
            controls: wp.Array,
            knots: wp.Array,
            n_substeps: int = 1
    ) -> Tuple[newton.State, TrajectoryNewton]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            n_substeps: The number of steps per rollout.
        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(x: newton.State, u: wp.Array) -> Tuple[
            newton.State, Tuple[newton.State, wp.Array, wp.Array, wp.Array]]:
            """Compute the cost and observation, then advance the state."""

            def valid_cb(args):
                x, u = args
                return self.ctrl_callback(x, u) if self.ctrl_callback else wp.zeros(model.joint_dof_count)

            def void_cb(args):
                x, u = args
                return u if u.size == model.joint_dof_count else wp.zeros(model.joint_dof_count)

            ctrl = wp.where(self.ctrl_callback is not None,
                            valid_cb,
                            void_cb,
                            (x, u))

            @wp.kernel
            def kernel_single_step(ctrl: wp.array(dtype=float),
                                   joint_target: wp.array(dtype=float),
                                   n_steps: int):
                tid = wp.tid()
                for _ in range(n_steps):
                    joint_target[tid] = ctrl[tid]

            wp.launch(
                kernel_single_step,
                dim=model.joint_dof_count,
                inputs=[ctrl, 1], outputs=[model.joint_target_pos]
            )

            x = model.state()
            cost = self.dt * self.task.running_cost(x, ctrl)
            sites = self.task.get_trace_sites(x)
            base_pose = self.task.get_base_pose(x)
            return x, (x, cost, sites, base_pose)

        final_state, (states, costs, trace_sites, base_poses) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)
        final_base_pose = self.task.get_base_pose(final_state)
        final_base_pose = wp.Array.reshape(final_base_pose, (1, final_base_pose.size))

        costs = wp.Array.append(costs, final_cost)
        trace_sites = wp.Array.append(trace_sites, final_trace_sites[None], axis=0)
        base_poses = wp.Array.append(base_poses, final_base_pose, axis=0)

        return states, TrajectoryNewton(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
            base_poses=base_poses
        )

    def init_params(
            self, initial_knots: wp.Array = None, seed: int = 0
    ) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_knots: The initial knots of the control spline.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = wp.rand_init(self.seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else wp.zeros((self.num_knots, self.num_ctrls))
        )
        assert mean.shape == (self.num_knots, self.num_ctrls), (
            f"Initial knots must have shape (num_knots, nu), got {mean.shape}"
        )
        tk = wp.array(np.linspace(0.0, self.plan_horizon, self.num_knots))
        return SamplingParams(tk=tk, mean=mean, rng=rng)

    @abstractmethod
    def sample_knots(self, params: Any) -> Tuple[wp.Array, Any]:
        """Sample a set of control spline knots U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            Control spline knots U, size (num rollouts, num_knots).
            Updated parameters (e.g., with a new PRNG key).
        """

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """

    def get_action(self, params: SamplingParams, t: wp.Array) -> wp.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The current time at which to query the spline. Spline times are
                continually evolving as the simulation progresses, so this
                number should roughly track mj_data.time.

        Returns:
            The control action u(t).
        """
        knots = params.mean[None, ...]  # (1, num_knots, nu)
        tk = params.tk
        u = self.interp_func(t, tk, knots)[0]  # (nu,)
        return u
