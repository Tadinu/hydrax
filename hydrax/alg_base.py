from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Literal, Tuple, Optional, Union

import numpy as np
import warp as wp

import jax
import jax.numpy as jnp
from flax.struct import dataclass

import mujoco as mj
from mujoco import mjx
import mujoco_warp as mjw

from hydrax.risk import AverageCost, RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.spline import get_interp_func


@dataclass
class Trajectory:
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

    controls: jax.Array
    knots: jax.Array
    costs: jax.Array
    trace_sites: jax.Array
    base_poses: jax.Array

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

    tk: jax.Array
    mean: jax.Array
    rng: jax.Array


@wp.kernel
def wp_default_mjw_ctrl_callback(ins: wp.array(dtype=float, ndim=2),
                                 outs: wp.array(dtype=float, ndim=2)):
    pass


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
            self,
            task: Task,
            num_samples: int,
            num_randomizations: int,
            risk_strategy: RiskStrategy,
            seed: int,
            plan_horizon: float,
            spline_type: Literal["zero", "linear", "cubic"] = "zero",
            num_knots: int = 4,
            iterations: int = 1,
            ctrl_callback: Optional[Callable[[mjx.Data, jax.Array], jax.Array]] = None
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_samples: The number of rollouts to sample
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
        self.num_samples = num_samples
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.mjx_model as the controller's
        # internal model, not the model used for simulation. dt is the
        # time between spline queries.
        self.plan_horizon = plan_horizon
        self.dt = self.task.dt
        self.ctrl_steps = int(round(self.plan_horizon / self.dt))

        # Spline setup for control interpolation
        self.spline_type = spline_type
        self.num_knots = num_knots
        self.interp_func = get_interp_func(spline_type)

        # MJ-JAX
        # Use a single model (no domain randomization) by default
        self.mjx_model = task.mjx_model
        self.mjx_randomized_model_template: mjx.Model = None

        # MJ-WARP
        self.mjw_model = task.mjw_model
        task.mjw_init_data(num_samples)
        self.mjw_rollout_graph = None
        self.mjw_rollout_initial_model: mjw.Model = None
        self.mjw_rollout_initial_state: mjw.Data = self.task.mjw_data
        self.mjw_rollout_control_inputs: wp.array(dtype=float) = wp.zeros(
            (self.num_samples, self.ctrl_steps, self.num_ctrls)) if self.mjw_model \
            else None
        self.mjw_rollout_control_outputs: wp.array(dtype=float) = wp.zeros(
            (self.num_samples, self.ctrl_steps, self.mjw_model.nu)) if self.mjw_model \
            else None
        self.mjw_substeps_num = 1
        self.mjw_costs = jnp.zeros((self.num_samples, self.ctrl_steps))
        self.mjw_trace_sites = jnp.zeros((self.num_samples, self.ctrl_steps, 1, 3))
        self.mjw_base_poses = jnp.zeros((self.num_samples, self.ctrl_steps, 7))

        # Control Callback
        self.ctrl_callback = ctrl_callback if self.mjx_model else wp_default_mjw_ctrl_callback

        # Number of optimization iterations
        if iterations < 1:
            raise ValueError("iterations must be greater than 0!")

        self.iterations = iterations

        if self.num_randomizations > 1:
            if self.mjx_model:
                # Make domain randomized models
                rng = jax.random.key(seed)
                rng, subrng = jax.random.split(rng)
                subrngs = jax.random.split(subrng, num_randomizations)
                randomizations: dict[str, jax.Array] = jax.vmap(self.task.domain_randomize_model)(subrngs)
                self.mjx_model = self.task.mjx_model.tree_replace(randomizations)
                # Keep track of which elements of the model have randomization
                self.mjx_randomized_model_template: mjx.Model = jax.tree.map(lambda x: None,
                                                                             self.task.mjx_model).tree_replace(
                    {key: 0 for key in randomizations.keys()}
                )
            else:
                randomizations = self.task.wp_domain_randomize_model(kernel_seed=seed)
                # Ref: https://mujoco.readthedocs.io/en/latest/mjwarp/index.html#batched-model-fields
                self.mjw_rollout_initial_model = self.task.mjw_model
                for field, value in randomizations.items():
                    wp.copy(getattr(self.mjw_rollout_initial_model, field), wp.from_jax(value))
                self.mjw_capture_rollout()

    def step_callback(self, state: Union[mjx.Data, mjw.Data]):
        self.task.step_callback(state)

    def optimize(self, state: Union[mjx.Data, mjw.Data], params: Any) -> Tuple[Any, Trajectory]:
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
        new_tk = (
                jnp.linspace(0.0, self.plan_horizon, self.num_knots) +
                (state.time if isinstance(state, mjx.Data) else wp.to_jax(state.time)[0])
        )

        # Clamp query times to the old spline's domain to avoid extrapolation,
        # which can produce wildly wrong values for linear/cubic splines.
        clamped_tk = jnp.clip(new_tk, tk[0], tk[-1])

        new_mean = self.interp_func(clamped_tk, tk, params.mean[None, ...])[0] 
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, iteration: Any) -> tuple[Any, Trajectory]:
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, self.num_ctrls)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return params, rollouts

        if isinstance(state, mjx.Data):
            params, rollouts = jax.lax.scan(
                f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
            )
        else:
            for _ in range(self.iterations):
                params, rollouts = _optimize_scan_body(params, iteration=_)

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        return params, rollouts_final

    def rollout_with_randomizations(
            self,
            initial_state: Union[mjx.Data, mjw.Data],
            tk: jax.Array,
            knots: jax.Array,
            rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            initial_state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        # (self.num_randomizations, initial_state.shape)
        mjx_initial_states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), initial_state
        ) if self.mjx_model else None

        if self.num_randomizations > 1:
            if self.mjx_model:
                # Randomize the initial states for each domain randomization
                subrngs = jax.random.split(rng, self.num_randomizations)
                randomizations: dict[str, jax.Array] = jax.vmap(self.task.domain_randomize_data)(
                    mjx_initial_states, subrngs
                )
                mjx_initial_states = mjx_initial_states.tree_replace(randomizations)
            else:
                randomizations: dict[str, jax.Array] = self.task.wp_domain_randomize_data(0)

        # Compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, self.ctrl_steps, self.num_ctrls)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        if self.mjx_model:
            _, rollouts = jax.vmap(
                self.eval_rollouts, in_axes=(self.mjx_randomized_model_template, 0, None, None)
            )(self.mjx_model, mjx_initial_states, controls, knots)
        else:
            # Ref: https://mujoco.readthedocs.io/en/latest/mjwarp/index.html#batched-model-fields
            for field, value in randomizations.items():
                wp.copy(getattr(initial_state, field), wp.from_jax(value))
            rollouts = self.wp_eval_rollouts(initial_state, controls, knots)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        # (self.num_randomizations, self.num_samples, self.ctrl_steps) -> (self.num_samples, self.ctrl_steps)
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st

        # (self.num_randomizations, self.num_samples, self.ctrl_steps, 7) -> (self.num_samples, self.ctrl_steps, 7)
        base_poses = jnp.mean(rollouts.base_poses, axis=0)
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites, base_poses=base_poses
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
            self,
            model: mjx.Model,
            state: mjx.Data,
            controls: jax.Array,
            knots: jax.Array,
            n_substeps: int = 1
    ) -> Tuple[mjx.Data, Trajectory]:
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

        def _scan_fn(x: Union[mjx.Data, mjw.Data], u: jax.Array) -> Tuple[
            mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""

            def valid_cb(args):
                x, u = args
                return self.ctrl_callback(x, u) if self.ctrl_callback else jnp.zeros(model.nu)

            def void_cb(args):
                x, u = args
                return u if u.size == model.nu else jnp.zeros(model.nu)

            ctrl = jax.lax.cond(self.ctrl_callback is not None,
                                valid_cb,
                                void_cb,
                                (x, u))

            def single_step(data, _):
                data = data.replace(ctrl=ctrl)
                data = mjx.step(model, data)
                return data, None

            x = jax.lax.scan(single_step, x, (), n_substeps)[0]  # 0 for data
            cost = self.dt * self.task.running_cost(x, ctrl)
            sites = self.task.get_trace_sites(x)
            base_pose = self.task.get_base_pose(x)
            next_phase = self.task.next_phase(x)
            x = x.replace(userdata=jnp.array([next_phase], dtype=jnp.float32))
            return x, (x, cost, sites, base_pose)

        final_state, (states, costs, trace_sites, base_poses) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)
        final_base_pose = self.task.get_base_pose(final_state)
        final_base_pose = jnp.reshape(final_base_pose, (1, final_base_pose.size))

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)
        base_poses = jnp.append(base_poses, final_base_pose, axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
            base_poses=base_poses
        )

    def mjw_capture_rollout(self):

        @wp.kernel
        def wp_kernel_copy_controls(src_ctrls: wp.array(dtype=float, ndim=2),
                                    dest_ctrls: wp.array(dtype=float, ndim=3),
                                    idx: int):
            i, j = wp.tid()

            # write into 3D array:
            # axis order assumed: [batch, slice, feature]
            dest_ctrls[i, idx, j] = src_ctrls[i, j]

        with wp.ScopedCapture() as capture:
            """Compute the cost and observation, then advance the state."""
            mjw_model = self.mjw_rollout_initial_model
            mjw_data = self.mjw_rollout_initial_state
            for _ in range(self.ctrl_steps):
                wp_ctrl_in = self.mjw_rollout_control_inputs[:, _, :]
                valid_cb = wp.ones(1, dtype=wp.int32) if self.ctrl_callback is not None else wp.zeros(0, dtype=wp.int32)
                wp_ctrl_out = wp.zeros((self.num_samples, mjw_model.nu),
                                       dtype=wp.float32) if self.ctrl_callback is not None else wp_ctrl_in
                wp.capture_if(valid_cb,
                              wp.launch(self.ctrl_callback, dim=len(wp_ctrl_in),
                                        inputs=[wp_ctrl_in],
                                        outputs=[wp_ctrl_out]))

                for i in range(self.mjw_substeps_num):
                    mjw_data.ctrl.assign(wp_ctrl_out)
                    mjw.forward(mjw_model, mjw_data)

                wp.launch(
                    wp_kernel_copy_controls,
                    dim=(self.num_samples, mjw_model.nu),
                    inputs=[wp_ctrl_out,
                            self.mjw_rollout_control_outputs,
                            _],
                    device=wp_ctrl_out.device
                )
        self.mjw_rollout_graph = capture.graph

    def wp_eval_rollouts(self,
                         initial_state: mjw.Data,
                         controls: jax.Array,
                         knots: jax.Array) -> Trajectory:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).
        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """
        self.mjw_rollout_initial_state = initial_state
        self.mjw_rollout_control_inputs = wp.from_jax(controls)
        wp.capture_launch(self.mjw_rollout_graph)
        for batch_idx in range(self.num_samples):
            self.mjw_trace_sites = jnp.concatenate([self.mjw_trace_sites,
                                                    jnp.tile(
                                                        self.task.get_trace_sites(self.mjw_rollout_initial_state,
                                                                                  batch_idx)
                                                        [None, None, ...],
                                                        (self.num_samples, self.ctrl_steps, 1, 1))],
                                                   axis=2)
            for step_idx in range(self.ctrl_steps):
                wp_ctrl = wp.clone(self.mjw_rollout_control_outputs[batch_idx, step_idx])
                self.mjw_costs.at[batch_idx, step_idx].set(
                    self.dt * self.task.running_cost(self.mjw_rollout_initial_state,
                                                     wp.to_jax(wp_ctrl),
                                                     batch_idx=batch_idx))
                self.mjw_base_poses.at[batch_idx, step_idx].set(self.task.get_base_pose(self.mjw_rollout_initial_state,
                                                                                        batch_idx))

        return Trajectory(
            controls=controls,
            knots=knots,
            costs=self.mjw_costs,
            trace_sites=self.mjw_trace_sites,
            base_poses=self.mjw_base_poses
        )

    def init_params(
            self, initial_knots: jax.Array = None, seed: int = 0
    ) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_knots: The initial knots of the control spline.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else jnp.zeros((self.num_knots, self.num_ctrls))
        )
        assert mean.shape == (self.num_knots, self.num_ctrls), (
            f"Initial knots must have shape (num_knots, nu), got {mean.shape}"
        )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        return SamplingParams(tk=tk, mean=mean, rng=rng)

    @abstractmethod
    def sample_knots(self, params: Any) -> Tuple[jax.Array, Any]:
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

    def get_action(self, params: SamplingParams, t: jax.Array) -> jax.Array:
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
