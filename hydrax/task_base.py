import os

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional
from etils import epath

import numpy as np
import jax
import jax.numpy as jnp

import mujoco as mj
from mujoco import mjx

jax.config.update("jax_check_tracer_leaks", True)


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def get_assets(self) -> Dict[str, bytes]:
        return {}

    def __init__(
            self,
            mj_model: Optional[mj.MjModel] = None,
            xml_path: Optional[epath.Path] = None,
            u_min: Optional[np.ndarray] = None,
            u_max: Optional[np.ndarray] = None,
            sim_dt: Optional[float] = 0.01,
            ctrl_dt: Optional[float] = 0.01,
            trace_sites: Optional[Sequence[str]] = None,
        impl: str = "warp",
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            u_min: Minimum control values.
            u_max: Maximum control values.
            impl: The backend implementation for rollouts ("jax" for standard
                  MJX or "warp" for MjWarp).

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        self._mj_model: mj.MjModel = None
        self._mj_data: mj.MjData = None
        self.warp_enabled = (impl == 'warp')
        self._mjx_model: mjx.Model = None
        self._xml_path: str = ""
        if not hasattr(self, "sim_dt"):
            self.sim_dt = sim_dt
        if not hasattr(self, "ctrl_dt"):
            self.ctrl_dt = ctrl_dt
        self.trace_sites = trace_sites if trace_sites else []
        self.u_min = u_min
        self.u_max = u_max
        self.num_ctrls: int = u_min.shape[0] if u_min is not None else 0

        # MJ-Model
        if mj_model is not None:
            assert isinstance(mj_model, mj.MjModel)
            self._mj_model = mj_model
        elif xml_path is not None:
            self._xml_path = xml_path.as_posix()
            xml = xml_path.read_text()
            self._mj_model = mj.MjModel.from_xml_string(xml, assets=self.get_assets())
        else:
            print("Mj/Mjx-Models will be created from spec later!")

        # NOTE: [_pos_init] is expected to be called independently up to specific child class

    def _post_init(self, obj_name: Optional[str] = None, keyframe: Optional[str] = None) -> None:
        self._obj_name = obj_name

        assert self._mj_model is not None
        self._mj_model.opt.timestep = self.sim_dt

        # MJ-Data
        self._mj_data = mj.MjData(self._mj_model)

        # MJX-Model
        # NOTE: Only create [mjx-model] here, [mjx-data] is dynamically made/updated at each rollout
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)

        # Set actuator limits
        if self.u_min is None:
            self.u_min = jnp.where(
                self.mj_model.actuator_ctrllimited,
                self.mj_model.actuator_ctrlrange[:, 0],
                -jnp.inf,
            )
            self.num_ctrls = self.mj_model.nu
        if self.u_max is None:
            self.u_max = jnp.where(
                self.mj_model.actuator_ctrllimited,
                self.mj_model.actuator_ctrlrange[:, 1],
                jnp.inf,
            )

        # Get site IDs for points we want to trace
        self.trace_sites += [obj_name]
        self.trace_site_ids = jnp.array(
            [self.mj_model.site(name).id for name in self.trace_sites]
        )

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def dt(self):
        return self.ctrl_dt

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def xml_path(self) -> str:
        return self._xml_path

    def next_phase(self, state: mjx.Data) -> jnp.int32:
        return 0

    def step_callback(self, state: mjx.Data):
        pass

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
            self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def make_data(self, **kwargs) -> mjx.Data:
        """Create a new state consistent with this task.

        By default, this just creates a new `mjx.Data` instance from the model.
        Specific tasks can override this method to set parameters that must be
        adjusted per task, e.g., nconmax and naconmax.

        TODO(vincekurtz): figure out a smarter place to set naconmax and njmax.
        N.B. when performing parallel rollouts with MjWarp, naconmax and
        njmax need to be set high enough to support constraint solving across
        *all* rollouts. This means that these parameters scale with the number
        of parallel rollouts/samples, as well as the complexity of the task.

        Args:
            **kwargs: Additional keyword arguments to pass to `mjx.make_data`.

        Returns:
            A new `mjx.Data` instance for this task.
        """
        return mjx.make_data(self.mj_model, impl=self.model.impl, **kwargs)
