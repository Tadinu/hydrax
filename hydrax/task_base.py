import os
# import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence, Optional, Union
from etils import epath
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp

import mujoco as mj
import mujoco.viewer
from mujoco import mjx

# robotsuite
from robosuite.utils.binding_utils import MjSimState

# hydrax
from hydrax import DATA_DIR
from hydrax.data_collector import DataCollector

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
            name: str,
            mj_model: Optional[mj.MjModel] = None,
            xml_path: Optional[epath.Path] = None,
            u_min: Optional[np.ndarray] = None,
            u_max: Optional[np.ndarray] = None,
            # NOTE: Purposefully zero here to only override [model.opt] upon valid sim_dt
            sim_dt: Optional[float] = 0.0,
            ctrl_dt: Optional[float] = 0.01,
            obj_name: Optional[str] = None,
            keyframe: Optional[str] = None,
            trace_sites: Optional[Sequence[str]] = None,
            warp_enabled: Optional[bool] = False,
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            u_min: Minimum control values.
            u_max: Maximum control values.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        self.name: str = name
        self._mj_model: mj.MjModel = None
        self._mj_data: mj.MjData = None
        self._mjx_model: mjx.Model = None
        self._xml_path: str = ""
        self._mj_viewer: mj.viewer = None
        if not hasattr(self, "sim_dt"):
            self.sim_dt = sim_dt
        if not hasattr(self, "ctrl_dt"):
            self.ctrl_dt = ctrl_dt
        self._obj_name: str = obj_name
        self._keyframe: str = keyframe
        self.trace_sites = trace_sites if trace_sites else []
        self.warp_enabled = warp_enabled
        self.u_min = u_min
        self.u_max = u_max
        self.num_ctrls: int = u_min.size if u_min is not None else 0
        self.ctrl_callback: Optional[Callable[[mjx.Data, jax.Array], jax.Array]] = None

        # MJ-Model
        if mj_model is not None:
            assert isinstance(mj_model, mj.MjModel)
            self._mj_model = mj_model
        elif xml_path is not None:
            self._xml_path = xml_path.as_posix()
            xml = xml_path.read_text()
            self._mj_model = mj.MjModel.from_xml_string(xml, assets=self.get_assets())
        else:
            self._mj_model = self._construct_system_model()

        # Ref trajectory qpos (for cost calculation)
        self.ref_qpos: np.ndarray = np.zeros(self._mj_model.nq)

        # Post init
        self._post_init()

        # Data collector
        # tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        self._data_collector = DataCollector(self, DATA_DIR)
        self._ep_meta = {}

    def _construct_system_model(self) -> Optional[mj.MjModel]:
        return None

    def _post_init(self) -> None:
        if self.sim_dt > 0.0:
            self._mj_model.opt.timestep = self.sim_dt

        # MJ-Data
        self._mj_data = mj.MjData(self._mj_model)

        # MJX-Model
        # NOTE: Only create [mjx-model] here, [mjx-data] is dynamically made/updated at each rollout
        self._mjx_model = mjx.put_model(self._mj_model, impl='warp') if self.warp_enabled \
            else mjx.put_model(self._mj_model)

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
        self._init_trace_sites()

    def _init_trace_sites(self):
        if self._obj_name:
            self.trace_sites += [self._obj_name]
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

    @property
    def mjx_impl(self) -> Optional[str]:
        return self._mjx_model.impl.value if self._mjx_model else None

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def home_qpos(self):
        return []

    def next_phase(self, state: mjx.Data) -> jnp.int32:
        return 0

    def step_callback(self, state: mjx.Data):
        pass

    def update_ref_qpos(self, ee_pose: Optional[Union[np.ndarray, jnp.ndarray]] = None) -> None:
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

    def check_success(self) -> bool:
        """
        Checks if the task has been completed. Should be implemented by subclasses
        Returns:
            bool: True if the task has been completed
        """
        pass

    def print_sensors_dim(self, sensor_ids: dict[str, int]) -> None:
        print({name: self._mj_model.sensor_dim[i] for name, i in sensor_ids.items()})

    def get_sensor_id(self, sensor_name: str) -> int:
        return self._mj_model.sensor(sensor_name).id

    def get_sensor_data(self, mjx_data: mjx.Data, sensor_id: int, start: int = 0, end: int = 0) -> jax.Array:
        """Get sensor data given sensor id."""
        # NOTE: Don't use [self.mjx_model], which may give incorrect adr if [warp_enabled] (This may be solved on future release)
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return mjx_data.sensordata[sensor_adr + start: sensor_adr + (end if end else sensor_dim)]

    def get_sensor_data_by_name(self, mjx_data: mjx.Data, sensor_name: str, start: int = 0, end: int = 0) -> jax.Array:
        """Get sensor data given sensor name."""
        sensor_id = self.mj_model.sensor(sensor_name).id
        return self.get_sensor_data(mjx_data, sensor_id, start, end)

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

    def get_base_pose(self, state: mjx.Data) -> jnp.ndarray:
        return jnp.zeros(7)

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self._mj_data.time,
            qpos=np.copy(self._mj_data.qpos),
            qvel=np.copy(self._mj_data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self._mj_data.time = value.time
        self._mj_data.qpos[:] = np.copy(value.qpos)
        self._mj_data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.

        See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjsimstate.pyx#L54
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self._mj_data.time = state.time
        self._mj_data.qpos[:] = state.qpos
        self._mj_data.qvel[:] = state.qvel

    def get_ep_meta(self):
        """
        Returns a dictionary containing episode metadata
        Returns:
            dict: episode metadata
        """
        return deepcopy(self._ep_meta)

    def set_ep_meta(self, meta):
        """
        Set episode meta data
        Args:
            meta (dict): containing episode metadata
        """
        self._ep_meta = meta

    def unset_ep_meta(self):
        """
        Unset episode meta data
        """
        self._ep_meta = {}

    def forward(self):
        mj.mj_forward(self._mj_model, self._mj_data)
        self._mj_viewer.sync()
