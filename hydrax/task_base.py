import os
# import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence, Optional, Union
from etils import epath
from copy import deepcopy

import numpy as np
import warp as wp
import jax
import jax.numpy as jnp

import mujoco as mj
import mujoco.viewer
from mujoco import mjx
from mujoco.mjx._src import types as mjx_types
import mujoco_warp as mjw

# hydrax
from hydrax import BackendType
from hydrax.utils.video import VideoRecorder

# mjmanip
from mjmanip.utils import mj_step

DATA_COLLECTION = False
if DATA_COLLECTION:
    # robotsuite
    from robosuite.utils.binding_utils import MjSimState
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

    FABRIC_ENV_WORLD_FILE_NAME: Optional[str] = None

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
            backend_type: Optional[BackendType] = BackendType.MJX
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
        self.name: str = name
        self._mj_model: mj.MjModel = None
        self._mj_data: mj.MjData = None
        self.warp_enabled = (impl == 'warp')
        self._mjx_model: mjx.Model = None
        self._mjw_model: mjw.Model = None
        self._xml_path: str = ""
        self._mj_viewer: mj.viewer.Handle = None
        self._mj_renderer: mj.Renderer = None
        self._mj_recorder: VideoRecorder = None
        if not hasattr(self, "sim_dt"):
            self.sim_dt: float = sim_dt
        if not hasattr(self, "ctrl_dt"):
            self.ctrl_dt: float = ctrl_dt
        self._obj_name: str = obj_name
        self._keyframe: str = keyframe
        self.trace_sites = trace_sites if trace_sites else []
        self.backend_type: BackendType = backend_type
        self.u_min: np.ndarray = u_min
        self.u_max: np.ndarray = u_max
        self.num_ctrls: int = u_min.size if u_min is not None else 0
        self.ctrl_callback: Optional[Callable[[mjx.Data, jax.Array], jax.Array]] = None
        self.num_samples: int = 0

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
        self._data_collector = DataCollector(self, DATA_DIR) if DATA_COLLECTION else None
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
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl) if (self.backend_type == BackendType.MJX or self.backend_type == BackendType.MJX_WARP) else None
        self._mjw_model = mjw.put_model(self.mj_model) if (self.backend_type == BackendType.MJW) else None
        self._mjw_data: mjw.Data = None
        self._mjw_step_graph = None
        self._mjw_forward_graph = None

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

    def mjw_init_data(self, num_samples: int):
        if self.mjw_model:
            self.num_samples = num_samples
            self._mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=num_samples,
                                          njmax=1000)
            self.mjw_create_graphs()

    def mjw_create_graphs(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            with wp.ScopedCapture() as capture:
                mjw.step(self.mjw_model, self.mjw_data)
            self._mjw_step_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mjw.forward(self.mjw_model, self.mjw_data)
            self._mjw_forward_graph = capture.graph

    def mjw_forward(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            if self._mjw_forward_graph is not None:
                wp.capture_launch(self._mjw_forward_graph)
            else:
                mjw.forward(self.mjw_model, self.mjw_data)

    def mjw_step(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            if self._mjw_step_graph is not None:
                wp.capture_launch(self._mjw_step_graph)
            else:
                mjw.step(self.mjw_model, self.mjw_data)

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def dt(self):
        return self.ctrl_dt

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mj_data(self) -> mj.MjData:
        return self._mj_data

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def mjx_impl(self) -> Optional[str]:
        return self._mjx_model.impl.value if self._mjx_model else None

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mjw_model(self) -> mjw.Model:
        return self._mjw_model

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def mjw_data(self) -> mjw.Data:
        return self._mjw_data

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def home_qpos(self):
        return []

    def next_phase(self, state: Union[mjx.Data, mjw.Data]) -> jnp.int32:
        return 0

    def step_callback(self, state: Union[mjx.Data, mjw.Data]):
        pass

    def step(self, action: np.ndarray, kinematics_only: bool = False):
        if kinematics_only:
            # TODO: Use specifically qpos_ids here instead
            self._mj_data.qpos[:self._mj_model.nu] = action[:self._mj_model.nu]
        else:
            self._mj_data.ctrl = action[:self._mj_model.nu]
        # Still step physically regardless to get physical interaction with objects
        mj.mj_step(self._mj_model, self._mj_data)
        if self._mj_viewer:
            self._mj_viewer.sync()
            print("STEP", action)
            if self._mj_renderer and self._mj_recorder and self._mj_recorder.is_recording:
                self._mj_renderer.update_scene(self._mj_data, self._mj_viewer.cam)
                frame = self._mj_renderer.render()
                self._mj_recorder.add_frame(frame.tobytes())

    def reset(self):
        pass

    def update_ref_qpos(self, ee_pose: Optional[Union[np.ndarray, jnp.ndarray]] = None) -> None:
        pass

    @abstractmethod
    def running_cost(self, state: Union[mjx.Data, mjw.Data], control: jax.Array,
                     batch_idx: Optional[int] = -1) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.
            step: The current step number.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
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

    def wp_domain_randomize_model(self, kernel_seed: int) -> Dict[str, jax.Array]:
        return {}

    def domain_randomize_data(self, data: Union[mjx.Data, mjw.Data], rng: jax.Array) -> Dict[str, jax.Array]:
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

    def wp_domain_randomize_data(self, kernel_seed: int) -> Dict[str, jax.Array]:
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

    def get_sensor_data(self, data: Union[mjx.Data, mjw.Data], sensor_id: int,
                        start: int = 0, end: int = 0,
                        batch_idx: int = 0) -> jax.Array:
        """Get sensor data given sensor id."""
        # NOTE: Don't use [self.mjx_model], which may give incorrect adr in case of [MJX_WARP] backend
        # (This may be solved on future release)
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        if isinstance(data, mjx.Data):
            return data.sensordata[sensor_adr + start: sensor_adr + (end if end else sensor_dim)]
        else:
            sensor_adr_start = sensor_adr + start
            sensor_adr_end = sensor_adr + (end if end else sensor_dim)
            return wp.to_jax(data.sensordata)[batch_idx, sensor_adr_start: sensor_adr_end]

    def get_sensor_data_by_name(self, data: Union[mjx.Data, mjw.Data], sensor_name: str, start: int = 0,
                                end: int = 0,
                                batch_idx: int = 0) -> jax.Array:
        """Get sensor data given sensor name."""
        sensor_id = self.mj_model.sensor(sensor_name).id
        return self.get_sensor_data(data, sensor_id, start, end, batch_idx=batch_idx)

    def get_trace_sites(self, state: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids] if self.mjx_model \
            else wp.to_jax(state.site_xpos)[batch_idx, self.trace_site_ids]

    def get_base_pose(self, state: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jnp.ndarray:
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
