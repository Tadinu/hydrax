from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence, Optional, Union
from etils import epath
from copy import deepcopy

import numpy as np
import newton
from newton._src.utils.recorder import RecorderModelAndState
import warp as wp

# hydrax
from hydrax.utils.video import VideoRecorder

DATA_COLLECTION = False
if DATA_COLLECTION:
    # robotsuite
    from robosuite.utils.binding_utils import MjSimState
    from hydrax import DATA_DIR
    from hydrax.data_collector import DataCollector


class TaskNewton(ABC):
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
            nt_model: Optional[newton.Model] = None,
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
            nt_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            u_min: Minimum control values.
            u_max: Maximum control values.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        self.name: str = name
        self._nt_model: newton.Model = None
        self._nt_state: newton.State = None
        self._xml_path: str = ""
        self._nt_viewer: newton.viewer.ViewerGL = None
        self._nt_recorder: RecorderModelAndState = None
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
        self.ctrl_callback: Optional[Callable[[newton.State, wp.Array], wp.Array]] = None

        # MJ-Model
        if nt_model is not None:
            assert isinstance(nt_model, newton.Model)
            self._nt_model = nt_model
        elif xml_path is not None:
            self._xml_path = xml_path.as_posix()
            xml = xml_path.read_text()
            self._nt_model_builder = newton.ModelBuilder()
            self._nt_model = self._nt_model_builder.add_mjcf(xml)
        else:
            self._nt_model = self._construct_system_model()

        # Ref trajectory qpos (for cost calculation)
        self.ref_qpos: np.ndarray = np.zeros(self._nt_model.joint_count)

        # Post init
        self._post_init()

        # Data collector
        # tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        self._data_collector = DataCollector(self, DATA_DIR) if DATA_COLLECTION else None
        self._ep_meta = {}

    def _construct_system_model(self) -> Optional[newton.Model]:
        return None

    def _post_init(self) -> None:
        # NT-State
        self._nt_state = newton.State()

        # Set actuator limits
        if self.u_min is None:
            self.u_min = wp.where(
                self.nt_model.actuator_ctrllimited,
                self.nt_model.actuator_ctrlrange[:, 0],
                -wp.inf,
            )
            self.num_ctrls = self.nt_model.joint_dof_count
        if self.u_max is None:
            self.u_max = wp.where(
                self.nt_model.actuator_ctrllimited,
                self.nt_model.actuator_ctrlrange[:, 1],
                wp.inf,
            )

        # Get site IDs for points we want to trace
        self._init_trace_sites()

    def _init_trace_sites(self):
        if self._obj_name:
            self.trace_sites += [self._obj_name]
        self.trace_site_ids = wp.array(
            [self.nt_model.shape_key.index(name) for name in self.trace_sites]
        )

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def dt(self):
        return self.ctrl_dt

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def nt_model(self) -> newton.Model:
        return self._nt_model

    @property
    def nt_state(self) -> newton.State:
        return self._nt_state

    @property  # -> Consistent with co-parent [mjx_env.MjxEnv]
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def home_qpos(self):
        return []

    def next_phase(self, state: newton.State) -> wp.int32:
        return 0

    def step_callback(self, state: newton.State):
        pass

    def step(self, action: np.ndarray, kinematics_only: bool = False):
        pass

    def reset(self):
        pass

    def update_ref_qpos(self, ee_pose: Optional[Union[np.ndarray, wp.Array]] = None) -> None:
        pass

    @abstractmethod
    def running_cost(self, state: newton.State, control: wp.Array) -> wp.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: newton.State) -> wp.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def domain_randomize_model(self, rng: wp.Array) -> newton.ModelBuilder:
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

    def domain_randomize_data(self, data: newton.State, rng: wp.Array) -> Dict[str, wp.Array]:
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
        print({name: self._nt_model.sensor_dim[i] for name, i in sensor_ids.items()})

    def get_sensor_id(self, sensor_name: str) -> int:
        return self._nt_model.sensor(sensor_name).id

    def get_sensor_data(self, mjx_data: mjx.Data, sensor_id: int, start: int = 0, end: int = 0) -> jax.Array:
        """Get sensor data given sensor id."""
        # NOTE: Don't use [self.mjx_model], which may give incorrect adr if [warp_enabled] (This may be solved on future release)
        sensor_adr = self.nt_model.sensor_adr[sensor_id]
        sensor_dim = self.nt_model.sensor_dim[sensor_id]
        return mjx_data.sensordata[sensor_adr + start: sensor_adr + (end if end else sensor_dim)]

    def get_sensor_data_by_name(self, mjx_data: mjx.Data, sensor_name: str, start: int = 0, end: int = 0) -> jax.Array:
        """Get sensor data given sensor name."""
        sensor_id = self.nt_model.sensor(sensor_name).id
        return self.get_sensor_data(mjx_data, sensor_id, start, end)

    def get_trace_sites(self, state: newton.State) -> wp.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return wp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def get_base_pose(self, state: newton.State) -> wp.Array:
        return wp.zeros(7)

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self._nt_state.time = value.time
        wp.copy(value.joint_q, self._nt_state.joint_q[:])
        wp.copy(self._nt_state.joint_qd[:], value.joint_qd)

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
        # mj.mj_forward(self._nt_model, self._nt_state)
        # self._nt_viewer.sync()
        pass
