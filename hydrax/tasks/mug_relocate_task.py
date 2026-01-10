from typing import Any, Dict, Optional, Callable, Union
from enum import IntEnum
from functools import partial
from etils import epath

import numpy as np
import warp as wp

# jax
import jax
from jax import jit
import jax.numpy as jnp

# mujoco
from mujoco import mjx
import mujoco_warp as mjw

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT, BackendType
from hydrax.task_base import Task

# mjmanip
from mjmanip.robot.leap_mjx import LeapMjx
from mjmanip.robot.leap_fabrics import LeapWithFabrics, HAND_XML_PATH
from mjmanip.utils import mj_get_joints_qids
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.leap_fabrics import LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES, \
    LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

HAND_BASE_POSE = [np.array([-0.25, 0.0, 0.25]), np.array([0.0, 1.0, 0.0, 0.0])]
HAND_HOME_QPOS = [
    0.8, 0, 0.8, 0.8,
    0.8, 0, 0.8, 0.8,
    0.8, 0, 0.8, 0.8,
    0.8, 0.8, 0.8, 0,
]


class RelocatePhase(IntEnum):
    INITIAL = 0
    REACHING = 1
    GRASPING = 2
    RELOCATING = 3


class MugRelocateTask(Task):
    """Mug rotation with the LEAP hand."""

    PALM_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES
    FINGER_FABRIC_CONTROL_FRAMES = LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        models_path = epath.Path(ROOT) / "models"
        path = models_path / "leap_hand"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")

        path = models_path / "mug"
        mjx_env.update_assets(assets, path)
        mjx_env.update_assets(assets, path, "*.xml")
        return assets

    def __init__(self, name: str,
                 fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
                 use_ctrl_callback: bool = False,
                 backend_type: BackendType = BackendType.MJX) -> None:
        """Load the MuJoCo model and set task parameters."""

        self.HAND_MODEL_NAME = "leap_rh_mjx"
        self.FINGER_TIPS_NAMES = ["if_tip", "mf_tip", "rf_tip", "th_tip"]

        # Fabrics
        self.fabrics_controller: FabricsController = None
        self.fabric_cfg: ArmHandPoseFabricConfig = fabric_cfg
        self.ctrl_callback = self.mjx_fabrics_convert_free_hand_to_full_arm_hand_ctrl \
            if use_ctrl_callback and fabric_cfg else None

        super().__init__(name,
                         xml_path=epath.Path(ROOT) / "models" / "leap_hand" / "scene_leap_rh_mjx_relocate_mug.xml",
                         obj_name="mug",
                         trace_sites=["grasp_site"] + self.FINGER_TIPS_NAMES,
                         backend_type=backend_type)

        # Move [base_body]
        base_body = self.mj_model.body("leap_mount")
        base_body.pos = HAND_BASE_POSE[0]
        base_body.quat = HAND_BASE_POSE[1]

        # Obj
        obj = self._mj_model.body(self._obj_name)
        self._init_obj_qpos = np.concatenate([obj.pos, obj.quat])

        # Get sensor ids
        self.mug_position_sensor = self.get_sensor_id("mug_position")
        self.mug_orientation_sensor = self.get_sensor_id("mug_orientation")
        self.mug_contact_with_palm_sensor = self.get_sensor_id("mug_contact_with_palm")
        self.mug_distance_to_grasp_sensor = self.get_sensor_id("mug_distance_to_grasp")
        self.obj_contact_with_finger_tip_sensors = {
            finger_tip: self.get_sensor_id(f"mug_contact_with_{finger_tip}")
            for finger_tip in self.FINGER_TIPS_NAMES
        }
        self.mug_distance_to_target_sensor = self.get_sensor_id("mug_distance_to_target")
        self.mug_orientation_from_target_sensor = self.get_sensor_id("mug_orientation_from_target")
        self.mug_linear_velocity_sensor = self.get_sensor_id("mug_linear_vel")
        self.mug_angular_velocity_sensor = self.get_sensor_id("mug_angular_vel")
        self.finger_tip_distance_to_mug_sensors = [self.get_sensor_id(f"{finger_tip}_distance_to_mug")
                                                   for finger_tip in self.FINGER_TIPS_NAMES]

        # Distance (m) beyond which we impose a high mug position cost
        self.grasp_threshold = 0.05  # 0.015
        self.target_distance_threshold = 0.001

        # Task phase
        self.phase: RelocatePhase = RelocatePhase.INITIAL

    def _post_init(self) -> None:
        super()._post_init()

        # Hand-specifics
        self._init_hand()

        # Fabrics
        if self.fabric_cfg is not None:
            self._init_fabrics()

    def _init_fabrics(self):
        self.fabrics_robot = LeapWithFabrics()
        self.fabrics_robot.main_model = self.mj_model
        self.fabrics_robot.main_data = self.mj_data
        self.fabrics_robot.robot_qpos_ids = mj_get_joints_qids(self.mj_model, LeapWithFabrics.JOINTS_NAMES,
                                                               is_qpos=True)
        # NOTE: MjData is created here-in if needed in robot's configuration
        self.fabrics_controller = FabricsController(robot=self.fabrics_robot,
                                                    palm_control_frames=self.PALM_FABRIC_CONTROL_FRAMES,
                                                    finger_control_frames=self.FINGER_FABRIC_CONTROL_FRAMES,
                                                    fabric_cfg=self.fabric_cfg,
                                                    object_model_paths=LeapWithFabrics.OBJECT_MODEL_PATHS,
                                                    object_collision_mesh_names=LeapWithFabrics.OBJECT_COLLISION_MESH_NAMES,
                                                    robot_path_or_xml=HAND_XML_PATH,
                                                    env_world_file_name=self.FABRIC_ENV_WORLD_FILE_NAME,
                                                    use_finger_fabrics=False,
                                                    use_cuda_graph=True,
                                                    batch_size=1)
        from warp.jax_experimental import jax_callable
        from warp._src.jax_experimental.ffi import GraphMode

        def wp_fabrics_step(in_arr: wp.array(dtype=float), out_arr: wp.array(dtype=float)):
            self.fabrics_controller.step(wp.to_torch(in_arr))
            out_arr.assign(wp.from_torch(self.fabrics_controller.q))

        self.jax_fabrics_step = jit(jax_callable(wp_fabrics_step, graph_mode=GraphMode.WARP))

    def _init_hand(self):
        pass

    @property
    def home_qpos(self):
        return (
            HAND_HOME_QPOS + self._init_obj_qpos.tolist() if self._obj_name else HAND_HOME_QPOS
        )

    def _get_mug_position(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Position of the mug in world frame."""
        return self.get_sensor_data(data, self.mug_position_sensor, batch_idx=batch_idx)

    def _get_mug_contact_with_palm(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Num of mug contacts with palm"""
        # [found: 0 or num_contacts]
        return self.get_sensor_data(data, self.mug_contact_with_palm_sensor, end=1, batch_idx=batch_idx)

    def _get_obj_contact_with_finger_tips(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        # Each return [found: 0 or num_contacts]
        return jnp.sum(jnp.array(
            [self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f], end=1, batch_idx=batch_idx)
             for f in self.FINGER_TIPS_NAMES]))

    def _get_obj_contact_force_with_finger_tips(self, data: Union[mjx.Data, mjw.Data],
                                                batch_idx: int = -1) -> jax.Array:
        return jnp.sum(jnp.square(jnp.array([self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f],
                                                                  start=1, end=4,
                                                                  batch_idx=batch_idx)
                                             for f in self.FINGER_TIPS_NAMES])))

    def _get_mug_distance_to_grasp(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Position of the mug relative to the grasp."""
        return self.get_sensor_data(data, self.mug_distance_to_grasp_sensor, batch_idx=batch_idx)

    def _get_mug_distance_to_target(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Position of the mug relative to the target."""
        return self.get_sensor_data(data, self.mug_distance_to_target_sensor, batch_idx=batch_idx)

    def _get_mug_orientation(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Orientation of the mug in world frame."""
        return self.get_sensor_data(data, self.mug_orientation_sensor, batch_idx=batch_idx)

    def _get_mug_orientation_distance_to_target(self, data: Union[mjx.Data, mjw.Data],
                                                batch_idx: int = -1) -> jax.Array:
        """Orientation of the mug relative to the target grasp orientation."""
        mug_relative_to_target_quat = self.get_sensor_data(data, self.mug_orientation_from_target_sensor,
                                                           batch_idx=batch_idx)

        # Quaternion subtraction gives us rotation relative to goal
        goal_relative_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return jnp.sum(jnp.square(mjx._src.math.quat_sub(mug_relative_to_target_quat, goal_relative_quat)))

    def _get_mug_linear_velocity(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Velocity of the mug in world."""
        return self.get_sensor_data(data, self.mug_linear_velocity_sensor, batch_idx=batch_idx)

    def _get_finger_tips_distance_to_mug(self, data: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        """Distance of the fingertips from the object."""
        return jnp.sum(
            jnp.square(jnp.array(
                [self.get_sensor_data(data, s, batch_idx=batch_idx) for s in self.finger_tip_distance_to_mug_sensors])))

    # Palm cost
    def _get_palm_cost(self, state: Union[mjx.Data, mjw.Data], encourage: bool, batch_idx: int = -1) -> jax.Array:
        return (-1 if encourage else 1) * 0.05 * self._get_mug_contact_with_palm(state, batch_idx=batch_idx)

    # Fingertips total cost
    def _get_fingertips_cost(self, state: Union[mjx.Data, mjw.Data], batch_idx: int = -1) -> jax.Array:
        # cost = 50 * self._get_finger_tips_distance_to_obj(state)
        cost = -0.05 * self._get_obj_contact_with_finger_tips(state, batch_idx=batch_idx)
        return cost

    @staticmethod
    def smooth_sigmoid(x, err, eps=0.005, steep=100.0):
        # logistic mask: ≈1 when err < eps, ≈0 otherwise
        mask = jax.nn.sigmoid((eps - err) * steep)
        return mask * (1.0 / x) ** 2

    def is_reaching(self, state: Union[mjx.Data, mjw.Data]) -> Any:
        return self.phase == RelocatePhase.REACHING and not self.is_in_object_proximity(state)

    def is_in_object_proximity(self, state: Union[mjx.Data, mjw.Data]) -> Any:
        grasp_position_err = self._get_mug_distance_to_grasp(state)
        grasp_squared_distance = jnp.sum(jnp.square(grasp_position_err[0:2]))
        # [0:2]ignore z since it can never be fully close to 0 spatially (3D)
        return grasp_squared_distance > self.grasp_threshold ** 2

    def is_relocating(self, state: Union[mjx.Data, mjw.Data]) -> Any:
        target_distance_err = self._get_mug_distance_to_target(state)
        target_squared_distance = jnp.sum(jnp.square(target_distance_err))
        return (~self.is_reaching(state)) & (target_squared_distance > self.grasp_threshold ** 2)

    def next_phase(self, state: Union[mjx.Data, mjw.Data]) -> jnp.int32:
        is_near_object = jnp.any(self.is_in_object_proximity(state))
        phase = state.userdata[0].astype(jnp.int32)
        return jax.lax.select(phase == RelocatePhase.INITIAL, RelocatePhase.REACHING,
                              jax.lax.select((phase == RelocatePhase.RELOCATING) & is_near_object,
                                             RelocatePhase.GRASPING,
                                             jax.lax.select(
                                                 (phase == RelocatePhase.GRASPING) & (~is_near_object),
                                                 RelocatePhase.RELOCATING,
                                                 jax.lax.select(is_near_object,
                                                                RelocatePhase.GRASPING,
                                                                phase))))

    @classmethod
    def get_cost(cls, cond: Callable, right_value: jnp.float32, wrong_value: jnp.float32) -> Any:
        return jax.lax.cond(
            cond(),
            lambda args: right_value,
            lambda args: wrong_value,
            right_value
        )

    def grasp_position_cost(self, data: Union[mjx.Data, mjw.Data], control: jax.Array) -> jax.Array:
        grasp_position_err = self._get_mug_distance_to_grasp(data)
        grasp_squared_distance = jnp.sum(jnp.square(grasp_position_err[0:2]))
        # [0:2]ignore z since it can never be fully close to 0 spatially (3D)
        grasp_proximity = grasp_squared_distance - self.grasp_threshold ** 2
        grasp_position_cost = 0.1 * grasp_squared_distance + 100 * jnp.maximum(grasp_proximity, 0.0)
        grasp_orientation_cost = self._get_mug_orientation_distance_to_target(data)

        k_grasp = 0.001
        grasp_control_cost = k_grasp * jnp.sum(jnp.square(control))
        return grasp_position_cost + grasp_orientation_cost + grasp_control_cost

    def bring_to_target_cost(self, data: Union[mjx.Data, mjw.Data], control: jax.Array) -> jax.Array:
        fingers_squared_distance = self._get_finger_tips_distance_to_mug(data)
        fingers_distance_cost = 10000 * fingers_squared_distance

        target_distance_err = self._get_mug_distance_to_target(data)
        target_squared_distance = jnp.sum(jnp.square(target_distance_err))
        target_proximity = target_squared_distance - self.target_distance_threshold ** 2
        target_distance_cost = 0.1 * target_squared_distance + 100 * jnp.maximum(target_proximity, 0.0)

        # smooth_sigmoid(self._get_mug_position(state)[2], squared_distance, 0.005)
        # squared_proximity_from_ground = 1.0 / jnp.square(self._get_mug_position(state)[2])
        # hover_cost = 0.1 * squared_proximity_from_ground + 100 * jnp.maximum(
        #    squared_proximity_from_ground - self.threshold ** 2, 0.0
        # )
        return target_distance_cost + fingers_distance_cost

    def running_cost(self, state: Union[mjx.Data, mjw.Data], control: jax.Array,
                     batch_idx: Optional[int] = -1) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_mug_distance_to_grasp(state, batch_idx)
        squared_distance = jnp.sum(jnp.square(position_err[0:2]))  # ignore z
        # Only highly weighed until reaching certain threshold, from which prioritize other costs (orientation, grasp, etc.)
        reaching_cost = 100 * jnp.maximum(
            squared_distance - self.grasp_threshold ** 2, 0.0
        )
        position_cost = 0.1 * squared_distance + reaching_cost
        orientation_cost = 50 * self._get_mug_orientation_distance_to_target(state, batch_idx)

        grasp_cost = 0.001 * jnp.sum(jnp.square(control)) + self._get_fingertips_cost(state, batch_idx)
        return position_cost + orientation_cost + grasp_cost

    def running_relocate_cost(self, data: Union[mjx.Data, mjw.Data], control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        phase = data.userdata[0].astype(jnp.int32)
        return jax.lax.select((phase == RelocatePhase.REACHING) | (phase == RelocatePhase.GRASPING),
                              jnp.array([self.grasp_position_cost(data, control)], dtype=jnp.float32),
                              jax.lax.select(phase == RelocatePhase.RELOCATING,
                                             self.bring_to_target_cost(data, control),
                                             jnp.array([100000000.0], dtype=jnp.float32)))

    def terminal_cost(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_mug_distance_to_grasp(data)
        return 100 * jnp.sum(jnp.square(position_err)) + self._get_fingertips_cost(data)

    def terminal_relocate_cost(self, data: Union[mjx.Data, mjw.Data]) -> Union[jax.Array, Any]:
        """The terminal cost ϕ(x_T)."""
        phase = data.userdata[0].astype(jnp.int32)
        grasp_distance_cost = 100 * jnp.sum(jnp.square(self._get_mug_distance_to_grasp(data)))
        return jax.lax.select((phase == RelocatePhase.REACHING) | (phase == RelocatePhase.GRASPING),
                              grasp_distance_cost,
                              jnp.reshape(100 * jnp.sum(jnp.square(self._get_mug_distance_to_target(
                                  data))) + 10000 * self._get_finger_tips_distance_to_mug(data),
                                          grasp_distance_cost.shape))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.mjx_model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.mjx_model.geom_friction.at[:, 0].set(
            self.mjx_model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(self, data: mjx.Data, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly shift the measured configurations."""
        shift = 0.005 * jax.random.normal(rng, (self.mjx_model.nq,))
        return {"qpos": data.qpos + shift}

    @partial(jit, static_argnums=(0,))
    def mjx_fabrics_convert_free_hand_to_full_arm_hand_ctrl(self, mjx_data: Union[mjx.Data, mjw.Data],
                                                            u: jnp.ndarray) -> jnp.ndarray:
        grasp_site_delta = jnp.zeros(7)
        grasp_site_ctrl = self.mjx_model.opt.timestep * u[:6]  # 6DOF in 3D

        # Delta grasp pose
        grasp_site_delta.at[:3].set(grasp_site_ctrl[:3])
        from brax import math
        grasp_site_delta.at[3:].set(math.euler_to_quat(grasp_site_ctrl[3:]))

        obj_pose = jnp.concatenate([self._get_obj_position(mjx_data), self._get_obj_orientation(mjx_data)])
        # self.jax_fabrics_step(mjx_mulPose(obj_pose, grasp_site_delta))
        return jax.dlpack.from_dlpack(self.fabrics_controller.q).squeeze()
