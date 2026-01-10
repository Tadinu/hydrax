from typing import Any, Dict, Optional, Union
from pathlib import Path
from functools import partial
from etils import epath
from ml_collections import config_dict
import numpy as np
import warp as wp

# jax
import jax
from jax import jit
import jax.numpy as jnp

# mujoco playground
import mujoco as mj
from mujoco import mjx
import mujoco_warp as mjw
from mujoco_playground._src import mjx_env

import mink
from mjmanip.utils import mj_add_mocap_body, mj_get_joints_qids, mj_get_mocap_pose, mj_set_body_tree_collision_enabled
from mjmanip.mjx_utils import mjx_mulPose
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig
from mjmanip.control.fabrics.fabrics_controller import FabricsController
from mjmanip.robot.panda_leap_fabrics import (PANDA_LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES,
                                              PANDA_LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES)

# hydrax
from hydrax import ROOT, BackendType, MODELS_DIR
from hydrax.tasks.panda.panda_base_task import PandaBaseEnv

_HERE = Path(__file__).parent

IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
ZERO_XYZ = np.zeros(3)
IDENTITY_POSE = np.concatenate([ZERO_XYZ, IDENTITY_WXYZ])


class PandaLeap:
    PANDA_LEAP_MODEL_DIR = f"{MODELS_DIR}/panda_leap"
    PANDA_LEAP_ASSETS_DIR = f"{PANDA_LEAP_MODEL_DIR}/assets"

    BASE_POSE = IDENTITY_POSE
    VISUAL_CLASS_NAMES: list[str] = []
    COLLISION_CLASS_NAMES: list[str] = []

    # panda
    ARM_HOME_QPOS = [0, 0.3, 0, -1.57079, 0, 2.0, -0.7853]
    ARM_HAND_ATTACHMENT_SITE_NAME = "attachment_site"
    ARM_DOFS_NO = len(ARM_HOME_QPOS)
    ARM_BODIES_NAMES = []
    ARM_JOINTS_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    ARM_GEOMS = ["link0_c", "link1_c", "link2_c", "link3_c", "link4_c", "link5_c0", "link5_c1", "link5_c2", "link6_c",
                 "link7_c"]

    # Get Fabrics joints info from FabricCSpaceInfo loaded from model description
    ARM_FABRICS_JOINTS_NAMES = [
        'link0_joint',
        'joint1',
        'joint2',
        'link2_fabric1_joint',  # Fabric-Fixed
        'link2_fabric2_joint',  # Fabric-Fixed
        'joint3',
        'link3_fabric_joint',  # Fabric-Fixed
        'joint4',
        'link4_fabric1_joint',  # Fabric-Fixed
        'link4_fabric2_joint',  # Fabric-Fixed
        'joint5',
        'link5_fabric1_joint',  # Fabric-Fixed
        'link5_fabric2_joint',  # Fabric-Fixed
        'joint6',
        'link6_fabric_joint',  # Fabric-Fixed
        'joint7',
        'link7_fabric_joint',  # Fabric-Fixed
    ]

    # leap hand
    # To be determined upon reading from xml in system construction
    HAND_MODEL_NAME = ""
    HAND_BASE_NAME = "leap_mount"
    HAND_HOME_QPOS = [
        0.8, 0, 0.8, 0.8,
        0.8, 0, 0.8, 0.8,
        0.8, 0, 0.8, 0.8,
        0.8, 0.8, 0.8, 0,
    ]
    HAND_DOFS_NO = len(HAND_HOME_QPOS)
    HAND_BASE_POSE = [np.array([0.0, 0.0, 0.13]), np.array([0.707, 0, -0.707, 0])]
    HAND_BODIES_NAMES = []
    HAND_JOINTS_NAMES = [
        "if_mcp", "if_rot", "if_pip", "if_dip",
        "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
        "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
        "th_cmc", "th_axl", "th_mcp", "th_ipl"
    ]

    HAND_FABRICS_BODIES_NAMES = HAND_BODIES_NAMES
    HAND_FABRICS_JOINTS_NAMES = [
        # LEAP-MOUNT
        f'{HAND_BASE_NAME}_joint',  # Fixed joint
        f'{HAND_BASE_NAME}_fabric_joint',  # Fixed joint

        # PALM
        'palm_joint',  # Fixed joint
        'palm_fabric_joint',  # Fabric-Fixed

        # INDEX
        'if_bs_fabric_joint',  # Fabric-Fixed
        'if_mcp',
        'if_rot',
        'if_px_fabric1_joint',  # Fabric-Fixed
        'if_px_fabric2_joint',  # Fabric-Fixed
        'if_pip',
        'if_dip',
        'if_ds_fabric1_joint',  # Fabric-Fixed
        'if_ds_fabric2_joint',  # Fabric-Fixed

        # MIDDLE
        'mf_bs_fabric_joint',  # Fabric-Fixed
        'mf_mcp',
        'mf_rot',
        'mf_px_fabric1_joint',  # Fabric-Fixed
        'mf_px_fabric2_joint',  # Fabric-Fixed
        'mf_pip',
        'mf_dip',
        'mf_ds_fabric1_joint',  # Fabric-Fixed
        'mf_ds_fabric2_joint',  # Fabric-Fixed

        # RING
        'rf_bs_fabric_joint',  # Fabric-Fixed
        'rf_mcp',
        'rf_rot',
        'rf_px_fabric1_joint',  # Fabric-Fixed
        'rf_px_fabric2_joint',  # Fabric-Fixed
        'rf_pip',
        'rf_dip',
        'rf_ds_fabric1_joint',  # Fabric-Fixed
        'rf_ds_fabric2_joint',  # Fabric-Fixed

        # THUMB
        'th_cmc',
        'th_axl',
        'th_mcp',
        'th_px_fabric_joint',  # Fabric-Fixed
        'th_ipl',
        'th_ds_fabric1_joint',  # Fabric-Fixed
        'th_ds_fabric2_joint',  # Fabric-Fixed

        # PALM-XYZ
        'palm_x_joint',  # Fabric-Fixed
        'palm_x_neg_joint',  # Fabric-Fixed
        'palm_y_joint',  # Fabric-Fixed
        'palm_y_neg_joint',  # Fabric-Fixed
        'palm_z_joint',  # Fabric-Fixed
        'palm_z_neg_joint',  # Fabric-Fixed
    ]

    HAND_GEOMS = ["leap_mount_collision_0", "leap_mount_collision_1"]
    HAND_GEOMS += [f"palm_collision_{i}" for i in range(1, 11)]

    # ee target
    EE_TARGET_MOCAP_NAME: str = "ee_target"

    # fingers
    FINGER_GEOMS = [
        "if_bs_collision",
        "if_px_collision",
        "if_md_collision_1",
        "if_md_collision_2",
        "if_ds_collision",
        "mf_bs_collision",
        "mf_px_collision",
        "mf_md_collision_1",
        "mf_md_collision_2",
        "mf_ds_collision",
        "rf_bs_collision",
        "rf_px_collision",
        "rf_md_collision_1",
        "rf_md_collision_2",
        "rf_ds_collision",
        "th_mp_collision",
        # "th_bs_collision",
        "th_px_collision",
        "th_ds_collision",
    ]

    # fingertips
    FINGER_TIPS: list[str] = ["rf_tip", "mf_tip", "if_tip", "th_tip"]
    FINGER_COLORS: dict[str, list[float]] = {
        FINGER_TIPS[0]: [0.9, 0, 0, 1],  # Red
        FINGER_TIPS[1]: [0, 0.9, 0, 1],  # Green
        FINGER_TIPS[2]: [0, 0, 0.9, 1],  # Blue
        FINGER_TIPS[3]: [0.9, 0.9, 0.9, 1],  # White
    }

    # Home qpos
    HOME_QPOS = ARM_HOME_QPOS + HAND_HOME_QPOS

    # Actuators
    ACTUATED_JOINTS_NO = len(HOME_QPOS)
    ACTS_NO = ACTUATED_JOINTS_NO

    # OBJECTS
    OBJECT_NAMES: list[str] = ['cube']  # ['mug']
    OBJECT_MODEL_PATHS: dict[str, str] = {
        # 'cube': f"{MODELS_DIR}/cube/reorientation_cube.xml",
        # 'mug': f"{MODELS_DIR}/objects/mug/mug.xml",
    }
    OBJECT_INIT_POSES: dict[str, np.ndarray] = {
        obj_name: np.hstack([np.array([0, 0.5, 0.01]), IDENTITY_WXYZ])
        for obj_name in OBJECT_NAMES
    }
    OBJECT_COLLISION_MESH_NAMES: dict[str, list[str]] = {
        OBJECT_NAMES[0]: []
    }

    def __init__(self):
        # Received from a client of this class, which is expected to have its own custom
        # [MjSpec] setting before compiling to [MjModel]
        self.mj_model: mj.MjModel = None

        # Created in setup given an already compiled [MjModel]
        self.data: mj.MjData = None

        # DiffIK tasks
        self.ee_task: mink.FrameTask = None
        self.posture_task: mink.PostureTask = None
        self.T_ee_prev: mink.SE3 = None
        self.N_DOFS = PandaLeap.ARM_DOFS_NO + PandaLeap.HAND_DOFS_NO

        # 1- EE (hand base)
        # mujoco::python::MjDataBodyViews -> NOTE: This holds runtime data, NOT [MjModelBodyViews]
        self.hand_base = None

        # 2- Targets
        self.targets_frame = 0
        # 2.1- EE target
        self.EE_TARGET_CENTER_DEFAULT = np.array([0.5, 0, 0.5])
        self.EE_TARGET_QUAT_DEFAULT = np.array([0, 1, 0, 0])
        self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT = 0.1

    @classmethod
    def attach_prefix(cls) -> str:
        return f"{cls.HAND_MODEL_NAME}/"

    @classmethod
    def hand_item_full_name(cls, hand_item_name: str) -> str:
        return f"{cls.attach_prefix()}{hand_item_name}"

    @classmethod
    def update_item_names_with_prefix(cls) -> None:
        prefix = cls.attach_prefix()
        cls.HAND_BASE_NAME = f"{prefix}{cls.HAND_BASE_NAME}"
        cls.HAND_BODIES_NAMES = [f"{prefix}{_}" for _ in cls.HAND_BODIES_NAMES]
        cls.HAND_JOINTS_NAMES = [f"{prefix}{_}" for _ in cls.HAND_JOINTS_NAMES]
        cls.HAND_GEOMS = [f"{prefix}{_}" for _ in cls.HAND_GEOMS]
        cls.FINGER_GEOMS = [f"{prefix}{_}" for _ in cls.FINGER_GEOMS]

    @classmethod
    def disable_arm_hand_collision(cls, spec: mj.MjSpec) -> None:
        for arm_body in cls.ARM_BODIES_NAMES:
            if not arm_body.endswith("world"):
                for hand_body in cls.HAND_BODIES_NAMES:
                    if not hand_body.endswith("world"):
                        spec.add_exclude(bodyname1=arm_body, bodyname2=hand_body)

    @classmethod
    def goal_name(cls, obj_name: str) -> str:
        return f"{obj_name}_goal"

    def setup(self, model: mj.MjModel, data: mj.MjData) -> None:
        self.mj_model = model
        self.data = data

        # 1- Create configs with tasks (frame, posture, etc.) & limits (collision, velocities, etc.)
        self._configure()

        # 2- EE (hand base)
        self.hand_base = self.data.body(self.HAND_BASE_NAME)

        # 3- Init targets for ee, fingertips, etc.
        self._init_targets()

    def _configure(self) -> None:
        # Robot kinematics
        self.configuration = mink.Configuration(model=self.mj_model, data=self.data)
        if self.HOME_QPOS:
            self.data.qpos[: len(self.HOME_QPOS)] = self.HOME_QPOS

        # Tasks
        self._config_tasks()

        # Limits (position/velocity, joints, collision, etc.)
        self._config_limits()

    def _config_tasks(self) -> None:
        # EE task
        self.ee_task = mink.FrameTask(
            frame_name=self.ARM_HAND_ATTACHMENT_SITE_NAME,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Posture task
        self.posture_task = mink.PostureTask(model=self.mj_model, cost=5e-2)
        self.tasks = [self.ee_task, self.posture_task]

    def _config_limits(self) -> None:
        # Enable collision avoidance between the following geoms
        collision_pairs = [
            # (["wrist_3_link"], ["floor", "wall"]),
            (["link3"], ["floor"]),
        ]
        # Max velocities
        max_velocities = {joint: np.pi for joint in self.ARM_JOINTS_NAMES}

        self.limits = [
            mink.ConfigurationLimit(model=self.mj_model),
            mink.CollisionAvoidanceLimit(
                model=self.mj_model, geom_pairs=collision_pairs
            ),
            mink.VelocityLimit(self.mj_model, max_velocities),
        ]

    def update_tasks(self) -> None:
        self._update_task_ee()

    def _update_task_ee(self) -> None:
        # Update kuka end-effector task, as [target]'s SE3
        T_wt = mink.SE3.from_mocap_name(
            self.mj_model, self.data, self.EE_TARGET_MOCAP_NAME
        )
        self.ee_task.set_target(T_wt)

    def _init_targets(self) -> None:
        # Init targets (ee_target + finger_targets)
        mink.utils.move_mocap_to_pose(
            self.mj_model,
            self.data,
            self.EE_TARGET_MOCAP_NAME,
            frame_pos=self.hand_base.xpos,
            frame_quat=self.hand_base.xquat,
        )
        self.T_ee_prev = self.configuration.get_transform_frame_to_world(
            self.hand_base.name, "body"
        )

    def update_targets(self) -> None:
        self.targets_frame += 1
        # Robot's [ee_target]
        delta = self.targets_frame / 360 * np.pi
        target_pos = (
                self.EE_TARGET_CENTER_DEFAULT
                + np.array([np.cos(delta), np.sin(delta), 0])
                * self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT
        )
        mink.utils.move_mocap_to_pose(
            self.mj_model,
            self.data,
            self.EE_TARGET_MOCAP_NAME,
            frame_pos=target_pos,
            frame_quat=self.EE_TARGET_QUAT_DEFAULT,
        )

    def get_joint_positions(self):
        return self.main_data.qpos.copy()

    def get_ee_target_mocap_pose(self) -> np.ndarray:
        ee_target_mocap_pose = mj_get_mocap_pose(self.main_data, self.EE_TARGET_MOCAP_NAME)
        ee_target_mocap_pose = np.concatenate([ee_target_mocap_pose[0], ee_target_mocap_pose[1]])
        # NOTE: This is unclear why incorrect (at least during the first steps after model building)!
        # bku_ee_target_mocap_pose = np.concatenate([self.ee_target_mocap.xpos, self.ee_target_mocap.xquat])
        return ee_target_mocap_pose


_ARM_DIR = "panda"
_GRIPPER_DIR = "leap_hand"


class PandaLeapEnv(PandaBaseEnv):
    """Base environment for Franka Emika Panda and Leap hand."""

    PALM_FABRIC_CONTROL_FRAMES = PANDA_LEAP_FABRIC_PALM_CONTROL_FRAME_NAMES
    FINGER_FABRIC_CONTROL_FRAMES = PANDA_LEAP_FABRIC_FINGER_CONTROL_FRAME_NAMES

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        models_path = epath.Path(ROOT) / "models"
        path = models_path / _ARM_DIR
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")
        path = models_path / _GRIPPER_DIR
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")
        return assets

    def __init__(
            self,
            name: str,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            xml_path: Optional[epath.Path] = None,
            obj_name: Optional[str] = None,
            keyframe: Optional[str] = None,
            fabric_cfg: Optional[ArmHandPoseFabricConfig] = None,
            use_ctrl_callback: bool = False,
            backend_type: BackendType = BackendType.MJX
    ):
        self.arm_xml: str = xml_path.as_posix() if xml_path \
            else MODELS_DIR + ("/panda/mjx_panda_leap_single_obj_fabric.xml" if fabric_cfg
                               else "/panda/mjx_panda_leap_single_cube.xml")
        self.HAND_MODEL_NAME = PandaLeap.HAND_MODEL_NAME = "leap_rh_mjx_fabric" if fabric_cfg else "leap_rh_mjx"
        self.hand_xml: str = MODELS_DIR + ("/leap_hand/leap_rh_mjx_fabric.xml" if fabric_cfg \
                                               else "/leap_hand/leap_rh_mjx.xml")
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None
        self.hand_base_spec: mj.MjsBody = None

        self.ARM_JOINTS_NAMES = PandaLeap.ARM_JOINTS_NAMES
        self.HAND_JOINTS_NAMES = [f"{PandaLeap.HAND_MODEL_NAME}/{joint}" for joint in PandaLeap.HAND_JOINTS_NAMES
                                  if not joint.startswith(PandaLeap.HAND_MODEL_NAME)]
        PandaLeap.JOINTS_NAMES = self.ARM_JOINTS_NAMES + self.HAND_JOINTS_NAMES

        self.FINGER_TIPS_NAMES = [f"{PandaLeap.HAND_MODEL_NAME}/if_tip", f"{PandaLeap.HAND_MODEL_NAME}/mf_tip",
                                  f"{PandaLeap.HAND_MODEL_NAME}/th_tip"]  # f"{PandaLeap.HAND_MODEL_NAME}/rf_tip",

        self.FINGER_IF_PALMS_NAMES = [  # f"{PandaLeap.HAND_MODEL_NAME}/if_bs",
            # f"{PandaLeap.HAND_MODEL_NAME}/if_px",
            f"{PandaLeap.HAND_MODEL_NAME}/if_md",
            f"{PandaLeap.HAND_MODEL_NAME}/if_ds"]

        self.FINGER_MF_PALMS_NAMES = [  # f"{PandaLeap.HAND_MODEL_NAME}/mf_bs",
            # f"{PandaLeap.HAND_MODEL_NAME}/mf_px",
            f"{PandaLeap.HAND_MODEL_NAME}/mf_md",
            f"{PandaLeap.HAND_MODEL_NAME}/mf_ds"]

        self.FINGER_RF_PALMS_NAMES = [  # f"{PandaLeap.HAND_MODEL_NAME}/rf_bs",
            # f"{PandaLeap.HAND_MODEL_NAME}/rf_px",
            f"{PandaLeap.HAND_MODEL_NAME}/rf_md",
            f"{PandaLeap.HAND_MODEL_NAME}/rf_ds"]

        self.FINGER_TH_PALMS_NAMES = [  # f"{PandaLeap.HAND_MODEL_NAME}/th_mp",
            # f"{PandaLeap.HAND_MODEL_NAME}/th_px",
            f"{PandaLeap.HAND_MODEL_NAME}/th_ds"]

        self.FINGER_PALMS_NAMES = (self.FINGER_IF_PALMS_NAMES + self.FINGER_MF_PALMS_NAMES +
                                   self.FINGER_RF_PALMS_NAMES + self.FINGER_TH_PALMS_NAMES)

        self.HAND_HOME_QPOS = [-0.0694123, 0.0551428, 0.986832, 0.671424,
                               -0.186261, -0.0866821, 1.01374, 0.728192,
                               -0.218949, -0.0318307, 1.25156, 0.840648,
                               1.0593, 0.638801, 0.391599, 0.57284]

        # Fabrics
        self.fabrics_controller: FabricsController = None
        self.fabric_cfg: ArmHandPoseFabricConfig = fabric_cfg

        # FABRICS
        FULL_HAND_FABRICS_JOINTS_NAMES = [f"{PandaLeap.HAND_MODEL_NAME}/{_}" for _ in
                                          PandaLeap.HAND_FABRICS_JOINTS_NAMES]
        FULL_HAND_FABRICS_BODIES_NAMES = [f"{PandaLeap.HAND_MODEL_NAME}/{_}" for _ in
                                          PandaLeap.HAND_BODIES_NAMES]
        PandaLeap.FABRICS_JOINTS_NAMES = PandaLeap.ARM_FABRICS_JOINTS_NAMES + FULL_HAND_FABRICS_JOINTS_NAMES
        PandaLeap.FABRICS_BODIES_NAMES = PandaLeap.ARM_BODIES_NAMES + FULL_HAND_FABRICS_BODIES_NAMES

        # NOTE: Don't pass [xml_path] to [PandaBaseEnv] here, since the arm+hand model will be programmingly composed
        # -> [self._construct_system_model()] invoked here-in!
        super().__init__(name, config, config_overrides,
                         obj_name=obj_name, keyframe=keyframe,
                         use_ctrl_callback=use_ctrl_callback,
                         backend_type=backend_type)

        self._init_sensors()

    def _init_sensors(self):
        self.obj_position_sensor = self.get_sensor_id(f"{self._obj_name}_position")
        self.obj_orientation_sensor = self.get_sensor_id(f"{self._obj_name}_orientation")

    def _init_trace_sites(self):
        self.trace_sites += [
            PandaLeap.hand_item_full_name(site)
            for site in ["if_tip", "mf_tip", "rf_tip", "th_tip", "grasp_site"]
        ]
        super()._init_trace_sites()

    @property
    def home_qpos(self):
        return (
            PandaLeap.HOME_QPOS + self._init_obj_qpos.tolist() if self._obj_name else PandaLeap.HOME_QPOS
        )

    def _init_objects(self):
        self._obj_mesh_name: str = "mj_mug.obj"
        # !NOTE: This is heavy -> unlikely to be runnable by mjx
        self._obj_pointcloud_name: str = None

        # Obj init pose
        rand_seed = 1
        np.random.seed(100 + rand_seed)
        init_obj_quat = np.zeros(4)
        init_obj_yaw = (np.pi * np.random.rand(1) - np.pi / 2).item()
        mj.mju_axisAngle2Quat(init_obj_quat, np.array([0.0, 0.0, 1.0]), init_obj_yaw)
        self._init_obj_pose = np.hstack((np.array([0.5, 0.5, 0.05]), init_obj_quat))
        init_obj_xy = np.array(
            [self._init_obj_pose[0], self._init_obj_pose[1]]
        ) + 0.005 * np.random.randn(2)
        init_obj_pos = np.hstack([init_obj_xy, self._init_obj_pose[2]])
        self._init_obj_qpos = np.hstack((init_obj_pos, self._init_obj_pose[3:]))

        # Goal
        self._goal_p = np.array(
            [init_obj_xy[0] + 0.01, init_obj_xy[1] - 0.01, self._init_obj_pose[2]]
        )
        self._goal_q = np.zeros(4)
        mj.mju_axisAngle2Quat(
            self._goal_q, np.array([0.0, 0.0, 1.0]), (np.pi * np.random.rand(1) - np.pi / 2).item()
        )

    def _get_obj_position(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the obj in world frame."""
        return self.get_sensor_data(data, self.obj_position_sensor)

    def _get_obj_orientation(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Orientation of the obj in world frame."""
        return self.get_sensor_data(data, self.obj_orientation_sensor)

    def has_objs(self) -> bool:
        return self._obj_name is not None

    def delete_specs(self) -> None:
        """Delete specs, eg. due to not being `picklable` by multiprocess"""
        del self.hand_base_spec
        del self.arm_spec
        del self.hand_spec

    def _construct_system_model(self) -> Optional[mj.MjModel]:
        # 0- Init objects (required for creating model spec)
        self._init_objects()

        # 1- Construct model spec
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        # https://mj.readthedocs.io/en/latest/python.html#construction
        self.arm_spec = mj.MjSpec.from_file(self.arm_xml)
        self.arm_spec.meshdir = PandaLeap.PANDA_LEAP_ASSETS_DIR
        self.arm_spec.texturedir = PandaLeap.PANDA_LEAP_ASSETS_DIR
        # For storing next_phase by [SamplingBasedController._scan_fn]
        self.arm_spec.nuserdata = 1
        self.arm_spec.option.disableflags |= mj.mjtDisableBit.mjDSBL_CLAMPCTRL
        # system_worldbody = self.arm_spec.worldbody
        print("SYSTEM MODEL NAME: ", self.arm_spec.modelname)
        print("OBJECT NAME: ", self._obj_name)
        PandaLeap.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies if body.name != self._obj_name]
        # Disable arm's bodies collision
        # NOTE: This may disrupt already-setup collision from XML
        # mj_set_body_tree_collision_enabled(self.arm_spec.bodies[1], False)

        # Name arm's body geoms
        # Enabled [gravcomp]
        for arm_body in self.arm_spec.bodies:
            for geom in arm_body.geoms:
                if geom.type == mj.mjtGeom.mjGEOM_MESH and not geom.name:
                    geom.name = geom.meshname

        self.hand_spec = mj.MjSpec.from_file(self.hand_xml)
        PandaLeap.HAND_MODEL_NAME = self.hand_spec.modelname
        PandaLeap.HAND_BODIES_NAMES = [body.name for body in self.hand_spec.bodies]
        self.hand_base_spec = self.hand_spec.worldbody.find_child(
            PandaLeap.HAND_BASE_NAME
        )
        self.hand_base_spec.pos = PandaLeap.HAND_BASE_POSE[0]
        self.hand_base_spec.quat = PandaLeap.HAND_BASE_POSE[1]

        # Attach [hand_spec] to [arm_spec]
        attach_site = self.arm_spec.site(PandaLeap.ARM_HAND_ATTACHMENT_SITE_NAME)
        attach_site.attach_body(self.hand_spec.worldbody, PandaLeap.attach_prefix())

        # Update [PandaLeap] item names after attachment
        if self.fabric_cfg:
            prefix = PandaLeap.attach_prefix()
            PandaLeap.HAND_BASE_NAME = f"{prefix}{PandaLeap.HAND_BASE_NAME}"
            PandaLeap.HAND_BODIES_NAMES = [f"{prefix}{_}" for _ in PandaLeap.HAND_BODIES_NAMES]
            PandaLeap.HAND_GEOMS = [f"{prefix}{_}" for _ in PandaLeap.HAND_GEOMS]
            PandaLeap.FINGER_GEOMS = [f"{prefix}{_}" for _ in PandaLeap.FINGER_GEOMS]
        else:
            PandaLeap.update_item_names_with_prefix()

        # Refetch [self.hand_base_spec], which seems to be just the same after attachment, in [self.arm_spec]
        self.hand_base_spec = self.arm_spec.body(PandaLeap.HAND_BASE_NAME)

        self.arm_spec.add_key(name="home", qpos=self.home_qpos)
        """
        <keyframe>
            <key name="home"
              qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04 0.04 0.7 0 0.03 1 0 0 0"
              ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 0.04"/>
            <key name="pickup"
              qpos="0.2897 0.50732 -0.140016 -2.176 -0.0310497 2.51592 -0.49251 0.04 0.0399982 0.511684 0.0645413 0.0298665 0.665781 2.76848e-17 -2.27527e-17 -0.746147"
              ctrl="0.2897 0.423 -0.144392 -2.13105 -0.0291743 2.52586 -0.492492 0.04"/>
            <key name="pickup1"
              qpos='0.2897 0.496673 -0.142836 -2.14746 -0.0295746 2.52378 -0.492496 0.04 0.0399988 0.529553 0.0731702 0.0299388 0.94209 8.84613e-06 -4.97524e-06 -0.335361'
              ctrl="0.2897 0.458 -0.144392 -2.13105 -0.0291743 2.52586 -0.492492 0.04"/>
        </keyframe>
        """

        # EE Target mocap body (under [arm_spec]'s worldbody)
        mj_add_mocap_body(
            world_spec=self.arm_spec,
            # target_body_spec=self.hand_base_spec,
            mocap_name=PandaLeap.EE_TARGET_MOCAP_NAME,
            mocap_geom_type=mj.mjtGeom.mjGEOM_BOX,
            mocap_size=[0.03] * 3,
            rgba=[1, 0, 1, 1]
        )

        # Enabled [gravcomp]
        for arm_body in self.arm_spec.bodies:
            if arm_body.name != self._obj_name:
                arm_body.gravcomp = 1

        # Add finger mocaps
        for fingertip in PandaLeap.FINGER_TIPS:
            mj_add_mocap_body(
                world_spec=self.arm_spec,
                # target_body_spec=self.hand_base_spec,
                mocap_name=f"{fingertip}_target",
                mocap_geom_type=mj.mjtGeom.mjGEOM_SPHERE,
                mocap_size=[0.02] * 3,
                rgba=PandaLeap.FINGER_COLORS[fingertip],
            )

        # Add contact excludes
        PandaLeap.disable_arm_hand_collision(self.arm_spec)

        # Add sensors
        self._add_sensors(self.arm_spec)

        # Compile [arm_spec] -> model
        self._mj_model = self.arm_spec.compile()

        return self._mj_model

    def _add_sensors(self, spec: mj.MjSpec):
        # Arm
        for geom in PandaLeap.ARM_GEOMS:
            spec.add_sensor(name=f"{geom}_contact_with_floor",
                            needstage=mj.mjtStage.mjSTAGE_POS,
                            type=mj.mjtSensor.mjSENS_CONTACT,
                            datatype=mj.mjtDataType.mjDATATYPE_REAL,
                            objtype=mj.mjtObj.mjOBJ_GEOM, objname=geom,
                            reftype=mj.mjtObj.mjOBJ_GEOM, refname="floor",
                            # NOTE: Refer to mjNCONDATA for contact bits
                            intprm=[1, 1, 1])  # "found"

        # Hand
        for geom in PandaLeap.HAND_GEOMS:
            spec.add_sensor(name=f"{geom}_contact_with_floor",
                            needstage=mj.mjtStage.mjSTAGE_POS,
                            type=mj.mjtSensor.mjSENS_CONTACT,
                            datatype=mj.mjtDataType.mjDATATYPE_REAL,
                            objtype=mj.mjtObj.mjOBJ_GEOM, objname=geom,
                            reftype=mj.mjtObj.mjOBJ_GEOM, refname="floor",
                            # NOTE: Refer to mjNCONDATA for contact bits
                            intprm=[1, 1, 1])  # "found"

        # Objects
        obj_name = self._obj_name

        # Palm tactile part
        palm_name = f"{PandaLeap.HAND_MODEL_NAME}/palm"
        palm_center_name = f"{PandaLeap.HAND_MODEL_NAME}/palm_center"
        palm_body = spec.body(palm_name)
        palm_body.add_geom(type=mj.mjtGeom.mjGEOM_BOX,
                           name=palm_name,
                           size=[0.03, 0.03, 0.01],
                           rgba=[0, 1, 0, 1],
                           pos=[-0.03, -0.035, -0.025], group=1)
        palm_body.add_site(type=mj.mjtGeom.mjGEOM_SPHERE,
                           name=palm_center_name,
                           size=[0.01, 0.01, 0.01],
                           rgba=[0, 0, 1, 1],
                           pos=[-0.03, -0.035, -0.04], group=1)
        spec.add_sensor(name=f"{obj_name}_contact_with_palm",
                        needstage=mj.mjtStage.mjSTAGE_POS,
                        type=mj.mjtSensor.mjSENS_CONTACT,
                        datatype=mj.mjtDataType.mjDATATYPE_REAL,
                        objtype=mj.mjtObj.mjOBJ_GEOM, objname=obj_name,
                        reftype=mj.mjtObj.mjOBJ_GEOM, refname=palm_name,
                        # NOTE: Refer to mjNCONDATA for contact bits
                        intprm=[1 | (1 << 3), 2, 1])  # "found dist"

        spec.add_sensor(name=f"grasp_direction",
                        needstage=mj.mjtStage.mjSTAGE_POS,
                        type=mj.mjtSensor.mjSENS_FRAMEPOS,  # [mjSENS_GEOMDIST] is not supported yet by [mjx]
                        datatype=mj.mjtDataType.mjDATATYPE_REAL,
                        objtype=mj.mjtObj.mjOBJ_SITE, objname=f"{PandaLeap.HAND_MODEL_NAME}/grasp_site",
                        reftype=mj.mjtObj.mjOBJ_SITE, refname=palm_center_name)
        # Finger tactile parts
        for finger_tip in self.FINGER_TIPS_NAMES:
            spec.add_sensor(name=f"{obj_name}_contact_with_{finger_tip}",
                            needstage=mj.mjtStage.mjSTAGE_POS,
                            type=mj.mjtSensor.mjSENS_CONTACT,
                            datatype=mj.mjtDataType.mjDATATYPE_REAL,
                            objtype=mj.mjtObj.mjOBJ_GEOM, objname=obj_name,
                            reftype=mj.mjtObj.mjOBJ_GEOM, refname=finger_tip,
                            # NOTE: Refer to mjNCONDATA for contact bits
                            intprm=[1 | (1 << 3), 2, 1])  # "found dist"

        for finger_palm in self.FINGER_PALMS_NAMES:
            finger_palm_body = spec.body(finger_palm)
            finger_palm_body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE,
                                      name=finger_palm,
                                      size=[0.01, 0.01, 0.01],
                                      rgba=[0, 1, 0, 1],
                                      pos=[-0.03, -0.04, 0.] if finger_palm in self.FINGER_TH_PALMS_NAMES
                                      else [-0.01, -0.03, 0.01], group=1)
            spec.add_sensor(name=f"{obj_name}_distance_to_{finger_palm}",
                            needstage=mj.mjtStage.mjSTAGE_POS,
                            type=mj.mjtSensor.mjSENS_FRAMEPOS,  # [mjSENS_GEOMDIST] is not supported yet by [mjx]
                            datatype=mj.mjtDataType.mjDATATYPE_REAL,
                            objtype=mj.mjtObj.mjOBJ_BODY, objname=obj_name,
                            reftype=mj.mjtObj.mjOBJ_GEOM, refname=finger_palm)

            spec.add_sensor(name=f"{obj_name}_contact_with_{finger_palm}",
                            needstage=mj.mjtStage.mjSTAGE_POS,
                            type=mj.mjtSensor.mjSENS_CONTACT,
                            datatype=mj.mjtDataType.mjDATATYPE_REAL,
                            objtype=mj.mjtObj.mjOBJ_GEOM, objname=obj_name,
                            reftype=mj.mjtObj.mjOBJ_GEOM, refname=finger_palm,
                            # NOTE: Refer to mjNCONDATA for contact bits
                            intprm=[1 | (1 << 3), 2, 1])  # "found dist"

    def main_robots_system_xml(self) -> str:
        return self.arm_spec.to_xml()

    def _post_init(self) -> None:
        super()._post_init()

        # Robot-specifics
        self._q_low_joint_pos_index = 0
        self._q_upper_joint_pos_index = 7
        self._qd_low_joint_pos_index = 0
        self._qd_upper_joint_pos_index = 7
        self._joint_limit_percentage = 0.9
        self._joint_vel_limit_percentage = 0.9
        self._jnt_range = self._mj_model.jnt_range
        self._jnt_vel_range = np.array(self.jnt_vel_range())
        self._joint_range_init_percent_limit = np.array(
            [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
        )
        self._max_torque = 8.0

        # Fabrics
        if self.fabric_cfg is not None:
            self._init_fabrics()

    def _init_fabrics(self):
        self.fabrics_robot = PandaLeap()
        self.fabrics_robot.main_model = self.mj_model
        self.fabrics_robot.main_data = self.mj_data
        self.fabrics_robot.robot_qpos_ids = mj_get_joints_qids(self.mj_model, PandaLeap.JOINTS_NAMES, is_qpos=True)
        # NOTE: MjData is created here-in if needed in robot's configuration
        self.fabrics_controller = FabricsController(robot=self.fabrics_robot,
                                                    palm_control_frames=self.PALM_FABRIC_CONTROL_FRAMES,
                                                    finger_control_frames=self.FINGER_FABRIC_CONTROL_FRAMES,
                                                    fabric_cfg=self.fabric_cfg,
                                                    object_model_paths=self.fabrics_robot.OBJECT_MODEL_PATHS,
                                                    object_collision_mesh_names=self.fabrics_robot.OBJECT_COLLISION_MESH_NAMES,
                                                    robot_path_or_xml=self.main_robots_system_xml(),
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
        self._palm_center = self.mj_model.site(PandaLeap.hand_item_full_name("palm_center")).id
        self._grasp_site = self.mj_model.site(PandaLeap.hand_item_full_name("grasp_site")).id
        self._direction_grasp_site = self.mj_model.site(PandaLeap.hand_item_full_name("direction_grasp_site")).id
        self._hand_geoms = [self.mj_model.geom(n).id for n in PandaLeap.HAND_GEOMS]
        self._finger_geoms = [self.mj_model.geom(n).id for n in PandaLeap.FINGER_GEOMS]
        self._hand_full_geoms = self._hand_geoms + self._finger_geoms
        self._arm_geoms = [self.mj_model.geom(n).id for n in PandaLeap.ARM_GEOMS]
        self._palm_geoms = [self.mj_model.geom(f"{PandaLeap.HAND_MODEL_NAME}/palm").id]
        self._finger_palm_geoms = [self.mj_model.geom(n).id for n in self.FINGER_PALMS_NAMES]

    def _free_joint_name(self, body_name: str):
        return f"{body_name}_freejoint"

    @partial(jit, static_argnums=(0,))
    def mjx_integrate_pos(self, qpos: jnp.ndarray, qvel: jnp.ndarray, dt: float) -> jnp.ndarray:
        return mjx._src.forward._integrate_pos(self.mjx_model.jnt_type, qpos, qvel, dt)

    @partial(jit, static_argnums=(0,))
    def mjx_convert_free_hand_to_full_arm_hand_ctrl(self, mjx_data: Union[mjx.Data, mjw.Data],
                                                    u: jnp.ndarray) -> jnp.ndarray:
        grasp_site_ctrl = u[:6]  # 6DOF in 3D
        hand_ctrl = u[6:]
        full_ctrl = None
        if True:
            # Jac of [self._grasp_site]
            jacp, jacr = mjx.jac(self.mjx_model, mjx_data,
                                 mjx_data.site_xpos[self._grasp_site].ravel(),
                                 self.mjx_model.site_bodyid[self._grasp_site])
            J = jnp.vstack([jacp.T, jacr.T])  # (6, nv)

            # damped least-squares solve: qvel = J^T (J J^T + λ^2 I)^-1 v
            damp = 0.01  # λ
            diag = (damp ** 2) * jnp.eye(6)
            JJt = J @ J.T
            full_ctrl = J.T @ jnp.linalg.solve(JJt + diag, grasp_site_ctrl)
            full_ctrl = self.mjx_integrate_pos(mjx_data.qpos.copy(), full_ctrl.copy(), self.mj_model.opt.timestep)
            full_ctrl = full_ctrl.at[7:23].set(hand_ctrl)
            full_ctrl = full_ctrl[:23]
        else:
            hand_base_pose = f(grasp_site_ctrl)
            self.diff_ik_mjx.last_solved_q = self.diff_ik_mjx.solve(hand_base_pose)
            full_ctrl = self.diff_ik_mjx.last_solved_q
        return full_ctrl

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
        self.jax_fabrics_step(mjx_mulPose(obj_pose, grasp_site_delta))
        return jax.dlpack.from_dlpack(self.fabrics_controller.q).squeeze()
