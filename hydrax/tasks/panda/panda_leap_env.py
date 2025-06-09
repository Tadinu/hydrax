from typing import Any, Dict, Optional, Union
from pathlib import Path
from etils import epath
from ml_collections import config_dict
import numpy as np

import mujoco as mj
from mujoco import mjx

import mink
from mjmanip.utils import mj_add_mocap_body, mj_set_body_tree_collision_enabled

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT
from hydrax.tasks.panda.panda_base_env import PandaBaseEnv

_HERE = Path(__file__).parent

IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
ZERO_XYZ = np.zeros(3)


class PandaLeap:
    # panda
    ARM_HOME_QPOS = [0, 0.3, 0, -1.57079, 0, 2.0, -0.7853]
    ARM_HAND_ATTACHMENT_SITE_NAME = "attachment_site"
    ARM_DOFS_NO = len(ARM_HOME_QPOS)
    ARM_BODIES_NAMES = []
    ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

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
    HAND_JOINTS = [
        "if_mcp", "if_rot", "if_pip", "if_dip",
        "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
        "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
        "th_cmc", "th_axl", "th_mcp", "th_ipl"
    ]

    HAND_GEOMS = ["leap_mount_collision_0", "leap_mount_collision_1"]
    HAND_GEOMS += [f"palm_collision_{i}" for i in range(1, 11)]

    # ee target
    EE_TARGET_MOCAP_NAME: str = "ee_target"

    # fingers
    FINGER_GEOMS = [
        "if_bs_collision_1",
        "if_px_collision",
        "if_md_collision_1",
        "if_md_collision_5",
        "if_ds_collision_1",
        "mf_bs_collision_1",
        "mf_px_collision",
        "mf_md_collision_1",
        "mf_md_collision_5",
        "mf_ds_collision_1",
        "rf_bs_collision_1",
        "rf_px_collision",
        "rf_md_collision_1",
        "rf_md_collision_5",
        "rf_ds_collision_1",
        "th_mp_collision",
        # "th_bs_collision_1",
        "th_px_collision_1",
        "th_ds_collision_1",
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
        cls.HAND_JOINTS = [f"{prefix}{_}" for _ in cls.HAND_JOINTS]
        cls.HAND_GEOMS = [f"{prefix}{_}" for _ in cls.HAND_GEOMS]
        cls.FINGER_GEOMS = [f"{prefix}{_}" for _ in cls.FINGER_GEOMS]

    @classmethod
    def disable_arm_hand_collision(cls, spec: mj.MjSpec) -> None:
        for arm_body in cls.ARM_BODIES_NAMES:
            if not arm_body.endswith("world"):
                for hand_body in cls.HAND_BODIES_NAMES:
                    if not hand_body.endswith("world"):
                        spec.add_exclude(bodyname1=arm_body, bodyname2=hand_body)

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
        max_velocities = {joint: np.pi for joint in self.ARM_JOINTS}

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


_ARM_DIR = "panda"
_GRIPPER_DIR = "leap_hand"


class PandaLeapEnv(PandaBaseEnv):
    """Base environment for Franka Emika Panda and Leap hand."""

    @staticmethod
    def get_assets() -> Dict[str, bytes]:
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
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            xml_path: Optional[epath.Path] = None,
    ):
        # NOTE: Don't pass [xml_path] to [PandaBaseEnv] here, since the arm+hand model will be programmingly composed
        super().__init__(config, config_overrides)
        self.arm_xml: str = xml_path.as_posix() if xml_path \
            else ROOT + "/models/panda/mjx_panda_nohand.xml"
        self.hand_xml: str = ROOT + "/models/cube/leap_rh.xml"
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None
        self.hand_base_spec: mj.MjsBody = None

        # Target obj
        self.obj_name_: str = "box"
        self.obj_mesh_name_: str = "mj_mug.obj"
        # !NOTE: This is heavy -> unlikely to be runnable by mjx
        self.obj_pointcloud_name_: str = None
        rand_seed = 1
        np.random.seed(100 + rand_seed)
        init_obj_quat = np.zeros(4)
        init_obj_yaw = np.pi * np.random.rand(1) - np.pi / 2
        mj.mju_axisAngle2Quat(init_obj_quat, [0.0, 0.0, 1.0], init_obj_yaw)
        self.init_obj_pose_ = np.hstack((np.array([0.5, 0.5, 0.05]), init_obj_quat))
        init_obj_xy = np.array(
            [self.init_obj_pose_[0], self.init_obj_pose_[1]]
        ) + 0.005 * np.random.randn(2)
        init_obj_pos = np.hstack([init_obj_xy, self.init_obj_pose_[2]])
        self.init_obj_qpos_ = np.hstack((init_obj_pos, self.init_obj_pose_[3:]))

        # Goal
        self.goal_p_ = np.array(
            [init_obj_xy[0] + 0.01, init_obj_xy[1] - 0.01, self.init_obj_pose_[2]]
        )
        self.goal_q_ = np.zeros(4)
        mj.mju_axisAngle2Quat(
            self.goal_q_, [0.0, 0.0, 1.0], np.pi * np.random.rand(1) - np.pi / 2
        )

        self.HOME_QPOS = (
            PandaLeap.HOME_QPOS + self.init_obj_qpos_.tolist()
            if self.obj_name_
            else PandaLeap.HOME_QPOS
        )

        # Construct model
        self._mj_model = self._construct_system_model()
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        # Joint names
        self.ARM_JOINTS = PandaLeap.ARM_JOINTS
        self.HAND_JOINTS = PandaLeap.HAND_JOINTS

        # Trace sites
        self.trace_sites = ["box"] + [
            PandaLeap.hand_item_full_name(site)
            for site in ["if_tip", "mf_tip", "rf_tip", "th_tip"]
        ]

    def has_objs(self) -> bool:
        return self.obj_name_ is not None

    def delete_specs(self) -> None:
        """Delete specs, eg. due to not being `picklable` by multiprocess"""
        del self.hand_base_spec
        del self.arm_spec
        del self.hand_spec

    def _construct_system_model(self) -> Optional[mj.MjModel]:
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        # https://mj.readthedocs.io/en/latest/python.html#construction
        self.arm_spec = mj.MjSpec.from_file(self.arm_xml)
        system_worldbody = self.arm_spec.worldbody
        print("SYSTEM MODEL NAME: ", self.arm_spec.modelname)
        PandaLeap.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]
        # Disable arm's bodies collision
        mj_set_body_tree_collision_enabled(self.arm_spec.bodies[1], False)

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
        PandaLeap.update_item_names_with_prefix()

        # Refetch [self.hand_base_spec], which seems to be just the same after attachment, in [self.arm_spec]
        self.hand_base_spec = self.arm_spec.body(PandaLeap.HAND_BASE_NAME)

        # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
        self.arm_spec.add_key(name="home", qpos=self.HOME_QPOS)
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
            self.arm_spec,
            self.hand_base_spec,
            PandaLeap.EE_TARGET_MOCAP_NAME,
            mocap_geom_type=mj.mjtGeom.mjGEOM_BOX,
            mocap_size=np.array([0.03] * 3),
            rgba=[1, 0, 1, 1]
        )

        # Enabled [gravcomp]
        for arm_body in self.arm_spec.bodies:
            arm_body.gravcomp = 1

        # Add finger mocaps
        for fingertip in PandaLeap.FINGER_TIPS:
            mj_add_mocap_body(
                self.arm_spec,
                self.hand_base_spec,
                f"{fingertip}_target",
                mocap_geom_type=mj.mjtGeom.mjGEOM_SPHERE,
                mocap_size=np.array([0.02] * 3),
                rgba=PandaLeap.FINGER_COLORS[fingertip],
            )

        # Add contact excludes
        PandaLeap.disable_arm_hand_collision(self.arm_spec)

        # Compile [arm_spec] -> model
        self._mjx_model = self.arm_spec.compile()
        return self._mjx_model

    def _post_init(self, obj_name: Optional[str] = None, keyframe: Optional[str] = None):
        super()._post_init(obj_name, keyframe)

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

    def _init_hand(self):
        self._grasp_site = self.mj_model.site(
            PandaLeap.hand_item_full_name("grasp_site")
        ).id
        self._hand_geoms = [self.mj_model.geom(n).id for n in PandaLeap.HAND_GEOMS]
        self._finger_geoms = [self.mj_model.geom(n).id for n in PandaLeap.FINGER_GEOMS]
        self._hand_full_geoms = self._hand_geoms + self._finger_geoms

    def _free_joint_name(self, body_name: str):
        return f"{body_name}_freejoint"
