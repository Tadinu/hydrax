# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Franka Emika Panda environment base class.
    Adapted from: https://github.com/google-deepmind/mujoco_playground
"""

from typing import Any, Dict, Optional, Union, Sequence

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
import numpy as np

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax.task_base import Task
from hydrax import ROOT


class PandaBaseEnv(mjx_env.MjxEnv, Task):
    """Base environment for Franka Emika Panda."""

    @staticmethod
    def default_config() -> config_dict.ConfigDict:
        return config_dict.create(
            ctrl_dt=0.02,
            sim_dt=0.005,
            episode_length=150,
            action_repeat=1,
            action_scale=0.04,
        )

    @staticmethod
    def get_assets() -> Dict[str, bytes]:
        assets = {}
        path = epath.Path(ROOT) / "models" / "panda"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")
        return assets

    def __init__(
            self,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            xml_path: Optional[epath.Path] = None,
            trace_sites: Optional[Sequence[str]] = None
    ):
        super().__init__(config, config_overrides)
        Task.__init__(self, trace_sites=trace_sites)

        self._mj_model: mj.MjModel = None
        self._mjx_model: mjx.Model = None
        self._xml_path: str = ""
        self._model_assets = self.get_assets()
        if xml_path is not None:
            self._xml_path = xml_path.as_posix()
            xml = xml_path.read_text()
            self._mj_model = mj.MjModel.from_xml_string(xml, assets=self._model_assets)
            self._mj_model.opt.timestep = self.sim_dt
            self._mjx_model = mjx.put_model(self._mj_model)
        self._action_scale = config.action_scale

        self.ARM_JOINTS = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.HAND_JOINTS = ["finger_joint1", "finger_joint2"]

    def _post_init(self, obj_name: Optional[str] = None, keyframe: Optional[str] = None):
        Task._post_init(self, obj_name, keyframe)

        # Robot-specifics
        all_joints = self.ARM_JOINTS + self.HAND_JOINTS
        self._robot_arm_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in self.ARM_JOINTS
        ])
        self._robot_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in all_joints
        ])
        self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

        # Hand-specifics
        self._init_hand()

        # Env-specifics
        self._obj_body = self._mj_model.body(obj_name).id
        self._obj_geom = self.mj_model.geom(obj_name).id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body(obj_name).jntadr[0]
        ]
        self._mocap_target = self._mj_model.body("mocap_target").mocapid
        self._floor_geom = self._mj_model.geom("floor").id
        self._init_q = self._mj_model.keyframe(keyframe).qpos
        self._init_obj_pos = jp.array(
            self._init_q[self._obj_qposadr: self._obj_qposadr + 3],
            dtype=jp.float32,
        )
        self._init_obj_quat = np.array(
            self._init_q[self._obj_qposadr + 3: self._obj_qposadr + 7],
            dtype=np.float32,
        )

    def _init_hand(self):
        self._grasp_site = self._mj_model.site("gripper").id
        self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
        self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
        self._hand_geom = self._mj_model.geom("hand_capsule").id
        self._hand_full_geoms = [self._left_finger_geom, self._right_finger_geom, self._hand_geom]

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    def jnt_vel_range(self):
        return [
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100],
        ]

    def ctrl_range(self):
        return [
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
