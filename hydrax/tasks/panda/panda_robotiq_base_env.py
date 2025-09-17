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
"""Franka Emika Panda base class.
Adapted from: https://github.com/google-deepmind/mujoco_playground
"""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
import numpy as np

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT
from hydrax.tasks.panda.panda_base_env import PandaBaseEnv

GRIPPER_GEOMS = [
    "left_coupler_col_1",
    "left_coupler_col_2",
    "left_follower_pad2",
    "right_coupler_col_1",
    "right_coupler_col_2",
    "right_follower_pad2",
]
GEAR = np.array([150.0, 150.0, 150.0, 150.0, 20.0, 20.0, 20.0])
_ARM_DIR = "panda"
_GRIPPER_DIR = "robotiq_2f85_v4"


class PandaRobotiqBaseEnv(PandaBaseEnv):
    """Base environment for Franka Emika Panda and Robotiq gripper."""

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
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            xml_path: Optional[epath.Path] = None
    ):
        super().__init__(config, config_overrides, xml_path)
        self.ARM_JOINTS = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.HAND_JOINTS = ["right_driver_joint", "left_driver_joint"]

    def _post_init(self, obj_name: Optional[str] = None, keyframe: Optional[str] = None):
        super()._post_init(obj_name, keyframe)
        # Robot-specifics
        self._q_low_joint_pos_index = 0
        self._q_upper_joint_pos_index = 7
        self._qd_low_joint_pos_index = 0
        self._qd_upper_joint_pos_index = 7
        self._gear = GEAR
        self._joint_limit_percentage = 0.9
        self._joint_vel_limit_percentage = 0.9
        self._jnt_range = self._mj_model.jnt_range
        self._jnt_vel_range = np.array(self.jnt_vel_range())
        self._joint_range_init_percent_limit = np.array(
            [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
        )
        self._max_torque = 8.0

    def _init_hand(self):
        super()._init_hand()
        self._grasp_geoms = [self.mj_model.geom(n).id for n in GRIPPER_GEOMS]

    @property
    def xml_path(self) -> str:
        raise self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu
