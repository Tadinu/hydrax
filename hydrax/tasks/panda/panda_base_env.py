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

import jax
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

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        path = epath.Path(ROOT) / "models" / "panda"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")
        return assets

    def __init__(self,
                 config: config_dict.ConfigDict,
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
                 xml_path: Optional[epath.Path] = None,
                 obj_name: Optional[str] = None,
                 keyframe: Optional[str] = None,
                 trace_sites: Optional[Sequence[str]] = None,
                 use_ctrl_callback: bool = False):
        super().__init__(config, config_overrides)
        if obj_name is None:
            obj_name = "cube"  # Required for creating model spec
        if keyframe is None:
            keyframe = "home"
        self.use_ctrl_callback = use_ctrl_callback
        self.ARM_JOINTS = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.HAND_JOINTS = []
        self._action_scale = config.action_scale
        Task.__init__(self, xml_path=xml_path, sim_dt=config.sim_dt,
                      obj_name=obj_name, keyframe=keyframe, trace_sites=trace_sites)

    def _post_init(self) -> None:
        # 1- Init [u_min, u_max]
        if self.use_ctrl_callback:
            self.u_min = np.concatenate([np.array([-1] * 3 + [-1.57] * 3),
                                         self.mj_model.actuator_ctrlrange[7:, 0]])
            self.u_max = np.concatenate([np.array([1] * 3 + [1.57] * 3),
                                         self.mj_model.actuator_ctrlrange[7:, 1]])
            self.num_ctrls = self.u_min.size

        # 2- Task's [_pos_init]
        # Init [u_min, u_max] here if not create above, so run later
        Task._post_init(self)

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
        self._init_ctrl = self._mj_model.keyframe(self._keyframe).ctrl
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

        # Hand-specifics
        self._init_hand()

        # Env-specifics
        self._obj_body = self._mj_model.body(self._obj_name).id
        self._obj_geom = self.mj_model.geom(self._obj_name).id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body(self._obj_name).jntadr[0]
        ]
        self._mocap_target = self._mj_model.body("target").mocapid
        self._floor_geom = self._mj_model.geom("floor").id
        self._init_q = self._mj_model.keyframe(self._keyframe).qpos
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

    @property  # Impl of parent abstract
    def xml_path(self) -> str:
        return self._xml_path

    @property  # Impl of parent abstract
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property  # Impl of parent abstract
    def mj_data(self) -> mj.MjData:
        return self._mj_data

    @property  # Impl of parent abstract
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property  # Impl of parent abstract
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
        if False:
            """Add noise to the state estimate."""
            rng, q_rng, v_rng = jax.random.split(rng, 3)
            q_err = 0.01 * jax.random.normal(q_rng, (self.mjx_model.nq,))
            v_err = 0.01 * jax.random.normal(v_rng, (self.mjx_model.nv,))
            return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}
        else:
            shift = 0.005 * jax.random.normal(rng, (self.mjx_model.nq,))
            return {"qpos": data.qpos + shift}
