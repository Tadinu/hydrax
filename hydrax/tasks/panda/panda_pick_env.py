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
"""Bring a box to a target and orientation.
Adapted from: https://github.com/google-deepmind/mujoco_playground
"""

from typing import Any, Dict, Optional, Union
from etils import epath

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

# mujoco playground
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
from mujoco_playground._src.reward import _sigmoids

# hydrax
from hydrax.tasks.panda.panda_leap_env import PandaLeapEnv
from hydrax.task_base import Task
from hydrax import ROOT


class PandaPickEnv(PandaLeapEnv, Task):
    """Bring a box to a target."""

    @staticmethod
    def default_config() -> config_dict.ConfigDict:
        """Returns the default config for bring_to_target tasks."""
        config = config_dict.create(
            ctrl_dt=0.02,
            sim_dt=0.005,
            episode_length=150,
            action_repeat=1,
            action_scale=0.04,
            reward_config=config_dict.create(
                scales=config_dict.create(
                    # Gripper goes to the box.
                    grasp_box=4.0,
                    # Box goes to the target mocap.
                    box_target=8.0,
                    # Do not collide the gripper with the floor.
                    no_floor_collision=0.25,
                    # Arm stays close to target pose.
                    robot_target_qpos=0.3,
                )
            ),
        )
        return config

    def __init__(self,
                 config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
                 xml_path: Optional[epath.Path] = None,
                 sample_orientation: bool = False):
        if xml_path is None:
            xml_path = epath.Path(ROOT) / "models" / "panda" / "mjx_panda_leap_single_cube.xml"
        super().__init__(config, config_overrides, xml_path)
        self._post_init(obj_name="box", keyframe="home")
        self._sample_orientation = sample_orientation

        # Get sensor ids
        self.cube_position_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_orientation_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )

        # Distance (m) beyond which we impose a high cube position cost
        self.delta = 0.015

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # intialize box position
        box_pos = (
                jax.random.uniform(
                    rng_box,
                    (3,),
                    minval=jnp.array([-0.2, -0.2, 0.0]),
                    maxval=jnp.array([0.2, 0.2, 0.0]),
                )
                + self._init_obj_pos
        )

        # initialize target position
        target_pos = (
                jax.random.uniform(
                    rng_target,
                    (3,),
                    minval=jnp.array([-0.2, -0.2, 0.2]),
                    maxval=jnp.array([0.2, 0.2, 0.4]),
                )
                + self._init_obj_pos
        )

        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._sample_orientation:
            # sample a random direction
            rng, rng_axis, rng_theta = jax.random.split(rng, 3)
            perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
            perturb_axis = perturb_axis / math.norm(perturb_axis)
            perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
            target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

        # initialize data
        init_q = (
            jnp.array(self._init_q)
            .at[self._obj_qposadr: self._obj_qposadr + 3]
            .set(box_pos)
        )
        data = mjx_env.init(
            self._mjx_model,
            init_q,
            jnp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
        )

        # set target mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jnp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2)
        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, state.info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jnp.any(jnp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = out_of_bounds | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = done.astype(float)

        state.metrics.update(
            **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
        )

        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        grasp_pos = data.site_xpos[self._grasp_site]
        pos_err = jnp.linalg.norm(target_pos - box_pos)
        box_mat = data.xmat[self._obj_body]
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        rot_err = jnp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

        box_target = 1 - jnp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        grasp_box = 1 - jnp.tanh(5 * jnp.linalg.norm(box_pos - grasp_pos))
        robot_target_qpos = 1 - jnp.tanh(
            jnp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Check for collisions with the floor
        hand_floor_collision = [
            collision.geoms_colliding(data, self._floor_geom, g)
            for g in self._hand_full_geoms
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = (1 - floor_collision).astype(float)

        info["reached_box"] = 1.0 * jnp.maximum(
            info["reached_box"],
            (jnp.linalg.norm(box_pos - grasp_pos) < 0.012),
        )

        rewards = {
            "grasp_box": grasp_box,
            "box_target": box_target * info["reached_box"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards

    def _get_cube_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target grasp position."""
        sensor_adr = self.mjx_model.sensor_adr[self.cube_position_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 3]

    def _get_cube_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        sensor_adr = self.mjx_model.sensor_adr[self.cube_orientation_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 4]

    def running_cost(self, data: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_cube_position_err(data)
        squared_distance = jnp.sum(jnp.square(position_err[0:2]))  # ignore z
        position_cost = 0.1 * squared_distance + 100 * jnp.maximum(
            squared_distance - self.delta ** 2, 0.0
        )

        orientation_err = self._get_cube_orientation_err(data)
        orientation_cost = jnp.sum(jnp.square(orientation_err))

        grasp_cost = 0.001 * jnp.sum(jnp.square(control))

        return position_cost + orientation_cost + grasp_cost

    def terminal_cost(self, data: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_cube_position_err(data)
        return 100 * jnp.sum(jnp.square(position_err))

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        grasp_pos = data.site_xpos[self._grasp_site]
        grasp_mat = data.site_xmat[self._grasp_site].ravel()
        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        obs = jnp.concatenate([
            data.qpos,
            data.qvel,
            grasp_pos,
            grasp_mat[3:],
            data.xmat[self._obj_body].ravel()[3:],
            data.xpos[self._obj_body] - data.site_xpos[self._grasp_site],
            info["target_pos"] - data.xpos[self._obj_body],
            target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
            data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        ])
        return obs

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.mjx_model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.mjx_model.geom_friction.at[:, 0].set(
            self.mjx_model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}


class PandaPickCubeOrientationEnv(PandaPickEnv):
    """Bring a box to a target and orientation."""

    def __init__(self,
                 config: config_dict.ConfigDict = PandaPickEnv.default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides, sample_orientation=True)
