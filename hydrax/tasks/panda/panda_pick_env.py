"""Bring a box to a target and orientation.
Adapted from: https://github.com/google-deepmind/mujoco_playground
"""

from typing import Any, Dict, Optional, Union, Tuple

from etils import epath
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from ml_collections import config_dict
import mujoco as mj
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

# mujoco playground
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

# hydrax
from hydrax.tasks.panda.panda_leap_env import PandaLeapEnv, PandaLeap
from hydrax import ROOT

# mjmanip
from mjmanip.robot.arm_hand import ArmHandDiffIK
from mjmanip.robot.arm_hand_mjx import ArmHandDiffIKMjx
from mjmanip.robot.panda_leap_mjx import PandaLeapMjx


class PandaPickEnv(PandaLeapEnv):
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
                 obj_name: Optional[str] = None,
                 keyframe: Optional[str] = None,
                 sample_orientation: bool = False,
                 use_ctrl_callback: bool = False,
                 warp_enabled: bool = False):
        if xml_path is None:
            xml_path = epath.Path(ROOT) / "models" / "panda" / "mjx_panda_leap_single_cube.xml"
        super().__init__(config, config_overrides,
                         xml_path=xml_path,
                         obj_name=obj_name,
                         keyframe=keyframe,
                         use_ctrl_callback=use_ctrl_callback,
                         warp_enabled=warp_enabled)
        self.FINGER_TIPS_NAMES = ["leap_rh/if_tip", "leap_rh/mf_tip", "leap_rh/rf_tip", "leap_rh/th_tip"]
        self._sample_orientation = sample_orientation
        self.ctrl_callback = self.mjx_convert_free_hand_to_full_arm_hand_ctrl
        self._init_ik()

        # Get sensor ids
        self.cube_position_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_contact_with_palm_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_contact_with_palm"
        )
        self.cube_distance_to_grasp_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_distance_to_grasp"
        )
        self.obj_contact_with_finger_tip_sensors = {finger_tip:
            mj.mj_name2id(
                self.mj_model, mj.mjtObj.mjOBJ_SENSOR, f"cube_contact_with_{finger_tip}",
            ) for finger_tip in self.FINGER_TIPS_NAMES
        }
        self.cube_distance_to_target_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_distance_to_target"
        )
        self.cube_orientation_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )
        self.cube_orientation_from_target_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_orientation_from_target"
        )
        self.cube_linear_velocity_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_linear_vel"
        )
        self.cube_angular_velocity_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_angular_vel"
        )
        self.finger_tip_distance_to_cube_sensors = [mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, f"{finger_tip}_position") for finger_tip in self.FINGER_TIPS_NAMES
        ]

        # Distance (m) beyond which we impose a high cube position cost
        self.grasp_threshold = 0.05  # 0.015
        self.target_distance_threshold = 0.001

    def _init_ik(self) -> None:
        PandaLeapMjx.HAND_MODEL_NAME = "leap_rh"
        PandaLeapMjx.NBATCHES = 1
        PandaLeapMjx.init_class_default()
        use_diff_ik_mjx = False
        self.diff_ik_mjx = None
        self.diff_ik = None
        if False:
            if use_diff_ik_mjx:
                self.diff_ik_mjx = ArmHandDiffIKMjx(model=self.mj_model, data=self.mj_data,
                                                    world_class=PandaLeapMjx,
                                                    q0=jnp.array(self.home_qpos),
                                                    mjx_model=self.mjx_model)
                self.diff_ik_mjx.init()
            else:
                self.diff_ik = ArmHandDiffIK(model=self.mj_model, data=self.mj_data,
                                             world_class=PandaLeapMjx,
                                             q0=PandaLeap.HOME_QPOS)
                self.diff_ik.init()

    def update_ref_qpos(self, ee_pose: Optional[Union[np.ndarray, jnp.ndarray]] = None):
        if self.diff_ik:
            if ee_pose is None:
                obj = self.mj_data.body(self._obj_name)
                DEFAULT_WXYZ = np.array([0., 1., 0., 0.])
                ee_pose = np.concatenate([obj.xpos, DEFAULT_WXYZ])
            self.ref_qpos = self.diff_ik.plan(task_ee_pose=np.asarray(ee_pose))

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
            self.mjx_model,
            init_q,
            jnp.zeros(self.mjx_model.nv, dtype=float),
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

    def step_callback(self, mjx_data: mjx.Data):
        if self.diff_ik_mjx:
            self.diff_ik_mjx.update_tasks(ee_pose=jnp.concatenate([self._get_cube_position(mjx_data),
                                                                   self._get_cube_orientation(mjx_data)]))

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

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
            self.geoms_colliding(data, self._floor_geom, g)
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

    def _get_cube_position(self, data: mjx.Data) -> jax.Array:
        """Position of the cube in world frame."""
        return self.get_sensor_data(data, self.cube_position_sensor)

    def _get_cube_contact_with_palm(self, data: mjx.Data) -> jax.Array:
        """Num of cube contacts with palm"""
        # [found: 0 or num_contacts]
        return self.get_sensor_data(data, self.cube_contact_with_palm_sensor, end=1) > 0

    def _get_obj_contact_with_finger_tips(self, data: mjx.Data) -> jax.Array:
        # Each return [found: 0 or num_contacts]
        return jnp.sum(jnp.array(
            [self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f]) for f in self.FINGER_TIPS_NAMES]))

    def _get_obj_contact_force_with_finger_tips(self, data: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(jnp.array([self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f],
                                                                  start=1, end=4) for f in self.FINGER_TIPS_NAMES])))

    def _get_cube_distance_to_grasp(self, data: mjx.Data) -> jax.Array:
        """Position of the cube relative to the grasp."""
        return self.get_sensor_data(data, self.cube_distance_to_grasp_sensor)

    def _get_cube_distance_to_target(self, data: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target."""
        return self.get_sensor_data(data, self.cube_distance_to_target_sensor)

    def _get_cube_orientation(self, data: mjx.Data) -> jax.Array:
        """Orientation of the cube in world frame."""
        return self.get_sensor_data(data, self.cube_orientation_sensor)

    def _get_cube_orientation_distance_to_target(self, data: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        cube_relative_to_target_quat = self.get_sensor_data(data, self.cube_orientation_from_target_sensor)

        # Quaternion subtraction gives us rotation relative to goal
        goal_relative_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return jnp.sum(jnp.square(mjx._src.math.quat_sub(cube_relative_to_target_quat, goal_relative_quat)))

    def _get_cube_linear_velocity(self, data: mjx.Data) -> jax.Array:
        """Velocity of the cube in world."""
        return self.get_sensor_data(data, self.cube_linear_velocity_sensor)

    def _get_finger_tips_distance_to_cube(self, data: mjx.Data) -> jax.Array:
        """Distance of the fingertips from the object."""
        return jnp.sum(
            jnp.square(jnp.array([self.get_sensor_data(data, s) for s in self.finger_tip_distance_to_cube_sensors])))

    # Arm cost
    def _get_arm_cost(self, data: mjx.Data) -> jax.Array:
        return jnp.zeros(1)  # 200 * self._get_arm_hand_contact_with_floor(data)

    def get_collision_info(self, state: mjx.Data, geom1: int, geom2: int) -> Tuple[jax.Array, jax.Array]:
        """Get the distance and normal of the collision between two geoms."""
        state = state._impl
        mask = (jnp.array([geom1, geom2]) == state.contact__geom).all(axis=1)
        mask |= (jnp.array([geom2, geom1]) == state.contact__geom).all(axis=1)
        idx = jnp.where(mask, state.contact__dist, 1e4).argmin()
        dist = state.contact__dist[idx] * mask[idx]
        normal = (dist < 0) * state.contact__frame[idx, 0, :3]
        return dist, normal

    def geoms_colliding(self, state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
        """Return True if the two geoms are colliding."""
        return self.get_collision_info(state, geom1, geom2)[0] < 0  # pylint: disable=protected-access

    def _get_contact_with_floor(self, data: mjx.Data, geoms: list[int]) -> jax.Array:
        # Check for collisions between arm with the floor
        floor_collision = [
            jnp.abs(self.get_collision_info(data, self._floor_geom, g)[0]) *
            self.geoms_colliding(data, self._floor_geom, g)
            for g in geoms
        ]
        return jnp.ones(1) * sum(floor_collision)

    def _get_arm_hand_contact_with_floor(self, data: mjx.Data) -> jax.Array:
        """Arm + Hand contact with floor."""
        # sensor_adr = self.mjx_warp_model.sensor_adr[self.wrist_contact_with_floor_sensor]
        # return data.sensordata[sensor_adr: sensor_adr + 1]  # 0 or 1
        return (self._get_contact_with_floor(data, self._arm_geoms) +
                self._get_contact_with_floor(data, self._hand_full_geoms))

    # Palm cost
    def _get_palm_cost(self, data: mjx.Data, encourage: bool) -> jax.Array:
        return (-1 if encourage else 1) * 0.05 * self._get_cube_contact_with_palm(data)

    # Fingertips total cost
    def _get_fingertips_cost(self, data: mjx.Data) -> jax.Array:
        # cost = 50 * self._get_finger_tips_distance_to_obj(data)
        cost = -0.5 * self._get_obj_contact_with_finger_tips(data)
        return cost

    # Ref traj cost
    def _get_ref_traj_cost(self, data: mjx.Data) -> jax.Array:
        if self.use_ctrl_callback:
            return jnp.zeros(1)
        else:
            return jnp.zeros(1)
            ref_qpos = jnp.array(self.ref_qpos[:7])
            square_ref_arm_pos_distance = jnp.sum(jnp.square(ref_qpos - data.qpos[:ref_qpos.size]))
            return 1000 * jnp.maximum(
                square_ref_arm_pos_distance - 1 ** 2, 0.0
            )

    def running_cost(self, data: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        ref_qpos_cost = self._get_ref_traj_cost(data)

        # Arm cost
        arm_cost = self._get_arm_cost(data)

        # Obj cost
        position_err = self._get_cube_distance_to_grasp(data)
        squared_distance = jnp.sum(jnp.square(position_err[0:2]))  # ignore z
        # Only highly weighed until reaching certain threshold, from which prioritize other costs (orientation, grasp, etc.)
        reaching_cost = 100 * jnp.maximum(
            squared_distance - self.grasp_threshold ** 2, 0.0
        )
        position_cost = 0.1 * squared_distance + reaching_cost
        orientation_cost = 50 * self._get_cube_orientation_distance_to_target(data)

        grasp_cost = 0.001 * jnp.sum(jnp.square(control)) + self._get_fingertips_cost(data)
        obj_vel_cost = 10 * jnp.sum(jnp.square(self._get_cube_linear_velocity(data)))
        return ref_qpos_cost + arm_cost + position_cost + orientation_cost + grasp_cost + obj_vel_cost

    def terminal_cost(self, data: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        ref_qpos_cost = self._get_ref_traj_cost(data)
        arm_cost = self._get_arm_cost(data)
        position_err = self._get_cube_distance_to_grasp(data)
        grasp_cost = self._get_fingertips_cost(data)
        obj_vel_cost = 10 * jnp.sum(jnp.square(self._get_cube_linear_velocity(data)))
        return ref_qpos_cost + 100 * jnp.sum(jnp.square(position_err)) + grasp_cost + obj_vel_cost

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
            data.xpos[self._obj_body] - grasp_pos,
            info["target_pos"] - data.xpos[self._obj_body],
            target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
            data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        ])
        return obs

    @partial(jit, static_argnums=(0,))
    def mjx_integrate_pos(self, qpos: jnp.ndarray, qvel: jnp.ndarray, dt: float) -> jnp.ndarray:
        return mjx._src.forward._integrate_pos(self.mjx_model.jnt_type, qpos, qvel, dt)

    @partial(jit, static_argnums=(0,))
    def mjx_convert_free_hand_to_full_arm_hand_ctrl(self, mjx_data: mjx.Data, u: jnp.ndarray) -> jnp.ndarray:
        hand_ctrl = u[6:]
        grasp_site_ctrl = u[:6]
        full_ctrl = None
        if True:
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


class PandaPickCubeOrientationEnv(PandaPickEnv):
    """Bring a box to a target and orientation."""

    def __init__(self,
                 config: config_dict.ConfigDict = PandaPickEnv.default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides, sample_orientation=True)
