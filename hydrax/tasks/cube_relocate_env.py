from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

HAND_BASE_POSE = [np.array([-0.25, 0.0, 0.25]), np.array([0.0, 1.0, 0.0, 0.0])]


class CubeRelocateEnv(Task):
    """Cube rotation with the LEAP hand."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mj.MjModel.from_xml_path(ROOT + "/models/leap_hand/scene_leap_rh_mjx_relocate_cube.xml")
        base_body = mj_model.body("leap_mount")
        base_body.pos = HAND_BASE_POSE[0]
        base_body.quat = HAND_BASE_POSE[1]
        super().__init__(
            mj_model,
            trace_sites=["grasp_site", "if_tip", "mf_tip", "rf_tip", "th_tip"],
        )

        # Get sensor ids
        self.cube_position_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_distance_to_grasp_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_distance_to_grasp"
        )
        self.cube_distance_from_target_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_distance_from_target"
        )
        self.cube_orientation_from_target_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "cube_orientation_from_target"
        )
        self.finger_tip_position_sensors = [mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, f"{fi}_tip_position") for fi in ["th", "if", "mf", "rf"]
        ]

        # Distance (m) beyond which we impose a high cube position cost
        self.threshold = 0.015

    def _post_init(self, obj_name: Optional[str] = None, keyframe: Optional[str] = None):
        Task._post_init(self, obj_name, keyframe)

        # Hand-specifics
        self._init_hand()

    def _init_hand(self):
        pass

    def _get_cube_position(self, state: mjx.Data) -> jax.Array:
        """Position of the cube in world frame."""
        position_adr = self.mjx_model.sensor_adr[self.cube_position_sensor]
        return state.sensordata[position_adr: position_adr + 3]

    def _get_cube_distance_from_target_sensor(self, state: mjx.Data) -> jax.Array:
        """Position of the cube relative to the target."""
        position_adr = self.mjx_model.sensor_adr[self.cube_distance_from_target_sensor]
        return state.sensordata[position_adr: position_adr + 3]

    def _get_cube_distance_from_grasp_err(self, state: mjx.Data) -> jax.Array:
        """Position of the cube relative to the grasp."""
        sensor_adr = self.mjx_model.sensor_adr[self.cube_distance_to_grasp_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 3]

    def _get_cube_orientation_from_target_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the cube relative to the target grasp orientation."""
        sensor_adr = self.mjx_model.sensor_adr[self.cube_orientation_from_target_sensor]
        cube_quat = state.sensordata[sensor_adr: sensor_adr + 4]

        # Quaternion subtraction gives us rotation relative to goal
        goal_quat = jp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(cube_quat, goal_quat)

    def _get_finger_tips_distance_from_obj(self, state):
        """Distance of the finger tips from the object."""
        sensor_adrs = [self.mjx_model.sensor_adr[s] for s in self.finger_tip_position_sensors]
        d = jp.zeros(1)
        for sensor_adr in sensor_adrs:
            d += jp.sum(jp.square(state.sensordata[sensor_adr: sensor_adr + 3]))
        return d

    @staticmethod
    def smooth_sigmoid(x, err, eps=0.005, steep=100.0):
        # logistic mask: ≈1 when err < eps, ≈0 otherwise
        mask = jax.nn.sigmoid((eps - err) * steep)
        return mask * (1.0 / x) ** 2

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        grasp_position_err = self._get_cube_distance_from_grasp_err(state)
        grasp_squared_distance = jp.sum(jp.square(grasp_position_err[0:2]))
        # [0:2]ignore z since it can never be fully close to 0 spatially (3D)
        grasp_proximity = grasp_squared_distance - self.threshold ** 2
        grasp_position_cost = 0.1 * grasp_squared_distance + 100 * jp.maximum(grasp_proximity, 0.0)

        grasp_orientation_err = self._get_cube_orientation_from_target_err(state)
        grasp_orientation_cost = jp.sum(jp.square(grasp_orientation_err))

        target_distance_err = self._get_cube_distance_from_target_sensor(state)
        target_squared_distance = jp.sum(jp.square(target_distance_err))
        target_proximity = target_squared_distance - self.threshold ** 2
        target_distance_cost = target_squared_distance + 100 * jp.maximum(target_proximity, 0.0)

        k_grasp = 0.001
        grasp_control_cost = k_grasp * jp.sum(jp.square(control))

        fingers_squared_distance = self._get_finger_tips_distance_from_obj(state)
        fingers_distance_cost = 100 * fingers_squared_distance
        # hover_cost = jax.lax.cond(
        #    grasp_proximity < 0.0,
        #    lambda args: args[0],
        #    lambda args: 0.0,
        #    (10.0 / jp.square(self._get_cube_position(state)[2]),)
        # )

        # smooth_sigmoid(self._get_cube_position(state)[2], squared_distance, 0.005)
        # squared_proximity_from_ground = 1.0 / jp.square(self._get_cube_position(state)[2])
        # hover_cost = 0.1 * squared_proximity_from_ground + 100 * jp.maximum(
        #    squared_proximity_from_ground - self.threshold ** 2, 0.0
        # )
        return (grasp_position_cost + grasp_orientation_cost + grasp_control_cost + target_distance_cost
                + fingers_distance_cost)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_cube_distance_from_grasp_err(state)
        distance_err = self._get_cube_distance_from_target_sensor(state)
        fingers_distance_err = self._get_finger_tips_distance_from_obj(state)
        return (100 * jp.sum(jp.square(position_err)) + 100 * jp.sum(jp.square(distance_err))
                + fingers_distance_err)

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.mjx_model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.mjx_model.geom_friction.at[:, 0].set(
            self.mjx_model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
            self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured configurations."""
        shift = 0.005 * jax.random.normal(rng, (self.mjx_model.nq,))
        return {"qpos": data.qpos + shift}
