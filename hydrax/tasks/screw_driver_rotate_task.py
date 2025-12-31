from typing import Any, Dict, Optional, Callable, Union
from etils import epath

import numpy as np
import jax
import jax.numpy as jnp

# mujoco
import mujoco as mj
from mujoco import mjx

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT
from hydrax.task_base import Task

# mjmanip
from mjmanip.utils import mj_body_qids


class ScrewDriverRotateTask(Task):
    """Screw Driver rotation with the LEAP hand."""
    HAND_BASE_POSE = [np.array([-0.15, 0.0, 0.17]),
                      np.array([0.0, 1.0, 0.0, 0.0])]  # np.array([0.000, 0.707, 0.0, 0.707])
    HAND_HOME_QPOS = [
        0.8, 0, 0.8, 0.8,
        0.8, 0, 0.8, 0.8,
        0.8, 0, 0.8, 0.8,
        0.8, 0.8, 0.8, 0,
    ]
    BASE_BODY_NAME = "leap_mount"

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        models_path = epath.Path(ROOT) / "models"
        path = models_path / "leap_hand"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")

        path = models_path / "screw_driver"
        mjx_env.update_assets(assets, path, "*.xml")

        path = models_path / "cube"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "reorientation_cube_textures")
        return assets

    def __init__(self, name: str, warp_enabled: bool = False,
                 xml_path=epath.Path(
                     ROOT) / "models" / "leap_hand" / "scene_leap_rh_mjx_rotate_screw_driver.xml") -> None:
        """Load the MuJoCo model and set task parameters."""

        self.HAND_MODEL_NAME = "leap_rh_mjx"
        self.FINGER_TIPS_NAMES = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
        super().__init__(name,
                         xml_path=xml_path,
                         obj_name="screw_driver",
                         trace_sites=["grasp_site"] + self.FINGER_TIPS_NAMES,
                         warp_enabled=warp_enabled)

        # Move [base_body]
        base_body = self.mj_model.body(self.BASE_BODY_NAME)
        base_body.pos = self.HAND_BASE_POSE[0]
        base_body.quat = self.HAND_BASE_POSE[1]

        # Obj
        self._init_obj_qpos = self._mj_data.qpos[mj_body_qids(self._mj_model, self._obj_name, is_qpos=True)]

        # Get sensor ids
        self.screw_driver_position_sensor = self.get_sensor_id("screw_driver_position")
        self.screw_driver_orientation_sensor = self.get_sensor_id("screw_driver_orientation")
        self.screw_driver_contact_with_palm_sensor = self.get_sensor_id("screw_driver_contact_with_palm")
        self.screw_driver_distance_to_grasp_sensor = self.get_sensor_id("screw_driver_distance_to_grasp")
        self.obj_contact_with_finger_tip_sensors = {
            finger_tip: self.get_sensor_id(f"screw_driver_contact_with_{finger_tip}")
            for finger_tip in self.FINGER_TIPS_NAMES
        }
        self.screw_driver_distance_to_target_sensor = self.get_sensor_id("screw_driver_distance_to_target")
        self.screw_driver_orientation_from_target_sensor = self.get_sensor_id("screw_driver_orientation_from_target")
        self.screw_driver_linear_velocity_sensor = self.get_sensor_id("screw_driver_linear_vel")
        self.screw_driver_angular_velocity_sensor = self.get_sensor_id("screw_driver_angular_vel")
        self.screw_driver_head_distance_to_target_heads_sensors = [
            self.get_sensor_id("screw_driver_head_distance_to_target_head_a"),
            self.get_sensor_id("screw_driver_head_distance_to_target_head_b")]
        self.finger_tip_distance_to_screw_driver_sensors = [self.get_sensor_id(f"{finger_tip}_distance_to_screw_driver")
                                                            for finger_tip in self.FINGER_TIPS_NAMES]

        # Distance (m) beyond which we impose a high screw_driver position cost
        self.grasp_threshold = 0.5  # 0.015
        self.target_distance_threshold = 0.001

    def _post_init(self) -> None:
        super()._post_init()

        # Hand-specifics
        self._init_hand()

    def _init_hand(self):
        pass

    @property
    def home_qpos(self):
        return (
            HAND_HOME_QPOS + self._init_obj_qpos.tolist() if self._obj_name else HAND_HOME_QPOS
        )

    def _get_screw_driver_position(self, data: mjx.Data) -> jax.Array:
        """Position of the screw_driver in world frame."""
        return self.get_sensor_data(data, self.screw_driver_position_sensor)

    def _get_screw_driver_contact_with_palm(self, data: mjx.Data) -> jax.Array:
        """Num of screw_driver contacts with palm"""
        # [found: 0 or num_contacts]
        return self.get_sensor_data(data, self.screw_driver_contact_with_palm_sensor, end=1)

    def _get_obj_contact_with_finger_tips(self, data: mjx.Data) -> jax.Array:
        # Each return [found: 0 or num_contacts]
        return jnp.sum(jnp.array(
            [self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f], end=1) for f in
             self.FINGER_TIPS_NAMES]))

    def _get_obj_contact_force_with_finger_tips(self, data: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(jnp.array([self.get_sensor_data(data, self.obj_contact_with_finger_tip_sensors[f],
                                                                  start=1, end=4) for f in self.FINGER_TIPS_NAMES])))

    def _get_screw_driver_distance_to_grasp(self, data: mjx.Data) -> jax.Array:
        """Position of the screw_driver relative to the grasp."""
        return self.get_sensor_data(data, self.screw_driver_distance_to_grasp_sensor)

    def _get_screw_driver_distance_to_target(self, data: mjx.Data) -> jax.Array:
        """Position of the screw_driver relative to the target."""
        return self.get_sensor_data(data, self.screw_driver_distance_to_target_sensor)

    def _get_screw_driver_orientation(self, data: mjx.Data) -> jax.Array:
        """Orientation of the screw_driver in world frame."""
        return self.get_sensor_data(data, self.screw_driver_orientation_sensor)

    def _get_screw_driver_orientation_distance_to_target(self, data: mjx.Data) -> jax.Array:
        """Orientation of the screw_driver relative to the target grasp orientation."""
        screw_driver_relative_to_target_quat = self.get_sensor_data(data,
                                                                    self.screw_driver_orientation_from_target_sensor)

        # Quaternion subtraction gives us rotation relative to goal
        goal_relative_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return jnp.sum(jnp.square(mjx._src.math.quat_sub(screw_driver_relative_to_target_quat, goal_relative_quat)))

    def _get_screw_driver_linear_velocity(self, data: mjx.Data) -> jax.Array:
        """Velocity of the screw_driver in world."""
        return self.get_sensor_data(data, self.screw_driver_linear_velocity_sensor)

    def _get_finger_tips_distance_to_screw_driver(self, data: mjx.Data) -> jax.Array:
        """Distance of the fingertips from the object."""
        return jnp.sum(
            jnp.square(
                jnp.array([self.get_sensor_data(data, s) for s in self.finger_tip_distance_to_screw_driver_sensors])))

    def _get_screw_driver_head_distance_to_target_heads(self, data: mjx.Data) -> jax.Array:
        """Position of the screw_driver head relative to the target heads"""
        return jnp.sum(
            jnp.square(jnp.array([self.get_sensor_data(data, sensor)
                                  for sensor in self.screw_driver_head_distance_to_target_heads_sensors])))

    # Palm cost
    def _get_palm_cost(self, state: mjx.Data, encourage: bool) -> jax.Array:
        return (-1 if encourage else 1) * 0.05 * self._get_screw_driver_contact_with_palm(state)

    # Fingertips total cost
    def _get_fingertips_cost(self, state: mjx.Data) -> jax.Array:
        # cost = 50 * self._get_finger_tips_distance_to_obj(state)
        cost = -0.01 * self._get_obj_contact_with_finger_tips(state)
        return cost

    @classmethod
    def get_cost(cls, cond: Callable, right_value: jnp.float32, wrong_value: jnp.float32) -> Any:
        return jax.lax.cond(
            cond(),
            lambda args: right_value,
            lambda args: wrong_value,
            right_value
        )

    def running_cost(self, data: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        squared_distance = self._get_finger_tips_distance_to_screw_driver(data)
        reaching_cost = 100 * jnp.maximum(
            squared_distance - self.grasp_threshold ** 2, 0.0
        )
        head_cost = 0  # 10000 * self._get_screw_driver_head_distance_to_target_heads(data) # Only for bringing/relocating
        position_cost = 0.1 * squared_distance + reaching_cost + head_cost
        orientation_cost = 5000000 * self._get_screw_driver_orientation_distance_to_target(data)
        grasp_cost = 0.001 * jnp.sum(jnp.square(control))  # + self._get_fingertips_cost(data)
        return position_cost + orientation_cost + grasp_cost

    def terminal_cost(self, data: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(data, jnp.zeros(1))

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
        if True:
            """Add noise to the state estimate."""
            rng, q_rng, v_rng = jax.random.split(rng, 3)
            q_err = 0.01 * jax.random.normal(q_rng, (self.mjx_model.nq,))
            v_err = 0.01 * jax.random.normal(v_rng, (self.mjx_model.nv,))
            return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}
        else:
            shift = 0.005 * jax.random.normal(rng, (self.mjx_model.nq,))
            return {"qpos": data.qpos + shift}
