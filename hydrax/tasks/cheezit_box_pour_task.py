from typing import Any, Dict, Optional, Callable, Union
from etils import epath

import numpy as np
import jax
import jax.numpy as jnp

# mujoco
import mujoco as mj
from mujoco import mjx
import mujoco_warp as mjw

# mujoco playground
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT, BackendType
from hydrax.task_base import Task

# mjmanip
from mjmanip.robot.world_base import IDENTITY_POSE

CHEEZIT_BOX_BASE_POSE = [np.array([-0.15, 0.0, 0.17]),
                         np.array([0.0, 1.0, 0.0, 0.0])]  # np.array([0.000, 0.707, 0.0, 0.707])


class CheezitBoxPouringTask(Task):
    """Grains pouring with Cheezit Box"""

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        models_path = epath.Path(ROOT) / "models"
        path = models_path / "cheezit_box"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path, "*.png")
        mjx_env.update_assets(assets, path, "*.obj")
        return assets

    def __init__(self, name: str, backend_type: BackendType = BackendType.MJX) -> None:
        """Load the MuJoCo model and set task parameters."""

        super().__init__(name,
                         xml_path=epath.Path(ROOT) / "models" / "cheezit_box" / "scene_cheezit_box_pour_grains.xml",
                         obj_name="cheezit_box",
                         backend_type=backend_type)

        # Move [cheezit_box]
        cheezit_box = self.mj_model.body("cheezit_box")
        # cheezit_box.pos = CHEEZIT_BOX_BASE_POSE[0]
        # cheezit_box.quat = CHEEZIT_BOX_BASE_POSE[1]

        # Get sensor ids
        self.cheezit_box_position_sensor = self.get_sensor_id("cheezit_box_position")
        self.cheezit_box_orientation_sensor = self.get_sensor_id("cheezit_box_orientation")
        self.cheezit_box_linear_velocity_sensor = self.get_sensor_id("cheezit_box_linear_vel")
        self.cheezit_box_angular_velocity_sensor = self.get_sensor_id("cheezit_box_angular_vel")

        self.cheezit_box_distance_to_target_sensor = self.get_sensor_id("cheezit_box_distance_to_target")
        self.cheezit_box_orientation_from_target_sensor = self.get_sensor_id("cheezit_box_orientation_from_target")

        # Distance thresholds
        self.position_distance_threshold = 0.01
        self.orientation_distance_threshold = 0.01

    @property
    def home_qpos(self):
        # return np.concatenate([IDENTITY_POSE, np.tile(IDENTITY_POSE, 180)]) # 180 grains
        return IDENTITY_POSE

    def _get_cheezit_box_position(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the cheezit_box in world frame."""
        return self.get_sensor_data(data, self.cheezit_box_position_sensor)

    def _get_cheezit_box_orientation(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Orientation of the cheezit_box in world frame."""
        return self.get_sensor_data(data, self.cheezit_box_orientation_sensor)

    def _get_cheezit_box_linear_velocity(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Linear Velocity of the cheezit_box in world."""
        return self.get_sensor_data(data, self.cheezit_box_linear_velocity_sensor)

    def _get_cheezit_box_angular_velocity(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Angular Velocity of the cheezit_box in world."""
        return self.get_sensor_data(data, self.cheezit_box_angular_velocity_sensor)

    def _get_cheezit_box_distance_to_target(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the cheezit_box relative to the target."""
        return self.get_sensor_data(data, self.cheezit_box_distance_to_target_sensor)

    def _get_cheezit_box_orientation_from_target(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Orientation of the cheezit_box relative to the target."""
        return self.get_sensor_data(data, self.cheezit_box_orientation_from_target_sensor)

    @classmethod
    def get_cost(cls, cond: Callable, right_value: jnp.float32, wrong_value: jnp.float32) -> Any:
        return jax.lax.cond(
            cond(),
            lambda args: right_value,
            lambda args: wrong_value,
            right_value
        )

    def running_cost(self, state: Union[mjx.Data, mjw.Data], control: jax.Array,
                     batch_idx: Optional[int] = -1) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Position error
        position_err = self._get_cheezit_box_distance_to_target(state)
        squared_distance = jnp.sum(jnp.square(position_err))
        reaching_cost = 100 * jnp.maximum(
            squared_distance - self.position_distance_threshold ** 2, 0.0
        )
        position_cost = 0.1 * squared_distance + reaching_cost

        # Orientation error
        orientation_err = self._get_cheezit_box_orientation_from_target(state)
        squared_distance = jnp.sum(jnp.square(orientation_err))
        orientation_cost = 50 * jnp.maximum(
            squared_distance - self.orientation_distance_threshold ** 2, 0.0
        )
        return position_cost + orientation_cost

    def terminal_cost(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, None)

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.mjx_model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.mjx_model.geom_friction.at[:, 0].set(
            self.mjx_model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
            self, data: Union[mjx.Data, mjw.Data], rng: jax.Array
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
