from typing import Dict, Optional

import jax
import jax.numpy as jnp
import mujoco as mj
from mujoco import mjx
import mujoco_warp as mjw

from hydrax import ROOT, BackendType
from hydrax.task_base import Task


class PushT(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(self, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mj.MjModel.from_xml_path(
            ROOT + "/models/pusht/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pusher"], impl=impl)

        # Get sensor ids
        self.block_position_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mj.mj_name2id(
            mj_model, mj.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def _get_position_err(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.mjx_model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 3]

    def _get_orientation_err(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.mjx_model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr: sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the pusher block relative to the block."""
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: Union[mjx.Data, mjw.Data], control: jax.Array,
                     batch_idx: Optional[int] = -1) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))

        return position_cost + orientation_cost + 0.01 * close_to_block_cost

    def terminal_cost(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        return self.running_cost(state, jnp.zeros(self.mjx_model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.mjx_model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.mjx_model.geom_friction.at[:, 0].set(
            self.mjx_model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        return super().make_data(nconmax=6000)
