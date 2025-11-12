from __future__ import annotations
from typing import Tuple, Optional, Sequence
import os
from pathlib import Path
import math
from dataclasses import dataclass

# Third-party
import torch

# hydrax
from hydrax.mfr.allegro_env import AllegroManipEnv, AllegroManipEnvCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = f"{CURRENT_DIR}/config"
MODELS_DIR = f"{CURRENT_DIR}/models"
ALLEGRO_MODEL_DIR = f"{MODELS_DIR}/allegro_xela"
VALVE_URDF_DIR = f"{MODELS_DIR}/valve"

# -----------------------------------------------------------------------------
# Task object (valve)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

VALVE_INIT_POS = [0.25, 0., 0.31]
VALVE_INIT_QUAT = [0.71, 0.0, 0.0, -0.71]

# -----------------------------------------------------------------------------
# Valve Turning environment config
# -----------------------------------------------------------------------------
"""
num_envs=num_envs,
control_mode='joint_impedance',
viewer=True,
steps_per_action=60,
friction_coefficient=1.0,
device=config['sim_device'],
video_save_path=img_save_dir,
joint_stiffness=config['kp'],
fingers=config['fingers'],
gravity=config['gravity'],
gradual_control=config['gradual_control'],
"""


@dataclass
class AllegroValveTurningCfg(AllegroManipEnvCfg):
    # valve (placeholder uses an instanceable USD box asset; replace if you have USD export of the URDF)
    # NOTE: Valve is an Articulation, not a RigidBody
    object_model_path: str = f"{VALVE_URDF_DIR}/valve_cross.urdf"

    # Contact with fingers
    valve_body_name: str = "valve"

    # action/observation sizes; action -> 16 allegro joint deltas
    action_space: int = 16
    observation_space: int = 64

    # default joint initial pose (derived from allegro.py default_dof_pos)
    default_q: float = 0.0

    def __post_init__(self):
        super().__post_init__()


# -----------------------------------------------------------------------------
# Environment implementation
# Original source: https://github.com/UM-ARM-Lab/MFR_benchmark
# -----------------------------------------------------------------------------

class AllegroValveTurningEnv(AllegroManipEnv):
    def __init__(self, task_cfg: dict, render_mode: Optional[str] = None, **kwargs):
        super().__init__(task_cfg=task_cfg,
                         cfg=AllegroValveTurningCfg(fingers=task_cfg['fingers']), render_mode=render_mode, **kwargs)

        # target yaw we want to achieve (per-env) — the task: rotate valve to this yaw
        self.target_yaw = torch.zeros((self.scene.num_envs,), device=self.device)

        self.default_dof_pos = torch.cat((torch.tensor([[0.0, 0.8, 0.4, 0.7]]).float(),
                                          torch.tensor([[-0.15, 0.9, 1.0, 0.9]]).float(),
                                          torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                          torch.tensor([[0.7, 1.0, 0.6, 1.05]]).float()),
                                         dim=1).to(self.device)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat(
            (self.default_dof_pos, torch.tensor([[0, 0, 0, 0, -0.523599, 0]]).float().to(device=self.device)),
            dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    # ---- Scene building -----------------------------------------------------

    def _setup_scene(self):
        assert isinstance(self.task_cfg, AllegroValveTurningCfg)
        super()._setup_scene()

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        assert len(self.actions.shape) == 2
        state = self.get_state()

        # goal cost
        obj_pos = self.object_pos
        obj_ori = self.object_rot

        valve_upright_cost = (obj_ori[:, 0] ** 2) + (obj_ori[:, 2] ** 2)
        reward -= 1000 * valve_upright_cost

        dropp_flag = state[:, -4] < -0.07
        # reward -= 1000 * dropp_flag.to(self.device)
        # dropping cost
        reward -= 1e6 * (dropp_flag * state[:, -4]) ** 2

        # action_cost
        reward -= 50.0 * (torch.norm(self.actions, dim=-1) ** 2)

        # small penalty for high joint velocities
        reward -= 0.01 * torch.linalg.norm(self.hand_dof_vel)
        return reward

    def get_state(self):
        results = super().get_state()
        valve_pos = self.valve.data.root_pos_w - self.scene.env_origins
        angles = euler_xyz_from_quat(self.object.data.root_quat_w)
        valve_rot = torch.tensor([angles[0], angles[1], angles[2]], device=self.device).reshape(1,
                                                                                                len(angles))
        q = []
        for finger in self.finger_names:
            q.append(results[f'{finger}_q'])

        q.append(valve_pos)
        q.append(valve_rot)
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)

        N = env_ids.numel()
        # randomize valve pose, maintaining Z
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos = torch.zeros((N, 3), device=self.device)
        pos[:, 0] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
        pos[:, 1] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
        pos[:, 2] = 0.31
        object_default_state[:, 0:3] = pos

        yaw = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        quat = torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), sy, cy], dim=-1)
        object_default_state[:, 3:7] = quat

        self.valve.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.valve.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids=env_ids)

        # set per-env target yaw (e.g. random target in [-pi,pi])
        self.target_yaw[env_ids] = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi
