from __future__ import annotations
from typing import Any
import os
from dataclasses import dataclass

import mujoco
from etils import epath

# Third-party
import torch

# hydrax
from hydrax import ROOT, BackendType
from hydrax.mfr.allegro_env import AllegroManipEnv, AllegroManipEnvCfg, euler_xyz_from_quat

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = f"{CURRENT_DIR}/config"
MODELS_DIR = f"{ROOT}/models"
ALLEGRO_MODEL_DIR = f"{MODELS_DIR}/allegro_xela"

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cuboid Turning environment config
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
class AllegroCuboidTurningCfg(AllegroManipEnvCfg):
    # URDF is required for Pytorch-kinematics
    # object_model_path: str = f"{MODELS_DIR}/screw_driver/screw_driver_6d.xml"
    object_model_path: str = f"{MODELS_DIR}/screw_driver/screw_driver_6d.urdf"
    cuboid_body_name: str = "cuboid"

    def __post_init__(self):
        super().__post_init__()
        robot_init_state_cfg = self.robot_cfg['init_state']
        robot_init_state_cfg[:3] = [-0.1, 0.0, 0.30]
        robot_init_state_cfg[3:] = [1, 0, 0, 0]


# -----------------------------------------------------------------------------
# Environment implementation
# Original source: https://github.com/UM-ARM-Lab/MFR_benchmark
# -----------------------------------------------------------------------------

class AllegroCuboidTurningEnv(AllegroManipEnv):
    def __init__(self, name: str, task_cfg: dict,
                 backend_type: BackendType = BackendType.MJX):
        super().__init__(name, task_cfg,
                         cfg=AllegroCuboidTurningCfg(fingers=task_cfg['fingers']),
                         backend_type=backend_type)

        # target yaw we want to achieve (per-env) — the task: rotate cuboid to this yaw
        self.target_yaw = torch.zeros((1,), device=self.device)

        self.default_dof_pos = torch.cat((torch.tensor([[0.0, 0.8, 0.4, 0.7]]).float(),
                                          torch.tensor([[-0.15, 0.9, 1.0, 0.9]]).float(),
                                          torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                          torch.tensor([[0.7, 1.0, 0.6, 1.05]]).float()),
                                         dim=1).to(self.device)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat(
            (self.default_dof_pos, torch.tensor([[0, 0, 0, 0, -0.523599, 0]]).float().to(device=self.device)),
            dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(1, 1)
        self.reset()

    # ---- Scene building -----------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        assert len(self.actions.shape) == 2
        state = self.get_state()

        # goal cost
        obj_pos = self.object_data.xpos.copy()
        obj_rot = euler_xyz_from_quat(torch.tensor(self.object_data.xquat.copy(), device=self.device))

        cuboid_upright_cost = (obj_pos[:, 0] ** 2) + (obj_rot[:, 2] ** 2)
        reward -= 1000 * cuboid_upright_cost

        dropp_flag = state[:, -4] < -0.07
        # reward -= 1000 * dropp_flag.to(self.device)
        # dropping cost
        reward -= 1e6 * (dropp_flag * state[:, -4]) ** 2

        # action_cost
        reward -= 50.0 * (torch.norm(self.actions, dim=-1) ** 2)

        # small penalty for high joint velocities
        reward -= 0.01 * torch.linalg.norm(self.mj_data.qvel[:4 * len(self.finger_names)])
        return reward

    def get_state(self):
        results = super().get_state()
        cuboid_pos = torch.tensor(self.object_data.xpos.copy(), device=self.device).float()

        angles = euler_xyz_from_quat(torch.tensor(self.object_data.xquat.copy(), device=self.device))
        cuboid_rot = torch.tensor([angles[0], angles[1], angles[2]], device=self.device).float()
        # print(cuboid_pos, cuboid_rot)
        q = []
        for finger in self.finger_names:
            q.append(results[f'{finger}_q'])
        q.append(cuboid_pos)
        q.append(cuboid_rot)
        q = torch.cat(q)
        return q
