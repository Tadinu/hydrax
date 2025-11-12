from typing import Optional, Sequence
from dataclasses import dataclass

# Third-party
import torch

# hydrax
from hydrax.mfr.allegro_cuboid_turning_env import AllegroCuboidTurningEnv, AllegroCuboidTurningCfg


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cuboid Alignment environment config
# -----------------------------------------------------------------------------

@dataclass
class AllegroCuboidAlignmentCfg(AllegroCuboidTurningCfg):
    pass


# -----------------------------------------------------------------------------
# Environment implementation
# Original source: https://github.com/UM-ARM-Lab/MFR_benchmark
# -----------------------------------------------------------------------------

class AllegroCuboidAlignmentEnv(AllegroCuboidTurningEnv):
    def __init__(self, task_cfg: dict, render_mode: Optional[str] = None, **kwargs):
        super().__init__(task_cfg=task_cfg,
                         cfg=AllegroCuboidAlignmentCfg(fingers=task_cfg['fingers']), render_mode=render_mode, **kwargs)
        self.wall_pose = torch.tensor([0, -0.25, 0.19]).float().to(device=self.device)
        self.wall_dims = torch.tensor([0.1, 0.5, 0.12]).float().to(device=self.device)
