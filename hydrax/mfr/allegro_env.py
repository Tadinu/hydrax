"""
Allegro Hand manipulation environments
-----------------------------------------------------------------
This file ports a custom Isaac Gym Allegro-hand task to the Isaac Lab
"direct RL" workflow. It targets Isaac Sim 5.0 and the latest
Isaac Lab API (see docs linked in the chat response for references).
- Uses `DirectRLEnv` / `DirectRLEnvCfg` and the InteractiveScene.
- Spawns an Allegro hand articulation and task objects
- Supports joint-position or joint-velocity control through PD actuators.
- Vectorized over N environments; designed for RL (Gymnasium-like API).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import mujoco
from etils import epath
import yaml
import pickle as pkl
from functools import partial

# Third-party
import numpy as np
import torch
from torch.func import vmap, jacrev, hessian, jacfwd
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

# ccai
from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.faster_csvto_dev.faster_csvto.faster_csvto import FasterConstrainedSteinTrajOpt
from ccai.kernels import rbf_kernel, structured_rbf_kernel
from ccai.problem import ConstrainedSVGDProblem
from ccai.mpc.csvgd import Constrained_SVGD_MPC

BaseConstrainedSteinTrajOpt = ConstrainedSteinTrajOpt

# mujoco
import jax
from mujoco import mjx
from mujoco_playground._src import mjx_env

# hydrax
from hydrax import ROOT
from hydrax.tasks.screw_driver_rotate_task import ScrewDriverRotateTask
from hydrax.mfr.utils.allegro_utils import all_finger_constraints, partial_to_full_state

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = f"{CURRENT_DIR}/config"
MODELS_DIR = f"{ROOT}/models"
ALLEGRO_MODEL_DIR = f"{MODELS_DIR}/allegro_xela"
ALLEGRO_MESH_DIR = f"{ALLEGRO_MODEL_DIR}/mesh"


def get_task_config(task_name: Optional[str] = None):
    if not task_name:
        task_name = 'cuboid_turning'
    if task_name == 'screwdriver_turning':
        config = yaml.safe_load(Path(f'{CONFIG_DIR}/allegro_screwdriver.yaml').read_text())
        config['num_env_force'] = 1
    elif task_name == 'valve_turning':
        config = yaml.safe_load(Path(f'{CONFIG_DIR}/allegro_valve.yaml').read_text())
        config['num_env_force'] = 0
    elif task_name == 'cuboid_turning':
        config = yaml.safe_load(Path(f'{CONFIG_DIR}/allegro_cuboid_turning.yaml').read_text())
        config['num_env_force'] = 0
    elif task_name == 'cuboid_alignment':
        config = yaml.safe_load(Path(f'{CONFIG_DIR}/allegro_cuboid_alignment.yaml').read_text())
        config['num_env_force'] = 1
    elif task_name == 'reorientation':
        config = yaml.safe_load(Path(f'{CONFIG_DIR}/allegro_reorientation.yaml').read_text())
        config['num_env_force'] = 0
    else:
        raise ValueError(f'Unknown task {task_name}')
    config['task'] = task_name
    return config


class PositionControlConstrainedSteinTrajOpt(BaseConstrainedSteinTrajOpt):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.torque_limit = params.get('torque_limit', 1)
        self.kp = params['kp']
        self.fingers = problem.fingers
        self.num_fingers = len(self.fingers)

    def _clamp_in_bounds(self, xuz):
        N = xuz.shape[0]
        min_x = self.problem.x_min.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        max_x = self.problem.x_max.reshape(1, 1, -1).repeat(1, self.problem.T, 1)
        if self.problem.dz > 0:
            min_x = torch.cat((min_x, -1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)
            max_x = torch.cat((max_x, 1e3 * torch.ones(1, self.problem.T, self.problem.dz)), dim=-1)

        torch.clamp_(xuz, min=min_x.to(device=xuz.device).reshape(1, -1),
                     max=max_x.to(device=xuz.device).reshape(1, -1))

        if self.problem.du > 0:
            xuz_copy = xuz.reshape((N, self.problem.T, -1))
            robot_joint_angles = xuz_copy[:, :-1, :self.problem.robot_dof]
            robot_joint_angles = torch.cat(
                (self.problem.start[:self.problem.robot_dof].reshape((1, 1, self.problem.robot_dof)).repeat((N, 1, 1)),
                 robot_joint_angles), dim=1)

            # make the commanded delta position respect the joint limits
            min_u_jlim = self.problem.robot_joint_x_min.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles
            max_u_jlim = self.problem.robot_joint_x_max.repeat((N, self.problem.T, 1)).to(
                xuz.device) - robot_joint_angles

            # make the commanded delta position respect the torque limits
            min_u_tlim = -self.torque_limit / self.kp * torch.ones_like(min_u_jlim)
            max_u_tlim = self.torque_limit / self.kp * torch.ones_like(max_u_jlim)

            # overall commanded delta position limits
            min_u = torch.where(min_u_jlim > min_u_tlim, min_u_jlim, min_u_tlim)
            max_u = torch.where(max_u_tlim > max_u_jlim, max_u_jlim, max_u_tlim)
            min_x = min_x.repeat((N, 1, 1)).to(device=xuz.device)
            max_x = max_x.repeat((N, 1, 1)).to(device=xuz.device)
            min_x[:, :, self.problem.dx:self.problem.dx + self.problem.robot_dof] = min_u
            max_x[:, :, self.problem.dx:self.problem.dx + self.problem.robot_dof] = max_u
            torch.clamp_(xuz, min=min_x.reshape((N, -1)), max=max_x.reshape((N, -1)))

    def resample(self, xuz):
        xuz = xuz.to(dtype=torch.float32)
        self.problem._preprocess(xuz)
        return super().resample(xuz)


class PositionControlConstrainedSVGDMPC(Constrained_SVGD_MPC):

    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.solver = PositionControlConstrainedSteinTrajOpt(problem, params)


class AllegroObjectProblem(ConstrainedSVGDProblem):

    def __init__(self,
                 dx,
                 du,
                 start,  # start should include both robot and obj states
                 goal,
                 T,
                 chain,
                 world_trans,  # transformation from the world to robot frame
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 obj_dof_code=[0, 0, 0, 0, 0, 0],
                 obj_joint_dim=0,
                 device='cuda:0'):
        """
        obj_dof: DoF of the object, The max number is 6, It's the DoF for the rigid body, not including any joints within the object.
        obj_joint_dim: It's the DoF of the joints within the object, excluding those are rigid body DoF.
        """
        super().__init__(start, goal, T, device)
        self.dx, self.du = dx, du
        self.dg_per_t = 0
        self.dg_constant = 0
        self.device = device
        self.dt = 0.1
        self.T = T
        self.start = start
        self.goal = goal
        self.K = rbf_kernel
        self.squared_slack = True
        self.compute_hess = False
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.obj_dof = np.sum(obj_dof_code)
        self.obj_translational_code = obj_dof_code[:3]
        self.obj_rotational_code = obj_dof_code[3:]
        self.obj_translational_dim = np.sum(self.obj_translational_code)
        self.obj_rotational_dim = np.sum(self.obj_rotational_code)
        self.obj_joint_dim = obj_joint_dim
        self.arm_dof = 0
        self.robot_dof = self.arm_dof + 4 * self.num_fingers
        self.chain = chain
        self.joint_index = {
            'index': list(np.array([0, 1, 2, 3]) + self.arm_dof),
            'middle': list(np.array([4, 5, 6, 7]) + self.arm_dof),
            'ring': list(np.array([8, 9, 10, 11]) + self.arm_dof),
            'thumb': list(np.array([12, 13, 14, 15]) + self.arm_dof)
        }
        self.all_joint_index = sum([self.joint_index[finger] for finger in self.fingers], [])
        self.collision_checking_ee_names = {
            'index': 'allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'middle': 'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'ring': 'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'thumb': 'allegro_hand_oya_finger_3_aftc_base_link',
        }
        self.ee_names = {
            'index': 'hitosashi_ee',
            'middle': 'naka_ee',
            'ring': 'kusuri_ee',
            'thumb': 'oya_ee',
        }
        self.ee_link_idx = {finger: chain.frame_to_idx[ee_name] for finger, ee_name in self.ee_names.items()}
        self.frame_indices = torch.tensor([self.ee_link_idx[finger] for finger in self.fingers])

        self.grad_kernel = jacrev(rbf_kernel, argnums=0)

        self.world_trans = world_trans.to(device=device)
        self.alpha = 10
        self.env_force = False
        # for honda hand
        index_x_max = torch.tensor([0.47, 1.6099999999, 1.7089999, 1.61799999]) + 0.05
        index_x_min = torch.tensor([-0.47, -0.195999999999, -0.174000000, -0.227]) - 0.05
        thumb_x_max = torch.tensor([1.396, 1.1629999999999, 1.644, 1.71899999]) + 0.05
        thumb_x_min = torch.tensor([0.26, -0.1049999999, -0.1889999999, -0.162]) - 0.05
        joint_min = {'index': index_x_min, 'middle': index_x_min, 'ring': index_x_min, 'thumb': thumb_x_min}
        joint_max = {'index': index_x_max, 'middle': index_x_max, 'ring': index_x_max, 'thumb': thumb_x_max}
        self.x_max = torch.cat([joint_max[finger] for finger in self.fingers])
        self.x_min = torch.cat([joint_min[finger] for finger in self.fingers])

        self.robot_joint_x_max = self.x_max.clone()
        self.robot_joint_x_min = self.x_min.clone()
        if self.du > 0:
            self.u_max = torch.ones(self.robot_dof) * np.pi / 5
            self.u_min = - torch.ones(self.robot_dof) * np.pi / 5
            self.x_max = torch.cat((self.x_max, self.u_max))
            self.x_min = torch.cat((self.x_min, self.u_min))
        self.data = {}

        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

        self.singularity_constr = vmap(self._singularity_constr)
        self.grad_singularity_constr = vmap(jacrev(self._singularity_constr))

        self.grad_euler_to_angular_velocity = jacrev(euler_to_angular_velocity, argnums=(0, 1))

        # DEBUG ONLY
        self.J_list = []
        self.contact_con = []
        self.force_con = []
        self.kinematics_con = []
        self.friction_con = []

        self.contact_con_mean = []
        self.force_con_mean = []
        self.kinematics_con_mean = []
        self.friction_con_mean = []

    def finger_id(self, finger_name: str) -> int:
        return self.fingers.index(finger_name) if finger_name in self.fingers else -1

    @property
    def start_obj_pose(self) -> torch.Tensor:
        return self.start[-self.obj_dof:]

    def is_free6d_obj(self) -> bool:
        return self.obj_dof == 6

    def forward_kinematics(self, q):
        return self.chain.forward_kinematics(partial_to_full_state(q[:, :4 * self.num_fingers], fingers=self.fingers),
                                             frame_indices=self.frame_indices)

    def save_history(self, save_dir):
        result_dict = {
            'J': self.J_list,
            'contact_con': self.contact_con,
            'force_con': self.force_con,
            'kinematics_con': self.kinematics_con,
            'friction_con': self.friction_con,
            'contact_con_mean': self.contact_con_mean,
            'force_con_mean': self.force_con_mean,
            'kinematics_con_mean': self.kinematics_con_mean,
            'friction_con_mean': self.friction_con_mean
        }
        with open(save_dir, 'wb') as f:
            pkl.dump(result_dict, f)

    def _cost(self, x, start, goal):
        raise NotImplementedError

    def _objective(self, x):
        x = x[:, :, :self.dx + self.du]
        N = x.shape[0]
        J, grad_J, hess_J = self.cost(x), self.grad_cost(x), self.hess_cost(x)

        N = x.shape[0]
        return (self.alpha * J.reshape(N),
                self.alpha * grad_J.reshape(N, -1),
                self.alpha * hess_J.reshape(N, self.T * (self.dx + self.du), self.T * (self.dx + self.du)))

    def _step_size_limit(self, xu):
        N, T, _ = xu.shape
        u = xu[:, :, -self.du:]

        max_step_size = 0.1
        h_plus = u - max_step_size
        h_minus = -u - max_step_size
        h = torch.stack((h_plus, h_minus), dim=2)  # N x T x 2 x du

        grad_h = torch.zeros(N, T, 2, self.du, T, self.dx + self.du, device=xu.device)
        hess_h = torch.zeros(N, T * 2 * self.du, T * (self.dx + self.du), device=xu.device)
        # assign gradients
        T_range = torch.arange(0, T, device=xu.device)
        grad_h[:, T_range, 0, :, T_range, -self.du:] = torch.eye(self.du, device=xu.device)
        grad_h[:, T_range, 1, :, T_range, -self.du:] = -torch.eye(self.du, device=xu.device)

        return h.reshape(N, -1), grad_h.reshape(N, -1, T * (self.dx + self.du)), hess_h

    def _singularity_constr(self, contact_jac):
        # this will be vmapped
        A = contact_jac @ contact_jac.transpose(-1, -2)
        eig = torch.linalg.eigvals(A).abs()
        eig = torch.topk(eig, 2, dim=-1).values
        manipulability = eig[0] / eig[1] - 50
        return manipulability
        # manipulability = torch.sqrt(torch.prod(eig, dim=-1))
        # return 0.0001 - manipulability

    @all_finger_constraints
    def _singularity_constraint(self, xu, finger_name, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        q = xu[:, :, :self.robot_dof]
        contact_jac = self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, self.robot_dof)[:, 1:]

        # compute constraint value
        h = self.singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))
        h = h.reshape(N, -1)
        dh = 1
        # compute the gradient
        if compute_grads:
            dh_djac = self.grad_singularity_constr(contact_jac.reshape(-1, 3, self.num_fingers * 4))

            djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[
                :, 1:]

            dh_dq = dh_djac.reshape(N, T, dh, -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, :4 * self.num_fingers] = dh_dq[:, 1:].transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None

    @staticmethod
    def get_rotation_from_normal(normal_vector):
        """
        :param normal_vector: (batch_size, 3)
        :return: (batch_size, 3, 3) rotation matrix with normal vector as the z-axis
        """
        z_axis = normal_vector / torch.norm(normal_vector, dim=1, keepdim=True)
        # y_axis = torch.randn_like(z_axis)
        y_axis = torch.tensor([1.0, 1.0, 1.0], device=normal_vector.device) \
                     .unsqueeze(0).repeat(normal_vector.shape[0], 1) / torch.sqrt(torch.tensor(3))
        y_axis = y_axis - torch.sum(y_axis * z_axis, dim=1).unsqueeze(-1) * z_axis
        y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)
        x_axis = torch.linalg.cross(y_axis, z_axis, dim=-1)
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)
        R = torch.stack((x_axis, y_axis, z_axis), dim=2)
        return R

    def _ee_locations_in_screwdriver(self, q_rob, q_env, queried_fingers, object_frame_name='screwdriver_body'):

        assert q_rob.shape[-1] == 16
        assert q_env.shape[-1] == self.obj_dof

        _q_env = q_env.clone()
        if self.obj_dof == 3:
            _q_env = torch.cat((q_env, torch.zeros_like(q_env[..., :1])), dim=-1)

        robot_trans = self.contact_scene.robot_sdf.chain.forward_kinematics(q_rob.reshape(-1, 16))
        ee_locs = []

        for finger in queried_fingers:
            ee_locs.append(robot_trans[self.ee_names[finger]].get_matrix()[:, :3, -1])

        ee_locs = torch.stack(ee_locs, dim=1)

        # convert to scene base frame
        ee_locs = self.contact_scene.scene_transform.inverse().transform_points(ee_locs)

        # convert to scene ee frame
        # Note, the FK here does not consider the change of the link center, specifically, it's the position of the joint connecting this link
        object_trans = self.contact_scene.scene_sdf.chain.forward_kinematics(
            _q_env.reshape(-1, _q_env.shape[-1]))
        ee_locs = object_trans[object_frame_name].inverse().transform_points(ee_locs)

        return ee_locs

    def eval(self, augmented_trajectory):
        N = augmented_trajectory.shape[0]
        augmented_trajectory = augmented_trajectory.clone().reshape(N, self.T, -1)
        x = augmented_trajectory[:, :, :self.dx + self.du]

        # preprocess fingers
        self._preprocess(x)

        # compute objective
        J, grad_J, hess_J = self._objective(x)
        hess_J = None
        grad_J = torch.cat((grad_J.reshape(N, self.T, -1),
                            torch.zeros(N, self.T, self.dz, device=x.device)), dim=2).reshape(N, -1)

        Xk = x.reshape(N, self.T, -1)
        K = self.K(Xk, Xk, None)  # hess_J.mean(dim=0))
        grad_K = -self.grad_kernel(Xk, Xk, None)  # @hess_J.mean(dim=0))
        grad_K = grad_K.reshape(N, N, N, self.T * (self.dx + self.du))
        grad_K = torch.einsum('nmmi->nmi', grad_K)
        grad_K = torch.cat((grad_K.reshape(N, N, self.T, self.dx + self.du),
                            torch.zeros(N, N, self.T, self.dz, device=x.device)), dim=-1)
        grad_K = grad_K.reshape(N, N, -1)
        G, dG, hessG = self.combined_constraints(augmented_trajectory, compute_hess=self.compute_hess)
        # DEBUG ONLY
        # augmented_x = augmented_trajectory.reshape(N, self.T, self.dx + self.du + self.dz)
        # xu = augmented_x[:, :, :(self.dx + self.du)]
        # try:
        #     g = self._con_eq(xu, compute_grads=False, compute_hess=False, verbose=True)
        #     h = self._con_ineq(xu, compute_grads=False, compute_hess=False, verbose=True)
        #     self.J_list.append(J.mean().item())
        #     self.contact_con.append(g['contact'])
        #     self.force_con.append(g['force'])
        #     self.kinematics_con.append(g['kinematics'])
        #     self.friction_con.append(h['friction'])
        #     # self.friction_con.append(g['friction'])
        #     self.contact_con_mean.append(g['contact_mean'])
        #     self.force_con_mean.append(g['force_mean'])
        #     self.kinematics_con_mean.append(g['kinematics_mean'])
        #     self.friction_con_mean.append(h['friction_mean'])
        #     # self.friction_con_mean.append(g['friction_mean'])
        # except:
        #     pass

        if hessG is not None:
            hessG.detach_()

        return grad_J.detach(), hess_J, K.detach(), grad_K.detach(), G.detach(), dG.detach(), hessG

    def update(self, start, goal=None, T=None):
        self.start = start
        if goal is not None:
            self.goal = goal

        # update functions that require start
        self.cost = vmap(partial(self._cost, start=self.start, goal=self.goal))
        self.grad_cost = vmap(jacrev(partial(self._cost, start=self.start, goal=self.goal)))
        self.hess_cost = vmap(hessian(partial(self._cost, start=self.start, goal=self.goal)))

        if T is not None:
            self.T = T
            self.dh = self.dz * T
            self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics

        # DEBUG ONLY
        self.J_list = []
        self.contact_con = []
        self.force_con = []
        self.kinematics_con = []
        self.friction_con = []

        self.contact_con_mean = []
        self.force_con_mean = []
        self.kinematics_con_mean = []
        self.friction_con_mean = []

    def get_initial_xu(self, N):
        """
        use delta joint movement to get the initial trajectory
        the action (force at the fingertip) is not used. it is randomly initialized
        the actual dynamics model is not used
        initialize with object not moving at all
        """

        u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u[:, :, :self.arm_dof] = u[:, :, :self.arm_dof] * 0.1

        x = [self.start[:self.dx].reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :self.dx] + u[:, t, :self.dx]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        if self.dx == (self.robot_dof + self.obj_dof):
            theta = self.start_obj_pose.unsqueeze(0).repeat((N, self.T, 1))
            x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def check_validity(self, state):
        return True


class AllegroContactProblem(AllegroObjectProblem):

    def get_constraint_dim(self, T):
        self.dg_per_t = 0
        self.dg_constant = self.num_fingers
        self.dg = self.dg_per_t * T + self.dg_constant
        self.dz = 0  # one friction constraints per finger
        self.dh = self.dz * T  # inequality

    def __init__(self,
                 dx,
                 du,
                 start,
                 goal,
                 T,
                 chain,
                 object_type: str,
                 world_trans,
                 object_asset_pos,
                 object_model_path: str,
                 fingers: list[str] = ['index', 'middle', 'ring', 'thumb'],
                 obj_dof_code: list[int] = [0, 0, 0, 0, 0, 0],
                 obj_joint_dim: int = 0,
                 collision_checking: bool = False,
                 device='cuda:0'):
        # object_location is different from object_asset_pos. object_asset_pos is
        # used for pytorch volumetric. The asset of valve might contain something else such as a wall, a table
        # object_location is the location of the object joint, which is what we care for motion planning
        super().__init__(dx=dx, du=du, start=start, goal=goal, T=T, chain=chain,
                         world_trans=world_trans, fingers=fingers, obj_dof_code=obj_dof_code,
                         obj_joint_dim=obj_joint_dim, device=device)
        self.collision_checking = collision_checking
        self.get_constraint_dim(T)

        # add collision checking
        # collision check all of the non-finger tip links
        # collision_check_oya = ['allegro_hand_oya_finger_link_13',
        #                        'allegro_hand_oya_finger_link_14',
        #                        ]
        # collision_check_hitosashi = [
        #     'allegro_hand_hitosashi_finger_finger_link_2',
        #     'allegro_hand_hitosashi_finger_finger_link_1'
        # ]
        self.object_type = object_type
        if object_model_path.endswith('.xml'):
            self.object_chain = pk.build_chain_from_mjcf(object_model_path).to(device=self.device)
        else:
            self.object_chain = pk.build_chain_from_urdf(open(object_model_path).read()).to(device=self.device)
        self.object_asset_pos = object_asset_pos.clone().detach().to(self.device).float()

        self._init_contact_scenes(collision_checking)

    def _init_contact_scenes(self, collision_checking: bool = False, visualize: bool = False):
        object_sdf = pv.RobotSDF(self.object_chain,
                                 # since we are using primitive shapes for the object, there's no need to define path for stl
                                 path_prefix=ALLEGRO_MESH_DIR if self.object_chain.is_from_mjcf else ALLEGRO_MODEL_DIR,
                                 use_collision_geometry=True,
                                 exclude_fixed_joints=True)
        robot_sdf = pv.RobotSDF(self.chain,
                                path_prefix=ALLEGRO_MESH_DIR if self.chain.is_from_mjcf else ALLEGRO_MODEL_DIR,
                                use_collision_geometry=True,
                                exclude_fixed_joints=False)

        scene_trans = self.world_trans.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.object_asset_pos[0], self.object_asset_pos[1],
                                                         self.object_asset_pos[2]))

        # self.index_collision_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
        #                                            collision_check_links=collision_check_hitosashi,
        #                                            softmin_temp=100.0)
        # self.thumb_collision_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
        #                                            collision_check_links=collision_check_oya,
        #                                            softmin_temp=100.0)
        # contact checking
        collision_check_links = [self.collision_checking_ee_names[finger] for finger in self.fingers]
        # grad_smooth_points = 50
        if collision_checking:
            collision_check_links.append('allegro_hand_hitosashi_finger_finger_link_2')
            collision_check_links.append('allegro_hand_hitosashi_finger_finger_link_3')
        self.contact_scene = pv.RobotScene(robot_sdf, object_sdf, scene_trans,
                                           collision_check_links=collision_check_links,
                                           # [self.collision_checking_ee_names['thumb']]
                                           softmin_temp=1.0e3,
                                           points_per_link=1000,
                                           partial_patch=False,
                                           # grad_smooth_points=grad_smooth_points,
                                           )
        if visualize:
            self.contact_scene.visualize_robot(
                partial_to_full_state(self.start[:self.robot_dof], fingers=self.fingers, arm_dof=self.arm_dof), None)

    def _preprocess(self, xu):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        x_expanded = torch.cat([self.start[:self.dx].reshape(1, 1, -1).repeat(N, 1, 1), x], dim=1)

        q = x_expanded[:, :, :self.robot_dof]
        if self.dx == self.robot_dof:
            theta = self.start_obj_pose.unsqueeze(0).repeat((N, self.T + 1, 1))
        else:
            theta = x_expanded[:, :, self.robot_dof: self.robot_dof + self.obj_dof]
        self._preprocess_fingers(q, theta)

    def _preprocess_fingers(self, q, theta):
        N, _, _ = q.shape

        # reshape to batch across time
        q_b = q.reshape(-1, self.robot_dof)
        theta_b = theta.reshape(-1, self.obj_dof)
        if self.obj_joint_dim > 0:
            theta_obj_joint = torch.zeros((theta_b.shape[0], self.obj_joint_dim),
                                          device=theta_b.device)  # add an additional dimension for the cap of the screw driver
            # the cap does not matter for the task, but needs to be included in the state for the model
            theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)
        full_q = partial_to_full_state(q_b, fingers=self.fingers, arm_dof=self.arm_dof)
        ret_scene = self.contact_scene.scene_collision_check(full_q, env_q=theta_b,
                                                             compute_gradient=True,
                                                             compute_hessian=False)
        full_robot_dof = self.arm_dof + 16
        for i, finger in enumerate(self.fingers):
            self.data[finger] = {}
            self.data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, self.T + 1)
            # reshape and throw away data for unused fingers
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, self.T + 1, full_robot_dof)[
                :, :, self.all_joint_index]

            # contact jacobian
            contact_jacobian = ret_scene.get('contact_jacobian', None)
            self.data[finger]['contact_jacobian'] = contact_jacobian[:, i].reshape(N, self.T + 1, 3, full_robot_dof)[
                :, :, :, self.all_joint_index]

            # contact hessian
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_hessian = contact_hessian[:, i].reshape(N, self.T + 1, 3, full_robot_dof, full_robot_dof)[
                :, :, :, self.all_joint_index]
            contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            # contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # shape (N, T+1, 3, 8, 8)

            # gradient of contact point
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
            d_contact_loc_dq = d_contact_loc_dq[:, i].reshape(N, self.T + 1, 3, full_robot_dof)[
                :, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            self.data[finger]['closest_pt_q_grad'] = d_contact_loc_dq
            self.data[finger]['contact_hessian'] = contact_hessian
            self.data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][
                :, i]  # the contact points are in the robot frame
            self.data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

            # gradient of contact normal
            self.data[finger]['dnormal_dq'] = ret_scene['dnormal_dq'][:, i].reshape(N, self.T + 1, 3, full_robot_dof)[
                :, :, :, self.all_joint_index]  # [:, :, :,
            # self.all_joint_index]

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :self.obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :self.obj_dof]
            dJ_dq = contact_hessian
            self.data[finger]['dJ_dq'] = dJ_dq  # Jacobian of the contact point
        if self.collision_checking:
            self.data['allegro_hand_hitosashi_finger_finger_link_2'] = {}
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['sdf'] = ret_scene['sdf'][:, -2].reshape(N,
                                                                                                              self.T + 1)
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['grad_sdf'] = \
                grad_g_q[:, -2].reshape(N, self.T + 1, full_robot_dof)[:, :, self.all_joint_index]
            self.data['allegro_hand_hitosashi_finger_finger_link_2']['grad_env_sdf'] = ret_scene['grad_env_sdf'][
                :, -2, :self.obj_dof]

            self.data['allegro_hand_hitosashi_finger_finger_link_3'] = {}
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['sdf'] = ret_scene['sdf'][:, -1].reshape(N,
                                                                                                              self.T + 1)
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['grad_sdf'] = \
                grad_g_q[:, -1].reshape(N, self.T + 1, full_robot_dof)[:, :, self.all_joint_index]
            self.data['allegro_hand_hitosashi_finger_finger_link_3']['grad_env_sdf'] = ret_scene['grad_env_sdf'][
                :, -1, :self.obj_dof]

    @all_finger_constraints
    def _contact_constraints(self, xu, finger_name, compute_grads=True, compute_hess=False, terminal=False):
        """
            Computes contact constraints
            constraint that sdf value is zero
        """
        N, T, _ = xu.shape
        # Retrieve pre-processed data
        ret_scene = self.data[finger_name]
        g = ret_scene.get('sdf').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_q = ret_scene.get('grad_sdf', None)
        hess_g_q = ret_scene.get('hess_sdf', None)
        grad_g_theta = ret_scene.get('grad_env_sdf', None)
        hess_g_theta = ret_scene.get('hess_env_sdf', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)
        # g = g + 2e-3

        # If terminal, only consider last state
        if terminal:
            g = g[:, -1].reshape(N, 1)

        if compute_grads:
            T_range = torch.arange(T, device=xu.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=xu.device)
            grad_g[:, T_range, T_range, :self.robot_dof] = grad_g_q[:, 1:]
            # is valve in state
            if self.dx == self.robot_dof + self.obj_dof:
                grad_g[:, T_range, T_range, self.robot_dof: self.robot_dof + self.obj_dof] = \
                    grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
            if terminal:
                grad_g = grad_g[:, -1].reshape(N, 1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None

    def _cost(self, xu, start, goal):
        state = xu[:, :self.dx]
        state = torch.cat([start[:self.dx].reshape(1, self.dx), state], dim=0)  # combine the first time step into it
        action = xu[:, self.dx:]
        action_cost = torch.sum(action ** 2)
        smoothness_cost = 10 * torch.sum((state[1:] - state[:-1]) ** 2)
        # smoothness_cost += 1000 *  torch.sum((state[1:, :self.arm_dof] - state[:-1, :self.arm_dof]) ** 2) # penalize the arm movement
        smoothness_cost += 100 * torch.sum(
            (state[1:, :self.arm_dof] - state[:-1, :self.arm_dof]) ** 2)  # penalize the arm movement
        return smoothness_cost + 10 * action_cost

    def _con_eq(self, xu, compute_grads=True, compute_hess=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess,
                                                                              terminal=True)
        return g_contact, grad_g_contact, hess_g_contact

    def _con_ineq(self, x, compute_grads=True, compute_hess=False):
        return None, None, None


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

@dataclass
class AllegroManipEnvCfg:
    # Simulation & runtime
    decimation = 2  # control at sim_dt * decimation
    episode_length_s = 10.0  # per-episode horizon (seconds)

    # Spaces (update if you change obs/action later)
    action_space = 16  # Allegro has 16 actuated joints
    observation_space = 64  # joint pos/vel + object pose/vel, etc.
    state_space = 0  # no privileged state by default

    # Allegro hand with Xela sensors
    hand_body_names: list[str] = field(default_factory=list)
    fingertip_body_names: list[str] = field(default_factory=list)
    finger_ee_names: dict[str, str] = field(default_factory=dict)

    # Actuated fingers
    fingers: list[str] = field(default_factory=list)  # ['index', 'middle', 'ring', 'thumb']

    arm_type: str = field(default_factory=str)
    hand_model_path: str = field(default_factory=str)

    # robot
    robot_cfg: dict[str, Any] = field(default_factory=dict)
    actuated_joint_names: list[str] = field(default_factory=list)

    # Default joint targets used at reset
    default_q: float = 0.5

    def __post_init__(self):
        self.hand_model_path: str = f"{ALLEGRO_MODEL_DIR}/allegro_xela_right.urdf"
        self.robot_cfg = {"init_state": [0] * 7}
        fingertip_body_names = [
            "allegro_hand_hitosashi_finger_finger_0_aftc_base_link",
            "allegro_hand_naka_finger_finger_1_aftc_base_link",
            "allegro_hand_kusuri_finger_finger_2_aftc_base_link",
            "allegro_hand_oya_finger_3_aftc_base_link",
        ]
        self.finger_ee_names = {
            'index': fingertip_body_names[0],
            'middle': fingertip_body_names[1],
            'ring': fingertip_body_names[2],
            'thumb': fingertip_body_names[3],
        }

        # Actuated fingers
        # NOTE: Order does not matter here!
        if 'index' in self.fingers:
            self.actuated_joint_names += [
                'allegro_hand_hitosashi_finger_finger_joint_0',
                'allegro_hand_hitosashi_finger_finger_joint_1',
                'allegro_hand_hitosashi_finger_finger_joint_2',
                'allegro_hand_hitosashi_finger_finger_joint_3']

        if 'middle' in self.fingers:
            self.actuated_joint_names += [
                'allegro_hand_naka_finger_finger_joint_4',
                'allegro_hand_naka_finger_finger_joint_5',
                'allegro_hand_naka_finger_finger_joint_6',
                'allegro_hand_naka_finger_finger_joint_7']

        if 'ring' in self.fingers:
            self.actuated_joint_names += [
                'allegro_hand_kusuri_finger_finger_joint_8',
                'allegro_hand_kusuri_finger_finger_joint_9',
                'allegro_hand_kusuri_finger_finger_joint_10',
                'allegro_hand_kusuri_finger_finger_joint_11']

        if 'thumb' in self.fingers:
            self.actuated_joint_names += [
                'allegro_hand_oya_finger_joint_12',
                'allegro_hand_oya_finger_joint_13',
                'allegro_hand_oya_finger_joint_14',
                'allegro_hand_oya_finger_joint_15']


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class AllegroManipEnv(ScrewDriverRotateTask):
    """Allegro-hand manipulation task in Isaac Lab's direct RL workflow.

    Observations
    -----------
    Concatenated vector with:
    - Allegro joint positions/velocities (normalized)
    - Object pose (p, q) and velocities
    - (Optional) fingertip poses

    Actions
    -------
    - By default: joint position deltas in [-1, 1], scaled to per-joint range.
      Switch to velocity control by changing `_apply_action`.
    """

    # TODO: Move this to cfg.robot_cfg['init_state']
    HAND_BASE_POSE = [np.array([0, 0.0, 0.2]),
                      np.array([0.000, -0.707, 0.0, 0.707])]  # np.array([1.0, 0.0, 0.0, 0.0]
    BASE_BODY_NAME = "allegro_hand_base_link"

    def get_assets(self) -> Dict[str, bytes]:
        assets = super().get_assets()
        models_path = epath.Path(ROOT) / "models"
        path = models_path / "allegro_xela"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "mesh")
        mjx_env.update_assets(assets, path / "mesh" / "allegro")
        return assets

    def __init__(self, name: str,
                 task_cfg: dict, cfg: AllegroManipEnvCfg,
                 xml_path=epath.Path(
                     ROOT) / "models" / "allegro_xela" / "scene_allegro_xela_right_rotate_cuboid.xml",
                 warp_enabled: bool = False,
                 device: str = 'cuda:0') -> None:
        """Load the MuJoCo model and set task parameters."""

        super().__init__(name, warp_enabled, xml_path)

        # Config
        self.task_cfg = task_cfg
        self.manip_cfg = cfg
        self.device = device

        # Robot init transf in world
        self.robot_init_transf = tf.Transform3d(pos=torch.tensor(self.HAND_BASE_POSE[0], device=device).float(),
                                                rot=torch.tensor(self.HAND_BASE_POSE[1], device=device).float(),
                                                device=device)

        # Others
        self.finger_names = cfg.fingers
        self.finger_to_joint_index = {
            'index': [0, 4, 5, 6],
            'middle': [1, 7, 8, 9],
            'ring': [2, 10, 11, 12],
            'thumb': [3, 13, 14, 15]
        }

        self._rb_states, self.rb_states = None, None
        self._actor_rb_states, self.actor_rb_states = None, None
        self._dof_states, self.dof_states = None, None
        self._q, self._qd = None, None
        self._ft_data, self.ft_data = None, None
        # self._forces, self.forces = None, None
        # self._jacobian, self.jacobian = None, None
        self.J_ee = None
        self._massmatrix, self.M = None, None
        self.default_dof_pos = None

        self.save_image_fpath = None
        self.frame_id = 0

        # Object
        self.object_data = self._mj_data.body(self._obj_name)

    def get_state(self) -> dict[str, Any]:
        arm_q = {
            'arm_q': torch.zeros(1).float()
        }

        # Finger joint positions (map finger names to their joint indices in self.finger_to_joint_index)
        finger_q = {
            f"{finger}_q": torch.tensor(self._mj_data.qpos[self.finger_to_joint_index[finger]].copy(),
                                        device=self.device).float() for finger in self.finger_names
        }

        # Finger end-effector positions via body links in Articulation data
        finger_ee_pos = {
            f"{finger}_ee_pos": torch.tensor(
                self._mj_data.body(self.manip_cfg.finger_ee_names[finger]).xpos.copy(),
                device=self.device).float() for finger in self.finger_names
        }

        # Merge results
        results = {**arm_q, **finger_q, **finger_ee_pos}
        return results

    @cached_property
    def obj_dof_code(self) -> list[int]:
        dofs = [0] * 6
        obj = self.mj_model.body(self._obj_name)
        for i in range(int(obj.jntnum)):
            jnt_id = obj.jntadr[i]
            jnt_type = self.mj_model.jnt_type[jnt_id]
            jnt_axis = self.mj_model.jnt_axis[jnt_id]
            match jnt_type:
                case mujoco.mjtJoint.mjJNT_FREE:
                    return [1] * 6
                case mujoco.mjtJoint.mjJNT_SLIDE:
                    for j in range(3):
                        dofs[j] = int(jnt_axis[j] == 1)
                case mujoco.mjtJoint.mjJNT_HINGE:
                    for j in range(3, 6):
                        dofs[j] = int(jnt_axis[j % 3] == 1)
        return dofs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor, wrap_to_2pi: bool = False) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x,y,z,w) to yaw angle around Z.
    Expects shape (..., 4) in IsaacLab/Isaac Sim convention (x,y,z,w).
    Returns angles in radians shape (...,).
    """
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  assuming q=(x,y,z,w)
    x, y, z, w = q.unbind(-1)
    num = 2 * (w * z + x * y)
    den = 1 - 2 * (y * y + z * z)
    return torch.atan2(num, den)


def euler_to_quat(euler):
    matrix = tf.euler_angles_to_matrix(euler, convention='XYZ')
    quat = tf.matrix_to_quaternion(matrix)
    return quat


def euler_to_angular_velocity(current_euler, next_euler):
    # using matrix

    current_mat = tf.euler_angles_to_matrix(current_euler, convention='XYZ')
    next_mat = tf.euler_angles_to_matrix(next_euler, convention='XYZ')
    dmat = next_mat - current_mat
    omega_mat = dmat @ current_mat.transpose(-1, -2)
    omega_x = (omega_mat[..., 2, 1] - omega_mat[..., 1, 2]) / 2
    omega_y = (omega_mat[..., 0, 2] - omega_mat[..., 2, 0]) / 2
    omega_z = (omega_mat[..., 1, 0] - omega_mat[..., 0, 1]) / 2
    omega = torch.stack((omega_x, omega_y, omega_z), dim=-1)

    # R.from_euler('XYZ', current_euler.cpu().detach().numpy().reshape(-1, 3)).as_quat().reshape(3, 12, 4)

    # quaternion
    # current_quat = euler_to_quat(current_euler)
    # next_quat = euler_to_quat(next_euler)
    # dquat = next_quat - current_quat
    # con_quat = - current_quat # conjugate
    # con_quat[..., 0] = current_quat[..., 0]
    # omega = 2 * tf.quaternion_raw_multiply(dquat, con_quat)[..., 1:]
    return omega
