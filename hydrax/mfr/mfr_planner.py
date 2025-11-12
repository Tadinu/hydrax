import os
import logging
import time
from typing import cast

logging.getLogger().setLevel(logging.INFO)

# !NOTE: All related to Isaac must be imported after [AppLauncher]

# import pytorch_volumetric as pv
import pytorch_kinematics as pk

# torch
import torch

torch.set_printoptions(precision=2, sci_mode=False)

# hydrax
from hydrax.mfr.allegro_env import (AllegroContactProblem, AllegroManipEnv, PositionControlConstrainedSVGDMPC)
from hydrax.mfr.allegro_cuboid_turning import AllegroCuboidTurning
from hydrax.mfr.allegro_cuboid_alignment_w_force import AllegroCuboidAlignment
from hydrax.mfr.allegro_cuboid_turning_env import AllegroCuboidTurningEnv, AllegroCuboidTurningCfg
from hydrax.mfr.allegro_cuboid_alignment_env import AllegroCuboidAlignmentEnv, AllegroCuboidAlignmentCfg
from hydrax.mfr.allegro_reorientation import AllegroReorientation
from hydrax.mfr.allegro_valve_turning import AllegroValveTurning
from hydrax.mfr.allegro_valve_turning_env import AllegroValveTurningEnv, AllegroValveTurningCfg
from hydrax.mfr.allegro_screwdriver import AllegroScrewdriver
from hydrax.mfr.allegro_screwdriver_env import AllegroScrewdriverEnv, AllegroScrewdriverCfg
from hydrax.mfr.utils.allegro_utils import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_task(task: str, task_config: dict) -> AllegroManipEnv:
    if task == 'screwdriver_turning':
        return AllegroScrewdriverEnv(name=task, task_cfg=task_config,
                                     # control_mode='joint_impedance',
                                     # viewer=True,
                                     # steps_per_action=60,
                                     # friction_coefficient=1.0,
                                     # device=task_config['sim_device'],
                                     # video_save_path=img_save_dir,
                                     # joint_stiffness=task_config['kp'],
                                     # gradual_control=task_config['gradual_control'],
                                     # gravity=task_config['gravity']
                                     )
    elif task == 'valve_turning':
        return AllegroValveTurningEnv(name=task, task_cfg=task_config,
                                      # control_mode='joint_impedance',
                                      # viewer=True,
                                      # steps_per_action=60,
                                      # friction_coefficient=1.0,
                                      # device=task_config['sim_device'],
                                      # valve_type=task_config['object_type'],
                                      # video_save_path=img_save_dir,
                                      # joint_stiffness=task_config['kp'],
                                      # gravity=task_config['gravity'],
                                      # random_robot_pose=task_config['random_robot_pose']
                                      )
    elif task == 'cuboid_turning':
        return AllegroCuboidTurningEnv(name=task, task_cfg=task_config)
    elif task == 'cuboid_alignment':
        return AllegroCuboidAlignmentEnv(name=task, task_cfg=task_config)
    elif task == 'reorientation':
        """
        return AllegroReorientationEnv()
        """
    return None


class MFRPlanner(object):

    def __init__(self, env: AllegroManipEnv, task_config: dict, args):
        task_config['task'] = args.task

        # Task
        self.task: AllegroManipEnv = env
        self.device = task_config['device']

        # Configs
        #
        # 1- Set up the kinematic chain
        self.robot_dof = 4 * len(task_config['fingers'])
        hand_model_path = env.manip_cfg.hand_model_path
        if hand_model_path.endswith('.xml'):
            self.chain = pk.build_chain_from_mjcf(hand_model_path).to(device=self.device)
        else:
            self.chain = pk.build_chain_from_urdf(open(hand_model_path).read()).to(device=self.device)
        task_config['chain'] = self.chain

        # 2- Goal
        task_config['goal'] = torch.tensor(task_config['goal'], device=self.device).float()

        # 3- Controller
        task_config.update(task_config['controllers'])
        task_config.pop('controllers')

        # 4- Object
        # NOTE: this is true for the tasks we have now. We need to pay attention if the root joint is not the root of the asset
        task_config['object_location'] = torch.tensor(self.task.object_data.xpos,
                                                      device=self.device).float()
        self.obj_dof = sum(env.obj_dof_code)
        task_config['obj_dof'] = self.obj_dof
        # Only turn the screwdriver once, compensating for the screwdriver cap
        self.obj_joint_dim = 1 if task_config['object_type'] == 'screwdriver' else 0

        # 5- Trials
        self.trial_count = 0
        self.fpath = pathlib.Path(
            f'{CURRENT_DIR}/data/experiments/{task_config["experiment_name"]}/trial_{self.trial_count + 1}')
        pathlib.Path.mkdir(self.fpath, parents=True, exist_ok=True)
        if task_config['visualize']:
            self.task.frame_fpath = self.fpath
            self.task.frame_id = 0
        else:
            self.task.frame_fpath = None
            self.task.frame_id = None

        # Pregrasp
        self.task.reset()
        self.task_config: dict = task_config
        self.pregrasp_action_list = []
        self.pregrasp()

    def pregrasp(self):
        cfg = self.task_config
        start = self.task.get_state().to(device=cfg['device'])

        # Pregrasp problem
        task = cfg['task']
        pregrasp_flag = not (task == 'reorientation')
        if pregrasp_flag:
            print("Pregrasping...")
            pregrasp_succ = False
            while pregrasp_succ == False:
                pregrasp_dx = pregrasp_du = self.robot_dof
                pregrasp_problem = AllegroContactProblem(
                    dx=pregrasp_dx,
                    du=pregrasp_du,
                    start=start,  # start[:pregrasp_dx + obj_dof]
                    goal=None,
                    T=4,
                    chain=cfg['chain'],
                    device=cfg['device'],
                    object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                    object_model_path=self.task.manip_cfg.object_model_path,
                    object_type=cfg['object_type'],
                    world_trans=self.task.robot_init_transf,
                    fingers=cfg['fingers'],
                    obj_dof_code=self.task.obj_dof_code,
                    obj_joint_dim=self.obj_joint_dim,
                )
                self.task.manip_problem = pregrasp_problem

                pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, cfg)
                pregrasp_planner.warmup_iters = 50

                start_time = time.time()
                best_traj, _ = pregrasp_planner.step(start)
                print(f"pregrasp solve time: {time.time() - start_time}")

                if cfg['visualize_plan']:
                    traj_for_viz = best_traj[:, :pregrasp_problem.dx]
                    tmp = start[pregrasp_dx:pregrasp_dx + self.obj_dof].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
                    tmp_2 = torch.zeros((traj_for_viz.shape[0], 1)).to(traj_for_viz.device)  # the top jint
                    traj_for_viz = torch.cat((traj_for_viz, tmp, tmp_2), dim=1)
                    viz_fpath = pathlib.PurePath.joinpath(self.fpath, "pregrasp")
                    img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                    gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                    pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                    pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                    visualize_trajectory(traj_for_viz, pregrasp_problem.contact_scene, viz_fpath,
                                         pregrasp_problem.fingers, pregrasp_problem.obj_dof + self.obj_joint_dim,
                                         arm_dof=0)

                for x in best_traj[:]:
                    action = x.reshape(-1, x.shape[0]).to(device=self.task.device)  # move the rest fingers
                    self.pregrasp_action_list.append(action.cpu().numpy().squeeze())
                if cfg['mode'] == 'simulation':
                    pregrasp_succ = self.task.manip_problem.check_validity(self.task.get_state())
                if pregrasp_succ == False:
                    print("pregrasp failed, replanning")
                    self.task.reset()

    def plan(self, step_env: bool = True, kinematics_only: bool = False):
        cfg = self.task_config
        num_fingers = len(cfg['fingers'])
        robot_dof = 4 * num_fingers
        obj_dof = cfg['obj_dof']

        state = self.task.get_state()
        start = state.to(device=cfg['device'])
        task = cfg['task']
        if task == 'screwdriver_turning':
            self.task = cast(AllegroScrewdriverEnv, self.task)
            self.task.manip_cfg = cast(AllegroScrewdriverCfg, self.task.manip_cfg)
            manipulation_problem = AllegroScrewdriver(
                start=start[:robot_dof + obj_dof],
                goal=cfg['goal'],
                T=cfg['T'],
                chain=cfg['chain'],
                device=cfg['device'],
                object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                object_model_path=self.task.manip_cfg.object_model_path,
                object_location=cfg['object_location'],
                object_type=cfg['object_type'],
                friction_coefficient=cfg['friction_coefficient'],
                finger_stiffness=cfg['kp'],
                arm_stiffness=500,
                world_trans=self.task.robot_init_transf,
                fingers=cfg['fingers'],
                force_balance=False,
                collision_checking=cfg['collision_checking'],
                obj_gravity=cfg['obj_gravity'],
                contact_region=cfg['contact_region'],
            )
        elif task == 'valve_turning':
            self.task = cast(AllegroValveTurningEnv, self.task)
            self.task.manip_cfg = cast(AllegroValveTurningCfg, self.task.manip_cfg)
            manipulation_problem = AllegroValveTurning(
                start=start,
                goal=cfg['goal'],
                T=cfg['T'],
                chain=cfg['chain'],
                device=cfg['device'],
                object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                object_model_path=self.task.manip_cfg.object_model_path,
                object_location=cfg['object_location'],
                object_type=cfg['object_type'],
                friction_coefficient=cfg['friction_coefficient'],
                world_trans=self.task.robot_init_transf,
                fingers=cfg['fingers'],
                obj_dof_code=self.task.obj_dof_code,
            )
        elif task == 'cuboid_turning':
            self.task = cast(AllegroCuboidTurningEnv, self.task)
            self.task.manip_cfg = cast(AllegroCuboidTurningCfg, self.task.manip_cfg)
            manipulation_problem = AllegroCuboidTurning(
                start=start,
                goal=cfg['goal'],
                T=cfg['T'],
                chain=cfg['chain'],
                object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                object_model_path=self.task.manip_cfg.object_model_path,
                world_trans=self.task.robot_init_transf,
                object_location=cfg['object_location'],
                object_type=cfg['object_type'],
                friction_coefficient=cfg['friction_coefficient'],
                device=cfg['device'],
                fingers=cfg['fingers'],
                obj_dof_code=self.task.obj_dof_code,
                obj_gravity=cfg['obj_gravity'],
            )
        elif task == 'cuboid_alignment':
            self.task = cast(AllegroCuboidAlignmentEnv, self.task)
            self.task.manip_cfg = cast(AllegroCuboidAlignmentCfg, self.task.manip_cfg)
            manipulation_problem = AllegroCuboidAlignment(
                start=start,
                goal=cfg['goal'],
                T=cfg['T'],
                chain=cfg['chain'],
                device=cfg['device'],
                object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                object_model_path=self.task.manip_cfg.object_model_path,
                wall_asset_pos=self.task.wall_pose,
                wall_dims=self.task.wall_dims,
                object_location=cfg['object_location'],
                object_type=cfg['object_type'],
                friction_coefficient=cfg['friction_coefficient'],
                world_trans=self.task.robot_init_transf,
                fingers=cfg['fingers'],
                obj_gravity=cfg['obj_gravity'],
                collision_checking=cfg['collision_checking'],
            )
        elif task == 'reorientation':
            manipulation_problem = AllegroReorientation(
                start=start,
                goal=cfg['goal'],
                T=cfg['T'],
                chain=cfg['chain'],
                object_asset_pos=torch.tensor(self.task.object_data.xpos.copy()),
                object_model_path=self.task.manip_cfg.object_model_path,
                world_trans=self.task.robot_init_transf,
                object_location=cfg['object_location'],
                object_type=cfg['object_type'],
                friction_coefficient=cfg['friction_coefficient'],
                device=cfg['device'],
                fingers=cfg['fingers'],
                obj_dof_code=self.task.obj_dof_code,
                obj_gravity=cfg['obj_gravity'],
            )
        else:
            raise ValueError(f'Unknown task: {task}')

        print("---------------------------------------------------")
        print("Start planning:", task)
        manipulation_planner = PositionControlConstrainedSVGDMPC(manipulation_problem, cfg)
        actual_trajectory = []
        duration = 0

        action_full = torch.zeros(self.task.num_ctrls, device=self.task.device)
        for k in range(cfg['num_steps']):
            state = self.task.get_state()
            start = state.to(device=cfg['device'])
            current_theta = state[-obj_dof:].detach().cpu().numpy()
            actual_trajectory.append(start.clone())
            start_time = time.time()
            best_traj, trajectories = manipulation_planner.step(start)

            solve_time = time.time() - start_time
            print(f"Planner solving time: {solve_time}", k)
            if k == 0:
                warmup_time = solve_time
            else:
                duration += solve_time
            planned_theta_traj = best_traj[:, robot_dof: robot_dof + obj_dof].detach().cpu().numpy()
            print(f"current theta: {current_theta}")
            print(f"planned theta: {planned_theta_traj}")

            if cfg['visualize_plan']:
                traj_for_viz = best_traj[:, :manipulation_problem.dx]
                traj_for_viz = torch.cat((start[:manipulation_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                # compensate for the screwdriver cap
                obj_joint_dim = 1 if cfg['object_type'] == 'screwdriver' else 0
                if obj_joint_dim > 0:
                    tmp = torch.zeros((traj_for_viz.shape[0], obj_joint_dim),
                                      device=best_traj.device)  # add the joint for the screwdriver cap
                    traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)

                viz_fpath = pathlib.PurePath.joinpath(self.task.frame_fpath, f"timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, manipulation_problem.contact_scene, viz_fpath,
                                     manipulation_problem.fingers, manipulation_problem.obj_dof + obj_joint_dim,
                                     arm_dof=0)

            x = best_traj[0, :manipulation_problem.dx + manipulation_problem.du]
            x = x.reshape(1, manipulation_problem.dx + manipulation_problem.du)
            manipulation_problem._preprocess(best_traj.unsqueeze(0))
            equality_constr_dict = manipulation_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False,
                                                                compute_hess=False, verbose=True)
            inequality_constr_dict = manipulation_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False,
                                                                    compute_hess=False, verbose=True)
            print("--------------------------------------")

            action = x[:, manipulation_problem.dx:manipulation_problem.dx + manipulation_problem.du].to(
                device=self.task.device)
            print("planned force")
            print(action[:, robot_dof:].reshape(num_fingers + cfg['num_env_force'], 3))
            action = action[:, :robot_dof]
            # NOTE: this is required since we define action as delta action
            action = action + start.unsqueeze(0)[:, :robot_dof].to(action.device)
            if step_env:
                print("Planner stepping", k, action.reshape(num_fingers, 4))
                action_full[:action.shape[1]] = action
                self.task.step(action_full.cpu().numpy().squeeze(), kinematics_only)
        return action_full
