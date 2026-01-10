"""An Adroit arm environment with ball relocation task using the Gymnasium API.
https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/adroit_hand/adroit_relocate.py

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Optional, Any, Dict, Sequence, Union
from etils import epath

from ml_collections import config_dict

import numpy as np
import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx
import mujoco_warp as mjw

# mujoco playground
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

# hydrax
from hydrax.task_base import Task
from hydrax import ROOT, BackendType

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}


class AdroitHandRelocateEnv(mjx_env.MjxEnv, Task):
    """
    ## Description

    This environment was introduced in ["Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"](https://arxiv.org/abs/1709.10087)
    by Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.

    The environment is based on the [Adroit manipulation platform](https://github.com/vikashplus/Adroit), a30 degree of freedom system which consists of a 24 degrees of freedom
    ShadowHand and a 6 degree of freedom arm. The task to be completed consists on moving the blue ball to the green target. The positions of the ball and target are randomized over the entire
    workspace. The task will be considered successful when the object is within epsilon-ball of the target.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (30,), float32)`. The control actions are absolute angular positions of the Adroit hand joints. The input of the control actions is set to a range between -1 and 1 by scaling the real actuator angle ranges in radians.
    The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Linear translation of the full arm in x direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTx                           | slide | position (m)|
    | 1   | Linear translation of the full arm in y direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTy                           | slide | position (m)|
    | 2   | Linear translation of the full arm in z direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTz                           | slide | position (m)|
    | 3   | Angular up and down movement of the full arm                                            | -1          | 1           | -0.4 (rad)   | 0.25 (rad)  | A_ARRx                           | hinge | angle (rad) |
    | 4   | Angular left and right and down movement of the full arm                                | -1          | 1           | -0.3 (rad)   | 0.3 (rad)   | A_ARRy                           | hinge | angle (rad) |
    | 5   | Roll angular movement of the full arm                                                   | -1          | 1           | -1.0 (rad)   | 2.0 (rad)   | A_ARRz                           | hinge | angle (rad) |
    | 6   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.524 (rad) | 0.175 (rad) | A_WRJ1                           | hinge | angle (rad) |
    | 7   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.79 (rad)  | 0.61 (rad)  | A_WRJ0                           | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_FFJ3                           | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ2                           | hinge | angle (rad) |
    | 10  | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ1                           | hinge | angle (rad) |
    | 11  | Angular position of the DIP joint of the forefinger                                     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ0                           | hinge | angle (rad) |
    | 12  | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_MFJ3                           | hinge | angle (rad) |
    | 13  | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ2                           | hinge | angle (rad) |
    | 14  | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ1                           | hinge | angle (rad) |
    | 15  | Angular position of the DIP joint of the middle finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ0                           | hinge | angle (rad) |
    | 16  | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_RFJ3                           | hinge | angle (rad) |
    | 17  | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ2                           | hinge | angle (rad) |
    | 18  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ1                           | hinge | angle (rad) |
    | 19  | Angular position of the DIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ0                           | hinge | angle (rad) |
    | 20  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.7(rad)    | A_LFJ4                           | hinge | angle (rad) |
    | 21  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_LFJ3                           | hinge | angle (rad) |
    | 22  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ2                           | hinge | angle (rad) |
    | 23  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ1                           | hinge | angle (rad) |
    | 24  | Angular position of the DIP joint of the little finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ0                           | hinge | angle (rad) |
    | 25  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | A_THJ4                           | hinge | angle (rad) |
    | 26  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.3 (rad)   | A_THJ3                           | hinge | angle (rad) |
    | 27  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.26 (rad)  | 0.26(rad)   | A_THJ2                           | hinge | angle (rad) |
    | 28  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.52 (rad)  | 0.52 (rad)  | A_THJ1                           | hinge | angle (rad) |
    | 29  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | A_THJ0                           | hinge | angle (rad) |


    ## Observation Space

    The observation space is of the type `Box(-inf, inf, (39,), float64)`. It contains information about the angular position of the finger joints, the pose of the palm of the hand, as well as kinematic information about the ball and target.

    | Num | Observation                                                                 | Min    | Max    | Joint Name (in corresponding XML file) | Site/Body Name (in corresponding XML file) | Joint Type| Unit                     |
    |-----|-----------------------------------------------------------------------------|--------|--------|----------------------------------------|--------------------------------------------|-----------|------------------------- |
    | 0   | Translation of the arm in the x direction                                   | -Inf   | Inf    | ARTx                                   | -                                          | slide     | position (m)             |
    | 1   | Translation of the arm in the y direction                                   | -Inf   | Inf    | ARTy                                   | -                                          | slide     | position (m)             |
    | 2   | Translation of the arm in the z direction                                   | -Inf   | Inf    | ARTz                                   | -                                          | slide     | position (m)             |
    | 3   | Angular position of the vertical arm joint                                  | -Inf   | Inf    | ARRx                                   | -                                          | hinge     | angle (rad)              |
    | 4   | Angular position of the horizontal arm joint                                | -Inf   | Inf    | ARRy                                   | -                                          | hinge     | angle (rad)              |
    | 5   | Roll angular value of the arm                                               | -Inf   | Inf    | ARRz                                   | -                                          | hinge     | angle (rad)              |
    | 6   | Angular position of the horizontal wrist joint                              | -Inf   | Inf    | WRJ1                                   | -                                          | hinge     | angle (rad)              |
    | 7   | Angular position of the vertical wrist joint                                | -Inf   | Inf    | WRJ0                                   | -                                          | hinge     | angle (rad)              |
    | 8   | Horizontal angular position of the MCP joint of the forefinger              | -Inf   | Inf    | FFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 9   | Vertical angular position of the MCP joint of the forefinge                 | -Inf   | Inf    | FFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 10  | Angular position of the PIP joint of the forefinger                         | -Inf   | Inf    | FFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 11  | Angular position of the DIP joint of the forefinger                         | -Inf   | Inf    | FFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 12  | Horizontal angular position of the MCP joint of the middle finger           | -Inf   | Inf    | MFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 13  | Vertical angular position of the MCP joint of the middle finger             | -Inf   | Inf    | MFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 14  | Angular position of the PIP joint of the middle finger                      | -Inf   | Inf    | MFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 15  | Angular position of the DIP joint of the middle finger                      | -Inf   | Inf    | MFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 16  | Horizontal angular position of the MCP joint of the ring finger             | -Inf   | Inf    | RFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 17  | Vertical angular position of the MCP joint of the ring finger               | -Inf   | Inf    | RFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 18  | Angular position of the PIP joint of the ring finger                        | -Inf   | Inf    | RFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 19  | Angular position of the DIP joint of the ring finger                        | -Inf   | Inf    | RFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 20  | Angular position of the CMC joint of the little finger                      | -Inf   | Inf    | LFJ4                                   | -                                          | hinge     | angle (rad)              |
    | 21  | Horizontal angular position of the MCP joint of the little finger           | -Inf   | Inf    | LFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 22  | Vertical angular position of the MCP joint of the little finger             | -Inf   | Inf    | LFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 23  | Angular position of the PIP joint of the little finger                      | -Inf   | Inf    | LFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 24  | Angular position of the DIP joint of the little finger                      | -Inf   | Inf    | LFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 25  | Horizontal angular position of the CMC joint of the thumb finger            | -Inf   | Inf    | THJ4                                   | -                                          | hinge     | angle (rad)              |
    | 26  | Vertical Angular position of the CMC joint of the thumb finger              | -Inf   | Inf    | THJ3                                   | -                                          | hinge     | angle (rad)              |
    | 27  | Horizontal angular position of the MCP joint of the thumb finger            | -Inf   | Inf    | THJ2                                   | -                                          | hinge     | angle (rad)              |
    | 28  | Vertical angular position of the MCP joint of the thumb finger              | -Inf   | Inf    | THJ1                                   | -                                          | hinge     | angle (rad)              |
    | 29  | Angular position of the IP joint of the thumb finger                        | -Inf   | Inf    | THJ0                                   | -                                          | hinge     | angle (rad)              |
    | 30  | x positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 31  | y positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 32  | z positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 33  | x positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 34  | y positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 35  | z positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 36  | x positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 37  | y positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 38  | z positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |

    ## Rewards

    The environment can be initialized in either a `dense` or `sparse` reward variant.

    In the `dense` reward setting, the environment returns a `dense` reward function that consists of the following parts:
    - `get_to_ball`: increasing negative reward the further away the palm of the hand is from the ball. This is computed as the 3 dimensional Euclidean distance between both body frames.
        This penalty is scaled by a factor of `0.1` in the final reward.
    - `ball_off_table`: add a positive reward of 1 if the ball is lifted from the table (`z` greater than `0.04` meters). If this condition is met two additional rewards are added:
        - `make_hand_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the palm to the target ball position. This reward is scaled by a factor of `0.5`.
        -` make_ball_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the ball to its target position. This reward is also scaled by a factor of `0.5`.
    - `ball_close_to_target`: bonus of `10` if the ball's Euclidean distance to its target is less than `0.1` meters. Bonus of `20` if the distance is less than `0.05` meters.

    The `sparse` reward variant of the environment can be initialized by calling `gym.make('AdroitHandReloateSparse-v1')`.
    In this variant, the environment returns a reward of 10 for environment success and -0.1 otherwise.

    ## Starting State

    The ball is set randomly over the table at reset. The ranges of the uniform distribution from which the position is samples are `[-0.15,0.15]` for the `x` coordinate, and `[-0.15,0.3]` got the `y` coordinate.
    The target position is also sampled from uniform distributions with ranges `[-0.2,0.2]` for the `x` coordinate, `[-0.2,0.2]` for the `y` coordinate, and `[0.15,0.35]` for the `z` coordinate.

    The joint values of the environment are deterministically initialized to a zero.

    For reproducibility, the starting state of the environment can also be set when calling `env.reset()` by passing the `options` dictionary argument (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
    with the `initial_state_dict` key. The `initial_state_dict` key must be a dictionary with the following items:

    * `qpos`: jp.ndarray with shape `(36,)`, MuJoCo simulation joint positions
    * `qvel`: jp.ndarray with shape `(36,)`, MuJoCo simulation joint velocities
    * `obj_pos`: jp.ndarray with shape `(3,)`, cartesian coordinates of the ball object
    * `target_pos`: jp.ndarray with shape `(3,)`, cartesian coordinates of the goal ball location

    The state of the simulation can also be set at any step with the `env.set_env_state(initial_state_dict)` method.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 200 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 400 make the environment as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('AdroitHandRelocate-v1', max_episode_steps=400)
    ```

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

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
                    grasp_obj=4.0,
                    # Do not collide with the floor.
                    no_floor_collision=0.25,
                    # Arm stays close to target pose.
                    robot_target_qpos=0.3,
                )
            ),
        )
        return config

    def get_assets(self) -> Dict[str, bytes]:
        assets = {}
        path = epath.Path(ROOT) / "models" / "adroit_hand"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "resources" / "meshes")
        mjx_env.update_assets(assets, path / "resources" / "textures")
        return assets

    def __init__(self,
                 config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        Task.__init__(self, xml_path=epath.Path(ROOT) / "models" / "adroit_hand" / "adroit_relocate.xml",
                      sim_dt=config.sim_dt,
                      trace_sites=["S_fftip", "S_mftip", "S_rftip", "S_lftip", "S_thtip"])

        self._action_scale = config.action_scale

        self.ARM_JOINTS = [
            "ARTx",
            "ARTy",
            "ARTz",
            "ARRx",
            "ARRy",
            "ARRz",
            "WRJ1", "WRJ0",
        ]
        self.HAND_JOINTS = [
            "FFJ3", "FFJ2", "FFJ1", "FFJ0",
            "MFJ3", "MFJ2", "MFJ1", "MFJ0",
            "RFJ3", "RFJ2", "RFJ1", "RFJ0",
            "LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0",
            "THJ4", "THJ3", "THJ2", "THJ1", "THJ0"
        ]

        # Get sensor ids
        self.obj_position_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "obj_position"
        )
        self.obj_orientation_sensor = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SENSOR, "obj_orientation"
        )

        # Distance (m) beyond which we impose a high obj position cost
        self.delta = 0.015

    def _post_init(self) -> None:
        Task._post_init(self)

        # Robot-specifics
        all_joints = self.ARM_JOINTS + self.HAND_JOINTS
        self._robot_arm_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in self.ARM_JOINTS
        ])
        self._robot_qposadr = np.array([
            self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
            for j in all_joints
        ])
        self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl if keyframe else None
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

        # Hand-specifics
        self._init_hand()

        # Env-specifics
        self._target_obj_site = self._mj_model.site("target").id
        self._grasp_site = self._mj_model.site("S_grasp").id
        self._obj_body = self._mj_model.body(obj_name).id
        self._obj_geom = self.mj_model.geom(obj_name).id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body(obj_name).jntadr[0]
        ]
        self._floor_geom = self._mj_model.geom("floor").id
        self._init_q = self._mj_model.keyframe(keyframe).qpos if keyframe else None
        self._init_obj_pos = jp.array(
            self._init_q[self._obj_qposadr: self._obj_qposadr + 3],
            dtype=jp.float32,
        ) if self._init_q else None
        self._init_obj_quat = np.array(
            self._init_q[self._obj_qposadr + 3: self._obj_qposadr + 7],
            dtype=np.float32,
        ) if self._init_q else None

        self.act_mean = np.mean(self._mj_model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
                self._mj_model.actuator_ctrlrange[:, 1] - self._mj_model.actuator_ctrlrange[:, 0]
        )

    def _init_hand(self):
        ARM_GEOMS = ["C_forearm1", "C_wrist", "C_palm0", "C_palm1"]
        HAND_GEOMS = [
            "C_wrist",
            "C_palm0", "C_palm1",
            "C_ffproximal", "C_ffmiddle", "C_ffdistal",
            "C_mfproximal", "C_mfmiddle", "C_mfdistal",
            "C_rfproximal", "C_rfmiddle", "C_rfdistal",
            "C_lfmetacarpal", "C_lfproximal", "C_lfmiddle", "C_lfdistal",
            "C_thproximal", "C_thmiddle", "C_thdistal"
        ]
        arm_geoms = [self.mj_model.geom(n).id for n in ARM_GEOMS]
        hand_geoms = [self.mj_model.geom(n).id for n in HAND_GEOMS]
        self._full_geoms = arm_geoms + hand_geoms

        # change actuator sensitivity
        self._mj_model.actuator("A_WRJ0").gainprm[:3] = jp.array([10, 0, 0])
        self._mj_model.actuator("A_WRJ1").gainprm[:3] = jp.array([10, 0, 0])

        self._mj_model.actuator("A_WRJ0").biasprm[:3] = jp.array([0, -10, 0])
        self._mj_model.actuator("A_WRJ1").biasprm[:3] = jp.array([0, -10, 0])

        self._mj_model.actuator("A_FFJ3").gainprm[:3] = jp.array([1, 0, 0])
        self._mj_model.actuator("A_THJ0").gainprm[:3] = jp.array([1, 0, 0])

        self._mj_model.actuator("A_FFJ3").biasprm[:3] = jp.array([0, -1, 0])
        self._mj_model.actuator("A_THJ0").biasprm[:3] = jp.array([0, -1, 0])

    @property
    def xml_path(self) -> str:
        return self._xml_path.as_posix()

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    def _get_obj_position_err(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Position of the obj relative to the target grasp position."""
        sensor_adr = self.mjx_model.sensor_adr[self.obj_position_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 3]

    def _get_obj_orientation_err(self, state: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """Orientation of the obj relative to the target grasp orientation."""
        sensor_adr = self.mjx_model.sensor_adr[self.obj_orientation_sensor]
        return state.sensordata[sensor_adr: sensor_adr + 4]

    def running_cost(self, state: Union[mjx.Data, mjw.Data], control: jax.Array,
                     batch_idx: Optional[int] = -1) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_obj_position_err(state)
        squared_distance = jp.sum(jp.square(position_err[0:2]))  # ignore z
        position_cost = 0.1 * squared_distance + 100 * jp.maximum(
            squared_distance - self.delta ** 2, 0.0
        )

        orientation_err = self._get_obj_orientation_err(state)
        orientation_cost = jp.sum(jp.square(orientation_err))

        grasp_cost = 0.001 * jp.sum(jp.square(control))

        return position_cost + orientation_cost + grasp_cost

    def terminal_cost(self, data: Union[mjx.Data, mjw.Data]) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_err = self._get_obj_position_err(data)
        return 100 * jp.sum(jp.square(position_err))

    def _get_obs(self, data: Union[mjx.Data, mjw.Data], info: dict[str, Any]) -> jax.Array:
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = data.qpos
        obj_pos = data.xpos[self._obj_body]
        palm_pos = data.site_xpos[self._grasp_site]
        target_pos = data.site_xpos[self._target_obj_site]
        return jp.concatenate(
            [qpos[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos]
        )

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)
        # initialize data
        data = mjx_env.init(self._mjx_model)

        target_pos = (
                jax.random.uniform(
                    rng_target,
                    (3,),
                    minval=jp.array([-0.2, -0.2, 0.2]),
                    maxval=jp.array([0.2, 0.2, 0.4]),
                )
                + self._init_obj_pos
        )
        info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, state.info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in raw_rewards.items()
        }
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        state.metrics.update(
            **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
        )

        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def _get_reward(self, data: Union[mjx.Data, mjw.Data], info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        grasp_pos = data.site_xpos[self._grasp_site]
        pos_err = jp.linalg.norm(target_pos - box_pos)
        box_mat = data.xmat[self._obj_body]
        target_mat = data.site_xmat[self._target_obj_site]
        rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

        obj_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        grasp_obj = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - grasp_pos))
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Check for collisions with the floor
        floor_collision = [
            collision.geoms_colliding(data, self._floor_geom, g)
            for g in self._full_geoms
        ]
        floor_collided = sum(floor_collision) > 0
        no_floor_collision = (1 - floor_collided).astype(float)

        info["reached_obj"] = 1.0 * jp.maximum(
            info["reached_obj"],
            (jp.linalg.norm(box_pos - grasp_pos) < 0.012),
        )

        rewards = {
            "grasp_obj": grasp_obj,
            "obj_target": obj_target * info["reached_obj"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return rewards
