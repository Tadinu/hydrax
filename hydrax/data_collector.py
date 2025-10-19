from __future__ import annotations
from typing import TYPE_CHECKING
import json
import os
import time
import datetime
import h5py
import glob

import numpy as np

# hydrax
from hydrax import DATA_DIR

if TYPE_CHECKING:
    from hydrax.task_base import Task


class DataCollector:
    def __init__(self, task: Task, directory, collect_freq=1, save_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            task (Task): The task to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            save_freq (int): How frequently to save data to disk, in terms of environment steps.
        """
        self.task = task

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.successful = False  # stores success state of demonstration

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to save data to disk, in terms of environment steps
        self.save_freq = save_freq

        if not os.path.exists(directory):
            print("DataCollector: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._save_data()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)

        # NOTE: was previously self.env.model.get_xml(). Was causing the following issue in rare cases:
        # ValueError: Error: eigenvalues of mesh inertia violate A + B >= C
        # switching to self.env.sim.model.get_xml() does not create this issue
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        # self.task.reset_from_xml_string(self._current_task_instance_xml)
        # self.task.reset()
        # self.task.set_state_from_flattened(self._current_task_instance_state)
        # self.task.forward()

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollector: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # save the episode info to json file
        ep_meta_path = os.path.join(self.ep_directory, "ep_meta.json")
        with open(ep_meta_path, "w") as f:
            json.dump(self.task.get_ep_meta(), f)

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)

    def _save_data(self):
        """
        Method to save internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            successful=self.successful,
            env=self.task.name,
        )
        self.states = []
        self.action_infos = []
        self.successful = False

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        self._start_new_episode()

    def collect(self, action):
        """
        Collect data after action has been performed in the task.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.task.get_state().flatten()
            self.states.append(state)

            info = {}
            info["actions"] = np.array(action)
            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.task.check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.save_freq == 0:
            self._save_data()

    def save_demos_as_hdf5(demo_dir: str, env_info):
        """
        Save the demonstrations saved in @demo_dir into a single hdf5 file.

        The strucure of the hdf5 file is as follows.

        data (group)
            date (attribute) - date of collection
            time (attribute) - time of collection
            repository_version (attribute) - repository version used during collection
            env (attribute) - environment name on which demos were collected

            demo1 (group) - every demonstration has a group
                model_file (attribute) - model xml string for demonstration
                states (dataset) - flattened mujoco states
                actions (dataset) - actions applied during demonstration

            demo2 (group)
            ...

        Args:
            demo_dir (str): Path to the directory containing raw demonstrations.
            env_info (str): JSON-encoded string containing environment information,
                including controller and robot info
        """

        hdf5_path = os.path.join(DATA_DIR, "demo.hdf5")
        f = h5py.File(hdf5_path, "w")

        # store some metadata in the attributes of one group
        grp = f.create_group("data")

        num_eps = 0
        env_name = None  # will get populated at some point

        for ep_directory in os.listdir(demo_dir):
            state_paths = os.path.join(demo_dir, ep_directory, "state_*.npz")
            states = []
            actions = []
            success = False

            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])

                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or dic["successful"]

            if len(states) == 0:
                continue

            # Add only the successful demonstration to dataset
            if success:
                print("Demonstration is successful and has been saved")
                # Delete the last state. This is because when the DataCollector wrapper
                # recorded the states and actions, the states were recorded AFTER playing that action,
                # so we end up with an extra state at the end.
                del states[-1]
                assert len(states) == len(actions)

                num_eps += 1
                ep_data_grp = grp.create_group("demo_{}".format(num_eps))

                # store model xml as an attribute
                xml_path = os.path.join(demo_dir, ep_directory, "model.xml")
                with open(xml_path, "r") as f:
                    xml_str = f.read()
                ep_data_grp.attrs["model_file"] = xml_str

                # write datasets for states and actions
                ep_data_grp.create_dataset("states", data=np.array(states))
                ep_data_grp.create_dataset("actions", data=np.array(actions))
            else:
                print("Demonstration is unsuccessful and has NOT been saved")

        # write dataset attributes (metadata)
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info

        f.close()

    def playback_demos(self, ep_dir, max_fr=None):
        """Playback data from an episode.

        Args:
            ep_dir (str): The path to the directory containing data for an episode.
        """
        state_paths = os.path.join(ep_dir, "state_*.npz")

        # read states back, load them one by one, and render
        t = 0
        for state_file in sorted(glob(state_paths)):
            print(state_file)
            dic = np.load(state_file)
            states = dic["states"]
            for state in states:
                start = time.time()
                self.task.set_state_from_flattened(state)
                self.task.forward()
                t += 1
                if t % 100 == 0:
                    print(t)

                if max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1 / max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)

    def close(self):
        """
        Close, flushing left over data
        """
        if self.has_interaction:
            self._save_data()
