import multiprocessing as mp
import time
from multiprocessing import Event, shared_memory

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from mjmanip.robot.arm_hand import ArmHandDiffIK

"""
Utilities for asynchronous simulation, with the simulator and controller running
in separate processes. The controller runs as fast as possible, while the
simulator runs in real time.

This is more realistic than deterministic simulation, but supports a limited set
of features (e.g., no trace visualization, zero-order-hold interpolation only).
"""


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""

    def __init__(self, arr: np.ndarray, ctx: mp.context.BaseContext):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
            ctx: The multiprocessing context to use for shared memory.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        shared_arr[:] = arr[:]
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.lock = ctx.Lock()

    def __getitem__(self, key: int) -> np.ndarray:
        """Get an item from the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return np.copy(arr[key])  # Need to copy here to avoid segfaults

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Set an item in the shared array."""
        with self.lock:
            shm = shared_memory.SharedMemory(name=self.shm.name)
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            arr[key] = value

    def __str__(self) -> str:
        """Return the string representation of the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return str(arr)

    def __del__(self) -> None:
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()


class SharedMemoryMujocoData:
    """Helper class for passing mujoco data between concurrent processes."""

    def __init__(self, mj_data: mj.MjData, ctx: mp.context.BaseContext):
        """Create shared memory objects for state and control data.

        Note that this does not copy the full mj_data object, only those fields
        that we want to share between the simulator and controller.

        Args:
            mj_data: The mujoco data object to store in shared memory.
            ctx: The multiprocessing context to use.
        """
        # N.B. we use float32 to match JAX's default precision
        self.qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32), ctx
        )
        self.ref_arm_qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32), ctx
        )
        self.qvel = SharedMemoryNumpyArray(
            np.array(mj_data.qvel, dtype=np.float32), ctx
        )
        self.xpos = SharedMemoryNumpyArray(
            np.array(mj_data.xpos, dtype=np.float32), ctx
        )
        self.xquat = SharedMemoryNumpyArray(
            np.array(mj_data.xquat, dtype=np.float32), ctx
        )
        self.ctrl = SharedMemoryNumpyArray(
            np.zeros(mj_data.ctrl.shape, dtype=np.float32), ctx
        )
        self.wrist_ctrl = SharedMemoryNumpyArray(
            np.zeros((6,), dtype=np.float32), ctx
        )
        self.hand_ctrl = SharedMemoryNumpyArray(
            np.zeros((16,), dtype=np.float32), ctx
        )
        if len(mj_data.mocap_pos) > 0:
            self.mocap_pos = SharedMemoryNumpyArray(
                np.array(mj_data.mocap_pos, dtype=np.float32), ctx
            )
            self.mocap_quat = SharedMemoryNumpyArray(
                np.array(mj_data.mocap_quat, dtype=np.float32), ctx
            )


def run_controller(
        ctrl: SamplingBasedController,
        shm_data: SharedMemoryMujocoData,
        ready: Event,
        finished: Event,
        initial_knots: jax.Array = None,
) -> None:
    """Run the controller, communicating with the simulator over shared memory.

    Args:
        ctrl: The controller instance (which includes the task definition).
        shm_data: Shared memory object for state and control action data.
        ready: Shared flag for signaling that the controller is ready.
        finished: Shared flag for stopping the simulation.
        initial_knots: The initial control to use for the controller.
    """
    # Initialize the policy parameters and state estimate
    mjx_data = mjx.make_data(ctrl.task.mj_model, impl='warp') if ctrl.task.warp_enabled \
        else mjx.make_data(ctrl.task.mjx_model)
    policy_params = ctrl.init_params(initial_knots=initial_knots)

    # Print out some planning horizon information
    print(
        f"Planning with {ctrl.ctrl_steps} steps "
        f"over a {ctrl.plan_horizon} second horizon "
        f"with {ctrl.num_knots} knots."
    )

    # Jit the optimizer step, then signal that we're ready to go
    print("Jitting controller...")
    st = time.time()
    jit_optimize = jax.jit(
        lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,)
    )
    get_action = jax.jit(ctrl.get_action)
    policy_params = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st}")

    # Signal that we're ready to start
    ready.set()

    start_time = time.time()
    while not finished.is_set():
        curr_time = time.time()

        # Set the start state for the controller, reading the lastest state info
        # from shared memory
        mjx_data = mjx_data.replace(
            time=jnp.array(curr_time - start_time, dtype=jnp.float32),
            qpos=jnp.array(shm_data.qpos[:]),
            qvel=jnp.array(shm_data.qvel[:]),
        )
        if len(mjx_data.mocap_pos) > 0:
            mjx_data = mjx_data.replace(
                mocap_pos=jnp.array(shm_data.mocap_pos[:]),
                mocap_quat=jnp.array(shm_data.mocap_quat[:]),
            )

        # Do a planning step
        ctrl.task.ref_qpos[:7] = shm_data.ref_arm_qpos[:7]
        policy_params = jit_optimize(mjx_data, policy_params)

        # Send the action to the simulator.
        action = get_action(policy_params, mjx_data.time)
        if action.shape == shm_data.ctrl[:].shape:
            shm_data.ctrl[:] = np.array(action, dtype=np.float32)
        else:
            shm_data.wrist_ctrl[:] = np.array(action[:6], dtype=np.float32)
            shm_data.hand_ctrl[:] = np.array(action[6:], dtype=np.float32)

        # Print the current planning frequency
        print(
            f"Controller running at {1 / (time.time() - st):.2f} Hz", end="\r"
        )

    # Preserve the last printed line
    print("")


def run_simulator(
        ctrl: SamplingBasedController,
        mj_model: mj.MjModel,
        mj_data: mj.MjData,
        shm_data: SharedMemoryMujocoData,
        ready: Event,
        finished: Event,
) -> None:
    """Run a simulation, communicating with the controller over shared memory.

    Args:
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        shm_data: Shared memory object for state and control action data.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
    """
    # Wait for the controller to be ready
    ready.wait()

    obj_id = ctrl.task.mj_model.body("cube").id
    mocap_obj_id = ctrl.task.mj_model.body("target").id
    mocap_id = ctrl.task.mj_model.body_mocapid[obj_id]
    DEFAULT_WXYZ = np.array([0., 1., 0., 0.])
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # Write the latest state to shared memory for the controller to read
            shm_data.qpos[:] = mj_data.qpos
            shm_data.qvel[:] = mj_data.qvel
            shm_data.xpos[:] = mj_data.xpos
            shm_data.xquat[:] = mj_data.xquat

            if len(mj_data.mocap_pos) > 0:
                shm_data.mocap_pos[:] = mj_data.mocap_pos
                shm_data.mocap_quat[:] = mj_data.mocap_quat

            ctrl.task.update_ref_qpos(np.concatenate([mj_data.xpos[obj_id], DEFAULT_WXYZ]))
            shm_data.ref_arm_qpos[:] = np.asarray(ctrl.task.ref_qpos)

            # Read the lastest control values from shared memory
            # TODO: actually query the spline rather than assuming zero-order
            # hold and a sufficiently high control rate
            if ctrl.ctrl_callback:
                mj_data.ctrl[7:] = shm_data.hand_ctrl[:]
                mj_data.ctrl[:7] = ArmHandDiffIK.dls_ik(mj_model, mj_data, shm_data.wrist_ctrl[:],
                                                        grasp_site_name="leap_rh/grasp_site",
                                                        dt=mj_model.opt.timestep)
            else:
                mj_data.ctrl[:] = shm_data.ctrl[:]

            # Step the simulation
            mj.mj_step(mj_model, mj_data)
            viewer.sync()

            # Try to run in roughly real-time
            elapsed_time = time.time() - start_time
            if elapsed_time < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed_time)

    # Signal that the simulation is done
    finished.set()


def run_interactive(
        controller: SamplingBasedController,
        mj_model: mj.MjModel,
        mj_data: mj.MjData,
        initial_knots: jax.Array = None,
) -> None:
    """Run an asynchronous interactive simulation.

    This is similar to `simulation.deterministic.run_interactive`, but runs the
    controller and simulator in separate processes. This is more realistic, but
    offers fewer features (e.g., no trace visualization).

    Args:
        controller: The controller to use for planning.
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        initial_knots: The initial knot points for the control spline at t=0
    """
    ctx = mp.get_context("spawn")  # Need to use spawn for jax compatibility

    # Create shared_memory data
    shm_data = SharedMemoryMujocoData(mj_data, ctx)
    ready = ctx.Event()
    finished = ctx.Event()

    # Set up the simulator and controller processes
    sim = ctx.Process(
        target=run_simulator,
        args=(controller, mj_model, mj_data, shm_data, ready, finished),
    )
    control = ctx.Process(
        target=run_controller,
        args=(controller, shm_data, ready, finished, initial_knots),
    )

    # Run the simulation and controller in parallel
    sim.start()
    control.start()

    # Clean up when done (e.g. the visualizer is closed)
    sim.join()
    control.join()
