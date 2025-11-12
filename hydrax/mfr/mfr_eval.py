import argparse
import time
import os

from evosax.algorithms.distribution_based import Sep_CMA_ES

import mujoco as mj
import mujoco.viewer
import numpy as np

# hydrax
from hydrax import ROOT
from hydrax.utils.video import VideoRecorder
from hydrax.algs import CEM, ICEM, MPPI, Evosax, PredictiveSampling
from hydrax.mfr.mfr_planner import MFRPlanner, get_task
from hydrax.mfr.allegro_env import get_task_config

"""
Run an interactive simulation of the cube relocating task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

from mfr_common import DEFAULT_MFR_TASK_NAME, MFR_HEADLESS

parser = argparse.ArgumentParser(description="MFR Planner")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument('--task', type=str, default=DEFAULT_MFR_TASK_NAME, help='task to evaluate')
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)",
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("mfr", help="MFR")
args = parser.parse_args()

# Task Config
task_config = get_task_config(args.task)

# Task
task = get_task(args.task, task_config)

args.algorithm = "mfr"

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.2,
        num_randomizations=32,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.2,
        temperature=0.001,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_start=0.5,
        sigma_min=0.5,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
        # explore_fraction=0.3
    )
elif args.algorithm == "icem":
    print("Running ICEM")
    ctrl = ICEM(
        task,
        num_samples=128,
        num_elites=10,
        sigma_start=1.0,
        sigma_min=1.0,
        num_randomizations=8,
        plan_horizon=2.0,
        spline_type="zero",
        num_knots=4,
        iterations=1
    )
elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        Sep_CMA_ES,
        num_samples=128,
        # elite_ratio=0.5,
        num_randomizations=8,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mfr":
    pass
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = task.mj_data
mj_model.opt.timestep = 0.01


def main(frequency: float = 25, fixed_camera_id: int = None, record_video: bool = False) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        frequency: The requested control frequency (Hz) for replanning.
        fixed_camera_id: The camera ID to use for the fixed camera view.
        record_video: Whether to record a video of the simulation.
    """
    # Report the planning horizon in seconds for debugging
    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz "
        f"sim_steps_per_replan: {sim_steps_per_replan}"
    )

    # Initialize video recording if enabled
    renderer = None
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mj.Renderer(mj_model, height=height, width=width)

    # Start the simulation
    with mj.viewer.launch_passive(mj_model, mj_data) as viewer:
        task._mj_viewer = viewer
        task._mj_renderer = renderer
        task._mj_recorder = recorder
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Planner
        planner = MFRPlanner(task, task_config, args)

        accum_elapsed = 0
        while viewer.is_running():
            start_time = time.time()

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            # sim_dt = mj_model.opt.timestep
            # t_curr = mj_data.time

            # simulate the system between spline replanning steps
            kinematics_only = True
            if planner.pregrasp_action_list:
                task.step(planner.pregrasp_action_list.pop(0), kinematics_only)
            else:
                planner.plan(step_env=True, kinematics_only=kinematics_only)
            viewer.sync()

            # Capture frame if recording
            if record_video and recorder.is_recording:
                renderer.update_scene(mj_data, viewer.cam)
                frame = renderer.render()
                recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            accum_elapsed += elapsed
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {elapsed:.4f}s, accumulated time: {accum_elapsed:.2f}s",
                end="\r",
            )

            if record_video and accum_elapsed >= 2:
                break

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()


if __name__ == "__main__":
    main()
