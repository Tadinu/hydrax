import argparse

import evosax
import mujoco as mj

from hydrax.algs import CEM, ICEM, MPPI, Evosax, PredictiveSampling, DIAL
from hydrax.simulation.asynchronous import run_interactive as async_run_interactive
from hydrax.simulation.deterministic import run_interactive as sync_run_interactive
from hydrax.tasks.panda.panda_open_cabinet_env import PandaOpenCabinetEnv
from hydrax.tasks.panda.panda_pick_env import PandaPickEnv
from hydrax.risk import BestCase

# Asynchronous simulations must be wrapped in a __main__ block
# https://docs.python.org/3/library/multiprocessing.html
if __name__ == "__main__":
    """
    Run an interactive simulation of the panda picking object.
    """

    # Define the task (cost and dynamics)
    task = PandaPickEnv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the cube rotation task."
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("ps", help="Predictive Sampling")
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("cem", help="Cross-Entropy Method")
    subparsers.add_parser("cmaes", help="CMA-ES")
    args = parser.parse_args()

    args.algorithm = "cmaes"
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
            num_samples=256,
            num_elites=10,
            sigma_start=0.5,
            sigma_min=0.5,
            num_randomizations=8,
            plan_horizon=2.0,
            spline_type="zero",
            num_knots=4,
            iterations=1
        )
    elif args.algorithm == "icem":
        print("Running ICEM")
        ctrl = ICEM(
            task,
            num_samples=128,
            num_elites=5,
            sigma_start=0.5,
            sigma_min=0.5,
            num_randomizations=32,
            explore_fraction=0.5,
            # risk_strategy=BestCase(),
            plan_horizon=0.12,
            spline_type="zero",
            num_knots=4,
            iterations=1
        )
    elif args.algorithm == "cmaes":
        print("Running CMA-ES")
        ctrl = Evosax(
            task,
            evosax.Sep_CMA_ES,
            num_samples=128,
            elite_ratio=0.5,
            num_randomizations=8,
            plan_horizon=0.25,
            spline_type="zero",
            num_knots=4,
        )
    elif args.algorithm == "dial":
        print("Running Diffusion-Inspired Annealing for Legged MPC (DIAL)")
        ctrl = DIAL(
            task,
            num_samples=1024,
            noise_level=0.4,
            beta_opt_iter=1.0,
            beta_horizon=1.0,
            temperature=0.001,
            plan_horizon=0.25,
            spline_type="zero",
            num_knots=11,
            iterations=5,
        )
    else:
        parser.error("Invalid algorithm")

    # Define the model used for simulation (with more realistic parameters)
    mj_model = task.mj_model
    mj_data = mj.MjData(mj_model)

    # Run the interactive simulation
    synchronous = False
    if synchronous:
        sync_run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=25,
            fixed_camera_id=None,
            show_traces=False,
            max_traces=1,
            trace_color=[1.0, 1.0, 1.0, 1.0],
        )
    else:
        mj_model.opt.timestep = 0.005
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
        # Delete all internal MjSpecs due to not being `picklable` by multiprocess
        task.delete_specs()
        async_run_interactive(
            ctrl,
            mj_model,
            mj_data
        )
