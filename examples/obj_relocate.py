import argparse
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# MuJoCo
import mujoco as mj
from evosax.algorithms.distribution_based import Sep_CMA_ES

# hydrax
from hydrax import ROOT, BackendType
from hydrax.algs import CEM, ICEM, MPPI, Evosax, PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive as async_run_interactive
from hydrax.simulation.deterministic import run_interactive as sync_run_interactive
from hydrax.tasks.mug_relocate_task import MugRelocateTask

# mjmanip
from mjmanip.control.fabrics.fabrics.arm_hand_pose_fabric import ArmHandPoseFabricConfig

"""
Run an interactive simulation of the obj relocating task.

Double click on the floating target obj, then change the goal orientation with
[ctrl + left click].
"""

FABRICS_CONFIGS_DIR = f"{ROOT}/control/fabrics/configs"

cs = ConfigStore.instance()
cs.store(name="leap_mujoco", node=ArmHandPoseFabricConfig)

cfg_name = "leap_mujoco"

fabric_cfg = None


@hydra.main(version_base=None, config_path=FABRICS_CONFIGS_DIR, config_name=cfg_name)
def fetch_fabric_config(cfg: DictConfig) -> None:
    global fabric_cfg
    fabric_cfg = OmegaConf.to_object(cfg)
    assert isinstance(fabric_cfg, ArmHandPoseFabricConfig)
    # print(OmegaConf.to_yaml(fabric_cfg))


# Asynchronous simulations must be wrapped in a __main__ block
# https://docs.python.org/3/library/multiprocessing.html
if __name__ == "__main__":
    """
    Run an interactive simulation of the panda picking object.
    """

    # Define the task (cost and dynamics)
    use_ctrl_callback = False
    fetch_fabric_config()
    # Define the task (cost and dynamics)
    task = MugRelocateTask(name="Mug Relocate",
                           fabric_cfg=fabric_cfg if use_ctrl_callback else None,
                           backend_type=BackendType.MJW)  # Not enough memory for Warp with mug mesh

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the obj rotation task."
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("ps", help="Predictive Sampling")
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("cem", help="Cross-Entropy Method")
    subparsers.add_parser("cmaes", help="CMA-ES")
    args = parser.parse_args()

    args.algorithm = "cem"
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
            num_samples=32,
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
    else:
        parser.error("Invalid algorithm")

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mj.MjData(mj_model)

    # Run the interactive simulation
    synchronous = True
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
