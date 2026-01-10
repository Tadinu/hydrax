import os
import enum
from pathlib import Path

import jax

# package root
ROOT = str(Path(__file__).parent.absolute())

# data dir
DATA_DIR = os.path.join(ROOT, "data")

# models dir
MODELS_DIR = os.path.join(ROOT, "models")

# cache
CACHE_DIR = f"{ROOT}/.tmp/warp_cache"

# Set XLA flags for better performance
# https://docs.jax.dev/en/latest/gpu_performance_tips.html#xla-performance-flags
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", f"{ROOT}/.tmp/jax_cache")


class BackendType(enum.IntEnum):
    MJX = enum.auto()
    MJX_WARP = enum.auto()
    MJW = enum.auto()


# Declare globally devices
if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("Hydrax CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
HYDRAX_DEVICE_INT = 0
HYDRAX_DEVICE = f"cuda:{HYDRAX_DEVICE_INT}"

# Torch
import torch

torch.set_default_device(HYDRAX_DEVICE)
a = torch.zeros(4, device=HYDRAX_DEVICE)
torch.set_printoptions(precision=4)
print("HYDRAX", a)


# Set the warp cache directory based on device int
def initialize_warp(warp_cache_name):
    """
    Explicitly setting the directory for codegen and compilation. Need this for multi-gpu settings.
    See https://omniverse.gitlab-master-pages.nvidia.com/warp/basics.html#example-cache-management
    ----------------------------------------------
    :param warp_cache_name: str, subdirectory name for placing generated code and compilation.
    """
    import warp as wp

    wp.config.kernel_cache_dir = f"{CACHE_DIR}/warp/{wp.config.version}/warpcache_{warp_cache_name}"
    wp.init()

    # clear kernel cache (forces fresh kernel builds every time)
    # wp.build.clear_kernel_cache()


initialize_warp(str(HYDRAX_DEVICE_INT))
