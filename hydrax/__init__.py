import os
import jax
from pathlib import Path

# package root
ROOT = str(Path(__file__).parent.absolute())

# data dir
DATA_DIR = os.path.join(ROOT, "data")

# Set XLA flags for better performance
# https://docs.jax.dev/en/latest/gpu_performance_tips.html#xla-performance-flags
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", f"{ROOT}/tmp/jax_cache")
