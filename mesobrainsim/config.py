"""
Backend configuration: switches between NumPy (CPU) and CuPy (GPU).
All modules should import `xp` from here instead of importing numpy/cupy directly.
"""

USE_GPU = False

def set_backend(use_gpu: bool):
    global USE_GPU, xp
    USE_GPU = use_gpu
    if use_gpu:
        try:
            import cupy as xp
        except ImportError:
            raise ImportError(
                "CuPy is not installed. Install it with: pip install cupy-cuda12x"
            )
    else:
        import numpy as xp


# Initialize default backend
import numpy as xp
