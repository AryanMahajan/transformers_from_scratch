"""Optional GPU backend via CuPy (a NumPy-compatible GPU array library).

Default is NumPy on the CPU: all tests, gradient checks and the demos use it.
`--device gpu` moves the model's arrays to the GPU as cupy.ndarray. The maths
and hand-written backward passes are unchanged:
  * NumPy functions called on CuPy arrays (np.exp, np.sum, np.where, np.sqrt,
    np.zeros_like, ...) dispatch to CuPy automatically (NEP 18 / __array_function__).
  * Only array CREATION from nothing (np.arange, np.tril(np.ones(...))) and a few
    special ops (np.add.at) must pick the right library: `xp_of(x)` returns
    numpy or cupy depending on where x lives.
  * Mixing a NumPy and a CuPy array raises (no hidden GPU<->CPU copies), so
    every transfer is explicit: to_device / to_numpy.
Notes: docs/07_real_training.md §2
"""

import numpy as np

try:
    import cupy as cp
except ImportError:                        # CPU-only install: everything still works
    cp = None


def gpu_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def is_gpu_array(a) -> bool:
    return cp is not None and isinstance(a, cp.ndarray)


def xp_of(a):
    """The array module (numpy or cupy) that `a` belongs to."""
    return cp if is_gpu_array(a) else np


def to_device(a, device: str):
    if device == "cpu":
        return to_numpy(a)
    if device == "gpu":
        if cp is None:
            raise RuntimeError("device='gpu' needs CuPy: pip install -r requirements-gpu.txt")
        return cp.asarray(a)
    raise ValueError(f"unknown device {device!r} (use 'cpu' or 'gpu')")


def to_numpy(a):
    return cp.asnumpy(a) if is_gpu_array(a) else np.asarray(a)


def like(a, ref):
    """Put `a` on the same device as `ref`."""
    return cp.asarray(a) if is_gpu_array(ref) else to_numpy(a)


def scatter_add(target, idx, values) -> None:
    """target[idx] += values, with repeated indices ADDING (np.add.at semantics)."""
    xp_of(target).add.at(target, idx, values)
