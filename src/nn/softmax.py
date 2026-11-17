"""Numerically stable softmax and log-softmax.

Both functions operate along a single axis (default: the last one), so the
same code works for attention scores [B, H, T, T] and LM logits [B, T, V].

Derivations: docs/01_numerical_foundations.md
"""

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """softmax(z)_i = exp(z_i) / sum_j exp(z_j), computed without overflow.

    Uses shift invariance: subtracting max(z) leaves the result unchanged but
    makes the largest exponent exp(0) = 1.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_shifted = np.exp(z - z_max)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def softmax_backward(p: np.ndarray, dout: np.ndarray, axis: int = -1) -> np.ndarray:
    """Vector-Jacobian product of softmax: given p = softmax(z) and
    dout = dL/dp, return dL/dz = p * (dout - <dout, p>).

    Equivalent to dout @ (diag(p) - p p^T) per row, but O(V) instead of
    O(V^2) and never materialises the Jacobian. Needs only the forward
    *output* p, not the input z.
    """
    dot = np.sum(dout * p, axis=axis, keepdims=True)  # <g, p> per row
    return p * (dout - dot)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log softmax(z)_i = (z_i - m) - log sum_j exp(z_j - m), with m = max(z).

    Stays in log space (log-sum-exp trick) so tiny probabilities never
    underflow to 0 before the log is taken.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    shifted = z - z_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp
