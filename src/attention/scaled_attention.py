"""Scaled dot-product attention, forward and backward.

Forward (any leading dims, e.g. [B, H, T, d]):
    S = Q K^T / sqrt(d)           [..., T, T]
    A = softmax(S masked to -inf)  future keys get exactly 0 weight
    O = A V                        [..., T, d]
Backward:
    dA = dO V^T,    dV = A^T dO
    dS = softmax_backward(A, dA)   (masked entries: A = 0 -> dS = 0 automatically)
    dQ = dS K / sqrt(d),   dK = dS^T Q / sqrt(d)
Derivations: docs/04_attention.md §1
"""

import numpy as np

from src.attention.causal_mask import causal_mask
from src.core.backend import xp_of
from src.nn.softmax import softmax, softmax_backward


def _t(x: np.ndarray) -> np.ndarray:
    """Transpose the last two axes (batched matrix transpose)."""
    return np.swapaxes(x, -1, -2)


def attention_forward(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True):
    """Returns (out [..., T, d], cache). cache["A"] holds the attention weights."""
    d = q.shape[-1]
    T = q.shape[-2]
    scale = 1.0 / np.sqrt(d)

    scores = (q @ _t(k)) * scale                          # [..., T, T]
    if causal:
        scores = np.where(causal_mask(T, xp_of(q)), scores, -np.inf)   # mask on q's device
    A = softmax(scores, axis=-1)
    out = A @ v                                           # [..., T, d]
    return out, {"q": q, "k": k, "v": v, "A": A, "scale": scale}


def attention_backward(dout: np.ndarray, cache: dict):
    """Returns (dq, dk, dv), each the same shape as its input."""
    q, k, v, A, scale = cache["q"], cache["k"], cache["v"], cache["A"], cache["scale"]

    dA = dout @ _t(v)                                     # [..., T, T]
    dv = _t(A) @ dout                                     # [..., T, d]
    dS = softmax_backward(A, dA, axis=-1) * scale         # fold the 1/sqrt(d) in once
    dq = dS @ k                                           # [..., T, d]
    dk = _t(dS) @ q                                       # [..., T, d]
    return dq, dk, dv
