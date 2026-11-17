"""Cross-entropy loss fused with log-softmax.

Forward:  L = mean_n [ LSE(z_n) - z_n[y_n] ]
Backward: dL/dz = (softmax(z) - onehot(y)) / N

Derivations: docs/01_numerical_foundations.md
"""

import numpy as np

from src.nn.softmax import log_softmax


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over all positions.

    logits:  [..., V] float (e.g. [B, T, V])
    targets: [...]    int   (e.g. [B, T]), values in [0, V)
    Returns (loss, probs); probs [..., V] is the cache for the backward pass.
    """
    V = logits.shape[-1]
    z = logits.reshape(-1, V)            # [N, V]
    y = targets.reshape(-1)              # [N]
    N = z.shape[0]

    log_probs = log_softmax(z, axis=-1)  # stable (LSE trick)
    loss = -log_probs[np.arange(N), y].mean()

    probs = np.exp(log_probs).reshape(logits.shape)
    return float(loss), probs


def cross_entropy_backward(probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """dL/dlogits = (probs - onehot(targets)) / N, same shape as logits."""
    V = probs.shape[-1]
    grad = probs.reshape(-1, V).copy()   # copy: don't mutate the cache
    y = targets.reshape(-1)
    N = grad.shape[0]

    grad[np.arange(N), y] -= 1.0
    grad /= N
    return grad.reshape(probs.shape)
