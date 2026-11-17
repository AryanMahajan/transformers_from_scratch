"""Global gradient-norm clipping.

    total = sqrt( sum over ALL parameters of sum(p.grad^2) )
    if total > max_norm:  every p.grad *= max_norm / total

One scale factor for all gradients: the update DIRECTION is unchanged, only
its length is capped. (Clipping each tensor separately would change the
direction.) Returns the pre-clip norm -- log it: spikes show instability.
Derivations: docs/06_training.md §2
"""

import numpy as np


def global_grad_norm(params) -> float:
    return float(np.sqrt(sum(float(np.sum(p.grad.astype(np.float64) ** 2)) for p in params)))


def clip_grad_norm(params, max_norm: float, eps: float = 1e-6) -> float:
    """Scale all grads in place so their global L2 norm is <= max_norm."""
    params = list(params)
    total = global_grad_norm(params)
    if total > max_norm:
        scale = max_norm / (total + eps)
        for p in params:
            p.grad *= scale
    return total
