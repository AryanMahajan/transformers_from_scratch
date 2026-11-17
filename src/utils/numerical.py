"""Finite-difference gradient checking: the ground truth for every backward.

Central differences, float64 only. See docs/01_numerical_foundations.md §6.
"""

from typing import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar f() w.r.t. array x.

    f takes NO arguments: it must read x by reference (closure / layer
    parameter). x is perturbed IN PLACE one entry at a time and restored,
    which is exactly how we'll check parameters inside layers later.
    """
    if x.dtype != np.float64:
        raise TypeError(f"gradient check needs float64, got {x.dtype}")
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f()
        x[idx] = orig - eps
        f_minus = f()
        x[idx] = orig                    # restore!
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / (||a|| + ||b||), norm-wise, so safe when entries are ~0."""
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def check_module(module, x: np.ndarray, rng: np.random.Generator, eps: float = 1e-6) -> dict[str, float]:
    """Gradient-check a Module: dL/dx and dL/dp for every parameter.

    L = sum(g * module(x)) for a random g. The module must already be float64
    (module.astype(np.float64)). Returns {"input": err, "<param name>": err, ...}.
    """
    y = module.forward(x)
    g = rng.standard_normal(y.shape)
    module.zero_grad()
    dx = module.backward(g)              # fills p.grad; before any re-forward

    def loss() -> float:
        return float(np.sum(g * module.forward(x)))

    errors = {}
    if np.issubdtype(x.dtype, np.floating):  # integer inputs (token ids) have no grad
        errors["input"] = relative_error(dx, numerical_gradient(loss, x, eps))
    for name, p in module.named_parameters():
        errors[name] = relative_error(p.grad, numerical_gradient(loss, p.data, eps))
    return errors


def check_vjp(
    forward: Callable[[np.ndarray], np.ndarray],
    backward: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> float:
    """Check backward(g) == d/dx sum(g * forward(x)) for a random g.

    forward(x) -> y        (any shape)
    backward(g) -> dL/dx   (g has y's shape; may use state from forward)
    Returns the relative error.
    """
    y = forward(x)
    g = rng.standard_normal(y.shape)
    analytic = backward(g)               # uses forward state from x as-is
    numeric = numerical_gradient(lambda: float(np.sum(g * forward(x))), x, eps)
    return relative_error(analytic, numeric)
