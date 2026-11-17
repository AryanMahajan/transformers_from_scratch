"""Activation functions.

GELU uses the tanh approximation (Hendrycks & Gimpel 2016), as GPT-2 does.
Derivations: docs/01_numerical_foundations.md
"""

import numpy as np

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_C = 0.044715


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))."""
    u = _SQRT_2_OVER_PI * (x + _GELU_C * x * x * x)   # x*x*x: ~6x faster than the general power x**3
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """dL/dx = dout * GELU'(x). Element-wise (diagonal Jacobian).

    GELU'(x) = 0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) * sqrt(2/pi) (1 + 3c x^2)
    Needs the forward *input* x.
    """
    u = _SQRT_2_OVER_PI * (x + _GELU_C * x * x * x)   # x*x*x: ~6x faster than the general power x**3
    t = np.tanh(u)
    du_dx = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_C * x * x)
    grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
    return dout * grad
