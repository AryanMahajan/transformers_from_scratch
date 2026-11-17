import math

import numpy as np

from src.nn.activations import gelu, gelu_backward


def _gelu_exact(x):
    """x * Phi(x) using math.erf (scalar), for reference only."""
    phi = np.vectorize(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))))
    return x * phi(x)


def test_known_values_and_limits():
    assert gelu(np.array(0.0)) == 0.0
    x = np.array([-10.0, 10.0])
    np.testing.assert_allclose(gelu(x), [0.0, 10.0], atol=1e-6)  # gate -> 0 / 1


def test_tanh_approx_close_to_exact():
    x = np.linspace(-6, 6, 2001)
    err = np.abs(gelu(x) - _gelu_exact(x)).max()
    assert err < 1e-3


def test_non_monotonic_dip():
    x = np.linspace(-3, 0, 3001)
    y = gelu(x)
    assert y.min() < -0.16  # minimum ~ -0.17 near x ~ -0.75
    assert -0.80 < x[y.argmin()] < -0.70


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8)) * 3  # [B, T, F], float64, spans the dip
    g = rng.standard_normal(x.shape)
    eps = 1e-6
    numeric = (gelu(x + eps) - gelu(x - eps)) / (2 * eps) * g  # valid: element-wise fn
    analytic = gelu_backward(x, g)
    rel_err = np.abs(analytic - numeric).max() / np.abs(numeric).max()
    assert rel_err < 1e-6


def test_slope_at_zero_is_half():
    # GELU'(0) = Phi(0) = 0.5
    np.testing.assert_allclose(gelu_backward(np.array(0.0), np.array(1.0)), 0.5)
