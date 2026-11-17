import numpy as np

from src.nn.softmax import log_softmax, softmax, softmax_backward


def test_rows_sum_to_one_and_positive():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((2, 3, 5))  # [B, T, V]
    p = softmax(z)
    assert p.shape == z.shape
    assert np.all(p > 0)
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


def test_shift_invariance():
    z = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(softmax(z), softmax(z + 100.0), rtol=1e-12)


def test_extreme_logits_no_nan_or_inf():
    z = np.array([[1000.0, 999.0, -1000.0], [-1000.0, -1000.0, -1000.0]])
    p = softmax(z)
    lp = log_softmax(z)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(lp))
    np.testing.assert_allclose(p[1], 1 / 3)  # all-equal row -> uniform
    # log p of the -1000 entry is about -2000, not -inf
    assert lp[0, 2] < -1999


def test_log_softmax_matches_log_of_softmax_on_safe_inputs():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 7))
    np.testing.assert_allclose(log_softmax(z), np.log(softmax(z)), rtol=1e-12)


def test_naive_softmax_actually_breaks():
    """Documents *why* the max-shift exists."""
    z = np.array([1000.0, 999.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    assert np.any(np.isnan(naive))
    assert np.all(np.isfinite(softmax(z)))


# ---------------------------------------------------------------- backward

def test_backward_matches_explicit_jacobian():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(6)
    g = rng.standard_normal(6)          # upstream dL/dp
    p = softmax(z)
    J = np.diag(p) - np.outer(p, p)     # J[i, j] = dp_i / dz_j
    np.testing.assert_allclose(softmax_backward(p, g), g @ J, rtol=1e-12)


def test_backward_matches_finite_differences_batched():
    """Scalar loss L = sum(g * softmax(z)); compare dL/dz to central diffs."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2, 3, 5))  # [B, T, V], float64
    g = rng.standard_normal(z.shape)
    analytic = softmax_backward(softmax(z), g)

    eps = 1e-6
    numeric = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        numeric[idx] = (np.sum(g * softmax(zp)) - np.sum(g * softmax(zm))) / (2 * eps)

    rel_err = np.abs(analytic - numeric).max() / np.abs(numeric).max()
    assert rel_err < 1e-6


def test_backward_rows_sum_to_zero():
    rng = np.random.default_rng(4)
    z = rng.standard_normal((4, 8))
    dz = softmax_backward(softmax(z), rng.standard_normal(z.shape))
    np.testing.assert_allclose(dz.sum(axis=-1), 0.0, atol=1e-12)
