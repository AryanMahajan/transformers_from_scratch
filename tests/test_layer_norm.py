import numpy as np

from src.nn.layer_norm import LayerNorm
from src.utils.numerical import check_module


def _ln(D=6, rng=None):
    ln = LayerNorm(D).astype(np.float64)
    if rng is not None:                       # non-trivial gamma/beta so dout*gamma is exercised
        ln.gamma.data[:] = rng.standard_normal(D)
        ln.beta.data[:] = rng.standard_normal(D)
    return ln


def test_output_is_normalised_per_position():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 16)) * 7 + 3          # arbitrary scale/shift
    y = LayerNorm(16).astype(np.float64)(x)              # gamma=1, beta=0
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, rtol=1e-5)


def test_matches_explicit_formula_and_no_cross_position_mixing():
    rng = np.random.default_rng(1)
    ln = _ln(rng=rng)
    x = rng.standard_normal((2, 3, 6))
    y = ln(x)
    r = x[1, 2]
    ref = ln.gamma.data * (r - r.mean()) / np.sqrt(r.var() + ln.eps) + ln.beta.data
    np.testing.assert_allclose(y[1, 2], ref, rtol=1e-12)
    x2 = x.copy()
    x2[0, 0] += 100.0                                    # change one position...
    np.testing.assert_allclose(ln(x2)[1], y[1])          # ...others unaffected


def test_gradient_check():
    rng = np.random.default_rng(2)
    errs = check_module(_ln(rng=rng), rng.standard_normal((2, 3, 6)), rng)
    assert set(errs) == {"input", "gamma", "beta"}
    assert max(errs.values()) < 1e-8


def test_dx_orthogonal_to_ones_and_x_hat():
    """LN ignores shifts and (with eps->0) scales of x, so dx has no component along 1 or x_hat."""
    rng = np.random.default_rng(3)
    ln = _ln(D=8, rng=rng)
    ln.eps = 0.0
    x = rng.standard_normal((4, 8))
    ln(x)
    dx = ln.backward(rng.standard_normal((4, 8)))
    np.testing.assert_allclose(dx.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose((dx * ln._x_hat).sum(axis=-1), 0.0, atol=1e-12)


def test_invariant_to_shift_and_scale_of_input():
    rng = np.random.default_rng(4)
    ln = _ln(rng=rng)
    x = rng.standard_normal((3, 6))
    np.testing.assert_allclose(ln(x + 5.0), ln(x), atol=1e-10)   # shift: exact
    # scale: exact only when eps = 0; eps adds an O(eps / var) deviation
    diff_eps = np.abs(ln(x * 10.0) - ln(x)).max()
    ln.eps = 0.0
    np.testing.assert_allclose(ln(x * 10.0), ln(x), atol=1e-10)
    assert 1e-7 < diff_eps < 1e-3


def test_constant_row_is_finite_thanks_to_eps():
    ln = LayerNorm(4)
    y = ln(np.full((1, 4), 3.0, dtype=np.float32))
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, 0.0)


def test_two_pass_variance_beats_naive_in_float32():
    """E[x^2] - mu^2 cancels catastrophically for large-mean rows; mean((x-mu)^2) doesn't."""
    rng = np.random.default_rng(5)
    x = (1e4 + rng.standard_normal((64, 32)) * 0.01).astype(np.float32)
    true_var = x.astype(np.float64).var(axis=-1)
    naive = (x * x).mean(axis=-1) - x.mean(axis=-1) ** 2
    xc = x - x.mean(axis=-1, keepdims=True)
    two_pass = (xc * xc).mean(axis=-1)
    assert np.abs(naive - true_var).max() > 1.0                  # garbage (true var ~1e-4)
    assert np.abs(two_pass - true_var).max() < 1e-5
