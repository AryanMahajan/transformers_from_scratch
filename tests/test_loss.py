import numpy as np

from src.nn.loss import cross_entropy, cross_entropy_backward
from src.nn.softmax import softmax, softmax_backward
from src.utils.numerical import numerical_gradient, relative_error


def test_uniform_logits_give_log_V():
    B, T, V = 2, 3, 65
    loss, _ = cross_entropy(np.zeros((B, T, V)), np.zeros((B, T), dtype=int))
    np.testing.assert_allclose(loss, np.log(V), rtol=1e-12)


def test_confident_correct_prediction_has_near_zero_loss():
    logits = np.array([[50.0, 0.0, 0.0]])
    loss, probs = cross_entropy(logits, np.array([0]))
    assert loss < 1e-12
    assert np.abs(cross_entropy_backward(probs, np.array([0]))).max() < 1e-12


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    B, T, V = 2, 3, 5
    z = rng.standard_normal((B, T, V))
    y = rng.integers(0, V, size=(B, T))
    _, probs = cross_entropy(z, y)
    analytic = cross_entropy_backward(probs, y)
    numeric = numerical_gradient(lambda: cross_entropy(z, y)[0], z)
    assert relative_error(analytic, numeric) < 1e-8


def test_fused_equals_chained_on_safe_inputs():
    """Fused p - onehot == softmax_backward(-1/p_y at target), when finite."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 6))
    y = rng.integers(0, 6, size=4)
    N = z.shape[0]
    _, probs = cross_entropy(z, y)

    p = softmax(z)
    dL_dp = np.zeros_like(p)
    dL_dp[np.arange(N), y] = -1.0 / (N * p[np.arange(N), y])
    chained = softmax_backward(p, dL_dp)

    np.testing.assert_allclose(cross_entropy_backward(probs, y), chained, atol=1e-12)


def test_chained_breaks_where_fused_survives():
    """Badly wrong prediction: p_y underflows to 0 -> chained gives nan."""
    z = np.array([[1000.0, 0.0]])  # target is index 1, p_1 = e^-1000 -> 0.0
    y = np.array([1])
    loss, probs = cross_entropy(z, y)
    fused = cross_entropy_backward(probs, y)
    assert np.isfinite(loss) and np.isclose(loss, 1000.0)
    np.testing.assert_allclose(fused, [[1.0, -1.0]])

    p = softmax(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        dL_dp = np.array([[0.0, -1.0 / p[0, 1]]])  # -1/0 = -inf
        chained = softmax_backward(p, dL_dp)
    assert np.any(np.isnan(chained))


def test_gradient_rows_sum_to_zero_and_cache_untouched():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((3, 4, 7))
    y = rng.integers(0, 7, size=(3, 4))
    _, probs = cross_entropy(z, y)
    before = probs.copy()
    g = cross_entropy_backward(probs, y)
    np.testing.assert_allclose(g.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_array_equal(probs, before)
    assert g.shape == z.shape
