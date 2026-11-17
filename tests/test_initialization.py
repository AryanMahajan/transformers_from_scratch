import numpy as np

from src.core import initialization as init
from src.nn.loss import cross_entropy


def test_same_seed_same_weights_different_seed_different():
    a = init.normal(np.random.default_rng(42), (8, 8))
    b = init.normal(np.random.default_rng(42), (8, 8))
    c = init.normal(np.random.default_rng(43), (8, 8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    W = init.xavier_uniform(rng, 64, 256)
    assert W.shape == (64, 256) and W.dtype == np.float32  # [in, out]
    assert init.normal(rng, (3,), dtype=np.float64).dtype == np.float64
    np.testing.assert_array_equal(init.zeros(4), 0.0)
    np.testing.assert_array_equal(init.ones(4), 1.0)


def test_empirical_variances_match_formulas():
    rng = np.random.default_rng(1)
    fi, fo = 400, 600
    np.testing.assert_allclose(init.xavier_normal(rng, fi, fo).var(), 2 / (fi + fo), rtol=0.02)
    np.testing.assert_allclose(init.xavier_uniform(rng, fi, fo).var(), 2 / (fi + fo), rtol=0.02)
    np.testing.assert_allclose(init.he_normal(rng, fi, fo).var(), 2 / fi, rtol=0.02)
    np.testing.assert_allclose(init.normal(rng, (fi, fo)).std(), 0.02, rtol=0.02)
    a = np.sqrt(6 / (fi + fo))
    assert np.abs(init.xavier_uniform(rng, fi, fo)).max() <= a


def test_variance_preserved_through_deep_linear_stack():
    """1/sqrt(fan_in) keeps Var ~1 over 20 layers; std=1 explodes; 0.01 vanishes."""
    D, L = 256, 20

    def run(std):
        rng = np.random.default_rng(2)
        h = rng.standard_normal((512, D))
        for _ in range(L):
            h = h @ init.normal(rng, (D, D), std, dtype=np.float64)
        return h.var()

    assert 0.5 < run(1 / np.sqrt(D)) < 2.0
    assert run(1.0) > 1e40
    assert run(0.01) < 1e-30


def test_he_preserves_variance_through_relu_stack():
    D, L = 256, 20
    rng = np.random.default_rng(3)
    h = rng.standard_normal((512, D))
    for _ in range(L):
        h = np.maximum(h @ init.he_normal(rng, D, D, dtype=np.float64), 0.0)
    assert 0.3 < np.mean(h**2) < 3.0  # second moment stays O(1)


def test_residual_std_and_small_init_gives_log_V_loss():
    np.testing.assert_allclose(init.gpt2_residual_std(6), 0.02 / np.sqrt(12))
    # GPT-2-style small LM head -> logits ~ 0 -> loss ~ log V at step 0
    rng = np.random.default_rng(4)
    D, V = 64, 65
    h = rng.standard_normal((4, 16, D))  # post-LayerNorm features, Var ~1
    logits = h @ init.normal(rng, (D, V), dtype=np.float64)
    loss, _ = cross_entropy(logits, rng.integers(0, V, size=(4, 16)))
    assert abs(loss - np.log(V)) < 0.05
