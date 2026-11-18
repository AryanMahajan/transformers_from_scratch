import numpy as np
import pytest

from src.generation.greedy import greedy_generate
from src.generation.sampling import filter_logits, generate, sample_next
from src.nn.softmax import softmax
from src.transformer.model import GPT, GPTConfig

Z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])


def _model(max_len=8, seed=0):
    return GPT(GPTConfig(vocab_size=5, d_model=16, n_heads=2, n_layers=1, max_len=max_len),
               np.random.default_rng(seed))


def _freqs(n=40000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    draws = [sample_next(Z, rng, **kw) for _ in range(n)]
    return np.bincount(draws, minlength=len(Z)) / n


# --- filter_logits ---------------------------------------------------------

def test_temperature_divides_logits():
    np.testing.assert_allclose(filter_logits(Z, temperature=0.5), Z / 0.5)


def test_low_temperature_sharpens_high_flattens():
    h = lambda p: -(p * np.log(p)).sum()                  # entropy
    p1, plow, phigh = (softmax(filter_logits(Z, t)) for t in (1.0, 0.3, 3.0))
    assert h(plow) < h(p1) < h(phigh)
    assert plow.argmax() == p1.argmax() == phigh.argmax()  # ranking never changes


def test_top_k_keeps_exactly_the_k_largest():
    z = filter_logits(Z, top_k=2)
    assert np.isfinite(z).sum() == 2
    assert set(np.flatnonzero(np.isfinite(z))) == {0, 1}
    np.testing.assert_array_equal(filter_logits(Z, top_k=99), Z)   # k >= V: no-op


def test_top_k_renormalises_kept_probabilities():
    p = softmax(filter_logits(Z, top_k=3))
    full = softmax(Z)
    np.testing.assert_allclose(p[:3], full[:3] / full[:3].sum())
    assert (p[3:] == 0).all()


def test_top_p_keeps_smallest_set_reaching_p():
    p = softmax(Z)                         # ~[.53,.20,.12,.07,.03] sorted already
    for top_p in (0.3, 0.6, 0.8, 0.95):
        kept = np.isfinite(filter_logits(Z, top_p=top_p)).sum()
        cum = np.cumsum(p)
        assert cum[kept - 1] >= top_p                          # enough mass
        assert kept == 1 or cum[kept - 2] < top_p              # and minimal
    assert np.isfinite(filter_logits(Z, top_p=0.01)).sum() == 1   # top token always kept
    np.testing.assert_array_equal(filter_logits(Z, top_p=1.0), Z)


def test_top_p_works_on_unsorted_logits():
    perm = np.array([3, 0, 4, 1, 2])
    z = filter_logits(Z[perm], top_p=0.6)
    assert set(np.flatnonzero(np.isfinite(z))) == {1, 3}       # where Z[0], Z[1] went


def test_bad_arguments_rejected():
    for kw in ({"temperature": -1.0}, {"top_k": 0}, {"top_p": 0.0}, {"top_p": 1.5}):
        with pytest.raises(ValueError):
            filter_logits(Z, **kw)


# --- sample_next -----------------------------------------------------------

def test_empirical_frequencies_match_softmax():
    for t in (1.0, 2.0):
        np.testing.assert_allclose(_freqs(temperature=t), softmax(Z / t), atol=0.01)


def test_masked_tokens_are_never_drawn():
    f = _freqs(n=20000, top_k=2)
    assert f[2:].sum() == 0
    np.testing.assert_allclose(f[:2], softmax(Z[:2]), atol=0.015)


def test_temperature_zero_is_argmax_and_tiny_temperature_matches_it():
    rng = np.random.default_rng(0)
    assert sample_next(Z, rng, temperature=0) == 0
    assert all(sample_next(Z, rng, temperature=1e-3) == 0 for _ in range(200))


def test_same_seed_same_draws():
    a = [sample_next(Z, np.random.default_rng(7)) for _ in range(5)]
    b = [sample_next(Z, np.random.default_rng(7)) for _ in range(5)]
    assert a == b


# --- generate --------------------------------------------------------------

def test_generate_shape_reproducible_and_restores_train_mode():
    m = _model(max_len=4)
    a = generate(m, [1, 2], 20, np.random.default_rng(3), temperature=1.0, top_k=3)
    b = generate(m, [1, 2], 20, np.random.default_rng(3), temperature=1.0, top_k=3)
    assert a.shape == (22,) and a.min() >= 0 and a.max() < 5   # also crosses max_len
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:2], [1, 2])
    assert m.training


def test_generate_temperature_zero_equals_greedy():
    m = _model()
    np.testing.assert_array_equal(generate(m, [3], 15, np.random.default_rng(0), temperature=0),
                                  greedy_generate(m, [3], 15))


def test_generate_top_k_1_equals_greedy():
    m = _model()
    np.testing.assert_array_equal(generate(m, [3], 15, np.random.default_rng(0), top_k=1),
                                  greedy_generate(m, [3], 15))


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        generate(_model(), [], 3, np.random.default_rng(0))
