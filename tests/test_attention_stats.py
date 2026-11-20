import numpy as np
import pytest

from src.analysis.attention_stats import distance_profile, head_stats, mass_beyond, reach_cdf
from src.transformer.model import GPT, GPTConfig

T = 10
i = np.arange(T)


def _stack(*maps):
    """[T, T] maps -> [B=2, H=len(maps), T, T] (same maps in both batch rows)."""
    A = np.stack(maps)[None]
    return np.concatenate([A, A])


def prev_head():
    A = np.zeros((T, T)); A[0, 0] = 1; A[i[1:], i[1:] - 1] = 1
    return A


def sink_head():
    A = np.zeros((T, T)); A[:, 0] = 1
    return A


def uniform_head():
    return np.tril(np.ones((T, T))) / (i + 1)[:, None]


def test_known_patterns():
    s = head_stats(_stack(prev_head(), sink_head(), uniform_head()))
    q = i[1:]                                                    # queries used
    np.testing.assert_allclose(s["mean_distance"], [1, q.mean(), (q / 2).mean()])
    np.testing.assert_allclose(s["prev_token"], [1, 1 / (T - 1), (1 / (q + 1)).mean()])
    np.testing.assert_allclose(s["self"], [0, 0, (1 / (q + 1)).mean()])
    np.testing.assert_allclose(s["first_token"], [1 / (T - 1), 1, (1 / (q + 1)).mean()])
    np.testing.assert_allclose(s["norm_entropy"], [0, 0, 1], atol=1e-12)


def test_distance_profile_and_mass_beyond():
    A = _stack(prev_head(), uniform_head())
    prof = distance_profile(A)
    np.testing.assert_allclose(prof[0], np.eye(T)[1])            # all mass at distance 1
    d = np.arange(T)                                             # uniform: mass 1/(i+1), averaged over i >= d
    np.testing.assert_allclose(prof[1], [(1 / (np.arange(max(k, 1), T) + 1)).mean() for k in d])
    for dmin in (1, 3):                                          # uniform: (i - d + 1) of i + 1 keys are far
        q = np.arange(dmin, T)
        np.testing.assert_allclose(mass_beyond(A, dmin), [dmin == 1, ((q - dmin + 1) / (q + 1)).mean()])


def test_reach_cdf():
    A = _stack(prev_head(), uniform_head())
    cdf = reach_cdf(A, q_from=4)
    np.testing.assert_allclose(cdf[0], [0] + [1] * (T - 1))     # everything at distance 1
    q = np.arange(4, T)                                          # uniform: min(d+1, i+1) of i+1 keys
    np.testing.assert_allclose(cdf[1], [(np.minimum(d + 1, q + 1) / (q + 1)).mean() for d in range(T)])
    with pytest.raises(ValueError):
        reach_cdf(A, 0)


def test_on_real_model_attention():
    """Stats of a real model's maps are in range; the maps are causal and normalised."""
    m = GPT(GPTConfig(vocab_size=20, d_model=16, n_heads=4, n_layers=2, max_len=12), np.random.default_rng(0))
    m(np.random.default_rng(1).integers(0, 20, size=(3, 12)))
    for block in m.blocks:
        A = block.attn.attn_weights
        np.testing.assert_allclose(A.sum(-1), 1, atol=1e-6)
        assert np.all(np.triu(A, 1) == 0)
        s = head_stats(A)
        assert all(v.shape == (4,) for v in s.values())
        assert np.all((s["norm_entropy"] > 0.9) & (s["norm_entropy"] <= 1 + 1e-9))   # init ~ uniform
        assert np.all((s["mean_distance"] >= 0) & (s["mean_distance"] <= 11))


def test_bad_input():
    with pytest.raises(ValueError):
        head_stats(np.ones((1, 1, 3, 4)))
    with pytest.raises(ValueError):
        mass_beyond(_stack(uniform_head()), T)
