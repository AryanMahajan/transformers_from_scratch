import numpy as np
import pytest

from src.attention.causal_mask import causal_mask
from src.attention.scaled_attention import attention_backward, attention_forward
from src.nn.softmax import softmax
from src.utils.numerical import numerical_gradient, relative_error


def _qkv(rng, shape=(2, 3, 5, 4)):                       # [B, H, T, d]
    return (rng.standard_normal(shape) for _ in range(3))


def test_causal_mask_layout():
    m = causal_mask(4)
    assert m.dtype == bool and m.shape == (4, 4)
    np.testing.assert_array_equal(m, np.tril(np.ones((4, 4), dtype=bool)))
    assert m.diagonal().all()                            # no fully-masked row


def test_forward_matches_naive_loop():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((4, 3)) for _ in range(3))   # T=4, d=3
    out, _ = attention_forward(q, k, v)
    for i in range(4):                                   # position i sees keys 0..i
        w = softmax(q[i] @ k[: i + 1].T / np.sqrt(3))
        np.testing.assert_allclose(out[i], w @ v[: i + 1], rtol=1e-12)


def test_future_weights_exactly_zero_and_rows_sum_to_one():
    rng = np.random.default_rng(1)
    _, cache = attention_forward(*_qkv(rng))
    A = cache["A"]
    upper = ~causal_mask(5)
    assert np.all(A[..., upper] == 0.0)                  # EXACTLY zero, not ~1e-9
    np.testing.assert_allclose(A.sum(axis=-1), 1.0, rtol=1e-12)
    assert np.all(np.isfinite(A))


@pytest.mark.parametrize("causal", [True, False])
def test_gradient_check_q_k_v(causal):
    rng = np.random.default_rng(2)
    q, k, v = _qkv(rng, (2, 2, 4, 3))
    out, cache = attention_forward(q, k, v, causal)
    g = rng.standard_normal(out.shape)
    dq, dk, dv = attention_backward(g, cache)

    def loss():
        return float(np.sum(g * attention_forward(q, k, v, causal)[0]))

    for analytic, x in ((dq, q), (dk, k), (dv, v)):
        assert relative_error(analytic, numerical_gradient(loss, x)) < 1e-8


def test_causality_perturbing_future_leaves_past_unchanged():
    """Exit criterion: changing token t+k (in q, k AND v) can't affect outputs <= t."""
    rng = np.random.default_rng(3)
    q, k, v = _qkv(rng, (1, 1, 8, 4))
    out, _ = attention_forward(q, k, v)
    t = 4
    for x in (q, k, v):
        x2 = x.copy()
        x2[..., t + 1:, :] += rng.standard_normal(x2[..., t + 1:, :].shape) * 10
        args = [x2 if a is x else a for a in (q, k, v)]
        out2, _ = attention_forward(*args)
        np.testing.assert_array_equal(out2[..., : t + 1, :], out[..., : t + 1, :])
        assert not np.allclose(out2[..., t + 1:, :], out[..., t + 1:, :])


def test_first_position_copies_its_own_value():
    rng = np.random.default_rng(4)
    q, k, v = _qkv(rng)
    out, _ = attention_forward(q, k, v)
    np.testing.assert_allclose(out[..., 0, :], v[..., 0, :])   # only key 0 is visible


def test_sqrt_d_scaling_keeps_softmax_unsaturated():
    """Unscaled scores have std ~ sqrt(d): softmax ~one-hot, gradients vanish."""
    rng = np.random.default_rng(5)
    d, T = 256, 64
    q, k = rng.standard_normal((T, d)), rng.standard_normal((T, d))
    raw = q @ k.T
    np.testing.assert_allclose(raw.std(), np.sqrt(d), rtol=0.1)          # Var(q.k) = d
    np.testing.assert_allclose((raw / np.sqrt(d)).std(), 1.0, rtol=0.1)

    def mean_max_prob_and_jac(s):
        p = softmax(s)
        jac = (p * (1 - p)).mean()                       # diagonal of the softmax Jacobian
        return p.max(axis=-1).mean(), jac

    p_raw, j_raw = mean_max_prob_and_jac(raw)
    p_scaled, j_scaled = mean_max_prob_and_jac(raw / np.sqrt(d))
    assert p_raw > 0.6 and p_scaled < 0.2                # saturated vs spread out
    assert j_scaled > 3 * j_raw                          # unscaled -> much weaker gradients
