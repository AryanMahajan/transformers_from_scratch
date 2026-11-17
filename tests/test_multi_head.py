import numpy as np
import pytest

from src.attention.multi_head import MultiHeadAttention, merge_heads, split_heads
from src.attention.scaled_attention import attention_forward
from src.utils.numerical import check_module


def _mha(D=8, H=2, seed=0, causal=True):
    return MultiHeadAttention(D, H, np.random.default_rng(seed), causal=causal).astype(np.float64)


def test_split_merge_shapes_and_inverse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 12))
    h = split_heads(x, 3)
    assert h.shape == (2, 3, 5, 4)
    np.testing.assert_array_equal(merge_heads(h), x)
    # head j of token t is features [j*D_h : (j+1)*D_h] of that same token
    np.testing.assert_array_equal(h[1, 2, 4], x[1, 4, 8:12])


def test_output_shape_param_count_and_names():
    mha = _mha(D=8, H=2)
    x = np.random.default_rng(1).standard_normal((3, 6, 8))
    assert mha(x).shape == (3, 6, 8)
    assert mha.attn_weights.shape == (3, 2, 6, 6)
    assert mha.num_parameters() == 4 * 8 * 8 + 3 * 8                # 4 W's, 3 biases (no key bias); independent of H
    assert {n.split(".")[0] for n, _ in mha.named_parameters()} == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_matches_explicit_per_head_loop():
    rng = np.random.default_rng(2)
    D, H, Dh = 8, 2, 4
    mha = _mha(D, H)
    x = rng.standard_normal((2, 5, D))
    y = mha(x)
    q, k, v = (p(x) for p in (mha.q_proj, mha.k_proj, mha.v_proj))
    heads = []
    for h in range(H):                                              # slice features per head
        sl = slice(h * Dh, (h + 1) * Dh)
        out, _ = attention_forward(q[..., sl], k[..., sl], v[..., sl])
        heads.append(out)
    ref = np.concatenate(heads, axis=-1) @ mha.o_proj.W.data + mha.o_proj.b.data
    np.testing.assert_allclose(y, ref, rtol=1e-12)


@pytest.mark.parametrize("causal", [True, False])
def test_gradient_check_all_params_and_input(causal):
    rng = np.random.default_rng(3)
    errs = check_module(_mha(D=6, H=3, causal=causal), rng.standard_normal((2, 4, 6)), rng)
    assert len(errs) == 1 + 4 + 3                                   # input + 4 W + 3 b
    assert max(errs.values()) < 1e-8


def test_causality_through_full_mha():
    rng = np.random.default_rng(4)
    mha = _mha()
    x = rng.standard_normal((1, 8, 8))
    y = mha(x)
    t = 3
    x2 = x.copy()
    x2[:, t + 1:] += 10 * rng.standard_normal(x2[:, t + 1:].shape)
    y2 = mha(x2)
    np.testing.assert_array_equal(y2[:, : t + 1], y[:, : t + 1])
    future_w = mha.attn_weights[..., np.triu_indices(8, k=1)[0], np.triu_indices(8, k=1)[1]]
    assert np.all(future_w == 0.0)


def test_single_head_equals_plain_attention():
    rng = np.random.default_rng(5)
    mha = _mha(D=4, H=1)
    x = rng.standard_normal((2, 3, 4))
    q, k, v = (p(x) for p in (mha.q_proj, mha.k_proj, mha.v_proj))
    out, _ = attention_forward(q, k, v)
    np.testing.assert_allclose(mha(x), mha.o_proj(out), rtol=1e-12)


def test_naive_reshape_without_transpose_is_wrong():
    """reshape(B,H,T,Dh) directly splits by POSITION, not by feature."""
    rng = np.random.default_rng(6)
    B, T, D, H = 1, 4, 4, 2
    x = rng.standard_normal((B, T, D))
    naive = x.reshape(B, H, T, D // H)
    correct = split_heads(x, H)
    assert not np.array_equal(naive, correct)
    # naive "head 1" contains only tokens 2 and 3; token 0 never appears in it
    assert set(map(tuple, naive[0, 1].reshape(-1, D)))  == {tuple(x[0, 2]), tuple(x[0, 3])}
    # correct head 1 sees every token (its second half of features)
    np.testing.assert_array_equal(correct[0, 1], x[0, :, 2:])


def test_key_bias_would_be_useless():
    """Adding any vector to all keys shifts each score row by a constant -> output unchanged."""
    rng = np.random.default_rng(7)
    mha = _mha()
    assert mha.k_proj.b is None
    x = rng.standard_normal((2, 5, 8))
    y = mha(x)
    b = rng.standard_normal(8) * 5
    q = mha.q_proj(x)
    k = mha.k_proj(x)
    v = mha.v_proj(x)
    out_b, _ = attention_forward(split_heads(q, 2), split_heads(k + b, 2), split_heads(v, 2))
    np.testing.assert_allclose(mha.o_proj(merge_heads(out_b)), y, atol=1e-12)


def test_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3, np.random.default_rng(0))
