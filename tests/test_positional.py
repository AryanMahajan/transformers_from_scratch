import numpy as np
import pytest

from src.attention.multi_head import MultiHeadAttention
from src.nn.positional import SinusoidalPositionalEncoding, sinusoidal_encoding
from src.utils.numerical import check_module


def test_matches_explicit_formula():
    T, D = 7, 6
    pe = sinusoidal_encoding(T, D, dtype=np.float64)
    assert pe.shape == (T, D)
    for t in range(T):
        for i in range(D // 2):
            w = 10000.0 ** (-2 * i / D)
            assert pe[t, 2 * i] == pytest.approx(np.sin(t * w), abs=1e-15)
            assert pe[t, 2 * i + 1] == pytest.approx(np.cos(t * w), abs=1e-15)


def test_position_zero_bounded_and_constant_norm():
    pe = sinusoidal_encoding(100, 16, dtype=np.float64)
    np.testing.assert_array_equal(pe[0, 0::2], 0.0)                  # sin 0
    np.testing.assert_array_equal(pe[0, 1::2], 1.0)                  # cos 0
    assert np.abs(pe).max() <= 1.0
    # sin^2 + cos^2 = 1 per pair -> every row has norm sqrt(D/2)
    np.testing.assert_allclose(np.linalg.norm(pe, axis=1), np.sqrt(8), rtol=1e-12)


def test_frequencies_form_geometric_series():
    D = 16
    pe = sinusoidal_encoding(2, D, dtype=np.float64)
    omega = np.arcsin(pe[1, 0::2])                                   # t = 1 -> angle = omega (all < pi/2)
    ratios = omega[1:] / omega[:-1]
    np.testing.assert_allclose(ratios, 10000.0 ** (-2 / D), rtol=1e-10)
    assert omega[0] == pytest.approx(1.0)                            # fastest: period 2*pi
    # slowest wavelength = 2*pi * 10000^((D-2)/D)  (~ 2*pi*10^4 for large D)


def test_relative_shift_is_a_rotation_independent_of_t():
    """PE[t+k] = R(k) PE[t] for one fixed matrix R(k), for every t."""
    T, D, k = 50, 8, 5
    pe = sinusoidal_encoding(T + k, D, dtype=np.float64)
    omega = 10000.0 ** (-np.arange(0, D, 2) / D)
    R = np.zeros((D, D))
    for i, w in enumerate(omega):                                    # [sin; cos] pair rotation
        c, s = np.cos(k * w), np.sin(k * w)
        R[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = [[c, s], [-s, c]]
    np.testing.assert_allclose(pe[k:] , pe[:T] @ R.T, atol=1e-12)


def test_dot_product_depends_only_on_offset():
    pe = sinusoidal_encoding(64, 16, dtype=np.float64)
    G = pe @ pe.T                                                    # G[t, t+k] = sum_i cos(k w_i)
    for k in range(10):
        diag = np.diagonal(G, offset=k)
        np.testing.assert_allclose(diag, diag[0], atol=1e-10)
    assert G[0, 0] > G[0, 1] > G[0, 2]                               # nearby positions more similar


def test_all_positions_distinct():
    pe = sinusoidal_encoding(512, 32, dtype=np.float64)
    d2 = ((pe[:, None, :] - pe[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    assert d2.min() > 1e-3


def test_module_adds_table_has_no_params_and_identity_backward():
    rng = np.random.default_rng(0)
    pos = SinusoidalPositionalEncoding(16, 8).astype(np.float64)
    x = rng.standard_normal((3, 10, 8))
    np.testing.assert_array_equal(pos(x), x + sinusoidal_encoding(16, 8, dtype=np.float64)[:10])
    assert pos.num_parameters() == 0 and pos.state_dict() == {}
    g = rng.standard_normal(x.shape)
    assert pos.backward(g) is g
    errs = check_module(pos, x, rng)
    assert list(errs) == ["input"] and errs["input"] < 1e-8


def test_rejects_odd_dim_and_too_long_sequence():
    with pytest.raises(ValueError):
        sinusoidal_encoding(10, 7)
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(4, 8)(np.zeros((1, 5, 8)))


def test_attention_is_order_blind_without_positions():
    """Non-causal MHA is permutation-equivariant; adding PE breaks that."""
    rng = np.random.default_rng(1)
    T, D = 6, 8
    mha = MultiHeadAttention(D, 2, rng, causal=False).astype(np.float64)
    x = rng.standard_normal((1, T, D))
    perm = rng.permutation(T)
    # no positions: permuting the input just permutes the output
    np.testing.assert_allclose(mha(x[:, perm]), mha(x)[:, perm], atol=1e-12)
    # with positions: the token at a new place gets a different vector
    pos = SinusoidalPositionalEncoding(T, D).astype(np.float64)
    assert not np.allclose(mha(pos(x[:, perm])), mha(pos(x))[:, perm], atol=1e-6)
