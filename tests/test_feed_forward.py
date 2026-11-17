import numpy as np

from src.attention.multi_head import MultiHeadAttention
from src.nn.activations import gelu
from src.transformer.feed_forward import FeedForward
from src.utils.numerical import check_module


def _ffn(D=6, d_ff=None, seed=0):
    return FeedForward(D, np.random.default_rng(seed), d_ff=d_ff).astype(np.float64)


def test_shapes_default_width_and_param_count():
    D = 8
    ffn = _ffn(D)
    assert ffn.fc1.W.shape == (D, 4 * D) and ffn.fc2.W.shape == (4 * D, D)
    x = np.random.default_rng(1).standard_normal((2, 5, D))
    assert ffn(x).shape == x.shape
    F = 4 * D
    assert ffn.num_parameters() == 2 * D * F + F + D              # = 8D^2 + 5D


def test_matches_manual_formula():
    ffn = _ffn(D=4, d_ff=10)
    x = np.random.default_rng(2).standard_normal((3, 4))
    W1, b1, W2, b2 = (p.data for p in ffn.parameters())
    np.testing.assert_allclose(ffn(x), gelu(x @ W1 + b1) @ W2 + b2, rtol=1e-12)


def test_gradient_check_input_and_all_params():
    rng = np.random.default_rng(3)
    errs = check_module(_ffn(D=5, d_ff=7), rng.standard_normal((2, 3, 5)), rng)
    assert set(errs) == {"input", "fc1.W", "fc1.b", "fc2.W", "fc2.b"}
    assert max(errs.values()) < 1e-8


def test_position_wise_tokens_do_not_interact():
    """Changing token t changes output t only; jacobian across positions is 0."""
    rng = np.random.default_rng(4)
    ffn = _ffn(D=6)
    x = rng.standard_normal((1, 7, 6))
    y = ffn(x)
    x2 = x.copy()
    x2[0, 3] += 5.0
    y2 = ffn(x2)
    changed = ~np.all(y2 == y, axis=-1)[0]
    np.testing.assert_array_equal(changed, np.arange(7) == 3)
    # equivalently: running each token alone gives identical results
    np.testing.assert_allclose(np.stack([ffn(x[:, t]) for t in range(7)], axis=1), y, rtol=1e-12)


def test_out_std_controls_fc2_init_only():
    ffn = FeedForward(256, np.random.default_rng(5), out_std=0.002)
    assert abs(ffn.fc1.W.data.std() - 0.02) < 1e-3
    assert abs(ffn.fc2.W.data.std() - 0.002) < 1e-4


def test_ffn_holds_two_thirds_of_block_weights():
    D = 64
    ffn = FeedForward(D, np.random.default_rng(6))
    mha = MultiHeadAttention(D, 4, np.random.default_rng(6))
    w = lambda m: sum(p.data.size for n, p in m.named_parameters() if n.endswith("W"))
    assert w(ffn) == 8 * D * D and w(mha) == 4 * D * D               # 8D^2 vs 4D^2
