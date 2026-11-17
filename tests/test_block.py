import numpy as np
import pytest

from src.transformer.block import TransformerBlock
from src.utils.numerical import check_module


def _block(D=8, H=2, seed=0, **kw):
    return TransformerBlock(D, H, np.random.default_rng(seed), **kw).astype(np.float64)


def test_shape_param_count_and_names():
    D = 8
    blk = _block(D)
    x = np.random.default_rng(1).standard_normal((2, 5, D))
    assert blk(x).shape == x.shape
    # MHA 4D^2+3D, FFN 8D^2+5D, two LayerNorms 2*2D
    assert blk.num_parameters() == 12 * D * D + 12 * D
    assert {n.split(".")[0] for n, _ in blk.named_parameters()} == {"ln1", "attn", "ln2", "ffn"}


def test_matches_manual_composition():
    blk = _block()
    x = np.random.default_rng(2).standard_normal((2, 4, 8))
    x1 = x + blk.attn(blk.ln1(x))
    ref = x1 + blk.ffn(blk.ln2(x1))
    np.testing.assert_allclose(blk(x), ref, rtol=1e-12)


def _scale_weights(module, factor):
    """Grad checks need O(1) gradients. At std 0.02 the q-path gradient is ~2e-4,
    so finite-difference noise becomes a large *relative* error (6.7e-6) even though
    backward is correct. Backward's formulas don't depend on the weight values."""
    for p in module.parameters():
        if p.data.ndim == 2:
            p.data *= factor
    return module


@pytest.mark.parametrize("causal", [True, False])
def test_gradient_check_input_and_all_params(causal):
    rng = np.random.default_rng(3)
    blk = _scale_weights(_block(D=6, H=2, causal=causal), 20.0)       # std 0.02 -> 0.4
    errs = check_module(blk, rng.standard_normal((2, 4, 6)), rng)
    assert len(errs) == 1 + 4 + 7 + 4                      # input, 2 LNs, MHA, FFN
    assert max(errs.values()) < 1e-8


def test_causality_through_block():
    rng = np.random.default_rng(4)
    blk = _block()
    x = rng.standard_normal((1, 8, 8))
    y = blk(x)
    x2 = x.copy()
    x2[:, 5:] += 10 * rng.standard_normal(x2[:, 5:].shape)
    np.testing.assert_array_equal(blk(x2)[:, :5], y[:, :5])


def test_zero_branches_make_block_the_identity():
    """If W_o and fc2 output 0, only the residual path remains: y = x, dx = dy."""
    blk = _block()
    for lin in (blk.attn.o_proj, blk.ffn.fc2):
        lin.W.data[...] = 0.0
        lin.b.data[...] = 0.0
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 4, 8))
    np.testing.assert_array_equal(blk(x), x)
    g = rng.standard_normal(x.shape)
    np.testing.assert_allclose(blk.backward(g), g, atol=1e-15)


def _input_grad_ratio(residual: bool, n_layers=12, D=64, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [_block(D, 4, seed=seed + i) for i in range(n_layers)]
    h = rng.standard_normal((2, 16, D))
    for b in blocks:
        h = b(h) if residual else b.ffn(b.ln2(b.attn(b.ln1(h))))
    g = rng.standard_normal(h.shape)
    d = g
    for b in reversed(blocks):
        if residual:
            d = b.backward(d)
        else:
            d = b.ln1.backward(b.attn.backward(b.ln2.backward(b.ffn.backward(d))))
    return np.linalg.norm(d) / np.linalg.norm(g)


def test_residual_is_a_gradient_highway():
    """12 blocks: with residuals ||dx|| ~ ||dy||; without, the input gets starved."""
    with_res = _input_grad_ratio(True)
    without = _input_grad_ratio(False)
    assert 0.5 < with_res < 2.0
    assert without < 0.1
