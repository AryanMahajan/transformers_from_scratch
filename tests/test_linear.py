import numpy as np

from src.nn.linear import Linear
from src.utils.numerical import check_module


def _layer(d_in=4, d_out=3, bias=True, seed=0):
    return Linear(d_in, d_out, np.random.default_rng(seed), bias=bias).astype(np.float64)


def test_forward_shapes_and_values():
    rng = np.random.default_rng(1)
    lin = _layer()
    lin.b.data[:] = rng.standard_normal(3)
    x = rng.standard_normal((2, 5, 4))                    # [B, T, D_in]
    y = lin(x)
    assert y.shape == (2, 5, 3)
    np.testing.assert_allclose(y[1, 2], x[1, 2] @ lin.W.data + lin.b.data)


def test_gradient_check_batched_3d():
    rng = np.random.default_rng(2)
    errs = check_module(_layer(), rng.standard_normal((2, 3, 4)), rng)
    assert set(errs) == {"input", "W", "b"}
    assert max(errs.values()) < 1e-8


def test_gradient_check_2d_and_no_bias():
    rng = np.random.default_rng(3)
    lin = _layer(bias=False)
    assert [n for n, _ in lin.named_parameters()] == ["W"]
    assert max(check_module(lin, rng.standard_normal((6, 4)), rng).values()) < 1e-8


def test_dW_is_sum_over_every_position():
    """dW = sum over all (b, t) of outer(x[b,t], dout[b,t])."""
    rng = np.random.default_rng(4)
    lin = _layer()
    x, g = rng.standard_normal((2, 3, 4)), rng.standard_normal((2, 3, 3))
    lin(x)
    lin.backward(g)
    expected = sum(np.outer(x[b, t], g[b, t]) for b in range(2) for t in range(3))
    np.testing.assert_allclose(lin.W.grad, expected, rtol=1e-12)
    np.testing.assert_allclose(lin.b.grad, g.sum(axis=(0, 1)), rtol=1e-12)


def test_grad_shapes_match_params_and_dx_matches_x():
    rng = np.random.default_rng(5)
    lin = _layer(8, 16)
    x = rng.standard_normal((2, 3, 8))
    lin(x)
    dx = lin.backward(np.ones((2, 3, 16)))
    assert dx.shape == x.shape
    assert lin.W.grad.shape == lin.W.shape == (8, 16)
    assert lin.b.grad.shape == lin.b.shape == (16,)


def test_default_init_is_gpt2_style_float32():
    lin = Linear(256, 512, np.random.default_rng(6))
    assert lin.W.data.dtype == np.float32
    np.testing.assert_allclose(lin.W.data.std(), 0.02, rtol=0.02)
    assert not lin.b.data.any()
