import numpy as np
import pytest

from src.core.parameter import Parameter
from src.optim.adamw import AdamW


def _p(values, grad=None, dtype=np.float64):
    p = Parameter(np.array(values, dtype=dtype))
    if grad is not None:
        p.grad[...] = grad
    return p


def test_first_step_is_lr_times_sign_thanks_to_bias_correction():
    """t=1: m_hat = g, v_hat = g^2 -> step = lr * g/|g|, whatever |g| is."""
    g = np.array([1e-4, -3.0, 250.0])
    p = _p(np.zeros(3), g)
    AdamW([("w", p)], lr=0.01).step()
    np.testing.assert_allclose(p.data, -0.01 * np.sign(g), rtol=1e-3)
    # without bias correction the first step would be lr * 0.1|g| / sqrt(0.001 g^2) = 3.16 lr
    assert 0.1 / np.sqrt(0.001) == pytest.approx(3.162, abs=1e-3)


def test_matches_written_out_formulas_for_several_steps():
    rng = np.random.default_rng(0)
    p = _p(rng.standard_normal(5))
    ref = p.data.copy()
    m = np.zeros(5); v = np.zeros(5)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    opt = AdamW([("w", p)], lr=lr)
    for t in range(1, 6):
        g = rng.standard_normal(5)
        p.grad[...] = g
        opt.step()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        ref = ref - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    np.testing.assert_allclose(p.data, ref, rtol=1e-12)


def test_invariant_to_gradient_scale():
    """Multiplying the loss by 1000 leaves Adam's steps (almost) unchanged."""
    rng = np.random.default_rng(1)
    grads = rng.standard_normal((10, 4))
    finals = []
    for scale in (1.0, 1000.0):
        p = _p(np.zeros(4))
        opt = AdamW([("w", p)], lr=0.01)
        for g in grads:
            p.grad[...] = scale * g
            opt.step()
        finals.append(p.data.copy())
    np.testing.assert_allclose(finals[0], finals[1], rtol=1e-6)


def test_minimises_ill_conditioned_quadratic():
    """f = sum a_i x_i^2 with curvatures spanning 1e4: per-parameter steps cope."""
    a = np.array([1.0, 1e2, 1e4])
    p = _p(np.ones(3))
    opt = AdamW([("x", p)], lr=0.05)
    for _ in range(2000):
        p.grad[...] = 2 * a * p.data
        opt.step()
    assert np.abs(p.data).max() < 1e-2


def test_weight_decay_shrinks_matrices_only():
    W = _p(np.ones((2, 2)))                      # decayed
    b = _p(np.ones(2))                           # bias / LN params: not decayed
    opt = AdamW([("W", W), ("b", b)], lr=0.1, weight_decay=0.5)
    for _ in range(10):                          # zero gradients: decay is the only force
        opt.step()
    np.testing.assert_allclose(W.data, (1 - 0.1 * 0.5) ** 10, rtol=1e-12)
    np.testing.assert_array_equal(b.data, 1.0)


def test_decoupled_decay_differs_from_l2_in_adam():
    """Two weights, gradients 100x apart. Decay effect = (no-decay run) - (decay run),
    same noise, so the random walk cancels out.
    AdamW: both shrink by ~1-(1-lr*wd)^n. L2-in-Adam: wd*p is divided by sqrt(v), so the
    small-gradient weight is decayed far more -- regularisation depends on gradient scale."""
    noise = np.random.default_rng(2).standard_normal((200, 2)) * np.array([0.01, 1.0])
    lr, wd = 0.01, 0.1

    def run(mode):
        p = _p(np.ones((1, 2)))
        opt = AdamW([("W", p)], lr=lr, weight_decay=wd if mode == "adamw" else 0.0)
        for g in noise:
            p.grad[...] = g + (wd * p.data if mode == "l2" else 0.0)
            opt.step()
        return p.data[0]

    base = run("none")
    adamw, l2 = base - run("adamw"), base - run("l2")
    theory = 1 - (1 - lr * wd) ** 200                              # 0.181
    np.testing.assert_allclose(adamw, theory, rtol=0.25)            # same for both weights
    assert 0.6 < adamw[0] / adamw[1] < 1.5
    assert l2[0] / l2[1] > 3                                        # measured 4.4


def test_state_dict_resume_is_bit_identical():
    rng = np.random.default_rng(3)
    grads = rng.standard_normal((10, 3))
    p_full = _p(np.zeros(3), dtype=np.float32)
    opt_full = AdamW([("w", p_full)], lr=0.01, weight_decay=0.0)
    p_res = _p(np.zeros(3), dtype=np.float32)
    opt_res = AdamW([("w", p_res)], lr=0.01)
    for i, g in enumerate(grads):
        p_full.grad[...] = g
        opt_full.step()
        if i < 5:
            p_res.grad[...] = g
            opt_res.step()
    saved, saved_w = opt_res.state_dict(), p_res.data.copy()
    p_new = _p(saved_w, dtype=np.float32)        # "restart the process"
    opt_new = AdamW([("w", p_new)], lr=0.01)
    opt_new.load_state_dict(saved)
    for g in grads[5:]:
        p_new.grad[...] = g
        opt_new.step()
    np.testing.assert_array_equal(p_new.data, p_full.data)
    assert p_new.data.dtype == np.float32 and opt_new.m["w"].dtype == np.float32


def test_updates_in_place_and_rejects_duplicates():
    p = _p(np.zeros(3), np.ones(3), dtype=np.float32)
    buf = p.data
    AdamW([("w", p)]).step()
    assert p.data is buf and p.data.dtype == np.float32
    with pytest.raises(ValueError):
        AdamW([("a", p), ("b", p)])
