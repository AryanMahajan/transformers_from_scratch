import math

import numpy as np
import pytest

from src.core.parameter import Parameter
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm, global_grad_norm
from src.optim.schedule import cosine_lr


def _params(rng, shapes=((3, 4), (4,), (2, 2, 2)), scale=1.0):
    ps = [Parameter(np.zeros(s)) for s in shapes]
    for p in ps:
        p.grad[...] = scale * rng.standard_normal(p.shape)
    return ps


# ------------------------------------------------------------------ clipping
def test_global_norm_is_norm_of_all_grads_concatenated():
    ps = _params(np.random.default_rng(0))
    flat = np.concatenate([p.grad.ravel() for p in ps])
    assert global_grad_norm(ps) == pytest.approx(np.linalg.norm(flat), rel=1e-12)


def test_below_threshold_untouched_and_returns_norm():
    ps = _params(np.random.default_rng(1))
    before = [p.grad.copy() for p in ps]
    n = clip_grad_norm(ps, max_norm=1e6)
    assert n == pytest.approx(global_grad_norm(ps))
    for p, b in zip(ps, before):
        np.testing.assert_array_equal(p.grad, b)


def test_above_threshold_caps_norm_and_keeps_direction():
    ps = _params(np.random.default_rng(2), scale=10.0)
    before = np.concatenate([p.grad.ravel() for p in ps])
    pre = clip_grad_norm(ps, max_norm=1.0)
    after = np.concatenate([p.grad.ravel() for p in ps])
    assert pre == pytest.approx(np.linalg.norm(before))           # returns PRE-clip norm
    assert np.linalg.norm(after) == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(after / np.linalg.norm(after), before / np.linalg.norm(before), rtol=1e-10)


def test_per_tensor_clipping_would_change_direction():
    """Why GLOBAL: clipping each tensor to the same norm separately rotates the update."""
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(5) * 10, rng.standard_normal(5) * 0.1
    g = np.concatenate([a, b])
    per_tensor = np.concatenate([a / np.linalg.norm(a), b / max(np.linalg.norm(b), 1.0)])
    cos = per_tensor @ g / (np.linalg.norm(per_tensor) * np.linalg.norm(g))
    assert cos < 0.99                                            # direction changed
    ps = [Parameter(np.zeros(5)), Parameter(np.zeros(5))]
    ps[0].grad[...], ps[1].grad[...] = a, b
    clip_grad_norm(ps, 1.0)
    glob = np.concatenate([ps[0].grad, ps[1].grad])
    assert glob @ g / (np.linalg.norm(glob) * np.linalg.norm(g)) == pytest.approx(1.0)


def test_zero_gradients_are_safe():
    ps = [Parameter(np.zeros(3))]
    assert clip_grad_norm(ps, 1.0) == 0.0
    assert np.all(ps[0].grad == 0)


def _adam_step_sizes(clip: bool):
    rng = np.random.default_rng(0)
    G = rng.standard_normal((400, 50))
    p = Parameter(np.zeros(50))
    opt = AdamW([("w", p)], lr=0.01)
    sizes = []
    for t, g in enumerate(G):
        p.grad[...] = g * (1000.0 if t == 100 else 1.0)          # one bad batch at step 100
        if clip:
            clip_grad_norm([p], 2 * np.sqrt(50))
        prev = p.data.copy()
        opt.step()
        sizes.append(np.abs(p.data - prev).mean())
    return np.array(sizes)


def test_one_spike_stalls_adam_without_clipping():
    """v remembers the spike for ~1/(1-b2) = 1000 steps -> later steps shrink ~sqrt(1000)."""
    no, yes = _adam_step_sizes(False), _adam_step_sizes(True)
    normal = no[50:100].mean()
    assert no[100] > 4 * normal                                  # oversized spike step
    assert no[201:400].mean() < normal / 10                      # long stall afterwards
    assert yes[201:400].mean() == pytest.approx(normal, rel=0.2) # clipped: unaffected


# ------------------------------------------------------------------ schedule
def test_warmup_linear_then_peak():
    lrs = [cosine_lr(s, 1.0, 10, 100) for s in range(10)]
    np.testing.assert_allclose(lrs, np.arange(1, 11) / 10)
    assert lrs[0] > 0                                            # step 0 still learns
    assert cosine_lr(10, 1.0, 10, 100) == pytest.approx(1.0)


def test_cosine_midpoint_end_and_after():
    mx, mn = 3e-4, 3e-5
    assert cosine_lr(55, mx, 10, 100, mn) == pytest.approx((mx + mn) / 2)
    assert cosine_lr(100, mx, 10, 100, mn) == pytest.approx(mn)
    assert cosine_lr(500, mx, 10, 100, mn) == mn


def test_monotone_decreasing_after_warmup():
    lrs = [cosine_lr(s, 1.0, 10, 100, 0.1) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert min(lrs) >= 0.1


def test_no_warmup_and_validation():
    assert cosine_lr(0, 1.0, 0, 10) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_lr(0, 1.0, 10, 10)
    assert math.isfinite(cosine_lr(10, 1.0, 0, 10))
