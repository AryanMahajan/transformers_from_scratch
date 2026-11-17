import numpy as np
import pytest

from src.core.module import Module
from src.core.parameter import Parameter
from src.utils.numerical import check_module


class Scale(Module):
    """Toy layer: y = x * w + b, with w, b of shape [D] broadcast over [..., D]."""

    def __init__(self, D, assign_instead_of_accumulate=False):
        self.w = Parameter(np.linspace(0.5, 1.5, D))
        self.b = Parameter(np.zeros(D))
        self._assign = assign_instead_of_accumulate

    def forward(self, x):
        self.x = x
        return x * self.w.data + self.b.data

    def backward(self, dout):
        lead = tuple(range(dout.ndim - 1))       # sum over all batch axes
        dw = np.sum(dout * self.x, axis=lead)
        db = np.sum(dout, axis=lead)
        if self._assign:                         # the bug we want to catch
            self.w.grad = dw
            self.b.grad = db
        else:
            self.w.grad += dw
            self.b.grad += db
        return dout * self.w.data


class Stack(Module):
    def __init__(self, D, n, tie=False, buggy=False):
        self.layers = [Scale(D, buggy) for _ in range(n)]
        self.head = Scale(D, buggy)
        if tie:
            self.head.w = self.layers[0].w       # same Parameter object

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def backward(self, dout):
        dout = self.head.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


def test_registry_names_order_and_count():
    m = Stack(D=3, n=2)
    names = [n for n, _ in m.named_parameters()]
    assert names == ["layers.0.w", "layers.0.b", "layers.1.w", "layers.1.b", "head.w", "head.b"]
    assert m.num_parameters() == 6 * 3
    assert len(list(m.modules())) == 4               # Stack + 2 layers + head


def test_tied_parameter_listed_once():
    m = Stack(D=3, n=2, tie=True)
    names = [n for n, _ in m.named_parameters()]
    assert "head.w" not in names                     # deduped (first path wins)
    assert len(m.parameters()) == 5


def test_backward_accumulates_and_zero_grad_resets():
    rng = np.random.default_rng(0)
    m = Scale(4)
    x, g = rng.standard_normal((2, 4)), rng.standard_normal((2, 4))
    m.forward(x); m.backward(g)
    once = m.w.grad.copy()
    m.forward(x); m.backward(g)
    np.testing.assert_allclose(m.w.grad, 2 * once)   # +=, not =
    m.zero_grad()
    assert not m.w.grad.any() and not m.b.grad.any()


def test_check_module_passes_for_correct_layer():
    rng = np.random.default_rng(1)
    errs = check_module(Stack(D=4, n=2).astype(np.float64), rng.standard_normal((2, 3, 4)), rng)
    assert set(errs) == {"input", "layers.0.w", "layers.0.b", "layers.1.w", "layers.1.b", "head.w", "head.b"}
    assert max(errs.values()) < 1e-8


def test_tied_weight_gradient_is_sum_of_both_uses():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 4))
    good = check_module(Stack(D=4, n=2, tie=True), x, rng)
    assert max(good.values()) < 1e-8
    # If backward assigned (=) instead of accumulating (+=), the head's
    # contribution to the shared w gets overwritten -> check fails.
    bad = check_module(Stack(D=4, n=2, tie=True, buggy=True), x, rng)
    assert bad["layers.0.w"] > 1e-2


def test_state_dict_roundtrip_and_copy_semantics():
    a, b = Stack(D=3, n=2), Stack(D=3, n=2)
    for p in a.parameters():
        p.data += 1.0
    sd = a.state_dict()
    sd["head.w"][0] = 999.0                          # mutating snapshot must not touch model
    assert a.head.w.data[0] != 999.0
    b.load_state_dict(a.state_dict())
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        np.testing.assert_array_equal(pa.data, pb.data)


def test_load_state_dict_rejects_mismatch():
    m = Stack(D=3, n=2)
    sd = m.state_dict()
    with pytest.raises(KeyError):
        m.load_state_dict({k: v for k, v in sd.items() if k != "head.b"})
    sd["head.b"] = np.zeros(5)
    with pytest.raises(ValueError):
        m.load_state_dict(sd)


def test_astype_and_train_eval_propagate():
    m = Stack(D=3, n=2).astype(np.float32)
    assert all(p.data.dtype == np.float32 and p.grad.dtype == np.float32 for p in m.parameters())
    m.eval()
    assert not any(sub.training for sub in m.modules())
    m.train()
    assert all(sub.training for sub in m.modules())
