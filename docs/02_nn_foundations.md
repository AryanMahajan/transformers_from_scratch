# Phase 2: Neural-Network Foundations (study notes)

Revision notes for each Phase 2 component: what it's for, the math and
design, gotchas, and self-check questions.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

**Notation:** as in `TECHNICAL_DOC.md`: `[B,T,D]` activations, `y = x @ W + b`
with `W ∈ R^{D_in×D_out}`, `dout = ∂L/∂y`.

---

## 1. `Parameter` and `Module`: `src/core/parameter.py`, `src/core/module.py`

### What it's for
This is the skeleton every layer plugs into. The optimizer and trainer only
ever need: "all weights", "reset gradients", "save/load". They never need
layer-specific knowledge.

### `Parameter`
`data` (the values) plus `grad` (∂L/∂data, same shape). An optimizer step is
`p.data -= lr · f(p.grad)` for each `p`.

### Module contract
```python
y  = m.forward(x)     # cache what backward needs
dx = m.backward(dy)   # p.grad += ∂L/∂p ; return ∂L/∂x (same shape as x)
```

### Why `+=` (accumulate) instead of `=`
The multivariable chain rule: if $W$ is used in several places,
$\partial L/\partial W = \sum_{\text{uses}} \partial L/\partial W|_{\text{use}}$.
- **Tied embeddings:** $E$ is used at the input lookup and in the output
  projection. Both backward passes must add into the same `E.grad`.
- So `zero_grad()` is mandatory **once per step**. Otherwise gradients from
  earlier steps pile up.
- `test_tied_weight_gradient_is_sum_of_both_uses` shows an `=` bug failing
  the gradient check (rel. err > 1e-2).

### Parameter registry
`named_parameters()` walks `vars(self)` recursively, in definition order:
- `Parameter` → yield it as `"prefix.name"`
- `Module` → recurse
- `list`/`tuple` of Modules → recurse with an index (`blocks.0.attn.W_q`)

**Dedupe by `id()`:** a tied parameter is reachable via two paths but is
listed **once**. Otherwise the optimizer would update it twice per step (and
Adam's moment estimates would be wrong).

### Utilities
| Method | Purpose |
|---|---|
| `zero_grad()` | `grad.fill(0)` **in place**, so the buffer identity is kept |
| `state_dict()` | `{name: data.copy()}`. The copy means later training doesn't change the snapshot |
| `load_state_dict()` | checks names and shapes, fails loudly, writes **in place** (`p.data[...] = …`) so references held by the optimizer stay valid |
| `astype(dtype)` | float32 to train, float64 to grad-check |
| `train()`/`eval()` | flag that propagates to all submodules (for dropout later) |

### `check_module` (`src/utils/numerical.py`)
The VJP trick applied to a whole layer. Draw a random `g`, set
$L = \sum g\odot m(x)$, run backward once, then use finite differences on
**x and every parameter**, nudging `p.data` in place. This is why
`numerical_gradient` takes a zero-argument closure. Integer inputs (token
ids) are skipped automatically.

### Gotchas
- `load_state_dict` must write in place. If you rebind `p.data = new`, it
  still works here, but in-place is the safer habit.
- Shared parameters must be the **same object** (`head.w = layers[0].w`),
  not a copy.
- The forward cache is overwritten on every forward, so you get one backward
  per forward. That's why `check_module` runs backward **before** the
  numeric loop.

### Tests (`tests/test_module.py`, toy `Scale` layer `y = x⊙w + b`)
- Registry names and order.
- The tied parameter is listed once.
- `+=` accumulation, and `zero_grad`.
- `check_module` passes (< 1e-8).
- The tied gradient is the sum of both uses, and an `=` bug is caught.
- `state_dict` round-trip and copy semantics.
- Mismatches are rejected.
- `astype`, `train`/`eval` propagate.

---

## Self-check questions
1. Why does `backward` accumulate into `grad` instead of assigning? Give a concrete case from our model.
2. What goes wrong if `zero_grad()` is forgotten?
3. Why must `parameters()` dedupe tied weights? What would Adam do otherwise?
4. Why does `state_dict()` return copies?
5. How does `check_module` check parameter gradients without any change to the layer's code?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| A minimal Module/parameters/zero_grad design | Karpathy, micrograd `nn.py`: https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py |
| Manual backprop through layers | Karpathy, *Becoming a Backprop Ninja*: https://www.youtube.com/watch?v=q8SA3rM6ckI |
