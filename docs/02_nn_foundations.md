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

## 2. Linear: `src/nn/linear.py`

### What it's for
Q/K/V/O projections, both FFN layers, and the LM head. Almost all of the
model's FLOPs are here.

### Forward
$y = xW + b$, with `x:[B,T,D_in]`, `W:[D_in,D_out]`, `b:[D_out]`. `@`
broadcasts over the leading axes, so all $N = B\cdot T$ positions share one $W$.

### Backward (index notation, flattened positions n)
$y_{nj} = \sum_i x_{ni}W_{ij} + b_j$

| Gradient | Index form | Matrix form | Shapes |
|---|---|---|---|
| $dx$ | $\sum_j \text{dout}_{nj}W_{ij}$ | `dout @ W.T` | `[N,Dout]·[Dout,Din]` |
| $dW$ | $\sum_n x_{ni}\,\text{dout}_{nj}$ | `x.T @ dout` | `[Din,N]·[N,Dout]` |
| $db$ | $\sum_n \text{dout}_{nj}$ | `dout.sum(0)` | `[Dout]` |

### Key ideas
1. **Shapes force the answer.** `dW` must be `[D_in,D_out]`, and only
   `xᵀ·dout` fits. Use this to place transposes.
2. **The sum over n is accumulation.** $W$ is reused at every one of the B·T
   positions, so its gradient is the sum of B·T contributions (§1's rule).
   The matmul does the sum.
3. **Flatten `[B,T,D] → [B·T,D]`** so that one matmul contracts over both batch
   axes. `reshape` is free for contiguous arrays.
4. **Cost:** forward is 1 matmul, backward is 2, so training ≈ 3× forward,
   which gives the "≈ 6·params FLOPs per training token" rule.
5. **Cache:** only `x` (needed for `dW`). `dx` doesn't need `x`.

### Code notes
- `bias=False` sets `self.b = None`, which the registry skips automatically.
  Useful for Q/K/V projections and the tied LM head.
- Default init: $\mathcal N(0, 0.02^2)$ weights (GPT-2), zero bias, float32.
  `std` can be overridden, e.g. `gpt2_residual_std(N)` for `W_o` and `W_2`.
- The cache is a reference to `x`, not a copy. That's safe because callers
  don't mutate activations in place.

### Tests (`tests/test_linear.py`)
- Forward values at one position.
- `check_module` on 3-D input (measured errors: input 8.7e-11, W 9.5e-13,
  b 1.2e-12) and on 2-D input with no bias.
- `dW` equals an explicit sum of outer products over every (b, t).
- Gradient shapes match parameter shapes.
- Default init.

---

## 3. Embedding: `src/nn/embedding.py`

### What it's for
- Token embedding: ids `[B,T]` → `[B,T,D]`.
- Learned positional embedding: the same class with `arange(T)`, broadcast over B.
- Later, the tied LM head: `logits = h @ Eᵀ` reuses the same $E$, so both
  gradient contributions **accumulate** (§1).

### Forward = one-hot matmul, without building the one-hot
$$\text{out}[n] = E[\text{ids}_n] \iff \text{out} = \text{onehot}(\text{ids})\cdot E$$

Indexing costs O(N·D). The explicit one-hot matmul would cost O(N·V·D).

### Backward: Linear's formula with x = one-hot
$$dE = \text{onehot}(\text{ids})^\top\cdot\text{dout}\ \Rightarrow\ dE[v] = \sum_{n:\,\text{ids}_n=v}\text{dout}[n]$$

- This is a **scatter-add**. Tokens absent from the batch get 0 gradient.
- **There's no input gradient.** Ids are discrete, so `backward` returns
  `None` and backprop stops here.

### Trap 1: buffered fancy-index `+=` ⚠️
`E.grad[ids] += dout` runs as `tmp = E.grad[ids] + dout; E.grad[ids] = tmp`.
With repeated ids, **the last write wins** and the other contributions are
lost. There's no error, only silently wrong gradients, and repeats are
constant in real text. **Fix:** `np.add.at(E.grad, ids, dout)`, an unbuffered
version where every occurrence adds. The NumPy docs give `a[[0,0]] += 1`
(adds once) against `np.add.at(a, [0,0], 1)` (adds twice).

### Trap 2: negative ids
`E[-1]` silently returns the last row. The forward pass checks
`0 ≤ ids < V` and integer dtype.

### Tests (`tests/test_embedding.py`)
- The lookup equals `onehot @ E`.
- `check_module` with heavily repeated ids (measured: 9.2e-13).
- Scatter-add sums all three occurrences, unused rows stay zero, and backward
  returns `None`.
- A demo of the buffered `+=` bug (gets 1.0 where it should be 3.0).
- Rejects −1, V, and float ids.
- Positional-embedding broadcast.

---

## Self-check questions
1. Why does `backward` accumulate into `grad` instead of assigning? Give a concrete case from our model.
2. What goes wrong if `zero_grad()` is forgotten?
3. Why must `parameters()` dedupe tied weights? What would Adam do otherwise?
4. Why does `state_dict()` return copies?
5. How does `check_module` check parameter gradients without any change to the layer's code?
6. Derive dx, dW and db for y = xW + b in index notation. Check the shapes.
7. Why does dW sum over both B and T? Connect it to the accumulation rule.
8. Why flatten to [B·T, D] in backward but not in forward?
9. Why does training cost about 3× a forward pass?
10. What does Linear cache, and why doesn't it need to cache W?
11. Show that an embedding lookup equals onehot(ids)·E, and derive dE from Linear's dW.
12. Why is `E.grad[ids] += dout` wrong? What exactly happens with repeated ids?
13. Why does Embedding.backward return None?
14. Why validate that ids ≥ 0?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| A minimal Module/parameters/zero_grad design | Karpathy, micrograd `nn.py`: https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py |
| Linear-layer backprop with shapes | Justin Johnson, *Derivatives, Backpropagation, and Vectorization* (§4, matrix multiply): https://cs231n.stanford.edu/handouts/derivatives.pdf |
| Unbuffered scatter-add | NumPy docs, `ufunc.at`: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html |
| Manual backprop through layers | Karpathy, *Becoming a Backprop Ninja*: https://www.youtube.com/watch?v=q8SA3rM6ckI |
