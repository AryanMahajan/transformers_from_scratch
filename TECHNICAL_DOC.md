# Technical Documentation

This is the living reference for architecture, conventions, and design
decisions. Every section is updated as its phase lands. Derivations
live in the per-phase notes in [`docs/`](docs/).

---

## 1. Notation and shape conventions

| Symbol | Meaning |
|--------|---------|
| `B` | batch size |
| `T` | sequence length (≤ context size) |
| `D` | model width, `d_model` |
| `H` | number of attention heads |
| `D_h` | head width, `D / H` |
| `F` | FFN hidden width, `d_ff` (default `4D`) |
| `V` | vocabulary size |

- **Row-vector convention:** activations are `[..., D]` and a linear layer
  computes `y = x @ W + b` with `W ∈ R^{D_in × D_out}`. (Papers often write
  `Wx`; ours is the transpose, so `W` is stored as `[in, out]`.)
- **Tensor layout:** `[B, T, D]` everywhere. Attention internals use
  `[B, H, T, D_h]` and scores are `[B, H, T, T]`.
- **dtype:** `float32` for training and `float64` for gradient checks.

## 2. Module contract

Every layer implements the following (details land in Phase 2):

```python
out = layer.forward(x)        # caches whatever backward needs
dx  = layer.backward(dout)    # accumulates into param.grad, returns dL/dx
```

- `backward` **accumulates** (`+=`) into parameter gradients. Call
  `zero_grad()` once per step. Accumulation matters because a parameter
  used in several places (e.g. tied embeddings) gets gradients from each use.
- `backward` returns the gradient with respect to the layer's input, with
  the same shape as `x`.
- One forward must be followed by at most one backward, because the cache
  is overwritten on the next forward.

## 3. Architecture (target)

Decoder-only, **Pre-LN** (see Xiong et al. 2020 in `papers/`):

```
ids [B,T]
 → TokenEmbedding [B,T,D] + PositionalEncoding [T,D]
 → N × Block:
       x = x + MHA(LN1(x))      # causal
       x = x + FFN(LN2(x))
 → LN_final
 → LM head (Linear D→V) → logits [B,T,V]
```

## 4. Design decisions log

| # | Decision | Reason |
|---|----------|--------|
| 1 | Pre-LN rather than Post-LN | More stable gradients at init, so no LR warmup is strictly needed. This matters for a hand-written trainer. |
| 2 | GELU uses the tanh approximation | Its derivative is closed-form and cheap. This matches GPT-2. |
| 3 | Cross-entropy is fused with log-softmax | Gives `∂L/∂z = p − y`, which is stable and avoids a V×V Jacobian. |
| 4 | Gradients are checked in float64 | With float32 the finite-difference error (~1e-4) hides real bugs. |
| 5 | Papers are not committed | Redistribution rights vary, so they are fetched with a script. |

## 5. Reproducibility

- Every experiment config records the seed, data version, tokenizer config, and hyperparameters.
- RNG: use `np.random.default_rng(seed)` and pass it explicitly. There is no global `np.random.seed` state.

## 6. Environment

- Python 3.11, NumPy 1.26.4, Matplotlib 3.8.3, pytest 9.0.2 (`requirements*.txt`)
- Virtual env: `.venv/` (git-ignored)
