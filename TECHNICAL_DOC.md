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
| 6 | Gradient checks use norm-wise relative error | Element-wise ratios blow up where the true gradient is ~0. |
| 7 | GPT-2 init: N(0, 0.02²), residual projections 0.02/√(2N) | Matches Pre-LN; logits ≈ 0 at init so step-0 loss ≈ log V. |
| 8 | Embedding backward uses `np.add.at` | Buffered fancy-index `+=` drops repeated ids silently. |
| 9 | LayerNorm uses two-pass variance, ε = 1e-5, biased var | `E[x²]−μ²` cancels catastrophically in float32; matches GPT-2/PyTorch. |
| 10 | Dataset: Tiny Shakespeare, contiguous 90/10 split, vocab from train only | Public domain, standard baselines; contiguous split avoids window leakage. |
| 11 | No key bias in attention (q, v, o keep theirs) | `q·(k+b)` shifts each score row by a constant; softmax ignores it, so the gradient is exactly 0. |
| 12 | Sinusoidal positions as a fixed buffer, interleaved sin/cos, built in float64 | No parameters (identity backward); `PE[t+k]` is a rotation of `PE[t]`; a float64 master avoids f32→f64 casts carrying rounding error into grad checks. Learned positions (an `Embedding` over `arange(T)`) are the Experiment D alternative. |
| 13 | Pre-LN blocks + final LN; grad-check blocks at std 0.4 | Untouched residual stream = clean gradient highway, no warmup needed. At std 0.02 some gradients are ~1e-4 and finite-difference noise dominates the relative error. |
| 14 | Tied input/output embeddings by default (`tie_weights` flag) | Saves V·D params, standard in GPT-2; the gradient is the sum of both uses via `+=`. Implemented as a matmul with `tok_emb.E`, not a shared Parameter. |
| 15 | AdamW with decoupled decay on matrices only (ndim ≥ 2) | L2-in-Adam divides the decay by √v (measured 4.4× uneven); decaying LayerNorm γ/biases is pointless. State keyed by parameter name for exact resume. |
| 16 | Global-norm clipping (c = 1.0) + linear warmup → cosine to max_lr/10 | Clipping preserves direction and prevents the measured post-spike Adam stall (18×); warmup covers noisy early v̂ estimates. |

## 5. Reproducibility

- Every experiment config records the seed, data version, tokenizer config, and hyperparameters.
- RNG: use `np.random.default_rng(seed)` and pass it explicitly. There is no global `np.random.seed` state.

## 6. Environment

- Python 3.11, NumPy 1.26.4, Matplotlib 3.8.3, pytest 9.0.2 (`requirements*.txt`)
- Virtual env: `.venv/` (git-ignored)
