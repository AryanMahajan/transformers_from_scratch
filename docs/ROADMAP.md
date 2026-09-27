# Roadmap

Every phase ends at a **checkpoint**. We move on only when that phase's
exit criteria pass and the concepts can be explained without notes.

Each component follows the same loop:

```
explain (math, shapes, params, fwd/bwd, pitfalls, test plan)
   → implement (~50–150 lines)
   → test (shapes, numerics, gradient check)
   → document in docs/NN_*.md
   → commit
```

Conventions used throughout: see [`../TECHNICAL_DOC.md`](../TECHNICAL_DOC.md).

---

## Phase 0: Setup ✅

- Repo skeleton, `.gitignore`, venv, pinned `requirements*.txt`, pytest config
- Reference papers downloaded (`scripts/download_papers.py`)

**Exit:** `pytest` runs; papers present locally.

---

## Phase 1: Numerical foundations

| Component | File | Key idea |
|-----------|------|----------|
| Softmax (stable) | `src/nn/softmax.py` | shift-invariance → subtract max |
| LogSoftmax | `src/nn/softmax.py` | log-sum-exp trick |
| Softmax backward | `src/nn/softmax.py` | Jacobian-vector product without forming the Jacobian |
| GELU (+ derivative) | `src/nn/activations.py` | tanh approximation, analytic derivative |
| Cross-entropy (fused) | `src/nn/loss.py` | `∂L/∂z = softmax(z) − onehot(y)` |
| Gradient checker | `src/utils/numerical.py` | central differences in float64 |
| Initialisation | `src/core/initialization.py` | variance preservation; seeded RNG |

**Exit:**
- softmax is correct for logits around ±1000 (no `nan`/`inf`)
- every backward matches finite differences (rel. err < 1e-6, float64)
- cross-entropy on uniform logits equals `log(V)`

**Checkpoint A:** derive `∂CE/∂z` on paper and explain why the fused version is both faster and more stable.

---

## Phase 2: Neural-network foundations

| Component | File |
|-----------|------|
| `Parameter` (data + grad) | `src/core/parameter.py` |
| `Module` (forward/backward, parameter registry, zero_grad, state_dict) | `src/core/module.py` |
| Linear | `src/nn/linear.py` |
| Embedding (scatter-add backward) | `src/nn/embedding.py` |
| LayerNorm | `src/nn/layer_norm.py` |

**Exit:** gradient checks pass for all weights and inputs; shapes hold for
`[B, T, D]` inputs.

**Checkpoint B:** derive the LayerNorm backward pass. Explain why Embedding
backward must use `np.add.at` and not fancy-index assignment.

---

## Phase 3: Tokenization

- Character-level tokenizer (encode/decode, special tokens, `<unk>`)
- Save/load vocab in `data/tokenizer/`
- Choose and document the dataset (source, licence, sizes, split)
- BPE comes later, after the model trains end to end

**Exit:** `decode(encode(s)) == s` for in-vocab text; dataset stats recorded.

---

## Phase 4: Attention

| Component | File |
|-----------|------|
| Causal mask | `src/attention/causal_mask.py` |
| Scaled dot-product attention (fwd/bwd) | `src/attention/scaled_attention.py` |
| Multi-head attention (split/merge heads, W_Q/K/V/O) | `src/attention/multi_head.py` |
| Sinusoidal positional encoding | `src/nn/embedding.py` or `src/transformer/` |

**Exit:**
- gradient check passes for attention on small tensors
- **causality test:** perturbing token `t+k` leaves outputs at positions `≤ t` unchanged
- future positions get exactly 0 attention weight

**Checkpoint C:** walk through every reshape/transpose in MHA with shapes.
Explain the `√d_k` scaling from a variance argument.

---

## Phase 5: Transformer model

- FFN (`Linear → GELU → Linear`, `d_ff = 4·d_model`)
- Pre-LN block: `x + MHA(LN(x))`, then `x + FFN(LN(x))`
- N-block stack, final LayerNorm, LM head → logits `[B, T, V]`
- Parameter counting

**Exit:** gradient check passes for a full tiny model (1–2 blocks, D=8);
the causality test passes on the full model.

---

## Phase 6: Training (midterm gate)

- Adam (then AdamW), gradient clipping
- Batching and sequence creation, training loop, logging
- **Overfit a tiny corpus:** the loss must drop close to 0

**Exit / midterm demo:** a randomly initialised model memorises a tiny
dataset and the loss curve is plotted.

---

## Phase 7: Real training

- Real corpus (e.g. TinyStories subset), train/val split
- Checkpointing (config, tokenizer, params, optimizer state, step): train → save → exit → reload → resume
- Autoregressive generation (greedy, temperature, top-k)
- Validation loss and perplexity, tokens/sec, parameter count
- BPE tokenizer

---

## Phase 8: Experiments and report (endterm)

| Exp | Variable | Values |
|-----|----------|--------|
| A | Layers | 2 / 4 / 6 |
| B | Heads | 2 / 4 / 8 |
| C | Context | 64 / 128 / 256 / 512 |
| D | Positional encoding | sinusoidal vs learned |
| E | Model size | ~1M / ~5M / ~10–30M params |

Also: attention heatmaps, error analysis, reproducible configs.
