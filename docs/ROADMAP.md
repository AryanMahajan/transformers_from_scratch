# Roadmap

Every phase ends at a **checkpoint**. I move on only when that phase's
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

## Phase 1: Numerical foundations ✅

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

Study notes: [`01_numerical_foundations.md`](01_numerical_foundations.md)

---

## Phase 2: Neural-network foundations ✅

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

Study notes: [`02_nn_foundations.md`](02_nn_foundations.md)

---

## Phase 3: Tokenization ✅

- Character-level tokenizer (encode/decode, special tokens, `<unk>`)
- Save/load vocab in `data/tokenizer/`
- Choose and document the dataset (source, licence, sizes, split)
- BPE comes later, after the model trains end to end

**Exit:** `decode(encode(s)) == s` for in-vocab text; dataset stats recorded.

Study notes: [`03_tokenization.md`](03_tokenization.md)

---

## Phase 4: Attention ✅

| Component | File |
|-----------|------|
| Causal mask | `src/attention/causal_mask.py` |
| Scaled dot-product attention (fwd/bwd) | `src/attention/scaled_attention.py` |
| Multi-head attention (split/merge heads, W_Q/K/V/O) | `src/attention/multi_head.py` |
| Sinusoidal positional encoding | `src/nn/positional.py` |

**Exit:**
- gradient check passes for attention on small tensors
- **causality test:** perturbing token `t+k` leaves outputs at positions `≤ t` unchanged
- future positions get exactly 0 attention weight

**Checkpoint C:** walk through every reshape/transpose in MHA with shapes.
Explain the `√d_k` scaling from a variance argument.

Study notes: [`04_attention.md`](04_attention.md)

---

## Phase 5: Transformer model ✅

- FFN (`Linear → GELU → Linear`, `d_ff = 4·d_model`)
- Pre-LN block: `x + MHA(LN(x))`, then `x + FFN(LN(x))`
- N-block stack, final LayerNorm, LM head → logits `[B, T, V]`
- Parameter counting

**Exit:** gradient check passes for a full tiny model (1–2 blocks, D=8);
the causality test passes on the full model.

Code: `src/transformer/{feed_forward,block,model}.py`.
Study notes: [`05_transformer.md`](05_transformer.md)

---

## Phase 6: Training (midterm gate) ✅

- Adam (then AdamW), gradient clipping
- Batching and sequence creation, training loop, logging
- **Overfit a tiny corpus:** the loss must drop close to 0

**Exit / midterm demo:** a randomly initialised model memorises a tiny
dataset and the loss curve is plotted.

**Result:** 257 chars, loss 4.19 → 0.0019, greedy reproduces 257/257 chars
(`python scripts/overfit_tiny.py`, `experiments/overfit_tiny/`).
Study notes: [`06_training.md`](06_training.md)

---

## Phase 7: Real training ✅

- Real corpus (Tiny Shakespeare, contiguous 90/10 split) ✅
- Checkpointing (config, tokenizer, params, optimizer state, step, RNG): train → save → exit → reload → resume, bit-identical ✅
- Autoregressive generation (greedy, temperature, top-k, top-p) ✅
- Validation loss, bits/char, perplexity, tokens/sec, parameter count, n-gram baselines ✅
- BPE tokenizer + char-vs-BPE comparison in bits/char ✅
- Extra: optional CuPy GPU backend (`--device gpu`)

**Results:** char model 2.81 bits/char, BPE-512 model 2.53 bits/char (bigram baselines 3.58 / 2.67).

Study notes: [`07_real_training.md`](07_real_training.md)

---

## Phase 8: Experiments and report (endterm) ✅

| Exp | Variable | Values | 3-seed result (val bits/char; centre 2.225 ± 0.018) |
|-----|----------|--------|--------|
| A | Layers | 2 / **4** / 6 | 2.273 / 2.225 / 2.210: 2 layers worse, 6 no clear gain |
| B | Heads | 2 / **4** / 8 | 2.208 / 2.225 / 2.225: no effect |
| C | Context | 64 / **128** / 256 / 512 | 2.219 / 2.225 / 2.252 / 2.312: 512 clearly worse |
| D | Positional encoding | **sinusoidal** vs learned | 2.225 / 2.216: tie (with √D) |
| E | Model size | **0.23M** / 0.86M / 3.3M / 10.8M params | finals 2.225 / 2.299 / 2.599 / 3.075: overfitting; best periodic within 0.04 |

- Seed-noise study first (3 seeds per cheap config) ✅
- Design fix found by it: token embeddings × √D with sinusoidal positions (2.583 → 2.225 bits/char, seed std 4.5× lower) ✅
- Attention analysis: heatmaps, per-head distance / prev-token / sink / entropy, loss vs context ✅
- Error analysis: loss by token kind / frequency / speaker names, calibration ✅
- Reproducible configs: restartable grid runner, `docs/REPRODUCE.md`, environment in every summary ✅
- Report: [`reports/REPORT.md`](../reports/REPORT.md) ✅

Study notes: [`08_experiments.md`](08_experiments.md)
