# Transformers From Scratch

A decoder-only Transformer language model built **from fundamental numerical
operations**. NumPy is the only numerical backend. It does not use autograd,
deep-learning frameworks, or pretrained weights. Every forward pass, backward
pass, and optimizer step is derived and written by hand.

> **Research question.** Can a decoder-only Transformer language model be
> implemented and trained from fundamental numerical operations, and how do
> architectural choices (depth, heads, context length, positional encoding,
> model size) affect its language-modeling behaviour?

This is an educational implementation and experimental study. It is not an
attempt at a state-of-the-art LLM.

---

## Status

| Phase | Topic | Status |
|------:|-------|--------|
| 0 | Project setup, environment, papers | ✅ done |
| 1 | Numerical foundations (Softmax, LogSoftmax, GELU, cross-entropy) | ✅ done |
| 2 | NN foundations (Parameter, Module, Linear, Embedding, LayerNorm) | ✅ done |
| 3 | Tokenization (char-level → BPE) | ✅ done |
| 4 | Attention (causal mask, scaled dot-product, multi-head, sinusoidal positions) | ✅ done |
| 5 | Transformer (FFN, block, stack, LM head) + gradient checks | ✅ done |
| 6 | Optimizer + training loop + **tiny-dataset overfit** (midterm gate) | ✅ done |
| 7 | Real corpus, checkpointing, generation (temperature/top-k/top-p), BPE, evaluation | ✅ done |
| 8 | Experiments A–E, attention analysis, error analysis, report | ✅ done |

[`docs/ROADMAP.md`](docs/ROADMAP.md) has the full plan and the exit
criteria for each phase.

---

## Quick start

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python scripts/download_papers.py   # fetch reference PDFs into papers/
python scripts/prepare_data.py      # download + split + tokenize Tiny Shakespeare
python scripts/overfit_tiny.py      # midterm gate: memorise 257 chars, plot loss curve
python scripts/train.py --run baseline [--device gpu] [--resume]   # real run (GPU: pip install -r requirements-gpu.txt)
python scripts/train.py --run bpe512 --tokenizer bpe --vocab-size 512 [--device gpu]
python scripts/sample.py --run baseline --temperature 0.8 --top-k 10   # generate text (no flags: compare settings)
pytest                              # run the test suite
```

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Tested with Python 3.11, NumPy 1.26.4.

---

## Repository layout

```
src/
  core/         Parameter, Module base class, weight initialisation
  nn/           Linear, Embedding, LayerNorm, activations, softmax, loss
  attention/    causal mask, scaled dot-product attention, multi-head attention
  transformer/  feed-forward, Transformer block, full model
  tokenizer/    character-level and BPE tokenizers
  optim/        Adam / AdamW
  training/     dataset batching, trainer, checkpointing
  generation/   autoregressive sampling
  analysis/     attention statistics, error analysis
  utils/        numerical helpers (gradient checking), logging
tests/          pytest suite (shapes, numerics, gradients, causality, serialization)
docs/           per-phase learning notes and derivations
papers/         reference papers (downloaded, not committed)
experiments/    configs + results for experiments A–E
reports/        figures and write-ups
```

## Documentation

- [`docs/ROADMAP.md`](docs/ROADMAP.md): phases, checkpoints, exit criteria
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): the exact command behind every reported number and figure
- [`reports/REPORT.md`](reports/REPORT.md): the endterm report (implementation, experiments, analysis)
- [`TECHNICAL_DOC.md`](TECHNICAL_DOC.md): architecture, conventions, shapes, design decisions
- [`docs/`](docs/): one learning note per phase covering the maths, derivations, shapes, pitfalls, and tests
- [`papers/README.md`](papers/README.md): reading list and reading order

## Constraints

Allowed: Python standard library, NumPy, Matplotlib (visualisation only), pytest (testing only).
Not allowed: PyTorch, TensorFlow, JAX, Keras, HuggingFace, any autodiff library, pretrained models.
