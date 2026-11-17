# Phase 7: Real training (study notes)

Revision notes for Phase 7. This phase moves from the memorisation demo to
training on the full corpus: checkpointing, a real run, generation and
evaluation.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

---

## 1. Checkpointing and exact resume: `src/training/checkpoint.py`

### Goal
**train → save → exit → reload → resume**, and the resumed run must be
**bit-identical** to one that never stopped. "Approximately the same" hides
bugs. Bit-identical is testable with `==`.

### What the future of a run depends on
| State | Why it's needed | If you forget it |
|---|---|---|
| model config + weights | rebuild the network | — |
| AdamW m, v | per-parameter step sizes | first steps after resume use v = 0, so every step ≈ lr·sign(g): a jolt |
| AdamW t | bias correction $1-\beta^t$ | correction restarts as if t = 1 |
| step | LR schedule position | warmup runs again |
| **batch RNG state** | which windows come next | **silent drift**: different batches (tested) |
| eval RNG state | which eval batches | eval numbers not comparable |
| train config | loop settings | — |
| tokenizer | decode generated text | ids are meaningless without it |
| history | continuous plots | curves start over |

The RNG is the one people forget. Re-seeding with the original seed replays
the batches from step 0, not from step s. The test
`test_forgetting_the_rng_state_breaks_resume` shows the losses diverge.

### File format: one `.npz`, no pickle
```
param/<name>    adam_m/<name>    adam_v/<name>     arrays (float32)
meta            JSON string: configs, tokenizer, step, adam_t, RNG states, history
```
- **No pickle** (`allow_pickle=False`): loading a pickle can execute arbitrary
  code. Arrays + JSON can't.
- NumPy's `bit_generator.state` is a dict of Python ints, so it's JSON-safe.
- `format_version` lets old checkpoints be detected if the format changes.
- **Size ≈ 3× the weights** (weights + m + v). Measured: 801k params →
  9.67 MB file vs 3.21 MB of weights. For inference only the weights are
  needed (`load_model`).

### Atomic save
Write `latest.npz.tmp`, then `os.replace(tmp, path)`. A crash mid-save leaves
the previous checkpoint intact; the rename is atomic on one filesystem.
(Writing through a file handle stops `np.savez` from appending `.npz` to the
tmp name.)

### The Trainer refactor
`train()` kept its state in local variables, so nothing could be saved.
`Trainer` now owns **model, opt, rng, eval_rng, step, history**, which is
exactly what's checkpointed. `train()` is a thin wrapper, so old code and
tests are unchanged.

**Loop order** (what makes resume exact):
```
[eval if step % eval_interval == 0] → [stop?] → train step → step += 1 → [save if step % ckpt_interval == 0]
```
A checkpoint saved after step s contains evals < s. The resumed run evaluates
at s with the restored eval RNG, exactly what the uninterrupted run did next.
Saving *before* the eval would double-record it; saving after the eval but
restoring at s would skip it.

`run(stop_at=k)` simulates being killed after k steps (used by the tests).

### Also added
- `tokens_per_sec` logged every `log_interval` steps (Phase 7 metric).
- `CharTokenizer.to_dict()/from_dict()` so the tokenizer embeds in the
  checkpoint (`save`/`load` reuse them).

### Tests (`tests/test_checkpoint.py`)
- **40 straight == 20 → save → rebuild from file → 20 more:** weights, m,
  v, and every history entry `==`, for tied + sinusoidal and untied + learned.
- Forgetting the RNG state → losses diverge.
- Round-trip contents: configs, step, t, RNG state, 3 arrays per parameter,
  tokenizer, same outputs, float32 preserved.
- Atomic: no `.tmp` left, overwrites `latest.npz`.
- Final checkpoint at the end; unknown format rejected.

### Housekeeping: line endings
Python's `write_text` on Windows writes `\r\n`. Earlier scripted edits had
committed 10 files with CRLF, so their diffs showed every line changed.
Fix: `.gitattributes` with `* text=auto eol=lf` + `git add --renormalize .`
(verified: no content changes, `git diff --ignore-cr-at-eol` empty).

---

## 2. First real run on Tiny Shakespeare: `scripts/train.py`, `src/training/baselines.py`, `src/core/backend.py`

### Baselines first (fit on train, scored on val)
| | nats | bits/char | perplexity |
|---|---|---|---|
| uniform (ln 66) | 4.190 | 6.04 | 66 |
| unigram | 3.347 | 4.83 | 28.4 |
| **bigram** (1 char of context, add-α smoothing) | **2.482** | **3.58** | **12.0** |

- **bits per char = nats / ln 2**; **perplexity = e^nats** = the effective
  number of equally likely next characters.
- The bigram is the "no attention needed" level. A Transformer that doesn't
  clearly beat it isn't using context.
- `np.add.at` again for the bigram counts (repeated pairs must add).

### Performance detour 1: a 10× CPU slowdown (`src/nn/linear.py`)
The first benchmark predicted ~3 hours for an 801k model. Profiling showed
every op slow, but a micro-benchmark gave 98 GFLOP/s for a plain matmul, so the
problem was *how* NumPy was called:
- `x @ W` with x `[16,128,128]` → NumPy **loops over the batch**: 61 ms.
  `x.reshape(-1,128) @ W` → one BLAS GEMM: **4.7 ms (13×)**.
- Fix: Linear flattens leading dims, does one 2-D GEMM, reshapes back. Same
  maths; all gradient checks still pass.
- `x**3` → `x*x*x` in GELU (general power is ~6× slower).
- **Step 6.7 s → 0.7 s.** Lesson: profile before shrinking the model.

### Performance detour 2: optional GPU backend (CuPy)
NumPy is CPU-only. The laptop has an RTX 3050 (6 GB). **CuPy** mirrors the
NumPy API on NVIDIA GPUs.
- **Default stays NumPy.** Tests, gradient checks and demos are pure NumPy.
  `--device gpu` is opt-in (`pip install -r requirements-gpu.txt`).
- NumPy functions on CuPy arrays (`np.exp`, `np.sum`, `np.where`, …)
  **dispatch automatically** (NEP 18, `__array_function__`). Only array
  *creation* (`np.arange`, `np.tril(np.ones())`) and `np.add.at` must pick the
  library: `xp_of(x)` returns numpy or cupy.
- NumPy + CuPy arrays never mix silently (CuPy raises), so every transfer is
  explicit: `to_device`, `to_numpy`, `like(a, ref)`.
- `Module.to(device)` moves parameters and buffers (`_move_buffers` hook: the
  sinusoidal table). **Model first, then optimizer**, so AdamW's m, v are
  created on the same device.
- Batches are still sampled on the CPU with the NumPy RNG (exact resume), and
  ids are copied to the device inside Embedding / cross_entropy.
- Checkpoints always hold NumPy arrays: train on GPU, load on CPU (tested).
- **Verification:** in float64, GPU gradients match CPU gradients to 1e-8 for
  every parameter. The CPU ones are finite-difference checked, so the GPU
  backward is correct by transitivity.
- Gotchas met:
  - Installing CuPy 14 silently **upgraded NumPy to 2.x**, whose NEP 50
    promotion rules change float32 × float64-scalar results. Pinned
    `numpy==1.26.4` + `cupy-cuda12x<14`.
  - Benchmarks with a short warm-up measured CuPy **compiling kernels** on
    first use. Warm up before timing.
  - Laptop timings vary a lot with background load (Chrome/Docker). Trust the
    run's own tokens/sec log over short benchmarks.

### The run (`experiments/baseline/`)
204,032 params (D = 64, H = 4, N = 4, T = 128, tied, sinusoidal); batch 32,
3000 steps = **12.3M tokens ≈ 11 passes** over train; AdamW 3e-3 → 3e-4
cosine, 100 warmup steps, weight decay 0.1, clip 1.0; GPU, median
**7,800 tok/s**, 28.6 min.

| | val nats | bits/char | perplexity |
|---|---|---|---|
| bigram baseline | 2.482 | 3.58 | 12.0 |
| **our model (final, 100-batch eval)** | **1.948** | **2.81** | **7.02** |

Train 1.843 → generalisation gap **0.106**.

### Reading the curve
1. **0 → ~50:** ln V → unigram (3.35). Learns character frequencies.
2. **~50 → 250: plateau at the unigram level**, the same tied-weights plateau
   as the toy corpus (Phase 6 §3), much longer here. The gradient norm jumped
   0.13 → 1.06 right before the breakthrough.
3. **250 → 1000:** fast drop. Crosses the bigram line at ~step 950: from here
   it uses more than one character of context.
4. **1000 → 3000:** steady improvement as the LR decays, with clipping
   occasionally active (norm spikes up to 4.6 absorbed).
5. **Gap opens after ~1750** (0.01 → 0.10). Partly memorising training text
   (~11 passes), partly **distribution shift** from the contiguous split (val
   = the last plays; even the unigram baseline is worse on val).

### Greedy sample: degenerate repetition
```
ROMEO:
I will the so the so the sould the her the sould
The shall the the some the the some the his the songe.
```
Right *form* (speaker names in capitals, colon, line breaks, word-length
chunks, common words), but greedy decoding falls into **loops**. It always
picks the single most likely character, and "the" is the most likely
continuation in many contexts. This is a known failure of greedy decoding, not
(only) of the model, and the motivation for sampling with temperature / top-k
(§3).

### Generation speed vs a 120B model
Generation is sequential (1 token per forward pass, no KV cache):
**GPU 127 tok/s, CPU 12–40 tok/s**. That's similar to the 40–50 tok/s a 120B
model decodes on an L40 cluster, but for a model 600,000× smaller. Big models
are limited by memory bandwidth (reading ~240 GB of weights per token); ours by
kernel-launch and Python overhead.

Training throughput (7,800 tok/s) is a different quantity: 4,096 tokens are
processed **in parallel** per step (teacher forcing), and each costs only
6N ≈ 1.2 MFLOP.

---

## Self-check questions
1. List everything a checkpoint must contain for an exact resume, and what goes wrong if each is missing.
2. Why does forgetting the RNG state cause drift, even with the same seed?
3. What happens to the first steps after resume if Adam's m and v are not restored? If t is not restored?
4. Why avoid pickle for checkpoints?
5. Why write to a temp file and rename?
6. Why is a training checkpoint ~3× the size of the weights? What does inference need?
7. In the loop order eval → stop → step → save, why does resume reproduce the eval at step s exactly once?
8. Convert 1.948 nats to bits per char and perplexity. What does perplexity 7 mean intuitively?
9. Why is the bigram the key baseline for a Transformer? What does crossing it show?
10. Why was a 3-D `x @ W` 13× slower than the 2-D version? How was it fixed without changing the maths?
11. How does the code run on GPU without rewriting every `np.` call? Which calls *did* need changes, and why?
12. How do we know the GPU backward pass is correct without a GPU gradient check?
13. Why did the train/val gap open late in training? Name two causes.
14. Why does greedy decoding produce "the the the..." loops?
15. Training throughput is 7,800 tok/s but generation is 127 tok/s. Why so different?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| CuPy: NumPy-compatible arrays on the GPU | CuPy docs, *Basics of CuPy*: https://docs.cupy.dev/en/stable/user_guide/basic.html |
| NumPy array-function dispatch (NEP 18) | https://numpy.org/neps/nep-0018-array-function-protocol.html |
| NumPy `.npz` format and `allow_pickle` | NumPy docs, `numpy.load`: https://numpy.org/doc/stable/reference/generated/numpy.load.html |
| Saving/restoring RNG state | NumPy docs, `BitGenerator.state`: https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.state.html |
