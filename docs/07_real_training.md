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

## 3. Sampling: temperature, top-k, top-p: `src/generation/sampling.py`, `scripts/sample.py`

### Where it sits
Same model, same loop as greedy (crop to `max_len`, use the last position's
logits). Only the **picker** changes: instead of `argmax`, draw from a
reshaped distribution.

### The maths
For last-position logits $z \in \mathbb{R}^V$:

**Temperature:** $p_i = \dfrac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$. Two tokens with
gap $\Delta = z_a - z_b$ get probability ratio $e^{\Delta/\tau}$.
- τ → 0: ratios → ∞, all mass on the argmax = **greedy** (implemented exactly as `temperature=0`).
- τ = 1: the model's own distribution (what cross-entropy trained).
- τ → ∞: ratios → 1, **uniform**.
- The **ranking never changes**; only the sharpness does (entropy rises with τ).

**Top-k:** keep the k largest logits, set the rest to −∞; after softmax the
kept tokens are renormalised ($p_i/\sum_{kept} p_j$) and the rest are exactly
0. Why: 60 tail characters at 0.3% each = 18% chance of junk *every step*.

**Top-p (nucleus):** sort by probability, keep the **smallest** set with
mass ≥ p. Rule used: drop token i if the mass strictly above it is already
≥ p (so the top token always survives). Unlike top-k the set **adapts**: one
token after "Q", dozens after a space.

**Drawing (inverse CDF):** $u \sim U[0, 1)$, pick the first i with
$\text{cdf}_i > u$. Token i owns an interval of width $p_i$, so it's picked with
probability exactly $p_i$; a masked token owns width 0 and is never picked.

Order: temperature → top-k → top-p → softmax → draw.

### Code notes
- Logits may be on the GPU; sampling one `[V]` vector is done on the CPU in float64 (`to_numpy`).
- `np.partition(z, V-k)[V-k]` finds the k-th largest in O(V), no full sort.
- `mass_before = cumsum(p_sorted) - p_sorted` gives the top-p mask in one line.
- `u = rng.random() * cdf[-1]` guards against `cdf[-1] = 0.9999999999999998`.
- The `rng` is passed in explicitly, so the same seed gives the same text.

### Tests (`tests/test_sampling.py`, 15)
- Temperature divides logits; low τ means lower entropy, high τ higher, same argmax.
- Top-k keeps exactly the k largest and renormalises; k ≥ V is a no-op.
- Top-p keeps the *minimal* set reaching p (checked for several p), on unsorted input too; the top token always kept.
- **40,000 draws match softmax(z/τ)** within 0.01 (τ = 1, 2); masked tokens never drawn.
- `temperature=0` and `top_k=1` both reproduce `greedy_generate` exactly.
- Same seed means the same output; invalid arguments raise.

### Demo on the baseline checkpoint (`experiments/baseline/samples.md`)
300 chars after "ROMEO:", mean ± std over 5 seeds.
**Diversity** = distinct words / words (low = loops); **real-word** = share
of generated words that occur in the training text (spelling).

| setting | diversity | real-word |
|---|---|---|
| greedy | 0.20 ± 0.00 | **0.88** |
| τ = 0.5 | 0.72 ± 0.04 | 0.79 ± 0.05 |
| τ = 0.8 | 0.87 ± 0.03 | 0.63 ± 0.03 |
| τ = 1.0 | 0.92 ± 0.03 | 0.44 ± 0.04 |
| τ = 1.5 | **0.97** ± 0.03 | 0.28 ± 0.07 |
| τ = 1.0, k = 5 | 0.76 ± 0.05 | 0.69 ± 0.05 |
| τ = 0.8, k = 10 | 0.77 ± 0.03 | 0.74 ± 0.03 |
| τ = 1.0, p = 0.9 | 0.86 ± 0.02 | 0.59 ± 0.04 |

Reading it:
- **Temperature is a single dial between two failure modes**: greedy loops
  ("the the shall the the some") vs τ = 1.5 gibberish ("tuuYy; ifst livarm",
  and it even sampled the `<unk>` token, because a flat distribution gives the
  tail real mass).
- **Truncation cuts the junk tail.** At τ = 1, top-k = 5 raises real-word
  0.44 → 0.69; at τ = 0.8, k = 10 raises it 0.63 → 0.74 (diversity
  0.87 → 0.77). Honest reading: on this small model the truncated settings
  land near the *same* diversity/real-word frontier as lowering τ
  (τ = 0.5: 0.72 / 0.79 vs τ = 0.8, k = 10: 0.77 / 0.74). The benefit is that
  truncation guarantees rare junk tokens (`<unk>`, stray capitals) get
  probability 0, while lower τ only shrinks them. Samples show proper speaker turns:
  ```
  COMIUCE:
  No, would his my lord or sene the him hear:
  ```
- τ = 1 text is "as the model believes", and the model (2.81 bpc) still puts
  a lot of mass on wrong characters. Better models need less truncation.
- Caveats: ~55 words per sample, crude metrics (real words ≠ grammar), one
  prompt. Good enough to show the trade-off, not to rank close settings.

### Why greedy is not "the best text"
Greedy maximises each step's probability, not the sequence's, and real
text is not made of the most probable tokens. Holtzman et al. show human
text has much lower per-token probability than beam/greedy output, which
falls into repetition loops. Sampling with a truncated tail matches human
text statistics better.

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
16. Show that τ → 0 gives greedy and τ → ∞ gives uniform, using the ratio $e^{\Delta/\tau}$. Can temperature change which token is most likely?
17. Why can sampling at τ = 1 produce junk even though τ = 1 is exactly the distribution the model was trained to output?
18. Top-k vs top-p: when does a fixed k keep too many tokens, and when too few? How does top-p adapt?
19. Prove that inverse-CDF sampling picks token i with probability $p_i$. Why is a −∞ logit never drawn?
20. Why must the RNG be passed in explicitly? Which test relies on it?
21. In the demo table, which setting would you pick for a demo, and why? What are the limits of the two metrics?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| CuPy: NumPy-compatible arrays on the GPU | CuPy docs, *Basics of CuPy*: https://docs.cupy.dev/en/stable/user_guide/basic.html |
| NumPy array-function dispatch (NEP 18) | https://numpy.org/neps/nep-0018-array-function-protocol.html |
| NumPy `.npz` format and `allow_pickle` | NumPy docs, `numpy.load`: https://numpy.org/doc/stable/reference/generated/numpy.load.html |
| Nucleus (top-p) sampling, why greedy/beam degenerate | Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020): https://arxiv.org/abs/1904.09751 |
| Saving/restoring RNG state | NumPy docs, `BitGenerator.state`: https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.state.html |
