# Phase 8: Experiments (study notes)

Revision notes for Phase 8: how architecture choices affect a small
Transformer language model, measured carefully.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

---

## 1. Method: one variable at a time: `scripts/run_experiments.py`, `scripts/analyze_experiments.py`

### Centre configuration
Every experiment changes **one** thing relative to the Phase 7 BPE-512 run:

| | value |
|---|---|
| tokenizer | BPE, V = 512 (2.07 chars/token on val) |
| model | D = 64, H = 4, N = 4, sinusoidal positions **+ tokens × √D** (§2), tied embeddings, pre-LN |
| context | T = 128 tokens |
| training | 3000 steps × **4096 tokens/step**, AdamW 3e-3 → 3e-4 cosine, warmup 100, wd 0.1, clip 1.0 |
| params | 232,576 |

### The grid
| Exp | Variable | Values (centre in bold) | Note |
|---|---|---|---|
| noise | seed | **0**, 1, 2 | measures run-to-run variation |
| A | layers N | 2 / **4** / 6 | |
| B | heads H | 2 / **4** / 8 | D fixed, so head width D/H = 32 / 16 / 8 |
| C | context T | 64 / **128** / 256 / 512 | batch = 4096 / T: same tokens per step |
| D | positions | **sinusoidal** / learned | |
| E | width D (N) | **64** / 128 / 256 / 384 (N = 6) | 0.23M / 0.86M / 3.3M / 10.8M params; lr 3e-3 / 3e-3 / 1e-3 / 6e-4 |

### Design rules
- **Seeds first.** Without knowing the run-to-run spread, a 0.01 bpc difference
  could be noise. Differences below ~2 seed-std are reported as "no effect".
- **What the seeds showed:** the first (unscaled) centre gave 2.527 / 2.671 /
  2.551 bpc (std 0.077), larger than several effects we want to measure.
  Investigating that, plus a suspicious Experiment D result, led to the
  finding in §2. With the fix, seed std is **0.018**. Every cheap config (A–D,
  and D = 128 in E) still runs with **seeds 0, 1, 2** and is reported as
  mean ± std (a 3-seed mean has std ≈ 0.010). The two expensive models (3.3M,
  10.8M) stay single-seed; treat their differences below ~0.04 as inconclusive.
- **Same tokens per step** in C: a longer context with the same batch size
  would also see more data per step, mixing two effects.
- **Same number of steps and tokens** for every run (12.3M tokens).
- **Bits per character** everywhere (Phase 7 §4), so numbers compare to the
  char model and baselines.
- **Best periodic val** is recorded alongside the final val. Larger models may
  overfit (only ~485k training tokens, ~25 passes, no dropout); their final
  number alone would understate what they reached.
- **Known confounds** are stated, not hidden: E lowers the LR for the two
  biggest models (3e-3 risks divergence at D ≥ 256), so E measures "size with
  a sensible LR", not size alone.

### Tooling
- `run_experiments.py` is **restartable**: a run with a `summary.json` is
  skipped and a run with a checkpoint resumes, so a crash or reboot costs at
  most 250 steps.
- `train.py` summaries now include every periodic eval and the best one.
- `analyze_experiments.py` builds per-experiment tables
  (`experiments/<exp>/results.md`), validation curves
  (`reports/figures/<exp>_curves.png`), a summary figure with the seed-noise
  band (`reports/figures/summary.png`) and `reports/results.md`.
- Figures use a colour-blind-checked palette: an ordinal one-hue ramp for
  ordered sweeps (light = small), distinct hues for unordered ones, plus direct
  labels, so colour never carries identity alone.

---

## 2. A finding before the experiments: positions were drowning the tokens

### Symptom
In the first grid, learned positions scored **2.207** bpc vs **2.527** for
sinusoidal: 0.32 bpc, twice the seed-noise band. Far too big for a choice of
positional encoding, so it was investigated before being reported.

### Diagnosis: measure the two signals that get added
`h = TokenEmb(ids) + PE[:T]`, so compare their row norms:

| | token row norm | position row norm | position / token |
|---|---|---|---|
| at init | 0.16 | 5.66 | **35×** |
| trained, sinusoidal | 1.00 | 5.66 (fixed) | **5.7×** |
| trained, learned | 1.09 | 0.47 | 0.43× |

- Every sinusoidal row has entries $\sin, \cos \in [-1, 1]$, so its norm is
  $\sqrt{D/2}$ (5.66 at D = 64).
- Token rows use GPT-2's init, $\mathcal N(0, 0.02^2)$: norm ≈ $0.02\sqrt{D}$ = 0.16.
  Weight decay and the tied output layer keep them small.
- The first LayerNorm normalises the sum, so the **token identity is a small
  component** of what every block sees; the model must dig it out from under
  a signal 35× larger. Learned positions start at 0.02 too, so they stay balanced.

### Fix: Vaswani et al. §3.4
"In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$."
`GPTConfig(scale_emb=True)`:

$$h = \sqrt{D}\cdot \text{TokenEmb}(ids) + PE[:T], \qquad
\frac{\partial L}{\partial e} = \sqrt{D}\cdot\frac{\partial L}{\partial h}$$

The tied output layer still uses the unscaled $E$ (as in the paper). Default
off, so old checkpoints load unchanged.

Tests: full-model gradient check with the scale (tied/untied × both position
types), with the embedding gradient at ~1e-10. Removing the $\sqrt D$ from
backward on purpose gives a 0.17–0.48 error, so the test does catch that bug.
Also: the forward scales only the token part, and position/token ratio ≈ 35×
without the scale vs 2–8× with it.

### Result: 2×2 ablation, 3 seeds each (val bits/char)
| | no scale | tokens × √D |
|---|---|---|
| sinusoidal | 2.583 ± 0.077 | **2.225 ± 0.018** |
| learned | 2.215 ± 0.010 | 2.216 ± 0.008 |

- The scale **recovers 0.36 bpc** for sinusoidal positions and does nothing
  for learned ones: exactly what the explanation predicts, because learned
  positions can set their own scale.
- **Seed noise fell 4.5×** (0.077 → 0.018). How well a run dug the token
  signal out from under the positions depended on the seed; remove the
  handicap and runs agree. A 3-seed mean now has std ≈ 0.010.
- With the fix, sinusoidal and learned positions **tie** (0.009 apart, within noise).

### Consequence for the experiments
Every run in the first grid used unscaled sinusoidal positions, so all of them
carried the handicap. The grid was **rerun around the fixed centre**
(sinusoidal + √D); the first-grid runs are kept in
`experiments/ablation_unscaled/` as the "without the scale" ablation.

### Lessons
- A result that's "too good" for the change that produced it is a reason to
  investigate, not to report.
- Measure the quantities being added. A one-line norm comparison found what
  a gradient check never could: the code was *correct*, the *design* was not.
- Seed variance can be a symptom. Large run-to-run spread pointed at an
  optimisation handicap, not just at the need for more seeds.
- Implementation details in papers (a √D factor in one sentence) matter.
- The better model also **memorises more**: the train/val gap doubled
  (0.15 → 0.32 bpc), relevant for the large models in Experiment E.

---

## Self-check questions
1. Why change one variable at a time? What would a run that changed depth and width together tell you?
2. Why run extra seeds before the experiments? How do you decide whether a difference is real?
3. In Experiment C, why scale the batch size as 4096 / T?
4. Why record the best periodic validation loss as well as the final one?
5. What confound does Experiment E carry, and why was it accepted?
6. Compute the row norm of a sinusoidal encoding and of a token embedding at init (D = 64, std 0.02). Why does the ratio matter after LayerNorm?
7. Why does the √D scale help sinusoidal positions but not learned ones?
8. Derive the backward pass of $h = \sqrt D\, e + p$ with respect to $e$. What would a missing factor do to training?
9. Why did fixing the scale also shrink the seed-to-seed variance?
10. The gradient checks all passed with the handicap in place. Why couldn't they find it?
