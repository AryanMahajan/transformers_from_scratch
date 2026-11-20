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

## 3. Inside the model: what the attention heads do: `src/analysis/attention_stats.py`, `scripts/attention_viz.py`

### Purpose
Loss numbers say *how well* a model predicts, not *how*. Every
`MultiHeadAttention` keeps its last attention map `attn_weights` `[B, H, T, T]`
(row $i$ = how query $i$ spreads its weight over keys $j \le i$). Reading these
maps answers two questions: what does each head look at, and why did a
longer context (Exp C) make the model *worse*?

### Per-head statistics
All are averaged over the batch and over queries $i \ge 1$ (query 0 can only
attend to itself, so it carries no information):

| statistic | formula | reads as |
|---|---|---|
| mean distance | $\sum_j A_{ij}(i-j)$ | how far back the head looks; uniform attention gives $i/2$ (≈ $T/4$ averaged) |
| previous-token weight | $A_{i,i-1}$ | ≈ 1 for a "look at the last token" head |
| first-token weight | $A_{i,0}$ | an *attention sink*: parking unused weight on token 0 |
| normalised entropy | $H_i / \ln(i+1)$, $H_i = -\sum_j A_{ij}\ln A_{ij}$ | 0 = one key, 1 = uniform over all allowed keys |
| mass beyond $d$ | $\sum_{j \le i-d} A_{ij}$ over $i \ge d$ | share of weight on keys ≥ d tokens back |
| reach CDF | $\sum_{j \ge i-d} A_{ij}$ over late queries $i \ge T/2$ | share within distance d; late queries all have ≥ T/2 history, so windows of different T compare fairly |

Dividing the entropy by $\ln(i+1)$ matters: a uniform row over $i+1$ keys has
entropy $\ln(i+1)$, which grows with $i$, so raw entropy would mostly measure
the position, not the head.

Tests (`tests/test_attention_stats.py`) use hand-built heads with known
answers: a previous-token head (distance 1, entropy 0), a sink head, and the
uniform causal head (entropy exactly 1, distance $i/2$), plus a real model's
maps (rows sum to 1, zero above the diagonal, ≈ uniform at init).

### The centre model (T = 128), 421 validation windows
Figures: `reports/figures/attention_pe_scale_sin_scaled_s0_heatmaps.png`
(every layer × head on "GREMIO:↵Good morrow, neighbour Baptista") and
`..._stats.png`.

| layer | mean distance (h0–h3) | prev-token | norm. entropy |
|---|---|---|---|
| 0 | 2.0 / 6.7 / 21.8 / 2.1 | 0.41 / 0.11 / 0.05 / 0.33 | 0.32 / 0.63 / 0.82 / 0.31 |
| 1 | 7.2 / 14.8 / 10.1 / 12.1 | 0.15 / 0.08 / 0.12 / 0.09 | 0.58 / 0.60 / 0.49 / 0.53 |
| 2 | 9.3 / 16.2 / 11.0 / 14.7 | 0.13 / 0.05 / 0.12 / 0.06 | 0.54 / 0.59 / 0.63 / 0.61 |
| 3 | 13.8 / 11.3 / 15.3 / 17.2 | 0.09 / 0.09 / 0.11 / 0.07 | 0.67 / 0.51 / 0.61 / 0.72 |

(Uniform attention would give mean distance 32 and entropy 1.)

- **Layer 0 specialises sharply.** h0 and h3 are local (distance ≈ 2, entropy
  ≈ 0.3). In the heatmap, **h3 attends to the first piece of the current
  word**: on every piece of `·B|a|pt|is|t|a` it looks at `·B`, on `or|row`
  at `·m`. BPE cut the word into pieces, and this head puts it back together.
  h0 is a previous-token / local head; h2 is broad (entropy 0.82), a cheap
  "summary of the recent past".
- **Higher layers look further** (mean distance 7 → 17) but stay well below
  uniform: they are local-to-mid-range. Several of them lock onto the
  speaker line (`IO`, `:`, `↵`): a structural anchor for *who is speaking*.
- **No attention sinks** at this scale: first-token weight is 0.01–0.02
  everywhere. Sinks are reported in large models; this one doesn't need a
  parking spot.
- Only 4% of attention reaches ≥ 64 tokens back.

### Why longer context hurt (Exp C)
Seed 0: T = 64 / 128 / 256 / 512 gave 2.226 / 2.225 / 2.249 / 2.279 bits/char.
Two measurements (figures `attention_distance.png`, `attention_position_loss.png`):

**1. Longer windows look much further back.** Upper layers (1–3):

| T | mean distance | uniform would be | mass ≥ 64 back | norm. entropy |
|---|---|---|---|---|
| 64 | 2.7–8.1 | 16 | n/a | 0.44–0.61 |
| 128 | 7–17 | 32 | 0.03–0.05 | 0.49–0.72 |
| 256 | 22–59 | 64 | 0.31–0.48 | 0.30–0.81 |
| 512 | 99–127 | 128 | 0.64–0.68 | 0.50–0.85 |

Mean distance grows roughly in proportion to T, reaching the uniform value at
T = 512, even though several of those heads are fairly sharp (entropy ≈ 0.5).
So they aren't spreading weight evenly; they pick specific tokens *far away*.
That fits heads that are mostly **content-addressed** ("find tokens like
this one") with only a weak recency bias: the longer the window, the more of
the matches lie far back. Layer 0 stays local at every T.

**2. That far-reaching attention buys almost nothing.** Validation loss
binned by how many tokens of context the prediction had (SE across windows):

| context (tokens) | T = 64 | T = 128 | T = 256 | T = 512 |
|---|---|---|---|---|
| 1–2 | 2.81 | 2.95 | 2.96 | 3.19 |
| 4–8 | 2.24 | 2.35 | 2.30 | 2.44 |
| 16–32 | 2.23 | 2.25 | 2.29 | 2.33 |
| 32–64 | **2.20** | 2.22 | 2.27 | 2.26 |
| 64–128 | | 2.23 | 2.25 | 2.28 |
| 128–256 | | | 2.24 | 2.26 |
| 256–512 | | | | 2.29 |

(SE ≈ 0.02–0.04 per bin for 32+ tokens.)

- **Within every model, loss stops improving after ~32–64 tokens** of context
  (≈ 65–130 characters, about one or two lines of verse). Beyond that the
  curves are flat within noise.
- **At the same context, longer-window models are worse**: in nearly every
  bin the order is T = 64 ≤ 128 ≤ 256 ≤ 512. Per-bin differences are only
  1–2 SE, but the order is consistent across bins and matches the overall
  validation numbers, which come from the same text.
- So the long models aren't hurt by *seeing* more; they are simply worse
  models, even when predicting from short context.

**Interpretation (hypotheses, not proven here).** At fixed 4096 tokens/step,
T = 512 means a batch of only 8 windows instead of 32. The tokens per step are
equal, but tokens inside one window are strongly correlated, so the gradient
averages over fewer *independent* samples. The model also has to learn
behaviour for 512 positions with the same budget, and its upper heads spend
weight on distant tokens that carry little information for next-token
prediction on this data. Separating these would need e.g. T = 512 with a
batch of 32 (4× the tokens, so a different budget); not run.

**Takeaway.** For Tiny Shakespeare at this scale, the useful context is about
one to two lines. A longer window costs training efficiency and is not repaid.
Attention maps *showed* the cause (far-reaching heads); the loss-by-position
curve *measured* that the reach was useless. Neither alone would have been
convincing.

### Pitfalls met while building this
- `distance_profile` first included query 0 at distance 0 while every other
  statistic excluded it. A hand-built test caught the 0.1 mismatch.
- The first "share within distance d" curve normalised a profile whose entries
  were averaged over *different* query sets, so it was not a true CDF.
  Replaced by `reach_cdf`, which accumulates each query's own distribution.
- Per-position loss is far too noisy at a few hundred windows (± 0.1).
  Log-spaced bins with standard errors made it readable.

---

## 4. Error analysis: where the bits go: `src/analysis/error_analysis.py`, `scripts/error_analysis.py`

### Purpose
The loss is a mean of per-token losses $\ell_t = -\ln p(x_{t+1} \mid x_{\le t})$.
Grouping the *sum* by a property of the target token shows what the
remaining loss is spent on. Run on the centre model over **every**
non-overlapping validation window: 421 × 127 = 53,467 predictions.

Mean **2.245** bits/char. (The training summary's 2.225 comes from 100 random
batches; this is an exact pass. All grid runs use the same estimator, so
comparisons between them are unaffected.) Top-1 accuracy 31.9 %, top-5 53.4 %.

### By kind of target token
A BPE token is classified from its string and the token before it (tests use
a toy vocabulary with known answers):

| category | share of tokens | share of loss | bits/token | top-1 |
|---|---|---|---|---|
| word start (`␣the`, a word after punctuation) | 31.2 % | **44.4 %** | 6.60 | 7.6 % |
| word continuation (`ing` after `walk`) | 42.0 % | 36.9 % | 4.08 | 40.2 % |
| line start (after `↵`: speaker names, verse openings) | 6.5 % | 7.2 % | 5.17 | 10.6 % |
| punctuation | 10.4 % | 7.9 % | 3.55 | 40.7 % |
| newline | 8.3 % | 1.5 % | 0.85 | 90.3 % |

- **Choosing the next word is the hard part.** Word starts are under a third
  of the tokens but almost half of the loss, with 7.6 % top-1 accuracy.
  Once a word has started, finishing it is far easier (40 %).
- **Layout is nearly solved.** Newlines cost 0.85 bits and are predicted
  right 90 % of the time: the model knows the verse line length and the
  "blank line, NAME:, newline" structure.

### By training frequency of the target
Targets split into four bins holding equal numbers of validation occurrences:
the 14 most frequent tokens cost 2.51 bits each (60 % top-1); the 329 rarest
cost 6.06 (21 %). Rare tokens are a quarter of the targets but a third of the loss.

### A distribution shift in the split
The most confident mistakes are all speaker names: after `↵↵FERDINA` the
model says `:` with p ≈ 1.00, after `↵↵PR` it says `INC` (PRINCE). The
validation split is the **last 10 % of the file**, mostly *The Tempest*, and
**316 of its 871 speaker turns are by characters who never appear in training**
(PROSPERO 62, SEBASTIAN 42, ANTONIO 38, GONZALO 37, MIRANDA 32, ARIEL 21, …).

| target tokens | share of tokens | share of loss | bits/token | top-1 |
|---|---|---|---|---|
| speaker name, seen in training | 6.0 % | 2.3 % | 1.76 | 68 % |
| speaker name, **unseen** in training | 3.7 % | 6.7 % | **8.41** | 14 % |
| not a name | 90.4 % | 91.1 % | 4.68 | 30 % |

If unseen names cost what seen ones do, validation would fall by about
**0.12 bits/char** (2.245 → ≈ 2.13). That part of the absolute number is a
property of the split, not of the model. It is identical for every model, so
the experiment comparisons stand. A shuffled-by-play split would remove it,
but the contiguous split is the standard for this dataset, so it is kept and
stated.

### Calibration
Bin predictions by confidence (probability of the top token) and compare with
how often the top token was right. Perfect calibration lies on the diagonal.

$$\text{ECE} = \sum_b \frac{n_b}{N}\,\bigl|\text{acc}_b - \text{conf}_b\bigr|$$

ECE = **0.040**. The model is **slightly overconfident in every bin** (e.g.
says 0.55, right 0.46; says 0.97, right 0.92). That fits the train/val gap
(0.32 bits/char, §2): probabilities are sharpened on the training text and
carry over to validation text where they are less deserved. It is still
fairly well calibrated: cross-entropy training rewards honest probabilities.
(Tests: synthetic "right with probability = confidence" data gives ECE < 0.01;
"says 90 %, right 60 %" gives exactly 0.3.)

### Costliest single tokens
`,` (2.3 % of all loss), `↵` (1.5 %), `.` (1.5 %), `:` (1.3 %) and `O`
(1.3 %: 11.4 bits each, mostly the second half of PROSPERO and *O* as an
interjection). Punctuation is cheap per token but frequent; `O` is rare and
very expensive.

### Takeaways
- The remaining loss is mostly **content** (which word comes next), not
  **form** (spelling, line breaks, punctuation), the ordering one expects
  from a small model on a small corpus.
- A model's worst errors can come from the **data split**, not the model.
  Reading the most confident mistakes found it in one look.
- Calibration is a cheap extra lens: the mild overconfidence is the same
  memorisation that the train/val gap shows.

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
11. Why exclude query 0 from every attention statistic?
12. Why normalise entropy by $\ln(i+1)$? What would raw entropy averaged over positions measure instead?
13. What mean distance and normalised entropy does uniform causal attention give? A pure previous-token head?
14. How can a head have entropy ≈ 0.5 and still a mean distance near the uniform value? What does that say about how it chooses keys?
15. Why does the reach CDF use only queries $i \ge T/2$ when comparing models with different T?
16. The T = 512 model looks much further back but is worse. Which measurement shows the far attention is not useful, and why is it needed in addition to the attention maps?
17. Give two reasons a longer window could hurt at a fixed number of tokens per step. What experiment would separate them?
18. What is an attention sink, and how would you detect one with these statistics?
19. Why group the *sum* of per-token losses rather than compare mean losses per category?
20. Word starts are 31 % of tokens but 44 % of the loss. What does that say about what the model has and hasn't learned?
21. Why do the frequency bins hold equal numbers of target *occurrences* rather than equal numbers of distinct tokens?
22. Define ECE. Why can a model be well calibrated and still have high loss?
23. Unseen speaker names cost 8.4 bits/token. Why doesn't that invalidate comparisons between models? What would you change to remove it?
24. Why is the training summary's validation number (2.225) slightly different from the full pass (2.245)?
