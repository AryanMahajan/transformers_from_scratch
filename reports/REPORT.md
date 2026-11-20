# A Decoder-Only Transformer from Fundamental Operations: Implementation and Experiments

*Endterm report, draft.* Code: this repository. Every number below has a
command in [`docs/REPRODUCE.md`](../docs/REPRODUCE.md); derivations are in the
per-phase notes in [`docs/`](../docs/).

---

## Abstract

We implement a GPT-style decoder-only Transformer language model using NumPy
alone: no autograd, no deep-learning framework, no pretrained weights. Every
forward pass, backward pass and optimiser step is derived by hand and verified
by finite-difference gradient checks (220 tests). Trained on Tiny Shakespeare,
a 0.23M-parameter model reaches 2.81 bits per character with a character
tokenizer and 2.53 with a BPE-512 tokenizer, against 3.58 for a character
bigram. A controlled study then varies depth, heads, context length,
positional encoding and width one at a time, with three seeds per
configuration. The study found a design flaw before it found any architectural
effect: fixed sinusoidal positions were 35× larger than the token embeddings
at initialisation, which cost 0.36 bits/char and inflated seed variance 4.5×.
Scaling token embeddings by √d_model (Vaswani et al. §3.4) fixes both.
Attention analysis shows interpretable heads (previous-token, word-start,
speaker-line) and explains why longer contexts hurt: heads reach far back but
the loss stops improving after ~32–64 tokens. Error analysis locates the
remaining loss in word choice rather than form, and finds that 36 % of
validation speaker turns are characters absent from training.

---

## 1. Introduction

**Research question.** Can a decoder-only Transformer language model be
implemented and trained from fundamental numerical operations, and how do
architectural choices (depth, heads, context length, positional encoding,
model size) affect its language-modelling behaviour?

Frameworks hide the parts of a Transformer that are easiest to get subtly
wrong: the backward pass of attention and LayerNorm, gradient accumulation for
shared weights, numerical stability of softmax and cross-entropy. Writing them
by hand makes each one explicit and testable. The cost is speed, so the study
works at small scale (0.13M–10.8M parameters, 12.3M training tokens per run)
and puts its effort into measurement quality instead: one variable at a time,
multiple seeds, and noise bands on every comparison.

**Contributions.**
1. A complete, hand-derived implementation with a gradient check for every
   layer and for the full model (§2).
2. A reproducible training pipeline: AdamW, warmup + cosine schedule, global
   clipping, bit-identical checkpoint/resume, optional GPU via CuPy (§2–3).
3. A controlled architecture study with seed-noise estimates (§5).
4. The diagnosis and fix of a positional-encoding scale mismatch (§4).
5. Attention and error analyses that explain *why* results came out as they
   did (§6–7).

---

## 2. Implementation

### 2.1 Architecture
Pre-LN decoder-only Transformer (Xiong et al. 2020), as in GPT-2:

```
ids [B,T] → √D·TokenEmb + PosEnc → N × [ x += MHA(LN(x));  x += FFN(LN(x)) ] → LN → logits (tied to TokenEmb)
```

Causal multi-head attention with $\text{softmax}(QK^\top/\sqrt{d_h} + M)V$,
GELU feed-forward of width 4D, sinusoidal or learned positions, tied
input/output embeddings, GPT-2 initialisation ($\mathcal N(0, 0.02^2)$, residual
projections scaled by $1/\sqrt{2N}$).

### 2.2 Backpropagation by hand
Every module implements `forward` (caching what backward needs) and `backward`
(accumulating parameter gradients, returning the input gradient). Key
derivations, each in the notes:
- **Fused softmax + cross-entropy:** $\partial L/\partial z = p - y$, avoiding
  a V × V Jacobian.
- **LayerNorm:** the three-term input gradient; two-pass variance, because
  $E[x^2]-\mu^2$ cancels catastrophically in float32.
- **Attention:** softmax backward per row $dS = A \odot (dA - \text{rowsum}(dA \odot A))$,
  then the two matmuls; the input gradient of x sums contributions through q, k and v.
- **Embedding:** `np.add.at`, because buffered fancy-index `+=` silently drops
  repeated token ids.
- **Tied embeddings:** the gradient is the sum of the lookup and output-layer
  uses, a consequence of the accumulate-don't-overwrite contract.

One small derived result: the key projection has no bias, because
$q\cdot(k+b)$ shifts each score row by a constant that softmax ignores, so the
bias would receive exactly zero gradient.

### 2.3 Verification
- **Gradient checks** in float64 with norm-wise relative error, for every
  layer and for full 2-block models (tied/untied × sinusoidal/learned): errors
  below 1e-8.
- **Property tests:** causality (changing future tokens never changes past
  logits), initial loss ≈ ln V, shift-by-one batching, sampling frequencies
  matching softmax(z/τ) over 40k draws, BPE trainer identical to a brute-force
  reference.
- **Midterm gate:** a randomly initialised model memorises the first 257
  characters (final loss 0.0019) and greedy decoding from "F" reproduces all
  257 characters exactly.
- **GPU backend:** CuPy gradients match CPU gradients to 1e-8 in float64;
  training is ~8× faster on an RTX 3050.

220 tests in total (`pytest`).

### 2.4 Training
AdamW (decoupled weight decay 0.1 on matrices only), linear warmup (100 steps)
then cosine decay from 3e-3 to 3e-4, global-norm clipping at 1.0.
Checkpoints are single pickle-free `.npz` files including optimiser and RNG
state, so an interrupted run resumes **bit-identically** (tested with `==`).

---

## 3. Data, tokenizers and first results

**Data.** Tiny Shakespeare (1.1M characters), contiguous 90/10 split; the
vocabulary comes from the training split only.

**Tokenizers.** Character-level (V = 65 + `<unk>`) and BPE (Sennrich et al. 2016)
with a character base and GPT-2-style pre-tokenisation. BPE-512 compresses the
validation text to 2.07 characters per token. Losses are reported in **bits
per character** (nats / ln 2 / chars per token), the only unit that compares
across tokenizers.

| model (0.23M params, D=64, H=4, N=4, T=128) | val bits/char |
|---|---|
| unigram, char | 4.83 |
| bigram, char | 3.58 |
| bigram, BPE-512 | 2.67 |
| Transformer, char | 2.81 |
| Transformer, BPE-512 | 2.53 |
| Transformer, BPE-512 + √D fix (§4), 3-seed mean | **2.225 ± 0.018** |

**Sampling.** Temperature, top-k and nucleus (top-p; Holtzman et al. 2020)
sampling trade diversity against real-word rate: greedy decoding is 88 % real
words but only 20 % distinct words (it loops); τ = 1 is 92 % distinct but only
44 % real words; τ = 0.8 with top-k = 10 gives 77 % / 74 %.

---

## 4. A design flaw found by the seed study

The first grid used three seeds for the centre configuration and found a
seed standard deviation of 0.077 bits/char, larger than several effects to be
measured. In the same grid, learned positions beat sinusoidal ones by 0.32
bits/char, far too large for that choice. Both pointed to one cause.

**Diagnosis.** The input is $h = E[\text{ids}] + PE$. Sinusoidal rows have norm
$\sqrt{D/2} = 5.66$; token rows initialised at $\mathcal N(0, 0.02^2)$ have norm
≈ 0.16. Positions were **35×** larger at initialisation, and still 5.7× after
training. After the first LayerNorm, token identity was a small component of
every block's input. Learned positions start at the same small scale, so they
never had this problem.

**Fix.** Vaswani et al. (§3.4) multiply embeddings by $\sqrt{d_{model}}$:
$h = \sqrt D\, E[\text{ids}] + PE$, with backward $\partial L/\partial E \mathrel{+}= \sqrt D\,\partial L/\partial h$
(gradient-checked; removing the factor gives errors of 0.17–0.48).

| 3 seeds each | no scale | tokens × √D |
|---|---|---|
| sinusoidal | 2.583 ± 0.077 | **2.225 ± 0.018** |
| learned | 2.215 ± 0.010 | 2.216 ± 0.008 |

The scale recovers 0.36 bits/char for sinusoidal positions and does nothing for
learned ones, as the diagnosis predicts. Seed noise falls 4.5×. With the fix,
the two position types tie. All gradient checks passed with the flaw in place:
the code was correct, the design was not. The grid was rerun around the fixed
centre.

---

## 5. Architecture experiments

**Method.** Centre: BPE-512, D = 64, H = 4, N = 4, T = 128, sinusoidal + √D,
232,576 parameters. Each experiment changes one variable; every run trains
3000 steps × 4096 tokens (12.3M tokens). The context experiment keeps tokens
per step fixed (batch = 4096 / T). Cheap configurations use seeds 0, 1, 2
(mean ± std); the two largest models are single-seed. A difference smaller
than about two seed standard deviations is reported as no effect. The best
periodic validation loss is recorded alongside the final one, because larger
models overfit before the end.

> **[To fill when the grid finishes: 3-seed tables for A (depth), B (heads),
> C (context), E (width), from `reports/results.md`; summary figure
> `reports/figures/summary.png`.]**

Seed-0 results so far (val bits/char, centre 2.232 for seed 0, 3-seed mean 2.225):

| experiment | values → final val (best periodic) |
|---|---|
| A depth N | 2: 2.262 · 6: 2.200 (2.187) |
| B heads H | 2: 2.195 · 8: 2.229 |
| C context T | 64: 2.226 · 256: 2.249 · 512: 2.279 (2.258) |
| E width D (params) | 128 (0.86M): 2.286 (2.214) · 256 (3.3M): 2.599 (2.195) · 384, N=6 (10.8M): 3.075 (2.217) |

Already clear from seed 0:
- **Width does not pay with this much data.** The 3.3M and 10.8M models reach
  their best validation loss early (steps 1250 and 750) and then overfit badly:
  ~485k training tokens are seen ~25 times, without dropout. Their best points
  barely beat the 0.23M centre. This is consistent with the compute-optimal
  view of Hoffmann et al. (2022) that parameters and training tokens should
  grow together; here the tokens are fixed and repeated, so extra parameters
  mostly memorise.
- **Longer context hurts** at fixed tokens per step; §6 explains why.

---

## 6. What the attention heads do

Each attention layer keeps its map $A \in \mathbb R^{B \times H \times T \times T}$.
Per head we measure mean attention distance, previous-token weight,
first-token weight (attention sinks; Xiao et al. 2024) and entropy normalised
by $\ln(i+1)$, the entropy of uniform attention over the allowed keys.

**Interpretable heads (centre model).** In layer 0, two heads are sharply local
(distance ≈ 2, entropy ≈ 0.3), and one **attends to the first piece of the
current word**: BPE splits *Baptista* into `·B|a|pt|is|t|a`, and on every
piece the head looks at `·B`. It reassembles words from sub-word pieces. Higher
layers look further (7–17 tokens, against 32 for uniform attention) and
several lock onto the speaker line (`NAME:↵`). No attention sinks appear at
this scale (first-token weight 0.01–0.02).

**Why longer context hurt.** In the T = 256 and T = 512 models the upper heads
reach far: 45–68 % of their attention lies ≥ 64 tokens back, against 4 % at
T = 128. Mean distance reaches the uniform value at T = 512, although many of
those heads are fairly sharp, so they are content-addressed with little recency
bias. Validation loss binned by available context shows this reach is useless:
**every model stops improving after ~32–64 tokens** (one or two lines of
verse), and at equal context the longer-window models are consistently worse.
They aren't hurt by seeing more; they are worse models overall. The likely
reason is training efficiency: at 4096 tokens per step, T = 512 means 8
windows per batch instead of 32, so each gradient averages fewer independent
samples. (This is a hypothesis; separating it from other explanations would
need a different token budget.)

---

## 7. Where the remaining loss goes

Per-token losses over all 53,467 validation predictions, grouped by the kind
of target token:

| target | share of tokens | share of loss | top-1 accuracy |
|---|---|---|---|
| word start | 31 % | **44 %** | 8 % |
| word continuation | 42 % | 37 % | 40 % |
| line start | 6 % | 7 % | 11 % |
| punctuation | 10 % | 8 % | 41 % |
| newline | 8 % | 2 % | 90 % |

The model has learned **form** (spelling, verse line length, the
`NAME:` / newline layout: newlines are 90 % correct) far better than
**content** (which word comes next: 8 % correct).

**A distribution shift in the standard split.** The model's most confident
mistakes are all speaker names: `FERDINA` completed with `:`, `PR` with `INC`
(PRINCE). The validation split is the end of the file, mostly *The Tempest*,
and 316 of its 871 speaker turns belong to characters who never appear in
training. Their name tokens cost 8.4 bits each, against 1.8 for names seen in
training, roughly 0.12 bits/char of the validation loss. The shift is the same
for every model, so the comparisons stand, but the absolute numbers are
pessimistic.

**Calibration.** Expected calibration error is 0.040, with slight
overconfidence in every confidence bin (e.g. says 0.55, right 0.46). This is
consistent with the train/validation gap: modern networks tend to be
miscalibrated in this direction (Guo et al. 2017).

---

## 8. Limitations

- **Scale.** 12.3M training tokens per run on a 1.1M-character corpus; results
  describe the small-data regime and need not transfer to large models.
- **Seeds.** Three seeds for cheap configurations, one for the two largest
  models; differences below ~0.04 bits/char between those two are
  inconclusive.
- **Confounds stated, not removed.** The width experiment lowers the learning
  rate for the two largest models; the context experiment changes batch size
  with T, by design.
- **No regularisation sweep.** Dropout was never implemented, so the large
  models' overfitting is reported rather than fixed.
- **Validation split shift** (§7) inflates absolute losses by ~0.12 bits/char.

---

## 9. Conclusion

A decoder-only Transformer can be built and trained from NumPy primitives with
every gradient derived by hand and verified numerically. The main lessons came
from measurement rather than architecture. Seed variance exposed a design flaw
that every correctness test had missed. The fix (one √D factor from a single
sentence of the original paper) mattered more than any architecture choice in
the grid. Analysis tools turned surprising results (long context hurting,
large models doing no better) into explanations: unused attention reach, and
too few tokens for the parameters.

> **[To fill when the grid finishes: final ranking of architectural effects
> with 3-seed confidence.]**

---

## References

- Ba, Kiros & Hinton (2016). *Layer Normalization*. arXiv:1607.06450
- Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks*. ICML. arXiv:1706.04599
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556
- Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. arXiv:1904.09751
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization*. arXiv:1711.05101
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units*. arXiv:1508.07909
- Vaswani et al. (2017). *Attention Is All You Need*. arXiv:1706.03762
- Xiao et al. (2024). *Efficient Streaming Language Models with Attention Sinks*. ICLR. arXiv:2309.17453
- Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture*. arXiv:2002.04745
