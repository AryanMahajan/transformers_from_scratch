# Phase 6: Training (study notes)

Revision notes for Phase 6, which makes the model learn. Exit criterion
(midterm gate): a randomly initialised model memorises a tiny corpus.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

---

## 1. Adam and AdamW: `src/optim/adamw.py`

### Why not plain SGD?
SGD steps by $-\eta g$: the step size is proportional to the gradient size,
with one learning rate for every parameter. With very different curvatures
(Transformers have them: embeddings, LayerNorm gains, attention matrices), the
largest curvature forces a tiny η, and flat directions crawl.

**Measured** on $f = x_1^2 + 100x_2^2 + 10^4x_3^2$, 2000 steps from (1,1,1):
- SGD at its best stable η (< 1/a_max): $x_1$ still **0.835**.
- Adam (lr 0.05): all coordinates ≈ 0.

### The update (per parameter, step t, g = gradient)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}\qquad
\theta \leftarrow \theta - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

- **m (momentum):** an exponential moving average (EMA) of g. It averages out
  noise from minibatches, so it gives the consistent *direction*. Memory
  ≈ $1/(1-\beta_1)$ = 10 steps.
- **v:** an EMA of g², the typical *scale* of g for that parameter. Memory
  ≈ 1000 steps.
- **m̂/√v̂ ≈ a normalised gradient**, roughly in [−1, 1]. So **every
  parameter moves about η per step**, whatever its gradient scale. That means
  per-parameter learning rates for free, and invariance to rescaling the loss
  (tested: ×1000 loss gives the same steps).
- Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (ε only prevents division by 0).

### Why bias correction
m and v start at 0, so early EMAs are biased toward 0:
$E[m_t] = (1-\beta_1^t)\,E[g]$ (sum the geometric series
$(1-\beta_1)\sum_{k<t}\beta_1^k$). Dividing by $1-\beta_1^t$ removes it.
- **At t = 1:** m̂ = g and v̂ = g², so the step is exactly **η·sign(g)**
  (tested with g = 1e-4, −3 and 250: all step η).
- **Without correction:** step = η·0.1|g|/√(0.001 g²) = **3.16η** at t = 1.
  It's distorted for the first few hundred steps (v's bias lasts ~1/(1−β₂)
  steps).

### AdamW: decoupled weight decay
Weight decay shrinks weights toward 0: $\theta \leftarrow \theta - \eta\lambda\theta$.

**L2-in-Adam** (the "obvious" way: add λθ to the gradient) is *not* the
same thing. The λθ term goes through m̂/√v̂ as well, so it's **divided by
√v̂**. Weights with large gradients barely get regularised, and weights with
small gradients get hammered. **AdamW** applies decay to the weights
directly, outside the adaptive part.

**Measured** (two weights, gradient scale 0.01 vs 1, 200 steps, η = 0.01,
λ = 0.1). Decay effect = no-decay run − decay run, with the same noise:

| | small-grad weight | large-grad weight |
|---|---|---|
| AdamW | 0.164 | 0.204 (theory for both: 1−(1−ηλ)²⁰⁰ = 0.181) |
| L2-in-Adam | **0.857** | 0.197 (**4.4×** uneven) |

### What gets decayed
Only matrices (`ndim ≥ 2`: Linear W, embeddings). **Not** biases, LayerNorm
γ and β: shrinking γ toward 0 would fight the normalisation and doesn't
reduce capacity meaningfully. This is the same rule as nanoGPT/GPT-2.

### Code notes
- Takes `model.named_parameters()`. State `m`, `v` keyed by name, so
  `state_dict()` is a plain dict for checkpoints (Phase 7 resume; tested to be
  **bit-identical** to an uninterrupted run).
- All updates in place (`*=`, `+=`, `-=`): float32 stays float32, and the
  model's arrays stay the same objects.
- Decay is applied before the Adam step, using the current θ (PyTorch's order).
  The decay is multiplied by η, so it follows the LR schedule.
- `lr` is a public attribute so a schedule can set it each step.
- **Memory:** m and v are two extra copies of every parameter, so training
  memory ≈ 3× the weights (+ gradients = 4×).

### Tests (`tests/test_adamw.py`)
- First step = η·sign(g) for gradients spanning 1e-4..250.
- Matches the written-out formulas over 5 steps (rtol 1e-12).
- Loss ×1000 → same trajectory.
- Converges on the ill-conditioned quadratic.
- Zero-gradient decay = $(1-\eta\lambda)^n$ exactly; biases untouched.
- AdamW vs L2-in-Adam (table above).
- Save/load mid-run → bit-identical continuation, float32 preserved.
- In place; rejects duplicate parameters.

---

## 2. Gradient clipping and the LR schedule: `src/optim/clip.py`, `src/optim/schedule.py`

### Global-norm clipping

$$\|g\| = \sqrt{\textstyle\sum_{\text{all params}} \sum g^2},\qquad \text{if } \|g\| > c:\ g \leftarrow g\cdot\frac{c}{\|g\|}$$

- **One factor for all parameters**: the update direction is unchanged, only
  its length is capped. Per-tensor clipping would rescale tensors differently
  and **rotate** the update (tested: cos < 0.99).
- Returns the **pre-clip** norm. Log it: a norm spiking during training is
  the earliest sign of instability.
- The norm is summed in float64 (B·T·D-sized float32 sums lose digits).
- Typical c = 1.0 (GPT-2/nanoGPT).

### "Adam is scale-invariant, so why clip?"
Clipping *every* step by the same factor would do nothing to Adam. But a single
**spike** (one bad batch) is different:
1. It enters m, so the spike step is large and points in a garbage
   direction, and momentum carries it for ~10 steps.
2. It enters v (scaled by g² = 10⁶×) and **stays there for ~1/(1−β₂) = 1000
   steps**, so every later step is divided by a huge √v and **training
   stalls**.

**Measured** (one 1000× gradient at step 100, mean |step|):

| | before spike | spike step | steps 201–400 |
|---|---|---|---|
| no clipping | 0.0018 | 0.0098 (5×) | **0.0001 (18× stalled)** |
| clipping | 0.0018 | 0.0018 | 0.0019 |

### Learning-rate schedule: warmup + cosine

$$\text{lr}(s) = \begin{cases}\eta_\max\frac{s+1}{W} & s<W\[2pt]
\eta_\min + \tfrac12(\eta_\max-\eta_\min)\left(1+\cos\pi\frac{s-W}{S-W}\right) & W\le s\le S\[2pt]
\eta_\min & s > S\end{cases}$$

- **Warmup (why, even with pre-LN):** in the first steps v̂ is estimated
  from very few gradients, so Adam's per-parameter step size is high-variance
  (Liu et al. 2019, RAdam). The initial gradients are also the largest, as the
  model leaves its init. A short linear ramp lets the statistics settle before
  full-size steps. Pre-LN makes warmup *less critical* (decision 1), not
  useless.
- **Cosine decay:** large steps early (fast progress), small steps late
  (settle into a minimum instead of bouncing around it). Smooth, with no
  step-schedule hyperparameters. It spends a long time near both ends.
- **η_min ≈ η_max/10** (nanoGPT convention) so learning never fully stops.
- **`s+1`** so step 0 already has lr > 0 (otherwise the first step is wasted).
- Midpoint of the cosine = $(\eta_\max+\eta_\min)/2$ exactly (tested).

### Order inside one training step
```
loss = forward → backward → norm = clip_grad_norm(...) → opt.lr = cosine_lr(step) → opt.step() → zero_grad
```
Clip **after** backward and **before** step: the optimizer must see the
clipped gradients.

### Tests (`tests/test_clip_schedule.py`)
- Global norm = norm of all gradients concatenated.
- Below threshold: bit-identical, returns the norm.
- Above: norm = c, direction preserved, returns the pre-clip norm.
- Per-tensor clipping would rotate the update; global doesn't.
- Zero gradients safe.
- **Spike experiment** (table above).
- Warmup linear with lr(0) > 0; peak at W; midpoint, end and after-end values;
  monotone decay ≥ η_min; validation.

---

## 3. Batching and the training loop: `src/training/dataset.py` (`get_batch`), `src/training/trainer.py`

### Batches: random windows, shifted by one
```
data:  t h e q u i c k b r o w n ...
x   =  data[i   : i+T]      [B,T]
y   =  data[i+1 : i+T+1]    [B,T]      y[t] = the token after x[t]
```
- **One window = T training examples.** Logits at position t see
  x[0..t] (causal mask) and predict y[t]. So a window trains contexts of
  length 1, 2, …, T at once. The mask is what makes this legal: without it,
  position t could just read x[t+1] = y[t].
- Starts are uniform over **every valid window**, i ∈ [0, N−T−1]
  (`rng.integers(0, N-T)`, high exclusive). Tested: all starts reached, and the
  last token is used as a target.
- Implemented as one gather: `idx = starts[:,None] + arange(T)`, then
  `data[idx]` and `data[idx+1]`.
- Random sampling (not sequential epochs): with ~1M tokens and T = 64 there
  are ~1M overlapping windows, so "epoch" isn't a natural unit. Steps are.

### One training step (`train_step`)
```
logits = model(x)                         [B,T,V]
loss, probs = cross_entropy(logits, y)    mean over B·T predictions
opt.zero_grad()                           grads accumulate with += → must clear
model.backward(dlogits.reshape(B,T,V))
norm = clip_grad_norm(params, c)          returns the pre-clip norm
opt.step()
```
`opt.lr = cosine_lr(step)` is set just before each step.

### The loop (`train`)
- **Eval before the step**, so step 0 records the **init loss ≈ ln V**
  (first sanity check), and the final eval is at `max_steps`.
- `estimate_loss`: mean over `eval_iters` random batches, forward only,
  `model.eval()` then back to `train()`. A single batch is too noisy to
  compare train and val. (Tested: it changes no parameters.)
- **Two RNGs:** one for training batches, one for eval. Adding or removing
  evals doesn't change which training batches are seen, so runs stay
  comparable.
- **Fail fast:** a non-finite loss raises immediately instead of silently
  training on nan.
- Returns `history`: per-step loss/lr/grad_norm, per-eval train/val loss, and
  the optimizer (for checkpointing in Phase 7). Same seed → identical history
  (tested).

### Observation: the plateau, then a sudden drop
Toy corpus `"thequickbrownfox" × 200` (period 16, 15 distinct letters), 1 block, D = 32:

| | eval loss at steps 0, 100, 200, 300, 400 |
|---|---|
| tied, lr 3e-3 | 3.26, **2.70**, 0.92, 0.06, 0.02 |
| tied, lr 1e-2 | 3.26, **2.70**, 0.15, 0.01, 0.006 |
| untied, lr 3e-3 | 3.27, 0.11, 0.01, 0.01, 0.008 |

- **3.26 = ln 26:** uniform at init.
- **2.70 ≈ the unigram entropy** of the string. The model first learns letter
  frequencies only.
- **Why only the tied model plateaus:** at init the branches are tiny, so the
  final h ≈ E[x_t], and logits = h·Eᵀ are largest for **the current token
  itself** (its dot product with itself). The tied model starts biased toward
  "repeat the current letter" and must first learn to move h away from
  E[x_t]. The untied head has no such bias.
- **Floor ≈ 0.005, not 0:** "o" is followed by "w" or "x". Position 0 of a
  window sees one letter, so after a lone "o" the next letter is a coin flip.
  Expected floor ≈ (1/24)·(2/16)·ln 2 ≈ 0.004.
- Lesson: **a flat loss curve isn't necessarily a bug.** Check what the
  plateau value means (here: unigram entropy) and train longer before
  debugging.

### Tests (`tests/test_trainer.py`)
- `get_batch`: shapes, `y == x+1` on `arange` data, contiguous rows, all
  valid starts reached, last token used, rejects short data, seeded.
- `estimate_loss` at init ≈ ln V; parameters unchanged; train mode restored.
- History lengths, eval steps, lr follows `cosine_lr` exactly, optimizer t.
- Deterministic given the seed.
- Learns the periodic corpus: ln V → < 0.1 in 300 steps.
- Logging works.

---

## 4. Midterm gate: memorise a tiny corpus (`scripts/overfit_tiny.py`)

### Why this test
If a model **can't** drive the loss to ~0 on a tiny dataset, something in
forward, backward, the optimizer or the data pipeline is broken. Memorisation
is the easiest thing a model can do. It's the standard first sanity check
before any real training (Karpathy's "overfit one batch").

### Setup
- The first **257 characters** of Tiny Shakespeare. **One** window of
  T = 256 (batch size 1; with N = T+1 there is exactly one valid window).
- Every position sees its **entire prefix**, so the next character is fully
  determined and the minimum loss is 0. (With random short windows it wouldn't
  be: §3's floor.)
- Model: D = 64, H = 4, N = 2, max_len = 256, tied, sinusoidal:
  **104,192 parameters**.
- AdamW lr 3e-3 → 3e-4 cosine, 20 warmup steps, **weight decay 0** (I *want*
  to overfit), clip 1.0, 400 steps.

### Result
`experiments/overfit_tiny/loss_curve.png`, `summary.json`

| | value |
|---|---|
| init loss | **4.1902** (ln 66 = 4.1897) |
| unigram entropy of the passage | 3.16 |
| final loss (step 400) | **0.0019** |
| teacher-forced accuracy | **100%** |
| greedy generation from just "F" | **257/257 characters exact** |
| time | 157 s (CPU, NumPy) |

### Reading the curve
1. **Step 0 at ln V:** the init is correct (Phase 5 §3).
2. **Bend near the unigram line (steps 15–25):** it first learns character
   frequencies, the same plateau as §3, much shorter here.
3. **Steep drop (steps 60–100):** it starts using context. Attention learns
   to look back.
4. **Slow tail:** pushing already-correct probabilities from 0.99 to 0.998.
   The loss keeps falling as the logits grow, but accuracy is already 100%.

### Why it needs attention, not just bigrams
"First Citizen:" appears 3× and "All:" 3×, and the text after each
"All:\n" differs ("Speak", "Resolved", "W…"). A bigram or short-context model
can't tell these occurrences apart. Only the long prefix can. Free-running
greedy decoding (feeding back its own outputs) reproducing all 257 characters
shows every one of those decisions was learned.

### Greedy decoding: `src/generation/greedy.py`
```
repeat n_new times:
    ctx = ids[-max_len:]                  crop: no positions beyond max_len
    next = argmax(model(ctx)[0, -1])      only the LAST position predicts the new token
    ids.append(next)
```
- Deterministic, so it's the right decoder for checking memorisation.
- Cost: one full forward per token, O(T²) each, because there's no KV cache.
  Fine at this size.
- Teacher-forced accuracy vs free-running: teacher forcing feeds the *true*
  prefix. Free-running feeds the model's own outputs, so one mistake would
  derail everything after it. 257/257 is the stronger result.
- Sampling with temperature/top-k comes in Phase 7.

### Tests
- `tests/test_greedy.py`: length, prompt kept, crops past max_len, deterministic,
  restores train mode, next = argmax of the last position, reproduces a learned
  cycle, empty prompt rejected.
- `tests/test_overfit_script.py`: smoke run (33 chars, 60 steps) writes the
  plot and summary, init ≈ ln V, loss halves. Skipped if the data isn't downloaded.

### For the demo
1. `python scripts/overfit_tiny.py`: watch the loss go from 4.19 to ~0.002.
2. Show `loss_curve.png` and explain the four regions above.
3. Show the generated text = the original passage.
4. Explain why this proves forward, backward, the optimizer and the data
   pipeline are all correct together, and why it says nothing about
   generalisation (val loss would be terrible). That comes in Phase 7.

---

## Self-check questions
1. Why does one SGD learning rate struggle with very different curvatures? What did the quadratic experiment show?
2. Write the Adam update. What do m and v estimate, and over roughly how many steps?
3. Derive $E[m_t] = (1-\beta_1^t)E[g]$. What is the step at t = 1 with and without bias correction?
4. Why is Adam invariant to rescaling the loss? Why is SGD not?
5. Why is L2 regularisation inside Adam not the same as weight decay? Which weights does it under-regularise?
6. Which parameters should not be decayed, and why?
7. How much memory does Adam add? What must be saved to resume training exactly?
8. Why clip the *global* norm instead of each tensor separately?
9. Adam is invariant to gradient scale, so why does clipping still help it? Describe what one spike does to m and to v.
10. Write the warmup + cosine schedule. Why warm up, even with pre-LN?
11. Where in the training step does clipping go, and why there?
12. Why log the pre-clip gradient norm?
13. How are x and y built from one window? Why does one window give T training examples, and why does that need the causal mask?
14. What is the range of valid window starts? What off-by-one bug would make the last token never be a target?
15. Why must `zero_grad` come before `backward` in every step?
16. Why average the eval loss over several batches, and why use a separate RNG for eval?
17. The loss sits at 2.70 for 100 steps, then drops. What does 2.70 correspond to? Why does the tied model plateau but the untied one doesn't?
18. Why is overfitting a tiny corpus the right first test? What bugs would it catch, and what does it *not* tell you?
19. Why use one window of T = N−1 instead of random short windows for the memorisation test?
20. Describe the four regions of the memorisation loss curve.
21. Why can't a bigram model memorise this passage? What does that say about attention?
22. Teacher-forced accuracy vs free-running generation: which is the stronger test, and why?
23. Why turn weight decay off here?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| The "overfit a tiny batch first" habit | Karpathy, *A Recipe for Training Neural Networks*: https://karpathy.github.io/2019/04/25/recipe/ |
| Adam (algorithm + bias-correction derivation) | Kingma & Ba 2014, *Adam: A Method for Stochastic Optimization*: https://arxiv.org/abs/1412.6980 |
| Gradient-norm clipping (original proposal) | Pascanu, Mikolov & Bengio 2012, *On the difficulty of training Recurrent Neural Networks*: https://arxiv.org/abs/1211.5063 |
| Why warmup helps Adam (variance of the adaptive LR) | Liu et al. 2019, *On the Variance of the Adaptive Learning Rate and Beyond* (RAdam): https://arxiv.org/abs/1908.03265 |
| AdamW (decoupled weight decay) | Loshchilov & Hutter 2017, *Decoupled Weight Decay Regularization*: https://arxiv.org/abs/1711.05101 |
