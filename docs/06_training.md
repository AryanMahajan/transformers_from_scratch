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

## Self-check questions
1. Why does one SGD learning rate struggle with very different curvatures? What did the quadratic experiment show?
2. Write the Adam update. What do m and v estimate, and over roughly how many steps?
3. Derive $E[m_t] = (1-\beta_1^t)E[g]$. What is the step at t = 1 with and without bias correction?
4. Why is Adam invariant to rescaling the loss? Why is SGD not?
5. Why is L2 regularisation inside Adam not the same as weight decay? Which weights does it under-regularise?
6. Which parameters should not be decayed, and why?
7. How much memory does Adam add? What must be saved to resume training exactly?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Adam (algorithm + bias-correction derivation) | Kingma & Ba 2014, *Adam: A Method for Stochastic Optimization*: https://arxiv.org/abs/1412.6980 |
| AdamW (decoupled weight decay) | Loshchilov & Hutter 2017, *Decoupled Weight Decay Regularization*: https://arxiv.org/abs/1711.05101 |
