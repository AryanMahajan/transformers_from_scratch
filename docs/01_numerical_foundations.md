# Phase 1: Numerical Foundations (study notes)

These are revision notes for each function in Phase 1: what it is for, the
math, what to watch out for, and questions to test yourself. Each section
maps to code you can read alongside it.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

**Notation:** $z$ = logits (input), $p$ = probabilities (output),
$g = \partial L/\partial p$ = upstream gradient, $V$ = size of the axis we
normalise over.

---

## 1. Softmax (stable): `src/nn/softmax.py::softmax`

### What it's for
It turns a vector of real numbers into a probability distribution. In our
model it is used in two places:
- **Attention:** scores `[B,H,T,T]` → attention weights. Row *i* says how
  much token *i* attends to each token *j*.
- **Output:** logits `[B,T,V]` → the distribution over the next token.

### Math

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The outputs are positive and sum to 1.

### The problem
In float32, `exp(x)` overflows to `inf` for x ≳ 88. With logits of about
1000 you get `inf/inf = nan`. With very negative logits everything
underflows to 0, giving `0/0 = nan`.

### The fix: shift invariance

$$\frac{e^{z_i-c}}{\sum_j e^{z_j-c}} = \frac{e^{-c}e^{z_i}}{e^{-c}\sum_j e^{z_j}} = \text{softmax}(z)_i$$

Pick $c = \max_j z_j$. Then:
- the largest exponent is $e^0 = 1$, so nothing overflows
- the denominator is ≥ 1, so you never divide by zero

### Code notes
- `keepdims=True` keeps the reduced axis as size 1 (`[B,T,V]` → `[B,T,1]`)
  so broadcasting lines up for the subtract and the divide.
- An `axis` parameter keeps one function usable for both attention and LM
  logits.

### Gotchas
- Shifting by the **mean** doesn't work: the largest value can still be far
  above the mean and overflow.
- Masked attention uses `-inf` scores. `exp(-inf - max) = 0`, which is
  fine. But if an **entire row** is `-inf`, then `max = -inf` and
  `-inf - (-inf) = nan`. A causal mask always leaves the diagonal
  unmasked, so this can't happen there, but padding masks can trigger it.

---

## 2. LogSoftmax: `src/nn/softmax.py::log_softmax`

### What it's for
Cross-entropy needs $\log p_y$. Computing `np.log(softmax(z))` fails when
$p_i$ underflows to 0, because $\log 0 = -\infty$.

### Math: the log-sum-exp (LSE) trick

$$\log\text{softmax}(z)_i = z_i - \log\sum_j e^{z_j}
= (z_i - m) - \log\sum_j e^{z_j - m},\qquad m = \max_j z_j$$

This stays in log space the whole time, so a tiny probability like
$e^{-2000}$ is represented as $-2000$ rather than 0.

### Fact to remember
$\nabla_z \text{LSE}(z) = \text{softmax}(z)$. This is why the gradient of
cross-entropy comes out so neatly (§4).

---

## 3. Softmax backward: `src/nn/softmax.py::softmax_backward`

### What it's for
In attention, $A = \text{softmax}(S)$. Backprop gives us $dA$, and we need
$dS$.

### Jacobian (quotient rule)
- $i = j$: $\partial p_i/\partial z_i = p_i - p_i^2$
- $i \ne j$: $\partial p_i/\partial z_j = -p_ip_j$

$$J_{ij} = p_i(\delta_{ij} - p_j)\qquad J = \text{diag}(p) - pp^\top$$

### Vector–Jacobian product (what we actually compute)

$$\frac{\partial L}{\partial z_j} = \sum_i g_i p_i(\delta_{ij}-p_j)
= p_j\big(g_j - \langle g,p\rangle\big)$$

$$\boxed{dz = p \odot (g - \langle g,p\rangle)}$$

This costs O(V) per row, against O(V²) memory and compute to build $J$. For
attention that saves `B·H·T` matrices of size T×T.

### Properties (all tested)
- It needs only the **output** $p$, so the forward pass caches `p`, not `z`.
- Each row of $dz$ **sums to 0**, because shifting $z$ by a constant
  doesn't change $p$, so the gradient has no component along
  $\mathbf{1}$.

### How it's tested (`tests/test_softmax.py`)
1. Compared against the explicit Jacobian `g @ (diag(p) − ppᵀ)`.
2. Compared against central finite differences on `[B,T,V]` in float64,
   with relative error < 1e-6.
3. Each row sums to 0.

---

## 4. GELU and its derivative: `src/nn/activations.py`

### What it's for
GELU is the non-linearity in the FFN: `Linear(D→4D) → GELU → Linear(4D→D)`.
Without it, the two linear layers would collapse into one.

### Math
Exact form: $\text{GELU}(x) = x\,\Phi(x)$, where $\Phi$ is the standard
normal CDF. It acts as a *soft* gate: $x$ is scaled by "the probability a
standard normal is ≤ $x$". ReLU's gate is hard (0 or 1).

Tanh approximation (Hendrycks & Gimpel; used in GPT-2):

$$\text{GELU}(x) \approx \tfrac12 x(1+\tanh u),\quad u = \sqrt{2/\pi}\,(x + 0.044715x^3)$$

- $\sqrt{2/\pi}$ makes the slope at 0 match: $\tfrac12\sqrt{2/\pi} = 1/\sqrt{2\pi} = \Phi'(0)$.
- $0.044715$ is fitted numerically. **Measured max error vs exact: 4.7e-4**
  (at x ≈ 2.7).
- We use the approximation because NumPy has no vectorised `erf`.

### GELU vs ReLU
| | ReLU | GELU |
|---|---|---|
| Smooth at 0 | ✗ (kink) | ✓ |
| Gradient for x < 0 | 0 (units can "die") | small but non-zero |
| Monotonic | ✓ | ✗: dip to **−0.170 at x ≈ −0.75** |

### Derivative (product rule + chain rule)

$$\text{GELU}'(x) = \tfrac12(1+\tanh u) + \tfrac12 x\,(1-\tanh^2 u)\,\sqrt{2/\pi}\,(1+3\cdot0.044715\,x^2)$$

Sanity check: $\text{GELU}'(0) = \tfrac12 = \Phi(0)$.

### Backward
`dx = dout * GELU'(x)`. GELU is **element-wise**, so its Jacobian is
diagonal and the VJP is just a multiply. Compare softmax, whose Jacobian is
dense.

### Gotchas
- The backward pass needs the **input** `x` (softmax needed the output).
  GELU isn't invertible near the dip, so the FFN must cache the
  pre-activation.
- Reuse `tanh u` from the forward computation in the backward. Don't call
  `tanh` twice.

### How it's tested (`tests/test_activations.py`)
Limits (−10 → 0, 10 → 10); error vs exact erf GELU < 1e-3; the dip's
location and depth; finite differences on `[B,T,F]` (rel. err < 1e-6);
GELU'(0) = 0.5.

---

## 5. Fused cross-entropy: `src/nn/loss.py` (Checkpoint A)

### What it's for
This is the training objective. Logits `[B,T,V]` and targets `[B,T]` (the
true next token) go in, one scalar comes out. Every gradient in the model
starts here.

### Forward
With $N = B\cdot T$:

$$L = -\frac1N\sum_n \log p_{n,y_n} = \frac1N\sum_n\big(\text{LSE}(z_n) - z_{n,y_n}\big)$$

- It is computed with `log_softmax` (§2), so it's stable for any logits.
- It takes the **mean** over all tokens, so the loss scale doesn't depend on
  B or T and learning rates stay comparable.
- **Uniform logits give $L = \log V$.** Use this to sanity-check
  initialisation: step-0 loss should be ≈ log V (≈ 4.17 for V = 65).

### Backward: derivation to know by heart
For one position: $L = \text{LSE}(z) - z_y$. Using $\nabla\text{LSE} = \text{softmax}$:

$$\frac{\partial L}{\partial z_j} = p_j - \delta_{jy}\quad\Rightarrow\quad \boxed{\frac{\partial L}{\partial z} = \frac1N(p - \text{onehot}(y))}$$

Intuition: the target logit is pushed **up** by $1-p_y$ and every other logit
is pushed **down** by $p_j$, in proportion to how wrong the model is. A
confident correct prediction gives a gradient of about 0.

### Why fused (the Checkpoint A answer)
Chained route: $\partial L/\partial p_y = -1/p_y$, then `softmax_backward`.
It is the same algebra, but:
- **Stability:** when the model is badly wrong, $p_y$ underflows to 0
  (logit gap ≳ 745 in float64, ≳ 104 in float32). Then $-1/p_y = -\infty$
  and $\infty\cdot0 = $ **nan**. Before underflow, a huge × tiny product
  loses precision. The fused version never divides by $p$.
- **Speed and memory:** one subtract per row against building $g$, a dot
  product and a multiply. Only $p$ is cached.
- It fails exactly where the gradient matters most: on badly wrong predictions.

### Code notes
- Reshape `[B,T,V]` → `[N,V]` and `[B,T]` → `[N]`, then use fancy indexing
  `log_probs[np.arange(N), y]` to pick each row's target entry.
- The forward returns `probs` as the cache, so the backward needs no
  recomputation.
- The backward **copies** `probs` before the in-place `-= 1`. Otherwise it
  would corrupt the cache. A test checks this.

### How it's tested (`tests/test_loss.py`)
- Uniform logits give log V.
- A confident correct prediction gives loss ≈ 0 and grad ≈ 0.
- Finite differences on `[B,T,V]`.
- Fused equals chained on safe inputs.
- A demo where chained gives **nan** at logits `[1000, 0]` with target 1,
  while fused gives `[1, −1]`.
- Gradient rows sum to 0, and the cache isn't mutated.

### Not handled yet (later phases)
- `ignore_index`/padding mask. It isn't needed with fixed-length
  next-token batches.
- Label smoothing.

---

## Self-check questions
1. Why subtract the max rather than the mean or the min?
2. What happens to masked (`-inf`) scores in softmax, and when does masking produce `nan`?
3. Why is `log_softmax` not the same as `np.log(softmax(z))` numerically?
4. Derive $J = \text{diag}(p) - pp^\top$ from the quotient rule without notes.
5. Why do the rows of the softmax gradient sum to zero? Give a one-line argument.
6. Why does `softmax_backward` take `p` and not `z`?
7. What is GELU's gate, and how does it differ from ReLU's?
8. Show that $\sqrt{2/\pi}$ makes the tanh approximation's slope at 0 match $\Phi'(0)$.
9. Derive GELU'(x) from the tanh form. Why is the backward a plain element-wise multiply?
10. Why does GELU backward need the input, while softmax backward needs the output?
11. **(Checkpoint A)** Derive $\partial L/\partial z = p - 	ext{onehot}(y)$ on paper using $
abla	ext{LSE} = 	ext{softmax}$.
12. **(Checkpoint A)** Give two reasons the fused version beats chaining `-1/p_y` through softmax backward.
13. What should the step-0 loss be for V = 65, and what does it mean if it's 20?
14. Why mean over tokens rather than sum?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Softmax numeric stability | CS231n, Linear classification, *"Practical issues: Numeric stability"*: https://cs231n.github.io/linear-classify/#softmax-classifier |
| LSE trick; ∇LSE = softmax | Wikipedia, LogSumExp: https://en.wikipedia.org/wiki/LogSumExp |
| Softmax Jacobian derivation | Eli Bendersky, *The Softmax function and its derivative*: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ |
| Softmax + CE gradient | Bendersky (link above), section *"Softmax and cross-entropy loss"* |
| Jacobians and VJPs in backprop | Justin Johnson, *Derivatives, Backpropagation, and Vectorization* (CS231n): https://cs231n.stanford.edu/handouts/derivatives.pdf |
| GELU paper | Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*: https://arxiv.org/abs/1606.08415 (also in `papers/s1_hendrycks2016_gelu.pdf`) |
| Manual backprop, start to finish | Karpathy, *Building makemore Part 4: Becoming a Backprop Ninja*: https://www.youtube.com/watch?v=q8SA3rM6ckI |
