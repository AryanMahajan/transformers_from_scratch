# Phase 5: The Transformer model (study notes)

Revision notes for Phase 5. This phase assembles the parts from Phases 2–4
into a full decoder-only language model.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

**Notation:** `D` = d_model, `F` = d_ff (default 4D), `x : [B, T, D]`.

---

## 1. Feed-forward network: `src/transformer/feed_forward.py`

### Where it sits
Each block has two halves: **attention** (tokens exchange information) and
the **FFN** (each token processes what it gathered). The FFN never mixes
positions.

### Forward

$$h = xW_1 + b_1\ [B,T,F],\qquad a = \text{GELU}(h),\qquad y = aW_2 + b_2\ [B,T,D]$$

- **Expand then contract:** D → 4D → D. The wide hidden layer gives room for
  many features (it's often read as a key-value memory: rows of $W_1$ detect
  patterns, rows of $W_2$ write the corresponding output).
- **The non-linearity is essential:** without GELU,
  $xW_1W_2 = x(W_1W_2)$ collapses to one D×D linear map. Attention's
  only non-linearity is the softmax over *positions*, so the FFN is where most
  feature-wise non-linear computation happens.

### Backward (chain in reverse)
| Step | Forward | Backward |
|---|---|---|
| ① | $y = aW_2 + b_2$ | `fc2.backward`: $da = dy\,W_2^\top$, fills $dW_2 = a^\top dy$, $db_2$ |
| ② | $a = \text{GELU}(h)$ | $dh = da \odot \text{GELU}'(h)$ (element-wise, diagonal Jacobian) |
| ③ | $h = xW_1 + b_1$ | `fc1.backward`: $dx = dh\,W_1^\top$, fills $dW_1, db_1$ |

- Needs the **pre-activation h** cached (`gelu_backward` takes the input, not
  the output). Linear caches its own inputs.
- No new math here. It's composition: Phase 1 (GELU) + Phase 2 (Linear).

### Position-wise
Treat `[B,T]` as one big batch of B·T independent vectors. So
$\partial y_t / \partial x_s = 0$ for $s \ne t$. The test perturbs token 3
and checks that only output 3 changes.

### Size and cost
- Parameters: $2DF + F + D = 8D^2 + 5D$ (F = 4D).
- Attention has $4D^2$ weights, so **the FFN holds ⅔ of each block's
  weights** and most of its matmul FLOPs (for T ≪ D).
- `out_std` lets the model initialise $W_2$ with `gpt2_residual_std(N)`
  (Phase 1 §7), since fc2 writes into the residual stream (like MHA's `W_o`).

### Tests (`tests/test_feed_forward.py`)
- Shapes, d_ff = 4D default, parameter count `8D² + 5D`.
- Equals `gelu(x@W1+b1)@W2+b2`.
- **Gradient check** of input + 4 parameters: all < 3e-10.
- Position-wise: perturbing token 3 changes only output 3; per-token = batched.
- `out_std` affects fc2 only.
- FFN weights = 8D², MHA weights = 4D².

---

## 2. Pre-LN Transformer block: `src/transformer/block.py`

### Forward

$$x_1 = x + \text{MHA}(\text{LN}_1(x)),\qquad y = x_1 + \text{FFN}(\text{LN}_2(x_1))$$

- The **residual stream** `x` runs straight through. Each sub-layer *reads* a
  normalised copy and *adds* its result back ("writes to the stream").
- Shape `[B,T,D]` in and out, so blocks stack.
- Parameters: MHA $4D^2+3D$ + FFN $8D^2+5D$ + 2 LNs $4D$ = **$12D^2 + 12D$**.

### Backward of a residual connection
$y = x + f(x) \Rightarrow \frac{\partial y}{\partial x} = I + J_f$, so

$$dx = dy + J_f^\top dy$$

The first term passes the gradient through **unchanged**: no weights, no
activations. Applied to the block:
```
dx1 = dy  + LN2.backward(FFN.backward(dy))
dx  = dx1 + LN1.backward(MHA.backward(dx1))
```
Over N blocks, $\partial y_N/\partial x_0 = \prod (I + J_i)$. Expanding it
always includes the identity term, so the gradient has a direct path to every
layer (**gradient highway**). Without residuals it's $\prod J_i$, a product of
N Jacobians that tends to vanish or explode.

**Measured** (12 blocks, D = 64, GPT-2 init), $\|dx_0\| / \|dy\|$:

| | ratio at the input |
|---|---|
| with residuals | **1.008** |
| without residuals | **0.018** (≈ 55× smaller) |

Subtlety: without residuals the ratio inside the stack looks fine
(0.4–0.9), because LayerNorm's backward multiplies by $1/\sigma$ of its input,
which cancels the shrinking (each layer's output has tiny σ ≈ 0.02-ish). At the
first layer the input has σ = 1, nothing cancels, and the **embeddings get
starved**. Residuals make every layer's gradient healthy for free.

### Identity at init
If $W_o$ and $W_2$ output zero, the block is exactly the identity and
backward is `dx = dy` (tested). With std 0.02 (or the smaller
`gpt2_residual_std`) the branches start small, so a deep model starts near
the identity and each block learns a small correction. This is why the
residual-writing layers get the smaller std.

### Pre-LN vs post-LN
| | Post-LN (original 2017) | Pre-LN (GPT-2 onward, ours) |
|---|---|---|
| Formula | $\text{LN}(x + f(x))$ | $x + f(\text{LN}(x))$ |
| Residual path | goes through LN every block | untouched sum |
| Gradient at init | large near the output; needs **LR warmup** | well-behaved; trains without warmup |
| Extra | — | needs a **final LN** before the LM head (stream is never normalised) |

(Xiong et al. 2020 analyse this.) Pre-LN keeps the highway clean: the
`dy +` term really is the identity.

### Gotcha: gradient checks at small init
The first block gradient check failed: rel. err **6.7e-6** on
`attn.q_proj.b`. The gradient there was only ~2e-4 (it flows through
several std-0.02 matrices), so finite-difference noise was a large *relative*
error. Scaling the weights ×20 (std 0.4) made all gradients O(1) and the error
fell to 7e-10. **Backward formulas don't depend on the weight values, so check
at a scale where gradients are O(1).** Same lesson as the key bias: look at the
magnitudes before assuming a bug.

### Tests (`tests/test_block.py`)
- Shape, parameter count `12D² + 12D`, sub-module names.
- Equals manual composition.
- **Gradient check** of input + 15 parameters (at std 0.4): < 1e-8, causal and not.
- Causality through the block.
- Zeroed branches → exact identity, forward and backward.
- **Gradient highway:** 12 blocks, ratio in (0.5, 2) with residuals, < 0.1 without.

---

## Self-check questions
1. Write the FFN forward and backward with shapes. Which forward value must be cached, and why?
2. Why does the FFN need a non-linearity? What does it collapse to without one?
3. Prove $\partial y_t/\partial x_s = 0$ for $s \ne t$. What does "position-wise" mean, and which layer *does* mix positions?
4. Count the FFN's parameters for D = 64. What fraction of a block's weights is that?
5. Why is fc2 (and not fc1) initialised with the residual std?
6. Write the pre-LN block forward and backward. Where does the "gradient highway" appear?
7. Expand $\prod_i (I + J_i)$ for two blocks. Why does it never vanish the way $\prod_i J_i$ can?
8. Without residuals, why do the middle layers' gradients look fine but the embeddings get starved?
9. Pre-LN vs post-LN: write both formulas. Why does pre-LN need a final LayerNorm, and why does post-LN need warmup?
10. Count a block's parameters for D = 64.
11. A grad check gives 6.7e-6 on one parameter whose gradient is ~2e-4. Bug or not? How do you tell?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Residual connections (the original idea) | He et al. 2015, *Deep Residual Learning for Image Recognition*: https://arxiv.org/abs/1512.03385 |
| Pre-LN vs post-LN analysis | Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture*: https://arxiv.org/abs/2002.04745 |
| The original FFN definition (§3.3) | Vaswani et al. 2017: https://arxiv.org/abs/1706.03762 |
