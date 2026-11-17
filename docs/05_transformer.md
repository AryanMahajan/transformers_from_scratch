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

## 3. The full model: `src/transformer/model.py`

### Forward
```
ids [B,T]  (int)
h = TokenEmb(ids) + Pos[:T]      [B,T,D]   sinusoidal buffer or learned Embedding(max_len, D)
h = block_N(...block_1(h))       [B,T,D]
h = LN_f(h)                      [B,T,D]   pre-LN never normalises the stream, so do it once here
logits = h Eᵀ                    [B,T,V]   tied: the LM head IS the embedding matrix
```
Logits at position t score the token at t+1. The loss (Phase 1) is applied
outside the model: `cross_entropy(logits[:, :-1], ids[:, 1:])` on a
training window.

### Weight tying
Use the same $E\ [V,D]$ for the input lookup and the output projection.
- **Intuition:** "which vector represents token v" and "which direction
  predicts token v" should be similar. It also saves V·D parameters (Press &
  Wolf 2017; GPT-2 does it).
- **Gradient = sum of both uses** (multivariable chain rule):
  $$\frac{\partial L}{\partial E} = \underbrace{dlogits^\top h}_{\text{LM head: } [V,D]} + \underbrace{\text{scatter-add of } dh_0 \text{ rows}}_{\text{embedding lookup}}$$
  Our `Parameter.grad` accumulates with `+=`, so both contributions add
  automatically. Phase 2's design pays off here. **Assigning (`=`) in either
  place would silently drop one of them.**
- The test builds an untied twin with `lm_head.W = Eᵀ`. The two compute the
  same function, and the tied `E.grad` equals
  `emb.E.grad + lm_head.W.gradᵀ` from the twin.
- No `Parameter` is shared between modules here: the tied head is just a
  matmul in `GPT.forward` using `tok_emb.E`. So the registry needs no dedup
  for it (dedup still protects other sharing patterns).

### Backward
1. Tied head: `E.grad += dlogits2ᵀ @ h2`, then `dh = dlogits @ E`
   (untied: `lm_head.backward`).
2. `ln_f.backward`, then the blocks in **reverse** order.
3. Learned positions: `pos_emb.backward(dh summed over the batch)`. Row t is
   used by every sequence in the batch, so its gradients add.
4. `tok_emb.backward(dh)`: scatter-add into `E.grad`, the second use of E.
5. No input gradient: ids are discrete.

### Initialisation
- Embeddings and most weights: std 0.02.
- `W_o` and `fc2` in **every** block: `gpt2_residual_std(N)` =
  $0.02/\sqrt{2N}$. There are 2N residual writes, variances add, so this keeps
  the stream's variance independent of depth (Phase 1 §7). Tested.
- **Consequence: initial loss ≈ ln V.** Logits ≈ 0 means a uniform
  prediction. Measured: **4.214 vs ln 66 = 4.190.** This is the first sanity
  check of every training run (Phase 6). If the start is far above ln V, the
  init is wrong.

### Parameter counting
$$N_\text{params} = \underbrace{VD}_{\text{tok emb}} + \underbrace{[T_\max D]}_{\text{learned pos}} + N(12D^2 + 12D) + \underbrace{2D}_{\text{LN}_f} + \underbrace{[VD]}_{\text{untied head}}$$

Example: V = 66, D = 128, N = 4, sinusoidal, tied. Embeddings 8,448, blocks
792,576, LN_f 256, **total 801,280.** The embedding is ~1%, so tying barely
matters at char level. With BPE (V ≈ 50k) the embedding would dominate a model
this size, and tying matters a lot.

### Config (`GPTConfig`)
`vocab_size, d_model, n_heads, n_layers, max_len, d_ff, pos, tie_weights`, as
a dataclass with `to_dict()` (it will be saved in checkpoints in Phase 7). The
`pos` and `n_heads/n_layers` fields are Experiments D and A–C.

### Tests (`tests/test_model.py`)
- Logits shape; parameter count matches the formula for all 4
  {sinusoidal, learned} × {tied, untied} variants.
- **Full-model gradient check (exit criterion):** every parameter (33–35
  tensors), all variants, max error 6.4e-9 (weights at std 0.4, §2 gotcha).
- **Full-model causality (exit criterion):** for every t, changing tokens
  after t leaves logits ≤ t bit-identical and changes later ones.
- Tied gradient = sum of both uses (untied-twin test).
- Initial loss ≈ ln V.
- One step of gradient descent lowers the loss (forward, loss and backward
  all agree end to end).
- Residual std used for `W_o`/`fc2` in every block.
- Too-long input and unknown `pos` rejected.

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
12. Write the model's forward with shapes from ids to logits. Which position's logits predict which token?
13. Derive the tied-embedding gradient. What goes wrong if either use assigns `grad =` instead of `+=`?
14. Why should the initial loss be ≈ ln V? What would a much larger initial loss tell you?
15. Write the full parameter-count formula and evaluate it for V = 66, D = 128, N = 4.
16. Why does the learned positional embedding's gradient sum over the batch?
17. Why is the full-model gradient check done with weights at std 0.4 instead of 0.02?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Residual connections (the original idea) | He et al. 2015, *Deep Residual Learning for Image Recognition*: https://arxiv.org/abs/1512.03385 |
| Pre-LN vs post-LN analysis | Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture*: https://arxiv.org/abs/2002.04745 |
| Weight tying | Press & Wolf 2017, *Using the Output Embedding to Improve Language Models*: https://arxiv.org/abs/1608.05859 |
| GPT-2 (pre-LN decoder, residual init, tying) | Radford et al. 2019, *Language Models are Unsupervised Multitask Learners*: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| The original FFN definition (§3.3) | Vaswani et al. 2017: https://arxiv.org/abs/1706.03762 |
