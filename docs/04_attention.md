# Phase 4: Attention (study notes)

Revision notes for Phase 4, the core of the Transformer. Attention is the
**only** place where positions exchange information. Every other layer acts
on each token independently.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

**Notation:** `Q, K, V : [..., T, d]` (inside MHA: `[B, H, T, D_h]`),
`S, A : [..., T, T]`.

---

## 1. Causal mask and scaled dot-product attention
`src/attention/causal_mask.py`, `src/attention/scaled_attention.py`

### Forward

$$S = \frac{QK^\top}{\sqrt d},\qquad A = \text{softmax}_{\text{row}}(S + M),\qquad O = AV$$

- $S_{ij}$: how well query *i* matches key *j*.
- Row *i* of $A$: a distribution over the positions *i* reads from.
- $O_i = \sum_j A_{ij}V_j$: a weighted average of values.

### Causal mask
$M_{ij} = -\infty$ for $j > i$, else 0. So $A_{ij} = 0$ **exactly** for
future keys.
- **Why:** an LM predicts token t+1 from ≤ t. Seeing t+1 means copying the
  answer: the train loss → ~0 and generation is garbage.
- **No nan:** the diagonal is never masked, so every row has a finite max.
  A fully masked row would give `-inf − (-inf) = nan` (Phase 1, Q2).
- We use a real `-inf`, not `-1e9`, so the weights are exactly 0 (exit
  criterion; the test uses `== 0.0`).
- Implementation: boolean `np.tril` and `np.where(mask, S, -inf)`.

### Why √d (Checkpoint C)
If $q_i, k_i$ are i.i.d. with mean 0 and variance 1:
$\text{Var}(q\cdot k) = \sum_{i=1}^d E[q_i^2]E[k_i^2] = d$.
The raw scores have std √d, so softmax is ~one-hot, the Jacobian
$p_i(\delta_{ij}-p_j)\approx 0$, and **gradients vanish**. Dividing by √d
restores variance 1. It's the same idea as initialisation.

**Measured** (d = 256, T = 64):

| | score std | mean max prob | mean p(1−p) (Jacobian diagonal) |
|---|---|---|---|
| unscaled | 15.9 (≈ √256 = 16) | **0.86** (≈ one-hot) | 0.0029 |
| scaled by 1/√d | 0.99 | 0.10 | 0.0150 (**≈ 5× larger**) |

### Backward

| Step | Forward | Backward |
|---|---|---|
| ① | $O = AV$ | $dA = dO\,V^\top$, $dV = A^\top dO$ |
| ② | $A = \text{softmax}(S+M)$ | $dS = A\odot(dA - \text{rowsum}(dA\odot A))$, Phase 1's `softmax_backward` |
| ③ | $S = QK^\top/\sqrt d$ | $dQ = dS\,K/\sqrt d$, $dK = dS^\top Q/\sqrt d$ |

- **Mask in backward:** masked entries are constants, so they get 0
  gradient, and that happens **for free**. $A_{ij}=0$ there, and
  `softmax_backward` multiplies by $A$.
- **Shapes:** $dK$ must be `[T,d]`. Only `dSᵀ·Q` fits ($K$ enters $S$
  transposed).
- The `1/√d` is folded into `dS` once, rather than applied to both dQ and dK.
- **Cache:** Q, K, V and A. **A is `[B,H,T,T]`**, which is the O(T²) memory
  cost of attention.

### Code notes
- `_t(x) = swapaxes(x, -1, -2)` is a batched transpose, so the same code
  works for `[T,d]`, `[B,T,d]` and `[B,H,T,d]`.
- These are functions returning `(out, cache)` (like `loss.py`), not a
  Module. MHA (§2) owns the parameters and calls them.
- `causal=False` is supported (bidirectional, BERT-style) and tested, which
  is useful for comparisons.

### Tests (`tests/test_attention.py`)
- Mask layout.
- Forward equals a naive per-position loop.
- Future weights `== 0.0`, rows sum to 1, all finite.
- **Gradient check of dQ, dK, dV** (causal and non-causal) < 1e-8.
- **Causality:** perturbing positions > t in Q, K or V leaves outputs ≤ t
  bit-identical.
- Position 0 outputs exactly V₀.
- The √d measurements above.

---

## 2. Multi-head attention: `src/attention/multi_head.py` (Checkpoint C)

### Every shape
```
x                          [B, T, D]
q,k,v = Linear(x)          [B, T, D]
split: reshape             [B, T, H, D_h]   cut each token's features into H chunks
       transpose(0,2,1,3)  [B, H, T, D_h]   heads next to batch; T is attention's axis
attention per head         [B, H, T, D_h]   scale 1/√D_h
merge: transpose(0,2,1,3)  [B, T, H, D_h]
       reshape             [B, T, D]        concatenate heads per token
out = Linear_o(merged)     [B, T, D]        mix across heads
```

### Why transpose, not just reshape
`reshape` never moves data. `[B,T,D]` memory is token-major, so
`reshape(B,H,T,D_h)` cuts it into H consecutive **blocks of positions**:
"head 1" would only contain tokens T/2..T−1. Reshape to `[B,T,H,D_h]` first
(this splits within each token), then transpose. The test shows the naive
head 1 holding only tokens 2 and 3.

### Why several heads
One head gives one distribution, so one weighted average per position. H
heads let a position look in H places for H reasons, which `W_o` then
combines.
- **Same parameters:** the projections are D×D whatever H is.
- **Same attention FLOPs:** $H\cdot T^2\cdot D_h = T^2 D$.
- **Trade-off:** each head lives in a $D_h = D/H$ subspace. Experiment C
  (heads) tests this.

### Backward = forward chain in reverse
1. `o_proj.backward` → `d_merged [B,T,D]`
2. split (undoes merge) → `[B,H,T,D_h]`
3. `attention_backward` → dq, dk, dv
4. merge each (undoes split) → `[B,T,D]`
5. `dx = q_proj.backward(dq) + k_proj.backward(dk) + v_proj.backward(dv)`

- **Reshape/transpose backward = the inverse rearrangement.** Their Jacobian
  is a permutation $P$, and $P^\top = P^{-1}$.
- **Why dx is a sum:** x feeds three projections. It's the multivariable
  chain rule again.

### Finding: the key bias is useless (so we removed it)
The first gradient check failed with rel. err **1.0** on `k_proj.b`. The
analytic gradient was ~1e-20 and the numeric gradient exactly 0, so the
relative error of two zeros was meaningless. **The gradient really is 0:**

$$q_i\cdot(k_j+b) = q_i\cdot k_j + \underbrace{q_i\cdot b}_{\text{same for all } j}$$

That shifts each score row by a constant, which softmax ignores. The key
bias never affects the output and would never train. The **query** bias is
different: $(q_i+b)\cdot k_j$ varies with j. **Decision:** `k_proj` has no
bias; q, v and o keep theirs. A test proves that adding any vector to all
keys leaves the output unchanged.

Lesson for gradient checking: a relative error of ~1 with **both** gradients
≈ 0 means "zero gradient", not necessarily "bug". Look at the magnitudes.

### Code notes
- `out_std` lets the model initialise `W_o` with `gpt2_residual_std(N)` (Phase 1 §7).
- `attn_weights` keeps `A [B,H,T,T]` from the last forward, for Phase 8's
  attention visualisation.
- Raises if `D % H != 0`.

### Tests (`tests/test_multi_head.py`)
- split/merge are inverses, and head j = feature slice j.
- Shapes, and the parameter count `4D² + 3D` (independent of H).
- Equals an explicit per-head loop.
- **Gradient check** of input + 7 parameters (all < 4e-9), causal and not.
- Causality through the full MHA, with future weights `== 0`.
- H = 1 equals plain attention.
- The naive-reshape demo.
- Key-bias invariance.
- Indivisible H is rejected.

---

## 3. Sinusoidal positional encoding: `src/nn/positional.py`

### Why we need positions at all
Attention only sees a **set**: no step in §1–§2 uses the index t. Shuffle the
input tokens and the outputs shuffle the same way (*permutation
equivariance*). "dog bites man" and "man bites dog" would look identical.
- The causal mask leaks *some* order (position t sees exactly t+1 tokens),
  but that's weak and indirect. Real models add explicit positions.
- Fix: `h_0 = TokenEmbedding(ids) + PE[:T]`. Each token is tagged with "where".

### Formula (Vaswani §3.5)

$$\omega_i = 10000^{-2i/D},\qquad PE_{t,2i} = \sin(t\,\omega_i),\qquad PE_{t,2i+1} = \cos(t\,\omega_i),\qquad i = 0..\tfrac D2-1$$

- **Geometric frequencies:** consecutive ratios are $10000^{-2/D}$. Periods
  run from $2\pi$ (pair 0, changes every token) to $2\pi\cdot 10000^{(D-2)/D}$
  (≈ 47,000 for D = 64: changes slowly). Like a clock or binary counter: fast
  hands for fine position, slow hands for coarse position.
- **Bounded:** values in [−1, 1], so they don't swamp embeddings. Every row
  has norm $\sqrt{D/2}$ (each pair has sin²+cos² = 1).
- **Unique:** no two positions (within a sane max_len) share a vector.

### The key property: shifts are rotations
Each pair $(\sin t\omega, \cos t\omega)$ is a point on the unit circle. By the
angle-addition formulas:

$$\begin{pmatrix}\sin (t+k)\omega\ \cos (t+k)\omega\end{pmatrix} =
\begin{pmatrix}\cos k\omega & \sin k\omega\ -\sin k\omega & \cos k\omega\end{pmatrix}
\begin{pmatrix}\sin t\omega\ \cos t\omega\end{pmatrix}$$

So $PE_{t+k} = R(k)\,PE_t$ with $R(k)$ block-diagonal and **independent of t**.
"k tokens back" is one fixed linear map, which a learned $W_Q, W_K$ can exploit.
Consequence: $PE_t\cdot PE_{t+k} = \sum_i \cos(k\omega_i)$ depends only on the
offset k, and falls off with |k|.

**Measured** (D = 64): similarity at offset 0, 1, 2, 5, 20, 100 =
32.0, 30.9, 28.3, 23.5, 18.9, 17.9.

### Backward
`out = x + PE` and PE is a constant, so `dx = dout` exactly. No parameters, no
gradient to store. It's the cheapest layer in the model.

### Sinusoidal vs learned (Experiment D)
| | Sinusoidal | Learned (`Embedding(max_len, D)` on `arange(T)`) |
|---|---|---|
| Parameters | 0 | max_len·D |
| Longer sequences than trained | formula defined (quality not guaranteed) | impossible (no row) |
| Flexibility | fixed | learns whatever helps |
GPT-2 uses learned; the original Transformer used sinusoidal and reported
nearly identical results. Modern LLMs mostly use RoPE, which applies this same
rotation idea *inside* attention to q and k.

### Gotchas
- **D must be even** (sin/cos pairs). Raises otherwise.
- **Interleaved vs concatenated:** we interleave (sin at even columns, cos at
  odd), like the paper. Some code uses `[sin…, cos…]` halves. Both are fine,
  but you can't mix tables between conventions.
- **Precision bug caught by the tests:** the first version built the table in
  float32 and then `.astype(float64)` for grad checks. Up-casting can't
  recover lost digits (error ~3e-8). Fix: keep a float64 master and cast
  *down* from it.
- The table is a plain array, not a `Parameter`: not trained, not in
  `state_dict` (it's recomputed from the formula).
- `T > max_len` raises.

### Tests (`tests/test_positional.py`)
- Matches the formula element-wise; PE[0] = (0,1,0,1,…); bounded; row norm √(D/2).
- Frequencies are geometric (ratio $10000^{-2/D}$).
- **Shift = fixed rotation:** `PE[k:] == PE[:T] @ R(k)ᵀ` for all t.
- Dot product depends only on the offset, decreasing nearby.
- 512 positions all distinct.
- Module: adds the table, 0 parameters, backward is identity, grad check < 1e-8.
- **Order blindness:** non-causal MHA on permuted input = permuted output; with
  PE added, this breaks.

---

## Self-check questions
1. Why must an LM's attention be causal? What happens to training loss without the mask?
2. Why can't a causal softmax row produce `nan`? When could a mask cause one?
3. Derive Var(q·k) = d. What does that do to softmax and its gradients without scaling?
4. Write the three backward steps of attention with shapes. Why is dK = dSᵀQ/√d and not dS·Q?
5. Why do masked positions get zero gradient without any special code?
6. What in attention costs O(T²) memory?
7. **(Checkpoint C)** Walk through every reshape and transpose in MHA with shapes, forward and backward.
8. Why does `reshape(B,H,T,D_h)` without a transpose give wrong heads?
9. Why do multiple heads cost no extra parameters or attention FLOPs?
10. Why is the backward of a reshape/transpose just the inverse rearrangement?
11. Prove the key bias has zero gradient. Why doesn't the same argument apply to the query bias?
12. Show that attention without positions is permutation-equivariant. Why is that bad for language?
13. Write the sinusoidal formula. Why geometric frequencies, and what are the shortest and longest periods?
14. Derive $PE_{t+k} = R(k)PE_t$. Why does it matter that R(k) doesn't depend on t?
15. Why is the backward of positional encoding the identity? How many parameters does it have?
16. Sinusoidal vs learned positions: give one advantage of each.

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| The original paper (§3.2) | Vaswani et al. 2017, *Attention Is All You Need*: https://arxiv.org/abs/1706.03762 (also `papers/02_vaswani2017_attention_is_all_you_need.pdf`) |
| Visual walkthrough of Q/K/V and multi-head | Jay Alammar, *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/ |
| Positional encoding intuition and the rotation proof | Amirhossein Kazemnejad, *Transformer Architecture: The Positional Encoding*: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ |
| Building causal self-attention in code | Karpathy, *Let's build GPT: from scratch, in code, spelled out*: https://www.youtube.com/watch?v=kCc8FmEb1nY |
