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

## Self-check questions
1. Why must an LM's attention be causal? What happens to training loss without the mask?
2. Why can't a causal softmax row produce `nan`? When could a mask cause one?
3. Derive Var(q·k) = d. What does that do to softmax and its gradients without scaling?
4. Write the three backward steps of attention with shapes. Why is dK = dSᵀQ/√d and not dS·Q?
5. Why do masked positions get zero gradient without any special code?
6. What in attention costs O(T²) memory?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| The original paper (§3.2) | Vaswani et al. 2017, *Attention Is All You Need*: https://arxiv.org/abs/1706.03762 (also `papers/02_vaswani2017_attention_is_all_you_need.pdf`) |
| Visual walkthrough of Q/K/V and multi-head | Jay Alammar, *The Illustrated Transformer*: https://jalammar.github.io/illustrated-transformer/ |
| Building causal self-attention in code | Karpathy, *Let's build GPT: from scratch, in code, spelled out*: https://www.youtube.com/watch?v=kCc8FmEb1nY |
