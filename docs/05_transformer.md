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

## Self-check questions
1. Write the FFN forward and backward with shapes. Which forward value must be cached, and why?
2. Why does the FFN need a non-linearity? What does it collapse to without one?
3. Prove $\partial y_t/\partial x_s = 0$ for $s \ne t$. What does "position-wise" mean, and which layer *does* mix positions?
4. Count the FFN's parameters for D = 64. What fraction of a block's weights is that?
5. Why is fc2 (and not fc1) initialised with the residual std?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| The original FFN definition (§3.3) | Vaswani et al. 2017: https://arxiv.org/abs/1706.03762 |
