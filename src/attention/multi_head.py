"""Multi-head (causal) self-attention.

    x [B,T,D] -> q,k,v = Linear(x)            [B,T,D]
              -> split_heads                  [B,H,T,D_h]   reshape then transpose
              -> scaled dot-product attention [B,H,T,D_h]
              -> merge_heads                  [B,T,D]       transpose then reshape
              -> W_o                          [B,T,D]
Backward runs the same chain in reverse; split/merge are each other's
backward (permutations: P^T = P^-1). x feeds q, k, v, so dx is the sum of
three contributions.

No key bias: q_i . (k_j + b) = q_i . k_j + q_i . b shifts every score in row i
by the same constant, and softmax is shift-invariant -> the key bias has no
effect and exactly zero gradient. (q, v, o keep their biases.)
Derivations: docs/04_attention.md §2
"""

import numpy as np

from src.attention.scaled_attention import attention_backward, attention_forward
from src.core import initialization as init
from src.core.module import Module
from src.nn.linear import Linear


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """[B, T, D] -> [B, H, T, D_h]. Cut each token's features, THEN move H forward."""
    B, T, D = x.shape
    return x.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """[B, H, T, D_h] -> [B, T, D]. Exact inverse of split_heads."""
    B, H, T, Dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator,
        causal: bool = True,
        bias: bool = True,
        out_std: float = 0.02,                     # pass gpt2_residual_std(N) in the model
        dtype=init.DEFAULT_DTYPE,
    ):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.causal = causal
        self.q_proj = Linear(d_model, d_model, rng, bias=bias, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, rng, bias=False, dtype=dtype)  # see docstring
        self.v_proj = Linear(d_model, d_model, rng, bias=bias, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, rng, bias=bias, std=out_std, dtype=dtype)
        self._cache = None
        self.attn_weights = None                   # [B, H, T, T], kept for visualisation

    def forward(self, x: np.ndarray) -> np.ndarray:
        H = self.n_heads
        q = split_heads(self.q_proj(x), H)
        k = split_heads(self.k_proj(x), H)
        v = split_heads(self.v_proj(x), H)
        out, self._cache = attention_forward(q, k, v, self.causal)
        self.attn_weights = self._cache["A"]
        return self.o_proj(merge_heads(out))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        H = self.n_heads
        d_heads = split_heads(self.o_proj.backward(dout), H)          # undo merge
        dq, dk, dv = attention_backward(d_heads, self._cache)
        dx = self.q_proj.backward(merge_heads(dq))                    # undo split
        dx = dx + self.k_proj.backward(merge_heads(dk))               # x used 3 times:
        dx = dx + self.v_proj.backward(merge_heads(dv))               # contributions add
        return dx
