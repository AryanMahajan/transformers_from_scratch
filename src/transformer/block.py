"""Pre-LN Transformer block (GPT-2 style).

Forward:
    x1 = x  + MHA(LN1(x))        attention sub-layer + residual
    y  = x1 + FFN(LN2(x1))       feed-forward sub-layer + residual
Backward (residual: y = x + f(x)  =>  dx = dy + f'(x)^T dy):
    dx1 = dy  + LN2.backward(FFN.backward(dy))
    dx  = dx1 + LN1.backward(MHA.backward(dx1))
The bare "dy +" term is the gradient highway: it reaches x without passing
through any weight matrix, so gradients don't vanish in deep stacks.

Pre-LN (normalise the branch input) rather than post-LN (normalise after the
add) keeps the residual stream an untouched sum, which trains stably without
warmup. Derivations: docs/05_transformer.md §2
"""

import numpy as np

from src.attention.multi_head import MultiHeadAttention
from src.core import initialization as init
from src.core.module import Module
from src.nn.layer_norm import LayerNorm
from src.transformer.feed_forward import FeedForward


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator,
        d_ff: int | None = None,
        causal: bool = True,
        out_std: float = 0.02,                 # residual-writing layers (W_o, fc2)
        dtype=init.DEFAULT_DTYPE,
    ):
        self.ln1 = LayerNorm(d_model, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, n_heads, rng, causal=causal, out_std=out_std, dtype=dtype)
        self.ln2 = LayerNorm(d_model, dtype=dtype)
        self.ffn = FeedForward(d_model, rng, d_ff=d_ff, out_std=out_std, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x1 = x + self.attn(self.ln1(x))
        return x1 + self.ffn(self.ln2(x1))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx1 = dout + self.ln2.backward(self.ffn.backward(dout))   # highway + FFN branch
        return dx1 + self.ln1.backward(self.attn.backward(dx1))   # highway + attention branch
