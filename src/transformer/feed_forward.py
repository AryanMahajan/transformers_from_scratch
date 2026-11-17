"""Position-wise feed-forward network (the "MLP" half of a Transformer block).

    x [B,T,D] -> h = x W1 + b1      [B,T,F]   expand   (F = d_ff, default 4D)
              -> a = GELU(h)        [B,T,F]   non-linearity
              -> y = a W2 + b2      [B,T,D]   project back
Backward is the same chain reversed:
    da = fc2.backward(dy);  dh = GELU'(h) * da;  dx = fc1.backward(dh)

"Position-wise": no step mixes tokens, so token t's output depends only on
token t's input. Attention moves information between positions; the FFN
processes it at each position.
Derivations: docs/05_transformer.md §1
"""

import numpy as np

from src.core import initialization as init
from src.core.module import Module
from src.nn.activations import gelu, gelu_backward
from src.nn.linear import Linear


class FeedForward(Module):
    def __init__(
        self,
        d_model: int,
        rng: np.random.Generator,
        d_ff: int | None = None,
        out_std: float = 0.02,                    # pass gpt2_residual_std(N) in the model
        dtype=init.DEFAULT_DTYPE,
    ):
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.fc1 = Linear(d_model, d_ff, rng, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, rng, std=out_std, dtype=dtype)
        self._h = None                            # pre-activation, needed by GELU'

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._h = self.fc1(x)
        return self.fc2(gelu(self._h))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        da = self.fc2.backward(dout)              # also fills fc2.W.grad, fc2.b.grad
        dh = gelu_backward(self._h, da)           # element-wise: diagonal Jacobian
        return self.fc1.backward(dh)              # also fills fc1.W.grad, fc1.b.grad
