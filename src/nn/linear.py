"""Linear layer: y = x @ W + b   (row-vector convention, W is [D_in, D_out]).

Backward (N = product of leading dims):
    dx = dout @ W^T            [N, D_out] @ [D_out, D_in]
    dW = x^T @ dout            [D_in, N] @ [N, D_out]   (sums over B and T)
    db = sum_n dout            [D_out]
Derivations: docs/02_nn_foundations.md §2
"""

import numpy as np

from src.core import initialization as init
from src.core.module import Module
from src.core.parameter import Parameter


class Linear(Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rng: np.random.Generator,
        bias: bool = True,
        std: float = 0.02,
        dtype=init.DEFAULT_DTYPE,
    ):
        self.W = Parameter(init.normal(rng, (d_in, d_out), std, dtype))
        self.b = Parameter(init.zeros(d_out, dtype)) if bias else None
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x                                   # cache for dW (reference, no copy)
        d_in, d_out = self.W.shape
        # flatten leading dims -> ONE 2-D GEMM. (A 3-D `x @ W` makes NumPy loop over the
        # batch: measured 13x slower for [16,128,128] @ [128,512].) Same maths, same shapes.
        y = (x.reshape(-1, d_in) @ self.W.data).reshape(*x.shape[:-1], d_out)
        if self.b is not None:
            y = y + self.b.data
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        d_in, d_out = self.W.shape
        x2 = x.reshape(-1, d_in)                      # [N, D_in]
        d2 = dout.reshape(-1, d_out)                  # [N, D_out]

        self.W.grad += x2.T @ d2                      # [D_in, D_out]
        if self.b is not None:
            self.b.grad += d2.sum(axis=0)             # [D_out]
        return (d2 @ self.W.data.T).reshape(x.shape)  # [..., D_in]  (2-D GEMM, see forward)
