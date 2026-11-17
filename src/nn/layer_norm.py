"""Layer normalisation over the last axis (Ba et al. 2016).

Forward:  x_hat = (x - mean) / sqrt(var + eps),   y = gamma * x_hat + beta
Backward: g_hat = dout * gamma
          dx    = rstd * (g_hat - mean(g_hat) - x_hat * mean(g_hat * x_hat))
          dgamma = sum_n dout * x_hat,   dbeta = sum_n dout
Derivations: docs/02_nn_foundations.md §4
"""

import numpy as np

from src.core import initialization as init
from src.core.module import Module
from src.core.parameter import Parameter


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype=init.DEFAULT_DTYPE):
        self.gamma = Parameter(init.ones(dim, dtype))
        self.beta = Parameter(init.zeros(dim, dtype))
        self.eps = eps
        self._x_hat = None
        self._rstd = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)   # two-pass: no E[x^2]-mu^2 cancellation
        rstd = 1.0 / np.sqrt(var + self.eps)           # 1/sigma, [..., 1]
        x_hat = xc * rstd
        self._x_hat, self._rstd = x_hat, rstd
        return self.gamma.data * x_hat + self.beta.data

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_hat, rstd = self._x_hat, self._rstd
        D = x_hat.shape[-1]

        self.gamma.grad += (dout * x_hat).reshape(-1, D).sum(axis=0)
        self.beta.grad += dout.reshape(-1, D).sum(axis=0)

        g = dout * self.gamma.data                      # dL/dx_hat
        g_mean = g.mean(axis=-1, keepdims=True)
        gx_mean = (g * x_hat).mean(axis=-1, keepdims=True)
        return rstd * (g - g_mean - x_hat * gx_mean)
