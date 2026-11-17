"""Embedding: out[b, t] = E[ids[b, t]]   ==   onehot(ids) @ E.

Backward is a scatter-add:  dE[v] = sum of dout[n] over positions with ids[n] == v.
Uses np.add.at because `E.grad[ids] += dout` silently drops repeated ids.
No input gradient (ids are discrete); backward returns None.
Derivations: docs/02_nn_foundations.md §3
"""

import numpy as np

from src.core import initialization as init
from src.core.module import Module
from src.core.backend import like, scatter_add
from src.core.parameter import Parameter


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        rng: np.random.Generator,
        std: float = 0.02,
        dtype=init.DEFAULT_DTYPE,
    ):
        self.E = Parameter(init.normal(rng, (num_embeddings, dim), std, dtype))
        self._ids = None

    def forward(self, ids: np.ndarray) -> np.ndarray:
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"ids must be integers, got {ids.dtype}")
        V = self.E.shape[0]
        if ids.size and (ids.min() < 0 or ids.max() >= V):   # E[-1] would silently wrap
            raise IndexError(f"ids out of range [0, {V}): min={ids.min()}, max={ids.max()}")
        ids = like(ids, self.E.data)                         # ids may arrive on the CPU
        self._ids = ids
        return self.E.data[ids]                              # [..., D]

    def backward(self, dout: np.ndarray) -> None:
        D = self.E.shape[1]
        scatter_add(self.E.grad, self._ids.reshape(-1), dout.reshape(-1, D))   # np.add.at / cupy add.at
        return None
