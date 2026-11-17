"""Parameter: a trainable array plus its accumulated gradient."""

import numpy as np


class Parameter:
    """data: the values. grad: dL/d(data), same shape, accumulated with +=."""

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad.fill(0.0)      # in place: keeps the same buffer

    def astype(self, dtype) -> None:
        self.data = self.data.astype(dtype)
        self.grad = self.grad.astype(dtype)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, dtype={self.data.dtype})"
