"""Sinusoidal positional encoding (Vaswani et al. 2017, §3.5).

Attention is permutation-equivariant: shuffle the tokens and the outputs
shuffle the same way. It has no idea of order. We fix that by adding a
position-dependent vector to every token embedding:

    omega_i      = base^(-2i/D)                  i = 0 .. D/2-1   (geometric series)
    PE[t, 2i]    = sin(t * omega_i)
    PE[t, 2i+1]  = cos(t * omega_i)

Each (sin, cos) pair is a point on the unit circle rotating at speed omega_i,
so  PE[t+k] = R(k) PE[t]  with R(k) block-diagonal 2x2 rotations that do NOT
depend on t -> relative offsets are a linear map, easy for attention to use.

No parameters: forward is x + PE[:T], backward is the identity (dx = dout).
Derivations: docs/04_attention.md §3
"""

import numpy as np

from src.core import initialization as init
from src.core.module import Module


def sinusoidal_encoding(max_len: int, dim: int, base: float = 10000.0, dtype=init.DEFAULT_DTYPE) -> np.ndarray:
    """Return the [max_len, dim] table. dim must be even (sin/cos pairs)."""
    if dim % 2 != 0:
        raise ValueError(f"dim={dim} must be even")
    t = np.arange(max_len, dtype=np.float64)[:, None]               # [T, 1]
    omega = base ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)  # [D/2]  = base^(-2i/D)
    angles = t * omega                                               # [T, D/2]  broadcast
    pe = np.empty((max_len, dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)                                     # even columns
    pe[:, 1::2] = np.cos(angles)                                     # odd columns
    return pe.astype(dtype)                                          # compute in f64, then cast


class SinusoidalPositionalEncoding(Module):
    """x [B, T, D] -> x + PE[:T]. A fixed buffer, not a Parameter."""

    def __init__(self, max_len: int, dim: int, base: float = 10000.0, dtype=init.DEFAULT_DTYPE):
        self.max_len = max_len
        self._pe64 = sinusoidal_encoding(max_len, dim, base, np.float64)  # exact master copy
        self.pe = self._pe64.astype(dtype)                           # plain array: not trained/saved

    def forward(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-2]
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        return x + self.pe[:T]                                       # broadcast over batch

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout                                                  # d(x + c)/dx = I

    def astype(self, dtype) -> "SinusoidalPositionalEncoding":
        self.pe = self._pe64.astype(dtype)   # re-cast from f64: f32 -> f64 can't recover lost digits
        return self
