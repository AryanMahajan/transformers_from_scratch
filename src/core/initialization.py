"""Weight initialisation. Every function takes an explicit np.random.Generator.

Weights follow the row-vector convention W in R^{fan_in x fan_out}.
Derivations: docs/01_numerical_foundations.md §7
"""

import numpy as np

DEFAULT_DTYPE = np.float32


def normal(rng: np.random.Generator, shape, std: float = 0.02, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """N(0, std^2). GPT-2 default std = 0.02."""
    return (rng.standard_normal(shape) * std).astype(dtype)


def xavier_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """Glorot normal: Var = 2 / (fan_in + fan_out)."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(rng, (fan_in, fan_out), std, dtype)


def xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """Glorot uniform: U(-a, a), a = sqrt(6 / (fan_in + fan_out)) so Var = a^2/3."""
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out)).astype(dtype)


def he_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """He/Kaiming normal for ReLU-like activations: Var = 2 / fan_in."""
    std = np.sqrt(2.0 / fan_in)
    return normal(rng, (fan_in, fan_out), std, dtype)


def gpt2_residual_std(n_layers: int, base_std: float = 0.02) -> float:
    """Std for residual output projections (attn W_o, FFN W_2): base / sqrt(2N).

    Each block adds 2 branches to the residual stream; variances add, so
    scaling each by 1/sqrt(2N) keeps the stream's variance bounded in depth.
    """
    return base_std / np.sqrt(2 * n_layers)


def zeros(shape, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """Biases, LayerNorm beta."""
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=DEFAULT_DTYPE) -> np.ndarray:
    """LayerNorm gamma."""
    return np.ones(shape, dtype=dtype)
