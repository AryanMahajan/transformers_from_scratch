"""Causal (lower-triangular) attention mask."""

import numpy as np


def causal_mask(T: int) -> np.ndarray:
    """Boolean [T, T]: True where query i MAY attend to key j (j <= i).

    The diagonal is always True, so no softmax row is entirely masked
    (which would give max = -inf and -inf - (-inf) = nan).
    """
    return np.tril(np.ones((T, T), dtype=bool))
