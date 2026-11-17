"""Dataset preparation helpers.

Notes: docs/03_tokenization.md §2, docs/06_training.md §3
"""

import numpy as np


def train_val_split(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Contiguous split: first (1 - val_fraction) is train, the rest is val.

    Contiguous (not random chars/lines) so val windows never overlap train
    text; only <= T chars of context leak at the single seam.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    cut = int(len(text) * (1.0 - val_fraction))
    return text[:cut], text[cut:]


def get_batch(data, batch_size: int, block_size: int, rng):
    """Sample B random windows from a 1-D token array.

        x = data[i : i+T]        inputs        [B, T]
        y = data[i+1 : i+T+1]    targets       [B, T]   (x shifted left by one)
    Position t of x predicts y[t] = the next token, so ONE window gives T
    training examples (contexts of length 1..T) -- the causal mask makes that
    legal. Starts i are uniform over every valid window: 0 .. len-T-1.
    Notes: docs/06_training.md §3
    """
    n = len(data)
    if n <= block_size:
        raise ValueError(f"need more than block_size={block_size} tokens, got {n}")
    starts = rng.integers(0, n - block_size, size=batch_size)       # high is exclusive
    offsets = np.arange(block_size)
    idx = starts[:, None] + offsets[None, :]                        # [B, T] gather indices
    return data[idx], data[idx + 1]
