"""Dataset preparation helpers.

Notes: docs/03_tokenization.md §2
"""


def train_val_split(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Contiguous split: first (1 - val_fraction) is train, the rest is val.

    Contiguous (not random chars/lines) so val windows never overlap train
    text; only <= T chars of context leak at the single seam.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    cut = int(len(text) * (1.0 - val_fraction))
    return text[:cut], text[cut:]
