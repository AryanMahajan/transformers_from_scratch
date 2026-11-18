"""Greedy autoregressive decoding (always take the most likely next token).

    for each new token:
        ctx    = ids[-max_len:]          crop: the model has no position beyond max_len
        logits = model(ctx[None])        [1, T, V]
        next   = argmax(logits[0, -1])   only the LAST position predicts the unseen token
        ids.append(next)
Deterministic, so it's the right decoder for checking memorisation. Sampling
with temperature / top-k / top-p: src/generation/sampling.py.
Notes: docs/06_training.md §4
"""

import numpy as np


def greedy_generate(model, prompt_ids, n_new: int) -> np.ndarray:
    ids = list(np.asarray(prompt_ids).reshape(-1))
    if not ids:
        raise ValueError("need at least one prompt token")
    max_len = model.cfg.max_len
    was_training = model.training
    model.eval()
    for _ in range(n_new):
        ctx = np.array(ids[-max_len:], dtype=np.int64)[None]       # [1, T<=max_len]
        logits = model(ctx)
        ids.append(int(np.argmax(logits[0, -1])))
    model.train(was_training)
    return np.array(ids, dtype=np.int64)
