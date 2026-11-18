"""Stochastic decoding: temperature, top-k and top-p (nucleus) sampling.

Greedy decoding takes argmax every step and falls into loops ("the the the").
Sampling draws the next token from a (reshaped) probability distribution:

    z        = logits[-1] / tau            temperature: tau<1 sharpens, tau>1 flattens
    z        = top_k(z, k)                 keep the k largest, others -> -inf
    z        = top_p(z, p)                 keep the smallest set with prob mass >= p
    probs    = softmax(z)                  masked tokens get exactly 0
    next     ~ Categorical(probs)          inverse-CDF with one uniform draw

tau -> 0 recovers greedy (handled exactly: temperature=0 means argmax).
Notes: docs/07_real_training.md §3
"""

import numpy as np

from src.core.backend import to_numpy
from src.nn.softmax import softmax


def filter_logits(logits, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None) -> np.ndarray:
    """Apply temperature, then top-k, then top-p to a 1-D logit vector [V].

    Returns float64 logits with removed tokens set to -inf (softmax -> 0).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 here (use 0 only via sample_next = greedy)")
    z = np.asarray(to_numpy(logits), dtype=np.float64).reshape(-1) / temperature
    V = z.shape[0]

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k < V:
            kth = np.partition(z, V - top_k)[V - top_k]       # k-th largest value
            z = np.where(z < kth, -np.inf, z)                  # ties at kth are kept

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        order = np.argsort(-z)                                 # most likely first
        p_sorted = softmax(z[order])
        mass_before = np.cumsum(p_sorted) - p_sorted           # prob mass strictly above each token
        drop = order[mass_before >= top_p]                     # already reached p without it
        z = z.copy()
        z[drop] = -np.inf                                      # the top token always survives
    return z


def sample_next(logits, rng: np.random.Generator, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> int:
    """Draw one token id from the last-position logits [V]."""
    if temperature == 0:
        return int(np.argmax(to_numpy(logits)))
    probs = softmax(filter_logits(logits, temperature, top_k, top_p))
    cdf = np.cumsum(probs)
    u = rng.random() * cdf[-1]                                 # scale guards against cdf[-1] = 1 - 1e-16
    return int(min(np.searchsorted(cdf, u, side="right"), len(cdf) - 1))


def generate(model, prompt_ids, n_new: int, rng: np.random.Generator,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None) -> np.ndarray:
    """Autoregressive sampling; same loop as greedy_generate, different picker."""
    ids = list(np.asarray(prompt_ids).reshape(-1))
    if not ids:
        raise ValueError("need at least one prompt token")
    max_len = model.cfg.max_len
    was_training = model.training
    model.eval()
    for _ in range(n_new):
        ctx = np.array(ids[-max_len:], dtype=np.int64)[None]       # [1, T<=max_len]
        logits = model(ctx)[0, -1]                                  # [V], may live on GPU
        ids.append(sample_next(logits, rng, temperature, top_k, top_p))
    model.train(was_training)
    return np.array(ids, dtype=np.int64)
