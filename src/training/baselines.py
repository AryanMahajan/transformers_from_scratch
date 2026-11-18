"""Count-based n-gram baselines and loss-unit conversions.

Fit on TRAIN, evaluate cross-entropy on VAL (the same metric as the model):
    unigram:  p(c)            = (count(c) + a) / (N + aV)
    bigram:   p(c | prev)     = (count(prev, c) + a) / (count(prev) + aV)
Add-a (Laplace/Lidstone) smoothing keeps unseen events from getting p = 0
(-> infinite loss). A bigram is "one character of context": the level a
model reaches with no attention at all. A Transformer must beat it.

Units:  bits per char = nats / ln 2  (/ chars-per-token for BPE);   perplexity = e^nats
        (perplexity = the effective number of equally likely next chars)
Notes: docs/07_real_training.md §2
"""

import math

import numpy as np


def bpc(loss_nats: float, chars_per_token: float = 1.0) -> float:
    """Bits per CHARACTER. For a tokenizer whose tokens average c chars,
    loss per token / c = loss per char, so different tokenizers compare fairly."""
    return loss_nats / math.log(2) / chars_per_token


def perplexity(loss_nats: float) -> float:
    return math.exp(loss_nats)


def unigram_loss(train_ids: np.ndarray, val_ids: np.ndarray, V: int, alpha: float = 1.0) -> float:
    counts = np.bincount(train_ids, minlength=V).astype(np.float64)
    p = (counts + alpha) / (counts.sum() + alpha * V)
    return float(-np.log(p[val_ids]).mean())


def bigram_loss(train_ids: np.ndarray, val_ids: np.ndarray, V: int, alpha: float = 0.1) -> float:
    counts = np.zeros((V, V))
    np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1.0)          # duplicates must add: np.add.at
    with np.errstate(invalid="ignore", divide="ignore"):            # alpha=0: unseen rows are 0/0, never looked up
        p = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * V)
        return float(-np.log(p[val_ids[:-1], val_ids[1:]]).mean())
