"""Per-head summaries of causal attention maps.

A[b, h, i, j] = how much query position i attends to key position j (rows sum to 1,
zero above the diagonal). Every statistic is averaged over the batch and over query
positions i >= 1: query 0 can only attend to itself, so it carries no information.

    mean distance      sum_j A_ij (i - j)          how far back the head looks
    prev-token weight  A_{i, i-1}                  "look at the last token" heads
    self weight        A_{i, i}
    first-token weight A_{i, 0}                    attention sinks (parking mass on token 0)
    norm. entropy      H_i / ln(i + 1)             0 = one token, 1 = uniform over the allowed keys
Derivations and interpretation: docs/08_experiments.md §3
"""

import numpy as np


def head_stats(A: np.ndarray) -> dict[str, np.ndarray]:
    """A [B, H, T, T] causal attention -> {stat name: [H] array}."""
    A = np.asarray(A, dtype=np.float64)
    B, H, T, T2 = A.shape
    if T != T2 or T < 2:
        raise ValueError(f"expected [B, H, T, T] with T >= 2, got {A.shape}")
    i = np.arange(T)
    dist = (i[:, None] - i[None, :]).astype(np.float64)          # [T, T]; >0 below the diagonal
    rows = i[1:]                                                 # queries 1..T-1

    ent = -np.sum(A * np.log(np.where(A > 0, A, 1.0)), axis=-1)  # [B, H, T]; 0 log 0 = 0
    return {
        "mean_distance": (A * dist).sum(-1)[..., 1:].mean(axis=(0, 2)),
        "prev_token": A[:, :, rows, rows - 1].mean(axis=(0, 2)),
        "self": A[:, :, rows, rows].mean(axis=(0, 2)),
        "first_token": A[:, :, rows, 0].mean(axis=(0, 2)),
        "norm_entropy": (ent[..., 1:] / np.log(rows + 1.0)).mean(axis=(0, 2)),
    }


def distance_profile(A: np.ndarray) -> np.ndarray:
    """A [B, H, T, T] -> [H, T]: average attention mass at each distance d = i - j.

    Averaged over the queries that can reach distance d (i >= d, and i >= 1 as in
    head_stats), so long distances aren't penalised for having fewer queries.
    Entry [h, 0] is the self weight."""
    A = np.asarray(A, dtype=np.float64)
    B, H, T, _ = A.shape
    prof = np.zeros((H, T))
    for d in range(T):
        i = np.arange(max(d, 1), T)
        prof[:, d] = A[:, :, i, i - d].mean(axis=(0, 2))
    return prof


def reach_cdf(A: np.ndarray, q_from: int) -> np.ndarray:
    """A [B, H, T, T] -> [H, T]: share of attention within distance d (entry d), for the
    queries i >= q_from. Each query's own distance distribution is accumulated, so it is
    an exact CDF (ends at 1). Using only late queries compares models of different T
    fairly: every query used has at least q_from tokens of history."""
    A = np.asarray(A, dtype=np.float64)
    B, H, T, _ = A.shape
    if not 1 <= q_from < T:
        raise ValueError(f"need 1 <= q_from < T={T}, got {q_from}")
    out = np.zeros((H, T))
    for i in range(q_from, T):
        c = np.cumsum(A[:, :, i, i::-1], axis=-1).mean(axis=0)  # distances 0..i -> [H, i+1]
        out[:, : i + 1] += c
        out[:, i + 1:] += c[:, -1:]                              # nothing further back exists
    return out / (T - q_from)


def mass_beyond(A: np.ndarray, d: int) -> np.ndarray:
    """A [B, H, T, T] -> [H]: share of attention spent on keys at least d tokens back,
    averaged over the queries that have such keys (i >= d)."""
    A = np.asarray(A, dtype=np.float64)
    T = A.shape[-1]
    if not 1 <= d < T:
        raise ValueError(f"need 1 <= d < T={T}, got {d}")
    i = np.arange(T)
    far = (i[:, None] - i[None, :]) >= d                         # [T, T]
    return (A * far).sum(-1)[..., d:].mean(axis=(0, 2))
