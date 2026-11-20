"""Where does a language model lose its bits? Per-token losses broken down by type,
frequency and confidence.

    nll_t = -log p(target_t | context)     (nats; the loss averages exactly these)
Splitting the total sum of nll by a property of the target token shows which kinds of
tokens the remaining loss is spent on: a class can be rare but expensive.
Calibration asks whether the model's confidence means what it says:
    ECE = sum_b (n_b / N) |accuracy_b - confidence_b|      (bins b of max-probability)
Interpretation: docs/08_experiments.md §4
"""

import numpy as np

CATEGORIES = ("word start", "word continuation", "line start", "punctuation", "newline", "other")


def token_scores(logits: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """logits [N, V], targets [N] -> nll (nats), confidence (max prob), prediction,
    whether it was right, and the rank of the target (0 = top prediction)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    tgt_logp = np.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    pred = logp.argmax(-1)
    return {
        "nll": -tgt_logp,
        "confidence": np.exp(logp.max(-1)),
        "pred": pred,
        "correct": pred == targets,
        "rank": (logp > tgt_logp[:, None]).sum(-1),              # tokens strictly more likely
    }


def token_kinds(strings: list[str]) -> np.ndarray:
    """Vocabulary strings -> kind per id: 'newline' (contains \\n), 'space-word' (' ' + letter),
    'letters' (starts with a letter), 'punct' (no letter, digit or whitespace), else 'other'."""
    kinds = []
    for s in strings:
        if "\n" in s:
            kinds.append("newline")
        elif len(s) > 1 and s[0] == " " and s[1].isalpha():
            kinds.append("space-word")
        elif s[:1].isalpha():
            kinds.append("letters")
        elif s and not any(c.isalnum() or c.isspace() for c in s):
            kinds.append("punct")
        else:
            kinds.append("other")
    return np.array(kinds)


def categorize(prev_ids: np.ndarray, tgt_ids: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """Category of each target token, using the token before it:
    a letters-token continues a word if the previous token ended inside a word, starts a
    line after a newline (speaker names, first word of a verse line), else starts a word."""
    k, kp = kinds[tgt_ids], kinds[prev_ids]
    cat = np.full(len(tgt_ids), "other", dtype=object)
    cat[k == "newline"] = "newline"
    cat[k == "punct"] = "punctuation"
    cat[k == "space-word"] = "word start"
    letters = k == "letters"
    cat[letters & np.isin(kp, ["letters", "space-word"])] = "word continuation"
    cat[letters & (kp == "newline")] = "line start"
    cat[letters & ~np.isin(kp, ["letters", "space-word", "newline"])] = "word start"
    return cat.astype(str)


def breakdown(nll: np.ndarray, correct: np.ndarray, groups: np.ndarray, order=None) -> list[dict]:
    """Per group: share of tokens, share of the total loss, mean nll, top-1 accuracy."""
    total, n = nll.sum(), len(nll)
    order = order if order is not None else sorted(set(groups.tolist()))
    rows = []
    for g in order:
        m = groups == g
        if m.any():
            rows.append({"group": g, "tokens": int(m.sum()), "token_share": m.sum() / n,
                         "loss_share": nll[m].sum() / total, "mean_nll": nll[m].mean(),
                         "accuracy": correct[m].mean()})
    return rows


def calibration(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Equal-width confidence bins -> (rows [(lo, hi, n, mean confidence, accuracy)], ECE)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)   # bin b: [edge_b, edge_b+1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf, acc = confidence[m].mean(), correct[m].mean()
            rows.append((edges[b], edges[b + 1], int(m.sum()), conf, acc))
            ece += m.mean() * abs(acc - conf)
    return rows, ece


def frequency_bins(counts: np.ndarray, targets: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Label each target by the training frequency of its token, in bins holding roughly
    equal numbers of *target occurrences* (so each bin is equally well measured).
    Returns labels 0 (most frequent) .. n_bins-1 (rarest)."""
    order = np.argsort(-counts, kind="stable")                    # ids, most frequent first
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    occ = np.bincount(targets, minlength=len(counts))[order]      # occurrences in frequency order
    cum = np.cumsum(occ) / occ.sum()
    bin_of_rank = np.minimum((cum - occ / occ.sum() / 2) * n_bins, n_bins - 1).astype(int)
    return bin_of_rank[rank[targets]]


def char_spans_mask(token_lengths: np.ndarray, ids: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Which tokens of the sequence `ids` start inside one of the character spans [a, b)?
    token_lengths[id] = number of characters that token decodes to (decoding concatenates)."""
    starts = np.concatenate([[0], np.cumsum(token_lengths[ids])[:-1]])
    mask = np.zeros(len(ids), dtype=bool)
    for a, b in spans:
        lo, hi = np.searchsorted(starts, [a, b])
        mask[lo:hi] = True
    return mask
