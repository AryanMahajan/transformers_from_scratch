import math

import numpy as np
import pytest

from src.training.baselines import bigram_loss, bpc, perplexity, unigram_loss


def test_unit_conversions():
    assert bpc(math.log(2)) == pytest.approx(1.0)
    assert perplexity(math.log(66)) == pytest.approx(66.0)        # uniform over V -> perplexity V


def test_unigram_uniform_data_gives_log_v():
    ids = np.tile(np.arange(5), 200)
    assert unigram_loss(ids, ids, 5, alpha=0.0) == pytest.approx(math.log(5))


def test_bigram_learns_deterministic_successor():
    ids = np.tile(np.arange(5), 200)                                # 0 1 2 3 4 0 1 ...
    assert bigram_loss(ids, ids, 5, alpha=1e-6) < 1e-4             # next char fully determined
    assert unigram_loss(ids, ids, 5) == pytest.approx(math.log(5), rel=1e-3)


def test_bigram_counts_duplicate_pairs():
    """np.add.at, not fancy-index +=: repeated (prev, next) pairs must all count."""
    ids = np.array([0, 1, 0, 1, 0, 2])                              # (0,1) twice, (0,2) once
    loss = bigram_loss(ids, np.array([0, 1]), 3, alpha=0.0)
    assert loss == pytest.approx(-math.log(2 / 3))


def test_smoothing_keeps_unseen_finite():
    tr = np.array([0, 1, 0, 1])
    va = np.array([1, 2])                                           # (1,2) never seen in train
    assert math.isfinite(bigram_loss(tr, va, 3, alpha=0.1))
    with np.errstate(divide="ignore"):
        assert bigram_loss(tr, va, 3, alpha=0.0) == math.inf


def test_bpc_per_character_for_multi_char_tokens():
    # 2 chars per token: the same per-token loss is half as many bits per char
    assert bpc(math.log(2) * 3, chars_per_token=2.0) == pytest.approx(1.5)
    assert bpc(1.0) == bpc(1.0, chars_per_token=1.0)
