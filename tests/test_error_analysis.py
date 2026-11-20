import numpy as np

from src.analysis.error_analysis import (breakdown, calibration, categorize, char_spans_mask,
                                         frequency_bins, token_kinds, token_scores)


def test_token_scores_match_explicit_softmax():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    targets = np.array([1, 2])
    s = token_scores(logits, targets)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    np.testing.assert_allclose(s["nll"], -np.log(p[[0, 1], targets]))
    np.testing.assert_allclose(s["confidence"], p.max(1))
    np.testing.assert_array_equal(s["correct"], [False, True])
    np.testing.assert_array_equal(s["rank"], [1, 0])


def test_kinds_and_categories():
    vocab = ["\n", " the", "ing", ",", "ROM", " ", "3", "!\n"]
    kinds = token_kinds(vocab)
    assert kinds.tolist() == ["newline", "space-word", "letters", "punct", "letters", "other", "other", "newline"]
    #           prev:   " the"  "ing"  "\n"   ","    " the"  "ROM"
    prev = np.array([1, 2, 0, 3, 1, 4])
    tgt = np.array([2, 2, 4, 4, 3, 0])
    assert categorize(prev, tgt, kinds).tolist() == [
        "word continuation", "word continuation", "line start", "word start", "punctuation", "newline"]


def test_breakdown_shares_sum_to_one():
    nll = np.array([1.0, 3.0, 2.0, 2.0])
    rows = breakdown(nll, np.array([1, 0, 1, 1]), np.array(["a", "b", "a", "a"]))
    assert [r["group"] for r in rows] == ["a", "b"]
    np.testing.assert_allclose([r["loss_share"] for r in rows], [5 / 8, 3 / 8])
    np.testing.assert_allclose([r["token_share"] for r in rows], [3 / 4, 1 / 4])
    np.testing.assert_allclose([r["accuracy"] for r in rows], [1, 0])


def test_calibration_perfect_and_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200_000)
    _, ece = calibration(conf, rng.uniform(0, 1, conf.size) < conf)      # right with prob = confidence
    assert ece < 0.01
    _, ece = calibration(np.full(1000, 0.9), np.arange(1000) < 600)     # says 90 %, right 60 %
    np.testing.assert_allclose(ece, 0.3)


def test_frequency_bins_equal_occurrence():
    counts = np.array([100, 50, 10, 1])                           # id 0 most frequent
    targets = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)     # each id = a quarter of targets
    np.testing.assert_array_equal(frequency_bins(counts, targets, 4), [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    counts2 = np.array([1, 100, 10, 50])                          # labels follow frequency, not id
    np.testing.assert_array_equal(frequency_bins(counts2, np.array([1, 3, 2, 0])), [0, 1, 2, 3])


def test_char_spans_mask():
    lengths = np.array([1, 3, 2])                                 # id 0 = "a", id 1 = "bcd", id 2 = "ef"
    ids = np.array([1, 0, 2, 1])                                  # "bcd" "a" "ef" "bcd": starts 0, 3, 4, 6
    np.testing.assert_array_equal(char_spans_mask(lengths, ids, [(3, 5)]), [False, True, True, False])
    np.testing.assert_array_equal(char_spans_mask(lengths, ids, [(0, 1), (6, 9)]), [True, False, False, True])
