from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.tokenizer.bpe_tokenizer import UNK, BPETokenizer, _apply_merge, pretokenize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tinyshakespeare.txt"
TOY = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"


def _reference_merges(text, vocab_size, specials=(UNK,)):
    """Slow, obviously-correct trainer: recount every pair from scratch each merge."""
    tokens = list(specials) + sorted(set(text) - set(specials))
    stoi = {t: i for i, t in enumerate(tokens)}
    counts = Counter(pretokenize(text))
    words = {tuple(stoi[c] for c in w): f for w, f in counts.items()}
    merges = []
    while len(tokens) < vocab_size:
        pc = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                pc[p] += f
        if not pc:
            break
        best = min(pc, key=lambda p: (-pc[p], p))
        if pc[best] < 2:
            break
        new = len(tokens)
        tokens.append(tokens[best[0]] + tokens[best[1]])
        merges.append(best)
        nw = Counter()
        for w, f in words.items():
            nw[tuple(_apply_merge(list(w), best, new))] += f
        words = nw
    return merges


def test_pretokenize_is_lossless_and_keeps_leading_space():
    s = "ROMEO:\nI'll go,  sir -- 1599!\n\n  end"
    pieces = pretokenize(s)
    assert "".join(pieces) == s
    assert " go" in pieces and "'ll" in pieces


def test_apply_merge_left_to_right_non_overlapping():
    assert _apply_merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert _apply_merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]


def test_first_merges_on_the_classic_example():
    tok = BPETokenizer.train(TOY, vocab_size=100)
    first = [tok.itos[tok.merged[m]] for m in tok.merges[:3]]
    assert first[0] == "es"                               # "es" appears in newest(6) + widest(3) = 9
    assert first[1] == "est"


def test_incremental_trainer_matches_brute_force():
    text = RAW.read_text(encoding="utf-8")[:20000] if RAW.exists() else TOY * 50
    tok = BPETokenizer.train(text, vocab_size=200)
    assert tok.merges == _reference_merges(text, 200)


def test_roundtrip_and_compression():
    text = RAW.read_text(encoding="utf-8")[:50000] if RAW.exists() else TOY * 200
    tok = BPETokenizer.train(text, vocab_size=300)
    assert tok.vocab_size == 300
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < len(text) / 1.5                     # real compression
    unseen = "Hello, world:  new text!\n"
    assert tok.decode(tok.encode(unseen)) == "".join(c if c in tok.stoi else UNK for c in unseen)


def test_encoding_training_words_replays_training_merges():
    """A frequent word must come out as a single token if training merged it fully."""
    text = (" the cat sat on the mat" * 50)
    tok = BPETokenizer.train(text, vocab_size=60)
    assert len(tok.encode(" the")) == 1
    assert tok.decode(tok.encode(" the")) == " the"


def test_merges_never_cross_word_boundaries():
    tok = BPETokenizer.train("ab ab ab ab ab", vocab_size=50)
    for t in tok.itos[len(tok.specials):]:
        assert " " not in t[1:]                           # a space can only be the first char


def test_unknown_chars_map_to_unk():
    tok = BPETokenizer.train(TOY, vocab_size=40)
    ids = tok.encode("low!")
    assert tok.unk_id in ids.tolist()
    nounk = BPETokenizer.train(TOY, vocab_size=40, specials=())
    with pytest.raises(KeyError):
        nounk.encode("low!")


def test_save_load_roundtrip(tmp_path):
    tok = BPETokenizer.train(TOY, vocab_size=40)
    tok.save(tmp_path / "bpe.json")
    tok2 = BPETokenizer.load(tmp_path / "bpe.json")
    assert tok2.itos == tok.itos and tok2.merges == tok.merges
    np.testing.assert_array_equal(tok2.encode(TOY), tok.encode(TOY))
    with pytest.raises(ValueError):
        BPETokenizer.from_dict({"type": "char"})


def test_too_small_vocab_rejected_and_stops_when_nothing_to_merge():
    with pytest.raises(ValueError):
        BPETokenizer.train(TOY, vocab_size=3)
    tok = BPETokenizer.train("abc", vocab_size=1000)     # no pair occurs twice
    assert tok.vocab_size == 4 and tok.merges == []


def test_decode_rejects_bad_ids():
    tok = BPETokenizer.train(TOY, vocab_size=40)
    with pytest.raises(IndexError):
        tok.decode([tok.vocab_size])
