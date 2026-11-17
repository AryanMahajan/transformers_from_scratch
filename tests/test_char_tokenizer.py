import numpy as np
import pytest

from src.tokenizer.char_tokenizer import UNK, CharTokenizer

CORPUS = "First Citizen:\nBefore we proceed any further, hear me speak.\n"


def test_round_trip_in_vocab_text():
    tok = CharTokenizer.train(CORPUS)
    for s in [CORPUS, "hear me", "", "\n\n", "First"]:
        assert tok.decode(tok.encode(s)) == s


def test_vocab_layout_specials_first_then_sorted_chars():
    tok = CharTokenizer.train(CORPUS)
    assert tok.itos[0] == UNK and tok.unk_id == 0
    chars = tok.itos[1:]
    assert chars == sorted(set(CORPUS))
    assert tok.vocab_size == 1 + len(set(CORPUS))


def test_deterministic_regardless_of_char_order():
    a = CharTokenizer.train("abcabc")
    b = CharTokenizer.train("cbacba")
    assert a.itos == b.itos


def test_encode_dtype_shape_and_ids_are_contiguous():
    tok = CharTokenizer.train(CORPUS)
    ids = tok.encode(CORPUS)
    assert ids.dtype == np.int32 and ids.shape == (len(CORPUS),)
    assert ids.min() >= 0 and ids.max() < tok.vocab_size
    assert set(np.unique(ids)) == set(range(1, tok.vocab_size))  # every char used, unk not


def test_unknown_chars_map_to_unk():
    tok = CharTokenizer.train("abc")
    ids = tok.encode("abz")
    assert ids[-1] == tok.unk_id
    assert tok.decode(ids) == "ab" + UNK                 # round trip only for in-vocab text


def test_no_unk_raises_on_unknown():
    tok = CharTokenizer.train("abc", specials=())
    assert tok.unk_id is None and tok.vocab_size == 3
    with pytest.raises(KeyError):
        tok.encode("abz")


def test_decode_rejects_out_of_range_and_accepts_2d():
    tok = CharTokenizer.train("abc")
    with pytest.raises(IndexError):
        tok.decode([1, -1])                              # -1 would silently be last char
    with pytest.raises(IndexError):
        tok.decode([tok.vocab_size])
    assert tok.decode(np.array([[1, 2], [3, 1]])) == "abca"


def test_unicode_code_points():
    text = "café — naïve 🙂"
    tok = CharTokenizer.train(text)
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("🙂")) == 1                    # one code point -> one token


def test_save_load_round_trip(tmp_path):
    tok = CharTokenizer.train(CORPUS + "é")
    path = tmp_path / "tok" / "char.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.itos == tok.itos and loaded.specials == tok.specials
    np.testing.assert_array_equal(loaded.encode(CORPUS), tok.encode(CORPUS))


def test_constructor_validates_vocab():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"], [])
    with pytest.raises(ValueError):
        CharTokenizer(["a", UNK], [UNK])
