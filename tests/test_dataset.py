import pytest

from src.training.dataset import train_val_split


def test_split_is_contiguous_and_lossless():
    text = "".join(chr(97 + i % 26) for i in range(1000))
    train, val = train_val_split(text, 0.1)
    assert train + val == text                     # nothing lost, order kept
    assert len(train) == 900 and len(val) == 100
    assert text.startswith(train) and text.endswith(val)


def test_split_rejects_bad_fraction():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            train_val_split("abc", bad)
