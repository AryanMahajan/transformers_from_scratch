"""Tokenizers. `tokenizer_from_dict` rebuilds either kind from its saved dict
(the `type` field), so checkpoints can hold a char or a BPE tokenizer."""

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer

_TYPES = {"char": CharTokenizer, "bpe": BPETokenizer}


def tokenizer_from_dict(data: dict):
    kind = data.get("type")
    if kind not in _TYPES:
        raise ValueError(f"unknown tokenizer type {kind!r}")
    return _TYPES[kind].from_dict(data)
