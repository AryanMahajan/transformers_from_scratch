"""Character-level tokenizer (Unicode code points).

Vocab layout: special tokens first (fixed ids), then corpus characters sorted
(deterministic: same corpus -> same ids). Unseen characters map to <unk>.
Guarantee: decode(encode(s)) == s for in-vocab text.
Notes: docs/03_tokenization.md §1
"""

import json
from pathlib import Path

import numpy as np

UNK = "<unk>"


class CharTokenizer:
    def __init__(self, tokens: list[str], specials: list[str]):
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens in vocab")
        if tokens[: len(specials)] != specials:
            raise ValueError("special tokens must come first")
        self.specials = list(specials)
        self.itos = list(tokens)                         # id -> token
        self.stoi = {t: i for i, t in enumerate(tokens)}  # token -> id
        self.unk_id = self.stoi.get(UNK)

    # ------------------------------------------------------------ build
    @classmethod
    def train(cls, text: str, specials: tuple[str, ...] = (UNK,)) -> "CharTokenizer":
        chars = sorted(set(text) - set(specials))
        return cls(list(specials) + chars, list(specials))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> np.ndarray:
        stoi, unk = self.stoi, self.unk_id
        if unk is None:
            missing = set(text) - stoi.keys()
            if missing:
                raise KeyError(f"characters not in vocab and no {UNK}: {sorted(missing)[:10]}")
        return np.fromiter((stoi.get(c, unk) for c in text), dtype=np.int32, count=len(text))

    def decode(self, ids) -> str:
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError(f"ids out of range [0, {self.vocab_size})")
        return "".join(self.itos[i] for i in ids.reshape(-1).tolist())

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"type": "char", "specials": list(self.specials), "tokens": list(self.itos)}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if data.get("type") != "char":
            raise ValueError(f"not a char tokenizer: type={data.get('type')}")
        return cls(data["tokens"], data["specials"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
