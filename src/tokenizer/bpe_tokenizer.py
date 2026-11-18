"""Byte-Pair-Encoding tokenizer (Sennrich et al. 2016), character base.

Training (greedy compression):
    vocab  = specials + all characters of the corpus
    words  = pre-tokenise the corpus (GPT-2-style regex) -> {word: frequency}
    repeat until len(vocab) == vocab_size:
        (a, b) = the most frequent adjacent pair, counted over words x frequency
        add token a+b; replace every (a, b) by it      <- one "merge", rank = order learned
Encoding a word: start from characters, repeatedly merge the adjacent pair with
the LOWEST rank (learned earliest) until no learned pair is left. This replays
training's order, so encode(training text) gives the same split training saw.
Decoding: concatenate the token strings. decode(encode(s)) == s for in-vocab text.

Pre-tokenisation keeps merges inside words (" the", "ing", "'s") so tokens
don't glue across word boundaries ("e t"); the leading space is part of the word.
Notes: docs/07_real_training.md §4
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UNK = "<unk>"
# GPT-2's split pattern, ASCII version: contractions | ?letters | ?digits | ?other | whitespace
PRETOKENIZE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    return PRETOKENIZE.findall(text)          # pieces concatenate back to text (pattern covers every char)


class BPETokenizer:
    def __init__(self, tokens: list[str], merges: list[tuple[int, int]], specials: list[str]):
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens in vocab")
        if tokens[: len(specials)] != specials:
            raise ValueError("special tokens must come first")
        self.specials = list(specials)
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.unk_id = self.stoi.get(UNK)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}   # pair -> rank
        self.merged = {pair: self.stoi[tokens[pair[0]] + tokens[pair[1]]] for pair in self.merges}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------ train
    @classmethod
    def train(cls, text: str, vocab_size: int, specials: tuple[str, ...] = (UNK,)) -> "BPETokenizer":
        tokens = list(specials) + sorted(set(text) - set(specials))
        if vocab_size < len(tokens):
            raise ValueError(f"vocab_size {vocab_size} < base vocab {len(tokens)}")
        stoi = {t: i for i, t in enumerate(tokens)}
        counts = Counter(pretokenize(text))
        words = [[stoi[c] for c in w] for w in counts]       # unique words as id lists
        freqs = list(counts.values())

        # pair statistics, kept up to date incrementally
        pair_count: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)   # pair -> words containing it
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_count[pair] += freqs[wi]
                where[pair].add(wi)

        merges = []
        while len(tokens) < vocab_size and pair_count:
            # most frequent pair; ties -> smallest ids (deterministic)
            best = min(pair_count, key=lambda p: (-pair_count[p], p))
            if pair_count[best] < 2:
                break                                         # merging a pair seen once compresses nothing useful
            new_id = len(tokens)
            tokens.append(tokens[best[0]] + tokens[best[1]])
            merges.append(best)
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):                    # remove this word's old pairs
                    pair_count[pair] -= f
                    if pair_count[pair] == 0:
                        del pair_count[pair]
                    where[pair].discard(wi)
                words[wi] = w = _apply_merge(w, best, new_id)
                for pair in zip(w, w[1:]):                    # add its new pairs
                    pair_count[pair] += f
                    where[pair].add(wi)
            where.pop(best, None)
        return cls(tokens, merges, list(specials))

    # ------------------------------------------------------------ encode / decode
    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        stoi, unk = self.stoi, self.unk_id
        if unk is None and any(c not in stoi for c in word):
            raise KeyError(f"characters not in vocab and no {UNK}: {word!r}")
        ids = [stoi.get(c, unk) for c in word]
        while len(ids) > 1:
            ranked = [(self.ranks.get(p, np.inf), p) for p in zip(ids, ids[1:])]
            rank, pair = min(ranked)
            if rank == np.inf:
                break                                         # no learned merge applies
            ids = _apply_merge(ids, pair, self.merged[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> np.ndarray:
        out: list[int] = []
        for word in pretokenize(text):
            out.extend(self._encode_word(word))
        return np.array(out, dtype=np.int32)

    def decode(self, ids) -> str:
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise IndexError(f"ids out of range [0, {self.vocab_size})")
        return "".join(self.itos[i] for i in ids.reshape(-1).tolist())

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"type": "bpe", "specials": self.specials, "tokens": self.itos,
                "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        if data.get("type") != "bpe":
            raise ValueError(f"not a BPE tokenizer: type={data.get('type')}")
        return cls(data["tokens"], [tuple(m) for m in data["merges"]], data["specials"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8",
                        newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right."""
    out, i, a, b = [], 0, pair[0], pair[1]
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
