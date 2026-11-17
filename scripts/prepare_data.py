"""Download Tiny Shakespeare, split, tokenize (char-level), and record stats.

    python scripts/prepare_data.py

Outputs:
    data/raw/tinyshakespeare.txt                 (git-ignored)
    data/processed/train.npy, val.npy            int32 ids (git-ignored)
    data/processed/meta.json                     stats (git-ignored)
    data/tokenizer/char_tinyshakespeare.json     vocab (committed)
"""

import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from src.training.dataset import train_val_split  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Same file via jsDelivr's GitHub CDN; raw.githubusercontent.com is blocked on some networks.
MIRRORS = [URL, "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"]
EXPECTED_BYTES = 1_115_394
RAW = ROOT / "data" / "raw" / "tinyshakespeare.txt"
PROCESSED = ROOT / "data" / "processed"
TOKENIZER = ROOT / "data" / "tokenizer" / "char_tinyshakespeare.json"
VAL_FRACTION = 0.1


def download() -> None:
    if RAW.exists():
        print(f"[skip] {RAW.relative_to(ROOT)} already exists")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    part = RAW.with_suffix(".txt.part")
    for url in MIRRORS:
        print(f"[get ] {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except OSError as e:                         # URLError, timeouts, SSL errors
            print(f"[fail] {e}")
            continue
        if len(data) != EXPECTED_BYTES:
            print(f"[fail] got {len(data)} bytes, expected {EXPECTED_BYTES}")
            continue
        part.write_bytes(data)
        part.replace(RAW)                            # atomic: no half-written RAW
        return
    sys.exit(f"all downloads failed; save the file manually as {RAW}")


def main() -> None:
    download()
    raw_bytes = RAW.read_bytes()
    text = raw_bytes.decode("utf-8")
    train_text, val_text = train_val_split(text, VAL_FRACTION)

    tok = CharTokenizer.train(train_text)            # vocab from TRAIN only
    tok.save(TOKENIZER)
    train_ids, val_ids = tok.encode(train_text), tok.encode(val_text)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED / "train.npy", train_ids)
    np.save(PROCESSED / "val.npy", val_ids)

    counts = Counter(train_text)
    meta = {
        "source": URL,
        "licence": "Public domain (Shakespeare); compiled by A. Karpathy for char-rnn",
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "chars": len(text),
        "lines": text.count("\n"),
        "words_whitespace": len(text.split()),
        "split": {"method": "contiguous", "val_fraction": VAL_FRACTION,
                  "train_chars": len(train_text), "val_chars": len(val_text)},
        "tokenizer": {"type": "char", "file": str(TOKENIZER.relative_to(ROOT)).replace("\\", "/"),
                      "vocab_size": tok.vocab_size, "specials": tok.specials},
        "val_unk_count": int((val_ids == tok.unk_id).sum()),
        "val_chars_unseen_in_train": sorted(set(val_text) - set(train_text)),
        "uniform_baseline": {"loss_nats": float(np.log(tok.vocab_size)),
                             "bpc": float(np.log2(tok.vocab_size))},
        "unigram_entropy_bpc": float(-sum(c / len(train_text) * np.log2(c / len(train_text))
                                          for c in counts.values())),
        "top_chars": [[repr(c), n] for c, n in counts.most_common(10)],
    }
    (PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    assert tok.decode(train_ids) == train_text, "round trip failed on train split"
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
