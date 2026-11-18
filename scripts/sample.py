"""Generate text from a trained checkpoint with greedy / temperature / top-k / top-p.

    python scripts/sample.py                                   # compare a grid of settings
    python scripts/sample.py --temperature 0.8 --top-k 10      # one setting
    python scripts/sample.py --prompt "JULIET:" --n 500 --device gpu

Grid mode writes experiments/<run>/samples.md and samples.json with two
metrics per sample:
    diversity = distinct words / total words           (1.0 = no repetition)
    real-word = share of generated words seen in train  (spelling quality)
Notes: docs/07_real_training.md §3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.sampling import generate  # noqa: E402
from src.training.checkpoint import load_model  # noqa: E402

GRID = [  # (label, temperature, top_k, top_p)
    ("greedy", 0.0, None, None),
    ("tau=0.5", 0.5, None, None),
    ("tau=0.8", 0.8, None, None),
    ("tau=1.0", 1.0, None, None),
    ("tau=1.5", 1.5, None, None),
    ("tau=1.0, k=5", 1.0, 5, None),
    ("tau=0.8, k=10", 0.8, 10, None),
    ("tau=1.0, p=0.9", 1.0, None, 0.9),
]
WORD = re.compile(r"[a-z']+")


def word_metrics(text: str, vocab: set) -> dict:
    words = WORD.findall(text.lower())
    if not words:
        return {"words": 0, "diversity": 0.0, "real_word": 0.0}
    return {"words": len(words),
            "diversity": len(set(words)) / len(words),
            "real_word": sum(w in vocab for w in words) / len(words)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="grid mode: average metrics over this many seeds")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    a = ap.parse_args(argv)

    model, tok, _ = load_model(ROOT / "checkpoints" / a.run / "latest.npz", device=a.device)
    prompt = tok.encode(a.prompt)

    if a.temperature is not None or a.top_k is not None or a.top_p is not None:
        t = 1.0 if a.temperature is None else a.temperature
        ids = generate(model, prompt, a.n, np.random.default_rng(a.seed), t, a.top_k, a.top_p)
        print(tok.decode(ids))
        return

    train_text = tok.decode(np.load(ROOT / "data" / "processed" / "train.npy"))
    vocab = set(WORD.findall(train_text.lower()))
    rows = []
    for label, t, k, p in GRID:
        texts = [tok.decode(generate(model, prompt, a.n, np.random.default_rng(a.seed + s), t, k, p))
                 for s in range(a.seeds)]
        ms = [word_metrics(x[len(a.prompt):], vocab) for x in texts]
        m = {key: float(np.mean([d[key] for d in ms])) for key in ("diversity", "real_word")}
        m |= {key + "_std": float(np.std([d[key] for d in ms])) for key in ("diversity", "real_word")}
        rows.append({"setting": label, "temperature": t, "top_k": k, "top_p": p, **m, "text": texts[0]})
        print(f"{label:15s} diversity {m['diversity']:.2f}+/-{m['diversity_std']:.2f}  "
              f"real-word {m['real_word']:.2f}+/-{m['real_word_std']:.2f}")

    out = ROOT / "experiments" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")
    md = [f"# Samples from `{a.run}` (prompt {a.prompt!r}, {a.n} chars)\n",
          f"Metrics: mean ± std over seeds {a.seed}..{a.seed + a.seeds - 1}; text shown for seed {a.seed}.\n",
          "| setting | diversity | real-word |", "|---|---|---|"]
    md += [f"| {r['setting']} | {r['diversity']:.2f} ± {r['diversity_std']:.2f} "
           f"| {r['real_word']:.2f} ± {r['real_word_std']:.2f} |" for r in rows]
    for r in rows:
        md += [f"\n## {r['setting']}\n", "```", r["text"], "```"]
    (out / "samples.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out / 'samples.md'}")


if __name__ == "__main__":
    main()
