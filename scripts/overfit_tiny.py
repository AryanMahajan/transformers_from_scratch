"""Midterm gate: a randomly initialised model memorises a tiny piece of Tiny Shakespeare.

Setup: the first N characters, ONE window of length T = N-1 (batch size 1),
so every position sees its whole prefix and the next character is fully
determined -> the loss can go to ~0. No weight decay (we WANT to overfit).

Outputs (experiments/overfit_tiny/):
    loss_curve.png   training loss (log scale) with ln V and unigram reference lines
    summary.json     configs, init/final loss, time, teacher-forced accuracy,
                     greedy reproduction from the first character

Run:  python scripts/overfit_tiny.py            (≈ a few minutes on a laptop CPU)
Notes: docs/06_training.md §4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.greedy import greedy_generate  # noqa: E402
from src.tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from src.training.trainer import TrainConfig, train  # noqa: E402
from src.transformer.model import GPT, GPTConfig  # noqa: E402

RAW = ROOT / "data" / "raw" / "tinyshakespeare.txt"
TOKENIZER = ROOT / "data" / "tokenizer" / "char_tinyshakespeare.json"


def unigram_entropy(ids: np.ndarray) -> float:
    counts = np.bincount(ids)
    p = counts[counts > 0] / len(ids)
    return float(-(p * np.log(p)).sum())


def plot(hist: dict, ln_v: float, unigram: float, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")                                         # no window: just write the file
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["step"], hist["loss"], lw=1.2, label="training loss")
    ax.axhline(ln_v, ls="--", c="gray", lw=1, label=f"uniform: ln V = {ln_v:.2f}")
    ax.axhline(unigram, ls=":", c="gray", lw=1, label=f"unigram entropy = {unigram:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats, log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=257)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "overfit_tiny")
    args = ap.parse_args(argv)

    tok = CharTokenizer.load(TOKENIZER)
    text = RAW.read_text(encoding="utf-8")[: args.chars]
    ids = tok.encode(text).astype(np.int64)
    T = len(ids) - 1                                              # exactly one window

    mcfg = GPTConfig(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=T)
    tcfg = TrainConfig(max_steps=args.steps, batch_size=1, block_size=T, max_lr=args.lr,
                       min_lr=args.lr / 10, warmup_steps=20, weight_decay=0.0, grad_clip=1.0,
                       eval_interval=50, eval_iters=1, log_interval=25, seed=args.seed)
    model = GPT(mcfg, np.random.default_rng(args.seed))
    print(f"memorising {len(ids)} chars | {model.num_parameters():,} params | V = {tok.vocab_size}")

    t0 = time.perf_counter()
    hist = train(model, ids, None, tcfg)
    seconds = time.perf_counter() - t0

    # teacher-forced accuracy: does argmax at each position equal the true next char?
    x, y = ids[None, :-1], ids[None, 1:]
    acc = float((model(x).argmax(-1) == y).mean())
    # free-running: start from the first char only, feed back own predictions
    gen = greedy_generate(model, ids[:1], T)
    gen_text = tok.decode(gen)
    n_match = next((i for i, (a, b) in enumerate(zip(gen_text, text)) if a != b), len(text))

    ln_v, uni = float(np.log(tok.vocab_size)), unigram_entropy(ids)
    args.out.mkdir(parents=True, exist_ok=True)
    plot(hist, ln_v, uni, args.out / "loss_curve.png",
         f"Memorising {len(ids)} chars of Tiny Shakespeare ({model.num_parameters():,} params)")
    summary = {
        "model": mcfg.to_dict(), "train": tcfg.to_dict(), "n_params": model.num_parameters(),
        "chars": len(ids), "ln_V": ln_v, "unigram_entropy": uni,
        "init_loss": hist["train_loss"][0], "final_loss": hist["train_loss"][-1],
        "min_step_loss": min(hist["loss"]), "seconds": round(seconds, 1),
        "teacher_forced_accuracy": acc, "greedy_exact_prefix_chars": n_match,
        "greedy_exact": gen_text == text, "text": text, "generated": gen_text,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ninit loss {summary['init_loss']:.4f} (ln V = {ln_v:.4f}) -> final {summary['final_loss']:.5f}"
          f" | {seconds:.0f}s | teacher-forced acc {acc:.1%}"
          f" | greedy reproduces {n_match}/{len(text)} chars")
    print("\n--- generated from the first character ---\n" + gen_text)
    return summary


if __name__ == "__main__":
    main()
