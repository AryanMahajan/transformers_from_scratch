"""Train the char-level GPT on Tiny Shakespeare, with checkpoint/resume.

    python scripts/train.py --run baseline                 # fresh run
    python scripts/train.py --run baseline --resume        # continue from checkpoints/baseline/latest.npz
    python scripts/train.py --run baseline --device gpu    # optional CuPy backend (pip install -r requirements-gpu.txt)
    python scripts/train.py --run bpe512 --tokenizer bpe --vocab-size 512   # BPE trained on the train split

Outputs:
    checkpoints/<run>/latest.npz        full training state (gitignored)
    experiments/<run>/loss_curve.png    train/val loss vs step, with baselines
    experiments/<run>/summary.json      configs, final metrics (nats, bpc, perplexity), baselines, sample
Losses are per TOKEN (nats); bits per CHARACTER divide by chars/token so
char and BPE runs compare fairly.
Notes: docs/07_real_training.md §2, §4
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
from src.tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402
from src.tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from src.training.baselines import bigram_loss, bpc, perplexity, unigram_loss  # noqa: E402
from src.training.checkpoint import load_trainer  # noqa: E402
from src.training.trainer import Trainer, TrainConfig, estimate_loss  # noqa: E402
from src.transformer.model import GPT, GPTConfig  # noqa: E402

DATA = ROOT / "data" / "processed"
TOKENIZER = ROOT / "data" / "tokenizer" / "char_tinyshakespeare.json"


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    ap.add_argument("--vocab-size", type=int, default=512, help="BPE only")
    # model
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--pos", default="sinusoidal", choices=["sinusoidal", "learned"])
    ap.add_argument("--scale-emb", action="store_true", help="tokens x sqrt(d_model) (Vaswani 3.4)")
    # training
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--ckpt-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def plot(hist, refs: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(hist["step"], hist["loss"], lw=0.6, alpha=0.35, c="C0", label="train (per batch)")
    ax.plot(hist["eval_step"], hist["train_loss"], "-o", ms=3, c="C0", label="train (eval)")
    ax.plot(hist["eval_step"], hist["val_loss"], "-o", ms=3, c="C3", label="val (eval)")
    for (name, v), ls in zip(refs.items(), ["--", ":", "-."]):
        ax.axhline(v, ls=ls, c="gray", lw=1, label=f"{name} = {v:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats per token)")
    ax.set_ylim(top=max(refs.values()) + 0.3)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_data(kind: str, vocab_size: int):
    """(tokenizer, train ids, val ids, chars per val token). BPE is trained on the
    train split only (deterministic, <1 s) and saved next to the char vocab."""
    char_tok = CharTokenizer.load(TOKENIZER)
    train_ids = np.load(DATA / "train.npy").astype(np.int64)
    val_ids = np.load(DATA / "val.npy").astype(np.int64)
    if kind == "char":
        return char_tok, train_ids, val_ids, 1.0
    train_text, val_text = char_tok.decode(train_ids), char_tok.decode(val_ids)
    tok = BPETokenizer.train(train_text, vocab_size)
    tok.save(TOKENIZER.parent / f"bpe{vocab_size}_tinyshakespeare.json")
    tr, va = tok.encode(train_text).astype(np.int64), tok.encode(val_text).astype(np.int64)
    return tok, tr, va, len(val_text) / len(va)


def main(argv=None) -> dict:
    a = parse(argv)
    tok, train_ids, val_ids, cpt = load_data(a.tokenizer, a.vocab_size)
    print(f"tokenizer {a.tokenizer}: V={tok.vocab_size}, {len(train_ids):,} train / {len(val_ids):,} val tokens,"
          f" {cpt:.3f} chars/token")
    ckpt = ROOT / "checkpoints" / a.run / "latest.npz"
    out = ROOT / "experiments" / a.run
    out.mkdir(parents=True, exist_ok=True)

    if a.resume:
        trainer = load_trainer(ckpt, train_ids, val_ids, device=a.device)
        print(f"resumed {a.run} at step {trainer.step}/{trainer.cfg.max_steps}")
    else:
        mcfg = GPTConfig(vocab_size=tok.vocab_size, d_model=a.d_model, n_heads=a.n_heads,
                         n_layers=a.n_layers, max_len=a.block_size, pos=a.pos,
                         scale_emb=a.scale_emb)
        tcfg = TrainConfig(max_steps=a.steps, batch_size=a.batch_size, block_size=a.block_size,
                           max_lr=a.lr, min_lr=a.lr / 10, warmup_steps=a.warmup,
                           weight_decay=a.weight_decay, eval_interval=a.eval_interval,
                           eval_iters=a.eval_iters, log_interval=10, ckpt_interval=a.ckpt_interval,
                           seed=a.seed)
        trainer = Trainer(GPT(mcfg, np.random.default_rng(a.seed)), train_ids, val_ids, tcfg, tokenizer=tok,
                          device=a.device)
    model, cfg = trainer.model, trainer.cfg
    print(f"{model.num_parameters():,} params on {trainer.device} | {model.cfg} | {cfg}")

    t0 = time.perf_counter()
    hist = trainer.run(ckpt_path=ckpt)
    seconds = time.perf_counter() - t0

    # final evaluation with more batches than the periodic evals (less noise)
    final_cfg = TrainConfig(**{**cfg.to_dict(), "eval_iters": 100})
    rng = np.random.default_rng(12345)
    val = estimate_loss(model, val_ids, final_cfg, rng)
    trn = estimate_loss(model, train_ids, final_cfg, rng)
    V = tok.vocab_size
    refs = {"uniform ln V": float(np.log(V)), "unigram": unigram_loss(train_ids, val_ids, V),
            "bigram": bigram_loss(train_ids, val_ids, V)}
    prompt = tok.encode("ROMEO:")
    sample = tok.decode(greedy_generate(model, prompt, 300))

    plot(hist, refs, out / "loss_curve.png",
         f"{a.run}: {model.num_parameters():,} params, {a.tokenizer} (V={V}) Tiny Shakespeare")
    summary = {
        "run": a.run, "device": trainer.device, "model": model.cfg.to_dict(), "train": cfg.to_dict(),
        "n_params": model.num_parameters(), "steps": trainer.step,
        "seconds_this_session": round(seconds, 1),
        "tokens_seen": trainer.step * cfg.batch_size * cfg.block_size,
        "median_tokens_per_sec": float(np.median(hist["tokens_per_sec"])) if hist["tokens_per_sec"] else None,
        "tokenizer": {"type": a.tokenizer, "vocab_size": tok.vocab_size, "val_chars_per_token": cpt},
        "final": {"train_loss": trn, "val_loss": val, "val_bpc": bpc(val, cpt),
                  "val_perplexity_per_token": perplexity(val), "val_perplexity_per_char": 2 ** bpc(val, cpt),
                  "generalisation_gap": val - trn},
        "baselines_val": {k: {"nats": v, "bpc": bpc(v, cpt), "perplexity": perplexity(v)} for k, v in refs.items()},
        # periodic evals (eval_iters batches each): for curves and for runs that overfit before the end
        "evals": {"step": hist["eval_step"], "train_loss": hist["train_loss"], "val_loss": hist["val_loss"]},
        "best_periodic_val": (lambda i: {"step": hist["eval_step"][i], "val_loss": hist["val_loss"][i],
                                         "val_bpc": bpc(hist["val_loss"][i], cpt)})(int(np.argmin(hist["val_loss"]))),
        "sample_greedy": sample,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    print(f"\nval {val:.4f} nats/token = {bpc(val, cpt):.3f} bits/char | train {trn:.4f}"
          f" | bigram {bpc(refs['bigram'], cpt):.3f} bits/char | {seconds / 60:.1f} min")
    print("\n--- greedy sample ---\n" + sample)
    return summary


if __name__ == "__main__":
    main()
