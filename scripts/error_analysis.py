"""Phase 8 Step 3: where does the model lose its bits?

    python scripts/error_analysis.py                      # centre model
    python scripts/error_analysis.py --run model_size/d256

Evaluates every non-overlapping validation window and breaks the per-token loss down by
token category, training frequency and confidence (calibration), and lists the most
confident mistakes. -> experiments/error_analysis/<tag>/report.md (+ .json),
reports/figures/error_<tag>_{categories,calibration}.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.attention_viz import GRID, INK, INK2, SURFACE, val_tokens  # noqa: E402
from src.analysis.error_analysis import (CATEGORIES, breakdown, calibration, categorize,  # noqa: E402
                                         char_spans_mask, frequency_bins, token_kinds, token_scores)
from src.tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from src.training.checkpoint import load_model  # noqa: E402

OUT = ROOT / "experiments" / "error_analysis"
FIG = ROOT / "reports" / "figures"
CATEGORICAL = ["#2a78d6", "#eb6834"]
FREQ_NAMES = ["most frequent", "frequent", "less frequent", "rarest"]


def evaluate(run: str, batch: int = 16):
    """Every validation window -> per-token scores, previous ids, targets, context ids."""
    model, tok, meta = load_model(ROOT / "checkpoints" / run / "latest.npz")
    model = model.astype(np.float64)
    model.eval()
    T = model.cfg.max_len
    ids, text = val_tokens(tok)
    windows = ids[: (len(ids) // T) * T].reshape(-1, T)
    parts = []
    for s in range(0, len(windows), batch):
        w = windows[s:s + batch]
        logits = np.asarray(model(w))[:, :-1]                     # position t predicts t+1
        parts.append(token_scores(logits.reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1)))
    scores = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cpt = len(text) / len(ids)
    return model, tok, windows, scores, cpt, ids, text


SPEAKER = re.compile(r"(?:^|\n\n)([A-Z][A-Za-z ]*[A-Z]):\n")    # 'NAME:' on its own line after a blank line


def train_text() -> str:
    char_tok = CharTokenizer.load(ROOT / "data" / "tokenizer" / "char_tinyshakespeare.json")
    return char_tok.decode(np.load(ROOT / "data" / "processed" / "train.npy"))


def train_counts(tok, V: int, text: str) -> np.ndarray:
    return np.bincount(np.asarray(tok.encode(text), dtype=np.int64), minlength=V)


def speaker_groups(tok, strings, ids, text, windows, train: str) -> np.ndarray:
    """Per target token: 'name, seen in train' / 'name, unseen in train' / 'not a name'."""
    known = set(m.group(1) for m in SPEAKER.finditer(train))
    seen, unseen = [], []
    for m in SPEAKER.finditer(text):
        (seen if m.group(1) in known else unseen).append(m.span(1))
    lengths = np.array([len(s) for s in strings])
    n = windows.size                                              # windows cover ids[:n]
    flat = lambda spans: char_spans_mask(lengths, ids[:n], spans).reshape(windows.shape)[:, 1:].reshape(-1)
    g = np.full(windows.shape[0] * (windows.shape[1] - 1), "not a name", dtype=object)
    g[flat(seen)] = "name, seen in train"
    g[flat(unseen)] = "name, unseen in train"
    return g.astype(str), len(seen), len(unseen)


def show(tok, ids) -> str:
    return tok.decode([int(i) for i in ids]).replace("\n", "↵")


def plot_categories(plt, rows, tag):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    names = [r["group"] for r in rows][::-1]
    y = np.arange(len(names))
    h = 0.38
    for k, (key, lab) in enumerate((("token_share", "share of tokens"), ("loss_share", "share of total loss"))):
        vals = [r[key] for r in rows][::-1]
        ax.barh(y + (0.5 - k) * (h + 0.03), vals, height=h, color=CATEGORICAL[k], label=lab)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi + (0.5 - k) * (h + 0.03), f"{v:.0%}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, names, fontsize=8, color=INK2)
    ax.set_xlim(0, max(max(r["token_share"], r["loss_share"]) for r in rows) * 1.18)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    ax.set_title(f"Where the loss goes, by kind of target token ({tag})", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    path = FIG / f"error_{tag}_categories.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    return path


def plot_calibration(plt, cal_rows, ece, tag):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfectly calibrated")
    conf = [r[3] for r in cal_rows]; acc = [r[4] for r in cal_rows]
    ax.plot(conf, acc, color=CATEGORICAL[0], lw=2, marker="o", ms=5, label=f"model (ECE {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("confidence (probability of top prediction)", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy of top prediction", color=INK2, fontsize=9)
    ax.set_title(f"Calibration ({tag})", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    path = FIG / f"error_{tag}_calibration.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pe_scale/sin_scaled_s0")
    p.add_argument("--examples", type=int, default=12)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (plt.matplotlib.ticker)

    model, tok, windows, s, cpt, ids, text = evaluate(a.run)
    train = train_text()
    V, T = model.cfg.vocab_size, model.cfg.max_len
    tag = a.run.replace("/", "_")
    tgt = windows[:, 1:].reshape(-1)
    prev = windows[:, :-1].reshape(-1)
    strings = [tok.decode([i]) for i in range(V)]
    to_bpc = 1 / np.log(2) / cpt
    total_bpc = s["nll"].mean() * to_bpc

    cats = categorize(prev, tgt, token_kinds(strings))
    cat_rows = breakdown(s["nll"], s["correct"], cats, order=[c for c in CATEGORIES])
    fbin = frequency_bins(train_counts(tok, V, train), tgt)
    spk, n_seen, n_unseen = speaker_groups(tok, strings, ids, text, windows, train)
    spk_rows = breakdown(s["nll"], s["correct"], spk, order=["name, seen in train", "name, unseen in train", "not a name"])
    freq_rows = breakdown(s["nll"], s["correct"], fbin, order=list(range(4)))
    cal_rows, ece = calibration(s["confidence"], s["correct"])

    # tokens that cost the most in total
    per_tok = np.bincount(tgt, weights=s["nll"], minlength=V)
    costly = np.argsort(-per_tok)[:10]
    # confident mistakes: wrong, highest confidence
    wrong = np.flatnonzero(~s["correct"])
    conf_wrong = wrong[np.argsort(-s["confidence"][wrong])][: a.examples]

    lines = [f"# Error analysis: `{a.run}`", "",
             f"{len(windows)} validation windows × {T - 1} predictions = {len(tgt):,} tokens. "
             f"Mean {total_bpc:.3f} bits/char ({s['nll'].mean():.3f} nats/token, {cpt:.2f} chars/token). "
             f"Top-1 accuracy {s['correct'].mean():.1%}, top-5 {np.mean(s['rank'] < 5):.1%}.", "",
             "## By kind of target token", "",
             "| category | tokens | share of tokens | share of loss | bits/token | top-1 acc |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['group']} | {r['tokens']:,} | {r['token_share']:.1%} | {r['loss_share']:.1%} | "
              f"{r['mean_nll'] / np.log(2):.2f} | {r['accuracy']:.1%} |" for r in cat_rows]
    lines += ["", "## By training frequency of the target (equal-occurrence quarters)", "",
              "| bin | distinct tokens | share of loss | bits/token | top-1 acc |", "|---|---|---|---|---|"]
    lines += [f"| {FREQ_NAMES[r['group']]} | {len(set(tgt[fbin == r['group']].tolist()))} | {r['loss_share']:.1%} | "
              f"{r['mean_nll'] / np.log(2):.2f} | {r['accuracy']:.1%} |" for r in freq_rows]
    lines += ["", f"## Speaker names ({n_unseen} of {n_seen + n_unseen} speaker turns are by characters never seen in training)", "",
              "| tokens | count | share of tokens | share of loss | bits/token | top-1 acc |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['group']} | {r['tokens']:,} | {r['token_share']:.1%} | {r['loss_share']:.1%} | "
              f"{r['mean_nll'] / np.log(2):.2f} | {r['accuracy']:.1%} |" for r in spk_rows]
    lines += ["", f"## Calibration (ECE = {ece:.3f})", "", "| confidence | tokens | mean confidence | accuracy |",
              "|---|---|---|---|"]
    lines += [f"| {lo:.1f}–{hi:.1f} | {n:,} | {c:.3f} | {acc:.3f} |" for lo, hi, n, c, acc in cal_rows]
    lines += ["", "## Tokens costing the most loss in total", "", "| token | count | share of loss | bits/token |",
              "|---|---|---|---|"]
    lines += [f"| `{strings[t]!r}` | {int((tgt == t).sum())} | {per_tok[t] / s['nll'].sum():.1%} | "
              f"{per_tok[t] / max((tgt == t).sum(), 1) / np.log(2):.2f} |" for t in costly]
    lines += ["", "## Most confident mistakes", "", "| context (last 40 chars) | predicted (p) | actual (p) |",
              "|---|---|---|"]
    for k in conf_wrong:
        w, t = divmod(k, T - 1)
        ctx = show(tok, windows[w, max(0, t - 30): t + 1])[-40:]
        p_actual = np.exp(-s["nll"][k])
        lines.append(f"| `{ctx}` | `{show(tok, [s['pred'][k]])}` ({s['confidence'][k]:.2f}) | "
                     f"`{show(tok, [tgt[k]])}` ({p_actual:.3f}) |")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps({
        "run": a.run, "tokens": int(len(tgt)), "bits_per_char": round(total_bpc, 4),
        "top1": round(float(s["correct"].mean()), 4), "top5": round(float(np.mean(s["rank"] < 5)), 4),
        "ece": round(ece, 4),
        "speaker_turns": {"seen": n_seen, "unseen": n_unseen},
        "speakers": [{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in r.items()} for r in spk_rows],
        "categories": [{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in r.items()} for r in cat_rows],
        "frequency": [{k: (round(float(v), 4) if isinstance(v, float) else int(v) if k in ("group", "tokens") else v)
                       for k, v in r.items()} for r in freq_rows],
        "calibration": [[round(float(x), 4) for x in r] for r in cal_rows]}, indent=1))
    print("\n".join(lines))
    FIG.mkdir(parents=True, exist_ok=True)
    print("->", plot_categories(plt, cat_rows, tag).relative_to(ROOT))
    print("->", plot_calibration(plt, cal_rows, ece, tag).relative_to(ROOT))


if __name__ == "__main__":
    main()
