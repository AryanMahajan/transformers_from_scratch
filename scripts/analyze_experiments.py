"""Phase 8: collect every run's summary into tables and figures.

    python scripts/analyze_experiments.py

Reads  experiments/<exp>/<run>[_s1|_s2]/summary.json (+ experiments/bpe512 and seeds/ for the centre)
Writes experiments/<exp>/results.md        one table per experiment (mean ± std over seeds)
       reports/figures/<exp>_curves.png     val bits/char vs step (seed-0 run of each config)
       reports/figures/summary.png          final val bits/char vs the varied quantity, seed-noise band
       reports/results.md                   all tables in one file
All losses are reported in bits per character (comparable across runs).
Notes: docs/08_experiments.md
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.checkpoint import read_checkpoint  # noqa: E402

EXP = ROOT / "experiments"
FIG = ROOT / "reports" / "figures"

# palette (validated with the dataviz validator, light surface): ordinal blue ramp, categorical slots 1-3
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# experiment -> (title, x label, [(seed-0 run dir, x value, label)], ordinal?)
# Centre = sinusoidal + sqrt(D) token scale; its seeds and Experiment D come from experiments/pe_scale/.
CENTRE = "pe_scale/sin_scaled_s0"
LEARNED = "pe_scale/learned_scaled_s0"
SEED_RUNS = {CENTRE: [CENTRE, "pe_scale/sin_scaled_s1", "pe_scale/sin_scaled_s2"],
             LEARNED: [LEARNED, "pe_scale/learned_scaled_s1", "pe_scale/learned_scaled_s2"]}
# 2x2 ablation: token scale x position type (the unscaled runs predate the fix)
FACTORIAL = {
    ("sinusoidal", "no scale"): ["bpe512", "ablation_unscaled/seeds/s1", "ablation_unscaled/seeds/s2"],
    ("sinusoidal", "x sqrt(D)"): SEED_RUNS[CENTRE],
    ("learned", "no scale"): ["ablation_unscaled/positional_encoding/learned", "pe_scale/learned_s1", "pe_scale/learned_s2"],
    ("learned", "x sqrt(D)"): SEED_RUNS[LEARNED],
}
EXPERIMENTS = {
    "seeds": ("Seed noise (centre)", "seed", [(r, i, f"seed {i}") for i, r in enumerate(SEED_RUNS[CENTRE])], False),
    "depth": ("A: depth", "layers N", [("depth/n2", 2, "N = 2"), (CENTRE, 4, "N = 4"), ("depth/n6", 6, "N = 6")], True),
    "heads": ("B: heads", "heads H (D = 64)", [("heads/h2", 2, "H = 2"), (CENTRE, 4, "H = 4"), ("heads/h8", 8, "H = 8")], True),
    "context": ("C: context", "context T (tokens)", [("context/t64", 64, "T = 64"), (CENTRE, 128, "T = 128"),
                                                     ("context/t256", 256, "T = 256"), ("context/t512", 512, "T = 512")], True),
    "positional_encoding": ("D: positions", "encoding", [(CENTRE, 0, "sinusoidal"), (LEARNED, 1, "learned")], False),
    "model_size": ("E: model size", "parameters", [(CENTRE, 0, "D = 64"), ("model_size/d128", 0, "D = 128"),
                                                    ("model_size/d256", 0, "D = 256"),
                                                    ("model_size/d384_n6", 0, "D = 384, N = 6")], True),
}


def load_run(run: str) -> dict | None:
    path = EXP / run / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    cpt = s["tokenizer"]["val_chars_per_token"]
    to_bpc = lambda nats: nats / math.log(2) / cpt
    if "evals" in s:
        steps, val, trn = s["evals"]["step"], s["evals"]["val_loss"], s["evals"]["train_loss"]
    else:                                   # older summary (Phase 7 centre run): read the checkpoint history
        meta, _ = read_checkpoint(ROOT / "checkpoints" / run / "latest.npz")
        steps, val, trn = meta["history"]["eval_step"], meta["history"]["val_loss"], meta["history"]["train_loss"]
    val_bpc = np.array([to_bpc(v) for v in val])
    i = int(np.argmin(val_bpc))
    f = s["final"]
    return {"params": s["n_params"], "val": to_bpc(f["val_loss"]), "train": to_bpc(f["train_loss"]),
            "gap": to_bpc(f["val_loss"] - f["train_loss"]), "best": float(val_bpc[i]), "best_step": steps[i],
            "steps": np.array(steps), "val_curve": val_bpc, "minutes": s["seconds_this_session"] / 60,
            "lr": s["train"]["max_lr"], "sample": s["sample_greedy"]}


def replicas(run: str) -> list[str]:
    """A config's seed-0 run plus its seed-1/2 replicas."""
    return SEED_RUNS.get(run, [run, f"{run}_s1", f"{run}_s2"])


def aggregate(runs: list[dict]) -> dict:
    """Mean over seeds (sample std, ddof=1, when there are >= 2 seeds) of the scalar metrics."""
    out = {"n": len(runs), "params": runs[0]["params"], "lr": runs[0]["lr"], "first": runs[0],
           "best_step": runs[0]["best_step"]}
    for k in ("val", "best", "train", "gap", "minutes"):
        v = np.array([r[k] for r in runs])
        out[k], out[k + "_std"] = float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else None)
    return out


def pm(m, sd) -> str:
    return f"{m:.3f}" if sd is None else f"{m:.3f} ± {sd:.3f}"


def table(exp, rows) -> str:
    title = EXPERIMENTS[exp][0]
    out = [f"### {title}\n",
           "| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |",
           "|---|---|---|---|---|---|---|---|"]
    for label, g in rows:
        out.append(f"| {label} | {g['params']:,} | {g['lr']:g} | {g['n']} | **{pm(g['val'], g['val_std'])}** "
                   f"| {pm(g['best'], g['best_std'])} | {pm(g['train'], g['train_std'])} | {pm(g['gap'], g['gap_std'])} |")
    return "\n".join(out) + "\n"


def style(ax, title, xlabel, ylabel="val bits per character"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.stdout.reconfigure(encoding="utf-8")            # Windows console default (cp1252) has no σ / ±
    FIG.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for exp, spec in EXPERIMENTS.items():
        loaded[exp] = []
        for run, x, label in spec[2]:
            runs = [r for r in map(load_run, [run] if exp == "seeds" else replicas(run)) if r]
            loaded[exp].append((label, x, aggregate(runs) if runs else None))
    seeds = [g["val"] for _, _, g in loaded["seeds"] if g]
    noise = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None

    md = ["# Phase 8 results\n",
          "All numbers are validation **bits per character** (BPE-512 tokens, 2.07 chars/token). "
          "Final = 100-batch eval after 3000 steps; best periodic = minimum of the 20-batch evals every "
          "250 steps (noisier, shows overfitting). `a ± b` = mean ± sample std over seeds.\n"]
    if noise is not None:
        md.append(f"**Seed noise:** final val bpc over {len(seeds)} seeds of the centre config = "
                  f"{np.mean(seeds):.3f} ± {noise:.3f}. Single-seed differences below ~2σ = {2 * noise:.3f} "
                  f"are not meaningful; a 3-seed mean has std ≈ σ/√3 = {noise / 3 ** 0.5:.3f}.\n")

    fact = {k: [r for r in map(load_run, v) if r] for k, v in FACTORIAL.items()}
    if all(fact.values()):
        f = {k: aggregate(v) for k, v in fact.items()}
        md += ["### Ablation: token-embedding scale x positional encoding\n",
               "| | no scale | tokens x sqrt(D) |", "|---|---|---|"]
        for pos in ("sinusoidal", "learned"):
            md.append(f"| {pos} | {pm(f[(pos, 'no scale')]['val'], f[(pos, 'no scale')]['val_std'])} "
                      f"| {pm(f[(pos, 'x sqrt(D)')]['val'], f[(pos, 'x sqrt(D)')]['val_std'])} |")
        md.append("")

    for exp, (title, xlabel, _, ordinal) in EXPERIMENTS.items():
        rows = [(label, g) for label, _, g in loaded[exp] if g]
        if not rows:
            continue
        t = table(exp, rows)
        md.append(t)
        (EXP / exp).mkdir(parents=True, exist_ok=True)
        (EXP / exp / "results.md").write_text(t, encoding="utf-8", newline="\n")

        colors = ORDINAL[-len(rows):] if ordinal else CATEGORICAL
        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
        ends = []
        for (label, g), c in zip(rows, colors):
            r = g["first"]                                  # curves: the seed-0 run of each config
            m = r["steps"] >= 500
            ax.plot(r["steps"][m], r["val_curve"][m], color=c, lw=2, label=label)
            ends.append((r["val_curve"][-1], label))
        ends.sort()                                         # direct labels at line ends, nudged apart
        ys = [e[0] for e in ends]
        lo, hi = ax.get_ylim()
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + 0.04 * (hi - lo))
        for y, (_, label) in zip(ys, ends):
            ax.annotate(label, (3000, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        style(ax, f"{title}: validation loss during training (seed 0)", "step")
        ax.set_xlim(500, 3450)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG / f"{exp}_curves.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # summary: final val bpc vs the varied quantity, one panel per experiment, seed band around the centre
    panels = [e for e in ("depth", "heads", "context", "positional_encoding", "model_size")
              if sum(g is not None for _, _, g in loaded[e]) > 1]
    if panels:
        centre = loaded["depth"][1][2]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.5), facecolor=SURFACE, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, exp in zip(axes, panels):
            title, xlabel, _, _ = EXPERIMENTS[exp]
            pts = [(x, label, g) for label, x, g in loaded[exp] if g]
            if exp == "model_size":
                xs = [g["params"] for _, _, g in pts]
                ax.set_xscale("log")
            elif exp == "positional_encoding":
                xs = list(range(len(pts)))
                ax.set_xticks(xs, [label for _, label, _ in pts])
                ax.set_xlim(-0.5, len(pts) - 0.5)
            elif exp == "context":
                xs = [x for x, _, _ in pts]
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs, [str(x) for x in xs])
            else:
                xs = [x for x, _, _ in pts]
                ax.set_xticks(xs)
            if noise is not None:
                ax.axhspan(centre["val"] - 2 * noise, centre["val"] + 2 * noise, color=GRID, alpha=0.7, lw=0)
            ax.errorbar(xs, [g["val"] for _, _, g in pts], yerr=[g["val_std"] or 0 for _, _, g in pts],
                        color=CATEGORICAL[0], lw=2, marker="o", ms=7, mec=SURFACE, mew=2, capsize=3,
                        elinewidth=1.2, label="final")
            if exp == "model_size":
                ax.plot(xs, [g["best"] for _, _, g in pts], color=CATEGORICAL[1], lw=2, ls="--", marker="o",
                        ms=7, mec=SURFACE, mew=2, label="best periodic")
                ax.legend(frameon=False, fontsize=7, labelcolor=INK)
            style(ax, title, xlabel if exp != "positional_encoding" else "",
                  "final val bits/char" if ax is axes[0] else "")
        fig.suptitle("Final validation bits/char, mean ± std over seeds (grey band: centre mean ± 2σ seed noise)",
                     fontsize=10, color=INK2, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(FIG / "summary.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
