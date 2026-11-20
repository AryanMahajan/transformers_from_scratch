"""Phase 8 Step 2: what do the attention heads of a trained model do?

    python scripts/attention_viz.py                                    # centre model
    python scripts/attention_viz.py --run context/t256 --compare pe_scale/sin_scaled_s0 context/t512

1. Heatmaps of every layer x head on one short validation passage.
2. Per-head statistics (mean distance, prev-token / first-token weight, entropy) averaged
   over non-overlapping validation windows -> experiments/attention/<tag>/stats.{json,md}.
3. With --compare: how attention mass spreads over distance for several runs (e.g. the
   context-length experiment), -> reports/figures/attention_distance.png.
4. Validation loss vs tokens of context available (log-spaced bins, SE across windows):
   does a longer window actually help? -> reports/figures/attention_position_loss.png.
Use --windows 1000 to evaluate every validation window.
Runs on CPU in float64: a forward pass of these models takes well under a second.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.attention_stats import distance_profile, head_stats, mass_beyond, reach_cdf  # noqa: E402
from src.tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from src.training.checkpoint import load_model  # noqa: E402

EXP = ROOT / "experiments" / "attention"
FIG = ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STATS = [("mean_distance", "mean distance (tokens)"), ("prev_token", "previous-token weight"),
         ("first_token", "first-token weight"), ("norm_entropy", "normalised entropy")]


def val_tokens(tok) -> np.ndarray:
    """Validation split encoded with the run's own tokenizer (char or BPE)."""
    char_tok = CharTokenizer.load(ROOT / "data" / "tokenizer" / "char_tinyshakespeare.json")
    text = char_tok.decode(np.load(ROOT / "data" / "processed" / "val.npy"))
    return np.asarray(tok.encode(text), dtype=np.int64), text


def position_nll(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """logits [B, T, V] at positions 0..T-2 predict ids 1..T-1 -> NLL (nats) [B, T-1]."""
    z = np.asarray(logits[:, :-1], dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    tgt = ids[:, 1:]
    return -np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]


def context_bins(bpc: np.ndarray):
    """bpc [windows, T-1] (position p has p+1 tokens of context) -> [(lo, hi, mean, se)]
    over log-spaced context bins [lo, hi). The SE is across windows (independent samples)."""
    ctx = np.arange(1, bpc.shape[1] + 1)
    out, lo = [], 1
    while lo <= ctx[-1]:
        hi = 2 * lo
        per_window = bpc[:, (ctx >= lo) & (ctx < hi)].mean(1)
        out.append((lo, hi, float(per_window.mean()), float(per_window.std(ddof=1) / np.sqrt(len(per_window)))))
        lo = hi
    return out


def attention_maps(model, ids: np.ndarray) -> list[np.ndarray]:
    """ids [B, T] -> one [B, H, T, T] attention array per layer."""
    model(ids)
    return [np.asarray(b.attn.attn_weights) for b in model.blocks]


def collect(run: str, n_windows: int, batch: int = 16):
    """Per-layer stats, distance profiles and mass-beyond over n validation windows."""
    model, tok, meta = load_model(ROOT / "checkpoints" / run / "latest.npz")
    model = model.astype(np.float64)
    model.eval()
    T = model.cfg.max_len
    ids, text = val_tokens(tok)
    windows = ids[: (len(ids) // T) * T].reshape(-1, T)[:n_windows]
    acc, nll = None, []
    for s in range(0, len(windows), batch):
        logits = np.asarray(model(windows[s:s + batch]))
        nll.append(position_nll(logits, windows[s:s + batch]))
        maps = [np.asarray(b.attn.attn_weights) for b in model.blocks]
        part = [dict(head_stats(A), profile=distance_profile(A), reach=reach_cdf(A, T // 2),
                     **{f"beyond_{d}": mass_beyond(A, d) for d in (16, 64, 128) if d < T}) for A in maps]
        w = len(windows[s:s + batch])                             # weight by batch size
        acc = ([{k: v * w for k, v in p.items()} for p in part] if acc is None else
               [{k: a[k] + v * w for k, v in p.items()} for a, p in zip(acc, part)])
    layers = [{k: v / len(windows) for k, v in a.items()} for a in acc]
    cpt = len(text) / len(ids)                                    # chars per token -> bits per char
    bpc_pos = np.concatenate(nll) / np.log(2) / cpt              # [windows, T-1]
    return model, tok, text, layers, len(windows), bpc_pos


def write_stats(run, layers, n_windows, T):
    out = EXP / run.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    data = {"run": run, "windows": n_windows, "block_size": T,
            "layers": [{k: np.round(v, 5).tolist() for k, v in L.items() if k not in ("profile", "reach")} for L in layers]}
    (out / "stats.json").write_text(json.dumps(data, indent=1))
    H = len(layers[0]["mean_distance"])
    lines = [f"# Attention statistics: `{run}`", "",
             f"{n_windows} validation windows of T = {T} tokens; queries i >= 1. "
             f"Uniform attention would give mean distance {np.mean(np.arange(1, T) / 2):.1f} "
             f"and normalised entropy 1.", ""]
    for key, label in STATS:
        lines += [f"## {label}", "", "| layer | " + " | ".join(f"h{h}" for h in range(H)) + " |",
                  "|---|" + "---|" * H]
        lines += [f"| {li} | " + " | ".join(f"{v:.3f}" for v in L[key]) + " |" for li, L in enumerate(layers)]
        lines.append("")
    (out / "stats.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def passage(tok, text: str, n_tokens: int) -> np.ndarray:
    """The first n_tokens of the first speech that starts after a blank line."""
    start = text.index("\n\n") + 2
    return np.asarray(tok.encode(text[start:start + 8 * n_tokens]), dtype=np.int64)[:n_tokens]


def label(tok, t: int) -> str:
    return tok.decode([int(t)]).replace(" ", "·").replace("\n", "↵")


def plot_heatmaps(plt, colors, model, tok, text, tag, n_tokens=24):
    ids = passage(tok, text, n_tokens)[None]
    maps = attention_maps(model, ids)
    N, H = len(maps), maps[0].shape[1]
    labels = [label(tok, t) for t in ids[0]]
    cmap = colors.LinearSegmentedColormap.from_list("seq", [SURFACE, ORDINAL[1], ORDINAL[3]])
    fig, axes = plt.subplots(N, H, figsize=(2.6 * H + 1.2, 2.6 * N + 0.8), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    norm = colors.PowerNorm(0.5, vmin=0, vmax=1)
    for li in range(N):
        for h in range(H):
            ax = axes[li, h]
            im = ax.imshow(maps[li][0, h], cmap=cmap, norm=norm)
            ax.set_xticks([]); ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"layer {li}", color=INK2, fontsize=9)
            if li == 0:
                ax.set_title(f"head {h}", color=INK2, fontsize=9)
            for s in ax.spines.values():
                s.set_color(GRID)
    for ax in axes[-1]:                                           # key labels on the bottom row
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6, color=INK2)
    for ax in axes[:, 0]:                                         # query labels on the left column
        ax.set_yticks(range(len(labels)), labels, fontsize=6, color=INK2)
    fig.suptitle(f"Attention maps, {tag}: rows = query, columns = key", x=0.01, ha="left", color=INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.4)
    cb.set_label("attention weight (square-root scale)", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    path = FIG / f"attention_{tag}_heatmaps.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stats(plt, colors, layers, tag):
    cmap = colors.LinearSegmentedColormap.from_list("seq", [SURFACE, ORDINAL[1], ORDINAL[3]])
    N, H = len(layers), len(layers[0]["mean_distance"])
    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.2 + 0.4 * H, 0.55 * N + 1.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, STATS):
        M = np.array([L[key] for L in layers])
        vmax = M.max() if key == "mean_distance" else 1.0
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for li in range(N):
            for h in range(H):
                v = M[li, h]
                ax.text(h, li, f"{v:.1f}" if key == "mean_distance" else f"{v:.2f}", ha="center",
                        va="center", fontsize=7, color="white" if v > 0.55 * vmax else INK)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xticks(range(H), [f"h{h}" for h in range(H)], fontsize=7, color=INK2)
        ax.set_yticks(range(N), [f"L{li}" for li in range(N)], fontsize=7, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Per-head attention statistics, {tag}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = FIG / f"attention_{tag}_stats.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_distance(plt, results):
    """Share of attention within distance d (queries in the second half of the window,
    averaged over every head and layer): an exact CDF, see reach_cdf."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    order = sorted(results, key=lambda r: r[1])
    for k, (run, T, layers, _) in enumerate(order):
        cum = np.mean([L["reach"] for L in layers], axis=(0, 1))             # [T]
        c = ORDINAL[min(k + (4 - len(order)), 3)] if len(order) <= 4 else ORDINAL[k % 4]
        d = np.arange(T)
        ax.plot(d, cum, color=c, lw=2, label=f"T = {T}")
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.set_title("How far back attention reaches (late queries, all heads and layers)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("distance d (tokens; 0 = itself, log scale beyond 1)", color=INK2, fontsize=9)
    ax.set_ylabel("share of attention within d", color=INK2, fontsize=9)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    path = FIG / "attention_distance.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_position_loss(plt, results):
    """Validation bits/char at each position in the window: does more context help?"""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    order = sorted(results, key=lambda r: r[1])
    for k, (run, T, _, bpc) in enumerate(order):
        c = ORDINAL[min(k + (4 - len(order)), 3)] if len(order) <= 4 else ORDINAL[k % 4]
        bins = context_bins(bpc)
        x = np.array([np.sqrt(lo * min(hi, T)) for lo, hi, _, _ in bins])   # geometric bin centre
        m = np.array([b[2] for b in bins]); se = np.array([b[3] for b in bins])
        ax.fill_between(x, m - 2 * se, m + 2 * se, color=c, alpha=0.15, lw=0)
        ax.plot(x, m, color=c, lw=2, marker="o", ms=4, label=f"T = {T}")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    ax.set_title("Loss vs context available (log-spaced bins, band = 2 SE)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("tokens of context available (log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("val bits per character", color=INK2, fontsize=9)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    path = FIG / "attention_position_loss.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="pe_scale/sin_scaled_s0")
    p.add_argument("--compare", nargs="*", default=[], help="more runs for the distance figure")
    p.add_argument("--windows", type=int, default=64)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    results = []
    for k, run in enumerate([a.run] + a.compare):
        model, tok, text, layers, n, bpc_pos = collect(run, a.windows)
        T = model.cfg.max_len
        out = write_stats(run, layers, n, T)
        data = json.loads((out / "stats.json").read_text())
        data["bpc_by_position"] = np.round(bpc_pos.mean(0), 4).tolist()
        data["bpc_by_context_bin"] = [dict(context=[lo, hi], mean=round(m, 4), se=round(se, 4))
                                      for lo, hi, m, se in context_bins(bpc_pos)]
        (out / "stats.json").write_text(json.dumps(data, indent=1))
        results.append((run, T, layers, bpc_pos))
        print("  bits/char by context (tokens seen): " + ", ".join(
            f"{lo}-{hi}: {m:.3f}+/-{se:.3f}" for lo, hi, m, se in context_bins(bpc_pos)))
        print(f"{run}: {n} windows of {T}; stats -> {out.relative_to(ROOT)}")
        for li, L in enumerate(layers):
            print(f"  L{li} dist " + " ".join(f"{v:5.1f}" for v in L["mean_distance"])
                  + " | prev " + " ".join(f"{v:.2f}" for v in L["prev_token"])
                  + " | first " + " ".join(f"{v:.2f}" for v in L["first_token"])
                  + " | ent " + " ".join(f"{v:.2f}" for v in L["norm_entropy"]))
        if k == 0:
            tag = run.replace("/", "_")
            print("  ->", plot_heatmaps(plt, colors, model, tok, text, tag).relative_to(ROOT))
            print("  ->", plot_stats(plt, colors, layers, tag).relative_to(ROOT))
        for d in (16, 64, 128):
            key = f"beyond_{d}"
            if key in layers[0]:
                v = np.mean([L[key] for L in layers])
                print(f"  attention mass >= {d} tokens back (mean over heads/layers): {v:.3f}")
    if a.compare:
        print("  ->", plot_distance(plt, results).relative_to(ROOT))
        print("  ->", plot_position_loss(plt, results).relative_to(ROOT))


if __name__ == "__main__":
    main()
