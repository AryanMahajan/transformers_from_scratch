"""Phase 8: run the experiment grid (one variable at a time around a centre config).

    python scripts/run_experiments.py --device gpu            # all runs not yet done
    python scripts/run_experiments.py --only depth heads      # some experiments
    python scripts/run_experiments.py --list                  # print the grid

Centre = BPE-512, D=64, H=4, N=4, T=128, sinusoidal + sqrt(D) token scale, 3000 steps x
4096 tokens, lr 3e-3. Each run writes experiments/<exp>/<name>/summary.json; a run with a
summary is skipped, and an interrupted run resumes from its checkpoint, so this
script can be stopped and restarted at any time.
Notes: docs/08_experiments.md
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402

TOKENS_PER_STEP = 4096
# Centre = BPE-512, D=64, H=4, N=4, T=128, sinusoidal positions WITH tokens x sqrt(D) (Vaswani 3.4).
# The first grid ran without the scale; those runs are kept in experiments/ablation_unscaled/.
CENTRE = {"d-model": 64, "n-heads": 4, "n-layers": 4, "pos": "sinusoidal", "scale-emb": True,
          "block-size": 128, "lr": 3e-3, "seed": 0}

# experiment -> {run name: overrides of CENTRE}. The centre itself (3 seeds) and Experiment D
# (learned + scale, 3 seeds) were run by the scale-confirmation set in experiments/pe_scale/.
GRID = {
    "depth":   {"n2": {"n-layers": 2}, "n6": {"n-layers": 6}},
    "heads":   {"h2": {"n-heads": 2}, "h8": {"n-heads": 8}},
    "context": {"t64": {"block-size": 64}, "t256": {"block-size": 256}, "t512": {"block-size": 512}},
    "model_size": {"d128": {"d-model": 128},
                   "d256": {"d-model": 256, "n-heads": 8, "lr": 1e-3},
                   "d384_n6": {"d-model": 384, "n-heads": 12, "n-layers": 6, "lr": 6e-4}},
}
# Every cheap config gets seeds 1 and 2 ("<name>_s1", "<name>_s2"), run after the grid above.
REPLICATE = {"depth": ["n2", "n6"], "heads": ["h2", "h8"], "context": ["t64", "t256", "t512"],
             "model_size": ["d128"]}
REPLICAS = {exp: {f"{name}_s{s}": {**GRID[exp][name], "seed": s} for name in names for s in (1, 2)}
            for exp, names in REPLICATE.items()}


def argv_for(exp: str, name: str, overrides: dict, device: str) -> list[str]:
    cfg = {**CENTRE, **overrides}
    batch = TOKENS_PER_STEP // cfg["block-size"]               # same tokens per step for every run
    args = ["--run", f"{exp}/{name}", "--tokenizer", "bpe", "--vocab-size", "512", "--device", device,
            "--steps", "3000", "--warmup", "100", "--eval-interval", "250", "--ckpt-interval", "250",
            "--batch-size", str(batch)]
    for k, v in cfg.items():
        if isinstance(v, bool):                                # store_true flags
            args += [f"--{k}"] if v else []
        else:
            args += [f"--{k}", str(v)]
    return args


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    plan = [(exp, name, ov) for grid in (GRID, REPLICAS) for exp, runs in grid.items() for name, ov in runs.items()]
    for exp, name, ov in plan:
        if a.only and exp not in a.only:
            continue
        args = argv_for(exp, name, ov, a.device)
        done = (ROOT / "experiments" / exp / name / "summary.json").exists()
        if a.list:
            print(("done " if done else "todo ") + " ".join(args))
            continue
        if done:
            continue
        ckpt = ROOT / "checkpoints" / exp / name / "latest.npz"
        if ckpt.exists():
            args.append("--resume")
        print(f"=== start {exp}/{name}", flush=True)
        s = train.main(args)
        print(f"=== done {exp}/{name}: {s['n_params']:,} params, val {s['final']['val_bpc']:.3f} bits/char"
              f" (best periodic {s['best_periodic_val']['val_bpc']:.3f} @ {s['best_periodic_val']['step']}),"
              f" {s['seconds_this_session'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
