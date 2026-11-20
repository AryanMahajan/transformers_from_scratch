# Reproducing every result

Every number in the notes comes from a command below. Each training run writes
`experiments/<run>/summary.json` with its full model config, training config,
seed, device and tokenizer, so a run can be rebuilt from its summary alone.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt          # numpy 1.26.4, matplotlib 3.8.3, pytest 9.0.2
pip install -r requirements-gpu.txt          # optional: cupy-cuda12x 13.6.0 for --device gpu
python scripts/prepare_data.py               # Tiny Shakespeare -> data/processed/{train,val}.npy (90/10 split)
python -m pytest -q                          # all tests should pass
```

Python 3.11. With the venv active, `python` below is the venv's interpreter;
the system Python has no CuPy, so `--device gpu` fails with it.

## Determinism guarantees

| claim | checked by |
|---|---|
| same seed + same device -> identical training | `tests/test_trainer.py::test_training_is_deterministic_given_seed` |
| interrupted + resumed run == uninterrupted run, bit for bit | `tests/test_checkpoint.py::test_interrupted_run_resumes_bit_identically` |
| the RNG state is part of the checkpoint (forgetting it breaks resume) | `tests/test_checkpoint.py::test_forgetting_the_rng_state_breaks_resume` |
| BPE training is deterministic: no randomness, count ties broken by smallest ids | by construction; `tests/test_bpe_tokenizer.py::test_incremental_trainer_matches_brute_force` checks the merge list against an independent trainer |

Training runs use float32 (all reported runs were on GPU via CuPy). CPU and GPU
use different matmul kernels and summation orders, so the same run on the other
device is **not** bit-identical: per-step differences start at float32 rounding
and can grow over 3000 steps. This was not measured here; compare across devices
only within the seed-noise band (std ≈ 0.018 bits/char, docs/08 §2).

## Phases 1–6 (building blocks, midterm)

```powershell
python scripts/overfit_tiny.py               # midterm gate: memorise 257 chars -> experiments/overfit_tiny/
python -m pytest -q                          # gradient checks for every layer and the full model
```

## Phase 7 (real training)

| result (docs/07) | command |
|---|---|
| char model, 2.81 bits/char | `python scripts/train.py --run baseline --steps 3000 --batch-size 32 --eval-interval 250 --ckpt-interval 250 --device gpu` |
| BPE-512 model, 2.53 bits/char | same + `--run bpe512 --tokenizer bpe --vocab-size 512` |
| sampling table (diversity vs real words) | `python scripts/sample.py --run baseline --seeds 5` |

## Phase 8 (experiments)

All Phase 8 runs share: BPE-512, 3000 steps × 4096 tokens/step, AdamW 3e-3 → 3e-4
cosine, warmup 100, weight decay 0.1, clip 1.0, eval every 250 steps.

| result (docs/08) | command |
|---|---|
| §1 the full grid (26 runs, restartable) | `python scripts/run_experiments.py --device gpu` (`--list` shows each run's exact `train.py` arguments and whether it is done) |
| §2 centre, 3 seeds (sinusoidal + √D) | `python scripts/train.py --run pe_scale/sin_scaled_s<k> --tokenizer bpe --vocab-size 512 --steps 3000 --batch-size 32 --eval-interval 250 --ckpt-interval 250 --scale-emb --seed <k> --device gpu`, k = 0, 1, 2 |
| §2 learned + √D | same with `--pos learned`, run `pe_scale/learned_scaled_s<k>` |
| §2 learned, no scale | same with `--pos learned`, no `--scale-emb`, run `pe_scale/learned_s<k>` (seed 0 = `ablation_unscaled/positional_encoding/learned`) |
| §2 sinusoidal, no scale | seed 0 = `bpe512`; seeds 1, 2 = `ablation_unscaled/seeds/s1`, `s2` |
| §1–2 tables and figures | `python scripts/analyze_experiments.py` -> `experiments/<exp>/results.md`, `reports/figures/*_curves.png`, `reports/figures/summary.png`, `reports/results.md` |
| §3 attention heatmaps and per-head stats | `python scripts/attention_viz.py --windows 1000 --compare context/t64 context/t256 context/t512` |
| §4 error analysis | `python scripts/error_analysis.py` |

The analysis scripts run on CPU in float64 and are deterministic: they evaluate
fixed, non-overlapping validation windows (no sampling).

## Where things are

| path | contents |
|---|---|
| `checkpoints/<run>/latest.npz` | weights, optimiser state, RNG state, configs, tokenizer (not in git: large) |
| `experiments/<run>/summary.json` | config, final + periodic evals, best periodic val, baselines, a greedy sample; from Phase 8 step 4b on also `environment`: git commit (+ whether the tree had uncommitted changes), Python / NumPy / CuPy versions, GPU name |
| `experiments/<run>/loss_curve.png` | train/val curves of that run |
| `experiments/attention/`, `experiments/error_analysis/` | Phase 8 analysis outputs (`stats.md`, `report.md`) |
| `reports/figures/` | every figure used in the notes |
