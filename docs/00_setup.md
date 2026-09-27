# Phase 0: Setup

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # bash: source .venv/Scripts/activate
pip install -r requirements-dev.txt
```

| File | Contents |
|------|----------|
| `requirements.txt` | runtime deps: `numpy`, `matplotlib` (visualisation only) |
| `requirements-dev.txt` | runtime deps plus `pytest` |
| `pytest.ini` | runs tests from `tests/` with the repo root on `sys.path`, so `import src...` works |

Versions are pinned exactly so experiment results are reproducible.

## What is git-ignored and why

| Path | Reason |
|------|--------|
| `.venv/` | Machine-specific and rebuildable from the requirements files |
| `papers/*.pdf` | Redistribution rights vary; use `scripts/download_papers.py` |
| `data/raw/*`, `data/processed/*` | Large and reproducible from the documented source |
| `checkpoints/*`, `*.npz`, `*.npy`, `*.pkl` | Binary training artifacts |
| `experiments/**/runs/` | Per-run outputs. Configs and summary results **are** committed |

`data/tokenizer/` is **not** ignored because tokenizer vocabularies are
small and are needed to reproduce results.

## Checkpoint

- [x] `pytest` runs (0 tests collected so far)
- [x] 13 papers downloaded and verified as valid PDFs
