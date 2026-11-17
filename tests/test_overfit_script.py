import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data" / "raw" / "tinyshakespeare.txt"


@pytest.mark.skipif(not RAW.exists(), reason="run scripts/prepare_data.py first")
def test_overfit_script_smoke(tmp_path):
    """Tiny, fast run: the script trains, writes the plot + summary, and the loss drops."""
    import overfit_tiny

    s = overfit_tiny.main(["--chars", "33", "--steps", "60", "--lr", "1e-2", "--out", str(tmp_path)])
    assert (tmp_path / "loss_curve.png").stat().st_size > 10_000
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["chars"] == 33 and saved["model"]["max_len"] == 32
    assert abs(s["init_loss"] - s["ln_V"]) < 0.05
    assert s["final_loss"] < 0.5 * s["init_loss"]
