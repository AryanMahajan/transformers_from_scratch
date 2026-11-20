import re

import numpy as np

from src.utils import environment as envmod
from src.utils.environment import environment


def test_records_code_and_library_versions():
    env = environment("cpu")
    assert env["numpy"] == np.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["python"])
    assert env["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])
    assert env["git_dirty"] in (None, True, False)
    assert env["cupy"] is None and env["gpu"] is None             # not asked for on CPU


def test_missing_git_degrades_to_none(monkeypatch):
    monkeypatch.setattr(envmod, "_git", lambda *a: None)
    env = environment("cpu")
    assert env["git_commit"] is None and env["git_dirty"] is None
