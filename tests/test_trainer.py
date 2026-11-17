import numpy as np
import pytest

from src.optim.schedule import cosine_lr
from src.training.dataset import get_batch
from src.training.trainer import TrainConfig, estimate_loss, train
from src.transformer.model import GPT, GPTConfig

PERIODIC = np.array([ord(c) - 97 for c in "thequickbrownfox" * 200], dtype=np.int64)   # V <= 26


def _model(seed=0, **kw):
    cfg = dict(vocab_size=26, d_model=32, n_heads=2, n_layers=1, max_len=32)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg), np.random.default_rng(seed))


def _cfg(**kw):
    base = dict(max_steps=40, batch_size=8, block_size=24, max_lr=1e-2, min_lr=1e-3,
                warmup_steps=10, eval_interval=20, eval_iters=3, log_interval=10**9)
    base.update(kw)
    return TrainConfig(**base)


# ------------------------------------------------------------------ get_batch
def test_get_batch_shapes_and_shift_by_one():
    data = np.arange(100)
    x, y = get_batch(data, 5, 8, np.random.default_rng(0))
    assert x.shape == y.shape == (5, 8)
    np.testing.assert_array_equal(y, x + 1)                         # y is x shifted by one token
    np.testing.assert_array_equal(np.diff(x, axis=1), 1)            # each row a contiguous window


def test_get_batch_covers_every_valid_window_and_never_overruns():
    data = np.arange(12)
    T = 4
    x, y = get_batch(data, 5000, T, np.random.default_rng(1))
    starts = set(x[:, 0].tolist())
    assert starts == set(range(len(data) - T))                     # 0..7 all reachable
    assert y.max() == len(data) - 1                                 # last token used as a target


def test_get_batch_rejects_short_data_and_is_seeded():
    with pytest.raises(ValueError):
        get_batch(np.arange(8), 2, 8, np.random.default_rng(0))
    a = get_batch(np.arange(50), 3, 5, np.random.default_rng(7))
    b = get_batch(np.arange(50), 3, 5, np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])


# ------------------------------------------------------------------ loop
def test_estimate_loss_at_init_is_log_v_and_changes_nothing():
    m = _model()
    before = m.state_dict()
    loss = estimate_loss(m, PERIODIC, _cfg(), np.random.default_rng(0))
    assert abs(loss - np.log(26)) < 0.1
    after = m.state_dict()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert m.training                                               # restored to train mode


def test_history_schedule_and_lengths():
    h = train(_model(), PERIODIC, PERIODIC, _cfg(), log=lambda s: None)
    assert h["step"] == list(range(40))
    assert h["eval_step"] == [0, 20, 40]
    np.testing.assert_allclose(h["lr"], [cosine_lr(s, 1e-2, 10, 40, 1e-3) for s in range(40)])
    assert all(np.isfinite(h["grad_norm"])) and h["optimizer"].t == 40


def test_training_is_deterministic_given_seed():
    h1 = train(_model(), PERIODIC, None, _cfg(), log=lambda s: None)
    h2 = train(_model(), PERIODIC, None, _cfg(), log=lambda s: None)
    assert h1["loss"] == h2["loss"]


def test_learns_periodic_sequence():
    """Tied model: unigram plateau (~2.69) first, then breaks through to ~0."""
    h = train(_model(), PERIODIC, None, _cfg(max_steps=300, eval_interval=100), log=lambda s: None)
    assert h["train_loss"][0] == pytest.approx(np.log(26), abs=0.1)
    assert h["train_loss"][-1] < 0.1


def test_logs_are_written():
    lines = []
    train(_model(), PERIODIC, PERIODIC, _cfg(max_steps=20, log_interval=10), log=lines.append)
    assert any("eval train" in s for s in lines) and any("grad norm" in s for s in lines)
