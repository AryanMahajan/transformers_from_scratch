import numpy as np
import pytest

from src.generation.greedy import greedy_generate
from src.training.trainer import TrainConfig, train
from src.transformer.model import GPT, GPTConfig


def _model(max_len=8, seed=0):
    return GPT(GPTConfig(vocab_size=5, d_model=16, n_heads=2, n_layers=1, max_len=max_len),
               np.random.default_rng(seed))


def test_length_prompt_kept_and_ids_in_range():
    out = greedy_generate(_model(), [1, 2, 3], 10)
    assert out.shape == (13,)
    np.testing.assert_array_equal(out[:3], [1, 2, 3])
    assert out.min() >= 0 and out.max() < 5


def test_crops_context_beyond_max_len():
    """Generating past max_len must not crash: only the last max_len ids are fed."""
    out = greedy_generate(_model(max_len=4), [0], 20)
    assert len(out) == 21


def test_deterministic_and_restores_train_mode():
    m = _model()
    a = greedy_generate(m, [2], 6)
    b = greedy_generate(m, [2], 6)
    np.testing.assert_array_equal(a, b)
    assert m.training


def test_next_token_is_argmax_of_last_position():
    m = _model()
    out = greedy_generate(m, [3, 1], 1)
    assert out[-1] == np.argmax(m(np.array([[3, 1]]))[0, -1])


def test_reproduces_a_learned_cycle():
    data = np.array([0, 1, 2, 3, 4] * 100)
    m = _model(max_len=8)
    train(m, data, None, TrainConfig(max_steps=150, batch_size=8, block_size=8, max_lr=1e-2,
                                    min_lr=1e-3, warmup_steps=10, eval_interval=150, eval_iters=1),
          log=lambda s: None)
    np.testing.assert_array_equal(greedy_generate(m, [2], 12), [2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        greedy_generate(_model(), [], 3)
