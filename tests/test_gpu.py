"""GPU backend (CuPy): same maths as the CPU path. Skipped without a GPU."""

import numpy as np
import pytest

from src.core.backend import gpu_available, is_gpu_array, to_numpy
from src.nn.loss import cross_entropy, cross_entropy_backward
from src.training.checkpoint import load_model, save_checkpoint
from src.training.trainer import Trainer, TrainConfig
from src.transformer.model import GPT, GPTConfig

pytestmark = pytest.mark.skipif(not gpu_available(), reason="no CUDA GPU / CuPy")


def _pair(dtype=np.float64, **kw):
    cfg = dict(vocab_size=11, d_model=16, n_heads=2, n_layers=2, max_len=12)
    cfg.update(kw)
    cpu = GPT(GPTConfig(**cfg), np.random.default_rng(0)).astype(dtype)
    gpu = GPT(GPTConfig(**cfg), np.random.default_rng(0)).astype(dtype).to("gpu")
    return cpu, gpu


@pytest.mark.parametrize("pos,tie", [("sinusoidal", True), ("learned", False)])
def test_gpu_gradients_match_cpu_in_float64(pos, tie):
    """CPU grads are finite-difference verified; GPU == CPU => GPU backward is correct."""
    cpu, gpu = _pair(pos=pos, tie_weights=tie)
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 11, size=(3, 12))
    x, y = ids[:, :-1], ids[:, 1:]
    for m in (cpu, gpu):
        logits = m(x)
        loss, probs = cross_entropy(logits, y)
        m.zero_grad()
        m.backward(cross_entropy_backward(probs, y).reshape(logits.shape))
    assert is_gpu_array(gpu.tok_emb.E.data)
    np.testing.assert_allclose(to_numpy(gpu(x)), cpu(x), rtol=1e-10, atol=1e-12)
    for (n, p), (_, q) in zip(cpu.named_parameters(), gpu.named_parameters()):
        np.testing.assert_allclose(to_numpy(q.grad), p.grad, rtol=1e-8, atol=1e-12, err_msg=n)


def test_trainer_on_gpu_learns_and_checkpoint_loads_on_cpu(tmp_path):
    data = np.array([0, 1, 2, 3, 4] * 200)
    m = GPT(GPTConfig(vocab_size=5, d_model=16, n_heads=2, n_layers=1, max_len=8), np.random.default_rng(0))
    tr = Trainer(m, data, None, TrainConfig(max_steps=150, batch_size=8, block_size=8, max_lr=1e-2,
                                            min_lr=1e-3, warmup_steps=10, eval_interval=150, eval_iters=2,
                                            log_interval=10**9), log=lambda s: None, device="gpu")
    tr.run()
    assert is_gpu_array(tr.opt.m["tok_emb.E"])                      # optimizer state lives on the GPU
    assert tr.history["train_loss"][-1] < 0.1
    ck = tmp_path / "g.npz"
    save_checkpoint(ck, tr)
    cpu_model, _, _ = load_model(ck, device="cpu")                  # trained on GPU, inspected on CPU
    x = data[None, :8]
    np.testing.assert_allclose(cpu_model(x), to_numpy(tr.model(x)), rtol=1e-4, atol=1e-5)


def test_to_cpu_roundtrip_and_buffers_move():
    _, gpu = _pair(dtype=np.float32)
    assert is_gpu_array(gpu.pos_enc.pe) and gpu.device == "gpu"
    gpu.to("cpu")
    assert not is_gpu_array(gpu.pos_enc.pe) and not is_gpu_array(gpu.blocks[0].attn.q_proj.W.data)
    gpu(np.zeros((1, 4), dtype=np.int64))                          # runs on CPU again
