import math

import numpy as np
import pytest

from src.tokenizer.char_tokenizer import CharTokenizer
from src.training.checkpoint import load_model, load_trainer, read_checkpoint, save_checkpoint
from src.training.trainer import Trainer, TrainConfig
from src.transformer.model import GPT, GPTConfig

TEXT = "to be or not to be that is the question " * 30


def _setup(pos="sinusoidal", tie=True):
    tok = CharTokenizer.train(TEXT)
    ids = tok.encode(TEXT).astype(np.int64)
    cut = int(0.9 * len(ids))
    mcfg = GPTConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, max_len=16,
                     pos=pos, tie_weights=tie)
    tcfg = TrainConfig(max_steps=40, batch_size=4, block_size=16, max_lr=1e-2, min_lr=1e-3,
                       warmup_steps=5, eval_interval=10, eval_iters=2, log_interval=10**9, ckpt_interval=10)
    return tok, ids[:cut], ids[cut:], mcfg, tcfg


def _fresh_trainer(pos="sinusoidal", tie=True):
    tok, tr, va, mcfg, tcfg = _setup(pos, tie)
    return Trainer(GPT(mcfg, np.random.default_rng(0)), tr, va, tcfg, log=lambda s: None, tokenizer=tok), tr, va


def _same_floats(a, b):
    return all((x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b)) and len(a) == len(b)


@pytest.mark.parametrize("pos,tie", [("sinusoidal", True), ("learned", False)])
def test_interrupted_run_resumes_bit_identically(tmp_path, pos, tie):
    """train 40 straight  ==  train 20 -> save -> rebuild from file -> train 20 more."""
    full, _, _ = _fresh_trainer(pos, tie)
    full.run()

    part, tr, va = _fresh_trainer(pos, tie)
    ck = tmp_path / "ckpt.npz"
    part.run(ckpt_path=ck, stop_at=20)                              # "killed" after step 20
    del part
    resumed = load_trainer(ck, tr, va, log=lambda s: None)          # "new process"
    assert resumed.step == 20
    resumed.run()

    for (n, p), (_, q) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        np.testing.assert_array_equal(p.data, q.data, err_msg=n)
    for n in full.opt.m:
        np.testing.assert_array_equal(full.opt.m[n], resumed.opt.m[n])
        np.testing.assert_array_equal(full.opt.v[n], resumed.opt.v[n])
    for key in ("step", "loss", "lr", "grad_norm", "eval_step", "train_loss", "val_loss"):
        assert _same_floats(full.history[key], resumed.history[key]), key


def test_forgetting_the_rng_state_breaks_resume(tmp_path):
    """Why RNG state is in the checkpoint: without it the resumed run sees other batches."""
    full, _, _ = _fresh_trainer()
    full.run()
    part, tr, va = _fresh_trainer()
    ck = tmp_path / "c.npz"
    part.run(ckpt_path=ck, stop_at=20)
    resumed = load_trainer(ck, tr, va, log=lambda s: None)
    resumed.rng = np.random.default_rng(resumed.cfg.seed)           # simulate a checkpoint without RNG state
    resumed.run()
    assert full.history["loss"][20:] != resumed.history["loss"][20:]


def test_roundtrip_contents(tmp_path):
    t, _, _ = _fresh_trainer()
    t.run(stop_at=7)
    ck = tmp_path / "x.npz"
    save_checkpoint(ck, t)
    meta, arrays = read_checkpoint(ck)                              # allow_pickle=False inside
    assert meta["step"] == 7 and meta["adam_t"] == 7
    assert meta["model_config"] == t.model.cfg.to_dict()
    assert meta["train_config"] == t.cfg.to_dict()
    assert meta["rng_state"] == t.rng.bit_generator.state
    n_params = len(list(t.model.named_parameters()))
    assert len(arrays) == 3 * n_params                              # param, adam_m, adam_v
    model, tok, _ = load_model(ck)
    assert tok.itos == t.tokenizer.itos
    x = t.train_data[None, :16]
    np.testing.assert_array_equal(model(x), t.model(x))             # same function
    assert model.tok_emb.E.data.dtype == np.float32


def test_save_is_atomic_and_overwrites(tmp_path):
    t, _, _ = _fresh_trainer()
    ck = tmp_path / "sub" / "latest.npz"
    t.run(ckpt_path=ck, stop_at=20)                                 # saves at 10 and 20
    assert ck.exists() and not (tmp_path / "sub" / "latest.npz.tmp").exists()
    assert read_checkpoint(ck)[0]["step"] == 20
    assert [p.name for p in ck.parent.iterdir()] == ["latest.npz"]


def test_final_checkpoint_written_at_end(tmp_path):
    t, _, _ = _fresh_trainer()
    t.cfg.ckpt_interval = 0                                         # periodic saving off
    ck = tmp_path / "end.npz"
    t.run(ckpt_path=ck)
    assert read_checkpoint(ck)[0]["step"] == 40


def test_rejects_unknown_format(tmp_path):
    ck = tmp_path / "bad.npz"
    with open(ck, "wb") as f:
        np.savez(f, meta=np.array('{"format_version": 99}'))
    with pytest.raises(ValueError):
        read_checkpoint(ck)


def test_bpe_tokenizer_survives_checkpoint(tmp_path):
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    text = "to be or not to be, that is the question " * 20
    bpe = BPETokenizer.train(text, vocab_size=40)
    ids = bpe.encode(text).astype(np.int64)
    mcfg = GPTConfig(vocab_size=bpe.vocab_size, d_model=16, n_heads=2, n_layers=1, max_len=8)
    tcfg = TrainConfig(max_steps=2, batch_size=2, block_size=8, warmup_steps=1, eval_interval=100, eval_iters=1)
    t = Trainer(GPT(mcfg, np.random.default_rng(0)), ids, None, tcfg, log=lambda s: None, tokenizer=bpe)
    t.run(ckpt_path=tmp_path / "c.npz")
    _, tok, _ = load_model(tmp_path / "c.npz")
    assert isinstance(tok, BPETokenizer) and tok.merges == bpe.merges
    assert tok.decode(tok.encode(text)) == text
