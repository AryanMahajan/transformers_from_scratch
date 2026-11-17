"""Training loop: everything from Phases 1-6 in one place, resumable.

One step:
    x, y   = get_batch(train)                     [B,T], [B,T]
    logits = model(x)                             [B,T,V]
    loss   = cross_entropy(logits, y)             mean over B*T predictions
    model.backward(cross_entropy_backward(...))   fills every p.grad
    norm   = clip_grad_norm(params, grad_clip)    cap the update length
    opt.lr = cosine_lr(step)                      warmup -> cosine
    opt.step(); opt.zero_grad()
Every eval_interval steps: mean loss over eval_iters random batches of train
and val (forward only).

The Trainer owns ALL state that affects the future of a run -- model, AdamW
moments, step counter, both RNGs, history -- so a checkpoint of it resumes
bit-identically (src/training/checkpoint.py).
Notes: docs/06_training.md §3, docs/07_real_training.md §1
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from src.nn.loss import cross_entropy, cross_entropy_backward
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm
from src.optim.schedule import cosine_lr
from src.training.dataset import get_batch


@dataclass
class TrainConfig:
    max_steps: int = 1000
    batch_size: int = 16
    block_size: int = 64                # T: context length used in training (<= model max_len)
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    ckpt_interval: int = 0              # save every k steps (0 = only at the end, if a path is given)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def train_step(model, opt, x, y, grad_clip: float) -> tuple[float, float]:
    """forward, loss, backward, clip, optimizer step. Returns (loss, pre-clip grad norm)."""
    logits = model(x)
    loss, probs = cross_entropy(logits, y)
    opt.zero_grad()                                                # grads accumulate (+=): clear first
    model.backward(cross_entropy_backward(probs, y).reshape(logits.shape))
    norm = clip_grad_norm([p for _, p in opt.named], grad_clip)
    opt.step()
    return loss, norm


def estimate_loss(model, data, cfg: TrainConfig, rng) -> float:
    """Mean loss over eval_iters random batches. Forward only: no grads, no updates."""
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, rng)
        losses.append(cross_entropy(model(x), y)[0])
    model.train()
    return float(np.mean(losses))


def _empty_history() -> dict:
    return {"step": [], "loss": [], "lr": [], "grad_norm": [],
            "eval_step": [], "train_loss": [], "val_loss": [], "tokens_per_sec": []}


class Trainer:
    def __init__(self, model, train_data, val_data, cfg: TrainConfig, log=print, tokenizer=None):
        self.model, self.train_data, self.val_data = model, train_data, val_data
        self.cfg, self.log, self.tokenizer = cfg, log, tokenizer
        self.opt = AdamW(model.named_parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)                 # training batches
        self.eval_rng = np.random.default_rng(cfg.seed + 1)        # separate: evals don't shift training batches
        self.step = 0                                              # number of optimizer steps done
        self.history = _empty_history()

    # ------------------------------------------------------------------ loop
    def _evaluate(self) -> None:
        tr = estimate_loss(self.model, self.train_data, self.cfg, self.eval_rng)
        va = (estimate_loss(self.model, self.val_data, self.cfg, self.eval_rng)
              if self.val_data is not None else float("nan"))
        h = self.history
        h["eval_step"].append(self.step); h["train_loss"].append(tr); h["val_loss"].append(va)
        self.log(f"step {self.step:5d} | eval train {tr:.4f} val {va:.4f}")

    def run(self, ckpt_path=None, stop_at: int | None = None) -> dict:
        """Train until cfg.max_steps (or stop_at, to simulate an interruption).

        Order per iteration: [eval if due] -> [stop?] -> train step -> step += 1 -> [save if due].
        A checkpoint saved after step s holds evals < s; the resumed run evaluates at s with
        the restored eval RNG, exactly as the uninterrupted run would have.
        """
        from src.training.checkpoint import save_checkpoint

        cfg, h = self.cfg, self.history
        end = cfg.max_steps if stop_at is None else min(stop_at, cfg.max_steps)
        self.model.train()
        t_last, tok_count = time.perf_counter(), 0
        while True:
            s = self.step
            if s % cfg.eval_interval == 0 or s == cfg.max_steps:
                self._evaluate()
            if s >= end:
                break
            self.opt.lr = cosine_lr(s, cfg.max_lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr)
            x, y = get_batch(self.train_data, cfg.batch_size, cfg.block_size, self.rng)
            loss, norm = train_step(self.model, self.opt, x, y, cfg.grad_clip)
            if not np.isfinite(loss):
                raise FloatingPointError(f"loss became {loss} at step {s}")
            tok_count += x.size
            h["step"].append(s); h["loss"].append(loss); h["lr"].append(self.opt.lr); h["grad_norm"].append(norm)
            self.step = s + 1
            if s % cfg.log_interval == 0:
                now = time.perf_counter()
                tps = tok_count / (now - t_last)
                h["tokens_per_sec"].append(tps)
                t_last, tok_count = now, 0
                self.log(f"step {s:5d} | loss {loss:.4f} | lr {self.opt.lr:.2e} | grad norm {norm:.3f} | {tps:,.0f} tok/s")
            if ckpt_path and cfg.ckpt_interval and self.step % cfg.ckpt_interval == 0:
                save_checkpoint(ckpt_path, self)
                self.log(f"saved checkpoint at step {self.step} -> {ckpt_path}")
        if ckpt_path and self.step == cfg.max_steps:
            save_checkpoint(ckpt_path, self)                        # final state
        return h


def train(model, train_data, val_data, cfg: TrainConfig, log=print) -> dict:
    """Convenience wrapper: fresh Trainer, run to the end. history["optimizer"] = the AdamW."""
    tr = Trainer(model, train_data, val_data, cfg, log=log)
    hist = tr.run()
    hist["optimizer"] = tr.opt
    return hist
