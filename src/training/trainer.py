"""Training loop: everything from Phases 1-6 in one place.

One step:
    x, y   = get_batch(train)                     [B,T], [B,T]
    logits = model(x)                             [B,T,V]
    loss   = cross_entropy(logits, y)             mean over B*T predictions
    model.backward(cross_entropy_backward(...))   fills every p.grad
    norm   = clip_grad_norm(params, grad_clip)    cap the update length
    opt.lr = cosine_lr(step)                      warmup -> cosine
    opt.step(); opt.zero_grad()
Every eval_interval steps: mean loss over eval_iters random batches of train
and val (forward only). Returns a history dict for plotting/reporting.
Notes: docs/06_training.md §3
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


def train(model, train_data, val_data, cfg: TrainConfig, log=print) -> dict:
    rng = np.random.default_rng(cfg.seed)                          # batches
    eval_rng = np.random.default_rng(cfg.seed + 1)                 # separate: eval doesn't shift training batches
    opt = AdamW(model.named_parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)
    hist = {"step": [], "loss": [], "lr": [], "grad_norm": [], "eval_step": [], "train_loss": [], "val_loss": []}
    model.train()
    t0 = time.perf_counter()
    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:  # eval before the step: step 0 = init loss
            tr = estimate_loss(model, train_data, cfg, eval_rng)
            va = estimate_loss(model, val_data, cfg, eval_rng) if val_data is not None else float("nan")
            hist["eval_step"].append(step); hist["train_loss"].append(tr); hist["val_loss"].append(va)
            log(f"step {step:5d} | eval train {tr:.4f} val {va:.4f} | {time.perf_counter() - t0:.1f}s")
        if step == cfg.max_steps:
            break
        opt.lr = cosine_lr(step, cfg.max_lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr)
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, rng)
        loss, norm = train_step(model, opt, x, y, cfg.grad_clip)
        if not np.isfinite(loss):
            raise FloatingPointError(f"loss became {loss} at step {step}")
        hist["step"].append(step); hist["loss"].append(loss); hist["lr"].append(opt.lr); hist["grad_norm"].append(norm)
        if step % cfg.log_interval == 0:
            log(f"step {step:5d} | loss {loss:.4f} | lr {opt.lr:.2e} | grad norm {norm:.3f}")
    hist["optimizer"] = opt                                        # for checkpointing (Phase 7)
    return hist
