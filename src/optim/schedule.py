"""Learning-rate schedule: linear warmup, then cosine decay to min_lr.

    step < W:        lr = max_lr * (step + 1) / W                       ramp up
    W <= step <= S:  progress = (step - W) / (S - W)                    0 -> 1
                     lr = min_lr + 0.5 (max_lr - min_lr)(1 + cos(pi * progress))
    step > S:        lr = min_lr
(W = warmup_steps, S = total_steps.) "+1" so step 0 already learns a little.
Derivations: docs/06_training.md §2
"""

import math


def cosine_lr(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("need 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
