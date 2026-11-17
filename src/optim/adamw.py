"""Adam / AdamW (Kingma & Ba 2014; Loshchilov & Hutter 2017).

Per parameter, at step t (g = p.grad):
    m = b1*m + (1-b1)*g              1st moment: running mean of g     (direction)
    v = b2*v + (1-b2)*g^2            2nd moment: running mean of g^2   (scale)
    m_hat = m / (1 - b1^t)           bias correction: m, v start at 0,
    v_hat = v / (1 - b2^t)           so early averages are too small
    p -= lr * wd * p                 decoupled weight decay (AdamW); wd=0 -> Adam
    p -= lr * m_hat / (sqrt(v_hat) + eps)

The step is ~lr in size whatever the gradient's scale (m_hat/sqrt(v_hat) is
roughly a signed, normalised gradient). Weight decay is applied to the
weights directly, NOT added to g: inside Adam an L2 term would be divided by
sqrt(v_hat) too, so heavily-updated weights would barely be regularised.

By default only matrices (ndim >= 2: Linear W, embeddings) are decayed;
biases and LayerNorm gamma/beta are not (shrinking gamma toward 0 makes no sense).
Derivations: docs/06_training.md §1
"""

import numpy as np

from src.core.backend import like
from src.core.parameter import Parameter


class AdamW:
    def __init__(
        self,
        named_params,                          # iterable of (name, Parameter), e.g. model.named_parameters()
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.named = list(named_params)
        if len({id(p) for _, p in self.named}) != len(self.named):
            raise ValueError("duplicate parameters")
        self.lr = lr                           # public: a schedule may change it every step
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {n: np.zeros_like(p.data) for n, p in self.named}
        self.v = {n: np.zeros_like(p.data) for n, p in self.named}

    @staticmethod
    def decays(p: Parameter) -> bool:
        return p.data.ndim >= 2

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t          # bias-correction denominators
        bc2 = 1.0 - self.b2 ** self.t
        for name, p in self.named:
            g, m, v = p.grad, self.m[name], self.v[name]
            m *= self.b1
            m += (1.0 - self.b1) * g           # in place: keeps dtype and buffers
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            if self.weight_decay and self.decays(p):
                p.data *= 1.0 - self.lr * self.weight_decay
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

    def zero_grad(self) -> None:
        for _, p in self.named:
            p.zero_grad()

    # --------------------------------------------- checkpointing (Phase 7 resume)
    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "m": {n: a.copy() for n, a in self.m.items()},
            "v": {n: a.copy() for n, a in self.v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if state["m"].keys() != self.m.keys():
            raise KeyError("optimizer state does not match parameters")
        self.t = int(state["t"])
        for n in self.m:
            self.m[n][...] = like(state["m"][n], self.m[n])      # checkpoint arrays are NumPy
            self.v[n][...] = like(state["v"][n], self.v[n])
