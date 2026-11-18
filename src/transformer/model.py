"""Decoder-only Transformer language model (GPT-style, pre-LN).

    ids [B,T] -> h = s * TokenEmb(ids) + Pos[:T]  [B,T,D]   s = sqrt(D) if scale_emb else 1
              -> N x TransformerBlock              [B,T,D]
              -> LN_f                              [B,T,D]   pre-LN never normalises the stream
              -> logits = h E^T  (tied)            [B,T,V]   or h W_lm (untied)
Backward walks the same chain in reverse. With tied weights E is used twice
(lookup at the bottom, projection at the top), so E.grad is the SUM of both
contributions -- the reason Parameter.grad accumulates with +=.
scale_emb (Vaswani et al. 2017, sec. 3.4): token rows start at std 0.02 (norm ~0.16)
but every sinusoidal row has norm sqrt(D/2) (5.66 at D=64): positions would drown
the token identity. Multiplying the token embedding by sqrt(D) balances the two.
Backward: d(s*e)/de = s, so the lookup gradient is s * dh.
Derivations: docs/05_transformer.md §3
"""

from dataclasses import asdict, dataclass

import numpy as np

from src.core import initialization as init
from src.core.backend import xp_of
from src.core.module import Module
from src.nn.embedding import Embedding
from src.nn.layer_norm import LayerNorm
from src.nn.linear import Linear
from src.nn.positional import SinusoidalPositionalEncoding
from src.transformer.block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 128             # context length T_max
    d_ff: int | None = None        # default 4 * d_model
    pos: str = "sinusoidal"        # "sinusoidal" | "learned"   (Experiment D)
    tie_weights: bool = True
    scale_emb: bool = False        # multiply token embeddings by sqrt(d_model) (Phase 8 fix)

    def to_dict(self) -> dict:
        return asdict(self)


class GPT(Module):
    def __init__(self, cfg: GPTConfig, rng: np.random.Generator, dtype=init.DEFAULT_DTYPE):
        if cfg.pos not in ("sinusoidal", "learned"):
            raise ValueError(f"unknown pos={cfg.pos!r}")
        self.cfg = cfg
        D, V = cfg.d_model, cfg.vocab_size
        self.tok_emb = Embedding(V, D, rng, dtype=dtype)
        if cfg.pos == "learned":
            self.pos_emb = Embedding(cfg.max_len, D, rng, dtype=dtype)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(cfg.max_len, D, dtype=dtype)
        res_std = init.gpt2_residual_std(cfg.n_layers)          # W_o and fc2 in every block
        self.blocks = [
            TransformerBlock(D, cfg.n_heads, rng, d_ff=cfg.d_ff, out_std=res_std, dtype=dtype)
            for _ in range(cfg.n_layers)
        ]
        self.ln_f = LayerNorm(D, dtype=dtype)
        if not cfg.tie_weights:
            self.lm_head = Linear(D, V, rng, bias=False, dtype=dtype)
        self._h = None                                           # ln_f output, for the tied dE
        self._emb_scale = float(np.sqrt(D))

    def astype(self, dtype) -> "GPT":
        super().astype(dtype)
        if self.cfg.pos == "sinusoidal":
            self.pos_enc.astype(dtype)                           # buffer, not a Parameter
        return self

    # ------------------------------------------------------------------ forward
    def forward(self, ids: np.ndarray) -> np.ndarray:
        T = ids.shape[-1]
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.cfg.max_len}")
        h = self.tok_emb(ids)                                    # [B,T,D]
        if self.cfg.scale_emb:
            h = h * self._emb_scale
        if self.cfg.pos == "learned":
            h = h + self.pos_emb(xp_of(h).arange(T))             # [T,D] broadcast over B
        else:
            h = self.pos_enc(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        if self.cfg.tie_weights:
            self._h = h
            return h @ self.tok_emb.E.data.T                     # [B,T,V]
        return self.lm_head(h)

    # ----------------------------------------------------------------- backward
    def backward(self, dlogits: np.ndarray) -> None:
        if self.cfg.tie_weights:
            E = self.tok_emb.E
            D = E.shape[1]
            d2 = dlogits.reshape(-1, dlogits.shape[-1])          # [B*T, V]
            E.grad += d2.T @ self._h.reshape(-1, D)              # top use of E: dE = dlogits^T h
            dh = dlogits @ E.data                                # [B,T,D]
        else:
            dh = self.lm_head.backward(dlogits)
        dh = self.ln_f.backward(dh)
        for block in reversed(self.blocks):
            dh = block.backward(dh)
        if self.cfg.pos == "learned":
            T, D = dh.shape[-2:]                                 # same row used by every batch item
            self.pos_emb.backward(dh.reshape(-1, T, D).sum(axis=0))
        if self.cfg.scale_emb:
            dh = dh * self._emb_scale                            # chain rule through h = s * e
        self.tok_emb.backward(dh)                                # bottom use of E: scatter-add (+=)
        return None                                              # ids are discrete: no input grad
