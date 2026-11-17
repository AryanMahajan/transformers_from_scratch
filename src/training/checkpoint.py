"""Checkpointing: save a Trainer to ONE .npz file and resume it bit-identically.

What the future of a run depends on (all saved):
    model config  + weights          -> rebuild the network
    AdamW m, v, t                     -> same adaptive step sizes (t drives bias correction)
    step                              -> same LR schedule position
    batch RNG + eval RNG states       -> same future batches (forgetting this = silent drift)
    train config, tokenizer, history  -> same loop settings, decode text, continuous plots
Layout inside the .npz (no pickle: loads with allow_pickle=False):
    param/<name>, adam_m/<name>, adam_v/<name>   float arrays
    meta                                          JSON string with everything else
Written to <path>.tmp then renamed: a crash mid-save never corrupts the old checkpoint.
Notes: docs/07_real_training.md §1
"""

import json
import os
from pathlib import Path

import numpy as np

from src.tokenizer.char_tokenizer import CharTokenizer
from src.training.trainer import Trainer, TrainConfig, _empty_history
from src.transformer.model import GPT, GPTConfig

FORMAT_VERSION = 1


def save_checkpoint(path, trainer: Trainer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opt = trainer.opt
    arrays = {}
    for name, p in trainer.model.named_parameters():
        arrays[f"param/{name}"] = p.data
        arrays[f"adam_m/{name}"] = opt.m[name]
        arrays[f"adam_v/{name}"] = opt.v[name]
    meta = {
        "format_version": FORMAT_VERSION,
        "model_config": trainer.model.cfg.to_dict(),
        "train_config": trainer.cfg.to_dict(),
        "tokenizer": trainer.tokenizer.to_dict() if trainer.tokenizer is not None else None,
        "step": trainer.step,
        "adam_t": opt.t,
        "rng_state": trainer.rng.bit_generator.state,          # dict of Python ints: JSON-safe
        "eval_rng_state": trainer.eval_rng.bit_generator.state,
        "history": trainer.history,
        "dtype": str(next(iter(arrays.values())).dtype),
    }
    arrays["meta"] = np.array(json.dumps(meta))                 # 0-d unicode array, no pickle
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:                                  # file handle: savez won't append ".npz"
        np.savez(f, **arrays)
    os.replace(tmp, path)                                       # atomic on the same filesystem


def read_checkpoint(path) -> tuple[dict, dict]:
    """Raw contents: (meta dict, {array name: array})."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {meta.get('format_version')}")
    return meta, arrays


def load_model(path) -> tuple[GPT, CharTokenizer | None, dict]:
    """For inference: (model, tokenizer, meta)."""
    meta, arrays = read_checkpoint(path)
    model = GPT(GPTConfig(**meta["model_config"]), np.random.default_rng(0), dtype=np.dtype(meta["dtype"]))
    model.load_state_dict({k[len("param/"):]: v for k, v in arrays.items() if k.startswith("param/")})
    tok = CharTokenizer.from_dict(meta["tokenizer"]) if meta["tokenizer"] else None
    return model, tok, meta


def load_trainer(path, train_data, val_data, log=print) -> Trainer:
    """Rebuild a Trainer exactly as it was when saved, ready for .run()."""
    meta, arrays = read_checkpoint(path)
    model, tok, _ = load_model(path)
    tr = Trainer(model, train_data, val_data, TrainConfig(**meta["train_config"]), log=log, tokenizer=tok)
    tr.opt.load_state_dict({
        "t": meta["adam_t"],
        "m": {k[len("adam_m/"):]: v for k, v in arrays.items() if k.startswith("adam_m/")},
        "v": {k[len("adam_v/"):]: v for k, v in arrays.items() if k.startswith("adam_v/")},
    })
    tr.rng.bit_generator.state = meta["rng_state"]
    tr.eval_rng.bit_generator.state = meta["eval_rng_state"]
    tr.step = meta["step"]
    tr.history = {**_empty_history(), **meta["history"]}
    return tr
