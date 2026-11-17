"""Module: base class for every layer.

Contract (TECHNICAL_DOC.md §2):
    y  = m.forward(x)     caches what backward needs
    dx = m.backward(dy)   ACCUMULATES (+=) into each param.grad, returns dL/dx

Parameters and sub-modules are discovered automatically from attributes
(Parameter, Module, or list/tuple of Modules). Shared Parameters (e.g. tied
embeddings) are yielded once.
"""

from typing import Iterator

import numpy as np

from src.core.parameter import Parameter


class Module:
    training: bool = True

    # ------------------------------------------------------------ contract
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------ registry
    def _children(self) -> Iterator[tuple[str, object]]:
        """Yield (name, Parameter | Module) for direct attributes, in definition order."""
        for name, value in vars(self).items():
            if isinstance(value, (Parameter, Module)):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, (Parameter, Module)):
                        yield f"{name}.{i}", item

    def named_parameters(self, prefix: str = "", _seen: set | None = None) -> Iterator[tuple[str, Parameter]]:
        seen = set() if _seen is None else _seen
        for name, child in self._children():
            full = f"{prefix}{name}"
            if isinstance(child, Parameter):
                if id(child) not in seen:            # dedupe tied weights
                    seen.add(id(child))
                    yield full, child
            else:
                yield from child.named_parameters(prefix=f"{full}.", _seen=seen)

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator["Module"]:
        yield self
        for _, child in self._children():
            if isinstance(child, Module):
                yield from child.modules()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    # ------------------------------------------------------------ utilities
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def astype(self, dtype) -> "Module":
        """Convert all parameters (float32 for training, float64 for grad checks)."""
        for p in self.parameters():
            p.astype(dtype)
        return self

    # ------------------------------------------------------------ (de)serialisation
    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        missing = params.keys() - state.keys()
        unexpected = state.keys() - params.keys()
        if missing or unexpected:
            raise KeyError(f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for name, p in params.items():
            if state[name].shape != p.shape:
                raise ValueError(f"{name}: shape {state[name].shape} != {p.shape}")
            p.data[...] = state[name]   # in place: optimizer refs stay valid
