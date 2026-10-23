"""Registry for interchangeable vision/text encoder pairs.

Adapters expose dim, encode_video(B,T,3,H,W)->(B,T,D),
encode_text(list[K])->(K,L,D), freeze_vision() and freeze_text().
Prompt strings remain configurable outside encoders.
"""
from collections.abc import Callable
from typing import Any

import torch.nn as nn


class BackboneRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[Any], nn.Module]] = {}

    def register(self, name: str, factory: Callable[[Any], nn.Module]) -> None:
        if name in self._factories:
            raise ValueError(f"Backbone provider already registered: {name}")
        self._factories[name] = factory

    def build(self, name: str, cfg: Any) -> nn.Module:
        if name not in self._factories:
            raise ValueError(f"Unknown backbone provider: {name}")
        return self._factories[name](cfg)


backbones = BackboneRegistry()
