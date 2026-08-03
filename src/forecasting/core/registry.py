"""A tiny registry enabling Open/Closed extension.

Components register themselves by name; the pipeline looks them up from config.
Adding a new model (ARIMA, Prophet, TFT...) means writing a class and decorating
it with @register_model("name") — no changes to the pipeline or selector.
"""
from __future__ import annotations

from typing import Callable, Type

_MODEL_REGISTRY: dict[str, Type] = {}


def register_model(name: str) -> Callable:
    def _wrap(cls):
        _MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls
    return _wrap


def get_model(name: str):
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name]


def available_models() -> list[str]:
    return list(_MODEL_REGISTRY)