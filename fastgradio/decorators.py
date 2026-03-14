from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FunctionMeta:
    compute_type: Literal["gpu", "cpu"] | None = None
    device: int | None = None
    concurrency_limit: int | None = None
    batch_size: int | None = None
    batch_timeout: float | None = None
    api_name: str | None = None
    api_method: str = "POST"
    is_generator: bool = False


def _get_or_create_meta(func) -> FunctionMeta:
    if not hasattr(func, "_fastgradio_meta"):
        func._fastgradio_meta = FunctionMeta()
    return func._fastgradio_meta


def _detect_generator(func) -> bool:
    return inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func)
