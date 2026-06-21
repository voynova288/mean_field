from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StateLabel:
    """System-agnostic label for a projected one-particle state."""

    index: int
    spin: str | int | None = None
    valley: int | None = None
    band: int | None = None
    sector: int | None = None
    fold: int | None = None
    layer: int | None = None
    sublattice: str | None = None
    active: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderParameterResult:
    """Reusable container for scalar/array/table order-parameter diagnostics."""

    scalars: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    classification: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = ["OrderParameterResult", "StateLabel"]
