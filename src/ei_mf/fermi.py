"""Fermi functions in meV/K units."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .units import KB_MEV_PER_K

Array = NDArray[np.float64]


def fermi_mev(x_mev: Array, temperature_K: float) -> Array:
    x = np.asarray(x_mev, dtype=np.float64)
    if temperature_K <= 0:
        return (x < 0.0).astype(np.float64)
    y = x / (KB_MEV_PER_K * float(temperature_K))
    return 1.0 / (np.exp(np.clip(y, -700.0, 700.0)) + 1.0)
