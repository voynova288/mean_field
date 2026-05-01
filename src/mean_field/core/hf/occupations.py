from __future__ import annotations

import numpy as np


def find_chemical_potential(energies: np.ndarray, filling_fraction: float) -> float:
    flattened = np.sort(np.ravel(energies))
    occupancies = np.arange(1, flattened.size + 1, dtype=float) / float(flattened.size)
    idx = 0
    while idx < flattened.size - 1 and filling_fraction > occupancies[idx]:
        idx += 1
    if idx < flattened.size - 1:
        return float((flattened[idx + 1] + flattened[idx]) / 2.0)
    return float(flattened[idx])


def occupied_state_linear_indices(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    flattened = np.ravel(np.asarray(energies, dtype=float), order="F")
    if total_occupied <= 0:
        return np.empty(0, dtype=int)
    if total_occupied >= flattened.size:
        return np.arange(flattened.size, dtype=int)
    # Match Julia's column-major `sortperm` tie-breaking for near-degenerate occupancies.
    return np.argsort(flattened, kind="stable")[:total_occupied]


def occupied_state_mask(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    occupied = occupied_state_linear_indices(energies, total_occupied)
    mask = np.zeros(energies.size, dtype=bool)
    mask[occupied] = True
    return mask.reshape(energies.shape, order="F")


def calculate_norm_convergence(updated_density: np.ndarray, previous_density: np.ndarray) -> float:
    numerator = float(np.linalg.norm(previous_density - updated_density))
    denominator = float(np.linalg.norm(updated_density))
    if denominator < 1e-15:
        return 0.0 if numerator < 1e-15 else float("inf")
    return numerator / denominator
