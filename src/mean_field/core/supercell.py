from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntegerSupercell:
    """System-agnostic integer moire supercell convention.

    Real-space supercell vectors are
    ``Rs1 = n11 R1 + n12 R2`` and ``Rs2 = n21 R1 + n22 R2``.
    If primitive reciprocal vectors are ``b1,b2``, then
    ``G1 = (n22 b1 - n21 b2) / Nc`` and
    ``G2 = (n11 b2 - n12 b1) / Nc`` with ``Nc = det(n)``.
    """

    n11: int
    n12: int
    n21: int
    n22: int

    @property
    def area_ratio(self) -> int:
        det = int(self.n11 * self.n22 - self.n12 * self.n21)
        if det <= 0:
            raise ValueError(f"Expected positive supercell determinant, got {det} for {self}")
        return det

    def reciprocal_vectors(self, b1: complex, b2: complex) -> tuple[complex, complex]:
        nc = float(self.area_ratio)
        g1 = (self.n22 * complex(b1) - self.n21 * complex(b2)) / nc
        g2 = (self.n11 * complex(b2) - self.n12 * complex(b1)) / nc
        return complex(g1), complex(g2)

    def primitive_shift_to_supercell(self, dm: int, dn: int) -> tuple[int, int]:
        """Coordinates of ``dm*b1 + dn*b2`` in the supercell reciprocal basis."""

        return (int(dm * self.n11 + dn * self.n12), int(dm * self.n21 + dn * self.n22))

    def as_dict(self) -> dict[str, int]:
        return {
            "n11": int(self.n11),
            "n12": int(self.n12),
            "n21": int(self.n21),
            "n22": int(self.n22),
            "area_ratio": int(self.area_ratio),
        }


def folded_band_count(primitive_band_count: int, area_ratio: int) -> int:
    primitive_band_count = int(primitive_band_count)
    area_ratio = int(area_ratio)
    if primitive_band_count <= 0:
        raise ValueError(f"primitive_band_count must be positive, got {primitive_band_count}")
    if area_ratio <= 0:
        raise ValueError(f"area_ratio must be positive, got {area_ratio}")
    return primitive_band_count * area_ratio


def fixed_sector_occupation_counts(
    *,
    n_spin: int,
    n_eta: int,
    default_count: int,
    overrides: Mapping[tuple[int, int], int] | None = None,
    n_band: int | None = None,
) -> np.ndarray:
    """Build integer occupation counts for spin/valley fixed-sector HF.

    The returned array has shape ``(n_spin, n_eta)``.  ``default_count`` is used
    for every sector and entries in ``overrides`` replace selected
    ``(spin, valley)`` sectors.  If ``n_band`` is given, counts are checked to
    lie in ``[0, n_band]``.
    """

    n_spin = int(n_spin)
    n_eta = int(n_eta)
    if n_spin <= 0 or n_eta <= 0:
        raise ValueError(f"Expected positive n_spin/n_eta, got {(n_spin, n_eta)}")
    occ = np.full((n_spin, n_eta), int(default_count), dtype=int)
    for key, value in (overrides or {}).items():
        ispin, ieta = (int(key[0]), int(key[1]))
        if not (0 <= ispin < n_spin and 0 <= ieta < n_eta):
            raise ValueError(f"Occupation override sector {(ispin, ieta)} outside {(n_spin, n_eta)}")
        occ[ispin, ieta] = int(value)
    if n_band is not None:
        n_band = int(n_band)
        if np.any(occ < 0) or np.any(occ > n_band):
            raise ValueError(f"Occupation counts must lie in [0, {n_band}], got {occ.tolist()}")
    return occ


def reference_diagonal_array(reference_diagonal: np.ndarray | float, *, n_band: int | None = None) -> np.ndarray:
    """Normalize a scalar or per-band reference occupation vector."""

    values = np.asarray(reference_diagonal, dtype=float)
    if values.ndim == 0:
        if n_band is None:
            raise ValueError("n_band is required when reference_diagonal is scalar")
        return np.full((int(n_band),), float(values), dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected scalar or 1D reference_diagonal, got shape {values.shape}")
    if n_band is not None and values.shape != (int(n_band),):
        raise ValueError(f"Expected reference_diagonal shape {(int(n_band),)}, got {values.shape}")
    return np.asarray(values, dtype=float)


def primitive_filling_from_occupation_counts(
    occupation_counts: np.ndarray,
    *,
    reference_diagonal: np.ndarray | float,
    area_ratio: int,
    n_band: int | None = None,
) -> float:
    """Return primitive-cell filling from folded-cell sector occupations.

    Formula: ``nu = (N_occ - N_ref) / Nc``.  ``occupation_counts`` contains the
    number of occupied folded bands in each spin/valley sector at one supercell
    k point.  ``reference_diagonal`` gives the reference occupation of folded
    bands in one sector.
    """

    counts = np.asarray(occupation_counts, dtype=int)
    reference = reference_diagonal_array(reference_diagonal, n_band=n_band)
    reference_total = float(counts.size) * float(np.sum(reference))
    occupied_total = float(np.sum(counts))
    return float(occupied_total - reference_total) / float(area_ratio)


def occupied_count_from_primitive_filling(
    primitive_filling: float,
    *,
    reference_diagonal: np.ndarray | float,
    area_ratio: int,
    n_sector: int,
    n_band: int | None = None,
    atol: float = 1.0e-9,
) -> int:
    """Integer occupied folded states implied by a primitive filling."""

    reference = reference_diagonal_array(reference_diagonal, n_band=n_band)
    raw = float(n_sector) * float(np.sum(reference)) + float(area_ratio) * float(primitive_filling)
    rounded = int(round(raw))
    if abs(raw - rounded) > float(atol):
        raise ValueError(
            f"primitive_filling={primitive_filling} gives non-integer supercell occupation {raw}"
        )
    return rounded


def folded_reference_diagonal_by_primitive_index(
    projected_indices: tuple[int, ...],
    *,
    target_band_index: int,
    folds_per_primitive: int,
    lower_reference: float = 1.0,
    target_reference: float = 0.0,
    upper_reference: float = 0.0,
) -> np.ndarray:
    """Reference occupations for folded bands selected by primitive band index."""

    folds_per_primitive = int(folds_per_primitive)
    if folds_per_primitive <= 0:
        raise ValueError(f"folds_per_primitive must be positive, got {folds_per_primitive}")
    target = int(target_band_index)
    values: list[float] = []
    for index in tuple(int(value) for value in projected_indices):
        if index < target:
            ref = float(lower_reference)
        elif index == target:
            ref = float(target_reference)
        else:
            ref = float(upper_reference)
        values.extend([ref] * folds_per_primitive)
    return np.asarray(values, dtype=float)


def folded_indices_for_primitive_band(
    projected_indices: tuple[int, ...],
    *,
    target_band_index: int,
    folds_per_primitive: int,
) -> tuple[int, ...]:
    indices = tuple(int(value) for value in projected_indices)
    target = int(target_band_index)
    if target not in indices:
        raise ValueError(f"target_band_index={target} is not present in projected_indices={indices}")
    folds_per_primitive = int(folds_per_primitive)
    if folds_per_primitive <= 0:
        raise ValueError(f"folds_per_primitive must be positive, got {folds_per_primitive}")
    start = indices.index(target) * folds_per_primitive
    return tuple(range(start, start + folds_per_primitive))


__all__ = [
    "IntegerSupercell",
    "fixed_sector_occupation_counts",
    "folded_band_count",
    "folded_indices_for_primitive_band",
    "folded_reference_diagonal_by_primitive_index",
    "occupied_count_from_primitive_filling",
    "primitive_filling_from_occupation_counts",
    "reference_diagonal_array",
]
