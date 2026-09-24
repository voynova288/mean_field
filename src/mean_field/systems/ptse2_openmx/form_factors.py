"""Finite-Q form factors from active reciprocal coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .reciprocal import ActiveReciprocalStates


@dataclass(frozen=True)
class ReciprocalShiftMatch:
    """Intersection implementing G_target = G_source + integer_shift."""

    integer_shift: np.ndarray
    target_indices: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        shift = np.asarray(self.integer_shift, dtype=np.int64)
        target = np.asarray(self.target_indices, dtype=np.int64)
        source = np.asarray(self.source_indices, dtype=np.int64)
        if shift.shape != (3,):
            raise ValueError("integer_shift must have shape (3,)")
        if target.ndim != 1 or source.ndim != 1 or target.shape != source.shape:
            raise ValueError("target/source match indices must be equal-length vectors")
        object.__setattr__(self, "integer_shift", shift)
        object.__setattr__(self, "target_indices", target)
        object.__setattr__(self, "source_indices", source)

    @property
    def match_count(self) -> int:
        return int(self.target_indices.size)


@dataclass(frozen=True)
class FormFactorResult:
    values: np.ndarray
    match: ReciprocalShiftMatch


def match_reciprocal_shift(
    target: ActiveReciprocalStates,
    source: ActiveReciprocalStates,
    integer_shift: Iterable[int],
) -> ReciprocalShiftMatch:
    """Find the zero-filled support intersection for one physical reciprocal shift."""

    shift = _strict_integer_shift(integer_shift)
    shifted_source = source.grid.integer_indices + shift
    target_keys = _structured_rows(target.grid.integer_indices)
    source_keys = _structured_rows(shifted_source)
    _, target_indices, source_indices = np.intersect1d(
        target_keys,
        source_keys,
        assume_unique=True,
        return_indices=True,
    )
    return ReciprocalShiftMatch(
        integer_shift=shift,
        target_indices=target_indices,
        source_indices=source_indices,
    )


def form_factor(
    target: ActiveReciprocalStates,
    source: ActiveReciprocalStates,
    integer_shift: Iterable[int],
) -> FormFactorResult:
    """Compute Lambda[target band, source band] with non-cyclic zero fill.

    The integer shift is ``G_f(target, source) + g_local``.  The reduced
    continuous momentum is already represented by ``k_target-k_source``.
    """

    if target.spin_count != source.spin_count:
        raise ValueError("target and source spin counts differ")
    if target.source_id != source.source_id:
        raise ValueError("target and source reciprocal states have different source IDs")
    if not np.array_equal(target.reciprocal_bohr, source.reciprocal_bohr):
        raise ValueError("target and source reciprocal bases differ")
    match = match_reciprocal_shift(target, source, integer_shift)
    target_values = target.coefficients[match.target_indices]
    source_values = source.coefficients[match.source_indices]
    target_matrix = target_values.reshape(-1, target.band_count)
    source_matrix = source_values.reshape(-1, source.band_count)
    values = target_matrix.conj().T @ source_matrix
    return FormFactorResult(values=values, match=match)


def reverse_form_factor_residual(
    target: ActiveReciprocalStates,
    source: ActiveReciprocalStates,
    integer_shift: Iterable[int],
) -> float:
    """Return max|Lambda_ts(shift)^dagger-Lambda_st(-shift)|."""

    shift = _strict_integer_shift(integer_shift)
    forward = form_factor(target, source, shift).values
    reverse = form_factor(source, target, -shift).values
    if forward.shape[::-1] != reverse.shape:
        raise RuntimeError("reverse form-factor dimensions are inconsistent")
    return float(np.max(np.abs(forward.conj().T - reverse), initial=0.0))


def _strict_integer_shift(integer_shift: Iterable[int]) -> np.ndarray:
    raw = np.asarray(tuple(integer_shift))
    if raw.shape != (3,):
        raise ValueError("integer_shift must contain three components")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("integer_shift components must use an integer dtype")
    return np.asarray(raw, dtype=np.int64)


def _structured_rows(rows: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(rows, dtype=np.int64)
    dtype = np.dtype([("g0", np.int64), ("g1", np.int64), ("g2", np.int64)])
    return contiguous.view(dtype).reshape(-1)
