"""Gauge-safe helpers for microscopic HTQG two-wall postprocessing.

These helpers do not identify topology.  They only fix the state-gauge,
periodic-position, and valley-label contracts used by accepted-state boundary
analysis.  Crossing counts and bulk-branch lineage remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .microscopic_supercell import (
    BatchAPhysicalLabelSupport,
    MicroscopicBasisLayout,
)

Array = np.ndarray
PauliComponent = Literal["x", "y", "z"]


@dataclass(frozen=True)
class DegenerateSubspaceLocalization:
    """Eigenvectors after wall-difference localization inside energy clusters."""

    vectors: Array
    clusters: tuple[tuple[int, ...], ...]
    rotated_clusters: tuple[tuple[int, ...], ...]
    degeneracy_tolerance_ev: float
    localization_operator: str = "Pi_wall_1_minus_Pi_wall_2"


@dataclass(frozen=True)
class PeriodicCenterResult:
    """Periodic center and RMS distance, with an explicit undefined state."""

    center: float | None
    resultant_magnitude: float
    rms_distance: float | None
    period: float
    resultant_tolerance: float
    status: Literal["defined", "undefined_near_zero_resultant"]


@dataclass(frozen=True)
class PlaneWaveValleyMatching:
    """Cross-valley index map at equal physical plane-wave label."""

    partner_index: Array
    matched_count: int
    unmatched_count: int
    complete: bool


def _energy_clusters(eigenvalues: Array, tolerance_ev: float) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    tolerance = float(tolerance_ev)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy tolerance must be finite and nonnegative")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("eigenvalues must be sorted in nondecreasing order")

    clusters: list[tuple[int, ...]] = []
    start = 0
    for stop in range(1, values.size + 1):
        if stop == values.size or values[stop] - values[start] > tolerance:
            clusters.append(tuple(range(start, stop)))
            start = stop
    return tuple(clusters)


def localize_degenerate_two_wall_subspaces(
    eigenvalues: Array,
    eigenvectors: Array,
    wall_1_minus_wall_2: Array,
    *,
    degeneracy_tolerance_ev: float,
    hermiticity_tolerance: float = 1.0e-12,
) -> DegenerateSubspaceLocalization:
    """Diagonalize ``Pi_W1-Pi_W2`` only inside strictly degenerate clusters.

    Eigenvectors are columns.  Singleton clusters are copied bit-for-bit and
    are never rotated.  The routine deliberately does not use
    ``Pi_W1+Pi_W2``, which cannot distinguish the two walls.
    """

    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    operator = np.asarray(wall_1_minus_wall_2, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[1] != values.size:
        raise ValueError("eigenvectors must have shape (dimension, nstate)")
    if operator.shape != (vectors.shape[0], vectors.shape[0]):
        raise ValueError("wall-difference operator has incompatible shape")
    scale = max(1.0, float(np.max(np.abs(operator), initial=0.0)))
    residual = float(np.max(np.abs(operator - operator.conj().T), initial=0.0))
    tolerance = float(hermiticity_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("hermiticity tolerance must be finite and nonnegative")
    if residual > tolerance * scale:
        raise ValueError("wall-difference operator is not Hermitian")

    clusters = _energy_clusters(values, degeneracy_tolerance_ev)
    localized = vectors.copy()
    rotated: list[tuple[int, ...]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        columns = np.asarray(cluster, dtype=np.int64)
        subspace = vectors[:, columns]
        projected = subspace.conj().T @ operator @ subspace
        projected = 0.5 * (projected + projected.conj().T)
        _wall_eigenvalues, rotation = np.linalg.eigh(projected)
        localized[:, columns] = subspace @ rotation
        rotated.append(cluster)
    return DegenerateSubspaceLocalization(
        vectors=localized,
        clusters=clusters,
        rotated_clusters=tuple(rotated),
        degeneracy_tolerance_ev=float(degeneracy_tolerance_ev),
    )


def periodic_center_and_rms(
    positions: Array,
    weights: Array,
    *,
    period: float,
    resultant_tolerance: float,
) -> PeriodicCenterResult:
    """Return a circular center, or explicitly mark it undefined.

    The resultant is normalized by total weight.  If it is at or below the
    declared tolerance, both the center and every center-dependent length are
    returned as ``None`` rather than selecting an arbitrary angle.
    """

    x = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != w.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("positions and weights must be equal nonempty vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("positions and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("periodic-center weights must be nonnegative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("periodic-center weights must have positive sum")
    resolved_period = float(period)
    if not np.isfinite(resolved_period) or resolved_period <= 0.0:
        raise ValueError("period must be finite and positive")
    threshold = float(resultant_tolerance)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("resultant tolerance must lie in [0,1]")

    moment = np.sum(w * np.exp(2.0j * np.pi * x / resolved_period)) / total
    resultant = float(abs(moment))
    if resultant <= threshold:
        return PeriodicCenterResult(
            center=None,
            resultant_magnitude=resultant,
            rms_distance=None,
            period=resolved_period,
            resultant_tolerance=threshold,
            status="undefined_near_zero_resultant",
        )
    center = float(
        (np.angle(moment) * resolved_period / (2.0 * np.pi)) % resolved_period
    )
    displacement = (x - center + 0.5 * resolved_period) % resolved_period
    displacement -= 0.5 * resolved_period
    rms = float(np.sqrt(np.sum(w * displacement * displacement) / total))
    return PeriodicCenterResult(
        center=center,
        resultant_magnitude=resultant,
        rms_distance=rms,
        period=resolved_period,
        resultant_tolerance=threshold,
        status="defined",
    )


def apply_raw_index_valley_pauli(
    vectors: Array,
    layout: MicroscopicBasisLayout,
    component: PauliComponent,
) -> Array:
    """Apply Pauli matrices on the stored valley *array index*.

    This is intentionally named ``raw_index_valley_pauli``.  Its transverse
    components pair equal ``(fold,g)`` indices and are not physical-envelope
    valley coherence operators unless an independent momentum-sewing proof is
    supplied.
    """

    values = np.asarray(vectors, dtype=np.complex128)
    if values.shape[0] != layout.dimension:
        raise ValueError("vectors have an incompatible leading dimension")
    trailing = values.shape[1:]
    shaped = values.reshape(layout.shape + trailing)
    out = np.empty_like(shaped)
    plus = shaped[:, :, :, :, 0, ...]
    minus = shaped[:, :, :, :, 1, ...]
    if component == "x":
        out[:, :, :, :, 0, ...] = minus
        out[:, :, :, :, 1, ...] = plus
    elif component == "y":
        out[:, :, :, :, 0, ...] = -1.0j * minus
        out[:, :, :, :, 1, ...] = 1.0j * plus
    elif component == "z":
        out[:, :, :, :, 0, ...] = plus
        out[:, :, :, :, 1, ...] = -minus
    else:
        raise ValueError("valley Pauli component must be 'x', 'y', or 'z'")
    return out.reshape(values.shape)


def build_physical_plane_wave_valley_matching(
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    *,
    reduced_k: int,
) -> PlaneWaveValleyMatching:
    """Pair valleys at equal support-resolved physical labels for one k point."""

    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("physical-label support does not match the basis layout")
    if type(reduced_k) is not int:
        raise TypeError("reduced_k must be an exact integer")
    resolved_k = reduced_k
    if not 0 <= resolved_k < support.labels.shape[0]:
        raise IndexError("reduced_k is outside the physical-label support")

    lookup: list[dict[tuple[int, int, int, int, int], int]] = [{}, {}]
    for index in range(layout.dimension):
        fold, g, layer, sublattice, valley, spin = layout.unravel_index(index)
        n1, n2 = support.physical_plane_wave_label(
            reduced_k=resolved_k,
            fold=fold,
            g=g,
            valley=valley,
        )
        key = (int(n1), int(n2), layer, sublattice, spin)
        if key in lookup[valley]:
            raise RuntimeError("physical plane-wave labels are not unique")
        lookup[valley][key] = index

    partner = np.full(layout.dimension, -1, dtype=np.int64)
    for valley in range(2):
        opposite = 1 - valley
        for key, index in lookup[valley].items():
            partner[index] = lookup[opposite].get(key, -1)
    matched = int(np.count_nonzero(partner >= 0))
    unmatched = int(layout.dimension - matched)
    if matched:
        indices = np.flatnonzero(partner >= 0)
        if not np.array_equal(partner[partner[indices]], indices):
            raise RuntimeError("physical plane-wave valley matching is not involutive")
    return PlaneWaveValleyMatching(
        partner_index=partner,
        matched_count=matched,
        unmatched_count=unmatched,
        complete=unmatched == 0,
    )


def apply_physical_plane_wave_matched_valley_pauli(
    vectors: Array,
    layout: MicroscopicBasisLayout,
    support: BatchAPhysicalLabelSupport,
    component: Literal["x", "y"],
    *,
    reduced_k: int,
    require_complete: bool = True,
) -> Array:
    """Apply a transverse valley Pauli operator using physical-label matching.

    Missing partners are rejected by default.  Allowing an incomplete map is a
    diagnostic projection and must not be reported as a full physical valley
    observable.
    """

    values = np.asarray(vectors, dtype=np.complex128)
    if values.shape[0] != layout.dimension:
        raise ValueError("vectors have an incompatible leading dimension")
    if component not in {"x", "y"}:
        raise ValueError("matched transverse valley component must be 'x' or 'y'")
    matching = build_physical_plane_wave_valley_matching(
        layout,
        support,
        reduced_k=reduced_k,
    )
    if require_complete and not matching.complete:
        raise ValueError(
            "physical plane-wave valley matching is incomplete: "
            f"{matching.unmatched_count} unmatched basis states"
        )
    out = np.zeros_like(values)
    for output_index, input_index in enumerate(matching.partner_index):
        if input_index < 0:
            continue
        _fold, _g, _layer, _sub, output_valley, _spin = layout.unravel_index(
            output_index
        )
        if component == "x":
            factor = 1.0 + 0.0j
        else:
            factor = -1.0j if output_valley == 0 else 1.0j
        out[output_index] = factor * values[int(input_index)]
    return out


__all__ = [
    "DegenerateSubspaceLocalization",
    "PeriodicCenterResult",
    "PlaneWaveValleyMatching",
    "apply_physical_plane_wave_matched_valley_pauli",
    "apply_raw_index_valley_pauli",
    "build_physical_plane_wave_valley_matching",
    "localize_degenerate_two_wall_subspaces",
    "periodic_center_and_rms",
]
