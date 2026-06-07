"""Array-only full-flavor IVC contract helpers for Polshyn tMBG.

This module is intentionally lightweight: it defines labels, density-layout
round trips, filling counts, and projector/order-parameter diagnostics needed
before any full-flavor Hartree-Fock updater is trusted.  It does not build a
TMBG Hamiltonian, run self-consistency, compute topology, or calibrate the
paper displacement field.

Evidence/contract sources in the project workspace:

- ``reports/polshyn2021_orbital_ivc_no_scout_physics_understanding_20260605.md``
- ``plans/polshyn2021_orbital_ivc_no_scout_reproduction_contract_20260605.md``
- ``reports/polshyn2021_orbital_ivc_lane_status_20260605.md``

Main uncertainty carried here: phase winding and TRS diagnostics are placeholders
until a sewing/Wilson-link convention is implemented on actual wavefunctions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...core.hf import conventional_projector_to_stored, stored_projector_to_conventional

SPIN_LABELS: tuple[str, str] = ("up", "down")
VALLEY_VALUES: tuple[int, int] = (1, -1)
VALLEY_LABELS: tuple[str, str] = ("K", "Kprime")


@dataclass(frozen=True)
class FullFlavorLabel:
    """One flattened full-flavor label ``a=(spin, valley, band/fold)``."""

    index: int
    spin_index: int
    spin_label: str
    valley_index: int
    valley: int
    valley_label: str
    band_index: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": int(self.index),
            "spin_index": int(self.spin_index),
            "spin_label": self.spin_label,
            "valley_index": int(self.valley_index),
            "valley": int(self.valley),
            "valley_label": self.valley_label,
            "band_index": int(self.band_index),
        }


@dataclass(frozen=True)
class FullFlavorLayout:
    """Full ``(spin, valley, band/fold)`` layout for valley-coherent densities.

    The flat Wang-compatible index follows the Polshyn no-scout contract:

    ``idx(s, eta, mu) = s + n_spin * (eta + n_valley * mu)``.
    """

    n_band: int
    n_spin: int = 2
    n_valley: int = 2
    spin_labels: tuple[str, ...] = SPIN_LABELS
    valley_values: tuple[int, ...] = VALLEY_VALUES
    valley_labels: tuple[str, ...] = VALLEY_LABELS

    def __post_init__(self) -> None:
        if int(self.n_spin) <= 0 or int(self.n_valley) <= 0 or int(self.n_band) <= 0:
            raise ValueError("n_spin, n_valley, and n_band must be positive")
        if len(self.spin_labels) != int(self.n_spin):
            raise ValueError(f"spin_labels length {len(self.spin_labels)} incompatible with n_spin={self.n_spin}")
        if len(self.valley_values) != int(self.n_valley):
            raise ValueError(f"valley_values length {len(self.valley_values)} incompatible with n_valley={self.n_valley}")
        if len(self.valley_labels) != int(self.n_valley):
            raise ValueError(f"valley_labels length {len(self.valley_labels)} incompatible with n_valley={self.n_valley}")

    @property
    def nt(self) -> int:
        return int(self.n_spin) * int(self.n_valley) * int(self.n_band)

    def full_density_shape(self, nk: int) -> tuple[int, int, int, int, int, int, int]:
        """Return the unflattened full-density shape for ``nk`` k points."""

        return (
            int(self.n_spin),
            int(self.n_valley),
            int(self.n_band),
            int(self.n_spin),
            int(self.n_valley),
            int(self.n_band),
            int(nk),
        )

    def flat_index(self, spin_index: int, valley_index: int, band_index: int) -> int:
        s = int(spin_index)
        eta = int(valley_index)
        band = int(band_index)
        if not 0 <= s < int(self.n_spin):
            raise IndexError(f"spin_index={s} outside [0, {self.n_spin})")
        if not 0 <= eta < int(self.n_valley):
            raise IndexError(f"valley_index={eta} outside [0, {self.n_valley})")
        if not 0 <= band < int(self.n_band):
            raise IndexError(f"band_index={band} outside [0, {self.n_band})")
        return int(s + int(self.n_spin) * (eta + int(self.n_valley) * band))

    def label(self, index: int) -> FullFlavorLabel:
        idx = int(index)
        if not 0 <= idx < self.nt:
            raise IndexError(f"flat index={idx} outside [0, {self.nt})")
        s = idx % int(self.n_spin)
        rest = idx // int(self.n_spin)
        eta = rest % int(self.n_valley)
        band = rest // int(self.n_valley)
        return FullFlavorLabel(
            index=idx,
            spin_index=int(s),
            spin_label=self.spin_labels[s],
            valley_index=int(eta),
            valley=int(self.valley_values[eta]),
            valley_label=self.valley_labels[eta],
            band_index=int(band),
        )

    def labels(self) -> tuple[FullFlavorLabel, ...]:
        return tuple(self.label(index) for index in range(self.nt))

    def sector_indices(
        self,
        *,
        spin_index: int | None = None,
        valley_index: int | None = None,
        band_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return flat indices for a spin/valley/band subset."""

        spins = range(int(self.n_spin)) if spin_index is None else (int(spin_index),)
        valleys = range(int(self.n_valley)) if valley_index is None else (int(valley_index),)
        bands = range(int(self.n_band)) if band_indices is None else tuple(int(v) for v in band_indices)
        return np.asarray([self.flat_index(s, eta, band) for band in bands for eta in valleys for s in spins], dtype=int)


def infer_full_flavor_layout_from_blocks(blocks: np.ndarray) -> FullFlavorLayout:
    """Infer a default two-spin/two-valley layout from a full block tensor."""

    arr = np.asarray(blocks)
    if arr.ndim != 7:
        raise ValueError(f"Expected a 7D full-density block tensor, got shape {arr.shape}")
    n_spin, n_valley, n_band, n_spin_rhs, n_valley_rhs, n_band_rhs, _nk = arr.shape
    if (n_spin, n_valley, n_band) != (n_spin_rhs, n_valley_rhs, n_band_rhs):
        raise ValueError(f"Full-density bra/ket dimensions are inconsistent: {arr.shape}")
    return FullFlavorLayout(n_band=int(n_band), n_spin=int(n_spin), n_valley=int(n_valley))


def _coerce_layout_for_blocks(blocks: np.ndarray, layout: FullFlavorLayout | None) -> FullFlavorLayout:
    inferred = infer_full_flavor_layout_from_blocks(blocks)
    if layout is None:
        return inferred
    if (int(layout.n_spin), int(layout.n_valley), int(layout.n_band)) != (
        int(inferred.n_spin),
        int(inferred.n_valley),
        int(inferred.n_band),
    ):
        raise ValueError(f"Layout {(layout.n_spin, layout.n_valley, layout.n_band)} incompatible with blocks {blocks.shape}")
    return layout


def flatten_full_density(blocks: np.ndarray, layout: FullFlavorLayout | None = None) -> np.ndarray:
    """Flatten full density blocks to ``(nt, nt, nk)`` without dropping IVC blocks.

    Input shape is
    ``(spin, valley, band, spin', valley', band', k)``.  Unlike the existing
    sector-block layout, this representation includes ``K-K'`` and other
    off-sector blocks.
    """

    arr = np.asarray(blocks, dtype=np.complex128)
    resolved = _coerce_layout_for_blocks(arr, layout)
    nk = int(arr.shape[-1])
    out = np.zeros((resolved.nt, resolved.nt, nk), dtype=np.complex128)
    for s0 in range(int(resolved.n_spin)):
        for v0 in range(int(resolved.n_valley)):
            for b0 in range(int(resolved.n_band)):
                i = resolved.flat_index(s0, v0, b0)
                for s1 in range(int(resolved.n_spin)):
                    for v1 in range(int(resolved.n_valley)):
                        for b1 in range(int(resolved.n_band)):
                            j = resolved.flat_index(s1, v1, b1)
                            out[i, j, :] = arr[s0, v0, b0, s1, v1, b1, :]
    return out


def unflatten_full_density(flat: np.ndarray, layout: FullFlavorLayout) -> np.ndarray:
    """Unflatten ``(nt, nt, nk)`` density to full spin/valley/band blocks."""

    arr = np.asarray(flat, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(f"Expected flattened density shape (nt, nt, nk), got {arr.shape}")
    nt, nt_rhs, nk = arr.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square flattened density, got {arr.shape}")
    if nt != int(layout.nt):
        raise ValueError(f"Flattened dimension {nt} incompatible with layout nt={layout.nt}")
    out = np.zeros(layout.full_density_shape(nk), dtype=np.complex128)
    for s0 in range(int(layout.n_spin)):
        for v0 in range(int(layout.n_valley)):
            for b0 in range(int(layout.n_band)):
                i = layout.flat_index(s0, v0, b0)
                for s1 in range(int(layout.n_spin)):
                    for v1 in range(int(layout.n_valley)):
                        for b1 in range(int(layout.n_band)):
                            j = layout.flat_index(s1, v1, b1)
                            out[s0, v0, b0, s1, v1, b1, :] = arr[i, j, :]
    return out


def conventional_flat_to_wang_stored(flat: np.ndarray) -> np.ndarray:
    """Convert a conventional full density to Wang/Xiaoyu stored convention.

    The Polshyn contract records the Wang boundary as ``D_store = D*``.  For a
    Hermitian density this is equivalent to a transpose, but using the explicit
    conjugate prevents ambiguity for contract tests.
    """

    return conventional_projector_to_stored(flat)


def wang_stored_to_conventional_flat(stored: np.ndarray) -> np.ndarray:
    """Inverse of :func:`conventional_flat_to_wang_stored`."""

    return stored_projector_to_conventional(stored)


def full_density_to_wang_stored(blocks: np.ndarray, layout: FullFlavorLayout | None = None) -> np.ndarray:
    """Flatten full blocks and convert to Wang/Xiaoyu stored convention."""

    return conventional_flat_to_wang_stored(flatten_full_density(blocks, layout=layout))


def wang_stored_to_full_density(stored: np.ndarray, layout: FullFlavorLayout) -> np.ndarray:
    """Convert Wang/Xiaoyu stored density back to conventional full blocks."""

    return unflatten_full_density(wang_stored_to_conventional_flat(stored), layout)


@dataclass(frozen=True)
class OddIntegerFillingSummary:
    """No-compute occupation contract for Polshyn odd-integer IVC fillings."""

    projected_indices: tuple[int, ...]
    target_band_index: int
    filling_nu: int
    area_ratio: int
    n_spin: int
    n_valley: int
    n_band_slots: int
    target_fold_indices: tuple[int, ...]
    lower_primitive_count: int
    reference_diagonal: np.ndarray
    reference_total_per_k: int
    target_occupied_total_per_k: int
    n_occupied_total_per_k: int
    primitive_nu_from_total: float
    convention: str = "Polshyn conduction-band filling: lower remote filled, target empty"

    @property
    def matches_odd_integer_contract(self) -> bool:
        return bool(
            self.filling_nu % 2 == 1
            and np.isclose(float(self.primitive_nu_from_total), float(self.filling_nu), atol=1.0e-12)
            and self.n_occupied_total_per_k == self.reference_total_per_k + self.target_occupied_total_per_k
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "projected_indices": [int(v) for v in self.projected_indices],
            "target_band_index": int(self.target_band_index),
            "filling_nu": int(self.filling_nu),
            "area_ratio": int(self.area_ratio),
            "n_spin": int(self.n_spin),
            "n_valley": int(self.n_valley),
            "n_band_slots": int(self.n_band_slots),
            "target_fold_indices": [int(v) for v in self.target_fold_indices],
            "lower_primitive_count": int(self.lower_primitive_count),
            "reference_diagonal": [float(v) for v in np.asarray(self.reference_diagonal, dtype=float)],
            "reference_total_per_k": int(self.reference_total_per_k),
            "target_occupied_total_per_k": int(self.target_occupied_total_per_k),
            "n_occupied_total_per_k": int(self.n_occupied_total_per_k),
            "primitive_nu_from_total": float(self.primitive_nu_from_total),
            "matches_odd_integer_contract": bool(self.matches_odd_integer_contract),
            "convention": self.convention,
            "sector_counts_valid_for_ivc": False,
        }


def folded_reference_diagonal(
    projected_indices: Sequence[int],
    target_band_index: int,
    *,
    area_ratio: int = 2,
) -> np.ndarray:
    """Return spinless folded reference occupations for the Polshyn convention."""

    projected = tuple(int(v) for v in projected_indices)
    if not projected:
        raise ValueError("projected_indices must not be empty")
    area = int(area_ratio)
    if area <= 0:
        raise ValueError(f"area_ratio must be positive, got {area_ratio}")
    target = int(target_band_index)
    if target not in projected:
        raise ValueError(f"target_band_index={target} not in projected_indices={projected}")
    values: list[float] = []
    for band in projected:
        ref = 1.0 if int(band) < target else 0.0
        values.extend([ref] * area)
    return np.asarray(values, dtype=float)


def odd_integer_filling_summary(
    projected_indices: Sequence[int],
    target_band_index: int,
    *,
    filling_nu: int,
    area_ratio: int = 2,
    n_spin: int = 2,
    n_valley: int = 2,
) -> OddIntegerFillingSummary:
    """Return the total occupied count for odd-integer Polshyn IVC fillings.

    Formula encoded from the no-scout contract:

    ``N_occ_total = n_spin * n_valley * A * L + A * nu``.

    This is a total full-flavor trace, not a fixed spin/valley sector-count
    matrix.  It is therefore compatible with IVC projectors whose valley
    diagonal occupations are fractional.
    """

    projected = tuple(int(v) for v in projected_indices)
    target = int(target_band_index)
    if target not in projected:
        raise ValueError(f"target_band_index={target} not in projected_indices={projected}")
    nu = int(filling_nu)
    if float(filling_nu) != float(nu):
        raise ValueError(f"filling_nu must be an integer odd filling, got {filling_nu!r}")
    if nu % 2 != 1:
        raise ValueError(f"filling_nu must be odd for this contract, got {nu}")
    if nu < 0 or nu > int(n_spin) * int(n_valley):
        raise ValueError(f"filling_nu={nu} outside target flavor range [0, {int(n_spin) * int(n_valley)}]")
    area = int(area_ratio)
    if area <= 0:
        raise ValueError(f"area_ratio must be positive, got {area_ratio}")
    lower_count = sum(1 for index in projected if int(index) < target)
    target_positions = [pos for pos, index in enumerate(projected) if int(index) == target]
    target_fold_indices = tuple(int(area * pos + fold) for pos in target_positions for fold in range(area))
    reference = folded_reference_diagonal(projected, target, area_ratio=area)
    reference_total = int(round(float(np.sum(reference)) * int(n_spin) * int(n_valley)))
    target_total = int(area * nu)
    occupied_total = int(reference_total + target_total)
    nt = int(n_spin) * int(n_valley) * int(area * len(projected))
    if occupied_total > nt:
        raise ValueError(f"occupied total {occupied_total} exceeds full flavor dimension {nt}")
    primitive_nu = (float(occupied_total) - float(reference_total)) / float(area)
    return OddIntegerFillingSummary(
        projected_indices=projected,
        target_band_index=target,
        filling_nu=nu,
        area_ratio=area,
        n_spin=int(n_spin),
        n_valley=int(n_valley),
        n_band_slots=int(area * len(projected)),
        target_fold_indices=target_fold_indices,
        lower_primitive_count=int(lower_count),
        reference_diagonal=reference,
        reference_total_per_k=reference_total,
        target_occupied_total_per_k=target_total,
        n_occupied_total_per_k=occupied_total,
        primitive_nu_from_total=float(primitive_nu),
    )


@dataclass(frozen=True)
class ProjectorValidation:
    hermitian_error: float
    idempotency_error: float
    trace_per_k: np.ndarray
    expected_trace: float | None
    trace_error: float
    is_valid: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "hermitian_error": float(self.hermitian_error),
            "idempotency_error": float(self.idempotency_error),
            "trace_per_k": [float(v) for v in np.asarray(self.trace_per_k, dtype=float)],
            "expected_trace": None if self.expected_trace is None else float(self.expected_trace),
            "trace_error": float(self.trace_error),
            "is_valid": bool(self.is_valid),
        }


def validate_projector(projector_flat: np.ndarray, *, expected_trace: float | None = None, atol: float = 1.0e-10) -> ProjectorValidation:
    """Check Hermiticity, idempotency, and trace without diagonalizing."""

    p = np.asarray(projector_flat, dtype=np.complex128)
    if p.ndim != 3 or p.shape[0] != p.shape[1]:
        raise ValueError(f"Expected projector shape (nt, nt, nk), got {p.shape}")
    hermitian_error = 0.0
    idempotency_error = 0.0
    traces = np.zeros(p.shape[2], dtype=float)
    for ik in range(p.shape[2]):
        pk = p[:, :, ik]
        hermitian_error = max(hermitian_error, float(np.linalg.norm(pk - pk.conjugate().T)))
        idempotency_error = max(idempotency_error, float(np.linalg.norm(pk @ pk - pk)))
        traces[ik] = float(np.real(np.trace(pk)))
    if expected_trace is None:
        trace_error = 0.0
    else:
        trace_error = float(np.max(np.abs(traces - float(expected_trace))))
    return ProjectorValidation(
        hermitian_error=float(hermitian_error),
        idempotency_error=float(idempotency_error),
        trace_per_k=traces,
        expected_trace=None if expected_trace is None else float(expected_trace),
        trace_error=trace_error,
        is_valid=bool(hermitian_error <= atol and idempotency_error <= atol and trace_error <= atol),
    )


def uniform_ivc_projector(
    layout: FullFlavorLayout,
    *,
    spin_index: int = 0,
    band_indices: Sequence[int] = (0,),
    phase: complex | float | Sequence[complex | float] | np.ndarray = 0.0,
    nk: int = 1,
    valley_pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Build a conventional flat uniform IVC projector for array-level tests.

    For each requested band/fold ``mu`` this creates
    ``(|K,mu> + exp(i phi_k)|K',mu>)/sqrt(2)`` in one spin sector.  The result
    has rank ``len(band_indices)`` at every k.  Chiral winding and sewing are not
    encoded here; callers may supply a k-dependent raw ``phase`` only as a local
    placeholder/control.
    """

    nk = int(nk)
    if nk <= 0:
        raise ValueError(f"nk must be positive, got {nk}")
    bands = tuple(int(v) for v in band_indices)
    if not bands:
        raise ValueError("band_indices must not be empty")
    phases = np.asarray(phase, dtype=np.complex128)
    if phases.ndim == 0:
        phases = np.full((len(bands), nk), phases.item(), dtype=np.complex128)
    elif phases.ndim == 1:
        if phases.shape[0] != nk:
            raise ValueError(f"1D phase must have length nk={nk}, got {phases.shape}")
        phases = np.broadcast_to(phases[None, :], (len(bands), nk)).astype(np.complex128)
    elif phases.shape != (len(bands), nk):
        raise ValueError(f"phase shape must be scalar, (nk,), or {(len(bands), nk)}, got {phases.shape}")

    projector = np.zeros((layout.nt, layout.nt, nk), dtype=np.complex128)
    v0, v1 = (int(valley_pair[0]), int(valley_pair[1]))
    for ib, band in enumerate(bands):
        i0 = layout.flat_index(int(spin_index), v0, band)
        i1 = layout.flat_index(int(spin_index), v1, band)
        for ik in range(nk):
            vec = np.zeros(layout.nt, dtype=np.complex128)
            vec[i0] = 1.0 / np.sqrt(2.0)
            vec[i1] = np.exp(1.0j * phases[ib, ik]) / np.sqrt(2.0)
            projector[:, :, ik] += np.outer(vec, vec.conjugate())
    return projector


@dataclass(frozen=True)
class IVCOrderMetrics:
    ivc_abs_mean: float
    ivc_abs_max: float
    per_k_frobenius: np.ndarray
    per_spin_frobenius_mean: np.ndarray
    raw_phase_field: np.ndarray
    phase_field_status: str
    trs_residual: float
    trs_residual_status: str
    target_band_indices: tuple[int, ...]
    spin_indices: tuple[int, ...]
    valley_pair: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "ivc_abs_mean": float(self.ivc_abs_mean),
            "ivc_abs_max": float(self.ivc_abs_max),
            "per_k_frobenius": [float(v) for v in np.asarray(self.per_k_frobenius, dtype=float)],
            "per_spin_frobenius_mean": [float(v) for v in np.asarray(self.per_spin_frobenius_mean, dtype=float)],
            "raw_phase_field": [None if np.isnan(v) else float(v) for v in np.asarray(self.raw_phase_field, dtype=float)],
            "phase_field_status": self.phase_field_status,
            "trs_residual": float(self.trs_residual),
            "trs_residual_status": self.trs_residual_status,
            "target_band_indices": [int(v) for v in self.target_band_indices],
            "spin_indices": [int(v) for v in self.spin_indices],
            "valley_pair": [int(v) for v in self.valley_pair],
        }


def _blocks_from_flat_or_blocks(density: np.ndarray, layout: FullFlavorLayout | None) -> tuple[np.ndarray, FullFlavorLayout]:
    arr = np.asarray(density, dtype=np.complex128)
    if arr.ndim == 7:
        resolved = _coerce_layout_for_blocks(arr, layout)
        return arr, resolved
    if arr.ndim == 3:
        if layout is None:
            raise ValueError("layout is required when density is flattened")
        return unflatten_full_density(arr, layout), layout
    raise ValueError(f"Expected density shape (nt, nt, nk) or full 7D blocks, got {arr.shape}")


def ivc_order_metrics(
    density: np.ndarray,
    *,
    layout: FullFlavorLayout | None = None,
    target_band_indices: Sequence[int] | None = None,
    spin_indices: Sequence[int] | None = None,
    valley_pair: tuple[int, int] = (0, 1),
    time_reversal_partner: Sequence[int] | None = None,
    atol: float = 1.0e-14,
) -> IVCOrderMetrics:
    """Compute lightweight IVC-amplitude metrics from full valley blocks.

    The amplitude is the k-averaged Frobenius norm of the target-projected
    ``rho_{K,K'}`` blocks, summed in quadrature over requested spins.  The phase
    field is a raw ``arg(trace rho_{K,K'})`` placeholder and is **not** a
    gauge-safe winding diagnostic.
    """

    blocks, resolved = _blocks_from_flat_or_blocks(density, layout)
    nk = int(blocks.shape[-1])
    bands = tuple(range(int(resolved.n_band))) if target_band_indices is None else tuple(int(v) for v in target_band_indices)
    spins = tuple(range(int(resolved.n_spin))) if spin_indices is None else tuple(int(v) for v in spin_indices)
    v0, v1 = int(valley_pair[0]), int(valley_pair[1])
    for band in bands:
        resolved.flat_index(0, 0, band)  # bounds check band
    for spin in spins:
        resolved.flat_index(spin, 0, 0)  # bounds check spin
    if not 0 <= v0 < int(resolved.n_valley) or not 0 <= v1 < int(resolved.n_valley):
        raise IndexError(f"valley_pair={valley_pair} incompatible with n_valley={resolved.n_valley}")

    per_k = np.zeros(nk, dtype=float)
    per_spin_k = np.zeros((len(spins), nk), dtype=float)
    complex_order = np.zeros(nk, dtype=np.complex128)
    max_abs = 0.0
    band_idx = np.asarray(bands, dtype=int)
    for ik in range(nk):
        total_sq = 0.0
        for ispin_pos, spin in enumerate(spins):
            sub = blocks[int(spin), v0, :, int(spin), v1, :, ik][np.ix_(band_idx, band_idx)]
            fro = float(np.linalg.norm(sub))
            per_spin_k[ispin_pos, ik] = fro
            total_sq += fro * fro
            if sub.size:
                max_abs = max(max_abs, float(np.max(np.abs(sub))))
                complex_order[ik] += complex(np.trace(sub))
        per_k[ik] = float(np.sqrt(total_sq))

    raw_phase = np.full(nk, np.nan, dtype=float)
    finite = np.abs(complex_order) > float(atol)
    raw_phase[finite] = np.angle(complex_order[finite])
    phase_status = "raw_trace_phase_not_gauge_safe_placeholder_requires_wilson_link_sewing"

    if time_reversal_partner is None:
        trs_residual = float("nan")
        trs_status = "not_computed_placeholder_requires_time_reversal_sewing_and_k_to_minus_k_map"
    else:
        partner = np.asarray(time_reversal_partner, dtype=int)
        if partner.shape != (nk,):
            raise ValueError(f"time_reversal_partner must have shape {(nk,)}, got {partner.shape}")
        if np.any(partner < 0) or np.any(partner >= nk):
            raise ValueError("time_reversal_partner contains out-of-range k indices")
        numerator = float(np.linalg.norm(complex_order - complex_order[partner]))
        denominator = max(float(np.linalg.norm(complex_order)), float(atol))
        trs_residual = numerator / denominator
        trs_status = "raw_unsewn_placeholder_not_a_physical_trs_metric"

    return IVCOrderMetrics(
        ivc_abs_mean=float(np.mean(per_k)) if nk else 0.0,
        ivc_abs_max=float(max_abs),
        per_k_frobenius=per_k,
        per_spin_frobenius_mean=np.mean(per_spin_k, axis=1) if len(spins) else np.zeros(0, dtype=float),
        raw_phase_field=raw_phase,
        phase_field_status=phase_status,
        trs_residual=float(trs_residual),
        trs_residual_status=trs_status,
        target_band_indices=bands,
        spin_indices=spins,
        valley_pair=(v0, v1),
    )


@dataclass(frozen=True)
class SectorBlockRepresentability:
    """Norms of full-density pieces lost by sector-block storage."""

    off_sector_norm: float
    valley_offdiag_norm: float
    spin_offdiag_norm: float
    is_representable: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "off_sector_norm": float(self.off_sector_norm),
            "valley_offdiag_norm": float(self.valley_offdiag_norm),
            "spin_offdiag_norm": float(self.spin_offdiag_norm),
            "is_representable": bool(self.is_representable),
            "message": self.message,
        }


def sector_block_representability(
    density: np.ndarray,
    *,
    layout: FullFlavorLayout | None = None,
    atol: float = 1.0e-12,
) -> SectorBlockRepresentability:
    """Report whether old ``(spin, valley, band, band, k)`` blocks can store density.

    The old sector-block layout stores only same-spin, same-valley square band
    blocks.  Any nonzero ``rho_KK'`` or spin-off-diagonal density is therefore
    representability loss for IVC.
    """

    blocks, resolved = _blocks_from_flat_or_blocks(density, layout)
    off_sector_sq = 0.0
    valley_sq = 0.0
    spin_sq = 0.0
    for s0 in range(int(resolved.n_spin)):
        for v0 in range(int(resolved.n_valley)):
            for s1 in range(int(resolved.n_spin)):
                for v1 in range(int(resolved.n_valley)):
                    if s0 == s1 and v0 == v1:
                        continue
                    block = blocks[s0, v0, :, s1, v1, :, :]
                    norm_sq = float(np.linalg.norm(block) ** 2)
                    off_sector_sq += norm_sq
                    if v0 != v1:
                        valley_sq += norm_sq
                    if s0 != s1:
                        spin_sq += norm_sq
    off_sector = float(np.sqrt(off_sector_sq))
    valley_offdiag = float(np.sqrt(valley_sq))
    spin_offdiag = float(np.sqrt(spin_sq))
    ok = bool(off_sector <= float(atol))
    if ok:
        message = "density is sector-block representable: no spin/valley off-diagonal blocks above tolerance"
    else:
        message = (
            "sector-block layout cannot represent full-flavor off-sector density; "
            f"rho_KKprime/off-valley norm={valley_offdiag:.6g}, spin-offdiag norm={spin_offdiag:.6g}"
        )
    return SectorBlockRepresentability(
        off_sector_norm=off_sector,
        valley_offdiag_norm=valley_offdiag,
        spin_offdiag_norm=spin_offdiag,
        is_representable=ok,
        message=message,
    )


def validate_sector_block_representable(
    density: np.ndarray,
    *,
    layout: FullFlavorLayout | None = None,
    atol: float = 1.0e-12,
) -> SectorBlockRepresentability:
    """Raise if a full density contains IVC/off-sector blocks lost by sector storage."""

    report = sector_block_representability(density, layout=layout, atol=atol)
    if not report.is_representable:
        raise ValueError(report.message)
    return report
