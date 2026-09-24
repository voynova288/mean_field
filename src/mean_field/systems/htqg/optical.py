from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from analysis.optical import OpticalResponseKindLike, OpticalResponseTensors
from .hf import (
    HTQGProjectedHFData,
    HTQGProjectedHFTargetData,
    SPIN_LABELS,
    VALLEY_SEQUENCE,
    build_htqg_hf_target_hamiltonian,
    build_htqg_overlap_blocks,
    build_htqg_target_data,
)
from mean_field.core.hf.overlap import HFOverlapBlockSet


@dataclass(frozen=True)
class HTQGHFOpticalPoint:
    """One HTQG projected-HF k point prepared for generic optical APIs.

    The Hamiltonian and its derivatives are represented in a locally parallel-
    transported active-band basis at ``kvec``.  The derivative stencil is local
    in Cartesian momentum and does not wrap through the mBZ torus; this avoids
    seam artifacts before data are passed to :mod:`analysis.optical`.
    """

    kvec: complex
    hamiltonian: np.ndarray
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    dhdk: np.ndarray
    d2hdk: np.ndarray
    step_nm_inv: float
    min_basis_overlap_singular_value: float
    band_indices: tuple[int, ...] | None = None


def _label_index(spin_index: int, valley_index: int, band_position: int) -> int:
    return int(spin_index + len(SPIN_LABELS) * (valley_index + len(VALLEY_SEQUENCE) * band_position))


def _target_full_spin_valley_basis(data: HTQGProjectedHFData, target: HTQGProjectedHFTargetData, ik: int) -> np.ndarray:
    """Return full spin/valley active basis columns for one target k point.

    The column order matches the projected-HF active index convention used by
    ``HTQGProjectedHFData.h0`` and saved HF Hamiltonians.
    """

    basis_dim = int(data.wavefunctions.shape[0])
    full_dim = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * basis_dim
    target_n_band = int(target.wavefunctions.shape[1])
    out = np.zeros((full_dim, int(target.nt)), dtype=np.complex128)
    for iband in range(target_n_band):
        for ivalley in range(len(VALLEY_SEQUENCE)):
            vec = np.asarray(target.wavefunctions[:, iband, ivalley, int(ik)], dtype=np.complex128)
            for ispin in range(len(SPIN_LABELS)):
                col = _label_index(ispin, ivalley, iband)
                block = ispin * len(VALLEY_SEQUENCE) + ivalley
                out[block * basis_dim : (block + 1) * basis_dim, col] = vec
    return out


def _polar_unitary(overlap: np.ndarray) -> tuple[np.ndarray, float]:
    u, singular_values, vh = np.linalg.svd(np.asarray(overlap, dtype=np.complex128), full_matrices=False)
    return u @ vh, float(np.min(singular_values)) if singular_values.size else 0.0


def _spin_valley_sector_indices(n_band: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([_label_index(ispin, ivalley, iband) for iband in range(int(n_band))], dtype=int)
        for ispin in range(len(SPIN_LABELS))
        for ivalley in range(len(VALLEY_SEQUENCE))
    )


def _spin_valley_block_polar_unitary(overlap: np.ndarray, n_band: int) -> tuple[np.ndarray, float]:
    """Return the polar transporter without numerically mixing exact sectors.

    The projected HTQG basis has exactly conserved spin and valley blocks. A
    full SVD may choose arbitrary vectors across equal singular-value blocks;
    those roundoff-scale off-sector entries are then amplified by second finite
    differences. Compute the equivalent polar factor block by block after
    verifying that the supplied overlap is sector diagonal.
    """

    matrix = np.asarray(overlap, dtype=np.complex128)
    sectors = _spin_valley_sector_indices(int(n_band))
    allowed = np.zeros(matrix.shape, dtype=bool)
    for indices in sectors:
        allowed[np.ix_(indices, indices)] = True
    leakage = float(np.max(np.abs(matrix[~allowed]))) if np.any(~allowed) else 0.0
    if leakage > 1.0e-12:
        raise ValueError(f"HTQG target overlap violates exact spin/valley sectors: {leakage}")
    transporter = np.zeros_like(matrix)
    minimum_sv = 1.0
    for indices in sectors:
        block, singular_value = _polar_unitary(matrix[np.ix_(indices, indices)])
        transporter[np.ix_(indices, indices)] = block
        minimum_sv = min(minimum_sv, float(singular_value))
    return transporter, minimum_sv


def _transport_target_hamiltonians_to_base(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    hamiltonians: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Parallel-transport target Hamiltonians to the first target k basis."""

    h = np.asarray(hamiltonians, dtype=np.complex128)
    if h.ndim != 3 or h.shape[:2] != (int(target.nt), int(target.nt)):
        raise ValueError(
            f"Expected target Hamiltonians with shape {(target.nt, target.nt, target.nk)}, "
            f"got {h.shape}"
        )
    base = _target_full_spin_valley_basis(data, target, 0)
    transported = np.empty_like(h)
    min_sv = 1.0
    for ik in range(h.shape[2]):
        if ik == 0:
            out = h[:, :, ik]
            sv = 1.0
        else:
            nb = _target_full_spin_valley_basis(data, target, ik)
            overlap = base.conjugate().T @ nb
            q, sv = _spin_valley_block_polar_unitary(overlap, int(target.wavefunctions.shape[1]))
            # q maps neighbor-basis coordinates into the base active-basis frame.
            out = q @ h[:, :, ik] @ q.conjugate().T
        transported[:, :, ik] = 0.5 * (out + out.conjugate().T)
        min_sv = min(min_sv, float(sv))
    return transported, min_sv


def _optical_point_from_target_hamiltonians(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    hamiltonians: np.ndarray,
    *,
    step_nm_inv: float,
) -> HTQGHFOpticalPoint:
    """Transport a nine-point target stencil and form Cartesian derivatives."""

    step = float(step_nm_inv)
    h, min_sv = _transport_target_hamiltonians_to_base(data, target, hamiltonians)
    if h.shape[2] != 9:
        raise ValueError(f"Expected the fixed nine-point derivative stencil, got {h.shape[2]} points")
    h0 = h[:, :, 0]
    hp_x, hm_x = h[:, :, 1], h[:, :, 2]
    hp_y, hm_y = h[:, :, 3], h[:, :, 4]
    hpp, hpm, hmp, hmm = h[:, :, 5], h[:, :, 6], h[:, :, 7], h[:, :, 8]

    target_nt = int(target.nt)
    dhdk = np.empty((2, target_nt, target_nt), dtype=np.complex128)
    dhdk[0] = (hp_x - hm_x) / (2.0 * step)
    dhdk[1] = (hp_y - hm_y) / (2.0 * step)
    d2hdk = np.empty((2, 2, target_nt, target_nt), dtype=np.complex128)
    d2hdk[0, 0] = (hp_x + hm_x - 2.0 * h0) / (step * step)
    d2hdk[1, 1] = (hp_y + hm_y - 2.0 * h0) / (step * step)
    cross = (hpp - hpm - hmp + hmm) / (4.0 * step * step)
    d2hdk[0, 1] = cross
    d2hdk[1, 0] = cross
    for a in range(2):
        dhdk[a] = 0.5 * (dhdk[a] + dhdk[a].conjugate().T)
        for b in range(2):
            d2hdk[a, b] = 0.5 * (d2hdk[a, b] + d2hdk[a, b].conjugate().T)

    h0 = 0.5 * (h0 + h0.conjugate().T)
    energies, eigenvectors = np.linalg.eigh(h0)
    return HTQGHFOpticalPoint(
        kvec=complex(target.kvec[0]),
        hamiltonian=h0,
        energies_ev=np.asarray(energies, dtype=float),
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        dhdk=dhdk,
        d2hdk=d2hdk,
        step_nm_inv=step,
        min_basis_overlap_singular_value=float(min_sv),
        band_indices=(
            tuple(int(index) for index in data.band_indices)
            if target.band_indices is None
            else tuple(int(index) for index in target.band_indices)
        ),
    )


def noninteracting_optical_derivative_bundle(
    data: HTQGProjectedHFData,
    kvec: complex,
    *,
    step_nm_inv: float = 1.0e-4,
    target_band_indices: Sequence[int] | None = None,
) -> HTQGHFOpticalPoint:
    """Build the seam-safe nine-point derivative bundle for projected H0.

    The source and every Cartesian stencil point use the resolved explicit
    ``d12``/``d34`` stored in ``data``.  Occupations and response conventions
    remain workflow-level choices.
    """

    step = float(step_nm_inv)
    if step <= 0.0:
        raise ValueError(f"step_nm_inv must be positive, got {step_nm_inv}")
    k0 = complex(kvec)
    offsets: Sequence[complex] = (
        0.0 + 0.0j,
        step + 0.0j,
        -step + 0.0j,
        0.0 + 1.0j * step,
        0.0 - 1.0j * step,
        step + 1.0j * step,
        step - 1.0j * step,
        -step + 1.0j * step,
        -step - 1.0j * step,
    )
    target = build_htqg_target_data(
        data,
        np.asarray([k0 + offset for offset in offsets], dtype=np.complex128),
        band_indices=target_band_indices,
    )
    return _optical_point_from_target_hamiltonians(
        data,
        target,
        target.h0,
        step_nm_inv=step,
    )


def hf_optical_derivative_bundle(
    data: HTQGProjectedHFData,
    density: np.ndarray,
    kvec: complex,
    *,
    step_nm_inv: float = 1.0e-4,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    target_band_indices: Sequence[int] | None = None,
) -> HTQGHFOpticalPoint:
    """Build seam-safe finite-difference derivatives for HTQG projected HF.

    The stencil uses local Cartesian points ``k ± h x`` and ``k ± h y`` plus
    mixed-derivative corners.  These points are not wrapped into the mBZ; each
    target Hamiltonian is evaluated directly and then parallel-transported back
    to the active basis at ``k`` by the polar part of the active-basis overlap.
    The returned derivatives can be passed directly to ``analysis.optical``.

    ``target_band_indices`` may request a boundary-complete ambient target
    window while retaining the supplied source HF density.  This evaluates a
    fixed-density target operator; it is not an HF rerun or an active-space
    convergence claim.  Callers must validate center parity, target-window
    isolation, h convergence, and target-window dependence before response
    integration.
    """

    step = float(step_nm_inv)
    if step <= 0.0:
        raise ValueError(f"step_nm_inv must be positive, got {step_nm_inv}")
    k0 = complex(kvec)
    offsets: Sequence[complex] = (
        0.0 + 0.0j,
        step + 0.0j,
        -step + 0.0j,
        0.0 + 1.0j * step,
        0.0 - 1.0j * step,
        step + 1.0j * step,
        step - 1.0j * step,
        -step + 1.0j * step,
        -step - 1.0j * step,
    )
    target = build_htqg_target_data(
        data,
        np.asarray([k0 + off for off in offsets], dtype=np.complex128),
        band_indices=target_band_indices,
    )
    source_blocks = build_htqg_overlap_blocks(data) if source_overlap_blocks is None else source_overlap_blocks
    h_target = build_htqg_hf_target_hamiltonian(
        data,
        target,
        np.asarray(density, dtype=np.complex128),
        source_overlap_blocks=source_blocks,
    )
    return _optical_point_from_target_hamiltonians(
        data,
        target,
        h_target,
        step_nm_inv=step,
    )


def optical_tensors_at_hf_k(
    kind: "OpticalResponseKindLike | str",
    data: HTQGProjectedHFData,
    density: np.ndarray,
    kvec: complex,
    *,
    mu_ev: float,
    temperature_k: float = 0.0,
    step_nm_inv: float = 1.0e-4,
    denominator_cutoff_ev: float = 1.0e-10,
    principal_value_eta_ev: float | None = None,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    target_band_indices: Sequence[int] | None = None,
) -> tuple[HTQGHFOpticalPoint, "OpticalResponseTensors"]:
    """Evaluate one HTQG projected-HF k point and route it to ``analysis.optical``."""

    from analysis.optical import precompute_optical_tensors

    point = hf_optical_derivative_bundle(
        data,
        density,
        kvec,
        step_nm_inv=step_nm_inv,
        source_overlap_blocks=source_overlap_blocks,
        target_band_indices=target_band_indices,
    )
    tensors = precompute_optical_tensors(
        kind,
        point.energies_ev,
        point.eigenvectors,
        point.dhdk,
        mu_ev=float(mu_ev),
        temperature_k=float(temperature_k),
        denominator_cutoff_ev=float(denominator_cutoff_ev),
        d2hdk=point.d2hdk,
        principal_value_eta_ev=principal_value_eta_ev,
    )
    return point, tensors


__all__ = [
    "HTQGHFOpticalPoint",
    "hf_optical_derivative_bundle",
    "noninteracting_optical_derivative_bundle",
    "optical_tensors_at_hf_k",
]
