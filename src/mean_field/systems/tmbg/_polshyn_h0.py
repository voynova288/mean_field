from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mean_field.core.hf import (
    HFOverlapBlockSet,
    build_projected_interaction_hamiltonian,
    shift_wavefunction_grid,
)

from ._polshyn_types import PolshynProjectedBasis
from ._polshyn_wang import (
    overlap_blocks_with_hartree_q0_zeroed,
    unflatten_sector_blocks,
    wang_stored_density_from_sector_blocks,
)
from .hamiltonian import dirac_block

_POLSHYN_H0_MODES = {"none", "active-reference", "minus-full-p0"}
_POLSHYN_P0_REFERENCES = {"decoupled-layers"}


def _normalize_h0_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized in {"", "off", "false", "no"}:
        normalized = "none"
    if normalized not in _POLSHYN_H0_MODES:
        raise ValueError(f"Unsupported Polshyn h0_subtraction mode {mode!r}; expected {sorted(_POLSHYN_H0_MODES)}")
    return normalized


def _normalize_p0_reference(p0_reference: str) -> str:
    normalized = str(p0_reference).strip().lower().replace("_", "-")
    if normalized not in _POLSHYN_P0_REFERENCES:
        raise ValueError(
            "Polshyn h0_subtraction currently exposes only p0_reference='decoupled-layers'; "
            f"got {p0_reference!r}"
        )
    return normalized


def _polshyn_h0_sign(mode: str) -> float:
    normalized = _normalize_h0_mode(mode)
    if normalized == "active-reference":
        return 1.0
    if normalized == "minus-full-p0":
        return -1.0
    return 0.0


@dataclass(frozen=True)
class PolshynH0SubtractionConfig:
    """Configuration for reviewed Polshyn projected-HF one-body subtraction.

    This is a TMBG/Polshyn system adapter, not a generic core-HF feature.  The
    public modes intentionally cover only the two Slurm-validated conventions:
    ``active-reference`` and ``minus-full-p0``.  The application sign is fixed
    by the mode and cannot be overridden by callers.
    """

    mode: str = "none"
    p0_reference: str = "decoupled-layers"
    zero_hartree_q0: bool = True
    include_active_reference: bool = True
    hartree_scale: float = 1.0
    fock_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _normalize_h0_mode(self.mode))
        object.__setattr__(self, "p0_reference", _normalize_p0_reference(self.p0_reference))
        object.__setattr__(self, "zero_hartree_q0", bool(self.zero_hartree_q0))
        object.__setattr__(self, "include_active_reference", bool(self.include_active_reference))
        object.__setattr__(self, "hartree_scale", float(self.hartree_scale))
        object.__setattr__(self, "fock_scale", float(self.fock_scale))
        if float(self.hartree_scale) < 0.0 or float(self.fock_scale) < 0.0:
            raise ValueError("Polshyn h0_subtraction Hartree/Fock scales must be non-negative")

    @property
    def applied_sign(self) -> float:
        return _polshyn_h0_sign(self.mode)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "p0_reference": self.p0_reference,
            "zero_hartree_q0": bool(self.zero_hartree_q0),
            "include_active_reference": bool(self.include_active_reference),
            "hartree_scale": float(self.hartree_scale),
            "fock_scale": float(self.fock_scale),
            "applied_sign": float(self.applied_sign),
        }


@dataclass(frozen=True)
class PolshynH0SubtractionResult:
    corrected_basis: PolshynProjectedBasis
    raw_correction_blocks: np.ndarray
    applied_correction_blocks: np.ndarray
    diagnostics: dict[str, Any]


def _validate_h0_block_shape(basis: PolshynProjectedBasis, blocks: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(blocks, dtype=np.complex128)
    if arr.shape != np.asarray(basis.h0_blocks).shape:
        raise ValueError(f"{name} shape {arr.shape} incompatible with Polshyn basis.h0_blocks shape {basis.h0_blocks.shape}")
    return arr


def _h0_norm_diagnostics(blocks: np.ndarray) -> dict[str, float]:
    arr = np.asarray(blocks, dtype=np.complex128)
    return {
        "h0_correction_norm_ev": float(np.linalg.norm(arr)),
        "h0_correction_max_abs_mev": float(1000.0 * np.max(np.abs(arr))) if arr.size else 0.0,
    }


def polshyn_reference_projector_blocks(basis: PolshynProjectedBasis) -> np.ndarray:
    """Return conventional sector reference projectors from ``basis.reference_diagonal``."""

    reference_diagonal = np.asarray(basis.reference_diagonal, dtype=float).reshape(-1)
    if reference_diagonal.shape != (int(basis.nb),):
        raise ValueError(f"Polshyn reference_diagonal shape {reference_diagonal.shape} does not match nb={basis.nb}")
    ref = np.diag(reference_diagonal).astype(np.complex128)
    out = np.zeros((int(basis.n_spin), int(basis.n_eta), int(basis.nb), int(basis.nb), int(basis.nk)), dtype=np.complex128)
    out[:, :, :, :, :] = ref[None, None, :, :, None]
    return out


def compute_polshyn_active_reference_h0_correction(
    basis: PolshynProjectedBasis,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    zero_hartree_q0: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the reviewed Polshyn ``active-reference`` h0 correction.

    The input reference is conventional sector-block ``P_ref``; it is converted
    to the Wang/Xiaoyu stored ``P*`` orientation before using the common core-HF
    projected interaction builder.
    """

    active_blocks = overlap_blocks_with_hartree_q0_zeroed(overlap_blocks) if bool(zero_hartree_q0) else overlap_blocks
    reference_blocks = polshyn_reference_projector_blocks(basis)
    reference_flat = wang_stored_density_from_sector_blocks(reference_blocks)
    correction_flat = build_projected_interaction_hamiltonian(reference_flat, active_blocks, v0=float(v0), beta=1.0)
    correction = unflatten_sector_blocks(correction_flat, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)
    correction = 0.5 * (correction + np.swapaxes(correction.conjugate(), 2, 3))
    trace_ref = np.trace(reference_blocks, axis1=2, axis2=3).real
    diagnostics: dict[str, Any] = {
        "mode": "active-reference",
        "applied_sign": 1.0,
        "p0_reference": "decoupled-layers",
        "zero_hartree_q0": bool(zero_hartree_q0),
        "include_active_reference": True,
        "reference_density_trace_mean": float(np.mean(trace_ref)),
        "reference_density_trace_min": float(np.min(trace_ref)),
        "reference_density_trace_max": float(np.max(trace_ref)),
    }
    diagnostics.update(_h0_norm_diagnostics(correction))
    return correction, diagnostics


def basis_with_polshyn_h0_correction(
    basis: PolshynProjectedBasis,
    correction_blocks: np.ndarray,
) -> PolshynProjectedBasis:
    """Return a basis whose ``h0_blocks`` include a symmetrized correction."""

    correction = _validate_h0_block_shape(basis, correction_blocks, name="Polshyn h0 correction")
    corrected = np.asarray(basis.h0_blocks, dtype=np.complex128) + correction
    corrected = 0.5 * (corrected + np.swapaxes(corrected.conjugate(), 2, 3))
    return replace(basis, h0_blocks=corrected)


def _single_layer_valence_projector(h2: np.ndarray, *, degeneracy_tol: float = 1.0e-12) -> np.ndarray:
    h = np.asarray(h2, dtype=np.complex128)
    if h.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 Dirac block, got {h.shape}")
    h = 0.5 * (h + h.conjugate().T)
    evals, evecs = np.linalg.eigh(h)
    if float(np.max(evals) - np.min(evals)) < float(degeneracy_tol):
        return 0.5 * np.eye(2, dtype=np.complex128)
    vec = evecs[:, [0]]
    return vec @ vec.conjugate().T


def _decoupled_layers_cnp_block(
    basis: PolshynProjectedBasis,
    k_super: complex,
    *,
    n1: int,
    n2: int,
    fold: int,
    valley: int,
) -> np.ndarray:
    lattice = basis.model.lattice
    params = basis.model.params
    k_site = complex(k_super + int(fold) * basis.super_b1 + int(n1) * lattice.g_m1 + int(n2) * lattice.g_m2)
    k_bottom = complex(k_site - int(valley) * lattice.k_m)
    k_top = complex(k_site - int(valley) * lattice.kprime_m)
    h_bottom = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, int(valley))
    h_middle = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, int(valley))
    h_top = dirac_block(k_top, lattice.theta_rad / 2.0, params.vf, int(valley))
    out = np.zeros((6, 6), dtype=np.complex128)
    out[0:2, 0:2] = _single_layer_valence_projector(h_bottom)
    out[2:4, 2:4] = _single_layer_valence_projector(h_middle)
    out[4:6, 4:6] = _single_layer_valence_projector(h_top)
    return out


def _shift_polshyn_grid(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    return shift_wavefunction_grid(values, dm, dn, boundary_mode="zero_fill", grid_axes=(1, 2))


def _p0_times_wavefunction_grid(basis: PolshynProjectedBasis, *, valley_index: int) -> tuple[np.ndarray, np.ndarray]:
    ieta = int(valley_index)
    valley = (1, -1)[ieta]
    nb = int(basis.nb)
    nx, ny = basis.embedding_shape
    u_grid = np.asarray(basis.wavefunctions, dtype=np.complex128)[:, :, ieta, :].reshape(
        basis.local_basis_size,
        nx,
        ny,
        nb,
        basis.nk,
        order="F",
    )
    p0u_grid = np.zeros_like(u_grid)
    p0_band = np.zeros((nb, nb, basis.nk), dtype=np.complex128)
    for ik, kval in enumerate(basis.kvec):
        for (n1, n2, fold), (ix, iy) in tuple(basis.embedding_positions.items()):
            u_site = u_grid[:, int(ix), int(iy), :, ik]
            if not np.any(u_site):
                continue
            p0_local = _decoupled_layers_cnp_block(
                basis,
                complex(kval),
                n1=int(n1),
                n2=int(n2),
                fold=int(fold),
                valley=int(valley),
            )
            p0u_grid[:, int(ix), int(iy), :, ik] = p0_local @ u_site
        p0_band[:, :, ik] = np.einsum(
            "lxyb,lxya->ab",
            np.conj(u_grid[:, :, :, :, ik]),
            p0u_grid[:, :, :, :, ik],
            optimize=True,
        )
        p0_band[:, :, ik] = 0.5 * (p0_band[:, :, ik] + p0_band[:, :, ik].conjugate().T)
    return p0u_grid, p0_band


def _compact_overlap_between(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    if target.nb != source.nb or target.embedding_shape != source.embedding_shape:
        raise ValueError("target/source Polshyn basis mismatch")
    nb = int(target.nb)
    nx, ny = target.embedding_shape
    target_cols = nb * target.nk
    source_cols = nb * source.nk
    ul = np.asarray(target.wavefunctions, dtype=np.complex128)[:, :, valley_index, :].reshape(
        target.basis_dimension,
        target_cols,
        order="F",
    )
    ur_grid = np.asarray(source.wavefunctions, dtype=np.complex128)[:, :, valley_index, :].reshape(
        source.local_basis_size,
        nx,
        ny,
        source_cols,
        order="F",
    )
    shifted = _shift_polshyn_grid(ur_grid, -int(shift[0]), -int(shift[1])).reshape(
        source.basis_dimension,
        source_cols,
        order="F",
    )
    return ul.conj().T @ shifted


def _compact_diagonal_overlap(basis: PolshynProjectedBasis, shift: tuple[int, int], *, valley_index: int) -> np.ndarray:
    nb = int(basis.nb)
    nx, ny = basis.embedding_shape
    w_grid = np.asarray(basis.wavefunctions, dtype=np.complex128)[:, :, valley_index, :].reshape(
        basis.local_basis_size,
        nx,
        ny,
        nb,
        basis.nk,
        order="F",
    )
    shifted = _shift_polshyn_grid(w_grid, -int(shift[0]), -int(shift[1]))
    return np.einsum("lxyak,lxybk->abk", np.conj(w_grid), shifted, optimize=True)


def _compact_overlap_to_source_grid(
    target: PolshynProjectedBasis,
    source_grid: np.ndarray,
    shift: tuple[int, int],
    *,
    valley_index: int,
) -> np.ndarray:
    nb = int(target.nb)
    nx, ny = target.embedding_shape
    source = np.asarray(source_grid, dtype=np.complex128)
    expected = (target.local_basis_size, nx, ny, nb, target.nk)
    if source.shape != expected:
        raise ValueError(f"Expected source_grid shape {expected}, got {source.shape}")
    target_cols = nb * target.nk
    source_cols = nb * target.nk
    ul = np.asarray(target.wavefunctions, dtype=np.complex128)[:, :, valley_index, :].reshape(
        target.basis_dimension,
        target_cols,
        order="F",
    )
    shifted = _shift_polshyn_grid(
        source.reshape(target.local_basis_size, nx, ny, source_cols, order="F"),
        -int(shift[0]),
        -int(shift[1]),
    ).reshape(target.basis_dimension, source_cols, order="F")
    return ul.conj().T @ shifted


def _compact_block(compact: np.ndarray, nb: int, kt: int, ks: int) -> np.ndarray:
    row = slice(int(kt) * int(nb), (int(kt) + 1) * int(nb))
    col = slice(int(ks) * int(nb), (int(ks) + 1) * int(nb))
    return np.asarray(compact[row, col], dtype=np.complex128)


def _p0_trace_diagnostics(p0_band_by_valley: list[np.ndarray]) -> dict[str, float]:
    if not p0_band_by_valley:
        return {
            "projected_p0_trace_mean": 0.0,
            "projected_p0_trace_min": 0.0,
            "projected_p0_trace_max": 0.0,
        }
    p0_traces = [np.trace(p0_band, axis1=0, axis2=1).real for p0_band in p0_band_by_valley]
    p0_trace_all = np.concatenate([arr.reshape(-1) for arr in p0_traces])
    return {
        "projected_p0_trace_mean": float(np.mean(p0_trace_all)),
        "projected_p0_trace_min": float(np.min(p0_trace_all)),
        "projected_p0_trace_max": float(np.max(p0_trace_all)),
    }


def compute_polshyn_minus_full_p0_h0_correction(
    basis: PolshynProjectedBasis,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    p0_reference: str = "decoupled-layers",
    zero_hartree_q0: bool = True,
    include_active_reference: bool = True,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
    progress_prefix: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the raw full-P0 correction used by ``minus-full-p0``.

    The returned raw correction is ``HF[D]`` for the full active-remote P0
    subtraction density.  The public ``minus-full-p0`` mode applies it with the
    fixed sign ``-1`` via :func:`apply_polshyn_h0_subtraction`.
    """

    _normalize_p0_reference(p0_reference)
    if len(overlap_blocks.shifts) != int(np.asarray(overlap_blocks.gvecs).size):
        raise ValueError("Polshyn h0_subtraction overlap shifts and g-vectors must have the same length")
    nb = int(basis.nb)
    nk = int(basis.nk)
    ref = np.diag(np.asarray(basis.reference_diagonal, dtype=float)).astype(np.complex128)
    ref_for_m = ref if bool(include_active_reference) else np.zeros_like(ref)
    scale = float(v0) / float(nk)
    correction = np.zeros_like(np.asarray(basis.h0_blocks, dtype=np.complex128))

    p0u_by_valley: list[np.ndarray] = []
    p0_band_by_valley: list[np.ndarray] = []
    m_by_valley: list[np.ndarray] = []
    for ieta in range(int(basis.n_eta)):
        p0u_grid, p0_band = _p0_times_wavefunction_grid(basis, valley_index=ieta)
        p0u_by_valley.append(p0u_grid)
        p0_band_by_valley.append(p0_band)
        m_blocks = np.zeros_like(p0_band)
        for ik in range(nk):
            m_blocks[:, :, ik] = ref_for_m + p0_band[:, :, ik]
        m_by_valley.append(m_blocks)

    hartree_norm = 0.0
    fock_norm = 0.0
    trace_abs_max = 0.0
    for ishift, shift in enumerate(tuple(overlap_blocks.shifts), start=1):
        shift = (int(shift[0]), int(shift[1]))
        if progress_prefix and (ishift == 1 or ishift == len(overlap_blocks.shifts) or ishift % 5 == 0):
            print(f"{progress_prefix} full-p0 shift {ishift}/{len(overlap_blocks.shifts)} shift={shift}", flush=True)
        target_diagonal = np.asarray(
            [_compact_diagonal_overlap(basis, shift, valley_index=ieta) for ieta in range(int(basis.n_eta))],
            dtype=np.complex128,
        )
        k_p0 = [
            _compact_overlap_to_source_grid(basis, p0u_by_valley[ieta], shift, valley_index=ieta)
            for ieta in range(int(basis.n_eta))
        ]
        k_p0_dagger_shift = [
            _compact_overlap_to_source_grid(basis, p0u_by_valley[ieta], (-shift[0], -shift[1]), valley_index=ieta)
            for ieta in range(int(basis.n_eta))
        ]

        trace_total = 0.0 + 0.0j
        for _ispin in range(int(basis.n_spin)):
            for ieta in range(int(basis.n_eta)):
                diag = target_diagonal[ieta]
                for ik in range(nk):
                    lam = diag[:, :, ik]
                    m_block = m_by_valley[ieta][:, :, ik]
                    k_same = _compact_block(k_p0[ieta], nb, ik, ik)
                    j_same = _compact_block(k_p0_dagger_shift[ieta], nb, ik, ik)
                    trace = np.einsum("ab,ab->", m_block, np.conj(lam), optimize=True)
                    trace -= np.conj(np.trace(k_same))
                    trace -= np.trace(j_same)
                    trace_total += trace
        trace_abs_max = max(trace_abs_max, float(abs(trace_total)))

        hartree_kernel = overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            hartree_value = float(hartree_scale) * float(hartree_kernel)
            if bool(zero_hartree_q0) and shift == (0, 0):
                hartree_value = 0.0
            if hartree_value != 0.0:
                hartree_piece = np.zeros_like(correction)
                coeff_h = scale * hartree_value * trace_total
                for ispin in range(int(basis.n_spin)):
                    for ieta in range(int(basis.n_eta)):
                        hartree_piece[ispin, ieta] += coeff_h * target_diagonal[ieta]
                correction += hartree_piece
                hartree_norm += float(np.linalg.norm(hartree_piece))

        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is None:
            continue
        fock_kernel = float(fock_scale) * np.asarray(fock_kernel, dtype=float)
        if fock_kernel.shape != (nk, nk):
            raise ValueError(f"Polshyn h0_subtraction fock kernel for shift {shift} has shape {fock_kernel.shape}, expected {(nk, nk)}")
        for ieta in range(int(basis.n_eta)):
            lam_compact = _compact_overlap_between(basis, basis, shift, valley_index=ieta)
            k_compact = k_p0[ieta]
            for ispin in range(int(basis.n_spin)):
                for kt in range(nk):
                    fock_block = np.zeros((nb, nb), dtype=np.complex128)
                    for ks in range(nk):
                        coeff = scale * float(fock_kernel[kt, ks])
                        if coeff == 0.0:
                            continue
                        lam = _compact_block(lam_compact, nb, kt, ks)
                        kblk = _compact_block(k_compact, nb, kt, ks)
                        m_block = m_by_valley[ieta][:, :, ks]
                        density_projected = lam @ m_block @ lam.conjugate().T
                        density_projected -= lam @ kblk.conjugate().T
                        density_projected -= kblk @ lam.conjugate().T
                        fock_block -= coeff * density_projected
                    correction[ispin, ieta, :, :, kt] += fock_block
                    fock_norm += float(np.linalg.norm(fock_block))
    correction = 0.5 * (correction + np.swapaxes(correction.conjugate(), 2, 3))
    diagnostics: dict[str, Any] = {
        "mode": "minus-full-p0",
        "applied_sign": -1.0,
        "p0_reference": "decoupled-layers",
        "zero_hartree_q0": bool(zero_hartree_q0),
        "include_active_reference": bool(include_active_reference),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
        "hartree_accumulated_norm_ev": float(hartree_norm),
        "fock_accumulated_norm_ev": float(fock_norm),
        "source_trace_abs_max": float(trace_abs_max),
    }
    diagnostics.update(_p0_trace_diagnostics(p0_band_by_valley))
    diagnostics.update(_h0_norm_diagnostics(correction))
    return correction, diagnostics


def apply_polshyn_h0_subtraction(
    basis: PolshynProjectedBasis,
    overlap_blocks: HFOverlapBlockSet,
    *,
    config: PolshynH0SubtractionConfig | str | None = None,
    v0: float,
    progress_prefix: str | None = None,
) -> PolshynH0SubtractionResult:
    """Apply a reviewed Polshyn h0-subtraction mode with fixed sign policy."""

    resolved = PolshynH0SubtractionConfig() if config is None else (PolshynH0SubtractionConfig(config) if isinstance(config, str) else config)
    if not isinstance(resolved, PolshynH0SubtractionConfig):
        raise TypeError(f"config must be PolshynH0SubtractionConfig, str, or None; got {type(resolved).__name__}")
    if resolved.mode == "none":
        raw = np.zeros_like(np.asarray(basis.h0_blocks, dtype=np.complex128))
        diagnostics: dict[str, Any] = resolved.to_dict()
        diagnostics.update(_h0_norm_diagnostics(raw))
        return PolshynH0SubtractionResult(
            corrected_basis=basis,
            raw_correction_blocks=raw,
            applied_correction_blocks=raw,
            diagnostics=diagnostics,
        )
    if resolved.mode == "active-reference":
        raw, diagnostics = compute_polshyn_active_reference_h0_correction(
            basis,
            overlap_blocks,
            v0=float(v0),
            zero_hartree_q0=bool(resolved.zero_hartree_q0),
        )
    elif resolved.mode == "minus-full-p0":
        raw, diagnostics = compute_polshyn_minus_full_p0_h0_correction(
            basis,
            overlap_blocks,
            v0=float(v0),
            p0_reference=resolved.p0_reference,
            zero_hartree_q0=bool(resolved.zero_hartree_q0),
            include_active_reference=bool(resolved.include_active_reference),
            hartree_scale=float(resolved.hartree_scale),
            fock_scale=float(resolved.fock_scale),
            progress_prefix=progress_prefix,
        )
    else:  # pragma: no cover - protected by config validation.
        raise ValueError(f"Unsupported Polshyn h0_subtraction mode {resolved.mode!r}")
    applied = float(resolved.applied_sign) * _validate_h0_block_shape(basis, raw, name="raw Polshyn h0 correction")
    corrected = basis_with_polshyn_h0_correction(basis, applied)
    merged_diagnostics = resolved.to_dict()
    merged_diagnostics.update(dict(diagnostics))
    merged_diagnostics["applied_sign"] = float(resolved.applied_sign)
    merged_diagnostics.update({f"applied_{key}": value for key, value in _h0_norm_diagnostics(applied).items()})
    return PolshynH0SubtractionResult(
        corrected_basis=corrected,
        raw_correction_blocks=raw,
        applied_correction_blocks=applied,
        diagnostics=merged_diagnostics,
    )


__all__ = [name for name in globals() if not name.startswith("_")]
