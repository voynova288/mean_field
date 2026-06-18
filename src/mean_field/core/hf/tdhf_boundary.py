from __future__ import annotations

"""Canonical HFState/HFRunResult boundary normalizer for TDHF bookkeeping.

This module is deliberately system-agnostic.  It normalizes a canonical
:class:`mean_field.core.contracts.HFState` (or ``HFRunResult.final_state``) into
HF orbital arrays, a stable global-index convention, and an occupied mask.  It
does not construct finite-q sectors or two-body TDHF matrix elements; those
require system-specific momentum wrapping, form factors, and interactions.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from mean_field.core.contracts import HFRunResult, HFState, assert_hermitian_field, assert_matrix_field_shape

TDHFOccupationPolicy = Literal["projector", "energy_sort"]


@dataclass(frozen=True)
class TDHFCanonicalOrbitals:
    """Canonical TDHF orbital bookkeeping derived from a converged HF state.

    ``eigenvectors[basis_index, hf_index, k]`` stores the per-k unitary whose
    columns are HF orbitals.  The global TDHF orbital index is
    ``local_hf_index + nt * k_index``, matching Fortran flattening of
    ``energies[local_hf_index, k_index]``.
    """

    energies: np.ndarray
    eigenvectors: np.ndarray
    occupied_mask: np.ndarray
    k_grid_frac: np.ndarray
    mu: float
    flavor_labels: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        if energies.ndim != 2:
            raise ValueError(f"energies must have shape (n_state, n_k), got {energies.shape}")
        nt, nk = (int(energies.shape[0]), int(energies.shape[1]))
        if nt <= 0 or nk <= 0:
            raise ValueError(f"energies must contain at least one state and k point, got {energies.shape}")

        eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)
        if eigenvectors.shape != (nt, nt, nk):
            raise ValueError(
                "eigenvectors must have shape (n_state, n_state, n_k), "
                f"got {eigenvectors.shape} for energies shape {energies.shape}"
            )
        unitary_residual = _max_unitary_residual(eigenvectors)
        if unitary_residual > 1.0e-8:
            raise ValueError(f"eigenvectors must be unitary at every k; max residual {unitary_residual:.6e}")

        occupied_mask = np.asarray(self.occupied_mask, dtype=bool)
        if occupied_mask.shape != (nt, nk):
            raise ValueError(
                "occupied_mask must have shape (n_state, n_k), "
                f"got {occupied_mask.shape} for energies shape {energies.shape}"
            )

        k_grid_frac = np.asarray(self.k_grid_frac, dtype=float)
        if k_grid_frac.shape != (nk, 2):
            raise ValueError(f"k_grid_frac must have shape ({nk}, 2), got {k_grid_frac.shape}")

        flavor_labels = tuple(self.flavor_labels)
        if flavor_labels and len(flavor_labels) != nt:
            raise ValueError(f"flavor_labels length {len(flavor_labels)} must be 0 or n_state={nt}")

        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "occupied_mask", occupied_mask)
        object.__setattr__(self, "k_grid_frac", k_grid_frac)
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "flavor_labels", flavor_labels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def nt(self) -> int:
        return int(self.energies.shape[0])

    @property
    def nk(self) -> int:
        return int(self.energies.shape[1])

    @property
    def global_energies(self) -> np.ndarray:
        return np.asarray(self.energies, dtype=float).reshape(-1, order="F")

    @property
    def occupied_global_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.occupied_mask, dtype=bool).reshape(-1, order="F"))

    @property
    def unoccupied_global_indices(self) -> np.ndarray:
        return np.flatnonzero((~np.asarray(self.occupied_mask, dtype=bool)).reshape(-1, order="F"))

    def global_index(self, local_index: int, k_index: int) -> int:
        local = int(local_index)
        ik = int(k_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        if ik < 0 or ik >= self.nk:
            raise IndexError(f"k_index={ik} outside [0, {self.nk})")
        return local + self.nt * ik

    def decode_global_index(self, global_index: int) -> tuple[int, int]:
        index = int(global_index)
        if index < 0 or index >= self.nt * self.nk:
            raise IndexError(f"global_index={index} outside [0, {self.nt * self.nk})")
        return index % self.nt, index // self.nt

    def flavor_label(self, local_index: int) -> Any | None:
        local = int(local_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        if not self.flavor_labels:
            return None
        return self.flavor_labels[local]


def canonical_tdhf_orbitals_from_hf_state(
    state: HFState,
    *,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
) -> TDHFCanonicalOrbitals:
    """Normalize a canonical ``HFState`` into TDHF orbital bookkeeping.

    By default the converged HF Hamiltonian ``state.hamiltonian.total`` is
    diagonalized per k point.  Occupations are then read from the canonical
    density projector after transforming it into that HF eigenbasis.  This
    validates that the projector is diagonal and integer-valued in the HF basis.

    ``occupation_policy="energy_sort"`` is an explicit fallback for toy or
    legacy states whose projector is unavailable or intentionally mixed.  It
    fills ``state.density.n_occupied_total`` lowest HF energies and rejects a
    Fermi-level degeneracy within ``degeneracy_tolerance``.
    """

    if occupation_policy not in {"projector", "energy_sort"}:
        raise ValueError("occupation_policy must be 'projector' or 'energy_sort'")
    if projector_tolerance < 0.0:
        raise ValueError("projector_tolerance must be non-negative")
    if degeneracy_tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be non-negative")

    hamiltonian = np.asarray(state.hamiltonian.total, dtype=np.complex128)
    assert_hermitian_field(hamiltonian, name="state.hamiltonian.total")
    energies, eigenvectors = _diagonalize_hamiltonian_field(hamiltonian)

    k_grid_frac = np.asarray(state.basis.k_grid_frac, dtype=float)
    if k_grid_frac.shape != (energies.shape[1], 2):
        raise ValueError(f"state.basis.k_grid_frac must have shape ({energies.shape[1]}, 2), got {k_grid_frac.shape}")

    if occupation_policy == "projector":
        occupied_mask, occupation_metadata = _occupied_mask_from_projector(
            state.density.projector,
            eigenvectors,
            expected_n_occupied=int(state.density.n_occupied_total),
            tolerance=float(projector_tolerance),
        )
    else:
        occupied_mask, occupation_metadata = _occupied_mask_from_energy_sort(
            energies,
            expected_n_occupied=int(state.density.n_occupied_total),
            degeneracy_tolerance=float(degeneracy_tolerance),
        )

    computed_mu = _chemical_potential_from_mask(energies, occupied_mask)
    state_mu = float(state.mu)
    mu = state_mu if np.isfinite(state_mu) else computed_mu

    flavor_labels = tuple(getattr(state.basis, "flavor_labels", ()))
    metadata: dict[str, Any] = {
        "source": "HFState",
        "diagonalization": "numpy.linalg.eigh(state.hamiltonian.total)",
        "occupation_policy": occupation_policy,
        "projector_tolerance": float(projector_tolerance),
        "degeneracy_tolerance": float(degeneracy_tolerance),
        "n_occupied_total": int(np.count_nonzero(occupied_mask)),
        "state_n_occupied_total": int(state.density.n_occupied_total),
        "state_mu": state_mu,
        "computed_mu": computed_mu,
        "active_band_indices": tuple(int(index) for index in state.basis.active_band_indices),
        "active_valence_bands": int(state.basis.active_valence_bands),
        "active_conduction_bands": int(state.basis.active_conduction_bands),
    }
    metadata.update(occupation_metadata)

    return TDHFCanonicalOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occupied_mask,
        k_grid_frac=k_grid_frac,
        mu=mu,
        flavor_labels=flavor_labels,
        metadata=metadata,
    )


def canonical_tdhf_orbitals_from_hf_run_result(
    result: HFRunResult,
    **kwargs: Any,
) -> TDHFCanonicalOrbitals:
    """Normalize ``HFRunResult.final_state`` into canonical TDHF orbitals."""

    orbitals = canonical_tdhf_orbitals_from_hf_state(result.final_state, **kwargs)
    metadata = dict(orbitals.metadata)
    metadata.update(
        {
            "source": "HFRunResult.final_state",
            "hf_run_converged": bool(result.converged),
            "hf_run_exit_reason": str(result.exit_reason),
            "hf_run_best_seed": int(result.best_seed),
            "hf_run_init_mode": str(result.init_mode),
        }
    )
    return replace(orbitals, metadata=metadata)


def _diagonalize_hamiltonian_field(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nt, _nt_rhs, nk = assert_matrix_field_shape(hamiltonian, name="hamiltonian")
    energies = np.empty((nt, nk), dtype=float)
    eigenvectors = np.empty((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        eigenvectors[:, :, ik] = eigvecs
    return energies, eigenvectors


def _occupied_mask_from_projector(
    projector: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    expected_n_occupied: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    projector_array = np.asarray(projector, dtype=np.complex128)
    assert_hermitian_field(projector_array, name="state.density.projector", tol=max(float(tolerance), 1.0e-12))
    nt, _nt_rhs, nk = assert_matrix_field_shape(projector_array, name="state.density.projector")
    if eigenvectors.shape != (nt, nt, nk):
        raise ValueError(
            "state.density.projector shape does not match HF eigenvectors: "
            f"projector {projector_array.shape}, eigenvectors {eigenvectors.shape}"
        )

    projector_hf = np.einsum(
        "aik,abk,bjk->ijk",
        np.conjugate(eigenvectors),
        projector_array,
        eigenvectors,
        optimize=True,
    )
    diagonal = np.einsum("iik->ik", projector_hf)
    off_diagonal = projector_hf.copy()
    index = np.arange(nt)
    off_diagonal[index, index, :] = 0.0

    offdiag_residual = _max_abs(off_diagonal)
    diag_imag_residual = _max_abs(diagonal.imag)
    diagonal_real = diagonal.real
    integer_residual = _max_abs(np.minimum(np.abs(diagonal_real), np.abs(diagonal_real - 1.0)))
    trace_total = float(np.sum(diagonal_real))
    trace_residual = abs(trace_total - float(expected_n_occupied))

    if offdiag_residual > tolerance:
        raise ValueError(
            "density projector is not diagonal in the HF eigenbasis; "
            f"max off-diagonal residual {offdiag_residual:.6e} exceeds {tolerance:.6e}"
        )
    if diag_imag_residual > tolerance:
        raise ValueError(
            "density projector has complex diagonal occupations in the HF eigenbasis; "
            f"max imaginary residual {diag_imag_residual:.6e} exceeds {tolerance:.6e}"
        )
    if integer_residual > tolerance:
        raise ValueError(
            "density projector does not define integer occupations in the HF eigenbasis; "
            f"max 0/1 residual {integer_residual:.6e} exceeds {tolerance:.6e}"
        )
    if trace_residual > max(tolerance * max(1, nt * nk), tolerance):
        raise ValueError(
            "density projector trace does not match DensityState.n_occupied_total; "
            f"trace {trace_total:.12g}, expected {expected_n_occupied}"
        )

    occupied_mask = diagonal_real > 0.5
    n_occupied = int(np.count_nonzero(occupied_mask))
    if n_occupied != int(expected_n_occupied):
        raise ValueError(
            "projector occupation mask does not match DensityState.n_occupied_total; "
            f"mask count {n_occupied}, expected {expected_n_occupied}"
        )

    return occupied_mask, {
        "projector_hf_offdiag_residual": offdiag_residual,
        "projector_hf_diag_imag_residual": diag_imag_residual,
        "projector_hf_integer_residual": integer_residual,
        "projector_hf_trace_residual": trace_residual,
    }


def _occupied_mask_from_energy_sort(
    energies: np.ndarray,
    *,
    expected_n_occupied: int,
    degeneracy_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim != 2:
        raise ValueError(f"energies must have shape (n_state, n_k), got {energy_array.shape}")
    n_total = int(energy_array.size)
    n_occupied = int(expected_n_occupied)
    if n_occupied < 0 or n_occupied > n_total:
        raise ValueError(f"n_occupied_total={n_occupied} outside [0, {n_total}]")

    flat = energy_array.reshape(-1, order="F")
    order = np.argsort(flat, kind="stable")
    fermi_gap = np.inf
    if 0 < n_occupied < n_total:
        highest_occupied = float(flat[order[n_occupied - 1]])
        lowest_unoccupied = float(flat[order[n_occupied]])
        fermi_gap = lowest_unoccupied - highest_occupied
        if fermi_gap <= float(degeneracy_tolerance):
            raise ValueError(
                "energy_sort occupation is ambiguous at a degenerate Fermi level; "
                f"gap {fermi_gap:.6e} is <= {degeneracy_tolerance:.6e}"
            )

    mask_flat = np.zeros(n_total, dtype=bool)
    if n_occupied:
        mask_flat[order[:n_occupied]] = True
    return mask_flat.reshape(energy_array.shape, order="F"), {"energy_sort_fermi_gap": float(fermi_gap)}


def _chemical_potential_from_mask(energies: np.ndarray, occupied_mask: np.ndarray) -> float:
    energy_array = np.asarray(energies, dtype=float)
    mask = np.asarray(occupied_mask, dtype=bool)
    if np.any(mask) and np.any(~mask):
        return 0.5 * (float(np.max(energy_array[mask])) + float(np.min(energy_array[~mask])))
    return float(np.mean(energy_array))


def _max_unitary_residual(eigenvectors: np.ndarray) -> float:
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    if vectors.ndim != 3 or vectors.shape[0] != vectors.shape[1]:
        return np.inf
    nt = vectors.shape[0]
    eye = np.eye(nt, dtype=np.complex128)
    max_residual = 0.0
    for ik in range(vectors.shape[2]):
        residual = vectors[:, :, ik].conjugate().T @ vectors[:, :, ik] - eye
        max_residual = max(max_residual, _max_abs(residual))
    return max_residual


def _max_abs(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


__all__ = [
    "TDHFCanonicalOrbitals",
    "TDHFOccupationPolicy",
    "canonical_tdhf_orbitals_from_hf_run_result",
    "canonical_tdhf_orbitals_from_hf_state",
]
