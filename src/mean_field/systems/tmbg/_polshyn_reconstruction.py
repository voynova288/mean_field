from __future__ import annotations

"""Private projected-HF microscopic reconstruction adapters for the Polshyn tMBG cell.

This module is deliberately kept behind the private ``_polshyn_reconstruction``
boundary.  The doubled-cell microscopic row order can be reconstructed for
flat-k diagnostics, but Polshyn boundary sewing has not been derived or
validated, so the adapter is not exported from the public ``polshyn_supercell``
facade and returned bundles are topology-ineligible.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mean_field.core.contracts import MicroscopicWavefunctionBundle

from ._polshyn_types import PolshynProjectedBasis, PolshynWangHFState

_POLSHYN_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS = 5_000_000


def polshyn_projected_hf_active_index(n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    """Return Polshyn-Wang active-state indices as ``(spin, valley, folded_band)``."""

    resolved = (int(n_spin), int(n_eta), int(nb))
    if any(value <= 0 for value in resolved):
        raise ValueError(f"n_spin, n_eta, and nb must be positive, got {resolved}")
    return np.arange(int(np.prod(resolved)), dtype=int).reshape(resolved, order="F")


def _basis_shape(basis: PolshynProjectedBasis) -> tuple[int, int, int, int, int]:
    raw = np.asarray(basis.wavefunctions, dtype=np.complex128)
    if raw.ndim != 4:
        raise ValueError(
            "Polshyn basis wavefunctions must have shape (basis, folded_band, valley, k), "
            f"got {raw.shape}"
        )
    basis_dim, nb, n_eta, nk = (int(value) for value in raw.shape)
    n_spin = int(basis.n_spin)
    if int(basis.nb) != nb:
        raise ValueError(f"basis.nb={basis.nb} does not match raw folded-band axis {nb}")
    if int(basis.n_eta) != n_eta:
        raise ValueError(f"basis.n_eta={basis.n_eta} does not match raw valley axis {n_eta}")
    if int(basis.nk) != nk:
        raise ValueError(f"basis.nk={basis.nk} does not match raw k axis {nk}")
    if n_spin <= 0:
        raise ValueError(f"basis.n_spin must be positive, got {n_spin}")
    embedding_shape = tuple(int(value) for value in basis.embedding_shape)
    if len(embedding_shape) != 2 or embedding_shape[0] <= 0 or embedding_shape[1] <= 0:
        raise ValueError(f"Polshyn embedding_shape must contain two positive entries, got {basis.embedding_shape}")
    expected_basis_dim = int(basis.local_basis_size) * embedding_shape[0] * embedding_shape[1]
    if basis_dim != expected_basis_dim:
        raise ValueError(
            "Polshyn basis dimension does not match Fortran-flattened doubled-cell embedding: "
            f"raw basis axis {basis_dim} != local_basis_size={int(basis.local_basis_size)} * "
            f"embedding_shape={embedding_shape} -> {expected_basis_dim}"
        )
    return basis_dim, nb, n_eta, nk, n_spin


def _folded_band_labels(basis: PolshynProjectedBasis) -> tuple[dict[str, object], ...]:
    labels: list[dict[str, object]] = []
    for primitive_position, primitive_band_index in enumerate(tuple(int(value) for value in basis.projected_indices)):
        for fold_index in range(2):
            labels.append(
                {
                    "folded_band_index": int(2 * primitive_position + fold_index),
                    "primitive_position": int(primitive_position),
                    "primitive_band_index": int(primitive_band_index),
                    "fold_index": int(fold_index),
                    "fold_momentum": "k" if int(fold_index) == 0 else "k+super_b1",
                    "is_target_band": bool(int(primitive_band_index) == int(basis.target_band_index)),
                }
            )
    if len(labels) != int(basis.nb):
        raise ValueError(f"Polshyn folded band labels length {len(labels)} does not match nb={basis.nb}")
    return tuple(labels)


def _active_basis_labels(basis: PolshynProjectedBasis) -> tuple[dict[str, object], ...]:
    basis_dim, nb, n_eta, _nk, n_spin = _basis_shape(basis)
    del basis_dim
    active = polshyn_projected_hf_active_index(n_spin, n_eta, nb)
    folded_labels = _folded_band_labels(basis)
    labels: list[dict[str, object]] = [dict(active_basis_index=index) for index in range(n_spin * n_eta * nb)]
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(nb):
                active_index = int(active[ispin, ieta, iband])
                labels[active_index] = {
                    "active_basis_index": active_index,
                    "spin_index": int(ispin),
                    "valley_index": int(ieta),
                    **folded_labels[iband],
                }
    return tuple(labels)


def expand_polshyn_projected_micro_basis(basis: PolshynProjectedBasis) -> np.ndarray:
    """Expand raw Polshyn basis arrays to ``(k, micro_row, active_basis)``.

    Raw Polshyn projected bases are stored as
    ``(basis, folded_band, valley, k)`` where ``basis`` is the Fortran
    flattening of the doubled-cell ``(local=6, embed_x, embed_y)`` grid.  The
    expanded microscopic rows are direct-summed as
    ``spin-major -> valley-inner -> basis_F(local,embed_x,embed_y)``.
    """

    raw = np.asarray(basis.wavefunctions, dtype=np.complex128)
    basis_dim, nb, n_eta, nk, n_spin = _basis_shape(basis)
    active = polshyn_projected_hf_active_index(n_spin, n_eta, nb)
    expanded = np.zeros((nk, n_spin * n_eta * basis_dim, n_spin * n_eta * nb), dtype=np.complex128)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            row_start = (ispin * n_eta + ieta) * basis_dim
            row_stop = row_start + basis_dim
            for iband in range(nb):
                active_col = int(active[ispin, ieta, iband])
                expanded[:, row_start:row_stop, active_col] = raw[:, iband, ieta, :].T
    return expanded


def _sector_offdiag_residual(hamiltonian: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> float:
    active = polshyn_projected_hf_active_index(n_spin, n_eta, nb)
    residual = 0.0
    for ik in range(int(hamiltonian.shape[2])):
        block = hamiltonian[:, :, ik]
        for spin_a in range(int(n_spin)):
            for eta_a in range(int(n_eta)):
                rows = np.asarray(active[spin_a, eta_a, :], dtype=int)
                for spin_b in range(int(n_spin)):
                    for eta_b in range(int(n_eta)):
                        if spin_a == spin_b and eta_a == eta_b:
                            continue
                        cols = np.asarray(active[spin_b, eta_b, :], dtype=int)
                        off = block[np.ix_(rows, cols)]
                        if off.size:
                            residual = max(residual, float(np.max(np.abs(off))))
    return residual


def polshyn_wang_active_eigenvectors_from_state(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState,
    *,
    off_sector_atol: float = 1.0e-8,
    hermiticity_atol: float = 1.0e-8,
    stored_energy_atol: float | None = 1.0e-7,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Diagonalize a Polshyn-Wang final Hamiltonian in fixed spin/valley sectors.

    Wang/Polshyn HF stores density in a ``P*`` orientation, but reconstruction
    needs ket coefficients from the final Hermitian HF Hamiltonian.  This helper
    never uses the stored density.  It rejects non-Hermitian or spin/valley
    off-sector Hamiltonians instead of silently symmetrizing them, and it records
    residuals against the stored final-state energies.
    """

    basis_dim, nb, n_eta, nk, n_spin = _basis_shape(basis)
    del basis_dim
    nt = n_spin * n_eta * nb
    hamiltonian = np.asarray(state.hamiltonian, dtype=np.complex128)
    if hamiltonian.shape != (nt, nt, nk):
        raise ValueError(f"Polshyn-Wang state.hamiltonian shape {hamiltonian.shape} does not match {(nt, nt, nk)}")
    if float(off_sector_atol) < 0.0:
        raise ValueError(f"off_sector_atol must be non-negative, got {off_sector_atol}")
    if float(hermiticity_atol) < 0.0:
        raise ValueError(f"hermiticity_atol must be non-negative, got {hermiticity_atol}")
    if stored_energy_atol is not None and float(stored_energy_atol) < 0.0:
        raise ValueError(f"stored_energy_atol must be non-negative or None, got {stored_energy_atol}")

    hermiticity_residual = float(np.max(np.abs(hamiltonian - hamiltonian.conjugate().swapaxes(0, 1))))
    if hermiticity_residual > float(hermiticity_atol):
        raise ValueError(
            "Polshyn-Wang final Hamiltonian is not Hermitian enough for reconstruction; "
            f"max residual {hermiticity_residual:.6e} exceeds {float(hermiticity_atol):.6e}"
        )
    off_sector_residual = _sector_offdiag_residual(hamiltonian, n_spin=n_spin, n_eta=n_eta, nb=nb)
    if off_sector_residual > float(off_sector_atol):
        raise ValueError(
            "Polshyn-Wang sector reconstruction requires a spin/valley block-diagonal HF Hamiltonian; "
            f"max off-sector element {off_sector_residual:.6e} exceeds {float(off_sector_atol):.6e}"
        )

    stored_energies = np.asarray(state.energies, dtype=float)
    if stored_energies.shape != (nt, nk):
        raise ValueError(f"Polshyn-Wang state.energies shape {stored_energies.shape} does not match {(nt, nk)}")

    active = polshyn_projected_hf_active_index(n_spin, n_eta, nb)
    coeffs = np.zeros((nt, nt, nk), dtype=np.complex128)
    energies = np.zeros((nt, nk), dtype=float)
    max_eigen_residual = 0.0
    max_unitarity_residual = 0.0
    for ik in range(nk):
        for ispin in range(n_spin):
            for ieta in range(n_eta):
                indices = np.asarray(active[ispin, ieta, :], dtype=int)
                block = hamiltonian[:, :, ik][np.ix_(indices, indices)]
                evals, evecs = np.linalg.eigh(block)
                energies[indices, ik] = evals
                coeffs[np.ix_(indices, indices, [ik])] = evecs[:, :, None]
                max_eigen_residual = max(
                    max_eigen_residual,
                    float(np.max(np.abs(block @ evecs - evecs * evals[None, :]))),
                )
                max_unitarity_residual = max(
                    max_unitarity_residual,
                    float(np.max(np.abs(evecs.conjugate().T @ evecs - np.eye(nb, dtype=np.complex128)))),
                )
    stored_energy_residual = float(np.max(np.abs(stored_energies - energies)))
    if stored_energy_atol is not None and stored_energy_residual > float(stored_energy_atol):
        raise ValueError(
            "Polshyn-Wang stored energies do not match the final-Hamiltonian sector eigensystem; "
            f"max residual {stored_energy_residual:.6e} exceeds {float(stored_energy_atol):.6e}"
        )
    diagnostics = {
        "hamiltonian_hermiticity_residual": float(hermiticity_residual),
        "off_sector_hamiltonian_residual": float(off_sector_residual),
        "active_eigensystem_residual": float(max_eigen_residual),
        "active_eigenvector_unitarity_residual": float(max_unitarity_residual),
        "stored_energy_eigh_residual": float(stored_energy_residual),
    }
    return coeffs, energies, diagnostics


def _normalize_polshyn_reconstruction_state_indices(
    state_indices: int | Iterable[int] | None,
    n_state: int,
) -> tuple[int, ...]:
    n_state_i = int(n_state)
    if n_state_i <= 0:
        raise ValueError(f"n_state must be positive, got {n_state}")
    if state_indices is None:
        return tuple(range(n_state_i))
    if isinstance(state_indices, (str, bytes)):
        raise TypeError("state_indices must be an integer, an iterable of integers, or None")
    if isinstance(state_indices, (int, np.integer)):
        out = (int(state_indices),)
    else:
        out = tuple(int(index) for index in state_indices)
    if not out:
        raise ValueError("At least one selected HF state index is required")
    if min(out) < 0 or max(out) >= n_state_i:
        raise ValueError(f"HF state indices {out} outside [0, {n_state_i})")
    return out


def _select_active_eigenvectors(
    active_eigenvectors: np.ndarray,
    *,
    n_active: int,
    nk: int,
    state_indices: int | Iterable[int] | None,
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
    if coeffs.ndim != 3 or coeffs.shape[0] != int(n_active) or coeffs.shape[2] != int(nk):
        raise ValueError(
            "Polshyn active_eigenvectors must have shape "
            f"({int(n_active)}, n_reconstructed_state, {int(nk)}), got {coeffs.shape}"
        )
    if coeffs.shape[1] == int(n_active):
        selected = _normalize_polshyn_reconstruction_state_indices(state_indices, int(n_active))
        return coeffs[:, list(selected), :], selected, True
    if state_indices is None:
        raise ValueError(
            "Rectangular selected active_eigenvectors require explicit state_indices so metadata can retain "
            "the original final-HF state labels"
        )
    selected = _normalize_polshyn_reconstruction_state_indices(state_indices, int(n_active))
    if coeffs.shape[1] != len(selected):
        raise ValueError(
            "Rectangular selected active_eigenvectors second axis must match state_indices length; "
            f"got {coeffs.shape[1]} coefficient columns for selected states {selected}"
        )
    return coeffs, selected, False


def _polshyn_reconstruction_output_element_count(
    basis: PolshynProjectedBasis,
    *,
    n_output_states: int,
) -> int:
    basis_dim, _nb, n_eta, nk, n_spin = _basis_shape(basis)
    micro_dim = int(n_spin * n_eta * basis_dim)
    return int(nk) * micro_dim * int(n_output_states)


def _validate_polshyn_reconstruction_output_size(
    basis: PolshynProjectedBasis,
    *,
    n_output_states: int,
    max_dense_elements: int | None,
) -> int:
    dense_elements = _polshyn_reconstruction_output_element_count(basis, n_output_states=int(n_output_states))
    if max_dense_elements is not None:
        max_elements = int(max_dense_elements)
        if max_elements < 0:
            raise ValueError("max_dense_elements must be non-negative or None")
        if dense_elements > max_elements:
            raise ValueError(
                "Polshyn projected-HF dense reconstruction would exceed the explicit size guard: "
                f"estimated {dense_elements} complex output elements for {int(n_output_states)} "
                f"selected state(s) > max_dense_elements={max_elements}. Increase max_dense_elements "
                "only for an intentional reconstruction call."
            )
    return dense_elements


def _selected_unitarity_residual(
    selected_coeffs: np.ndarray,
    *,
    unitarity_atol: float | None,
) -> float | None:
    if unitarity_atol is None:
        return None
    if float(unitarity_atol) < 0.0:
        raise ValueError(f"unitarity_atol must be non-negative or None, got {unitarity_atol}")
    n_selected = int(selected_coeffs.shape[1])
    gram = np.einsum("ahk,amk->hmk", selected_coeffs.conjugate(), selected_coeffs, optimize=True)
    residual = float(np.max(np.abs(gram - np.eye(n_selected, dtype=np.complex128)[:, :, None])))
    if residual > float(unitarity_atol):
        raise ValueError(
            "Polshyn selected active_eigenvectors must be orthonormal at each k point; "
            f"max column-Gram residual {residual:.6e} exceeds {float(unitarity_atol):.6e}"
        )
    return residual


def _contract_polshyn_selected_micro_wavefunctions(
    basis: PolshynProjectedBasis,
    selected_coeffs: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(basis.wavefunctions, dtype=np.complex128)
    basis_dim, nb, n_eta, nk, n_spin = _basis_shape(basis)
    active = polshyn_projected_hf_active_index(n_spin, n_eta, nb)
    n_selected = int(selected_coeffs.shape[1])
    micro_dim = int(n_spin * n_eta * basis_dim)
    psi_flat = np.zeros((nk, micro_dim, n_selected), dtype=np.complex128)
    for ik in range(nk):
        for ispin in range(n_spin):
            for ieta in range(n_eta):
                row_start = (ispin * n_eta + ieta) * basis_dim
                row_stop = row_start + basis_dim
                active_rows = np.asarray(active[ispin, ieta, :], dtype=int)
                psi_flat[ik, row_start:row_stop, :] = raw[:, :, ieta, ik] @ selected_coeffs[active_rows, :, ik]
    return psi_flat


def _k_grid_frac(basis: PolshynProjectedBasis) -> np.ndarray | None:
    raw = basis.k_grid_frac
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.size != int(basis.nk) * 2:
        raise ValueError(f"Polshyn k_grid_frac shape {arr.shape} is incompatible with nk={basis.nk}")
    return arr.reshape((int(basis.nk), 2))


def _embedding_positions_metadata(basis: PolshynProjectedBasis) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for key, value in sorted(dict(basis.embedding_positions).items()):
        n1, n2, fold = (int(key[0]), int(key[1]), int(key[2]))
        ix, iy = (int(value[0]), int(value[1]))
        out.append({"primitive_n1": n1, "primitive_n2": n2, "fold": fold, "embed_x": ix, "embed_y": iy})
    return out


def _state_labels(
    labels: Sequence[Mapping[str, object]] | None,
    selected_state_indices: Sequence[int],
    *,
    eigenvector_source: str,
) -> tuple[dict[str, object], ...]:
    selected = tuple(int(index) for index in selected_state_indices)
    if labels is None:
        return tuple(
            {
                "hf_state_index": int(index),
                "state_source": str(eigenvector_source),
            }
            for index in selected
        )
    out = tuple(dict(label) for label in labels)
    if len(out) != len(selected):
        raise ValueError(f"state_labels length {len(out)} must match selected state count={len(selected)}")
    return out


def _metadata(
    basis: PolshynProjectedBasis,
    *,
    eigenvector_source: str,
    eigensystem_diagnostics: Mapping[str, float] | None,
    selected_state_indices: Sequence[int],
    n_reconstructed_states: int,
    dense_elements: int,
    all_dense_elements: int,
    max_dense_elements: int | None,
    selected_coefficients_from_full_eigensystem: bool,
) -> dict[str, object]:
    basis_dim, nb, n_eta, nk, n_spin = _basis_shape(basis)
    diagnostics = {key: float(value) for key, value in dict(eigensystem_diagnostics or {}).items()}
    return {
        "system": "tmbg_polshyn_doubled",
        "reconstruction_api_status": "private_experimental_not_exported_from_polshyn_supercell",
        "public_facade_exported": False,
        "reconstruction_adapter": "mean_field.systems.tmbg._polshyn_reconstruction.reconstruct_polshyn_wang_hf_micro_wavefunctions",
        "selected_state_contraction": "system-specific rectangular contraction; common helper currently requires square all-state coefficients",
        "raw_wavefunctions_axis_order": "basis,folded_band,valley,k",
        "expanded_micro_basis_axis_order": "k,microscopic_basis,active_basis",
        "microscopic_row_order": "spin_major,valley_inner,basis_F(local=6,embed_x,embed_y)",
        "active_column_order": "np.arange(n_spin*n_eta*nb).reshape((n_spin,n_eta,nb), order='F')",
        "active_eigenvectors_axis_order": "active_basis,hf_state,k",
        "raw_basis_dim": int(basis_dim),
        "n_spin": int(n_spin),
        "n_valley": int(n_eta),
        "n_folded_band": int(nb),
        "n_k": int(nk),
        "projected_indices": [int(value) for value in basis.projected_indices],
        "target_band_index": int(basis.target_band_index),
        "folded_band_rule": "folded_index = 2*primitive_position + fold; fold 0 at k, fold 1 at k+super_b1",
        "folded_band_labels": _folded_band_labels(basis),
        "active_basis_labels": _active_basis_labels(basis),
        "supercell": basis.supercell.as_dict(),
        "supercell_reciprocal_vectors_nm_inv": [
            [float(complex(basis.super_b1).real), float(complex(basis.super_b1).imag)],
            [float(complex(basis.super_b2).real), float(complex(basis.super_b2).imag)],
        ],
        "embedding_shape": [int(value) for value in basis.embedding_shape],
        "embedding_origin": [int(value) for value in basis.embedding_origin],
        "embedding_positions": _embedding_positions_metadata(basis),
        "k_flat_order": "_supercell_k_grid loops iy/f2 outer and ix/f1 inner; adapter keeps flat k order and does not attach grid_shape",
        "grid_shape_attached": False,
        "sewing_available": False,
        "sewing_transforms_attached": False,
        "sewing_blocker": "Polshyn doubled-cell projected-micro sewing has not been derived/tested",
        "topology_eligible": False,
        "topology_ineligible_reason": "Polshyn doubled-cell sewing is unavailable; flat-k reconstructed bundles are not validated torus bundles",
        "topology_policy": "refuse/avoid FHS torus topology unless a future explicit diagnostic path derives sewing",
        "eigenvector_source": str(eigenvector_source),
        "selected_hf_state_indices": [int(index) for index in selected_state_indices],
        "n_reconstructed_states": int(n_reconstructed_states),
        "selected_coefficients_from_full_eigensystem": bool(selected_coefficients_from_full_eigensystem),
        "dense_reconstruction_estimated_elements": int(dense_elements),
        "dense_reconstruction_estimated_all_state_elements": int(all_dense_elements),
        "max_dense_elements": None if max_dense_elements is None else int(max_dense_elements),
        **diagnostics,
    }


def reconstruct_polshyn_wang_hf_micro_wavefunctions(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState | None = None,
    active_eigenvectors: np.ndarray | None = None,
    *,
    include_sewing: bool = False,
    state_indices: int | Iterable[int] | None = None,
    state_labels: Sequence[Mapping[str, object]] | None = None,
    basis_metadata: Mapping[str, Any] | None = None,
    max_dense_elements: int | None = _POLSHYN_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS,
    off_sector_atol: float = 1.0e-8,
    hermiticity_atol: float = 1.0e-8,
    stored_energy_atol: float | None = 1.0e-7,
    unitarity_atol: float | None = 1.0e-8,
) -> MicroscopicWavefunctionBundle:
    """Reconstruct flat-k Polshyn-Wang projected-HF wavefunctions in microscopic rows.

    This is a private, explicit adapter.  It supports selected final-HF states
    and a dense-output size guard, but it intentionally does not attach boundary
    sewing or a 2D grid shape.  Returned bundles are diagnostic flat-k objects,
    not topology-ready torus wavefunction bundles.
    """

    if bool(include_sewing):
        raise NotImplementedError(
            "Polshyn doubled-cell projected-micro sewing is not implemented; "
            "reconstruction can return flat wavefunctions only"
        )
    basis_dim, nb, n_eta, nk, n_spin = _basis_shape(basis)
    del basis_dim
    n_active = int(n_spin * n_eta * nb)
    eigensystem_diagnostics: dict[str, float] | None = None
    if active_eigenvectors is None:
        if state is None:
            raise ValueError("Either state or active_eigenvectors must be supplied for Polshyn reconstruction")
        coeffs, _energies, eigensystem_diagnostics = polshyn_wang_active_eigenvectors_from_state(
            basis,
            state,
            off_sector_atol=off_sector_atol,
            hermiticity_atol=hermiticity_atol,
            stored_energy_atol=stored_energy_atol,
        )
        eigenvector_source = "sector_np.linalg.eigh(final_polshyn_wang_hamiltonian)"
    else:
        coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
        eigenvector_source = "caller_provided_active_eigenvectors"

    selected_coeffs, selected, selected_from_full = _select_active_eigenvectors(
        coeffs,
        n_active=n_active,
        nk=nk,
        state_indices=state_indices,
    )
    dense_elements = _validate_polshyn_reconstruction_output_size(
        basis,
        n_output_states=len(selected),
        max_dense_elements=max_dense_elements,
    )
    all_dense_elements = _polshyn_reconstruction_output_element_count(basis, n_output_states=n_active)
    unitarity_residual = _selected_unitarity_residual(selected_coeffs, unitarity_atol=unitarity_atol)
    psi_flat = _contract_polshyn_selected_micro_wavefunctions(basis, selected_coeffs)

    kvec_arr = np.asarray(basis.kvec, dtype=np.complex128).reshape(-1)
    if kvec_arr.shape != (nk,):
        raise ValueError(f"Polshyn kvec must have shape ({nk},), got {kvec_arr.shape}")
    k_grid_frac = _k_grid_frac(basis)
    labels = _state_labels(state_labels, selected, eigenvector_source=eigenvector_source)

    metadata = dict(basis_metadata or {})
    metadata.update(
        _metadata(
            basis,
            eigenvector_source=eigenvector_source,
            eigensystem_diagnostics=eigensystem_diagnostics,
            selected_state_indices=selected,
            n_reconstructed_states=len(selected),
            dense_elements=dense_elements,
            all_dense_elements=all_dense_elements,
            max_dense_elements=max_dense_elements,
            selected_coefficients_from_full_eigensystem=selected_from_full,
        )
    )
    metadata.update(
        {
            "micro_basis_axis_order": "k,microscopic_basis,active_basis",
            "input_micro_basis_axes": {
                "raw_projected_basis_axis_order": "basis,folded_band,valley,k",
                "spin_axis": "direct_sum_repeated_projected_basis",
                "basis_flattening": "Fortran order over (local=6,embed_x,embed_y)",
            },
            "psi_micro_axis_order": "k,microscopic_basis,hf_state",
            "microscopic_basis_dim": int(psi_flat.shape[1]),
            "n_active": int(n_active),
            "state_labels": labels,
            "kvec_provided": True,
            "selected_state_allocation": "output_axis_contains_only_selected_hf_states",
        }
    )
    if k_grid_frac is not None:
        metadata["k_grid_frac_shape"] = [int(nk), 2]
    if unitarity_residual is not None:
        metadata["active_eigenvectors_unitarity_residual"] = float(unitarity_residual)

    return MicroscopicWavefunctionBundle(
        kvec=kvec_arr,
        psi_micro=psi_flat,
        sewing_transforms=(),
        basis_metadata=metadata,
        source="hf_reconstructed",
    )


# Intentionally not re-exported by mean_field.systems.tmbg.polshyn_supercell.
# ``__all__`` documents the testable helpers inside this private module only.
__all__ = [
    "expand_polshyn_projected_micro_basis",
    "polshyn_projected_hf_active_index",
    "polshyn_wang_active_eigenvectors_from_state",
    "reconstruct_polshyn_wang_hf_micro_wavefunctions",
]
