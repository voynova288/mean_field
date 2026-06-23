from __future__ import annotations

"""Projected-HF microscopic reconstruction adapters for RnG/hBN.

This module is a system layer around the common
``mean_field.core.hf.reconstruction`` contraction helper.  It only expands the
RnG/hBN projected single-particle basis into the direct-sum spin/flavor
microscopic row convention used by ``sewing.py``.  Full-state reconstruction
uses the common ``basis @ HF-eigenvector`` helper; selected-state reconstruction
uses the same axis contract without allocating the full state bundle.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mean_field.core.contracts import MicroscopicWavefunctionBundle
from mean_field.core.hf import reconstruct_projected_micro_wavefunctions

from ._hf_types import RLGhBNHartreeFockRun, RLGhBNHartreeFockState, RLGhBNProjectedBasisData
from .sewing import rlg_hbn_projected_micro_sewing_transforms

_RLG_HBN_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS = 5_000_000


def rlg_hbn_projected_hf_active_index(n_spin: int, n_eta: int, n_band: int) -> np.ndarray:
    """Return the RnG/hBN active-state index tensor ``(spin, flavor, band)``.

    The order is the core-HF Fortran convention used when the projected H0 is
    built in ``_hf_basis.py``:
    ``np.arange(nt).reshape((n_spin, n_eta, n_band), order="F")``.
    """

    resolved = (int(n_spin), int(n_eta), int(n_band))
    if any(value <= 0 for value in resolved):
        raise ValueError(f"n_spin, n_eta, and n_band must be positive, got {resolved}")
    return np.arange(int(np.prod(resolved)), dtype=int).reshape(resolved, order="F")


def _final_hf_hermiticity_residual(hamiltonian: np.ndarray) -> float:
    hmat = np.asarray(hamiltonian, dtype=np.complex128)
    if hmat.ndim != 3 or hmat.shape[0] != hmat.shape[1]:
        raise ValueError(f"Expected final HF Hamiltonian shape (active, active, k), got {hmat.shape}")
    return float(np.max(np.abs(hmat - hmat.conjugate().swapaxes(0, 1)))) if hmat.size else 0.0


def build_rlg_hbn_final_hf_eigensystem(
    state: RLGhBNHartreeFockState,
    *,
    hermiticity_atol: float | None = 1.0e-8,
) -> Any:
    """Return the final-HF eigensystem used by microscopic reconstruction.

    The existing RnG/hBN orbital implementation lives in the TDHF module because
    TDHF also needs the same converged-HF single-particle orbitals.  This neutral
    wrapper is the reconstruction boundary: callers ask for a final-HF
    eigensystem, not for a TDHF object, and the legacy helper remains an
    implementation detail.
    """

    residual = _final_hf_hermiticity_residual(np.asarray(state.hamiltonian, dtype=np.complex128))
    if hermiticity_atol is not None:
        tolerance = float(hermiticity_atol)
        if tolerance < 0.0:
            raise ValueError(f"hermiticity_atol must be non-negative or None, got {hermiticity_atol}")
        if residual > tolerance:
            raise ValueError(
                "RnG/hBN final HF Hamiltonian is not Hermitian enough for microscopic reconstruction; "
                f"max residual {residual:.6e} exceeds {tolerance:.6e}"
            )

    from ._tdhf_orbitals import build_rlg_hbn_tdhf_orbitals

    return build_rlg_hbn_tdhf_orbitals(state)


def _basis_shape(data: RLGhBNProjectedBasisData) -> tuple[int, int, int, int, int]:
    raw = np.asarray(data.basis.wavefunctions, dtype=np.complex128)
    if raw.ndim != 4:
        raise ValueError(f"RnG/hBN projected basis wavefunctions must have shape (basis, band, flavor, k), got {raw.shape}")
    basis_dim, n_band, n_eta, nk = (int(value) for value in raw.shape)
    n_spin = int(data.basis.n_spin)
    if int(data.basis.n_band) != n_band:
        raise ValueError(f"basis.n_band={data.basis.n_band} does not match raw wavefunction band axis {n_band}")
    if int(data.basis.n_flavor) != n_eta:
        raise ValueError(f"basis.n_flavor={data.basis.n_flavor} does not match raw wavefunction flavor axis {n_eta}")
    if int(data.basis.nk) != nk:
        raise ValueError(f"basis.nk={data.basis.nk} does not match raw wavefunction k axis {nk}")
    return basis_dim, n_band, n_eta, nk, n_spin


def _normalize_reconstruction_state_indices(
    state_indices: int | Iterable[int] | None,
    n_state: int,
) -> tuple[int, ...]:
    if state_indices is None:
        return tuple(range(int(n_state)))
    if isinstance(state_indices, (str, bytes)):
        raise TypeError("state_indices must be an integer, an iterable of integers, or None")
    if isinstance(state_indices, (int, np.integer)):
        out = (int(state_indices),)
    else:
        out = tuple(int(index) for index in state_indices)
    if not out:
        raise ValueError("At least one HF state index is required")
    if min(out) < 0 or max(out) >= int(n_state):
        raise ValueError(f"HF state indices {out} outside [0, {int(n_state)})")
    return out


def _reconstruction_output_element_count(data: RLGhBNProjectedBasisData, n_state: int) -> int:
    basis_dim, _n_band, n_eta, nk, n_spin = _basis_shape(data)
    micro_dim = int(n_spin * n_eta * basis_dim)
    return int(nk) * int(micro_dim) * int(n_state)


def _validate_reconstruction_size(
    data: RLGhBNProjectedBasisData,
    n_state: int,
    max_dense_elements: int | None,
) -> int:
    dense_elements = _reconstruction_output_element_count(data, int(n_state))
    if max_dense_elements is not None:
        max_elements = int(max_dense_elements)
        if max_elements < 0:
            raise ValueError("max_dense_elements must be non-negative or None")
        if dense_elements > max_elements:
            raise ValueError(
                "RnG/hBN projected-HF dense reconstruction would exceed the explicit size guard: "
                f"estimated {dense_elements} complex output elements > max_dense_elements={max_elements}. "
                "Pass state_indices to reconstruct selected HF states, or increase max_dense_elements only "
                "for an intentional dense reconstruction call."
            )
    return dense_elements


def _prepare_active_coefficients(
    data: RLGhBNProjectedBasisData,
    active_eigenvectors: np.ndarray,
    state_indices: int | Iterable[int] | None,
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    _basis_dim, n_band, n_eta, nk, n_spin = _basis_shape(data)
    n_active = int(n_spin * n_eta * n_band)
    coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError(
            "active_eigenvectors must have shape (active_basis, hf_state, k); "
            f"got {coeffs.shape}"
        )
    if coeffs.shape[0] != n_active or coeffs.shape[2] != nk:
        raise ValueError(
            f"active_eigenvectors must have shape ({n_active}, hf_state, {nk}); got {coeffs.shape}"
        )
    if coeffs.shape[1] <= 0:
        raise ValueError("active_eigenvectors must include at least one HF state column")

    selected = _normalize_reconstruction_state_indices(state_indices, n_active)
    if state_indices is None:
        if coeffs.shape[1] != n_active:
            raise ValueError(
                "Rectangular selected active_eigenvectors require explicit state_indices so state labels remain unambiguous; "
                f"got second axis {coeffs.shape[1]} for n_active={n_active}"
            )
        return coeffs, selected, True

    if coeffs.shape[1] == n_active:
        return coeffs[:, list(selected), :], selected, True
    if coeffs.shape[1] == len(selected):
        return coeffs, selected, False
    raise ValueError(
        "active_eigenvectors second axis must be either the full active-state count "
        f"{n_active} or len(state_indices)={len(selected)}; got {coeffs.shape[1]}"
    )


def _selected_state_labels(
    state_labels: Sequence[Mapping[str, object]] | None,
    *,
    selected: tuple[int, ...],
    n_full_state: int,
    eigenvector_source: str,
) -> tuple[dict[str, object], ...]:
    if state_labels is None:
        return tuple(
            {"hf_state_index": int(index), "state_source": str(eigenvector_source)}
            for index in selected
        )
    labels = tuple(dict(label) for label in state_labels)
    if len(labels) == int(n_full_state):
        return tuple(labels[index] for index in selected)
    if len(labels) == len(selected):
        return labels
    raise ValueError(
        "state_labels length must match either the full active-state count "
        f"{int(n_full_state)} or selected state count {len(selected)}; got {len(labels)}"
    )


def _active_coefficients_unitarity_residual(
    coeffs: np.ndarray,
    *,
    unitarity_atol: float | None,
) -> float | None:
    if unitarity_atol is None:
        return None
    tolerance = float(unitarity_atol)
    if tolerance < 0.0:
        raise ValueError(f"unitarity_atol must be non-negative or None, got {unitarity_atol}")
    n_state = int(coeffs.shape[1])
    gram = np.einsum("ahk,amk->hmk", coeffs.conjugate(), coeffs, optimize=True)
    residual = float(np.max(np.abs(gram - np.eye(n_state, dtype=np.complex128)[:, :, None])))
    if residual > tolerance:
        raise ValueError(
            "selected active_eigenvectors must be orthonormal at each k point; "
            f"max column-Gram residual {residual:.6e} exceeds {tolerance:.6e}"
        )
    return residual


def _active_basis_labels(data: RLGhBNProjectedBasisData) -> tuple[dict[str, object], ...]:
    basis_dim, n_band, n_eta, _nk, n_spin = _basis_shape(data)
    del basis_dim
    active = rlg_hbn_projected_hf_active_index(n_spin, n_eta, n_band)
    valleys = tuple(int(value) for value in getattr(data, "valleys", tuple(range(n_eta))))
    physical_bands = tuple(int(value) for value in getattr(data, "active_band_indices", tuple(range(n_band))))
    labels: list[dict[str, object]] = [dict(active_basis_index=index) for index in range(n_spin * n_eta * n_band)]
    for ispin in range(n_spin):
        for iflavor in range(n_eta):
            valley = valleys[iflavor] if iflavor < len(valleys) else iflavor
            for iband in range(n_band):
                active_index = int(active[ispin, iflavor, iband])
                labels[active_index] = {
                    "active_basis_index": active_index,
                    "spin_index": int(ispin),
                    "flavor_index": int(iflavor),
                    "valley": int(valley),
                    "projected_band_index": int(iband),
                    "physical_band_index": int(physical_bands[iband]) if iband < len(physical_bands) else int(iband),
                }
    return tuple(labels)


def expand_rlg_hbn_projected_micro_basis(data: RLGhBNProjectedBasisData) -> np.ndarray:
    """Expand raw RnG/hBN basis arrays to ``(k, micro_row, active_basis)``.

    Raw RnG/hBN projected bases are stored as ``(basis, band, flavor, k)`` where
    ``basis`` is the Fortran flattening of ``(local, nx, ny)``.  The expanded
    microscopic rows are direct-summed as
    ``spin-major -> flavor-inner -> basis_F(local,nx,ny)`` so that the first
    axis matches ``rlg_hbn_projected_micro_sewing_transforms``.
    """

    raw = np.asarray(data.basis.wavefunctions, dtype=np.complex128)
    basis_dim, n_band, n_eta, nk, n_spin = _basis_shape(data)
    active = rlg_hbn_projected_hf_active_index(n_spin, n_eta, n_band)
    expanded = np.zeros((nk, n_spin * n_eta * basis_dim, n_spin * n_eta * n_band), dtype=np.complex128)
    for ispin in range(n_spin):
        for iflavor in range(n_eta):
            row_start = (ispin * n_eta + iflavor) * basis_dim
            row_stop = row_start + basis_dim
            for iband in range(n_band):
                active_col = int(active[ispin, iflavor, iband])
                expanded[:, row_start:row_stop, active_col] = raw[:, iband, iflavor, :].T
    return expanded


def _contract_selected_micro_wavefunctions(data: RLGhBNProjectedBasisData, coeffs: np.ndarray) -> np.ndarray:
    """Contract raw RnG/hBN basis rows with a selected rectangular HF eigensystem."""

    raw = np.asarray(data.basis.wavefunctions, dtype=np.complex128)
    basis_dim, n_band, n_eta, nk, n_spin = _basis_shape(data)
    n_selected = int(coeffs.shape[1])
    active = rlg_hbn_projected_hf_active_index(n_spin, n_eta, n_band)
    psi_flat = np.zeros((nk, n_spin * n_eta * basis_dim, n_selected), dtype=np.complex128)
    for ispin in range(n_spin):
        for iflavor in range(n_eta):
            row_start = (ispin * n_eta + iflavor) * basis_dim
            row_stop = row_start + basis_dim
            for iband in range(n_band):
                active_col = int(active[ispin, iflavor, iband])
                basis_by_k = raw[:, iband, iflavor, :].T
                weights_by_k_state = coeffs[active_col, :, :].T
                psi_flat[:, row_start:row_stop, :] += basis_by_k[:, :, None] * weights_by_k_state[:, None, :]
    return psi_flat


def _regular_grid_shape(data: RLGhBNProjectedBasisData, *, as_grid: bool) -> tuple[int, int] | None:
    if not bool(as_grid):
        return None
    mesh = int(getattr(data, "mesh_size", 0))
    nk = int(data.basis.nk)
    if mesh <= 0:
        return None
    if mesh * mesh != nk:
        raise ValueError(f"RnG/hBN mesh_size={mesh} is incompatible with nk={nk}")
    return (mesh, mesh)


def _k_grid_frac(data: RLGhBNProjectedBasisData) -> np.ndarray | None:
    raw = getattr(data, "k_grid_frac", None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    nk = int(data.basis.nk)
    if arr.size != nk * 2:
        raise ValueError(f"RnG/hBN k_grid_frac shape {arr.shape} is incompatible with nk={nk}")
    return arr.reshape((nk, 2))


def _metadata(
    data: RLGhBNProjectedBasisData,
    *,
    eigenvector_source: str,
    include_sewing: bool,
    as_grid: bool,
) -> dict[str, object]:
    basis_dim, n_band, n_eta, nk, n_spin = _basis_shape(data)
    return {
        "system": "RnG_hBN",
        "system_alias": "RLG-hBN",
        "reconstruction_adapter": "mean_field.systems.RnG_hBN.hf.reconstruct_rlg_hbn_projected_hf_micro_wavefunctions",
        "implementation_module": "mean_field.systems.RnG_hBN._hf_reconstruction",
        "final_hf_eigensystem_helper": "mean_field.systems.RnG_hBN.hf.build_rlg_hbn_final_hf_eigensystem",
        "common_helper": "mean_field.core.hf.reconstruction.reconstruct_projected_micro_wavefunctions",
        "raw_wavefunctions_axis_order": "basis,band,flavor,k",
        "expanded_micro_basis_axis_order": "k,microscopic_basis,active_basis",
        "microscopic_row_order": "spin_major,flavor_inner,basis_F(local,nx,ny)",
        "active_column_order": "np.arange(nt).reshape((n_spin,n_flavor,n_band), order='F')",
        "active_eigenvectors_axis_order": "active_basis,hf_state,k",
        "raw_basis_dim": int(basis_dim),
        "n_spin": int(n_spin),
        "n_flavor": int(n_eta),
        "n_band": int(n_band),
        "n_k": int(nk),
        "local_basis_size": int(data.basis.local_basis_size),
        "reciprocal_grid_shape": [int(value) for value in data.basis.grid_shape],
        "reciprocal_grid_origin": [int(value) for value in getattr(data, "reciprocal_grid_origin", (0, 0))],
        "valleys": [int(value) for value in getattr(data, "valleys", tuple(range(n_eta)))],
        "active_band_indices_per_band": [int(value) for value in getattr(data, "active_band_indices", tuple(range(n_band)))],
        "flat_band_indices": [int(value) for value in getattr(data, "flat_band_indices", ())],
        "k_flat_order": "C flatten of build_moire_k_grid fractional grid: f1/indexing-ij axis outer, f2 axis inner",
        "requested_grid_output": bool(as_grid),
        "sewing_transforms_attached": bool(include_sewing),
        "sewing_row_order_contract": "matches src/mean_field/systems/RnG_hBN/sewing.py spin-major blocks",
        "sewing_validation_status": "software row-order tests only; physical seam/topology validation remains separate",
        "eigenvector_source": str(eigenvector_source),
        "active_basis_labels": _active_basis_labels(data),
    }


def _resolve_data_and_state(
    run_or_basis_data: RLGhBNHartreeFockRun | RLGhBNProjectedBasisData,
) -> tuple[RLGhBNProjectedBasisData, Any | None]:
    if isinstance(run_or_basis_data, RLGhBNHartreeFockRun):
        return run_or_basis_data.basis_data, run_or_basis_data.state
    if isinstance(run_or_basis_data, RLGhBNProjectedBasisData):
        return run_or_basis_data, None
    if hasattr(run_or_basis_data, "basis_data") and hasattr(run_or_basis_data, "state"):
        return run_or_basis_data.basis_data, run_or_basis_data.state
    return run_or_basis_data, None


def _selected_bundle(
    data: RLGhBNProjectedBasisData,
    coeffs: np.ndarray,
    *,
    k_grid_shape: tuple[int, int] | None,
    state_labels: Sequence[Mapping[str, object]] | None,
    selected: tuple[int, ...],
    eigenvector_source: str,
    sewing_transforms: Sequence[Any],
    metadata: Mapping[str, Any],
    unitarity_atol: float | None,
) -> MicroscopicWavefunctionBundle:
    basis_dim, n_band, n_eta, nk, n_spin = _basis_shape(data)
    n_full_state = int(n_spin * n_eta * n_band)
    unitarity_residual = _active_coefficients_unitarity_residual(coeffs, unitarity_atol=unitarity_atol)
    labels = _selected_state_labels(
        state_labels,
        selected=selected,
        n_full_state=n_full_state,
        eigenvector_source=eigenvector_source,
    )
    kvec_arr = np.asarray(data.kvec, dtype=np.complex128).reshape(-1)
    if kvec_arr.shape != (nk,):
        raise ValueError(f"kvec must have shape ({nk},), got {kvec_arr.shape}")
    psi_flat = _contract_selected_micro_wavefunctions(data, coeffs)
    micro_dim = int(n_spin * n_eta * basis_dim)
    psi = psi_flat if k_grid_shape is None else psi_flat.reshape((*k_grid_shape, micro_dim, len(selected)), order="C")
    bundle_metadata = dict(metadata)
    bundle_metadata.update(
        {
            "micro_basis_axis_order": "k,microscopic_basis,active_basis",
            "active_eigenvectors_axis_order": "active_basis,hf_state,k",
            "psi_micro_axis_order": "k,microscopic_basis,hf_state"
            if k_grid_shape is None
            else "mesh_1,mesh_2,microscopic_basis,hf_state",
            "n_k": int(nk),
            "microscopic_basis_dim": int(micro_dim),
            "n_active_basis": int(n_full_state),
            "n_reconstructed_states": int(len(selected)),
            "state_labels": labels,
            "kvec_provided": True,
            "selected_state_contraction": "raw_basis_rectangular_einsum_compatible_with_common_helper",
        }
    )
    if unitarity_residual is not None:
        bundle_metadata["active_eigenvectors_unitarity_residual"] = float(unitarity_residual)
    if k_grid_shape is not None:
        bundle_metadata["grid_shape"] = tuple(int(value) for value in k_grid_shape)
    if _k_grid_frac(data) is not None:
        bundle_metadata["k_grid_frac_shape"] = [int(nk), 2]
    return MicroscopicWavefunctionBundle(
        kvec=kvec_arr,
        psi_micro=psi,
        sewing_transforms=tuple(sewing_transforms),
        basis_metadata=bundle_metadata,
        source="hf_reconstructed",
    )


def reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(
    run_or_basis_data: RLGhBNHartreeFockRun | RLGhBNProjectedBasisData,
    active_eigenvectors: np.ndarray | None = None,
    *,
    include_sewing: bool = True,
    as_grid: bool = True,
    state_indices: int | Iterable[int] | None = None,
    max_dense_elements: int | None = _RLG_HBN_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS,
    state_labels: Sequence[Mapping[str, object]] | None = None,
    basis_metadata: Mapping[str, Any] | None = None,
    unitarity_atol: float | None = 1.0e-8,
    hermiticity_atol: float | None = 1.0e-8,
) -> MicroscopicWavefunctionBundle:
    """Reconstruct RnG/hBN projected-HF wavefunctions in microscopic rows.

    If ``active_eigenvectors`` is omitted and a raw
    :class:`RLGhBNHartreeFockRun`-like object is supplied, the adapter uses
    :func:`build_rlg_hbn_final_hf_eigensystem` to obtain ket coefficients from
    the converged HF Hamiltonian.  Bare basis-data objects require explicit ket
    coefficients with shape ``(active_basis, hf_state, k)``.  ``state_indices``
    can request a selected rectangular subset before dense output allocation;
    ``max_dense_elements`` guards the resulting ``psi_micro`` size.
    """

    data, state = _resolve_data_and_state(run_or_basis_data)
    if active_eigenvectors is None:
        if state is None:
            raise ValueError("active_eigenvectors are required when reconstructing from RLGhBNProjectedBasisData alone")
        orbitals = build_rlg_hbn_final_hf_eigensystem(state, hermiticity_atol=hermiticity_atol)
        coeffs = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
        eigenvector_source = "build_rlg_hbn_final_hf_eigensystem(final_hf_hamiltonian)"
        if state_labels is None:
            state_labels = tuple(
                {
                    "hf_state_index": int(index),
                    "state_source": "final_hf_eigensystem",
                    "occupied_any_k": bool(np.any(orbitals.occupied_mask[index, :])),
                    "occupied_all_k": bool(np.all(orbitals.occupied_mask[index, :])),
                }
                for index in range(int(orbitals.nt))
            )
    else:
        coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
        eigenvector_source = "caller_provided_active_eigenvectors"

    selected_coeffs, selected, selected_from_full = _prepare_active_coefficients(data, coeffs, state_indices)
    dense_elements = _validate_reconstruction_size(data, len(selected), max_dense_elements)
    metadata = _metadata(data, eigenvector_source=eigenvector_source, include_sewing=include_sewing, as_grid=as_grid)
    metadata.update(
        {
            "projected_hf_reconstruction": "explicit_dense_opt_in",
            "dense_reconstruction_estimated_output_elements": int(dense_elements),
            "max_dense_elements": None if max_dense_elements is None else int(max_dense_elements),
            "selected_hf_state_indices": [int(index) for index in selected],
            "n_reconstructed_states": int(len(selected)),
            "selected_coefficients_from_full_eigensystem": bool(selected_from_full),
        }
    )
    metadata.update(dict(basis_metadata or {}))
    grid_shape = _regular_grid_shape(data, as_grid=as_grid)
    if grid_shape is not None:
        metadata["k_grid_output_order"] = "psi_micro.reshape((mesh_size, mesh_size, micro, state), order='C')"
    sewing = ()
    if bool(include_sewing):
        sewing = rlg_hbn_projected_micro_sewing_transforms(
            local_basis_size=int(data.basis.local_basis_size),
            grid_shape=tuple(int(value) for value in data.basis.grid_shape),
            spin_count=int(data.basis.n_spin),
            valley_signs=tuple(int(value) for value in getattr(data, "valleys", (1, -1))),
        )

    n_active = int(data.basis.n_spin * data.basis.n_flavor * data.basis.n_band)
    reconstructs_full_square = selected == tuple(range(n_active)) and selected_coeffs.shape[1] == n_active
    if reconstructs_full_square:
        return reconstruct_projected_micro_wavefunctions(
            expand_rlg_hbn_projected_micro_basis(data),
            selected_coeffs,
            kvec=np.asarray(data.kvec, dtype=np.complex128),
            k_grid_frac=_k_grid_frac(data),
            grid_shape=grid_shape,
            state_labels=_selected_state_labels(
                state_labels,
                selected=selected,
                n_full_state=n_active,
                eigenvector_source=eigenvector_source,
            ),
            sewing_transforms=sewing,
            basis_metadata=metadata,
            unitarity_atol=unitarity_atol,
        )

    return _selected_bundle(
        data,
        selected_coeffs,
        k_grid_shape=grid_shape,
        state_labels=state_labels,
        selected=selected,
        eigenvector_source=eigenvector_source,
        sewing_transforms=sewing,
        metadata=metadata,
        unitarity_atol=unitarity_atol,
    )


__all__ = [
    "build_rlg_hbn_final_hf_eigensystem",
    "expand_rlg_hbn_projected_micro_basis",
    "reconstruct_rlg_hbn_projected_hf_micro_wavefunctions",
    "rlg_hbn_projected_hf_active_index",
]
