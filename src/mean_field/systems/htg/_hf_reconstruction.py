from __future__ import annotations

"""HTG-specific projected-HF microscopic reconstruction helpers.

These helpers keep HTG row-order and compact-basis conventions inside the HTG
system layer.  They do not run SCF or change the final-HF Hamiltonian; callers
provide an already-built projected basis and final active Hamiltonian.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mean_field.core.contracts import MicroscopicWavefunctionBundle, MicroscopicWavefunctionSource
from mean_field.core.hf.reconstruction import (
    active_eigenvector_unitarity_residual,
    contract_direct_sum_projected_micro_wavefunctions,
    direct_sum_active_index,
    normalize_reconstruction_state_indices,
)


def htg_direct_sum_state_index(n_spin: int, n_eta: int, n_band: int) -> np.ndarray:
    """Return HTG active-basis indices in spin/eta/band Fortran order."""

    return direct_sum_active_index((n_spin, n_eta, n_band), context="HTG active-basis index")


def htg_hf_state_labels(n_state: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {"hf_state_index": int(index), "state_source": "final_hamiltonian_eigh"}
        for index in range(int(n_state))
    )


def normalize_htg_reconstruction_state_indices(
    band_indices: int | Iterable[int] | None,
    n_state: int,
) -> tuple[int, ...]:
    """Normalize selected final-HF state indices with explicit bounds checks."""

    return normalize_reconstruction_state_indices(band_indices, n_state, label="HF band/state")


def htg_reconstruction_output_element_count(
    data: object,
    *,
    n_output_states: int,
) -> int:
    """Estimate allocated ``psi_micro`` elements for an HTG reconstruction call."""

    basis = getattr(data, "basis")
    basis_dimension = int(basis.basis_dimension)
    micro_dim = int(basis.n_spin * basis.n_flavor * basis_dimension)
    return int(getattr(data, "nk")) * micro_dim * int(n_output_states)


def validate_htg_reconstruction_output_size(
    data: object,
    *,
    n_output_states: int,
    max_dense_elements: int | None,
    context: str,
) -> int:
    """Apply the explicit HTG dense-output guard before allocating ``psi_micro``."""

    dense_elements = htg_reconstruction_output_element_count(data, n_output_states=int(n_output_states))
    if max_dense_elements is not None:
        max_elements = int(max_dense_elements)
        if max_elements < 0:
            raise ValueError("max_dense_elements must be non-negative or None")
        if dense_elements > max_elements:
            raise ValueError(
                f"{context} projected-HF dense reconstruction would exceed the explicit size guard: "
                f"estimated {dense_elements} complex output elements for {int(n_output_states)} "
                f"reconstructed state(s) > max_dense_elements={max_elements}. "
                "Increase max_dense_elements only for an intentional reconstruction call."
            )
    return dense_elements


def htg_active_hamiltonian_state_count(hamiltonian: np.ndarray, *, context: str) -> int:
    hmat = np.asarray(hamiltonian, dtype=np.complex128)
    if hmat.ndim != 3 or hmat.shape[0] != hmat.shape[1]:
        raise ValueError(f"Expected active Hamiltonian shape (n_state, n_state, n_k), got {hmat.shape}")
    if hmat.shape[0] <= 0 or hmat.shape[2] <= 0:
        raise ValueError(f"{context} active Hamiltonian must have positive n_state and n_k, got {hmat.shape}")
    return int(hmat.shape[0])


def diagonalize_htg_active_hamiltonian_field(
    hamiltonian: np.ndarray,
    *,
    context: str,
    reference_energies: np.ndarray | None = None,
    hermiticity_atol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Diagonalize an already-built HTG final active Hamiltonian with guards."""

    hmat = np.asarray(hamiltonian, dtype=np.complex128)
    n_state = htg_active_hamiltonian_state_count(hmat, context=context)
    nk = int(hmat.shape[2])
    hermitian_residual_value = float(np.max(np.abs(hmat - hmat.conjugate().swapaxes(0, 1))))
    if hermitian_residual_value > float(hermiticity_atol):
        raise ValueError(
            f"{context} final Hamiltonian is not Hermitian enough for reconstruction; "
            f"max residual {hermitian_residual_value:.6e} exceeds {float(hermiticity_atol):.6e}"
        )
    evals = np.zeros((n_state, nk), dtype=float)
    evecs = np.zeros((n_state, n_state, nk), dtype=np.complex128)
    max_eigen_residual = 0.0
    max_unitarity_residual = 0.0
    identity = np.eye(n_state, dtype=np.complex128)
    for ik in range(nk):
        evals[:, ik], evecs[:, :, ik] = np.linalg.eigh(hmat[:, :, ik])
        max_eigen_residual = max(
            max_eigen_residual,
            float(np.max(np.abs(hmat[:, :, ik] @ evecs[:, :, ik] - evecs[:, :, ik] * evals[None, :, ik]))),
        )
        max_unitarity_residual = max(
            max_unitarity_residual,
            float(np.max(np.abs(evecs[:, :, ik].conjugate().T @ evecs[:, :, ik] - identity))),
        )
    diagnostics = {
        "hamiltonian_hermiticity_residual": hermitian_residual_value,
        "active_eigensystem_residual": float(max_eigen_residual),
        "active_eigenvector_unitarity_residual": float(max_unitarity_residual),
    }
    if reference_energies is not None:
        ref = np.asarray(reference_energies, dtype=float)
        if ref.shape == evals.shape:
            diagnostics["stored_energy_eigh_residual"] = float(np.max(np.abs(ref - evals)))
    return evals, evecs, diagnostics


def _grid_shape(shape: tuple[int, int] | None, n_k: int) -> tuple[int, int] | None:
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError(f"grid_shape must have length two, got {shape}")
    out = (int(shape[0]), int(shape[1]))
    if out[0] <= 0 or out[1] <= 0 or out[0] * out[1] != int(n_k):
        raise ValueError(f"grid_shape={shape} is incompatible with n_k={n_k}")
    return out


def _state_labels(
    labels: Sequence[Mapping[str, object]] | None,
    n_state: int,
) -> tuple[dict[str, object], ...]:
    if labels is None:
        return htg_hf_state_labels(n_state)
    out = tuple(dict(label) for label in labels)
    if len(out) != int(n_state):
        raise ValueError(f"state_labels length {len(out)} must match reconstructed state count={n_state}")
    return out


def reconstruct_htg_projected_micro_bundle(
    raw_projected_basis: np.ndarray,
    active_eigenvectors: np.ndarray,
    *,
    n_spin: int,
    selected_state_indices: Sequence[int],
    kvec: np.ndarray | None = None,
    k_grid_frac: np.ndarray | None = None,
    grid_shape: tuple[int, int] | None = None,
    state_labels: Sequence[Mapping[str, object]] | None = None,
    sewing_transforms: Sequence[Callable[..., Any]] = (),
    basis_metadata: Mapping[str, Any] | None = None,
    source: MicroscopicWavefunctionSource = "hf_reconstructed",
    unitarity_atol: float | None = 1.0e-8,
    context: str = "HTG",
) -> MicroscopicWavefunctionBundle:
    """Contract compact HTG projected basis into selected final-HF states.

    Unlike the square common helper, this HTG adapter accepts a selected set of
    final-HF columns and allocates only the requested output state axis.  The
    compact basis row order is ``(basis, band, flavor, k)`` and the microscopic
    output row order is ``spin, eta, basis_flat_F(...)``.
    """

    raw = np.asarray(raw_projected_basis, dtype=np.complex128)
    if raw.ndim != 4:
        raise ValueError(f"{context} projected basis must have shape (basis, band, flavor, k), got {raw.shape}")
    basis_dimension, n_band, n_eta, nk = (int(value) for value in raw.shape)
    n_spin_i = int(n_spin)
    state_index = htg_direct_sum_state_index(n_spin_i, n_eta, n_band)
    n_active = int(state_index.size)
    coeffs = np.asarray(active_eigenvectors, dtype=np.complex128)
    if coeffs.shape != (n_active, n_active, nk):
        raise ValueError(
            f"{context} active_eigenvectors must have shape ({n_active}, {n_active}, {nk}), "
            f"got {coeffs.shape}"
        )
    selected = normalize_htg_reconstruction_state_indices(tuple(int(index) for index in selected_state_indices), n_active)
    n_selected = int(len(selected))
    selected_coeffs = coeffs[:, list(selected), :]
    residual = active_eigenvector_unitarity_residual(
        selected_coeffs,
        unitarity_atol=unitarity_atol,
        context=f"{context} selected active_eigenvectors",
    )

    kvec_arr = np.arange(nk, dtype=np.complex128) if kvec is None else np.asarray(kvec, dtype=np.complex128).reshape(-1)
    if kvec_arr.shape != (nk,):
        raise ValueError(f"kvec must have shape ({nk},), got {kvec_arr.shape}")
    if k_grid_frac is not None and np.asarray(k_grid_frac, dtype=float).shape != (nk, 2):
        raise ValueError(f"k_grid_frac must have shape ({nk}, 2), got {np.asarray(k_grid_frac).shape}")
    shape = _grid_shape(grid_shape, nk)

    micro_dim = int(n_spin_i * n_eta * basis_dimension)
    psi_flat = contract_direct_sum_projected_micro_wavefunctions(raw, selected_coeffs, state_index)

    psi = psi_flat if shape is None else psi_flat.reshape((*shape, micro_dim, n_selected), order="C")
    labels = _state_labels(state_labels, n_selected)
    metadata = dict(basis_metadata or {})
    metadata.update(
        {
            "micro_basis_axis_order": "k,microscopic_basis,active_basis",
            "input_micro_basis_axes": {
                "raw_projected_basis_axis_order": "basis,band,flavor,k",
                "spin_axis": "direct_sum_repeated_projected_basis",
            },
            "active_eigenvectors_axis_order": "active_basis,hf_state,k",
            "psi_micro_axis_order": "k,microscopic_basis,hf_state"
            if shape is None
            else "mesh_1,mesh_2,microscopic_basis,hf_state",
            "n_k": nk,
            "microscopic_basis_dim": micro_dim,
            "n_active": n_active,
            "n_reconstructed_hf_states": n_selected,
            "state_labels": labels,
            "kvec_provided": kvec is not None,
            "selected_state_allocation": "output_axis_contains_only_selected_hf_states",
        }
    )
    if residual is not None:
        metadata["active_eigenvectors_unitarity_residual"] = residual
    if shape is not None:
        metadata["grid_shape"] = shape
    if k_grid_frac is not None:
        metadata["k_grid_frac_shape"] = [nk, 2]

    return MicroscopicWavefunctionBundle(
        kvec=kvec_arr,
        psi_micro=psi,
        sewing_transforms=tuple(sewing_transforms),
        basis_metadata=metadata,
        source=source,
    )


__all__ = [
    "diagonalize_htg_active_hamiltonian_field",
    "htg_active_hamiltonian_state_count",
    "htg_direct_sum_state_index",
    "htg_hf_state_labels",
    "htg_reconstruction_output_element_count",
    "normalize_htg_reconstruction_state_indices",
    "reconstruct_htg_projected_micro_bundle",
    "validate_htg_reconstruction_output_size",
]
