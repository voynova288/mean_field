from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import canonicalize_projected_micro_basis, reconstruct_projected_micro_wavefunctions


def _rotation(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.complex128,
    )


def test_reconstruct_projected_micro_wavefunctions_identity_basis_to_topology_grid() -> None:
    n_k = 4
    micro_basis = np.repeat(np.eye(2, dtype=np.complex128)[np.newaxis, :, :], n_k, axis=0)
    active_eigenvectors = np.empty((2, 2, n_k), dtype=np.complex128)
    for ik in range(n_k):
        active_eigenvectors[:, :, ik] = _rotation(0.1 * ik)
    k_grid_frac = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )

    bundle = reconstruct_projected_micro_wavefunctions(
        micro_basis,
        active_eigenvectors,
        kvec=np.asarray([0.0, 1.0, 1.0j, 1.0 + 1.0j], dtype=np.complex128),
        k_grid_frac=k_grid_frac,
        grid_shape=(2, 2),
        state_labels=(
            {"hf_state_index": 0, "occupation": "occupied"},
            {"hf_state_index": 1, "occupation": "empty"},
        ),
        basis_metadata={"system": "toy"},
    )

    expected_flat = np.moveaxis(active_eigenvectors, 2, 0)
    assert bundle.psi_micro.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(bundle.psi_micro.reshape(n_k, 2, 2), expected_flat, atol=1.0e-14)
    assert bundle.source == "hf_reconstructed"
    assert bundle.basis_metadata["system"] == "toy"
    assert bundle.basis_metadata["psi_micro_axis_order"] == "mesh_1,mesh_2,microscopic_basis,hf_state"
    assert bundle.basis_metadata["grid_shape"] == (2, 2)
    assert bundle.basis_metadata["k_grid_frac_shape"] == [4, 2]
    assert bundle.basis_metadata["state_labels"][0]["occupation"] == "occupied"
    assert bundle.basis_metadata["active_eigenvectors_unitarity_residual"] < 1.0e-14


def test_reconstruct_projected_micro_wavefunctions_handles_noncanonical_micro_axes() -> None:
    canonical = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0j], [0.5, -0.5j]],
            [[2.0, 0.0], [0.0, -1.0]],
        ],
        dtype=np.complex128,
    )
    raw = np.transpose(canonical, axes=(1, 2, 0))
    active_eigenvectors = np.repeat(np.eye(2, dtype=np.complex128)[:, :, np.newaxis], 3, axis=2)

    np.testing.assert_allclose(
        canonicalize_projected_micro_basis(raw, k_axis=2, microscopic_basis_axis=0, active_axis=1),
        canonical,
    )
    bundle = reconstruct_projected_micro_wavefunctions(
        raw,
        active_eigenvectors,
        k_axis=2,
        microscopic_basis_axis=0,
        active_axis=1,
        basis_metadata={"axis_order_source": "unit-test"},
    )

    assert bundle.psi_micro.shape == (3, 2, 2)
    np.testing.assert_allclose(bundle.psi_micro, canonical)
    assert bundle.basis_metadata["psi_micro_axis_order"] == "k,microscopic_basis,hf_state"
    assert bundle.basis_metadata["input_micro_basis_axes"] == {
        "k_axis": 2,
        "microscopic_basis_axis": 0,
        "active_axis": 1,
    }
    assert bundle.basis_metadata["state_labels"] == ({"hf_state_index": 0}, {"hf_state_index": 1})


def test_reconstruct_projected_micro_wavefunctions_rejects_unsafe_inputs() -> None:
    micro_basis = np.repeat(np.eye(2, dtype=np.complex128)[np.newaxis, :, :], 2, axis=0)
    active_eigenvectors = np.repeat(np.eye(2, dtype=np.complex128)[:, :, np.newaxis], 2, axis=2)

    nonunitary = active_eigenvectors.copy()
    nonunitary[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="unitary"):
        reconstruct_projected_micro_wavefunctions(micro_basis, nonunitary)

    with pytest.raises(ValueError, match="grid_shape"):
        reconstruct_projected_micro_wavefunctions(micro_basis, active_eigenvectors, grid_shape=(3, 1))

    with pytest.raises(ValueError, match="state_labels length"):
        reconstruct_projected_micro_wavefunctions(micro_basis, active_eigenvectors, state_labels=({"only": 0},))

    with pytest.raises(ValueError, match="exactly three axes"):
        canonicalize_projected_micro_basis(np.zeros((2, 2), dtype=np.complex128))
