from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.htg import topology as htg_topology

def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    k_grid_frac = np.stack(np.meshgrid(np.arange(mesh) / float(mesh), np.arange(mesh) / float(mesh), indexing="ij"), axis=-1)
    for ix in range(mesh):
        kx = 2.0 * np.pi * ix / mesh
        for iy in range(mesh):
            ky = 2.0 * np.pi * iy / mesh
            dz = mass + np.cos(kx) + np.cos(ky)
            hamiltonian = np.asarray([[dz, np.sin(kx) - 1j * np.sin(ky)], [np.sin(kx) + 1j * np.sin(ky), -dz]])
            _vals, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions, k_grid_frac

def _constant_wavefunction_grid(mesh_1: int, mesh_2: int, basis_dim: int, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.zeros((mesh_1, mesh_2, basis_dim, n_states), dtype=np.complex128)
    for state in range(n_states):
        wavefunctions[:, :, state, state] = 1.0
    k_grid_frac = np.stack(np.meshgrid(np.arange(mesh_1) / float(mesh_1), np.arange(mesh_2) / float(mesh_2), indexing="ij"), axis=-1)
    return wavefunctions, k_grid_frac

def test_htg_topology_from_eigenvectors_delegates_to_common_system_adapter() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    result = htg_topology.compute_topology_from_eigenvectors(
        wavefunctions,
        0,
        valley=-1,
        k_grid_frac=k_grid_frac,
        orientation_sign=-1.0,
    )

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == -1
    assert result.is_nearly_integer
    assert result.index_metadata is not None
    assert result.index_metadata["system"] == "htg"
    assert result.index_metadata["valley"] == -1
    assert result.index_metadata["metadata"] == {"boundary_sewing": False}

def test_htg_topology_from_grid_result_maps_absolute_band_indices_to_columns() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac, band_indices=(100, 101))

    result = htg_topology.compute_topology_from_grid_result(grid, 101, valley=1)

    assert result.band_indices == (101,)
    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)
    assert result.index_metadata is not None
    metadata = result.index_metadata["metadata"]
    assert metadata["absolute_band_indices"] == [101]
    assert metadata["column_indices"] == [1]
    assert metadata["grid_result_band_indices"] == [100, 101]

def test_htg_topology_on_grid_builds_explicit_contiguous_eigenvector_window(monkeypatch) -> None:
    wavefunctions, k_grid_frac = _constant_wavefunction_grid(4, 5, 3, 3)
    calls: dict[str, object] = {}

    def fake_compute_bands_on_grid(
        mesh_size,
        lattice,
        params,
        *,
        valley,
        d_top,
        d_bot,
        band_indices,
        return_eigenvectors,
        endpoint,
        frac_shift,
    ):
        calls.update(
            {
                "mesh_size": mesh_size,
                "lattice": lattice,
                "params": params,
                "valley": valley,
                "d_top": d_top,
                "d_bot": d_bot,
                "band_indices": band_indices,
                "return_eigenvectors": return_eigenvectors,
                "endpoint": endpoint,
                "frac_shift": frac_shift,
            }
        )
        return SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac, band_indices=tuple(band_indices))

    monkeypatch.setattr(htg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)
    lattice, params = object(), object()

    result = htg_topology.compute_topology_on_grid(
        5,
        lattice,
        params,
        (10, 12),
        valley=-1,
        d_top=1.0 + 0.5j,
        d_bot=-0.25j,
        frac_shift=(0.25, 0.5),
        boundary_sewing=False,
    )

    assert result.band_indices == (10, 12)
    assert result.rounded_chern_number == 0
    assert calls == {
        "mesh_size": 5,
        "lattice": lattice,
        "params": params,
        "valley": -1,
        "d_top": 1.0 + 0.5j,
        "d_bot": -0.25j,
        "band_indices": (10, 11, 12),
        "return_eigenvectors": True,
        "endpoint": False,
        "frac_shift": (0.25, 0.5),
    }
    assert result.index_metadata is not None
    metadata = result.index_metadata["metadata"]
    assert metadata["absolute_band_indices"] == [10, 12]
    assert metadata["column_indices"] == [0, 2]

def test_htg_boundary_sewing_transform_uses_reciprocal_translation() -> None:
    lattice = SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int), n_g=2)
    sew_plus, sew_y = htg_topology.boundary_sewing_transforms(lattice)
    values = np.arange(12, dtype=float).astype(np.complex128)
    np.testing.assert_allclose(sew_plus(values), np.concatenate((values[6:12], np.zeros(6, dtype=np.complex128))))
    np.testing.assert_allclose(sew_y(values), 0.0)
    with pytest.raises(ValueError, match="Expected first axis"):
        sew_plus(np.zeros((11,), dtype=np.complex128))

def test_htg_topology_on_grid_rejects_endpoint_mesh() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        htg_topology.compute_topology_on_grid(3, object(), object(), 0, endpoint=True, boundary_sewing=False)
