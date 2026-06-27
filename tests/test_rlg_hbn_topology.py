from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import compute_lattice_topology, sewing_transforms_from_block_spec
from mean_field.systems.RnG_hBN import topology as rlg_topology


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


def test_rlg_hbn_fhs_state_from_eigenvectors_delegates_with_orientation_metadata() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    state = rlg_topology.fhs_state_from_eigenvectors(wavefunctions, 0, valley=-1, k_grid_frac=k_grid_frac, paper_orientation=True)
    result = compute_lattice_topology(state)

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == -1
    assert result.index_metadata["system"] == "RLG_hBN"
    assert result.index_metadata["metadata"] == {"boundary_sewing": False, "orientation_sign": -1.0}


def test_rlg_hbn_fhs_state_from_grid_result_uses_optional_boundary_metadata() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)
    state = rlg_topology.fhs_state_from_grid_result(grid, 0, valley=1, use_boundary_sewing=False)
    result = compute_lattice_topology(state)

    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)
    metadata = result.index_metadata["metadata"]
    assert metadata["boundary_sewing"] is False
    assert metadata["orientation_sign"] == 1.0
    assert metadata["column_indices"] == [0]


def test_rlg_hbn_generic_basis_sewing_relabels_plane_wave_g_blocks() -> None:
    lattice = SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int), n_g=2)
    params = SimpleNamespace(layer_count=1)
    sew_plus, _ = sewing_transforms_from_block_spec(rlg_topology.rlg_hbn_basis_sewing(lattice, params, valley=1))
    values = np.arange(4, dtype=np.complex128)
    np.testing.assert_allclose(sew_plus(values), np.asarray([2, 3, 0, 0], dtype=np.complex128))

    sew_minus, _ = sewing_transforms_from_block_spec(rlg_topology.rlg_hbn_basis_sewing(lattice, params, valley=-1))
    np.testing.assert_allclose(sew_minus(values), np.asarray([0, 0, 0, 1], dtype=np.complex128))
    with pytest.raises(ValueError, match="Expected first axis"):
        sew_plus(np.zeros((3,), dtype=np.complex128))
    with pytest.raises(ValueError, match="Expected valley"):
        rlg_topology.rlg_hbn_basis_sewing(lattice, params, valley=0)


def test_rlg_hbn_fhs_state_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=9, mass=1.0)
    calls: dict[str, object] = {}

    def fake_compute_bands_on_grid(mesh_size, lattice, params, *, valley, n_bands, return_eigenvectors, endpoint, frac_shift):
        calls.update({"mesh_size": mesh_size, "valley": valley, "n_bands": n_bands, "return_eigenvectors": return_eigenvectors, "endpoint": endpoint, "frac_shift": frac_shift})
        return SimpleNamespace(eigenvectors=wavefunctions[:, :, :, :n_bands], k_grid_frac=k_grid_frac)

    monkeypatch.setattr(rlg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)
    state = rlg_topology.fhs_state_on_grid(9, SimpleNamespace(g_indices=np.asarray([[0, 0]], dtype=int), n_g=1), SimpleNamespace(layer_count=1), 0, valley=-1, frac_shift=(0.25, 0.5), use_boundary_sewing=False)
    result = compute_lattice_topology(state)

    assert result.rounded_chern_number == 1
    metadata = result.index_metadata["metadata"]
    assert metadata["boundary_sewing"] is False
    assert metadata["orientation_sign"] == 1.0
    assert metadata["column_indices"] == [0]
    assert calls == {"mesh_size": 9, "valley": -1, "n_bands": None, "return_eigenvectors": True, "endpoint": False, "frac_shift": (0.25, 0.5)}


def test_rlg_hbn_fhs_state_on_grid_rejects_endpoint_mesh_and_missing_band_window() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        rlg_topology.fhs_state_on_grid(3, object(), object(), 0, endpoint=True, use_boundary_sewing=False)
    with pytest.raises(ValueError, match="does not include requested band index"):
        rlg_topology.fhs_state_on_grid(3, object(), object(), 2, n_bands=2, use_boundary_sewing=False)
