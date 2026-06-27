from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import compute_lattice_topology
from mean_field.systems.tmbg import topology as tmbg_topology


def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    k_grid_frac = np.stack(np.meshgrid(np.arange(mesh) / float(mesh), np.arange(mesh) / float(mesh), indexing="ij"), axis=-1)
    for ix in range(mesh):
        kx = 2.0 * np.pi * ix / mesh
        for iy in range(mesh):
            ky = 2.0 * np.pi * iy / mesh
            dz = mass + np.cos(kx) + np.cos(ky)
            hamiltonian = np.asarray([[dz, np.sin(kx) - 1j * np.sin(ky)], [np.sin(kx) + 1j * np.sin(ky), -dz]], dtype=np.complex128)
            _vals, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions, k_grid_frac


def test_tmbg_fhs_state_from_eigenvectors_delegates_to_common_fhs() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    state = tmbg_topology.fhs_state_from_eigenvectors(wavefunctions, 0, valley=-1, k_grid_frac=k_grid_frac, orientation_sign=-1.0)
    result = compute_lattice_topology(state)

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == -1
    assert result.is_nearly_integer
    assert result.index_metadata["system"] == "tmbg"


def test_tmbg_fhs_state_from_grid_result_requires_eigenvectors() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)
    result = compute_lattice_topology(tmbg_topology.fhs_state_from_grid_result(grid, 0, valley=1))
    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)


def test_tmbg_fhs_state_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=9, mass=1.0)
    calls: dict[str, object] = {}

    def fake_compute_bands_on_grid(mesh_size, lattice, params, *, valley, n_bands, return_eigenvectors, endpoint, frac_shift):
        calls.update({"mesh_size": mesh_size, "lattice": lattice, "params": params, "valley": valley, "n_bands": n_bands, "return_eigenvectors": return_eigenvectors, "endpoint": endpoint, "frac_shift": frac_shift})
        return SimpleNamespace(eigenvectors=wavefunctions[:, :, :, :n_bands], k_grid_frac=k_grid_frac)

    monkeypatch.setattr(tmbg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)
    state = tmbg_topology.fhs_state_on_grid(9, object(), object(), 0, valley=-1, frac_shift=(0.25, 0.5), use_boundary_sewing=False)
    result = compute_lattice_topology(state)

    assert result.rounded_chern_number == 1
    assert state.metadata["boundary_sewing"] is False
    assert calls["mesh_size"] == 9
    assert calls["valley"] == -1
    assert calls["n_bands"] is None
    assert calls["return_eigenvectors"] is True
    assert calls["endpoint"] is False
    assert calls["frac_shift"] == (0.25, 0.5)


def test_tmbg_fhs_state_on_grid_rejects_endpoint_mesh_and_missing_band_window() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        tmbg_topology.fhs_state_on_grid(3, object(), object(), 0, endpoint=True)
    with pytest.raises(ValueError, match="does not include requested band index"):
        tmbg_topology.fhs_state_on_grid(3, object(), object(), 2, n_bands=2)


def test_polshyn_projected_hf_fhs_state_reshapes_flat_k_order(monkeypatch) -> None:
    mesh_shape = (2, 3)
    nk = mesh_shape[0] * mesh_shape[1]
    micro_dim = 2 * 2 * 6
    psi_flat = np.zeros((nk, micro_dim, 2), dtype=np.complex128)
    for ik in range(nk):
        psi_flat[ik, :, :] = ik + 0.01 * np.arange(micro_dim)[:, None] + 0.001 * np.arange(2)[None, :]
    k_grid_frac = np.asarray([(ix / mesh_shape[0], iy / mesh_shape[1]) for iy in range(mesh_shape[1]) for ix in range(mesh_shape[0])], dtype=float)
    basis = SimpleNamespace(local_basis_size=6, embedding_shape=(1, 1), basis_dimension=6, n_spin=2, n_eta=2, k_grid_frac=k_grid_frac)
    captured: dict[str, object] = {}

    def fake_reconstruct_polshyn_wang_hf_micro_wavefunctions(*args, **kwargs):
        captured["reconstruct_kwargs"] = dict(kwargs)
        return SimpleNamespace(psi_micro=psi_flat, basis_metadata={"selected_hf_state_indices": [1, 6], "topology_eligible": False})

    monkeypatch.setattr(tmbg_topology, "reconstruct_polshyn_wang_hf_micro_wavefunctions", fake_reconstruct_polshyn_wang_hf_micro_wavefunctions)
    state = tmbg_topology.fhs_state_from_polshyn_projected_hf(basis, active_eigenvectors=np.zeros((8, 8, nk), dtype=np.complex128), state_indices=(1, 6))

    assert isinstance(state, tmbg_topology.FHSState)
    assert captured["reconstruct_kwargs"]["include_sewing"] is False
    assert state.wavefunctions.shape == mesh_shape + (micro_dim, 2)
    np.testing.assert_allclose(state.wavefunctions[0, 0], psi_flat[0])
    np.testing.assert_allclose(state.wavefunctions[1, 0], psi_flat[1])
    np.testing.assert_allclose(state.wavefunctions[0, 1], psi_flat[2])
    np.testing.assert_allclose(state.k_grid_frac, k_grid_frac.reshape(mesh_shape + (2,), order="F"))
    assert state.reported_indices == (1, 6)
    assert state.metadata["topology_adapter"] == "mean_field.systems.tmbg.topology.fhs_state_from_polshyn_projected_hf"


def test_polshyn_projected_hf_fhs_state_rejects_missing_mesh_metadata(monkeypatch) -> None:
    psi_flat = np.ones((4, 2 * 2 * 6, 1), dtype=np.complex128)

    def fake_reconstruct(*_args, **_kwargs):
        return SimpleNamespace(psi_micro=psi_flat, basis_metadata={"selected_hf_state_indices": [0]})

    monkeypatch.setattr(tmbg_topology, "reconstruct_polshyn_wang_hf_micro_wavefunctions", fake_reconstruct)
    basis = SimpleNamespace(local_basis_size=6, embedding_shape=(1, 1), basis_dimension=6, n_spin=2, n_eta=2, k_grid_frac=None)
    with pytest.raises(ValueError, match="mesh_shape or basis.k_grid_frac"):
        tmbg_topology.fhs_state_from_polshyn_projected_hf(basis, active_eigenvectors=np.zeros((8, 1, 4), dtype=np.complex128), state_indices=0)
