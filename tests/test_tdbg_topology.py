from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import compute_lattice_topology
from mean_field.systems.tdbg import topology as tdbg_topology


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


def test_tdbg_fhs_state_from_eigenvectors_delegates_to_common_fhs() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    state = tdbg_topology.fhs_state_from_eigenvectors(wavefunctions, 0, valley=-1, k_grid_frac=k_grid_frac, metadata={"boundary_sewing": False})
    result = compute_lattice_topology(state)

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == 1
    assert result.index_metadata["system"] == "tdbg"


def test_tdbg_fhs_state_from_grid_result_uses_grid_fractional_coordinates() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)
    result = compute_lattice_topology(tdbg_topology.fhs_state_from_grid_result(grid, 0, valley=1))
    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)


def test_tdbg_basis_sewing_uses_q_site_metadata() -> None:
    lattice = SimpleNamespace(q_sites=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float), g_m1=1.0 + 0.0j, g_m2=0.0 + 1.0j)
    spec = tdbg_topology.tdbg_basis_sewing(lattice)  # type: ignore[arg-type]
    assert spec.local_block_size == 4
    assert spec.block_labels.tolist() == [0, 0]
    assert spec.translations == ((1.0, 0.0), (0.0, 1.0))


def test_tdbg_projected_hf_fhs_state_reconstructs_reshapes_and_preserves_sewing(monkeypatch) -> None:
    psi_flat = (np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2) + 1j).astype(np.complex128)
    k_grid_flat = np.asarray([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]], dtype=float)

    def sew_1(values):
        return np.asarray(values)

    def sew_2(values):
        return np.asarray(values)

    calls: dict[str, object] = {}

    def fake_reconstruct(result, **kwargs):
        calls["result"] = result
        calls["reconstruct_kwargs"] = kwargs
        return SimpleNamespace(
            psi_micro=psi_flat,
            sewing_transforms=(sew_1, sew_2),
            basis_metadata={"topology_grid_shape": [2, 2], "topology_grid_shape_source": "unit-test metadata", "selected_hf_state_indices": [2, 5], "topology_eligible": True},
        )

    monkeypatch.setattr(tdbg_topology, "reconstruct_tdbg_projected_hf_micro_wavefunctions", fake_reconstruct)
    fake_result = SimpleNamespace(data=SimpleNamespace(k_grid_frac=k_grid_flat))
    state = tdbg_topology.fhs_state_from_projected_hf(fake_result, state_indices=(2, 5), valley=7, max_dense_elements=123, metadata={"fixture": "projected-hf-state-wiring"})

    assert calls["result"] is fake_result
    assert calls["reconstruct_kwargs"]["state_indices"] == (2, 5)
    assert calls["reconstruct_kwargs"]["band_indices"] is None
    assert calls["reconstruct_kwargs"]["max_dense_elements"] == 123
    np.testing.assert_allclose(state.wavefunctions, psi_flat.reshape(2, 2, 3, 2))
    np.testing.assert_allclose(state.k_grid_frac, k_grid_flat.reshape(2, 2, 2))
    assert state.sewing_transforms == (sew_1, sew_2)
    assert state.reported_indices == (2, 5)
    assert state.labels == ("hf_state=2", "hf_state=5")
    assert state.metadata["topology_input_axis_order"] == "mesh,mesh,basis,state"
    assert state.metadata["fixture"] == "projected-hf-state-wiring"


def test_tdbg_projected_hf_fhs_state_requires_mesh_shape_metadata(monkeypatch) -> None:
    def fake_reconstruct(_result, **_kwargs):
        return SimpleNamespace(psi_micro=np.ones((3, 2, 1), dtype=np.complex128), sewing_transforms=(), basis_metadata={"selected_hf_state_indices": [0]})

    monkeypatch.setattr(tdbg_topology, "reconstruct_tdbg_projected_hf_micro_wavefunctions", fake_reconstruct)
    with pytest.raises(ValueError, match="topology_grid_shape/grid_shape"):
        tdbg_topology.fhs_state_from_projected_hf(SimpleNamespace(data=SimpleNamespace(k_grid_frac=np.zeros((3, 2)))))


def test_tdbg_fhs_state_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=9, mass=1.0)
    calls: dict[str, object] = {}

    def fake_compute_bands_on_grid(mesh_size, lattice, params, *, valley, n_bands, return_eigenvectors, endpoint, frac_shift):
        calls.update({"mesh_size": mesh_size, "lattice": lattice, "params": params, "valley": valley, "n_bands": n_bands, "return_eigenvectors": return_eigenvectors, "endpoint": endpoint, "frac_shift": frac_shift})
        return SimpleNamespace(eigenvectors=wavefunctions[:, :, :, :n_bands], k_grid_frac=k_grid_frac)

    monkeypatch.setattr(tdbg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)
    state = tdbg_topology.fhs_state_on_grid(9, object(), object(), 0, valley=-1, frac_shift=(0.25, 0.5), boundary_sewing=False)
    result = compute_lattice_topology(state)

    assert result.rounded_chern_number == 1
    assert state.metadata["boundary_sewing"] is False
    assert calls["mesh_size"] == 9
    assert calls["valley"] == -1
    assert calls["n_bands"] is None
    assert calls["return_eigenvectors"] is True
    assert calls["endpoint"] is False
    assert calls["frac_shift"] == (0.25, 0.5)


def test_tdbg_fhs_state_on_grid_rejects_endpoint_mesh_and_missing_band_window() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        tdbg_topology.fhs_state_on_grid(3, object(), object(), 0, endpoint=True, boundary_sewing=False)
    with pytest.raises(ValueError, match="does not include requested band index"):
        tdbg_topology.fhs_state_on_grid(3, object(), object(), 2, n_bands=2, boundary_sewing=False)
