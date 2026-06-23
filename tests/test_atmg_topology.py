from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.atmg import topology as atmg_topology


def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    k_grid_frac = np.stack(
        np.meshgrid(np.arange(mesh) / float(mesh), np.arange(mesh) / float(mesh), indexing="ij"),
        axis=-1,
    )
    for ix in range(mesh):
        kx = 2.0 * np.pi * ix / mesh
        for iy in range(mesh):
            ky = 2.0 * np.pi * iy / mesh
            dz = mass + np.cos(kx) + np.cos(ky)
            hamiltonian = np.asarray(
                [[dz, np.sin(kx) - 1j * np.sin(ky)], [np.sin(kx) + 1j * np.sin(ky), -dz]],
                dtype=np.complex128,
            )
            _vals, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions, k_grid_frac


def test_atmg_topology_from_eigenvectors_delegates_to_common_system_adapter() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    result = atmg_topology.compute_topology_from_eigenvectors(
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
    assert result.index_metadata["system"] == "atmg"
    assert result.index_metadata["valley"] == -1


def test_atmg_topology_from_grid_result_uses_grid_fractional_coordinates() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)

    result = atmg_topology.compute_topology_from_grid_result(grid, 0, valley=1)

    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)


def test_atmg_topology_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=9, mass=1.0)
    calls: dict[str, object] = {}

    def fake_compute_bands_on_grid(
        mesh_size,
        lattice,
        params,
        *,
        valley,
        n_bands,
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
                "n_bands": n_bands,
                "return_eigenvectors": return_eigenvectors,
                "endpoint": endpoint,
                "frac_shift": frac_shift,
            }
        )
        return SimpleNamespace(eigenvectors=wavefunctions[:, :, :, :n_bands], k_grid_frac=k_grid_frac)

    monkeypatch.setattr(atmg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)

    result = atmg_topology.compute_topology_on_grid(
        9,
        object(),
        object(),
        0,
        valley=-1,
        frac_shift=(0.25, 0.5),
    )

    assert result.rounded_chern_number == 1
    assert calls["mesh_size"] == 9
    assert calls["valley"] == -1
    assert calls["n_bands"] == 1
    assert calls["return_eigenvectors"] is True
    assert calls["endpoint"] is False
    assert calls["frac_shift"] == (0.25, 0.5)


def test_atmg_topology_on_grid_rejects_endpoint_mesh() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        atmg_topology.compute_topology_on_grid(3, object(), object(), 0, endpoint=True)


def test_atmg_topology_on_grid_rejects_n_bands_that_excludes_target_band() -> None:
    with pytest.raises(ValueError, match="does not include requested band index"):
        atmg_topology.compute_topology_on_grid(3, object(), object(), 2, n_bands=2)
