from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.tdbg import topology as tdbg_topology


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
                [
                    [dz, np.sin(kx) - 1j * np.sin(ky)],
                    [np.sin(kx) + 1j * np.sin(ky), -dz],
                ],
                dtype=np.complex128,
            )
            _vals, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions, k_grid_frac


def test_tdbg_topology_from_eigenvectors_delegates_to_common_system_adapter() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    result = tdbg_topology.compute_topology_from_eigenvectors(
        wavefunctions,
        0,
        valley=-1,
        k_grid_frac=k_grid_frac,
        metadata={"boundary_sewing": False},
        orientation_sign=-1.0,
    )

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == -1
    assert result.is_nearly_integer
    assert result.index_metadata is not None
    assert result.index_metadata["system"] == "tdbg"
    assert result.index_metadata["valley"] == -1
    assert result.index_metadata["metadata"] == {"boundary_sewing": False}


def test_tdbg_topology_from_grid_result_uses_grid_fractional_coordinates() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)

    result = tdbg_topology.compute_topology_from_grid_result(grid, 0, valley=1)

    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)


def test_tdbg_boundary_sewing_transform_uses_translation_srcmap() -> None:
    lattice = SimpleNamespace(
        q_sites=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        n_q=2,
        g_m1=1.0 + 0.0j,
        g_m2=0.0 + 1.0j,
    )

    sew_1, sew_2 = tdbg_topology.boundary_sewing_transforms(lattice)  # type: ignore[arg-type]
    values = np.arange(8, dtype=float).astype(np.complex128)

    shifted = sew_1(values)
    np.testing.assert_allclose(shifted[:4], values[4:8])
    np.testing.assert_allclose(shifted[4:8], 0.0)

    no_match = sew_2(values)
    np.testing.assert_allclose(no_match, 0.0)

    with pytest.raises(ValueError, match="Expected first axis"):
        sew_1(np.zeros((7,), dtype=np.complex128))


def test_tdbg_topology_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
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

    monkeypatch.setattr(tdbg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)

    result = tdbg_topology.compute_topology_on_grid(
        9,
        object(),
        object(),
        0,
        valley=-1,
        frac_shift=(0.25, 0.5),
        boundary_sewing=False,
    )

    assert result.rounded_chern_number == 1
    assert result.index_metadata is not None
    metadata = result.index_metadata["metadata"]
    assert metadata["boundary_sewing"] is False
    assert metadata["absolute_band_indices"] == [0]
    assert metadata["column_indices"] == [0]
    assert calls["mesh_size"] == 9
    assert calls["valley"] == -1
    assert calls["n_bands"] is None
    assert calls["return_eigenvectors"] is True
    assert calls["endpoint"] is False
    assert calls["frac_shift"] == (0.25, 0.5)


def test_tdbg_topology_on_grid_rejects_endpoint_mesh() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        tdbg_topology.compute_topology_on_grid(3, object(), object(), 0, endpoint=True, boundary_sewing=False)


def test_tdbg_topology_on_grid_rejects_n_bands_that_excludes_target_band() -> None:
    with pytest.raises(ValueError, match="does not include requested band index"):
        tdbg_topology.compute_topology_on_grid(3, object(), object(), 2, n_bands=2, boundary_sewing=False)
