from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.tmbg import topology as tmbg_topology


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


def test_tmbg_topology_from_eigenvectors_delegates_to_common_system_adapter() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    result = tmbg_topology.compute_topology_from_eigenvectors(
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
    assert result.index_metadata["system"] == "tmbg"
    assert result.index_metadata["valley"] == -1


def test_tmbg_topology_from_grid_result_requires_eigenvectors() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)

    result = tmbg_topology.compute_topology_from_grid_result(grid, 0, valley=1)

    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)


def test_tmbg_topology_on_grid_builds_single_explicit_eigenvector_grid(monkeypatch) -> None:
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

    monkeypatch.setattr(tmbg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)

    result = tmbg_topology.compute_topology_on_grid(
        9,
        object(),
        object(),
        0,
        valley=-1,
        frac_shift=(0.25, 0.5),
        boundary_sewing=False,
    )

    assert result.rounded_chern_number == 1
    assert calls["mesh_size"] == 9
    assert calls["valley"] == -1
    assert calls["n_bands"] is None
    assert calls["return_eigenvectors"] is True
    assert calls["endpoint"] is False
    assert calls["frac_shift"] == (0.25, 0.5)


def test_tmbg_boundary_sewing_transform_uses_reciprocal_translation() -> None:
    lattice = SimpleNamespace(g_indices=np.asarray([[0, 0], [1, 0]], dtype=int), n_g=2)
    sew_plus, sew_y = tmbg_topology.boundary_sewing_transforms(lattice)
    values = np.arange(12, dtype=float).astype(np.complex128)
    np.testing.assert_allclose(sew_plus(values), np.concatenate((values[6:12], np.zeros(6, dtype=np.complex128))))
    np.testing.assert_allclose(sew_y(values), 0.0)
    with pytest.raises(ValueError, match="Expected first axis"):
        sew_plus(np.zeros((11,), dtype=np.complex128))


def _manual_polshyn_row_shift(values: np.ndarray, *, shift: tuple[int, int]) -> np.ndarray:
    local = 6
    embed_x = 2
    embed_y = 3
    n_spin = 2
    n_valley = 2
    basis_dim = local * embed_x * embed_y
    micro_dim = n_spin * n_valley * basis_dim
    array = np.asarray(values, dtype=np.complex128)
    frames = int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1
    matrix = array.reshape((micro_dim, frames), order="C")
    expected = np.zeros_like(matrix)
    for ispin in range(n_spin):
        for ivalley in range(n_valley):
            start = (ispin * n_valley + ivalley) * basis_dim
            block = matrix[start : start + basis_dim, :].reshape((local, embed_x, embed_y, frames), order="F")
            shifted = np.zeros_like(block)
            for target_x in range(embed_x):
                source_x = target_x + int(shift[0])
                if source_x < 0 or source_x >= embed_x:
                    continue
                for target_y in range(embed_y):
                    source_y = target_y + int(shift[1])
                    if source_y < 0 or source_y >= embed_y:
                        continue
                    shifted[:, target_x, target_y, :] = block[:, source_x, source_y, :]
            expected[start : start + basis_dim, :] = shifted.reshape((basis_dim, frames), order="F")
    return expected.reshape(array.shape, order="C")

def test_polshyn_projected_hf_boundary_sewing_transforms_shift_rows_over_spin_valley_basis() -> None:
    basis = SimpleNamespace(
        local_basis_size=6,
        embedding_shape=(2, 3),
        basis_dimension=6 * 2 * 3,
        n_spin=2,
        n_eta=2,
    )
    sew_b1, sew_b2 = tmbg_topology.polshyn_projected_hf_boundary_sewing_transforms(basis)
    micro_dim = 2 * 2 * 6 * 2 * 3
    labelled_rows = np.arange(micro_dim * 3, dtype=float).reshape((micro_dim, 3), order="C").astype(np.complex128)

    np.testing.assert_allclose(sew_b1(labelled_rows), _manual_polshyn_row_shift(labelled_rows, shift=(1, 0)))
    np.testing.assert_allclose(sew_b2(labelled_rows), _manual_polshyn_row_shift(labelled_rows, shift=(0, 1)))
    np.testing.assert_allclose(sew_b1(labelled_rows[:, 0]), _manual_polshyn_row_shift(labelled_rows[:, 0], shift=(1, 0)))
    with pytest.raises(ValueError, match="Expected first axis"):
        sew_b1(np.zeros((micro_dim - 1, 3), dtype=np.complex128))

def test_polshyn_projected_hf_topology_adapter_reshapes_flat_k_order_and_wires_common_api(monkeypatch) -> None:
    mesh_shape = (2, 3)
    nk = mesh_shape[0] * mesh_shape[1]
    micro_dim = 2 * 2 * 6
    psi_flat = np.zeros((nk, micro_dim, 2), dtype=np.complex128)
    for ik in range(nk):
        psi_flat[ik, :, :] = ik + 0.01 * np.arange(micro_dim)[:, None] + 0.001 * np.arange(2)[None, :]
    k_grid_frac = np.asarray(
        [(ix / mesh_shape[0], iy / mesh_shape[1]) for iy in range(mesh_shape[1]) for ix in range(mesh_shape[0])],
        dtype=float,
    )
    basis = SimpleNamespace(
        local_basis_size=6,
        embedding_shape=(1, 1),
        basis_dimension=6,
        n_spin=2,
        n_eta=2,
        k_grid_frac=k_grid_frac,
    )
    captured: dict[str, object] = {}

    def fake_reconstruct_polshyn_wang_hf_micro_wavefunctions(*args, **kwargs):
        captured["reconstruct_kwargs"] = dict(kwargs)
        return SimpleNamespace(
            psi_micro=psi_flat,
            basis_metadata={
                "selected_hf_state_indices": [1, 6],
                "topology_eligible": False,
                "topology_ineligible_reason": "flat diagnostic remains ineligible",
            },
        )

    def fake_compute_system_topology_from_eigenvectors(eigenvectors, band_indices, **kwargs):
        captured["eigenvectors"] = np.asarray(eigenvectors)
        captured["band_indices"] = tuple(band_indices)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(marker="common-topology-called")

    monkeypatch.setattr(
        tmbg_topology,
        "reconstruct_polshyn_wang_hf_micro_wavefunctions",
        fake_reconstruct_polshyn_wang_hf_micro_wavefunctions,
    )
    monkeypatch.setattr(tmbg_topology, "compute_system_topology_from_eigenvectors", fake_compute_system_topology_from_eigenvectors)

    result = tmbg_topology.compute_polshyn_projected_hf_topology(
        basis,
        active_eigenvectors=np.zeros((8, 8, nk), dtype=np.complex128),
        state_indices=(1, 6),
        metadata={"caller": "unit-test", "topology_eligible": False, "physical_validation_status": "caller-claimed-physical"},
    )

    assert result.marker == "common-topology-called"
    assert captured["reconstruct_kwargs"]["include_sewing"] is False
    assert captured["reconstruct_kwargs"]["state_indices"] == (1, 6)
    psi_grid = captured["eigenvectors"]
    assert psi_grid.shape == mesh_shape + (micro_dim, 2)
    np.testing.assert_allclose(psi_grid[0, 0], psi_flat[0])
    np.testing.assert_allclose(psi_grid[1, 0], psi_flat[1])
    np.testing.assert_allclose(psi_grid[0, 1], psi_flat[2])
    np.testing.assert_allclose(psi_grid[1, 2], psi_flat[5])
    assert captured["band_indices"] == (0, 1)

    kwargs = captured["kwargs"]
    np.testing.assert_allclose(kwargs["k_grid_frac"], k_grid_frac.reshape(mesh_shape + (2,), order="F"))
    assert kwargs["result_band_indices"] == (1, 6)
    assert kwargs["role"] == "hf_state"
    assert kwargs["labels"] == ("hf_state=1", "hf_state=6")
    assert len(kwargs["sewing_transforms"]) == 2
    metadata = kwargs["index_metadata"]
    assert metadata["topology_eligible"] is True
    assert metadata["flat_diagnostic_bundle_topology_eligible"] is False
    assert metadata["flat_diagnostic_topology_ineligible_reason"] == "flat diagnostic remains ineligible"
    assert metadata["topology_grid_shape"] == [2, 3]
    assert "order='F'" in metadata["topology_flat_grid_order"]
    assert metadata["topology_sewing_axes"] == ["B1", "B2"]
    assert metadata["topology_eligible"] is True
    assert metadata["physical_validation_status"] == "software_api_only_pending_slurm_paper_validation"
    assert metadata["absolute_band_indices"] == [1, 6]
    assert metadata["column_indices"] == [0, 1]
    assert metadata["caller_metadata"] == {"caller": "unit-test", "topology_eligible": False, "physical_validation_status": "caller-claimed-physical"}
    assert "src/mean_field/systems/tmbg/topology.py" in metadata["evidence_paths"]

def test_tmbg_topology_on_grid_rejects_endpoint_mesh() -> None:
    with pytest.raises(ValueError, match="endpoint=False"):
        tmbg_topology.compute_topology_on_grid(3, object(), object(), 0, endpoint=True)


def test_tmbg_topology_on_grid_rejects_n_bands_that_excludes_target_band() -> None:
    with pytest.raises(ValueError, match="does not include requested band index"):
        tmbg_topology.compute_topology_on_grid(3, object(), object(), 2, n_bands=2)


def test_tmbg_topology_accepts_explicit_common_sewing_transforms() -> None:
    wavefunctions, k_grid_frac = _qiwuzhang_wavefunctions(mesh=9, mass=1.0)
    result = tmbg_topology.compute_topology_from_eigenvectors(
        wavefunctions,
        0,
        k_grid_frac=k_grid_frac,
        sewing_transforms=(None, None),
    )
    assert result.index_metadata is not None
    assert result.index_metadata["metadata"]["boundary_sewing"] is True

def test_polshyn_projected_hf_topology_rejects_no_sewing_unless_diagnostic(monkeypatch) -> None:
    mesh_shape = (2, 2)
    nk = 4
    micro_dim = 2 * 2 * 6
    psi_flat = np.ones((nk, micro_dim, 1), dtype=np.complex128)
    k_grid_frac = np.asarray([(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)], dtype=float)
    basis = SimpleNamespace(
        local_basis_size=6,
        embedding_shape=(1, 1),
        basis_dimension=6,
        n_spin=2,
        n_eta=2,
        k_grid_frac=k_grid_frac,
    )

    def fake_reconstruct(*_args, **_kwargs):
        return SimpleNamespace(
            psi_micro=psi_flat,
            basis_metadata={"selected_hf_state_indices": [0], "topology_eligible": False},
        )

    captured: dict[str, object] = {}

    def fake_common(_eigenvectors, _band_indices, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(marker="diagnostic-topology-called")

    monkeypatch.setattr(tmbg_topology, "reconstruct_polshyn_wang_hf_micro_wavefunctions", fake_reconstruct)
    monkeypatch.setattr(tmbg_topology, "compute_system_topology_from_eigenvectors", fake_common)

    with pytest.raises(ValueError, match="requires doubled-cell B1/B2 boundary sewing"):
        tmbg_topology.compute_polshyn_projected_hf_topology(
            basis,
            active_eigenvectors=np.zeros((8, 1, nk), dtype=np.complex128),
            state_indices=0,
            boundary_sewing=False,
        )
    with pytest.raises(ValueError, match="requires doubled-cell B1/B2 boundary sewing"):
        tmbg_topology.compute_polshyn_projected_hf_topology(
            basis,
            active_eigenvectors=np.zeros((8, 1, nk), dtype=np.complex128),
            state_indices=0,
            sewing_transforms=(lambda values: values, None),
        )

    result = tmbg_topology.compute_polshyn_projected_hf_topology(
        basis,
        active_eigenvectors=np.zeros((8, 1, nk), dtype=np.complex128),
        state_indices=0,
        boundary_sewing=False,
        diagnostic_no_sewing=True,
    )
    assert result.marker == "diagnostic-topology-called"
    metadata = captured["kwargs"]["index_metadata"]
    assert metadata["topology_eligible"] is False
    assert metadata["topology_status"] == "diagnostic-no-sewing-not-physical"
    assert metadata["physical_validation_status"] == "not_physical_no_sewing_diagnostic"
    assert metadata["topology_sewing_transforms_count"] == 0


def test_polshyn_projected_hf_topology_rejects_unvalidated_mesh_order_and_no_sqrt_fallback(monkeypatch) -> None:
    psi_flat = np.ones((4, 2 * 2 * 6, 1), dtype=np.complex128)

    def fake_reconstruct(*_args, **_kwargs):
        return SimpleNamespace(psi_micro=psi_flat, basis_metadata={"selected_hf_state_indices": [0]})

    monkeypatch.setattr(tmbg_topology, "reconstruct_polshyn_wang_hf_micro_wavefunctions", fake_reconstruct)

    bad_order_basis = SimpleNamespace(
        local_basis_size=6,
        embedding_shape=(1, 1),
        basis_dimension=6,
        n_spin=2,
        n_eta=2,
        # ix/f1 outer, iy/f2 inner: wrong for Polshyn flat storage.
        k_grid_frac=np.asarray([(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)], dtype=float),
    )
    with pytest.raises(ValueError, match="iy/f2 outer and ix/f1 inner"):
        tmbg_topology.compute_polshyn_projected_hf_topology(
            bad_order_basis,
            active_eigenvectors=np.zeros((8, 1, 4), dtype=np.complex128),
            state_indices=0,
        )

    no_grid_basis = SimpleNamespace(
        local_basis_size=6,
        embedding_shape=(1, 1),
        basis_dimension=6,
        n_spin=2,
        n_eta=2,
        k_grid_frac=None,
    )
    with pytest.raises(ValueError, match="Refusing to infer a topology torus from sqrt"):
        tmbg_topology.compute_polshyn_projected_hf_topology(
            no_grid_basis,
            active_eigenvectors=np.zeros((8, 1, 4), dtype=np.complex128),
            state_indices=0,
        )
