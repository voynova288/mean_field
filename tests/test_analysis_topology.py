from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import (
    WavefunctionIndex,
    WavefunctionLayout,
    assert_topology_eligible,
    canonicalize_wavefunction_grid,
    compute_lattice_topology,
    compute_quantum_geometry,
    compute_lattice_topology_for_state_groups,
    compute_system_topology_from_bundle,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    fubini_study_trace,
    normalize_quantum_geometry_maps,
    reshape_flat_mesh_to_grid,
    split_state_indices_by_direct_gaps,
    wavefunction_index_from_state_labels,
)


def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    energies = np.empty((2, mesh, mesh), dtype=float)
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
            vals, vecs = np.linalg.eigh(hamiltonian)
            energies[:, ix, iy] = vals
            wavefunctions[ix, iy] = vecs
    return wavefunctions, energies


def test_fhs_chern_for_qiwuzhang_single_bands_and_subspace() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=21, mass=1.0)

    lower = compute_lattice_topology(
        wavefunctions,
        0,
        index=WavefunctionIndex(indices=(0,), role="band", labels=("lower",), system="qiwuzhang"),
    )
    upper = compute_lattice_topology(wavefunctions, 1)
    full_subspace = compute_lattice_topology(wavefunctions, (0, 1), link_method="determinant")

    assert lower.rounded_chern_number == 1
    assert lower.is_nearly_integer
    assert upper.rounded_chern_number == -1
    assert np.isclose(lower.chern_number + upper.chern_number, 0.0, atol=1.0e-12)
    assert full_subspace.rounded_chern_number == 0
    assert full_subspace.is_nearly_integer
    assert lower.min_link_magnitude > 0.9


def test_fhs_chern_distinguishes_trivial_and_topological_mass_regions() -> None:
    topological, _ = _qiwuzhang_wavefunctions(mesh=21, mass=-1.0)
    trivial, _ = _qiwuzhang_wavefunctions(mesh=21, mass=3.0)

    assert compute_lattice_topology(topological, 0).rounded_chern_number == -1
    assert compute_lattice_topology(trivial, 0).rounded_chern_number == 0


def test_wavefunction_layout_helpers_flatten_state_axes_with_labels() -> None:
    raw = np.arange(2 * 3 * 5 * 2 * 2, dtype=float).reshape((2, 3, 5, 2, 2))
    layout = WavefunctionLayout(
        basis_axis=2,
        state_axes=(3, 4),
        state_axis_names=("band", "flavor"),
        state_axis_labels={"band": ("v", "c"), "flavor": ("K", "Kprime")},
    )

    canonical = canonicalize_wavefunction_grid(raw, layout)
    assert canonical.wavefunctions.shape == (2, 3, 5, 4)
    assert canonical.state_labels == (
        {"band": "v", "flavor": "K"},
        {"band": "v", "flavor": "Kprime"},
        {"band": "c", "flavor": "K"},
        {"band": "c", "flavor": "Kprime"},
    )

    index = canonical.index_for((0, 3), role="band_flavor", system="toy", valley=1)
    assert index.labels == ("band=v/flavor=K", "band=c/flavor=Kprime")
    assert index.metadata["selected_state_labels"] == [
        {"band": "v", "flavor": "K"},
        {"band": "c", "flavor": "Kprime"},
    ]


def test_flat_mesh_and_state_label_helpers() -> None:
    values = np.arange(2 * 6 * 3).reshape((2, 6, 3))
    grid = reshape_flat_mesh_to_grid(values, (2, 3), k_axis=1)
    assert grid.shape == (2, 3, 2, 3)
    np.testing.assert_array_equal(grid.reshape(6, 2, 3), np.moveaxis(values, 1, 0))

    index = wavefunction_index_from_state_labels(
        1,
        ({"band": "v"}, {"band": "c"}),
        role="band",
        system="toy",
        metadata={"source": "unit-test"},
    )
    assert index.indices == (1,)
    assert index.labels == ("band=c",)
    assert index.metadata["source"] == "unit-test"


def test_gap_grouping_and_group_topology_api() -> None:
    wavefunctions, energies = _qiwuzhang_wavefunctions(mesh=15, mass=1.0)
    groups = split_state_indices_by_direct_gaps(energies, (0, 1), min_gap=0.5)

    assert groups == ((0,), (1,))

    results = compute_lattice_topology_for_state_groups(
        wavefunctions,
        groups,
        base_index=WavefunctionIndex(indices=(0, 1), labels=("lower", "upper"), system="qiwuzhang"),
    )

    assert tuple(result.rounded_chern_number for result in results) == (1, -1)
    assert all(result.is_nearly_integer for result in results)


def test_lattice_topology_orientation_sign_flips_links_connection_and_curvature() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    positive = compute_lattice_topology(wavefunctions, 0)
    negative = compute_lattice_topology(wavefunctions, 0, orientation_sign=-1.0)

    np.testing.assert_allclose(negative.link_1, positive.link_1.conjugate())
    np.testing.assert_allclose(negative.link_2, positive.link_2.conjugate())
    np.testing.assert_allclose(negative.berry_connection, -positive.berry_connection)
    np.testing.assert_allclose(negative.berry_curvature, -positive.berry_curvature)
    assert negative.rounded_chern_number == -positive.rounded_chern_number

    with pytest.raises(ValueError, match="orientation_sign"):
        compute_lattice_topology(wavefunctions, 0, orientation_sign=0.5)


def test_system_topology_adapter_attaches_metadata_and_orientation() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    result = compute_system_topology_from_eigenvectors(
        wavefunctions,
        0,
        system="qiwuzhang",
        valley=-1,
        labels=("lower",),
        index_metadata={"mesh_source": "unit_test"},
        orientation_sign=-1.0,
    )

    assert result.band_indices == (0,)
    assert result.valley == -1
    assert result.rounded_chern_number == -1
    assert result.is_nearly_integer
    assert result.berry_connection is not None
    assert result.min_link_magnitude is not None and result.min_link_magnitude > 0.9
    assert result.index_metadata is not None
    assert result.index_metadata["system"] == "qiwuzhang"
    assert result.index_metadata["valley"] == -1
    assert result.index_metadata["labels"] == ["lower"]
    assert result.index_metadata["metadata"] == {"mesh_source": "unit_test"}


def test_system_topology_adapter_from_grid_result_and_error_path() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=17, mass=-1.0)
    k_grid_frac = np.stack(
        np.meshgrid(np.arange(17) / 17.0, np.arange(17) / 17.0, indexing="ij"),
        axis=-1,
    )
    grid = SimpleNamespace(eigenvectors=wavefunctions, k_grid_frac=k_grid_frac)

    result = compute_system_topology_from_grid_result(grid, 0, system="qiwuzhang", valley=1)

    assert result.rounded_chern_number == -1
    np.testing.assert_allclose(result.k_grid_frac, k_grid_frac)
    assert result.to_dict()["rounded_chern_number"] == -1

    with pytest.raises(ValueError, match="Grid eigenvectors are required"):
        compute_system_topology_from_grid_result(SimpleNamespace(), 0, system="qiwuzhang")


def test_quantum_geometry_constant_wavefunction_is_flat() -> None:
    wavefunctions = np.zeros((4, 5, 2, 1), dtype=np.complex128)
    wavefunctions[:, :, 0, 0] = 1.0

    result = compute_quantum_geometry(wavefunctions, 0, include_fhs=True)

    assert result.quantum_geometric_tensor.shape == (2, 2, 4, 5)
    assert result.quantum_metric.shape == (2, 2, 4, 5)
    assert result.berry_curvature_density.shape == (4, 5)
    np.testing.assert_allclose(result.quantum_metric, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.berry_curvature_density, 0.0, atol=1.0e-12)
    assert np.isclose(result.projector_chern_number, 0.0, atol=1.0e-12)
    assert result.fhs_chern_number == pytest.approx(0.0, abs=1.0e-12)
    assert result.min_link_magnitude == pytest.approx(1.0, abs=1.0e-12)


def test_quantum_geometry_qiwuzhang_matches_fhs_sign_and_metric_shapes() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=21, mass=1.0)

    result = compute_quantum_geometry(wavefunctions, 0, include_fhs=True)

    assert result.fhs_chern_number == pytest.approx(1.0, abs=1.0e-12)
    assert result.projector_chern_number == pytest.approx(1.0, abs=5.0e-2)
    assert result.quantum_metric.shape == (2, 2, 21, 21)
    assert result.berry_curvature_density.shape == (21, 21)
    np.testing.assert_allclose(fubini_study_trace(result.quantum_metric), result.fubini_study_trace)
    assert float(np.min(result.trace_metric)) >= -1.0e-12

    normalized = normalize_quantum_geometry_maps(
        result,
        bz_area=1.0,
        metadata={"model": "qiwuzhang"},
    )
    assert normalized.integrated_berry_curvature == pytest.approx(result.projector_chern_number, abs=5.0e-2)
    assert normalized.integrated_fubini_study_trace > 0.0
    assert normalized.metadata == {"model": "qiwuzhang"}


def test_system_topology_grid_result_maps_absolute_band_indices_to_columns() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    grid = SimpleNamespace(eigenvectors=wavefunctions, band_indices=(100, 101))

    result = compute_system_topology_from_grid_result(grid, 101, system="mapped", valley=1)

    assert result.band_indices == (101,)
    assert result.rounded_chern_number == -1
    assert result.index_metadata is not None
    assert result.index_metadata["metadata"]["absolute_band_indices"] == [101]
    assert result.index_metadata["metadata"]["column_indices"] == [1]
    assert result.index_metadata["metadata"]["grid_result_band_indices"] == [100, 101]

    with pytest.raises(ValueError, match="not available"):
        compute_system_topology_from_grid_result(grid, 102, system="mapped")

def test_topology_bundle_guard_blocks_explicit_ineligible_metadata_before_fhs() -> None:
    bundle = SimpleNamespace(
        wavefunctions=np.zeros((1,), dtype=np.complex128),
        metadata={
            "topology_eligible": False,
            "topology_ineligible_reason": "unit-test no validated torus sewing",
            "evidence_paths": ["tests/test_analysis_topology.py"],
        },
    )

    with pytest.raises(ValueError, match="topology_eligible=False.*unit-test no validated torus sewing"):
        assert_topology_eligible(bundle, context="unit-test")
    with pytest.raises(ValueError, match="topology_eligible=False.*unit-test no validated torus sewing"):
        compute_system_topology_from_bundle(bundle, 0, system="toy")

def test_topology_bundle_helper_rejects_flat_reconstructed_bundle_with_clear_error() -> None:
    bundle = SimpleNamespace(
        wavefunctions=np.ones((6, 2, 1), dtype=np.complex128),
        metadata={"topology_eligible": True, "psi_micro_axis_order": "k,microscopic_basis,hf_state"},
    )
    with pytest.raises(ValueError, match="requires a 4D torus wavefunction grid.*system topology adapter"):
        compute_system_topology_from_bundle(bundle, 0, system="toy")


def test_topology_bundle_helper_allows_eligible_metadata_without_changing_low_level_arrays() -> None:
    wavefunctions = np.ones((2, 3, 1, 1), dtype=np.complex128)
    bundle = SimpleNamespace(
        wavefunctions=wavefunctions,
        metadata={"topology_eligible": True, "fixture": "constant-line-bundle"},
    )

    result = compute_system_topology_from_bundle(bundle, 0, system="toy", valley=0)

    assert result.rounded_chern_number == 0
    assert result.is_nearly_integer
    assert result.index_metadata is not None
    assert result.index_metadata["metadata"]["fixture"] == "constant-line-bundle"

    low_level = compute_lattice_topology(wavefunctions, 0, metadata={"topology_eligible": False})
    assert low_level.rounded_chern_number == 0
    assert low_level.metadata["topology_eligible"] is False
