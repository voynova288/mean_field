from __future__ import annotations

import numpy as np
import pytest

from analysis.topology import (
    BlockSewingSpec,
    FHSState,
    compute_lattice_topology,
    fhs_state_from_grid_result,
    fhs_state_from_wavefunctions,
    matrix_sewing_transform,
    sewing_transforms_from_block_spec,
)


def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> np.ndarray:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
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
    return wavefunctions


def test_fhs_state_chern_for_qiwuzhang_single_bands_and_subspace() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=21, mass=1.0)

    lower = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, system="qiwuzhang", labels=("lower",)))
    upper = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 1, system="qiwuzhang", labels=("upper",)))
    full_subspace = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, (0, 1), link_method="determinant"))

    assert lower.rounded_chern_number == 1
    assert upper.rounded_chern_number == -1
    assert full_subspace.rounded_chern_number == 0
    assert lower.is_nearly_integer
    assert lower.min_link_magnitude > 0.9
    assert lower.index_metadata["system"] == "qiwuzhang"
    assert lower.index_metadata["labels"] == ["lower"]


def test_fhs_chern_distinguishes_trivial_and_topological_mass_regions() -> None:
    topological = _qiwuzhang_wavefunctions(mesh=21, mass=-1.0)
    trivial = _qiwuzhang_wavefunctions(mesh=21, mass=3.0)

    assert compute_lattice_topology(fhs_state_from_wavefunctions(topological, 0)).rounded_chern_number == -1
    assert compute_lattice_topology(fhs_state_from_wavefunctions(trivial, 0)).rounded_chern_number == 0


def test_grid_state_maps_absolute_band_labels_to_columns() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    grid = type("Grid", (), {"eigenvectors": wavefunctions, "band_indices": (100, 101)})()

    state = fhs_state_from_grid_result(grid, 101, system="mapped", valley=1)
    result = compute_lattice_topology(state)

    assert state.state_indices == (1,)
    assert state.reported_indices == (101,)
    assert result.band_indices == (101,)
    assert result.metadata["absolute_band_indices"] == [101]
    assert result.metadata["column_indices"] == [1]
    with pytest.raises(ValueError, match="Requested band labels"):
        fhs_state_from_grid_result(grid, 102, system="mapped")


def test_lattice_topology_orientation_sign_flips_links_connection_and_curvature() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    positive = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0))
    negative = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, orientation_sign=-1.0))

    np.testing.assert_allclose(negative.link_1, positive.link_1.conjugate())
    np.testing.assert_allclose(negative.link_2, positive.link_2.conjugate())
    np.testing.assert_allclose(negative.berry_connection, -positive.berry_connection)
    np.testing.assert_allclose(negative.berry_curvature, -positive.berry_curvature)
    assert negative.rounded_chern_number == -positive.rounded_chern_number
    with pytest.raises(ValueError, match="orientation_sign"):
        compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, orientation_sign=0.5))


def test_matrix_sewing_transform_is_target_side_on_wrapped_links() -> None:
    swap = matrix_sewing_transform(np.asarray([[0, 1], [1, 0]], dtype=np.complex128))
    np.testing.assert_allclose(swap(np.asarray([1, 2], dtype=np.complex128)), np.asarray([2, 1], dtype=np.complex128))
    with pytest.raises(ValueError, match="Expected first axis"):
        swap(np.ones((3,), dtype=np.complex128))


def test_block_sewing_spec_generates_common_target_side_sewing() -> None:
    spec = BlockSewingSpec(
        block_coordinates=np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        local_block_size=2,
        translations=((1.0,), (0.0,)),
    )
    sew_1, sew_2 = sewing_transforms_from_block_spec(spec)
    vector = np.arange(6, dtype=np.complex128)

    np.testing.assert_array_equal(sew_1(vector), np.asarray([2, 3, 4, 5, 0, 0], dtype=np.complex128))
    np.testing.assert_array_equal(sew_2(vector), vector)

    wavefunctions = np.ones((3, 2, 6, 1), dtype=np.complex128)
    from_state = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, basis_sewing=spec))
    explicit = compute_lattice_topology(wavefunctions, 0, sewing_transforms=(sew_1, sew_2))

    np.testing.assert_allclose(from_state.link_1, explicit.link_1)
    np.testing.assert_allclose(from_state.link_2, explicit.link_2)
    np.testing.assert_allclose(from_state.berry_curvature, explicit.berry_curvature)
    assert from_state.chern_number == pytest.approx(explicit.chern_number)


def test_removed_topology_adapter_public_api_names_are_absent() -> None:
    import analysis.topology as topology

    for name in (
        "compute_system_topology_from_eigenvectors",
        "compute_system_topology_from_grid_result",
        "compute_system_topology_from_bundle",
        "WavefunctionLayout",
        "canonicalize_wavefunction_grid",
    ):
        assert not hasattr(topology, name)

    assert isinstance(fhs_state_from_wavefunctions(np.ones((2, 2, 1, 1)), 0), FHSState)
