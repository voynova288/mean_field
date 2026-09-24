from __future__ import annotations

import numpy as np
import pytest

import analysis.topology as topology_api
from analysis.topology import (
    BlockSewingSpec,
    berry_curvature_from_links,
    chern_number_from_berry_curvature,
    compute_lattice_topology,
    compute_link_variables,
    fhs_state_from_wavefunctions,
    link_variable_from_overlap_matrix,
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
                [
                    [dz, np.sin(kx) - 1j * np.sin(ky)],
                    [np.sin(kx) + 1j * np.sin(ky), -dz],
                ],
                dtype=np.complex128,
            )
            _, vecs = np.linalg.eigh(hamiltonian)
            wavefunctions[ix, iy] = vecs
    return wavefunctions


def test_public_topology_api_is_fhs_only() -> None:
    for name in (
        "compute_lattice_topology",
        "compute_link_variables",
        "berry_curvature_from_links",
        "chern_number_from_berry_curvature",
        "link_variable_from_overlap_matrix",
    ):
        assert hasattr(topology_api, name)

    for removed in (
        "compute_quantum_geometry",
        "make_topology_adapter",
        "compute_system_topology_from_eigenvectors",
        "WavefunctionLayout",
        "canonicalize_wavefunction_grid",
        "projector_qgt_forward_difference",
        "projector_qgt_central_difference",
        "qgt_to_metric_and_berry",
        "TopologyResult",
    ):
        assert not hasattr(topology_api, removed)


def test_compute_lattice_topology_requires_canonical_state() -> None:
    wavefunctions = np.ones((2, 2, 1, 1), dtype=np.complex128)
    with pytest.raises(TypeError, match="expects FHSState"):
        compute_lattice_topology(wavefunctions)  # type: ignore[arg-type]


def test_full_state_pipeline_rejects_zero_nonfinite_and_singular_links() -> None:
    scalar = np.zeros((2, 2, 2, 1), dtype=np.complex128)
    scalar[..., 0, 0] = 1.0
    scalar[1, 0, :, 0] = np.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="magnitude"):
        compute_lattice_topology(fhs_state_from_wavefunctions(scalar, 0))

    nonfinite = np.zeros((2, 2, 1, 1), dtype=np.complex128)
    nonfinite[..., 0, 0] = 1.0
    nonfinite[1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_lattice_topology(fhs_state_from_wavefunctions(nonfinite, 0))

    singular = np.zeros((2, 2, 3, 2), dtype=np.complex128)
    singular[..., 0, 0] = 1.0
    singular[..., 1, 1] = 1.0
    singular[1, 0, :, 1] = np.asarray([0.0, 0.0, 1.0])
    for method in ("polar", "determinant"):
        with pytest.raises(ValueError, match="singular value"):
            compute_lattice_topology(
                fhs_state_from_wavefunctions(
                    singular, (0, 1), link_method=method
                )
            )

    zero_fill_spec = BlockSewingSpec(
        block_coordinates=np.asarray([[0.0]]),
        local_block_size=1,
        translations=((1.0,), (0.0,)),
    )
    with pytest.raises(ValueError, match="magnitude"):
        compute_lattice_topology(
            fhs_state_from_wavefunctions(
                np.ones((2, 2, 1, 1), dtype=np.complex128),
                0,
                basis_sewing=zero_fill_spec,
            )
        )


def test_state_builder_rejects_ambiguous_sewing_sources() -> None:
    spec = BlockSewingSpec(
        block_coordinates=np.asarray([[0.0]]),
        local_block_size=1,
        translations=((0.0,), (0.0,)),
    )
    identity = matrix_sewing_transform(np.eye(1, dtype=np.complex128))
    with pytest.raises(ValueError, match="either sewing_transforms or basis_sewing"):
        fhs_state_from_wavefunctions(
            np.ones((2, 2, 1, 1), dtype=np.complex128),
            0,
            sewing_transforms=(identity, identity),
            basis_sewing=spec,
        )


def test_fhs_chern_for_qiwuzhang_single_bands_and_subspace() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=21, mass=1.0)

    lower = compute_lattice_topology(
        fhs_state_from_wavefunctions(
            wavefunctions, 0, system="qiwuzhang", labels=("lower",)
        )
    )
    upper = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 1))
    full_subspace = compute_lattice_topology(
        fhs_state_from_wavefunctions(
            wavefunctions, (0, 1), link_method="determinant"
        )
    )

    assert lower.rounded_chern_number == 1
    assert lower.is_nearly_integer
    assert upper.rounded_chern_number == -1
    assert np.isclose(lower.chern_number + upper.chern_number, 0.0, atol=1.0e-12)
    assert full_subspace.rounded_chern_number == 0
    assert full_subspace.is_nearly_integer
    assert lower.min_link_magnitude > 0.9
    assert lower.minimum_plaquette_branch_margin == pytest.approx(
        np.pi - np.max(np.abs(lower.berry_curvature))
    )
    assert lower.to_dict()["minimum_plaquette_branch_margin"] == pytest.approx(
        lower.minimum_plaquette_branch_margin
    )
    assert lower.band_indices == (0,)
    assert lower.index_metadata["system"] == "qiwuzhang"
    assert lower.index_metadata["labels"] == ["lower"]


def test_full_grid_local_frame_covariance_and_link_method_parity() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=11, mass=1.0)
    rng = np.random.default_rng(831)

    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(11, 11)))
    gauged_single = wavefunctions.copy()
    gauged_single[..., 0] *= phase[..., None]
    base_single = compute_lattice_topology(
        fhs_state_from_wavefunctions(wavefunctions, 0)
    )
    transformed_single = compute_lattice_topology(
        fhs_state_from_wavefunctions(gauged_single, 0)
    )
    np.testing.assert_allclose(
        transformed_single.berry_curvature,
        base_single.berry_curvature,
        atol=1.0e-12,
        rtol=0.0,
    )

    gauged_full = wavefunctions.copy()
    for ix in range(11):
        for iy in range(11):
            q, _ = np.linalg.qr(
                rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            )
            gauged_full[ix, iy] = gauged_full[ix, iy] @ q
    results = {}
    for method in ("polar", "determinant"):
        results[("base", method)] = compute_lattice_topology(
            fhs_state_from_wavefunctions(
                wavefunctions, (0, 1), link_method=method
            )
        )
        results[("gauged", method)] = compute_lattice_topology(
            fhs_state_from_wavefunctions(
                gauged_full, (0, 1), link_method=method
            )
        )
        np.testing.assert_allclose(
            results[("gauged", method)].berry_curvature,
            results[("base", method)].berry_curvature,
            atol=1.0e-12,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        results[("base", "polar")].berry_curvature,
        results[("base", "determinant")].berry_curvature,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_fhs_chern_distinguishes_trivial_and_topological_regions() -> None:
    topological = _qiwuzhang_wavefunctions(mesh=21, mass=-1.0)
    trivial = _qiwuzhang_wavefunctions(mesh=21, mass=3.0)

    assert compute_lattice_topology(
        fhs_state_from_wavefunctions(topological, 0)
    ).rounded_chern_number == -1
    assert compute_lattice_topology(
        fhs_state_from_wavefunctions(trivial, 0)
    ).rounded_chern_number == 0


def test_overlap_matrix_adapter_uses_common_fhs_link_normalization() -> None:
    scalar_link, scalar_magnitude = link_variable_from_overlap_matrix(
        np.asarray([[3.0 + 4.0j]])
    )
    assert scalar_link == pytest.approx(0.6 + 0.8j)
    assert scalar_magnitude == pytest.approx(5.0)

    phase_left = 0.3
    phase_right = -0.1
    unitary = np.diag(np.exp(1j * np.asarray([phase_left, phase_right])))
    overlap = unitary @ np.diag([2.0, 0.5])
    subspace_link, minimum_singular = link_variable_from_overlap_matrix(overlap)
    assert subspace_link == pytest.approx(np.exp(1j * (phase_left + phase_right)))
    assert minimum_singular == pytest.approx(0.5)

    with pytest.raises(ValueError, match="square endpoint-overlap"):
        link_variable_from_overlap_matrix(np.ones((2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="magnitude"):
        link_variable_from_overlap_matrix(np.zeros((1, 1), dtype=np.complex128))
    with pytest.raises(ValueError, match="singular"):
        link_variable_from_overlap_matrix(np.diag([1.0, 0.0]).astype(np.complex128))

    rng = np.random.default_rng(17)
    left_q, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    )
    right_q, _ = np.linalg.qr(
        rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    )
    transformed_link, transformed_minimum = link_variable_from_overlap_matrix(
        left_q.conj().T @ overlap @ right_q
    )
    expected_covariance = (
        np.linalg.det(left_q).conjugate()
        * subspace_link
        * np.linalg.det(right_q)
    )
    assert transformed_link == pytest.approx(expected_covariance)
    assert transformed_minimum == pytest.approx(minimum_singular)


def test_links_curvature_and_chern_are_the_only_berry_path() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)
    selected = wavefunctions[..., 0]

    links = compute_link_variables(selected)
    curvature = berry_curvature_from_links(links.link_1, links.link_2)
    chern = chern_number_from_berry_curvature(curvature)
    result = compute_lattice_topology(
        fhs_state_from_wavefunctions(wavefunctions, 0)
    )

    np.testing.assert_allclose(curvature, result.berry_curvature)
    assert chern == pytest.approx(result.chern_number, abs=1.0e-12)
    assert result.rounded_chern_number == 1


def test_k_grid_metadata_shape_and_finiteness_fail_closed() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=5, mass=1.0)
    with pytest.raises(ValueError, match="k_grid_frac must have shape"):
        compute_lattice_topology(
            fhs_state_from_wavefunctions(
                wavefunctions,
                0,
                k_grid_frac=np.zeros((5, 5, 3), dtype=float),
            )
        )
    invalid = np.zeros((5, 5, 2), dtype=float)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_lattice_topology(
            fhs_state_from_wavefunctions(wavefunctions, 0, k_grid_frac=invalid)
        )

    endpoint_axis = np.linspace(0.0, 1.0, 5)
    endpoint_grid = np.stack(
        np.meshgrid(endpoint_axis, endpoint_axis, indexing="ij"), axis=-1
    )
    with pytest.raises(ValueError, match="duplicate reciprocal-equivalent"):
        compute_lattice_topology(
            fhs_state_from_wavefunctions(
                wavefunctions,
                0,
                k_grid_frac=endpoint_grid,
            )
        )


def test_orientation_sign_conjugates_links_and_flips_curvature() -> None:
    wavefunctions = _qiwuzhang_wavefunctions(mesh=17, mass=1.0)

    positive = compute_lattice_topology(
        fhs_state_from_wavefunctions(wavefunctions, 0)
    )
    negative = compute_lattice_topology(
        fhs_state_from_wavefunctions(wavefunctions, 0, orientation_sign=-1.0)
    )

    np.testing.assert_allclose(negative.link_1, positive.link_1.conjugate())
    np.testing.assert_allclose(negative.link_2, positive.link_2.conjugate())
    np.testing.assert_allclose(negative.berry_connection, -positive.berry_connection)
    np.testing.assert_allclose(negative.berry_curvature, -positive.berry_curvature)
    assert negative.rounded_chern_number == -positive.rounded_chern_number

    with pytest.raises(ValueError, match="orientation_sign"):
        compute_lattice_topology(
            fhs_state_from_wavefunctions(wavefunctions, 0, orientation_sign=0.5)
        )


def test_block_sewing_spec_generates_common_target_side_sewing() -> None:
    block_coordinates = np.asarray([[0], [1], [2]], dtype=float)
    spec = BlockSewingSpec(
        block_coordinates=block_coordinates,
        local_block_size=2,
        translations=((1.0,), (0.0,)),
    )
    vector = np.arange(6, dtype=np.complex128)
    sew_1, _ = sewing_transforms_from_block_spec(spec)

    np.testing.assert_array_equal(sew_1(vector), np.asarray([2, 3, 4, 5, 0, 0], dtype=np.complex128))

    wavefunctions = np.ones((3, 2, 6, 1), dtype=np.complex128)
    from_state = compute_lattice_topology(fhs_state_from_wavefunctions(wavefunctions, 0, basis_sewing=spec))
    explicit = compute_lattice_topology(
        fhs_state_from_wavefunctions(
            wavefunctions,
            0,
            sewing_transforms=(sew_1, sewing_transforms_from_block_spec(spec)[1]),
        )
    )

    np.testing.assert_allclose(from_state.link_1, explicit.link_1)
    np.testing.assert_allclose(from_state.link_2, explicit.link_2)
    np.testing.assert_allclose(from_state.berry_curvature, explicit.berry_curvature)
    assert from_state.chern_number == pytest.approx(explicit.chern_number)


def test_matrix_sewing_transform_is_target_side_on_wrapped_links() -> None:
    # A constant line bundle with an anti-periodic first boundary needs a -1
    # target-side sewing matrix to recover a trivial torus Chern/link structure.
    wavefunctions = np.ones((3, 4, 1, 1), dtype=np.complex128)
    wavefunctions[0, :, 0, 0] = -1.0

    without_sewing = compute_lattice_topology(
        fhs_state_from_wavefunctions(wavefunctions, 0)
    )
    with_sewing = compute_lattice_topology(
        fhs_state_from_wavefunctions(
            wavefunctions,
            0,
            sewing_transforms=(
                matrix_sewing_transform(
                    np.asarray([[-1.0]], dtype=np.complex128)
                ),
                None,
            ),
        )
    )

    assert without_sewing.min_link_magnitude == pytest.approx(1.0, abs=1.0e-12)
    assert with_sewing.min_link_magnitude == pytest.approx(1.0, abs=1.0e-12)
    assert with_sewing.rounded_chern_number == 0
