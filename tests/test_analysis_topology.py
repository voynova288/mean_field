from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.topology import (
    WavefunctionIndex,
    WavefunctionLayout,
    canonicalize_wavefunction_grid,
    compute_lattice_topology,
    compute_quantum_geometry,
    infer_berry_sign_from_chern,
    integrated_fubini_study_metric,
    normalize_quantum_geometry_maps,
    normalized_chern_density,
    reconstruct_projected_micro_wavefunctions,
    trace_condition_violation,
)


def _qiwuzhang_eigenvectors(mesh_size: int, *, mass: float) -> np.ndarray:
    eigenvectors = np.zeros((mesh_size, mesh_size, 2, 2), dtype=np.complex128)
    frac = np.arange(mesh_size, dtype=float) / float(mesh_size)
    for ix, fx in enumerate(frac):
        kx = 2.0 * math.pi * fx
        for iy, fy in enumerate(frac):
            ky = 2.0 * math.pi * fy
            dx = math.sin(kx)
            dy = math.sin(ky)
            dz = mass + math.cos(kx) + math.cos(ky)
            hamiltonian = np.asarray(
                [
                    [dz, dx - 1j * dy],
                    [dx + 1j * dy, -dz],
                ],
                dtype=np.complex128,
            )
            _, vecs = np.linalg.eigh(hamiltonian)
            eigenvectors[ix, iy] = vecs
    return eigenvectors


def test_unified_topology_computes_connection_flux_and_chern_with_index_metadata() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(21, mass=-1.0)

    result = compute_lattice_topology(
        eigenvectors,
        0,
        index=WavefunctionIndex(indices=(0,), role="band", labels=("lower",), system="qwz", valley=1),
    )

    assert result.wavefunction_index.labels == ("lower",)
    assert result.berry_connection.shape == (2, 21, 21)
    assert result.berry_curvature.shape == (21, 21)
    assert result.is_nearly_integer
    assert abs(result.chern_number) == pytest.approx(1.0, abs=1.0e-8)


def test_unified_topology_is_invariant_under_local_band_phases() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(17, mass=-1.0)
    rng = np.random.default_rng(20240528)
    phases = np.exp(2j * np.pi * rng.random(eigenvectors.shape[:2] + eigenvectors.shape[-1:]))
    gauged = eigenvectors * phases[:, :, np.newaxis, :]

    reference = compute_lattice_topology(eigenvectors, 0)
    transformed = compute_lattice_topology(gauged, 0)

    assert transformed.chern_number == pytest.approx(reference.chern_number, abs=1.0e-10)


def test_unified_topology_total_two_band_subspace_is_trivial() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(21, mass=-1.0)

    result = compute_lattice_topology(eigenvectors, (0, 1))

    assert result.is_nearly_integer
    assert result.chern_number == pytest.approx(0.0, abs=1.0e-8)


def test_unified_topology_multiband_subspace_is_invariant_under_local_unitary_frames() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(13, mass=-1.0)
    rng = np.random.default_rng(8675309)
    rotated = np.empty_like(eigenvectors)
    for ix in range(eigenvectors.shape[0]):
        for iy in range(eigenvectors.shape[1]):
            random_matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            unitary, _ = np.linalg.qr(random_matrix)
            rotated[ix, iy] = eigenvectors[ix, iy] @ unitary

    reference = compute_lattice_topology(eigenvectors, (0, 1))
    transformed = compute_lattice_topology(rotated, (0, 1))

    assert transformed.chern_number == pytest.approx(reference.chern_number, abs=1.0e-10)
    assert transformed.is_nearly_integer


def test_boundary_sewing_transform_changes_only_wrapping_connection() -> None:
    vectors = np.zeros((3, 4, 2), dtype=np.complex128)
    vectors[:, :, 0] = 1.0

    result = compute_lattice_topology(
        vectors,
        index=WavefunctionIndex(indices=(0,), role="toy_line_bundle", labels=("constant",)),
        sewing_transforms=(lambda target: -target, None),
        link_method="determinant",
    )

    assert np.allclose(result.berry_connection[0, :-1, :], 0.0, atol=1.0e-12)
    assert np.allclose(np.abs(result.berry_connection[0, -1, :]), np.pi, atol=1.0e-12)
    assert result.chern_number == pytest.approx(0.0, abs=1.0e-12)


def test_quantum_geometry_projector_qgt_matches_fhs_chern_and_metric_bounds() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(31, mass=-1.0)

    result = compute_quantum_geometry(
        eigenvectors,
        0,
        index=WavefunctionIndex(indices=(0,), role="band", labels=("lower",), system="qwz"),
    )

    assert result.quantum_geometric_tensor.shape == (2, 2, 31, 31)
    assert result.quantum_metric.shape == (2, 2, 31, 31)
    assert result.berry_curvature_density.shape == (31, 31)
    assert result.fhs_chern_number == pytest.approx(-1.0, abs=1.0e-10)
    assert result.projector_chern_number == pytest.approx(result.fhs_chern_number, abs=2.5e-2)
    assert np.min(result.trace_metric) >= -1.0e-10
    assert np.min(result.determinant_condition_residual) >= -1.0e-8


def _qiwuzhang_exact_projector_geometry(mesh_size: int, *, mass: float) -> tuple[np.ndarray, np.ndarray]:
    metric = np.zeros((2, 2, mesh_size, mesh_size), dtype=float)
    omega = np.zeros((mesh_size, mesh_size), dtype=float)
    two_pi = 2.0 * math.pi
    for ix in range(mesh_size):
        kx = two_pi * float(ix) / float(mesh_size)
        for iy in range(mesh_size):
            ky = two_pi * float(iy) / float(mesh_size)
            d_vec = np.asarray(
                [math.sin(kx), math.sin(ky), mass + math.cos(kx) + math.cos(ky)],
                dtype=float,
            )
            norm = float(np.linalg.norm(d_vec))
            n_vec = d_vec / norm
            d_kx = np.asarray([math.cos(kx), 0.0, -math.sin(kx)], dtype=float)
            d_ky = np.asarray([0.0, math.cos(ky), -math.sin(ky)], dtype=float)
            dn_kx = (d_kx - n_vec * float(np.dot(n_vec, d_kx))) / norm
            dn_ky = (d_ky - n_vec * float(np.dot(n_vec, d_ky))) / norm

            # For the lower-band projector P=(1-n.sigma)/2:
            # g_ab = (1/4) d_a n . d_b n and, in the framework's FHS-oriented
            # convention, Omega_xy = -1/2 n . (d_x n x d_y n).
            scale = two_pi * two_pi  # convert k-derivatives to fractional-coordinate derivatives
            metric[0, 0, ix, iy] = 0.25 * float(np.dot(dn_kx, dn_kx)) * scale
            metric[1, 1, ix, iy] = 0.25 * float(np.dot(dn_ky, dn_ky)) * scale
            metric[0, 1, ix, iy] = 0.25 * float(np.dot(dn_kx, dn_ky)) * scale
            metric[1, 0, ix, iy] = metric[0, 1, ix, iy]
            omega[ix, iy] = -0.5 * float(np.dot(n_vec, np.cross(dn_kx, dn_ky))) * scale
    return metric, omega


def test_quantum_geometry_central_stencil_matches_exact_projector_geometry() -> None:
    mesh_size = 41
    eigenvectors = _qiwuzhang_eigenvectors(mesh_size, mass=-1.0)
    exact_metric, exact_omega = _qiwuzhang_exact_projector_geometry(mesh_size, mass=-1.0)

    result = compute_quantum_geometry(eigenvectors, 0, finite_difference="central")

    assert result.projector_chern_number == pytest.approx(-1.0, abs=1.2e-2)
    assert np.sqrt(np.mean((result.quantum_metric - exact_metric) ** 2)) < 8.0e-2
    assert np.sqrt(np.mean((result.berry_curvature_density - exact_omega) ** 2)) < 1.8e-1


def test_quantum_geometry_cartesian_qgt_matches_exact_components_and_normalizations() -> None:
    mesh_size = 41
    eigenvectors = _qiwuzhang_eigenvectors(mesh_size, mass=-1.0)
    exact_metric_frac, exact_omega_frac = _qiwuzhang_exact_projector_geometry(mesh_size, mass=-1.0)
    fractional_to_cartesian_scale = (2.0 * math.pi) ** 2
    exact_metric = exact_metric_frac / fractional_to_cartesian_scale
    exact_omega = exact_omega_frac / fractional_to_cartesian_scale

    result = compute_quantum_geometry(
        eigenvectors,
        0,
        finite_difference="central",
        coordinate_system="cartesian",
        reciprocal_vectors=2.0 * math.pi * np.eye(2),
        include_fhs=True,
    )

    assert result.fhs_chern_number == pytest.approx(-1.0, abs=1.0e-10)
    assert result.projector_chern_number == pytest.approx(-1.0, abs=1.2e-2)
    assert np.sqrt(np.mean((result.quantum_metric[0, 0] - exact_metric[0, 0]) ** 2)) < 2.5e-3
    assert np.sqrt(np.mean((result.quantum_metric[1, 1] - exact_metric[1, 1]) ** 2)) < 2.5e-3
    assert np.sqrt(np.mean((result.quantum_metric[0, 1] - exact_metric[0, 1]) ** 2)) < 2.5e-3
    assert np.sqrt(np.mean((result.fubini_study_trace - (exact_metric[0, 0] + exact_metric[1, 1])) ** 2)) < 5.0e-3
    assert np.sqrt(np.mean((result.berry_curvature_density - exact_omega) ** 2)) < 5.0e-3
    assert result.integrated_fubini_study_metric == pytest.approx(
        float(np.mean(result.normalized_fubini_study_trace)),
        abs=1.0e-12,
    )
    assert result.average_trace_condition_violation == pytest.approx(
        float(np.mean(result.normalized_fubini_study_trace - np.abs(result.normalized_berry_curvature))),
        abs=1.0e-12,
    )
    assert np.max(np.abs(result.determinant_condition_residual)) < 1.0e-5
    fhs_density = result.fhs_berry_curvature / result.momentum_area_element
    assert np.sqrt(np.mean((fhs_density - exact_omega) ** 2)) < 3.0e-2


def test_quantum_geometry_direct_cartesian_derivatives_match_fractional_transform() -> None:
    mesh_size = 31
    eigenvectors = _qiwuzhang_eigenvectors(mesh_size, mass=-1.0)
    reciprocal_vectors = 2.0 * math.pi * np.eye(2)

    transformed = compute_quantum_geometry(
        eigenvectors,
        0,
        finite_difference="central",
        coordinate_system="cartesian",
        reciprocal_vectors=reciprocal_vectors,
        include_fhs=False,
    )
    direct = compute_quantum_geometry(
        eigenvectors,
        0,
        finite_difference="central",
        coordinate_system="cartesian",
        derivative_coordinates="cartesian",
        coordinate_steps=(2.0 * math.pi / mesh_size, 2.0 * math.pi / mesh_size),
        reciprocal_vectors=reciprocal_vectors,
        include_fhs=False,
    )

    assert direct.derivative_coordinates == "cartesian"
    assert direct.momentum_area_element == pytest.approx((2.0 * math.pi / mesh_size) ** 2)
    assert np.max(np.abs(direct.quantum_metric - transformed.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(direct.berry_curvature_density - transformed.berry_curvature_density)) < 1.0e-10
    assert direct.integrated_fubini_study_metric == pytest.approx(transformed.integrated_fubini_study_metric, abs=1.0e-12)


def test_quantum_geometry_normalized_maps_api_handles_paper_signs() -> None:
    mesh_size = 41
    eigenvectors = _qiwuzhang_eigenvectors(mesh_size, mass=-1.0)
    result = compute_quantum_geometry(
        eigenvectors,
        0,
        finite_difference="central",
        coordinate_system="cartesian",
        reciprocal_vectors=2.0 * math.pi * np.eye(2),
        include_fhs=False,
    )

    common_maps = normalize_quantum_geometry_maps(result)
    sign = infer_berry_sign_from_chern(common_maps.integrated_berry_curvature, expected_chern=1.0)
    paper_maps = result.normalized_maps(berry_sign=sign, metadata={"reference": "positive C convention"})

    assert sign == -1.0
    assert paper_maps.berry_sign == -1.0
    assert paper_maps.integrated_berry_curvature == pytest.approx(1.0, abs=1.2e-2)
    assert paper_maps.integrated_fubini_study_trace == pytest.approx(result.integrated_fubini_study_metric, abs=1.0e-12)
    assert paper_maps.average_trace_condition_violation == pytest.approx(
        float(np.mean(paper_maps.fubini_study_trace - np.abs(paper_maps.berry_curvature))),
        abs=1.0e-12,
    )
    assert paper_maps.metadata["reference"] == "positive C convention"


def test_quantum_geometry_central_stencil_is_gauge_invariant_and_improves_local_maps() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(31, mass=-1.0)
    rng = np.random.default_rng(20260605)
    phases = np.exp(2j * np.pi * rng.random(eigenvectors.shape[:2] + eigenvectors.shape[-1:]))

    reference = compute_quantum_geometry(eigenvectors, 0, finite_difference="central")
    transformed = compute_quantum_geometry(eigenvectors * phases[:, :, np.newaxis, :], 0, finite_difference="central")

    assert reference.fhs_chern_number == pytest.approx(-1.0, abs=1.0e-10)
    assert reference.projector_chern_number == pytest.approx(reference.fhs_chern_number, abs=2.0e-2)
    assert np.max(np.abs(transformed.quantum_metric - reference.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(transformed.berry_curvature_density - reference.berry_curvature_density)) < 1.0e-10


def test_quantum_geometry_central_stencil_requires_backward_sewing_for_boundaries() -> None:
    vectors = np.zeros((3, 4, 2), dtype=np.complex128)
    vectors[:, :, 0] = 1.0

    with pytest.raises(ValueError, match="backward_sewing_transforms"):
        compute_quantum_geometry(
            vectors,
            finite_difference="central",
            sewing_transforms=(lambda target: -target, None),
        )


def test_quantum_geometry_is_invariant_under_local_band_phases() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(23, mass=-1.0)
    rng = np.random.default_rng(20260604)
    phases = np.exp(2j * np.pi * rng.random(eigenvectors.shape[:2] + eigenvectors.shape[-1:]))

    reference = compute_quantum_geometry(eigenvectors, 0)
    transformed = compute_quantum_geometry(eigenvectors * phases[:, :, np.newaxis, :], 0)

    assert transformed.fhs_chern_number == pytest.approx(reference.fhs_chern_number, abs=1.0e-10)
    assert np.max(np.abs(transformed.quantum_metric - reference.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(transformed.berry_curvature_density - reference.berry_curvature_density)) < 1.0e-10


def test_quantum_geometry_full_two_band_subspace_is_trivial() -> None:
    eigenvectors = _qiwuzhang_eigenvectors(17, mass=-1.0)

    result = compute_quantum_geometry(eigenvectors, (0, 1))

    assert result.fhs_chern_number == pytest.approx(0.0, abs=1.0e-10)
    assert result.projector_chern_number == pytest.approx(0.0, abs=1.0e-10)
    assert np.max(np.abs(result.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(result.berry_curvature_density)) < 1.0e-10


def test_quantum_geometry_nontrivial_rank_two_subspace_matches_line_bundle_and_is_u2_gauge_invariant() -> None:
    mesh_size = 23
    eigenvectors = _qiwuzhang_eigenvectors(mesh_size, mass=-1.0)
    embedded = np.zeros((mesh_size, mesh_size, 3, 2), dtype=np.complex128)
    embedded[:, :, :2, 0] = eigenvectors[:, :, :, 0]
    embedded[:, :, 2, 1] = 1.0

    line_reference = compute_quantum_geometry(eigenvectors, 0, finite_difference="central")
    subspace_reference = compute_quantum_geometry(embedded, finite_difference="central")

    assert subspace_reference.fhs_chern_number == pytest.approx(line_reference.fhs_chern_number, abs=1.0e-10)
    assert subspace_reference.projector_chern_number == pytest.approx(line_reference.projector_chern_number, abs=1.0e-10)
    assert np.max(np.abs(subspace_reference.quantum_metric - line_reference.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(subspace_reference.berry_curvature_density - line_reference.berry_curvature_density)) < 1.0e-10

    rng = np.random.default_rng(20260604)
    rotated = np.empty_like(embedded)
    for ix in range(mesh_size):
        for iy in range(mesh_size):
            random_matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            unitary, _ = np.linalg.qr(random_matrix)
            rotated[ix, iy] = embedded[ix, iy] @ unitary

    transformed = compute_quantum_geometry(rotated, finite_difference="central")

    assert transformed.fhs_chern_number == pytest.approx(subspace_reference.fhs_chern_number, abs=1.0e-10)
    assert transformed.projector_chern_number == pytest.approx(subspace_reference.projector_chern_number, abs=1.0e-10)
    assert np.max(np.abs(transformed.quantum_metric - subspace_reference.quantum_metric)) < 1.0e-10
    assert np.max(np.abs(transformed.berry_curvature_density - subspace_reference.berry_curvature_density)) < 1.0e-10


def test_wavefunction_layout_flattens_band_flavor_axes_and_preserves_index_labels() -> None:
    wavefunctions = np.zeros((2, 3, 4, 2, 2), dtype=np.complex128)
    canonical = canonicalize_wavefunction_grid(
        wavefunctions,
        WavefunctionLayout(
            basis_axis=2,
            state_axes=(3, 4),
            state_axis_names=("band", "flavor"),
            state_axis_labels={"band": ("valence", "conduction"), "flavor": ("K_up", "K_down")},
        ),
    )

    index = canonical.index_for((0, 3), role="band_flavor_subspace", system="toy")

    assert canonical.wavefunctions.shape == (2, 3, 4, 4)
    assert index.indices == (0, 3)
    assert index.labels == ("band=valence/flavor=K_up", "band=conduction/flavor=K_down")
    assert index.metadata["selected_state_labels"] == [
        {"band": "valence", "flavor": "K_up"},
        {"band": "conduction", "flavor": "K_down"},
    ]


def test_fubini_study_paper_normalizations_are_available() -> None:
    trace = np.asarray([[2.0, 4.0], [6.0, 8.0]])
    curvature = np.asarray([[1.0, -2.0], [3.0, -4.0]])

    assert integrated_fubini_study_metric(trace, area_element=0.5) == pytest.approx(20.0 * 0.5 / (2.0 * np.pi))
    assert normalized_chern_density(curvature, bz_area=2.0 * np.pi) == pytest.approx(curvature)
    assert trace_condition_violation(trace, curvature, bz_area=2.0 * np.pi) == pytest.approx(
        trace - np.abs(curvature)
    )


def test_projected_hf_micro_wavefunction_reconstruction_preserves_spin_flavor_blocks() -> None:
    # basis_dim=2, n_band=2, n_flavor=2, nk=4; n_spin=2 gives nt=8.
    basis = np.zeros((2, 2, 2, 4), dtype=np.complex128)
    basis[0, 0, :, :] = 1.0
    basis[1, 1, :, :] = 1.0
    mixing = np.zeros((8, 1, 4), dtype=np.complex128)
    # Flattening order F means idx[spin, flavor, band] = spin + 2*flavor + 4*band.
    mixing[0, 0, :] = 1.0  # spin0, flavor0, band0
    mixing[7, 0, :] = 2.0  # spin1, flavor1, band1

    reconstructed = reconstruct_projected_micro_wavefunctions(
        basis,
        mixing,
        (2, 2),
        n_spin=2,
        flatten_order="F",
    )

    assert reconstructed.shape == (2, 2, 8, 1)
    assert np.allclose(reconstructed[:, :, 0, 0], 1.0)
    assert np.allclose(reconstructed[:, :, 7, 0], 2.0)
    assert np.count_nonzero(np.abs(reconstructed[:, :, :, 0]) > 1.0e-12) == 8
