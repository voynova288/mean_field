from __future__ import annotations

from dataclasses import replace
from math import sqrt

import numpy as np
import pytest
from scipy.fft import ifftn, next_fast_len

from mean_field.systems.tise2.folded_hf import (
    FoldedTiSe2Geometry,
    physical_momentum_routes,
    physical_q_vectors,
    precompute_folded_coulomb_kernel,
)
from mean_field.systems.tise2.local_full_hf import local_full_h0
from mean_field.systems.tise2.local_ws_fft import (
    LocalWSFFTKernel,
    LocalWSMesh,
    LocalWSNormalState,
    build_local_ws_fft_kernel,
    build_local_ws_mesh,
    build_local_ws_normal_state,
    estimate_local_ws_fft_peak_bytes,
    local_ws_fft_fock_action,
    local_ws_fft_hartree_action,
    local_ws_fft_interaction_action,
    local_ws_fft_interaction_energy,
    local_ws_fft_ordered_kernel_action,
)


_HARD_CODED_ROUTES = (
    (((0, 0), (1, 1), (2, 2), (3, 3)), ((1, 0),), ((2, 0),), ((3, 0),)),
    (((0, 1),), ((0, 0), (1, 1), (2, 2), (3, 3)), ((2, 1),), ((3, 1),)),
    (((0, 2),), ((1, 2),), ((0, 0), (1, 1), (2, 2), (3, 3)), ((3, 2),)),
    (((0, 3),), ((1, 3),), ((2, 3),), ((0, 0), (1, 1), (2, 2), (3, 3))),
)

def _random_complex(nk: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))


def _random_hermitian(nk: int, *, seed: int) -> np.ndarray:
    raw = _random_complex(nk, seed=seed)
    return 0.5 * (raw + raw.conj().swapaxes(0, 1))


@pytest.fixture(scope="module")
def ws_mesh() -> LocalWSMesh:
    return build_local_ws_mesh(M=1, L=1)


@pytest.fixture(scope="module")
def ws_kernel(ws_mesh: LocalWSMesh) -> LocalWSFFTKernel:
    return build_local_ws_fft_kernel(ws_mesh, epsilon_r=11.0)


@pytest.fixture(scope="module")
def dense_kernel(ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel):
    return precompute_folded_coulomb_kernel(
        ws_mesh,
        q_vectors_Ainv=ws_kernel.q_vectors_Ainv,
        epsilon_r=ws_kernel.epsilon_r,
        max_dense_memory_gb=0.05,
    )


def _dense_actions(density: np.ndarray, dense) -> tuple[np.ndarray, np.ndarray]:
    routes = _HARD_CODED_ROUTES
    fock = np.zeros_like(density)
    hartree = np.zeros_like(density)
    for s in range(4):
        for t in range(4):
            for u, v in routes[s][t]:
                fock[s, t] -= dense.shifted_weighted_ev[s, v] @ density[u, v]
                hartree[s, t] += dense.hartree_ev[s, t] * np.einsum(
                    "p,p->",
                    density[u, v],
                    dense.mesh.weights_Ainv3,
                    optimize=True,
                )
    return hartree, fock


@pytest.mark.parametrize("M,L", [(0, 0), (1, 2), (3, 1)])
def test_ws_cubature_exact_geometry_volume_and_label_map(M: int, L: int) -> None:
    geometry = FoldedTiSe2Geometry(a_angstrom=3.61, c_angstrom=6.12)
    mesh = build_local_ws_mesh(M=M, L=L, geometry=geometry)
    q1 = physical_q_vectors(geometry)[1]
    B = float(np.linalg.norm(q1[:2]))
    Qz = float(q1[2])
    expected_nk = (3 * M * (M + 1) + 1) * (2 * L + 1)
    expected_area = sqrt(3.0) * B**2 / 2.0
    expected_volume = expected_area * Qz

    assert mesh.nk == expected_nk
    assert mesh.geometry == geometry
    np.testing.assert_array_equal(mesh.q_vectors_Ainv, physical_q_vectors(geometry))
    assert mesh.B_Ainv == mesh.Qparallel_Ainv == B
    assert mesh.Qz_Ainv == Qz
    assert mesh.d == 2 * M + 1
    assert mesh.dz == 2 * L + 1
    assert mesh.cell_volume_Ainv3 == expected_volume / expected_nk
    np.testing.assert_allclose(
        np.sum(mesh.weights_Ainv3),
        expected_volume / (2.0 * np.pi) ** 3,
        rtol=4e-16,
        atol=0.0,
    )
    assert np.all(mesh.weights_Ainv3 == mesh.weights_Ainv3[0])
    assert np.unique(mesh.labels, axis=0).shape[0] == expected_nk
    assert np.count_nonzero(mesh.box_mask) == expected_nk

    m, n, ell = mesh.labels.T
    assert np.all(2 * np.abs(m) < mesh.d)
    assert np.all(2 * np.abs(n) < mesh.d)
    assert np.all(2 * np.abs(m + n) < mesh.d)
    assert np.all(2 * np.abs(ell) < mesh.dz)
    expected_points = np.column_stack(
        (
            B * m / mesh.d,
            B * (m + 2 * n) / (sqrt(3.0) * mesh.d),
            Qz * ell / mesh.dz,
        )
    )
    np.testing.assert_array_equal(mesh.points_Ainv, expected_points)
    x, y, z = mesh.points_Ainv.T
    assert np.all(2.0 * np.abs(x) < B)
    assert np.all(np.abs(-x + sqrt(3.0) * y) < B)
    assert np.all(np.abs(x + sqrt(3.0) * y) < B)
    assert np.all(2.0 * np.abs(z) < Qz)

    labels = {tuple(int(value) for value in row) for row in mesh.labels}
    assert {(-m0, -n0, -l0) for m0, n0, l0 in labels} == labels
    assert {(-n0, m0 + n0, l0) for m0, n0, l0 in labels} == labels
    np.testing.assert_allclose(
        np.einsum("k,kd->d", mesh.weights_Ainv3, mesh.points_Ainv),
        np.zeros(3),
        rtol=0.0,
        atol=2e-19,
    )
    np.testing.assert_allclose(
        np.einsum("k,kd->d", mesh.weights_Ainv3, mesh.points_Ainv**3),
        np.zeros(3),
        rtol=0.0,
        atol=2e-20,
    )


def test_kernel_inventory_padding_hex_mask_and_no_shifted_zero_alias(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel
) -> None:
    assert ws_kernel.ordered_kernel_fft_ev.shape == (4, 4, *ws_kernel.fft_shape)
    assert ws_kernel.fft_shape == tuple(
        next_fast_len(2 * size - 1) for size in ws_mesh.box_shape
    )
    assert all(
        actual >= 2 * box - 1
        for actual, box in zip(ws_kernel.fft_shape, ws_mesh.box_shape)
    )
    assert ws_kernel.singular_inventory == tuple((s, s, 0, 0, 0) for s in range(4))
    assert sum(len(block) for row in physical_momentum_routes() for block in row) == 28

    differences = (
        ws_mesh.points_Ainv[:, None, :] - ws_mesh.points_Ainv[None, :, :]
    )
    tolerance = 64.0 * np.finfo(float).eps
    for s in range(4):
        for v in range(4):
            transfer = differences + ws_kernel.q_vectors_Ainv[s] - ws_kernel.q_vectors_Ainv[v]
            singular = np.einsum("...d,...d->...", transfer, transfer) <= tolerance
            if s == v:
                assert np.array_equal(singular, np.eye(ws_mesh.nk, dtype=bool))
            else:
                assert not np.any(singular)

    padded = ifftn(ws_kernel.ordered_kernel_fft_ev[0, 1]).real
    for dm in range(-(ws_mesh.d - 1), ws_mesh.d):
        for dn in range(-(ws_mesh.d - 1), ws_mesh.d):
            if max(abs(dm), abs(dn), abs(dm + dn)) <= 2 * ws_mesh.M:
                continue
            for dl in range(-(ws_mesh.dz - 1), ws_mesh.dz):
                assert padded[
                    dm % ws_kernel.fft_shape[0],
                    dn % ws_kernel.fft_shape[1],
                    dl % ws_kernel.fft_shape[2],
                ] == pytest.approx(0.0, abs=2e-14)


def test_fft_peak_memory_estimate_is_recorded_and_cap_fails_before_allocation(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel
) -> None:
    expected = estimate_local_ws_fft_peak_bytes(
        ws_mesh, fft_shape=ws_kernel.fft_shape
    )
    assert ws_kernel.estimated_peak_bytes == expected
    assert ws_kernel.estimated_peak_bytes > ws_kernel.ordered_kernel_fft_ev.nbytes
    with pytest.raises(MemoryError, match="peak-memory cap"):
        build_local_ws_fft_kernel(
            ws_mesh,
            epsilon_r=11.0,
            max_fft_memory_gb=(expected - 1) / 1.0e9,
        )


def test_mesh_constructor_and_kernel_geometry_mismatches_fail_closed(
    ws_mesh: LocalWSMesh,
) -> None:
    shifted_points = ws_mesh.points_Ainv.copy()
    shifted_points[ws_mesh.gamma_index, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="label-to-point map"):
        replace(ws_mesh, points_Ainv=shifted_points)

    other_geometry = FoldedTiSe2Geometry(
        a_angstrom=ws_mesh.geometry.a_angstrom * 1.01,
        c_angstrom=ws_mesh.geometry.c_angstrom,
    )
    with pytest.raises(ValueError, match="geometry/q vectors do not match"):
        build_local_ws_fft_kernel(
            ws_mesh, epsilon_r=11.0, geometry=other_geometry
        )


def test_all_sixteen_ordered_fft_kernels_match_existing_dense_oracle(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    rng = np.random.default_rng(20260924)
    source = rng.normal(size=ws_mesh.nk) + 1j * rng.normal(size=ws_mesh.nk)
    for s in range(4):
        for v in range(4):
            actual = local_ws_fft_ordered_kernel_action(
                source, kernel=ws_kernel, sector=s, density_col_sector=v
            )
            expected = dense_kernel.shifted_weighted_ev[s, v] @ source
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_nonhermitian_route_orientation_impulses_match_hard_coded_oracle(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    source_index = int(np.argmax(ws_mesh.labels[:, 0] - 2 * ws_mesh.labels[:, 2]))
    amplitude = 0.73 - 1.21j
    for density_row in range(4):
        for density_col in range(4):
            density = np.zeros((4, 4, ws_mesh.nk), dtype=np.complex128)
            density[density_row, density_col, source_index] = amplitude
            expected_h = np.zeros_like(density)
            expected_f = np.zeros_like(density)
            for sector in range(4):
                for target_sector in range(4):
                    if (density_row, density_col) not in _HARD_CODED_ROUTES[sector][target_sector]:
                        continue
                    expected_f[sector, target_sector] = -(
                        dense_kernel.shifted_weighted_ev[sector, density_col, :, source_index]
                        * amplitude
                    )
                    expected_h[sector, target_sector] = (
                        dense_kernel.hartree_ev[sector, target_sector]
                        * ws_mesh.weights_Ainv3[source_index]
                        * amplitude
                    )
            np.testing.assert_allclose(
                local_ws_fft_fock_action(density, kernel=ws_kernel),
                expected_f,
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                local_ws_fft_hartree_action(density, kernel=ws_kernel),
                expected_h,
                rtol=2e-13,
                atol=2e-13,
            )


def test_boundary_impulses_have_no_circular_wrap(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    labels = ws_mesh.labels
    boundary = np.flatnonzero(
        (np.max(np.abs(labels[:, :2]), axis=1) == ws_mesh.M)
        | (np.abs(labels[:, 0] + labels[:, 1]) == ws_mesh.M)
        | (np.abs(labels[:, 2]) == ws_mesh.L)
    )
    assert boundary.size > 0
    for source_index in boundary:
        impulse = np.zeros(ws_mesh.nk, dtype=np.complex128)
        impulse[source_index] = 1.0 - 0.25j
        actual = local_ws_fft_ordered_kernel_action(
            impulse, kernel=ws_kernel, sector=2, density_col_sector=1
        )
        expected = dense_kernel.shifted_weighted_ev[2, 1] @ impulse
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_asymmetric_padded_mesh_all_kernels_and_corner_impulses_dense_parity() -> None:
    mesh = build_local_ws_mesh(M=3, L=1)
    kernel = build_local_ws_fft_kernel(
        mesh, epsilon_r=13.0, max_fft_memory_gb=0.05
    )
    minimum = tuple(2 * size - 1 for size in mesh.box_shape)
    assert kernel.fft_shape[0] > minimum[0]
    assert kernel.fft_shape[1] > minimum[1]
    assert kernel.fft_shape[2] == minimum[2]
    dense = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=kernel.q_vectors_Ainv,
        epsilon_r=kernel.epsilon_r,
        max_dense_memory_gb=0.05,
    )
    vertices = {
        (mesh.M, 0),
        (0, mesh.M),
        (-mesh.M, mesh.M),
        (-mesh.M, 0),
        (0, -mesh.M),
        (mesh.M, -mesh.M),
    }
    corners = [
        index
        for index, (m, n, ell) in enumerate(mesh.labels)
        if (int(m), int(n)) in vertices and abs(int(ell)) == mesh.L
    ]
    assert len(corners) == 12
    for sector in range(4):
        for density_col in range(4):
            for source_index in corners:
                impulse = np.zeros(mesh.nk, dtype=np.complex128)
                impulse[source_index] = 0.41 + 0.67j
                actual = local_ws_fft_ordered_kernel_action(
                    impulse,
                    kernel=kernel,
                    sector=sector,
                    density_col_sector=density_col,
                )
                expected = dense.shifted_weighted_ev[sector, density_col] @ impulse
                np.testing.assert_allclose(
                    actual, expected, rtol=3e-13, atol=3e-13
                )


def test_self_cell_matches_dense_oracle(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    assert ws_kernel.self_cell_ev == dense_kernel.self_cell_ev
    for s in range(4):
        diagonal = np.diag(dense_kernel.shifted_weighted_ev[s, s])
        np.testing.assert_array_equal(
            diagonal, np.full(ws_mesh.nk, ws_kernel.self_cell_ev)
        )


def test_arbitrary_complex_full_actions_match_dense_oracle_and_are_linear(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    density = _random_complex(ws_mesh.nk, seed=17)
    other = _random_complex(ws_mesh.nk, seed=18)
    expected_h, expected_f = _dense_actions(density, dense_kernel)
    actual_h = local_ws_fft_hartree_action(density, kernel=ws_kernel)
    actual_f = local_ws_fft_fock_action(density, kernel=ws_kernel)
    np.testing.assert_allclose(actual_h, expected_h, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual_f, expected_f, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        local_ws_fft_interaction_action(density, kernel=ws_kernel),
        expected_h + expected_f,
        rtol=2e-13,
        atol=2e-13,
    )

    alpha = -0.37 + 0.29j
    beta = 0.51 - 0.13j
    combined = local_ws_fft_interaction_action(
        alpha * density + beta * other, kernel=ws_kernel
    )
    separate = alpha * local_ws_fft_interaction_action(
        density, kernel=ws_kernel
    ) + beta * local_ws_fft_interaction_action(other, kernel=ws_kernel)
    np.testing.assert_allclose(combined, separate, rtol=3e-13, atol=3e-13)


def test_fft_worker_parallelism_matches_serial_without_target_pool(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel
) -> None:
    density = _random_complex(ws_mesh.nk, seed=19)
    serial = local_ws_fft_interaction_action(
        density, kernel=ws_kernel, fft_workers=1
    )
    parallel = local_ws_fft_interaction_action(
        density, kernel=ws_kernel, fft_workers=2
    )
    np.testing.assert_allclose(parallel, serial, rtol=0.0, atol=3e-13)


def test_dagger_covariance_and_hermitian_input(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    density = _random_complex(ws_mesh.nk, seed=23)
    dagger = density.conj().swapaxes(0, 1)
    action = local_ws_fft_interaction_action(density, kernel=ws_kernel)
    action_dagger = local_ws_fft_interaction_action(dagger, kernel=ws_kernel)
    np.testing.assert_allclose(
        action_dagger,
        action.conj().swapaxes(0, 1),
        rtol=3e-13,
        atol=3e-13,
    )

    hermitian = _random_hermitian(ws_mesh.nk, seed=24)
    expected_h, expected_f = _dense_actions(hermitian, dense_kernel)
    actual = local_ws_fft_interaction_action(hermitian, kernel=ws_kernel)
    np.testing.assert_allclose(actual, expected_h + expected_f, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        actual,
        actual.conj().swapaxes(0, 1),
        rtol=0.0,
        atol=3e-13,
    )


def test_interaction_energy_and_derivative_match_dense_oracle(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel, dense_kernel
) -> None:
    density = 0.02 * _random_hermitian(ws_mesh.nk, seed=31)
    direction = _random_hermitian(ws_mesh.nk, seed=32)
    dense_h, dense_f = _dense_actions(density, dense_kernel)
    dense_sigma = dense_h + dense_f
    dense_energy = 0.5 * np.einsum(
        "abk,abk,k->", dense_sigma, density, ws_mesh.weights_Ainv3, optimize=True
    ).real
    assert local_ws_fft_interaction_energy(
        density, kernel=ws_kernel
    ) == pytest.approx(dense_energy, rel=3e-13, abs=3e-15)

    step = 2.0e-6
    finite_difference = (
        local_ws_fft_interaction_energy(density + step * direction, kernel=ws_kernel)
        - local_ws_fft_interaction_energy(density - step * direction, kernel=ws_kernel)
    ) / (2.0 * step)
    analytic = np.einsum(
        "abk,abk,k->", dense_sigma, direction, ws_mesh.weights_Ainv3, optimize=True
    ).real
    assert finite_difference == pytest.approx(analytic, rel=2e-8, abs=2e-11)


def test_epsilon_inverse_scaling_is_exact_for_all_interaction_data(
    ws_mesh: LocalWSMesh, ws_kernel: LocalWSFFTKernel
) -> None:
    doubled = build_local_ws_fft_kernel(
        ws_mesh, epsilon_r=2.0 * ws_kernel.epsilon_r
    )
    np.testing.assert_allclose(
        doubled.ordered_kernel_fft_ev,
        0.5 * ws_kernel.ordered_kernel_fft_ev,
        rtol=3e-15,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        doubled.hartree_ev, 0.5 * ws_kernel.hartree_ev, rtol=0.0, atol=0.0
    )
    assert doubled.self_cell_ev == 0.5 * ws_kernel.self_cell_ev


def test_raw_finite_temperature_normal_update_has_common_mesh_and_d0_closure() -> None:
    adapter = build_local_ws_normal_state(
        M=1,
        L=0,
        epsilon_r=17.0,
        temperature_K=80.0,
        max_fft_memory_gb=0.05,
    )
    assert isinstance(adapter, LocalWSNormalState)
    assert adapter.kernel.mesh is adapter.mesh
    np.testing.assert_array_equal(
        adapter.h0_ev,
        local_full_h0(adapter.mesh.points_Ainv, params=adapter.params),
    )
    np.testing.assert_array_equal(
        adapter.q_vectors_Ainv, physical_q_vectors(adapter.geometry)
    )
    assert np.sum(adapter.mesh.normalized_weights) == pytest.approx(1.0)
    reference_particles = np.einsum(
        "bk,k->", adapter.reference.occupations, adapter.mesh.normalized_weights
    )
    assert reference_particles == pytest.approx(1.0, abs=2e-12)

    density_delta = np.zeros_like(adapter.reference_projector_stored)
    interaction = adapter.interaction_callback(density_delta)
    np.testing.assert_array_equal(interaction, np.zeros_like(interaction))
    raw = adapter.raw_finite_temperature_update(density_delta)
    np.testing.assert_allclose(
        raw.projector_stored,
        adapter.reference_projector_stored,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        raw.projector_stored - adapter.reference_projector_stored,
        np.zeros_like(density_delta),
        rtol=0.0,
        atol=2e-15,
    )
    raw_particles = np.einsum(
        "bk,k->", raw.occupations, adapter.mesh.normalized_weights
    )
    assert raw_particles == pytest.approx(1.0, abs=2e-12)


def test_tise2_package_exports_only_stable_ws_fft_surface() -> None:
    from mean_field.systems import tise2

    assert tise2.LocalWSMesh is LocalWSMesh
    assert tise2.LocalWSFFTKernel is LocalWSFFTKernel
    assert tise2.build_local_ws_mesh is build_local_ws_mesh
    assert tise2.build_local_ws_fft_kernel is build_local_ws_fft_kernel
    assert tise2.local_ws_fft_interaction_action is local_ws_fft_interaction_action
