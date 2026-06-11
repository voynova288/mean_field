from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import ProjectedWavefunctionBasis, calculate_projected_overlap_between
from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
    average_scheme_density_delta,
    build_rlg_hbn_density_from_hamiltonian,
    build_kpath_from_nodes,
    build_rlg_hbn_interaction_components,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    build_rlg_hbn_projected_basis_for_kvec,
    evaluate_rlg_hbn_hf_path,
    calculate_layer_projected_overlap_between,
    diagonal_layer_overlap_blocks,
    interaction_shifts_for_cutoff,
    rlg_hbn_filling_from_density,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
    rlg_hbn_occupied_bands_per_k,
    rlg_hbn_projector_idempotency_residual,
    rlg_hbn_average_reference_density,
    rlg_hbn_projected_micro_sewing_transforms,
    rlg_hbn_reference_density,
    run_rlg_hbn_hartree_fock,
)
from mean_field.systems.RnG_hBN.hf import rlg_hbn_layer_component_groups


def _small_model() -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )


def _small_interaction(**overrides: object) -> RLGhBNInteractionParams:
    values = {
        "active_valence_bands": 1,
        "active_conduction_bands": 1,
        "k_mesh_size": 1,
        "interaction_cutoff_q1": 1.0,
        "use_screened_basis": False,
    }
    values.update(overrides)
    return RLGhBNInteractionParams(**values)


def _wavefunction_grid(basis_data):
    return basis_data.basis.wavefunctions.reshape(
        (
            basis_data.basis.local_basis_size,
            *basis_data.reciprocal_grid_shape,
            basis_data.basis.n_band,
            basis_data.basis.n_flavor,
            basis_data.basis.nk,
        ),
        order="F",
    )


def _screened_energy_diagonal_h0(basis_data) -> np.ndarray:
    diagonal_h0 = np.zeros_like(basis_data.h0)
    n_band = basis_data.basis.n_band
    idx = np.arange(basis_data.nt, dtype=int).reshape(
        (basis_data.basis.n_spin, basis_data.basis.n_flavor, n_band),
        order="F",
    )
    for ik in range(basis_data.nk):
        for ispin in range(basis_data.basis.n_spin):
            for iflavor in range(basis_data.basis.n_flavor):
                block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                diagonal_h0[:, :, ik][np.ix_(block_indices, block_indices)] = np.diag(
                    basis_data.band_energies[:, iflavor, ik]
                )
    return diagonal_h0


def _grid_position(basis_data, pair: tuple[int, int]) -> tuple[int, int]:
    ix = int(pair[0] - basis_data.reciprocal_grid_origin[0])
    iy = int(pair[1] - basis_data.reciprocal_grid_origin[1])
    assert 0 <= ix < basis_data.reciprocal_grid_shape[0]
    assert 0 <= iy < basis_data.reciprocal_grid_shape[1]
    return ix, iy


def test_rlg_hbn_projected_micro_sewing_transform_relabels_valley_blocks() -> None:
    local_basis_size = 1
    grid_shape = (3, 2)
    block_dim = local_basis_size * grid_shape[0] * grid_shape[1]
    sew_x, _ = rlg_hbn_projected_micro_sewing_transforms(
        local_basis_size=local_basis_size,
        grid_shape=grid_shape,
        spin_count=1,
        valley_signs=(1, -1),
    )

    vector = np.zeros(2 * block_dim, dtype=np.complex128)
    k_block = np.zeros((local_basis_size, *grid_shape), dtype=np.complex128)
    kprime_block = np.zeros_like(k_block)
    k_block[0, 1, 0] = 2.0
    kprime_block[0, 1, 1] = 3.0
    vector[:block_dim] = k_block.reshape(-1, order="F")
    vector[block_dim:] = kprime_block.reshape(-1, order="F")

    shifted = sew_x(vector)
    shifted_k = shifted[:block_dim].reshape((local_basis_size, *grid_shape), order="F")
    shifted_kprime = shifted[block_dim:].reshape((local_basis_size, *grid_shape), order="F")

    assert shifted_k[0, 0, 0] == 2.0
    assert shifted_kprime[0, 2, 1] == 3.0
    assert np.count_nonzero(shifted) == 2


def _manual_zero_fill_overlap_for_raw_shift(grid: np.ndarray, raw_shift: tuple[int, int]) -> complex:
    shifted = np.zeros_like(grid)
    _, nx, ny, _ = grid.shape
    dm, dn = raw_shift
    for ix in range(nx):
        src_ix = ix + int(dm)
        if src_ix < 0 or src_ix >= nx:
            continue
        for iy in range(ny):
            src_iy = iy + int(dn)
            if src_iy < 0 or src_iy >= ny:
                continue
            shifted[:, ix, iy, :] = grid[:, src_ix, src_iy, :]
    return complex(np.sum(np.conj(grid) * shifted))


def test_rlg_hbn_layer_form_factor_uses_valley_signed_physical_g_shift() -> None:
    local_basis_size = 2
    grid_shape = (4, 3)
    n_band = 1
    n_flavor = 2
    nk = 1
    grid = np.zeros((local_basis_size, *grid_shape, n_band, n_flavor, nk), dtype=np.complex128)
    for iflavor in range(n_flavor):
        for ix in range(grid_shape[0]):
            for iy in range(grid_shape[1]):
                for local in range(local_basis_size):
                    grid[local, ix, iy, 0, iflavor, 0] = (
                        1.0 + 0.2 * local + 0.3 * ix - 0.4j * iy + (0.7 + 0.5j) * iflavor
                    )
    basis = ProjectedWavefunctionBasis(
        wavefunctions=grid.reshape(local_basis_size * grid_shape[0] * grid_shape[1], n_band, n_flavor, nk, order="F"),
        grid_shape=grid_shape,
        n_spin=1,
        local_basis_size=local_basis_size,
        name="valley-signed-overlap-fixture",
    )

    overlap = calculate_layer_projected_overlap_between(
        basis,
        basis,
        1,
        0,
        layer_count=1,
        valleys=(1, -1),
    )

    # The helper's physical shift is G=(1,0). Eq. (18) and the TR-relabelled
    # K' basis imply raw shifts s=-eta*G: K uses (-1,0), K' uses (+1,0).
    k_grid = grid[:, :, :, :, 0, :].reshape(local_basis_size, *grid_shape, n_band * nk, order="F")
    kp_grid = grid[:, :, :, :, 1, :].reshape(local_basis_size, *grid_shape, n_band * nk, order="F")
    expected_k = _manual_zero_fill_overlap_for_raw_shift(k_grid, (-1, 0))
    expected_kprime = _manual_zero_fill_overlap_for_raw_shift(kp_grid, (1, 0))

    np.testing.assert_allclose(overlap[0, 0, 0, 0, 0], expected_k, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(overlap[0, 1, 0, 1, 0], expected_kprime, rtol=1.0e-13, atol=1.0e-13)
    assert overlap[0, 0, 0, 1, 0] == 0.0
    assert overlap[0, 1, 0, 0, 0] == 0.0


def test_rlg_hbn_projected_basis_uses_periodic_gauge_relabel_for_both_valleys() -> None:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=2,
    )
    interaction = _small_interaction()
    lattice = model.lattice
    k0 = 0.13 * lattice.g_m1 + 0.17 * lattice.g_m2

    basis_data = build_rlg_hbn_projected_basis_for_kvec(
        model,
        interaction,
        np.asarray([k0, k0 + lattice.g_m1], dtype=np.complex128),
        valleys=(1, -1),
    )

    grid = _wavefunction_grid(basis_data)

    assert np.allclose(basis_data.kvec[0], k0, atol=1.0e-12)
    assert np.allclose(basis_data.kvec[1], k0 + lattice.g_m1, atol=1.0e-12)
    assert np.allclose(basis_data.band_energies[:, :, 0], basis_data.band_energies[:, :, 1], atol=1.0e-12)
    assert np.allclose(basis_data.h0[:, :, 0], basis_data.h0[:, :, 1], atol=1.0e-12)

    for iflavor, valley in enumerate(basis_data.valleys):
        for pair in np.asarray(lattice.g_indices, dtype=int):
            can_pair = (int(pair[0]), int(pair[1]))
            raw_pair = (int(pair[0] - int(valley)), int(pair[1]))
            can_ix, can_iy = _grid_position(basis_data, can_pair)
            raw_ix, raw_iy = _grid_position(basis_data, raw_pair)
            assert np.allclose(
                grid[:, raw_ix, raw_iy, :, iflavor, 1],
                grid[:, can_ix, can_iy, :, iflavor, 0],
                atol=1.0e-12,
            )

    for ik in range(basis_data.nk):
        for iflavor in range(basis_data.basis.n_flavor):
            flattened = grid[:, :, :, :, iflavor, ik].reshape(-1, basis_data.basis.n_band, order="F")
            assert np.allclose(
                flattened.conjugate().T @ flattened,
                np.eye(basis_data.basis.n_band),
                atol=1.0e-12,
            )


def test_rlg_hbn_projected_basis_uses_screened_interlayer_potential() -> None:
    model = _small_model()
    interaction = _small_interaction(use_screened_basis=True)

    basis_data = build_rlg_hbn_projected_basis(
        model,
        interaction,
        mesh_size=1,
        screening_mesh_size=1,
        screening_max_iter=20,
        screening_tolerance_mev=1.0e-5,
    )

    assert basis_data.screening is not None
    assert basis_data.screening.converged
    assert not np.isclose(basis_data.screened_u_mev, model.params.displacement_field_mev)
    assert np.isclose(basis_data.basis_model.params.displacement_field_mev, basis_data.screening.screened_u_mev)
    assert basis_data.active_band_indices == basis_data.flat_band_indices
    assert basis_data.basis.n_spin == 2
    assert basis_data.basis.n_flavor == 2
    assert basis_data.basis.n_band == 2
    assert basis_data.h0.shape == (8, 8, 1)
    assert basis_data.band_energies.shape == (2, 2, 1)


def test_rlg_hbn_screened_basis_projects_physical_external_field_h0() -> None:
    model = _small_model()
    interaction = _small_interaction(use_screened_basis=True, scheme="cn")

    basis_data = build_rlg_hbn_projected_basis(
        model,
        interaction,
        mesh_size=1,
        screening_mesh_size=1,
        screening_max_iter=12,
        screening_tolerance_mev=1.0e-5,
    )

    assert basis_data.screening is not None
    assert not np.isclose(basis_data.screened_u_mev, model.params.displacement_field_mev)
    assert basis_data.physical_h0 is not None
    assert basis_data.fixed_remote_hamiltonian is not None
    assert np.allclose(basis_data.h0, basis_data.physical_h0)
    assert np.allclose(basis_data.fixed_remote_hamiltonian, 0.0)
    assert not np.allclose(basis_data.h0, _screened_energy_diagonal_h0(basis_data))


def test_rlg_hbn_average_scheme_adds_fixed_remote_band_potential() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)

    assert basis_data.physical_h0 is not None
    assert basis_data.fixed_remote_hamiltonian is not None
    assert np.linalg.norm(basis_data.fixed_remote_hamiltonian) > 0.0
    assert np.allclose(basis_data.h0, basis_data.physical_h0 + basis_data.fixed_remote_hamiltonian)
    assert not np.allclose(basis_data.h0, basis_data.physical_h0)


def test_rlg_hbn_projected_basis_can_skip_screening_and_validates_active_window() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)

    assert basis_data.screening is None
    assert np.isclose(basis_data.screened_u_mev, 24.0)

    with pytest.raises(ValueError, match="Active RLG/hBN band window"):
        build_rlg_hbn_projected_basis(
            _small_model(),
            RLGhBNInteractionParams(use_screened_basis=False),
            mesh_size=1,
        )



def test_rlg_hbn_layer_component_groups_match_local_basis_convention() -> None:
    groups = rlg_hbn_layer_component_groups(3)
    assert tuple(group.name for group in groups) == ("layer_0", "layer_1", "layer_2")
    assert [group.indices.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5]]

    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(k_mesh_size=1), mesh_size=1)
    assert tuple(group.name for group in basis_data.basis.component_groups) == ("layer_0", "layer_1", "layer_2")
    assert [group.indices.tolist() for group in basis_data.basis.component_groups] == [[0, 1], [2, 3], [4, 5]]


def test_rlg_hbn_layer_form_factors_sum_to_total_charge_overlap() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(k_mesh_size=2), mesh_size=2)

    layer_overlap = calculate_layer_projected_overlap_between(
        basis_data.basis,
        basis_data.basis,
        0,
        0,
        layer_count=basis_data.basis_model.params.layer_count,
    )
    total_overlap = calculate_projected_overlap_between(basis_data.basis, basis_data.basis, 0, 0)
    diagonal = diagonal_layer_overlap_blocks(layer_overlap)

    assert layer_overlap.shape == (3, basis_data.nt, basis_data.nk, basis_data.nt, basis_data.nk)
    assert diagonal.shape == (3, basis_data.nt, basis_data.nt, basis_data.nk)
    assert np.allclose(np.sum(layer_overlap, axis=0), total_overlap, atol=1.0e-10)
    for ik in range(basis_data.nk):
        assert np.allclose(np.sum(diagonal[:, :, :, ik], axis=0), np.eye(basis_data.nt), atol=1.0e-10)


def test_rlg_hbn_layer_overlap_blocks_include_layer_coulomb_tables() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(k_mesh_size=2), mesh_size=2)
    shifts = interaction_shifts_for_cutoff(basis_data.basis_model.lattice, basis_data.interaction)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))

    assert shifts == ((0, 0),)
    assert blocks.shifts == ((0, 0),)
    assert blocks.gvecs.shape == (1,)
    assert blocks.layer_diagonal_overlaps[(0, 0)].shape == (3, basis_data.nt, basis_data.nt, basis_data.nk)
    assert blocks.hartree_layer_coulomb[(0, 0)].shape == (3, 3)
    assert blocks.fock_layer_coulomb[(0, 0)].shape == (basis_data.nk, basis_data.nk, 3, 3)


def test_rlg_hbn_average_density_delta_feeds_layer_interaction_components() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(k_mesh_size=2), mesh_size=2)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))
    occupation_density = np.zeros((basis_data.nt, basis_data.nt, basis_data.nk), dtype=np.complex128)
    occupation_density[0, 0, :] = 1.0
    density_delta = average_scheme_density_delta(occupation_density)
    reference = rlg_hbn_average_reference_density(basis_data.nt, basis_data.nk)

    components = build_rlg_hbn_interaction_components(density_delta, blocks, v0=basis_data.v0)

    assert np.allclose(reference[:, :, 0], 0.5 * np.eye(basis_data.nt))
    assert np.allclose(density_delta, occupation_density - reference)
    assert components.hartree.shape == density_delta.shape
    assert components.fock.shape == density_delta.shape
    assert np.allclose(components.total, components.hartree + components.fock)
    for ik in range(basis_data.nk):
        assert np.max(np.abs(components.total[:, :, ik] - components.total[:, :, ik].conjugate().T)) < 1.0e-8


def test_rlg_hbn_reference_density_distinguishes_average_and_cn_schemes() -> None:
    nt = 8
    nk = 2
    average = rlg_hbn_reference_density(nt, nk, scheme="average", active_valence_bands=1)
    cn = rlg_hbn_reference_density(nt, nk, scheme="cn", active_valence_bands=1)

    assert np.allclose(np.diag(average[:, :, 0]), 0.5)
    assert np.isclose(np.trace(cn[:, :, 0]).real, 4.0)
    assert set(np.unique(np.diag(cn[:, :, 0]).real)) == {0.0, 1.0}


def test_rlg_hbn_density_builder_tracks_nu_relative_to_active_valence() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)
    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=1.0,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    reference = rlg_hbn_reference_density(
        basis_data.nt,
        basis_data.nk,
        scheme=basis_data.interaction.scheme,
        active_valence_bands=basis_data.interaction.active_valence_bands,
    )

    density, energies, mu, occ_mask = build_rlg_hbn_density_from_hamiltonian(
        basis_data.h0,
        nu=1.0,
        reference_density=reference,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        occupation_counts=counts,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )

    assert counts == (2, 1, 1, 1)
    assert rlg_hbn_occupied_bands_per_k(
        1.0,
        basis_data.nt,
        active_valence_bands=basis_data.interaction.active_valence_bands,
    ) == 5
    assert density.shape == basis_data.h0.shape
    assert energies.shape == (basis_data.nt, basis_data.nk)
    assert np.isfinite(mu)
    assert int(np.sum(occ_mask)) == 5
    assert np.isclose(
        rlg_hbn_filling_from_density(
            density,
            reference,
            active_valence_bands=basis_data.interaction.active_valence_bands,
        ),
        1.0,
    )
    assert rlg_hbn_projector_idempotency_residual(density, reference) < 1.0e-10



def test_rlg_hbn_projected_hf_smoke_run_produces_hermitian_state() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))

    run = run_rlg_hbn_hartree_fock(
        basis_data,
        overlap_blocks=blocks,
        nu=1.0,
        init_mode="flavor",
        seed=1,
        max_iter=2,
        precision=1.0e-12,
    )

    assert run.state.density.shape == basis_data.h0.shape
    assert run.iterations >= 1
    assert run.state.diagnostics["projector_idempotency_residual"] < 1.0e-8
    assert run.state.diagnostics["density_hermitian_residual"] < 1.0e-8
    assert run.state.diagnostics["hamiltonian_hermitian_residual"] < 1.0e-8
    assert np.isclose(run.state.diagnostics["filling"], 1.0)


def test_rlg_hbn_hf_path_evaluator_smoke() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))
    run = run_rlg_hbn_hartree_fock(
        basis_data,
        overlap_blocks=blocks,
        nu=1.0,
        init_mode="flavor",
        seed=1,
        max_iter=1,
        precision=1.0e-12,
    )
    lattice = basis_data.basis_model.lattice
    path = build_kpath_from_nodes(
        (lattice.gamma_m, lattice.k_m, lattice.gamma_m),
        ("Gamma", "K", "Gamma"),
        (1, 1),
    )

    result = evaluate_rlg_hbn_hf_path(run, path, chunk_size=1)

    assert result.hamiltonian.shape == (basis_data.nt, basis_data.nt, 3)
    assert result.energies.shape == (basis_data.nt, 3)
    assert result.basis_data.physical_h0 is not None
    assert result.basis_data.fixed_remote_hamiltonian is not None
    assert np.linalg.norm(result.basis_data.fixed_remote_hamiltonian) > 0.0
    assert np.allclose(
        result.basis_data.h0,
        result.basis_data.physical_h0 + result.basis_data.fixed_remote_hamiltonian,
    )
    for ik in range(result.hamiltonian.shape[2]):
        assert np.allclose(result.hamiltonian[:, :, ik], result.hamiltonian[:, :, ik].conjugate().T)
    assert np.all(np.isfinite(result.energies))
