from __future__ import annotations

import numpy as np

from mean_field.devtools.plot_rlg_hbn_paper_hf_bands import _paper_hf_path
from mean_field.core.hf import conventional_projector_to_stored, stored_projector_to_conventional
from mean_field.systems.RnG_hBN import (
    InterlayerHartreeResult,
    LayerChargeResult,
    RLGhBNInteractionParams,
    RLGhBNModel,
    ScreenedInterlayerPotentialResult,
    ScreeningIteration,
    build_hamiltonian,
    build_rlg_hbn_projected_basis,
    diagonalize_hamiltonian,
    initialize_rlg_hbn_density,
    normalize_rlg_hbn_init_mode,
    rlg_hbn_projector_from_density,
    rlg_hbn_projector_idempotency_residual,
    rlg_hbn_reference_density,
)


def test_fig6_paper_path_uses_canonical_kprime_representative() -> None:
    model = RLGhBNModel.from_config(layer_count=5, xi=1, theta_deg=0.77, shell_count=1)

    path = _paper_hf_path(model, points_per_segment=6)
    node_kvec = np.asarray(path.kvec, dtype=np.complex128)[np.asarray(path.node_indices, dtype=int) - 1]

    assert np.isclose(node_kvec[1], model.lattice.k_m)
    assert np.isclose(node_kvec[2], model.lattice.kprime_m)
    assert not np.isclose(node_kvec[2], model.lattice.k_m)
    assert np.isclose(node_kvec[4], model.lattice.g_m2 / 2.0)
    assert np.isclose(node_kvec[5], model.lattice.m_m)


def _fake_screening(model: RLGhBNModel, *, screened_u_mev: float) -> ScreenedInterlayerPotentialResult:
    layer_count = model.params.layer_count
    charge = LayerChargeResult(
        layer_charge=np.zeros(layer_count, dtype=float),
        reference_layer_charge=np.zeros(layer_count, dtype=float),
        delta_layer_charge=np.zeros(layer_count, dtype=float),
        mesh_size=1,
        n_spin=2,
        valleys=(1, -1),
        n_valence_bands=0,
    )
    hartree = InterlayerHartreeResult(
        layer_potential_mev=np.zeros(layer_count, dtype=float),
        interlayer_slope_mev=float(screened_u_mev - model.params.displacement_field_mev),
        delta_layer_charge=np.zeros(layer_count, dtype=float),
        moire_cell_area_nm2=1.0,
    )
    return ScreenedInterlayerPotentialResult(
        external_v_mev=float(model.params.displacement_field_mev),
        screened_u_mev=float(screened_u_mev),
        converged=True,
        iterations=(
            ScreeningIteration(
                iteration=0,
                screened_u_mev=float(screened_u_mev),
                interlayer_hartree_mev=float(hartree.interlayer_slope_mev),
                candidate_u_mev=float(screened_u_mev),
                residual_mev=0.0,
            ),
        ),
        layer_charge=charge,
        hartree=hartree,
        residual_mev=0.0,
        method="test_fixture",
        mesh_size=1,
    )


def _screened_diagonal_h0(basis_data) -> np.ndarray:
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


def test_fig6_screened_basis_projects_physical_hamiltonian_not_screened_energies() -> None:
    physical_model = RLGhBNModel.from_config(
        layer_count=3,
        xi=0,
        theta_deg=0.77,
        displacement_field_mev=64.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        scheme="cn",
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        use_screened_basis=True,
    )
    screened_u_mev = 21.0

    basis_data = build_rlg_hbn_projected_basis(
        physical_model,
        interaction,
        mesh_size=1,
        screening_result=_fake_screening(physical_model, screened_u_mev=screened_u_mev),
    )

    assert basis_data.physical_h0 is not None
    assert np.isclose(basis_data.basis_model.params.displacement_field_mev, screened_u_mev)
    assert not np.allclose(basis_data.physical_h0, _screened_diagonal_h0(basis_data))

    k_value = complex(basis_data.kvec[0])
    valley = int(basis_data.valleys[0])
    _evals, evecs = diagonalize_hamiltonian(
        k_value,
        basis_data.basis_model.lattice,
        basis_data.basis_model.params,
        valley=valley,
    )
    selected = evecs[:, np.asarray(basis_data.active_band_indices, dtype=int)]
    expected = selected.conjugate().T @ build_hamiltonian(
        k_value,
        physical_model.lattice,
        physical_model.params,
        valley=valley,
    ) @ selected
    expected = 0.5 * (expected + expected.conjugate().T)

    idx = np.arange(basis_data.nt, dtype=int).reshape(
        (basis_data.basis.n_spin, basis_data.basis.n_flavor, basis_data.basis.n_band),
        order="F",
    )
    block_indices = np.asarray(idx[0, 0, :], dtype=int)
    assert np.allclose(
        basis_data.physical_h0[:, :, 0][np.ix_(block_indices, block_indices)],
        expected,
        atol=1.0e-10,
    )


def test_fig6_average_scheme_h0_includes_fixed_remote_hamiltonian() -> None:
    physical_model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=64.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        scheme="average",
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        use_screened_basis=True,
    )

    basis_data = build_rlg_hbn_projected_basis(
        physical_model,
        interaction,
        mesh_size=1,
        screening_result=_fake_screening(physical_model, screened_u_mev=28.0),
    )

    assert basis_data.physical_h0 is not None
    assert basis_data.fixed_remote_hamiltonian is not None
    assert np.linalg.norm(basis_data.fixed_remote_hamiltonian) > 0.0
    assert np.allclose(
        basis_data.h0,
        basis_data.physical_h0 + basis_data.fixed_remote_hamiltonian,
        atol=1.0e-10,
    )
    assert not np.allclose(basis_data.h0, basis_data.physical_h0)


def test_rlg_hbn_density_archive_convention_recovers_ket_projector_with_transpose() -> None:
    occupied = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    ket_projector = occupied[:, None] @ occupied.conjugate()[None, :]
    stored_projector = conventional_projector_to_stored(ket_projector)
    reference = rlg_hbn_reference_density(2, 1, scheme="average", active_valence_bands=0, n_spin=1, n_eta=1)
    density_delta = stored_projector[:, :, None] - reference

    archived_stored_projector = rlg_hbn_projector_from_density(density_delta, reference)[:, :, 0]
    recovered_ket_projector = stored_projector_to_conventional(archived_stored_projector)

    np.testing.assert_allclose(archived_stored_projector, stored_projector, atol=1.0e-12)
    np.testing.assert_allclose(recovered_ket_projector, ket_projector, atol=1.0e-12)
    assert rlg_hbn_projector_idempotency_residual(density_delta, reference) < 1.0e-12


def test_fig6_random_init_mode_alias_builds_valid_projector() -> None:
    h0 = np.zeros((8, 8, 1), dtype=np.complex128)
    reference = rlg_hbn_reference_density(8, 1, scheme="average", active_valence_bands=1)

    density = initialize_rlg_hbn_density(
        h0,
        nu=1.0,
        reference_density=reference,
        active_valence_bands=1,
        init_mode="diag_random",
        seed=4,
        n_spin=2,
        n_eta=2,
        n_band=2,
    )

    assert normalize_rlg_hbn_init_mode("random") == "random"
    assert density.shape == h0.shape
    assert rlg_hbn_projector_idempotency_residual(density, reference) < 1.0e-10
