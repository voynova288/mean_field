from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mean_field.systems.tpt.xue_style_hf import (
    COULOMB_EV_ANGSTROM,
    TPTGammaPatch,
    TPT_SAVED_GRID_SHA256,
    TPTXueLikeParameters,
    build_tpt_xue_h0,
    build_tpt_xue_like_hf_state,
    load_tpt_full_bz_energy_rank_diagnostic,
    load_tpt_gamma_patch,
    precompute_tpt_xue_coulomb_kernel,
    rectangular_coulomb_self_cell_average_ev_angstrom2,
    tpt_xue_absolute_density_residual,
    tpt_xue_fock,
    tpt_xue_fock_at_momenta,
    tpt_xue_global_neutral_projector,
    tpt_xue_hartree,
    tpt_xue_interaction,
    tpt_xue_reference_density,
    tpt_xue_reference_relative_energy_density,
    run_tpt_xue_like_hf,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/tpt/TPT_NONINTERACTING_WANNIER_BANDS_V2.npz"


def _synthetic_patch() -> TPTGammaPatch:
    x = np.asarray([-0.1, 0.0, 0.1])
    y = np.asarray([-0.05, 0.0, 0.05])
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()])
    k2 = np.sum(points**2, axis=1)
    energies = np.vstack(
        [
            -0.25 - 0.4 * k2,
            -0.10 - 0.3 * k2,
            +0.10 + 0.2 * k2,
            +0.30 + 0.5 * k2,
        ]
    )
    weights = np.full(points.shape[0], 0.1 * 0.05 / (2.0 * np.pi) ** 2)
    patch = TPTGammaPatch(
        k_cartesian_angstrom_inv=points,
        weights_angstrom_minus2=weights,
        cell_widths_angstrom_inv=(0.1, 0.05),
        mesh_shape=(3, 3),
        active_energies_ev=energies,
        source_path=Path("synthetic.npz"),
        source_sha256="synthetic",
    )
    patch.validate()
    return patch


def test_physical_xue_kernel_prefactor_and_zero_distance_limit() -> None:
    dx, dy, epsilon = 0.04, 0.02, 10.0
    intra, inter = rectangular_coulomb_self_cell_average_ev_angstrom2(
        delta_kx_angstrom_inv=dx,
        delta_ky_angstrom_inv=dy,
        dielectric_constant=epsilon,
        electron_hole_separation_angstrom=0.0,
    )
    a, b = dx / 2.0, dy / 2.0
    integral = 4.0 * (a * np.arcsinh(b / a) + b * np.arcsinh(a / b))
    expected = 2.0 * np.pi * COULOMB_EV_ANGSTROM / epsilon * integral / (dx * dy)
    assert intra == pytest.approx(expected, rel=2e-14)
    assert inter == pytest.approx(intra, rel=2e-14)


def test_interlayer_kernel_is_exponentially_suppressed_without_q_floor() -> None:
    patch = _synthetic_patch()
    params = TPTXueLikeParameters(
        dielectric_constant=10.0, electron_hole_separation_angstrom=100.0
    )
    kernel = precompute_tpt_xue_coulomb_kernel(patch, params)
    assert kernel.self_cell_intra_ev_angstrom2 > 0.0
    assert 0.0 < kernel.self_cell_inter_ev_angstrom2 < kernel.self_cell_intra_ev_angstrom2
    q01 = np.linalg.norm(patch.k_cartesian_angstrom_inv[0] - patch.k_cartesian_angstrom_inv[1])
    assert kernel.inter_ev_angstrom2[0, 1] / kernel.intra_ev_angstrom2[0, 1] == pytest.approx(
        np.exp(-100.0 * q01)
    )


def test_periodic_kernel_wraps_reciprocal_seams_and_target_action_replays_source() -> None:
    base = _synthetic_patch()
    patch = TPTGammaPatch(
        **{**base.__dict__, "reciprocal_periods_angstrom_inv": (0.3, 0.15)}
    )
    patch.validate()
    params = TPTXueLikeParameters(
        dielectric_constant=4.0, electron_hole_separation_angstrom=20.0
    )
    kernel = precompute_tpt_xue_coulomb_kernel(patch, params)
    # x-major mesh: indices 0 and 6 are x=-0.1 and +0.1 at the same y.
    expected_q = 0.1
    assert kernel.intra_ev_angstrom2[0, 6] == pytest.approx(
        2.0 * np.pi * COULOMB_EV_ANGSTROM / (4.0 * expected_q)
    )
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(8, 8, patch.nk)) + 1j * rng.normal(size=(8, 8, patch.nk))
    density = 1.0e-3 * (raw + np.swapaxes(raw.conj(), 0, 1))
    direct = tpt_xue_fock(density, patch=patch, kernel=kernel)
    replay = tpt_xue_fock_at_momenta(
        density,
        source_patch=patch,
        params=params,
        source_kernel=kernel,
        target_k_cartesian_angstrom_inv=patch.k_cartesian_angstrom_inv,
    )
    assert np.max(np.abs(direct - replay)) < 1.0e-13


def test_patch_validation_rejects_inconsistent_declared_spacing() -> None:
    patch = _synthetic_patch()
    with pytest.raises(ValueError, match="delta_kx"):
        TPTGammaPatch(
            k_cartesian_angstrom_inv=patch.k_cartesian_angstrom_inv,
            weights_angstrom_minus2=np.full(
                patch.nk, 0.2 * 0.05 / (2.0 * np.pi) ** 2
            ),
            cell_widths_angstrom_inv=(0.2, 0.05),
            mesh_shape=patch.mesh_shape,
            active_energies_ev=patch.active_energies_ev,
            source_path=patch.source_path,
            source_sha256="synthetic",
        ).validate()


def test_reference_subtraction_and_global_occupation_return_exact_zero_for_h0() -> None:
    patch = _synthetic_patch()
    h0 = build_tpt_xue_h0(patch)
    reference = tpt_xue_reference_density(patch.nk)
    update = tpt_xue_global_neutral_projector(h0, reference_density=reference)
    assert np.max(np.abs(update.density)) == 0.0
    assert np.asarray(update.observables["occupied_rank_per_k"]).tolist() == [4] * patch.nk
    projector = np.asarray(update.observables["raw_projector_stored"])
    for ik in range(patch.nk):
        assert np.max(np.abs(projector[:, :, ik] @ projector[:, :, ik] - projector[:, :, ik])) < 1e-14


def test_global_filling_can_transfer_one_state_between_k_points() -> None:
    patch = _synthetic_patch()
    h = build_tpt_xue_h0(patch)
    h[2, 2, 0] = -1.0
    h[0, 0, 1] = +1.0
    update = tpt_xue_global_neutral_projector(
        h, reference_density=tpt_xue_reference_density(patch.nk)
    )
    ranks = np.asarray(update.observables["occupied_rank_per_k"])
    assert ranks[0] == 5
    assert ranks[1] == 3
    assert int(ranks.sum()) == 4 * patch.nk


def test_complex_projector_is_converted_to_core_stored_orientation() -> None:
    patch = _synthetic_patch()
    h = build_tpt_xue_h0(patch)
    h[0, 2, :] = 0.03j
    h[2, 0, :] = -0.03j
    update = tpt_xue_global_neutral_projector(
        h, reference_density=tpt_xue_reference_density(patch.nk)
    )
    values, vectors = np.linalg.eigh(h[:, :, 0])
    projector_ket = vectors[:, :4] @ vectors[:, :4].conj().T
    stored = np.asarray(update.observables["raw_projector_stored"])
    assert stored[0, 2, 0] == pytest.approx(projector_ket[2, 0])
    assert abs(stored[0, 2, 0] - projector_ket[0, 2]) > 1e-4


def test_fock_uses_stored_density_transpose_for_complex_coherence() -> None:
    patch = _synthetic_patch()
    params = TPTXueLikeParameters(
        dielectric_constant=10.0, electron_hole_separation_angstrom=30.0
    )
    kernel = precompute_tpt_xue_coulomb_kernel(patch, params)
    density = np.zeros((8, 8, patch.nk), dtype=np.complex128)
    z = 0.2 + 0.35j
    density[0, 2, :] = z
    density[2, 0, :] = z.conjugate()
    sigma = tpt_xue_fock(density, patch=patch, kernel=kernel)
    expected = -(kernel.inter_ev_angstrom2 @ (patch.weights_angstrom_minus2 * z.conjugate()))
    assert np.max(np.abs(sigma[0, 2] - expected)) < 1e-13
    assert np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))) < 1e-13


def test_hartree_capacitor_sign_and_neutrality_gate() -> None:
    patch = _synthetic_patch()
    params = TPTXueLikeParameters(
        dielectric_constant=10.0, electron_hole_separation_angstrom=100.0
    )
    density = np.zeros((8, 8, patch.nk), dtype=np.complex128)
    density[2, 2, :] = 0.25
    density[0, 0, :] = -0.25
    sigma = tpt_xue_hartree(density, patch=patch, params=params)
    assert np.all(sigma[2, 2].real > 0.0)
    assert np.all(sigma[0, 0].real < 0.0)
    assert sigma[2, 2, 0] == pytest.approx(-sigma[0, 0, 0])
    density[0, 0, :] = 0.0
    with pytest.raises(ValueError, match="neutrality"):
        tpt_xue_hartree(density, patch=patch, params=params)


def test_interaction_is_energy_functional_derivative_for_neutral_hermitian_chord() -> None:
    patch = _synthetic_patch()
    params = TPTXueLikeParameters(
        dielectric_constant=10.0, electron_hole_separation_angstrom=20.0
    )
    kernel = precompute_tpt_xue_coulomb_kernel(patch, params)
    rng = np.random.default_rng(7)

    def neutral_offdiagonal() -> np.ndarray:
        raw = rng.normal(size=(8, 8, patch.nk)) + 1j * rng.normal(size=(8, 8, patch.nk))
        return 0.02 * (raw + np.swapaxes(raw.conj(), 0, 1))

    density = neutral_offdiagonal()
    direction = neutral_offdiagonal()
    for array in (density, direction):
        for index in range(8):
            array[index, index, :] = 0.0
    density[2, 2, :] = 0.03
    density[0, 0, :] = -0.03
    direction[3, 3, :] = -0.02
    direction[1, 1, :] = 0.02

    def interaction_energy(test_density: np.ndarray) -> float:
        sigma = tpt_xue_interaction(test_density, patch=patch, params=params, kernel=kernel)
        zero_h0 = np.zeros_like(test_density)
        return tpt_xue_reference_relative_energy_density(
            sigma, zero_h0, test_density, patch=patch
        )

    step = 1.0e-6
    finite_difference = (interaction_energy(density + step * direction) - interaction_energy(density - step * direction)) / (2.0 * step)
    sigma = tpt_xue_interaction(density, patch=patch, params=params, kernel=kernel)
    analytic = np.einsum(
        "abk,abk,k->", sigma, direction, patch.weights_angstrom_minus2, optimize=True
    ).real
    assert finite_difference == pytest.approx(analytic, rel=2e-8, abs=2e-11)


def test_absolute_density_residual_is_well_defined_at_normal_reference() -> None:
    patch = _synthetic_patch()
    zero = np.zeros((8, 8, patch.nk), dtype=np.complex128)
    delta = zero.copy()
    delta[0, 2, :] = 3.0e-9
    delta[2, 0, :] = 3.0e-9
    assert tpt_xue_absolute_density_residual(zero, zero) == 0.0
    assert tpt_xue_absolute_density_residual(delta, zero) == pytest.approx(
        np.sqrt(2.0) * 3.0e-9
    )


def test_runner_rejects_nonvariational_clipped_oda_and_exposes_diagnostic_lineage() -> None:
    patch = _synthetic_patch()
    state = build_tpt_xue_like_hf_state(
        patch,
        dielectric_constant=10.0,
        electron_hole_separation_angstrom=100.0,
        accept_provisional_scalar_spin_duplication=True,
        accept_energy_sector_layer_proxy=True,
        precision=1e-20,
    )
    with pytest.raises(ValueError, match="constrained line search"):
        run_tpt_xue_like_hf(state, init_mode="normal", max_iter=1, max_oda_lambda=0.5)
    diagnostic = run_tpt_xue_like_hf(
        state,
        init_mode="exciton_spin_conserving",
        max_iter=1,
        allow_unconverged_diagnostic=True,
    )
    assert not diagnostic.run.converged
    assert diagnostic.density_delta_stored.shape == (8, 8, patch.nk)
    assert diagnostic.raw_density_delta_stored.shape == diagnostic.density_delta_stored.shape
    assert diagnostic.final_raw_residual > 0.0


def test_full_step_discriminator_records_only_unit_mixing() -> None:
    patch = _synthetic_patch()
    state = build_tpt_xue_like_hf_state(
        patch,
        dielectric_constant=10.0,
        electron_hole_separation_angstrom=100.0,
        accept_provisional_scalar_spin_duplication=True,
        accept_energy_sector_layer_proxy=True,
        precision=1e-20,
    )
    diagnostic = run_tpt_xue_like_hf(
        state,
        init_mode="exciton_spin_conserving",
        max_iter=2,
        force_full_step=True,
        allow_unconverged_diagnostic=True,
    )
    assert diagnostic.run.iter_oda.tolist() == [1.0, 1.0]


def test_effective_model_scope_requires_explicit_acknowledgements() -> None:
    patch = _synthetic_patch()
    with pytest.raises(ValueError, match="provisional"):
        build_tpt_xue_like_hf_state(
            patch,
            dielectric_constant=10.0,
            electron_hole_separation_angstrom=100.0,
            accept_provisional_scalar_spin_duplication=False,
            accept_energy_sector_layer_proxy=True,
        )
    with pytest.raises(ValueError, match="model proxy"):
        build_tpt_xue_like_hf_state(
            patch,
            dielectric_constant=10.0,
            electron_hole_separation_angstrom=100.0,
            accept_provisional_scalar_spin_duplication=True,
            accept_energy_sector_layer_proxy=False,
        )


def test_full_bz_energy_rank_diagnostic_is_periodic_and_explicitly_unqualified() -> None:
    patch = load_tpt_full_bz_energy_rank_diagnostic(SOURCE)
    assert patch.mesh_shape == (51, 51)
    assert patch.nk == 2601
    assert patch.reciprocal_periods_angstrom_inv is not None
    assert patch.quadrature_policy == "uniform_periodic_full_bz_raw_energy_ranks"
    assert patch.projector_authority == "user_requested_unsewn_global_energy_ranks_not_material_projector"
    assert np.sum(patch.weights_angstrom_minus2) == pytest.approx(
        1.0 / (3.7037999629999998 * 18.5991001129000004)
    )


def test_authoritative_saved_grid_extracts_exact_qualified_gamma_patch() -> None:
    patch = load_tpt_gamma_patch(SOURCE)
    assert patch.source_sha256 == TPT_SAVED_GRID_SHA256
    assert patch.source_hr_sha256 is not None
    assert patch.projector_audit_sha256 is not None
    assert patch.projector_authority == "raw_energy_indices_qualified_only_in_gamma_patch"
    assert patch.mesh_shape == (9, 15)
    assert patch.nk == 135
    assert np.min(np.linalg.norm(patch.k_cartesian_angstrom_inv, axis=1)) < 1e-15
    gamma = int(np.argmin(np.linalg.norm(patch.k_cartesian_angstrom_inv, axis=1)))
    assert patch.active_energies_ev[:, gamma] == pytest.approx(
        [-0.69324, -0.50051, -0.40537, -0.25105], abs=6e-6
    )
