from __future__ import annotations

import numpy as np

from mean_field.systems.inas_gasb.axial import (
    AxialRotationSpec,
    expand_axial_radial_bundle,
)
from mean_field.systems.inas_gasb.angular_fock import (
    CoRotatingHarmonicFockOperator,
    PolarHarmonicProjectedFockOperator,
    direct_full_2d_co_rotating_fock_mode_contributions,
    direct_full_2d_co_rotating_fock_modes,
)
from mean_field.systems.inas_gasb.kane4_bundle import Kane4Bundle
from mean_field.systems.inas_gasb.projected_fock import (
    ProjectedFockOperator,
    fock_energy_density_mev_nm2,
    uniform_dielectric_green_on_mesh_mev_nm2,
)


def _minimal_radial_bundle() -> Kane4Bundle:
    nr = 2
    frame = np.zeros((2, 8, 4), dtype=np.complex128)
    frame[:, 0, 0] = 1.0
    frame[:, 1, 1] = 1.0
    frame[:, 2, 2] = 1.0
    frame[:, 5, 3] = 1.0
    h0 = np.repeat(
        np.diag([-1.2, -1.2, 0.8, 0.8])[:, :, None], nr, axis=2
    ).astype(np.complex128)
    active_tr = np.kron(
        np.eye(2), np.asarray([[0.0, -1.0], [1.0, 0.0]])
    ).astype(np.complex128)
    return Kane4Bundle(
        k_cart_nm_inv=np.array([[0.02, 0.0], [0.04, 0.0]]),
        weights_nm2=np.array([2.0e-4, 2.0e-4]),
        z_nm=np.array([0.0, 1.0]),
        z_weights_nm=np.array([0.5, 0.5]),
        h0_mev=h0,
        micro_wavefunctions=np.repeat(frame[None, :, :, :], nr, axis=0),
        time_reversal_unitary=active_tr,
        provenance={"source": "minimal-angular-fock-oracle", "radial_only": True},
    )


def _exact_frame_angular_perturbation(
    axial_bundle: Kane4Bundle, *, nphi: int, strength: float
) -> Kane4Bundle:
    phi = np.asarray(axial_bundle.micro_wavefunctions, dtype=np.complex128).copy()
    nr = axial_bundle.nk // nphi
    for ir in range(nr):
        for iphi in range(nphi):
            angle = 2.0 * np.pi * iphi / nphi
            mixing_angle = float(strength * np.cos(angle))
            cosine = np.cos(mixing_angle)
            sine = np.sin(mixing_angle)
            micro = np.eye(8, dtype=np.complex128)
            for first, second in ((0, 6), (1, 7)):
                micro[first, first] = cosine
                micro[second, second] = cosine
                micro[first, second] = -sine
                micro[second, first] = sine
            index = ir * nphi + iphi
            phi[index] = np.einsum(
                "mn,zna->zma", micro, phi[index], optimize=True
            )
    return Kane4Bundle(
        **{
            **axial_bundle.__dict__,
            "micro_wavefunctions": phi,
            "provenance": {
                **axial_bundle.provenance,
                "exact_frame_test_perturbation": {
                    "strength": float(strength),
                    "profile": "cos(theta) mixing of Gamma6/Gamma7 Kramers pairs",
                },
            },
        }
    )


def _full_operator(bundle: Kane4Bundle, *, epsilon_r: float) -> ProjectedFockOperator:
    green = uniform_dielectric_green_on_mesh_mev_nm2(
        bundle.k_cart_nm_inv,
        bundle.weights_nm2,
        bundle.z_nm,
        epsilon_r=epsilon_r,
    )
    return ProjectedFockOperator.from_bundle(
        bundle,
        green,
        self_cell_description="direct exact-frame full-2D coupled-mode oracle",
        precompute_exchange_tensor=True,
    )


def _all_axial_harmonics(
    radial: Kane4Bundle,
    target: Kane4Bundle,
    *,
    nphi: int,
    epsilon_r: float,
) -> PolarHarmonicProjectedFockOperator:
    modes = tuple(range(nphi // 2 + 1)) + tuple(
        range(-(nphi // 2 - 1), 0)
    )
    harmonics = tuple(
        CoRotatingHarmonicFockOperator.from_bundle_uniform_dielectric(
            radial,
            mode=mode,
            nphi=nphi,
            epsilon_r=epsilon_r,
        )
        for mode in modes
    )
    return PolarHarmonicProjectedFockOperator.from_harmonic_operators(
        radial,
        target,
        harmonics,
        target_nphi=nphi,
    )


def test_exact_frame_coupled_harmonic_oracle_matches_direct_full_2d() -> None:
    radial = _minimal_radial_bundle()
    nphi = 4
    epsilon_r = 15.0
    axial = expand_axial_radial_bundle(radial, nphi=nphi)
    exact = _exact_frame_angular_perturbation(axial, nphi=nphi, strength=0.23)
    exact.validate()
    full = _full_operator(exact, epsilon_r=epsilon_r)
    axial_harmonic = _all_axial_harmonics(
        radial, exact, nphi=nphi, epsilon_r=epsilon_r
    )

    rng = np.random.default_rng(811)
    raw = rng.normal(size=(4, 4, radial.nk)) + 1j * rng.normal(
        size=(4, 4, radial.nk)
    )
    mode_zero = 0.5 * (raw + np.swapaxes(raw.conj(), 0, 1))
    density_modes = np.zeros((4, 4, radial.nk, nphi), dtype=np.complex128)
    density_modes[:, :, :, 0] = mode_zero
    density_lab = axial_harmonic.reconstruct_modes(density_modes)

    direct_modes = direct_full_2d_co_rotating_fock_modes(
        full,
        exact,
        density_modes,
        nr=radial.nk,
        nphi=nphi,
    )
    contributions = direct_full_2d_co_rotating_fock_mode_contributions(
        full,
        exact,
        density_modes,
        nr=radial.nk,
        nphi=nphi,
    )
    np.testing.assert_allclose(
        np.sum(contributions, axis=4), direct_modes, rtol=2e-12, atol=2e-12
    )

    sigma_direct = full(density_lab)
    sigma_from_modes = axial_harmonic.reconstruct_modes(direct_modes)
    np.testing.assert_allclose(
        sigma_from_modes, sigma_direct, rtol=2e-12, atol=2e-12
    )
    energy_direct = fock_energy_density_mev_nm2(
        density_lab, sigma_direct, exact.weights_nm2
    )
    energy_modes = fock_energy_density_mev_nm2(
        density_lab, sigma_from_modes, exact.weights_nm2
    )
    np.testing.assert_allclose(energy_modes, energy_direct, rtol=2e-12, atol=2e-12)

    off_diagonal_coupling = np.max(np.abs(contributions[:, :, :, 1:, 0]))
    assert off_diagonal_coupling > 1e-5

    sigma_diagonal_axial = axial_harmonic(density_lab)
    assert np.max(np.abs(sigma_diagonal_axial - sigma_direct)) > 1e-5


def test_axial_frame_oracle_is_mode_diagonal_and_matches_axial_harmonics() -> None:
    radial = _minimal_radial_bundle()
    nphi = 4
    epsilon_r = 15.0
    axial = expand_axial_radial_bundle(radial, nphi=nphi)
    full = _full_operator(axial, epsilon_r=epsilon_r)
    harmonic = _all_axial_harmonics(
        radial, axial, nphi=nphi, epsilon_r=epsilon_r
    )

    rng = np.random.default_rng(823)
    raw = rng.normal(size=(4, 4, radial.nk)) + 1j * rng.normal(
        size=(4, 4, radial.nk)
    )
    mode_zero = 0.5 * (raw + np.swapaxes(raw.conj(), 0, 1))
    modes = np.zeros((4, 4, radial.nk, nphi), dtype=np.complex128)
    modes[:, :, :, 0] = mode_zero
    density = harmonic.reconstruct_modes(modes)

    contributions = direct_full_2d_co_rotating_fock_mode_contributions(
        full,
        axial,
        modes,
        nr=radial.nk,
        nphi=nphi,
    )
    assert np.max(np.abs(contributions[:, :, :, 1:, 0])) < 2e-11
    np.testing.assert_allclose(
        harmonic(density), full(density), rtol=3e-11, atol=3e-11
    )
