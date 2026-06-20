from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.htqg import (
    HTQGModel,
    HTQGParams,
    build_commensurate_geometry,
    build_hamiltonian,
    commensurate_twist_angles_deg,
    build_htqg_lattice,
    canonical_domain_key,
    diagonalize_hamiltonian,
    domain_displacements,
    fujimoto_2025_fig2_checkpoint,
)
from mean_field.systems.htqg.validation import run_lightweight_validation


def _light_params() -> HTQGParams:
    return HTQGParams.default(kappa=0.6, lambda_mdt_nm=0.0, include_dirac_rotation=False)


def test_htqg_commensurate_geometry_matches_fujimoto_fig2_checkpoint() -> None:
    geometry = build_commensurate_geometry(8, 7, 8, 8)
    np.testing.assert_allclose(geometry.twist_angles_deg, (2.13, 2.27, 2.13), atol=5.0e-3)
    assert tuple(round(value, 2) for value in geometry.twist_angles_deg) == (2.13, 2.27, 2.13)
    assert fujimoto_2025_fig2_checkpoint()


def test_htqg_commensurate_geometry_public_contract() -> None:
    geometry = build_commensurate_geometry(8, 7, 8, 8)

    assert geometry.integers == (8, 7, 8, 8)
    assert geometry.twist_angles_deg == (geometry.theta12_deg, geometry.theta23_deg, geometry.theta34_deg)
    assert geometry.twist_angles_rad == (geometry.theta12_rad, geometry.theta23_rad, geometry.theta34_rad)
    np.testing.assert_allclose(geometry.supermoire_period_factor_12, 13.0, atol=1.0e-12)
    np.testing.assert_allclose(geometry.supermoire_period_factor_23, np.sqrt(192.0), atol=1.0e-12)
    assert geometry.to_dict()["n12"] == 8
    assert geometry.to_dict()["theta23_deg"] == pytest.approx(2.2745253413468105)

    signed = commensurate_twist_angles_deg(8, 8, 8, 7, positive=False)
    positive = commensurate_twist_angles_deg(8, 8, 8, 7, positive=True)
    assert signed[0] < 0.0
    assert signed[1] < 0.0
    np.testing.assert_allclose(positive, tuple(abs(value) for value in signed), atol=1.0e-12)

    with pytest.raises(ValueError, match="must not be"):
        build_commensurate_geometry(0, 0, 8, 8)


def test_htqg_domain_aliases_and_static_hamiltonian_contract() -> None:
    params = _light_params()
    lattice = build_htqg_lattice(
        2.25,
        n_shells=0,
        graphene_lattice_constant_nm=params.graphene_lattice_constant_nm,
    )

    assert canonical_domain_key("αβα") == "alpha_beta_alpha"
    assert canonical_domain_key("abg") == "alpha_beta_gamma"

    for domain_key in ("alpha_beta_alpha", "alpha_beta_gamma"):
        domain = domain_displacements(lattice, domain_key)
        hamiltonian = build_hamiltonian(0.0 + 0.0j, lattice, params, domain=domain, valley=1)
        assert hamiltonian.shape == (lattice.matrix_dim, lattice.matrix_dim)
        np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-12)

        energies, eigenvectors = diagonalize_hamiltonian(0.0 + 0.0j, lattice, params, domain=domain, valley=1)
        assert energies.shape == (lattice.matrix_dim,)
        assert eigenvectors is not None
        assert eigenvectors.shape == hamiltonian.shape

    report = run_lightweight_validation(lattice, params, domain="alpha_beta_alpha")
    assert report.failure_count == 0


def test_htqg_model_band_helpers_return_central_band_shapes() -> None:
    model = HTQGModel.default(
        theta_deg=2.25,
        n_shells=0,
        domain="alpha_beta_alpha",
        params=_light_params(),
    )

    path = model.path_bands(points_per_segment=2, central_band_count=4)
    assert path.energies.shape[1] == 4
    assert path.eigenvectors is None
    assert tuple(path.band_indices) == (2, 3, 4, 5)

    grid = model.grid_bands(2, central_band_count=4)
    assert grid.energies.shape == (2, 2, 4)
    assert grid.eigenvectors is None
    assert tuple(grid.band_indices) == (2, 3, 4, 5)
