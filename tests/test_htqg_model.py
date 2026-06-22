from __future__ import annotations

import numpy as np

from mean_field.systems.htqg import (
    HTQGModel,
    HTQGParams,
    build_hamiltonian,
    build_htqg_lattice,
    canonical_domain_key,
    diagonalize_hamiltonian,
    domain_displacements,
)


def _light_params() -> HTQGParams:
    return HTQGParams.default(kappa=0.6, lambda_mdt_nm=0.0, include_dirac_rotation=False)


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

    h_plus = build_hamiltonian(0.0 + 0.0j, lattice, params, domain="alpha_beta_alpha", valley=1)
    h_minus = build_hamiltonian(0.0 + 0.0j, lattice, params, domain="alpha_beta_alpha", valley=-1)
    tr_residual = float(np.max(np.abs(np.linalg.eigvalsh(h_plus) - np.linalg.eigvalsh(h_minus))))
    assert tr_residual <= 1.0e-8


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
