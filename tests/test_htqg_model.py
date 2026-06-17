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
from mean_field.systems.htqg.hf import (
    HTQGInteractionSettings,
    HTQGProjectedHFConfig,
    build_htqg_interaction_components,
    build_htqg_overlap_blocks,
    build_htqg_projected_hf_data,
    initialize_htqg_density,
    occupation_by_label,
)
from mean_field.systems.htqg.validation import run_lightweight_validation


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


def test_htqg_projected_hf_tiny_data_overlap_and_density_contract() -> None:
    config = HTQGProjectedHFConfig(
        theta_deg=2.25,
        n_shells=0,
        mesh_size=1,
        active_band_count=2,
        domain="alpha_beta_alpha",
        filling=0,
        params=_light_params(),
        interaction=HTQGInteractionSettings(g_shells=0),
    )
    data = build_htqg_projected_hf_data(config)

    assert data.nt == 8
    assert data.nk == 1
    assert data.n_occupied_per_k == 4
    assert data.h0.shape == (data.nt, data.nt, data.nk)
    np.testing.assert_allclose(data.h0[:, :, 0], data.h0[:, :, 0].conj().T, atol=1.0e-12)

    overlap_blocks = build_htqg_overlap_blocks(data)
    assert overlap_blocks.shifts == ((0, 0),)

    density = initialize_htqg_density(data, init_mode="bm", seed=1)
    assert density.shape == data.h0.shape
    np.testing.assert_allclose(density[:, :, 0], density[:, :, 0].conj().T, atol=1.0e-12)
    assert np.isclose(np.trace(density[:, :, 0]).real, data.n_occupied_per_k)

    components = build_htqg_interaction_components(data, density, overlap_blocks=overlap_blocks)
    assert set(components) == {"hartree", "fock", "total"}
    for component in components.values():
        assert component.shape == data.h0.shape
        np.testing.assert_allclose(component[:, :, 0], component[:, :, 0].conj().T, atol=1.0e-12)

    occupations = occupation_by_label(data, density)
    assert len(occupations) == data.nt
    assert np.isclose(sum(occupations.values()), data.n_occupied_per_k)
