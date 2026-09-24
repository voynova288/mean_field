from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.htqg import (
    HTQGModel,
    HTQGParams,
)
from mean_field.systems.htqg.commensurate import (
    build_commensurate_geometry,
    commensurate_twist_angles_deg,
    fujimoto_2025_fig2_checkpoint,
)
from mean_field.systems.htqg.hamiltonian import (
    build_hamiltonian,
    diagonalize_hamiltonian,
)
from mean_field.systems.htqg.lattice import (
    build_htqg_lattice,
)
from mean_field.systems.htqg.domains import (
    canonical_domain_key,
    domain_displacements,
)
from mean_field.systems.htqg.validation import run_lightweight_validation
from analysis.topology import compute_lattice_topology, sewing_transforms_from_block_spec
from mean_field.systems.htqg.topology import fhs_state_from_eigenvectors, fhs_state_from_grid_result, htqg_basis_sewing
from mean_field.core.hf.overlap import ProjectedWavefunctionBasis
from mean_field.systems.htqg.hf import (
    HTQGInteractionSettings,
    HTQGProjectedHFConfig,
    _calculate_htqg_projected_overlap_between,
    build_htqg_projected_hf_data,
)


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

    assert canonical_domain_key("ααα") == "alpha_alpha_alpha"
    assert canonical_domain_key("aaa") == "alpha_alpha_alpha"
    assert canonical_domain_key("αβα") == "alpha_beta_alpha"
    assert canonical_domain_key("abg") == "alpha_beta_gamma"

    aa_like = domain_displacements(lattice, "alpha_alpha_alpha")
    assert aa_like.d12 == 0.0j
    assert aa_like.d34 == 0.0j
    assert aa_like.c2zt_partner == "alpha_alpha_alpha"
    assert aa_like.is_aa_like

    for domain_key in ("alpha_alpha_alpha", "alpha_beta_alpha", "alpha_beta_gamma"):
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


def test_htqg_projected_hf_transfer_shell_is_c3_closed() -> None:
    config = HTQGProjectedHFConfig(
        n_shells=0,
        mesh_size=1,
        active_band_count=2,
        domain="alpha_beta_alpha",
        params=_light_params(),
        interaction=HTQGInteractionSettings(g_shells=2),
        frac_shift=(0.0, 0.0),
    )
    data = build_htqg_projected_hf_data(config)
    shifts = {tuple(shift) for shift in data.shifts}
    c3 = lambda m, n: (-m - n, m)

    assert len(shifts) == 1 + 3 * 2 * (2 + 1)
    assert all(c3(*shift) in shifts for shift in shifts)


def test_htqg_projected_hf_default_mesh_shift_is_half_cell() -> None:
    config = HTQGProjectedHFConfig(
        n_shells=0,
        mesh_size=4,
        active_band_count=2,
        domain="alpha_beta_alpha",
        params=_light_params(),
        interaction=HTQGInteractionSettings(g_shells=0),
        frac_shift=None,
    )
    data = build_htqg_projected_hf_data(config)
    np.testing.assert_allclose(data.k_grid_frac[0, 0], (0.125, 0.125), atol=1.0e-14)
    np.testing.assert_allclose(data.k_grid_frac[1, 0], (0.375, 0.125), atol=1.0e-14)


def test_htqg_projected_hf_kprime_overlap_uses_opposite_transfer_shift() -> None:
    target_wf = np.zeros((9, 1, 2, 1), dtype=np.complex128)
    source_wf = np.zeros_like(target_wf)
    # Flattening is local + local_basis_size * (ix + nx * iy), with nx=ny=3.
    center = 1 + 3 * 1
    plus_x = 2 + 3 * 1
    minus_x = 0 + 3 * 1
    target_wf[center, 0, 0, 0] = 1.0  # K target at center.
    target_wf[center, 0, 1, 0] = 1.0  # K' target at center.
    source_wf[plus_x, 0, 0, 0] = 1.0  # K source couples for shift +(1,0).
    source_wf[minus_x, 0, 1, 0] = 1.0  # K' source must couple with the opposite shift.
    target = ProjectedWavefunctionBasis(target_wf, (3, 3), n_spin=1, local_basis_size=1, boundary_mode="zero_fill")
    source = ProjectedWavefunctionBasis(source_wf, (3, 3), n_spin=1, local_basis_size=1, boundary_mode="zero_fill")

    overlap = _calculate_htqg_projected_overlap_between(target, source, 1, 0)

    assert overlap.shape == (2, 1, 2, 1)
    assert overlap[0, 0, 0, 0] == pytest.approx(1.0)  # K block.
    assert overlap[1, 0, 1, 0] == pytest.approx(1.0)  # K' block would vanish with the old same-sign shift.
    assert overlap[0, 0, 1, 0] == pytest.approx(0.0)
    assert overlap[1, 0, 0, 0] == pytest.approx(0.0)


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



def test_htqg_topology_state_uses_half_cell_grid_and_valley_aware_sewing() -> None:
    model = HTQGModel.default(
        theta_deg=2.25,
        n_shells=0,
        domain="alpha_beta_gamma",
        params=_light_params(),
        valley=-1,
    )
    state = model.fhs_state_on_grid(
        2,
        model.lattice.matrix_dim // 2,
        central_band_count=2,
    )

    np.testing.assert_allclose(state.k_grid_frac[0, 0], (0.25, 0.25))
    assert state.basis_sewing is not None
    assert state.basis_sewing.translations == ((-1.0, 0.0), (0.0, -1.0))
    assert state.metadata["frac_shift"] == [0.5, 0.5]


def test_htqg_valley_sewing_matches_interior_hamiltonian_covariance() -> None:
    model = HTQGModel.default(
        theta_deg=2.25,
        n_shells=2,
        domain="alpha_beta_gamma",
        params=_light_params(),
    )
    momentum = 0.137 * model.lattice.b_m1 + 0.211 * model.lattice.b_m2
    for valley in (1, -1):
        spec = htqg_basis_sewing(model.lattice, valley=valley)
        seams = sewing_transforms_from_block_spec(spec)
        reciprocal_vectors = (model.lattice.b_m1, model.lattice.b_m2)
        for seam, reciprocal_vector in zip(
            seams,
            reciprocal_vectors,
            strict=True,
        ):
            transform = seam(
                np.eye(model.lattice.matrix_dim, dtype=np.complex128)
            )
            retained = np.flatnonzero(np.linalg.norm(transform, axis=1) > 0.5)
            source = model.build_hamiltonian(momentum, valley=valley)
            target = model.build_hamiltonian(
                momentum + reciprocal_vector,
                valley=valley,
            )
            mapped = transform @ source @ transform.conjugate().T
            np.testing.assert_allclose(
                target[np.ix_(retained, retained)],
                mapped[np.ix_(retained, retained)],
                atol=5.0e-13,
                rtol=0.0,
            )


def test_htqg_realistic_fhs_chern_matches_fig1_chern_checkpoint() -> None:
    """Reduced-faithful HTQG topology check using actual Fig. 1 parameters.

    This is not a toy/local-array snapshot: it uses the realistic Fujimoto HTQG
    parameter set, shell-6 plane-wave basis, mesh-9 FHS torus, and HTQG boundary
    sewing. The oracle is the paper/report checkpoint for the K-valley αβγ
    isolated central energy bands: valence C=-2 and conduction C=0.
    """

    mesh_size = 9
    model = HTQGModel.default(
        theta_deg=2.25,
        n_shells=6,
        domain="alpha_beta_gamma",
        params=HTQGParams.realistic(kappa=0.6),
        valley=1,
    )
    grid = model.grid_bands(
        mesh_size,
        central_band_count=4,
        return_eigenvectors=True,
        frac_shift=(0.5, 0.5),
    )

    assert tuple(grid.band_indices) == (506, 507, 508, 509)
    np.testing.assert_allclose(
        grid.k_grid_frac[0, 0],
        (0.5 / mesh_size, 0.5 / mesh_size),
        atol=1.0e-15,
    )
    adjacent_gaps = tuple(
        float(np.min(grid.energies[..., index + 1] - grid.energies[..., index]))
        for index in range(3)
    )
    assert adjacent_gaps[0] > 0.04
    assert adjacent_gaps[1] > 0.005
    assert adjacent_gaps[2] > 0.04
    valence = compute_lattice_topology(fhs_state_from_grid_result(
        grid,
        507,
        lattice=model.lattice,
        domain="alpha_beta_gamma",
        valley=1,
        metadata={"checkpoint": "reports/htqg_fig1_chern_comparison_20260611.md"},
    ))
    valence_from_vectors = compute_lattice_topology(fhs_state_from_eigenvectors(
        grid.eigenvectors,
        1,
        lattice=model.lattice,
        domain="alpha_beta_gamma",
        valley=1,
        k_grid_frac=grid.k_grid_frac,
        metadata={"checkpoint": "reports/htqg_fig1_chern_comparison_20260611.md", "column_index": 1},
    ))
    conduction = compute_lattice_topology(fhs_state_from_grid_result(
        grid,
        508,
        lattice=model.lattice,
        domain="alpha_beta_gamma",
        valley=1,
        metadata={"checkpoint": "reports/htqg_fig1_chern_comparison_20260611.md"},
    ))
    kprime_valence = compute_lattice_topology(fhs_state_from_eigenvectors(
        np.conjugate(grid.eigenvectors[::-1, ::-1]),
        1,
        lattice=model.lattice,
        domain="alpha_beta_gamma",
        valley=-1,
        k_grid_frac=grid.k_grid_frac,
        metadata={"time_reversal_source": "K(-k)*"},
    ))

    assert valence.band_indices == (507,)
    assert valence.rounded_chern_number == -2
    assert valence.chern_number == pytest.approx(-2.0, abs=1.0e-10)
    assert valence.min_link_magnitude > 0.5
    assert valence.minimum_plaquette_branch_margin > 0.05
    assert valence.metadata["boundary_sewing"] is True
    assert valence.metadata["absolute_band_indices"] == [507]

    assert valence_from_vectors.band_indices == (1,)
    assert valence_from_vectors.rounded_chern_number == valence.rounded_chern_number
    assert valence_from_vectors.chern_number == pytest.approx(valence.chern_number, abs=1.0e-12)
    np.testing.assert_allclose(valence_from_vectors.berry_curvature, valence.berry_curvature, atol=1.0e-12, rtol=0.0)

    assert conduction.band_indices == (508,)
    assert conduction.rounded_chern_number == 0
    assert conduction.chern_number == pytest.approx(0.0, abs=1.0e-10)
    assert conduction.min_link_magnitude > 0.5
    assert conduction.minimum_plaquette_branch_margin > 0.05
    assert conduction.metadata["boundary_sewing"] is True
    assert conduction.metadata["absolute_band_indices"] == [508]

    assert kprime_valence.rounded_chern_number == 2
    assert kprime_valence.chern_number == pytest.approx(2.0, abs=1.0e-10)
    assert kprime_valence.min_link_magnitude > 0.5
    assert kprime_valence.minimum_plaquette_branch_margin > 0.05

    # The tested Berry curvature is the FHS plaquette flux; its integral is the
    # Chern oracle above. Do not replace this by projector/QGT curvature.
    assert float(np.sum(valence.berry_curvature) / (2.0 * np.pi)) == pytest.approx(valence.chern_number)
    assert float(np.sum(conduction.berry_curvature) / (2.0 * np.pi)) == pytest.approx(conduction.chern_number)
