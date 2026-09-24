from __future__ import annotations

import numpy as np

from mean_field.core.hf.coulomb import screened_coulomb
from mean_field.systems.ahn_tmd import (
    AhnTMDHFConfig,
    AhnTMDModel,
    ahn_tmd_density_from_fixed_hole_occupations,
    build_ahn_tmd_lattice,
    build_ahn_tmd_projected_hf_data,
    hex_shell_indices,
)
from mean_field.systems.ahn_tmd.hf import _time_reverse_embedded_wavefunctions


def test_ahn_tmd_hex_shell_and_lattice_shape() -> None:
    assert len(hex_shell_indices(5)) == 91
    lattice = build_ahn_tmd_lattice(2.1, n_shell=5)
    assert lattice.n_g == 91
    assert lattice.matrix_dim == 182
    assert lattice.q0.real > 0.0
    assert abs(lattice.q0.imag) < 1.0e-14
    assert abs(lattice.q0 + lattice.q1 + lattice.q2) < 1.0e-12


def test_ahn_tmd_hamiltonian_is_hermitian_and_time_reversal_related() -> None:
    model = AhnTMDModel.from_config(theta_deg=2.1, n_shell=5)
    k = 0.173 * model.lattice.q0 + 0.217 * model.lattice.q1
    h_k = model.build_hamiltonian(k, valley=1)
    h_kp = model.build_hamiltonian(-k, valley=-1)
    assert np.linalg.norm(h_k - h_k.conjugate().T) < 1.0e-12
    assert np.linalg.norm(h_kp - h_k.conjugate()) < 1.0e-12


def test_ahn_tmd_path_and_grid_bands_have_expected_shapes() -> None:
    model = AhnTMDModel.from_config(theta_deg=2.1, n_shell=5)
    path_result = model.bands_along_standard_path(points_per_segment=2, n_bands=3)
    assert path_result.energies_eV.shape == (7, 3)
    grid_result = model.bands_on_rectangular_grid(2, 2, n_bands=2, return_eigenvectors=True)
    assert grid_result.energies_eV.shape == (4, 2)
    assert grid_result.eigenvectors is not None
    assert grid_result.eigenvectors.shape == (4, model.matrix_dim, 2)


def test_ahn_tmd_projected_hf_data_uses_core_shapes_and_paper_coulomb_limit() -> None:
    config = AhnTMDHFConfig(mesh=(1, 1), n_bands=2, g_shell=0, occupation_counts=(1, 1), max_iter=1)
    data = build_ahn_tmd_projected_hf_data(config)
    assert data.projected_basis.n_band == 2
    assert data.projected_basis.n_flavor == 2
    assert data.projected_basis.nt == 4
    assert data.h0_hole.shape == (4, 4, 1)
    assert data.initial_density.shape == data.h0_hole.shape
    assert len(data.overlap_blocks.shifts) == 1
    shift = data.overlap_blocks.shifts[0]
    assert data.overlap_blocks.hartree_screening[shift] == 0.0
    assert np.allclose(data.overlap_blocks.fock_screening[shift], 0.0)


def test_ahn_tmd_fock_screening_uses_target_minus_source_transfer() -> None:
    config = AhnTMDHFConfig(mesh=(3, 3), n_bands=2, g_shell=1, max_iter=1)
    data = build_ahn_tmd_projected_hf_data(config)
    # Pick a nonzero reciprocal image and an off-diagonal k pair so the sign of k_t-k_s matters.
    for shift, gvec in zip(data.overlap_blocks.shifts, data.overlap_blocks.gvecs, strict=True):
        if shift != (0, 0):
            break
    else:  # pragma: no cover - g_shell=1 should always provide nonzero shifts
        raise AssertionError("missing nonzero reciprocal shift")
    kt, ks = 1, 5
    actual = data.overlap_blocks.fock_screening[shift][kt, ks]
    expected = screened_coulomb(data.kvec[kt] - data.kvec[ks] + complex(gvec), config.coulomb_params)
    wrong_orientation = screened_coulomb(data.kvec[ks] - data.kvec[kt] + complex(gvec), config.coulomb_params)
    assert np.isclose(actual, expected)
    assert not np.isclose(actual, wrong_orientation)


def test_ahn_tmd_fixed_hole_density_has_nu2_trace() -> None:
    hamiltonian = np.zeros((4, 4, 1), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag([-2.0, -3.0, -1.0, 0.0])
    update = ahn_tmd_density_from_fixed_hole_occupations(hamiltonian, (1, 1), n_bands=2)
    assert update.density.shape == hamiltonian.shape
    assert np.isclose(np.trace(update.density[:, :, 0]).real, 2.0)
    assert update.energies.shape == (4, 1)


def test_ahn_tmd_time_reversal_embedded_wavefunctions_invert_g_grid() -> None:
    values = np.zeros((2 * 5 * 5, 1), dtype=np.complex128)
    # local=1, G=(m,n)=(+1,-2) in a width=5 grid with center offset=2.
    ix, iy = 3, 0
    flat = 1 + 2 * (ix + 5 * iy)
    values[flat, 0] = 2.0 + 3.0j
    transformed = _time_reverse_embedded_wavefunctions(values, width=5, local_basis_size=2)
    tx, ty = 1, 4  # (-m,-n)=(-1,+2)
    target_flat = 1 + 2 * (tx + 5 * ty)
    assert transformed[target_flat, 0] == 2.0 - 3.0j
    assert np.count_nonzero(transformed) == 1

    shifted = _time_reverse_embedded_wavefunctions(
        values,
        width=5,
        local_basis_size=2,
        reciprocal_shift=(-1, 0),
    )
    shifted_target_flat = 1 + 2 * (0 + 5 * ty)
    assert shifted[shifted_target_flat, 0] == 2.0 - 3.0j
    assert np.count_nonzero(shifted) == 1
