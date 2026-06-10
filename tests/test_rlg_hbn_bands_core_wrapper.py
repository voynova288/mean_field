from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh

from mean_field.systems.RnG_hBN.bands import compute_bands_along_path, compute_bands_on_grid, neutrality_energy_mev
from mean_field.systems.RnG_hBN.hamiltonian import build_hamiltonian, diagonalize_hamiltonian, flat_band_indices
from mean_field.systems.RnG_hBN.lattice import build_kpath_from_nodes, build_rlg_hbn_lattice
from mean_field.systems.RnG_hBN.params import RLGhBNParams


def test_rlg_hbn_path_band_wrapper_preserves_no_eigenvector_eigvalsh_path() -> None:
    params = RLGhBNParams.from_table(layer_count=3, xi=0, displacement_field_mev=20.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=1, layer_count=params.layer_count)
    path = build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m),
        ("K", "Gamma"),
        (2,),
    )

    result = compute_bands_along_path(
        path,
        lattice,
        params,
        valley=1,
        n_bands=5,
        return_eigenvectors=False,
    )

    assert result.energies.shape == (path.kvec.size, 5)
    assert result.eigenvectors is None
    assert result.metadata["system"] == "RLG_hBN"
    for ik, kval in enumerate(path.kvec):
        hamiltonian = build_hamiltonian(kval, lattice, params, valley=1)
        expected = eigvalsh(hamiltonian, subset_by_index=[0, 4])
        np.testing.assert_allclose(result.energies[ik], expected, atol=0.0, rtol=0.0)


def test_rlg_hbn_grid_band_wrapper_matches_direct_diagonalization_with_eigenvectors() -> None:
    params = RLGhBNParams.from_table(layer_count=3, xi=1, displacement_field_mev=12.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=1, layer_count=params.layer_count)

    result = compute_bands_on_grid(
        2,
        lattice,
        params,
        valley=-1,
        n_bands=4,
        return_eigenvectors=True,
        frac_shift=(0.125, 0.25),
    )

    assert result.k_grid_frac.shape == (2, 2, 2)
    assert result.kvec.shape == (2, 2)
    assert result.energies.shape == (2, 2, 4)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 2, lattice.matrix_dim, 4)
    assert result.metadata["valley"] == -1
    np.testing.assert_allclose(result.k_grid_frac[0, 0], [0.125, 0.25])

    evals, evecs = diagonalize_hamiltonian(result.kvec[0, 0], lattice, params, valley=-1, n_bands=4)
    np.testing.assert_allclose(result.energies[0, 0], evals, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.abs(result.eigenvectors[0, 0]), np.abs(evecs), atol=0.0, rtol=0.0)


def test_rlg_hbn_neutrality_energy_works_with_core_path_result() -> None:
    params = RLGhBNParams.from_table(layer_count=3, xi=1, displacement_field_mev=0.0)
    lattice = build_rlg_hbn_lattice(theta_deg=0.77, shell_count=1, layer_count=params.layer_count)
    path = build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m),
        ("K", "Gamma"),
        (2,),
    )
    result = compute_bands_along_path(path, lattice, params, valley=1)
    valence, conduction = flat_band_indices(lattice, params)
    expected = 0.5 * (float(np.max(result.energies[:, valence])) + float(np.min(result.energies[:, conduction])))
    assert neutrality_energy_mev(result, lattice, params) == expected
