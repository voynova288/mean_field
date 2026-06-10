from __future__ import annotations

import numpy as np

from mean_field.systems.atmg.bands import compute_bands_along_path, compute_bands_on_grid
from mean_field.systems.atmg.bilayer_map import build_atmg_via_tbg_sum
from mean_field.systems.atmg.hamiltonian import diagonalize_hamiltonian
from mean_field.systems.atmg.lattice import build_atmg_lattice, build_kpath_from_nodes
from mean_field.systems.atmg.params import ATMGParameters
from mean_field.systems.atmg.tbg import build_coupling_table


def test_atmg_path_band_wrapper_matches_direct_diagonalization_and_mapping() -> None:
    lattice = build_atmg_lattice(theta_deg=1.53, n_shells=1)
    params = ATMGParameters.chiral(3, 1.53)
    valley = 1
    path = build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m),
        ("K", "Gamma"),
        2,
    )

    result = compute_bands_along_path(
        path,
        lattice,
        params,
        valley=valley,
        n_bands=5,
        return_eigenvectors=True,
        include_mapped=True,
    )

    assert result.energies.shape == (path.kvec.size, 5)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (path.kvec.size, 2 * params.n_layers * lattice.n_g, 5)
    assert result.mapped_energies is not None
    assert result.mapped_energies.shape == result.energies.shape
    assert result.subspace_labels
    assert result.subspace_energies is not None
    assert result.metadata["system"] == "ATMG"

    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            n_bands=5,
            coupling_table=coupling_table,
        )
        np.testing.assert_allclose(result.energies[ik], evals, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.abs(result.eigenvectors[ik]), np.abs(evecs), atol=0.0, rtol=0.0)

        mapped = build_atmg_via_tbg_sum(kval, lattice, params, valley=valley)
        np.testing.assert_allclose(result.mapped_energies[ik], mapped.combined_energies[:5], atol=0.0, rtol=0.0)


def test_atmg_grid_band_wrapper_preserves_optional_mapped_fields() -> None:
    lattice = build_atmg_lattice(theta_deg=1.53, n_shells=0)
    params = ATMGParameters.realistic(2, 1.53, kappa=0.8)
    valley = -1

    result = compute_bands_on_grid(
        2,
        lattice,
        params,
        valley=valley,
        n_bands=4,
        return_eigenvectors=False,
        include_mapped=True,
        frac_shift=(0.25, 0.125),
    )

    assert result.k_grid_frac.shape == (2, 2, 2)
    assert result.kvec.shape == (2, 2)
    assert result.energies.shape == (2, 2, 4)
    assert result.eigenvectors is None
    assert result.mapped_energies is not None
    assert result.mapped_energies.shape == (2, 2, 4)
    assert result.metadata["system"] == "ATMG"
    assert result.metadata["valley"] == valley
    np.testing.assert_allclose(result.k_grid_frac[0, 0], [0.25, 0.125])

    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    evals, _ = diagonalize_hamiltonian(
        result.kvec[0, 0],
        lattice,
        params,
        valley=valley,
        n_bands=4,
        coupling_table=coupling_table,
    )
    np.testing.assert_allclose(result.energies[0, 0], evals, atol=0.0, rtol=0.0)

    mapped = build_atmg_via_tbg_sum(result.kvec[0, 0], lattice, params, valley=valley)
    np.testing.assert_allclose(result.mapped_energies[0, 0], mapped.combined_energies[:4], atol=0.0, rtol=0.0)
