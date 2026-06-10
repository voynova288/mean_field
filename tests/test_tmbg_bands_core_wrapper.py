from __future__ import annotations

import numpy as np

from mean_field.systems.tmbg.bands import compute_bands_along_path, compute_bands_on_grid
from mean_field.systems.tmbg.hamiltonian import build_coupling_table, diagonalize_hamiltonian
from mean_field.systems.tmbg.lattice import build_kpath_from_nodes, build_tmbg_lattice
from mean_field.systems.tmbg.params import TMBGParameters


def test_tmbg_path_band_wrapper_matches_cached_direct_diagonalization() -> None:
    lattice = build_tmbg_lattice(theta_deg=1.21, n_shells=1)
    params = TMBGParameters.minimal()
    valley = -1
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
        n_bands=4,
        return_eigenvectors=True,
    )

    assert result.energies.shape == (path.kvec.size, 4)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (path.kvec.size, lattice.matrix_dim, 4)
    assert result.metadata["system"] == "TMBG"
    assert result.metadata["valley"] == valley

    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            n_bands=4,
            return_eigenvectors=True,
            coupling_table=coupling_table,
        )
        np.testing.assert_allclose(result.energies[ik], evals, atol=0.0, rtol=0.0)
        assert evecs is not None
        np.testing.assert_allclose(np.abs(result.eigenvectors[ik]), np.abs(evecs), atol=0.0, rtol=0.0)


def test_tmbg_grid_band_wrapper_preserves_no_eigenvector_mode() -> None:
    lattice = build_tmbg_lattice(theta_deg=1.21, n_shells=0)
    params = TMBGParameters.full(interlayer_potential=0.01)
    valley = 1

    result = compute_bands_on_grid(
        2,
        lattice,
        params,
        valley=valley,
        n_bands=3,
        return_eigenvectors=False,
        frac_shift=(0.25, 0.125),
    )

    assert result.k_grid_frac.shape == (2, 2, 2)
    assert result.kvec.shape == (2, 2)
    assert result.energies.shape == (2, 2, 3)
    assert result.eigenvectors is None
    assert result.metadata["system"] == "TMBG"
    assert result.metadata["valley"] == valley
    np.testing.assert_allclose(result.k_grid_frac[0, 0], [0.25, 0.125])

    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    evals, evecs = diagonalize_hamiltonian(
        result.kvec[0, 0],
        lattice,
        params,
        valley=valley,
        n_bands=3,
        return_eigenvectors=False,
        coupling_table=coupling_table,
    )
    assert evecs is None
    np.testing.assert_allclose(result.energies[0, 0], evals, atol=0.0, rtol=0.0)
