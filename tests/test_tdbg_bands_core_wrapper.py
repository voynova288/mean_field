from __future__ import annotations

import numpy as np

from mean_field.systems.tdbg.bands import compute_bands_along_path, compute_bands_on_grid
from mean_field.systems.tdbg.hamiltonian import diagonalize_hamiltonian
from mean_field.systems.tdbg.lattice import build_kpath_from_nodes, build_tdbg_lattice
from mean_field.systems.tdbg.params import TDBGParameters


def test_tdbg_path_band_wrapper_matches_direct_diagonalization() -> None:
    lattice = build_tdbg_lattice(theta_deg=1.38, cut=1.0)
    params = TDBGParameters.minimal()
    path = build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m),
        ("K", "Gamma"),
        (2,),
    )

    result = compute_bands_along_path(path, lattice, params, valley=1, n_bands=3, return_eigenvectors=True)

    assert result.energies.shape == (path.kvec.size, 3)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (path.kvec.size, lattice.matrix_dim, 3)
    assert result.metadata["system"] == "TDBG"
    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(kval, lattice, params, valley=1, n_bands=3)
        np.testing.assert_allclose(result.energies[ik], evals, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.abs(result.eigenvectors[ik]), np.abs(evecs), atol=0.0, rtol=0.0)


def test_tdbg_grid_band_wrapper_preserves_grid_coordinates() -> None:
    lattice = build_tdbg_lattice(theta_deg=1.38, cut=1.0)
    params = TDBGParameters.minimal()

    result = compute_bands_on_grid(
        2,
        lattice,
        params,
        valley=-1,
        n_bands=2,
        return_eigenvectors=False,
        frac_shift=(0.125, 0.25),
    )

    assert result.k_grid_frac.shape == (2, 2, 2)
    assert result.kvec.shape == (2, 2)
    assert result.energies.shape == (2, 2, 2)
    assert result.eigenvectors is None
    assert result.metadata["system"] == "TDBG"
    assert result.metadata["valley"] == -1
    np.testing.assert_allclose(result.k_grid_frac[0, 0], [0.125, 0.25])

    evals, _ = diagonalize_hamiltonian(result.kvec[0, 0], lattice, params, valley=-1, n_bands=2)
    np.testing.assert_allclose(result.energies[0, 0], evals, atol=0.0, rtol=0.0)
