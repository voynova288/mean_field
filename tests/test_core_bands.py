from __future__ import annotations

import numpy as np

from mean_field.core.bands import compute_grid_bands, compute_path_bands, resolve_n_bands
from mean_field.core.lattice import KPath


def _toy_path() -> KPath:
    return KPath(
        kvec=np.asarray([0.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 0.5], dtype=float),
        labels=("G", "X"),
        node_indices=(1, 2),
    )


def _toy_diagonalize(k: complex, n_bands: int, return_eigenvectors: bool):
    h = np.asarray([[k.real, 0.0], [0.0, 1.0 + k.real]], dtype=float)
    evals, evecs = np.linalg.eigh(h)
    if return_eigenvectors:
        return evals[:n_bands], evecs[:, :n_bands]
    return evals[:n_bands], None


def test_resolve_n_bands_defaults_and_bounds() -> None:
    assert resolve_n_bands(4, None) == 4
    assert resolve_n_bands(4, 2) == 2


def test_compute_path_bands_with_eigenvectors() -> None:
    result = compute_path_bands(
        _toy_path(),
        matrix_dim=2,
        n_bands=1,
        return_eigenvectors=True,
        diagonalize=_toy_diagonalize,
    )

    assert result.energies.shape == (2, 1)
    assert result.band_indices == ()
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 2, 1)
    np.testing.assert_allclose(result.energies[:, 0], [0.0, 0.5])


def test_compute_path_bands_records_result_band_indices() -> None:
    result = compute_path_bands(
        _toy_path(),
        matrix_dim=2,
        n_bands=1,
        return_eigenvectors=False,
        diagonalize=_toy_diagonalize,
        result_band_indices=(10,),
    )

    assert result.band_indices == (10,)


def test_compute_grid_bands_without_eigenvectors() -> None:
    kvec = np.asarray([[0.0 + 0.0j, 0.25 + 0.0j], [0.5 + 0.0j, 0.75 + 0.0j]], dtype=np.complex128)
    kfrac = np.zeros((2, 2, 2), dtype=float)
    result = compute_grid_bands(
        k_grid_frac=kfrac,
        kvec=kvec,
        matrix_dim=2,
        n_bands=2,
        return_eigenvectors=False,
        diagonalize=_toy_diagonalize,
    )

    assert result.energies.shape == (2, 2, 2)
    assert result.band_indices == ()
    assert result.eigenvectors is None
    np.testing.assert_allclose(result.energies[..., 0], [[0.0, 0.25], [0.5, 0.75]])
