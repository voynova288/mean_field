from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.bands import solve_bands_along_path, solve_bands_on_grid
from mean_field.core.lattice import KPath


def _toy_diagonalizer(k: complex, n_bands: int, return_eigenvectors: bool):
    h = np.diag([float(k.real), float(k.imag), float(abs(k))]).astype(float)
    evals = np.linalg.eigvalsh(h)[:n_bands]
    if not return_eigenvectors:
        return evals, None
    evecs = np.eye(3, dtype=np.complex128)[:, :n_bands]
    return evals, evecs


def test_solve_bands_along_path_handles_optional_eigenvectors() -> None:
    path = KPath(
        kvec=np.asarray([0.0 + 0.0j, 1.0 + 2.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, np.sqrt(5.0)], dtype=float),
        labels=("A", "B"),
        node_indices=(1, 2),
    )

    result = solve_bands_along_path(
        path,
        basis_dim=3,
        diagonalize=_toy_diagonalizer,
        n_bands=2,
        return_eigenvectors=True,
        metadata={"system": "toy"},
    )

    assert result.path is path
    assert result.energies.shape == (2, 2)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 3, 2)
    assert result.metadata["system"] == "toy"
    np.testing.assert_allclose(result.energies[1], [1.0, 2.0])


def test_solve_bands_on_grid_validates_requested_band_count() -> None:
    k_grid_frac = np.zeros((1, 1, 2), dtype=float)
    kvec = np.zeros((1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="Requested 4 bands"):
        solve_bands_on_grid(
            k_grid_frac,
            kvec,
            basis_dim=3,
            diagonalize=_toy_diagonalizer,
            n_bands=4,
        )


def test_solve_bands_on_grid_shapes_match_grid_and_basis() -> None:
    k_grid_frac = np.zeros((2, 1, 2), dtype=float)
    kvec = np.asarray([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=np.complex128)

    result = solve_bands_on_grid(
        k_grid_frac,
        kvec,
        basis_dim=3,
        diagonalize=_toy_diagonalizer,
        n_bands=1,
        return_eigenvectors=True,
    )

    assert result.k_grid_frac.shape == (2, 1, 2)
    assert result.kvec.shape == (2, 1)
    assert result.energies.shape == (2, 1, 1)
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 1, 3, 1)
