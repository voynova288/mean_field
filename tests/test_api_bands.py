from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import KGrid, KPath, compute_bands


class _ToyModel:
    @property
    def matrix_dim(self) -> int:
        return 2

    def diagonalize(self, k_tilde: complex, *, n_bands: int | None = None, return_eigenvectors: bool = False, **kwargs: object):
        h = np.asarray([[k_tilde.real, 0.0], [0.0, 1.0 + k_tilde.imag]], dtype=float)
        evals, evecs = np.linalg.eigh(h)
        count = 2 if n_bands is None else int(n_bands)
        return evals[:count], evecs[:, :count] if return_eigenvectors else None

    def component_groups(self) -> tuple[object, ...]:
        return ()


def test_compute_bands_accepts_public_kgrid_for_non_square_direct_grid() -> None:
    kvec = np.asarray([[0.0 + 0.0j, 0.5 + 0.0j, 1.0 + 0.0j], [0.0 + 0.2j, 0.5 + 0.2j, 1.0 + 0.2j]])
    frac = np.zeros(kvec.shape + (2,), dtype=float)
    bundle = compute_bands(_ToyModel(), grid_mesh=KGrid(mesh=(2, 3), kvec=kvec, frac=frac), n_bands=1)
    assert bundle.source == "grid"
    assert bundle.k.shape == (2, 3)
    assert bundle.energies.shape == (2, 3, 1)
    np.testing.assert_allclose(bundle.energies[..., 0], [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])


def test_compute_bands_accepts_public_kpath_direct_path() -> None:
    path = KPath(kvec=np.asarray([0.0 + 0.0j, 0.5 + 0.0j]), labels=("G", "X"), node_indices=(1, 2))
    bundle = compute_bands(_ToyModel(), path=path, n_bands=2, return_eigenvectors=True)
    assert bundle.source == "path"
    assert bundle.energies.shape == (2, 2)
    assert bundle.eigenvectors is not None
    assert bundle.eigenvectors.shape == (2, 2, 2)
    assert bundle.basis_metadata["labels"] == ["G", "X"]


def test_compute_bands_non_square_tuple_points_to_kgrid() -> None:
    with pytest.raises(NotImplementedError, match="Pass a KGrid"):
        compute_bands(_ToyModel(), grid_mesh=(2, 3), n_bands=1)
