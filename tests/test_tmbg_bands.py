from __future__ import annotations

from mean_field.systems.tmbg import (
    TMBGParameters,
    build_standard_kpath,
    build_tmbg_lattice,
    compute_bands_along_path,
    compute_bands_on_grid,
)


def test_tmbg_band_helpers_export_expected_shapes() -> None:
    lattice = build_tmbg_lattice(1.21, n_shells=1)
    params = TMBGParameters.minimal()
    path = build_standard_kpath(lattice, points_per_segment=4)

    path_result = compute_bands_along_path(path, lattice, params, valley=1, n_bands=8)
    grid_result = compute_bands_on_grid(3, lattice, params, valley=1, n_bands=6)

    assert path_result.energies.shape == (path.kvec.size, 8)
    assert grid_result.k_grid_frac.shape == (3, 3, 2)
    assert grid_result.kvec.shape == (3, 3)
    assert grid_result.energies.shape == (3, 3, 6)
