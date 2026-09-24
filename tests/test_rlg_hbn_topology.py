from __future__ import annotations

import numpy as np

from analysis.topology import compute_lattice_topology, default_k_grid_frac, sewing_transforms_from_block_spec
from mean_field.systems.RnG_hBN.bands import GridBandsResult
from mean_field.systems.RnG_hBN.lattice import build_rlg_hbn_lattice
from mean_field.systems.RnG_hBN.params import RLGhBNParams
import mean_field.systems.RnG_hBN.topology as rlg_topology
from mean_field.systems.RnG_hBN.topology import rlg_hbn_basis_sewing


def test_rlg_hbn_generic_basis_sewing_relabels_plane_wave_g_blocks() -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=2, layer_count=params.layer_count)
    block = 2 * params.layer_count
    vector = np.arange(block * lattice.n_g, dtype=np.complex128)
    lookup = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    sew_1, sew_2 = sewing_transforms_from_block_spec(rlg_hbn_basis_sewing(lattice, params, valley=1))
    out_1 = sew_1(vector)
    out_2 = sew_2(vector)
    for target_index, (n1, n2) in enumerate(lattice.g_indices):
        for out, shift in ((out_1, (1, 0)), (out_2, (0, 1))):
            source_index = lookup.get((int(n1) + shift[0], int(n2) + shift[1]))
            target_slice = slice(block * target_index, block * (target_index + 1))
            if source_index is None:
                np.testing.assert_array_equal(out[target_slice], np.zeros(block, dtype=np.complex128))
            else:
                source_slice = slice(block * source_index, block * (source_index + 1))
                np.testing.assert_array_equal(out[target_slice], vector[source_slice])


def test_rlg_hbn_generic_basis_sewing_reverses_shift_for_kprime_valley() -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=2, layer_count=params.layer_count)
    block = 2 * params.layer_count
    vector = np.arange(block * lattice.n_g, dtype=np.complex128)
    lookup = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    sew_1, _ = sewing_transforms_from_block_spec(rlg_hbn_basis_sewing(lattice, params, valley=-1))
    out = sew_1(vector)
    for target_index, (n1, n2) in enumerate(lattice.g_indices):
        source_index = lookup.get((int(n1) - 1, int(n2)))
        target_slice = slice(block * target_index, block * (target_index + 1))
        if source_index is None:
            np.testing.assert_array_equal(out[target_slice], np.zeros(block, dtype=np.complex128))
        else:
            source_slice = slice(block * source_index, block * (source_index + 1))
            np.testing.assert_array_equal(out[target_slice], vector[source_slice])


def test_rlg_hbn_grid_state_attaches_boundary_sewing_and_paper_orientation() -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=2, layer_count=params.layer_count)
    grid = GridBandsResult(
        k_grid_frac=default_k_grid_frac(2, 2),
        kvec=np.zeros((2, 2), dtype=np.complex128),
        energies=np.zeros((2, 2, 1), dtype=float),
        eigenvectors=np.ones((2, 2, lattice.matrix_dim, 1), dtype=np.complex128),
        band_indices=(7,),
    )

    result = compute_lattice_topology(
        rlg_topology.fhs_state_from_grid_result(
            grid,
            7,
            lattice=lattice,
            params=params,
            valley=1,
            paper_orientation=True,
        )
    )

    assert result.band_indices == (7,)
    assert result.rounded_chern_number == 0
    assert result.valley == 1
    assert result.metadata["boundary_sewing"] is True
    assert result.metadata["orientation_sign"] == -1.0
    assert result.metadata["absolute_band_indices"] == [7]
    assert result.metadata["column_indices"] == [0]


def test_rlg_hbn_on_grid_state_uses_boundary_sewing_by_default(monkeypatch) -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=2, layer_count=params.layer_count)

    def fake_compute_bands_on_grid(*args, **kwargs):
        return GridBandsResult(
            k_grid_frac=default_k_grid_frac(2, 2),
            kvec=np.zeros((2, 2), dtype=np.complex128),
            energies=np.zeros((2, 2, 1), dtype=float),
            eigenvectors=np.ones((2, 2, lattice.matrix_dim, 1), dtype=np.complex128),
            band_indices=(0,),
        )

    monkeypatch.setattr(rlg_topology, "compute_bands_on_grid", fake_compute_bands_on_grid)
    result = compute_lattice_topology(rlg_topology.fhs_state_on_grid(3, lattice, params, 0, valley=-1, orientation_sign=1.0))

    assert result.band_indices == (0,)
    assert result.rounded_chern_number == 0
    assert result.valley == -1
    assert result.metadata["boundary_sewing"] is True
    assert result.metadata["orientation_sign"] == 1.0
