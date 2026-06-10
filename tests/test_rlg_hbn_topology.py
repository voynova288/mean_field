from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mean_field.systems.RnG_hBN.bands import GridBandsResult
from mean_field.systems.RnG_hBN.lattice import build_rlg_hbn_lattice
from mean_field.systems.RnG_hBN.params import RLGhBNParams
import mean_field.systems.RnG_hBN.topology as rlg_topology
from mean_field.systems.RnG_hBN.topology import rlg_hbn_boundary_sewing_transforms


def test_rlg_hbn_boundary_sewing_relabels_plane_wave_g_blocks() -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=1, layer_count=params.layer_count)
    block = 2 * params.layer_count
    vector = np.arange(block * lattice.n_g, dtype=np.complex128)
    lookup = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    sew_1, sew_2 = rlg_hbn_boundary_sewing_transforms(lattice, params, valley=1)
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


def test_rlg_hbn_boundary_sewing_reverses_shift_for_kprime_valley() -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=1, layer_count=params.layer_count)
    block = 2 * params.layer_count
    vector = np.arange(block * lattice.n_g, dtype=np.complex128)
    lookup = {tuple(int(value) for value in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    sew_1, _ = rlg_hbn_boundary_sewing_transforms(lattice, params, valley=-1)
    out = sew_1(vector)
    for target_index, (n1, n2) in enumerate(lattice.g_indices):
        source_index = lookup.get((int(n1) - 1, int(n2)))
        target_slice = slice(block * target_index, block * (target_index + 1))
        if source_index is None:
            np.testing.assert_array_equal(out[target_slice], np.zeros(block, dtype=np.complex128))
        else:
            source_slice = slice(block * source_index, block * (source_index + 1))
            np.testing.assert_array_equal(out[target_slice], vector[source_slice])


def test_rlg_hbn_grid_result_wrapper_passes_default_sewing_and_explicit_paper_orientation(monkeypatch) -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=1, layer_count=params.layer_count)
    grid = GridBandsResult(
        k_grid_frac=np.zeros((2, 2, 2), dtype=float),
        kvec=np.zeros((2, 2), dtype=np.complex128),
        energies=np.zeros((2, 2, 1), dtype=float),
        eigenvectors=np.zeros((2, 2, lattice.matrix_dim, 1), dtype=np.complex128),
    )
    captured: dict[str, object] = {}

    def fake_common_grid_result(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(rlg_topology, "compute_system_topology_from_grid_result", fake_common_grid_result)
    result = rlg_topology.compute_topology_from_grid_result(
        grid,
        0,
        lattice=lattice,
        params=params,
        valley=1,
        paper_orientation=True,
    )

    assert result.ok is True
    assert captured["orientation_sign"] == -1.0
    assert captured["sewing_transforms"] is not None
    assert captured["index_metadata"] == {"boundary_sewing": True, "orientation_sign": -1.0}


def test_rlg_hbn_on_grid_wrapper_uses_sewing_builder_by_default(monkeypatch) -> None:
    params = RLGhBNParams(layer_count=3, xi=1)
    lattice = build_rlg_hbn_lattice(shell_count=1, layer_count=params.layer_count)
    captured: dict[str, object] = {}

    def fake_common_on_grid(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(rlg_topology, "compute_system_topology_on_grid", fake_common_on_grid)
    result = rlg_topology.compute_topology_on_grid(3, lattice, params, 0, valley=-1, orientation_sign=1.0)

    assert result.ok is True
    assert captured["orientation_sign"] == 1.0
    assert captured["sewing_transforms"] is None
    builder = captured["sewing_transforms_builder"]
    assert callable(builder)
    sew_1, sew_2 = builder()
    assert callable(sew_1)
    assert callable(sew_2)
    assert captured["index_metadata"] == {"boundary_sewing": True, "orientation_sign": 1.0}
