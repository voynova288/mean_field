from __future__ import annotations

import mean_field.systems.tdbg.projected_hf_geometry as projected_hf_geometry



def test_tdbg_band_window_indices_remains_stable_after_geometry_split() -> None:
    assert projected_hf_geometry.tdbg_band_window_indices(10, "isolated_cb") == (5,)
    assert projected_hf_geometry.tdbg_band_window_indices(10, "two_flat") == (4, 5)
    assert projected_hf_geometry.tdbg_band_window_indices(10, "central4") == (3, 4, 5, 6)


def test_tdbg_embedded_component_groups_label_sector_layer_sublattice() -> None:
    groups = projected_hf_geometry.tdbg_embedded_component_groups()

    assert [group.name for group in groups] == [
        "sector_0",
        "sector_1",
        "layer_0",
        "layer_1",
        "layer_2",
        "layer_3",
        "sublattice_A",
        "sublattice_B",
    ]
    assert [group.indices.tolist() for group in groups] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
    ]
