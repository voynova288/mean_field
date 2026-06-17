from __future__ import annotations

from mean_field.systems.tdbg import hf as tdbg_hf
from mean_field.systems.tdbg import projected_hf, projected_hf_geometry


def test_tdbg_projected_hf_geometry_split_preserves_legacy_facade() -> None:
    assert projected_hf.tdbg_band_window_indices is projected_hf_geometry.tdbg_band_window_indices
    assert projected_hf.tdbg_moire_area_nm2 is projected_hf_geometry.tdbg_moire_area_nm2
    assert projected_hf._shift_table is projected_hf_geometry._shift_table
    assert projected_hf._TDBGQSiteEmbedding is projected_hf_geometry._TDBGQSiteEmbedding
    assert projected_hf._tdbg_q_site_embedding is projected_hf_geometry._tdbg_q_site_embedding
    assert projected_hf._tdbg_core_order_permutation is projected_hf_geometry._tdbg_core_order_permutation
    assert projected_hf._tdbg_projected_wavefunction_basis is projected_hf_geometry._tdbg_projected_wavefunction_basis
    assert projected_hf._tdbg_total_overlap_from_bases is projected_hf_geometry._tdbg_total_overlap_from_bases
    assert projected_hf._tdbg_total_overlap_between is projected_hf_geometry._tdbg_total_overlap_between
    assert projected_hf.tdbg_embedded_component_groups is projected_hf_geometry.tdbg_embedded_component_groups
    assert tdbg_hf.tdbg_moire_area_nm2 is projected_hf_geometry.tdbg_moire_area_nm2
    assert tdbg_hf.tdbg_embedded_component_groups is projected_hf_geometry.tdbg_embedded_component_groups


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
