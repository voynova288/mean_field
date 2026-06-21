from __future__ import annotations

from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.tdbg import projected_hf, projected_hf_data


def test_tdbg_projected_hf_data_split_preserves_legacy_facade() -> None:
    assert projected_hf._projected_orbital_g_matrix is projected_hf_data._projected_orbital_g_matrix
    assert projected_hf._projected_onebody_and_wavefunctions is projected_hf_data._projected_onebody_and_wavefunctions
    assert projected_hf.build_tdbg_projected_hf_data is projected_hf_data.build_tdbg_projected_hf_data
    assert tdbg_system.build_tdbg_projected_hf_data is projected_hf_data.build_tdbg_projected_hf_data
