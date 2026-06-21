from __future__ import annotations

from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.tdbg import projected_hf, projected_hf_run


def test_tdbg_projected_hf_run_split_preserves_legacy_facade() -> None:
    assert projected_hf.build_tdbg_projected_hf_state is projected_hf_run.build_tdbg_projected_hf_state
    assert projected_hf.build_tdbg_projected_hf_kernel is projected_hf_run.build_tdbg_projected_hf_kernel
    assert projected_hf.build_tdbg_projected_hf_problem is projected_hf_run.build_tdbg_projected_hf_problem
    assert projected_hf.run_tdbg_projected_hf is projected_hf_run.run_tdbg_projected_hf
    assert tdbg_system.run_tdbg_projected_hf is projected_hf_run.run_tdbg_projected_hf
