from __future__ import annotations

from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.tdbg import projected_hf, projected_hf_state


def test_tdbg_projected_hf_state_split_preserves_legacy_facade() -> None:
    assert projected_hf.TDBGStateLabel is projected_hf_state.TDBGStateLabel
    assert projected_hf.TDBGProjectedHFData is projected_hf_state.TDBGProjectedHFData
    assert projected_hf.TDBGProjectedHFState is projected_hf_state.TDBGProjectedHFState
    assert projected_hf.TDBGProjectedHFTargetData is projected_hf_state.TDBGProjectedHFTargetData
    assert projected_hf.TDBGProjectedHFResult is projected_hf_state.TDBGProjectedHFResult
    assert projected_hf.initialize_tdbg_density is projected_hf_state.initialize_tdbg_density
    assert projected_hf.initialize_tdbg_nu2_density is projected_hf_state.initialize_tdbg_nu2_density
    assert projected_hf.tdbg_density_from_hamiltonian is projected_hf_state.tdbg_density_from_hamiltonian
    assert projected_hf.tdbg_order_parameters is projected_hf_state.tdbg_order_parameters
    assert projected_hf._conventional_projector_to_stored is projected_hf_state._conventional_projector_to_stored
    assert projected_hf._stored_to_conventional is projected_hf_state._stored_to_conventional
    assert projected_hf._reference_subtracted_tdbg_density is projected_hf_state._reference_subtracted_tdbg_density
    assert projected_hf._hartree_density_for_policy is projected_hf_state._hartree_density_for_policy
    assert projected_hf._fock_density_for_policy is projected_hf_state._fock_density_for_policy
    assert tdbg_system.TDBGProjectedHFData is projected_hf_state.TDBGProjectedHFData
