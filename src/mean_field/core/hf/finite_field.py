"""Compatibility facade for generic finite-magnetic-field Hartree-Fock helpers."""

from __future__ import annotations

from ._finite_field_shared import InitMode, MagneticFlux
from ._finite_field_types import (
    FiniteFieldHartreeFockInputBundle,
    FiniteFieldHartreeFockInputs,
    FiniteFieldHartreeFockState,
    FiniteFieldHartreeFockSummary,
    FiniteFieldTLSymmetricHartreeFockInputs,
    MagneticOverlapData,
)
from ._finite_field_initialization import (
    build_h0_from_hofstadter_metadata,
    coulomb_unit_from_lattice,
    density_update_from_hamiltonian,
    finite_field_diophantine_filling,
    finite_field_filling,
    finite_field_occupied_state_count,
    initialize_density_from_h0,
    normalize_finite_field_init_mode,
    screened_coulomb_finite_b,
    state_index,
    zeeman_unit_from_area,
)
from ._finite_field_interaction import (
    apply_iks_phase_to_transposed_density,
    build_magnetic_hf_overlap_block_set,
    build_magnetic_interaction_hamiltonian,
    build_tl_symmetric_magnetic_interaction_hamiltonian,
    compute_finite_field_hf_energy,
    expand_valley_overlap_data_to_flavors,
)
from ._finite_field_kernel import (
    build_finite_field_hf_kernel,
    build_finite_field_hf_kernel_from_inputs,
    build_finite_field_hf_problem,
    build_tl_symmetric_finite_field_hf_kernel,
    build_tl_symmetric_finite_field_hf_kernel_from_inputs,
    calculate_valley_spin_order_parameters,
    run_finite_field_hartree_fock,
    run_finite_field_hartree_fock_from_inputs,
    run_tl_symmetric_finite_field_hartree_fock_from_inputs,
    summarize_finite_field_hartree_fock,
)

__all__ = [
    'FiniteFieldHartreeFockInputBundle',
    'FiniteFieldHartreeFockInputs',
    'FiniteFieldHartreeFockState',
    'FiniteFieldHartreeFockSummary',
    'FiniteFieldTLSymmetricHartreeFockInputs',
    'InitMode',
    'MagneticFlux',
    'MagneticOverlapData',
    'apply_iks_phase_to_transposed_density',
    'build_finite_field_hf_kernel',
    'build_finite_field_hf_kernel_from_inputs',
    'build_finite_field_hf_problem',
    'build_h0_from_hofstadter_metadata',
    'build_magnetic_hf_overlap_block_set',
    'build_magnetic_interaction_hamiltonian',
    'build_tl_symmetric_finite_field_hf_kernel',
    'build_tl_symmetric_finite_field_hf_kernel_from_inputs',
    'build_tl_symmetric_magnetic_interaction_hamiltonian',
    'calculate_valley_spin_order_parameters',
    'compute_finite_field_hf_energy',
    'coulomb_unit_from_lattice',
    'density_update_from_hamiltonian',
    'expand_valley_overlap_data_to_flavors',
    'finite_field_diophantine_filling',
    'finite_field_filling',
    'finite_field_occupied_state_count',
    'initialize_density_from_h0',
    'normalize_finite_field_init_mode',
    'run_finite_field_hartree_fock',
    'run_finite_field_hartree_fock_from_inputs',
    'run_tl_symmetric_finite_field_hartree_fock_from_inputs',
    'screened_coulomb_finite_b',
    'state_index',
    'summarize_finite_field_hartree_fock',
    'zeeman_unit_from_area',
]

for _name in __all__:
    _obj = globals()[_name]
    if getattr(_obj, "__module__", "").startswith("mean_field.core.hf._finite_field_"):
        _obj.__module__ = __name__
del _name, _obj
