"""Constrained RPA utilities for the zero-field TBG workflow."""

from .band_classifier import BandClassification, classify_flat_bands
from .bm import AllBandBMSolution, read_all_band_bm_solution, solve_all_band_bm_model, write_all_band_bm_solution
from .coulomb import CRPACoulombParams, coulomb_potential_mev, coulomb_potential_table_mev
from .dielectric import DielectricResult, compute_dielectric
from .diagnostics import write_all_epsilon_diagnostics, write_crpa_epsilon_diagnostics_csv
from .grid import CRPAKGrid, build_q_shift_table, build_uniform_crpa_grid
from .screened_coulomb import CRPAScreenedCoulomb
from .susceptibility import (
    compute_constrained_chi0,
    compute_constrained_chi0_by_subtraction,
    compute_flat_flat_chi0,
    compute_full_chi0,
    constrained_sum_identity_error,
)
from .workflow import CRPAResult, compute_crpa, load_crpa_result, write_crpa_outputs
from .hf_validation import has_full_crpa_q_table, validate_hf_compatible_crpa
from .hf_interface import (
    build_bare_split_full_hf_kernel,
    build_crpa_projected_interaction_components,
    build_crpa_projected_interaction_hamiltonian,
    build_crpa_projected_target_hamiltonian,
    build_fock_screened_overlap_blocks,
    build_full_crpa_hf_kernel,
    crpa_split_energy_functional,
    crpa_hf_energy_components,
    half_reference_delta_like,
    physical_projector_from_delta,
    run_bare_split_full_hartree_fock,
    run_full_crpa_hartree_fock,
    split_oda_parameter,
)

__all__ = [
    "AllBandBMSolution",
    "BandClassification",
    "CRPACoulombParams",
    "CRPAKGrid",
    "CRPAResult",
    "CRPAScreenedCoulomb",
    "DielectricResult",
    "build_q_shift_table",
    "build_bare_split_full_hf_kernel",
    "build_crpa_projected_interaction_components",
    "build_crpa_projected_interaction_hamiltonian",
    "build_crpa_projected_target_hamiltonian",
    "build_fock_screened_overlap_blocks",
    "build_full_crpa_hf_kernel",
    "build_uniform_crpa_grid",
    "classify_flat_bands",
    "compute_constrained_chi0",
    "compute_constrained_chi0_by_subtraction",
    "compute_crpa",
    "compute_dielectric",
    "compute_flat_flat_chi0",
    "compute_full_chi0",
    "constrained_sum_identity_error",
    "crpa_split_energy_functional",
    "coulomb_potential_mev",
    "coulomb_potential_table_mev",
    "crpa_hf_energy_components",
    "half_reference_delta_like",
    "has_full_crpa_q_table",
    "load_crpa_result",
    "physical_projector_from_delta",
    "read_all_band_bm_solution",
    "run_bare_split_full_hartree_fock",
    "run_full_crpa_hartree_fock",
    "solve_all_band_bm_model",
    "split_oda_parameter",
    "validate_hf_compatible_crpa",
    "write_all_band_bm_solution",
    "write_all_epsilon_diagnostics",
    "write_crpa_epsilon_diagnostics_csv",
    "write_crpa_outputs",
]
