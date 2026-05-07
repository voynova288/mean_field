"""Constrained RPA utilities for the zero-field TBG workflow."""

from .band_classifier import BandClassification, classify_flat_bands
from .bm import AllBandBMSolution, read_all_band_bm_solution, solve_all_band_bm_model, write_all_band_bm_solution
from .coulomb import CRPACoulombParams, coulomb_potential_mev, coulomb_potential_table_mev
from .dielectric import DielectricResult, compute_dielectric
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
from .hf_interface import (
    build_crpa_projected_interaction_hamiltonian,
    build_fock_screened_overlap_blocks,
    build_full_crpa_hf_kernel,
    run_full_crpa_hartree_fock,
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
    "build_crpa_projected_interaction_hamiltonian",
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
    "coulomb_potential_mev",
    "coulomb_potential_table_mev",
    "load_crpa_result",
    "read_all_band_bm_solution",
    "run_full_crpa_hartree_fock",
    "solve_all_band_bm_model",
    "write_all_band_bm_solution",
    "write_crpa_outputs",
]
