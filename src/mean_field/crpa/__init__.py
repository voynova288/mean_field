"""Constrained RPA utilities for the zero-field TBG workflow."""

from .band_classifier import BandClassification, classify_flat_bands
from .bm import AllBandBMSolution, read_all_band_bm_solution, solve_all_band_bm_model, write_all_band_bm_solution
from .coulomb import CRPACoulombParams, coulomb_potential_mev, coulomb_potential_table_mev
from .dielectric import DielectricResult, compute_dielectric
from .diagnostics import write_all_epsilon_diagnostics, write_crpa_epsilon_diagnostics_csv
from .flat_remote import (
    apply_chi0_energy_mode,
    apply_eq19_flat_remote_correction,
    apply_hf_active_flat_basis,
    normalize_chi0_energy_mode,
)
from .grid import CRPAKGrid, build_q_shift_table, build_uniform_crpa_grid
from .susceptibility import (
    compute_constrained_chi0,
    compute_constrained_chi0_by_subtraction,
    compute_flat_flat_chi0,
    compute_full_chi0,
    constrained_sum_identity_error,
)
from .workflow import CRPAResult, compute_crpa, load_crpa_result, write_crpa_outputs
__all__ = [
    "AllBandBMSolution",
    "BandClassification",
    "CRPACoulombParams",
    "CRPAKGrid",
    "CRPAResult",
    "DielectricResult",
    "apply_chi0_energy_mode",
    "apply_eq19_flat_remote_correction",
    "apply_hf_active_flat_basis",
    "build_q_shift_table",
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
    "normalize_chi0_energy_mode",
    "read_all_band_bm_solution",
    "solve_all_band_bm_model",
    "write_all_band_bm_solution",
    "write_all_epsilon_diagnostics",
    "write_crpa_epsilon_diagnostics_csv",
    "write_crpa_outputs",
]
