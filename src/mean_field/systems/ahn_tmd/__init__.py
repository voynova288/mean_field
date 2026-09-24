"""Ahn et al. 2024 tMoTe2 system adapter for Mean_Field.

This package owns only the physical continuum model and basis conventions.
HF/SCHF and topology must be connected through existing generic Mean_Field
interfaces; do not keep system-local HF loops or topology algorithms here.
"""

from .bands import AhnTMDGridBands, AhnTMDPathBands, compute_bands_along_path, compute_bands_on_rectangular_grid
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .hf import (
    AhnTMDHFConfig,
    AhnTMDHFState,
    AhnTMDProjectedHFData,
    ahn_tmd_density_from_fixed_hole_occupations,
    build_ahn_tmd_hf_state,
    build_ahn_tmd_projected_hf_data,
    run_ahn_tmd_projected_hf,
)
from .lattice import AhnTMDLattice, build_ahn_tmd_lattice, build_moire_k_grid, build_rectangular_moire_k_grid, hex_shell_indices, standard_kpath
from .model import AhnTMDModel
from .params import AhnTMDParameters, DEFAULT_AHN_TMD_PARAMETERS

__all__ = [
    "AhnTMDGridBands",
    "AhnTMDHFConfig",
    "AhnTMDHFState",
    "AhnTMDLattice",
    "AhnTMDModel",
    "AhnTMDParameters",
    "AhnTMDPathBands",
    "AhnTMDProjectedHFData",
    "DEFAULT_AHN_TMD_PARAMETERS",
    "ahn_tmd_density_from_fixed_hole_occupations",
    "build_ahn_tmd_hf_state",
    "build_ahn_tmd_lattice",
    "build_ahn_tmd_projected_hf_data",
    "build_hamiltonian",
    "build_moire_k_grid",
    "build_rectangular_moire_k_grid",
    "compute_bands_along_path",
    "compute_bands_on_rectangular_grid",
    "diagonalize_hamiltonian",
    "hex_shell_indices",
    "run_ahn_tmd_projected_hf",
    "standard_kpath",
]
