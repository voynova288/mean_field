"""Alternating-twist multilayer graphene noninteracting model."""

from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .bilayer_map import (
    ATMGSVDResult,
    MappedSpectrumResult,
    analytic_singular_values,
    build_W_matrix,
    build_atmg_via_tbg_sum,
    build_block_diagonal_mapped_hamiltonian,
    svd_decompose,
)
from .hamiltonian import build_diagonal_block, build_hamiltonian, diagonalize_hamiltonian
from .lattice import ATMGLattice, build_atmg_lattice, build_kpath_from_nodes, build_moire_k_grid, build_standard_kpath
from .model import ATMGModel
from .params import ATMGParameters
from .tbg import (
    TBGCouplingEntry,
    build_coupling_table,
    build_monolayer_hamiltonian,
    build_tbg_hamiltonian,
    diagonalize_tbg_hamiltonian,
    dirac_block,
    moire_coupling_matrix,
)
from .topology import TopologyResult, compute_topology_from_eigenvectors, compute_topology_from_grid_result, compute_topology_on_grid
from .validation import ValidationCheck, ValidationReport, reproduce_khalaf_checkpoints, validate_physics

__all__ = [
    "ATMGModel",
    "ATMGParameters",
    "ATMGLattice",
    "ATMGSVDResult",
    "GridBandsResult",
    "MappedSpectrumResult",
    "PathBandsResult",
    "TBGCouplingEntry",
    "TopologyResult",
    "ValidationCheck",
    "ValidationReport",
    "analytic_singular_values",
    "build_W_matrix",
    "build_atmg_lattice",
    "build_atmg_via_tbg_sum",
    "build_block_diagonal_mapped_hamiltonian",
    "build_coupling_table",
    "build_diagonal_block",
    "build_hamiltonian",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_monolayer_hamiltonian",
    "build_standard_kpath",
    "build_tbg_hamiltonian",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "diagonalize_hamiltonian",
    "diagonalize_tbg_hamiltonian",
    "dirac_block",
    "moire_coupling_matrix",
    "reproduce_khalaf_checkpoints",
    "svd_decompose",
    "validate_physics",
]
