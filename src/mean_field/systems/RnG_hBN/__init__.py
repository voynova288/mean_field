"""Rhombohedral L-layer graphene on hBN noninteracting continuum model."""

from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .charge_background import ChargeBackgroundResult, compute_valence_charge_background
from .hamiltonian import (
    MoireCouplingEntry,
    basis_index,
    build_coupling_table,
    build_hamiltonian,
    build_rlg_block,
    diagonalize_hamiltonian,
    dirac_block,
    flat_band_indices,
    hamiltonian_dimension,
    interlayer_coupling,
    layer_slice,
    moire_coupling_matrix,
    moire_potential,
    valence_band_count,
)
from .lattice import RLGhBNLattice, build_kpath_from_nodes, build_moire_k_grid, build_rlg_hbn_lattice, build_standard_kpath
from .model import RLGhBNModel
from .params import MOIRE_PARAMETER_TABLE, RLGhBNParams, table_ii_moire_parameters
from .plot import RLGhBNPathPlotTrace, path_bandwidth_mev, write_rlg_hbn_path_band_plot
from .topology import TopologyResult, compute_topology_from_eigenvectors, compute_topology_from_grid_result, compute_topology_on_grid
from .validation import ValidationCheck, ValidationReport, reproduce_paper_checkpoints, validate_physics

__all__ = [
    "ChargeBackgroundResult",
    "GridBandsResult",
    "MOIRE_PARAMETER_TABLE",
    "MoireCouplingEntry",
    "PathBandsResult",
    "RLGhBNLattice",
    "RLGhBNModel",
    "RLGhBNParams",
    "RLGhBNPathPlotTrace",
    "TopologyResult",
    "ValidationCheck",
    "ValidationReport",
    "basis_index",
    "build_coupling_table",
    "build_hamiltonian",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_rlg_block",
    "build_rlg_hbn_lattice",
    "build_standard_kpath",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "compute_valence_charge_background",
    "diagonalize_hamiltonian",
    "dirac_block",
    "flat_band_indices",
    "hamiltonian_dimension",
    "interlayer_coupling",
    "layer_slice",
    "moire_coupling_matrix",
    "moire_potential",
    "path_bandwidth_mev",
    "reproduce_paper_checkpoints",
    "table_ii_moire_parameters",
    "validate_physics",
    "valence_band_count",
    "write_rlg_hbn_path_band_plot",
]
