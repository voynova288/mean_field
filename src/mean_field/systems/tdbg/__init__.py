"""Twisted double bilayer graphene noninteracting model."""

from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import (
    build_bilayer_block,
    build_hamiltonian,
    build_site_block,
    diagonalize_hamiltonian,
    moire_coupling_matrix,
)
from .lattice import TDBGLattice, build_kpath_from_nodes, build_moire_k_grid, build_standard_kpath, build_tdbg_lattice
from .model import TDBGModel
from .params import TDBGParameters, delta_from_paper_ud, layer_potentials_from_delta, paper_ud_layer_potentials
from .plot import TDBGPathPlotTrace, write_tdbg_path_band_plot
from .topology import TopologyResult, compute_topology_from_eigenvectors, compute_topology_from_grid_result, compute_topology_on_grid
from .validation import (
    ReferenceComparisonResult,
    ValidationCheck,
    ValidationReport,
    compare_against_pytwist_reference,
    validate_physics,
)

__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "ReferenceComparisonResult",
    "TDBGModel",
    "TDBGLattice",
    "TDBGParameters",
    "TDBGPathPlotTrace",
    "TopologyResult",
    "ValidationCheck",
    "ValidationReport",
    "build_bilayer_block",
    "build_hamiltonian",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_site_block",
    "build_standard_kpath",
    "build_tdbg_lattice",
    "compare_against_pytwist_reference",
    "delta_from_paper_ud",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "diagonalize_hamiltonian",
    "layer_potentials_from_delta",
    "moire_coupling_matrix",
    "paper_ud_layer_potentials",
    "validate_physics",
    "write_tdbg_path_band_plot",
]
