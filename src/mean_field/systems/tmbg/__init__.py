"""Twisted monolayer-bilayer graphene noninteracting model."""

from .bands import GridBandsResult, PathBandsResult, compute_bands_along_path, compute_bands_on_grid
from .hamiltonian import (
    MoireCouplingEntry,
    blg_interlayer,
    build_coupling_table,
    build_diagonal_block,
    build_hamiltonian,
    diagonalize_hamiltonian,
    dirac_block,
    moire_coupling_matrix,
)
from .lattice import TMBGLattice, build_kpath_from_nodes, build_moire_k_grid, build_standard_kpath, build_tmbg_lattice
from .model import TMBGModel
from .params import TMBGParameters, VALID_BERNAL_CONVENTIONS, VALID_BLG_STACKINGS, default_omega, default_omega_prime, hopping_to_velocity
from .plot import (
    TMBGBandPlotPanel,
    infer_flat_band_indices,
    write_tmbg_band_plot,
    write_tmbg_berry_curvature_plot,
    write_tmbg_lattice_plot,
    write_tmbg_paper_band_figure,
)
from .topology import TopologyResult, compute_topology_from_eigenvectors, compute_topology_from_grid_result, compute_topology_on_grid
from .validation import (
    ValidationCheck,
    ValidationReport,
    diagnose_ktilde_symmetry,
    validate_physics,
)

__all__ = [
    "GridBandsResult",
    "MoireCouplingEntry",
    "PathBandsResult",
    "TMBGModel",
    "TMBGParameters",
    "TMBGLattice",
    "TMBGBandPlotPanel",
    "TopologyResult",
    "ValidationCheck",
    "ValidationReport",
    "VALID_BERNAL_CONVENTIONS",
    "VALID_BLG_STACKINGS",
    "blg_interlayer",
    "build_coupling_table",
    "build_diagonal_block",
    "build_hamiltonian",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_standard_kpath",
    "build_tmbg_lattice",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "default_omega",
    "default_omega_prime",
    "diagnose_ktilde_symmetry",
    "diagonalize_hamiltonian",
    "dirac_block",
    "hopping_to_velocity",
    "infer_flat_band_indices",
    "moire_coupling_matrix",
    "validate_physics",
    "write_tmbg_band_plot",
    "write_tmbg_berry_curvature_plot",
    "write_tmbg_lattice_plot",
    "write_tmbg_paper_band_figure",
]
