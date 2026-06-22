"""Minimal common topology helpers.

This restored public surface intentionally exposes only system-independent
Fukui-Hatsugai-Suzuki link/plaquette/Chern utilities. System wrappers and
quantum-metric/QGT helpers remain archived until they are reviewed separately.
"""

from .core import (
    DirectBandGapReport,
    LatticeTopologyResult,
    LinkMethod,
    LinkVariables,
    SewingTransform,
    WavefunctionIndex,
    adjacent_direct_gap_reports,
    berry_curvature_from_links,
    chern_number_from_berry_curvature,
    compute_lattice_topology,
    compute_lattice_topology_for_state_groups,
    compute_link_variables,
    default_k_grid_frac,
    direct_band_gap_report,
    matrix_sewing_transform,
    normalize_state_indices,
    select_wavefunction_subspace,
    split_state_indices_by_direct_gaps,
    wavefunction_index_for_state_group,
)
from .system import (
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    topology_result_from_lattice_result,
)
from .wavefunction import (
    CanonicalWavefunctionGrid,
    WavefunctionLayout,
    canonicalize_wavefunction_grid,
    reshape_flat_mesh_to_grid,
    wavefunction_index_from_state_labels,
)

__all__ = [
    "CanonicalWavefunctionGrid",
    "DirectBandGapReport",
    "LatticeTopologyResult",
    "LinkMethod",
    "LinkVariables",
    "SewingTransform",
    "TopologyResult",
    "WavefunctionIndex",
    "WavefunctionLayout",
    "adjacent_direct_gap_reports",
    "berry_curvature_from_links",
    "chern_number_from_berry_curvature",
    "compute_lattice_topology",
    "compute_lattice_topology_for_state_groups",
    "compute_system_topology_from_eigenvectors",
    "compute_system_topology_from_grid_result",
    "compute_link_variables",
    "canonicalize_wavefunction_grid",
    "default_k_grid_frac",
    "direct_band_gap_report",
    "matrix_sewing_transform",
    "normalize_state_indices",
    "reshape_flat_mesh_to_grid",
    "select_wavefunction_subspace",
    "split_state_indices_by_direct_gaps",
    "topology_result_from_lattice_result",
    "wavefunction_index_for_state_group",
    "wavefunction_index_from_state_labels",
]
