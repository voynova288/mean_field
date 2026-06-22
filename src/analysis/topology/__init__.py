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

__all__ = [
    "DirectBandGapReport",
    "LatticeTopologyResult",
    "LinkMethod",
    "LinkVariables",
    "SewingTransform",
    "WavefunctionIndex",
    "adjacent_direct_gap_reports",
    "berry_curvature_from_links",
    "chern_number_from_berry_curvature",
    "compute_lattice_topology",
    "compute_lattice_topology_for_state_groups",
    "compute_link_variables",
    "default_k_grid_frac",
    "direct_band_gap_report",
    "matrix_sewing_transform",
    "normalize_state_indices",
    "select_wavefunction_subspace",
    "split_state_indices_by_direct_gaps",
    "wavefunction_index_for_state_group",
]
