"""Unified Berry-geometry analysis framework.

Use :func:`compute_lattice_topology` for system-independent Berry connection,
Fukui-Hatsugai-Suzuki plaquette flux, and Chern-number calculations.  The only
system-specific inputs are the selected wavefunction columns, their index labels,
and optional boundary sewing transforms.
"""

from .core import (
    LatticeTopologyResult,
    LinkMethod,
    LinkVariables,
    SewingTransform,
    WavefunctionIndex,
    berry_curvature_from_links,
    chern_number_from_berry_curvature,
    compute_lattice_topology,
    compute_link_variables,
    default_k_grid_frac,
    matrix_sewing_transform,
    normalize_state_indices,
    select_wavefunction_subspace,
)

__all__ = [
    "LatticeTopologyResult",
    "LinkMethod",
    "LinkVariables",
    "SewingTransform",
    "WavefunctionIndex",
    "berry_curvature_from_links",
    "chern_number_from_berry_curvature",
    "compute_lattice_topology",
    "compute_link_variables",
    "default_k_grid_frac",
    "matrix_sewing_transform",
    "normalize_state_indices",
    "select_wavefunction_subspace",
]
