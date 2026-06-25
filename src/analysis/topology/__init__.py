"""Common topology and quantum-geometry public API."""

from . import core as _core
from . import system as _system
from . import wavefunction as _wavefunction

_CORE = (
    "DirectBandGapReport", "LatticeTopologyResult", "LinkMethod", "LinkVariables", "SewingTransform", "WavefunctionIndex",
    "adjacent_direct_gap_reports", "berry_curvature_from_links", "chern_number_from_berry_curvature", "compute_lattice_topology",
    "compute_lattice_topology_for_state_groups", "compute_link_variables", "default_k_grid_frac", "direct_band_gap_report",
    "matrix_sewing_transform", "normalize_state_indices", "select_wavefunction_subspace", "split_state_indices_by_direct_gaps",
    "wavefunction_index_for_state_group",
)
_SYSTEM = tuple(_system.__all__)
_WAVEFUNCTION = tuple(_wavefunction.__all__)

for _module, _names in (
    (_core, _CORE),
    (_system, _SYSTEM),
    (_wavefunction, _WAVEFUNCTION),
):
    globals().update({name: getattr(_module, name) for name in _names})

__all__ = [*_CORE, *_SYSTEM, *_WAVEFUNCTION]

del _core, _system, _wavefunction, _module, _names
