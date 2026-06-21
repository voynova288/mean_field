from __future__ import annotations

"""Compatibility facade for HTG folded-supercell HF helpers."""

from ._supercell_types import (
    HTGSupercell,
    HTGSupercellGroundStateScan,
    HTGSupercellHFWavefunctionGrid,
    HTGSupercellHartreeFockRun,
    HTGSupercellHartreeFockState,
    HTGSupercellPathResult,
    HTGSupercellProjectedBasisData,
    HTGSupercellSCFGridPathSamples,
)
from ._supercell_geometry import (
    build_htg_supercell_hf_wavefunction_grid,
    build_htg_supercell_uniform_grid,
    extract_htg_supercell_inspection_scf_grid_path,
    extract_htg_supercell_scf_grid_path,
    htg_common_area6_fractional_supercell,
    htg_default_fractional_supercell,
    htg_doubled_fractional_supercell,
    htg_minimal_fractional_supercell,
    htg_supercell_full_boundary_sewing_transform,
    htg_supercell_full_boundary_sewing_transforms,
    htg_tripled_fractional_supercell,
    supercell_fold_representatives,
)
from ._supercell_basis import (
    _supercell_reference_density_blocks,
    build_htg_supercell_overlap_blocks,
    build_htg_supercell_overlap_blocks_between,
    build_htg_supercell_projected_basis,
    build_htg_supercell_projected_basis_for_kvec,
    htg_supercell_filling_from_density,
    htg_supercell_occupied_count_per_k,
    htg_supercell_reference_diagonal,
)
from ._supercell_runner import (
    build_htg_supercell_hf_kernel,
    build_htg_supercell_hf_problem,
    initialize_htg_supercell_density,
    run_htg_supercell_hf,
    scan_htg_supercell_ground_state,
)
from ._supercell_archive import load_htg_supercell_hf_run_from_archive
from ._supercell_path_io import (
    build_htg_supercell_gamma_path,
    evaluate_htg_supercell_hf_path,
    save_htg_supercell_path_npz,
    save_htg_supercell_run_npz,
)

__all__ = [
    "HTGSupercell",
    "HTGSupercellGroundStateScan",
    "HTGSupercellHFWavefunctionGrid",
    "HTGSupercellHartreeFockRun",
    "HTGSupercellHartreeFockState",
    "HTGSupercellPathResult",
    "HTGSupercellProjectedBasisData",
    "HTGSupercellSCFGridPathSamples",
    "build_htg_supercell_gamma_path",
    "build_htg_supercell_hf_kernel",
    "build_htg_supercell_hf_problem",
    "build_htg_supercell_hf_wavefunction_grid",
    "build_htg_supercell_overlap_blocks",
    "build_htg_supercell_overlap_blocks_between",
    "build_htg_supercell_projected_basis",
    "build_htg_supercell_projected_basis_for_kvec",
    "build_htg_supercell_uniform_grid",
    "evaluate_htg_supercell_hf_path",
    "extract_htg_supercell_inspection_scf_grid_path",
    "extract_htg_supercell_scf_grid_path",
    "htg_common_area6_fractional_supercell",
    "htg_default_fractional_supercell",
    "htg_doubled_fractional_supercell",
    "htg_minimal_fractional_supercell",
    "htg_supercell_filling_from_density",
    "htg_supercell_full_boundary_sewing_transform",
    "htg_supercell_full_boundary_sewing_transforms",
    "htg_supercell_occupied_count_per_k",
    "htg_supercell_reference_diagonal",
    "htg_tripled_fractional_supercell",
    "initialize_htg_supercell_density",
    "load_htg_supercell_hf_run_from_archive",
    "run_htg_supercell_hf",
    "save_htg_supercell_path_npz",
    "save_htg_supercell_run_npz",
    "scan_htg_supercell_ground_state",
    "supercell_fold_representatives",
]
