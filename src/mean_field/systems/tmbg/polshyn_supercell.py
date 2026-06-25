from __future__ import annotations

"""Compatibility facade for Polshyn tMBG doubled-cell HF helpers."""

from ._polshyn_types import (
    PolshynDoubledCell,
    PolshynFillingSummary,
    PolshynProjectedBasis,
    PolshynWangHFState,
    polshyn_doubled_cell,
)
from ._polshyn_basis import build_polshyn_projected_basis
from ._polshyn_contracts import PolshynRunHFConfig, polshyn_wang_hf_bundle_to_hf_run_result, run_tmbg_polshyn_hf_config_adapter
from ._polshyn_filling import (
    cdw_density_blocks,
    occupation_counts_nu_7over2,
    polshyn_nu_7over2_filling_summary,
    primitive_nu_from_counts,
    reference_diagonal_for_projected_indices,
)
from ._polshyn_h0 import (
    PolshynH0SubtractionConfig,
    PolshynH0SubtractionResult,
    apply_polshyn_h0_subtraction,
    basis_with_polshyn_h0_correction,
    compute_polshyn_active_reference_h0_correction,
    compute_polshyn_minus_full_p0_h0_correction,
    polshyn_reference_projector_blocks,
)
from ._polshyn_reconstruction import reconstruct_polshyn_wang_hf_micro_wavefunctions
from ._polshyn_wang import (
    build_wang_hf_problem,
    build_wang_overlap_blocks,
    estimate_fermi_level_from_sector_energies,
    flatten_sector_blocks,
    moire_cell_area_nm2,
    overlap_blocks_with_hartree_q0_zeroed,
    run_projected_hf_scf_wang,
    scaled_overlap_blocks,
    translation_order_parameters,
    unflatten_sector_blocks,
    unflatten_sector_energies,
    wang_density_from_fixed_sector_occupations,
    wang_projected_wavefunction_basis,
    wang_sector_density_blocks,
    wang_sector_energy_blocks,
    wang_sector_hamiltonian_blocks,
    wang_stored_density_from_sector_blocks,
)

__all__ = [
    "PolshynDoubledCell",
    "PolshynFillingSummary",
    "PolshynH0SubtractionConfig",
    "PolshynH0SubtractionResult",
    "PolshynProjectedBasis",
    "PolshynRunHFConfig",
    "PolshynWangHFState",
    "apply_polshyn_h0_subtraction",
    "basis_with_polshyn_h0_correction",
    "build_polshyn_projected_basis",
    "build_wang_hf_problem",
    "build_wang_overlap_blocks",
    "cdw_density_blocks",
    "compute_polshyn_active_reference_h0_correction",
    "compute_polshyn_minus_full_p0_h0_correction",
    "estimate_fermi_level_from_sector_energies",
    "flatten_sector_blocks",
    "moire_cell_area_nm2",
    "occupation_counts_nu_7over2",
    "overlap_blocks_with_hartree_q0_zeroed",
    "polshyn_doubled_cell",
    "polshyn_nu_7over2_filling_summary",
    "polshyn_reference_projector_blocks",
    "polshyn_wang_hf_bundle_to_hf_run_result",
    "primitive_nu_from_counts",
    "reconstruct_polshyn_wang_hf_micro_wavefunctions",
    "reference_diagonal_for_projected_indices",
    "run_projected_hf_scf_wang",
    "run_tmbg_polshyn_hf_config_adapter",
    "scaled_overlap_blocks",
    "translation_order_parameters",
    "unflatten_sector_blocks",
    "unflatten_sector_energies",
    "wang_density_from_fixed_sector_occupations",
    "wang_projected_wavefunction_basis",
    "wang_sector_density_blocks",
    "wang_sector_energy_blocks",
    "wang_sector_hamiltonian_blocks",
    "wang_stored_density_from_sector_blocks",
]
