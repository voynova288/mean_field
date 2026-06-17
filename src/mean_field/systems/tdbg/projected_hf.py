from __future__ import annotations

import numpy as np

from .lattice import TDBGLattice, build_moire_k_grid
from .projected_hf_config import (
    SPIN_LABELS,
    TDBGInteractionSettings,
    TDBG_LOCAL_LABELS,
    TDBGPaperUdConvention,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
    VALID_PAPER_UD_CONVENTIONS,
    VALLEY_LABELS,
    VALLEY_SEQUENCE,
    tdbg_delta_from_paper_ud_for_valley,
    tdbg_parameters_from_paper_ud_for_valley,
    validate_tdbg_interaction_settings,
    validate_tdbg_projected_hf_config,
)
from .projected_hf_data import (
    _projected_onebody_and_wavefunctions,
    _projected_orbital_g_matrix,
    build_tdbg_projected_hf_data,
)
from .projected_hf_geometry import (
    _TDBGQSiteEmbedding,
    _shift_table,
    _tdbg_core_order_permutation,
    _tdbg_projected_wavefunction_basis,
    _tdbg_q_site_embedding,
    _tdbg_total_overlap_between,
    _tdbg_total_overlap_from_bases,
    tdbg_band_window_indices,
    tdbg_embedded_component_groups,
    tdbg_moire_area_nm2,
)
from .projected_hf_interactions import (
    TDBGProjectedHFInteractionBuilder,
    _local_lambda,
    _split_intersite_overlap_blocks,
    _stored_inner_ev,
    build_tdbg_interaction_builder,
    build_tdbg_interaction_components,
    build_tdbg_onsite_hamiltonian,
    build_tdbg_total_overlap_blocks,
    graphene_area_over_moire_area,
    tdbg_energy_components,
)
from .projected_hf_state import (
    TDBGProjectedHFData,
    TDBGProjectedHFDensityBuilder,
    TDBGProjectedHFInitializer,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    TDBGProjectedHFTargetData,
    TDBGStateLabel,
    _active_filling_indices,
    _conventional_projector_to_stored,
    _first_conduction_indices,
    _fock_density_for_policy,
    _hartree_density_for_policy,
    _numeric_order_parameters,
    _reference_projector,
    _reference_subtracted_tdbg_density,
    _stored_to_conventional,
    initialize_tdbg_density,
    initialize_tdbg_nu2_density,
    tdbg_density_from_hamiltonian,
    tdbg_order_parameters,
)
from .projected_hf_reports import (
    liu2022_default_projected_hf_config,
    liu2022_projected_hf_metadata,
    tdbg_hf_grid_band_summary,
)
from .projected_hf_run import (
    build_tdbg_projected_hf_kernel,
    build_tdbg_projected_hf_problem,
    build_tdbg_projected_hf_state,
    run_tdbg_projected_hf,
)
from .projected_hf_target import (
    _build_tdbg_target_overlap_block_sets,
    _local_lambda_from_wavefunctions,
    _target_source_total_overlap,
    _total_diagonal_overlap_from_wavefunctions,
    _with_fock_screening,
    build_tdbg_hf_target_hamiltonian,
    build_tdbg_onsite_target_hamiltonian,
    build_tdbg_projected_hf_target_data,
    diagonalize_tdbg_hf_target_hamiltonian,
)

def scan_tdbg_projected_hf_states(
    config: TDBGProjectedHFConfig,
    *,
    init_modes: tuple[str, ...] = ("sp", "sp_down", "vp_k", "vp_kprime", "ivc_even", "ivc_odd", "random"),
    seeds: tuple[int, ...] = (1, 2, 3),
) -> tuple[TDBGProjectedHFResult, ...]:
    data = build_tdbg_projected_hf_data(config)
    results: list[TDBGProjectedHFResult] = []
    for init_mode in init_modes:
        mode_seeds = seeds if init_mode.startswith("random") else (seeds[0],)
        for seed in mode_seeds:
            results.append(run_tdbg_projected_hf(data, init_mode=init_mode, seed=int(seed)))
    return tuple(results)
