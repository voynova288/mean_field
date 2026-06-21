from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from ....core.hf import (
    DensityUpdateResult,
    FlavorBandData,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    block_mask,
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    build_flavor_band_data,
    calculate_norm_convergence,
    compute_density_overlap_trace_from_diagonal,
    compute_hf_energy,
    compute_oda_parameter,
    contract_fock_term_from_overlap,
    empty_overlap_block_set,
    find_chemical_potential,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    occupied_state_linear_indices as _occupied_state_linear_indices,
    occupied_state_mask as _occupied_state_mask,
    project_to_flavor_diagonal,
    project_to_flavor_diagonal_inplace,
    run_hartree_fock_problem,
)

__all__ = [name for name in globals() if not name.startswith('__')]
