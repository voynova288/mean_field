from __future__ import annotations

from dataclasses import replace
import os

import numpy as np

from ...core.hf import (
    compute_oda_parameter,
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    build_projected_interaction_hamiltonian,
    flavor_block_indices,
    run_hartree_fock_problem,
)
from ...core.hf.overlap import compute_density_overlap_trace_from_diagonal, contract_fock_term_from_overlap
from ...systems.tbg.params import TBGParameters
from ...systems.tbg.zero_field.hf import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    _screened_coulomb_matrix,
    _with_tbg_overlap_screening,
    build_full_density_from_hamiltonian,
    coulomb_unit,
    initialize_full_state,
    normalize_full_init_mode,
    occupied_sigma_mean,
    offdiag_flavor_norm,
    restricted_filling,
    restricted_gap_estimate,
)
from ..screened_coulomb import CRPAScreenedCoulomb

__all__ = [name for name in globals() if name not in {"annotations", "__builtins__"}]
