from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
    compute_oda_parameter,
)
from mean_field.core.hf.flavors import (
    FlavorBandData,
    block_mask,
    build_flavor_band_data,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    project_to_flavor_diagonal,
    project_to_flavor_diagonal_inplace,
)
from mean_field.core.hf.overlap import (
    HFOverlapBlockSet,
    compute_density_overlap_trace_from_diagonal,
    contract_fock_term_from_overlap,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)
from mean_field.core.hf.interaction import (
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    compute_hf_energy,
    empty_overlap_block_set,
)
from mean_field.core.hf.occupations import (
    calculate_norm_convergence,
    conventional_projector_to_stored,
    find_chemical_potential,
    occupied_state_linear_indices as _occupied_state_linear_indices,
    occupied_state_mask as _occupied_state_mask,
    stored_projector_to_conventional,
)

__all__ = [name for name in globals() if not name.startswith('__')]
