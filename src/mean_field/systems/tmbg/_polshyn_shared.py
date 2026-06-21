from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from ...core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from ...core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    ProjectedWavefunctionBasis,
    build_projected_interaction_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    density_from_fixed_sector_occupations as _core_density_from_fixed_sector_occupations,
    diagonal_overlap_blocks,
    flat_sector_indices as _core_flat_sector_indices,
    flatten_sector_blocks as _core_flatten_sector_blocks,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb,
    screened_coulomb_matrix,
    unflatten_sector_blocks as _core_unflatten_sector_blocks,
    unflatten_sector_energies as _core_unflatten_sector_energies,
)
from ...core.hf.contracts_bridge import density_state_from_delta
from ...core.supercell import (
    IntegerSupercell,
    fixed_sector_occupation_counts,
    folded_indices_for_primitive_band,
    folded_reference_diagonal_by_primitive_index,
    primitive_filling_from_occupation_counts,
)
from .lattice import TMBGLattice

__all__ = [name for name in globals() if not name.startswith('__')]
