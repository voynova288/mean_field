from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np

from ...core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockProblem,
    HartreeFockRun,
    ProjectedWavefunctionBasis,
    build_projected_hf_kernel,
    build_projected_hf_problem,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    find_chemical_potential,
    occupied_state_mask,
    random_unitary_from_hermitian,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb_matrix,
)
from ...core.supercell import IntegerSupercell, folded_band_count, occupied_count_from_primitive_filling
from .hamiltonian import centered_band_indices
from .lattice import HTGLattice, KPath, build_kpath_from_nodes
from .mean_field_adapter import (
    _hybrid_projected_basis_at_k,
    _layer_potential_operator,
    centered_projection_band_indices,
    hermitian_residual,
    projector_idempotency_residual,
    reciprocal_shift_labels,
)
from .model import HTGModel
from .params import InteractionParams
from .hamiltonian import sublattice_sigma_z

__all__ = [name for name in globals() if not name.startswith('__')]
