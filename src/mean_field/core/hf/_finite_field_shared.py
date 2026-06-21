"""Generic finite-magnetic-field Hartree-Fock machinery.

This module owns the system-independent B-field HF calculation: finite-field
state containers, density initialization/update, screened interaction kernels,
full magnetic-BZ and magnetic-translation-reduced contractions, SCF problem
assembly, and compact summaries.  Physical-system layers should supply the
projected Hofstadter/spectrum arrays and overlap blocks, then call this module
instead of reimplementing finite-B HF in a system package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np

from ..magnetic_field import (
    MagneticFlux,
    diophantine_filling,
    in_hex_shell,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_reciprocal_vector,
)
from .engine import DensityUpdateResult, HartreeFockRun
from .interaction import build_projected_hf_kernel, build_projected_interaction_hamiltonian
from .occupations import (
    calculate_norm_convergence,
    find_chemical_potential,
    occupied_state_linear_indices,
)
from .overlap import (
    HFOverlapBlockSet,
    compute_density_overlap_trace_from_diagonal,
    contract_fock_term_from_overlap,
    diagonal_overlap_blocks,
)
from .problem import HartreeFockKernel, HartreeFockProblem, run_hartree_fock_problem

Array = np.ndarray
InitMode = Literal["bm", "random", "flavor", "bm_cascade", "sublattice"]

__all__ = [name for name in globals() if not name.startswith('__')]
