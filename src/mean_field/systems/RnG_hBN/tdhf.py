"""Compatibility facade for RLG/hBN TDHF/RPA helpers."""

from __future__ import annotations

from ._tdhf_support import (
    RLGhBNTDHFFiniteQSupport,
    rlg_hbn_tdhf_finite_q_mode_support,
)
from ._tdhf_types import (
    RLGhBNTDHFInteraction,
    RLGhBNTDHFMomentumShift,
    RLGhBNTDHFOrbitals,
)
from ._tdhf_orbitals import (
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_orbitals_from_canonical_hf,
    validate_rlg_hbn_tdhf_canonical_orbital_parity,
)
from ._tdhf_pairs import (
    build_rlg_hbn_tdhf_interaction,
    build_rlg_hbn_tdhf_q0_pairs,
    build_rlg_hbn_tdhf_q_pairs,
    required_rlg_hbn_tdhf_finite_q_overlap_shifts,
    required_rlg_hbn_tdhf_full_finite_q_overlap_shifts,
)
from ._tdhf_archive import load_rlg_hbn_tdhf_run_from_archive
from ._tdhf_q0 import build_rlg_hbn_tdhf_q0_matrices_from_pairs
from ._tdhf_fixed_quotient import (
    RLGhBNTDHFFiniteQQuotientContext,
    build_rlg_hbn_tdhf_finite_q_quotient_context,
)
from ._tdhf_finite_q import (
    RLGhBNTDHFFiniteQQuotientMatrixPair,
    build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs,
    build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs,
)
from ._tdhf_quotient_orbit import (
    RLGhBNTDHFQuotientCycleResult,
    RLGhBNTDHFQuotientOrbitResult,
    build_rlg_hbn_tdhf_c3_quotient_cycle,
    build_rlg_hbn_tdhf_c3_quotient_orbit,
)
from ._tdhf_dispatch import (
    build_rlg_hbn_tdhf_q_matrices,
    build_rlg_hbn_tdhf_q_matrices_from_canonical_hf,
    build_rlg_hbn_tdhf_q0_matrices,
    build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf,
)

__all__ = [
    "RLGhBNTDHFInteraction",
    "RLGhBNTDHFFiniteQSupport",
    "RLGhBNTDHFMomentumShift",
    "RLGhBNTDHFOrbitals",
    "RLGhBNTDHFQuotientCycleResult",
    "RLGhBNTDHFQuotientOrbitResult",
    "RLGhBNTDHFFiniteQQuotientContext",
    "RLGhBNTDHFFiniteQQuotientMatrixPair",
    "build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs",
    "build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs",
    "build_rlg_hbn_tdhf_finite_q_quotient_context",
    "build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs",
    "build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs",
    "build_rlg_hbn_tdhf_c3_quotient_cycle",
    "build_rlg_hbn_tdhf_c3_quotient_orbit",
    "build_rlg_hbn_tdhf_interaction",
    "build_rlg_hbn_tdhf_orbitals",
    "build_rlg_hbn_tdhf_orbitals_from_canonical_hf",
    "build_rlg_hbn_tdhf_q_matrices",
    "build_rlg_hbn_tdhf_q_matrices_from_canonical_hf",
    "build_rlg_hbn_tdhf_q_pairs",
    "build_rlg_hbn_tdhf_q0_matrices",
    "build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf",
    "build_rlg_hbn_tdhf_q0_matrices_from_pairs",
    "build_rlg_hbn_tdhf_q0_pairs",
    "load_rlg_hbn_tdhf_run_from_archive",
    "required_rlg_hbn_tdhf_finite_q_overlap_shifts",
    "required_rlg_hbn_tdhf_full_finite_q_overlap_shifts",
    "rlg_hbn_tdhf_finite_q_mode_support",
    "validate_rlg_hbn_tdhf_canonical_orbital_parity",
]
