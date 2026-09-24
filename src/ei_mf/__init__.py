"""Mean-field tools for reproducing the InAs/GaSb excitonic-insulator calculation."""

from .bands import effective_kF, kF_from_density_cm2, xi0_parabolic_mev
from .bfield import delta_k_parallel_nm_inv
from .gap_solver_stageA import StageAResult, solve_stageA, solve_stageA_case
from .params import EIParams
from .units import PAPER_CONVENTION

__all__ = [
    "EIParams",
    "PAPER_CONVENTION",
    "StageAResult",
    "delta_k_parallel_nm_inv",
    "effective_kF",
    "kF_from_density_cm2",
    "solve_stageA",
    "solve_stageA_case",
    "xi0_parabolic_mev",
]
