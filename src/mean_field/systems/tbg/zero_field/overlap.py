from __future__ import annotations

import numpy as np

from mean_field.core.hf.overlap import (
    ProjectedWavefunctionBasis,
    calculate_projected_overlap,
    calculate_projected_overlap_between,
    calculate_projected_overlap_compact,
)
from .model import BMSolution


def projected_basis_from_bm_solution(solution: BMSolution) -> ProjectedWavefunctionBasis:
    boundary_mode = "periodic" if bool(solution.periodic_g_grid) else "zero_fill"
    return ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(solution.uk, dtype=np.complex128),
        grid_shape=(solution.lg, solution.lg),
        n_spin=solution.n_spin,
        local_basis_size=solution.nlocal,
        name="tbg_bm",
        boundary_mode=boundary_mode,
    )


def calculate_overlap_compact(solution: BMSolution, m: int, n: int, *, valley_index: int = 0) -> np.ndarray:
    return calculate_projected_overlap_compact(
        projected_basis_from_bm_solution(solution),
        m,
        n,
        flavor_index=valley_index,
    )


def calculate_overlap_between(target: BMSolution, source: BMSolution, m: int, n: int) -> np.ndarray:
    return calculate_projected_overlap_between(
        projected_basis_from_bm_solution(target),
        projected_basis_from_bm_solution(source),
        m,
        n,
    )


def calculate_overlap(solution: BMSolution, m: int, n: int) -> np.ndarray:
    return calculate_projected_overlap(projected_basis_from_bm_solution(solution), m, n)
