"""Core reusable building blocks for mean-field solvers."""

from .magnetic_field import (
    MagneticFlux,
    choose_magnetic_nq,
    diophantine_branch_cases,
    diophantine_filling,
    in_hex_shell,
    magnetic_k_vectors,
    magnetic_normalization_count,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_reciprocal_vector,
    magnetic_shell_shifts,
)

__all__ = [
    "MagneticFlux",
    "choose_magnetic_nq",
    "diophantine_branch_cases",
    "diophantine_filling",
    "in_hex_shell",
    "magnetic_k_vectors",
    "magnetic_normalization_count",
    "magnetic_orbit_indices",
    "magnetic_r_orbit_positions",
    "magnetic_reciprocal_vector",
    "magnetic_shell_shifts",
]
