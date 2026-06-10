"""Core reusable building blocks for mean-field solvers."""

from .bands import (
    GridBandsResult,
    PathBandsResult,
    resolve_n_bands,
    solve_bands_along_path,
    solve_bands_on_grid,
)
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
from .validation import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
    format_validation_value,
    status_from_bool,
    validate_valley,
)

__all__ = [
    "GridBandsResult",
    "MagneticFlux",
    "PathBandsResult",
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "choose_magnetic_nq",
    "diophantine_branch_cases",
    "diophantine_filling",
    "format_validation_value",
    "in_hex_shell",
    "magnetic_k_vectors",
    "magnetic_normalization_count",
    "magnetic_orbit_indices",
    "magnetic_r_orbit_positions",
    "magnetic_reciprocal_vector",
    "magnetic_shell_shifts",
    "resolve_n_bands",
    "solve_bands_along_path",
    "solve_bands_on_grid",
    "status_from_bool",
    "validate_valley",
]
