"""Core reusable building blocks for mean-field solvers."""

from .bands import (
    GridBandsResult,
    PathBandsResult,
    compute_grid_bands,
    compute_path_bands,
    resolve_n_bands,
)
from .validation import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
    ValidationValue,
    format_validation_value,
    status_from_bool,
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

__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "compute_grid_bands",
    "compute_path_bands",
    "resolve_n_bands",
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "ValidationValue",
    "format_validation_value",
    "status_from_bool",
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
