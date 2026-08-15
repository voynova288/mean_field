"""Degenerate-shell occupation helpers for fixed-rank Hartree--Fock ODA.

The linear algebra here is system-independent. Physical systems remain
responsible for identifying an exact Fock shell, enforcing their block
structure, and deciding how unresolved broken-symmetry branches are fanned out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import numpy as np

from .engine import DensityUpdateResult


class DensityStateProtocol(Protocol):
    density: np.ndarray


@dataclass(frozen=True, slots=True)
class StateBoundPreviousDensityBuilder:
    """Unary engine adapter for a density update that needs the current state.

    One adapter is bound to exactly one mutable HF state. At each ordinary
    unary engine call it supplies private copies of the Hamiltonian and the
    state's current mixed density to ``callback``. This preserves the legacy
    SCF engine and its historical replay identity while making the state
    dependence explicit and mutation-safe.
    """

    state: DensityStateProtocol
    callback: Callable[[np.ndarray, np.ndarray], DensityUpdateResult]
    policy_fingerprint: str

    def __post_init__(self) -> None:
        if not hasattr(self.state, "density"):
            raise TypeError("state must expose a density array")
        if not callable(self.callback):
            raise TypeError("callback must be callable")
        if (
            not isinstance(self.policy_fingerprint, str)
            or len(self.policy_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.policy_fingerprint
            )
        ):
            raise ValueError("policy_fingerprint must be a lowercase SHA256 digest")

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        previous = np.asarray(self.state.density, dtype=np.complex128)
        return self.callback(
            np.asarray(hamiltonian, dtype=np.complex128).copy(),
            previous.copy(),
        )


@dataclass(frozen=True, slots=True)
class MaximumOverlapRankSelection:
    """Basis-independent rank selection inside a degenerate Fock shell.

    ``coefficient_projector`` acts in the supplied shell frame. It is absent
    when the overlap spectrum is tied at the requested rank, because returning
    an arbitrary eigenvector ordering would create a gauge-dependent branch.
    """

    selected_rank: int
    shell_dimension: int
    overlap_eigenvalues_descending: np.ndarray
    maximum_overlap_value: float
    overlap_cutoff_gap: float | None
    overlap_gap_tolerance: float
    hermiticity_residual: float
    unique: bool
    coefficient_projector: np.ndarray | None

    def __post_init__(self) -> None:
        values = np.asarray(self.overlap_eigenvalues_descending, dtype=np.float64)
        if values.shape != (self.shell_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("invalid overlap eigenvalue inventory")
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "overlap_eigenvalues_descending", values)
        if not 0 <= self.selected_rank <= self.shell_dimension:
            raise ValueError("selected rank is outside the shell")
        if type(self.unique) is not bool:
            raise TypeError("unique must be a native bool")
        if self.unique != (self.coefficient_projector is not None):
            raise ValueError("unique selection/projector mismatch")
        if self.coefficient_projector is not None:
            projector = np.asarray(self.coefficient_projector, dtype=np.complex128)
            if projector.shape != (self.shell_dimension, self.shell_dimension):
                raise ValueError("coefficient projector shape mismatch")
            projector = np.array(projector, copy=True)
            projector.setflags(write=False)
            object.__setattr__(self, "coefficient_projector", projector)


def select_maximum_overlap_rank_projector(
    overlap_matrix: np.ndarray,
    selected_rank: int,
    *,
    overlap_gap_tolerance: float,
    hermiticity_tolerance: float = 1.0e-12,
) -> MaximumOverlapRankSelection:
    """Select the closest fixed-rank shell projector, or report a tie.

    For shell frame ``V`` and current ODA density ``D``, callers pass
    ``V.conj().T @ D @ V``. The Ky Fan maximum principle selects the requested
    rank from its largest eigenvalues. A tied overlap cutoff returns
    ``unique=False`` and no arbitrary projector.
    """

    matrix = np.asarray(overlap_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("overlap_matrix must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("overlap_matrix must be finite")
    if isinstance(selected_rank, (bool, np.bool_)) or not isinstance(
        selected_rank, Integral
    ):
        raise TypeError("selected_rank must be an integer")
    rank = int(selected_rank)
    dimension = int(matrix.shape[0])
    if rank < 0 or rank > dimension:
        raise ValueError("selected_rank is outside the shell")
    for value, label in (
        (overlap_gap_tolerance, "overlap_gap_tolerance"),
        (hermiticity_tolerance, "hermiticity_tolerance"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{label} must be a real scalar")
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
    gap_tolerance = float(overlap_gap_tolerance)
    hermitian_tolerance = float(hermiticity_tolerance)
    hermiticity_residual = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermiticity_residual > hermitian_tolerance:
        raise ValueError("overlap_matrix is materially non-Hermitian")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    order = np.arange(dimension - 1, -1, -1)
    descending = np.asarray(eigenvalues[order], dtype=np.float64)
    maximum_overlap = float(np.sum(descending[:rank]))
    if rank in (0, dimension):
        cutoff_gap: float | None = None
        unique = True
    else:
        cutoff_gap = float(descending[rank - 1] - descending[rank])
        unique = cutoff_gap > gap_tolerance
    coefficient_projector: np.ndarray | None = None
    if unique:
        if rank == 0:
            coefficient_projector = np.zeros(
                (dimension, dimension), dtype=np.complex128
            )
        elif rank == dimension:
            coefficient_projector = np.eye(dimension, dtype=np.complex128)
        else:
            selected = np.asarray(
                eigenvectors[:, order[:rank]], dtype=np.complex128
            )
            coefficient_projector = selected @ selected.conj().T
        residual = max(
            float(
                np.max(
                    np.abs(
                        coefficient_projector - coefficient_projector.conj().T
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        coefficient_projector @ coefficient_projector
                        - coefficient_projector
                    )
                )
            ),
            abs(float(np.trace(coefficient_projector).real) - rank),
            abs(float(np.trace(coefficient_projector).imag)),
        )
        if residual > max(
            1.0e-12,
            32.0 * np.finfo(np.float64).eps * dimension,
        ):
            raise RuntimeError(
                "maximum-overlap coefficient projector failed invariants"
            )
    return MaximumOverlapRankSelection(
        selected_rank=rank,
        shell_dimension=dimension,
        overlap_eigenvalues_descending=descending,
        maximum_overlap_value=maximum_overlap,
        overlap_cutoff_gap=cutoff_gap,
        overlap_gap_tolerance=gap_tolerance,
        hermiticity_residual=hermiticity_residual,
        unique=bool(unique),
        coefficient_projector=coefficient_projector,
    )


__all__ = [
    "DensityStateProtocol",
    "MaximumOverlapRankSelection",
    "StateBoundPreviousDensityBuilder",
    "select_maximum_overlap_rank_projector",
]
