"""Gauge-covariant low-energy downfolding for Kane eigenvectors.

The routines here contain no kdotpy dependency.  They map an exact active
subspace at momentum k into a fixed orthonormal reference subspace using the
unitary polar factor of the overlap matrix.  This retains the full E1/H1 block,
including its off-diagonal hybridization, and reports principal-angle
completeness instead of hiding remote-subspace leakage by scalar reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LowdinDownfolding:
    h_eff_mev: ComplexArray
    overlap: ComplexArray
    polar_isometry: ComplexArray
    principal_cos2: FloatArray
    hermiticity_error_mev: float
    spectrum_error_mev: float

    @property
    def min_completeness(self) -> float:
        return float(np.min(self.principal_cos2))

    @property
    def condition_number(self) -> float:
        return float(np.max(self.principal_cos2) / np.min(self.principal_cos2))


def _orthonormality_error(vectors: ComplexArray) -> float:
    n = vectors.shape[1]
    return float(np.max(np.abs(vectors.conj().T @ vectors - np.eye(n))))


def lowdin_effective_hamiltonian(
    reference_vectors: ComplexArray,
    active_vectors: ComplexArray,
    active_energies_mev: FloatArray,
    *,
    orthonormality_tol: float = 1e-8,
    rank_tol: float = 1e-10,
) -> LowdinDownfolding:
    """Map exact active eigenstates into a fixed reference basis.

    ``reference_vectors`` and ``active_vectors`` have shape ``(nbasis, nlow)``
    and orthonormal columns.  If ``A=V_ref^dagger V_active`` and
    ``G=A^dagger A``, the polar isometry ``W=A G^{-1/2}`` maps active
    eigenstates to the reference basis.  ``H_eff=W diag(E) W^dagger`` is
    Hermitian and keeps the exact selected spectrum.  Eigenvalues of ``G`` are
    squared principal-angle cosines and diagnose whether the fixed low-energy
    reference subspace remains controlled.
    """

    vref = np.asarray(reference_vectors, dtype=np.complex128)
    vact = np.asarray(active_vectors, dtype=np.complex128)
    energies = np.asarray(active_energies_mev, dtype=np.float64)
    if vref.ndim != 2 or vact.ndim != 2 or vref.shape != vact.shape:
        raise ValueError("reference and active vectors must have the same 2D shape")
    if energies.shape != (vref.shape[1],):
        raise ValueError("active energies must match the low-energy subspace size")
    if _orthonormality_error(vref) > orthonormality_tol:
        raise ValueError("reference vectors are not orthonormal")
    if _orthonormality_error(vact) > orthonormality_tol:
        raise ValueError("active vectors are not orthonormal")

    overlap = vref.conj().T @ vact
    gram = overlap.conj().T @ overlap
    gram = 0.5 * (gram + gram.conj().T)
    cos2, eigenvectors = np.linalg.eigh(gram)
    if float(np.min(cos2)) <= rank_tol:
        raise ValueError(
            "active and reference subspaces have a rank-deficient overlap; "
            f"minimum principal cos^2={float(np.min(cos2)):.3e}"
        )
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(cos2))) @ eigenvectors.conj().T
    isometry = overlap @ inverse_sqrt
    h_eff = (isometry * energies[None, :]) @ isometry.conj().T
    hermiticity_error = float(np.max(np.abs(h_eff - h_eff.conj().T)))
    spectrum_error = float(
        np.max(np.abs(np.linalg.eigvalsh(h_eff) - np.sort(energies)))
    )
    return LowdinDownfolding(
        h_eff,
        overlap,
        isometry,
        np.asarray(cos2, dtype=np.float64),
        hermiticity_error,
        spectrum_error,
    )


def intersubspace_operator_diagnostics(
    lower_vectors: ComplexArray,
    upper_vectors: ComplexArray,
    operator: ComplexArray,
) -> dict[str, float]:
    """Norms of an operator projected between orthonormal subspaces."""

    lower = np.asarray(lower_vectors, dtype=np.complex128)
    upper = np.asarray(upper_vectors, dtype=np.complex128)
    op = np.asarray(operator, dtype=np.complex128)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[0] != upper.shape[0]:
        raise ValueError("lower and upper vectors must be compatible 2D arrays")
    if op.shape != (lower.shape[0], lower.shape[0]):
        raise ValueError("operator dimension does not match the subspaces")
    if _orthonormality_error(lower) > 1e-8 or _orthonormality_error(upper) > 1e-8:
        raise ValueError("lower and upper subspaces must be orthonormal")
    if float(np.max(np.abs(lower.conj().T @ upper))) > 1e-8:
        raise ValueError("lower and upper subspaces must be mutually orthogonal")
    operator_error = float(np.max(np.abs(op-op.conj().T)))
    if operator_error > 1e-8:
        raise ValueError("operator must be Hermitian")
    coupling = lower.conj().T @ op @ upper
    singular_values = np.linalg.svd(coupling, compute_uv=False)
    frobenius = float(np.linalg.norm(coupling))
    return {
        "intersubspace_frobenius": frobenius,
        "intersubspace_rms_per_lower_state": float(
            frobenius / np.sqrt(lower.shape[1])
        ),
        "intersubspace_max_singular": float(np.max(singular_values)),
        "operator_hermiticity_error": operator_error,
    }


def intersubspace_dhdk_diagnostics(
    h_eff_mev: ComplexArray,
    dhdk_mev_nm: ComplexArray,
    *,
    lower_dimension: int = 2,
) -> dict[str, float]:
    """Gauge-invariant derivative coupling between lower and upper subspaces.

    The Hamiltonian and derivative must be represented in the same fixed basis.
    After diagonalizing ``h_eff_mev``, this returns norms of
    ``U_lower^dagger (dH/dk) U_upper``.  Frobenius and singular-value norms are
    invariant under arbitrary unitary rotations within either subspace and
    under a common unitary change of the fixed basis.
    """

    h = np.asarray(h_eff_mev, dtype=np.complex128)
    derivative = np.asarray(dhdk_mev_nm, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or derivative.shape != h.shape:
        raise ValueError("h_eff and dhdk must be square matrices of matching shape")
    if not 0 < lower_dimension < h.shape[0]:
        raise ValueError("lower_dimension must split the eigenspace")
    h_error = float(np.max(np.abs(h - h.conj().T)))
    derivative_error = float(np.max(np.abs(derivative - derivative.conj().T)))
    if h_error > 1e-8 or derivative_error > 1e-8:
        raise ValueError("h_eff and dhdk must be Hermitian")
    energies, vectors = np.linalg.eigh(h)
    lower = vectors[:, :lower_dimension]
    upper = vectors[:, lower_dimension:]
    projected = intersubspace_operator_diagnostics(lower, upper, derivative)
    direct_gap = float(energies[lower_dimension] - energies[lower_dimension - 1])
    return {
        "direct_gap_meV": direct_gap,
        "intersubspace_frobenius_meV_nm": projected["intersubspace_frobenius"],
        "intersubspace_rms_per_lower_state_meV_nm": projected[
            "intersubspace_rms_per_lower_state"
        ],
        "intersubspace_max_singular_meV_nm": projected[
            "intersubspace_max_singular"
        ],
        "h_hermiticity_error_meV": h_error,
        "dhdk_hermiticity_error_meV_nm": derivative_error,
    }


def e1_h1_block_diagnostics(
    h_eff_mev: ComplexArray,
    *,
    electron_dimension: int = 2,
) -> dict[str, float]:
    """Return pair-averaged E1/H1 energies and hybridization norm."""

    h = np.asarray(h_eff_mev, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("effective Hamiltonian must be square")
    n = h.shape[0]
    if not 0 < electron_dimension < n:
        raise ValueError("electron_dimension must split the low-energy basis")
    he = h[:electron_dimension, :electron_dimension]
    hh = h[electron_dimension:, electron_dimension:]
    heh = h[:electron_dimension, electron_dimension:]
    return {
        "electron_pair_average_meV": float(np.trace(he).real / he.shape[0]),
        "hole_pair_average_valence_meV": float(np.trace(hh).real / hh.shape[0]),
        "hybridization_rms_meV": float(
            np.sqrt(np.trace(heh @ heh.conj().T).real / he.shape[0])
        ),
    }
