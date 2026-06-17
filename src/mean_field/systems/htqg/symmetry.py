from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .domains import HTQGDomain, canonical_domain_key
from .lattice import HTQGLattice

OMEGA = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))


@dataclass(frozen=True)
class SymmetryAction:
    name: str
    internal_matrix: np.ndarray
    antiunitary: bool
    full_plane_wave_action_implemented: bool
    description: str


def sigma_x() -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def sigma_z() -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def layer_mu_x() -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )


def c2x_internal() -> np.ndarray:
    return np.kron(layer_mu_x(), sigma_x())


def c2yT_internal() -> np.ndarray:
    return np.kron(layer_mu_x(), np.eye(2, dtype=np.complex128))


def c2zT_internal() -> np.ndarray:
    return np.kron(np.eye(4, dtype=np.complex128), sigma_x())


def c3z_internal(domain: str | HTQGDomain) -> np.ndarray:
    key = canonical_domain_key(domain)
    if key == "alpha_beta_gamma":
        layer_diag = np.asarray([1.0, OMEGA, OMEGA, 1.0], dtype=np.complex128)
    elif key == "alpha_beta_alpha":
        layer_diag = np.asarray([1.0, OMEGA.conjugate(), OMEGA.conjugate(), OMEGA], dtype=np.complex128)
    elif key == "beta_alpha_beta":
        # C2zT partner of alpha_beta_alpha; useful as a diagnostic convention.
        layer_diag = np.asarray([1.0, OMEGA, OMEGA, OMEGA.conjugate()], dtype=np.complex128)
    elif key == "gamma_beta_alpha":
        layer_diag = np.asarray([1.0, OMEGA.conjugate(), OMEGA.conjugate(), 1.0], dtype=np.complex128)
    else:  # pragma: no cover
        raise AssertionError(key)
    sub_diag = np.asarray([OMEGA, OMEGA.conjugate()], dtype=np.complex128)
    return np.kron(np.diag(layer_diag), np.diag(sub_diag))


def moire_translation_internal() -> np.ndarray:
    layer_diag = np.asarray([1.0, OMEGA.conjugate(), OMEGA, 1.0], dtype=np.complex128)
    return np.kron(np.diag(layer_diag), np.eye(2, dtype=np.complex128))


def particle_hole_internal() -> np.ndarray:
    layer = np.asarray(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    return np.kron(layer, sigma_x())


def expand_internal_matrix(lattice: HTQGLattice, internal_matrix: np.ndarray) -> np.ndarray:
    """Repeat an 8x8 internal matrix on every G block.

    This is only the internal part of a continuum-model symmetry.  C3, moiré
    translations, mirrors, and antiunitary operations also need momentum/G-grid
    maps for a Gate-A validation.
    """

    internal = np.asarray(internal_matrix, dtype=np.complex128)
    if internal.shape != (8, 8):
        raise ValueError(f"Expected internal matrix shape (8, 8), got {internal.shape}")
    return np.kron(np.eye(lattice.n_g, dtype=np.complex128), internal)


def unitarity_residual(matrix: np.ndarray) -> float:
    mat = np.asarray(matrix, dtype=np.complex128)
    ident = np.eye(mat.shape[0], dtype=np.complex128)
    return float(np.max(np.abs(mat.conjugate().T @ mat - ident)))


def internal_symmetry_actions(domain: str | HTQGDomain = "alpha_beta_alpha") -> tuple[SymmetryAction, ...]:
    return (
        SymmetryAction(
            "C2x_internal",
            c2x_internal(),
            False,
            False,
            "Internal μx⊗σx part; full action also mirrors k/G and maps domain displacements.",
        ),
        SymmetryAction(
            "C2yT_internal",
            c2yT_internal(),
            True,
            False,
            "Internal μx antiunitary part; full action includes My and complex conjugation.",
        ),
        SymmetryAction(
            "C2zT_internal",
            c2zT_internal(),
            True,
            False,
            "Internal σx antiunitary part; full action maps d -> -d.",
        ),
        SymmetryAction(
            "C3z_internal",
            c3z_internal(domain),
            False,
            False,
            "Appendix-C internal C3z matrix at r=0; full action rotates k/G and may include translations.",
        ),
        SymmetryAction(
            "TaM_internal",
            moire_translation_internal(),
            False,
            False,
            "Layer phase for a moiré translation; full action is not a standalone Gate-A check.",
        ),
        SymmetryAction(
            "P_internal",
            particle_hole_internal(),
            True,
            False,
            "Approximate particle-hole internal matrix, valid only with MDT and Dirac rotations disabled.",
        ),
    )


def validate_internal_unitarity(domain: str | HTQGDomain = "alpha_beta_alpha", *, atol: float = 1.0e-12) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for action in internal_symmetry_actions(domain):
        residuals[action.name] = unitarity_residual(action.internal_matrix)
    return residuals


__all__ = [
    "OMEGA",
    "SymmetryAction",
    "c2x_internal",
    "c2yT_internal",
    "c2zT_internal",
    "c3z_internal",
    "expand_internal_matrix",
    "internal_symmetry_actions",
    "layer_mu_x",
    "moire_translation_internal",
    "particle_hole_internal",
    "sigma_x",
    "sigma_z",
    "unitarity_residual",
    "validate_internal_unitarity",
]
