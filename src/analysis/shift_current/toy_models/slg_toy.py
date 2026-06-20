from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from mean_field.systems.htg.mao2025 import CARBON_BOND_NM


@dataclass(frozen=True)
class GappedSLGParams:
    """Nearest-neighbor gapped graphene toy model from Mao et al. Appendix A."""

    hopping_ev: float = 2.73
    mass_ev: float = 1.5
    bond_nm: float = CARBON_BOND_NM


def nn_vectors(params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Nearest-neighbor vectors d_j in nm, matching Appendix A."""

    d = float(params.bond_nm)
    vectors = []
    for j in range(3):
        angle = 2.0 * math.pi * j / 3.0
        vectors.append((-d * math.sin(angle), d * math.cos(angle)))
    return np.asarray(vectors, dtype=float)






def hamiltonian(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Bloch Hamiltonian H(k) in the embedded A/B basis.

    H_AB(k) = -t sum_j exp(i k.d_j), H_BA = H_AB^*, and the mass term is
    +m on A and -m on B.  This is a two-band finite-BZ model, not a continuum
    massive Dirac approximation.
    """

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    f = np.sum(phase)
    off = -float(params.hopping_ev) * f
    return np.asarray([[params.mass_ev, off], [off.conjugate(), -params.mass_ev]], dtype=np.complex128)


def dhdk(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    """Analytic partial_k H matrices in eV*nm."""

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    derivs: list[np.ndarray] = []
    for axis in range(2):
        df = np.sum(1.0j * dvec[:, axis] * phase)
        doff = -float(params.hopping_ev) * df
        derivs.append(np.asarray([[0.0, doff], [doff.conjugate(), 0.0]], dtype=np.complex128))
    return derivs[0], derivs[1]


def d2hdk(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> np.ndarray:
    """Analytic second partial derivatives of H in eV*nm^2.

    The hTTG continuum Dirac Hamiltonian is linear in k, so Mao Eq. (4) has no
    second-derivative term.  This tight-binding benchmark is nonlinear in k and
    therefore requires W^{ab}_{nm}=<u_n|partial_a partial_b H|u_m> in the
    generalized derivative.
    """

    k = np.asarray(k_xy_nm_inv, dtype=float).reshape(2)
    dvec = nn_vectors(params)
    phase = np.exp(1.0j * (dvec @ k))
    out = np.empty((2, 2, 2, 2), dtype=np.complex128)
    for axis_a in range(2):
        for axis_b in range(2):
            d2f = np.sum(-dvec[:, axis_a] * dvec[:, axis_b] * phase)
            d2off = -float(params.hopping_ev) * d2f
            out[axis_a, axis_b] = np.asarray(
                [[0.0, d2off], [d2off.conjugate(), 0.0]],
                dtype=np.complex128,
            )
    return out


def diagonalize(k_xy_nm_inv: np.ndarray | tuple[float, float], params: GappedSLGParams = GappedSLGParams()) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(hamiltonian(k_xy_nm_inv, params))
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
