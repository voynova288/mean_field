"""Basis conventions for projected InAs/GaSb E1/H1 matrix models.

The maintained many-body representation is an ordinary electron-band basis,
not a superconducting BdG doubling:

    (E1+, E1-, H1+, H1-).

Excitonic order is the interaction-induced E1--H1 block of the Fock
self-energy.  Kramers partners are explicit, so no additional spin factor is
allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KANE8_JZ = np.asarray((0.5, -0.5, 1.5, 0.5, -0.5, -1.5, 0.5, -0.5), dtype=float)


@dataclass(frozen=True)
class E1H1BasisSpec:
    """Declare the ordered active quartet and its E1/H1 block split."""

    labels: tuple[str, ...] = ("E1+", "E1-", "H1+", "H1-")
    electron_indices: tuple[int, ...] = (0, 1)
    hole_indices: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        n = len(self.labels)
        indices = self.electron_indices + self.hole_indices
        if n != 4:
            raise ValueError("the initial matrix-EI model requires one E1 and one H1 Kramers pair")
        if tuple(sorted(indices)) != tuple(range(n)):
            raise ValueError("electron_indices and hole_indices must partition the active basis")
        if len(self.electron_indices) != 2 or len(self.hole_indices) != 2:
            raise ValueError("E1 and H1 must each contain exactly one Kramers pair")

    @property
    def dimension(self) -> int:
        return len(self.labels)

    @property
    def electron_projector(self) -> np.ndarray:
        out = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        out[self.electron_indices, self.electron_indices] = 1.0
        return out

    @property
    def h1_electron_projector(self) -> np.ndarray:
        """Projector onto H1 electron states; holes are vacancies in this block."""

        out = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        out[self.hole_indices, self.hole_indices] = 1.0
        return out

    @property
    def hole_projector(self) -> np.ndarray:
        """Compatibility alias for the H1-electron subspace projector."""

        return self.h1_electron_projector

    @property
    def tau_z(self) -> np.ndarray:
        return self.electron_projector - self.h1_electron_projector

    def eh_block(self, matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.complex128)
        if arr.shape[:2] != (self.dimension, self.dimension):
            raise ValueError("matrix must begin with the active-basis axes")
        return arr[np.ix_(self.electron_indices, self.hole_indices, *[range(s) for s in arr.shape[2:]])]


def ordinary_electron_pair_channels_mev(
    conduction_energy_mev: np.ndarray,
    valence_energy_mev: np.ndarray,
    electron_mu_mev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ordinary-electron energies to electron-hole pair channels.

    With ``a`` a conduction electron and ``b`` a valence hole,
    ``A=E_c-mu_el`` and ``B=mu_el-E_v``.  The doubled pair convention then has
    ``xi=A+B=E_c-E_v`` and ``eta=A-B=E_c+E_v-2*mu_el``.  Therefore the
    absolute Kane--Poisson electron chemical potential enters the blocking or
    charge-asymmetry channel, not the pair-energy channel.  It must not be
    identified with the pair-density Lagrange multiplier denoted ``mu`` in
    the scalar BCS derivation.
    """

    conduction = np.asarray(conduction_energy_mev, dtype=float)
    valence = np.asarray(valence_energy_mev, dtype=float)
    if conduction.shape != valence.shape:
        raise ValueError("conduction and valence energies must have matching shapes")
    if not np.all(np.isfinite(conduction)) or not np.all(np.isfinite(valence)):
        raise ValueError("conduction and valence energies must be finite")
    if not np.isfinite(electron_mu_mev):
        raise ValueError("electron_mu_mev must be finite")
    electron_excitation = conduction - float(electron_mu_mev)
    hole_excitation = float(electron_mu_mev) - valence
    xi_pair = electron_excitation + hole_excitation
    eta_charge = electron_excitation - hole_excitation
    return xi_pair, eta_charge


def kane8_time_reversal_unitary() -> np.ndarray:
    """Unitary part of ``Theta=U_T K`` in the standard kdotpy Kane basis.

    The orbital order is ``Gamma6(+1/2,-1/2), Gamma8(+3/2,+1/2,-1/2,-3/2),
    Gamma7(+1/2,-1/2)`` and ``Theta|j,m> = (-1)^(j-m)|j,-m>``.
    """

    result = np.zeros((8, 8), dtype=np.complex128)
    result[1, 0] = 1.0
    result[0, 1] = -1.0
    result[5, 2] = 1.0
    result[4, 3] = -1.0
    result[3, 4] = 1.0
    result[2, 5] = -1.0
    result[7, 6] = 1.0
    result[6, 7] = -1.0
    return result


def spinful_time_reversal_errors(unitary: np.ndarray) -> tuple[float, float]:
    """Return unitarity and ``Theta^2=-1`` errors for an antiunitary matrix."""

    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("time-reversal unitary must be square")
    identity = np.eye(matrix.shape[0])
    return (
        float(np.max(np.abs(matrix.conj().T @ matrix - identity))),
        float(np.max(np.abs(matrix @ matrix.conj() + identity))),
    )


def block_unitary(
    electron: np.ndarray,
    hole: np.ndarray,
    basis: E1H1BasisSpec | None = None,
) -> np.ndarray:
    """Embed independent E1/H1 Kramers-pair rotations in the active basis."""

    ue = np.asarray(electron, dtype=np.complex128)
    uh = np.asarray(hole, dtype=np.complex128)
    if ue.shape != (2, 2) or uh.shape != (2, 2):
        raise ValueError("electron and hole rotations must both be 2x2")
    for name, matrix in (("electron", ue), ("hole", uh)):
        error = float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(2))))
        if error > 1e-10:
            raise ValueError(f"{name} rotation is not unitary: error={error:.3e}")
    spec = E1H1BasisSpec() if basis is None else basis
    out = np.zeros((spec.dimension, spec.dimension), dtype=np.complex128)
    out[np.ix_(spec.electron_indices, spec.electron_indices)] = ue
    out[np.ix_(spec.hole_indices, spec.hole_indices)] = uh
    return out


def active_electron_hole_areal_densities(
    density: np.ndarray,
    k_weights_nm2: np.ndarray,
    basis: E1H1BasisSpec | None = None,
) -> tuple[float, float]:
    """Return E1-electron and H1-hole densities from a ket-oriented density."""

    spec = E1H1BasisSpec() if basis is None else basis
    projector = np.asarray(density, dtype=np.complex128)
    weights = np.asarray(k_weights_nm2, dtype=float)
    if projector.shape[:2] != (spec.dimension, spec.dimension) or projector.ndim != 3:
        raise ValueError("density must have shape (4, 4, nk)")
    if weights.shape != (projector.shape[2],):
        raise ValueError("k weights do not match density")
    n_e = np.einsum(
        "k,ab,bak->",
        weights,
        spec.electron_projector,
        projector,
        optimize=True,
    )
    identity_minus_p = np.eye(spec.dimension)[:, :, None] - projector
    n_h = np.einsum(
        "k,ab,bak->",
        weights,
        spec.h1_electron_projector,
        identity_minus_p,
        optimize=True,
    )
    if abs(n_e.imag) > 1e-10 or abs(n_h.imag) > 1e-10:
        raise ValueError("electron/hole densities are not real")
    return float(n_e.real), float(n_h.real)
