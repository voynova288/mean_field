"""Monney et al. four-pocket low-energy model for bulk 1T-TiSe2.

The basis is ``(v_Gamma, c_L1, c_L2, c_L3)`` at one common local momentum
``p`` measured from each pocket center.  Equivalently the physical momentum
labels are ``{p, p+Q1, p+Q2, p+Q3}``, with ``Qi = Gamma->Li``.

This is the four-pocket truncation used in arXiv:0809.1930.  It is not the full
eight-sector reciprocal quotient of a literal 2x2x2 supercell.

The paper literally prints ``+ t_c cos(pi p_z/k_GammaA)`` and quotes
``t_c=+0.03 eV``.  That literal convention is the default.  Because the text
also calls the fitted energies band extrema, the alternative
``l_minimum_diagnostic`` convention is exposed only as a sensitivity test; it
must not be presented as the paper formula without an independent source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

HBAR2_OVER_2ME_EV_A2 = 3.80998212


@dataclass(frozen=True)
class MonneyTiSe2Parameters:
    """Parameters for the four-pocket model.

    ``c_angstrom`` is required to convert the paper's dimensionless
    ``p_z/k_GammaA`` to inverse Angstrom.  The default 6.008 Angstrom is an
    external experimental lattice constant, not a fit parameter supplied by
    arXiv:0809.1930.
    """

    epsilon_v0_ev: float = -0.030
    m_v_over_me: float = -0.23
    t_v_ev: float = 0.060
    epsilon_c0_ev: float = -0.010
    m_long_over_me: float = 5.5
    m_short_over_me: float = 2.2
    t_c_ev: float = 0.030
    c_angstrom: float = 6.008
    conduction_z_convention: Literal[
        "paper_literal", "l_minimum_diagnostic"
    ] = "paper_literal"
    valley_angles_rad: tuple[float, float, float] = (
        0.0,
        2.0 * np.pi / 3.0,
        4.0 * np.pi / 3.0,
    )

    def __post_init__(self) -> None:
        finite_values = (
            self.epsilon_v0_ev,
            self.m_v_over_me,
            self.t_v_ev,
            self.epsilon_c0_ev,
            self.m_long_over_me,
            self.m_short_over_me,
            self.t_c_ev,
            self.c_angstrom,
            *self.valley_angles_rad,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise ValueError("Monney parameters must be finite")
        if self.m_v_over_me >= 0.0:
            raise ValueError("the hole-like valence mass must be negative")
        if self.m_long_over_me <= 0.0 or self.m_short_over_me <= 0.0:
            raise ValueError("conduction masses must be positive")
        if self.t_v_ev < 0.0 or self.t_c_ev < 0.0:
            raise ValueError("t_v_ev and t_c_ev are nonnegative hopping magnitudes")
        if self.c_angstrom <= 0.0:
            raise ValueError("c_angstrom must be positive")
        if self.conduction_z_convention not in {
            "paper_literal",
            "l_minimum_diagnostic",
        }:
            raise ValueError(
                "conduction_z_convention must be 'paper_literal' or "
                "'l_minimum_diagnostic'"
            )
        if len(self.valley_angles_rad) != 3:
            raise ValueError("the Monney model requires exactly three L valleys")
        angles = np.mod(np.asarray(self.valley_angles_rad, dtype=float), 2.0 * np.pi)
        angles = np.sort(angles)
        spacings = np.diff(np.concatenate((angles, [angles[0] + 2.0 * np.pi])))
        if not np.allclose(spacings, 2.0 * np.pi / 3.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("valley_angles_rad must form one C3 orbit")

    @property
    def k_gamma_a_Ainv(self) -> float:
        return float(np.pi / self.c_angstrom)


def _validate_points(points_Ainv: np.ndarray) -> FloatArray:
    points = np.asarray(points_Ainv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_Ainv must have shape (nk,3), got {points.shape}")
    if points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("points_Ainv must be finite and nonempty")
    return points


def monney_bare_energies(
    points_Ainv: np.ndarray,
    params: MonneyTiSe2Parameters | None = None,
) -> FloatArray:
    """Return bare energies with shape ``(4,nk)`` in eV.

    ``points_Ainv`` is the common local momentum ``p``.  Row zero is the
    Gamma-centered valence band.  Rows one through three are the L-centered
    conduction valleys evaluated at physical momenta ``p+Qi``.
    """

    model = MonneyTiSe2Parameters() if params is None else params
    points = _validate_points(points_Ainv)
    px, py, pz = points.T
    cosine = np.cos(np.pi * pz / model.k_gamma_a_Ainv)

    energies = np.empty((4, points.shape[0]), dtype=np.float64)
    energies[0] = (
        HBAR2_OVER_2ME_EV_A2 * (px**2 + py**2) / model.m_v_over_me
        + model.t_v_ev * cosine
        + model.epsilon_v0_ev
    )

    conduction_cosine_sign = (
        1.0 if model.conduction_z_convention == "paper_literal" else -1.0
    )
    for valley, angle in enumerate(model.valley_angles_rad, start=1):
        c, s = np.cos(angle), np.sin(angle)
        p_long = c * px + s * py
        p_short = -s * px + c * py
        energies[valley] = (
            HBAR2_OVER_2ME_EV_A2
            * (
                p_long**2 / model.m_long_over_me
                + p_short**2 / model.m_short_over_me
            )
            + conduction_cosine_sign * model.t_c_ev * cosine
            + model.epsilon_c0_ev
        )
    return energies


def monney_bare_hamiltonian(
    points_Ainv: np.ndarray,
    params: MonneyTiSe2Parameters | None = None,
) -> ComplexArray:
    """Return the diagonal bare Hamiltonian with shape ``(4,4,nk)``."""

    energies = monney_bare_energies(points_Ainv, params=params)
    hamiltonian = np.zeros((4, 4, energies.shape[1]), dtype=np.complex128)
    indices = np.arange(4)
    hamiltonian[indices, indices, :] = energies
    return hamiltonian


def _normalize_delta(delta_ev: np.ndarray, nk: int) -> ComplexArray:
    delta = np.asarray(delta_ev, dtype=np.complex128)
    if delta.shape == (3,):
        delta = np.repeat(delta[:, None], int(nk), axis=1)
    if delta.shape != (3, int(nk)):
        raise ValueError(f"delta_ev must have shape (3,) or (3,{nk}), got {delta.shape}")
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta_ev must be finite")
    return delta


def monney_mean_field_hamiltonian(
    points_Ainv: np.ndarray,
    delta_ev: np.ndarray,
    params: MonneyTiSe2Parameters | None = None,
) -> ComplexArray:
    """Return the paper's Hermitian four-band mean-field Hamiltonian.

    The off-diagonal convention is ``H[v,ci] = -Delta_i``.  With
    ``Delta_i = sum_q V(q) <b_i^dagger a>``, this reproduces the denominator
    of Monney et al. Eq. (16).
    """

    hamiltonian = monney_bare_hamiltonian(points_Ainv, params=params)
    delta = _normalize_delta(delta_ev, hamiltonian.shape[2])
    for valley in range(3):
        conduction = valley + 1
        hamiltonian[0, conduction, :] = -delta[valley]
        hamiltonian[conduction, 0, :] = -delta[valley].conj()
    return hamiltonian


def monney_characteristic_denominator(
    z_ev: complex | np.ndarray,
    bare_energies_ev: np.ndarray,
    delta_ev: np.ndarray,
) -> np.ndarray:
    """Evaluate the characteristic denominator in Monney et al. Eq. (16)."""

    bare = np.asarray(bare_energies_ev, dtype=np.float64)
    if bare.ndim != 2 or bare.shape[0] != 4:
        raise ValueError("bare_energies_ev must have shape (4,nk)")
    delta = _normalize_delta(delta_ev, bare.shape[1])
    z = np.asarray(z_ev, dtype=np.complex128)
    if z.ndim == 0:
        z = np.full((bare.shape[1],), complex(z), dtype=np.complex128)
    if z.shape != (bare.shape[1],):
        raise ValueError(f"z_ev must be scalar or shape {(bare.shape[1],)}, got {z.shape}")

    conduction_factors = z[None, :] - bare[1:, :]
    result = (z - bare[0]) * np.prod(conduction_factors, axis=0)
    for valley in range(3):
        other = [index for index in range(3) if index != valley]
        result -= np.abs(delta[valley]) ** 2 * np.prod(
            conduction_factors[other, :], axis=0
        )
    return result


__all__ = [
    "HBAR2_OVER_2ME_EV_A2",
    "MonneyTiSe2Parameters",
    "monney_bare_energies",
    "monney_bare_hamiltonian",
    "monney_characteristic_denominator",
    "monney_mean_field_hamiltonian",
]
