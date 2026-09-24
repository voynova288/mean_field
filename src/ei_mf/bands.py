"""Band and detuning helpers for the Stage-A parabolic EI model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .params import EIParams
from .units import HBAR2_OVER_2M0_MEV_NM2

Array = NDArray[np.float64]


def density_cm2_to_nm2(n_cm2: float) -> float:
    """Convert cm^-2 to nm^-2."""

    return float(n_cm2) * 1e-14


def kF_from_density_cm2(n_cm2: float, spin_degeneracy: float = 2.0) -> float:
    """Fermi momentum for a circular 2D pocket with density n=g_s k_F^2/(4*pi)."""

    if n_cm2 < 0:
        raise ValueError("density must be non-negative")
    n_nm2 = density_cm2_to_nm2(n_cm2)
    return float(np.sqrt(4.0 * np.pi * n_nm2 / spin_degeneracy))


def effective_kF(params: EIParams) -> float:
    return float(params.kF_nm_inv) if params.kF_nm_inv is not None else kF_from_density_cm2(params.n0_cm2)


def alpha_pair_mev_nm2(params: EIParams) -> float:
    """Paper-convention coefficient for xi0=(eps_e+eps_h-2mu)."""

    return HBAR2_OVER_2M0_MEV_NM2 * (
        1.0 / params.m_e_over_m0 + 1.0 / params.m_h_over_m0
    )


def xi0_parabolic_mev(k_nm_inv: Array, params: EIParams) -> Array:
    """Stage-A pair detuning in paper convention.

    This intentionally has no extra 1/2 factor.  It implements the workdoc
    convention ``xi0 = alpha_pair * (k**2 - kF**2)``.
    """

    k = np.asarray(k_nm_inv, dtype=np.float64)
    kF = effective_kF(params)
    return alpha_pair_mev_nm2(params) * (k * k - kF * kF)


@dataclass(frozen=True)
class RadialBandData:
    """Radial conduction-electron and valence-hole energies for Stage-B work.

    The required CSV convention is intentionally explicit:
    ``k_nm_inv,epsilon_a_meV,epsilon_b_meV`` where ``epsilon_b_meV`` is a hole
    energy, not the raw valence-electron energy.  See ``docs/equations.md``.
    """

    k_nm_inv: Array
    epsilon_a_mev: Array
    epsilon_b_mev: Array
    source: str = "unknown"

    @classmethod
    def from_csv(cls, path: str | Path) -> "RadialBandData":
        path = Path(path)
        data = np.genfromtxt(path, delimiter=",", names=True)
        names = set(data.dtype.names or ())
        required = {"k_nm_inv", "epsilon_a_meV", "epsilon_b_meV"}
        if not required.issubset(names):
            raise ValueError(
                f"band CSV must contain columns {sorted(required)}; got {sorted(names)}"
            )
        k = np.asarray(data["k_nm_inv"], dtype=np.float64)
        eps_a = np.asarray(data["epsilon_a_meV"], dtype=np.float64)
        eps_b = np.asarray(data["epsilon_b_meV"], dtype=np.float64)
        order = np.argsort(k)
        out = cls(k[order], eps_a[order], eps_b[order], source=str(path))
        out.validate()
        return out

    def validate(self) -> None:
        if self.k_nm_inv.ndim != 1 or self.k_nm_inv.size < 4:
            raise ValueError("band data requires at least four radial k points")
        if not (self.epsilon_a_mev.shape == self.k_nm_inv.shape == self.epsilon_b_mev.shape):
            raise ValueError("band arrays must have matching shapes")
        if np.any(np.diff(self.k_nm_inv) <= 0):
            raise ValueError("band k grid must be strictly increasing")
        if not (
            np.all(np.isfinite(self.k_nm_inv))
            and np.all(np.isfinite(self.epsilon_a_mev))
            and np.all(np.isfinite(self.epsilon_b_mev))
        ):
            raise ValueError("band data contains non-finite values")

    def interpolate(self, k_nm_inv: Array) -> tuple[Array, Array]:
        """Interpolate epsilon_a and epsilon_b onto a solver grid."""

        k = np.asarray(k_nm_inv, dtype=np.float64)
        if k.min() < self.k_nm_inv[0] or k.max() > self.k_nm_inv[-1]:
            raise ValueError(
                "solver k grid extends outside band CSV range; increase CSV range or reduce k_max"
            )
        eps_a = np.interp(k, self.k_nm_inv, self.epsilon_a_mev)
        eps_b = np.interp(k, self.k_nm_inv, self.epsilon_b_mev)
        return eps_a, eps_b


def epsilon_P_M_from_band_data(bands: RadialBandData, k_nm_inv: Array) -> tuple[Array, Array]:
    """Return Supplementary Note 2 epsilon_P=epsilon_a+epsilon_b and epsilon_M=epsilon_a-epsilon_b."""

    eps_a, eps_b = bands.interpolate(k_nm_inv)
    return eps_a + eps_b, eps_a - eps_b
