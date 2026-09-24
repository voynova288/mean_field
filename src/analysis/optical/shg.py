from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import numpy as np

from analysis.shift_current import HBAR_EV_S
from analysis.optical.magneto import VACUUM_PERMITTIVITY_F_PER_M
from analysis.optical.linear import E2_OVER_HBAR_S, _selected_band_list

# Unit prefactor magnitudes for Liu-Dai-style 2D velocity-gauge Eq. (4) when
# callers use the repository convention ``velocity_h = dH/dk`` in eV nm,
# photon/denominator energies in eV, and k weights in nm^-2.  The SI sheet
# coefficient is A m/V^2.  The paper labels the 2D shift-current panel as
# microA/V^2; the natural 2D conversion is microA nm/V^2, so workflow code must
# state which convention it plots instead of silently using this as proof of
# paper normalization.
#
# By default, exact/small energy-denominator degeneracies are skipped below.
# Some printed velocity-gauge formulas include an explicit ``i eta`` broadening
# and do not state such an exclusion; set
# ``skip_degenerate_energy_denominators=False`` only after auditing that the
# diagonal/intraband terms should be eta-regularized for the target paper
# convention.
VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2 = E2_OVER_HBAR_S * 1.0e-9
VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2 = E2_OVER_HBAR_S * 1.0e6

# Liu-Dai Eq. (4) is odd in the electron charge.  With the convention that
# ``E_CHARGE_C``/``E2_OVER_HBAR_S`` are positive magnitudes, the paper's ``e^3``
# corresponds to the signed electron charge ``(-|e|)^3``.  Linear optics in
# Eq. (20) is even in e and is therefore unaffected, but Fig.4 is globally sign
# sensitive.
LIU_DAI_2020_EQ4_PREFAC_A_M_PER_V2 = -VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2
LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2 = -VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2
LIU_DAI_2020_EQ4_SHG_PREFAC_S = -E2_OVER_HBAR_S


@dataclass(frozen=True)
class SHGConductivityResult:
    """Liu--Dai Eq.(4) compatibility SHG photoconductivity tensor.

    ``conductivity`` has shape ``(n_omega, ndim, ndim, ndim)`` in component
    order ``(current_axis c, optical_axis a, optical_axis b)``.  Units are set by
    the caller's velocity, k-weight, and prefactor conventions.
    """

    photon_energies_ev: np.ndarray
    conductivity: np.ndarray
    skipped_small_denominators: int

@dataclass(frozen=True)
class ShiftCurrentVelocityGaugeResult:
    """Velocity-gauge dc shift-current photoconductivity tensor.

    ``conductivity`` has shape ``(n_omega, ndim, ndim, ndim)`` in component
    order ``(current_axis c, optical_axis a, optical_axis b)``.  It evaluates the
    real part in Liu-Dai 2020 Eq. (4).  Units are set by the caller's velocity,
    k-weight, and prefactor conventions.
    """

    photon_energies_ev: np.ndarray
    conductivity: np.ndarray
    skipped_small_denominators: int


def _polarization_factor_matrix(
    ndim: int,
    polarization_factor: np.ndarray | Sequence[Sequence[complex]] | None,
) -> np.ndarray:
    if polarization_factor is None:
        return np.ones((ndim, ndim), dtype=np.complex128)
    phi = np.asarray(polarization_factor, dtype=np.complex128)
    if phi.shape != (ndim, ndim):
        raise ValueError(f"polarization_factor must have shape {(ndim, ndim)}, got {phi.shape}")
    return phi


def _velocity_gauge_inputs(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    velocity_h: np.ndarray,
    occupations: np.ndarray,
    eta_ev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    omega = np.asarray(photon_energies_ev, dtype=float)
    energies = np.asarray(energies_ev, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    velocity = np.asarray(velocity_h, dtype=np.complex128)
    if velocity.ndim != 3 or velocity.shape[1:] != (energies.size, energies.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {velocity.shape}")
    if occ.shape != energies.shape:
        raise ValueError(f"occupations has shape {occ.shape}, expected {energies.shape}")
    if eta_ev <= 0.0:
        raise ValueError(f"eta_ev must be positive, got {eta_ev}")
    return omega, energies, velocity, occ, int(velocity.shape[0])


def _omega_prefactor(
    omega: np.ndarray,
    prefactor: complex,
    k_weight: float,
    ndim: int,
    include_bz_factor: bool,
    omega_power_regularizer_ev: float,
) -> np.ndarray:
    factor = complex(prefactor) * float(k_weight)
    if include_bz_factor:
        factor /= (2.0 * math.pi) ** int(ndim)
    omega2 = omega * omega + float(omega_power_regularizer_ev) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(omega2 > 0.0, factor / omega2, 0.0)


def shift_current_conductivity_velocity_gauge(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    velocity_h: np.ndarray,
    occupations: np.ndarray,
    *,
    k_weight: float = 1.0,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    polarization_factor: np.ndarray | Sequence[Sequence[complex]] | None = None,
    omega_power_regularizer_ev: float = 0.0,
    skip_degenerate_energy_denominators: bool = True,
) -> ShiftCurrentVelocityGaugeResult:
    """Velocity-gauge dc shift-current tensor for Liu-Dai 2020 Eq. (4).

    This implements the transition-sum structure

    ```text
    sigma^c_ab(0) = (prefactor / w^2) sum_{Omega=+-w} sum_{lmn}
        Re[ phi_ab (f_l - f_n) v^a_nl v^b_lm v^c_mn
            / ((E_n-E_m-i eta) (E_n-E_l+Omega-i eta)) ]
    ```

    Energies and photon energies are in eV.  ``velocity_h`` is whatever the
    system adapter supplies; for this repo's TBG/TDBG adapters it is ``dH/dk``
    in eV nm.  Use the explicit velocity-gauge prefactor constants in this
    module only after deciding whether the workflow wants strict SI sheet units
    (A m/V^2) or the conventional 2D plotting unit (microA nm/V^2).

    ``skip_degenerate_energy_denominators=True`` omits terms whose static energy
    denominators are smaller than ``denominator_cutoff_ev``.  Setting it to
    ``False`` keeps exact diagonal denominators and lets the explicit ``i eta``
    regularize them.  This is a convention-sensitive choice for Liu-Dai Fig.4/5.
    """

    omega, energies, velocity, occ, ndim = _velocity_gauge_inputs(
        photon_energies_ev,
        energies_ev,
        velocity_h,
        occupations,
        eta_ev,
    )
    bands = _selected_band_list(energies.size, selected_bands)
    idx = np.asarray(bands, dtype=int)
    e = energies[idx]
    f = occ[idx]
    v = velocity[:, idx, :][:, :, idx]
    phi = _polarization_factor_matrix(ndim, polarization_factor)
    omega_prefactor = _omega_prefactor(
        omega,
        prefactor,
        k_weight,
        ndim,
        include_bz_factor,
        omega_power_regularizer_ev,
    )

    # Explicit axes: n,l,m.  The numerator follows v^a_nl v^b_lm v^c_mn.
    en_m = e[:, None] - e[None, :]  # n,m
    en_l = e[:, None] - e[None, :]  # n,l
    fln = f[None, :] - f[:, None]  # n,l == f_l - f_n
    small_nm = np.abs(en_m) <= float(denominator_cutoff_ev)
    small_nl = np.abs(en_l) <= float(denominator_cutoff_ev)
    if skip_degenerate_energy_denominators:
        mask = (~small_nm[:, None, :]) & (~small_nl[:, :, None]) & (fln[:, :, None] != 0.0)
        skipped = int(np.count_nonzero((small_nm[:, None, :] | small_nl[:, :, None]) & (fln[:, :, None] != 0.0)))
    else:
        mask = fln[:, :, None] != 0.0
        skipped = 0
    safe_den1 = np.where(mask, en_m[:, None, :] - 1.0j * float(eta_ev), 1.0 + 0.0j)
    numerator = np.einsum("anl,blm,cmn,ab->cabnlm", v, v, v, phi, optimize=True)
    out = np.zeros((omega.size, ndim, ndim, ndim), dtype=float)
    for sign in (-1.0, 1.0):
        safe_den2 = np.where(
            mask[None, :, :, :],
            en_l[None, :, :, None] + sign * omega[:, None, None, None] - 1.0j * float(eta_ev),
            1.0 + 0.0j,
        )
        resonance = np.where(
            mask[None, :, :, :],
            omega_prefactor[:, None, None, None] * fln[None, :, :, None] / (safe_den1[None, :, :, :] * safe_den2),
            0.0,
        )
        out += np.real(np.einsum("cabnlm,wnlm->wcab", numerator, resonance, optimize=True))
    return ShiftCurrentVelocityGaugeResult(
        photon_energies_ev=omega,
        conductivity=out,
        skipped_small_denominators=skipped,
    )


def shg_conductivity_velocity_gauge(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    velocity_h: np.ndarray,
    occupations: np.ndarray,
    *,
    k_weight: float = 1.0,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    polarization_factor: np.ndarray | Sequence[Sequence[complex]] | None = None,
    omega_power_regularizer_ev: float = 0.0,
    skip_degenerate_energy_denominators: bool = True,
) -> SHGConductivityResult:
    """Liu--Dai Eq.(4) compatibility three-velocity SHG tensor.

    This implements the transition-sum structure of Liu-Dai 2020 Eq. (4) for
    ``sigma^c_ab(2 omega)``:

    ```text
    sigma^c_ab(2w) ~ (prefactor / w^2) sum_{Omega=+-w} sum_{lmn}
        phi_ab (f_l - f_n) v^a_nl v^b_lm v^c_mn
        / [(E_n-E_m-2 Omega-i eta) (E_n-E_l+Omega-i eta)]
    ```

    Energies and photon energies are in eV.  This is a formula-level building
    block, not a final paper-unit implementation: callers must audit the global
    SI prefactor, velocity units, band truncation, degeneracy policy, and the
    paper-specific symmetrization/polarization factor ``phi_ab`` before claiming
    a reproduction.  ``skip_degenerate_energy_denominators`` has the same meaning
    as in ``shift_current_conductivity_velocity_gauge``.
    """

    omega, energies, velocity, occ, ndim = _velocity_gauge_inputs(
        photon_energies_ev,
        energies_ev,
        velocity_h,
        occupations,
        eta_ev,
    )
    bands = _selected_band_list(energies.size, selected_bands)
    idx = np.asarray(bands, dtype=int)
    e = energies[idx]
    f = occ[idx]
    v = velocity[:, idx, :][:, :, idx]
    phi = _polarization_factor_matrix(ndim, polarization_factor)
    omega_prefactor = _omega_prefactor(
        omega,
        prefactor,
        k_weight,
        ndim,
        include_bz_factor,
        omega_power_regularizer_ev,
    )

    en_m = e[:, None] - e[None, :]  # n,m
    en_l = e[:, None] - e[None, :]  # n,l
    fln = f[None, :] - f[:, None]  # n,l == f_l - f_n
    small_nm = np.abs(en_m) <= float(denominator_cutoff_ev)
    small_nl = np.abs(en_l) <= float(denominator_cutoff_ev)
    if skip_degenerate_energy_denominators:
        mask_static = (~small_nm[:, None, :]) & (~small_nl[:, :, None]) & (fln[:, :, None] != 0.0)
        skipped = int(np.count_nonzero((small_nm[:, None, :] | small_nl[:, :, None]) & (fln[:, :, None] != 0.0)))
    else:
        mask_static = fln[:, :, None] != 0.0
        skipped = 0
    numerator = np.einsum("anl,blm,cmn,ab->cabnlm", v, v, v, phi, optimize=True)
    out = np.zeros((omega.size, ndim, ndim, ndim), dtype=np.complex128)
    for sign in (-1.0, 1.0):
        den1 = en_m[None, :, None, :] - 2.0 * sign * omega[:, None, None, None] - 1.0j * float(eta_ev)
        den2 = en_l[None, :, :, None] + sign * omega[:, None, None, None] - 1.0j * float(eta_ev)
        resonance = np.where(
            mask_static[None, :, :, :],
            omega_prefactor[:, None, None, None] * fln[None, :, :, None] / (den1 * den2),
            0.0,
        )
        out += np.einsum("cabnlm,wnlm->wcab", numerator, resonance, optimize=True)
    return SHGConductivityResult(
        photon_energies_ev=omega,
        conductivity=out,
        skipped_small_denominators=skipped,
    )


def shg_susceptibility_from_conductivity(
    shg_conductivity: np.ndarray,
    photon_energies_ev: np.ndarray,
    *,
    epsilon0_f_per_m: float = VACUUM_PERMITTIVITY_F_PER_M,
    output: str = "m_per_v",
) -> np.ndarray:
    """Convert SHG photoconductivity to susceptibility.

    Implements Liu-Dai 2020's relation

    ```text
    chi^c_ab(2 omega) = i sigma^c_ab(2 omega) / (2 epsilon0 omega)
    ```

    where ``omega`` is angular frequency in rad/s.  ``photon_energies_ev`` are
    converted using ``omega = E / hbar``.  Use ``output="pm_per_v"`` for the
    plotting unit used in the paper.
    """

    sigma = np.asarray(shg_conductivity, dtype=np.complex128)
    photon = np.asarray(photon_energies_ev, dtype=float)
    omega_rad_s = photon / HBAR_EV_S
    shape = (photon.size,) + (1,) * (sigma.ndim - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = 1.0j * sigma / (2.0 * float(epsilon0_f_per_m) * omega_rad_s.reshape(shape))
    if output == "m_per_v":
        return chi
    if output == "pm_per_v":
        return chi * 1.0e12
    raise ValueError(f"Unsupported SHG susceptibility output unit {output!r}; expected 'm_per_v' or 'pm_per_v'")


__all__ = [
    "SHGConductivityResult",
    "ShiftCurrentVelocityGaugeResult",
    "LIU_DAI_2020_EQ4_PREFAC_A_M_PER_V2",
    "LIU_DAI_2020_EQ4_PREFAC_UA_NM_PER_V2",
    "LIU_DAI_2020_EQ4_SHG_PREFAC_S",
    "VELOCITY_GAUGE_2D_PREFAC_A_M_PER_V2",
    "VELOCITY_GAUGE_2D_PREFAC_UA_NM_PER_V2",
    "shift_current_conductivity_velocity_gauge",
    "shg_conductivity_velocity_gauge",
    "shg_susceptibility_from_conductivity",
]
