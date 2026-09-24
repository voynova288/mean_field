from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import operator

import numpy as np

# SI conductance constants.  `E2_OVER_H_S` is the quantized Hall conductance
# unit.  `E2_OVER_HBAR_S = 2*pi*e^2/h` is the natural Kubo prefactor appearing
# in Liu-Dai 2020 Eq. (20) when the BZ integral is written with `(2*pi)^-2`.
# The response kernels still leave velocity/momentum units to the system adapter;
# use these constants only after the adapter's unit convention has been audited.
E_CHARGE_C = 1.602176634e-19
H_PLANCK_J_S = 6.62607015e-34
E2_OVER_H_S = E_CHARGE_C**2 / H_PLANCK_J_S
E2_OVER_HBAR_S = 2.0 * math.pi * E2_OVER_H_S


@dataclass(frozen=True)
class LinearConductivityResult:
    """Immutable frequency-dependent optical conductivity tensor.

    ``conductivity`` has shape ``(n_omega, ndim, ndim)``.  The units are set by
    the caller-supplied ``prefactor`` and by the units of ``velocity_h`` and
    ``k_weight``.  For Kerr/Faraday, callers should convert the result to sheet
    conductance in Siemens before calling :mod:`analysis.optical.magneto`.
    """

    photon_energies_ev: np.ndarray
    conductivity: np.ndarray
    skipped_small_denominators: int

    def __post_init__(self) -> None:
        omega = np.array(self.photon_energies_ev, dtype=float, copy=True)
        conductivity = np.array(self.conductivity, dtype=np.complex128, copy=True)
        if omega.ndim != 1 or np.any(~np.isfinite(omega)):
            raise ValueError(
                "photon_energies_ev must be a finite one-dimensional array"
            )
        if (
            conductivity.ndim != 3
            or conductivity.shape[0] != omega.size
            or conductivity.shape[1] != conductivity.shape[2]
        ):
            raise ValueError(
                "conductivity must have shape (n_omega,ndim,ndim), "
                f"got {conductivity.shape} for n_omega={omega.size}"
            )
        if np.any(~np.isfinite(conductivity)):
            raise ValueError("conductivity must contain only finite values")
        skipped = operator.index(self.skipped_small_denominators)
        if skipped < 0:
            raise ValueError("skipped_small_denominators must be non-negative")
        omega.setflags(write=False)
        conductivity.setflags(write=False)
        object.__setattr__(self, "photon_energies_ev", omega)
        object.__setattr__(self, "conductivity", conductivity)
        object.__setattr__(self, "skipped_small_denominators", skipped)


def _selected_band_list(nb: int, selected_bands: Iterable[int] | None) -> list[int]:
    if selected_bands is None:
        return list(range(nb))
    bands = []
    for value in selected_bands:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("selected_bands entries must be integer indices, not bool")
        try:
            bands.append(operator.index(value))
        except TypeError as error:
            raise TypeError(
                f"selected band {value!r} is not an integer index"
            ) from error
    if not bands:
        raise ValueError("selected_bands must not be empty")
    if len(set(bands)) != len(bands):
        raise ValueError("selected_bands must not contain duplicates")
    for band in bands:
        if band < 0 or band >= nb:
            raise ValueError(f"selected band {band} is outside [0,{nb})")
    return bands


def linear_conductivity_tensor_from_gauge_data(
    photon_energies_ev: np.ndarray,
    energies_ev: np.ndarray,
    velocity_h: np.ndarray,
    occupations: np.ndarray,
    *,
    k_weight: float = 1.0,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0j,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
) -> LinearConductivityResult:
    """Interband Kubo optical-conductivity tensor.

    This is the reusable transition-sum form of Liu-Dai 2020 Eq. (20):

    ```text
    sigma_ab(omega) ~ i sum_{m!=n} v^a_mn v^b_nm
        (f_n - f_m) / (E_n - E_m)
        / (E_n - E_m - hbar*omega - i*eta)
    ```

    The formula is evaluated with energies and photon energies in eV.  The
    caller controls the overall prefactor because continuum models in this repo
    may provide ``velocity_h`` as ``dH/dk`` in eV*length, while SI workflows may
    use true velocity matrix elements.  ``k_weight`` is the integration weight in
    the adapter's reciprocal-unit convention; when ``include_bz_factor=True`` it
    is divided by ``(2*pi)^ndim``.

    Intraband/Drude terms are not included.  Add them in a separate audited
    layer if needed for metallic low-frequency optics.
    """

    omega = np.asarray(photon_energies_ev, dtype=float)
    energies = np.asarray(energies_ev, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    velocity = np.asarray(velocity_h, dtype=np.complex128)
    if omega.ndim != 1 or np.any(~np.isfinite(omega)):
        raise ValueError("photon_energies_ev must be a finite one-dimensional array")
    if energies.ndim != 1 or np.any(~np.isfinite(energies)):
        raise ValueError("energies_ev must be a finite one-dimensional array")
    if np.any(~np.isfinite(occ)) or np.any(~np.isfinite(velocity)):
        raise ValueError("occupations and velocity_h must contain only finite values")
    if velocity.ndim != 3 or velocity.shape[1:] != (energies.size, energies.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {velocity.shape}")
    if occ.shape != energies.shape:
        raise ValueError(f"occupations has shape {occ.shape}, expected {energies.shape}")
    if eta_ev <= 0.0 or not np.isfinite(eta_ev):
        raise ValueError(f"eta_ev must be finite and positive, got {eta_ev}")
    if denominator_cutoff_ev < 0.0 or not np.isfinite(denominator_cutoff_ev):
        raise ValueError(
            "denominator_cutoff_ev must be finite and non-negative, "
            f"got {denominator_cutoff_ev}"
        )
    if not np.isfinite(k_weight):
        raise ValueError(f"k_weight must be finite, got {k_weight}")
    ndim = velocity.shape[0]
    factor = complex(prefactor) * float(k_weight)
    if include_bz_factor:
        factor /= (2.0 * math.pi) ** ndim
    bands = _selected_band_list(energies.size, selected_bands)
    e = energies[np.asarray(bands, dtype=int)]
    f = occ[np.asarray(bands, dtype=int)]
    v = velocity[:, np.asarray(bands, dtype=int), :][:, :, np.asarray(bands, dtype=int)]

    # Axes below are explicit: static[n,m] = E_n - E_m, while v[a,m,n]
    # and v[b,n,m] follow Liu-Dai Eq. (20).  Keeping this vectorized is
    # important for full-window TBG/TDBG benchmarks.
    static = e[:, None] - e[None, :]
    occupation_diff = f[:, None] - f[None, :]
    offdiag = ~np.eye(e.size, dtype=bool)
    small = np.abs(static) <= float(denominator_cutoff_ev)
    skipped = int(np.count_nonzero(offdiag & small))
    mask = offdiag & ~small & (occupation_diff != 0.0)
    safe_static = np.where(mask, static, 1.0)
    dynamic = static[None, :, :] - omega[:, None, None] - 1.0j * float(eta_ev)
    resonance = np.where(mask[None, :, :], occupation_diff[None, :, :] / safe_static[None, :, :] / dynamic, 0.0)
    sigma = factor * np.einsum("amn,bnm,wnm->wab", v, v, resonance, optimize=True)
    return LinearConductivityResult(
        photon_energies_ev=omega,
        conductivity=sigma,
        skipped_small_denominators=skipped,
    )


__all__ = [
    "E2_OVER_H_S",
    "E2_OVER_HBAR_S",
    "E_CHARGE_C",
    "H_PLANCK_J_S",
    "LinearConductivityResult",
    "linear_conductivity_tensor_from_gauge_data",
]
