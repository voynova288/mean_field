from __future__ import annotations

from typing import Mapping
import math

import numpy as np

from .conventions import SHIFT_CURRENT_PREFAC_UA_NM_PER_V2, ShiftCurrentConvention

def lorentzian_delta(
    photon_energies_ev: np.ndarray,
    transition_energy_ev: float,
    eta_ev: float,
    *,
    normalized: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> np.ndarray:
    """Lorentzian delta replacement in eV units.

    ``normalized=True`` returns ``(eta/pi)/(x^2+eta^2)``.  Joya's plotting text
    uses ``eta/(x^2+eta^2)``; pass ``JOYA_EQ7_GEOMETRIC_CONVENTION`` to make
    that choice explicit.
    """

    if convention is not None:
        normalized = bool(convention.normalized_lorentzian)
    photon = np.asarray(photon_energies_ev, dtype=float)
    eta = float(eta_ev)
    if eta <= 0.0:
        raise ValueError(f"eta_ev must be positive, got {eta_ev}")
    diff = photon - float(transition_energy_ev)
    out = eta / (diff * diff + eta * eta)
    if normalized:
        out = out / math.pi
    return out


def add_transitions_to_integral(
    integral: np.ndarray,
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
    normalized_lorentzian: bool = True,
    include_bz_factor: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> None:
    """Accumulate weighted transitions into a spectrum/integral in-place."""

    if np.asarray(transition_energies_ev).size == 0:
        return
    factor = float(k_weight_nm_inv_sq)
    if include_bz_factor:
        factor /= (2.0 * math.pi) ** 2
    for transition_ev, weight in zip(np.asarray(transition_energies_ev, dtype=float), np.asarray(transition_weights, dtype=np.complex128), strict=True):
        integral += factor * weight * lorentzian_delta(
            photon_energies_ev,
            float(transition_ev),
            eta_ev,
            normalized=bool(normalized_lorentzian),
            convention=convention,
        )


def conductivity_from_integral(
    integral: np.ndarray,
    *,
    prefactor: float = SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    phase: complex = -1.0j,
) -> np.ndarray:
    """Apply a caller-chosen conductivity prefactor to an integrated kernel."""

    return np.real(complex(phase) * float(prefactor) * np.asarray(integral))


def spectra_from_transition_table(
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
    prefactor: float = SHIFT_CURRENT_PREFAC_UA_NM_PER_V2,
    prefactor_phase: complex = -1.0j,
    normalized_lorentzian: bool = True,
    include_bz_factor: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> dict[str, np.ndarray]:
    """Build spectra for several named components from transition arrays."""

    spectra: dict[str, np.ndarray] = {}
    for name, (transition_energies, weights) in transition_table.items():
        integral = np.zeros_like(photon_energies_ev, dtype=np.complex128)
        add_transitions_to_integral(
            integral,
            photon_energies_ev,
            np.asarray(transition_energies, dtype=float),
            np.asarray(weights, dtype=np.complex128),
            k_weight_nm_inv_sq=k_weight_nm_inv_sq,
            eta_ev=eta_ev,
            normalized_lorentzian=normalized_lorentzian,
            include_bz_factor=include_bz_factor,
            convention=convention,
        )
        spectra[name] = conductivity_from_integral(integral, prefactor=prefactor, phase=prefactor_phase)
    return spectra


def fermi_window_indices(fermi_grid_ev: np.ndarray, initial_energy_ev: float, final_energy_ev: float) -> tuple[int, int]:
    """Return the Fermi-grid interval where ``initial <= E_F < final``."""

    grid = np.asarray(fermi_grid_ev, dtype=float)
    lo = min(float(initial_energy_ev), float(final_energy_ev))
    hi = max(float(initial_energy_ev), float(final_energy_ev))
    start = int(np.searchsorted(grid, lo, side="left"))
    end = int(np.searchsorted(grid, hi, side="left"))
    return max(0, min(start, grid.size)), max(0, min(end, grid.size))


def accumulate_fermi_omega_heatmap(
    heatmap: np.ndarray,
    fermi_grid_ev: np.ndarray,
    photon_energies_ev: np.ndarray,
    *,
    initial_energy_ev: float,
    final_energy_ev: float,
    transition_energy_ev: float,
    amplitude: complex | float,
    eta_ev: float,
    k_weight: float = 1.0,
    normalized_lorentzian: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> bool:
    """Accumulate one transition into an ``(E_F, omega)`` heatmap."""

    start, end = fermi_window_indices(fermi_grid_ev, initial_energy_ev, final_energy_ev)
    if start >= end:
        return False
    broaden = lorentzian_delta(
        photon_energies_ev,
        transition_energy_ev,
        eta_ev,
        normalized=normalized_lorentzian,
        convention=convention,
    )
    heatmap[start:end, :] += float(k_weight) * amplitude * broaden[None, :]
    return True


__all__ = [
    "accumulate_fermi_omega_heatmap",
    "add_transitions_to_integral",
    "conductivity_from_integral",
    "fermi_window_indices",
    "lorentzian_delta",
    "spectra_from_transition_table",
]
