from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from analysis.response_derivative_gauge import hamiltonian_gauge_data
from analysis.shift_current import fermi_occupation
from analysis.optical.linear import LinearConductivityResult, linear_conductivity_tensor_from_gauge_data
from analysis.optical.shg import (
    SHGConductivityResult,
    ShiftCurrentVelocityGaugeResult,
    shift_current_conductivity_velocity_gauge,
    shg_conductivity_velocity_gauge,
)


@dataclass(frozen=True)
class OpticalKPointData:
    """One k-point payload for generic optical grid integration.

    System adapters own Hamiltonians, eigenvectors, basis conventions, and unit
    conversions.  The common optical layer only needs energies, Hamiltonian-gauge
    velocity/``dH/dk`` matrices, occupations, and the k-point integration weight.
    """

    energies_ev: np.ndarray
    velocity_h: np.ndarray
    occupations: np.ndarray
    weight: float = 1.0


def optical_kpoint_data_from_eigensystem(
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: Sequence[np.ndarray] | np.ndarray,
    *,
    weight: float = 1.0,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
) -> OpticalKPointData:
    """Build :class:`OpticalKPointData` from a system eigensystem.

    ``dhdk`` is interpreted through the shared
    :mod:`analysis.response_derivative_gauge` Hamiltonian-gauge layer used by the
    shift/injection modules.  Final units are inherited from the system adapter
    and must be documented by the workflow before applying SI prefactors.
    """

    gauge = hamiltonian_gauge_data(
        np.asarray(energies_ev, dtype=float),
        np.asarray(eigenvectors, dtype=np.complex128),
        np.asarray(dhdk, dtype=np.complex128),
        denominator_cutoff=float(denominator_cutoff_ev),
    )
    return OpticalKPointData(
        energies_ev=gauge.energies,
        velocity_h=gauge.velocity_h,
        occupations=fermi_occupation(gauge.energies, mu_ev=mu_ev, temperature_k=temperature_k),
        weight=float(weight),
    )


def _nonempty_kpoints(kpoints: Iterable[OpticalKPointData]) -> list[OpticalKPointData]:
    points = list(kpoints)
    if not points:
        raise ValueError("At least one k-point is required for optical grid accumulation")
    return points


def _materialize_selected_bands(
    selected_bands: Iterable[int] | None,
) -> tuple[object, ...] | None:
    if selected_bands is None:
        return None
    return tuple(selected_bands)


def linear_conductivity_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0j,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
) -> LinearConductivityResult:
    """Accumulate Liu-Dai Eq. (20)-style optical conductivity over k points."""

    points = _nonempty_kpoints(kpoints)
    bands = _materialize_selected_bands(selected_bands)
    total: np.ndarray | None = None
    skipped = 0
    omega = np.asarray(photon_energies_ev, dtype=float)
    for point in points:
        partial = linear_conductivity_tensor_from_gauge_data(
            omega,
            point.energies_ev,
            point.velocity_h,
            point.occupations,
            k_weight=point.weight,
            eta_ev=eta_ev,
            prefactor=prefactor,
            include_bz_factor=include_bz_factor,
            denominator_cutoff_ev=denominator_cutoff_ev,
            selected_bands=bands,
        )
        total = partial.conductivity.copy() if total is None else total + partial.conductivity
        skipped += partial.skipped_small_denominators
    assert total is not None
    return LinearConductivityResult(
        photon_energies_ev=omega,
        conductivity=total,
        skipped_small_denominators=skipped,
    )


def shift_current_velocity_gauge_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    polarization_factor: np.ndarray | Sequence[Sequence[complex]] | None = None,
    omega_power_regularizer_ev: float = 0.0,
    skip_degenerate_energy_denominators: bool = True,
) -> ShiftCurrentVelocityGaugeResult:
    """Accumulate Liu-Dai Eq. (4) velocity-gauge dc shift tensor over k."""

    points = _nonempty_kpoints(kpoints)
    bands = _materialize_selected_bands(selected_bands)
    total: np.ndarray | None = None
    skipped = 0
    omega = np.asarray(photon_energies_ev, dtype=float)
    for point in points:
        partial = shift_current_conductivity_velocity_gauge(
            omega,
            point.energies_ev,
            point.velocity_h,
            point.occupations,
            k_weight=point.weight,
            eta_ev=eta_ev,
            prefactor=prefactor,
            include_bz_factor=include_bz_factor,
            denominator_cutoff_ev=denominator_cutoff_ev,
            selected_bands=bands,
            polarization_factor=polarization_factor,
            omega_power_regularizer_ev=omega_power_regularizer_ev,
            skip_degenerate_energy_denominators=skip_degenerate_energy_denominators,
        )
        total = partial.conductivity.copy() if total is None else total + partial.conductivity
        skipped += partial.skipped_small_denominators
    assert total is not None
    return ShiftCurrentVelocityGaugeResult(
        photon_energies_ev=omega,
        conductivity=total,
        skipped_small_denominators=skipped,
    )


def shg_conductivity_from_kpoint_data(
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[OpticalKPointData],
    *,
    eta_ev: float = 1.0e-3,
    prefactor: complex = 1.0,
    include_bz_factor: bool = True,
    denominator_cutoff_ev: float = 1.0e-12,
    selected_bands: Iterable[int] | None = None,
    polarization_factor: np.ndarray | Sequence[Sequence[complex]] | None = None,
    omega_power_regularizer_ev: float = 0.0,
    skip_degenerate_energy_denominators: bool = True,
) -> SHGConductivityResult:
    """Accumulate preliminary Liu-Dai Eq. (4) SHG tensor over k points."""

    points = _nonempty_kpoints(kpoints)
    bands = _materialize_selected_bands(selected_bands)
    total: np.ndarray | None = None
    skipped = 0
    omega = np.asarray(photon_energies_ev, dtype=float)
    for point in points:
        partial = shg_conductivity_velocity_gauge(
            omega,
            point.energies_ev,
            point.velocity_h,
            point.occupations,
            k_weight=point.weight,
            eta_ev=eta_ev,
            prefactor=prefactor,
            include_bz_factor=include_bz_factor,
            denominator_cutoff_ev=denominator_cutoff_ev,
            selected_bands=bands,
            polarization_factor=polarization_factor,
            omega_power_regularizer_ev=omega_power_regularizer_ev,
            skip_degenerate_energy_denominators=skip_degenerate_energy_denominators,
        )
        total = partial.conductivity.copy() if total is None else total + partial.conductivity
        skipped += partial.skipped_small_denominators
    assert total is not None
    return SHGConductivityResult(
        photon_energies_ev=omega,
        conductivity=total,
        skipped_small_denominators=skipped,
    )


__all__ = [
    "OpticalKPointData",
    "linear_conductivity_from_kpoint_data",
    "optical_kpoint_data_from_eigensystem",
    "shift_current_velocity_gauge_from_kpoint_data",
    "shg_conductivity_from_kpoint_data",
]
