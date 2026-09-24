from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import math
import numpy as np

from analysis.shift_current import (
    Component,
    ShiftCurrentComponent,
    add_transitions_to_integral,
    berry_connection_matrix,
    component_from_any,
    component_label,
    fermi_occupation,
    positive_transition_pairs,
    velocity_matrices,
)


@dataclass(frozen=True)
class InjectionCurrentTensors:
    """Reusable one-k-point ingredients for injection current / CPGE.

    The response kernel is the length-gauge injection-current object

    ``D_mn^{c,ab} = Delta_mn^c r_nm^b r_mn^a``

    for an occupied initial band ``n`` and empty final band ``m``.  The same
    Hamiltonian-gauge Berry connection ``r`` used by ``analysis.shift_current``
    is reused here; no raw eigenvector phase derivatives are taken.
    """

    energies_ev: np.ndarray
    occupations: np.ndarray
    velocity_h: np.ndarray
    berry_connection: np.ndarray

    @property
    def r(self) -> np.ndarray:
        return self.berry_connection

    @property
    def D(self) -> np.ndarray:
        return self.velocity_h


def precompute_injection_current_tensors(
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: Sequence[np.ndarray] | np.ndarray,
    *,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
) -> InjectionCurrentTensors:
    """Build Hamiltonian-gauge ingredients for injection current.

    ``dhdk`` should use the same momentum units as the system adapter.  In toy
    models with ``e = hbar = 1`` this is also the band velocity matrix; in SI
    workflows the final prefactor/unit conversion is a caller responsibility.
    """

    energies = np.asarray(energies_ev, dtype=float)
    velocity_h = velocity_matrices(eigenvectors, dhdk)
    berry_connection = berry_connection_matrix(
        velocity_h,
        energies,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    occupations = fermi_occupation(energies, mu_ev=mu_ev, temperature_k=temperature_k)
    return InjectionCurrentTensors(
        energies_ev=energies,
        occupations=occupations,
        velocity_h=velocity_h,
        berry_connection=berry_connection,
    )


def injection_component_kernel_from_gauge_pair(
    velocity_h: np.ndarray,
    berry_connection: np.ndarray,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
) -> complex:
    """Return ``D_mn^{c,ab}`` for one ordered optical transition.

    ``initial_band`` is the occupied band ``n`` and ``final_band`` is the empty
    band ``m`` in ``hbar omega_mn = E_m - E_n``.  The component label follows
    the shared convention ``c;ab`` / ``cab`` where ``c`` is output current and
    ``a,b`` are optical-field indices.
    """

    comp = component_from_any(component)
    c, a, b = comp.as_tuple
    V = np.asarray(velocity_h, dtype=np.complex128)
    r = np.asarray(berry_connection, dtype=np.complex128)
    n = int(initial_band)
    m = int(final_band)
    if V.ndim != 3 or r.ndim != 3 or V.shape != r.shape:
        raise ValueError(
            "velocity_h and berry_connection must both have shape "
            f"(ndim,nb,nb), got {V.shape} and {r.shape}"
        )
    if max(c, a, b) >= V.shape[0]:
        raise ValueError(f"component {component!r} needs axis {max(c, a, b)}, but ndim={V.shape[0]}")
    delta_v = V[c, m, m] - V[c, n, n]
    return complex(delta_v * r[b, n, m] * r[a, m, n])


def injection_component_kernel(
    tensors: InjectionCurrentTensors,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
) -> complex:
    """Return the occupation-independent injection kernel for one transition."""

    return injection_component_kernel_from_gauge_pair(
        tensors.velocity_h,
        tensors.berry_connection,
        initial_band,
        final_band,
        component,
    )


def injection_transition_weight(
    tensors: InjectionCurrentTensors,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
) -> complex:
    """Return ``(f_n - f_m) D_mn^{c,ab}`` for one transition."""

    n = int(initial_band)
    m = int(final_band)
    fnm = float(tensors.occupations[n] - tensors.occupations[m])
    return fnm * injection_component_kernel(tensors, n, m, component)


def positive_injection_transition_terms(
    tensors: InjectionCurrentTensors,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transition energies and complex injection weights.

    This mirrors ``analysis.shift_current.positive_transition_terms`` but uses
    the injection kernel ``Delta v * r * r`` instead of a generalized derivative.
    """

    transitions = positive_transition_pairs(
        tensors.energies_ev,
        tensors.occupations,
        selected_bands=selected_bands,
        min_transition_ev=min_transition_ev,
        max_transition_ev=max_transition_ev,
        min_abs_occupation_diff=min_abs_occupation_diff,
    )
    energies: list[float] = []
    weights: list[complex] = []
    for n, m, transition_ev, _fnm in transitions:
        energies.append(float(transition_ev))
        weights.append(injection_transition_weight(tensors, n, m, component))
    return np.asarray(energies, dtype=float), np.asarray(weights, dtype=np.complex128)


def accumulate_injection_spectrum(
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight: float,
    eta_ev: float,
    include_bz_factor: bool = True,
    normalized_lorentzian: bool = True,
    prefactor: complex = -2.0 * math.pi,
    relaxation_time: float = 1.0,
) -> np.ndarray:
    """Accumulate a complex injection spectrum from transition tables.

    The default prefactor is the toy-unit length-gauge factor ``-2*pi`` from
    Lin--Hsu Eq. (9), with ``e = hbar = tau = 1``.  Production SI conversion and
    scattering-time choices should be applied explicitly by callers.
    """

    integral = np.zeros_like(np.asarray(photon_energies_ev, dtype=float), dtype=np.complex128)
    add_transitions_to_integral(
        integral,
        photon_energies_ev,
        transition_energies_ev,
        transition_weights,
        k_weight_nm_inv_sq=float(k_weight),
        eta_ev=float(eta_ev),
        normalized_lorentzian=bool(normalized_lorentzian),
        include_bz_factor=bool(include_bz_factor),
    )
    return complex(prefactor) * float(relaxation_time) * integral


def injection_spectra_from_transition_table(
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight: float,
    eta_ev: float,
    include_bz_factor: bool = True,
    normalized_lorentzian: bool = True,
    prefactor: complex = -2.0 * math.pi,
    relaxation_time: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build complex injection spectra for several named components."""

    return {
        name: accumulate_injection_spectrum(
            photon_energies_ev,
            transitions,
            weights,
            k_weight=k_weight,
            eta_ev=eta_ev,
            include_bz_factor=include_bz_factor,
            normalized_lorentzian=normalized_lorentzian,
            prefactor=prefactor,
            relaxation_time=relaxation_time,
        )
        for name, (transitions, weights) in transition_table.items()
    }


def linear_injection_part(response: np.ndarray) -> np.ndarray:
    """Linear-polarization injection component (real part)."""

    return np.real(np.asarray(response))


def cpge_part(response: np.ndarray) -> np.ndarray:
    """Circular-polarization injection / CPGE component (imaginary part)."""

    return np.imag(np.asarray(response))


__all__ = [
    "InjectionCurrentTensors",
    "accumulate_injection_spectrum",
    "component_label",
    "cpge_part",
    "injection_component_kernel",
    "injection_component_kernel_from_gauge_pair",
    "injection_spectra_from_transition_table",
    "injection_transition_weight",
    "linear_injection_part",
    "positive_injection_transition_terms",
    "precompute_injection_current_tensors",
]
