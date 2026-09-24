"""Typed production facade for finite-band harmonic generation.

The production SHG/THG backend is the complete finite-band Passos velocity-gauge
recursion.  Its causal prescription is independent-input adiabatic switching:
each input energy is continued as ``E_j -> E_j + i*gamma``.  Gauge and causal
prescription are coupled in a typed formulation so a density-matrix scattering
width cannot be applied to the velocity-gauge recursion by changing only its
poles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, TypeAlias

import numpy as np

from analysis.optical.core import OpticalResponseKindLike, canonical_response_kind
from analysis.optical.passos import (
    PassosHarmonicConductivityResult,
    PassosKPointData,
    PassosSHGSusceptibilityResult,
    passos_harmonic_conductivity_from_kpoint_data,
    passos_shg_susceptibility,
)


HarmonicGenerationKind = Literal[
    "second_harmonic_generation",
    "third_harmonic_generation",
]


@dataclass(frozen=True, kw_only=True)
class IndependentInputAdiabaticSwitching:
    """Continue every input photon energy by the same positive ``i*gamma``."""

    gamma_ev: float
    kind: Literal["independent_input_adiabatic"] = field(
        default="independent_input_adiabatic",
        init=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.gamma_ev, (bool, np.bool_)):
            raise TypeError("gamma_ev must be a real scalar, not bool")
        gamma = float(self.gamma_ev)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma_ev must be finite and positive")
        object.__setattr__(self, "gamma_ev", gamma)


@dataclass(frozen=True, kw_only=True)
class PassosFiniteBandVelocityGauge:
    """Complete finite-band velocity gauge with its valid causal prescription."""

    switching: IndependentInputAdiabaticSwitching
    kind: Literal["passos_finite_band_velocity_gauge"] = field(
        default="passos_finite_band_velocity_gauge",
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.switching, IndependentInputAdiabaticSwitching):
            raise TypeError(
                "switching must be IndependentInputAdiabaticSwitching; "
                "density-matrix scattering is not implemented in velocity gauge"
            )


HarmonicFormulation: TypeAlias = PassosFiniteBandVelocityGauge


def _harmonic_order(kind: OpticalResponseKindLike | str) -> tuple[HarmonicGenerationKind, int]:
    resolved = canonical_response_kind(kind)
    if resolved == "second_harmonic_generation":
        return resolved, 2
    if resolved == "third_harmonic_generation":
        return resolved, 3
    raise ValueError("harmonic response supports only SHG or THG response kinds")


def harmonic_response_from_kpoint_data(
    kind: OpticalResponseKindLike | str,
    photon_energies_ev: np.ndarray,
    kpoints: Iterable[PassosKPointData],
    *,
    formulation: HarmonicFormulation,
) -> PassosHarmonicConductivityResult:
    """Compute production SHG/THG from a typed gauge/prescription pair."""

    _, order = _harmonic_order(kind)
    if not isinstance(formulation, PassosFiniteBandVelocityGauge):
        raise TypeError("formulation must be PassosFiniteBandVelocityGauge")
    return passos_harmonic_conductivity_from_kpoint_data(
        photon_energies_ev,
        kpoints,
        order=order,
        adiabatic_gamma_ev=formulation.switching.gamma_ev,
    )


def shg_susceptibility_from_response(
    response: PassosHarmonicConductivityResult,
    *,
    effective_thickness_m: float | None = None,
) -> PassosSHGSusceptibilityResult:
    """Convert production SHG conductivity to sheet/effective-bulk susceptibility."""

    return passos_shg_susceptibility(
        response,
        effective_thickness_m=effective_thickness_m,
    )


__all__ = [
    "HarmonicFormulation",
    "HarmonicGenerationKind",
    "IndependentInputAdiabaticSwitching",
    "PassosFiniteBandVelocityGauge",
    "harmonic_response_from_kpoint_data",
    "shg_susceptibility_from_response",
]
