"""Bulk parameters for the InAs/GaSb/AlSb Kane model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


# CODATA electron mass and hbar, converted to eV nm^2.
HBAR2_OVER_2M0_EV_NM2 = 0.0380998211615486

SUPPLEMENTARY_TABLE_1_PROVENANCE = (
    "Du et al., Nature Communications 8, 1971 (2017), "
    "Supplementary Table 1"
)
INFERRED_KANE_EP_EV = 22.5
INFERRED_EP_PROVENANCE = (
    "Common E_P = 22.5 eV inferred by inverse Kane renormalization of the "
    "Supplementary Table 1 gamma values against standard Luttinger gammas"
)
VURGAFTMAN_GAMMA_PROVENANCE = (
    "Standard Luttinger gammas from Vurgaftman, Meyer, and Ram-Mohan, "
    "J. Appl. Phys. 89, 5815 (2001)"
)
GASB_GAMMA2_CORRECTION = (
    "Supplementary Table 1 prints GaSb gamma2 = 8.18; inverse Kane "
    "renormalization with E_P = 22.5 eV identifies this as a typo, so the "
    "model uses gamma2 = 0.08."
)


@dataclass(frozen=True)
class KaneMaterial:
    """Material parameters in eV and nm units.

    The supplement's primed gammas obey ``gamma1' = gamma1_L - EP/(3 Eg)``
    and ``gamma(2,3)' = gamma(2,3)_L - EP/(6 Eg)``. Comparing them with the
    Vurgaftman standard gammas gives the common inferred ``EP`` below.
    ``P0 = sqrt(EP * hbar^2 / (2 m0))`` therefore has units of eV nm.
    """

    name: str
    Eg: float
    Ev: float
    delta_so: float
    Ac: float
    gamma1: float
    gamma2: float
    gamma3: float
    epsilon_r: float
    EP: float = INFERRED_KANE_EP_EV
    bulk_provenance: str = SUPPLEMENTARY_TABLE_1_PROVENANCE
    ep_provenance: str = INFERRED_EP_PROVENANCE
    standard_gamma_provenance: str = VURGAFTMAN_GAMMA_PROVENANCE
    printed_gamma2: float | None = None
    gamma2_correction: str | None = None

    @property
    def P0(self) -> float:
        return sqrt(self.EP * HBAR2_OVER_2M0_EV_NM2)


MATERIALS = {
    "InAs": KaneMaterial(
        name="InAs",
        Eg=0.417,
        Ev=-0.417,
        delta_so=0.390,
        Ac=-0.260,
        gamma1=2.01,
        gamma2=-0.49,
        gamma3=0.21,
        epsilon_r=14.55,
    ),
    "GaSb": KaneMaterial(
        name="GaSb",
        Eg=0.812,
        Ev=0.143,
        delta_so=0.760,
        Ac=0.090,
        gamma1=4.16,
        gamma2=0.08,
        gamma3=1.38,
        epsilon_r=15.69,
        printed_gamma2=8.18,
        gamma2_correction=GASB_GAMMA2_CORRECTION,
    ),
    "AlSb": KaneMaterial(
        name="AlSb",
        Eg=2.386,
        Ev=-0.237,
        delta_so=0.676,
        Ac=-0.060,
        gamma1=2.04,
        gamma2=-0.38,
        gamma3=0.40,
        epsilon_r=14.40,
    ),
}


__all__ = [
    "GASB_GAMMA2_CORRECTION",
    "HBAR2_OVER_2M0_EV_NM2",
    "INFERRED_EP_PROVENANCE",
    "INFERRED_KANE_EP_EV",
    "KaneMaterial",
    "MATERIALS",
    "SUPPLEMENTARY_TABLE_1_PROVENANCE",
    "VURGAFTMAN_GAMMA_PROVENANCE",
]
