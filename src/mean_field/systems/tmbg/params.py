from __future__ import annotations

from dataclasses import dataclass, field
import math


GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_T0_EV = -3.1
DEFAULT_T1_EV = 0.36
DEFAULT_T3_EV = 0.283
DEFAULT_T4_EV = 0.138
DEFAULT_DELTA_EV = 0.015
VALID_BLG_STACKINGS = ("AB", "BA")
VALID_BERNAL_CONVENTIONS = ("park", "polshyn2020")


def default_omega(t1: float = DEFAULT_T1_EV) -> float:
    return float(t1 / 3.0)


def default_omega_prime(t1: float = DEFAULT_T1_EV) -> float:
    return float((-0.1835 * t1**2 + 1.036 * t1 - 0.06736) / 3.0)


def hopping_to_velocity(hopping_ev: float, lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM) -> float:
    return float(math.sqrt(3.0) * abs(hopping_ev) * lattice_constant_nm / 2.0)


@dataclass(frozen=True)
class TMBGParameters:
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    t0: float = DEFAULT_T0_EV
    t1: float = DEFAULT_T1_EV
    t3: float = DEFAULT_T3_EV
    t4: float = DEFAULT_T4_EV
    delta: float = DEFAULT_DELTA_EV
    omega: float = field(default_factory=default_omega)
    omega_prime: float = field(default_factory=default_omega_prime)
    interlayer_potential: float = 0.0
    staggered_potential: float = 0.0
    blg_stacking: str = "AB"
    bernal_convention: str = "park"
    model_name: str = "full"

    vf: float = field(init=False)
    v3: float = field(init=False)
    v4: float = field(init=False)

    def __post_init__(self) -> None:
        if self.blg_stacking not in VALID_BLG_STACKINGS:
            raise ValueError(f"Expected blg_stacking in {VALID_BLG_STACKINGS}, got {self.blg_stacking!r}")
        if self.bernal_convention not in VALID_BERNAL_CONVENTIONS:
            raise ValueError(f"Expected bernal_convention in {VALID_BERNAL_CONVENTIONS}, got {self.bernal_convention!r}")
        a_nm = float(self.graphene_lattice_constant_nm)
        object.__setattr__(self, "vf", hopping_to_velocity(self.t0, a_nm))
        object.__setattr__(self, "v3", hopping_to_velocity(self.t3, a_nm))
        object.__setattr__(self, "v4", hopping_to_velocity(self.t4, a_nm))

    @classmethod
    def minimal(
        cls,
        *,
        interlayer_potential: float = 0.0,
        staggered_potential: float = 0.0,
        blg_stacking: str = "AB",
        bernal_convention: str = "park",
    ) -> "TMBGParameters":
        return cls(
            t3=0.0,
            t4=0.0,
            delta=0.0,
            interlayer_potential=interlayer_potential,
            staggered_potential=staggered_potential,
            blg_stacking=blg_stacking,
            bernal_convention=bernal_convention,
            model_name="minimal",
        )

    @classmethod
    def full(
        cls,
        *,
        interlayer_potential: float = 0.0,
        staggered_potential: float = 0.0,
        blg_stacking: str = "AB",
        bernal_convention: str = "park",
    ) -> "TMBGParameters":
        return cls(
            interlayer_potential=interlayer_potential,
            staggered_potential=staggered_potential,
            blg_stacking=blg_stacking,
            bernal_convention=bernal_convention,
            model_name="full",
        )

    @classmethod
    def polshyn2020(
        cls,
        *,
        interlayer_potential: float = -0.033,
        staggered_potential: float = 0.0,
        blg_stacking: str = "BA",
    ) -> "TMBGParameters":
        """Parameter/convention set from Polshyn 2020 supplementary Eq. (S1)-(S6)."""

        return cls(
            t0=-2.61,
            t1=0.361,
            t3=0.283,
            t4=0.138,
            delta=0.0,
            omega=0.117,
            omega_prime=0.7 * 0.117,
            interlayer_potential=interlayer_potential,
            staggered_potential=staggered_potential,
            blg_stacking=blg_stacking,
            bernal_convention="polshyn2020",
            model_name="polshyn2020",
        )
