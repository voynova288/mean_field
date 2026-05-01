from __future__ import annotations

from dataclasses import dataclass, field
import math


GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_HBAR_V_OVER_A_EV = 2.1354
DEFAULT_GAMMA1_EV = 0.4
DEFAULT_GAMMA3_EV = 0.32
DEFAULT_GAMMA4_EV = 0.044
DEFAULT_DELTA_PRIME_EV = 0.050
DEFAULT_U_EV = 0.0797
DEFAULT_U_PRIME_EV = 0.0975
DEFAULT_BETA = 3.14
DEFAULT_POISSON_RATIO = 0.16
VALID_STACKINGS = ("AB-AB", "AB-BA")
VALID_VALLEYS = (-1, 1)


def fermi_velocity_from_hbar_v_over_a(
    hbar_v_over_a: float,
    lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> float:
    return float(hbar_v_over_a * lattice_constant_nm)


def remote_velocity(
    hopping_ev: float,
    lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> float:
    return float(math.sqrt(3.0) * lattice_constant_nm * hopping_ev / 2.0)


@dataclass(frozen=True)
class TDBGParameters:
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    beta: float = DEFAULT_BETA
    poisson_ratio: float = DEFAULT_POISSON_RATIO
    phi_deg: float = 0.0
    epsilon: float = 0.0

    stacking: str = "AB-AB"
    valley: int = 1
    Delta: float = 0.0

    hbar_v_over_a: float = DEFAULT_HBAR_V_OVER_A_EV
    gamma1: float = DEFAULT_GAMMA1_EV
    gamma3: float = DEFAULT_GAMMA3_EV
    gamma4: float = DEFAULT_GAMMA4_EV
    delta_prime: float = DEFAULT_DELTA_PRIME_EV
    u: float = DEFAULT_U_EV
    u_prime: float = DEFAULT_U_PRIME_EV
    model_name: str = "full"

    vf: float = field(init=False)
    v3: float = field(init=False)
    v4: float = field(init=False)
    phi_rad: float = field(init=False)

    def __post_init__(self) -> None:
        if self.stacking not in VALID_STACKINGS:
            raise ValueError(f"Expected stacking in {VALID_STACKINGS}, got {self.stacking}")
        if int(self.valley) not in VALID_VALLEYS:
            raise ValueError(f"Expected valley in {VALID_VALLEYS}, got {self.valley}")

        a_nm = float(self.graphene_lattice_constant_nm)
        object.__setattr__(self, "vf", fermi_velocity_from_hbar_v_over_a(self.hbar_v_over_a, a_nm))
        object.__setattr__(self, "v3", remote_velocity(self.gamma3, a_nm))
        object.__setattr__(self, "v4", remote_velocity(self.gamma4, a_nm))
        object.__setattr__(self, "phi_rad", float(self.phi_deg) * math.pi / 180.0)

    @classmethod
    def minimal(
        cls,
        *,
        stacking: str = "AB-AB",
        valley: int = 1,
        Delta: float = 0.0,
        phi_deg: float = 0.0,
        epsilon: float = 0.0,
    ) -> "TDBGParameters":
        return cls(
            stacking=stacking,
            valley=valley,
            Delta=Delta,
            phi_deg=phi_deg,
            epsilon=epsilon,
            gamma3=0.0,
            gamma4=0.0,
            delta_prime=0.0,
            model_name="minimal",
        )

    @classmethod
    def full(
        cls,
        *,
        stacking: str = "AB-AB",
        valley: int = 1,
        Delta: float = 0.0,
        phi_deg: float = 0.0,
        epsilon: float = 0.0,
    ) -> "TDBGParameters":
        return cls(
            stacking=stacking,
            valley=valley,
            Delta=Delta,
            phi_deg=phi_deg,
            epsilon=epsilon,
            model_name="full",
        )

    @classmethod
    def no_corrugation(
        cls,
        *,
        stacking: str = "AB-AB",
        valley: int = 1,
        Delta: float = 0.0,
        phi_deg: float = 0.0,
        epsilon: float = 0.0,
    ) -> "TDBGParameters":
        return cls(
            stacking=stacking,
            valley=valley,
            Delta=Delta,
            phi_deg=phi_deg,
            epsilon=epsilon,
            u=DEFAULT_U_PRIME_EV,
            u_prime=DEFAULT_U_PRIME_EV,
            model_name="no_corrugation",
        )
