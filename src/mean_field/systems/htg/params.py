from __future__ import annotations

from dataclasses import dataclass, field
import math


GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_FERMI_VELOCITY_M_PER_S = 1.03e6
DEFAULT_TUNNELING_EV = 0.105
DEFAULT_KAPPA = 0.7
HBAR_EV_S = 6.582119569e-16
VALID_VALLEYS = (-1, 1)


def velocity_m_per_s_to_ev_nm(velocity_m_per_s: float) -> float:
    return float(HBAR_EV_S * velocity_m_per_s * 1.0e9)


@dataclass(frozen=True)
class HTGParams:
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    fermi_velocity_m_per_s: float = DEFAULT_FERMI_VELOCITY_M_PER_S
    w_ev: float = DEFAULT_TUNNELING_EV
    kappa: float = DEFAULT_KAPPA
    zeta_rad: float | None = None
    model_name: str = "default"

    vf_ev_nm: float = field(init=False)

    def __post_init__(self) -> None:
        if self.graphene_lattice_constant_nm <= 0.0:
            raise ValueError("graphene_lattice_constant_nm must be positive")
        if self.fermi_velocity_m_per_s <= 0.0:
            raise ValueError("fermi_velocity_m_per_s must be positive")
        if self.w_ev <= 0.0:
            raise ValueError("w_ev must be positive")
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        object.__setattr__(
            self,
            "vf_ev_nm",
            velocity_m_per_s_to_ev_nm(self.fermi_velocity_m_per_s),
        )

    @classmethod
    def default(cls, *, kappa: float = DEFAULT_KAPPA) -> "HTGParams":
        return cls(kappa=kappa, model_name="default")

    @classmethod
    def chiral(cls, *, zeta_rad: float = 0.0) -> "HTGParams":
        return cls(kappa=0.0, zeta_rad=float(zeta_rad), model_name="chiral")

    def vk_theta_ev(self, k_theta_nm_inv: float) -> float:
        return float(self.vf_ev_nm * k_theta_nm_inv)

    def alpha(self, k_theta_nm_inv: float) -> float:
        return float(self.w_ev / self.vk_theta_ev(k_theta_nm_inv))


@dataclass(frozen=True)
class InteractionParams:
    """HTG defaults for the reusable projected Hartree-Fock solver."""

    epsilon_r: float = 8.0
    d_sc_nm: float = 25.0
    U_ev: float = 0.0
    subtraction: str = "average"
    n_k: int = 12
    g_shells: int = 1
    finite_zero_limit: bool = True
    zero_cutoff_nm_inv: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")
        if self.d_sc_nm < 0.0:
            raise ValueError("d_sc_nm must be non-negative")
        if self.subtraction != "average":
            raise ValueError("Only the average subtraction scheme is currently supported")
        if self.n_k <= 0:
            raise ValueError("n_k must be positive")
        if self.g_shells < 0:
            raise ValueError("g_shells must be non-negative")
        if self.zero_cutoff_nm_inv < 0.0:
            raise ValueError("zero_cutoff_nm_inv must be non-negative")


def theta_deg_from_alpha(
    alpha: float,
    *,
    params: HTGParams | None = None,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> float:
    if alpha <= 0.0:
        raise ValueError(f"Expected positive alpha, got {alpha}")
    resolved = params if params is not None else HTGParams.chiral()
    k_graphene = 4.0 * math.pi / (3.0 * float(graphene_lattice_constant_nm))
    argument = resolved.w_ev / (2.0 * k_graphene * resolved.vf_ev_nm * float(alpha))
    if not 0.0 < argument < 1.0:
        raise ValueError(
            f"alpha={alpha} gives invalid asin argument {argument:.6g}; "
            "check the tunneling and velocity parameters."
        )
    return float(2.0 * math.asin(argument) * 180.0 / math.pi)
