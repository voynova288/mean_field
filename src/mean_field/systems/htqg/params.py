from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_FERMI_VELOCITY_M_PER_S = 8.4e5
DEFAULT_TUNNELING_EV = 0.110
DEFAULT_KAPPA = 0.6
DEFAULT_THETA_DEG = 2.25
# Momentum-dependent tunneling length in nm.  Fujimoto et al. cite the
# Kwan-Tan-Devakul MDT derivation; that reference uses -2.3 Angstrom
# = -0.23 nm.  The HTQG arXiv v1 sentence saying "-0.23 Angstrom" is
# inconsistent with the cited derivation and with the Fig. 8/1(e) gap scale.
PAPER_MDT_NM = -0.23
HBAR_EV_S = 6.582119569e-16
VALID_VALLEYS = (-1, 1)

MDTMomentumConvention = Literal["source", "target", "midpoint"]


def velocity_m_per_s_to_ev_nm(velocity_m_per_s: float) -> float:
    """Convert a Dirac velocity in m/s to hbar*v in eV nm."""

    return float(HBAR_EV_S * float(velocity_m_per_s) * 1.0e9)


@dataclass(frozen=True)
class HTQGParams:
    """Physical parameters for the Fujimoto et al. 2025 HTQG continuum model.

    Units follow the HTQG work document: energies are eV, momenta are nm^-1,
    and ``vf_ev_nm`` is hbar*v in eV nm.  The default parameter set is the
    first-pass convention-locked model: realistic ``kappa=0.6`` but with
    momentum-dependent tunneling and layer Dirac rotations disabled.  Use
    :meth:`realistic` when explicitly studying the weak particle-hole breaking
    terms from the paper.

    ``lambda_mdt_nm`` is a length multiplying the physical small momentum in
    the source-layer plane wave.  Its default is the Kwan-Tan-Devakul value
    -2.3 Angstrom = -0.23 nm used by the HTQG figures; this is intentionally
    not the typo-scale -0.023 nm.
    """

    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    fermi_velocity_m_per_s: float = DEFAULT_FERMI_VELOCITY_M_PER_S
    w_ev: float = DEFAULT_TUNNELING_EV
    kappa: float = DEFAULT_KAPPA
    lambda_mdt_nm: float = 0.0
    include_dirac_rotation: bool = False
    mdt_momentum: MDTMomentumConvention = "source"
    model_name: str = "fujimoto2025_convention_locked"
    vf_ev_nm: float = field(init=False)

    def __post_init__(self) -> None:
        if self.graphene_lattice_constant_nm <= 0.0:
            raise ValueError("graphene_lattice_constant_nm must be positive")
        if self.fermi_velocity_m_per_s <= 0.0:
            raise ValueError("fermi_velocity_m_per_s must be positive")
        if self.w_ev < 0.0:
            raise ValueError("w_ev must be non-negative")
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        if self.mdt_momentum not in {"source", "target", "midpoint"}:
            raise ValueError("mdt_momentum must be 'source', 'target', or 'midpoint'")
        object.__setattr__(self, "vf_ev_nm", velocity_m_per_s_to_ev_nm(self.fermi_velocity_m_per_s))

    @classmethod
    def default(
        cls,
        *,
        kappa: float = DEFAULT_KAPPA,
        lambda_mdt_nm: float = 0.0,
        include_dirac_rotation: bool = False,
    ) -> "HTQGParams":
        return cls(
            kappa=float(kappa),
            lambda_mdt_nm=float(lambda_mdt_nm),
            include_dirac_rotation=bool(include_dirac_rotation),
            model_name="fujimoto2025_convention_locked",
        )

    @classmethod
    def chiral(cls, *, w_ev: float = DEFAULT_TUNNELING_EV) -> "HTQGParams":
        return cls(
            w_ev=float(w_ev),
            kappa=0.0,
            lambda_mdt_nm=0.0,
            include_dirac_rotation=False,
            model_name="fujimoto2025_chiral",
        )

    @classmethod
    def realistic(
        cls,
        *,
        kappa: float = DEFAULT_KAPPA,
        lambda_mdt_nm: float = PAPER_MDT_NM,
        include_dirac_rotation: bool = True,
        mdt_momentum: MDTMomentumConvention = "source",
    ) -> "HTQGParams":
        return cls(
            kappa=float(kappa),
            lambda_mdt_nm=float(lambda_mdt_nm),
            include_dirac_rotation=bool(include_dirac_rotation),
            mdt_momentum=mdt_momentum,
            model_name="fujimoto2025_realistic_ph_breaking",
        )

    def vk_theta_ev(self, k_theta_nm_inv: float) -> float:
        return float(self.vf_ev_nm * float(k_theta_nm_inv))

    def alpha(self, k_theta_nm_inv: float) -> float:
        denom = self.vk_theta_ev(k_theta_nm_inv)
        if denom <= 0.0:
            raise ValueError("k_theta_nm_inv must be positive")
        return float(self.w_ev / denom)

    def to_dict(self) -> dict[str, object]:
        return {
            "graphene_lattice_constant_nm": float(self.graphene_lattice_constant_nm),
            "fermi_velocity_m_per_s": float(self.fermi_velocity_m_per_s),
            "vf_ev_nm": float(self.vf_ev_nm),
            "w_ev": float(self.w_ev),
            "kappa": float(self.kappa),
            "lambda_mdt_nm": float(self.lambda_mdt_nm),
            "include_dirac_rotation": bool(self.include_dirac_rotation),
            "mdt_momentum": str(self.mdt_momentum),
            "model_name": str(self.model_name),
        }


def theta_deg_from_alpha(
    alpha: float,
    *,
    params: HTQGParams | None = None,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> float:
    """Return the twist angle corresponding to ``alpha = w/(hbar v k_theta)``."""

    if alpha <= 0.0:
        raise ValueError(f"Expected positive alpha, got {alpha}")
    resolved = params if params is not None else HTQGParams.chiral()
    k_graphene = 4.0 * math.pi / (3.0 * float(graphene_lattice_constant_nm))
    argument = resolved.w_ev / (2.0 * k_graphene * resolved.vf_ev_nm * float(alpha))
    if not 0.0 < argument < 1.0:
        raise ValueError(
            f"alpha={alpha} gives invalid asin argument {argument:.6g}; "
            "check tunneling, velocity, and lattice constant."
        )
    return float(2.0 * math.asin(argument) * 180.0 / math.pi)


__all__ = [
    "DEFAULT_FERMI_VELOCITY_M_PER_S",
    "DEFAULT_KAPPA",
    "DEFAULT_THETA_DEG",
    "DEFAULT_TUNNELING_EV",
    "GRAPHENE_LATTICE_CONSTANT_NM",
    "HBAR_EV_S",
    "HTQGParams",
    "PAPER_MDT_NM",
    "VALID_VALLEYS",
    "velocity_m_per_s_to_ev_nm",
    "theta_deg_from_alpha",
]
