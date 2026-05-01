from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_W_AB_EV = 0.110
DEFAULT_KAPPA = 0.8
DEFAULT_GRAPHENE_HOPPING_EV = 3.1
DEFAULT_VF_EV_NM = math.sqrt(3.0) * DEFAULT_GRAPHENE_HOPPING_EV * GRAPHENE_LATTICE_CONSTANT_NM / 2.0


def _normalize_alpha_couplings(
    n_layers: int,
    alpha: float,
    alpha_couplings: tuple[float, ...] | list[float] | None,
) -> tuple[float, ...]:
    if n_layers <= 1:
        return tuple()
    if alpha_couplings is None:
        return tuple(float(alpha) for _ in range(n_layers - 1))
    normalized = tuple(float(value) for value in alpha_couplings)
    if len(normalized) != n_layers - 1:
        raise ValueError(
            f"Expected {n_layers - 1} interface couplings for n_layers={n_layers}, got {len(normalized)}"
        )
    if min(normalized) < 0.0:
        raise ValueError(f"Interface couplings must be non-negative, got {normalized}")
    return normalized


@dataclass(frozen=True)
class ATMGParameters:
    n_layers: int
    theta_deg: float
    w_ab: float = DEFAULT_W_AB_EV
    kappa: float = DEFAULT_KAPPA
    vf: float = DEFAULT_VF_EV_NM
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    alpha_couplings: tuple[float, ...] | None = None
    model_name: str = "uniform"

    theta_rad: float = field(init=False)
    graphene_k_mag: float = field(init=False)
    moire_energy_scale: float = field(init=False)
    alpha: float = field(init=False)
    resolved_alpha_couplings: tuple[float, ...] = field(init=False)
    resolved_w_ab_couplings: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_layers = int(self.n_layers)
        theta_deg = float(self.theta_deg)
        theta_rad = theta_deg * math.pi / 180.0
        if n_layers <= 0:
            raise ValueError(f"Expected a positive layer count, got {self.n_layers}")
        if theta_rad <= 0.0:
            raise ValueError(f"Expected a positive twist angle, got {self.theta_deg}")
        if self.w_ab < 0.0:
            raise ValueError(f"Expected a non-negative w_ab, got {self.w_ab}")
        if self.vf <= 0.0:
            raise ValueError(f"Expected a positive vf, got {self.vf}")

        graphene_k_mag = 4.0 * math.pi / (3.0 * float(self.graphene_lattice_constant_nm))
        moire_energy_scale = float(self.vf) * graphene_k_mag * theta_rad
        alpha = 0.0 if moire_energy_scale == 0.0 else float(self.w_ab) / moire_energy_scale
        resolved_alpha_couplings = _normalize_alpha_couplings(
            n_layers,
            alpha,
            self.alpha_couplings,
        )
        resolved_w_ab_couplings = tuple(value * moire_energy_scale for value in resolved_alpha_couplings)

        object.__setattr__(self, "n_layers", n_layers)
        object.__setattr__(self, "theta_deg", theta_deg)
        object.__setattr__(self, "theta_rad", theta_rad)
        object.__setattr__(self, "graphene_k_mag", graphene_k_mag)
        object.__setattr__(self, "moire_energy_scale", moire_energy_scale)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "resolved_alpha_couplings", resolved_alpha_couplings)
        object.__setattr__(self, "resolved_w_ab_couplings", resolved_w_ab_couplings)

    @property
    def n_odd(self) -> int:
        return (int(self.n_layers) + 1) // 2

    @property
    def n_even(self) -> int:
        return int(self.n_layers) // 2

    def layer_twist_rad(self, layer_index: int) -> float:
        index = int(layer_index)
        if index < 0 or index >= self.n_layers:
            raise IndexError(f"Layer index {layer_index} is out of range for n_layers={self.n_layers}")
        return -self.theta_rad / 2.0 if index % 2 == 0 else self.theta_rad / 2.0

    @property
    def is_uniform(self) -> bool:
        if len(self.resolved_alpha_couplings) <= 1:
            return True
        return max(self.resolved_alpha_couplings) - min(self.resolved_alpha_couplings) < 1.0e-14

    @classmethod
    def chiral(
        cls,
        n_layers: int,
        theta_deg: float,
        *,
        w_ab: float = DEFAULT_W_AB_EV,
        vf: float = DEFAULT_VF_EV_NM,
        graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
        alpha_couplings: tuple[float, ...] | list[float] | None = None,
    ) -> "ATMGParameters":
        return cls(
            n_layers=n_layers,
            theta_deg=theta_deg,
            w_ab=w_ab,
            kappa=0.0,
            vf=vf,
            graphene_lattice_constant_nm=graphene_lattice_constant_nm,
            alpha_couplings=tuple(alpha_couplings) if alpha_couplings is not None else None,
            model_name="chiral",
        )

    @classmethod
    def realistic(
        cls,
        n_layers: int,
        theta_deg: float,
        *,
        w_ab: float = DEFAULT_W_AB_EV,
        kappa: float = DEFAULT_KAPPA,
        vf: float = DEFAULT_VF_EV_NM,
        graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
        alpha_couplings: tuple[float, ...] | list[float] | None = None,
    ) -> "ATMGParameters":
        return cls(
            n_layers=n_layers,
            theta_deg=theta_deg,
            w_ab=w_ab,
            kappa=kappa,
            vf=vf,
            graphene_lattice_constant_nm=graphene_lattice_constant_nm,
            alpha_couplings=tuple(alpha_couplings) if alpha_couplings is not None else None,
            model_name="realistic",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_layers": int(self.n_layers),
            "theta_deg": float(self.theta_deg),
            "theta_rad": float(self.theta_rad),
            "w_ab": float(self.w_ab),
            "kappa": float(self.kappa),
            "vf": float(self.vf),
            "graphene_lattice_constant_nm": float(self.graphene_lattice_constant_nm),
            "graphene_k_mag": float(self.graphene_k_mag),
            "moire_energy_scale": float(self.moire_energy_scale),
            "alpha": float(self.alpha),
            "alpha_couplings": list(float(value) for value in self.resolved_alpha_couplings),
            "w_ab_couplings": list(float(value) for value in self.resolved_w_ab_couplings),
            "model_name": str(self.model_name),
        }


__all__ = [
    "ATMGParameters",
    "DEFAULT_GRAPHENE_HOPPING_EV",
    "DEFAULT_KAPPA",
    "DEFAULT_VF_EV_NM",
    "DEFAULT_W_AB_EV",
    "GRAPHENE_LATTICE_CONSTANT_NM",
]
