from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CRPAConfig:
    q_mesh: int | tuple[int, int]
    epsilon_bn: float = 4.0
    ds_angstrom: float = 400.0
    eta_mev: float = 1.0
    occupation_mode: str = "cnp_index"
    form_factor_mode: str = "k_periodic_zero_fill"
    metadata: dict[str, object] = field(default_factory=dict)


def compute_crpa(model_or_solution: object, config: CRPAConfig, **kwargs: Any) -> object:
    """Public cRPA façade.

    The stable API is frozen here; production logic remains in `mean_field.crpa`
    until system adapters expose a uniform method hook.
    """

    if hasattr(model_or_solution, "compute_crpa"):
        return model_or_solution.compute_crpa(config, **kwargs)  # type: ignore[attr-defined]
    raise NotImplementedError(
        "Unified compute_crpa requires a system/BM-solution adapter exposing compute_crpa(config)"
    )


__all__ = ["CRPAConfig", "compute_crpa"]
