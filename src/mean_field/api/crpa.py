from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
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


@dataclass(frozen=True)
class CRPAAdapterInfo:
    name: str
    system_name: str
    import_path: str
    description: str
    requires_explicit_inputs: tuple[str, ...] = ()


_CRPA_ADAPTERS: tuple[CRPAAdapterInfo, ...] = (
    CRPAAdapterInfo(
        name="tbg_workflow",
        system_name="tbg",
        import_path="mean_field.crpa.workflow:compute_crpa",
        description="TBG zero-field cRPA workflow adapter with explicit BM/TBG parameters.",
        requires_explicit_inputs=("theta_deg or TBGZeroFieldBMModel", "TBGParameters", "lk/lg/q_lg choices"),
    ),
)


def list_crpa_adapters(*, system_name: str | None = None) -> tuple[CRPAAdapterInfo, ...]:
    adapters = _CRPA_ADAPTERS
    if system_name is not None:
        key = str(system_name).lower().replace("-", "_")
        adapters = tuple(item for item in adapters if item.system_name.lower().replace("-", "_") == key)
    return adapters


def get_crpa_adapter_info(name: str) -> CRPAAdapterInfo:
    for item in _CRPA_ADAPTERS:
        if item.name == name:
            return item
    raise KeyError(f"Unknown CRPA adapter {name!r}; available: {[item.name for item in _CRPA_ADAPTERS]}")


def resolve_crpa_adapter(name: str) -> Callable[..., Any]:
    info = get_crpa_adapter_info(name)
    module_name, attr = info.import_path.split(":", 1)
    return getattr(import_module(module_name), attr)


def _compute_tbg_crpa(model_or_solution: object, config: CRPAConfig, **kwargs: Any) -> object:
    workflow_compute = resolve_crpa_adapter("tbg_workflow")
    params = kwargs.pop("params", getattr(model_or_solution, "params", None))
    theta_deg = kwargs.pop("theta_deg", getattr(model_or_solution, "theta_deg", None))
    if params is None or theta_deg is None:
        raise ValueError("TBG cRPA adapter requires explicit params and theta_deg or a TBGZeroFieldBMModel")
    q_mesh = config.q_mesh
    if isinstance(q_mesh, tuple):
        if len(q_mesh) != 2 or int(q_mesh[0]) != int(q_mesh[1]):
            raise ValueError("TBG cRPA adapter currently requires a square q_mesh or integer q_mesh")
        q_lg = int(q_mesh[0])
    else:
        q_lg = int(q_mesh)
    from mean_field.crpa.coulomb import CRPACoulombParams

    coulomb_params = kwargs.pop(
        "coulomb_params",
        CRPACoulombParams(epsilon_bn=float(config.epsilon_bn), ds_angstrom=float(config.ds_angstrom)),
    )
    return workflow_compute(
        params,
        theta_deg=float(theta_deg),
        q_lg=q_lg,
        eta_mev=float(config.eta_mev),
        occupation_mode=str(config.occupation_mode),
        form_factor_mode=str(config.form_factor_mode),
        coulomb_params=coulomb_params,
        **kwargs,
    )


def compute_crpa(model_or_solution: object, config: CRPAConfig, *, adapter: str | None = None, **kwargs: Any) -> object:
    """Public cRPA façade with explicit adapter registry.

    Objects may still provide ``compute_crpa(config)``.  Registry dispatch is
    explicit to avoid silently inferring production cRPA parameters.
    """

    if adapter is not None:
        if adapter == "tbg_workflow":
            return _compute_tbg_crpa(model_or_solution, config, **kwargs)
        resolved = resolve_crpa_adapter(adapter)
        return resolved(model_or_solution, config, **kwargs)
    if hasattr(model_or_solution, "compute_crpa"):
        return model_or_solution.compute_crpa(config, **kwargs)  # type: ignore[attr-defined]
    raise NotImplementedError(
        "Unified compute_crpa requires adapter='tbg_workflow' with explicit TBG inputs, "
        "or an object exposing compute_crpa(config)"
    )


__all__ = [
    "CRPAAdapterInfo",
    "CRPAConfig",
    "compute_crpa",
    "get_crpa_adapter_info",
    "list_crpa_adapters",
    "resolve_crpa_adapter",
]
