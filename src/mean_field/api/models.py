from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .artifacts import ModelRecord


@dataclass(frozen=True)
class BandEigenResult:
    energies: np.ndarray
    eigenvectors: np.ndarray | None = None
    metadata: dict[str, object] | None = None


class ContinuumModel(Protocol):
    @property
    def matrix_dim(self) -> int: ...

    def build_hamiltonian(self, k_tilde: complex, **kwargs: Any) -> np.ndarray: ...

    def diagonalize(self, k_tilde: complex, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]: ...

    def lattice_summary(self) -> dict[str, object]: ...

    def component_groups(self) -> tuple[object, ...]: ...


ModelBuilder = Callable[[str, dict[str, Any]], object]


@dataclass(frozen=True)
class ModelAdapterInfo:
    name: str
    aliases: tuple[str, ...]
    description: str
    builder: ModelBuilder

    @property
    def public_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def _normalize_model_key(system_name: str) -> str:
    return str(system_name).lower().replace("-", "_")


def _pop_alias(kwargs: dict[str, Any], canonical: str, *aliases: str, default: Any = None) -> Any:
    if canonical in kwargs:
        return kwargs.pop(canonical)
    for alias in aliases:
        if alias in kwargs:
            return kwargs.pop(alias)
    return default


def _reject_unknown_options(system_name: str, options: dict[str, Any], *, allowed: set[str]) -> None:
    unknown = sorted(key for key in options if key not in allowed)
    if unknown:
        raise TypeError(f"Unknown options for {system_name!r}: {unknown}")


def _build_htg_model(system_name: str, options: dict[str, Any]) -> object:
    from mean_field.systems.htg import HTGModel, HTGParams

    theta_deg = float(_pop_alias(options, "theta_deg", default=1.8))
    n_shells = int(_pop_alias(options, "n_shells", "shell_count", default=5))
    if "params" not in options:
        params_kwargs: dict[str, Any] = {}
        if "w_ev" in options:
            params_kwargs["w_ev"] = options.pop("w_ev")
        if "kappa" in options:
            params_kwargs["kappa"] = options.pop("kappa")
        if "fermi_velocity_m_per_s" in options:
            params_kwargs["fermi_velocity_m_per_s"] = options.pop("fermi_velocity_m_per_s")
        params = HTGParams.default() if not params_kwargs else HTGParams(**params_kwargs)
        options["params"] = params
    _reject_unknown_options(system_name, options, allowed={"params"})
    return HTGModel.from_config(theta_deg, n_shells=n_shells, params=options.get("params"))


def _build_rlg_hbn_model(system_name: str, options: dict[str, Any]) -> object:
    from mean_field.systems.RnG_hBN import RLGhBNModel

    layer_count = int(_pop_alias(options, "layer_count", "layers", default=5))
    xi = int(_pop_alias(options, "xi", default=1))
    theta_deg = float(_pop_alias(options, "theta_deg", default=0.77))
    displacement = float(_pop_alias(options, "displacement_field_mev", "displacement_mev", default=0.0))
    shell_count = int(_pop_alias(options, "shell_count", "n_shells", default=4))
    params = options.pop("params", None)
    _reject_unknown_options(system_name, options, allowed=set())
    return RLGhBNModel.from_config(
        layer_count=layer_count,
        xi=xi,
        theta_deg=theta_deg,
        displacement_field_mev=displacement,
        shell_count=shell_count,
        params=params,
    )


def _build_tbg_model(system_name: str, options: dict[str, Any]) -> object:
    from mean_field.systems.tbg import TBGZeroFieldBMModel

    variant = str(_pop_alias(options, "variant", "model", default="zero_field_bm")).lower().replace("-", "_")
    if variant not in {"zero_field_bm", "bm", "b0_bm"}:
        raise NotImplementedError(
            "The public TBG model adapter currently supports only variant='zero_field_bm' BM bands; "
            "TBG HF requires an explicit system workflow."
        )
    theta_deg = float(_pop_alias(options, "theta_deg", default=1.05))
    lg = int(_pop_alias(options, "lg", default=9))
    params = options.pop("params", None)
    sigma_rotation = bool(_pop_alias(options, "sigma_rotation", default=True))
    periodic_g_grid = bool(_pop_alias(options, "periodic_g_grid", default=True))
    _reject_unknown_options(system_name, options, allowed=set())
    return TBGZeroFieldBMModel.from_config(
        theta_deg,
        lg=lg,
        params=params,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )


def _build_tdbg_model(system_name: str, options: dict[str, Any]) -> object:
    from mean_field.systems.tdbg import TDBGModel

    theta_deg = float(_pop_alias(options, "theta_deg", default=1.33))
    cut = float(_pop_alias(options, "cut", default=4.0))
    params = options.pop("params", None)
    _reject_unknown_options(system_name, options, allowed=set())
    return TDBGModel.from_config(theta_deg, cut=cut, params=params)


def _build_tmbg_model(system_name: str, options: dict[str, Any]) -> object:
    from mean_field.systems.tmbg import TMBGModel

    theta_deg = float(_pop_alias(options, "theta_deg", default=1.25))
    n_shells = int(_pop_alias(options, "n_shells", "shell_count", default=5))
    params = options.pop("params", None)
    _reject_unknown_options(system_name, options, allowed=set())
    return TMBGModel.from_config(theta_deg, n_shells=n_shells, params=params)


_MODEL_ADAPTERS: tuple[ModelAdapterInfo, ...] = (
    ModelAdapterInfo(
        name="htg",
        aliases=("helical_trilayer_graphene",),
        description="Helical trilayer graphene continuum model adapter.",
        builder=_build_htg_model,
    ),
    ModelAdapterInfo(
        name="rlg_hbn",
        aliases=("rng_hbn", "rnghbn"),
        description="Rhombohedral graphene / hBN continuum model adapter.",
        builder=_build_rlg_hbn_model,
    ),
    ModelAdapterInfo(
        name="tbg",
        aliases=("twisted_bilayer_graphene",),
        description="TBG zero-field BM band model adapter.",
        builder=_build_tbg_model,
    ),
    ModelAdapterInfo(name="tdbg", aliases=(), description="Twisted double bilayer graphene model adapter.", builder=_build_tdbg_model),
    ModelAdapterInfo(name="tmbg", aliases=(), description="Twisted monolayer-bilayer graphene model adapter.", builder=_build_tmbg_model),
)

_MODEL_ADAPTER_BY_KEY: dict[str, ModelAdapterInfo] = {
    _normalize_model_key(public_name): adapter
    for adapter in _MODEL_ADAPTERS
    for public_name in adapter.public_names
}


def list_model_adapters() -> tuple[ModelAdapterInfo, ...]:
    return _MODEL_ADAPTERS


def get_model_adapter_info(system_name: str) -> ModelAdapterInfo:
    key = _normalize_model_key(system_name)
    try:
        return _MODEL_ADAPTER_BY_KEY[key]
    except KeyError as exc:
        supported = ", ".join(adapter.name for adapter in _MODEL_ADAPTERS)
        raise ValueError(f"Unsupported system_name={system_name!r}; supported: {supported}") from exc


def resolve_model_adapter(system_name: str) -> ModelBuilder:
    return get_model_adapter_info(system_name).builder


def make_model(system_name: str, **kwargs: Any) -> object:
    """Build a system model through the stable public façade registry."""

    adapter = get_model_adapter_info(system_name)
    return adapter.builder(system_name, dict(kwargs))


def model_record(model: object, *, system_name: str | None = None) -> ModelRecord:
    name = system_name or model.__class__.__name__.replace("Model", "").lower()
    lattice = model.lattice_summary() if hasattr(model, "lattice_summary") else {}
    params: dict[str, object] = {}
    if hasattr(model, "params") and hasattr(model.params, "to_summary_dict"):
        params = dict(model.params.to_summary_dict())
    return ModelRecord(system_name=name, params=params, lattice=dict(lattice))


def component_groups(model: object) -> tuple[object, ...]:
    """Return system-declared component groups, or an empty tuple if absent."""

    if hasattr(model, "component_groups"):
        return tuple(model.component_groups())  # type: ignore[attr-defined]
    return ()


def component_group_records(model: object) -> tuple[dict[str, object], ...]:
    """Return JSON-serializable records for system-declared component groups."""

    records: list[dict[str, object]] = []
    for group in component_groups(model):
        name = getattr(group, "name", None)
        indices = getattr(group, "indices", None)
        if name is None or indices is None:
            records.append({"repr": repr(group)})
            continue
        tolist = getattr(indices, "tolist", None)
        record: dict[str, object] = {"name": str(name), "indices": list(tolist() if callable(tolist) else indices)}
        for optional_key in ("index_space", "description"):
            value = getattr(group, optional_key, None)
            if value is not None:
                record[optional_key] = str(value)
        records.append(record)
    return tuple(records)


__all__ = [
    "BandEigenResult",
    "ContinuumModel",
    "ModelAdapterInfo",
    "component_group_records",
    "component_groups",
    "get_model_adapter_info",
    "list_model_adapters",
    "make_model",
    "model_record",
    "resolve_model_adapter",
]
