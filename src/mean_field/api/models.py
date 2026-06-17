from __future__ import annotations

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


def _pop_alias(kwargs: dict[str, Any], canonical: str, *aliases: str, default: Any = None) -> Any:
    if canonical in kwargs:
        return kwargs.pop(canonical)
    for alias in aliases:
        if alias in kwargs:
            return kwargs.pop(alias)
    return default


def make_model(system_name: str, **kwargs: Any) -> object:
    """Build a system model through the stable public façade.

    This is intentionally a thin compatibility layer over existing system model
    constructors.  It freezes public spelling and aliases without moving physics
    logic out of system modules yet.
    """

    key = str(system_name).lower().replace("-", "_")
    options = dict(kwargs)
    if key in {"htg", "helical_trilayer_graphene"}:
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
    if key in {"htqg", "helical_twisted_quadrilayer_graphene"}:
        from mean_field.systems.htqg import HTQGModel, HTQGParams

        theta_deg = float(_pop_alias(options, "theta_deg", default=2.25))
        n_shells = int(_pop_alias(options, "n_shells", "shell_count", default=4))
        domain = _pop_alias(options, "domain", default="alpha_beta_alpha")
        valley = int(_pop_alias(options, "valley", default=1))
        if "params" not in options:
            params_kwargs: dict[str, Any] = {}
            if "kappa" in options:
                params_kwargs["kappa"] = options.pop("kappa")
            if "lambda_mdt_nm" in options:
                params_kwargs["lambda_mdt_nm"] = options.pop("lambda_mdt_nm")
            if "include_dirac_rotation" in options:
                params_kwargs["include_dirac_rotation"] = options.pop("include_dirac_rotation")
            options["params"] = HTQGParams.default(**params_kwargs)
        _reject_unknown_options(system_name, options, allowed={"params"})
        return HTQGModel.default(
            theta_deg=theta_deg,
            n_shells=n_shells,
            domain=domain,
            params=options.get("params"),
            valley=valley,
        )
    if key in {"rlg_hbn", "rng_hbn", "rnghbn", "rlg-hbn"}:
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
    if key in {"tbg", "twisted_bilayer_graphene"}:
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
    if key == "tdbg":
        from mean_field.systems.tdbg import TDBGModel

        theta_deg = float(_pop_alias(options, "theta_deg", default=1.33))
        cut = float(_pop_alias(options, "cut", default=4.0))
        params = options.pop("params", None)
        _reject_unknown_options(system_name, options, allowed=set())
        return TDBGModel.from_config(theta_deg, cut=cut, params=params)
    if key == "tmbg":
        from mean_field.systems.tmbg import TMBGModel

        theta_deg = float(_pop_alias(options, "theta_deg", default=1.25))
        n_shells = int(_pop_alias(options, "n_shells", "shell_count", default=5))
        params = options.pop("params", None)
        _reject_unknown_options(system_name, options, allowed=set())
        return TMBGModel.from_config(theta_deg, n_shells=n_shells, params=params)
    if key == "atmg":
        from mean_field.systems.atmg import ATMGModel

        n_layers = int(_pop_alias(options, "n_layers", "layers", default=3))
        theta_deg = float(_pop_alias(options, "theta_deg", default=1.5))
        n_shells = int(_pop_alias(options, "n_shells", "shell_count", default=5))
        params = options.pop("params", None)
        _reject_unknown_options(system_name, options, allowed=set())
        return ATMGModel.from_config(n_layers, theta_deg, n_shells=n_shells, params=params)
    raise ValueError(f"Unsupported system_name={system_name!r}; supported: htg, htqg, rlg_hbn, tbg, tdbg, tmbg, atmg")


def _reject_unknown_options(system_name: str, options: dict[str, Any], *, allowed: set[str]) -> None:
    unknown = sorted(key for key in options if key not in allowed)
    if unknown:
        raise TypeError(f"Unknown options for {system_name!r}: {unknown}")


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
    "component_group_records",
    "component_groups",
    "make_model",
    "model_record",
]
