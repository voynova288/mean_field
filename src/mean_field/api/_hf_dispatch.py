from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_types import HFConfig
from ._hf_result import HFResult
from ._hf_registry import _call_registered_hf_adapter

def _run_tdbg_hf_if_explicit(model: object, config: HFConfig, kwargs: dict[str, Any]) -> HFResult | None:
    from mean_field.systems.tdbg import (
        TDBGModel,
        TDBGProjectedHFConfig,
        TDBGProjectedHFResult,
        build_tdbg_projected_hf_data,
        run_tdbg_projected_hf,
        tdbg_projected_hf_result_to_hf_run_result,
    )

    if not isinstance(model, TDBGModel):
        return None

    tdbg_config = kwargs.pop("tdbg_config", kwargs.pop("projected_config", None))
    if tdbg_config is None:
        raise NotImplementedError(
            "Unified run_hf has a TDBG adapter only for explicit "
            "tdbg_config=TDBGProjectedHFConfig(...) plus init_mode=...; "
            "generic HFConfig -> TDBGProjectedHFConfig mapping is not implemented"
        )
    if not isinstance(tdbg_config, TDBGProjectedHFConfig):
        raise TypeError(f"tdbg_config must be TDBGProjectedHFConfig, got {type(tdbg_config).__name__}")
    init_mode = kwargs.pop("init_mode", None)
    if init_mode is None:
        raise TypeError("TDBG public run_hf adapter requires explicit init_mode=...")
    seed = int(kwargs.pop("seed", 1))
    if kwargs:
        raise TypeError(f"Unsupported TDBG run_hf kwargs: {sorted(kwargs)}")

    _validate_tdbg_public_hf_config(model, config, tdbg_config)
    data = build_tdbg_projected_hf_data(tdbg_config)
    raw = run_tdbg_projected_hf(data, init_mode=str(init_mode), seed=seed)

    from .models import model_record

    record = model_record(model, system_name="tdbg")
    summary = raw.to_summary_dict() if hasattr(raw, "to_summary_dict") else {}
    canonical_run_result = (
        tdbg_projected_hf_result_to_hf_run_result(raw)
        if isinstance(raw, TDBGProjectedHFResult)
        else None
    )
    return HFResult(
        model=record,
        config=config,
        state=raw,
        observables=dict(summary),
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="eV",
                density_convention="projector",
                density_axis_order="abk",
                gauge="tdbg_projected_hf_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "tdbg.projected_hf.explicit_config",
                "system_name": "tdbg",
                "adapter": "mean_field.api.run_hf",
            },
        ),
        canonical_run_result=canonical_run_result,
    )


def _validate_tdbg_public_hf_config(model: object, config: HFConfig, tdbg_config: Any) -> None:
    if int(config.mesh[0]) != int(config.mesh[1]) or int(config.mesh[0]) != int(tdbg_config.mesh_size):
        raise ValueError(
            "TDBG public run_hf requires HFConfig.mesh=(mesh_size, mesh_size) matching "
            f"tdbg_config.mesh_size={tdbg_config.mesh_size}, got {config.mesh}"
        )
    if float(config.filling) != float(int(tdbg_config.filling)):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.filling={tdbg_config.filling}, got {config.filling}"
        )
    if config.max_iter != int(tdbg_config.max_iter):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.max_iter={tdbg_config.max_iter}, got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(tdbg_config.precision)):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.precision={tdbg_config.precision}, got {config.precision}"
        )
    if config.density_convention != "projector":
        raise ValueError(
            "TDBG projected HF stores an absolute projector density; set "
            "HFConfig.density_convention='projector' for this explicit adapter"
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "TDBG public run_hf takes the projected window from tdbg_config.window; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )
    model_theta = getattr(model, "theta_deg", None)
    model_cut = getattr(model, "cut", None)
    if model_theta is not None and not np.isclose(float(model_theta), float(tdbg_config.theta_deg)):
        raise ValueError(
            f"TDBG model theta_deg={model_theta} does not match tdbg_config.theta_deg={tdbg_config.theta_deg}"
        )
    if model_cut is not None and not np.isclose(float(model_cut), float(tdbg_config.cut)):
        raise ValueError(f"TDBG model cut={model_cut} does not match tdbg_config.cut={tdbg_config.cut}")


def _run_registered_hf_config_adapter_if_explicit(
    adapter_name: str,
    model: object,
    config: HFConfig,
    kwargs: Mapping[str, Any],
) -> HFResult | None:
    result = _call_registered_hf_adapter(adapter_name, model, config, **dict(kwargs))
    if result is None:
        return None
    if not isinstance(result, HFResult):
        raise TypeError(
            f"Registered run_hf adapter {adapter_name!r} returned {type(result).__name__}; expected HFResult or None"
        )
    return result

def run_hf(model: object, config: HFConfig, **kwargs: Any) -> HFResult:
    """Run HF through a system-provided public hook.

    This façade intentionally does not rewrite existing HF runners or infer
    missing system settings.  Supported systems expose explicit system-owned
    config adapters that return or can be wrapped as an `HFResult`.
    """

    if hasattr(model, "run_hf"):
        raw = model.run_hf(config, **kwargs)  # type: ignore[attr-defined]
        if isinstance(raw, HFResult):
            return raw
        from .models import model_record

        return HFResult(model=model_record(model), config=config, state=raw)

    explicit_result = _run_tdbg_hf_if_explicit(model, config, dict(kwargs))
    if explicit_result is not None:
        return explicit_result

    for adapter_name in (
        "htg_explicit_primitive_run_hf",
        "rlg_hbn_explicit_run_hf",
        "tbg_zero_field_explicit_run_hf",
        "tmbg_polshyn_explicit_run_hf",
    ):
        explicit_result = _run_registered_hf_config_adapter_if_explicit(adapter_name, model, config, kwargs)
        if explicit_result is not None:
            return explicit_result

    raise NotImplementedError(
        "Unified run_hf is frozen at the API level, but this model has no run_hf(config) adapter yet"
    )

__all__ = [name for name in globals() if not name.startswith('__')]
