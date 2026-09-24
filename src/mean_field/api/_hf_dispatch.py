from __future__ import annotations

from typing import Any, Mapping

from ._hf_registry import _call_registered_hf_adapter, list_hf_adapters
from ._hf_result import HFResult
from ._hf_types import HFConfig


def _run_registered_hf_config_adapter_if_explicit(
    adapter_name: str,
    model: object,
    config: HFConfig,
    kwargs: Mapping[str, Any],
) -> HFResult | None:
    result = _call_registered_hf_adapter(
        adapter_name,
        model,
        config,
        **dict(kwargs),
    )
    if result is None:
        return None
    if not isinstance(result, HFResult):
        raise TypeError(
            f"Registered run_hf adapter {adapter_name!r} returned "
            f"{type(result).__name__}; expected HFResult or None"
        )
    return result


def run_hf(model: object, config: HFConfig, **kwargs: Any) -> HFResult:
    """Run HF through the ordered, system-owned typed adapter registry.

    The registry is the only execution authority. Every supported system must
    provide an explicit ``adapter_type='run_hf'`` entry and must fail closed on
    missing or mismatched system configuration; arbitrary ``model.run_hf`` duck
    hooks are not trusted.
    """

    if type(config) is not HFConfig:
        raise TypeError("run_hf requires the exact public HFConfig type")

    for info in list_hf_adapters(adapter_type="run_hf"):
        if not info.supports_run_hf_config:
            raise RuntimeError(
                f"Registered run_hf adapter {info.name!r} does not declare "
                "supports_run_hf_config=True"
            )
        result = _run_registered_hf_config_adapter_if_explicit(
            info.name,
            model,
            config,
            kwargs,
        )
        if result is not None:
            return result

    raise NotImplementedError(
        "Unified run_hf has no run_hf(config) adapter registered as a typed "
        "system boundary for this model"
    )


__all__ = ["run_hf"]
