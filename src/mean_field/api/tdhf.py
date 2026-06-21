from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class TDHFConfig:
    q_sector: tuple[int, int] | str = "q0"
    channel: str = "all"
    max_pairs: int = 5000
    max_dense_memory_gb: float = 8.0
    assembly: str = "auto"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TDHFAdapterInfo:
    name: str
    system_name: str
    import_path: str
    description: str
    requires_explicit_inputs: tuple[str, ...] = ()


_TDHF_ADAPTERS: tuple[TDHFAdapterInfo, ...] = (
    TDHFAdapterInfo(
        name="rlg_hbn_q0",
        system_name="rlg_hbn",
        import_path="mean_field.systems.RnG_hBN.tdhf:build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf",
        description="RLG/hBN q=0 TDHF matrix assembly from a raw HF run plus canonical HF state/result.",
        requires_explicit_inputs=("raw RLGhBNHartreeFockRun", "canonical HF state/result"),
    ),
    TDHFAdapterInfo(
        name="rlg_hbn_finite_q",
        system_name="rlg_hbn",
        import_path="mean_field.systems.RnG_hBN.tdhf:build_rlg_hbn_tdhf_q_matrices_from_canonical_hf",
        description="RLG/hBN finite-q TDHF matrix assembly from a raw HF run plus canonical HF state/result.",
        requires_explicit_inputs=("raw RLGhBNHartreeFockRun", "canonical HF state/result", "integer q_shift"),
    ),
)


def list_tdhf_adapters(*, system_name: str | None = None) -> tuple[TDHFAdapterInfo, ...]:
    adapters = _TDHF_ADAPTERS
    if system_name is not None:
        key = str(system_name).lower().replace("-", "_")
        adapters = tuple(item for item in adapters if item.system_name.lower().replace("-", "_") == key)
    return adapters


def get_tdhf_adapter_info(name: str) -> TDHFAdapterInfo:
    for item in _TDHF_ADAPTERS:
        if item.name == name:
            return item
    raise KeyError(f"Unknown TDHF adapter {name!r}; available: {[item.name for item in _TDHF_ADAPTERS]}")


def resolve_tdhf_adapter(name: str) -> Callable[..., Any]:
    info = get_tdhf_adapter_info(name)
    module_name, attr = info.import_path.split(":", 1)
    return getattr(import_module(module_name), attr)


def _run_rlg_hbn_tdhf(hf_result_or_archive: object, config: TDHFConfig, *, adapter: str, **kwargs: Any) -> object:
    builder = resolve_tdhf_adapter(adapter)
    raw_run = kwargs.pop("run", None)
    canonical_hf = kwargs.pop("canonical_hf", None)
    if raw_run is None and isinstance(hf_result_or_archive, tuple) and len(hf_result_or_archive) == 2:
        raw_run, canonical_hf = hf_result_or_archive
    if raw_run is None:
        raw_run = hf_result_or_archive
    if canonical_hf is None:
        canonical_hf = getattr(hf_result_or_archive, "canonical_run_result", None)
    if canonical_hf is None:
        raise ValueError("RLG/hBN TDHF adapter requires canonical_hf=... or a (raw_run, canonical_hf) tuple")
    common = dict(
        beta=float(kwargs.pop("beta", 1.0)),
        max_pairs=int(kwargs.pop("max_pairs", config.max_pairs)),
        structure_tolerance=float(kwargs.pop("structure_tolerance", 1.0e-6)),
    )
    if adapter == "rlg_hbn_q0":
        assembly = str(config.assembly)
        if assembly == "auto":
            assembly = "vectorized"
        return builder(raw_run, canonical_hf, assembly=assembly, **common, **kwargs)
    q_sector = config.q_sector
    if isinstance(q_sector, str):
        raise ValueError("Finite-q RLG/hBN TDHF requires TDHFConfig.q_sector=(dq1,dq2), not a string")
    return builder(
        raw_run,
        canonical_hf,
        tuple(int(value) for value in q_sector),
        channel=str(config.channel),
        **common,
        **kwargs,
    )


def run_tdhf(hf_result_or_archive: object, config: TDHFConfig, *, adapter: str | None = None, **kwargs: Any) -> object:
    """Public TDHF/RPA façade with explicit adapter registry."""

    if adapter is not None:
        if adapter in {"rlg_hbn_q0", "rlg_hbn_finite_q"}:
            return _run_rlg_hbn_tdhf(hf_result_or_archive, config, adapter=adapter, **kwargs)
        resolved = resolve_tdhf_adapter(adapter)
        return resolved(hf_result_or_archive, config, **kwargs)
    if hasattr(hf_result_or_archive, "run_tdhf"):
        return hf_result_or_archive.run_tdhf(config, **kwargs)  # type: ignore[attr-defined]
    raise NotImplementedError(
        "Unified run_tdhf requires an explicit registered adapter such as 'rlg_hbn_q0'/'rlg_hbn_finite_q', "
        "or an object exposing run_tdhf(config)"
    )


__all__ = [
    "TDHFAdapterInfo",
    "TDHFConfig",
    "get_tdhf_adapter_info",
    "list_tdhf_adapters",
    "resolve_tdhf_adapter",
    "run_tdhf",
]
