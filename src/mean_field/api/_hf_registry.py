from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_types import *  # noqa: F401,F403


_HF_ADAPTER_REGISTRY: tuple[HFAdapterInfo, ...] = (
    HFAdapterInfo(
        name="tdbg_projected_hf_result_to_hf_run_result",
        system_name="tdbg",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.tdbg.projected_hf_contracts:tdbg_projected_hf_result_to_hf_run_result",
        description="Post-run canonical HFRunResult view for an explicit TDBGProjectedHFResult.",
        requires_explicit_inputs=("TDBGProjectedHFResult",),
        run_hf_config_reason="Post-run converter only; use tdbg_explicit_projected_run_hf for explicit TDBG config dispatch.",
    ),
    HFAdapterInfo(
        name="tdbg_explicit_projected_run_hf",
        system_name="tdbg",
        adapter_type="run_hf",
        import_path="mean_field.api.hf:run_hf",
        description="Public run_hf dispatch for an explicit TDBGProjectedHFConfig plus init_mode.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("tdbg_config=TDBGProjectedHFConfig", "init_mode"),
        run_hf_config_reason="Requires explicit tdbg_config=TDBGProjectedHFConfig plus init_mode; generic HFConfig inference is not implemented.",
    ),
    HFAdapterInfo(
        name="htg_hf_run_to_hf_run_result",
        system_name="htg",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.htg.mean_field_adapter:htg_hf_run_to_hf_run_result",
        description="Post-run canonical HFRunResult view for an existing primitive-cell HTG HF run.",
        requires_explicit_inputs=("HTGHartreeFockRun",),
        run_hf_config_reason="Post-run converter only; use htg_explicit_primitive_run_hf for explicit HTG config dispatch.",
    ),
    HFAdapterInfo(
        name="htg_hf_run_to_hf_result",
        system_name="htg",
        adapter_type="hf_result",
        import_path="mean_field.systems.htg.mean_field_adapter:htg_hf_run_to_hf_result",
        description="Public HFResult view of an existing primitive-cell HTG HF run.",
        requires_explicit_inputs=("HTGHartreeFockRun",),
        run_hf_config_reason="Post-run HFResult converter only; use htg_explicit_primitive_run_hf for explicit HTG config dispatch.",
    ),
    HFAdapterInfo(
        name="htg_explicit_primitive_run_hf",
        system_name="htg",
        adapter_type="run_hf",
        import_path="mean_field.systems.htg.mean_field_adapter:run_htg_hf_config_adapter",
        description="Public run_hf dispatch for an explicit primitive-cell HTGRunHFConfig.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("htg_config=HTGRunHFConfig",),
        run_hf_config_reason="Requires explicit htg_config=HTGRunHFConfig; generic HFConfig to HTG runner inference is not implemented.",
    ),
    HFAdapterInfo(
        name="htg_supercell_hf_run_to_hf_run_result",
        system_name="htg_supercell",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.htg.supercell_contracts:htg_supercell_hf_run_to_hf_run_result",
        description="Post-run canonical HFRunResult view for an existing HTG folded-supercell HF run.",
        requires_explicit_inputs=("HTGSupercellHartreeFockRun",),
        run_hf_config_reason="Post-run converter only; use htg_explicit_supercell_run_hf for explicit HTG supercell config dispatch.",
    ),
    HFAdapterInfo(
        name="htg_supercell_hf_run_to_hf_result",
        system_name="htg_supercell",
        adapter_type="hf_result",
        import_path="mean_field.systems.htg.supercell_contracts:htg_supercell_hf_run_to_hf_result",
        description="Public HFResult view of an existing HTG folded-supercell HF run.",
        requires_explicit_inputs=("HTGSupercellHartreeFockRun",),
        run_hf_config_reason="Post-run HFResult converter only; use htg_explicit_supercell_run_hf for explicit HTG supercell config dispatch.",
    ),
    HFAdapterInfo(
        name="htg_explicit_supercell_run_hf",
        system_name="htg_supercell",
        adapter_type="run_hf",
        import_path="mean_field.systems.htg.supercell_contracts:run_htg_supercell_hf_config_adapter",
        description="Public run_hf dispatch for an explicit folded-supercell HTGSupercellRunHFConfig.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("htg_supercell_config=HTGSupercellRunHFConfig",),
        run_hf_config_reason="Requires explicit htg_supercell_config=HTGSupercellRunHFConfig; generic fractional-filling inference is not implemented.",
    ),
    HFAdapterInfo(
        name="tbg_zero_field_hf_run_to_hf_run_result",
        system_name="tbg_zero_field",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.tbg.zero_field.hf_contracts:tbg_zero_field_hf_run_to_hf_run_result",
        description="Post-run canonical HFRunResult view for a TBG zero-field HF run plus matching BMSolution grid.",
        requires_explicit_inputs=("RestrictedHartreeFockRun", "grid_solution=BMSolution"),
        run_hf_config_reason="Post-run converter only; use tbg_zero_field_explicit_run_hf for explicit grid-owning config dispatch.",
    ),
    HFAdapterInfo(
        name="tbg_zero_field_hf_run_to_hf_result",
        system_name="tbg_zero_field",
        adapter_type="hf_result",
        import_path="mean_field.systems.tbg.zero_field.hf_contracts:tbg_zero_field_hf_run_to_hf_result",
        description="Public HFResult view of an existing TBG zero-field HF run plus matching BMSolution grid.",
        requires_explicit_inputs=("RestrictedHartreeFockRun", "grid_solution=BMSolution"),
        run_hf_config_reason="Post-run HFResult converter only; use tbg_zero_field_explicit_run_hf for explicit grid-owning config dispatch.",
    ),
    HFAdapterInfo(
        name="tbg_zero_field_explicit_run_hf",
        system_name="tbg_zero_field",
        adapter_type="run_hf",
        import_path="mean_field.systems.tbg.zero_field.hf_contracts:run_tbg_zero_field_hf_config_adapter",
        description="Public run_hf dispatch for an explicit TBGZeroFieldRunHFConfig carrying the matching BMSolution.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("tbg_zero_field_config=TBGZeroFieldRunHFConfig(grid_solution=...)",),
        run_hf_config_reason="Requires explicit tbg_zero_field_config=TBGZeroFieldRunHFConfig carrying the matching BMSolution; generic HFConfig to B0 grid inference is not implemented.",
    ),
    HFAdapterInfo(
        name="b0_hf_benchmark_run_to_hf_run_result",
        system_name="tbg_zero_field",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.tbg.zero_field.hf_contracts:b0_hf_benchmark_run_to_hf_run_result",
        description="Post-run canonical HFRunResult view for a B0 HF benchmark result carrying the matching grid_solution.",
        requires_explicit_inputs=("B0HFBenchmarkRun-like result",),
        run_hf_config_reason="Post-run benchmark converter only; no generic public HFConfig to B0 benchmark runner is frozen.",
    ),
    HFAdapterInfo(
        name="rlg_hbn_hf_run_to_hf_run_result",
        system_name="rlg_hbn",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.RnG_hBN.hf_contracts:rlg_hbn_hf_run_to_hf_run_result",
        description="Post-run canonical HFRunResult view for an existing RnG/hBN HF run.",
        requires_explicit_inputs=("RLGhBNHartreeFockRun",),
        run_hf_config_reason="Post-run converter only; use rlg_hbn_explicit_run_hf for explicit RnG/hBN config dispatch.",
    ),
    HFAdapterInfo(
        name="rlg_hbn_hf_run_to_hf_result",
        system_name="rlg_hbn",
        adapter_type="hf_result",
        import_path="mean_field.systems.RnG_hBN.hf_contracts:rlg_hbn_hf_run_to_hf_result",
        description="Public HFResult view of an existing RnG/hBN HF run.",
        requires_explicit_inputs=("RLGhBNHartreeFockRun",),
        run_hf_config_reason="Post-run HFResult converter only; use rlg_hbn_explicit_run_hf for explicit RnG/hBN config dispatch.",
    ),
    HFAdapterInfo(
        name="rlg_hbn_explicit_run_hf",
        system_name="rlg_hbn",
        adapter_type="run_hf",
        import_path="mean_field.systems.RnG_hBN.hf_contracts:run_rlg_hbn_hf_config_adapter",
        description="Public run_hf dispatch for an explicit RLGhBNRunHFConfig.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("rlg_hbn_config=RLGhBNRunHFConfig",),
        run_hf_config_reason="Requires explicit rlg_hbn_config=RLGhBNRunHFConfig; generic HFConfig to RnG/hBN runner inference is not implemented.",
    ),
    HFAdapterInfo(
        name="polshyn_wang_hf_bundle_to_hf_run_result",
        system_name="tmbg_polshyn",
        adapter_type="canonical_hf_run_result",
        import_path="mean_field.systems.tmbg.polshyn_supercell:polshyn_wang_hf_bundle_to_hf_run_result",
        description="Post-run canonical HFRunResult view for an explicit TMBG Polshyn-Wang (basis, state, info) bundle.",
        requires_explicit_inputs=("PolshynProjectedBasis", "PolshynWangHFState", "info"),
        run_hf_config_reason="Post-run bundle converter only; use tmbg_polshyn_explicit_run_hf for explicit Polshyn config dispatch.",
    ),
    HFAdapterInfo(
        name="tmbg_polshyn_explicit_run_hf",
        system_name="tmbg_polshyn",
        adapter_type="run_hf",
        import_path="mean_field.systems.tmbg.polshyn_supercell:run_tmbg_polshyn_hf_config_adapter",
        description="Public run_hf dispatch for an explicit TMBG Polshyn-Wang configuration.",
        supports_run_hf_config=True,
        requires_explicit_inputs=("tmbg_polshyn_config=PolshynRunHFConfig",),
        run_hf_config_reason="Requires explicit PolshynRunHFConfig with projected_indices, target_band_index, mesh, and interaction shifts; generic HFConfig inference is not implemented.",
    ),
)
_HF_ADAPTERS_BY_NAME: dict[str, HFAdapterInfo] = {info.name: info for info in _HF_ADAPTER_REGISTRY}


def list_hf_adapters(
    *,
    system_name: str | None = None,
    adapter_type: HFAdapterType | None = None,
) -> tuple[HFAdapterInfo, ...]:
    """Return registered safe public HF boundary adapters.

    The registry is intentionally descriptive. Entries with
    ``supports_run_hf_config=False`` are conversion helpers for already-computed
    system HF artifacts and must not be treated as generic ``run_hf(config)``
    support.
    """

    adapters = _HF_ADAPTER_REGISTRY
    if system_name is not None:
        key = str(system_name).lower()
        adapters = tuple(info for info in adapters if info.system_name.lower() == key)
    if adapter_type is not None:
        adapters = tuple(info for info in adapters if info.adapter_type == adapter_type)
    return tuple(adapters)


def get_hf_adapter_info(name: str) -> HFAdapterInfo:
    """Return a registered adapter descriptor by name."""

    try:
        return _HF_ADAPTERS_BY_NAME[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(_HF_ADAPTERS_BY_NAME))
        raise KeyError(f"Unknown HF adapter {name!r}; known adapters: {known}") from exc


def resolve_hf_adapter(name: str) -> Callable[..., Any]:
    """Resolve a registered adapter lazily without importing system modules at API import time."""

    info = get_hf_adapter_info(name)
    module_name, separator, attribute = info.import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Invalid HF adapter import path for {name!r}: {info.import_path!r}")
    module = import_module(module_name)
    adapter = getattr(module, attribute)
    if not callable(adapter):
        raise TypeError(f"Registered HF adapter {name!r} resolved to non-callable {info.import_path!r}")
    return adapter


def _call_registered_hf_adapter(name: str, *args: Any, **kwargs: Any) -> Any:
    return resolve_hf_adapter(name)(*args, **kwargs)


def tdbg_projected_hf_result_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("tdbg_projected_hf_result_to_hf_run_result", *args, **kwargs)


def htg_hf_run_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("htg_hf_run_to_hf_run_result", *args, **kwargs)


def htg_hf_run_to_hf_result(*args: Any, **kwargs: Any) -> Any:
    return _call_registered_hf_adapter("htg_hf_run_to_hf_result", *args, **kwargs)


def htg_supercell_hf_run_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("htg_supercell_hf_run_to_hf_run_result", *args, **kwargs)


def htg_supercell_hf_run_to_hf_result(*args: Any, **kwargs: Any) -> Any:
    return _call_registered_hf_adapter("htg_supercell_hf_run_to_hf_result", *args, **kwargs)


def tbg_zero_field_hf_run_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("tbg_zero_field_hf_run_to_hf_run_result", *args, **kwargs)


def tbg_zero_field_hf_run_to_hf_result(*args: Any, **kwargs: Any) -> Any:
    return _call_registered_hf_adapter("tbg_zero_field_hf_run_to_hf_result", *args, **kwargs)


def b0_hf_benchmark_run_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("b0_hf_benchmark_run_to_hf_run_result", *args, **kwargs)


def rlg_hbn_hf_run_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("rlg_hbn_hf_run_to_hf_run_result", *args, **kwargs)

def rlg_hbn_hf_run_to_hf_result(*args: Any, **kwargs: Any) -> Any:
    return _call_registered_hf_adapter("rlg_hbn_hf_run_to_hf_result", *args, **kwargs)


def polshyn_wang_hf_bundle_to_hf_run_result(*args: Any, **kwargs: Any) -> ContractHFRunResult:
    return _call_registered_hf_adapter("polshyn_wang_hf_bundle_to_hf_run_result", *args, **kwargs)

__all__ = [name for name in globals() if not name.startswith('__')]
