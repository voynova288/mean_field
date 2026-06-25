from __future__ import annotations

"""Public Hartree-Fock API facade.

Implementation is split across private ``mean_field.api._hf_*`` modules, but
this module preserves the stable ``mean_field.api.hf`` import path and lazy
adapter registry paths.
"""

from ._hf_types import (
    CoulombKernelName,
    DensityConventionName,
    HFAdapterInfo,
    HFAdapterType,
    HFConfig,
    HFState,
    InteractionSchemeName,
    WavefunctionBundle,
)
from ._hf_registry import (
    b0_hf_benchmark_run_to_hf_run_result,
    get_hf_adapter_info,
    htg_hf_run_to_hf_result,
    htg_hf_run_to_hf_run_result,
    list_hf_adapters,
    polshyn_wang_hf_bundle_to_hf_run_result,
    resolve_hf_adapter,
    rlg_hbn_hf_run_to_hf_result,
    rlg_hbn_hf_run_to_hf_run_result,
    tbg_zero_field_hf_run_to_hf_result,
    tbg_zero_field_hf_run_to_hf_run_result,
    tdbg_projected_hf_result_to_hf_run_result,
)
from ._hf_sidecars import (
    _canonical_hf_run_result_sidecar,
    _write_canonical_hf_array_payload,
    reconstruct_canonical_hf_run_result,
)
from ._hf_result import HFResult
from ._hf_dispatch import run_hf

__all__ = [
    "CoulombKernelName",
    "DensityConventionName",
    "HFAdapterInfo",
    "HFAdapterType",
    "HFConfig",
    "HFResult",
    "HFState",
    "InteractionSchemeName",
    "WavefunctionBundle",
    "b0_hf_benchmark_run_to_hf_run_result",
    "get_hf_adapter_info",
    "htg_hf_run_to_hf_result",
    "htg_hf_run_to_hf_run_result",
    "list_hf_adapters",
    "polshyn_wang_hf_bundle_to_hf_run_result",
    "reconstruct_canonical_hf_run_result",
    "resolve_hf_adapter",
    "rlg_hbn_hf_run_to_hf_result",
    "rlg_hbn_hf_run_to_hf_run_result",
    "run_hf",
    "tbg_zero_field_hf_run_to_hf_result",
    "tbg_zero_field_hf_run_to_hf_run_result",
    "tdbg_projected_hf_result_to_hf_run_result",
]
