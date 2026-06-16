from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TDHFConfig:
    q_sector: tuple[int, int] | str = "q0"
    channel: str = "all"
    max_pairs: int = 5000
    max_dense_memory_gb: float = 8.0
    assembly: str = "auto"
    metadata: dict[str, object] = field(default_factory=dict)


def run_tdhf(hf_result_or_archive: object, config: TDHFConfig, **kwargs: Any) -> object:
    """Public TDHF/RPA façade.

    Phase 1 freezes the call shape.  Existing dense q=0 pilots remain in
    system/devtool adapters until they can be moved behind this hook.
    """

    if hasattr(hf_result_or_archive, "run_tdhf"):
        return hf_result_or_archive.run_tdhf(config, **kwargs)  # type: ignore[attr-defined]
    raise NotImplementedError("Unified run_tdhf requires an HF result/archive adapter exposing run_tdhf(config)")


__all__ = ["TDHFConfig", "run_tdhf"]
