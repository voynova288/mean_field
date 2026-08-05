"""Public system-agnostic TDHF/RPA front door.

Physical systems provide a typed signed-momentum sector through
``TDHFSectorProviderProtocol``.  The public API owns generic structure,
static/dynamic, Wang assignment, and Ward-certificate analysis.  Legacy
objects exposing ``run_tdhf`` remain supported during migration.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from mean_field.core.hf.tdhf_goldstone import TDHFWardSubspaceCertificate
from mean_field.core.hf.tdhf_signed import (
    TDHFSectorProviderProtocol,
    TDHFTypedAnalysis,
    TDHFTypedSector,
    TDHFWardCertificate,
    analyze_tdhf_typed_sector,
)


@dataclass(frozen=True)
class TDHFConfig:
    q_sector: tuple[int, int] | str = "q0"
    channel: str = "all"
    max_pairs: int = 5000
    max_dense_memory_gb: float = 8.0
    assembly: str = "auto"
    metadata: dict[str, object] = field(default_factory=dict)
    _: KW_ONLY
    structure_tolerance: float = 1.0e-10
    hessian_tolerance: float = 1.0e-10
    imag_tolerance: float = 1.0e-10
    norm_tolerance: float = 1.0e-10
    zero_tolerance: float = 1.0e-10
    degeneracy_tolerance: float = 1.0e-10
    pairing_tolerance: float = 1.0e-9
    eigensolver_tolerance: float = 1.0e-9
    metric_gram_tolerance: float = 1.0e-9


def analyze_tdhf_sector(
    sector: TDHFTypedSector,
    config: TDHFConfig,
    *,
    ward_certificate: TDHFWardCertificate | None = None,
    ward_subspace_certificate: TDHFWardSubspaceCertificate | None = None,
) -> TDHFTypedAnalysis:
    """Analyze one already-assembled typed sector through the public API."""

    return analyze_tdhf_typed_sector(
        sector,
        structure_tolerance=config.structure_tolerance,
        hessian_tolerance=config.hessian_tolerance,
        imag_tolerance=config.imag_tolerance,
        norm_tolerance=config.norm_tolerance,
        zero_tolerance=config.zero_tolerance,
        degeneracy_tolerance=config.degeneracy_tolerance,
        pairing_tolerance=config.pairing_tolerance,
        eigensolver_tolerance=config.eigensolver_tolerance,
        metric_gram_tolerance=config.metric_gram_tolerance,
        ward=ward_certificate,
        ward_subspace=ward_subspace_certificate,
    )


def run_tdhf_typed(
    provider: TDHFSectorProviderProtocol,
    config: TDHFConfig,
    **kwargs: Any,
) -> TDHFTypedAnalysis:
    """Explicit typed-provider TDHF entry point."""

    ward_certificate = kwargs.pop("ward_certificate", None)
    ward_subspace_certificate = kwargs.pop("ward_subspace_certificate", None)
    if ward_certificate is not None and not isinstance(
        ward_certificate, TDHFWardCertificate
    ):
        raise TypeError("ward_certificate must be a TDHFWardCertificate")
    if ward_subspace_certificate is not None and not isinstance(
        ward_subspace_certificate, TDHFWardSubspaceCertificate
    ):
        raise TypeError(
            "ward_subspace_certificate must be a TDHFWardSubspaceCertificate"
        )
    sector = provider.build_tdhf_sector(config, **kwargs)
    from mean_field.core.hf.tdhf_signed import (
        TDHFGenericSignedQSector,
        TDHFSelfConjugateQSector,
    )

    if not isinstance(sector, (TDHFGenericSignedQSector, TDHFSelfConjugateQSector)):
        raise TypeError("typed provider returned an unsupported TDHF sector")
    return analyze_tdhf_sector(
        sector,
        config,
        ward_certificate=ward_certificate,
        ward_subspace_certificate=ward_subspace_certificate,
    )


def run_tdhf(
    hf_result_or_archive: object,
    config: TDHFConfig,
    **kwargs: Any,
) -> object:
    """Build and analyze TDHF through a system adapter.

    New adapters implement ``build_tdhf_sector(config, **kwargs)`` and return a
    typed generic or self-conjugate signed-q sector.  During migration, legacy
    objects with their own ``run_tdhf`` method retain their prior behavior.
    """

    if hasattr(hf_result_or_archive, "run_tdhf"):
        return hf_result_or_archive.run_tdhf(config, **kwargs)  # type: ignore[attr-defined]
    if isinstance(hf_result_or_archive, TDHFSectorProviderProtocol):
        return run_tdhf_typed(hf_result_or_archive, config, **kwargs)
    raise NotImplementedError(
        "run_tdhf requires TDHFSectorProviderProtocol.build_tdhf_sector(...) "
        "or a legacy run_tdhf(config) adapter"
    )


__all__ = ["TDHFConfig", "analyze_tdhf_sector", "run_tdhf", "run_tdhf_typed"]
