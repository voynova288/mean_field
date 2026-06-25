from __future__ import annotations

"""Boundary-only bridges from legacy HF density arrays to core contracts.

These helpers intentionally do not participate in the SCF loop.  They build
canonical :mod:`mean_field.core.contracts` views from existing stored-orientation
HF arrays so systems can add contract checks and sidecars without changing their
physics implementation.
"""

from collections.abc import Mapping
from typing import Any

import math
import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    ReferenceDensity as ContractReferenceDensity,
    ReferenceScheme,
)

from .density import (
    DensityConvention as HFDensityConvention,
    ReferenceDensity as HFDensityReference,
    density_to_stored_delta,
    validate_density_array,
)

def float_or_none(
    value: object,
    *,
    include_bool: bool = True,
    finite_only: bool = True,
) -> float | None:
    """Return a float scalar for contract metadata, or ``None`` if unsupported.

    This helper is for post-run contract/sidecar adapters only.  It does not
    participate in SCF logic.  ``include_bool`` and ``finite_only`` preserve the
    small historical differences between system adapters while centralizing the
    conversion policy.
    """

    if not include_bool and isinstance(value, bool | np.bool_):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if finite_only and not math.isfinite(out):
        return None
    return out


def finite_float_or_none(value: object, *, include_bool: bool = True) -> float | None:
    """Return a finite float scalar for contract metadata, or ``None``."""

    return float_or_none(value, include_bool=include_bool, finite_only=True)


def float_diagnostics(
    values: Mapping[str, Any],
    *,
    include_bool: bool = True,
    finite_only: bool = True,
) -> dict[str, float]:
    """Extract numeric scalar diagnostics from a mapping-like object."""

    out: dict[str, float] = {}
    for key, value in values.items():
        scalar = float_or_none(value, include_bool=include_bool, finite_only=finite_only)
        if scalar is not None:
            out[str(key)] = scalar
    return out


def basis_energies_from_h0(h0: np.ndarray) -> np.ndarray:
    """Return eigenvalue bands for a stored ``(state, state, k)`` h0 field."""

    h0_array = np.asarray(h0, dtype=np.complex128)
    out = np.zeros((h0_array.shape[0], h0_array.shape[2]), dtype=float)
    for ik in range(h0_array.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(h0_array[:, :, ik])
    return out


_REFERENCE_SCHEME_ALIASES: dict[str, ReferenceScheme] = {
    "average": "average",
    "avg": "average",
    "cn": "CN",
    "charge_neutral": "CN",
    "charge-neutral": "CN",
    "CN": "CN",
    "central_average": "central_average",
    "central-average": "central_average",
    "custom": "custom",
}


def normalize_contract_reference_scheme(scheme: ReferenceScheme | str) -> ReferenceScheme:
    """Normalize common system spellings to the canonical contract schemes."""

    key = str(scheme)
    try:
        return _REFERENCE_SCHEME_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_REFERENCE_SCHEME_ALIASES))
        raise ValueError(
            f"Unsupported reference scheme {scheme!r}; use one of {allowed} or pass 'custom'."
        ) from exc


def _legacy_reference_array_and_metadata(
    reference: np.ndarray | HFDensityReference,
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(reference, HFDensityReference):
        return reference.data, {
            "hf_density_reference_convention": reference.convention,
            "hf_density_axis_order": reference.axis_order,
        }
    return np.asarray(reference, dtype=np.complex128), {}


def make_contract_reference_density(
    reference: np.ndarray | HFDensityReference | ContractReferenceDensity,
    *,
    scheme: ReferenceScheme | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContractReferenceDensity:
    """Create a canonical contract reference density from a legacy reference.

    ``core.hf.density.ReferenceDensity`` and ``core.contracts.ReferenceDensity``
    are deliberately different classes.  This helper aliases them explicitly and
    preserves legacy reference metadata when converting.
    """

    extra_metadata: dict[str, Any] = {}
    if isinstance(reference, ContractReferenceDensity):
        resolved_scheme = reference.scheme if scheme is None else normalize_contract_reference_scheme(scheme)
        if resolved_scheme != reference.scheme:
            raise ValueError(
                f"Reference scheme override {resolved_scheme!r} does not match existing {reference.scheme!r}"
            )
        reference_array = reference.reference
        extra_metadata.update(reference.metadata)
    else:
        if scheme is None:
            raise ValueError("scheme is required when converting a non-contract reference density")
        resolved_scheme = normalize_contract_reference_scheme(scheme)
        reference_array, extra_metadata = _legacy_reference_array_and_metadata(reference)

    merged_metadata = dict(extra_metadata)
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    validate_density_array(np.asarray(reference_array, dtype=np.complex128))
    return ContractReferenceDensity(
        scheme=resolved_scheme,
        reference=reference_array,
        metadata=merged_metadata,
    )


def density_state_from_delta(
    density_delta: np.ndarray,
    reference: np.ndarray | HFDensityReference | ContractReferenceDensity,
    *,
    reference_scheme: ReferenceScheme | str | None = None,
    filling: float,
    n_occupied_total: int,
    reference_metadata: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContractDensityState:
    """Build a canonical ``DensityState`` from an existing stored delta ``P - R``.

    The input and returned arrays use the existing core-HF stored matrix-field
    convention ``(n_state, n_state, n_k)``.  No projector/idempotency check is
    performed here; use ``assert_density_state_consistent`` at the call site with
    the appropriate ``require_projector`` setting.
    """

    contract_reference = make_contract_reference_density(
        reference,
        scheme=reference_scheme,
        metadata=reference_metadata,
    )
    delta = np.asarray(density_delta, dtype=np.complex128)
    validate_density_array(delta)
    return ContractDensityState(
        density_delta=delta,
        reference=contract_reference,
        filling=float(filling),
        n_occupied_total=int(n_occupied_total),
        metadata={} if metadata is None else dict(metadata),
    )


def density_state_from_projector(
    projector: np.ndarray,
    reference: np.ndarray | HFDensityReference | ContractReferenceDensity,
    *,
    reference_scheme: ReferenceScheme | str | None = None,
    filling: float,
    n_occupied_total: int,
    reference_metadata: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContractDensityState:
    """Build a canonical ``DensityState`` from an existing stored projector ``P``.

    The raw projector is not relabelled.  The returned contract stores only
    ``density_delta = P - R`` and reconstructs ``P`` via ``DensityState.projector``.
    """

    contract_reference = make_contract_reference_density(
        reference,
        scheme=reference_scheme,
        metadata=reference_metadata,
    )
    delta = density_to_stored_delta(
        np.asarray(projector, dtype=np.complex128),
        HFDensityConvention.PROJECTOR,
        reference=contract_reference.reference,
        reference_policy="require",
    )
    return ContractDensityState(
        density_delta=delta,
        reference=contract_reference,
        filling=float(filling),
        n_occupied_total=int(n_occupied_total),
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "basis_energies_from_h0",
    "density_state_from_delta",
    "density_state_from_projector",
    "finite_float_or_none",
    "float_diagnostics",
    "float_or_none",
    "make_contract_reference_density",
    "normalize_contract_reference_scheme",
]
