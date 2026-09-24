"""Fail-closed interaction binding for the InAs/GaSb matrix-EI solver.

The solver consumes one normalized component contract and never infers physical
semantics from optional attributes or caller-supplied labels.  Concrete-class
awareness is confined to the closed registry in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .angular_fock import PolarHarmonicProjectedFockOperator
from .axial_fock import (
    AxialAveragedProjectedFockOperator,
    AxialProjectedFockOperator,
)
from .compressed_fock import CompressedProjectedFockOperator
from .hartree import ProjectedHartreeFockOperator
from .kane4_bundle import Kane4Bundle
from .projected_fock import ProjectedFockOperator

ComplexArray = np.ndarray
MatrixEIInteractionKind = Literal["exchange_only", "projected_hartree_fock"]
MatrixEIReferencePolicy = Literal[
    "normal_ordered_exchange_only",
    "normal_ordered_hartree_fock",
]
MatrixEIInteractionAuthority = Literal[
    "source_bound_explicit_kernel",
    "builder_attested_reduction",
    "diagnostic_reduction",
    "explicit_precomputed_tensor",
    "source_bound_common_electrostatics",
    "explicit_mixed_electrostatics_scaffold",
]


@dataclass(frozen=True)
class MatrixEIInteractionSpec:
    """Solver-visible physical semantics issued by the closed registry."""

    kind: MatrixEIInteractionKind
    required_reference_policy: MatrixEIReferencePolicy
    model_label: str
    electrostatic_ensemble: str
    authority: MatrixEIInteractionAuthority

    @property
    def projected_hartree_enabled(self) -> bool:
        return self.kind == "projected_hartree_fock"


@runtime_checkable
class MatrixEIInteractionProtocol(Protocol):
    """Normalized total/component action consumed by matrix-EI."""

    spec: MatrixEIInteractionSpec
    bundle_fingerprint: str

    def __call__(self, density_delta: ComplexArray) -> ComplexArray: ...

    def components(
        self, density_delta: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]: ...

    def fingerprint(self) -> str: ...


@dataclass(frozen=True)
class _BoundMatrixEIInteraction:
    operator: object
    spec: MatrixEIInteractionSpec
    bundle_fingerprint: str
    _fingerprint: str

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        return np.asarray(self.operator(density_delta), dtype=np.complex128)  # type: ignore[operator]

    def components(
        self, density_delta: ComplexArray
    ) -> tuple[ComplexArray, ComplexArray]:
        if self.spec.projected_hartree_enabled:
            hartree, fock = self.operator.components(density_delta)  # type: ignore[attr-defined]
            return (
                np.asarray(hartree, dtype=np.complex128),
                np.asarray(fock, dtype=np.complex128),
            )
        fock = np.asarray(self.operator(density_delta), dtype=np.complex128)  # type: ignore[operator]
        return np.zeros_like(fock), fock

    def fingerprint(self) -> str:
        return self._fingerprint


def _exchange_spec(
    model_label: str,
    authority: MatrixEIInteractionAuthority,
) -> MatrixEIInteractionSpec:
    return MatrixEIInteractionSpec(
        kind="exchange_only",
        required_reference_policy="normal_ordered_exchange_only",
        model_label=model_label,
        electrostatic_ensemble="hartree_disabled_exchange_only",
        authority=authority,
    )


_EXCHANGE_SPECS = {
    ProjectedFockOperator: _exchange_spec(
        "normal-ordered exchange-only", "source_bound_explicit_kernel"
    ),
    AxialProjectedFockOperator: _exchange_spec(
        "normal-ordered exchange-only", "builder_attested_reduction"
    ),
    AxialAveragedProjectedFockOperator: _exchange_spec(
        "normal-ordered co-rotating-m0 exchange diagnostic",
        "diagnostic_reduction",
    ),
    PolarHarmonicProjectedFockOperator: _exchange_spec(
        "normal-ordered selected-harmonic exchange diagnostic",
        "diagnostic_reduction",
    ),
    CompressedProjectedFockOperator: _exchange_spec(
        "normal-ordered precomputed-tensor exchange-only",
        "explicit_precomputed_tensor",
    ),
}
_REGISTERED_INTERACTION_TYPES = frozenset(
    {*_EXCHANGE_SPECS, ProjectedHartreeFockOperator}
)


def _registered_interaction_spec(operator: object) -> MatrixEIInteractionSpec:
    exchange_spec = _EXCHANGE_SPECS.get(type(operator))
    if exchange_spec is not None:
        return exchange_spec
    if type(operator) is not ProjectedHartreeFockOperator:
        raise TypeError("matrix-EI interaction type is not registered")
    hartree_fock = operator
    if hartree_fock.allow_mixed_electrostatics:
        ensemble = (
            "mixed_electrostatics_scaffold:hartree="
            f"{hartree_fock.hartree.boundary_condition};"
            f"fock_source={hartree_fock.fock.self_cell_description}"
        )
        authority: MatrixEIInteractionAuthority = (
            "explicit_mixed_electrostatics_scaffold"
        )
    else:
        ensemble = str(hartree_fock.hartree.boundary_condition)
        authority = "source_bound_common_electrostatics"
    return MatrixEIInteractionSpec(
        kind="projected_hartree_fock",
        required_reference_policy="normal_ordered_hartree_fock",
        model_label="normal-ordered Hartree-Fock",
        electrostatic_ensemble=ensemble,
        authority=authority,
    )


def bind_matrix_ei_interaction(
    bundle: Kane4Bundle,
    operator: object,
) -> MatrixEIInteractionProtocol:
    """Validate and normalize one registered concrete interaction.

    Structural duck typing is intentionally insufficient: an unregistered
    object cannot self-assert its reference policy, electrostatic ensemble, or
    scientific authority.
    """

    if not isinstance(bundle, Kane4Bundle):
        raise TypeError("matrix-EI interaction binding requires Kane4Bundle")
    if type(operator) not in _REGISTERED_INTERACTION_TYPES:
        raise TypeError(
            "matrix-EI solver requires an exact registered interaction type; "
            "unregistered subclasses and duck-typed objects are rejected"
        )
    bundle.validate()
    spec = _registered_interaction_spec(operator)
    operator.validate_against_bundle(bundle)  # type: ignore[attr-defined]
    fingerprint = operator.fingerprint()  # type: ignore[attr-defined]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("interaction operator fingerprint is required")
    bound = _BoundMatrixEIInteraction(
        operator=operator,
        spec=spec,
        bundle_fingerprint=bundle.fingerprint(),
        _fingerprint=fingerprint,
    )
    if not isinstance(bound, MatrixEIInteractionProtocol):
        raise TypeError("internal matrix-EI interaction binding is incomplete")
    return bound


__all__ = [
    "MatrixEIInteractionAuthority",
    "MatrixEIInteractionKind",
    "MatrixEIInteractionProtocol",
    "MatrixEIInteractionSpec",
    "bind_matrix_ei_interaction",
]
