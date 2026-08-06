"""Authority-limited local Vituri-2024 scalar-Hessian elements.

This module maps the already-derived ours-ordered antisymmetrized coefficient
from :mod:`vituri2024_vertex` to paper Eq. C9.  With a Thouless coordinate
``z[a,A]`` and the annihilator order used by the projected Hamiltonian,

``delta2E = z^dagger A z + Re[z^T B^* z]``,
``A[aA,bB] = (epsilon_a-epsilon_A) delta_ab delta_AB
             - vbar[a,B;A,b]/Area``,
``B[aA,bB] = -vbar[a,b;A,B]/Area``.

Thus A couples exactly equal transfers ``q_aA=q_bB`` and B couples exact
opposites ``q_aA=-q_bB``.  The antisymmetrized vertex has units
eV*Angstrom^2 and is divided by the finite area exactly once; the ``1/(4A)``
of the full four-index Hamiltonian is not applied again.  No post-Hermitizing
operation is performed.

The apparent paper-C3 inconsistency is not reinterpreted here: it is already
handled in :mod:`vituri2024_vertex`, which derives ``vbar`` from the projected
Hamiltonian instead of copying literal C3.  This file supplies local elements
only.  It does not assemble a dense RPA matrix, promote local transfers to a
typed signed-q sector, or run an eigensolver.

All diagonal-HF and finite-area provenance accepted here is caller-attested.
In particular, source stationarity, area/mesh/quadrature/torus authority, the
q=0 background, and domain/cutoff convergence remain uncertified.  Therefore
these receipts are not production RPA or paper-numerical authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Final, Literal, TypeAlias

import numpy as np

from .vituri2024 import SM_TEX_SHA256
from .vituri2024_interaction import InteractionInput
from .vituri2024_vertex import (
    VERTEX_AUTHORITY,
    Vituri2024AntisymmetrizedVertexReceipt,
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024Orbital,
    vituri2024_antisymmetrized_projected_vertex,
)

DIAGONAL_HF_SOURCE_PROVIDER_STATUS: Final[str] = (
    "caller_attested_immutable_diagonal_source_not_independently_verified"
)
FINITE_AREA_PROVIDER_STATUS: Final[str] = (
    "caller_attested_finite_area_not_independently_verified"
)
RPA_ELEMENT_AUTHORITY: Final[str] = (
    "paper_C9_index_sign_mapping_from_derived_projected_vertex_"
    "no_production_authority"
)
SCALAR_HESSIAN_EQUATION: Final[str] = (
    "delta2E=z^dagger A z+Re[z^T B^* z]"
)
RPA_A_ELEMENT_EQUATION: Final[str] = (
    "A_{aA,bB}=(epsilon_a-epsilon_A)delta_{ab}delta_{AB}"
    "-vbar_{aB;Ab}/Area"
)
RPA_B_ELEMENT_EQUATION: Final[str] = "B_{aA,bB}=-vbar_{ab;AB}/Area"
RPA_ELEMENT_UNITS: Final[str] = "eV"
RPA_VERTEX_UNITS: Final[str] = "eV*Angstrom^2"
RPA_AREA_UNITS: Final[str] = "Angstrom^2"
RPA_ELEMENT_NO_GO_LIMITS: Final[tuple[str, ...]] = (
    "no_hf_stationarity_certification",
    "no_area_mesh_quadrature_or_torus_authority",
    "no_q0_background_or_normal_ordering_authority",
    "no_domain_or_cutoff_convergence_authority",
    "no_dense_rpa_assembly",
    "no_typed_signed_q_sector_promotion",
    "no_rpa_eigensolver",
    "no_production_or_paper_numerical_authority",
)

RPAElementKind: TypeAlias = Literal["A", "B"]


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_finite_real(value: object, *, label: str) -> float:
    result = _finite_real(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_complex(value: object, *, label: str) -> complex:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a strict complex scalar")
    try:
        result = complex(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a complex scalar") from exc
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a lowercase SHA256 digest")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty caller-attested text")
    return value


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    return value


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _payload_without_fingerprint(receipt: object, fingerprint_name: str) -> dict[str, object]:
    return {
        item.name: getattr(receipt, item.name)
        for item in fields(receipt)  # type: ignore[arg-type]
        if item.name != fingerprint_name
    }


@dataclass(frozen=True, slots=True)
class Vituri2024DiagonalHFTransitionReceipt:
    """Caller-attested occupied-to-empty transition from one diagonal source.

    The particle occupation is fixed to zero and the hole occupation to one.
    Positive gap and immutable provenance are structural checks only; no
    mechanism in this adapter certifies that the supplied diagonal source is
    a stationary Hartree--Fock solution.
    """

    particle: Vituri2024Orbital
    hole: Vituri2024Orbital
    particle_energy_ev: float
    hole_energy_ev: float
    source_artifact_sha256: str
    source_text: str
    source_provider_status: str = field(
        default=DIAGONAL_HF_SOURCE_PROVIDER_STATUS, init=False
    )
    source_artifact_immutable: bool = field(default=True, init=False)
    particle_occupation: int = field(default=0, init=False)
    hole_occupation: int = field(default=1, init=False)
    gap_ev: float = field(init=False)
    caller_attested: bool = field(default=True, init=False)
    hf_stationarity_certified: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    source_fingerprint: str = field(init=False)
    transition_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.particle) is not Vituri2024Orbital:
            raise TypeError("particle must be a Vituri2024Orbital")
        if type(self.hole) is not Vituri2024Orbital:
            raise TypeError("hole must be a Vituri2024Orbital")
        if self.particle == self.hole:
            raise ValueError("particle and hole must be distinct fixed-occupation orbitals")
        particle_energy = _finite_real(
            self.particle_energy_ev, label="particle_energy_ev"
        )
        hole_energy = _finite_real(self.hole_energy_ev, label="hole_energy_ev")
        gap = particle_energy - hole_energy
        if not math.isfinite(gap):
            raise OverflowError("transition gap is outside finite float64 range")
        if gap <= 0.0:
            raise ValueError("diagonal-HF transition gap must be positive")
        object.__setattr__(self, "particle_energy_ev", particle_energy)
        object.__setattr__(self, "hole_energy_ev", hole_energy)
        object.__setattr__(self, "gap_ev", gap)
        _sha256(self.source_artifact_sha256, label="source_artifact_sha256")
        _nonempty_text(self.source_text, label="source_text")
        locked = (
            self.source_provider_status == DIAGONAL_HF_SOURCE_PROVIDER_STATUS,
            self.source_artifact_immutable is True,
            type(self.particle_occupation) is int and self.particle_occupation == 0,
            type(self.hole_occupation) is int and self.hole_occupation == 1,
            self.caller_attested is True,
            self.hf_stationarity_certified is False,
            self.production_ready is False,
        )
        if not all(locked):
            raise ValueError("diagonal-HF transition scope or occupations were changed")
        source_fingerprint = _fingerprint(
            {
                "source_artifact_sha256": self.source_artifact_sha256,
                "source_text": self.source_text,
                "source_provider_status": self.source_provider_status,
                "source_artifact_immutable": self.source_artifact_immutable,
            }
        )
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        object.__setattr__(
            self,
            "transition_fingerprint",
            _fingerprint(_payload_without_fingerprint(self, "transition_fingerprint")),
        )

    @property
    def fingerprint(self) -> str:
        return self.transition_fingerprint


@dataclass(frozen=True, slots=True)
class Vituri2024FiniteAreaReceipt:
    """Caller-attested positive finite area, without discretization authority."""

    area_angstrom_squared: float
    provider_sha256: str
    source_text: str
    provider_status: str = field(default=FINITE_AREA_PROVIDER_STATUS, init=False)
    caller_attested: bool = field(default=True, init=False)
    mesh_authority: bool = field(default=False, init=False)
    quadrature_authority: bool = field(default=False, init=False)
    torus_authority: bool = field(default=False, init=False)
    background_authority: bool = field(default=False, init=False)
    paper_authority: bool = field(default=False, init=False)
    production_authority: bool = field(default=False, init=False)
    area_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "area_angstrom_squared",
            _positive_finite_real(
                self.area_angstrom_squared, label="area_angstrom_squared"
            ),
        )
        _sha256(self.provider_sha256, label="provider_sha256")
        _nonempty_text(self.source_text, label="source_text")
        locked = (
            self.provider_status == FINITE_AREA_PROVIDER_STATUS,
            self.caller_attested is True,
            self.mesh_authority is False,
            self.quadrature_authority is False,
            self.torus_authority is False,
            self.background_authority is False,
            self.paper_authority is False,
            self.production_authority is False,
        )
        if not all(locked):
            raise ValueError("finite-area receipt authority was inflated")
        object.__setattr__(
            self,
            "area_fingerprint",
            _fingerprint(_payload_without_fingerprint(self, "area_fingerprint")),
        )

    @property
    def fingerprint(self) -> str:
        return self.area_fingerprint


def _validated_transition(
    receipt: Vituri2024DiagonalHFTransitionReceipt,
) -> Vituri2024DiagonalHFTransitionReceipt:
    if type(receipt) is not Vituri2024DiagonalHFTransitionReceipt:
        raise TypeError(
            "transition must be a Vituri2024DiagonalHFTransitionReceipt"
        )
    clean = Vituri2024DiagonalHFTransitionReceipt(
        particle=receipt.particle,
        hole=receipt.hole,
        particle_energy_ev=receipt.particle_energy_ev,
        hole_energy_ev=receipt.hole_energy_ev,
        source_artifact_sha256=receipt.source_artifact_sha256,
        source_text=receipt.source_text,
    )
    if clean != receipt:
        raise ValueError("diagonal-HF transition receipt is inconsistent or tampered")
    return clean


def _validated_area(receipt: Vituri2024FiniteAreaReceipt) -> Vituri2024FiniteAreaReceipt:
    if type(receipt) is not Vituri2024FiniteAreaReceipt:
        raise TypeError("area must be a Vituri2024FiniteAreaReceipt")
    clean = Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=receipt.area_angstrom_squared,
        provider_sha256=receipt.provider_sha256,
        source_text=receipt.source_text,
    )
    if clean != receipt:
        raise ValueError("finite-area receipt is inconsistent or tampered")
    return clean


def _require_same_source(
    left: Vituri2024DiagonalHFTransitionReceipt,
    right: Vituri2024DiagonalHFTransitionReceipt,
) -> None:
    if (
        left.source_artifact_sha256,
        left.source_text,
        left.source_provider_status,
        left.source_artifact_immutable,
        left.source_fingerprint,
    ) != (
        right.source_artifact_sha256,
        right.source_text,
        right.source_provider_status,
        right.source_artifact_immutable,
        right.source_fingerprint,
    ):
        raise ValueError(
            "left and right transitions must use the same immutable source receipt"
        )


def _require_consistent_duplicate_orbitals(
    left: Vituri2024DiagonalHFTransitionReceipt,
    right: Vituri2024DiagonalHFTransitionReceipt,
) -> None:
    entries = (
        (left.particle, left.particle_energy_ev, left.particle_occupation),
        (left.hole, left.hole_energy_ev, left.hole_occupation),
        (right.particle, right.particle_energy_ev, right.particle_occupation),
        (right.hole, right.hole_energy_ev, right.hole_occupation),
    )
    for index, (orbital, energy, occupation) in enumerate(entries):
        for previous_orbital, previous_energy, previous_occupation in entries[:index]:
            if orbital != previous_orbital:
                continue
            if energy != previous_energy:
                raise ValueError("duplicate orbital has inconsistent diagonal energies")
            if occupation != previous_occupation:
                raise ValueError("duplicate orbital has inconsistent fixed occupations")


def _transfer(
    transition: Vituri2024DiagonalHFTransitionReceipt,
) -> tuple[float, float]:
    particle = transition.particle.momentum_inverse_angstrom
    hole = transition.hole.momentum_inverse_angstrom
    result = (particle[0] - hole[0], particle[1] - hole[1])
    if not all(math.isfinite(component) for component in result):
        raise OverflowError("transition transfer is outside finite float64 range")
    return result


def _validated_vertex(
    receipt: Vituri2024AntisymmetrizedVertexReceipt,
) -> Vituri2024AntisymmetrizedVertexReceipt:
    if type(receipt) is not Vituri2024AntisymmetrizedVertexReceipt:
        raise TypeError("vertex must be a Vituri2024AntisymmetrizedVertexReceipt")
    expected_fingerprint = _fingerprint(
        _payload_without_fingerprint(receipt, "vertex_fingerprint")
    )
    if receipt.fingerprint != expected_fingerprint:
        raise ValueError("vertex fingerprint mismatch; receipt may be tampered")
    receipt.kinematics.require_conserving()
    return receipt


def _element_payload(receipt: "Vituri2024RPAElementReceipt") -> dict[str, object]:
    return _payload_without_fingerprint(receipt, "element_fingerprint")


@dataclass(frozen=True, slots=True)
class Vituri2024RPAElementReceipt:
    """One local C9 A or B scalar-Hessian element; never a dense RPA object."""

    element_kind: RPAElementKind
    value_ev: complex
    one_body_contribution_ev: float
    interaction_contribution_ev: complex
    left_transition: Vituri2024DiagonalHFTransitionReceipt
    right_transition: Vituri2024DiagonalHFTransitionReceipt
    left_transition_fingerprint: str
    right_transition_fingerprint: str
    source_fingerprint: str
    left_transfer_inverse_angstrom: tuple[float, float]
    right_transfer_inverse_angstrom: tuple[float, float]
    delta1_ev: float
    area: Vituri2024FiniteAreaReceipt
    area_fingerprint: str
    interaction_receipt_fingerprint: str
    interaction_binding_fingerprint: str | None
    context_fingerprint: str
    kinematics: Vituri2024FourPointKinematicsReceipt
    kinematics_fingerprint: str
    vertex: Vituri2024AntisymmetrizedVertexReceipt
    vertex_fingerprint: str
    equation: str
    quartet_index_order: str
    scalar_hessian_equation: str = field(
        default=SCALAR_HESSIAN_EQUATION, init=False
    )
    units: str = field(default=RPA_ELEMENT_UNITS, init=False)
    vertex_units: str = field(default=RPA_VERTEX_UNITS, init=False)
    area_units: str = field(default=RPA_AREA_UNITS, init=False)
    authority: str = field(default=RPA_ELEMENT_AUTHORITY, init=False)
    vertex_authority: str = field(default=VERTEX_AUTHORITY, init=False)
    c3_inconsistency_handled_by_vertex: bool = field(default=True, init=False)
    vertex_divided_by_area_exactly_once: bool = field(default=True, init=False)
    extra_half_factor_applied: bool = field(default=False, init=False)
    extra_quarter_factor_applied: bool = field(default=False, init=False)
    post_hermitized: bool = field(default=False, init=False)
    dense_rpa_assembly: bool = field(default=False, init=False)
    typed_signed_q_sector_promotion: bool = field(default=False, init=False)
    hf_stationarity_certified: bool = field(default=False, init=False)
    area_mesh_quadrature_torus_certified: bool = field(default=False, init=False)
    q0_background_certified: bool = field(default=False, init=False)
    domain_cutoff_convergence_certified: bool = field(default=False, init=False)
    production_rpa_authority: bool = field(default=False, init=False)
    paper_numerical_parity: bool = field(default=False, init=False)
    no_go_limits: tuple[str, ...] = field(
        default=RPA_ELEMENT_NO_GO_LIMITS, init=False
    )
    element_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.element_kind not in ("A", "B"):
            raise ValueError("element_kind must be exactly 'A' or 'B'")
        value = _finite_complex(self.value_ev, label="RPA element value")
        one_body = _finite_real(
            self.one_body_contribution_ev, label="one_body_contribution_ev"
        )
        interaction = _finite_complex(
            self.interaction_contribution_ev,
            label="interaction_contribution_ev",
        )
        object.__setattr__(self, "value_ev", value)
        object.__setattr__(self, "one_body_contribution_ev", one_body)
        object.__setattr__(self, "interaction_contribution_ev", interaction)

        left = _validated_transition(self.left_transition)
        right = _validated_transition(self.right_transition)
        object.__setattr__(self, "left_transition", left)
        object.__setattr__(self, "right_transition", right)
        _require_same_source(left, right)
        _require_consistent_duplicate_orbitals(left, right)
        for label, supplied, expected in (
            ("left transition", self.left_transition_fingerprint, left.fingerprint),
            ("right transition", self.right_transition_fingerprint, right.fingerprint),
            ("source", self.source_fingerprint, left.source_fingerprint),
        ):
            _sha256(supplied, label=f"{label}_fingerprint")
            if supplied != expected:
                raise ValueError(f"{label} fingerprint mismatch")

        left_transfer = _transfer(left)
        right_transfer = _transfer(right)
        if self.left_transfer_inverse_angstrom != left_transfer:
            raise ValueError("left transfer does not match particle minus hole momentum")
        if self.right_transfer_inverse_angstrom != right_transfer:
            raise ValueError("right transfer does not match particle minus hole momentum")
        if self.element_kind == "A":
            if left_transfer != right_transfer:
                raise ValueError("A element requires exact q_left=q_right")
            expected_equation = RPA_A_ELEMENT_EQUATION
            expected_quartet = (left.particle, right.hole, left.hole, right.particle)
            expected_order = "(a,B;A,b)"
            expected_one_body = (
                left.gap_ev
                if left.particle == right.particle and left.hole == right.hole
                else 0.0
            )
        else:
            if left_transfer != (-right_transfer[0], -right_transfer[1]):
                raise ValueError("B element requires exact q_left=-q_right")
            expected_equation = RPA_B_ELEMENT_EQUATION
            expected_quartet = (left.particle, right.particle, left.hole, right.hole)
            expected_order = "(a,b;A,B)"
            expected_one_body = 0.0
        if one_body != expected_one_body:
            raise ValueError("one-body contribution is inconsistent with element kind")

        delta1 = _finite_real(self.delta1_ev, label="Delta1")
        object.__setattr__(self, "delta1_ev", delta1)
        area = _validated_area(self.area)
        object.__setattr__(self, "area", area)
        _sha256(self.area_fingerprint, label="area_fingerprint")
        if self.area_fingerprint != area.fingerprint:
            raise ValueError("area fingerprint mismatch")

        if type(self.kinematics) is not Vituri2024FourPointKinematicsReceipt:
            raise TypeError("kinematics must be a Vituri2024FourPointKinematicsReceipt")
        kinematics = self.kinematics.require_conserving()
        if kinematics.momentum_tolerance_inverse_angstrom != 0.0:
            raise ValueError("RPA local element requires diagnostic tolerance exactly zero")
        if (
            kinematics.alpha,
            kinematics.beta,
            kinematics.gamma,
            kinematics.delta,
        ) != expected_quartet:
            raise ValueError("four-point kinematics does not match the C9 index order")
        _sha256(self.kinematics_fingerprint, label="kinematics_fingerprint")
        if self.kinematics_fingerprint != kinematics.fingerprint:
            raise ValueError("kinematics fingerprint mismatch")

        vertex = _validated_vertex(self.vertex)
        object.__setattr__(self, "vertex", vertex)
        if vertex.kinematics != kinematics:
            raise ValueError("vertex and element kinematics mismatch")
        if vertex.delta1_ev != delta1:
            raise ValueError("vertex and element Delta1 mismatch")
        _sha256(self.vertex_fingerprint, label="vertex_fingerprint")
        if self.vertex_fingerprint != vertex.fingerprint:
            raise ValueError("vertex fingerprint mismatch")
        _sha256(
            self.interaction_receipt_fingerprint,
            label="interaction_receipt_fingerprint",
        )
        if self.interaction_receipt_fingerprint != vertex.interaction_receipt_fingerprint:
            raise ValueError("element and vertex interaction fingerprint mismatch")
        if self.interaction_binding_fingerprint is None:
            if vertex.interaction_binding_fingerprint is not None:
                raise ValueError("element omitted the vertex interaction binding")
        else:
            _sha256(
                self.interaction_binding_fingerprint,
                label="interaction_binding_fingerprint",
            )
            if self.interaction_binding_fingerprint != vertex.interaction_binding_fingerprint:
                raise ValueError("element and vertex interaction binding mismatch")

        expected_interaction = -vertex.value / area.area_angstrom_squared
        if interaction != expected_interaction:
            raise ValueError("interaction contribution must be -vbar/Area exactly once")
        if value != complex(expected_one_body) + expected_interaction:
            raise ValueError("RPA element value is inconsistent with C9")
        expected_context_fingerprint = _fingerprint(
            {
                "delta1_ev": delta1,
                "interaction_receipt_fingerprint": (
                    vertex.interaction_receipt_fingerprint
                ),
                "interaction_binding_fingerprint": (
                    vertex.interaction_binding_fingerprint
                ),
                "area_fingerprint": area.fingerprint,
            }
        )
        _sha256(self.context_fingerprint, label="context_fingerprint")
        if self.context_fingerprint != expected_context_fingerprint:
            raise ValueError("Delta1/interaction/area context fingerprint mismatch")

        if self.equation != expected_equation:
            raise ValueError("element equation does not match element_kind")
        if self.quartet_index_order != expected_order:
            raise ValueError("quartet index-order label is inconsistent")
        locked = (
            self.scalar_hessian_equation == SCALAR_HESSIAN_EQUATION,
            self.units == RPA_ELEMENT_UNITS,
            self.vertex_units == RPA_VERTEX_UNITS,
            self.area_units == RPA_AREA_UNITS,
            self.authority == RPA_ELEMENT_AUTHORITY,
            self.vertex_authority == VERTEX_AUTHORITY,
            self.c3_inconsistency_handled_by_vertex is True,
            self.vertex_divided_by_area_exactly_once is True,
            self.extra_half_factor_applied is False,
            self.extra_quarter_factor_applied is False,
            self.post_hermitized is False,
            self.dense_rpa_assembly is False,
            self.typed_signed_q_sector_promotion is False,
            self.hf_stationarity_certified is False,
            self.area_mesh_quadrature_torus_certified is False,
            self.q0_background_certified is False,
            self.domain_cutoff_convergence_certified is False,
            self.production_rpa_authority is False,
            self.paper_numerical_parity is False,
            self.no_go_limits == RPA_ELEMENT_NO_GO_LIMITS,
        )
        if not all(locked):
            raise ValueError("RPA element authority or NO-GO scope was inflated")
        object.__setattr__(
            self, "element_fingerprint", _fingerprint(_element_payload(self))
        )

    @property
    def value(self) -> complex:
        return self.value_ev

    @property
    def fingerprint(self) -> str:
        return self.element_fingerprint


def _vituri2024_rpa_element(
    element_kind: RPAElementKind,
    left_transition: Vituri2024DiagonalHFTransitionReceipt,
    right_transition: Vituri2024DiagonalHFTransitionReceipt,
    area: Vituri2024FiniteAreaReceipt,
    Delta1: object,
    interaction: InteractionInput,
    *,
    kinematics_provider_sha256: str,
    kinematics_source_text: str,
) -> Vituri2024RPAElementReceipt:
    left = _validated_transition(left_transition)
    right = _validated_transition(right_transition)
    clean_area = _validated_area(area)
    _require_same_source(left, right)
    _require_consistent_duplicate_orbitals(left, right)
    delta1 = _finite_real(Delta1, label="Delta1")
    left_transfer = _transfer(left)
    right_transfer = _transfer(right)

    if element_kind == "A":
        if left_transfer != right_transfer:
            raise ValueError("A element requires exact q_left=q_right")
        quartet = (left.particle, right.hole, left.hole, right.particle)
        equation = RPA_A_ELEMENT_EQUATION
        quartet_index_order = "(a,B;A,b)"
        one_body = (
            left.gap_ev
            if left.particle == right.particle and left.hole == right.hole
            else 0.0
        )
    elif element_kind == "B":
        if left_transfer != (-right_transfer[0], -right_transfer[1]):
            raise ValueError("B element requires exact q_left=-q_right")
        quartet = (left.particle, right.particle, left.hole, right.hole)
        equation = RPA_B_ELEMENT_EQUATION
        quartet_index_order = "(a,b;A,B)"
        one_body = 0.0
    else:  # private fail-closed guard
        raise ValueError("element_kind must be exactly 'A' or 'B'")

    kinematics = Vituri2024FourPointKinematicsReceipt(
        alpha=quartet[0],
        beta=quartet[1],
        gamma=quartet[2],
        delta=quartet[3],
        momentum_tolerance_inverse_angstrom=0.0,
        provider_sha256=kinematics_provider_sha256,
        derivation_source_sm_sha256=SM_TEX_SHA256,
        source_text=kinematics_source_text,
    )
    kinematics.require_conserving()
    vertex = vituri2024_antisymmetrized_projected_vertex(
        kinematics, delta1, interaction
    )
    interaction_contribution = -vertex.value / clean_area.area_angstrom_squared
    context_fingerprint = _fingerprint(
        {
            "delta1_ev": delta1,
            "interaction_receipt_fingerprint": (
                vertex.interaction_receipt_fingerprint
            ),
            "interaction_binding_fingerprint": (
                vertex.interaction_binding_fingerprint
            ),
            "area_fingerprint": clean_area.fingerprint,
        }
    )
    return Vituri2024RPAElementReceipt(
        element_kind=element_kind,
        value_ev=complex(one_body) + interaction_contribution,
        one_body_contribution_ev=one_body,
        interaction_contribution_ev=interaction_contribution,
        left_transition=left,
        right_transition=right,
        left_transition_fingerprint=left.fingerprint,
        right_transition_fingerprint=right.fingerprint,
        source_fingerprint=left.source_fingerprint,
        left_transfer_inverse_angstrom=left_transfer,
        right_transfer_inverse_angstrom=right_transfer,
        delta1_ev=delta1,
        area=clean_area,
        area_fingerprint=clean_area.fingerprint,
        interaction_receipt_fingerprint=vertex.interaction_receipt_fingerprint,
        interaction_binding_fingerprint=vertex.interaction_binding_fingerprint,
        context_fingerprint=context_fingerprint,
        kinematics=kinematics,
        kinematics_fingerprint=kinematics.fingerprint,
        vertex=vertex,
        vertex_fingerprint=vertex.fingerprint,
        equation=equation,
        quartet_index_order=quartet_index_order,
    )


def vituri2024_rpa_a_element(
    left_transition: Vituri2024DiagonalHFTransitionReceipt,
    right_transition: Vituri2024DiagonalHFTransitionReceipt,
    area: Vituri2024FiniteAreaReceipt,
    Delta1: object,
    interaction: InteractionInput,
    *,
    kinematics_provider_sha256: str,
    kinematics_source_text: str,
) -> Vituri2024RPAElementReceipt:
    """Return local ``A[aA,bB]`` with exact equal-transfer enforcement."""

    return _vituri2024_rpa_element(
        "A",
        left_transition,
        right_transition,
        area,
        Delta1,
        interaction,
        kinematics_provider_sha256=kinematics_provider_sha256,
        kinematics_source_text=kinematics_source_text,
    )


def vituri2024_rpa_b_element(
    left_transition: Vituri2024DiagonalHFTransitionReceipt,
    right_transition: Vituri2024DiagonalHFTransitionReceipt,
    area: Vituri2024FiniteAreaReceipt,
    Delta1: object,
    interaction: InteractionInput,
    *,
    kinematics_provider_sha256: str,
    kinematics_source_text: str,
) -> Vituri2024RPAElementReceipt:
    """Return local ``B[aA,bB]`` with exact opposite-transfer enforcement."""

    return _vituri2024_rpa_element(
        "B",
        left_transition,
        right_transition,
        area,
        Delta1,
        interaction,
        kinematics_provider_sha256=kinematics_provider_sha256,
        kinematics_source_text=kinematics_source_text,
    )


__all__ = [
    "DIAGONAL_HF_SOURCE_PROVIDER_STATUS",
    "FINITE_AREA_PROVIDER_STATUS",
    "RPA_A_ELEMENT_EQUATION",
    "RPA_B_ELEMENT_EQUATION",
    "RPA_ELEMENT_AUTHORITY",
    "RPA_ELEMENT_NO_GO_LIMITS",
    "RPA_ELEMENT_UNITS",
    "RPA_VERTEX_UNITS",
    "RPA_AREA_UNITS",
    "SCALAR_HESSIAN_EQUATION",
    "RPAElementKind",
    "Vituri2024DiagonalHFTransitionReceipt",
    "Vituri2024FiniteAreaReceipt",
    "Vituri2024RPAElementReceipt",
    "vituri2024_rpa_a_element",
    "vituri2024_rpa_b_element",
]
