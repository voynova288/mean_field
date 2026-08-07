"""Diagnostic-only signed-q assembly for the local Vituri-2024 C9 A/B APIs.

This adapter accepts two independent, ordered transition inventories at one
nonzero local continuum ``q`` and its exact arithmetic opposite ``-q``.  It
calls :func:`vituri2024_rpa_a_element` and
:func:`vituri2024_rpa_b_element` separately for all four signed lanes and
then packages the unmodified results in the core typed signed-q classes.
There is no copying, averaging, symmetrization, Hermitization, torus
canonicalization, reciprocal carry, q=0, exact-M, or self-conjugate path.

The resulting ``static_hessian_authority='projected_signed_ab'`` is deliberate:
this module establishes only local projected A/B assembly and core structural
compatibility.  It does not establish HF stationarity, a physical mesh/area,
q=0 background or normal ordering, UV/domain convergence, CDW source
authority, a real-material response, paper parity, or production/executable
readiness.  Assembly does not invoke a TDHF/RPA eigenmode solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Final, Literal, TypeAlias

import numpy as np

from mean_field.core.hf import (
    ParticleHolePair,
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFSignedQBlocks,
    build_standard_nambu_sewing,
    build_tdhf_signed_q_matrices,
    classify_tdhf_signed_q,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
)

from .vituri2024_interaction import (
    InteractionInput,
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
from .vituri2024_rpa import (
    Vituri2024DiagonalHFTransitionReceipt,
    Vituri2024FiniteAreaReceipt,
    Vituri2024RPAElementReceipt,
    vituri2024_rpa_a_element,
    vituri2024_rpa_b_element,
)
from .vituri2024_vertex import Vituri2024Flavor, Vituri2024Orbital


VITURI2024_TDHF_ASSEMBLY_AUTHORITY: Final[str] = (
    "diagnostic_local_C9_projected_signed_ab_only"
)
VITURI2024_TDHF_RESPONSE_SCOPE: Final[str] = (
    "vituri2024_local_continuum_projected_signed_ab_diagnostic_only_v1"
)
VITURI2024_TDHF_Q_PROVENANCE: Final[str] = (
    "exact_local_continuum_q_equals_raw_equals_canonical;"
    "no_reciprocal_torus_or_carry_authority"
)
VITURI2024_TDHF_NO_GO_LIMITS: Final[tuple[str, ...]] = (
    "no_hf_stationarity_certification",
    "no_real_mesh_or_finite_area_authority",
    "no_q0_background_or_normal_ordering_authority",
    "no_uv_or_domain_convergence_authority",
    "no_cdw_source_authority",
    "no_torus_carry_exact_m_or_self_conjugate_authority",
    "no_scalar_hessian_authority",
    "no_default_tdhf_eigensolver",
    "no_production_executable_or_paper_numerical_readiness",
)

SignedLane: TypeAlias = Literal[
    "A_plus", "B_plus_minus", "A_minus", "B_minus_plus"
]


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_finite_real(value: object, *, label: str) -> float:
    result = _finite_real(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _momentum_tuple(value: object, *, label: str) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact two-tuple in 1/Angstrom")
    return (
        _finite_real(value[0], label=f"{label}[0]"),
        _finite_real(value[1], label=f"{label}[1]"),
    )


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a lowercase SHA256 digest")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
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


def _validated_orbital(orbital: Vituri2024Orbital) -> Vituri2024Orbital:
    if type(orbital) is not Vituri2024Orbital:
        raise TypeError("transition orbital must be a Vituri2024Orbital")
    if type(orbital.flavor) is not Vituri2024Flavor:
        raise TypeError("transition orbital flavor must be a Vituri2024Flavor")
    clean = Vituri2024Orbital(
        flavor=Vituri2024Flavor(
            valley=orbital.flavor.valley,
            spin=orbital.flavor.spin,
        ),
        momentum_inverse_angstrom=orbital.momentum_inverse_angstrom,
    )
    if clean != orbital:
        raise ValueError("Vituri orbital is inconsistent or tampered")
    return clean


def _validated_transition(
    transition: Vituri2024DiagonalHFTransitionReceipt,
) -> Vituri2024DiagonalHFTransitionReceipt:
    if type(transition) is not Vituri2024DiagonalHFTransitionReceipt:
        raise TypeError(
            "inventory entries must be Vituri2024DiagonalHFTransitionReceipt"
        )
    clean = Vituri2024DiagonalHFTransitionReceipt(
        particle=_validated_orbital(transition.particle),
        hole=_validated_orbital(transition.hole),
        particle_energy_ev=transition.particle_energy_ev,
        hole_energy_ev=transition.hole_energy_ev,
        source_artifact_sha256=transition.source_artifact_sha256,
        source_text=transition.source_text,
    )
    if clean != transition:
        raise ValueError("diagonal-HF transition is inconsistent or tampered")
    return clean


def _validated_area(area: Vituri2024FiniteAreaReceipt) -> Vituri2024FiniteAreaReceipt:
    if type(area) is not Vituri2024FiniteAreaReceipt:
        raise TypeError("area must be a Vituri2024FiniteAreaReceipt")
    clean = Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=area.area_angstrom_squared,
        provider_sha256=area.provider_sha256,
        source_text=area.source_text,
    )
    if clean != area:
        raise ValueError("finite-area receipt is inconsistent or tampered")
    return clean


def _validated_interaction(
    interaction: InteractionInput,
) -> tuple[
    InteractionInput,
    str,
    str | None,
]:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        receipt = interaction
        clean_receipt = Vituri2024InteractionChoiceReceipt(
            gate_distance_angstrom=receipt.gate_distance_angstrom,
            coulomb_e2_ev_angstrom=receipt.coulomb_e2_ev_angstrom,
            q0_evaluation=receipt.q0_evaluation,
            provider_sha256=receipt.provider_sha256,
            source_sha256=receipt.source_sha256,
            authority_kind=receipt.authority_kind,
            source_text=receipt.source_text,
        )
        if clean_receipt != receipt:
            raise ValueError("interaction receipt is inconsistent or tampered")
        return clean_receipt, clean_receipt.fingerprint, None
    if type(interaction) is Vituri2024InteractionBinding:
        receipt = interaction.receipt
        clean_receipt = Vituri2024InteractionChoiceReceipt(
            gate_distance_angstrom=receipt.gate_distance_angstrom,
            coulomb_e2_ev_angstrom=receipt.coulomb_e2_ev_angstrom,
            q0_evaluation=receipt.q0_evaluation,
            provider_sha256=receipt.provider_sha256,
            source_sha256=receipt.source_sha256,
            authority_kind=receipt.authority_kind,
            source_text=receipt.source_text,
        )
        clean_binding = Vituri2024InteractionBinding(receipt=clean_receipt)
        if clean_binding != interaction:
            raise ValueError("interaction binding is inconsistent or tampered")
        return clean_binding, clean_receipt.fingerprint, clean_binding.receipt_fingerprint
    raise TypeError(
        "interaction must be a Vituri2024InteractionChoiceReceipt or binding"
    )


def _transfer(
    transition: Vituri2024DiagonalHFTransitionReceipt,
) -> tuple[float, float]:
    particle = transition.particle.momentum_inverse_angstrom
    hole = transition.hole.momentum_inverse_angstrom
    result = (particle[0] - hole[0], particle[1] - hole[1])
    if not all(math.isfinite(component) for component in result):
        raise OverflowError("transition transfer is outside finite float64 range")
    return result


def _validate_global_orbitals(
    transitions: tuple[Vituri2024DiagonalHFTransitionReceipt, ...],
) -> None:
    roles: dict[Vituri2024Orbital, Literal["particle", "hole"]] = {}
    energies: dict[Vituri2024Orbital, float] = {}
    for transition in transitions:
        for orbital, role, energy in (
            (transition.particle, "particle", transition.particle_energy_ev),
            (transition.hole, "hole", transition.hole_energy_ev),
        ):
            previous_role = roles.get(orbital)
            if previous_role is not None and previous_role != role:
                raise ValueError(
                    "global fixed-occupation inconsistency: orbital appears as both "
                    "particle and hole"
                )
            previous_energy = energies.get(orbital)
            if previous_energy is not None and previous_energy != energy:
                raise ValueError("duplicate orbital has inconsistent diagonal energies")
            roles[orbital] = role
            energies[orbital] = energy


@dataclass(frozen=True, slots=True)
class Vituri2024TransitionInventory:
    """Ordered fixed-occupation transitions for one exact nonzero local q."""

    q_inverse_angstrom: tuple[float, float]
    transitions: tuple[Vituri2024DiagonalHFTransitionReceipt, ...]
    source_artifact_sha256: str = field(init=False)
    source_text: str = field(init=False)
    source_fingerprint: str = field(init=False)
    ordered_transition_fingerprints: tuple[str, ...] = field(init=False)
    reciprocal_torus_authority: bool = field(default=False, init=False)
    reciprocal_carry_authority: bool = field(default=False, init=False)
    canonicalization_authority: bool = field(default=False, init=False)
    inventory_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        q = _momentum_tuple(self.q_inverse_angstrom, label="q_inverse_angstrom")
        if q == (0.0, 0.0):
            raise ValueError("Vituri transition inventory requires nonzero local q")
        if type(self.transitions) is not tuple or not self.transitions:
            raise ValueError("Vituri transition inventory must be a nonempty tuple")
        clean = tuple(_validated_transition(item) for item in self.transitions)
        first = clean[0]
        source_record = (
            first.source_artifact_sha256,
            first.source_text,
            first.source_provider_status,
            first.source_artifact_immutable,
            first.source_fingerprint,
        )
        for transition in clean:
            if _transfer(transition) != q:
                raise ValueError("every inventory transition must have exactly q transfer")
            if (
                transition.source_artifact_sha256,
                transition.source_text,
                transition.source_provider_status,
                transition.source_artifact_immutable,
                transition.source_fingerprint,
            ) != source_record:
                raise ValueError(
                    "all inventory transitions must share one immutable source receipt "
                    "and source_fingerprint"
                )
        physical_pairs = tuple((item.particle, item.hole) for item in clean)
        ordered = tuple(item.fingerprint for item in clean)
        if len(set(ordered)) != len(ordered):
            raise ValueError("inventory contains duplicate transition fingerprints")
        if len(set(physical_pairs)) != len(physical_pairs):
            raise ValueError("inventory contains duplicate physical (particle, hole) pairs")
        _validate_global_orbitals(clean)
        object.__setattr__(self, "q_inverse_angstrom", q)
        object.__setattr__(self, "transitions", clean)
        object.__setattr__(self, "source_artifact_sha256", first.source_artifact_sha256)
        object.__setattr__(self, "source_text", first.source_text)
        object.__setattr__(self, "source_fingerprint", first.source_fingerprint)
        object.__setattr__(self, "ordered_transition_fingerprints", ordered)
        locked = (
            self.reciprocal_torus_authority is False,
            self.reciprocal_carry_authority is False,
            self.canonicalization_authority is False,
        )
        if not all(locked):
            raise ValueError("local inventory cannot claim torus/carry/canonical authority")
        object.__setattr__(
            self,
            "inventory_fingerprint",
            _fingerprint(
                {
                    "q_inverse_angstrom": q,
                    "ordered_transition_fingerprints": ordered,
                    "source_artifact_sha256": self.source_artifact_sha256,
                    "source_text": self.source_text,
                    "source_fingerprint": self.source_fingerprint,
                    "reciprocal_torus_authority": self.reciprocal_torus_authority,
                    "reciprocal_carry_authority": self.reciprocal_carry_authority,
                    "canonicalization_authority": self.canonicalization_authority,
                }
            ),
        )

    def _validate_live_state(self) -> None:
        clean = type(self)(
            q_inverse_angstrom=self.q_inverse_angstrom,
            transitions=self.transitions,
        )
        if clean != self:
            raise ValueError("transition inventory is inconsistent or tampered")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return self.inventory_fingerprint


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFAssemblyContext:
    """One independently supplied area/Delta1/interaction element context."""

    area: Vituri2024FiniteAreaReceipt
    Delta1: float
    interaction: InteractionInput
    kinematics_provider_sha256: str
    kinematics_source_text: str
    delta1_ev: float = field(init=False)
    area_fingerprint: str = field(init=False)
    interaction_receipt_fingerprint: str = field(init=False)
    interaction_binding_fingerprint: str | None = field(init=False)
    context_fingerprint: str = field(init=False)
    assembly_context_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        area = _validated_area(self.area)
        delta1 = _finite_real(self.Delta1, label="Delta1")
        interaction, receipt_fingerprint, binding_fingerprint = _validated_interaction(
            self.interaction
        )
        _sha256(self.kinematics_provider_sha256, label="kinematics_provider_sha256")
        _nonempty_text(self.kinematics_source_text, label="kinematics_source_text")
        context_fingerprint = _fingerprint(
            {
                "delta1_ev": delta1,
                "interaction_receipt_fingerprint": receipt_fingerprint,
                "interaction_binding_fingerprint": binding_fingerprint,
                "area_fingerprint": area.fingerprint,
            }
        )
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "Delta1", delta1)
        object.__setattr__(self, "delta1_ev", delta1)
        object.__setattr__(self, "interaction", interaction)
        object.__setattr__(self, "area_fingerprint", area.fingerprint)
        object.__setattr__(
            self, "interaction_receipt_fingerprint", receipt_fingerprint
        )
        object.__setattr__(
            self, "interaction_binding_fingerprint", binding_fingerprint
        )
        object.__setattr__(self, "context_fingerprint", context_fingerprint)
        object.__setattr__(
            self,
            "assembly_context_fingerprint",
            _fingerprint(
                {
                    "local_element_context_fingerprint": context_fingerprint,
                    "kinematics_provider_sha256": self.kinematics_provider_sha256,
                    "kinematics_source_text": self.kinematics_source_text,
                }
            ),
        )

    def _validate_live_state(self) -> None:
        clean = type(self)(
            area=self.area,
            Delta1=self.Delta1,
            interaction=self.interaction,
            kinematics_provider_sha256=self.kinematics_provider_sha256,
            kinematics_source_text=self.kinematics_source_text,
        )
        if clean != self:
            raise ValueError("TDHF assembly context is inconsistent or tampered")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return self.assembly_context_fingerprint


def vituri2024_tdhf_interaction_fingerprint(
    context: Vituri2024TDHFAssemblyContext,
) -> str:
    """Canonically bind the complete local interaction/assembly context."""

    if type(context) is not Vituri2024TDHFAssemblyContext:
        raise TypeError("Vituri TDHF interaction fingerprint requires the exact context type")
    context._validate_live_state()
    area = _validated_area(context.area)
    delta1 = _finite_real(context.Delta1, label="Delta1")
    _, interaction_receipt_fingerprint, interaction_binding_fingerprint = (
        _validated_interaction(context.interaction)
    )
    local_context_fingerprint = _fingerprint(
        {
            "delta1_ev": delta1,
            "interaction_receipt_fingerprint": interaction_receipt_fingerprint,
            "interaction_binding_fingerprint": interaction_binding_fingerprint,
            "area_fingerprint": area.fingerprint,
        }
    )
    local_assembly_context_fingerprint = _fingerprint(
        {
            "local_element_context_fingerprint": local_context_fingerprint,
            "kinematics_provider_sha256": _sha256(
                context.kinematics_provider_sha256,
                label="kinematics_provider_sha256",
            ),
            "kinematics_source_text": _nonempty_text(
                context.kinematics_source_text,
                label="kinematics_source_text",
            ),
        }
    )
    return _fingerprint(
        {
            "delta1_ev": delta1,
            "interaction_receipt_fingerprint": interaction_receipt_fingerprint,
            "interaction_binding_fingerprint": interaction_binding_fingerprint,
            "area_fingerprint": area.fingerprint,
            "local_element_context_fingerprint": local_context_fingerprint,
            "local_assembly_context_fingerprint": (
                local_assembly_context_fingerprint
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SignedQTransitionInventoryPair:
    """Independent ``+q``/``-q`` inventories with independently typed contexts."""

    plus_inventory: Vituri2024TransitionInventory
    minus_inventory: Vituri2024TransitionInventory
    plus_context: Vituri2024TDHFAssemblyContext
    minus_context: Vituri2024TDHFAssemblyContext
    source_fingerprint: str = field(init=False)
    context_fingerprint: str = field(init=False)
    assembly_context_fingerprint: str = field(init=False)
    pair_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name, inventory in (
            ("plus_inventory", self.plus_inventory),
            ("minus_inventory", self.minus_inventory),
        ):
            if type(inventory) is not Vituri2024TransitionInventory:
                raise TypeError(f"{name} must be a Vituri2024TransitionInventory")
            inventory._validate_live_state()
        for name, context in (
            ("plus_context", self.plus_context),
            ("minus_context", self.minus_context),
        ):
            if type(context) is not Vituri2024TDHFAssemblyContext:
                raise TypeError(f"{name} must be a Vituri2024TDHFAssemblyContext")
            context._validate_live_state()

        q_plus = self.plus_inventory.q_inverse_angstrom
        q_minus = self.minus_inventory.q_inverse_angstrom
        if q_minus != (-q_plus[0], -q_plus[1]):
            raise ValueError("signed Vituri inventories require exact q_minus=-q_plus")
        if self.plus_inventory.source_fingerprint != self.minus_inventory.source_fingerprint:
            raise ValueError("signed Vituri inventories have a source_fingerprint mismatch")
        if (
            self.plus_inventory.source_artifact_sha256,
            self.plus_inventory.source_text,
        ) != (
            self.minus_inventory.source_artifact_sha256,
            self.minus_inventory.source_text,
        ):
            raise ValueError("signed Vituri inventories must share one source receipt")
        if self.plus_context.context_fingerprint != self.minus_context.context_fingerprint:
            raise ValueError(
                "signed Vituri area/Delta1/interaction context_fingerprint mismatch"
            )
        if (
            self.plus_context.assembly_context_fingerprint
            != self.minus_context.assembly_context_fingerprint
        ):
            raise ValueError("signed Vituri full local assembly contexts differ")
        _validate_global_orbitals(
            self.plus_inventory.transitions + self.minus_inventory.transitions
        )
        object.__setattr__(
            self, "source_fingerprint", self.plus_inventory.source_fingerprint
        )
        object.__setattr__(
            self, "context_fingerprint", self.plus_context.context_fingerprint
        )
        object.__setattr__(
            self,
            "assembly_context_fingerprint",
            self.plus_context.assembly_context_fingerprint,
        )
        object.__setattr__(
            self,
            "pair_fingerprint",
            _fingerprint(
                {
                    "plus_inventory": self.plus_inventory.fingerprint,
                    "minus_inventory": self.minus_inventory.fingerprint,
                    "plus_context": self.plus_context.fingerprint,
                    "minus_context": self.minus_context.fingerprint,
                    "source_fingerprint": self.source_fingerprint,
                    "context_fingerprint": self.context_fingerprint,
                    "assembly_context_fingerprint": (
                        self.assembly_context_fingerprint
                    ),
                }
            ),
        )

    def _validate_live_state(self) -> None:
        clean = type(self)(
            plus_inventory=self.plus_inventory,
            minus_inventory=self.minus_inventory,
            plus_context=self.plus_context,
            minus_context=self.minus_context,
        )
        if clean != self:
            raise ValueError("signed-q transition inventory pair is inconsistent or tampered")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return self.pair_fingerprint


# Short public synonym; both names denote the same strict typed pair.
Vituri2024SignedQPair = Vituri2024SignedQTransitionInventoryPair


def _orbital_sort_key(orbital: Vituri2024Orbital) -> tuple[int, int, float, float]:
    return (
        orbital.flavor.valley,
        orbital.flavor.spin,
        orbital.momentum_inverse_angstrom[0],
        orbital.momentum_inverse_angstrom[1],
    )


def _core_pairs_and_orbitals(
    signed: Vituri2024SignedQTransitionInventoryPair,
) -> tuple[
    tuple[ParticleHolePair, ...],
    tuple[ParticleHolePair, ...],
    tuple[tuple[int, Vituri2024Orbital], ...],
]:
    transitions = (
        signed.plus_inventory.transitions + signed.minus_inventory.transitions
    )
    orbitals = sorted(
        {orbital for transition in transitions for orbital in (transition.particle, transition.hole)},
        key=_orbital_sort_key,
    )
    orbital_to_id = {orbital: index for index, orbital in enumerate(orbitals)}

    def convert(
        inventory: Vituri2024TransitionInventory,
    ) -> tuple[ParticleHolePair, ...]:
        return tuple(
            ParticleHolePair(
                particle=orbital_to_id[transition.particle],
                hole=orbital_to_id[transition.hole],
                particle_momentum=transition.particle.momentum_inverse_angstrom,
                hole_momentum=transition.hole.momentum_inverse_angstrom,
                particle_flavor=transition.particle.flavor,
                hole_flavor=transition.hole.flavor,
            )
            for transition in inventory.transitions
        )

    return (
        convert(signed.plus_inventory),
        convert(signed.minus_inventory),
        tuple((index, orbital) for index, orbital in enumerate(orbitals)),
    )


def _validated_element_fingerprint(element: Vituri2024RPAElementReceipt) -> str:
    if type(element) is not Vituri2024RPAElementReceipt:
        raise TypeError("assembled elements must be Vituri2024RPAElementReceipt")
    expected = _fingerprint(
        _payload_without_fingerprint(element, "element_fingerprint")
    )
    if element.fingerprint != expected:
        raise ValueError("local A/B element fingerprint mismatch; receipt may be tampered")
    return expected


def _element_assembly_context_fingerprint(
    element: Vituri2024RPAElementReceipt,
) -> str:
    return _fingerprint(
        {
            "local_element_context_fingerprint": element.context_fingerprint,
            "kinematics_provider_sha256": element.kinematics.provider_sha256,
            "kinematics_source_text": element.kinematics.source_text,
        }
    )

def _sewing_fingerprint(sewing: object) -> str:
    return _fingerprint(
        {
            "plus_to_minus": fingerprint_tdhf_matrix(sewing.plus_to_minus),  # type: ignore[attr-defined]
            "minus_to_plus": fingerprint_tdhf_matrix(sewing.minus_to_plus),  # type: ignore[attr-defined]
            "source_fingerprint": sewing.source_fingerprint,  # type: ignore[attr-defined]
            "plus_pairs_fingerprint": sewing.plus_pairs_fingerprint,  # type: ignore[attr-defined]
            "minus_pairs_fingerprint": sewing.minus_pairs_fingerprint,  # type: ignore[attr-defined]
            "construction": sewing.construction,  # type: ignore[attr-defined]
            "closure_residual": sewing.closure_residual,  # type: ignore[attr-defined]
        }
    )


def _readonly_matrix(values: list[list[complex]]) -> np.ndarray:
    result = np.array(values, dtype=np.complex128, copy=True, order="C")
    result.setflags(write=False)
    return result


def _lane_matrix(
    elements: tuple[Vituri2024RPAElementReceipt, ...],
    rows: int,
    columns: int,
) -> np.ndarray:
    return _readonly_matrix(
        [
            [elements[row * columns + column].value_ev for column in range(columns)]
            for row in range(rows)
        ]
    )


def _ordered_lane_fingerprints(
    elements: tuple[Vituri2024RPAElementReceipt, ...],
) -> tuple[str, ...]:
    return tuple(_validated_element_fingerprint(element) for element in elements)


def _validate_lane(
    *,
    lane: SignedLane,
    elements: tuple[Vituri2024RPAElementReceipt, ...],
    row_inventory: Vituri2024TransitionInventory,
    column_inventory: Vituri2024TransitionInventory,
    expected_kind: Literal["A", "B"],
    source_fingerprint: str,
    context_fingerprint: str,
    assembly_context_fingerprint: str,
) -> None:
    expected_count = len(row_inventory.transitions) * len(column_inventory.transitions)
    if len(elements) != expected_count:
        raise ValueError(f"{lane} local element count does not match ordered inventories")
    for index, element in enumerate(elements):
        _validated_element_fingerprint(element)
        row, column = divmod(index, len(column_inventory.transitions))
        if element.element_kind != expected_kind:
            raise ValueError(f"{lane} contains the wrong local element kind")
        if element.left_transition != row_inventory.transitions[row]:
            raise ValueError(f"{lane} left transition order drifted")
        if element.right_transition != column_inventory.transitions[column]:
            raise ValueError(f"{lane} right transition order drifted")
        # Both comparisons are mandatory: source equality does not imply
        # area/Delta1/interaction equality, or conversely.
        if element.source_fingerprint != source_fingerprint:
            raise ValueError(f"{lane} source_fingerprint compatibility failed")
        if element.context_fingerprint != context_fingerprint:
            raise ValueError(f"{lane} context_fingerprint compatibility failed")
        reconstructed_assembly_context = _element_assembly_context_fingerprint(element)
        if reconstructed_assembly_context != assembly_context_fingerprint:
            raise ValueError(
                f"{lane} assembly_context_fingerprint compatibility failed"
            )


@dataclass(frozen=True, slots=True)
class Vituri2024TDHFSignedQAssemblyReceipt:
    """Tamper-evident receipt for diagnostic projected signed A/B assembly."""

    signed_pair: Vituri2024SignedQTransitionInventoryPair
    blocks: TDHFSignedQBlocks
    sector: TDHFGenericSignedQSector
    orbital_id_map: tuple[tuple[int, Vituri2024Orbital], ...]
    A_plus_elements: tuple[Vituri2024RPAElementReceipt, ...]
    B_plus_minus_elements: tuple[Vituri2024RPAElementReceipt, ...]
    A_minus_elements: tuple[Vituri2024RPAElementReceipt, ...]
    B_minus_plus_elements: tuple[Vituri2024RPAElementReceipt, ...]
    A_plus_element_fingerprints: tuple[str, ...]
    B_plus_minus_element_fingerprints: tuple[str, ...]
    A_minus_element_fingerprints: tuple[str, ...]
    B_minus_plus_element_fingerprints: tuple[str, ...]
    A_plus_matrix_fingerprint: str
    B_plus_minus_matrix_fingerprint: str
    A_minus_matrix_fingerprint: str
    B_minus_plus_matrix_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    sewing_fingerprint: str
    sector_fingerprint: str
    interaction_fingerprint: str
    structure_tolerance: float
    source_fingerprint: str = field(init=False)
    context_fingerprint: str = field(init=False)
    assembly_context_fingerprint: str = field(init=False)
    authority: str = field(default=VITURI2024_TDHF_ASSEMBLY_AUTHORITY, init=False)
    response_scope: str = field(default=VITURI2024_TDHF_RESPONSE_SCOPE, init=False)
    static_hessian_authority: str = field(default="projected_signed_ab", init=False)
    full_assembly_compatibility_keys: tuple[str, str, str] = field(
        default=(
            "source_fingerprint",
            "context_fingerprint",
            "assembly_context_fingerprint",
        ),
        init=False,
    )
    post_symmetrized: bool = field(default=False, init=False)
    post_hermitized: bool = field(default=False, init=False)
    tdhf_eigensolver_called: bool = field(default=False, init=False)
    hf_stationarity_certified: bool = field(default=False, init=False)
    real_mesh_area_authority: bool = field(default=False, init=False)
    q0_background_authority: bool = field(default=False, init=False)
    uv_domain_convergence_authority: bool = field(default=False, init=False)
    cdw_source_authority: bool = field(default=False, init=False)
    paper_numerical_parity: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    executable_ready: bool = field(default=False, init=False)
    no_go_limits: tuple[str, ...] = field(default=VITURI2024_TDHF_NO_GO_LIMITS, init=False)
    assembly_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_fingerprint", self.signed_pair.source_fingerprint
        )
        object.__setattr__(
            self, "context_fingerprint", self.signed_pair.context_fingerprint
        )
        object.__setattr__(
            self,
            "assembly_context_fingerprint",
            self.signed_pair.assembly_context_fingerprint,
        )
        self._validate_live_state(check_assembly_fingerprint=False)
        object.__setattr__(self, "assembly_fingerprint", self._expected_fingerprint())

    def _expected_fingerprint(self) -> str:
        return _fingerprint(
            {
                "signed_pair": self.signed_pair.pair_fingerprint,
                "orbital_id_map": self.orbital_id_map,
                "A_plus_element_fingerprints": self.A_plus_element_fingerprints,
                "B_plus_minus_element_fingerprints": self.B_plus_minus_element_fingerprints,
                "A_minus_element_fingerprints": self.A_minus_element_fingerprints,
                "B_minus_plus_element_fingerprints": self.B_minus_plus_element_fingerprints,
                "matrix_fingerprints": (
                    self.A_plus_matrix_fingerprint,
                    self.B_plus_minus_matrix_fingerprint,
                    self.A_minus_matrix_fingerprint,
                    self.B_minus_plus_matrix_fingerprint,
                ),
                "pair_fingerprints": (
                    self.plus_pairs_fingerprint,
                    self.minus_pairs_fingerprint,
                ),
                "sewing_fingerprint": self.sewing_fingerprint,
                "sector_fingerprint": self.sector_fingerprint,
                "interaction_fingerprint": self.interaction_fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "context_fingerprint": self.context_fingerprint,
                "assembly_context_fingerprint": self.assembly_context_fingerprint,
                "structure_tolerance": self.structure_tolerance,
                "authority": self.authority,
                "response_scope": self.response_scope,
                "static_hessian_authority": self.static_hessian_authority,
                "full_assembly_compatibility_keys": self.full_assembly_compatibility_keys,
                "authority_locks": {
                    "post_symmetrized": self.post_symmetrized,
                    "post_hermitized": self.post_hermitized,
                    "tdhf_eigensolver_called": self.tdhf_eigensolver_called,
                    "hf_stationarity_certified": self.hf_stationarity_certified,
                    "real_mesh_area_authority": self.real_mesh_area_authority,
                    "q0_background_authority": self.q0_background_authority,
                    "uv_domain_convergence_authority": self.uv_domain_convergence_authority,
                    "cdw_source_authority": self.cdw_source_authority,
                    "paper_numerical_parity": self.paper_numerical_parity,
                    "production_ready": self.production_ready,
                    "executable_ready": self.executable_ready,
                },
                "no_go_limits": self.no_go_limits,
            }
        )

    def _validate_live_state(self, *, check_assembly_fingerprint: bool = True) -> None:
        self.signed_pair._validate_live_state()
        if self.blocks is not self.sector.blocks:
            raise ValueError("assembly blocks and sector blocks must be the same object")
        expected_plus_pairs, expected_minus_pairs, expected_orbitals = (
            _core_pairs_and_orbitals(self.signed_pair)
        )
        if self.blocks.plus_pairs != expected_plus_pairs:
            raise ValueError("plus core pair order/metadata drifted")
        if self.blocks.minus_pairs != expected_minus_pairs:
            raise ValueError("minus core pair order/metadata drifted")
        if self.orbital_id_map != expected_orbitals:
            raise ValueError("deterministic physical-orbital to core-ID map drifted")

        lane_specs = (
            (
                "A_plus",
                self.A_plus_elements,
                self.signed_pair.plus_inventory,
                self.signed_pair.plus_inventory,
                "A",
                self.A_plus_element_fingerprints,
                self.blocks.A_plus,
                self.A_plus_matrix_fingerprint,
            ),
            (
                "B_plus_minus",
                self.B_plus_minus_elements,
                self.signed_pair.plus_inventory,
                self.signed_pair.minus_inventory,
                "B",
                self.B_plus_minus_element_fingerprints,
                self.blocks.B_plus_minus,
                self.B_plus_minus_matrix_fingerprint,
            ),
            (
                "A_minus",
                self.A_minus_elements,
                self.signed_pair.minus_inventory,
                self.signed_pair.minus_inventory,
                "A",
                self.A_minus_element_fingerprints,
                self.blocks.A_minus,
                self.A_minus_matrix_fingerprint,
            ),
            (
                "B_minus_plus",
                self.B_minus_plus_elements,
                self.signed_pair.minus_inventory,
                self.signed_pair.plus_inventory,
                "B",
                self.B_minus_plus_element_fingerprints,
                self.blocks.B_minus_plus,
                self.B_minus_plus_matrix_fingerprint,
            ),
        )
        for lane, elements, rows, columns, kind, fingerprints, matrix, matrix_hash in lane_specs:
            _validate_lane(
                lane=lane,  # type: ignore[arg-type]
                elements=elements,
                row_inventory=rows,
                column_inventory=columns,
                expected_kind=kind,  # type: ignore[arg-type]
                source_fingerprint=self.signed_pair.source_fingerprint,
                context_fingerprint=self.signed_pair.context_fingerprint,
                assembly_context_fingerprint=(
                    self.signed_pair.assembly_context_fingerprint
                ),
            )
            if fingerprints != _ordered_lane_fingerprints(elements):
                raise ValueError(f"{lane} ordered element fingerprints drifted")
            expected_matrix = _lane_matrix(elements, len(rows.transitions), len(columns.transitions))
            if not np.array_equal(np.asarray(matrix), expected_matrix):
                raise ValueError(f"{lane} matrix no longer equals ordered local elements")
            if fingerprint_tdhf_matrix(np.asarray(matrix)) != matrix_hash:
                raise ValueError(f"{lane} matrix fingerprint mismatch")

        if fingerprint_tdhf_pairs(self.blocks.plus_pairs) != self.plus_pairs_fingerprint:
            raise ValueError("plus pair fingerprint mismatch")
        if fingerprint_tdhf_pairs(self.blocks.minus_pairs) != self.minus_pairs_fingerprint:
            raise ValueError("minus pair fingerprint mismatch")
        if _sewing_fingerprint(self.sector.sewing) != self.sewing_fingerprint:
            raise ValueError("Nambu sewing fingerprint mismatch")
        if self.source_fingerprint != self.signed_pair.source_fingerprint:
            raise ValueError("assembly source_fingerprint compatibility failed")
        if self.context_fingerprint != self.signed_pair.context_fingerprint:
            raise ValueError("assembly context_fingerprint compatibility failed")
        if (
            self.assembly_context_fingerprint
            != self.signed_pair.assembly_context_fingerprint
        ):
            raise ValueError(
                "assembly assembly_context_fingerprint compatibility failed"
            )
        if self.sector.sewing.source_fingerprint != self.source_fingerprint:
            raise ValueError("Nambu sewing source fingerprint mismatch")
        if self.sector.source_fingerprint != self.source_fingerprint:
            raise ValueError("sector source fingerprint mismatch")
        if self.sector.interaction_fingerprint != self.interaction_fingerprint:
            raise ValueError("sector interaction fingerprint mismatch")
        if self.sector.response_scope != VITURI2024_TDHF_RESPONSE_SCOPE:
            raise ValueError("sector response scope was widened")
        if self.sector.static_hessian_authority != "projected_signed_ab":
            raise ValueError("sector static-Hessian authority was inflated")
        if fingerprint_tdhf_sector(self.sector) != self.sector_fingerprint:
            raise ValueError("typed sector fingerprint mismatch")
        q = self.sector.q
        if type(q) is not TDHFGenericSignedQ:
            raise TypeError("Vituri local signed assembly requires generic q")
        if (
            q.plus_raw,
            q.plus_canonical,
            q.minus_raw,
            q.minus_canonical,
        ) != (
            self.signed_pair.plus_inventory.q_inverse_angstrom,
            self.signed_pair.plus_inventory.q_inverse_angstrom,
            self.signed_pair.minus_inventory.q_inverse_angstrom,
            self.signed_pair.minus_inventory.q_inverse_angstrom,
        ):
            raise ValueError("local q raw/canonical labels drifted")
        if VITURI2024_TDHF_Q_PROVENANCE not in q.provenance:
            raise ValueError("local q provenance lost the no-torus/no-carry scope")

        tolerance = _nonnegative_finite_real(
            self.structure_tolerance, label="structure_tolerance"
        )
        build_tdhf_signed_q_matrices(
            self.blocks,
            self.sector.sewing,
            structure_tolerance=tolerance,
            raise_on_structure_error=True,
        )
        locked = (
            self.authority == VITURI2024_TDHF_ASSEMBLY_AUTHORITY,
            self.response_scope == VITURI2024_TDHF_RESPONSE_SCOPE,
            self.static_hessian_authority == "projected_signed_ab",
            self.full_assembly_compatibility_keys
            == (
                "source_fingerprint",
                "context_fingerprint",
                "assembly_context_fingerprint",
            ),
            self.post_symmetrized is False,
            self.post_hermitized is False,
            self.tdhf_eigensolver_called is False,
            self.hf_stationarity_certified is False,
            self.real_mesh_area_authority is False,
            self.q0_background_authority is False,
            self.uv_domain_convergence_authority is False,
            self.cdw_source_authority is False,
            self.paper_numerical_parity is False,
            self.production_ready is False,
            self.executable_ready is False,
            self.no_go_limits == VITURI2024_TDHF_NO_GO_LIMITS,
        )
        if not all(locked):
            raise ValueError("Vituri TDHF diagnostic authority or NO-GO scope was inflated")
        if check_assembly_fingerprint:
            _sha256(self.assembly_fingerprint, label="assembly_fingerprint")
            if self.assembly_fingerprint != self._expected_fingerprint():
                raise ValueError("assembly fingerprint mismatch; receipt may be tampered")

    @property
    def fingerprint(self) -> str:
        self._validate_live_state()
        return self.assembly_fingerprint


def assemble_vituri2024_tdhf_signed_q(
    signed_pair: Vituri2024SignedQTransitionInventoryPair,
    *,
    structure_tolerance: float = 1.0e-10,
) -> Vituri2024TDHFSignedQAssemblyReceipt:
    """Assemble four independent local A/B lanes; do not solve TDHF modes."""

    if type(signed_pair) is not Vituri2024SignedQTransitionInventoryPair:
        raise TypeError(
            "signed_pair must be a Vituri2024SignedQTransitionInventoryPair"
        )
    signed_pair._validate_live_state()
    tolerance = _nonnegative_finite_real(
        structure_tolerance, label="structure_tolerance"
    )
    plus = signed_pair.plus_inventory.transitions
    minus = signed_pair.minus_inventory.transitions
    plus_context = signed_pair.plus_context
    minus_context = signed_pair.minus_context

    # These are intentionally four separate call sites.  No signed lane is
    # inferred from, copied from, averaged with, or repaired using another.
    A_plus_elements = tuple(
        vituri2024_rpa_a_element(
            left,
            right,
            plus_context.area,
            plus_context.Delta1,
            plus_context.interaction,
            kinematics_provider_sha256=plus_context.kinematics_provider_sha256,
            kinematics_source_text=plus_context.kinematics_source_text,
        )
        for left in plus
        for right in plus
    )
    B_plus_minus_elements = tuple(
        vituri2024_rpa_b_element(
            left,
            right,
            plus_context.area,
            plus_context.Delta1,
            plus_context.interaction,
            kinematics_provider_sha256=plus_context.kinematics_provider_sha256,
            kinematics_source_text=plus_context.kinematics_source_text,
        )
        for left in plus
        for right in minus
    )
    A_minus_elements = tuple(
        vituri2024_rpa_a_element(
            left,
            right,
            minus_context.area,
            minus_context.Delta1,
            minus_context.interaction,
            kinematics_provider_sha256=minus_context.kinematics_provider_sha256,
            kinematics_source_text=minus_context.kinematics_source_text,
        )
        for left in minus
        for right in minus
    )
    B_minus_plus_elements = tuple(
        vituri2024_rpa_b_element(
            left,
            right,
            minus_context.area,
            minus_context.Delta1,
            minus_context.interaction,
            kinematics_provider_sha256=minus_context.kinematics_provider_sha256,
            kinematics_source_text=minus_context.kinematics_source_text,
        )
        for left in minus
        for right in plus
    )

    lane_specs = (
        ("A_plus", A_plus_elements, signed_pair.plus_inventory, signed_pair.plus_inventory, "A"),
        ("B_plus_minus", B_plus_minus_elements, signed_pair.plus_inventory, signed_pair.minus_inventory, "B"),
        ("A_minus", A_minus_elements, signed_pair.minus_inventory, signed_pair.minus_inventory, "A"),
        ("B_minus_plus", B_minus_plus_elements, signed_pair.minus_inventory, signed_pair.plus_inventory, "B"),
    )
    for lane, elements, rows, columns, kind in lane_specs:
        _validate_lane(
            lane=lane,  # type: ignore[arg-type]
            elements=elements,
            row_inventory=rows,
            column_inventory=columns,
            expected_kind=kind,  # type: ignore[arg-type]
            source_fingerprint=signed_pair.source_fingerprint,
            context_fingerprint=signed_pair.context_fingerprint,
            assembly_context_fingerprint=signed_pair.assembly_context_fingerprint,
        )

    A_plus = _lane_matrix(A_plus_elements, len(plus), len(plus))
    B_plus_minus = _lane_matrix(B_plus_minus_elements, len(plus), len(minus))
    A_minus = _lane_matrix(A_minus_elements, len(minus), len(minus))
    B_minus_plus = _lane_matrix(B_minus_plus_elements, len(minus), len(plus))
    plus_pairs, minus_pairs, orbital_id_map = _core_pairs_and_orbitals(signed_pair)
    blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=A_plus,
        B_plus_minus=B_plus_minus,
        A_minus=A_minus,
        B_minus_plus=B_minus_plus,
    )
    sewing = build_standard_nambu_sewing(
        plus_pairs,
        minus_pairs,
        source_fingerprint=signed_pair.source_fingerprint,
        construction=(
            "standard_block_swap_v1;"
            f"plus_inventory={signed_pair.plus_inventory.fingerprint};"
            f"minus_inventory={signed_pair.minus_inventory.fingerprint};"
            "local_continuum_no_torus_or_carry"
        ),
    )
    q_kind = classify_tdhf_signed_q(
        plus_raw=signed_pair.plus_inventory.q_inverse_angstrom,
        minus_raw=signed_pair.minus_inventory.q_inverse_angstrom,
        plus_canonical=signed_pair.plus_inventory.q_inverse_angstrom,
        minus_canonical=signed_pair.minus_inventory.q_inverse_angstrom,
        provenance=(
            f"{VITURI2024_TDHF_Q_PROVENANCE};"
            f"plus_inventory={signed_pair.plus_inventory.fingerprint};"
            f"minus_inventory={signed_pair.minus_inventory.fingerprint}"
        ),
    )
    if type(q_kind) is not TDHFGenericSignedQ:
        raise ValueError("nonzero local continuum q unexpectedly classified self-conjugate")
    context = signed_pair.plus_context
    interaction_fingerprint = vituri2024_tdhf_interaction_fingerprint(context)
    sector = TDHFGenericSignedQSector(
        q=q_kind,
        blocks=blocks,
        sewing=sewing,
        source_fingerprint=signed_pair.source_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        response_scope=VITURI2024_TDHF_RESPONSE_SCOPE,
        static_hessian_authority="projected_signed_ab",
    )
    build_tdhf_signed_q_matrices(
        blocks,
        sewing,
        structure_tolerance=tolerance,
        raise_on_structure_error=True,
    )
    return Vituri2024TDHFSignedQAssemblyReceipt(
        signed_pair=signed_pair,
        blocks=blocks,
        sector=sector,
        orbital_id_map=orbital_id_map,
        A_plus_elements=A_plus_elements,
        B_plus_minus_elements=B_plus_minus_elements,
        A_minus_elements=A_minus_elements,
        B_minus_plus_elements=B_minus_plus_elements,
        A_plus_element_fingerprints=_ordered_lane_fingerprints(A_plus_elements),
        B_plus_minus_element_fingerprints=_ordered_lane_fingerprints(B_plus_minus_elements),
        A_minus_element_fingerprints=_ordered_lane_fingerprints(A_minus_elements),
        B_minus_plus_element_fingerprints=_ordered_lane_fingerprints(B_minus_plus_elements),
        A_plus_matrix_fingerprint=fingerprint_tdhf_matrix(A_plus),
        B_plus_minus_matrix_fingerprint=fingerprint_tdhf_matrix(B_plus_minus),
        A_minus_matrix_fingerprint=fingerprint_tdhf_matrix(A_minus),
        B_minus_plus_matrix_fingerprint=fingerprint_tdhf_matrix(B_minus_plus),
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(minus_pairs),
        sewing_fingerprint=_sewing_fingerprint(sewing),
        sector_fingerprint=fingerprint_tdhf_sector(sector),
        interaction_fingerprint=interaction_fingerprint,
        structure_tolerance=tolerance,
    )


__all__ = [
    "SignedLane",
    "VITURI2024_TDHF_ASSEMBLY_AUTHORITY",
    "VITURI2024_TDHF_NO_GO_LIMITS",
    "VITURI2024_TDHF_Q_PROVENANCE",
    "VITURI2024_TDHF_RESPONSE_SCOPE",
    "Vituri2024SignedQPair",
    "Vituri2024SignedQTransitionInventoryPair",
    "Vituri2024TDHFAssemblyContext",
    "Vituri2024TDHFSignedQAssemblyReceipt",
    "Vituri2024TransitionInventory",
    "assemble_vituri2024_tdhf_signed_q",
    "vituri2024_tdhf_interaction_fingerprint",
]
