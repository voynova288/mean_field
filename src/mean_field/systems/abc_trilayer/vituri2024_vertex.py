"""Authority-limited Vituri-2024 projected four-point vertex.

The fixed derivation source is arXiv:2408.10309v1, archive-relative
``SM.tex``.  Lines 73--96 imply the ordered coefficient

``U(alpha,beta;gamma,delta) = VTF(|k_beta-k_gamma|)
    F(alpha,delta) F(beta,gamma)``

with the two flavor Kronecker deltas and momentum conservation understood.
The ordered coefficient excludes ``1/(2A)``; its full ordered-index sum
reconstructs the projected Hamiltonian only after that prefactor is supplied.
The full antisymmetrized ``vbar`` sum instead excludes ``1/(4A)``.  Exported
evaluators omit the momentum delta only after requiring the local residual
vector to be exactly ``(0.0, 0.0)``; a declared tolerance is diagnostic only.

This module deliberately does *not* copy printed Eq. C3 (``SM.tex`` lines
165--170) literally.  Both arXiv v1 and the published PRB C3 repeat
``F(alpha,delta) F(beta,gamma)`` in the two terms, which generically
contradicts the immediately stated ket and bra antisymmetry.  Published-PRB
evidence is pinned to PDF SHA256
``2226e17ed95bd867607787b47343fe5fc77f2c30557023e349d86e55159c0765``.
Instead this layer derives
``vbar = U(alpha,beta;gamma,delta) - U(alpha,beta;delta,gamma)`` from the
earlier projected Hamiltonian.  Its authority label is therefore
``derived_from_projected_H_not_literal_C3_internally_inconsistent``.  No
production or paper numerical parity, reciprocal-torus/carry convention,
paper phase gauge, Hartree--Fock background, occupation factor, or executable
readiness is claimed.

Kinematics ``provider_sha256`` and ``source_text`` are caller-attested
quartet/tolerance metadata and are not independently verified.  They are
distinct from the fixed derivation-source ``SM.tex`` hash.  The complex
coefficient is gauge *covariant*, not gauge invariant.  Under independent
state phases it acquires
``exp[-i(phi_alpha+phi_beta)+i(phi_gamma+phi_delta)]``.  The committed
same-valley third-band form-factor and VTF providers remain the only physical
providers used by this layer.
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
from .vituri2024_interaction import (
    InteractionInput,
    Vituri2024DensityFormFactorReceipt,
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
    Vituri2024LocalBandGapInfo,
    third_band_density_form_factor,
    vituri2024_vtf,
)

VERTEX_AUTHORITY: Final[str] = (
    "derived_from_projected_H_not_literal_C3_internally_inconsistent"
)
VERTEX_GAUGE_BEHAVIOR: Final[str] = (
    "complex_coefficient_gauge_covariant_not_invariant_no_paper_phase_gauge"
)
PUBLISHED_PRB_PDF_SHA256: Final[str] = (
    "2226e17ed95bd867607787b47343fe5fc77f2c30557023e349d86e55159c0765"
)
ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR: Final[str] = "1/(2A)"
ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR: Final[str] = "1/(4A)"
VERTEX_NORMALIZATION_IDENTITY: Final[str] = (
    "(1/(2A))*full_sum(U*O)=(1/(4A))*full_sum(vbar*O), "
    "vbar=U-U_ket_swapped and O_ket_swapped=-O"
)
ORDERED_COEFFICIENT_EXCLUSIONS: Final[tuple[str, ...]] = (
    "no_1_over_2A",
    "no_momentum_delta_because_exact_local_conservation_is_required",
    "no_occupation_factors",
)
ANTISYMMETRIZED_VERTEX_EXCLUSIONS: Final[tuple[str, ...]] = (
    "no_1_over_4A",
    "no_momentum_delta_because_exact_local_conservation_is_required",
    "no_occupation_factors",
)
KINEMATICS_AUTHORITY_SCOPE: Final[str] = (
    "exact_local_continuum_momentum_conservation_only_no_reciprocal_torus_or_carry_authority"
)
KINEMATICS_PROVIDER_METADATA_STATUS: Final[str] = (
    "caller_attested_quartet_and_tolerance_not_independently_verified"
)

SelectionRule: TypeAlias = Literal[
    "allowed_by_both_flavor_deltas",
    "zero_by_alpha_delta_flavor_delta",
    "zero_by_beta_gamma_flavor_delta",
    "zero_by_both_flavor_deltas",
]
InteractionInputKind: TypeAlias = Literal["receipt", "binding"]
PauliShortCircuitReason: TypeAlias = Literal[
    "alpha_equals_beta",
    "gamma_equals_delta",
    "both_bra_and_ket_pairs_repeated",
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


def _finite_complex(value: object, *, label: str) -> complex:
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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _strict_sign_int(value: object, *, label: str) -> int:
    if type(value) is not int or value not in (-1, 1):
        raise ValueError(f"{label} must be exactly the integer +1 or -1")
    return value


def _momentum_tuple(value: object, *, label: str) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be a two-component tuple in 1/Angstrom")
    return (
        _finite_real(value[0], label=f"{label}[0]"),
        _finite_real(value[1], label=f"{label}[1]"),
    )


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
            _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _vector_difference(
    left: tuple[float, float], right: tuple[float, float], *, label: str
) -> tuple[float, float]:
    result = (left[0] - right[0], left[1] - right[1])
    if not all(math.isfinite(component) for component in result):
        raise OverflowError(f"{label} is outside finite float64 range")
    return result


def _vector_norm(vector: tuple[float, float], *, label: str) -> float:
    result = math.hypot(*vector)
    if not math.isfinite(result):
        raise OverflowError(f"{label} is outside finite float64 range")
    return result


@dataclass(frozen=True, slots=True)
class Vituri2024Flavor:
    """Strict spin/valley flavor ``lambda=(valley, spin)``."""

    valley: int
    spin: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "valley", _strict_sign_int(self.valley, label="valley")
        )
        object.__setattr__(self, "spin", _strict_sign_int(self.spin, label="spin"))


@dataclass(frozen=True, slots=True)
class Vituri2024Orbital:
    """One projected-band orbital with local continuum momentum in 1/Angstrom."""

    flavor: Vituri2024Flavor
    momentum_inverse_angstrom: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.flavor) is not Vituri2024Flavor:
            raise TypeError("flavor must be a Vituri2024Flavor")
        object.__setattr__(
            self,
            "momentum_inverse_angstrom",
            _momentum_tuple(
                self.momentum_inverse_angstrom,
                label="momentum_inverse_angstrom",
            ),
        )


def _kinematics_payload(receipt: "Vituri2024FourPointKinematicsReceipt") -> dict[str, object]:
    return {
        "alpha": receipt.alpha,
        "beta": receipt.beta,
        "gamma": receipt.gamma,
        "delta": receipt.delta,
        "momentum_tolerance_inverse_angstrom": (
            receipt.momentum_tolerance_inverse_angstrom
        ),
        "provider_sha256": receipt.provider_sha256,
        "derivation_source_sm_sha256": receipt.derivation_source_sm_sha256,
        "source_text": receipt.source_text,
        "residual_vector_inverse_angstrom": receipt.residual_vector_inverse_angstrom,
        "residual_norm_inverse_angstrom": receipt.residual_norm_inverse_angstrom,
        "within_declared_tolerance": receipt.within_declared_tolerance,
        "authority_scope": receipt.authority_scope,
        "provider_metadata_status": receipt.provider_metadata_status,
        "reciprocal_torus_authority": receipt.reciprocal_torus_authority,
        "reciprocal_carry_authority": receipt.reciprocal_carry_authority,
        "paper_direct_claim_allowed": receipt.paper_direct_claim_allowed,
    }


@dataclass(frozen=True, slots=True)
class Vituri2024FourPointKinematicsReceipt:
    """Typed local-momentum conservation receipt for ``alpha,beta;gamma,delta``.

    ``derivation_source_sm_sha256`` is pinned to scoped v1 ``SM.tex``.
    ``provider_sha256`` and non-empty ``source_text`` are only caller-attested
    metadata for the concrete quartet and diagnostic tolerance; this module
    does not independently verify either.  Conservation is purely local:
    reciprocal-torus equivalence and reciprocal-lattice carries are outside
    this receipt's authority.
    """

    alpha: Vituri2024Orbital
    beta: Vituri2024Orbital
    gamma: Vituri2024Orbital
    delta: Vituri2024Orbital
    momentum_tolerance_inverse_angstrom: float
    provider_sha256: str
    derivation_source_sm_sha256: str
    source_text: str
    residual_vector_inverse_angstrom: tuple[float, float] = field(init=False)
    residual_norm_inverse_angstrom: float = field(init=False)
    within_declared_tolerance: bool = field(init=False)
    authority_scope: str = field(default=KINEMATICS_AUTHORITY_SCOPE, init=False)
    provider_metadata_status: str = field(
        default=KINEMATICS_PROVIDER_METADATA_STATUS, init=False
    )
    reciprocal_torus_authority: bool = field(default=False, init=False)
    reciprocal_carry_authority: bool = field(default=False, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)
    kinematics_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta"):
            if type(getattr(self, name)) is not Vituri2024Orbital:
                raise TypeError(f"{name} must be a Vituri2024Orbital")
        object.__setattr__(
            self,
            "momentum_tolerance_inverse_angstrom",
            _nonnegative_finite_real(
                self.momentum_tolerance_inverse_angstrom,
                label="momentum_tolerance_inverse_angstrom",
            ),
        )
        _sha256(self.provider_sha256, label="provider_sha256")
        _sha256(
            self.derivation_source_sm_sha256,
            label="derivation_source_sm_sha256",
        )
        if self.derivation_source_sm_sha256 != SM_TEX_SHA256:
            raise ValueError(
                "derivation_source_sm_sha256 must match arXiv:2408.10309v1 SM.tex"
            )
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source_text must be non-empty caller-attested metadata")
        if self.authority_scope != KINEMATICS_AUTHORITY_SCOPE:
            raise ValueError("kinematics authority scope was changed")
        if self.provider_metadata_status != KINEMATICS_PROVIDER_METADATA_STATUS:
            raise ValueError("kinematics provider metadata status was changed")
        if self.reciprocal_torus_authority is not False:
            raise ValueError("kinematics cannot claim reciprocal-torus authority")
        if self.reciprocal_carry_authority is not False:
            raise ValueError("kinematics cannot claim reciprocal-carry authority")
        if self.paper_direct_claim_allowed is not False:
            raise ValueError("provider kinematics is not a paper-direct numerical claim")

        momenta = tuple(
            orbital.momentum_inverse_angstrom
            for orbital in (self.alpha, self.beta, self.gamma, self.delta)
        )
        residual = (
            momenta[0][0] + momenta[1][0] - momenta[2][0] - momenta[3][0],
            momenta[0][1] + momenta[1][1] - momenta[2][1] - momenta[3][1],
        )
        if not all(math.isfinite(component) for component in residual):
            raise OverflowError("momentum-conservation residual is not finite")
        object.__setattr__(self, "residual_vector_inverse_angstrom", residual)
        residual_norm = _vector_norm(
            residual, label="momentum-conservation residual norm"
        )
        object.__setattr__(self, "residual_norm_inverse_angstrom", residual_norm)
        object.__setattr__(
            self,
            "within_declared_tolerance",
            residual_norm <= self.momentum_tolerance_inverse_angstrom,
        )
        object.__setattr__(
            self, "kinematics_fingerprint", _fingerprint(_kinematics_payload(self))
        )

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint over quartet, tolerance, and provenance."""

        return self.kinematics_fingerprint

    def require_conserving(self) -> "Vituri2024FourPointKinematicsReceipt":
        """Return this receipt or fail if it is tampered/nonconserving."""

        clean = _validated_kinematics(self)
        if clean.residual_vector_inverse_angstrom != (0.0, 0.0):
            raise ValueError(
                "four-point vertex requires exact local momentum conservation; "
                "the declared tolerance is diagnostic only"
            )
        return self

    def ket_swapped(self) -> "Vituri2024FourPointKinematicsReceipt":
        """Return the same local receipt with ``gamma`` and ``delta`` exchanged."""

        self.require_conserving()
        return Vituri2024FourPointKinematicsReceipt(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.delta,
            delta=self.gamma,
            momentum_tolerance_inverse_angstrom=(
                self.momentum_tolerance_inverse_angstrom
            ),
            provider_sha256=self.provider_sha256,
            derivation_source_sm_sha256=self.derivation_source_sm_sha256,
            source_text=self.source_text,
        )


def _validated_kinematics(
    receipt: Vituri2024FourPointKinematicsReceipt,
) -> Vituri2024FourPointKinematicsReceipt:
    if type(receipt) is not Vituri2024FourPointKinematicsReceipt:
        raise TypeError("kinematics must be a Vituri2024FourPointKinematicsReceipt")
    expected_fingerprint = _fingerprint(_kinematics_payload(receipt))
    if receipt.kinematics_fingerprint != expected_fingerprint:
        raise ValueError("kinematics fingerprint mismatch; receipt may be tampered")
    clean = Vituri2024FourPointKinematicsReceipt(
        alpha=receipt.alpha,
        beta=receipt.beta,
        gamma=receipt.gamma,
        delta=receipt.delta,
        momentum_tolerance_inverse_angstrom=(
            receipt.momentum_tolerance_inverse_angstrom
        ),
        provider_sha256=receipt.provider_sha256,
        derivation_source_sm_sha256=receipt.derivation_source_sm_sha256,
        source_text=receipt.source_text,
    )
    if clean != receipt:
        raise ValueError("kinematics receipt fields are inconsistent")
    return clean


def _validated_interaction_receipt(
    receipt: Vituri2024InteractionChoiceReceipt,
) -> Vituri2024InteractionChoiceReceipt:
    if type(receipt) is not Vituri2024InteractionChoiceReceipt:
        raise TypeError("interaction receipt has the wrong type")
    clean = Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=receipt.gate_distance_angstrom,
        coulomb_e2_ev_angstrom=receipt.coulomb_e2_ev_angstrom,
        q0_evaluation=receipt.q0_evaluation,
        provider_sha256=receipt.provider_sha256,
        source_sha256=receipt.source_sha256,
        authority_kind=receipt.authority_kind,
        source_text=receipt.source_text,
    )
    if clean != receipt:
        raise ValueError("interaction receipt fields are inconsistent or tampered")
    return clean


def _resolve_interaction(
    interaction: InteractionInput,
) -> tuple[
    Vituri2024InteractionChoiceReceipt,
    str,
    str | None,
    InteractionInputKind,
]:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        receipt = _validated_interaction_receipt(interaction)
        return receipt, receipt.fingerprint, None, "receipt"
    if type(interaction) is Vituri2024InteractionBinding:
        receipt = _validated_interaction_receipt(interaction.receipt)
        if interaction.receipt_fingerprint != receipt.fingerprint:
            raise ValueError("interaction binding fingerprint mismatch")
        if interaction.paper_direct_claim_allowed is not False:
            raise ValueError("interaction binding inflated paper authority")
        if interaction.establishes_hf_q0_background is not False:
            raise ValueError("interaction binding inflated q=0 background authority")
        return (
            receipt,
            receipt.fingerprint,
            interaction.receipt_fingerprint,
            "binding",
        )
    raise TypeError(
        "interaction must be a Vituri2024InteractionChoiceReceipt or binding"
    )


def _validated_local_gap(
    gap: Vituri2024LocalBandGapInfo,
) -> Vituri2024LocalBandGapInfo:
    if type(gap) is not Vituri2024LocalBandGapInfo:
        raise TypeError("form-factor local-gap evidence has the wrong type")
    clean = Vituri2024LocalBandGapInfo(
        momentum_inverse_angstrom=gap.momentum_inverse_angstrom,
        valley=gap.valley,
        delta1_ev=gap.delta1_ev,
        energy_ev=gap.energy_ev,
        lower_gap_ev=gap.lower_gap_ev,
        upper_gap_ev=gap.upper_gap_ev,
        band_index_zero_based=gap.band_index_zero_based,
    )
    if clean != gap:
        raise ValueError("form-factor local-gap evidence is inconsistent or tampered")
    return clean


def _validated_form_factor(
    receipt: Vituri2024DensityFormFactorReceipt,
) -> Vituri2024DensityFormFactorReceipt:
    if type(receipt) is not Vituri2024DensityFormFactorReceipt:
        raise TypeError("form factor must be a typed third-band receipt")
    clean = Vituri2024DensityFormFactorReceipt(
        valley=receipt.valley,
        delta1_ev=receipt.delta1_ev,
        k_bra_inverse_angstrom=receipt.k_bra_inverse_angstrom,
        k_ket_inverse_angstrom=receipt.k_ket_inverse_angstrom,
        value=receipt.value,
        absolute_squared=receipt.absolute_squared,
        projector_trace_identity=receipt.projector_trace_identity,
        projector_trace_residual=receipt.projector_trace_residual,
        bra_local_gap=_validated_local_gap(receipt.bra_local_gap),
        ket_local_gap=_validated_local_gap(receipt.ket_local_gap),
    )
    if clean != receipt:
        raise ValueError("form-factor receipt is inconsistent or tampered")
    return clean


def _selection_rule(kinematics: Vituri2024FourPointKinematicsReceipt) -> SelectionRule:
    alpha_delta = kinematics.alpha.flavor == kinematics.delta.flavor
    beta_gamma = kinematics.beta.flavor == kinematics.gamma.flavor
    if alpha_delta and beta_gamma:
        return "allowed_by_both_flavor_deltas"
    if not alpha_delta and not beta_gamma:
        return "zero_by_both_flavor_deltas"
    if not alpha_delta:
        return "zero_by_alpha_delta_flavor_delta"
    return "zero_by_beta_gamma_flavor_delta"


@dataclass(frozen=True, slots=True)
class Vituri2024PauliShortCircuitRecord:
    """Typed proof that antisymmetry forces an exact zero before evaluation."""

    alpha_equals_beta: bool
    gamma_equals_delta: bool
    reason: PauliShortCircuitReason = field(init=False)
    exact_zero_returned: bool = field(default=True, init=False)
    vtf_evaluated: bool = field(default=False, init=False)
    form_factors_evaluated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.alpha_equals_beta) is not bool:
            raise TypeError("alpha_equals_beta must be a strict bool")
        if type(self.gamma_equals_delta) is not bool:
            raise TypeError("gamma_equals_delta must be a strict bool")
        if not (self.alpha_equals_beta or self.gamma_equals_delta):
            raise ValueError("Pauli short circuit requires a repeated bra or ket pair")
        if self.alpha_equals_beta and self.gamma_equals_delta:
            reason: PauliShortCircuitReason = "both_bra_and_ket_pairs_repeated"
        elif self.alpha_equals_beta:
            reason = "alpha_equals_beta"
        else:
            reason = "gamma_equals_delta"
        object.__setattr__(self, "reason", reason)
        if (
            self.exact_zero_returned is not True
            or self.vtf_evaluated is not False
            or self.form_factors_evaluated is not False
        ):
            raise ValueError("Pauli short-circuit scope was changed")


def _validated_pauli_short_circuit(
    record: Vituri2024PauliShortCircuitRecord,
) -> Vituri2024PauliShortCircuitRecord:
    if type(record) is not Vituri2024PauliShortCircuitRecord:
        raise TypeError("pauli_short_circuit has the wrong type")
    clean = Vituri2024PauliShortCircuitRecord(
        alpha_equals_beta=record.alpha_equals_beta,
        gamma_equals_delta=record.gamma_equals_delta,
    )
    if clean != record:
        raise ValueError("Pauli short-circuit record is inconsistent or tampered")
    return clean


def _ordered_payload(receipt: "Vituri2024OrderedCoefficientReceipt") -> dict[str, object]:
    return {
        item.name: getattr(receipt, item.name)
        for item in fields(receipt)
        if item.name != "coefficient_fingerprint"
    }


@dataclass(frozen=True, slots=True)
class Vituri2024OrderedCoefficientReceipt:
    """Receipt for one un-antisymmetrized projected coefficient ``U``."""

    value_ev_angstrom_squared: complex
    delta1_ev: float
    interaction_receipt: Vituri2024InteractionChoiceReceipt
    interaction_receipt_fingerprint: str
    interaction_binding_fingerprint: str | None
    interaction_input_kind: InteractionInputKind
    kinematics: Vituri2024FourPointKinematicsReceipt
    kinematics_fingerprint: str
    transfer_vector_inverse_angstrom: tuple[float, float]
    transfer_norm_inverse_angstrom: float
    form_factor_alpha_delta: Vituri2024DensityFormFactorReceipt | None
    form_factor_beta_gamma: Vituri2024DensityFormFactorReceipt | None
    form_factor_alpha_delta_fingerprint: str | None
    form_factor_beta_gamma_fingerprint: str | None
    kernel_value_ev_angstrom_squared: float | None
    selection_rule: SelectionRule
    exclusions: tuple[str, ...] = field(
        default=ORDERED_COEFFICIENT_EXCLUSIONS, init=False
    )
    authority: str = field(default=VERTEX_AUTHORITY, init=False)
    gauge_behavior: str = field(default=VERTEX_GAUGE_BEHAVIOR, init=False)
    derivation_source_sm_sha256: str = field(default=SM_TEX_SHA256, init=False)
    published_prb_pdf_sha256: str = field(
        default=PUBLISHED_PRB_PDF_SHA256, init=False
    )
    omitted_full_sum_hamiltonian_area_prefactor: str = field(
        default=ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR, init=False
    )
    normalization_identity: str = field(
        default=VERTEX_NORMALIZATION_IDENTITY, init=False
    )
    includes_area_prefactor: bool = field(default=False, init=False)
    includes_momentum_delta: bool = field(default=False, init=False)
    includes_occupation_factors: bool = field(default=False, init=False)
    establishes_hf_q0_background: bool = field(default=False, init=False)
    production_numerical_parity: bool = field(default=False, init=False)
    paper_numerical_parity: bool = field(default=False, init=False)
    coefficient_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        value = _finite_complex(
            self.value_ev_angstrom_squared, label="ordered coefficient"
        )
        object.__setattr__(self, "value_ev_angstrom_squared", value)
        delta1 = _finite_real(self.delta1_ev, label="Delta1")
        object.__setattr__(self, "delta1_ev", delta1)

        interaction = _validated_interaction_receipt(self.interaction_receipt)
        object.__setattr__(self, "interaction_receipt", interaction)
        _sha256(
            self.interaction_receipt_fingerprint,
            label="interaction_receipt_fingerprint",
        )
        if self.interaction_receipt_fingerprint != interaction.fingerprint:
            raise ValueError("interaction receipt fingerprint mismatch")
        if self.interaction_input_kind not in ("receipt", "binding"):
            raise ValueError("invalid interaction_input_kind")
        if self.interaction_input_kind == "receipt":
            if self.interaction_binding_fingerprint is not None:
                raise ValueError("raw interaction receipt cannot claim a binding fingerprint")
        else:
            _sha256(
                self.interaction_binding_fingerprint,
                label="interaction_binding_fingerprint",
            )
            if self.interaction_binding_fingerprint != interaction.fingerprint:
                raise ValueError("interaction binding fingerprint mismatch")

        kinematics = _validated_kinematics(self.kinematics)
        kinematics.require_conserving()
        object.__setattr__(self, "kinematics", kinematics)
        _sha256(self.kinematics_fingerprint, label="kinematics_fingerprint")
        if self.kinematics_fingerprint != kinematics.fingerprint:
            raise ValueError("ordered coefficient kinematics fingerprint mismatch")

        expected_transfer = _vector_difference(
            kinematics.beta.momentum_inverse_angstrom,
            kinematics.gamma.momentum_inverse_angstrom,
            label="ordered transfer vector",
        )
        transfer = _momentum_tuple(
            self.transfer_vector_inverse_angstrom,
            label="transfer_vector_inverse_angstrom",
        )
        if transfer != expected_transfer:
            raise ValueError("ordered transfer vector does not match k_beta-k_gamma")
        object.__setattr__(self, "transfer_vector_inverse_angstrom", transfer)
        transfer_norm = _nonnegative_finite_real(
            self.transfer_norm_inverse_angstrom,
            label="transfer_norm_inverse_angstrom",
        )
        if transfer_norm != _vector_norm(transfer, label="ordered transfer norm"):
            raise ValueError("ordered transfer norm is inconsistent")
        object.__setattr__(self, "transfer_norm_inverse_angstrom", transfer_norm)

        expected_selection = _selection_rule(kinematics)
        if self.selection_rule != expected_selection:
            raise ValueError("ordered flavor selection rule is inconsistent")
        if expected_selection == "allowed_by_both_flavor_deltas":
            ff_ad = _validated_form_factor(self.form_factor_alpha_delta)  # type: ignore[arg-type]
            ff_bg = _validated_form_factor(self.form_factor_beta_gamma)  # type: ignore[arg-type]
            object.__setattr__(self, "form_factor_alpha_delta", ff_ad)
            object.__setattr__(self, "form_factor_beta_gamma", ff_bg)
            expected_ff_ad_fingerprint = _fingerprint(ff_ad)
            expected_ff_bg_fingerprint = _fingerprint(ff_bg)
            if self.form_factor_alpha_delta_fingerprint != expected_ff_ad_fingerprint:
                raise ValueError("alpha-delta form-factor fingerprint mismatch")
            if self.form_factor_beta_gamma_fingerprint != expected_ff_bg_fingerprint:
                raise ValueError("beta-gamma form-factor fingerprint mismatch")
            if (
                ff_ad.valley != kinematics.alpha.flavor.valley
                or ff_ad.delta1_ev != delta1
                or ff_ad.k_bra_inverse_angstrom
                != kinematics.alpha.momentum_inverse_angstrom
                or ff_ad.k_ket_inverse_angstrom
                != kinematics.delta.momentum_inverse_angstrom
            ):
                raise ValueError("alpha-delta form-factor receipt binding mismatch")
            if (
                ff_bg.valley != kinematics.beta.flavor.valley
                or ff_bg.delta1_ev != delta1
                or ff_bg.k_bra_inverse_angstrom
                != kinematics.beta.momentum_inverse_angstrom
                or ff_bg.k_ket_inverse_angstrom
                != kinematics.gamma.momentum_inverse_angstrom
            ):
                raise ValueError("beta-gamma form-factor receipt binding mismatch")
            kernel = _finite_real(
                self.kernel_value_ev_angstrom_squared,
                label="kernel_value_ev_angstrom_squared",
            )
            if kernel <= 0.0:
                raise ValueError("nonzero ordered term requires a positive VTF value")
            object.__setattr__(self, "kernel_value_ev_angstrom_squared", kernel)
            expected_value = complex(kernel * ff_ad.value * ff_bg.value)
            if value != expected_value:
                raise ValueError("ordered value is inconsistent with VTF and form factors")
        else:
            if value != 0.0j:
                raise ValueError("flavor-forbidden ordered coefficient must be exact zero")
            if self.kernel_value_ev_angstrom_squared is not None:
                raise ValueError("flavor-forbidden term must not evaluate VTF")
            if any(
                item is not None
                for item in (
                    self.form_factor_alpha_delta,
                    self.form_factor_beta_gamma,
                    self.form_factor_alpha_delta_fingerprint,
                    self.form_factor_beta_gamma_fingerprint,
                )
            ):
                raise ValueError("flavor-forbidden term must not evaluate form factors")

        locked = (
            self.exclusions == ORDERED_COEFFICIENT_EXCLUSIONS,
            self.authority == VERTEX_AUTHORITY,
            self.gauge_behavior == VERTEX_GAUGE_BEHAVIOR,
            self.derivation_source_sm_sha256 == SM_TEX_SHA256,
            self.published_prb_pdf_sha256 == PUBLISHED_PRB_PDF_SHA256,
            self.omitted_full_sum_hamiltonian_area_prefactor
            == ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR,
            self.normalization_identity == VERTEX_NORMALIZATION_IDENTITY,
            self.includes_area_prefactor is False,
            self.includes_momentum_delta is False,
            self.includes_occupation_factors is False,
            self.establishes_hf_q0_background is False,
            self.production_numerical_parity is False,
            self.paper_numerical_parity is False,
        )
        if not all(locked):
            raise ValueError("ordered coefficient scope or authority was inflated")
        object.__setattr__(
            self, "coefficient_fingerprint", _fingerprint(_ordered_payload(self))
        )

    @property
    def value(self) -> complex:
        """Complex ordered coefficient in eV*Angstrom^2."""

        return self.value_ev_angstrom_squared

    @property
    def fingerprint(self) -> str:
        return self.coefficient_fingerprint

    @property
    def form_factor_receipts(self) -> tuple[Vituri2024DensityFormFactorReceipt, ...]:
        return tuple(
            receipt
            for receipt in (
                self.form_factor_alpha_delta,
                self.form_factor_beta_gamma,
            )
            if receipt is not None
        )


def _validated_ordered(
    receipt: Vituri2024OrderedCoefficientReceipt,
) -> Vituri2024OrderedCoefficientReceipt:
    if type(receipt) is not Vituri2024OrderedCoefficientReceipt:
        raise TypeError("ordered term must be a Vituri2024OrderedCoefficientReceipt")
    if receipt.coefficient_fingerprint != _fingerprint(_ordered_payload(receipt)):
        raise ValueError("ordered coefficient fingerprint mismatch; receipt may be tampered")
    return receipt


def vituri2024_ordered_projected_coefficient(
    kinematics: Vituri2024FourPointKinematicsReceipt,
    Delta1: object,
    interaction: InteractionInput,
) -> Vituri2024OrderedCoefficientReceipt:
    """Evaluate ordered ``U(alpha,beta;gamma,delta)`` from projected ``H_C``.

    Flavor selection is applied before VTF or form-factor evaluation.  Thus a
    flavor-forbidden term is exact zero even when its irrelevant transfer is
    q=0 and the interaction receipt rejects q=0 evaluation.
    """

    clean_kinematics = _validated_kinematics(kinematics)
    clean_kinematics.require_conserving()
    delta1 = _finite_real(Delta1, label="Delta1")
    (
        interaction_receipt,
        interaction_receipt_fingerprint,
        interaction_binding_fingerprint,
        interaction_input_kind,
    ) = _resolve_interaction(interaction)
    transfer = _vector_difference(
        clean_kinematics.beta.momentum_inverse_angstrom,
        clean_kinematics.gamma.momentum_inverse_angstrom,
        label="ordered transfer vector",
    )
    transfer_norm = _vector_norm(transfer, label="ordered transfer norm")
    selection = _selection_rule(clean_kinematics)

    ff_ad: Vituri2024DensityFormFactorReceipt | None = None
    ff_bg: Vituri2024DensityFormFactorReceipt | None = None
    ff_ad_fingerprint: str | None = None
    ff_bg_fingerprint: str | None = None
    kernel: float | None = None
    value = 0.0j
    if selection == "allowed_by_both_flavor_deltas":
        kernel = vituri2024_vtf(transfer_norm, interaction_receipt)
        ff_ad = third_band_density_form_factor(
            clean_kinematics.alpha.momentum_inverse_angstrom,
            clean_kinematics.delta.momentum_inverse_angstrom,
            clean_kinematics.alpha.flavor.valley,
            delta1,
        )
        ff_bg = third_band_density_form_factor(
            clean_kinematics.beta.momentum_inverse_angstrom,
            clean_kinematics.gamma.momentum_inverse_angstrom,
            clean_kinematics.beta.flavor.valley,
            delta1,
        )
        ff_ad_fingerprint = _fingerprint(ff_ad)
        ff_bg_fingerprint = _fingerprint(ff_bg)
        value = complex(kernel * ff_ad.value * ff_bg.value)

    return Vituri2024OrderedCoefficientReceipt(
        value_ev_angstrom_squared=value,
        delta1_ev=delta1,
        interaction_receipt=interaction_receipt,
        interaction_receipt_fingerprint=interaction_receipt_fingerprint,
        interaction_binding_fingerprint=interaction_binding_fingerprint,
        interaction_input_kind=interaction_input_kind,
        kinematics=clean_kinematics,
        kinematics_fingerprint=clean_kinematics.fingerprint,
        transfer_vector_inverse_angstrom=transfer,
        transfer_norm_inverse_angstrom=transfer_norm,
        form_factor_alpha_delta=ff_ad,
        form_factor_beta_gamma=ff_bg,
        form_factor_alpha_delta_fingerprint=ff_ad_fingerprint,
        form_factor_beta_gamma_fingerprint=ff_bg_fingerprint,
        kernel_value_ev_angstrom_squared=kernel,
        selection_rule=selection,
    )


def _antisym_payload(
    receipt: "Vituri2024AntisymmetrizedVertexReceipt",
) -> dict[str, object]:
    return {
        item.name: getattr(receipt, item.name)
        for item in fields(receipt)
        if item.name != "vertex_fingerprint"
    }


@dataclass(frozen=True, slots=True)
class Vituri2024AntisymmetrizedVertexReceipt:
    """Receipt for ``vbar=U(alpha,beta;gamma,delta)-U(alpha,beta;delta,gamma)``."""

    value_ev_angstrom_squared: complex
    raw_ordered_difference_ev_angstrom_squared: complex | None
    delta1_ev: float
    interaction_receipt: Vituri2024InteractionChoiceReceipt
    interaction_receipt_fingerprint: str
    interaction_binding_fingerprint: str | None
    interaction_input_kind: InteractionInputKind
    kinematics: Vituri2024FourPointKinematicsReceipt
    kinematics_fingerprint: str
    direct_ordered: Vituri2024OrderedCoefficientReceipt | None
    exchange_ordered: Vituri2024OrderedCoefficientReceipt | None
    pauli_short_circuit: Vituri2024PauliShortCircuitRecord | None
    antisymmetrization: str = field(
        default="U(alpha,beta;gamma,delta)-U(alpha,beta;delta,gamma)", init=False
    )
    exclusions: tuple[str, ...] = field(
        default=ANTISYMMETRIZED_VERTEX_EXCLUSIONS, init=False
    )
    authority: str = field(default=VERTEX_AUTHORITY, init=False)
    gauge_behavior: str = field(default=VERTEX_GAUGE_BEHAVIOR, init=False)
    derivation_source_sm_sha256: str = field(default=SM_TEX_SHA256, init=False)
    published_prb_pdf_sha256: str = field(
        default=PUBLISHED_PRB_PDF_SHA256, init=False
    )
    omitted_full_sum_hamiltonian_area_prefactor: str = field(
        default=ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR,
        init=False,
    )
    normalization_identity: str = field(
        default=VERTEX_NORMALIZATION_IDENTITY, init=False
    )
    includes_area_prefactor: bool = field(default=False, init=False)
    includes_momentum_delta: bool = field(default=False, init=False)
    includes_occupation_factors: bool = field(default=False, init=False)
    establishes_hf_q0_background: bool = field(default=False, init=False)
    production_numerical_parity: bool = field(default=False, init=False)
    paper_numerical_parity: bool = field(default=False, init=False)
    vertex_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        value = _finite_complex(
            self.value_ev_angstrom_squared, label="antisymmetrized vertex"
        )
        raw = (
            None
            if self.raw_ordered_difference_ev_angstrom_squared is None
            else _finite_complex(
                self.raw_ordered_difference_ev_angstrom_squared,
                label="raw ordered difference",
            )
        )
        object.__setattr__(self, "value_ev_angstrom_squared", value)
        object.__setattr__(self, "raw_ordered_difference_ev_angstrom_squared", raw)
        delta1 = _finite_real(self.delta1_ev, label="Delta1")
        object.__setattr__(self, "delta1_ev", delta1)

        interaction = _validated_interaction_receipt(self.interaction_receipt)
        object.__setattr__(self, "interaction_receipt", interaction)
        _sha256(
            self.interaction_receipt_fingerprint,
            label="interaction_receipt_fingerprint",
        )
        if self.interaction_receipt_fingerprint != interaction.fingerprint:
            raise ValueError("antisymmetrized interaction receipt fingerprint mismatch")
        if self.interaction_input_kind not in ("receipt", "binding"):
            raise ValueError("invalid interaction_input_kind")
        if self.interaction_input_kind == "receipt":
            if self.interaction_binding_fingerprint is not None:
                raise ValueError("raw interaction receipt cannot claim binding fingerprint")
        else:
            _sha256(
                self.interaction_binding_fingerprint,
                label="interaction_binding_fingerprint",
            )
            if self.interaction_binding_fingerprint != interaction.fingerprint:
                raise ValueError("antisymmetrized interaction binding fingerprint mismatch")

        kinematics = _validated_kinematics(self.kinematics)
        kinematics.require_conserving()
        object.__setattr__(self, "kinematics", kinematics)
        _sha256(self.kinematics_fingerprint, label="kinematics_fingerprint")
        if self.kinematics_fingerprint != kinematics.fingerprint:
            raise ValueError("antisymmetrized kinematics fingerprint mismatch")

        repeated_bra = kinematics.alpha == kinematics.beta
        repeated_ket = kinematics.gamma == kinematics.delta
        if repeated_bra or repeated_ket:
            if self.pauli_short_circuit is None:
                raise ValueError("repeated pair requires a typed Pauli short circuit")
            pauli = _validated_pauli_short_circuit(self.pauli_short_circuit)
            object.__setattr__(self, "pauli_short_circuit", pauli)
            if (
                pauli.alpha_equals_beta is not repeated_bra
                or pauli.gamma_equals_delta is not repeated_ket
            ):
                raise ValueError("Pauli short-circuit record does not match kinematics")
            if self.direct_ordered is not None or self.exchange_ordered is not None:
                raise ValueError("Pauli short circuit forbids ordered-term receipts")
            if raw is not None:
                raise ValueError("Pauli short circuit forbids a raw ordered difference")
            if value != 0.0j:
                raise ValueError("Pauli short circuit must return exact zero")
        else:
            if self.pauli_short_circuit is not None:
                raise ValueError("nonrepeated quartet cannot claim a Pauli short circuit")
            if self.direct_ordered is None or self.exchange_ordered is None:
                raise ValueError("non-Pauli vertex requires both ordered-term receipts")
            if raw is None:
                raise ValueError("non-Pauli vertex requires a raw ordered difference")
            direct = _validated_ordered(self.direct_ordered)
            exchange = _validated_ordered(self.exchange_ordered)
            object.__setattr__(self, "direct_ordered", direct)
            object.__setattr__(self, "exchange_ordered", exchange)
            if direct.kinematics_fingerprint != kinematics.fingerprint:
                raise ValueError("direct ordered term is not bound to input kinematics")
            expected_exchange_kinematics = kinematics.ket_swapped()
            if exchange.kinematics != expected_exchange_kinematics:
                raise ValueError(
                    "exchange ordered term is not the ket-swapped coefficient"
                )
            for term in (direct, exchange):
                if term.delta1_ev != delta1:
                    raise ValueError("ordered term Delta1 mismatch")
                if term.interaction_receipt_fingerprint != interaction.fingerprint:
                    raise ValueError("ordered term interaction fingerprint mismatch")
                if (
                    term.interaction_binding_fingerprint
                    != self.interaction_binding_fingerprint
                ):
                    raise ValueError("ordered term interaction binding mismatch")
            if raw != direct.value - exchange.value:
                raise ValueError("raw antisymmetrized difference is inconsistent")
            if value != raw:
                raise ValueError("antisymmetrized value is inconsistent")

        locked = (
            self.antisymmetrization
            == "U(alpha,beta;gamma,delta)-U(alpha,beta;delta,gamma)",
            self.exclusions == ANTISYMMETRIZED_VERTEX_EXCLUSIONS,
            self.authority == VERTEX_AUTHORITY,
            self.gauge_behavior == VERTEX_GAUGE_BEHAVIOR,
            self.derivation_source_sm_sha256 == SM_TEX_SHA256,
            self.published_prb_pdf_sha256 == PUBLISHED_PRB_PDF_SHA256,
            self.omitted_full_sum_hamiltonian_area_prefactor
            == ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR,
            self.normalization_identity == VERTEX_NORMALIZATION_IDENTITY,
            self.includes_area_prefactor is False,
            self.includes_momentum_delta is False,
            self.includes_occupation_factors is False,
            self.establishes_hf_q0_background is False,
            self.production_numerical_parity is False,
            self.paper_numerical_parity is False,
        )
        if not all(locked):
            raise ValueError("antisymmetrized vertex scope or authority was inflated")
        object.__setattr__(
            self, "vertex_fingerprint", _fingerprint(_antisym_payload(self))
        )

    @property
    def value(self) -> complex:
        """Complex antisymmetrized coefficient in eV*Angstrom^2."""

        return self.value_ev_angstrom_squared

    @property
    def fingerprint(self) -> str:
        return self.vertex_fingerprint


def vituri2024_antisymmetrized_projected_vertex(
    kinematics: Vituri2024FourPointKinematicsReceipt,
    Delta1: object,
    interaction: InteractionInput,
) -> Vituri2024AntisymmetrizedVertexReceipt:
    """Return the derived antisymmetrized projected four-point vertex receipt."""

    clean_kinematics = _validated_kinematics(kinematics)
    clean_kinematics.require_conserving()
    delta1 = _finite_real(Delta1, label="Delta1")
    (
        interaction_receipt,
        interaction_receipt_fingerprint,
        interaction_binding_fingerprint,
        interaction_input_kind,
    ) = _resolve_interaction(interaction)

    repeated_bra = clean_kinematics.alpha == clean_kinematics.beta
    repeated_ket = clean_kinematics.gamma == clean_kinematics.delta
    if repeated_bra or repeated_ket:
        return Vituri2024AntisymmetrizedVertexReceipt(
            value_ev_angstrom_squared=0.0j,
            raw_ordered_difference_ev_angstrom_squared=None,
            delta1_ev=delta1,
            interaction_receipt=interaction_receipt,
            interaction_receipt_fingerprint=interaction_receipt_fingerprint,
            interaction_binding_fingerprint=interaction_binding_fingerprint,
            interaction_input_kind=interaction_input_kind,
            kinematics=clean_kinematics,
            kinematics_fingerprint=clean_kinematics.fingerprint,
            direct_ordered=None,
            exchange_ordered=None,
            pauli_short_circuit=Vituri2024PauliShortCircuitRecord(
                alpha_equals_beta=repeated_bra,
                gamma_equals_delta=repeated_ket,
            ),
        )

    direct = vituri2024_ordered_projected_coefficient(
        clean_kinematics, delta1, interaction
    )
    exchange = vituri2024_ordered_projected_coefficient(
        clean_kinematics.ket_swapped(), delta1, interaction
    )
    raw = direct.value - exchange.value
    return Vituri2024AntisymmetrizedVertexReceipt(
        value_ev_angstrom_squared=raw,
        raw_ordered_difference_ev_angstrom_squared=raw,
        delta1_ev=direct.delta1_ev,
        interaction_receipt=direct.interaction_receipt,
        interaction_receipt_fingerprint=direct.interaction_receipt_fingerprint,
        interaction_binding_fingerprint=direct.interaction_binding_fingerprint,
        interaction_input_kind=direct.interaction_input_kind,
        kinematics=clean_kinematics,
        kinematics_fingerprint=clean_kinematics.fingerprint,
        direct_ordered=direct,
        exchange_ordered=exchange,
        pauli_short_circuit=None,
    )



__all__ = [
    "ANTISYMMETRIZED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR",
    "ANTISYMMETRIZED_VERTEX_EXCLUSIONS",
    "KINEMATICS_AUTHORITY_SCOPE",
    "KINEMATICS_PROVIDER_METADATA_STATUS",
    "ORDERED_COEFFICIENT_EXCLUSIONS",
    "ORDERED_FULL_SUM_HAMILTONIAN_AREA_PREFACTOR",
    "PUBLISHED_PRB_PDF_SHA256",
    "PauliShortCircuitReason",
    "SelectionRule",
    "VERTEX_AUTHORITY",
    "VERTEX_GAUGE_BEHAVIOR",
    "VERTEX_NORMALIZATION_IDENTITY",
    "Vituri2024AntisymmetrizedVertexReceipt",
    "Vituri2024Flavor",
    "Vituri2024FourPointKinematicsReceipt",
    "Vituri2024Orbital",
    "Vituri2024OrderedCoefficientReceipt",
    "Vituri2024PauliShortCircuitRecord",
    "vituri2024_antisymmetrized_projected_vertex",
    "vituri2024_ordered_projected_coefficient",
]
