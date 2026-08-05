"""Scoped Vituri-2024 screened kernel and same-valley band form factor.

The paper-direct authority is arXiv:2408.10309v1 ``SM.tex`` lines 58--63
for the screened Coulomb formulas and lines 76--97 for projection to the
third-lowest band.  The source does not specify the gate distance, the
numerical value/unit realization of ``e^2``, or a q=0 background convention.
Those inputs therefore require an explicit non-paper reproduction-choice
receipt.  This module implements no Hartree--Fock, intervalley-coherent raw
order parameter, or TDHF surface.

Units
-----
``q`` is a scalar magnitude in 1/Angstrom, ``d`` is in Angstrom, and ``e^2``
is in eV*Angstrom.  Consequently both ``V0`` and ``VTF`` are returned in
eV*Angstrom^2, as required by the two-dimensional momentum-space interaction
appearing with the paper's ``1/A`` normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
import operator
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .vituri2024 import (
    ACTIVE_BAND_INDEX_ZERO_BASED,
    SM_TEX_SHA256,
    VITURI2024_PARAMETERS,
    ActiveBandEigensolution,
    Vituri2024Parameters,
    state_projector,
    third_lowest_active_band,
)

InteractionAuthorityKind: TypeAlias = Literal[
    "reproduction_choice", "independent_provider_explicit"
]
Q0Evaluation: TypeAlias = Literal["reject", "analytic_kernel_limit_only"]
InteractionInput: TypeAlias = (
    "Vituri2024InteractionChoiceReceipt | Vituri2024InteractionBinding"
)

PAPER_EPSILON: Final[float] = 8.0
PAPER_Q_TF_PER_A0: Final[float] = 0.04
PAPER_Q_TF_INVERSE_ANGSTROM: Final[float] = (
    PAPER_Q_TF_PER_A0 / VITURI2024_PARAMETERS.a0
)
FORM_FACTOR_GAUGE_LABEL: Final[str] = (
    "numerical_eigh_phase_only_not_paper_gauge"
)


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


def _nonnegative_finite_real(value: object, *, label: str) -> float:
    result = _finite_real(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a lowercase SHA256 digest")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _strict_valley(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("valley must be the integer +1 or -1")
    try:
        valley = operator.index(value)
    except TypeError as exc:
        raise ValueError("valley must be the integer +1 or -1") from exc
    if valley not in (-1, 1):
        raise ValueError("valley must be the integer +1 or -1")
    return int(valley)


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Vituri2024InteractionChoiceReceipt:
    """Immutable authority receipt for inputs absent from ``SM.tex``.

    ``gate_distance_angstrom`` and ``coulomb_e2_ev_angstrom`` have no defaults:
    callers must supply them explicitly.  ``source_sha256`` is pinned to the
    authoritative v1 ``SM.tex`` artifact.  ``provider_sha256`` identifies the
    controlled reproduction choice or independent provider supplying the
    otherwise missing numerical inputs.

    Selecting ``analytic_kernel_limit_only`` at q=0 authorizes only evaluation
    of the continuous mathematical kernel limit.  It does not provide a
    neutralizing background, normal-ordering convention, or q=0 Hartree rule.
    """

    gate_distance_angstrom: float
    coulomb_e2_ev_angstrom: float
    q0_evaluation: Q0Evaluation
    provider_sha256: str
    source_sha256: str
    authority_kind: InteractionAuthorityKind
    source_text: str
    epsilon: float = field(default=PAPER_EPSILON, init=False)
    q_tf_per_a0: float = field(default=PAPER_Q_TF_PER_A0, init=False)
    q_tf_inverse_angstrom: float = field(
        default=PAPER_Q_TF_INVERSE_ANGSTROM, init=False
    )
    paper_direct_claim_allowed: bool = field(default=False, init=False)
    establishes_hf_q0_background: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gate_distance_angstrom",
            _positive_finite_real(
                self.gate_distance_angstrom, label="gate_distance_angstrom"
            ),
        )
        object.__setattr__(
            self,
            "coulomb_e2_ev_angstrom",
            _positive_finite_real(
                self.coulomb_e2_ev_angstrom,
                label="coulomb_e2_ev_angstrom",
            ),
        )
        if self.q0_evaluation not in ("reject", "analytic_kernel_limit_only"):
            raise ValueError(
                "q0_evaluation must be exactly 'reject' or "
                "'analytic_kernel_limit_only'"
            )
        _sha256(self.provider_sha256, label="provider_sha256")
        _sha256(self.source_sha256, label="source_sha256")
        if self.source_sha256 != SM_TEX_SHA256:
            raise ValueError("source_sha256 must match arXiv:2408.10309v1 SM.tex")
        if self.authority_kind not in (
            "reproduction_choice",
            "independent_provider_explicit",
        ):
            raise ValueError("invalid interaction authority_kind")
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source_text must be non-empty provenance text")
        locked = (
            self.epsilon == PAPER_EPSILON,
            self.q_tf_per_a0 == PAPER_Q_TF_PER_A0,
            self.q_tf_inverse_angstrom == PAPER_Q_TF_INVERSE_ANGSTROM,
            self.paper_direct_claim_allowed is False,
            self.establishes_hf_q0_background is False,
        )
        if not all(locked):
            raise ValueError("Vituri paper locks or authority limits were changed")

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA256 over all receipt fields."""

        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024InteractionBinding:
    """Bind one validated interaction choice without inflating its authority."""

    receipt: Vituri2024InteractionChoiceReceipt
    receipt_fingerprint: str = field(init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)
    establishes_hf_q0_background: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not Vituri2024InteractionChoiceReceipt:
            raise TypeError("binding requires a Vituri2024InteractionChoiceReceipt")
        if self.paper_direct_claim_allowed is not False:
            raise ValueError("binding cannot claim paper-direct authority")
        if self.establishes_hf_q0_background is not False:
            raise ValueError("binding cannot establish an HF q=0 background")
        object.__setattr__(self, "receipt_fingerprint", self.receipt.fingerprint)


def bind_vituri2024_interaction(
    receipt: Vituri2024InteractionChoiceReceipt,
) -> Vituri2024InteractionBinding:
    """Return a typed, authority-limited binding for ``receipt``."""

    return Vituri2024InteractionBinding(receipt=receipt)


def _resolve_receipt(interaction: InteractionInput) -> Vituri2024InteractionChoiceReceipt:
    if type(interaction) is Vituri2024InteractionChoiceReceipt:
        return interaction
    if type(interaction) is Vituri2024InteractionBinding:
        return interaction.receipt
    raise TypeError(
        "interaction must be a Vituri2024InteractionChoiceReceipt or binding"
    )


def _radial_gate_kernel(
    q_inverse_angstrom: object,
    receipt: Vituri2024InteractionChoiceReceipt,
) -> float:
    """Return stable ``r(q)=tanh(q*d)/q`` in Angstrom."""

    q = _nonnegative_finite_real(q_inverse_angstrom, label="q_inverse_angstrom")
    d = receipt.gate_distance_angstrom
    if q == 0.0:
        if receipt.q0_evaluation == "reject":
            raise ValueError(
                "q=0 evaluation rejected by the interaction reproduction choice"
            )
        return d

    qd = q * d
    if qd == 0.0:  # positive subnormal product underflow: tanh(qd)/qd -> 1
        ratio = 1.0
    elif math.isinf(qd):  # positive product overflow: tanh(qd) -> 1
        return 1.0 / q
    else:
        ratio = math.tanh(qd) / qd
    return d * ratio


def vituri2024_v0(q_inverse_angstrom: object, interaction: InteractionInput) -> float:
    """Return paper ``V0(q)`` in eV*Angstrom^2 for scalar ``q >= 0``.

    The stable radial factor is evaluated as
    ``d * tanh(q*d)/(q*d)``, with its exact q=0 mathematical limit when and
    only when the receipt selects ``analytic_kernel_limit_only``.
    """

    receipt = _resolve_receipt(interaction)
    radial = _radial_gate_kernel(q_inverse_angstrom, receipt)
    prefactor = (2.0 * math.pi / PAPER_EPSILON) * (
        receipt.coulomb_e2_ev_angstrom
    )
    value = prefactor * radial
    if not math.isfinite(value):
        raise OverflowError("V0 is outside finite float64 range")
    return value


def vituri2024_vtf(q_inverse_angstrom: object, interaction: InteractionInput) -> float:
    """Return paper ``VTF(q)`` in eV*Angstrom^2 for scalar ``q >= 0``.

    Using ``r=tanh(q*d)/q`` and ``C=2*pi*e^2/epsilon``, the paper expression
    is rearranged as ``C/(1/r + qTF)`` to avoid an overflowing ``C*r``
    intermediate.  q=0 analytic evaluation remains only a kernel limit and
    does not establish a background prescription.
    """

    receipt = _resolve_receipt(interaction)
    radial = _radial_gate_kernel(q_inverse_angstrom, receipt)
    prefactor = (2.0 * math.pi / PAPER_EPSILON) * (
        receipt.coulomb_e2_ev_angstrom
    )
    inverse_radial = 1.0 / radial
    value = prefactor / (inverse_radial + PAPER_Q_TF_INVERSE_ANGSTROM)
    if not math.isfinite(value):
        raise OverflowError("VTF is outside finite float64 range")
    return value


# Formula-shaped aliases retained for direct comparison with SM.tex notation.
V0 = vituri2024_v0
VTF = vituri2024_vtf
v0 = vituri2024_v0
vtf = vituri2024_vtf
v_tf = vituri2024_vtf


@dataclass(frozen=True, slots=True)
class Vituri2024LocalBandGapInfo:
    """Pointwise third-band evidence bound to valley and outer-layer Delta1."""

    momentum_inverse_angstrom: tuple[float, float]
    valley: int
    delta1_ev: float
    energy_ev: float
    lower_gap_ev: float
    upper_gap_ev: float
    band_index_zero_based: int = ACTIVE_BAND_INDEX_ZERO_BASED

    def __post_init__(self) -> None:
        if len(self.momentum_inverse_angstrom) != 2:
            raise ValueError("local-gap momentum must have two components")
        momentum = tuple(
            _finite_real(value, label="local-gap momentum component")
            for value in self.momentum_inverse_angstrom
        )
        object.__setattr__(self, "momentum_inverse_angstrom", momentum)
        object.__setattr__(self, "valley", _strict_valley(self.valley))
        object.__setattr__(
            self,
            "delta1_ev",
            _finite_real(self.delta1_ev, label="local-gap delta1_ev"),
        )
        object.__setattr__(
            self, "energy_ev", _finite_real(self.energy_ev, label="band energy")
        )
        object.__setattr__(
            self,
            "lower_gap_ev",
            _positive_finite_real(self.lower_gap_ev, label="lower local gap"),
        )
        object.__setattr__(
            self,
            "upper_gap_ev",
            _positive_finite_real(self.upper_gap_ev, label="upper local gap"),
        )
        if self.band_index_zero_based != ACTIVE_BAND_INDEX_ZERO_BASED:
            raise ValueError("local-gap record must describe the third-lowest band")


@dataclass(frozen=True, slots=True)
class Vituri2024StateOverlapInvariant:
    """Authority-neutral algebraic overlap for arbitrary six-spinors.

    This helper carries no band, valley, momentum, Hamiltonian, or eigensolver
    provenance.  It exists only to certify overlap phase covariance and the
    projector-trace identity; it cannot be used as a physical form-factor
    receipt by itself.
    """

    value: complex
    absolute_squared: float
    projector_trace_identity: float
    projector_trace_residual: float
    authority_scope: str = field(
        default="algebraic_overlap_no_band_valley_or_hamiltonian_authority",
        init=False,
    )
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        value = complex(self.value)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("state-overlap value must be finite")
        object.__setattr__(self, "value", value)
        absolute_squared = _nonnegative_finite_real(
            self.absolute_squared, label="state-overlap absolute_squared"
        )
        projector_trace = _nonnegative_finite_real(
            self.projector_trace_identity, label="projector_trace_identity"
        )
        residual = _nonnegative_finite_real(
            self.projector_trace_residual, label="projector_trace_residual"
        )
        object.__setattr__(self, "absolute_squared", absolute_squared)
        object.__setattr__(self, "projector_trace_identity", projector_trace)
        object.__setattr__(self, "projector_trace_residual", residual)
        expected = abs(value) ** 2
        tolerance = 256.0 * np.finfo(float).eps * max(1.0, expected)
        if abs(absolute_squared - expected) > tolerance:
            raise ValueError("absolute_squared is inconsistent with the overlap")
        if abs(projector_trace - absolute_squared) > tolerance:
            raise ValueError("projector trace identity is inconsistent")
        if residual > tolerance:
            raise ValueError("projector trace identity residual is too large")
        if self.authority_scope != (
            "algebraic_overlap_no_band_valley_or_hamiltonian_authority"
        ):
            raise ValueError("state-overlap authority scope was changed")
        if self.paper_direct_claim_allowed is not False:
            raise ValueError("algebraic state overlap is not paper-direct authority")


@dataclass(frozen=True, slots=True)
class Vituri2024DensityFormFactorReceipt:
    """Third-band ``F_tau(k_bra,k_ket)`` with pointwise gap evidence."""

    valley: int
    delta1_ev: float
    k_bra_inverse_angstrom: tuple[float, float]
    k_ket_inverse_angstrom: tuple[float, float]
    value: complex
    absolute_squared: float
    projector_trace_identity: float
    projector_trace_residual: float
    bra_local_gap: Vituri2024LocalBandGapInfo
    ket_local_gap: Vituri2024LocalBandGapInfo
    parent_model_scope: str = field(
        default="vituri2024_ham6_third_lowest_band_v1", init=False
    )
    source_sha256: str = field(default=SM_TEX_SHA256, init=False)
    gauge_label: str = field(default=FORM_FACTOR_GAUGE_LABEL, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)
    establishes_hf_q0_background: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        state_receipt = Vituri2024StateOverlapInvariant(
            value=self.value,
            absolute_squared=self.absolute_squared,
            projector_trace_identity=self.projector_trace_identity,
            projector_trace_residual=self.projector_trace_residual,
        )
        object.__setattr__(self, "valley", _strict_valley(self.valley))
        object.__setattr__(
            self,
            "delta1_ev",
            _finite_real(self.delta1_ev, label="form-factor delta1_ev"),
        )
        object.__setattr__(self, "value", state_receipt.value)
        object.__setattr__(self, "absolute_squared", state_receipt.absolute_squared)
        object.__setattr__(
            self,
            "projector_trace_identity",
            state_receipt.projector_trace_identity,
        )
        object.__setattr__(
            self, "projector_trace_residual", state_receipt.projector_trace_residual
        )
        for name in ("k_bra_inverse_angstrom", "k_ket_inverse_angstrom"):
            momentum = getattr(self, name)
            if len(momentum) != 2:
                raise ValueError(f"{name} must have two components")
            object.__setattr__(
                self,
                name,
                tuple(
                    _finite_real(component, label=f"{name} component")
                    for component in momentum
                ),
            )
        if not isinstance(self.bra_local_gap, Vituri2024LocalBandGapInfo):
            raise TypeError("bra_local_gap must be typed local-gap information")
        if not isinstance(self.ket_local_gap, Vituri2024LocalBandGapInfo):
            raise TypeError("ket_local_gap must be typed local-gap information")
        if self.bra_local_gap.momentum_inverse_angstrom != self.k_bra_inverse_angstrom:
            raise ValueError("bra local-gap momentum mismatch")
        if self.ket_local_gap.momentum_inverse_angstrom != self.k_ket_inverse_angstrom:
            raise ValueError("ket local-gap momentum mismatch")
        for label, local_gap in (
            ("bra", self.bra_local_gap),
            ("ket", self.ket_local_gap),
        ):
            if local_gap.valley != self.valley:
                raise ValueError(f"{label} local-gap valley mismatch")
            if local_gap.delta1_ev != self.delta1_ev:
                raise ValueError(f"{label} local-gap delta1 mismatch")
        if self.parent_model_scope != "vituri2024_ham6_third_lowest_band_v1":
            raise ValueError("form-factor parent-model scope was changed")
        if self.source_sha256 != SM_TEX_SHA256:
            raise ValueError("form-factor source identity was changed")
        if self.gauge_label != FORM_FACTOR_GAUGE_LABEL:
            raise ValueError("form-factor gauge label was changed")
        if self.paper_direct_claim_allowed is not False:
            raise ValueError("numerical-eigh form factor is not a paper-gauge claim")
        if self.establishes_hf_q0_background is not False:
            raise ValueError("form-factor receipt cannot establish an HF q=0 background")


def _normalized_state(state: ArrayLike) -> np.ndarray:
    raw = np.asarray(state)
    if raw.shape != (6,):
        raise ValueError("state must have shape (6,)")
    try:
        vector = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("state must be a finite complex vector") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError("state must be a finite complex vector")
    norm_squared = float(np.vdot(vector, vector).real)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("state must have nonzero finite norm")
    return vector / math.sqrt(norm_squared)


def state_overlap_invariant(
    state_bra: ArrayLike,
    state_ket: ArrayLike,
) -> Vituri2024StateOverlapInvariant:
    """Evaluate an authority-neutral overlap identity for two six-spinors.

    Independent phase changes produce the expected covariant overlap phase,
    while ``absolute_squared`` and ``projector_trace_identity`` are invariant.
    No claim is made that either input is a band state or that both belong to
    the same valley; physical provenance is added only by
    :func:`third_band_density_form_factor`.
    """

    bra = _normalized_state(state_bra)
    ket = _normalized_state(state_ket)
    value = complex(np.vdot(bra, ket))
    absolute_squared = float(abs(value) ** 2)
    projector_trace_complex = complex(
        np.trace(state_projector(bra) @ state_projector(ket))
    )
    projector_trace_identity = float(projector_trace_complex.real)
    projector_trace_residual = float(
        abs(projector_trace_complex - absolute_squared)
    )
    return Vituri2024StateOverlapInvariant(
        value=value,
        absolute_squared=absolute_squared,
        projector_trace_identity=projector_trace_identity,
        projector_trace_residual=projector_trace_residual,
    )


def _local_gap_info(
    momentum: ArrayLike,
    solution: ActiveBandEigensolution,
    *,
    valley: int,
    delta1_ev: float,
) -> Vituri2024LocalBandGapInfo:
    raw = np.asarray(momentum)
    momentum_tuple = (float(raw[0]), float(raw[1]))
    return Vituri2024LocalBandGapInfo(
        momentum_inverse_angstrom=momentum_tuple,
        valley=valley,
        delta1_ev=delta1_ev,
        energy_ev=solution.energy,
        lower_gap_ev=solution.lower_gap,
        upper_gap_ev=solution.upper_gap,
        band_index_zero_based=solution.band_index_zero_based,
    )


def third_band_density_form_factor(
    k_bra: ArrayLike,
    k_ket: ArrayLike,
    valley: int,
    Delta1: float,
    *,
    parameters: Vituri2024Parameters = VITURI2024_PARAMETERS,
) -> Vituri2024DensityFormFactorReceipt:
    """Return the same-valley third-band density form-factor receipt.

    ``F_tau(k_bra,k_ket) = u_tau(k_bra)^dagger u_tau(k_ket)`` is gauge
    covariant.  Its magnitude squared is independently represented as
    ``Tr[P_tau(k_bra) P_tau(k_ket)]``.  The eigenvectors come directly from
    ``numpy.linalg.eigh`` through :func:`third_lowest_active_band`; no paper
    gauge is imposed.
    """

    tau = _strict_valley(valley)
    delta1_ev = _finite_real(Delta1, label="Delta1")
    bra_solution = third_lowest_active_band(
        k_bra, tau, delta1_ev, parameters=parameters
    )
    ket_solution = third_lowest_active_band(
        k_ket, tau, delta1_ev, parameters=parameters
    )
    state_receipt = state_overlap_invariant(
        bra_solution.eigenvector, ket_solution.eigenvector
    )
    bra_gap = _local_gap_info(
        k_bra, bra_solution, valley=tau, delta1_ev=delta1_ev
    )
    ket_gap = _local_gap_info(
        k_ket, ket_solution, valley=tau, delta1_ev=delta1_ev
    )
    return Vituri2024DensityFormFactorReceipt(
        valley=tau,
        delta1_ev=delta1_ev,
        k_bra_inverse_angstrom=bra_gap.momentum_inverse_angstrom,
        k_ket_inverse_angstrom=ket_gap.momentum_inverse_angstrom,
        value=state_receipt.value,
        absolute_squared=state_receipt.absolute_squared,
        projector_trace_identity=state_receipt.projector_trace_identity,
        projector_trace_residual=state_receipt.projector_trace_residual,
        bra_local_gap=bra_gap,
        ket_local_gap=ket_gap,
    )


F_tau = third_band_density_form_factor


__all__ = [
    "FORM_FACTOR_GAUGE_LABEL",
    "F_tau",
    "InteractionAuthorityKind",
    "PAPER_EPSILON",
    "PAPER_Q_TF_INVERSE_ANGSTROM",
    "PAPER_Q_TF_PER_A0",
    "Q0Evaluation",
    "V0",
    "VTF",
    "Vituri2024DensityFormFactorReceipt",
    "Vituri2024InteractionBinding",
    "Vituri2024InteractionChoiceReceipt",
    "Vituri2024LocalBandGapInfo",
    "Vituri2024StateOverlapInvariant",
    "bind_vituri2024_interaction",
    "state_overlap_invariant",
    "third_band_density_form_factor",
    "v0",
    "v_tf",
    "vituri2024_v0",
    "vituri2024_vtf",
    "vtf",
]
