"""Typed, system-independent signed-momentum TDHF/RPA API.

This module separates self-conjugate momentum sectors from generic non-TRIM
``{q,-q}`` orbits.  System adapters own momentum wrapping, gauge sewing,
form factors, and interaction contractions; the core owns block algebra,
structure validation, indefinite-metric mode assignment, and static/dynamic
status records.

For a generic signed orbit the block convention is

    H(q) = [[A(q),       B(q)],
            [B(-q)^*, A(-q)^*]],
    L(q) = eta H(q),  eta = diag(+I_q, -I_-q).

No matrix is Hermitized, symmetrized, averaged, or copied between signs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Hashable, Literal, Protocol, runtime_checkable

import numpy as np
from scipy import linalg as scipy_linalg

from .tdhf import ParticleHolePair, signed_q_particle_hole_assignment_residual

TDHF_TYPED_API_VERSION = "typed_signed_q_v1"

TDHFStaticHessianAuthority = Literal[
    "scalar_hessian", "projected_signed_ab", "not_established"
]


@dataclass(frozen=True)
class TDHFGenericSignedQ:
    """A momentum orbit whose canonical ``q`` and ``-q`` labels are distinct."""

    plus_raw: Hashable
    minus_raw: Hashable
    plus_canonical: Hashable
    minus_canonical: Hashable
    provenance: str
    orbit_multiplicity: int = 2

    def __post_init__(self) -> None:
        if self.plus_canonical == self.minus_canonical:
            raise ValueError("generic signed q requires distinct canonical labels")
        if not self.provenance or self.orbit_multiplicity != 2:
            raise ValueError("generic signed q requires provenance and multiplicity two")


@dataclass(frozen=True)
class TDHFSelfConjugateQ:
    """A q=0 or sewn exact-boundary momentum with one canonical torus label."""

    plus_raw: Hashable
    minus_raw: Hashable
    canonical: Hashable
    provenance: str
    orbit_multiplicity: int = 1

    def __post_init__(self) -> None:
        if not self.provenance or self.orbit_multiplicity != 1:
            raise ValueError(
                "self-conjugate q requires provenance and multiplicity one"
            )


TDHFSignedQ = TDHFGenericSignedQ | TDHFSelfConjugateQ


def classify_tdhf_signed_q(
    *,
    plus_raw: Hashable,
    minus_raw: Hashable,
    plus_canonical: Hashable,
    minus_canonical: Hashable,
    provenance: str,
) -> TDHFSignedQ:
    """Classify a signed orbit without collapsing its raw endpoint aliases."""

    if not provenance:
        raise ValueError("signed-q provenance must be nonempty")
    if plus_canonical == minus_canonical:
        return TDHFSelfConjugateQ(
            plus_raw=plus_raw,
            minus_raw=minus_raw,
            canonical=plus_canonical,
            provenance=provenance,
        )
    return TDHFGenericSignedQ(
        plus_raw=plus_raw,
        minus_raw=minus_raw,
        plus_canonical=plus_canonical,
        minus_canonical=minus_canonical,
        provenance=provenance,
    )


@dataclass(frozen=True)
class TDHFSignedQBlocks:
    """Independent q/-q A/B blocks with explicit row and column pair spaces."""

    plus_pairs: tuple[ParticleHolePair, ...]
    minus_pairs: tuple[ParticleHolePair, ...]
    A_plus: np.ndarray
    B_plus_minus: np.ndarray
    A_minus: np.ndarray
    B_minus_plus: np.ndarray


@dataclass(frozen=True)
class TDHFNambuSewing:
    """Anti-linear Nambu sewing ``w_- = S w_+^*`` supplied by an adapter."""

    plus_to_minus: np.ndarray
    minus_to_plus: np.ndarray
    source_fingerprint: str
    plus_pairs_fingerprint: str
    minus_pairs_fingerprint: str
    construction: str
    closure_residual: float


@dataclass(frozen=True)
class TDHFSignedStructureResiduals:
    A_plus_hermitian: float
    A_minus_hermitian: float
    B_partner_transpose: float
    H_plus_hermitian: float
    H_minus_hermitian: float
    L_plus_pseudo_hermitian: float
    L_minus_pseudo_hermitian: float
    signed_liouvillian_covariance: float
    reverse_signed_liouvillian_covariance: float
    sewing_closure: float
    sewing_metric_anticovariance: float
    tolerance: float

    @property
    def ok(self) -> bool:
        values = (
            self.A_plus_hermitian,
            self.A_minus_hermitian,
            self.B_partner_transpose,
            self.H_plus_hermitian,
            self.H_minus_hermitian,
            self.L_plus_pseudo_hermitian,
            self.L_minus_pseudo_hermitian,
            self.signed_liouvillian_covariance,
            self.reverse_signed_liouvillian_covariance,
            self.sewing_closure,
            self.sewing_metric_anticovariance,
        )
        return bool(all(np.isfinite(value) and value <= self.tolerance for value in values))


@dataclass(frozen=True)
class TDHFSignedQMatrices:
    H_plus: np.ndarray
    H_minus: np.ndarray
    L_plus: np.ndarray
    L_minus: np.ndarray
    eta_plus: np.ndarray
    eta_minus: np.ndarray
    structure: TDHFSignedStructureResiduals


@dataclass(frozen=True)
class TDHFGenericSignedQSector:
    q: TDHFGenericSignedQ
    blocks: TDHFSignedQBlocks
    sewing: TDHFNambuSewing
    source_fingerprint: str
    interaction_fingerprint: str
    response_scope: str
    static_hessian_authority: TDHFStaticHessianAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.q, TDHFGenericSignedQ):
            raise TypeError("generic TDHF sector requires TDHFGenericSignedQ")
        if not self.source_fingerprint or not self.interaction_fingerprint:
            raise ValueError("typed TDHF sector fingerprints must be nonempty")
        if not self.response_scope:
            raise ValueError("typed TDHF response_scope must be nonempty")


@dataclass(frozen=True)
class TDHFSelfConjugateQStructureResiduals:
    A_hermitian: float
    B_symmetric: float
    H_hermitian: float
    L_pseudo_hermitian: float
    tolerance: float

    @property
    def ok(self) -> bool:
        values = (
            self.A_hermitian,
            self.B_symmetric,
            self.H_hermitian,
            self.L_pseudo_hermitian,
        )
        return bool(all(np.isfinite(value) and value <= self.tolerance for value in values))


@dataclass(frozen=True)
class TDHFSelfConjugateQMatrices:
    H: np.ndarray
    L: np.ndarray
    eta: np.ndarray
    structure: TDHFSelfConjugateQStructureResiduals
    raw_signed_diagnostic: TDHFSignedQMatrices | None


@dataclass(frozen=True)
class TDHFSelfConjugateQSector:
    q: TDHFSelfConjugateQ
    canonical_pairs: tuple[ParticleHolePair, ...]
    A: np.ndarray
    B: np.ndarray
    source_fingerprint: str
    interaction_fingerprint: str
    response_scope: str
    static_hessian_authority: TDHFStaticHessianAuthority
    canonical_sewing_provenance: str
    raw_signed_diagnostic_blocks: TDHFSignedQBlocks | None = None
    raw_signed_diagnostic_sewing: TDHFNambuSewing | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.q, TDHFSelfConjugateQ):
            raise TypeError(
                "self-conjugate TDHF sector requires TDHFSelfConjugateQ"
            )
        if not self.source_fingerprint or not self.interaction_fingerprint:
            raise ValueError("typed TDHF sector fingerprints must be nonempty")
        if not self.response_scope or not self.canonical_sewing_provenance:
            raise ValueError(
                "self-conjugate TDHF requires response and sewing provenance"
            )


TDHFTypedSector = TDHFGenericSignedQSector | TDHFSelfConjugateQSector


@dataclass(frozen=True)
class TDHFWangModeAssignment:
    """Raw eigensystem plus Wang norm/sign momentum assignment."""

    raw_eigenvalues: np.ndarray
    raw_metric_norms: np.ndarray
    raw_residuals: np.ndarray
    plus_raw_indices: np.ndarray
    minus_raw_indices: np.ndarray
    plus_energies: np.ndarray
    minus_energies: np.ndarray
    plus_vectors: np.ndarray
    minus_vectors: np.ndarray
    plus_residuals: np.ndarray
    minus_residuals: np.ndarray
    metric_gram_residual: float
    complex_indices: np.ndarray
    null_metric_indices: np.ndarray
    zero_eigenvalue_indices: np.ndarray


@dataclass(frozen=True)
class TDHFSelfConjugateModeAssignment:
    raw_eigenvalues: np.ndarray
    raw_metric_norms: np.ndarray
    raw_residuals: np.ndarray
    positive_metric_indices: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray
    metric_gram_residual: float
    complex_indices: np.ndarray
    null_metric_indices: np.ndarray
    zero_eigenvalue_indices: np.ndarray


@dataclass(frozen=True)
class TDHFStaticStatus:
    kind: Literal[
        "positive_definite",
        "positive_semidefinite",
        "indefinite",
        "invalid",
        "not_established",
    ]
    eigenvalues: np.ndarray
    negative_count: int
    zero_count: int
    positive_count: int
    min_eigenvalue: float
    hermitian_residual: float
    tolerance: float


@dataclass(frozen=True)
class TDHFDynamicStatus:
    kind: Literal["real", "complex", "invalid"]
    complex_count: int
    max_abs_imag: float
    signed_pairing_residual: float
    max_eigensolver_residual_plus: float
    max_eigensolver_residual_minus: float
    selected_mode_residual: float
    metric_gram_residual: float
    imag_tolerance: float
    pairing_tolerance: float
    eigensolver_tolerance: float
    degeneracy_tolerance: float
    structure_tolerance: float


@dataclass(frozen=True)
class TDHFZeroModeStatus:
    origin: Literal[
        "none",
        "ordinary_dynamic_zero",
        "ward_static_null",
        "uncertified_static_null",
    ]
    static_zero_count: int
    dynamic_zero_count: int
    ward_passed: bool


@dataclass(frozen=True)
class TDHFWardCertificate:
    passed: bool
    generator_label: str
    generator_provenance: str
    source_fingerprint: str
    interaction_fingerprint: str
    sector_fingerprint: str
    response_scope: str
    generator_fingerprint: str
    generator_vector: np.ndarray
    hessian_fingerprint: str
    liouvillian_fingerprint: str
    static_hessian_authority: TDHFStaticHessianAuthority
    static_null_tolerance: float
    source_stationarity_residual: float
    source_stationarity_tolerance: float
    hessian_action_residual: float
    liouvillian_action_residual: float
    action_tolerance: float
    static_null_overlap: float
    overlap_tolerance: float
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TDHFTypedAnalysis:
    sector: TDHFTypedSector
    matrices: TDHFSignedQMatrices | TDHFSelfConjugateQMatrices
    assignment: TDHFWangModeAssignment | TDHFSelfConjugateModeAssignment
    static: TDHFStaticStatus
    dynamic: TDHFDynamicStatus
    zero_mode: TDHFZeroModeStatus
    ward: TDHFWardCertificate | None


@runtime_checkable
class TDHFSectorProviderProtocol(Protocol):
    """System adapter consumed by :func:`mean_field.api.run_tdhf`."""

    def build_tdhf_sector(self, config: object, **kwargs: Any) -> TDHFTypedSector:
        ...


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array), initial=0.0))


def _validated_tolerance(name: str, value: float) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def fingerprint_tdhf_pairs(pairs: tuple[ParticleHolePair, ...]) -> str:
    """Deterministically bind pair order and adapter-owned metadata."""

    records = [
        {
            "particle": pair.particle,
            "hole": pair.hole,
            "particle_momentum": repr(pair.particle_momentum),
            "hole_momentum": repr(pair.hole_momentum),
            "particle_flavor": repr(pair.particle_flavor),
            "hole_flavor": repr(pair.hole_flavor),
        }
        for pair in pairs
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fingerprint_tdhf_matrix(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fingerprint_tdhf_sector(sector: TDHFTypedSector) -> str:
    """Bind q identity, ordered pairs, sewing, source, interaction, and scope."""

    if isinstance(sector, TDHFGenericSignedQSector):
        record = {
            "kind": "generic",
            "q": repr(sector.q),
            "plus_pairs": fingerprint_tdhf_pairs(sector.blocks.plus_pairs),
            "minus_pairs": fingerprint_tdhf_pairs(sector.blocks.minus_pairs),
            "sewing_plus": fingerprint_tdhf_matrix(sector.sewing.plus_to_minus),
            "sewing_minus": fingerprint_tdhf_matrix(sector.sewing.minus_to_plus),
            "sewing_construction": sector.sewing.construction,
        }
    else:
        record = {
            "kind": "self_conjugate",
            "q": repr(sector.q),
            "canonical_pairs": fingerprint_tdhf_pairs(sector.canonical_pairs),
            "canonical_A": fingerprint_tdhf_matrix(sector.A),
            "canonical_B": fingerprint_tdhf_matrix(sector.B),
            "canonical_sewing_provenance": sector.canonical_sewing_provenance,
        }
    record.update(
        {
            "source_fingerprint": sector.source_fingerprint,
            "interaction_fingerprint": sector.interaction_fingerprint,
            "response_scope": sector.response_scope,
            "static_hessian_authority": sector.static_hessian_authority,
        }
    )
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_blocks(blocks: TDHFSignedQBlocks) -> tuple[int, int]:
    n_plus = len(blocks.plus_pairs)
    n_minus = len(blocks.minus_pairs)
    if n_plus == 0 or n_minus == 0:
        raise ValueError("typed signed-q TDHF requires nonempty q and -q pair spaces")
    expected = {
        "A_plus": (n_plus, n_plus),
        "B_plus_minus": (n_plus, n_minus),
        "A_minus": (n_minus, n_minus),
        "B_minus_plus": (n_minus, n_plus),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(blocks, name))
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not bool(np.all(np.isfinite(value.real)) and np.all(np.isfinite(value.imag))):
            raise ValueError(f"{name} contains nonfinite values")
    return n_plus, n_minus


def build_standard_nambu_sewing(
    plus_pairs: tuple[ParticleHolePair, ...],
    minus_pairs: tuple[ParticleHolePair, ...],
    *,
    source_fingerprint: str,
    construction: str = "standard_block_swap_v1",
) -> TDHFNambuSewing:
    """Build the standard block swap and bind the exact pair inventories."""

    n_plus = len(plus_pairs)
    n_minus = len(minus_pairs)
    if n_plus <= 0 or n_minus <= 0:
        raise ValueError("Nambu sewing pair spaces must be nonempty")
    total = n_plus + n_minus
    plus_to_minus = np.zeros((total, total), dtype=np.complex128)
    plus_to_minus[:n_minus, n_plus:] = np.eye(n_minus)
    plus_to_minus[n_minus:, :n_plus] = np.eye(n_plus)
    minus_to_plus = np.conj(plus_to_minus.T)
    closure = _max_abs(minus_to_plus @ np.conj(plus_to_minus) - np.eye(total))
    return TDHFNambuSewing(
        plus_to_minus=plus_to_minus,
        minus_to_plus=minus_to_plus,
        source_fingerprint=source_fingerprint,
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(minus_pairs),
        construction=construction,
        closure_residual=closure,
    )


def build_tdhf_signed_q_matrices(
    blocks: TDHFSignedQBlocks,
    sewing: TDHFNambuSewing,
    *,
    structure_tolerance: float = 1.0e-10,
    raise_on_structure_error: bool = False,
) -> TDHFSignedQMatrices:
    """Build both signed Hessians/Liouvillians without post-hoc repair."""

    resolved_tolerance = _validated_tolerance(
        "structure_tolerance", structure_tolerance
    )
    n_plus, n_minus = _validate_blocks(blocks)
    ap = np.asarray(blocks.A_plus, dtype=np.complex128)
    bp = np.asarray(blocks.B_plus_minus, dtype=np.complex128)
    am = np.asarray(blocks.A_minus, dtype=np.complex128)
    bm = np.asarray(blocks.B_minus_plus, dtype=np.complex128)
    hp = np.block([[ap, bp], [np.conj(bm), np.conj(am)]])
    hm = np.block([[am, bm], [np.conj(bp), np.conj(ap)]])
    eta_plus = np.concatenate([np.ones(n_plus), -np.ones(n_minus)])
    eta_minus = np.concatenate([np.ones(n_minus), -np.ones(n_plus)])
    lp = eta_plus[:, None] * hp
    lm = eta_minus[:, None] * hm
    total = n_plus + n_minus
    spm = np.asarray(sewing.plus_to_minus, dtype=np.complex128)
    smp = np.asarray(sewing.minus_to_plus, dtype=np.complex128)
    if spm.shape != (total, total) or smp.shape != (total, total):
        raise ValueError("Nambu sewing matrices have the wrong signed-sector dimension")
    sewing_closure = max(
        _max_abs(smp @ np.conj(spm) - np.eye(total)),
        _max_abs(spm @ np.conj(smp) - np.eye(total)),
        float(sewing.closure_residual),
    )
    residuals = TDHFSignedStructureResiduals(
        A_plus_hermitian=_max_abs(ap - np.conj(ap.T)),
        A_minus_hermitian=_max_abs(am - np.conj(am.T)),
        B_partner_transpose=_max_abs(bp - bm.T),
        H_plus_hermitian=_max_abs(hp - np.conj(hp.T)),
        H_minus_hermitian=_max_abs(hm - np.conj(hm.T)),
        L_plus_pseudo_hermitian=_max_abs(np.conj(lp.T) * eta_plus[None, :] - eta_plus[:, None] * lp),
        L_minus_pseudo_hermitian=_max_abs(np.conj(lm.T) * eta_minus[None, :] - eta_minus[:, None] * lm),
        signed_liouvillian_covariance=_max_abs(lm @ spm + spm @ np.conj(lp)),
        reverse_signed_liouvillian_covariance=_max_abs(
            lp @ smp + smp @ np.conj(lm)
        ),
        sewing_closure=sewing_closure,
        sewing_metric_anticovariance=_max_abs(
            np.conj(spm.T) @ (eta_minus[:, None] * spm)
            + np.diag(eta_plus)
        ),
        tolerance=resolved_tolerance,
    )
    if raise_on_structure_error and not residuals.ok:
        raise ValueError(f"typed signed-q TDHF structure gate failed: {residuals}")
    return TDHFSignedQMatrices(
        H_plus=hp,
        H_minus=hm,
        L_plus=lp,
        L_minus=lm,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        structure=residuals,
    )


def build_tdhf_self_conjugate_matrices(
    sector: TDHFSelfConjugateQSector,
    *,
    structure_tolerance: float = 1.0e-10,
    raise_on_structure_error: bool = False,
) -> TDHFSelfConjugateQMatrices:
    """Build the canonical TRIM/q=0 Hessian and Liouvillian.

    Raw ``+M/-M`` diagnostics, when present, are not used to fabricate the
    canonical scalar block.
    """

    tolerance = _validated_tolerance("structure_tolerance", structure_tolerance)
    n_pairs = len(sector.canonical_pairs)
    if n_pairs == 0:
        raise ValueError("self-conjugate TDHF requires a nonempty canonical pair space")
    a = np.asarray(sector.A, dtype=np.complex128)
    b = np.asarray(sector.B, dtype=np.complex128)
    if a.shape != (n_pairs, n_pairs) or b.shape != (n_pairs, n_pairs):
        raise ValueError("self-conjugate A/B shapes must match canonical_pairs")
    if not bool(
        np.all(np.isfinite(a.real))
        and np.all(np.isfinite(a.imag))
        and np.all(np.isfinite(b.real))
        and np.all(np.isfinite(b.imag))
    ):
        raise ValueError("self-conjugate A/B blocks contain nonfinite values")
    h = np.block([[a, b], [np.conj(b), np.conj(a)]])
    eta = np.concatenate([np.ones(n_pairs), -np.ones(n_pairs)])
    l = eta[:, None] * h
    structure = TDHFSelfConjugateQStructureResiduals(
        A_hermitian=_max_abs(a - np.conj(a.T)),
        B_symmetric=_max_abs(b - b.T),
        H_hermitian=_max_abs(h - np.conj(h.T)),
        L_pseudo_hermitian=_max_abs(
            np.conj(l.T) * eta[None, :] - eta[:, None] * l
        ),
        tolerance=tolerance,
    )
    if raise_on_structure_error and not structure.ok:
        raise ValueError(
            f"self-conjugate TDHF structure gate failed: {structure}"
        )
    diagnostic_blocks = sector.raw_signed_diagnostic_blocks
    diagnostic_sewing = sector.raw_signed_diagnostic_sewing
    if (diagnostic_blocks is None) != (diagnostic_sewing is None):
        raise ValueError(
            "self-conjugate raw signed diagnostics require both blocks and sewing"
        )
    raw_diagnostic = None
    if diagnostic_blocks is not None and diagnostic_sewing is not None:
        if diagnostic_sewing.source_fingerprint != sector.source_fingerprint:
            raise ValueError("raw signed diagnostic sewing source mismatch")
        if diagnostic_sewing.plus_pairs_fingerprint != fingerprint_tdhf_pairs(
            diagnostic_blocks.plus_pairs
        ) or diagnostic_sewing.minus_pairs_fingerprint != fingerprint_tdhf_pairs(
            diagnostic_blocks.minus_pairs
        ):
            raise ValueError("raw signed diagnostic pair fingerprints mismatch")
        raw_diagnostic = build_tdhf_signed_q_matrices(
            diagnostic_blocks,
            diagnostic_sewing,
            structure_tolerance=structure_tolerance,
        )
    return TDHFSelfConjugateQMatrices(
        H=h,
        L=l,
        eta=eta,
        structure=structure,
        raw_signed_diagnostic=raw_diagnostic,
    )


def _metric_normalized_real_modes(
    liouvillian: np.ndarray,
    metric: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
    *,
    imag_tol: float,
    norm_tol: float,
    zero_tol: float,
    degeneracy_tol: float,
) -> tuple[list[tuple[int, complex, float, np.ndarray, float]], float]:
    """Resolve real degenerate eigenspaces into an eta-orthonormal basis."""

    candidate = np.nonzero(
        (np.abs(values.imag) <= imag_tol) & (np.abs(values) > zero_tol)
    )[0]
    if not candidate.size:
        return [], 0.0
    order = candidate[np.lexsort((values[candidate].imag, values[candidate].real))]
    groups: list[list[int]] = [[int(order[0])]]
    for index in order[1:]:
        if abs(values[index] - values[groups[-1][-1]]) <= degeneracy_tol:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    modes: list[tuple[int, complex, float, np.ndarray, float]] = []
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=np.int64)
        subspace = vectors[:, group]
        gram = np.conj(subspace.T) @ (metric[:, None] * subspace)
        gram = 0.5 * (gram + np.conj(gram.T))
        metric_eigenvalues, metric_vectors = scipy_linalg.eigh(gram)
        for metric_index in np.argsort(metric_eigenvalues)[::-1]:
            metric_eigenvalue = float(np.real(metric_eigenvalues[metric_index]))
            if abs(metric_eigenvalue) <= norm_tol:
                continue
            vector = subspace @ metric_vectors[:, metric_index]
            vector /= np.sqrt(abs(metric_eigenvalue))
            sign = 1.0 if metric_eigenvalue > 0.0 else -1.0
            # The eta-Rayleigh quotient remains valid for either metric sign.
            eigenvalue = complex(
                np.vdot(vector, metric * (liouvillian @ vector)) / sign
            )
            residual = float(
                np.linalg.norm(liouvillian @ vector - eigenvalue * vector)
            )
            pivot = int(np.argmax(np.abs(vector)))
            if abs(vector[pivot]) > 0.0:
                vector *= np.exp(-1j * np.angle(vector[pivot]))
            modes.append((int(group[0]), eigenvalue, sign, vector, residual))

    if modes:
        mode_matrix = np.vstack([mode[3] for mode in modes])
        signs = np.asarray([mode[2] for mode in modes])
        gram = np.conj(mode_matrix) @ (metric[:, None] * mode_matrix.T)
        metric_gram_residual = _max_abs(gram - np.diag(signs))
    else:
        metric_gram_residual = 0.0
    return modes, metric_gram_residual


def solve_tdhf_self_conjugate_modes(
    matrices: TDHFSelfConjugateQMatrices,
    *,
    imag_tol: float = 1.0e-10,
    norm_tol: float = 1.0e-10,
    zero_tol: float = 1.0e-10,
    degeneracy_tol: float = 1.0e-10,
) -> TDHFSelfConjugateModeAssignment:
    """Return eta-normalized self-conjugate modes without signed-q labels."""

    imag_tol = _validated_tolerance("imag_tol", imag_tol)
    norm_tol = _validated_tolerance("norm_tol", norm_tol)
    zero_tol = _validated_tolerance("zero_tol", zero_tol)
    degeneracy_tol = _validated_tolerance("degeneracy_tol", degeneracy_tol)
    values, vectors = scipy_linalg.eig(matrices.L)
    norms = np.real(
        np.sum(np.conj(vectors) * matrices.eta[:, None] * vectors, axis=0)
    )
    raw_residuals = np.linalg.norm(
        matrices.L @ vectors - vectors * values[None, :], axis=0
    )
    modes, gram_residual = _metric_normalized_real_modes(
        matrices.L,
        matrices.eta,
        values,
        vectors,
        imag_tol=imag_tol,
        norm_tol=norm_tol,
        zero_tol=zero_tol,
        degeneracy_tol=degeneracy_tol,
    )
    positive = [mode for mode in modes if mode[2] > 0.0]
    return TDHFSelfConjugateModeAssignment(
        raw_eigenvalues=np.asarray(values),
        raw_metric_norms=np.asarray(norms),
        raw_residuals=np.asarray(raw_residuals),
        positive_metric_indices=np.asarray(
            [mode[0] for mode in positive], dtype=np.int64
        ),
        energies=np.asarray([np.real(mode[1]) for mode in positive]),
        vectors=(
            np.vstack([mode[3] for mode in positive])
            if positive
            else np.empty((0, matrices.L.shape[0]), dtype=np.complex128)
        ),
        residuals=np.asarray([mode[4] for mode in positive]),
        metric_gram_residual=gram_residual,
        complex_indices=np.nonzero(np.abs(values.imag) > imag_tol)[0].astype(np.int64),
        null_metric_indices=np.nonzero(np.abs(norms) <= norm_tol)[0].astype(np.int64),
        zero_eigenvalue_indices=np.nonzero(np.abs(values) <= zero_tol)[0].astype(np.int64),
    )


def solve_tdhf_wang_signed_modes(
    matrices: TDHFSignedQMatrices,
    *,
    imag_tol: float = 1.0e-10,
    norm_tol: float = 1.0e-10,
    zero_tol: float = 1.0e-10,
    degeneracy_tol: float = 1.0e-10,
) -> TDHFWangModeAssignment:
    """Apply Wang Appendix-A assignment after eta-Gram normalization."""

    imag_tol = _validated_tolerance("imag_tol", imag_tol)
    norm_tol = _validated_tolerance("norm_tol", norm_tol)
    zero_tol = _validated_tolerance("zero_tol", zero_tol)
    degeneracy_tol = _validated_tolerance("degeneracy_tol", degeneracy_tol)
    values, vectors = scipy_linalg.eig(matrices.L_plus)
    metric = matrices.eta_plus
    norms = np.real(np.sum(np.conj(vectors) * metric[:, None] * vectors, axis=0))
    raw_residuals = np.linalg.norm(
        matrices.L_plus @ vectors - vectors * values[None, :], axis=0
    )
    modes, gram_residual = _metric_normalized_real_modes(
        matrices.L_plus,
        metric,
        values,
        vectors,
        imag_tol=imag_tol,
        norm_tol=norm_tol,
        zero_tol=zero_tol,
        degeneracy_tol=degeneracy_tol,
    )
    plus = [mode for mode in modes if mode[2] > 0.0]
    minus = [mode for mode in modes if mode[2] < 0.0]

    def vectors_for(
        selected: list[tuple[int, complex, float, np.ndarray, float]],
    ) -> np.ndarray:
        return (
            np.vstack([mode[3] for mode in selected])
            if selected
            else np.empty((0, matrices.L_plus.shape[0]), dtype=np.complex128)
        )

    return TDHFWangModeAssignment(
        raw_eigenvalues=np.asarray(values),
        raw_metric_norms=np.asarray(norms),
        raw_residuals=np.asarray(raw_residuals),
        plus_raw_indices=np.asarray([mode[0] for mode in plus], dtype=np.int64),
        minus_raw_indices=np.asarray([mode[0] for mode in minus], dtype=np.int64),
        plus_energies=np.asarray([np.real(mode[1]) for mode in plus]),
        minus_energies=np.asarray([-np.real(mode[1]) for mode in minus]),
        plus_vectors=vectors_for(plus),
        minus_vectors=vectors_for(minus),
        plus_residuals=np.asarray([mode[4] for mode in plus]),
        minus_residuals=np.asarray([mode[4] for mode in minus]),
        metric_gram_residual=gram_residual,
        complex_indices=np.nonzero(np.abs(values.imag) > imag_tol)[0].astype(np.int64),
        null_metric_indices=np.nonzero(np.abs(norms) <= norm_tol)[0].astype(np.int64),
        zero_eigenvalue_indices=np.nonzero(np.abs(values) <= zero_tol)[0].astype(np.int64),
    )


def certify_tdhf_ward_identity(
    *,
    hessian: np.ndarray,
    liouvillian: np.ndarray,
    generator: np.ndarray,
    generator_label: str,
    generator_provenance: str,
    source_fingerprint: str,
    expected_source_fingerprint: str,
    interaction_fingerprint: str,
    sector_fingerprint: str,
    response_scope: str,
    static_hessian_authority: TDHFStaticHessianAuthority,
    source_stationarity_residual: float,
    source_stationarity_tolerance: float,
    action_tolerance: float,
    null_eigenvalue_tolerance: float,
    overlap_tolerance: float,
) -> TDHFWardCertificate:
    """Certify a supplied broken-symmetry generator against a scalar Hessian."""

    source_stationarity_tolerance = _validated_tolerance(
        "source_stationarity_tolerance", source_stationarity_tolerance
    )
    action_tolerance = _validated_tolerance("action_tolerance", action_tolerance)
    null_eigenvalue_tolerance = _validated_tolerance(
        "null_eigenvalue_tolerance", null_eigenvalue_tolerance
    )
    overlap_tolerance = _validated_tolerance("overlap_tolerance", overlap_tolerance)
    if overlap_tolerance > 1.0:
        raise ValueError("overlap_tolerance must not exceed one")
    if not np.isfinite(source_stationarity_residual) or source_stationarity_residual < 0.0:
        raise ValueError("source_stationarity_residual must be finite and non-negative")
    h = np.asarray(hessian, dtype=np.complex128)
    l = np.asarray(liouvillian, dtype=np.complex128)
    g = np.asarray(generator, dtype=np.complex128).reshape(-1)
    if h.shape != l.shape or h.ndim != 2 or h.shape[0] != h.shape[1] or g.size != h.shape[0]:
        raise ValueError("Ward hessian, liouvillian, and generator shapes disagree")
    norm = float(np.linalg.norm(g))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Ward generator must be finite and nonzero")
    g = g / norm
    h_action = float(np.linalg.norm(h @ g))
    l_action = float(np.linalg.norm(l @ g))
    hermitian_residual = _max_abs(h - np.conj(h.T))
    if hermitian_residual <= action_tolerance:
        eigenvalues, eigenvectors = scipy_linalg.eigh(h)
        null = eigenvectors[:, np.abs(eigenvalues) <= null_eigenvalue_tolerance]
        overlap = (
            float(np.linalg.norm(np.conj(null.T) @ g) ** 2) if null.shape[1] else 0.0
        )
    else:
        overlap = 0.0
    failures: list[str] = []
    if source_fingerprint != expected_source_fingerprint:
        failures.append("source_fingerprint_mismatch")
    if static_hessian_authority != "scalar_hessian":
        failures.append("scalar_hessian_authority_missing")
    if not interaction_fingerprint:
        failures.append("missing_interaction_fingerprint")
    if not generator_label or not generator_provenance:
        failures.append("missing_generator_provenance")
    if not sector_fingerprint or not response_scope:
        failures.append("missing_sector_provenance")
    if source_stationarity_residual > source_stationarity_tolerance:
        failures.append("source_not_stationary")
    if hermitian_residual > action_tolerance:
        failures.append("hessian_not_hermitian")
    if h_action > action_tolerance or l_action > action_tolerance:
        failures.append("ward_action_residual")
    if overlap < overlap_tolerance:
        failures.append("static_null_overlap")
    return TDHFWardCertificate(
        passed=not failures,
        generator_label=generator_label,
        generator_provenance=generator_provenance,
        source_fingerprint=source_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        sector_fingerprint=sector_fingerprint,
        response_scope=response_scope,
        generator_fingerprint=fingerprint_tdhf_matrix(g[:, None]),
        generator_vector=np.asarray(g),
        hessian_fingerprint=fingerprint_tdhf_matrix(h),
        liouvillian_fingerprint=fingerprint_tdhf_matrix(l),
        static_hessian_authority=static_hessian_authority,
        static_null_tolerance=float(null_eigenvalue_tolerance),
        source_stationarity_residual=float(source_stationarity_residual),
        source_stationarity_tolerance=float(source_stationarity_tolerance),
        hessian_action_residual=h_action,
        liouvillian_action_residual=l_action,
        action_tolerance=float(action_tolerance),
        static_null_overlap=overlap,
        overlap_tolerance=float(overlap_tolerance),
        failure_reasons=tuple(failures),
    )


def analyze_tdhf_typed_sector(
    sector: TDHFTypedSector,
    *,
    structure_tolerance: float = 1.0e-10,
    hessian_tolerance: float = 1.0e-10,
    imag_tolerance: float = 1.0e-10,
    norm_tolerance: float = 1.0e-10,
    zero_tolerance: float = 1.0e-10,
    degeneracy_tolerance: float = 1.0e-10,
    pairing_tolerance: float = 1.0e-9,
    eigensolver_tolerance: float = 1.0e-9,
    ward: TDHFWardCertificate | None = None,
) -> TDHFTypedAnalysis:
    """Run structure, static, dynamic, and zero-origin analyses independently."""

    structure_tolerance = _validated_tolerance(
        "structure_tolerance", structure_tolerance
    )
    hessian_tolerance = _validated_tolerance(
        "hessian_tolerance", hessian_tolerance
    )
    imag_tolerance = _validated_tolerance("imag_tolerance", imag_tolerance)
    norm_tolerance = _validated_tolerance("norm_tolerance", norm_tolerance)
    zero_tolerance = _validated_tolerance("zero_tolerance", zero_tolerance)
    degeneracy_tolerance = _validated_tolerance(
        "degeneracy_tolerance", degeneracy_tolerance
    )
    pairing_tolerance = _validated_tolerance(
        "pairing_tolerance", pairing_tolerance
    )
    eigensolver_tolerance = _validated_tolerance(
        "eigensolver_tolerance", eigensolver_tolerance
    )

    if isinstance(sector, TDHFGenericSignedQSector):
        if sector.sewing.source_fingerprint != sector.source_fingerprint:
            raise ValueError("Nambu sewing source fingerprint does not match sector")
        if sector.sewing.plus_pairs_fingerprint != fingerprint_tdhf_pairs(
            sector.blocks.plus_pairs
        ) or sector.sewing.minus_pairs_fingerprint != fingerprint_tdhf_pairs(
            sector.blocks.minus_pairs
        ):
            raise ValueError("Nambu sewing pair fingerprints do not match sector")
        matrices: TDHFSignedQMatrices | TDHFSelfConjugateQMatrices = (
            build_tdhf_signed_q_matrices(
                sector.blocks,
                sector.sewing,
                structure_tolerance=structure_tolerance,
            )
        )
        assignment: TDHFWangModeAssignment | TDHFSelfConjugateModeAssignment = (
            solve_tdhf_wang_signed_modes(
                matrices,
                imag_tol=imag_tolerance,
                norm_tol=norm_tolerance,
                zero_tol=zero_tolerance,
                degeneracy_tol=degeneracy_tolerance,
            )
        )
        raw_plus_values = assignment.raw_eigenvalues
        residual_plus = assignment.raw_residuals
        raw_minus_values, raw_minus_vectors = scipy_linalg.eig(matrices.L_minus)
        residual_minus = np.linalg.norm(
            matrices.L_minus @ raw_minus_vectors
            - raw_minus_vectors * raw_minus_values[None, :],
            axis=0,
        )
        pairing = signed_q_particle_hole_assignment_residual(
            raw_plus_values, raw_minus_values
        )
        h = matrices.H_plus
        l = matrices.L_plus
        structure_ok = matrices.structure.ok
    else:
        matrices = build_tdhf_self_conjugate_matrices(
            sector,
            structure_tolerance=structure_tolerance,
        )
        assignment = solve_tdhf_self_conjugate_modes(
            matrices,
            imag_tol=imag_tolerance,
            norm_tol=norm_tolerance,
            zero_tol=zero_tolerance,
            degeneracy_tol=degeneracy_tolerance,
        )
        raw_plus_values = assignment.raw_eigenvalues
        raw_minus_values = assignment.raw_eigenvalues
        residual_plus = assignment.raw_residuals
        residual_minus = assignment.raw_residuals
        pairing = signed_q_particle_hole_assignment_residual(
            raw_plus_values, raw_minus_values
        )
        h = matrices.H
        l = matrices.L
        structure_ok = matrices.structure.ok

    finite_dynamic = bool(
        np.all(np.isfinite(raw_plus_values.real))
        and np.all(np.isfinite(raw_plus_values.imag))
        and np.all(np.isfinite(raw_minus_values.real))
        and np.all(np.isfinite(raw_minus_values.imag))
    )
    complex_count = int(
        np.count_nonzero(np.abs(raw_plus_values.imag) > imag_tolerance)
        + (
            np.count_nonzero(np.abs(raw_minus_values.imag) > imag_tolerance)
            if isinstance(sector, TDHFGenericSignedQSector)
            else 0
        )
    )
    max_imag = float(
        max(
            np.max(np.abs(raw_plus_values.imag), initial=0.0),
            np.max(np.abs(raw_minus_values.imag), initial=0.0),
        )
    )
    max_residual_plus = float(np.max(residual_plus, initial=0.0))
    max_residual_minus = float(np.max(residual_minus, initial=0.0))
    if isinstance(assignment, TDHFWangModeAssignment):
        selected_residual = float(
            max(
                np.max(assignment.plus_residuals, initial=0.0),
                np.max(assignment.minus_residuals, initial=0.0),
            )
        )
    else:
        selected_residual = float(np.max(assignment.residuals, initial=0.0))
    metric_gram_residual = float(assignment.metric_gram_residual)
    finite_residuals = bool(
        np.all(np.isfinite(residual_plus))
        and np.all(np.isfinite(residual_minus))
        and np.isfinite(selected_residual)
        and np.isfinite(metric_gram_residual)
    )
    if (
        not finite_dynamic
        or not finite_residuals
        or not structure_ok
        or not np.isfinite(pairing)
        or pairing > pairing_tolerance
        or max_residual_plus > eigensolver_tolerance
        or max_residual_minus > eigensolver_tolerance
        or selected_residual > eigensolver_tolerance
        or metric_gram_residual > norm_tolerance
    ):
        dynamic_kind: Literal["real", "complex", "invalid"] = "invalid"
    elif complex_count:
        dynamic_kind = "complex"
    else:
        dynamic_kind = "real"
    dynamic = TDHFDynamicStatus(
        kind=dynamic_kind,
        complex_count=complex_count,
        max_abs_imag=max_imag,
        signed_pairing_residual=pairing,
        max_eigensolver_residual_plus=max_residual_plus,
        max_eigensolver_residual_minus=max_residual_minus,
        selected_mode_residual=selected_residual,
        metric_gram_residual=metric_gram_residual,
        imag_tolerance=imag_tolerance,
        pairing_tolerance=pairing_tolerance,
        eigensolver_tolerance=eigensolver_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        structure_tolerance=structure_tolerance,
    )

    hermitian_residual = _max_abs(h - np.conj(h.T))
    if sector.static_hessian_authority != "scalar_hessian":
        static = TDHFStaticStatus(
            kind="not_established",
            eigenvalues=np.asarray([], dtype=float),
            negative_count=0,
            zero_count=0,
            positive_count=0,
            min_eigenvalue=float("nan"),
            hermitian_residual=hermitian_residual,
            tolerance=hessian_tolerance,
        )
    elif hermitian_residual > hessian_tolerance or not structure_ok:
        static = TDHFStaticStatus(
            kind="invalid",
            eigenvalues=np.asarray([], dtype=float),
            negative_count=0,
            zero_count=0,
            positive_count=0,
            min_eigenvalue=float("nan"),
            hermitian_residual=hermitian_residual,
            tolerance=hessian_tolerance,
        )
    else:
        eigenvalues = scipy_linalg.eigvalsh(h)
        negative = int(np.count_nonzero(eigenvalues < -hessian_tolerance))
        zero = int(np.count_nonzero(np.abs(eigenvalues) <= hessian_tolerance))
        positive = int(np.count_nonzero(eigenvalues > hessian_tolerance))
        if negative:
            kind: Literal["positive_definite", "positive_semidefinite", "indefinite", "invalid", "not_established"] = "indefinite"
        elif zero:
            kind = "positive_semidefinite"
        else:
            kind = "positive_definite"
        static = TDHFStaticStatus(
            kind=kind,
            eigenvalues=np.asarray(eigenvalues),
            negative_count=negative,
            zero_count=zero,
            positive_count=positive,
            min_eigenvalue=float(eigenvalues[0]),
            hermitian_residual=hermitian_residual,
            tolerance=hessian_tolerance,
        )

    ward_bound = False
    if ward is not None and ward.passed:
        mismatches = []
        if sector.static_hessian_authority != "scalar_hessian":
            mismatches.append("scalar_hessian_authority")
        if ward.source_fingerprint != sector.source_fingerprint:
            mismatches.append("source_fingerprint")
        if ward.interaction_fingerprint != sector.interaction_fingerprint:
            mismatches.append("interaction_fingerprint")
        if ward.sector_fingerprint != fingerprint_tdhf_sector(sector):
            mismatches.append("sector_fingerprint")
        if ward.response_scope != sector.response_scope:
            mismatches.append("response_scope")
        if ward.hessian_fingerprint != fingerprint_tdhf_matrix(h):
            mismatches.append("hessian_fingerprint")
        if ward.liouvillian_fingerprint != fingerprint_tdhf_matrix(l):
            mismatches.append("liouvillian_fingerprint")
        if ward.static_hessian_authority != sector.static_hessian_authority:
            mismatches.append("static_hessian_authority")
        if ward.static_null_tolerance != hessian_tolerance:
            mismatches.append("static_null_tolerance")
        if static.kind != "positive_semidefinite" or static.zero_count == 0:
            mismatches.append("static_null_status")
        if ward.failure_reasons:
            mismatches.append("certificate_failure_reasons")
        if (
            not ward.generator_label
            or not ward.generator_provenance
            or not ward.generator_fingerprint
        ):
            mismatches.append("generator_provenance")
        generator = np.asarray(ward.generator_vector, dtype=np.complex128).reshape(-1)
        generator_valid = bool(
            generator.size == h.shape[0]
            and np.all(np.isfinite(generator.real))
            and np.all(np.isfinite(generator.imag))
            and abs(np.linalg.norm(generator) - 1.0) <= 1.0e-12
            and ward.generator_fingerprint
            == fingerprint_tdhf_matrix(generator[:, None])
        )
        if not generator_valid:
            mismatches.append("generator_integrity")
        certificate_scalars = (
            ward.source_stationarity_residual,
            ward.source_stationarity_tolerance,
            ward.hessian_action_residual,
            ward.liouvillian_action_residual,
            ward.action_tolerance,
            ward.static_null_overlap,
            ward.overlap_tolerance,
            ward.static_null_tolerance,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in certificate_scalars):
            mismatches.append("certificate_nonfinite")
        elif ward.overlap_tolerance > 1.0 or ward.static_null_overlap > 1.0 + 1.0e-12:
            mismatches.append("certificate_overlap_range")
        elif generator_valid:
            h_action = float(np.linalg.norm(h @ generator))
            l_action = float(np.linalg.norm(l @ generator))
            null_vectors = scipy_linalg.eigh(h)[1][
                :, np.abs(scipy_linalg.eigvalsh(h)) <= ward.static_null_tolerance
            ]
            overlap = (
                float(np.linalg.norm(np.conj(null_vectors.T) @ generator) ** 2)
                if null_vectors.shape[1]
                else 0.0
            )
            if (
                ward.source_stationarity_residual
                > ward.source_stationarity_tolerance
                or h_action > ward.action_tolerance
                or l_action > ward.action_tolerance
                or ward.hessian_action_residual > ward.action_tolerance
                or ward.liouvillian_action_residual > ward.action_tolerance
                or overlap < ward.overlap_tolerance
                or ward.static_null_overlap < ward.overlap_tolerance
            ):
                mismatches.append("certificate_gate_recheck")
        if mismatches:
            raise ValueError(
                "Ward certificate is not bound to analyzed sector: "
                + ", ".join(mismatches)
            )
        ward_bound = True

    dynamic_zero_count = int(
        np.count_nonzero(np.abs(raw_plus_values) <= zero_tolerance)
    )
    if ward_bound:
        zero_origin: Literal["none", "ordinary_dynamic_zero", "ward_static_null", "uncertified_static_null"] = "ward_static_null"
    elif static.zero_count:
        zero_origin = "uncertified_static_null"
    elif dynamic_zero_count and dynamic.kind != "invalid":
        zero_origin = "ordinary_dynamic_zero"
    else:
        zero_origin = "none"
    zero_mode = TDHFZeroModeStatus(
        origin=zero_origin,
        static_zero_count=static.zero_count,
        dynamic_zero_count=dynamic_zero_count,
        ward_passed=ward_bound,
    )
    return TDHFTypedAnalysis(
        sector=sector,
        matrices=matrices,
        assignment=assignment,
        static=static,
        dynamic=dynamic,
        zero_mode=zero_mode,
        ward=ward,
    )


__all__ = [
    "TDHFDynamicStatus",
    "TDHF_TYPED_API_VERSION",
    "TDHFGenericSignedQ",
    "TDHFGenericSignedQSector",
    "TDHFNambuSewing",
    "TDHFSectorProviderProtocol",
    "TDHFSelfConjugateModeAssignment",
    "TDHFSelfConjugateQ",
    "TDHFSelfConjugateQMatrices",
    "TDHFSelfConjugateQSector",
    "TDHFSelfConjugateQStructureResiduals",
    "TDHFSignedQ",
    "TDHFSignedQBlocks",
    "TDHFSignedQMatrices",
    "TDHFSignedStructureResiduals",
    "TDHFStaticHessianAuthority",
    "TDHFStaticStatus",
    "TDHFTypedAnalysis",
    "TDHFTypedSector",
    "TDHFWangModeAssignment",
    "TDHFWardCertificate",
    "TDHFZeroModeStatus",
    "analyze_tdhf_typed_sector",
    "build_standard_nambu_sewing",
    "build_tdhf_self_conjugate_matrices",
    "build_tdhf_signed_q_matrices",
    "certify_tdhf_ward_identity",
    "classify_tdhf_signed_q",
    "fingerprint_tdhf_matrix",
    "fingerprint_tdhf_pairs",
    "fingerprint_tdhf_sector",
    "solve_tdhf_self_conjugate_modes",
    "solve_tdhf_wang_signed_modes",
]
