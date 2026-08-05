"""Multi-generator Ward gates and nonrelativistic Goldstone counting.

For broken charges with physical commutator matrix
``rho_ab = -i <[Q_a,Q_b]> / volume``, the Watanabe--Murayama counting theorem
(PRL 108, 251602; PRX 4, 031057) gives

    n_A = n_BG - rank(rho)
    n_B = rank(rho) / 2.

Type-A/type-B are the generic theorem labels.  Identifying them with linear
(type-I) and quadratic (type-II) dispersions additionally assumes the regular
spatial-gradient structure of the physical system; that mapping belongs in a
system adapter such as the Khalaf Fig. 3 contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
from scipy import linalg as scipy_linalg

from .tdhf_signed import (
    TDHFStaticHessianAuthority,
    TDHFTypedSector,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_sector,
)


@dataclass(frozen=True)
class TDHFGoldstoneCount:
    broken_generator_count: int
    symplectic_rank: int
    type_a_count: int
    type_b_count: int
    dynamic_branch_count: int
    rank_atol: float
    rank_rtol: float
    rank_threshold: float
    antisymmetry_residual: float
    singular_values: np.ndarray
    smallest_retained_margin: float
    largest_rejected_margin: float


@dataclass(frozen=True)
class TDHFSymplecticProvenance:
    commutator_definition: str
    volume_normalization: str
    generator_basis_order: tuple[str, ...]
    source_fingerprint: str
    interaction_fingerprint: str
    sector_fingerprint: str
    response_scope: str
    notes: str

    def __post_init__(self) -> None:
        text_fields = (
            self.commutator_definition,
            self.volume_normalization,
            self.source_fingerprint,
            self.interaction_fingerprint,
            self.sector_fingerprint,
            self.response_scope,
            self.notes,
        )
        if any(not str(value).strip() for value in text_fields):
            raise ValueError("symplectic provenance fields cannot be empty")
        if not self.generator_basis_order or any(
            not str(value).strip() for value in self.generator_basis_order
        ):
            raise ValueError("symplectic generator basis order cannot be empty")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "commutator_definition": self.commutator_definition,
                "volume_normalization": self.volume_normalization,
                "generator_basis_order": self.generator_basis_order,
                "source_fingerprint": self.source_fingerprint,
                "interaction_fingerprint": self.interaction_fingerprint,
                "sector_fingerprint": self.sector_fingerprint,
                "response_scope": self.response_scope,
                "notes": self.notes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TDHFWardSubspaceCertificate:
    passed: bool
    generator_labels: tuple[str, ...]
    generator_provenances: tuple[str, ...]
    generator_provenance_fingerprint: str
    symplectic_provenance: TDHFSymplecticProvenance
    symplectic_provenance_fingerprint: str
    source_fingerprint: str
    interaction_fingerprint: str
    sector_fingerprint: str
    response_scope: str
    static_hessian_authority: TDHFStaticHessianAuthority
    generator_matrix: np.ndarray
    generator_fingerprint: str
    orthonormal_generator_basis: np.ndarray
    orthonormal_generator_basis_fingerprint: str
    generator_singular_values: np.ndarray
    generator_rank_threshold: float
    generator_rank_atol: float
    generator_rank_rtol: float
    hessian_fingerprint: str
    liouvillian_fingerprint: str
    symplectic_matrix: np.ndarray
    symplectic_fingerprint: str
    goldstone_count: TDHFGoldstoneCount | None
    symplectic_rank_atol: float
    symplectic_rank_rtol: float
    antisymmetry_tolerance: float
    source_stationarity_residual: float
    source_stationarity_tolerance: float
    hessian_hermiticity_residual: float
    generator_gram_residual: float
    hessian_action_operator_norm: float
    liouvillian_action_operator_norm: float
    action_tolerance: float
    static_null_tolerance: float
    static_null_dimension: int
    static_null_min_principal_overlap: float
    overlap_tolerance: float
    failure_reasons: tuple[str, ...]


def _tolerance(name: str, value: float) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _strict_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _rank_threshold(
    singular_values: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> float:
    scale = float(singular_values[0]) if singular_values.size else 0.0
    return float(atol + rtol * scale)


def _rank_margins(
    singular_values: np.ndarray,
    rank: int,
    threshold: float,
) -> tuple[float, float]:
    retained = (
        float(singular_values[rank - 1] - threshold) if rank else float("inf")
    )
    rejected = (
        float(threshold - singular_values[rank])
        if rank < singular_values.size
        else float("inf")
    )
    return retained, rejected


def count_tdhf_goldstones_from_rank(
    broken_generator_count: int,
    symplectic_rank: int,
) -> TDHFGoldstoneCount:
    """Apply the type-A/type-B theorem to an independently certified rank."""

    n_broken = _strict_count("broken_generator_count", broken_generator_count)
    rank = _strict_count("symplectic_rank", symplectic_rank)
    if n_broken < 0 or rank < 0 or rank > n_broken:
        raise ValueError("Goldstone symplectic rank must lie in [0,n_broken]")
    if rank % 2:
        raise ValueError("a physical real antisymmetric symplectic rank must be even")
    type_a = n_broken - rank
    type_b = rank // 2
    singular_values = np.asarray([], dtype=float)
    singular_values.setflags(write=False)
    return TDHFGoldstoneCount(
        broken_generator_count=n_broken,
        symplectic_rank=rank,
        type_a_count=type_a,
        type_b_count=type_b,
        dynamic_branch_count=type_a + type_b,
        rank_atol=0.0,
        rank_rtol=0.0,
        rank_threshold=0.0,
        antisymmetry_residual=0.0,
        singular_values=singular_values,
        smallest_retained_margin=float("nan"),
        largest_rejected_margin=float("nan"),
    )


def analyze_tdhf_goldstone_symplectic_matrix(
    symplectic_matrix: np.ndarray,
    *,
    rank_atol: float,
    rank_rtol: float,
    antisymmetry_tolerance: float,
) -> TDHFGoldstoneCount:
    """Validate a raw physical rho and determine a scale-aware rank."""

    rank_atol = _tolerance("rank_atol", rank_atol)
    rank_rtol = _tolerance("rank_rtol", rank_rtol)
    antisymmetry_tolerance = _tolerance(
        "antisymmetry_tolerance", antisymmetry_tolerance
    )
    rho_complex = np.asarray(symplectic_matrix, dtype=np.complex128)
    if rho_complex.ndim != 2 or rho_complex.shape[0] != rho_complex.shape[1]:
        raise ValueError("symplectic matrix must be square")
    if not np.all(np.isfinite(rho_complex.real)) or not np.all(
        np.isfinite(rho_complex.imag)
    ):
        raise ValueError("symplectic matrix must be finite")
    rho = np.asarray(rho_complex.real)
    symmetric_contamination = 0.5 * (rho + rho.T)
    symmetric_singular_values = scipy_linalg.svdvals(symmetric_contamination)
    imaginary_singular_values = scipy_linalg.svdvals(rho_complex.imag)
    antisymmetry_residual = float(
        max(
            symmetric_singular_values[0]
            if symmetric_singular_values.size
            else 0.0,
            imaginary_singular_values[0]
            if imaginary_singular_values.size
            else 0.0,
        )
    )
    singular_values = np.asarray(scipy_linalg.svdvals(rho), dtype=float)
    threshold = _rank_threshold(
        singular_values,
        atol=rank_atol,
        rtol=rank_rtol,
    )
    # Symmetric/imaginary contamination must be below both the declared
    # structural tolerance and the numerical rank-resolution floor.
    if antisymmetry_residual > antisymmetry_tolerance or (
        antisymmetry_residual > threshold
    ):
        raise ValueError("symplectic matrix is not resolved as real antisymmetric")
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank % 2:
        raise ValueError("resolved symplectic rank is odd")
    retained_margin, rejected_margin = _rank_margins(
        singular_values, rank, threshold
    )
    if (
        retained_margin <= antisymmetry_residual
        or rejected_margin <= antisymmetry_residual
    ):
        raise ValueError(
            "symplectic rank is not separated from structural uncertainty"
        )
    singular_values.setflags(write=False)
    type_a = rho.shape[0] - rank
    type_b = rank // 2
    return TDHFGoldstoneCount(
        broken_generator_count=rho.shape[0],
        symplectic_rank=rank,
        type_a_count=type_a,
        type_b_count=type_b,
        dynamic_branch_count=type_a + type_b,
        rank_atol=rank_atol,
        rank_rtol=rank_rtol,
        rank_threshold=threshold,
        antisymmetry_residual=antisymmetry_residual,
        singular_values=singular_values,
        smallest_retained_margin=retained_margin,
        largest_rejected_margin=rejected_margin,
    )


def _generator_provenance_fingerprint(
    labels: tuple[str, ...],
    provenances: tuple[str, ...],
    generator_fingerprint: str,
    *,
    source_fingerprint: str,
    interaction_fingerprint: str,
    sector_fingerprint: str,
    response_scope: str,
) -> str:
    payload = json.dumps(
        {
            "labels": labels,
            "provenances": provenances,
            "generator_fingerprint": generator_fingerprint,
            "source_fingerprint": source_fingerprint,
            "interaction_fingerprint": interaction_fingerprint,
            "sector_fingerprint": sector_fingerprint,
            "response_scope": response_scope,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _operator_norm(matrix: np.ndarray) -> float:
    singular_values = scipy_linalg.svdvals(matrix)
    return float(singular_values[0]) if singular_values.size else 0.0


def certify_tdhf_ward_subspace(
    *,
    hessian: np.ndarray,
    liouvillian: np.ndarray,
    generators: np.ndarray,
    generator_labels: Sequence[str],
    generator_provenances: Sequence[str],
    symplectic_matrix: np.ndarray,
    symplectic_provenance: TDHFSymplecticProvenance,
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
    generator_rank_atol: float,
    generator_rank_rtol: float,
    symplectic_rank_atol: float,
    symplectic_rank_rtol: float,
    antisymmetry_tolerance: float,
) -> TDHFWardSubspaceCertificate:
    """Certify a broken-generator tangent subspace and its raw physical rho."""

    source_stationarity_tolerance = _tolerance(
        "source_stationarity_tolerance", source_stationarity_tolerance
    )
    action_tolerance = _tolerance("action_tolerance", action_tolerance)
    null_eigenvalue_tolerance = _tolerance(
        "null_eigenvalue_tolerance", null_eigenvalue_tolerance
    )
    overlap_tolerance = _tolerance("overlap_tolerance", overlap_tolerance)
    generator_rank_atol = _tolerance("generator_rank_atol", generator_rank_atol)
    generator_rank_rtol = _tolerance("generator_rank_rtol", generator_rank_rtol)
    symplectic_rank_atol = _tolerance(
        "symplectic_rank_atol", symplectic_rank_atol
    )
    symplectic_rank_rtol = _tolerance(
        "symplectic_rank_rtol", symplectic_rank_rtol
    )
    antisymmetry_tolerance = _tolerance(
        "antisymmetry_tolerance", antisymmetry_tolerance
    )
    if overlap_tolerance > 1.0:
        raise ValueError("overlap_tolerance must not exceed one")
    if (
        not np.isfinite(source_stationarity_residual)
        or source_stationarity_residual < 0.0
    ):
        raise ValueError("source stationarity residual must be finite and non-negative")

    h = np.asarray(hessian, dtype=np.complex128)
    l = np.asarray(liouvillian, dtype=np.complex128)
    g = np.asarray(generators, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or l.shape != h.shape:
        raise ValueError("Ward-subspace Hessian and Liouvillian shapes disagree")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("Ward-subspace Hessian must be finite")
    if not np.all(np.isfinite(l.real)) or not np.all(np.isfinite(l.imag)):
        raise ValueError("Ward-subspace Liouvillian must be finite")
    if g.ndim != 2 or g.shape[0] != h.shape[0] or g.shape[1] == 0:
        raise ValueError("Ward generators must be a nonempty (dimension,count) matrix")
    if not np.all(np.isfinite(g.real)) or not np.all(np.isfinite(g.imag)):
        raise ValueError("Ward generators must be finite")
    labels = tuple(str(label) for label in generator_labels)
    provenances = tuple(str(value) for value in generator_provenances)
    if len(labels) != g.shape[1] or len(provenances) != g.shape[1]:
        raise ValueError("Ward generator labels/provenances have wrong length")
    if any(not value for value in labels + provenances):
        raise ValueError("Ward generator labels/provenances cannot be empty")
    if not isinstance(symplectic_provenance, TDHFSymplecticProvenance):
        raise TypeError("physical rho requires typed symplectic provenance")
    if symplectic_provenance.generator_basis_order != labels:
        raise ValueError("symplectic provenance generator order mismatch")
    provenance_bindings = (
        (symplectic_provenance.source_fingerprint, source_fingerprint),
        (symplectic_provenance.interaction_fingerprint, interaction_fingerprint),
        (symplectic_provenance.sector_fingerprint, sector_fingerprint),
        (symplectic_provenance.response_scope, response_scope),
    )
    if any(actual != expected for actual, expected in provenance_bindings):
        raise ValueError("symplectic provenance is not bound to Ward metadata")

    column_norms = np.linalg.norm(g, axis=0)
    if np.any(~np.isfinite(column_norms)) or np.any(column_norms == 0.0):
        raise ValueError("Ward generators must be nonzero")
    normalized = g / column_norms[None, :]
    u, generator_singular_values, _ = scipy_linalg.svd(
        normalized,
        full_matrices=False,
    )
    generator_singular_values = np.asarray(generator_singular_values, dtype=float)
    generator_threshold = _rank_threshold(
        generator_singular_values,
        atol=generator_rank_atol,
        rtol=generator_rank_rtol,
    )
    generator_rank = int(
        np.count_nonzero(generator_singular_values > generator_threshold)
    )
    if generator_rank != normalized.shape[1]:
        raise ValueError("Ward generator tangents are linearly dependent")
    q_basis = np.asarray(u[:, : normalized.shape[1]])
    gram_residual = float(
        np.max(
            np.abs(np.conj(q_basis.T) @ q_basis - np.eye(q_basis.shape[1])),
            initial=0.0,
        )
    )

    rho = np.asarray(symplectic_matrix, dtype=np.complex128)
    if rho.shape != (g.shape[1], g.shape[1]):
        raise ValueError("symplectic matrix shape does not match generator count")
    if not np.all(np.isfinite(rho.real)) or not np.all(np.isfinite(rho.imag)):
        raise ValueError("symplectic matrix must be finite")
    try:
        count: TDHFGoldstoneCount | None = (
            analyze_tdhf_goldstone_symplectic_matrix(
                rho,
                rank_atol=symplectic_rank_atol,
                rank_rtol=symplectic_rank_rtol,
                antisymmetry_tolerance=antisymmetry_tolerance,
            )
        )
        symplectic_failure = False
    except ValueError:
        count = None
        symplectic_failure = True

    hermitian_residual = float(np.max(np.abs(h - np.conj(h.T)), initial=0.0))
    h_action = _operator_norm(h @ q_basis)
    l_action = _operator_norm(l @ q_basis)
    if hermitian_residual <= action_tolerance:
        eigenvalues, eigenvectors = scipy_linalg.eigh(h)
        null = eigenvectors[:, np.abs(eigenvalues) <= null_eigenvalue_tolerance]
    else:
        null = np.empty((h.shape[0], 0), dtype=np.complex128)
    if null.shape[1] >= q_basis.shape[1]:
        principal = scipy_linalg.svdvals(np.conj(null.T) @ q_basis)
        min_overlap = float(np.min(principal**2, initial=1.0))
    else:
        min_overlap = 0.0

    derived = (
        hermitian_residual,
        gram_residual,
        h_action,
        l_action,
        min_overlap,
    )
    if not all(np.isfinite(value) for value in derived):
        raise ValueError("Ward-subspace derived residuals must be finite")

    failures: list[str] = []
    if not source_fingerprint or source_fingerprint != expected_source_fingerprint:
        failures.append("source_fingerprint_mismatch")
    if static_hessian_authority != "scalar_hessian":
        failures.append("scalar_hessian_authority_missing")
    if not interaction_fingerprint or not sector_fingerprint or not response_scope:
        failures.append("missing_sector_provenance")
    if source_stationarity_residual > source_stationarity_tolerance:
        failures.append("source_not_stationary")
    if hermitian_residual > action_tolerance:
        failures.append("hessian_not_hermitian")
    if gram_residual > max(generator_rank_atol, generator_rank_rtol):
        failures.append("generator_gram_residual")
    if h_action > action_tolerance or l_action > action_tolerance:
        failures.append("ward_action_residual")
    if min_overlap < overlap_tolerance:
        failures.append("static_null_overlap")
    if symplectic_failure:
        failures.append("invalid_symplectic_matrix")

    normalized = np.asarray(normalized)
    normalized.setflags(write=False)
    q_basis.setflags(write=False)
    generator_singular_values.setflags(write=False)
    rho.setflags(write=False)
    generator_fingerprint = fingerprint_tdhf_matrix(normalized)
    return TDHFWardSubspaceCertificate(
        passed=not failures,
        generator_labels=labels,
        generator_provenances=provenances,
        generator_provenance_fingerprint=_generator_provenance_fingerprint(
            labels,
            provenances,
            generator_fingerprint,
            source_fingerprint=source_fingerprint,
            interaction_fingerprint=interaction_fingerprint,
            sector_fingerprint=sector_fingerprint,
            response_scope=response_scope,
        ),
        symplectic_provenance=symplectic_provenance,
        symplectic_provenance_fingerprint=symplectic_provenance.fingerprint,
        source_fingerprint=source_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        sector_fingerprint=sector_fingerprint,
        response_scope=response_scope,
        static_hessian_authority=static_hessian_authority,
        generator_matrix=normalized,
        generator_fingerprint=generator_fingerprint,
        orthonormal_generator_basis=q_basis,
        orthonormal_generator_basis_fingerprint=fingerprint_tdhf_matrix(q_basis),
        generator_singular_values=generator_singular_values,
        generator_rank_threshold=generator_threshold,
        generator_rank_atol=generator_rank_atol,
        generator_rank_rtol=generator_rank_rtol,
        hessian_fingerprint=fingerprint_tdhf_matrix(h),
        liouvillian_fingerprint=fingerprint_tdhf_matrix(l),
        symplectic_matrix=rho,
        symplectic_fingerprint=fingerprint_tdhf_matrix(rho),
        goldstone_count=count,
        symplectic_rank_atol=symplectic_rank_atol,
        symplectic_rank_rtol=symplectic_rank_rtol,
        antisymmetry_tolerance=antisymmetry_tolerance,
        source_stationarity_residual=float(source_stationarity_residual),
        source_stationarity_tolerance=source_stationarity_tolerance,
        hessian_hermiticity_residual=hermitian_residual,
        generator_gram_residual=gram_residual,
        hessian_action_operator_norm=h_action,
        liouvillian_action_operator_norm=l_action,
        action_tolerance=action_tolerance,
        static_null_tolerance=null_eigenvalue_tolerance,
        static_null_dimension=int(null.shape[1]),
        static_null_min_principal_overlap=min_overlap,
        overlap_tolerance=overlap_tolerance,
        failure_reasons=tuple(failures),
    )


def _goldstone_receipt_equal(
    left: TDHFGoldstoneCount,
    right: TDHFGoldstoneCount,
) -> bool:
    scalar_fields = (
        "broken_generator_count",
        "symplectic_rank",
        "type_a_count",
        "type_b_count",
        "dynamic_branch_count",
        "rank_atol",
        "rank_rtol",
        "rank_threshold",
        "antisymmetry_residual",
        "smallest_retained_margin",
        "largest_rejected_margin",
    )
    return all(getattr(left, field) == getattr(right, field) for field in scalar_fields) and np.array_equal(
        left.singular_values,
        right.singular_values,
    )


def validate_tdhf_ward_subspace_certificate(
    certificate: TDHFWardSubspaceCertificate,
    *,
    sector: TDHFTypedSector,
    hessian: np.ndarray,
    liouvillian: np.ndarray,
    static_zero_count: int,
    expected_static_null_tolerance: float,
) -> TDHFGoldstoneCount:
    """Rebind and recompute a passing certificate at the typed-sector boundary."""

    if not certificate.passed or certificate.failure_reasons:
        raise ValueError("Ward-subspace certificate is not passing")
    h = np.asarray(hessian, dtype=np.complex128)
    l = np.asarray(liouvillian, dtype=np.complex128)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(l)):
        raise ValueError("analyzed Ward-subspace matrices must be finite")
    mismatches: list[str] = []
    if certificate.source_fingerprint != sector.source_fingerprint:
        mismatches.append("source_fingerprint")
    if certificate.interaction_fingerprint != sector.interaction_fingerprint:
        mismatches.append("interaction_fingerprint")
    if certificate.sector_fingerprint != fingerprint_tdhf_sector(sector):
        mismatches.append("sector_fingerprint")
    if certificate.response_scope != sector.response_scope:
        mismatches.append("response_scope")
    if certificate.static_hessian_authority != sector.static_hessian_authority:
        mismatches.append("static_hessian_authority")
    if certificate.hessian_fingerprint != fingerprint_tdhf_matrix(h):
        mismatches.append("hessian_fingerprint")
    if certificate.liouvillian_fingerprint != fingerprint_tdhf_matrix(l):
        mismatches.append("liouvillian_fingerprint")
    if certificate.static_null_tolerance != expected_static_null_tolerance:
        mismatches.append("static_null_tolerance")
    if static_zero_count < len(certificate.generator_labels):
        mismatches.append("static_null_dimension")
    provenance = certificate.symplectic_provenance
    if (
        not isinstance(provenance, TDHFSymplecticProvenance)
        or certificate.symplectic_provenance_fingerprint != provenance.fingerprint
        or provenance.generator_basis_order != certificate.generator_labels
        or provenance.source_fingerprint != sector.source_fingerprint
        or provenance.interaction_fingerprint != sector.interaction_fingerprint
        or provenance.sector_fingerprint != fingerprint_tdhf_sector(sector)
        or provenance.response_scope != sector.response_scope
    ):
        mismatches.append("symplectic_provenance")

    certificate_scalars = (
        certificate.generator_rank_threshold,
        certificate.generator_rank_atol,
        certificate.generator_rank_rtol,
        certificate.symplectic_rank_atol,
        certificate.symplectic_rank_rtol,
        certificate.antisymmetry_tolerance,
        certificate.source_stationarity_residual,
        certificate.source_stationarity_tolerance,
        certificate.hessian_hermiticity_residual,
        certificate.generator_gram_residual,
        certificate.hessian_action_operator_norm,
        certificate.liouvillian_action_operator_norm,
        certificate.action_tolerance,
        certificate.static_null_tolerance,
        certificate.static_null_min_principal_overlap,
        certificate.overlap_tolerance,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in certificate_scalars):
        mismatches.append("certificate_nonfinite")

    if (
        len(certificate.generator_labels) != len(certificate.generator_provenances)
        or not certificate.generator_labels
        or any(
            not value
            for value in certificate.generator_labels
            + certificate.generator_provenances
        )
    ):
        mismatches.append("generator_provenance")
    generators = np.asarray(certificate.generator_matrix, dtype=np.complex128)
    basis = np.asarray(
        certificate.orthonormal_generator_basis,
        dtype=np.complex128,
    )
    rho = np.asarray(certificate.symplectic_matrix, dtype=np.complex128)
    recomputed_generator_fingerprint = fingerprint_tdhf_matrix(generators)
    if (
        generators.shape != (h.shape[0], len(certificate.generator_labels))
        or not np.all(np.isfinite(generators))
        or certificate.generator_fingerprint != recomputed_generator_fingerprint
    ):
        mismatches.append("generator_fingerprint")
    if certificate.generator_provenance_fingerprint != (
        _generator_provenance_fingerprint(
            certificate.generator_labels,
            certificate.generator_provenances,
            recomputed_generator_fingerprint,
            source_fingerprint=certificate.source_fingerprint,
            interaction_fingerprint=certificate.interaction_fingerprint,
            sector_fingerprint=certificate.sector_fingerprint,
            response_scope=certificate.response_scope,
        )
    ):
        mismatches.append("generator_provenance")
    if certificate.orthonormal_generator_basis_fingerprint != fingerprint_tdhf_matrix(
        basis
    ):
        mismatches.append("orthonormal_basis_fingerprint")
    if (
        rho.shape
        != (len(certificate.generator_labels), len(certificate.generator_labels))
        or certificate.symplectic_fingerprint != fingerprint_tdhf_matrix(rho)
    ):
        mismatches.append("symplectic_fingerprint")
    if basis.shape != (h.shape[0], len(certificate.generator_labels)):
        mismatches.append("orthonormal_basis_shape")
    elif not np.all(np.isfinite(basis)):
        mismatches.append("orthonormal_basis_nonfinite")
    else:
        gram = float(
            np.max(
                np.abs(np.conj(basis.T) @ basis - np.eye(basis.shape[1])),
                initial=0.0,
            )
        )
        generator_singular_values = scipy_linalg.svdvals(generators)
        generator_threshold = _rank_threshold(
            generator_singular_values,
            atol=certificate.generator_rank_atol,
            rtol=certificate.generator_rank_rtol,
        )
        generator_rank = int(
            np.count_nonzero(generator_singular_values > generator_threshold)
        )
        generator_column_norm_residual = float(
            np.max(np.abs(np.linalg.norm(generators, axis=0) - 1.0), initial=0.0)
        )
        span_residual = _operator_norm(
            generators - basis @ (np.conj(basis.T) @ generators)
        )
        h_action = _operator_norm(h @ basis)
        l_action = _operator_norm(l @ basis)
        eigenvalues, eigenvectors = scipy_linalg.eigh(h)
        null = eigenvectors[
            :, np.abs(eigenvalues) <= certificate.static_null_tolerance
        ]
        overlap = (
            float(
                np.min(
                    scipy_linalg.svdvals(np.conj(null.T) @ basis) ** 2,
                    initial=1.0,
                )
            )
            if null.shape[1] >= basis.shape[1]
            else 0.0
        )
        hermitian_residual = float(
            np.max(np.abs(h - np.conj(h.T)), initial=0.0)
        )
        receipt_mismatch = bool(
            not np.array_equal(
                certificate.generator_singular_values,
                generator_singular_values,
            )
            or certificate.generator_rank_threshold != generator_threshold
            or generator_rank != len(certificate.generator_labels)
            or certificate.generator_gram_residual != gram
            or certificate.hessian_action_operator_norm != h_action
            or certificate.liouvillian_action_operator_norm != l_action
            or certificate.hessian_hermiticity_residual != hermitian_residual
            or certificate.static_null_dimension != int(null.shape[1])
            or certificate.static_null_min_principal_overlap != overlap
        )
        if (
            not np.isfinite(gram)
            or not np.isfinite(generator_column_norm_residual)
            or not np.isfinite(span_residual)
            or not np.isfinite(h_action)
            or not np.isfinite(l_action)
            or not np.isfinite(overlap)
            or generator_column_norm_residual > certificate.generator_rank_atol
            or span_residual > generator_threshold
            or receipt_mismatch
            or gram > max(
                certificate.generator_rank_atol,
                certificate.generator_rank_rtol,
            )
            or h_action > certificate.action_tolerance
            or l_action > certificate.action_tolerance
            or overlap < certificate.overlap_tolerance
        ):
            mismatches.append("certificate_gate_recheck")

    try:
        count = analyze_tdhf_goldstone_symplectic_matrix(
            rho,
            rank_atol=certificate.symplectic_rank_atol,
            rank_rtol=certificate.symplectic_rank_rtol,
            antisymmetry_tolerance=certificate.antisymmetry_tolerance,
        )
    except ValueError as error:
        raise ValueError("stored physical rho is invalid") from error
    if certificate.goldstone_count is None or not _goldstone_receipt_equal(
        certificate.goldstone_count,
        count,
    ):
        mismatches.append("goldstone_count")
    if (
        certificate.source_stationarity_residual
        > certificate.source_stationarity_tolerance
    ):
        mismatches.append("source_stationarity")
    if mismatches:
        raise ValueError(
            "Ward-subspace certificate is not bound to analyzed sector: "
            + ", ".join(mismatches)
        )
    return count


__all__ = [
    "TDHFGoldstoneCount",
    "TDHFSymplecticProvenance",
    "TDHFWardSubspaceCertificate",
    "analyze_tdhf_goldstone_symplectic_matrix",
    "certify_tdhf_ward_subspace",
    "count_tdhf_goldstones_from_rank",
    "validate_tdhf_ward_subspace_certificate",
]
