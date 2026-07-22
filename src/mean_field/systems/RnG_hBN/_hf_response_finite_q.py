from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._hf_c3_quotient import (
    RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
    RLGhBNHFC3QuotientInteractionContext,
    build_rlg_hbn_hf_c3_quotient_interaction_components,
    build_rlg_hbn_hf_c3_quotient_interaction_context,
)
from ._hf_shared import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
)
from ._hf_types import RLGhBNHartreeFockRun


RLGhBNFiniteQDensityTangentRole = Literal["ph", "hp"]


@dataclass(frozen=True)
class RLGhBNFiniteQDensityTangent:
    """Stored-density tangent connecting source and target momentum fibers.

    RLG/hBN HF stores ``ΔD[a,b]=<c_a^† c_b>-R[a,b]``. For q=0, a ph/X
    tangent therefore has orbital-basis entry ``D[h,p]=1`` while an hp/Y
    tangent has ``D[p,h]=1``. This tangent-level API remains q=0-only;
    generic nonzero-q lifting is provided by the finite-q quotient matrix API.
    The endpoint arrays prevent either path from collapsing source/target k.
    """

    q_shift: tuple[int, int]
    target_k: np.ndarray
    source_k: np.ndarray
    blocks: np.ndarray
    role: RLGhBNFiniteQDensityTangentRole


@dataclass(frozen=True)
class RLGhBNFiniteQResponse:
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    provenance: dict[str, Any]


def _exact_integer_vector(values: object, *, name: str) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exact integers") from error
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain exact integers, got {numeric.tolist()}")
    return numeric.astype(int)


def _validated_q0_tangent(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = run.basis_data
    mesh = int(basis.mesh_size)
    if mesh <= 0:
        raise ValueError("RLG/hBN HF quotient response requires a regular mesh")
    q_values = _exact_integer_vector(tangent.q_shift, name="q_shift")
    if q_values.shape != (2,):
        raise ValueError(f"q_shift must have shape (2,), got {q_values.shape}")
    q = (int(q_values[0]), int(q_values[1]))
    if q != (0, 0):
        raise NotImplementedError(
            "Only the q=0 corrected-HF quotient response is implemented. "
            "Generic finite-q requires a derived all-copy endpoint lift."
        )
    if tangent.role not in ("ph", "hp"):
        raise ValueError(f"tangent role must be 'ph' or 'hp', got {tangent.role!r}")
    blocks = np.asarray(tangent.blocks, dtype=np.complex128)
    expected_shape = (basis.nt, basis.nt, basis.nk)
    if blocks.shape != expected_shape:
        raise ValueError(
            f"q=0 tangent blocks must have shape {expected_shape}, got {blocks.shape}"
        )
    target_k = _exact_integer_vector(tangent.target_k, name="target_k")
    source_k = _exact_integer_vector(tangent.source_k, name="source_k")
    expected_k = np.arange(basis.nk, dtype=int)
    if not np.array_equal(target_k, expected_k):
        raise ValueError(
            "q=0 dense tangent target_k must enumerate every stored k in order"
        )
    if not np.array_equal(source_k, expected_k):
        raise ValueError(
            "q=0 dense tangent source_k must enumerate every stored k in order"
        )
    return blocks, target_k, source_k


def _validate_interaction_provenance(
    run: RLGhBNHartreeFockRun,
    context: RLGhBNHFC3QuotientInteractionContext,
    *,
    beta: float,
) -> None:
    provenance = run.interaction_provenance
    if provenance is None:
        raise ValueError(
            "HF run has no typed interaction provenance; legacy/v1 archives cannot "
            "be used for variational-v2 response"
        )
    mismatches: dict[str, object] = {}
    if provenance.convention != RLG_HBN_HF_INTERACTION_CONVENTION_VERSION:
        mismatches["convention"] = provenance.convention
    if not provenance.quotient_enabled:
        mismatches["quotient_enabled"] = provenance.quotient_enabled
    if not np.isclose(float(provenance.beta), float(beta), rtol=0.0, atol=1.0e-15):
        mismatches["beta"] = provenance.beta
    if provenance.physical_shifts != tuple(context.physical_shifts):
        mismatches["physical_shifts"] = provenance.physical_shifts
    if provenance.zero_literal_q0_fock:
        mismatches["zero_literal_q0_fock"] = True
    if provenance.basis_periodic_gauge != RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION:
        mismatches["basis_periodic_gauge"] = provenance.basis_periodic_gauge
    if provenance.basis_periodic_gauge_padding != int(
        RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING
    ):
        mismatches["basis_periodic_gauge_padding"] = (
            provenance.basis_periodic_gauge_padding
        )
    if provenance.form_factor_convention != RLG_HBN_FORM_FACTOR_CONVENTION_VERSION:
        mismatches["form_factor_convention"] = provenance.form_factor_convention
    if mismatches:
        raise ValueError(
            "HF run interaction provenance does not match the response kernel: "
            f"{mismatches}"
        )


def validate_rlg_hbn_hf_quotient_source_closure(
    run: RLGhBNHartreeFockRun,
    *,
    context: RLGhBNHFC3QuotientInteractionContext | None = None,
    closure_tolerance_mev: float = 1.0e-3,
    stationarity_tolerance_mev: float = 1.0e-3,
) -> dict[str, float | str]:
    """Fail closed unless a saved HF state belongs to the current quotient.

    This is an intentionally heavy preflight: production callers should run it
    once per restored HF source, not once per tangent column.
    """

    if not run.converged:
        raise ValueError("HF quotient source closure requires a converged HF run")
    resolved_context = (
        build_rlg_hbn_hf_c3_quotient_interaction_context(
            run.basis_data,
            run.overlap_blocks,
        )
        if context is None
        else context
    )
    if resolved_context.basis_data is not run.basis_data:
        raise ValueError("source-closure context was not built from this HF run basis")
    if resolved_context.base_blocks is not run.overlap_blocks:
        raise ValueError("source-closure context was not built from this HF run overlaps")
    provenance = run.interaction_provenance
    beta = float("nan") if provenance is None else float(provenance.beta)
    _validate_interaction_provenance(run, resolved_context, beta=beta)
    components = build_rlg_hbn_hf_c3_quotient_interaction_components(
        run.state.density,
        resolved_context,
        v0=run.state.v0,
        beta=beta,
    )
    rebuilt = np.asarray(run.state.h0) + np.asarray(components.total)
    closure = float(np.max(np.abs(np.asarray(run.state.hamiltonian) - rebuilt)))
    stored_projector = np.asarray(run.state.density) + np.asarray(
        run.state.reference_density
    )
    projector = stored_projector.transpose(1, 0, 2)
    stationarity = 0.0
    for index in range(run.state.nk):
        hamiltonian = np.asarray(run.state.hamiltonian)[:, :, index]
        block = projector[:, :, index]
        stationarity = max(
            stationarity,
            float(np.max(np.abs(hamiltonian @ block - block @ hamiltonian))),
        )
    metrics: dict[str, float | str] = {
        "hf_interaction_convention": RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
        "hamiltonian_closure_mev": closure,
        "projector_commutator_mev": stationarity,
        "closure_tolerance_mev": float(closure_tolerance_mev),
        "stationarity_tolerance_mev": float(stationarity_tolerance_mev),
    }
    if closure > float(closure_tolerance_mev):
        raise ValueError(f"saved HF Hamiltonian fails quotient closure: {metrics}")
    if stationarity > float(stationarity_tolerance_mev):
        raise ValueError(f"saved HF density is not stationary: {metrics}")
    return metrics


def apply_rlg_hbn_hf_quotient_response(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
    *,
    context: RLGhBNHFC3QuotientInteractionContext | None = None,
    beta: float | None = None,
    require_converged: bool = True,
    require_provenance: bool = True,
) -> RLGhBNFiniteQResponse:
    """Apply the corrected HF quotient derivative to a stored-density tangent.

    At q=0 the accepted HF self-energy is exactly linear in the stored density,
    so ``K[dD] = Sigma[dD]``. This function deliberately delegates to the same
    interaction builder used by SCF/ODA. It does not use pair sewing,
    ``fixed_copy``, or post-assembly matrix transport.
    """

    if bool(require_converged) and not bool(run.converged):
        raise ValueError(
            "HF quotient response requires a converged HF run unless "
            "require_converged=False is explicitly used for a reduced diagnostic"
        )
    if not np.isclose(
        float(run.state.v0),
        float(run.basis_data.v0),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(
            f"HF state/basis v0 mismatch: {run.state.v0} != {run.basis_data.v0}"
        )
    blocks, target_k, source_k = _validated_q0_tangent(run, tangent)
    resolved_context = (
        build_rlg_hbn_hf_c3_quotient_interaction_context(
            run.basis_data,
            run.overlap_blocks,
        )
        if context is None
        else context
    )
    if resolved_context.basis_data is not run.basis_data:
        raise ValueError("response context was not built from this HF run basis")
    if resolved_context.base_blocks is not run.overlap_blocks:
        raise ValueError("response context was not built from this HF run overlaps")
    resolved_beta = (
        float(run.interaction_provenance.beta)
        if beta is None and run.interaction_provenance is not None
        else (1.0 if beta is None else float(beta))
    )
    if bool(require_provenance):
        _validate_interaction_provenance(
            run,
            resolved_context,
            beta=resolved_beta,
        )
    components = build_rlg_hbn_hf_c3_quotient_interaction_components(
        blocks,
        resolved_context,
        v0=run.state.v0,
        beta=resolved_beta,
    )
    provenance = {
        "hf_interaction_convention": RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
        "source_provenance_validated": bool(require_provenance),
        "response_scope": "q0_dense_stored_density_derivative",
        "q_shift": [0, 0],
        "role": str(tangent.role),
        "target_k_count": int(target_k.size),
        "source_k_count": int(source_k.size),
        "fixed_source_count": int(len(resolved_context.fixed_sources)),
        "ordinary_fock_key_count": int(len(resolved_context.ordinary_fock_weights)),
        "beta": resolved_beta,
        "generic_finite_q_matrix_api": (
            "build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs"
        ),
    }
    return RLGhBNFiniteQResponse(
        hartree=np.asarray(components.hartree, dtype=np.complex128),
        fock=np.asarray(components.fock, dtype=np.complex128),
        total=np.asarray(components.total, dtype=np.complex128),
        provenance=provenance,
    )


__all__ = [
    "RLGhBNFiniteQDensityTangent",
    "RLGhBNFiniteQDensityTangentRole",
    "RLGhBNFiniteQResponse",
    "apply_rlg_hbn_hf_quotient_response",
    "validate_rlg_hbn_hf_quotient_source_closure",
]
