from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from ._hf_basis import RLG_HBN_REMOTE_H0_POLICY_VERSION
from ._hf_interaction_path import (
    RLG_HBN_HF_PHYSICAL_SHIFT_POLICY_VERSION,
    RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION,
    interaction_shifts_for_cutoff,
    build_rlg_hbn_interaction_components,
)
from ._hf_types import RLGhBNHartreeFockRun
from ._hf_interaction_provider import RLGhBNTrackPInteractionProvider
from ._finite_q_geometry import (
    _mesh_shape_from_k_grid_frac,
    _shift_k_index_with_wrap,
)


RLGhBNFiniteQDensityTangentRole = Literal["ph", "hp"]


@dataclass(frozen=True)
class RLGhBNFiniteQDensityTangent:
    """Stored-density tangent connecting source and target momentum fibers.

    RLG/hBN HF stores ``ΔD[a,b]=<c_a^† c_b>-R[a,b]``. For q=0, a ph/X
    tangent therefore has orbital-basis entry ``D[h,p]=1`` while an hp/Y
    tangent has ``D[p,h]=1``. Track-P exposes a signed finite-q column action
    on this tangent; the fixed-quotient response remains q=0-only.
    ``target_k`` is density axis 0 (dagger) and ``source_k`` is density axis 1
    (annihilation), so the endpoint arrays cannot collapse source/target k.
    Each finite-q tangent must be role-homogeneous: a tangent contains either
    ph or hp columns, never a mixed ph/hp dense recomposition. Track-P finite-q
    does not establish a global dense scalar pairing.
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


def _validated_single_representative_finite_q_tangent(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    tuple[int, int],
    np.ndarray,
    np.ndarray,
]:
    """Validate one dense stored-density q-sector without reducing raw q aliases.

    ``target_k`` names the momentum fiber of density axis 0 (the dagger
    index), and ``source_k`` names density axis 1 (the annihilation index).
    For ``role='ph'`` these are ``k`` and ``k+q``; for ``role='hp'`` they are
    ``k-q`` and ``k``. The returned response is always the +q Hamiltonian
    block with output axis-0 fibers ``k+q`` and axis-1 fibers ``k``.
    """

    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    q_values = _exact_integer_vector(tangent.q_shift, name="q_shift")
    if q_values.shape != (2,):
        raise ValueError(f"q_shift must have shape (2,), got {q_values.shape}")
    q_shift = (int(q_values[0]), int(q_values[1]))
    for axis, (value, size) in enumerate(zip(q_shift, mesh_shape, strict=True)):
        limit = int(size) // 2
        if value < -limit or value > limit:
            raise ValueError(
                "finite-q response requires a signed first-zone mesh shift; "
                f"axis={axis} value={value} allowed=[{-limit},{limit}]"
            )
    if tangent.role not in ("ph", "hp"):
        raise ValueError(f"tangent role must be 'ph' or 'hp', got {tangent.role!r}")
    blocks = np.asarray(tangent.blocks, dtype=np.complex128)
    expected_prefix = (run.basis_data.nt, run.basis_data.nt, run.basis_data.nk)
    if blocks.ndim not in (3, 4) or blocks.shape[:3] != expected_prefix:
        raise ValueError(
            "finite-q tangent blocks must have shape "
            f"{expected_prefix} or {expected_prefix}+(n_rhs,), got {blocks.shape}"
        )
    if blocks.ndim == 4 and blocks.shape[3] <= 0:
        raise ValueError("finite-q tangent batch must contain at least one RHS")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("finite-q tangent blocks must be finite")
    target_k = _exact_integer_vector(tangent.target_k, name="target_k")
    source_k = _exact_integer_vector(tangent.source_k, name="source_k")
    expected_k = np.arange(run.basis_data.nk, dtype=int)
    shifted_plus = np.empty_like(expected_k)
    shifted_minus = np.empty_like(expected_k)
    wrap_plus = np.empty((expected_k.size, 2), dtype=int)
    wrap_minus = np.empty((expected_k.size, 2), dtype=int)
    minus_shift = (-q_shift[0], -q_shift[1])
    for index in expected_k:
        shifted_plus[index], wrap_plus[index] = _shift_k_index_with_wrap(
            int(index), q_shift, mesh_shape
        )
        shifted_minus[index], wrap_minus[index] = _shift_k_index_with_wrap(
            int(index), minus_shift, mesh_shape
        )
    if tangent.role == "ph":
        expected_target = expected_k
        expected_source = shifted_plus
    else:
        expected_target = shifted_minus
        expected_source = expected_k
    if not np.array_equal(target_k, expected_target):
        raise ValueError(
            "finite-q tangent target_k must enumerate density axis-0 fibers "
            f"for role={tangent.role!r}"
        )
    if not np.array_equal(source_k, expected_source):
        raise ValueError(
            "finite-q tangent source_k must enumerate density axis-1 fibers "
            f"for role={tangent.role!r}"
        )
    return (
        blocks,
        target_k,
        source_k,
        q_shift,
        mesh_shape,
        wrap_plus,
        wrap_minus,
    )


def _single_representative_finite_q_response_components(
    run: RLGhBNHartreeFockRun,
    blocks: np.ndarray,
    *,
    q_shift: tuple[int, int],
    role: RLGhBNFiniteQDensityTangentRole,
    wrap_plus: np.ndarray,
    wrap_minus: np.ndarray,
    physical_shifts: tuple[tuple[int, int], ...],
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract one fixed-G stored-density q-sector into its +q HF response."""

    nk = int(run.basis_data.nk)
    nt = int(run.basis_data.nt)
    squeeze_rhs = blocks.ndim == 3
    batched_blocks = blocks[..., None] if squeeze_rhs else blocks
    n_rhs = int(batched_blocks.shape[3])
    overlap_by_shift = {
        tuple(int(value) for value in key): np.asarray(data, dtype=np.complex128)
        for key, data in run.overlap_blocks.layer_overlaps.items()
    }
    kernel_by_shift = {
        tuple(int(value) for value in key): np.asarray(data, dtype=float)
        for key, data in run.overlap_blocks.fock_layer_coulomb.items()
    }
    plus_k = np.empty(nk, dtype=int)
    minus_k = np.empty(nk, dtype=int)
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    for index in range(nk):
        plus_k[index], _ = _shift_k_index_with_wrap(index, q_shift, mesh_shape)
        minus_k[index], _ = _shift_k_index_with_wrap(
            index, (-q_shift[0], -q_shift[1]), mesh_shape
        )
    hartree = np.zeros((nt, nt, nk, n_rhs), dtype=np.complex128)
    fock = np.zeros_like(hartree)
    active_sources = np.nonzero(
        np.any(
            np.max(np.abs(batched_blocks), axis=(0, 1)) > 0.0,
            axis=1,
        )
    )[0]
    scale = float(beta) * float(run.state.v0) / float(nk)
    missing_overlaps: set[tuple[int, int]] = set()
    missing_kernels: set[tuple[int, int]] = set()

    def add_shift(
        left: tuple[int, int], right: tuple[int, int]
    ) -> tuple[int, int]:
        return left[0] + right[0], left[1] + right[1]

    def sub_shift(
        left: tuple[int, int], right: tuple[int, int]
    ) -> tuple[int, int]:
        return left[0] - right[0], left[1] - right[1]

    for physical_shift in physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        source_trace: np.ndarray | None = None
        for source_base_raw in active_sources:
            source_base = int(source_base_raw)
            if role == "ph":
                input_key = add_shift(
                    g0,
                    tuple(int(value) for value in wrap_plus[source_base]),
                )
                input_axis0_k = source_base
                input_axis1_k = int(plus_k[source_base])
            else:
                input_key = sub_shift(
                    g0,
                    tuple(int(value) for value in wrap_minus[source_base]),
                )
                input_axis0_k = int(minus_k[source_base])
                input_axis1_k = source_base
            input_overlap = overlap_by_shift.get(input_key)
            if input_overlap is None:
                missing_overlaps.add(input_key)
                continue
            trace = np.einsum(
                "abr,mba->mr",
                batched_blocks[:, :, source_base, :],
                np.conj(
                    input_overlap[
                        :, :, input_axis1_k, :, input_axis0_k
                    ]
                ),
                optimize=True,
            )
            source_trace = trace if source_trace is None else source_trace + trace
        if source_trace is None:
            continue

        for target_base in range(nk):
            output_axis0_k = int(plus_k[target_base])
            output_axis1_k = target_base
            output_key = add_shift(
                g0,
                tuple(int(value) for value in wrap_plus[target_base]),
            )
            output_overlap = overlap_by_shift.get(output_key)
            output_kernel = kernel_by_shift.get(output_key)
            if output_overlap is None:
                missing_overlaps.add(output_key)
            if output_kernel is None:
                missing_kernels.add(output_key)
            if output_overlap is None or output_kernel is None:
                continue
            out_form = output_overlap[
                :, :, output_axis0_k, :, output_axis1_k
            ]
            kernel = np.asarray(
                output_kernel[output_axis0_k, output_axis1_k], dtype=float
            )
            hartree[:, :, target_base, :] += scale * np.einsum(
                "lm,lab,mr->abr",
                kernel,
                out_form,
                source_trace,
                optimize=True,
            )

            for source_base_raw in active_sources:
                source_base = int(source_base_raw)
                if role == "ph":
                    input_axis0_k = source_base
                    input_axis1_k = int(plus_k[source_base])
                    left_key = add_shift(
                        g0,
                        sub_shift(
                            tuple(int(value) for value in wrap_plus[target_base]),
                            tuple(int(value) for value in wrap_plus[source_base]),
                        ),
                    )
                    right_key = g0
                else:
                    input_axis0_k = int(minus_k[source_base])
                    input_axis1_k = source_base
                    left_key = output_key
                    right_key = sub_shift(
                        g0,
                        tuple(int(value) for value in wrap_minus[source_base]),
                    )
                left_overlap = overlap_by_shift.get(left_key)
                right_overlap = overlap_by_shift.get(right_key)
                left_kernel = kernel_by_shift.get(left_key)
                if left_overlap is None:
                    missing_overlaps.add(left_key)
                if left_kernel is None:
                    missing_kernels.add(left_key)
                if left_overlap is None or left_kernel is None:
                    continue
                if right_overlap is None:
                    missing_overlaps.add(right_key)
                    continue
                left_form = left_overlap[
                    :, :, output_axis0_k, :, input_axis1_k
                ]
                right_form = right_overlap[
                    :, :, output_axis1_k, :, input_axis0_k
                ]
                kernel = np.asarray(
                    left_kernel[output_axis0_k, input_axis1_k], dtype=float
                )
                fock[:, :, target_base, :] -= scale * np.einsum(
                    "lm,lab,mcd,dbr->acr",
                    kernel,
                    left_form,
                    np.conj(right_form),
                    batched_blocks[:, :, source_base, :],
                    optimize=True,
                )
    if missing_overlaps or missing_kernels:
        raise ValueError(
            "finite-q single-representative response requires missing cache "
            "entries: "
            f"overlap_shifts={sorted(missing_overlaps)[:20]}, "
            f"kernel_shifts={sorted(missing_kernels)[:20]}"
        )
    if squeeze_rhs:
        return hartree[..., 0], fock[..., 0]
    return hartree, fock


def _single_representative_projected_finite_q_response_components(
    run: RLGhBNHartreeFockRun,
    blocks: np.ndarray,
    *,
    q_shift: tuple[int, int],
    role: RLGhBNFiniteQDensityTangentRole,
    wrap_plus: np.ndarray,
    wrap_minus: np.ndarray,
    physical_shifts: tuple[tuple[int, int], ...],
    beta: float,
    output_bra: np.ndarray,
    output_ket: np.ndarray,
    output_base_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the same Track-P response during contraction, before nt² output."""

    nk = int(run.basis_data.nk)
    squeeze_rhs = blocks.ndim == 3
    batched_blocks = blocks[..., None] if squeeze_rhs else blocks
    n_rhs = int(batched_blocks.shape[3])
    n_output = int(output_base_k.size)
    overlap_by_shift = {
        tuple(int(value) for value in key): np.asarray(data, dtype=np.complex128)
        for key, data in run.overlap_blocks.layer_overlaps.items()
    }
    kernel_by_shift = {
        tuple(int(value) for value in key): np.asarray(data, dtype=float)
        for key, data in run.overlap_blocks.fock_layer_coulomb.items()
    }
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    plus_k = np.empty(nk, dtype=int)
    minus_k = np.empty(nk, dtype=int)
    for index in range(nk):
        plus_k[index], _ = _shift_k_index_with_wrap(index, q_shift, mesh_shape)
        minus_k[index], _ = _shift_k_index_with_wrap(
            index, (-q_shift[0], -q_shift[1]), mesh_shape
        )
    active_sources = np.nonzero(
        np.any(
            np.max(np.abs(batched_blocks), axis=(0, 1)) > 0.0,
            axis=1,
        )
    )[0]
    active_rhs_by_source = {
        int(source): np.nonzero(
            np.max(
                np.abs(batched_blocks[:, :, int(source), :]), axis=(0, 1)
            )
            > 0.0
        )[0]
        for source in active_sources
    }
    hartree = np.zeros((n_output, n_rhs), dtype=np.complex128)
    fock = np.zeros_like(hartree)
    scale = float(beta) * float(run.state.v0) / float(nk)
    missing_overlaps: set[tuple[int, int]] = set()
    missing_kernels: set[tuple[int, int]] = set()

    def add_shift(left, right):
        return int(left[0]) + int(right[0]), int(left[1]) + int(right[1])

    def sub_shift(left, right):
        return int(left[0]) - int(right[0]), int(left[1]) - int(right[1])

    output_indices_by_base = {
        base: np.nonzero(output_base_k == base)[0] for base in range(nk)
    }
    for physical_shift in physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        source_trace: np.ndarray | None = None
        for source_raw in active_sources:
            source_base = int(source_raw)
            if role == "ph":
                input_key = add_shift(g0, wrap_plus[source_base])
                input_axis0_k = source_base
                input_axis1_k = int(plus_k[source_base])
            else:
                input_key = sub_shift(g0, wrap_minus[source_base])
                input_axis0_k = int(minus_k[source_base])
                input_axis1_k = source_base
            input_overlap = overlap_by_shift.get(input_key)
            if input_overlap is None:
                missing_overlaps.add(input_key)
                continue
            trace = np.einsum(
                "abr,mba->mr",
                batched_blocks[:, :, source_base, :],
                np.conj(
                    input_overlap[
                        :, :, input_axis1_k, :, input_axis0_k
                    ]
                ),
                optimize=True,
            )
            source_trace = trace if source_trace is None else source_trace + trace
        if source_trace is None:
            continue

        for target_base in range(nk):
            output_indices = output_indices_by_base[target_base]
            if output_indices.size == 0:
                continue
            output_axis0_k = int(plus_k[target_base])
            output_axis1_k = target_base
            output_key = add_shift(g0, wrap_plus[target_base])
            output_overlap = overlap_by_shift.get(output_key)
            output_kernel = kernel_by_shift.get(output_key)
            if output_overlap is None:
                missing_overlaps.add(output_key)
            if output_kernel is None:
                missing_kernels.add(output_key)
            if output_overlap is None or output_kernel is None:
                continue
            output_form = np.einsum(
                "ai,lab,bi->li",
                np.conj(output_bra[:, output_indices]),
                output_overlap[
                    :, :, output_axis0_k, :, output_axis1_k
                ],
                output_ket[:, output_indices],
                optimize=True,
            )
            kernel = np.asarray(
                output_kernel[output_axis0_k, output_axis1_k], dtype=float
            )
            hartree[output_indices, :] += scale * np.einsum(
                "lm,li,mr->ir",
                kernel,
                output_form,
                source_trace,
                optimize=True,
            )

            for source_raw in active_sources:
                source_base = int(source_raw)
                active_rhs = active_rhs_by_source[source_base]
                if active_rhs.size == 0:
                    continue
                if role == "ph":
                    input_axis0_k = source_base
                    input_axis1_k = int(plus_k[source_base])
                    left_key = add_shift(
                        g0,
                        sub_shift(
                            wrap_plus[target_base], wrap_plus[source_base]
                        ),
                    )
                    right_key = g0
                else:
                    input_axis0_k = int(minus_k[source_base])
                    input_axis1_k = source_base
                    left_key = output_key
                    right_key = sub_shift(g0, wrap_minus[source_base])
                left_overlap = overlap_by_shift.get(left_key)
                right_overlap = overlap_by_shift.get(right_key)
                left_kernel = kernel_by_shift.get(left_key)
                if left_overlap is None:
                    missing_overlaps.add(left_key)
                if left_kernel is None:
                    missing_kernels.add(left_key)
                if left_overlap is None or left_kernel is None:
                    continue
                if right_overlap is None:
                    missing_overlaps.add(right_key)
                    continue
                left_projected = np.einsum(
                    "ai,lab->lib",
                    np.conj(output_bra[:, output_indices]),
                    left_overlap[
                        :, :, output_axis0_k, :, input_axis1_k
                    ],
                    optimize=True,
                )
                right_projected = np.einsum(
                    "ci,mcd->mid",
                    np.conj(output_ket[:, output_indices]),
                    right_overlap[
                        :, :, output_axis1_k, :, input_axis0_k
                    ],
                    optimize=True,
                )
                kernel = np.asarray(
                    left_kernel[output_axis0_k, input_axis1_k], dtype=float
                )
                fock[np.ix_(output_indices, active_rhs)] -= scale * np.einsum(
                    "lm,lib,mid,dbr->ir",
                    kernel,
                    left_projected,
                    np.conj(right_projected),
                    batched_blocks[
                        :, :, source_base, active_rhs
                    ],
                    optimize=True,
                )
    if missing_overlaps or missing_kernels:
        raise ValueError(
            "projected finite-q single-representative response requires missing "
            "cache entries: "
            f"overlap_shifts={sorted(missing_overlaps)[:20]}, "
            f"kernel_shifts={sorted(missing_kernels)[:20]}"
        )
    if squeeze_rhs:
        return hartree[:, 0], fock[:, 0]
    return hartree, fock


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


def validate_rlg_hbn_hf_single_representative_provenance(
    run: RLGhBNHartreeFockRun,
    *,
    beta: float | None = None,
    physical_shifts: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, object]:
    """Fail closed unless ``run`` uses the explicitly typed hybrid functional.

    The current candidate combines the C3-repaired frozen-remote one-body term
    with a fixed-|G|, one-stored-torus-representative active interaction. These
    two policies are recorded separately rather than hidden behind a generic
    ``single representative`` label.
    """

    provenance = run.interaction_provenance
    if provenance is None:
        raise ValueError(
            "HF run has no typed interaction provenance; an untyped legacy archive "
            "cannot be promoted to the single-representative production chain"
        )
    expected_shifts = interaction_shifts_for_cutoff(
        run.basis_data.basis_model.lattice,
        run.basis_data.interaction,
    )
    restored_shifts = tuple(
        (int(value[0]), int(value[1])) for value in run.overlap_blocks.shifts
    )
    supplied_shifts = (
        expected_shifts
        if physical_shifts is None
        else tuple((int(value[0]), int(value[1])) for value in physical_shifts)
    )
    resolved_beta = float(provenance.beta) if beta is None else float(beta)
    mismatches: dict[str, object] = {}
    provider = run.track_p_provider
    if not isinstance(provider, RLGhBNTrackPInteractionProvider):
        mismatches["track_p_provider"] = type(provider).__name__
    else:
        if provider.basis_data is not run.basis_data:
            mismatches["provider_basis_identity"] = False
        if provider.overlap_blocks is not run.overlap_blocks:
            mismatches["provider_overlap_identity"] = False
        if not np.isclose(
            provider.beta, resolved_beta, rtol=0.0, atol=1.0e-15
        ):
            mismatches["provider_beta"] = provider.beta
        if provenance.provider_schema_version not in (0, 1):
            mismatches["provider_schema_version"] = (
                provenance.provider_schema_version
            )
        if provenance.provider_schema_version == 1:
            if not provenance.provider_fingerprint:
                mismatches["provider_fingerprint"] = "blank"
            elif provenance.provider_fingerprint != provider.fingerprint:
                mismatches["provider_fingerprint"] = provider.fingerprint
        elif (
            provenance.provider_fingerprint
            and provenance.provider_fingerprint != provider.fingerprint
        ):
            mismatches["provider_fingerprint"] = provider.fingerprint
        provider.validate_state(run.state)
    if (
        provenance.convention
        != RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
    ):
        mismatches["convention"] = provenance.convention
    if provenance.quotient_enabled:
        mismatches["quotient_enabled"] = provenance.quotient_enabled
    if not np.isfinite(float(provenance.beta)) or not np.isfinite(resolved_beta):
        mismatches["beta_finite"] = provenance.beta
    elif not np.isclose(
        float(provenance.beta), resolved_beta, rtol=0.0, atol=1.0e-15
    ):
        mismatches["beta"] = provenance.beta
    if restored_shifts != expected_shifts:
        mismatches["restored_physical_shifts"] = restored_shifts
    if provenance.physical_shifts != expected_shifts:
        mismatches["archive_physical_shifts"] = provenance.physical_shifts
    if supplied_shifts != expected_shifts:
        mismatches["supplied_physical_shifts"] = supplied_shifts
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
    if provenance.remote_h0_policy != RLG_HBN_REMOTE_H0_POLICY_VERSION:
        mismatches["remote_h0_policy"] = provenance.remote_h0_policy
    if run.basis_data.fixed_remote_hamiltonian is None:
        mismatches["fixed_remote_hamiltonian"] = None
    else:
        remote_h0_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                run.basis_data.fixed_remote_hamiltonian,
                dtype=np.complex128,
            ).view(np.uint8)
        ).hexdigest()
        if provenance.remote_h0_sha256 != remote_h0_sha256:
            mismatches["remote_h0_sha256"] = provenance.remote_h0_sha256
    if (
        provenance.physical_shift_policy
        != RLG_HBN_HF_PHYSICAL_SHIFT_POLICY_VERSION
    ):
        mismatches["physical_shift_policy"] = provenance.physical_shift_policy
    if mismatches:
        raise ValueError(
            "HF run interaction provenance does not match the fixed-G active / "
            f"C3-repaired-remote response kernel: {mismatches}"
        )
    return {
        "hf_interaction_convention": provenance.convention,
        "remote_h0_policy": provenance.remote_h0_policy,
        "remote_h0_sha256": provenance.remote_h0_sha256,
        "physical_shift_policy": provenance.physical_shift_policy,
        "source_provenance_validated": True,
        "beta": resolved_beta,
        "physical_shift_count": len(expected_shifts),
        "provider_fingerprint": provider.fingerprint,
    }


def _require_attached_track_p_provider(
    run: RLGhBNHartreeFockRun,
    *,
    beta: float,
) -> RLGhBNTrackPInteractionProvider:
    provider = run.track_p_provider
    if not isinstance(provider, RLGhBNTrackPInteractionProvider):
        raise ValueError("Track-P response requires an attached provider")
    if provider.basis_data is not run.basis_data:
        raise ValueError("Track-P provider is bound to a different basis object")
    if provider.overlap_blocks is not run.overlap_blocks:
        raise ValueError("Track-P provider is stale for the run overlap object")
    if not np.isclose(provider.beta, float(beta), rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"Track-P provider beta mismatch: {provider.beta} != {float(beta)}"
        )
    provider.validate_state(run.state)
    return provider


def validate_rlg_hbn_hf_single_representative_source_closure(
    run: RLGhBNHartreeFockRun,
    *,
    closure_tolerance_mev: float = 1.0e-3,
    stationarity_tolerance_mev: float = 1.0e-3,
) -> dict[str, float | str]:
    """Validate saved-H closure and stationarity for the typed hybrid functional."""

    if not run.converged:
        raise ValueError(
            "single-representative source closure requires a converged HF run"
        )
    provenance_metrics = validate_rlg_hbn_hf_single_representative_provenance(run)
    provenance = run.interaction_provenance
    assert provenance is not None
    for name, values in (
        ("state.h0", run.state.h0),
        ("basis_data.h0", run.basis_data.h0),
        ("state.density", run.state.density),
        ("state.hamiltonian", run.state.hamiltonian),
        ("state.reference_density", run.state.reference_density),
    ):
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError(f"single-representative source contains nonfinite {name}")
    h0_basis_residual = float(
        np.max(
            np.abs(
                np.asarray(run.state.h0, dtype=np.complex128)
                - np.asarray(run.basis_data.h0, dtype=np.complex128)
            )
        )
    )
    provider = _require_attached_track_p_provider(
        run,
        beta=float(provenance.beta),
    )
    components = provider.scf_components(run.state.density)
    rebuilt = np.asarray(run.state.h0) + np.asarray(components.total)
    closure = float(
        np.max(np.abs(np.asarray(run.state.hamiltonian) - rebuilt))
    )
    stored_projector = np.asarray(run.state.density) + np.asarray(
        run.state.reference_density
    )
    projector = stored_projector.transpose(1, 0, 2)
    stationarity_values = []
    for index in range(run.state.nk):
        hamiltonian = np.asarray(run.state.hamiltonian)[:, :, index]
        block = projector[:, :, index]
        stationarity_values.append(
            float(
                np.max(
                    np.abs(hamiltonian @ block - block @ hamiltonian)
                )
            )
        )
    stationarity = float(max(stationarity_values, default=0.0))
    metrics: dict[str, float | str] = {
        "hf_interaction_convention": str(
            provenance_metrics["hf_interaction_convention"]
        ),
        "remote_h0_policy": str(provenance_metrics["remote_h0_policy"]),
        "physical_shift_policy": str(
            provenance_metrics["physical_shift_policy"]
        ),
        "hamiltonian_closure_mev": closure,
        "h0_basis_residual_mev": h0_basis_residual,
        "projector_commutator_mev": stationarity,
        "closure_tolerance_mev": float(closure_tolerance_mev),
        "stationarity_tolerance_mev": float(stationarity_tolerance_mev),
    }
    if not np.isfinite(closure) or not np.isfinite(stationarity):
        raise ValueError(
            f"single-representative source closure produced nonfinite metrics: {metrics}"
        )
    if h0_basis_residual > float(closure_tolerance_mev):
        raise ValueError(
            f"saved HF h0 differs from the restored basis h0: {metrics}"
        )
    if closure > float(closure_tolerance_mev):
        raise ValueError(
            f"saved HF Hamiltonian fails single-representative closure: {metrics}"
        )
    if stationarity > float(stationarity_tolerance_mev):
        raise ValueError(
            f"saved HF density is not stationary for the single-representative "
            f"functional: {metrics}"
        )
    return metrics


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


def apply_rlg_hbn_hf_single_representative_finite_q_response(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
    *,
    beta: float | None = None,
    require_converged: bool = True,
    require_provenance: bool = True,
) -> RLGhBNFiniteQResponse:
    """Apply the fixed-G Track-P HF derivative to one signed q-sector.

    The input stores density axis-0/axis-1 fibers explicitly. The output has
    shape ``(nt, nt, nk)`` and enumerates Hamiltonian blocks with axis 0 at
    ``k+q`` and axis 1 at ``k``. Hartree and Fock components are returned
    separately so TDHF A/B columns can be checked term by term. Each tangent
    must be role-homogeneous; mixed ph/hp dense recomposition and global scalar
    pairing are unsupported for Track-P finite-q.
    """

    if bool(require_converged) and not bool(run.converged):
        raise ValueError(
            "single-representative finite-q response requires a converged HF "
            "run unless require_converged=False is explicitly requested"
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
    (
        blocks,
        target_k,
        source_k,
        q_shift,
        mesh_shape,
        wrap_plus,
        wrap_minus,
    ) = _validated_single_representative_finite_q_tangent(run, tangent)
    resolved_beta = (
        float(run.interaction_provenance.beta)
        if beta is None and run.interaction_provenance is not None
        else (1.0 if beta is None else float(beta))
    )
    if bool(require_provenance):
        validate_rlg_hbn_hf_single_representative_provenance(
            run,
            beta=resolved_beta,
        )
    provider = _require_attached_track_p_provider(
        run,
        beta=resolved_beta,
    )
    physical_shifts = provider.physical_shifts
    hartree, fock = _single_representative_finite_q_response_components(
        run,
        blocks,
        q_shift=q_shift,
        role=tangent.role,
        wrap_plus=wrap_plus,
        wrap_minus=wrap_minus,
        physical_shifts=physical_shifts,
        beta=resolved_beta,
    )
    total = hartree + fock
    output_axis1_k = np.arange(run.basis_data.nk, dtype=int)
    output_axis0_k = np.empty_like(output_axis1_k)
    for index in output_axis1_k:
        output_axis0_k[index], _ = _shift_k_index_with_wrap(
            int(index), q_shift, mesh_shape
        )
    provenance = {
        "hf_interaction_convention": (
            RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
        ),
        "source_provenance_validated": bool(require_provenance),
        "response_scope": "track_p_role_resolved_dense_column_action_v2",
        "role_dependent_kernel": True,
        "projected_ab_column_action": False,
        "cross_role_dense_recomposition_authorized": False,
        "global_dense_scalar_extension": "not_established",
        "pairing_adjointness_scope": "assembled_signed_ab_pair",
        "response_cache_fingerprint": provider.response_cache_fingerprint,
        "q_shift_raw": [int(q_shift[0]), int(q_shift[1])],
        "q_shift_torus": [
            int(q_shift[0]) % int(mesh_shape[0]),
            int(q_shift[1]) % int(mesh_shape[1]),
        ],
        "mesh_shape": [int(mesh_shape[0]), int(mesh_shape[1])],
        "role": str(tangent.role),
        "density_axis0_k": target_k.tolist(),
        "density_axis1_k": source_k.tolist(),
        "output_axis0_k": output_axis0_k.tolist(),
        "output_axis1_k": output_axis1_k.tolist(),
        "physical_shift_count": len(physical_shifts),
        "physical_shifts": [list(value) for value in physical_shifts],
        "beta": resolved_beta,
        "provider_fingerprint": provider.fingerprint,
        "post_assembly_averaging": False,
    }
    return RLGhBNFiniteQResponse(
        hartree=hartree,
        fock=fock,
        total=total,
        provenance=provenance,
    )


def project_rlg_hbn_hf_single_representative_finite_q_response(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
    *,
    output_bra: np.ndarray,
    output_ket: np.ndarray,
    output_base_k: np.ndarray,
    beta: float | None = None,
    require_converged: bool = True,
    require_provenance: bool = True,
) -> RLGhBNFiniteQResponse:
    """Apply and project the Track-P response without materializing nt² outputs.

    For output ``i``, ``output_base_k[i]=k``: ``output_bra[:, i]`` belongs to
    the Hamiltonian row/dagger fiber ``k+q`` and ``output_ket[:, i]`` belongs
    to its column/annihilation fiber ``k``. The projection is a role-resolved
    A/B column action; mixed ph/hp dense recomposition and global scalar
    pairing are unsupported for Track-P finite-q.
    """

    if bool(require_converged) and not bool(run.converged):
        raise ValueError(
            "projected finite-q response requires a converged HF run unless "
            "require_converged=False is explicitly requested"
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
    (
        blocks,
        target_k,
        source_k,
        q_shift,
        mesh_shape,
        wrap_plus,
        wrap_minus,
    ) = _validated_single_representative_finite_q_tangent(run, tangent)
    bra = np.asarray(output_bra, dtype=np.complex128)
    ket = np.asarray(output_ket, dtype=np.complex128)
    if bra.ndim != 2 or bra.shape[0] != run.basis_data.nt:
        raise ValueError(
            "output_bra must have shape (nt, n_output), got "
            f"{bra.shape}"
        )
    if ket.shape != bra.shape:
        raise ValueError(
            f"output_ket shape {ket.shape} does not match output_bra {bra.shape}"
        )
    if not np.all(np.isfinite(bra)) or not np.all(np.isfinite(ket)):
        raise ValueError("output projection vectors must be finite")
    base_k = _exact_integer_vector(output_base_k, name="output_base_k")
    if base_k.shape != (bra.shape[1],):
        raise ValueError(
            "output_base_k must have one entry per output projection, got "
            f"{base_k.shape} for n_output={bra.shape[1]}"
        )
    if np.any(base_k < 0) or np.any(base_k >= run.basis_data.nk):
        raise ValueError("output_base_k contains an index outside the stored mesh")
    resolved_beta = (
        float(run.interaction_provenance.beta)
        if beta is None and run.interaction_provenance is not None
        else (1.0 if beta is None else float(beta))
    )
    if bool(require_provenance):
        validate_rlg_hbn_hf_single_representative_provenance(
            run,
            beta=resolved_beta,
        )
    provider = _require_attached_track_p_provider(
        run,
        beta=resolved_beta,
    )
    physical_shifts = provider.physical_shifts
    hartree, fock = _single_representative_projected_finite_q_response_components(
        run,
        blocks,
        q_shift=q_shift,
        role=tangent.role,
        wrap_plus=wrap_plus,
        wrap_minus=wrap_minus,
        physical_shifts=physical_shifts,
        beta=resolved_beta,
        output_bra=bra,
        output_ket=ket,
        output_base_k=base_k,
    )
    provenance = {
        "hf_interaction_convention": (
            RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
        ),
        "source_provenance_validated": bool(require_provenance),
        "response_scope": "track_p_role_resolved_projected_ab_column_action_v2",
        "role_dependent_kernel": True,
        "projected_ab_column_action": True,
        "cross_role_dense_recomposition_authorized": False,
        "global_dense_scalar_extension": "not_established",
        "pairing_adjointness_scope": "assembled_signed_ab_pair",
        "response_cache_fingerprint": provider.response_cache_fingerprint,
        "q_shift_raw": [int(q_shift[0]), int(q_shift[1])],
        "q_shift_torus": [
            int(q_shift[0]) % int(mesh_shape[0]),
            int(q_shift[1]) % int(mesh_shape[1]),
        ],
        "role": str(tangent.role),
        "density_axis0_k": target_k.tolist(),
        "density_axis1_k": source_k.tolist(),
        "output_base_k": base_k.tolist(),
        "output_count": int(base_k.size),
        "physical_shift_count": len(physical_shifts),
        "beta": resolved_beta,
        "provider_fingerprint": provider.fingerprint,
        "post_assembly_averaging": False,
    }
    return RLGhBNFiniteQResponse(
        hartree=hartree,
        fock=fock,
        total=hartree + fock,
        provenance=provenance,
    )


def apply_rlg_hbn_hf_single_representative_response(
    run: RLGhBNHartreeFockRun,
    tangent: RLGhBNFiniteQDensityTangent,
    *,
    beta: float | None = None,
    require_converged: bool = True,
    require_provenance: bool = True,
) -> RLGhBNFiniteQResponse:
    """Apply the derivative of the fixed-G single-representative HF functional."""

    if bool(require_converged) and not bool(run.converged):
        raise ValueError(
            "single-representative response requires a converged HF run unless "
            "require_converged=False is explicitly requested"
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
    resolved_beta = (
        float(run.interaction_provenance.beta)
        if beta is None and run.interaction_provenance is not None
        else (1.0 if beta is None else float(beta))
    )
    if bool(require_provenance):
        validate_rlg_hbn_hf_single_representative_provenance(
            run,
            beta=resolved_beta,
        )
    provider = _require_attached_track_p_provider(
        run,
        beta=resolved_beta,
    )
    components = provider.tangent_components(blocks)
    provenance = {
        "hf_interaction_convention": (
            RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
        ),
        "source_provenance_validated": bool(require_provenance),
        "response_scope": "q0_dense_stored_density_derivative",
        "response_cache_fingerprint": provider.response_cache_fingerprint,
        "q_shift": [0, 0],
        "role": str(tangent.role),
        "target_k_count": int(target_k.size),
        "source_k_count": int(source_k.size),
        "physical_shift_count": len(run.overlap_blocks.shifts),
        "beta": resolved_beta,
        "provider_fingerprint": provider.fingerprint,
        "generic_finite_q_matrix_api": (
            "build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs"
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
    "apply_rlg_hbn_hf_single_representative_finite_q_response",
    "apply_rlg_hbn_hf_single_representative_response",
    "project_rlg_hbn_hf_single_representative_finite_q_response",
    "validate_rlg_hbn_hf_quotient_source_closure",
    "validate_rlg_hbn_hf_single_representative_provenance",
    "validate_rlg_hbn_hf_single_representative_source_closure",
]
