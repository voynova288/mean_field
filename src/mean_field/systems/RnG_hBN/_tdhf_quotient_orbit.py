from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mean_field.core.hf import ParticleHolePair, TDHFMatrices, TDHFStructureResiduals

from ._hf_interaction_path import build_rlg_hbn_layer_overlap_blocks
from ._hf_types import (
    RLGhBNHartreeFockRun,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
)
from ._tdhf_finite_q_terms import (
    TDHFFiniteQTerms,
    build_rlg_hbn_tdhf_finite_q_intraflavor_terms,
    required_rlg_hbn_tdhf_ws_overlap_shifts,
    sum_finite_q_terms,
)
from ._tdhf_fixed_quotient import (
    RLGhBNTDHFEnergySewing,
    RLGhBNTDHFSparseFixedSource,
    build_energy_assigned_c3_sewing,
    build_periodic_gauge_basis_view,
    build_raw_pair_c3_sewing,
    build_sparse_fixed_source,
    c3_composed_direct_physical_shell,
    c3_direct_physical_shell,
    c3_reciprocal_index,
    c3_repeated_zone_offset,
    fixed_role_masks,
    pair_excitation_energies,
    required_sparse_fixed_context_padding,
    transport_fixed_terms_from_canonical_form_factors,
)
from ._tdhf_pairs import build_rlg_hbn_tdhf_q_pairs
from ._tdhf_types import RLGhBNTDHFOrbitals


@dataclass(frozen=True)
class RLGhBNTDHFQuotientOrbitResult:
    source_shift: tuple[int, int]
    target_shift: tuple[int, int]
    source_matrices: TDHFMatrices
    target_matrices: TDHFMatrices
    terms: dict[str, TDHFFiniteQTerms]
    plus_sewing: RLGhBNTDHFEnergySewing
    minus_sewing: RLGhBNTDHFEnergySewing
    metadata: dict[str, Any]
    internals: "RLGhBNTDHFQuotientOrbitInternals | None" = None


@dataclass(frozen=True)
class RLGhBNTDHFQuotientOrbitInternals:
    run: RLGhBNHartreeFockRun
    sewing_basis: RLGhBNProjectedBasisData
    pairs: dict[str, tuple[ParticleHolePair, ...]]
    physical_shifts: tuple[tuple[int, int], ...]
    source: RLGhBNTDHFSparseFixedSource
    minus_source: RLGhBNTDHFSparseFixedSource


@dataclass(frozen=True)
class RLGhBNTDHFQuotientCycleResult:
    shifts: tuple[tuple[int, int], ...]
    matrices: dict[tuple[int, int], TDHFMatrices]
    steps: tuple[RLGhBNTDHFQuotientOrbitResult, ...]
    closure_residuals: dict[str, float]


def _c3_shift_mod(shift: tuple[int, int], mesh_size: int) -> tuple[int, int]:
    rotated = c3_reciprocal_index(shift)
    return rotated[0] % int(mesh_size), rotated[1] % int(mesh_size)


def _minus_shift(shift: tuple[int, int]) -> tuple[int, int]:
    return -int(shift[0]), -int(shift[1])


def _intraflavor_pairs(
    orbitals: RLGhBNTDHFOrbitals,
    run: RLGhBNHartreeFockRun,
    shift: tuple[int, int],
) -> tuple[ParticleHolePair, ...]:
    result: list[ParticleHolePair] = []
    for pair in build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, shift):
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if particle is None or hole is None:
            raise ValueError("finite-q pairs must carry flavor metadata")
        if particle.spin == hole.spin and particle.valley == hole.valley:
            result.append(pair)
    return tuple(result)


def _merge_overlap_blocks(
    base: RLGhBNLayerOverlapBlockSet,
    extra: RLGhBNLayerOverlapBlockSet,
) -> RLGhBNLayerOverlapBlockSet:
    return RLGhBNLayerOverlapBlockSet(
        shifts=base.shifts,
        gvecs=base.gvecs,
        layer_overlaps={**base.layer_overlaps, **extra.layer_overlaps},
        layer_diagonal_overlaps={
            **base.layer_diagonal_overlaps,
            **extra.layer_diagonal_overlaps,
        },
        hartree_layer_coulomb={
            **base.hartree_layer_coulomb,
            **extra.hartree_layer_coulomb,
        },
        fock_layer_coulomb={
            **base.fock_layer_coulomb,
            **extra.fock_layer_coulomb,
        },
    )


def _run_with_overlap_keys(
    run: RLGhBNHartreeFockRun,
    required_keys: tuple[tuple[int, int], ...],
) -> tuple[RLGhBNHartreeFockRun, tuple[tuple[int, int], ...]]:
    available = {
        (int(key[0]), int(key[1]))
        for key in run.overlap_blocks.layer_overlaps
    }
    missing = tuple(key for key in required_keys if key not in available)
    if not missing:
        return run, ()
    extra = build_rlg_hbn_layer_overlap_blocks(run.basis_data, shifts=missing)
    merged = _merge_overlap_blocks(run.overlap_blocks, extra)
    provider = getattr(run, "track_p_provider", None)
    completed_provider = (
        None if provider is None else provider.with_overlap_blocks(merged)
    )
    return replace(
        run,
        overlap_blocks=merged,
        track_p_provider=completed_provider,
    ), missing


def _interaction_terms(terms: TDHFFiniteQTerms) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(terms[name], dtype=np.complex128)
        for name in ("A_direct", "A_exchange", "B_direct", "B_exchange")
    }


def _with_interactions(
    base: TDHFFiniteQTerms,
    interactions: dict[str, np.ndarray],
) -> TDHFFiniteQTerms:
    result = {name: np.array(value, copy=True) for name, value in base.items()}
    for name, value in interactions.items():
        result[name] = np.array(value, copy=True)
    return result


def _replace_source_fixed_terms(
    base_terms: TDHFFiniteQTerms,
    source: RLGhBNTDHFSparseFixedSource,
    *,
    partner_base_terms: TDHFFiniteQTerms | None = None,
) -> tuple[TDHFFiniteQTerms, dict[str, np.ndarray] | None]:
    pair_count = int(base_terms["A0"].shape[0])
    identity = np.eye(pair_count, dtype=np.complex128)
    transported = transport_fixed_terms_from_canonical_form_factors(
        _interaction_terms(base_terms),
        left_transform=identity,
        right_transform=identity,
        target_x_fixed=source.x_fixed,
        target_y_fixed=source.y_fixed,
        evaluators=source.term_evaluators,
        partner_target_terms=(
            None if partner_base_terms is None else _interaction_terms(partner_base_terms)
        ),
    )
    return _with_interactions(base_terms, transported.terms), transported.partner_terms


def _assemble_partner_liouvillian(
    plus_terms: TDHFFiniteQTerms,
    minus_terms: TDHFFiniteQTerms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus_a = sum_finite_q_terms(plus_terms, ("A0", "A_direct", "A_exchange"))
    plus_b = sum_finite_q_terms(plus_terms, ("B_direct", "B_exchange"))
    minus_a = sum_finite_q_terms(minus_terms, ("A0", "A_direct", "A_exchange"))
    minus_b = sum_finite_q_terms(minus_terms, ("B_direct", "B_exchange"))
    liouvillian = np.block(
        [
            [plus_a, plus_b],
            [-np.conj(minus_b), -np.conj(minus_a)],
        ]
    )
    return plus_a, plus_b, liouvillian


def _matrices_from_partner_terms(
    pairs: tuple[ParticleHolePair, ...],
    plus_terms: TDHFFiniteQTerms,
    minus_terms: TDHFFiniteQTerms,
    *,
    structure_tolerance: float,
) -> TDHFMatrices:
    plus_a, plus_b, liouvillian = _assemble_partner_liouvillian(
        plus_terms,
        minus_terms,
    )
    minus_a = sum_finite_q_terms(minus_terms, ("A0", "A_direct", "A_exchange"))
    minus_b = sum_finite_q_terms(minus_terms, ("B_direct", "B_exchange"))
    a_residual = max(
        float(np.max(np.abs(plus_a - plus_a.conj().T))) if plus_a.size else 0.0,
        float(np.max(np.abs(minus_a - minus_a.conj().T))) if minus_a.size else 0.0,
    )
    b_residual = (
        float(np.max(np.abs(plus_b - minus_b.T)))
        if plus_b.size
        else 0.0
    )
    structure = TDHFStructureResiduals(
        a_hermitian=a_residual,
        b_symmetric=b_residual,
        particle_hole_symmetry=0.0,
        tolerance=float(structure_tolerance),
    )
    return TDHFMatrices(
        pairs=pairs,
        A=plus_a,
        B=plus_b,
        L=liouvillian,
        structure=structure,
    )


def _pair_order_signature(
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
) -> tuple[tuple[int, int, int], ...]:
    result: list[tuple[int, int, int]] = []
    for pair in pairs:
        p_local, _ = orbitals.decode_global_index(pair.particle)
        h_local, h_k = orbitals.decode_global_index(pair.hole)
        result.append((int(p_local), int(h_local), int(h_k)))
    return tuple(result)


def build_rlg_hbn_tdhf_c3_quotient_orbit(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    source_shift: tuple[int, int],
    *,
    beta: float = 1.0,
    physical_shifts: tuple[tuple[int, int], ...] | None = None,
    fixed_copy: int = 0,
    periodic_gauge_padding: int | None = None,
    structure_tolerance: float = 1.0e-6,
    _retain_internals: bool = False,
) -> RLGhBNTDHFQuotientOrbitResult:
    """Build one q/C3q orbit with pre-assembly source-form-factor transport."""

    mesh = int(run.basis_data.mesh_size)
    q = (int(source_shift[0]) % mesh, int(source_shift[1]) % mesh)
    c3q = _c3_shift_mod(q, mesh)
    minus_q = _minus_shift(q)
    minus_c3q = _minus_shift(c3q)
    shifts = {
        "q": q,
        "c3q": c3q,
        "minus_q": minus_q,
        "minus_c3q": minus_c3q,
    }
    pairs = {
        name: _intraflavor_pairs(orbitals, run, shift)
        for name, shift in shifts.items()
    }
    pair_count = len(pairs["q"])
    if any(len(value) != pair_count for value in pairs.values()):
        raise ValueError("q/C3q and partner pair spaces have different dimensions")
    if _pair_order_signature(orbitals, pairs["q"]) != _pair_order_signature(
        orbitals,
        pairs["minus_q"],
    ):
        raise ValueError("q/-q pair ordering differs")
    if _pair_order_signature(orbitals, pairs["c3q"]) != _pair_order_signature(
        orbitals,
        pairs["minus_c3q"],
    ):
        raise ValueError("C3q/-C3q pair ordering differs")

    resolved_physical = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )
    c3q_offset = c3_repeated_zone_offset(q, c3q, mesh)
    minus_c3q_offset = c3_repeated_zone_offset(minus_q, minus_c3q, mesh)
    c3q_direct_shell = c3_direct_physical_shell(
        resolved_physical,
        repeated_zone_offset=c3q_offset,
    )
    minus_c3q_direct_shell = c3_direct_physical_shell(
        resolved_physical,
        repeated_zone_offset=minus_c3q_offset,
    )
    base_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        run,
        orbitals,
        {shifts[name]: pairs[name] for name in shifts},
        resolved_physical,
    )
    c3q_direct_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        run,
        orbitals,
        {c3q: pairs["c3q"]},
        c3q_direct_shell,
    )
    minus_c3q_direct_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        run,
        orbitals,
        {minus_c3q: pairs["minus_c3q"]},
        minus_c3q_direct_shell,
    )
    required_keys = tuple(
        sorted(set(base_keys) | set(c3q_direct_keys) | set(minus_c3q_direct_keys))
    )
    term_run, built_keys = _run_with_overlap_keys(run, required_keys)
    base_terms = {
        name: build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
            term_run,
            orbitals,
            pairs[name],
            shifts[name],
            beta=beta,
            physical_shifts=resolved_physical,
            require_complete_umklapp=True,
        )
        for name in shifts
    }
    c3q_shifted = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        term_run,
        orbitals,
        pairs["c3q"],
        c3q,
        beta=beta,
        physical_shifts=c3q_direct_shell,
        require_complete_umklapp=True,
    )
    minus_c3q_shifted = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        term_run,
        orbitals,
        pairs["minus_c3q"],
        minus_c3q,
        beta=beta,
        physical_shifts=minus_c3q_direct_shell,
        require_complete_umklapp=True,
    )
    for term_name in ("A_direct", "B_direct"):
        base_terms["c3q"][term_name] = np.array(c3q_shifted[term_name], copy=True)
        base_terms["minus_c3q"][term_name] = np.array(
            minus_c3q_shifted[term_name],
            copy=True,
        )

    q_source = build_sparse_fixed_source(
        term_run,
        orbitals,
        pairs["q"],
        q,
        resolved_physical,
        fixed_copy=fixed_copy,
        periodic_gauge_padding=periodic_gauge_padding,
        beta=beta,
    )
    minus_q_source = build_sparse_fixed_source(
        term_run,
        orbitals,
        pairs["minus_q"],
        minus_q,
        resolved_physical,
        fixed_copy=fixed_copy,
        periodic_gauge_padding=periodic_gauge_padding,
        beta=beta,
    )
    sewing_padding = max(
        int(q_source.context.periodic_gauge_padding),
        int(minus_q_source.context.periodic_gauge_padding),
    )
    sewing_basis = build_periodic_gauge_basis_view(
        term_run,
        periodic_gauge_padding=sewing_padding,
        name="rlg_hbn_tdhf_quotient_sewing_basis",
    )
    source_q_terms, source_partner = _replace_source_fixed_terms(
        base_terms["q"],
        q_source,
        partner_base_terms=base_terms["minus_q"],
    )
    source_minus_terms, _ = _replace_source_fixed_terms(
        base_terms["minus_q"],
        minus_q_source,
    )
    if source_partner is None:
        raise RuntimeError("source shared B provider did not return partner terms")
    for term_name in ("B_direct", "B_exchange"):
        source_minus_terms[term_name] = np.array(source_partner[term_name], copy=True)

    fixed_indices = {
        int(pair[0]) * mesh + int(pair[1])
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    target_x, target_y = fixed_role_masks(
        orbitals,
        pairs["c3q"],
        c3q,
        fixed_indices,
        mesh,
    )
    target_minus_x, target_minus_y = fixed_role_masks(
        orbitals,
        pairs["minus_c3q"],
        minus_c3q,
        fixed_indices,
        mesh,
    )
    sewing_vector_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray] = {}
    raw_plus = build_raw_pair_c3_sewing(
        sewing_basis,
        orbitals,
        source_pairs=pairs["q"],
        source_shift=q,
        target_pairs=pairs["c3q"],
        target_shift=c3q,
        vector_cache=sewing_vector_cache,
    )
    raw_minus = build_raw_pair_c3_sewing(
        sewing_basis,
        orbitals,
        source_pairs=pairs["minus_q"],
        source_shift=minus_q,
        target_pairs=pairs["minus_c3q"],
        target_shift=minus_c3q,
        vector_cache=sewing_vector_cache,
    )
    plus_sewing = build_energy_assigned_c3_sewing(
        raw_plus,
        source_fixed=q_source.x_fixed,
        target_fixed=target_x,
        source_energies=pair_excitation_energies(orbitals, pairs["q"]),
        target_energies=pair_excitation_energies(orbitals, pairs["c3q"]),
    )
    minus_sewing = build_energy_assigned_c3_sewing(
        raw_minus,
        source_fixed=minus_q_source.x_fixed,
        target_fixed=target_minus_x,
        source_energies=pair_excitation_energies(orbitals, pairs["minus_q"]),
        target_energies=pair_excitation_energies(orbitals, pairs["minus_c3q"]),
    )

    target_plus_transport = transport_fixed_terms_from_canonical_form_factors(
        _interaction_terms(base_terms["c3q"]),
        left_transform=plus_sewing.matrix,
        right_transform=minus_sewing.matrix,
        target_x_fixed=target_x,
        target_y_fixed=target_y,
        evaluators=q_source.term_evaluators,
        partner_target_terms=_interaction_terms(base_terms["minus_c3q"]),
    )
    target_minus_transport = transport_fixed_terms_from_canonical_form_factors(
        _interaction_terms(base_terms["minus_c3q"]),
        left_transform=minus_sewing.matrix,
        right_transform=plus_sewing.matrix,
        target_x_fixed=target_minus_x,
        target_y_fixed=target_minus_y,
        evaluators=minus_q_source.term_evaluators,
    )
    target_plus_terms = _with_interactions(
        base_terms["c3q"],
        target_plus_transport.terms,
    )
    target_minus_terms = _with_interactions(
        base_terms["minus_c3q"],
        target_minus_transport.terms,
    )
    if target_plus_transport.partner_terms is None:
        raise RuntimeError("target shared B provider did not return partner terms")
    for term_name in ("B_direct", "B_exchange"):
        target_minus_terms[term_name] = np.array(
            target_plus_transport.partner_terms[term_name],
            copy=True,
        )

    source_matrices = _matrices_from_partner_terms(
        pairs["q"],
        source_q_terms,
        source_minus_terms,
        structure_tolerance=structure_tolerance,
    )
    target_matrices = _matrices_from_partner_terms(
        pairs["c3q"],
        target_plus_terms,
        target_minus_terms,
        structure_tolerance=structure_tolerance,
    )
    return RLGhBNTDHFQuotientOrbitResult(
        source_shift=q,
        target_shift=c3q,
        source_matrices=source_matrices,
        target_matrices=target_matrices,
        terms={
            "q": source_q_terms,
            "minus_q": source_minus_terms,
            "c3q": target_plus_terms,
            "minus_c3q": target_minus_terms,
        },
        plus_sewing=plus_sewing,
        minus_sewing=minus_sewing,
        metadata={
            "provider_mode": "preassembly_source_form_factor_transport",
            "direct_shell_offsets": {
                "c3q": [int(x) for x in c3q_offset],
                "minus_c3q": [int(x) for x in minus_c3q_offset],
            },
            "required_overlap_key_count": len(required_keys),
            "built_overlap_key_count": len(built_keys),
            "source_q_padding": q_source.context.periodic_gauge_padding,
            "source_minus_q_padding": minus_q_source.context.periodic_gauge_padding,
            "sewing_basis_padding": sewing_padding,
            "source_q_provider": q_source.evaluator.metadata(),
            "source_minus_q_provider": minus_q_source.evaluator.metadata(),
            "source_q_leg_vector_cache_count": q_source.leg_vector_cache_count,
            "source_minus_q_leg_vector_cache_count": minus_q_source.leg_vector_cache_count,
        },
        internals=(
            RLGhBNTDHFQuotientOrbitInternals(
                run=term_run,
                sewing_basis=sewing_basis,
                pairs=pairs,
                physical_shifts=resolved_physical,
                source=q_source,
                minus_source=minus_q_source,
            )
            if _retain_internals
            else None
        ),
    )


def _covariance_residuals(
    source: TDHFMatrices,
    target: TDHFMatrices,
    plus_sewing: np.ndarray,
    minus_sewing: np.ndarray,
) -> dict[str, float]:
    transformed_a = plus_sewing @ source.A @ np.linalg.inv(plus_sewing)
    transformed_b = plus_sewing @ source.B @ np.linalg.inv(np.conj(minus_sewing))
    zeros = np.zeros(
        (plus_sewing.shape[0], minus_sewing.shape[1]),
        dtype=np.complex128,
    )
    full_sewing = np.block(
        [
            [plus_sewing, zeros],
            [zeros.T, np.conj(minus_sewing)],
        ]
    )
    transformed_l = full_sewing @ source.L @ np.linalg.inv(full_sewing)
    return {
        "A": float(np.max(np.abs(target.A - transformed_a))) if target.A.size else 0.0,
        "B": float(np.max(np.abs(target.B - transformed_b))) if target.B.size else 0.0,
        "L": float(np.max(np.abs(target.L - transformed_l))) if target.L.size else 0.0,
    }


def build_rlg_hbn_tdhf_c3_quotient_cycle(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    representative_shift: tuple[int, int],
    *,
    beta: float = 1.0,
    physical_shifts: tuple[tuple[int, int], ...] | None = None,
    fixed_copy: int = 0,
    periodic_gauge_padding: int | None = None,
    structure_tolerance: float = 1.0e-6,
    closure_tolerance: float = 1.0e-9,
    require_closure: bool = True,
) -> RLGhBNTDHFQuotientCycleResult:
    """Build a full C3 cycle from one canonical microscopic source provider."""

    mesh = int(run.basis_data.mesh_size)
    q0 = (int(representative_shift[0]) % mesh, int(representative_shift[1]) % mesh)
    q1 = _c3_shift_mod(q0, mesh)
    q2 = _c3_shift_mod(q1, mesh)
    cycle_shifts = (q0,) if q1 == q0 else (q0, q1, q2)
    cycle_pairs = {
        shift: _intraflavor_pairs(orbitals, run, shift)
        for shift in tuple(cycle_shifts)
        + tuple(_minus_shift(shift) for shift in cycle_shifts)
    }
    minimum_cycle_padding = max(
        required_sparse_fixed_context_padding(
            run,
            orbitals,
            cycle_pairs[shift],
            shift,
        )
        for shift in cycle_pairs
    )
    resolved_cycle_padding = (
        minimum_cycle_padding
        if periodic_gauge_padding is None
        else int(periodic_gauge_padding)
    )
    if resolved_cycle_padding < minimum_cycle_padding:
        raise ValueError(
            "periodic_gauge_padding is too small for the complete C3 cycle: "
            f"requested={resolved_cycle_padding}, required={minimum_cycle_padding}, "
            f"cycle={cycle_shifts}"
        )
    first = build_rlg_hbn_tdhf_c3_quotient_orbit(
        run,
        orbitals,
        q0,
        beta=beta,
        physical_shifts=physical_shifts,
        fixed_copy=fixed_copy,
        periodic_gauge_padding=resolved_cycle_padding,
        structure_tolerance=structure_tolerance,
        _retain_internals=True,
    )
    closure: dict[str, float] = {}
    first_residuals = _covariance_residuals(
        first.source_matrices,
        first.target_matrices,
        first.plus_sewing.matrix,
        first.minus_sewing.matrix,
    )
    for name, value in first_residuals.items():
        closure[f"step_0_{name}"] = value
    if q1 == q0:
        closure["max"] = max(closure.values(), default=0.0)
        if bool(require_closure) and closure["max"] > float(closure_tolerance):
            raise RuntimeError(
                "C3-fixed quotient sector failed covariance: "
                f"max_residual={closure['max']:.6e}"
            )
        return RLGhBNTDHFQuotientCycleResult(
            shifts=(q0,),
            matrices={q0: first.source_matrices},
            steps=(first,),
            closure_residuals=closure,
        )

    if first.internals is None:
        raise RuntimeError("quotient orbit internals were not retained")
    internal = first.internals
    if _c3_shift_mod(q2, mesh) != q0:
        raise RuntimeError(f"C3 momentum orbit does not close: {q0}, {q1}, {q2}")
    minus_q1 = _minus_shift(q1)
    minus_q2 = _minus_shift(q2)
    pairs_q2 = cycle_pairs[q2]
    pairs_minus_q2 = cycle_pairs[minus_q2]
    if len(pairs_q2) != len(first.source_matrices.pairs):
        raise ValueError("q2 pair-space dimension differs from canonical source")
    if _pair_order_signature(orbitals, pairs_q2) != _pair_order_signature(
        orbitals,
        pairs_minus_q2,
    ):
        raise ValueError("q2/-q2 pair ordering differs")

    physical = internal.physical_shifts
    q1_offset = c3_repeated_zone_offset(q0, q1, mesh)
    minus_q1_offset = c3_repeated_zone_offset(_minus_shift(q0), minus_q1, mesh)
    q2_offset = c3_repeated_zone_offset(q1, q2, mesh)
    minus_q2_offset = c3_repeated_zone_offset(minus_q1, minus_q2, mesh)
    q2_direct_shell = c3_composed_direct_physical_shell(
        physical,
        repeated_zone_offsets=(q1_offset, q2_offset),
    )
    minus_q2_direct_shell = c3_composed_direct_physical_shell(
        physical,
        repeated_zone_offsets=(minus_q1_offset, minus_q2_offset),
    )
    normal_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        internal.run,
        orbitals,
        {q2: pairs_q2, minus_q2: pairs_minus_q2},
        physical,
    )
    q2_direct_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        internal.run,
        orbitals,
        {q2: pairs_q2},
        q2_direct_shell,
    )
    minus_q2_direct_keys = required_rlg_hbn_tdhf_ws_overlap_shifts(
        internal.run,
        orbitals,
        {minus_q2: pairs_minus_q2},
        minus_q2_direct_shell,
    )
    cycle_run, _built = _run_with_overlap_keys(
        internal.run,
        tuple(sorted(set(normal_keys) | set(q2_direct_keys) | set(minus_q2_direct_keys))),
    )
    q2_terms = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        cycle_run,
        orbitals,
        pairs_q2,
        q2,
        beta=beta,
        physical_shifts=physical,
    )
    minus_q2_terms = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        cycle_run,
        orbitals,
        pairs_minus_q2,
        minus_q2,
        beta=beta,
        physical_shifts=physical,
    )
    q2_shifted = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        cycle_run,
        orbitals,
        pairs_q2,
        q2,
        beta=beta,
        physical_shifts=q2_direct_shell,
    )
    minus_q2_shifted = build_rlg_hbn_tdhf_finite_q_intraflavor_terms(
        cycle_run,
        orbitals,
        pairs_minus_q2,
        minus_q2,
        beta=beta,
        physical_shifts=minus_q2_direct_shell,
    )
    for term_name in ("A_direct", "B_direct"):
        q2_terms[term_name] = np.array(q2_shifted[term_name], copy=True)
        minus_q2_terms[term_name] = np.array(minus_q2_shifted[term_name], copy=True)

    fixed_indices = {
        int(pair[0]) * mesh + int(pair[1])
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    q1_x, _q1_y = fixed_role_masks(
        orbitals,
        internal.pairs["c3q"],
        q1,
        fixed_indices,
        mesh,
    )
    minus_q1_x, _minus_q1_y = fixed_role_masks(
        orbitals,
        internal.pairs["minus_c3q"],
        minus_q1,
        fixed_indices,
        mesh,
    )
    q2_x, q2_y = fixed_role_masks(
        orbitals,
        pairs_q2,
        q2,
        fixed_indices,
        mesh,
    )
    minus_q2_x, minus_q2_y = fixed_role_masks(
        orbitals,
        pairs_minus_q2,
        minus_q2,
        fixed_indices,
        mesh,
    )
    sewing_cache: dict[tuple[int, int, tuple[int, int]], np.ndarray] = {}
    raw_12 = build_raw_pair_c3_sewing(
        internal.sewing_basis,
        orbitals,
        source_pairs=internal.pairs["c3q"],
        source_shift=q1,
        target_pairs=pairs_q2,
        target_shift=q2,
        vector_cache=sewing_cache,
    )
    raw_minus_12 = build_raw_pair_c3_sewing(
        internal.sewing_basis,
        orbitals,
        source_pairs=internal.pairs["minus_c3q"],
        source_shift=minus_q1,
        target_pairs=pairs_minus_q2,
        target_shift=minus_q2,
        vector_cache=sewing_cache,
    )
    sewing_12 = build_energy_assigned_c3_sewing(
        raw_12,
        source_fixed=q1_x,
        target_fixed=q2_x,
        source_energies=pair_excitation_energies(orbitals, internal.pairs["c3q"]),
        target_energies=pair_excitation_energies(orbitals, pairs_q2),
    )
    minus_sewing_12 = build_energy_assigned_c3_sewing(
        raw_minus_12,
        source_fixed=minus_q1_x,
        target_fixed=minus_q2_x,
        source_energies=pair_excitation_energies(
            orbitals,
            internal.pairs["minus_c3q"],
        ),
        target_energies=pair_excitation_energies(orbitals, pairs_minus_q2),
    )
    sewing_02 = sewing_12.matrix @ first.plus_sewing.matrix
    minus_sewing_02 = minus_sewing_12.matrix @ first.minus_sewing.matrix
    q2_transport = transport_fixed_terms_from_canonical_form_factors(
        _interaction_terms(q2_terms),
        left_transform=sewing_02,
        right_transform=minus_sewing_02,
        target_x_fixed=q2_x,
        target_y_fixed=q2_y,
        evaluators=internal.source.term_evaluators,
        partner_target_terms=_interaction_terms(minus_q2_terms),
    )
    minus_q2_transport = transport_fixed_terms_from_canonical_form_factors(
        _interaction_terms(minus_q2_terms),
        left_transform=minus_sewing_02,
        right_transform=sewing_02,
        target_x_fixed=minus_q2_x,
        target_y_fixed=minus_q2_y,
        evaluators=internal.minus_source.term_evaluators,
    )
    q2_terms = _with_interactions(q2_terms, q2_transport.terms)
    minus_q2_terms = _with_interactions(minus_q2_terms, minus_q2_transport.terms)
    if q2_transport.partner_terms is None:
        raise RuntimeError("q2 shared B provider did not return partner terms")
    for term_name in ("B_direct", "B_exchange"):
        minus_q2_terms[term_name] = np.array(
            q2_transport.partner_terms[term_name],
            copy=True,
        )
    q2_matrices = _matrices_from_partner_terms(
        pairs_q2,
        q2_terms,
        minus_q2_terms,
        structure_tolerance=structure_tolerance,
    )

    step_1 = RLGhBNTDHFQuotientOrbitResult(
        source_shift=q1,
        target_shift=q2,
        source_matrices=first.target_matrices,
        target_matrices=q2_matrices,
        terms={
            "q": first.terms["c3q"],
            "minus_q": first.terms["minus_c3q"],
            "c3q": q2_terms,
            "minus_c3q": minus_q2_terms,
        },
        plus_sewing=sewing_12,
        minus_sewing=minus_sewing_12,
        metadata={
            "provider_mode": "composed_preassembly_source_form_factor_transport",
            "direct_shell_offsets": {
                "q1": [int(x) for x in q1_offset],
                "q2": [int(x) for x in q2_offset],
                "minus_q1": [int(x) for x in minus_q1_offset],
                "minus_q2": [int(x) for x in minus_q2_offset],
            },
            "direct_shell_transport": "G_next=C3(G_current)-offset",
        },
    )
    second_residuals = _covariance_residuals(
        step_1.source_matrices,
        step_1.target_matrices,
        sewing_12.matrix,
        minus_sewing_12.matrix,
    )
    for name, value in second_residuals.items():
        closure[f"step_1_{name}"] = value

    # Fix the quotient gauge globally around the cycle. Independent raw-phase
    # assignments on the closing edge can differ by pair-label gauge phases;
    # the physical C3^3 closure is the inverse of the composed q0->q2 sewing.
    sewing_20_matrix = np.linalg.inv(sewing_02)
    minus_sewing_20_matrix = np.linalg.inv(minus_sewing_02)
    sewing_20_gram = sewing_20_matrix.conj().T @ sewing_20_matrix
    minus_sewing_20_gram = minus_sewing_20_matrix.conj().T @ minus_sewing_20_matrix
    sewing_20 = RLGhBNTDHFEnergySewing(
        matrix=sewing_20_matrix,
        assignment_max_energy_delta=0.0,
        assignment_mean_energy_delta=0.0,
        unitarity_residual_max=float(
            np.max(np.abs(sewing_20_gram - np.eye(sewing_20_gram.shape[0])))
        ),
        condition_number=float(np.linalg.cond(sewing_20_matrix)),
    )
    minus_sewing_20 = RLGhBNTDHFEnergySewing(
        matrix=minus_sewing_20_matrix,
        assignment_max_energy_delta=0.0,
        assignment_mean_energy_delta=0.0,
        unitarity_residual_max=float(
            np.max(
                np.abs(
                    minus_sewing_20_gram
                    - np.eye(minus_sewing_20_gram.shape[0])
                )
            )
        ),
        condition_number=float(np.linalg.cond(minus_sewing_20_matrix)),
    )
    third_residuals = _covariance_residuals(
        q2_matrices,
        first.source_matrices,
        sewing_20.matrix,
        minus_sewing_20.matrix,
    )
    for name, value in third_residuals.items():
        closure[f"step_2_{name}"] = value
    step_2 = RLGhBNTDHFQuotientOrbitResult(
        source_shift=q2,
        target_shift=q0,
        source_matrices=q2_matrices,
        target_matrices=first.source_matrices,
        terms={
            "q": q2_terms,
            "minus_q": minus_q2_terms,
            "c3q": first.terms["q"],
            "minus_c3q": first.terms["minus_q"],
        },
        plus_sewing=sewing_20,
        minus_sewing=minus_sewing_20,
        metadata={"provider_mode": "cycle_closure_acceptance"},
    )
    closure["max"] = max(closure.values(), default=0.0)
    if bool(require_closure) and closure["max"] > float(closure_tolerance):
        raise RuntimeError(
            "C3 quotient cycle failed covariance closure: "
            f"max_residual={closure['max']:.6e}, "
            f"tolerance={float(closure_tolerance):.6e}"
        )
    return RLGhBNTDHFQuotientCycleResult(
        shifts=(q0, q1, q2),
        matrices={
            q0: first.source_matrices,
            q1: first.target_matrices,
            q2: q2_matrices,
        },
        steps=(first, step_1, step_2),
        closure_residuals=closure,
    )
