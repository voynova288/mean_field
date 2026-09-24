from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from .engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
    compute_oda_parameter,
)
from .overlap import (
    HFOverlapBlockSet,
    compute_density_overlap_trace_from_diagonal,
    contract_fock_term_from_overlap,
)
from .problem import (
    HartreeFockInitializerProtocol,
    HartreeFockKernel,
    HartreeFockProblem,
    ProjectedDensityBuilderProtocol,
    run_hartree_fock_problem,
)


def empty_overlap_block_set() -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=(),
        gvecs=np.asarray([], dtype=np.complex128),
        overlaps={},
    )


def compute_hf_energy(interaction_hamiltonian: np.ndarray, h0: np.ndarray, density: np.ndarray) -> float:
    # Match Julia B0's stored-projector convention:
    # `tr(H * transpose(P)) == sum_ab H[a,b] * P[a,b]`.
    # This is not the ordinary `tr(H @ rho)` contraction for a conventional density matrix.
    total = np.einsum("abk,abk->", interaction_hamiltonian, density, optimize=True) / 2.0
    total += np.einsum("abk,abk->", h0, density, optimize=True)
    return float(total.real / h0.shape[2])


def _state_interaction_scale(state: HartreeFockStateProtocol, v0: float | None) -> float:
    if v0 is not None:
        return float(v0)
    state_v0 = getattr(state, "v0", None)
    if state_v0 is None:
        raise ValueError("Projected HF interaction scale `v0` must be provided when the state has no `v0` attribute.")
    return float(state_v0)


def _validate_overlap_shift_table(overlap_blocks: HFOverlapBlockSet) -> None:
    if len(overlap_blocks.shifts) != int(overlap_blocks.gvecs.size):
        raise ValueError("Overlap shifts and g-vectors must have the same length.")
    if len(set(overlap_blocks.shifts)) != len(overlap_blocks.shifts):
        raise ValueError("Overlap shifts must be unique and ordered explicitly.")
    if set(overlap_blocks.overlaps) != set(overlap_blocks.shifts):
        raise ValueError("Overlap block keys must exactly match the shift inventory.")


def _validate_matching_shift_inventories(*block_sets: HFOverlapBlockSet) -> None:
    if not block_sets:
        return
    reference = block_sets[0]
    _validate_overlap_shift_table(reference)
    reference_gvecs = np.asarray(reference.gvecs, dtype=np.complex128)
    for blocks in block_sets[1:]:
        _validate_overlap_shift_table(blocks)
        if blocks.shifts != reference.shifts:
            raise ValueError("Source, target, and target/source shift order must match exactly.")
        if not np.array_equal(
            np.asarray(blocks.gvecs, dtype=np.complex128), reference_gvecs
        ):
            raise ValueError("Source, target, and target/source g-vectors must match exactly.")


def build_projected_interaction_hamiltonian(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")

    interaction = np.zeros_like(density)
    if len(overlap_blocks.shifts) == 0:
        return interaction
    _validate_overlap_shift_table(overlap_blocks)

    scale = float(beta) * float(v0) / nk
    for shift in overlap_blocks.shifts:
        overlap = overlap_blocks.overlaps[shift]
        if overlap.shape != (nt, nk, nt, nk):
            raise ValueError(f"Expected overlap block shape {(nt, nk, nt, nk)}, got {overlap.shape} for shift {shift}")

        diagonal_overlap = overlap_blocks.diagonal_overlaps.get(shift)
        hartree_kernel = overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            if diagonal_overlap is None:
                raise ValueError(f"Missing diagonal overlap for active Hartree shift {shift}")
            if diagonal_overlap.shape != (nt, nt, nk):
                raise ValueError(f"Expected diagonal overlap shape {(nt, nt, nk)}, got {diagonal_overlap.shape} for shift {shift}")
            hartree_prefactor = scale * float(hartree_kernel)
            if hartree_prefactor != 0.0:
                tr_pg = compute_density_overlap_trace_from_diagonal(density, diagonal_overlap, use_numba=use_numba)
                interaction += hartree_prefactor * tr_pg * diagonal_overlap

        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            if fock_kernel.shape != (nk, nk):
                raise ValueError(f"Expected fock kernel shape {(nk, nk)}, got {fock_kernel.shape} for shift {shift}")
            coeff_matrix = scale * fock_kernel
            interaction -= contract_fock_term_from_overlap(overlap, density, coeff_matrix, use_numba=use_numba)

    return interaction


def build_projected_target_hamiltonian(
    base_hamiltonian: np.ndarray,
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet,
    target_overlap_blocks: HFOverlapBlockSet,
    target_source_overlap_blocks: HFOverlapBlockSet,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Evaluate a source-density HF operator in a target projected basis.

    The target and source flavor dimensions may differ.  Their only bridge is
    the rectangular target-source overlap; Hartree traces remain source-space
    scalars and Fock contraction returns a square target-space operator.
    """

    target_hamiltonian = np.asarray(base_hamiltonian, dtype=np.complex128).copy()
    nt_source, nt_source_rhs, nk_source = density.shape
    if nt_source != nt_source_rhs:
        raise ValueError(f"Expected square source density blocks, got {density.shape}")
    if (
        target_hamiltonian.ndim != 3
        or target_hamiltonian.shape[0] != target_hamiltonian.shape[1]
    ):
        raise ValueError(f"Expected square target Hamiltonian blocks, got {target_hamiltonian.shape}")

    nt_target = int(target_hamiltonian.shape[0])
    nk_target = int(target_hamiltonian.shape[2])
    scale = float(beta) * float(v0) / nk_source
    _validate_matching_shift_inventories(
        source_overlap_blocks,
        target_overlap_blocks,
        target_source_overlap_blocks,
    )

    for shift in target_source_overlap_blocks.shifts:
        target_source_overlap = target_source_overlap_blocks.overlaps[shift]
        expected_target_source_shape = (
            nt_target,
            nk_target,
            nt_source,
            nk_source,
        )
        if target_source_overlap.shape != expected_target_source_shape:
            raise ValueError(
                f"Expected target-source overlap shape {expected_target_source_shape}, "
                f"got {target_source_overlap.shape} for shift {shift}"
            )

        hartree_kernel = source_overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            source_diagonal = source_overlap_blocks.diagonal_overlaps.get(shift)
            target_diagonal = target_overlap_blocks.diagonal_overlaps.get(shift)
            if source_diagonal is None or target_diagonal is None:
                raise ValueError(f"Missing source/target diagonal overlap for active Hartree shift {shift}")
            if source_diagonal.shape != (nt_source, nt_source, nk_source):
                raise ValueError(
                    f"Expected source diagonal shape {(nt_source, nt_source, nk_source)}, "
                    f"got {source_diagonal.shape}"
                )
            if target_diagonal.shape != (nt_target, nt_target, nk_target):
                raise ValueError(
                    f"Expected target diagonal shape {(nt_target, nt_target, nk_target)}, "
                    f"got {target_diagonal.shape}"
                )
            hartree_prefactor = scale * float(hartree_kernel)
            if hartree_prefactor != 0.0:
                tr_pg = compute_density_overlap_trace_from_diagonal(density, source_diagonal, use_numba=use_numba)
                target_hamiltonian += hartree_prefactor * tr_pg * target_diagonal

        fock_kernel = target_source_overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            if fock_kernel.shape != (nk_target, nk_source):
                raise ValueError(f"Expected fock kernel shape {(nk_target, nk_source)}, got {fock_kernel.shape}")
            target_hamiltonian -= contract_fock_term_from_overlap(
                target_source_overlap,
                density,
                scale * fock_kernel,
                use_numba=use_numba,
            )

    return target_hamiltonian


def compute_projected_oda_parameter(
    state: HartreeFockStateProtocol,
    delta_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> float:
    interaction_scale = _state_interaction_scale(state, v0)
    return compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: build_projected_interaction_hamiltonian(
            density,
            overlap_blocks,
            v0=interaction_scale,
            beta=beta,
            use_numba=use_numba,
        ),
    )


def build_projected_hf_kernel(
    state: HartreeFockStateProtocol,
    overlap_blocks: HFOverlapBlockSet,
    *,
    density_builder: ProjectedDensityBuilderProtocol,
    v0: float | None = None,
    beta: float = 1.0,
    energy_functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = compute_hf_energy,
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None | Literal["default"] = "default",
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None,
    density_postprocessor: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None,
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    convergence_rule: Literal["raw", "mixed"] = "raw",
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    interaction_scale = _state_interaction_scale(state, v0)
    interaction_builder = lambda density: build_projected_interaction_hamiltonian(
        density,
        overlap_blocks,
        v0=interaction_scale,
        beta=beta,
        use_numba=use_numba,
    )
    if oda_parameterizer == "default":
        resolved_oda_parameterizer = None
        resolved_oda_delta_interaction_builder = interaction_builder
    else:
        resolved_oda_parameterizer = oda_parameterizer
        resolved_oda_delta_interaction_builder = None

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        energy_functional=energy_functional,
        oda_parameterizer=resolved_oda_parameterizer,
        oda_delta_interaction_builder=resolved_oda_delta_interaction_builder,
        hamiltonian_postprocessor=hamiltonian_postprocessor,
        density_postprocessor=density_postprocessor,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_metric=convergence_metric,
        convergence_rule=convergence_rule,
    )


def build_projected_hf_problem(
    *,
    initializer: HartreeFockInitializerProtocol,
    kernel: HartreeFockKernel,
) -> HartreeFockProblem:
    return HartreeFockProblem(initializer=initializer, kernel=kernel)


def run_projected_hartree_fock(
    state: HartreeFockStateProtocol,
    *,
    initializer: HartreeFockInitializerProtocol,
    density_builder: ProjectedDensityBuilderProtocol,
    overlap_blocks: HFOverlapBlockSet,
    init_mode: str,
    seed: int,
    v0: float | None = None,
    beta: float = 1.0,
    energy_functional: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = compute_hf_energy,
    oda_parameterizer: Callable[[HartreeFockStateProtocol, np.ndarray], float] | None | Literal["default"] = "default",
    hamiltonian_postprocessor: Callable[[np.ndarray], None] | None = None,
    density_postprocessor: Callable[[np.ndarray], None] | None = None,
    step_callback: Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None = None,
    convergence_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    convergence_rule: Literal["raw", "mixed"] = "raw",
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    max_oda_lambda: float | None = None,
    use_numba: bool | None = None,
) -> HartreeFockRun:
    kernel = build_projected_hf_kernel(
        state,
        overlap_blocks,
        density_builder=density_builder,
        v0=v0,
        beta=beta,
        energy_functional=energy_functional,
        oda_parameterizer=oda_parameterizer,
        hamiltonian_postprocessor=hamiltonian_postprocessor,
        density_postprocessor=density_postprocessor,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_metric=convergence_metric,
        convergence_rule=convergence_rule,
        use_numba=use_numba,
    )
    problem = build_projected_hf_problem(initializer=initializer, kernel=kernel)
    return run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        max_oda_lambda=max_oda_lambda,
    )
