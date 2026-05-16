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
    target_hamiltonian = np.asarray(base_hamiltonian, dtype=np.complex128).copy()
    nt, nt_rhs, nk_source = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")
    if target_hamiltonian.shape[0] != nt or target_hamiltonian.shape[1] != nt:
        raise ValueError(f"Expected target Hamiltonian flavor dimension {nt}, got {target_hamiltonian.shape}")

    nk_target = target_hamiltonian.shape[2]
    scale = float(beta) * float(v0) / nk_source
    for blocks in (source_overlap_blocks, target_overlap_blocks, target_source_overlap_blocks):
        _validate_overlap_shift_table(blocks)

    for shift in target_source_overlap_blocks.shifts:
        target_source_overlap = target_source_overlap_blocks.overlaps[shift]
        if target_source_overlap.shape != (nt, nk_target, nt, nk_source):
            raise ValueError(
                f"Expected target-source overlap shape {(nt, nk_target, nt, nk_source)}, "
                f"got {target_source_overlap.shape} for shift {shift}"
            )

        hartree_kernel = source_overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            source_diagonal = source_overlap_blocks.diagonal_overlaps.get(shift)
            target_diagonal = target_overlap_blocks.diagonal_overlaps.get(shift)
            if source_diagonal is None or target_diagonal is None:
                raise ValueError(f"Missing source/target diagonal overlap for active Hartree shift {shift}")
            if source_diagonal.shape != (nt, nt, nk_source):
                raise ValueError(f"Expected source diagonal shape {(nt, nt, nk_source)}, got {source_diagonal.shape}")
            if target_diagonal.shape != (nt, nt, nk_target):
                raise ValueError(f"Expected target diagonal shape {(nt, nt, nk_target)}, got {target_diagonal.shape}")
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
    convergence_rule: Literal["raw", "mixed"] = "raw",
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
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
    )
