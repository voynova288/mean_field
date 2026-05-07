from __future__ import annotations

import argparse

import numpy as np

from mean_field import load_b0_suite
from mean_field.benchmarks import load_complex_stack_tsv
from mean_field.systems.tbg.zero_field import build_overlap_block_set
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    build_full_density_from_hamiltonian,
    build_interaction_hamiltonian,
    calculate_norm_convergence,
    initialize_full_state,
    oda_parametrization_restricted,
)
from mean_field.systems.tbg.zero_field.runners import _build_benchmark_grid_solution, build_b0_reference_parameters


def _matrix_diff(label: str, reference: np.ndarray, computed: np.ndarray) -> None:
    diff = computed - reference
    print(
        f"{label}: "
        f"reference_fro={np.linalg.norm(reference):.12e} "
        f"computed_fro={np.linalg.norm(computed):.12e} "
        f"diff_fro={np.linalg.norm(diff):.12e} "
        f"diff_max_abs={np.max(np.abs(diff)):.12e}"
    )


def compare_case(benchmark_id: str, *, iteration: int) -> None:
    if iteration < 1:
        raise ValueError(f"Expected iteration >= 1, got {iteration}")

    case = load_b0_suite().get(benchmark_id)
    params = build_b0_reference_parameters(case.theta_deg)
    solution = _build_benchmark_grid_solution(case, params, lk=case.lk, lg=case.lg)
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=case.nu)
    overlap_blocks = build_overlap_block_set(solution, lg=case.lg)

    initial_density_path = case.initial_density_override_path()
    initial_density = None
    if initial_density_path.is_file():
        initial_density = load_complex_stack_tsv(initial_density_path, shape=state.density.shape)
    initialize_full_state(state, init_mode=case.init_mode, seed=case.seed, initial_density=initial_density)

    snapshot_input_density = None
    snapshot_interaction = None
    snapshot_hamiltonian = None
    snapshot_updated_density = None
    snapshot_mu = None
    snapshot_lambda = None
    snapshot_norm_raw = None
    snapshot_norm_mixed = None
    snapshot_energies = None

    for current_iteration in range(1, iteration + 1):
        previous_density = state.density.copy()
        interaction_h = build_interaction_hamiltonian(
            previous_density,
            overlap_blocks,
            solution.lattice_kvec,
            solution.params,
            state.v0,
        )
        hamiltonian_total = state.h0 + interaction_h
        # Keep the cached Hamiltonian synchronized with the local step so the
        # ODA branch matches `run_full_hartree_fock(...)`.
        state.hamiltonian[:, :, :] = hamiltonian_total
        density_new, energies, sigma_ztauz, mu = build_full_density_from_hamiltonian(
            state.hamiltonian,
            state.sigma_z,
            nu=state.nu,
        )
        delta_density = density_new - previous_density
        oda_lambda = oda_parametrization_restricted(
            state,
            delta_density,
            overlap_blocks,
            solution.lattice_kvec,
            solution.params,
        )
        mixed_density = oda_lambda * density_new + (1.0 - oda_lambda) * previous_density
        norm_raw = calculate_norm_convergence(density_new, previous_density)
        norm_mixed = calculate_norm_convergence(mixed_density, previous_density)

        state.density[:, :, :] = mixed_density
        state.energies[:, :] = energies
        state.sigma_ztauz[:, :] = sigma_ztauz
        state.mu = float(mu)

        if current_iteration == iteration:
            snapshot_input_density = previous_density
            snapshot_interaction = interaction_h
            snapshot_hamiltonian = hamiltonian_total
            snapshot_updated_density = mixed_density
            snapshot_mu = float(mu)
            snapshot_lambda = float(oda_lambda)
            snapshot_norm_raw = float(norm_raw)
            snapshot_norm_mixed = float(norm_mixed)
            snapshot_energies = energies

    assert snapshot_input_density is not None
    assert snapshot_interaction is not None
    assert snapshot_hamiltonian is not None
    assert snapshot_updated_density is not None
    assert snapshot_mu is not None
    assert snapshot_lambda is not None
    assert snapshot_norm_raw is not None
    assert snapshot_norm_mixed is not None
    assert snapshot_energies is not None

    reference_input_path = case.reference_iteration_input_density_path(iteration)
    reference_interaction_path = case.reference_iteration_interaction_path(iteration)
    reference_hamiltonian_path = case.reference_iteration_hamiltonian_path(iteration)
    reference_updated_density_path = case.reference_iteration_updated_density_path(iteration)
    missing = [
        path
        for path in (
            reference_input_path,
            reference_interaction_path,
            reference_hamiltonian_path,
            reference_updated_density_path,
        )
        if not path.is_file()
    ]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing iteration snapshot reference files for {benchmark_id}: {missing_str}")

    ref_input_density = load_complex_stack_tsv(reference_input_path, shape=snapshot_input_density.shape)
    ref_interaction = load_complex_stack_tsv(reference_interaction_path, shape=snapshot_interaction.shape)
    ref_hamiltonian = load_complex_stack_tsv(reference_hamiltonian_path, shape=snapshot_hamiltonian.shape)
    ref_updated_density = load_complex_stack_tsv(reference_updated_density_path, shape=snapshot_updated_density.shape)

    print(f"benchmark_id={case.benchmark_id}")
    print(f"theta_deg={case.theta_deg:.2f}")
    print(f"nu={case.nu}")
    print(f"init_mode={case.init_mode}")
    print(f"seed={case.seed}")
    print(f"iteration={iteration}")
    print(f"initial_density_override={initial_density_path.is_file()}")
    _matrix_diff("input_density", ref_input_density, snapshot_input_density)
    _matrix_diff("interaction_h", ref_interaction, snapshot_interaction)
    _matrix_diff("hamiltonian_total", ref_hamiltonian, snapshot_hamiltonian)
    _matrix_diff("updated_density", ref_updated_density, snapshot_updated_density)
    print(f"computed_mu={snapshot_mu:.12f}")
    print(f"computed_oda_lambda={snapshot_lambda:.12f}")
    print(f"computed_norm_raw={snapshot_norm_raw:.12e}")
    print(f"computed_norm_mixed={snapshot_norm_mixed:.12e}")
    print(f"lowest_eigs_k0={','.join(f'{val:.12f}' for val in snapshot_energies[:8, 0])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python full-HF iteration snapshots against Julia matrix references.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    parser.add_argument("--iteration", type=int, default=10, help="Iteration number to compare against the Julia snapshot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_case(args.benchmark_id, iteration=args.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
