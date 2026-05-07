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
    initialize_full_state,
)
from mean_field.systems.tbg.zero_field.runners import build_b0_reference_parameters
from mean_field.systems.tbg.zero_field.runners import _build_benchmark_grid_solution


def _matrix_diff(label: str, reference: np.ndarray, computed: np.ndarray) -> None:
    diff = computed - reference
    print(
        f"{label}: "
        f"reference_fro={np.linalg.norm(reference):.12e} "
        f"computed_fro={np.linalg.norm(computed):.12e} "
        f"diff_fro={np.linalg.norm(diff):.12e} "
        f"diff_max_abs={np.max(np.abs(diff)):.12e}"
    )


def compare_case(benchmark_id: str) -> None:
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

    interaction_h = build_interaction_hamiltonian(
        state.density,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        state.v0,
    )
    hamiltonian_total = state.h0 + interaction_h
    density_new, energies, _, mu = build_full_density_from_hamiltonian(
        hamiltonian_total,
        state.sigma_z,
        nu=state.nu,
    )

    reference_interaction_path = case.reference_first_iteration_interaction_path()
    reference_hamiltonian_path = case.reference_first_iteration_hamiltonian_path()
    reference_density_path = case.reference_first_iteration_density_path()
    missing = [
        path
        for path in (
            reference_interaction_path,
            reference_hamiltonian_path,
            reference_density_path,
        )
        if not path.is_file()
    ]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing first-iteration reference files for {benchmark_id}: {missing_str}")

    ref_interaction = load_complex_stack_tsv(reference_interaction_path, shape=interaction_h.shape)
    ref_hamiltonian = load_complex_stack_tsv(reference_hamiltonian_path, shape=hamiltonian_total.shape)
    ref_density = load_complex_stack_tsv(reference_density_path, shape=density_new.shape)

    print(f"benchmark_id={case.benchmark_id}")
    print(f"theta_deg={case.theta_deg:.2f}")
    print(f"nu={case.nu}")
    print(f"init_mode={case.init_mode}")
    print(f"seed={case.seed}")
    print(f"initial_density_override={initial_density_path.is_file()}")
    _matrix_diff("interaction_h", ref_interaction, interaction_h)
    _matrix_diff("hamiltonian_total", ref_hamiltonian, hamiltonian_total)
    _matrix_diff("density_new", ref_density, density_new)
    print(f"computed_mu={mu:.12f}")
    print(f"lowest_eigs_k0={','.join(f'{val:.12f}' for val in energies[:8, 0])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python full-HF first-step matrices against machine-readable Julia references.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_case(args.benchmark_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
