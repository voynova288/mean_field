from __future__ import annotations

import argparse

import numpy as np

from mean_field import load_b0_suite
from mean_field.benchmarks import load_complex_stack_tsv
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    build_full_density_from_hamiltonian,
    occupied_sigma_mean,
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

    reference_hamiltonian_path = case.reference_iteration_hamiltonian_path(iteration)
    reference_updated_density_path = case.reference_iteration_updated_density_path(iteration)
    missing = [path for path in (reference_hamiltonian_path, reference_updated_density_path) if not path.is_file()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing reference files for {benchmark_id} iteration {iteration}: {missing_str}")

    reference_hamiltonian = load_complex_stack_tsv(reference_hamiltonian_path, shape=state.hamiltonian.shape)
    reference_updated_density = load_complex_stack_tsv(reference_updated_density_path, shape=state.density.shape)
    computed_density, energies, sigma_ztauz, mu = build_full_density_from_hamiltonian(
        reference_hamiltonian,
        state.sigma_z,
        nu=case.nu,
    )

    print(f"benchmark_id={case.benchmark_id}")
    print(f"iteration={iteration}")
    print(f"theta_deg={case.theta_deg:.2f}")
    print(f"nu={case.nu}")
    print(f"init_mode={case.init_mode}")
    print(f"seed={case.seed}")
    _matrix_diff("updated_density", reference_updated_density, computed_density)
    print(f"computed_mu={mu:.12f}")
    print(f"occupied_sigma_mean={occupied_sigma_mean(energies, sigma_ztauz, case.nu):.12f}")
    print(f"lowest_eigs_k0={','.join(f'{val:.12f}' for val in energies[:8, 0])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Python full-HF projector update against a stored Julia iteration Hamiltonian."
    )
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number whose stored Hamiltonian should be used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_case(args.benchmark_id, iteration=args.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
