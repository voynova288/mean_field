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
    build_restricted_density_from_hamiltonian,
    calculate_norm_convergence,
    compute_hf_energy,
    initialize_full_state,
    initialize_restricted_state,
    occupied_sigma_mean,
    oda_parametrization_restricted,
    offdiag_flavor_norm,
    restricted_filling,
    restricted_gap_estimate,
)
from mean_field.systems.tbg.zero_field.runners import _build_benchmark_grid_solution, build_b0_reference_parameters


def inspect_case(benchmark_id: str, *, mode: str = "full", max_iter: int = 8) -> None:
    suite = load_b0_suite()
    case = suite.get(benchmark_id)
    params = build_b0_reference_parameters(case.theta_deg)
    solution = _build_benchmark_grid_solution(case, params, lk=case.lk, lg=case.lg)
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=case.nu)
    overlap_blocks = build_overlap_block_set(solution, lg=case.lg)
    override_path = case.initial_density_override_path()
    initial_density = None
    if mode == "full" and override_path.is_file():
        initial_density = load_complex_stack_tsv(override_path, shape=state.density.shape)

    if mode == "full":
        initialize_full_state(state, init_mode=case.init_mode, seed=case.seed, initial_density=initial_density)
        density_builder = build_full_density_from_hamiltonian
    elif mode == "restricted":
        initialize_restricted_state(state, init_mode=case.init_mode, seed=case.seed)
        density_builder = build_restricted_density_from_hamiltonian
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    print(f"benchmark_id={case.benchmark_id}")
    print(f"mode={mode}")
    print(f"theta_deg={case.theta_deg:.2f}")
    print(f"nu={case.nu}")
    print(f"init_mode={case.init_mode}")
    print(f"seed={case.seed}")
    print(f"lk={case.lk}")
    print(f"lg={case.lg}")
    print(f"nk={solution.nk}")
    print(f"initial_density_override={override_path.is_file()}")
    print(
        "iter=0 "
        f"filling={restricted_filling(state.density):.12f} "
        f"offdiag_flavor={offdiag_flavor_norm(state.density):.12e}"
    )

    for iteration in range(1, max_iter + 1):
        previous_density = state.density.copy()
        state.hamiltonian[:, :, :] = state.h0
        interaction_h = build_interaction_hamiltonian(
            state.density,
            overlap_blocks,
            solution.lattice_kvec,
            solution.params,
            state.v0,
        )
        state.hamiltonian[:, :, :] += interaction_h

        energy = compute_hf_energy(interaction_h, state.h0, state.density)
        density_new, energies, sigma_ztauz, mu = density_builder(
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

        print(
            f"iter={iteration} "
            f"energy={energy:.12f} "
            f"mu={mu:.12f} "
            f"lambda={oda_lambda:.12f} "
            f"norm_raw={norm_raw:.12e} "
            f"norm_mixed={norm_mixed:.12e} "
            f"offdiag_flavor={offdiag_flavor_norm(state.density):.12e} "
            f"gap={restricted_gap_estimate(state.energies, state.nu):.12f} "
            f"occupied_sigma_mean={occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu):.12f} "
            f"filling={restricted_filling(state.density):.12f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace the first few HF iterations for a bundled B0 benchmark case.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    parser.add_argument("--mode", choices=("full", "restricted"), default="full")
    parser.add_argument("--max-iter", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_case(args.benchmark_id, mode=args.mode, max_iter=args.max_iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
