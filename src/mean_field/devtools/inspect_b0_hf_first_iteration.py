from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mean_field import load_b0_suite
from mean_field.systems.tbg.zero_field import build_b0_uniform_lattice, build_overlap_block_set
from mean_field.systems.tbg.zero_field.hf import (
    _hex_shell_contains,
    _screened_coulomb_matrix,
    RestrictedHartreeFockState,
    build_full_density_from_hamiltonian,
    build_interaction_hamiltonian,
    build_restricted_density_from_hamiltonian,
    contract_fock_term_from_overlap,
    initialize_full_state,
    initialize_restricted_state,
    offdiag_flavor_norm,
    oda_parametrization_restricted,
    screened_coulomb,
)
from mean_field.systems.tbg.zero_field.model import solve_bm_model
from mean_field.systems.tbg.zero_field.overlap import compute_density_overlap_trace
from mean_field.systems.tbg.zero_field.runners import build_b0_reference_parameters


@dataclass(frozen=True)
class MatrixNorms:
    fro_norm: float
    offdiag_total_norm: float
    max_abs: float
    max_abs_offdiag: float


def matrix_norms(stack: np.ndarray) -> MatrixNorms:
    if stack.ndim != 3:
        raise ValueError(f"Expected a rank-3 stack, got shape {stack.shape}")
    offdiag = stack.copy()
    for ik in range(stack.shape[2]):
        diag = np.diag(np.diag(stack[:, :, ik]))
        offdiag[:, :, ik] -= diag
    return MatrixNorms(
        fro_norm=float(np.linalg.norm(stack)),
        offdiag_total_norm=float(np.linalg.norm(offdiag)),
        max_abs=float(np.max(np.abs(stack))),
        max_abs_offdiag=float(np.max(np.abs(offdiag))),
    )


def print_matrix_norms(label: str, stack: np.ndarray) -> None:
    norms = matrix_norms(stack)
    print(
        f"{label}: "
        f"fro={norms.fro_norm:.6e} "
        f"offdiag_total={norms.offdiag_total_norm:.6e} "
        f"max_abs={norms.max_abs:.6e} "
        f"max_abs_offdiag={norms.max_abs_offdiag:.6e}"
    )


def _load_initial_density_override(path: Path, *, nt: int, nk: int) -> np.ndarray:
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, real_s, imag_s = stripped.split("\t")
            density[int(row_s), int(col_s), int(ik_s)] = complex(float(real_s), float(imag_s))
    return density


def inspect_case(benchmark_id: str, *, mode: str = "full") -> None:
    suite = load_b0_suite()
    case = suite.get(benchmark_id)
    params = build_b0_reference_parameters(case.theta_deg)
    grid = build_b0_uniform_lattice(params, case.lk)
    solution = solve_bm_model(params, grid.kvec, lg=case.lg, sigma_rotation=True)
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=case.nu)
    overlap_blocks = build_overlap_block_set(solution, lg=case.lg)
    override_path = case.case_dir / f"initial_density_{case.init_mode}_seed_{case.seed:03d}.tsv"
    initial_density = None
    if mode == "full" and override_path.is_file():
        initial_density = _load_initial_density_override(override_path, nt=state.nt, nk=state.nk)

    if mode == "full":
        init_mode = case.init_mode
        initialize_full_state(state, init_mode=init_mode, seed=case.seed, initial_density=initial_density)
        density_builder = build_full_density_from_hamiltonian
    elif mode == "restricted":
        init_mode = case.init_mode
        initialize_restricted_state(state, init_mode=init_mode, seed=case.seed)
        density_builder = build_restricted_density_from_hamiltonian
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    print(f"benchmark_id={case.benchmark_id}")
    print(f"mode={mode}")
    print(f"theta_deg={case.theta_deg:.2f}")
    print(f"nu={case.nu}")
    print(f"init_mode={init_mode}")
    print(f"seed={case.seed}")
    print(f"lk={case.lk}")
    print(f"lg={case.lg}")
    print(f"nk={solution.nk}")
    print(f"initial_density_override={override_path.is_file()}")

    print(f"initial_offdiag_flavor_norm={offdiag_flavor_norm(state.density):.6e}")
    print_matrix_norms("density_initial", state.density)

    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)))
    for shift in ((0, 0), (1, 0), (0, 1)):
        if shift not in overlap_blocks.overlaps:
            continue
        overlap = overlap_blocks.overlaps[shift]
        gvec_index = overlap_blocks.shifts.index(shift)
        gvec = complex(overlap_blocks.gvecs[gvec_index])
        coeff_matrix = state.v0 * _screened_coulomb_matrix(
            solution.lattice_kvec[None, :] - solution.lattice_kvec[:, None] + gvec,
            lm,
        ) / solution.nk
        fock_piece = contract_fock_term_from_overlap(overlap, state.density, coeff_matrix)
        tr_pg = compute_density_overlap_trace(state.density, overlap)
        print(
            f"shift={shift} "
            f"g_abs={abs(gvec):.6e} "
            f"in_shell={_hex_shell_contains(params, gvec)} "
            f"overlap_fro={np.linalg.norm(overlap):.6e} "
            f"overlap_max_abs={np.max(np.abs(overlap)):.6e} "
            f"hartree_coeff={state.v0 * screened_coulomb(gvec, lm) / solution.nk:.6e} "
            f"tr_pg_abs={abs(tr_pg):.6e} "
            f"fock_fro={np.linalg.norm(fock_piece):.6e} "
            f"fock_max_abs={np.max(np.abs(fock_piece)):.6e} "
            f"coeff_max_abs={np.max(np.abs(coeff_matrix)):.6e}"
        )

    interaction_h = build_interaction_hamiltonian(
        state.density,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        state.v0,
    )
    state.hamiltonian[:, :, :] = state.h0 + interaction_h
    print_matrix_norms("interaction_h", interaction_h)
    print_matrix_norms("hamiltonian_total", state.hamiltonian)

    density_new, energies, sigma_ztauz, mu = density_builder(
        state.hamiltonian,
        state.sigma_z,
        nu=state.nu,
    )
    delta_density = density_new - state.density
    oda_lambda = oda_parametrization_restricted(
        state,
        delta_density,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
    )

    print(f"updated_mu={mu:.12f}")
    print(f"updated_offdiag_flavor_norm={offdiag_flavor_norm(density_new):.6e}")
    print_matrix_norms("density_new", density_new)
    print_matrix_norms("delta_density", delta_density)
    print(f"delta_density_fro={np.linalg.norm(delta_density):.6e}")
    print(f"delta_density_max_abs={np.max(np.abs(delta_density)):.6e}")
    print(f"oda_lambda={oda_lambda:.12f}")
    print(f"occupied_sigma_mean={np.mean(sigma_ztauz[:2, 0]):.6e}")
    print(f"lowest_eigs_k0={','.join(f'{val:.12f}' for val in energies[:8, 0])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the first HF iteration for a bundled B0 benchmark case.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    parser.add_argument("--mode", choices=("full", "restricted"), default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_case(args.benchmark_id, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
