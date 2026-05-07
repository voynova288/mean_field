from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mean_field import load_b0_suite
from mean_field.systems.tbg.zero_field import build_b0_uniform_lattice, build_overlap_block_set
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    _hex_shell_contains,
    _screened_coulomb_matrix,
    contract_fock_term_from_overlap,
    initialize_full_state,
    screened_coulomb,
)
from mean_field.systems.tbg.zero_field.model import solve_bm_model
from mean_field.systems.tbg.zero_field.overlap import compute_density_overlap_trace
from mean_field.systems.tbg.zero_field.runners import build_b0_reference_parameters


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


def inspect_case(benchmark_id: str) -> None:
    suite = load_b0_suite()
    case = suite.get(benchmark_id)
    params = build_b0_reference_parameters(case.theta_deg)
    grid = build_b0_uniform_lattice(params, case.lk)
    solution = solve_bm_model(params, grid.kvec, lg=case.lg, sigma_rotation=True)
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=case.nu)
    overlap_blocks = build_overlap_block_set(solution, lg=case.lg)

    override_path = case.case_dir / f"initial_density_{case.init_mode}_seed_{case.seed:03d}.tsv"
    initial_density = None
    if override_path.is_file():
        initial_density = _load_initial_density_override(override_path, nt=state.nt, nk=state.nk)
    initialize_full_state(state, init_mode=case.init_mode, seed=case.seed, initial_density=initial_density)

    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)))
    print("m\tn\ting_shell\thartree_coeff\ttr_pg_real\ttr_pg_imag\ttr_pg_abs\tfock_fro\tfock_max_abs")
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        overlap = overlap_blocks.overlaps[shift]
        coeff_matrix = state.v0 * _screened_coulomb_matrix(
            solution.lattice_kvec[None, :] - solution.lattice_kvec[:, None] + complex(gvec),
            lm,
        ) / solution.nk
        fock_piece = contract_fock_term_from_overlap(overlap, state.density, coeff_matrix)
        tr_pg = compute_density_overlap_trace(state.density, overlap)
        print(
            f"{shift[0]}\t{shift[1]}\t{str(_hex_shell_contains(params, complex(gvec))).lower()}\t"
            f"{state.v0 * screened_coulomb(complex(gvec), lm) / solution.nk:.16e}\t"
            f"{tr_pg.real:.16e}\t{tr_pg.imag:.16e}\t{abs(tr_pg):.16e}\t"
            f"{np.linalg.norm(fock_piece):.16e}\t{np.max(np.abs(fock_piece)):.16e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump first-step HF shift metrics for all reciprocal shifts.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_case(args.benchmark_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
