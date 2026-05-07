from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mean_field import load_b0_suite
from mean_field.systems.tbg.zero_field import build_b0_uniform_lattice, build_overlap_block_set
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    _hex_shell_contains,
    build_interaction_hamiltonian,
    initialize_full_state,
    screened_coulomb,
)
from mean_field.systems.tbg.zero_field.model import solve_bm_model
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


def _screened_coulomb_matrix_direct(qvals: np.ndarray, lm: float) -> np.ndarray:
    values = np.zeros_like(qvals, dtype=float)
    for index, q in np.ndenumerate(qvals):
        values[index] = screened_coulomb(complex(q), lm)
    return values


def _direct_density_overlap_trace(density: np.ndarray, overlap: np.ndarray) -> complex:
    # This script mirrors the current Julia Hartree contraction exactly; it is a
    # parity check for the vectorized builder, not an independent physics audit.
    total = 0.0 + 0.0j
    for ik in range(density.shape[2]):
        total += np.trace(density[:, :, ik] @ np.conj(overlap[:, ik, :, ik]))
    return complex(total)


def _build_interaction_hamiltonian_direct(
    density: np.ndarray,
    overlaps: dict[tuple[int, int], np.ndarray],
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    lattice_kvec: np.ndarray,
    params,
    v0: float,
) -> np.ndarray:
    nt, _, nk = density.shape
    interaction = np.zeros((nt, nt, nk), dtype=np.complex128)
    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)))
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if not _hex_shell_contains(params, complex(gvec)):
            continue
        overlap = overlaps[shift]

        hartree_prefactor = v0 * screened_coulomb(complex(gvec), lm) / nk
        if hartree_prefactor != 0.0:
            tr_pg = _direct_density_overlap_trace(density, overlap)
            for ik in range(nk):
                interaction[:, :, ik] += hartree_prefactor * tr_pg * overlap[:, ik, :, ik]

        coeff_matrix = v0 * _screened_coulomb_matrix_direct(
            lattice_kvec[None, :] - lattice_kvec[:, None] + complex(gvec),
            lm,
        ) / nk
        for ik in range(nk):
            tmp_fock = np.zeros((nt, nt), dtype=np.complex128)
            for ip in range(nk):
                tmp_fock += coeff_matrix[ik, ip] * (
                    overlap[:, ik, :, ip] @ density[:, :, ip].T @ overlap[:, ik, :, ip].conj().T
                )
            interaction[:, :, ik] -= tmp_fock
    return interaction


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

    interaction_vectorized = build_interaction_hamiltonian(
        state.density,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        state.v0,
    )
    interaction_direct = _build_interaction_hamiltonian_direct(
        state.density,
        overlap_blocks.overlaps,
        overlap_blocks.shifts,
        overlap_blocks.gvecs,
        solution.lattice_kvec,
        solution.params,
        state.v0,
    )
    diff = interaction_vectorized - interaction_direct

    print(f"benchmark_id={case.benchmark_id}")
    print(f"initial_density_override={override_path.is_file()}")
    print(f"vectorized_fro={np.linalg.norm(interaction_vectorized):.12e}")
    print(f"direct_fro={np.linalg.norm(interaction_direct):.12e}")
    print(f"diff_fro={np.linalg.norm(diff):.12e}")
    print(f"diff_max_abs={np.max(np.abs(diff)):.12e}")
    print(f"diff_max_abs_offdiag={np.max(np.abs(diff - np.einsum('iik->iik', np.zeros_like(diff)))) if False else np.max(np.abs(diff)):.12e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vectorized and direct Julia-style HF interaction builders.")
    parser.add_argument("benchmark_id", help="Benchmark case id from the bundled B0 suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_case(args.benchmark_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
