#!/usr/bin/env python3

from __future__ import annotations

import argparse

import numpy as np

from mean_field.systems.tbg.zero_field import build_b0_reference_parameters, build_b0_uniform_lattice, calculate_overlap, calculate_overlap_compact, solve_bm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Python BM grid overlap blocks for selected reciprocal shifts.")
    parser.add_argument("--theta-deg", type=float, default=1.20)
    parser.add_argument("--lk", type=int, default=19)
    parser.add_argument("--lg", type=int, default=9)
    return parser.parse_args()


def summarize(name: str, matrix: np.ndarray) -> str:
    mid = matrix.shape[0] // 2
    return (
        f"{name}\t"
        f"fro_norm={np.linalg.norm(matrix):.16e}\t"
        f"max_abs={np.max(np.abs(matrix)):.16e}\t"
        f"trace_real={np.trace(matrix).real:.16e}\t"
        f"trace_imag={np.trace(matrix).imag:.16e}\t"
        f"entry_11=({matrix[0,0].real:.16e},{matrix[0,0].imag:.16e})\t"
        f"entry_mid=({matrix[mid,mid].real:.16e},{matrix[mid,mid].imag:.16e})"
    )


def main() -> int:
    args = parse_args()
    params = build_b0_reference_parameters(args.theta_deg)
    lattice = build_b0_uniform_lattice(params, args.lk)
    solution = solve_bm_model(params, lattice.kvec, lg=args.lg, sigma_rotation=True)

    print(f"python_grid\ttheta={args.theta_deg:.2f}\tlk={args.lk}\tlg={args.lg}\tnk={solution.nk}")
    for shift in ((0, 0), (1, 0), (0, 1)):
        full = calculate_overlap(solution, shift[0], shift[1])
        compact = calculate_overlap_compact(solution, shift[0], shift[1], valley_index=0)
        print(summarize(f"python_full\tG=({shift[0]},{shift[1]})", full))
        print(summarize(f"python_compact\tG=({shift[0]},{shift[1]})", compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
