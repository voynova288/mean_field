#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from mean_field import load_bm_unstrained_overlap_references
from mean_field.systems.tbg.zero_field import calculate_overlap_compact, run_bm_unstrained, summarize_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect deterministic BM gauge choices against the compact overlap reference.")
    parser.add_argument("theta_deg", type=float)
    parser.add_argument("lattice_kind", choices=("path", "grid"))
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    return parser.parse_args()


def _load_reference(theta_deg: float, lattice_kind: str, m: int, n: int):
    rounded = round(theta_deg, 2)
    matches = [
        row
        for row in load_bm_unstrained_overlap_references()
        if round(row.theta_deg, 2) == rounded and row.lattice_kind == lattice_kind and row.m == m and row.n == n and row.valley_label == "K"
    ]
    if not matches:
        raise SystemExit(f"No overlap reference for theta={theta_deg}, lattice_kind={lattice_kind}, G=({m},{n})")
    return matches[0]


def _phase_from_mode(vec: np.ndarray, mode: str, *, tol: float = 1e-12) -> complex:
    if mode == "raw":
        return 1.0 + 0.0j
    if mode == "anchor_max":
        anchor = vec[int(np.argmax(np.abs(vec)))]
    elif mode == "anchor_first":
        nz = np.flatnonzero(np.abs(vec) > tol)
        if nz.size == 0:
            return 1.0 + 0.0j
        anchor = vec[int(nz[0])]
    elif mode == "anchor_sum":
        anchor = np.sum(vec)
    else:
        raise ValueError(f"Unsupported gauge mode: {mode}")
    if abs(anchor) <= tol:
        return 1.0 + 0.0j
    return anchor / abs(anchor)


def _apply_gauge(uk: np.ndarray, mode: str) -> np.ndarray:
    gauged = np.array(uk, copy=True, order="F")
    if mode == "raw":
        return gauged
    _, nb, n_eta, nk = gauged.shape
    for ik in range(nk):
        for ieta in range(n_eta):
            for ib in range(nb):
                vec = gauged[:, ib, ieta, ik]
                phase = _phase_from_mode(vec, mode)
                vec /= phase
    return gauged


def _max_scalar_error(diag, ref) -> float:
    return max(
        abs(diag.fro_norm - ref.fro_norm),
        abs(diag.max_abs - ref.max_abs),
        abs(diag.trace_real - ref.trace_real),
        abs(diag.trace_imag - ref.trace_imag),
        abs(diag.entry_11_real - ref.entry_11_real),
        abs(diag.entry_11_imag - ref.entry_11_imag),
        abs(diag.entry_mid_real - ref.entry_mid_real),
        abs(diag.entry_mid_imag - ref.entry_mid_imag),
    )


def _print_mode(mode: str, diag, ref) -> None:
    print(
        f"mode={mode}\t"
        f"max_abs_scalar_error={_max_scalar_error(diag, ref):.16e}\t"
        f"fro_norm={diag.fro_norm:.16e}\t"
        f"max_abs={diag.max_abs:.16e}\t"
        f"trace_real={diag.trace_real:.16e}\t"
        f"trace_imag={diag.trace_imag:.16e}\t"
        f"entry_11=({diag.entry_11_real:.16e},{diag.entry_11_imag:.16e})\t"
        f"entry_mid=({diag.entry_mid_real:.16e},{diag.entry_mid_imag:.16e})"
    )


def main() -> int:
    args = parse_args()
    theta_deg = round(args.theta_deg, 2)
    ref = _load_reference(theta_deg, args.lattice_kind, args.m, args.n)
    run = run_bm_unstrained(theta_deg, points_per_segment=120, lg=9, grid_lk=33 if args.lattice_kind == "grid" else 0)
    solution = run.path_solution if args.lattice_kind == "path" else run.grid_solution
    if solution is None:
        raise SystemExit("Grid solution was not computed.")

    print(
        f"reference\t"
        f"fro_norm={ref.fro_norm:.16e}\t"
        f"max_abs={ref.max_abs:.16e}\t"
        f"trace_real={ref.trace_real:.16e}\t"
        f"trace_imag={ref.trace_imag:.16e}\t"
        f"entry_11=({ref.entry_11_real:.16e},{ref.entry_11_imag:.16e})\t"
        f"entry_mid=({ref.entry_mid_real:.16e},{ref.entry_mid_imag:.16e})"
    )
    for mode in ("raw", "anchor_max", "anchor_first", "anchor_sum"):
        gauged_solution = replace(solution, uk=_apply_gauge(solution.uk, mode))
        overlap = calculate_overlap_compact(gauged_solution, args.m, args.n, valley_index=0)
        diag = summarize_overlap(theta_deg, args.lattice_kind, overlap, args.m, args.n, valley_label="K")
        _print_mode(mode, diag, ref)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
