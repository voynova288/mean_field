from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mean_field.crpa import CRPACoulombParams, write_crpa_outputs
from mean_field.crpa.band_classifier import classify_flat_bands
from mean_field.crpa.bm import read_all_band_bm_solution
from mean_field.crpa.grid import build_uniform_crpa_grid
from mean_field.crpa.plotting import write_epsilon_vs_q_plot
from mean_field.crpa.validation import validation_summary, write_validation_report
from mean_field.crpa.workflow import compute_crpa_from_solution


def _parse_range(value: str) -> tuple[int, int]:
    pieces = str(value).replace(",", ":").split(":")
    if len(pieces) != 2:
        raise ValueError(f"Expected q range as start:stop, got {value!r}")
    start, stop = int(pieces[0]), int(pieces[1])
    if stop <= start:
        raise ValueError(f"Expected q range stop > start, got {value!r}")
    return start, stop


def _chunk_range(n_items: int, chunk_index: int, chunk_count: int) -> tuple[int, int]:
    if chunk_count <= 0:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError(f"chunk_index must be in [0, {chunk_count}), got {chunk_index}")
    start = (n_items * chunk_index) // chunk_count
    stop = (n_items * (chunk_index + 1)) // chunk_count
    return start, stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute one q-point chunk from a cached all-band BM solution.")
    parser.add_argument("--bm-solution", type=Path, required=True)
    parser.add_argument("--q-lg", type=int, required=True)
    parser.add_argument("--epsilon-bn", type=float, default=4.0)
    parser.add_argument("--ds-angstrom", type=float, default=400.0)
    parser.add_argument("--eta-mev", type=float, default=1.0)
    parser.add_argument("--q-range", default=None, help="Flat q-index range start:stop, stop exclusive.")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument(
        "--form-factor-mode",
        choices=("zhang_zero_fill", "hf_periodic"),
        default="zhang_zero_fill",
        help="Plane-wave form-factor convention for this chunk.",
    )
    parser.add_argument(
        "--hf-compatible",
        action="store_true",
        help="Alias for --form-factor-mode=hf_periodic; requires a periodic-G BM cache.",
    )
    parser.add_argument(
        "--occupation-mode",
        choices=("cnp_index", "energy_step"),
        default="cnp_index",
        help="Reference occupation for cRPA. Production Zhang/HF chunks use cnp_index.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    solution = read_all_band_bm_solution(args.bm_solution)
    lk_float = math.sqrt(float(solution.nk))
    lk = int(round(lk_float))
    if lk * lk != solution.nk:
        raise ValueError(f"Cached BM solution nk={solution.nk} is not a square k grid")
    grid = build_uniform_crpa_grid(solution.params, lk)
    if not np.allclose(grid.kvec, solution.lattice_kvec):
        raise ValueError("Cached BM k vectors do not match the reconstructed CRPA grid")

    if args.q_range is not None:
        start, stop = _parse_range(args.q_range)
    elif args.chunk_index is not None and args.chunk_count is not None:
        start, stop = _chunk_range(grid.nk, int(args.chunk_index), int(args.chunk_count))
    else:
        start, stop = 0, grid.nk
    start = max(0, int(start))
    stop = min(grid.nk, int(stop))
    if stop <= start:
        raise ValueError(f"Empty q chunk after clipping: {start}:{stop}")

    q_indices = [grid.unravel_index(iq) for iq in range(start, stop)]
    form_factor_mode = "hf_periodic" if args.hf_compatible else str(args.form_factor_mode)
    if form_factor_mode == "hf_periodic" and not bool(solution.periodic_g_grid):
        raise ValueError("HF-compatible cRPA chunks require a BM cache prepared with --periodic-g-grid.")
    classification = classify_flat_bands(solution.spectrum, method="center")
    coulomb = CRPACoulombParams(epsilon_bn=float(args.epsilon_bn), ds_angstrom=float(args.ds_angstrom))
    bands_per_valley = None if solution.nb == solution.basis_dimension else int(solution.nb)
    theta_deg = float(solution.params.dtheta_rad) * 180.0 / math.pi

    print(
        "[crpa-chunk] start "
        f"bm={args.bm_solution} lk={lk} lg={solution.lg} q_lg={args.q_lg} "
        f"q_range={start}:{stop} q_points={len(q_indices)} form_factor_mode={form_factor_mode} "
        f"occupation_mode={args.occupation_mode}",
        flush=True,
    )
    result = compute_crpa_from_solution(
        solution,
        classification,
        grid,
        theta_deg=theta_deg,
        q_lg=int(args.q_lg),
        bands_per_valley=bands_per_valley,
        q_indices=q_indices,
        coulomb_params=coulomb,
        eta_mev=float(args.eta_mev),
        form_factor_mode=form_factor_mode,
        occupation_mode=str(args.occupation_mode),
        flat_method="center",
    )
    out = write_crpa_outputs(result, args.output_dir)
    write_epsilon_vs_q_plot(result, out / "epsilon_vs_q.pdf")
    report_path = write_validation_report(result, out / "validation_report.md")
    summary = validation_summary(result)
    print(f"[crpa-chunk] wrote outputs to {out}", flush=True)
    print(f"[crpa-chunk] report={report_path}", flush=True)
    print(
        "[crpa-chunk] summary "
        f"eps_times_bn_max={summary['effective_epsilon_times_bn_max']:.6g} "
        f"q_points={len(q_indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
