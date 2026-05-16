from __future__ import annotations

import argparse
import time
from pathlib import Path

from mean_field.crpa.bm import solve_all_band_bm_model, write_all_band_bm_solution
from mean_field.crpa.grid import build_uniform_crpa_grid
from mean_field.systems.tbg.params import TBGParameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and save a reusable all-band BM eigensystem for TBG cRPA chunks.")
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.4)
    parser.add_argument("--lk", type=int, required=True)
    parser.add_argument("--lg", type=int, required=True)
    parser.add_argument("--bands-per-valley", type=int, default=None)
    parser.add_argument(
        "--periodic-g-grid",
        action="store_true",
        help="Use the legacy B0/Jacobian benchmark periodic G-grid tunneling convention instead of physical zero-fill.",
    )
    parser.add_argument(
        "--hf-compatible",
        action="store_true",
        help="Alias for --periodic-g-grid when preparing a BM cache for HF-compatible cRPA chunks.",
    )
    parser.add_argument("--compressed", action="store_true", help="Use compressed npz output. Smaller but slower for large lg.")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    params = TBGParameters.from_degrees(
        args.theta_deg,
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )
    grid = build_uniform_crpa_grid(params, int(args.lk))
    periodic_g_grid = bool(args.periodic_g_grid or args.hf_compatible)
    print(
        "[bm-cache] solving "
        f"theta={args.theta_deg} lk={args.lk} lg={args.lg} "
        f"bands_per_valley={args.bands_per_valley} periodic_g_grid={str(periodic_g_grid).lower()}",
        flush=True,
    )
    start = time.perf_counter()
    solution = solve_all_band_bm_model(
        params,
        grid.kvec,
        lg=int(args.lg),
        bands_per_valley=args.bands_per_valley,
        sigma_rotation=True,
        periodic_g_grid=periodic_g_grid,
    )
    solve_elapsed = time.perf_counter() - start
    write_start = time.perf_counter()
    path = write_all_band_bm_solution(solution, args.output_path, compressed=bool(args.compressed))
    write_elapsed = time.perf_counter() - write_start
    print(f"[bm-cache] wrote {path}", flush=True)
    print(
        "[bm-cache] summary "
        f"nk={solution.nk} nb={solution.nb} basis_dim={solution.basis_dimension} "
        f"solve_elapsed_sec={solve_elapsed:.3f} write_elapsed_sec={write_elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
