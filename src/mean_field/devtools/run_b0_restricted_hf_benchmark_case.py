#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import socket
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from mean_field import load_b0_suite
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    compare_hf_path_to_reference,
    evaluate_restricted_hf_path,
    run_restricted_hf_from_bm_solution,
    solve_bm_model,
    write_hf_path_nodes_tsv,
    write_hf_path_summary,
    write_hf_path_tsv,
)
from mean_field.systems.tbg.zero_field.model import build_b0_uniform_lattice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one packaged B0 benchmark case with the current Python restricted-HF candidate and export path bands."
    )
    parser.add_argument("benchmark_id", help="Benchmark case identifier from benchmarks/b0/benchmark_manifest.tsv")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "python_restricted_hf_benchmark"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum number of SCF iterations.")
    parser.add_argument("--precision", type=float, default=1e-5, help="SCF convergence threshold.")
    parser.add_argument(
        "--points-per-segment",
        type=int,
        default=None,
        help="Optional override for the exported path discretization.",
    )
    return parser.parse_args()


def case_tag(theta_deg: float, nu: int, init_mode: str, seed: int, lk: int, lg: int) -> str:
    return f"theta_{round(theta_deg * 100):03d}_nu_{round(nu * 1000):+05d}_init_{init_mode}_seed_{seed:03d}_lk{lk}_lg{lg}"


def write_key_value_file(path: Path, entries: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")


def main() -> int:
    args = parse_args()
    suite = load_b0_suite()
    case = suite.get(args.benchmark_id)

    output_root = Path(args.output_root).resolve()
    case_dir = output_root / case.benchmark_id
    state_dir = case_dir / "states"
    path_dir = case_dir / "path_bands"
    state_dir.mkdir(parents=True, exist_ok=True)
    path_dir.mkdir(parents=True, exist_ok=True)

    tag = case_tag(case.theta_deg, case.nu, case.init_mode, case.seed, case.lk, case.lg)
    state_path = state_dir / f"{tag}.npz"
    path_tsv = path_dir / f"{tag}_hf_path.tsv"
    node_tsv = path_dir / f"{tag}_hf_path_nodes.tsv"
    summary_txt = path_dir / f"{tag}_hf_path_summary.txt"
    parity_txt = case_dir / "parity_to_reference_summary.txt"
    runtime_txt = case_dir / "runtime_summary.txt"

    points_per_segment = case.points_per_segment if args.points_per_segment is None else int(args.points_per_segment)

    params = TBGParameters.from_degrees(case.theta_deg, vf=2482.0, w0=77.0, w1=110.0, strain=0.0, alpha=0.5, deformation_potential=0.0)
    grid = build_b0_uniform_lattice(params, case.lk)

    wall_start = datetime.now()
    total_t0 = time.perf_counter()

    print(f"[stage] bm:start benchmark_id={case.benchmark_id} lk={case.lk} lg={case.lg}", flush=True)
    bm_t0 = time.perf_counter()
    bm_solution = solve_bm_model(params, grid.kvec, lg=case.lg, sigma_rotation=True)
    bm_elapsed = time.perf_counter() - bm_t0
    print(f"[stage] bm:done elapsed_sec={bm_elapsed:.3f}", flush=True)

    print(
        f"[stage] hf:start nu={case.nu} init_mode={case.init_mode} seed={case.seed} max_iter={args.max_iter} precision={args.precision}",
        flush=True,
    )
    hf_t0 = time.perf_counter()
    hf_run = run_restricted_hf_from_bm_solution(
        bm_solution,
        nu=float(case.nu),
        init_mode=case.init_mode,
        seed=case.seed,
        max_iter=args.max_iter,
        precision=args.precision,
    )
    hf_elapsed = time.perf_counter() - hf_t0
    print(
        f"[stage] hf:done elapsed_sec={hf_elapsed:.3f} iterations={hf_run.iterations} converged={str(hf_run.converged).lower()} exit_reason={hf_run.exit_reason}",
        flush=True,
    )

    np.savez_compressed(
        state_path,
        density=hf_run.state.density,
        hamiltonian=hf_run.state.hamiltonian,
        h0=hf_run.state.h0,
        energies=hf_run.state.energies,
        sigma_ztauz=hf_run.state.sigma_ztauz,
        sigma_z=hf_run.state.sigma_z,
        mu=np.asarray([hf_run.state.mu], dtype=float),
        iter_energy=hf_run.iter_energy,
        iter_err=hf_run.iter_err,
        iter_oda=hf_run.iter_oda,
        nu=np.asarray([hf_run.state.nu], dtype=float),
        init_mode=np.asarray([case.init_mode]),
        normalized_init_mode=np.asarray([hf_run.init_mode]),
        seed=np.asarray([case.seed], dtype=int),
        converged=np.asarray([hf_run.converged]),
        exit_reason=np.asarray([hf_run.exit_reason]),
        theta_deg=np.asarray([case.theta_deg], dtype=float),
        lk=np.asarray([case.lk], dtype=int),
        lg=np.asarray([case.lg], dtype=int),
    )
    print(f"[stage] state:done path={state_path}", flush=True)

    print(f"[stage] path:start points_per_segment={points_per_segment}", flush=True)
    path_t0 = time.perf_counter()
    path_result = evaluate_restricted_hf_path(
        hf_run,
        bm_solution,
        points_per_segment=points_per_segment,
        lg=case.lg,
        init_mode=case.init_mode,
    )
    path_elapsed = time.perf_counter() - path_t0
    print(f"[stage] path:done elapsed_sec={path_elapsed:.3f}", flush=True)

    write_hf_path_tsv(path_tsv, path_result)
    write_hf_path_nodes_tsv(node_tsv, path_result)
    write_hf_path_summary(summary_txt, path_result, hf_state_path=str(state_path))
    print(f"[stage] export:done path_tsv={path_tsv}", flush=True)

    print(f"[stage] parity:start reference={case.reference_path_tsv_path}", flush=True)
    reference = case.load_reference_path()
    parity = compare_hf_path_to_reference(reference, path_result)
    write_key_value_file(
        parity_txt,
        [
            ("benchmark_id", case.benchmark_id),
            ("implementation", "python_restricted_hf_candidate"),
            ("reference_impl", "b0_reference"),
            ("reference_path_tsv", str(case.reference_path_tsv_path)),
            ("kdist_max_abs_diff", f"{parity.kdist_max_abs_diff}"),
            ("max_abs_band_diff_mev", f"{parity.max_abs_band_diff_mev}"),
            ("rms_band_diff_mev", f"{parity.rms_band_diff_mev}"),
            ("mean_abs_band_diff_mev", f"{parity.mean_abs_band_diff_mev}"),
            ("energy_sorting", parity.energy_sorting),
        ],
    )
    print(
        f"[stage] parity:done max_abs_band_diff_mev={parity.max_abs_band_diff_mev:.6e} rms_band_diff_mev={parity.rms_band_diff_mev:.6e}",
        flush=True,
    )

    total_elapsed = time.perf_counter() - total_t0
    wall_end = datetime.now()
    write_key_value_file(
        runtime_txt,
        [
            ("benchmark_id", case.benchmark_id),
            ("implementation", "python_restricted_hf_candidate"),
            ("theta_deg", f"{case.theta_deg:.2f}"),
            ("nu", str(case.nu)),
            ("init_mode", case.init_mode),
            ("normalized_init_mode", hf_run.init_mode),
            ("seed", str(case.seed)),
            ("lk", str(case.lk)),
            ("lg", str(case.lg)),
            ("points_per_segment", str(points_per_segment)),
            ("start_time", wall_start.strftime("%Y-%m-%dT%H:%M:%S")),
            ("end_time", wall_end.strftime("%Y-%m-%dT%H:%M:%S")),
            ("bm_elapsed_sec", f"{bm_elapsed}"),
            ("hf_elapsed_sec", f"{hf_elapsed}"),
            ("path_elapsed_sec", f"{path_elapsed}"),
            ("total_elapsed_sec", f"{total_elapsed}"),
            ("hostname", socket.gethostname()),
            ("state_path", str(state_path)),
            ("path_tsv", str(path_tsv)),
            ("path_exit_reason", hf_run.exit_reason),
            ("converged", str(hf_run.converged).lower()),
        ],
    )
    print(f"[stage] runtime:done total_elapsed_sec={total_elapsed:.3f}", flush=True)

    print(f"Completed benchmark_id={case.benchmark_id}")
    print(f"State: {state_path}")
    print(f"Path bands: {path_tsv}")
    print(f"Parity summary: {parity_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
