#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from mean_field.runtime import collect_runtime_environment, current_timestamp
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import build_b0_benchmark_kpath, solve_bm_model, write_path_band_plot


def _ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().strip().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit(
            f"Refusing to run {workload_name} on login node {hostname}; submit it through Slurm from login002."
        )


def _format_tag(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def _default_output_dir(args: argparse.Namespace) -> Path:
    stem = (
        "custom_bm_band_"
        f"theta_{_format_tag(args.theta_deg)}_"
        f"vf_{_format_tag(args.vf)}_"
        f"w0_{_format_tag(args.w0)}_"
        f"w1_{_format_tag(args.w1)}_"
        f"mu_{_format_tag(args.chemical_potential)}_"
        f"strain_{_format_tag(args.strain)}"
    )
    return REPO_ROOT / "results" / "BM" / stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a custom zero-field BM path-band calculation and write plot/artifact files.")
    parser.add_argument("--theta-deg", type=float, required=True, help="Twist angle in degrees.")
    parser.add_argument("--vf", type=float, default=2416.0, help="Dirac velocity prefactor.")
    parser.add_argument("--chemical-potential", type=float, default=0.0, help="Chemical potential in meV.")
    parser.add_argument("--w0", type=float, default=88.0, help="AA tunneling in meV.")
    parser.add_argument("--w1", type=float, default=110.0, help="AB tunneling in meV.")
    parser.add_argument("--delta", type=float, default=0.0, help="Staggered potential. Only delta=0 is currently supported here.")
    parser.add_argument("--strain", type=float, default=0.0, help="Uniaxial strain amplitude.")
    parser.add_argument("--strain-angle-rad", type=float, default=0.0, help="Strain angle in radians.")
    parser.add_argument("--poisson", type=float, default=0.16, help="Poisson ratio.")
    parser.add_argument("--beta-g", type=float, default=3.14, help="Gauge-coupling beta_g.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Layer strain partition factor.")
    parser.add_argument("--deformation-potential", type=float, default=0.0, help="Deformation potential term.")
    parser.add_argument("--lg", type=int, default=9, help="Reciprocal-space shell cutoff.")
    parser.add_argument("--points-per-segment", type=int, default=120, help="Number of samples on each M-K-Gamma/Gamma-M segment.")
    parser.add_argument("--stem", default="band_plot", help="Output figure stem.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to results/BM/<parameterized-stem>.")
    return parser.parse_args()


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _write_key_value_summary(path: Path, entries: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def main() -> int:
    args = parse_args()
    if args.theta_deg <= 0.0:
        raise SystemExit("theta_deg must be positive.")
    if args.points_per_segment <= 0:
        raise SystemExit("points_per_segment must be positive.")
    if args.lg <= 0:
        raise SystemExit("lg must be positive.")
    if abs(args.delta) > 0.0:
        raise SystemExit("This runner currently supports only delta=0.0 because the zero-field BM solver path does not consume nonzero delta.")

    _ensure_not_running_compute_on_login_node("custom zero-field BM band plot")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = TBGParameters(
        dtheta_rad=float(args.theta_deg) * math.pi / 180.0,
        convention="b0",
        vf=float(args.vf),
        chemical_potential=float(args.chemical_potential),
        w0=float(args.w0),
        w1=float(args.w1),
        delta=float(args.delta),
        strain=float(args.strain),
        strain_angle_rad=float(args.strain_angle_rad),
        poisson=float(args.poisson),
        beta_g=float(args.beta_g),
        alpha=float(args.alpha),
        deformation_potential=float(args.deformation_potential),
    )

    start_time = current_timestamp()
    solve_start = perf_counter()
    path = build_b0_benchmark_kpath(params, int(args.points_per_segment))
    path_solution = solve_bm_model(params, path.kvec, lg=int(args.lg), sigma_rotation=True)
    elapsed_sec = perf_counter() - solve_start
    end_time = current_timestamp()

    energy_array = np.asarray(path_solution.flattened_energies(), dtype=float)
    energy_rows = energy_array.T
    node_k_index = path.node_indices[1] - 1
    k_energies = np.sort(energy_array[:, node_k_index])
    k_middle_gap_mev = float(k_energies[4] - k_energies[3])

    path_rows: list[dict[str, str]] = []
    band_columns = [f"band_{index + 1:02d}_meV" for index in range(energy_array.shape[0])]
    for index, (kvec, kdist, row) in enumerate(zip(path.kvec, path.kdist, energy_rows, strict=True), start=1):
        entry = {
            "sample_index": str(index),
            "k_dist": f"{float(kdist):.16e}",
            "kx": f"{float(np.real(kvec)):.16e}",
            "ky": f"{float(np.imag(kvec)):.16e}",
        }
        entry.update({column: f"{float(value):.16e}" for column, value in zip(band_columns, row, strict=True)})
        path_rows.append(entry)

    node_rows = [
        {
            "label": node.label,
            "index": str(node.index),
            "k_dist": f"{node.k_dist:.16e}",
            "kx": f"{node.kx:.16e}",
            "ky": f"{node.ky:.16e}",
        }
        for node in path.nodes
    ]

    path_tsv_path = _write_tsv(
        output_dir / "computed_bm_path.tsv",
        ["sample_index", "k_dist", "kx", "ky", *band_columns],
        path_rows,
    )
    nodes_tsv_path = _write_tsv(
        output_dir / "computed_nodes.tsv",
        ["label", "index", "k_dist", "kx", "ky"],
        node_rows,
    )
    plot_paths = write_path_band_plot(
        output_dir,
        stem=args.stem,
        kdist=np.asarray(path.kdist, dtype=float),
        energies=energy_rows,
        path=path,
        title=f"theta={args.theta_deg:.2f}°, vf={args.vf:.1f}, w0/w1={args.w0:.1f}/{args.w1:.1f} meV",
    )

    summary_path = _write_key_value_summary(
        output_dir / "computed_summary.txt",
        [
            ("theta_deg", f"{args.theta_deg:.16e}"),
            ("vf", f"{args.vf:.16e}"),
            ("chemical_potential", f"{args.chemical_potential:.16e}"),
            ("w0", f"{args.w0:.16e}"),
            ("w1", f"{args.w1:.16e}"),
            ("delta", f"{args.delta:.16e}"),
            ("strain", f"{args.strain:.16e}"),
            ("strain_angle_rad", f"{args.strain_angle_rad:.16e}"),
            ("poisson", f"{args.poisson:.16e}"),
            ("beta_g", f"{args.beta_g:.16e}"),
            ("alpha", f"{args.alpha:.16e}"),
            ("deformation_potential", f"{args.deformation_potential:.16e}"),
            ("lg", str(args.lg)),
            ("points_per_segment", str(args.points_per_segment)),
            ("path_point_count", str(path.kvec.size)),
            ("band_count", str(energy_array.shape[0])),
            ("k_middle_gap_meV", f"{k_middle_gap_mev:.16e}"),
            ("elapsed_sec", f"{elapsed_sec:.16e}"),
            ("path_tsv", str(path_tsv_path)),
            ("nodes_tsv", str(nodes_tsv_path)),
            ("band_plot_png", str(plot_paths["band_plot_png"])),
            ("band_plot_pdf", str(plot_paths["band_plot_pdf"])),
        ],
    )

    metadata = {
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_sec": elapsed_sec,
        "runtime_environment": asdict(collect_runtime_environment()),
        "parameters": {
            "theta_deg": args.theta_deg,
            "vf": args.vf,
            "chemical_potential": args.chemical_potential,
            "w0": args.w0,
            "w1": args.w1,
            "delta": args.delta,
            "strain": args.strain,
            "strain_angle_rad": args.strain_angle_rad,
            "poisson": args.poisson,
            "beta_g": args.beta_g,
            "alpha": args.alpha,
            "deformation_potential": args.deformation_potential,
            "lg": args.lg,
            "points_per_segment": args.points_per_segment,
        },
        "artifacts": {
            "path_tsv": str(path_tsv_path),
            "nodes_tsv": str(nodes_tsv_path),
            "summary_txt": str(summary_path),
            "band_plot_png": str(plot_paths["band_plot_png"]),
            "band_plot_pdf": str(plot_paths["band_plot_pdf"]),
        },
        "derived": {
            "path_point_count": int(path.kvec.size),
            "band_count": int(energy_array.shape[0]),
            "k_middle_gap_meV": k_middle_gap_mev,
        },
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"path_tsv={path_tsv_path}")
    print(f"nodes_tsv={nodes_tsv_path}")
    print(f"summary_txt={summary_path}")
    print(f"band_plot_png={plot_paths['band_plot_png']}")
    print(f"band_plot_pdf={plot_paths['band_plot_pdf']}")
    print(f"run_metadata_json={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
