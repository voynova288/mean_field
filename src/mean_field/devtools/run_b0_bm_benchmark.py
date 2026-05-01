#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from mean_field import load_bm_unstrained_overlap_references, load_bm_unstrained_references
from mean_field.systems.tbg.zero_field import calculate_overlap_compact, run_bm_unstrained, summarize_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the active B0 zero-field BM benchmark comparisons and write machine-readable reports.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where benchmark comparison TSV and summary files will be written.",
    )
    parser.add_argument(
        "--benchmark-root",
        default=None,
        help="Optional benchmark root. Defaults to the active packaged BM helper benchmark under benchmarks/b0/bm_inputs/unstrained_path.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of twist angles to evaluate, for example: --theta 1.20 1.28",
    )
    return parser.parse_args()


def rounded_theta_set(values: list[float] | None) -> set[float] | None:
    if values is None:
        return None
    return {round(value, 2) for value in values}


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_root = None if args.benchmark_root is None else Path(args.benchmark_root).resolve()
    theta_filter = rounded_theta_set(args.theta)

    refs = [ref for ref in load_bm_unstrained_references(root=benchmark_root) if theta_filter is None or round(ref.theta_deg, 2) in theta_filter]
    overlap_refs = [
        ref
        for ref in load_bm_unstrained_overlap_references(root=benchmark_root)
        if theta_filter is None or round(ref.theta_deg, 2) in theta_filter
    ]
    if not refs:
        raise SystemExit("No BM benchmark references selected.")

    path_rows: list[dict[str, str]] = []
    overlap_rows: list[dict[str, str]] = []

    overlap_by_theta: dict[float, list] = {}
    for ref in overlap_refs:
        overlap_by_theta.setdefault(round(ref.theta_deg, 2), []).append(ref)

    for ref in refs:
        summary = ref.load_summary()
        theta = round(ref.theta_deg, 2)
        points_per_segment = int(summary["points_per_segment"])
        lg = int(summary["lg"])
        grid_lk = int(summary["grid_lk"])
        need_grid = any(row.lattice_kind == "grid" for row in overlap_by_theta.get(theta, ()))
        run = run_bm_unstrained(theta, points_per_segment=points_per_segment, lg=lg, grid_lk=grid_lk if need_grid else 0)

        ref_kdist, ref_energies = ref.load_path_data()
        ref_energy_array = np.asarray(ref_energies, dtype=float).T
        model_energy_array = run.path_solution.flattened_energies()
        max_err = float(np.max(np.abs(model_energy_array - ref_energy_array)))

        path_rows.append(
            {
                "theta_deg": f"{theta:.2f}",
                "path_points": str(model_energy_array.shape[1]),
                "band_count": str(model_energy_array.shape[0]),
                "max_abs_path_energy_error_meV": f"{max_err:.16e}",
                "k_middle_gap_meV": f"{run.k_middle_gap_mev:.16e}",
                "reference_k_middle_gap_meV": summary.get("K_middle_gap_meV", ""),
                "valence_bandwidth_meV": "" if run.valence_bandwidth_mev is None else f"{run.valence_bandwidth_mev:.16e}",
                "reference_valence_bandwidth_meV": summary.get("valence_bandwidth_meV", ""),
                "conduction_bandwidth_meV": "" if run.conduction_bandwidth_mev is None else f"{run.conduction_bandwidth_mev:.16e}",
                "reference_conduction_bandwidth_meV": summary.get("conduction_bandwidth_meV", ""),
            }
        )

        for row in overlap_by_theta.get(theta, ()):
            if row.lattice_kind == "path":
                solution = run.path_solution
            elif row.lattice_kind == "grid":
                if run.grid_solution is None:
                    raise RuntimeError(f"Grid overlap requested for theta={theta:.2f} but no grid solution was computed.")
                solution = run.grid_solution
            else:
                raise ValueError(f"Unsupported lattice_kind={row.lattice_kind}")

            overlap = calculate_overlap_compact(solution, row.m, row.n, valley_index=0)
            diag = summarize_overlap(theta, row.lattice_kind, overlap, row.m, row.n, valley_label="K")
            scalar_errors = {
                "fro_norm": abs(diag.fro_norm - row.fro_norm),
                "max_abs": abs(diag.max_abs - row.max_abs),
                "trace_real": abs(diag.trace_real - row.trace_real),
                "trace_imag": abs(diag.trace_imag - row.trace_imag),
                "entry_11_real": abs(diag.entry_11_real - row.entry_11_real),
                "entry_11_imag": abs(diag.entry_11_imag - row.entry_11_imag),
                "entry_mid_real": abs(diag.entry_mid_real - row.entry_mid_real),
                "entry_mid_imag": abs(diag.entry_mid_imag - row.entry_mid_imag),
            }
            overlap_rows.append(
                {
                    "theta_deg": f"{theta:.2f}",
                    "lattice_kind": row.lattice_kind,
                    "valley_label": row.valley_label,
                    "m": str(row.m),
                    "n": str(row.n),
                    "max_abs_scalar_error": f"{max(scalar_errors.values()):.16e}",
                    "fro_norm_error": f"{scalar_errors['fro_norm']:.16e}",
                    "max_abs_error": f"{scalar_errors['max_abs']:.16e}",
                    "trace_real_error": f"{scalar_errors['trace_real']:.16e}",
                    "trace_imag_error": f"{scalar_errors['trace_imag']:.16e}",
                    "entry_11_real_error": f"{scalar_errors['entry_11_real']:.16e}",
                    "entry_11_imag_error": f"{scalar_errors['entry_11_imag']:.16e}",
                    "entry_mid_real_error": f"{scalar_errors['entry_mid_real']:.16e}",
                    "entry_mid_imag_error": f"{scalar_errors['entry_mid_imag']:.16e}",
                }
            )

    write_tsv(
        output_dir / "path_benchmark.tsv",
        [
            "theta_deg",
            "path_points",
            "band_count",
            "max_abs_path_energy_error_meV",
            "k_middle_gap_meV",
            "reference_k_middle_gap_meV",
            "valence_bandwidth_meV",
            "reference_valence_bandwidth_meV",
            "conduction_bandwidth_meV",
            "reference_conduction_bandwidth_meV",
        ],
        path_rows,
    )
    write_tsv(
        output_dir / "overlap_benchmark.tsv",
        [
            "theta_deg",
            "lattice_kind",
            "valley_label",
            "m",
            "n",
            "max_abs_scalar_error",
            "fro_norm_error",
            "max_abs_error",
            "trace_real_error",
            "trace_imag_error",
            "entry_11_real_error",
            "entry_11_imag_error",
            "entry_mid_real_error",
            "entry_mid_imag_error",
        ],
        overlap_rows,
    )

    max_path_error = max(float(row["max_abs_path_energy_error_meV"]) for row in path_rows)
    max_overlap_error = max((float(row["max_abs_scalar_error"]) for row in overlap_rows), default=0.0)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"benchmark_root={benchmark_root if benchmark_root is not None else 'active_packaged_b0_benchmark'}\n")
        handle.write(f"angles={','.join(row['theta_deg'] for row in path_rows)}\n")
        handle.write(f"max_abs_path_energy_error_meV={max_path_error:.16e}\n")
        handle.write(f"max_abs_overlap_scalar_error={max_overlap_error:.16e}\n")

    print(f"Wrote {output_dir / 'path_benchmark.tsv'}")
    print(f"Wrote {output_dir / 'overlap_benchmark.tsv'}")
    print(f"Wrote {output_dir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
