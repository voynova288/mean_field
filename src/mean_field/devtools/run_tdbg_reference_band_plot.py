from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.core.lattice import KPath, cumulative_distance
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.tdbg import (
    PathBandsResult,
    TDBGModel,
    TDBGParameters,
    TDBGPathPlotTrace,
    compare_against_pytwist_reference,
    write_tdbg_path_band_plot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_NPZ = (
    Path("/data/home/ziyuzhu/pytwist/custom_outputs")
    / "tdbg_theta_1p33_phi_0p0_epsilon_0p0_D_0p0"
    / "band_data.npz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TDBG noninteracting band plots and compare against the pytwist reference.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.33)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--delta-ev", type=float, default=0.0)
    parser.add_argument("--cut", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--stacking", choices=("AB-AB", "AB-BA"), default="AB-AB")
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE_NPZ)
    parser.add_argument("--window-ev", type=float, default=0.07)
    return parser.parse_args()


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"tdbg_reference_alignment_{job_id}"
    else:
        stem = f"tdbg_reference_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return REPO_ROOT / "results" / "TDBG" / stem


def _reference_path_from_npz(path: Path, labels: tuple[str, ...], node_indices: tuple[int, ...]) -> tuple[PathBandsResult, PathBandsResult]:
    payload = np.load(path)
    kxy = np.asarray(payload["kpath"], dtype=float)
    kvec = np.asarray(kxy[:, 0] + 1j * kxy[:, 1], dtype=np.complex128)
    kpath = KPath(
        kvec=kvec,
        kdist=cumulative_distance(kvec),
        labels=labels,
        node_indices=node_indices,
    )
    return (
        PathBandsResult(path=kpath, energies=np.asarray(payload["evals_m"], dtype=float)),
        PathBandsResult(path=kpath, energies=np.asarray(payload["evals_p"], dtype=float)),
    )


def main() -> None:
    total_start = perf_counter()
    args = _parse_args()
    ensure_not_running_compute_on_login_node("TDBG reference band plot")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = TDBGParameters.full(
        stacking=args.stacking,
        Delta=args.delta_ev,
        phi_deg=args.phi_deg,
        epsilon=args.epsilon,
    )
    model = TDBGModel.from_config(args.theta_deg, cut=args.cut, params=params)
    path = model.standard_kpath(resolution=args.resolution)

    result_minus = model.bands_along_path(path, valley=-1, n_bands=model.matrix_dim)
    result_plus = model.bands_along_path(path, valley=1, n_bands=model.matrix_dim)

    np.savez_compressed(
        output_dir / "computed_band_data.npz",
        evals_m=result_minus.energies,
        evals_p=result_plus.energies,
        kpath=np.stack([path.kvec.real, path.kvec.imag], axis=-1),
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        epsilon=args.epsilon,
        delta_ev=args.delta_ev,
        stacking=args.stacking,
        cut=args.cut,
        resolution=args.resolution,
    )

    computed_paths = write_tdbg_path_band_plot(
        output_dir,
        (
            TDBGPathPlotTrace(label="K'", path_result=result_minus, color="tab:blue", linestyle="--", linewidth=0.55, alpha=0.8),
            TDBGPathPlotTrace(label="K", path_result=result_plus, color="tab:orange", linestyle="-", linewidth=0.55, alpha=0.8),
        ),
        stem="tdbg_bands",
        title=(
            f"TDBG {args.stacking}, theta={args.theta_deg:.2f}°, "
            f"phi={args.phi_deg:.2f}°, epsilon={args.epsilon:.4f}, Delta={args.delta_ev * 1.0e3:.1f} meV"
        ),
        ylim=(-abs(args.window_ev), abs(args.window_ev)),
    )

    summary: dict[str, object] = {
        "parameters": {
            "theta_deg": args.theta_deg,
            "phi_deg": args.phi_deg,
            "epsilon": args.epsilon,
            "delta_ev": args.delta_ev,
            "cut": args.cut,
            "resolution": args.resolution,
            "stacking": args.stacking,
        },
        "lattice_summary": model.lattice_summary(),
        "artifacts": {
            "computed_band_data_npz": str(output_dir / "computed_band_data.npz"),
            "tdbg_bands_png": str(computed_paths["band_plot_png"]),
            "tdbg_bands_pdf": str(computed_paths["band_plot_pdf"]),
        },
    }

    overlay_paths: dict[str, Path] = {}
    if args.reference_npz.exists():
        comparison = compare_against_pytwist_reference(model, args.reference_npz)
        reference_minus, reference_plus = _reference_path_from_npz(
            args.reference_npz,
            path.labels,
            path.node_indices,
        )
        overlay_paths = write_tdbg_path_band_plot(
            output_dir,
            (
                TDBGPathPlotTrace(label="ref K'", path_result=reference_minus, color="#9ecae1", linestyle="--", linewidth=0.55, alpha=0.55),
                TDBGPathPlotTrace(label="ref K", path_result=reference_plus, color="#fdd0a2", linestyle="-", linewidth=0.55, alpha=0.55),
                TDBGPathPlotTrace(label="calc K'", path_result=result_minus, color="tab:blue", linestyle="--", linewidth=0.45, alpha=0.9),
                TDBGPathPlotTrace(label="calc K", path_result=result_plus, color="tab:orange", linestyle="-", linewidth=0.45, alpha=0.9),
            ),
            stem="tdbg_reference_overlay",
            title=f"TDBG reference overlay, theta={args.theta_deg:.2f}°, stacking={args.stacking}",
            ylim=(-abs(args.window_ev), abs(args.window_ev)),
        )
        summary["reference_npz"] = str(args.reference_npz)
        summary["reference_alignment"] = {
            "resolution": comparison.resolution,
            "kpath_shape": list(comparison.kpath_shape),
            "band_shape": list(comparison.band_shape),
            "kpath_max_abs_diff_nm_inv": comparison.kpath_max_abs_diff,
            "evals_minus_max_abs_diff_ev": comparison.evals_minus_max_abs_diff,
            "evals_plus_max_abs_diff_ev": comparison.evals_plus_max_abs_diff,
        }
        summary["artifacts"].update(
            {
                "tdbg_reference_overlay_png": str(overlay_paths["band_plot_png"]),
                "tdbg_reference_overlay_pdf": str(overlay_paths["band_plot_pdf"]),
            }
        )

    total_elapsed = perf_counter() - total_start
    summary["runtime"] = {
        "total_elapsed_sec": total_elapsed,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    write_json(output_dir / "alignment_summary.json", summary)

    report_lines = [
        "# TDBG Reference Alignment",
        "",
        "## Parameters",
        "",
        f"- `theta_deg = {args.theta_deg}`",
        f"- `phi_deg = {args.phi_deg}`",
        f"- `epsilon = {args.epsilon}`",
        f"- `delta_ev = {args.delta_ev}`",
        f"- `cut = {args.cut}`",
        f"- `resolution = {args.resolution}`",
        f"- `stacking = {args.stacking}`",
        "",
        "## Artifacts",
        "",
        f"- `tdbg_bands.png = {computed_paths['band_plot_png']}`",
        f"- `tdbg_bands.pdf = {computed_paths['band_plot_pdf']}`",
    ]
    if overlay_paths:
        report_lines.extend(
            [
                f"- `tdbg_reference_overlay.png = {overlay_paths['band_plot_png']}`",
                f"- `tdbg_reference_overlay.pdf = {overlay_paths['band_plot_pdf']}`",
            ]
        )
    if "reference_alignment" in summary:
        alignment = summary["reference_alignment"]
        report_lines.extend(
            [
                "",
                "## Reference Alignment",
                "",
                f"- `kpath_max_abs_diff_nm_inv = {alignment['kpath_max_abs_diff_nm_inv']:.6e}`",
                f"- `evals_minus_max_abs_diff_ev = {alignment['evals_minus_max_abs_diff_ev']:.6e}`",
                f"- `evals_plus_max_abs_diff_ev = {alignment['evals_plus_max_abs_diff_ev']:.6e}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- `total_elapsed_sec = {total_elapsed:.6f}`",
            "",
        ]
    )
    (output_dir / "alignment_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[done] output_dir={output_dir}")
    print(f"tdbg_bands_png={computed_paths['band_plot_png']}")
    if overlay_paths:
        print(f"tdbg_reference_overlay_png={overlay_paths['band_plot_png']}")
    print(f"alignment_summary_json={output_dir / 'alignment_summary.json'}")


if __name__ == "__main__":
    main()
