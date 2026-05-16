from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, select_flat_pair_window, write_json
from mean_field.systems.tmbg import (
    TMBGBandPlotPanel,
    TMBGModel,
    TMBGParameters,
    infer_flat_band_indices,
    write_tmbg_paper_band_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Fig. 2-like tMBG band plot on the standard K-Gamma-M-K' path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.21)
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--valley", type=int, default=1)
    parser.add_argument("--bands-per-side", type=int, default=6)
    return parser.parse_args()


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"tmbg_fig2_like_{job_id}"
    else:
        stem = f"tmbg_fig2_like_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return REPO_ROOT / "results" / "TMBG" / stem


def _panel_label(delta_ev: float) -> str:
    delta_mev = int(round(delta_ev * 1000.0))
    if delta_mev > 0:
        return f"Δ = +{delta_mev} meV"
    if delta_mev < 0:
        return f"Δ = {delta_mev} meV"
    return "Δ = 0 meV"


def _display_k_label(label: str) -> str:
    return {"Gamma": "Γ", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(label, label)


def _gap_annotation(delta_ev: float, path_result, flat_pair: tuple[int, int]) -> tuple[str, float, str]:
    energies = np.asarray(path_result.energies, dtype=float)
    flat_gaps = energies[:, flat_pair[1]] - energies[:, flat_pair[0]]
    gap_index = int(np.argmin(flat_gaps))
    gap_mev = float(flat_gaps[gap_index] * 1.0e3)

    location = None
    for node in path_result.path.nodes:
        if int(node.index - 1) == gap_index:
            location = _display_k_label(node.label)
            break
    if location is None:
        kvec = complex(path_result.path.kvec[gap_index])
        location = f"({kvec.real:+.4f}, {kvec.imag:+.4f}) nm^-1"

    delta_label = _panel_label(delta_ev).replace("Δ = ", "")
    annotation = f"flat_gap @ Δ={delta_label}: {gap_mev:.2f} meV at k={location}"
    return annotation, gap_mev, location


def main() -> None:
    start_time = datetime.now().isoformat(timespec="seconds")
    total_start = perf_counter()
    args = _parse_args()
    ensure_not_running_compute_on_login_node("tMBG Fig. 2-like band plot")
    args.output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valley_label = "K" if args.valley == 1 else "K'"

    delta_values = (0.0, 0.060, -0.040)
    panels: list[TMBGBandPlotPanel] = []
    metadata: dict[str, object] = {
        "theta_deg": args.theta_deg,
        "n_shells": args.n_shells,
        "points_per_segment": args.points_per_segment,
        "valley": args.valley,
        "bands_per_side": args.bands_per_side,
        "window_mev": [-100.0, 100.0],
        "panels": [],
    }
    report_lines = [
        "# tMBG Fig. 2-like Band Plot",
        "",
        f"生成时间：{start_time}",
        "",
        "## Parameters",
        "",
        f"- `theta_deg = {args.theta_deg}`",
        f"- `n_shells = {args.n_shells}`",
        f"- `points_per_segment = {args.points_per_segment}`",
        f"- `valley = {args.valley}`",
        f"- `bands_per_side = {args.bands_per_side}`",
        "- `y_window = [-100, +100] meV`",
        "",
        "## Panels",
        "",
    ]

    for delta_ev in delta_values:
        params = TMBGParameters.full(interlayer_potential=delta_ev, staggered_potential=0.0)
        model = TMBGModel.from_config(args.theta_deg, n_shells=args.n_shells, params=params)
        path_result = model.bands_along_standard_path(
            points_per_segment=args.points_per_segment,
            valley=args.valley,
            n_bands=model.lattice.matrix_dim,
        )
        flat_pair = infer_flat_band_indices(path_result.energies)
        selected_indices = select_flat_pair_window(
            path_result.energies.shape[1],
            flat_pair,
            args.bands_per_side,
            mode="center",
        )
        selected_energies = np.asarray(path_result.energies[:, selected_indices], dtype=float)
        local_lookup = {int(index): ilocal for ilocal, index in enumerate(selected_indices)}
        local_flat_pair = tuple(int(local_lookup[int(index)]) for index in flat_pair)
        flat_gaps = selected_energies[:, local_flat_pair[1]] - selected_energies[:, local_flat_pair[0]]
        min_gap_mev = float(np.min(flat_gaps) * 1.0e3)

        panel_dir = args.output_dir / f"delta_{int(round(delta_ev * 1000.0)):+04d}mev"
        panel_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            panel_dir / "bands_path.npz",
            k_distance=np.asarray(path_result.path.kdist, dtype=float),
            energies=selected_energies,
            kvec_nm_inv=np.stack(
                [
                    np.asarray(path_result.path.kvec.real, dtype=float),
                    np.asarray(path_result.path.kvec.imag, dtype=float),
                ],
                axis=-1,
            ),
            band_indices=np.asarray(selected_indices, dtype=int),
            flat_band_indices=np.asarray(flat_pair, dtype=int),
            k_labels=np.asarray(path_result.path.labels, dtype=object),
        )

        annotation, min_gap_mev, min_gap_location = _gap_annotation(delta_ev, path_result, flat_pair)

        panels.append(
            TMBGBandPlotPanel(
                label=_panel_label(delta_ev),
                path_result=path_result,
                band_indices=selected_indices,
                flat_band_indices=flat_pair,
                annotation=annotation,
            )
        )
        metadata["panels"].append(
            {
                "delta_ev": delta_ev,
                "label": _panel_label(delta_ev),
                "selected_band_indices": list(selected_indices),
                "flat_band_indices": list(flat_pair),
                "flat_gap_mev": min_gap_mev,
                "flat_gap_location": min_gap_location,
                "annotation": annotation,
            }
        )
        report_lines.extend(
            [
                f"### {_panel_label(delta_ev)}",
                "",
                f"- `selected_band_indices = {list(selected_indices)}`",
                f"- `flat_band_indices = {list(flat_pair)}`",
                f"- `flat_gap_meV = {min_gap_mev:.3f}`",
                f"- `flat_gap_location = {min_gap_location}`",
                "",
            ]
        )

    plot_paths = write_tmbg_paper_band_figure(
        args.output_dir,
        tuple(panels),
        title=f"tMBG full model, theta={args.theta_deg:.2f}°, valley={valley_label}",
        ylim=(-0.100, 0.100),
    )
    total_elapsed = perf_counter() - total_start
    report_path = args.output_dir / "fig2_like_bands_report.md"
    report_lines.extend(
        [
            "## Artifacts",
            "",
            f"- `fig2_like_bands.png = {plot_paths['paper_band_plot_png']}`",
            f"- `fig2_like_bands.pdf = {plot_paths['paper_band_plot_pdf']}`",
            f"- `run_metadata.json = {args.output_dir / 'run_metadata.json'}`",
            "",
            "## Runtime",
            "",
            f"- `total_elapsed_sec = {total_elapsed:.6f}`",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metadata["artifacts"] = {
        "paper_band_plot_png": str(plot_paths["paper_band_plot_png"]),
        "paper_band_plot_pdf": str(plot_paths["paper_band_plot_pdf"]),
        "fig2_like_bands_report_md": str(report_path),
    }
    metadata["runtime"] = {
        "start_time": start_time,
        "total_elapsed_sec": total_elapsed,
    }

    write_json(args.output_dir / "run_metadata.json", metadata, sort_keys=False)

    print(f"[done] output_dir={args.output_dir}")
    for key, value in plot_paths.items():
        print(f"{key}={value}")
    print(f"fig2_like_bands_report_md={report_path}")


if __name__ == "__main__":
    main()
