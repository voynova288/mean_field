from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.systems.htg import (
    HTGModel,
    HTGParams,
    HTGPathPlotTrace,
    MAGIC_ALPHA_ZETA0,
    build_chiral_lattice_from_alpha,
    compute_chern_basis_on_grid,
    estimate_central_band_metrics,
    theta_deg_from_alpha,
    validate_lattice,
    validate_static_hamiltonian,
    write_htg_fig3b_plot,
    write_htg_path_band_plot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "HTG"


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Devakul 2023 h-HTG Fig. 2b and Fig. 3b band panels."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.5)
    parser.add_argument("--kappa", type=float, default=0.7)
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--points-per-segment", type=int, default=96)
    parser.add_argument("--central-band-count", type=int, default=28)
    parser.add_argument("--fig2-window-mev", type=float, default=150.0)
    parser.add_argument("--fig3-alphas", type=_parse_csv_floats, default=MAGIC_ALPHA_ZETA0[:2])
    parser.add_argument("--fig3-n-shells", type=int, default=None)
    parser.add_argument("--fig3-points-per-segment", type=int, default=96)
    parser.add_argument("--fig3-central-band-count", type=int, default=36)
    parser.add_argument("--fig3-window", type=float, default=0.7)
    parser.add_argument("--topology-mesh", type=int, default=9)
    parser.add_argument("--skip-topology", action="store_true")
    return parser.parse_args()


def _ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().strip().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit(
            f"Refusing to run {workload_name} on login node {hostname}; submit it through Slurm from login002."
        )


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"htg_fig2b_fig3b_{job_id}"
    else:
        stem = f"htg_fig2b_fig3b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_path_npz(path: Path, result, **metadata: object) -> None:
    np.savez_compressed(
        path,
        energies_ev=np.asarray(result.energies, dtype=float),
        kdist=np.asarray(result.path.kdist, dtype=float),
        kvec=np.stack([result.path.kvec.real, result.path.kvec.imag], axis=-1),
        labels=np.asarray(result.path.labels),
        node_indices=np.asarray(result.path.node_indices, dtype=int),
        band_indices=np.asarray(result.band_indices, dtype=int),
        **metadata,
    )


def _validation_payload(checks) -> list[dict[str, object]]:
    return [check.to_dict() for check in checks]


def _format_metric_mev(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000.0:.2f}"


def main() -> None:
    total_start = perf_counter()
    args = _parse_args()
    _ensure_not_running_compute_on_login_node("HTG Fig. 2b/Fig. 3b reproduction")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }

    params_fig2 = HTGParams.default(kappa=args.kappa)
    model_fig2 = HTGModel.from_config(args.theta_deg, n_shells=args.n_shells, params=params_fig2)
    path_fig2 = model_fig2.standard_kpath(points_per_segment=args.points_per_segment)
    fig2_result = model_fig2.bands_along_path(
        path_fig2,
        valley=1,
        central_band_count=args.central_band_count,
        return_eigenvectors=False,
    )
    fig2_metrics = estimate_central_band_metrics(fig2_result, model_fig2.matrix_dim)
    _save_json(output_dir / "lattice_info.json", model_fig2.lattice_summary())
    _save_path_npz(
        output_dir / "fig2b_bands_path.npz",
        fig2_result,
        theta_deg=float(args.theta_deg),
        kappa=float(args.kappa),
        n_shells=int(args.n_shells),
    )

    annotate = (
        f"theta={args.theta_deg:.3f} deg, kappa={args.kappa:.3f}\n"
        f"W_path={_format_metric_mev(fig2_metrics['central_bandwidth_ev'])} meV, "
        f"Egap_path={_format_metric_mev(fig2_metrics['remote_gap_ev'])} meV"
    )
    fig2_paths = write_htg_path_band_plot(
        output_dir,
        (
            HTGPathPlotTrace(
                label="K",
                path_result=fig2_result,
                color="#1f1f1f",
                linewidth=0.72,
                alpha=0.9,
                energy_scale=1000.0,
            ),
        ),
        stem="fig2b_htg_bands",
        title="h-HTG Fig. 2b reproduction",
        ylabel="Energy (meV)",
        ylim=(-abs(args.fig2_window_mev), abs(args.fig2_window_mev)),
        annotate=annotate,
    )

    validation = {
        "lattice": _validation_payload(validate_lattice(model_fig2.lattice)),
        "static_hamiltonian": _validation_payload(validate_static_hamiltonian(model_fig2.lattice, model_fig2.params)),
    }

    topology_payload: dict[str, object] | None = None
    if not args.skip_topology and args.topology_mesh > 1:
        topology_result = compute_chern_basis_on_grid(
            args.topology_mesh,
            model_fig2.lattice,
            model_fig2.params,
            valley=1,
            frac_shift=(0.5, 0.5),
        )
        topology_payload = topology_result.to_dict()
        _save_json(output_dir / "chern_numbers.json", topology_payload)

    fig3_panels = []
    fig3_summaries: list[dict[str, object]] = []
    fig3_n_shells = args.n_shells if args.fig3_n_shells is None else int(args.fig3_n_shells)
    params_chiral = HTGParams.chiral(zeta_rad=0.0)
    for alpha in args.fig3_alphas:
        theta_from_alpha = theta_deg_from_alpha(alpha, params=params_chiral)
        lattice = build_chiral_lattice_from_alpha(alpha, n_shells=fig3_n_shells, params=params_chiral)
        model = HTGModel(lattice=lattice, params=params_chiral)
        path = model.standard_kpath(points_per_segment=args.fig3_points_per_segment)
        result = model.bands_along_path(
            path,
            valley=1,
            central_band_count=args.fig3_central_band_count,
            return_eigenvectors=False,
        )
        key = f"alpha_{alpha:.3f}".replace(".", "p")
        _save_path_npz(
            output_dir / f"fig3b_{key}_bands_path.npz",
            result,
            alpha=float(alpha),
            theta_deg=float(theta_from_alpha),
            n_shells=int(fig3_n_shells),
            kappa=0.0,
            zeta_rad=0.0,
        )
        vk_theta = params_chiral.vk_theta_ev(lattice.k_theta)
        fig3_panels.append((f"alpha={alpha:.3f}", result, 1.0 / vk_theta))
        fig3_summaries.append(
            {
                "alpha": float(alpha),
                "theta_deg": float(theta_from_alpha),
                "vk_theta_ev": float(vk_theta),
                "lattice_summary": model.lattice_summary(),
                "path_band_metrics": estimate_central_band_metrics(result, model.matrix_dim),
            }
        )

    fig3_paths = write_htg_fig3b_plot(
        output_dir,
        tuple(fig3_panels),
        stem="fig3b_chiral_bands",
        ylim=(-abs(args.fig3_window), abs(args.fig3_window)),
    )

    elapsed = perf_counter() - total_start
    summary: dict[str, object] = {
        "parameters": {
            "theta_deg": float(args.theta_deg),
            "kappa": float(args.kappa),
            "n_shells": int(args.n_shells),
            "points_per_segment": int(args.points_per_segment),
            "central_band_count": int(args.central_band_count),
            "fig3_alphas": [float(value) for value in args.fig3_alphas],
            "fig3_n_shells": int(fig3_n_shells),
            "fig3_points_per_segment": int(args.fig3_points_per_segment),
            "fig3_central_band_count": int(args.fig3_central_band_count),
            "topology_mesh": int(args.topology_mesh),
            "skip_topology": bool(args.skip_topology),
        },
        "runtime": {**runtime, "total_elapsed_sec": float(elapsed)},
        "fig2b": {
            "lattice_summary": model_fig2.lattice_summary(),
            "path_band_metrics": fig2_metrics,
            "artifacts": {
                "bands_npz": str(output_dir / "fig2b_bands_path.npz"),
                "bands_png": str(fig2_paths["band_plot_png"]),
                "bands_pdf": str(fig2_paths["band_plot_pdf"]),
            },
        },
        "fig3b": {
            "panels": fig3_summaries,
            "artifacts": {
                "bands_png": str(fig3_paths["band_plot_png"]),
                "bands_pdf": str(fig3_paths["band_plot_pdf"]),
            },
        },
        "validation": validation,
    }
    if topology_payload is not None:
        summary["fig2b"]["chern_basis"] = topology_payload

    _save_json(output_dir / "run_metadata.json", summary)

    report_lines = [
        "# HTG Fig. 2b / Fig. 3b Reproduction",
        "",
        "## Runtime",
        "",
        f"- `hostname = {runtime['hostname']}`",
        f"- `slurm_job_id = {runtime['slurm_job_id']}`",
        f"- `elapsed_sec = {elapsed:.3f}`",
        "",
        "## Fig. 2b",
        "",
        f"- `theta_deg = {args.theta_deg}`",
        f"- `kappa = {args.kappa}`",
        f"- `n_shells = {args.n_shells}`",
        f"- `matrix_dim = {model_fig2.matrix_dim}`",
        f"- `alpha = {model_fig2.params.alpha(model_fig2.lattice.k_theta):.6f}`",
        f"- `W_path_mev = {_format_metric_mev(fig2_metrics['central_bandwidth_ev'])}`",
        f"- `Egap_path_mev = {_format_metric_mev(fig2_metrics['remote_gap_ev'])}`",
        f"- `bands_png = {fig2_paths['band_plot_png']}`",
        f"- `bands_pdf = {fig2_paths['band_plot_pdf']}`",
        "",
        "## Fig. 3b",
        "",
    ]
    for panel in fig3_summaries:
        metrics = panel["path_band_metrics"]
        report_lines.extend(
            [
                f"- `alpha = {panel['alpha']:.6f}`, `theta_deg = {panel['theta_deg']:.6f}`, "
                f"`W_path_over_vk = {metrics['central_bandwidth_ev'] / panel['vk_theta_ev'] if metrics['central_bandwidth_ev'] is not None else 'n/a'}`",
            ]
        )
    report_lines.extend(
        [
            f"- `bands_png = {fig3_paths['band_plot_png']}`",
            f"- `bands_pdf = {fig3_paths['band_plot_pdf']}`",
            "",
            "## Topology",
            "",
        ]
    )
    if topology_payload is None:
        report_lines.append("- `skipped = true`")
    else:
        report_lines.extend(
            [
                f"- `mesh_size = {topology_payload['mesh_size']}`",
                f"- `chern_a = {topology_payload['chern_a']:.8f}`",
                f"- `chern_b = {topology_payload['chern_b']:.8f}`",
                f"- `total_chern = {topology_payload['total_chern']:.8f}`",
                f"- `rounded_chern_a = {topology_payload['rounded_chern_a']}`",
                f"- `rounded_chern_b = {topology_payload['rounded_chern_b']}`",
                f"- `rounded_total_chern = {topology_payload['rounded_total_chern']}`",
                f"- `raw_chern_a = {topology_payload['raw_chern_a']:.8f}`",
                f"- `raw_chern_b = {topology_payload['raw_chern_b']:.8f}`",
            ]
        )
    report_lines.extend(["", "## Validation", ""])
    for group_name, checks in validation.items():
        for check in checks:
            report_lines.append(f"- `{group_name}.{check['name']} = {check['passed']}` (`value = {check['value']}`)")
    report_lines.append("")
    (output_dir / "validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[done] output_dir={output_dir}")
    print(f"fig2b_png={fig2_paths['band_plot_png']}")
    print(f"fig3b_png={fig3_paths['band_plot_png']}")
    print(f"run_metadata_json={output_dir / 'run_metadata.json'}")
    print(f"validation_report_md={output_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
