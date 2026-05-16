from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.systems.RnG_hBN import (
    PathBandsResult,
    RLGhBNModel,
    RLGhBNPathPlotTrace,
    neutrality_energy_mev,
    validate_physics,
    write_rlg_hbn_path_band_plot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_REFERENCE = REPO_ROOT / "reference" / "2312.11617v1.pdf"
DEFAULT_V_VALUES = (-48.0, -24.0, 0.0, 24.0, 48.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the R5G/hBN paper Fig. 2 single-particle band plot.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=0.77)
    parser.add_argument("--shell-count", type=int, default=4)
    parser.add_argument("--points-per-segment", type=int, default=24)
    parser.add_argument("--bands-per-side", type=int, default=48)
    parser.add_argument("--layer-count", type=int, default=5)
    parser.add_argument("--valley", type=int, default=1)
    parser.add_argument("--source-dir", type=Path, default=None, help="Replot an existing raw Fig. 2 output directory.")
    return parser.parse_args()


def _load_plot_backend():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    return plt


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"r5g_theta0p77_paper_fig2_neutrality_{job_id or timestamp}"
    return REPO_ROOT / "results" / "RnG_hBN" / stem


def _paper_fig2_path(model: RLGhBNModel, points_per_segment: int):
    return model.build_kpath(
        (model.lattice.gamma_m, model.lattice.k_m, model.lattice.m_m, model.lattice.gamma_m, model.lattice.kprime_m),
        ("Gamma", "K", "M", "Gamma", "Kprime"),
        (
            int(points_per_segment),
            int(points_per_segment),
            int(points_per_segment),
            int(points_per_segment),
        ),
    )


def _select_band_window(total_bands: int, flat_pair: tuple[int, int], bands_per_side: int) -> tuple[int, ...]:
    start = max(0, int(flat_pair[0]) - int(bands_per_side))
    stop = min(int(total_bands), int(flat_pair[1]) + int(bands_per_side) + 2)
    return tuple(range(start, stop))


def _v_tag(value: float) -> str:
    rounded = int(round(float(value)))
    if rounded < 0:
        return f"Vm{abs(rounded)}"
    return f"V{rounded}"


def _title(xi: int, displacement_mev: float) -> str:
    return f"xi = {xi}, V = {int(round(displacement_mev))} meV"


def _node_ticks(path_result: PathBandsResult) -> tuple[list[float], list[str]]:
    ticks = [float(node.k_dist) for node in path_result.path.nodes]
    labels = [r"$\tilde{\Gamma}_M$", r"$\tilde{K}_M$", r"$\tilde{M}_M$", r"$\tilde{\Gamma}_M$", r"$\tilde{K}'_M$"]
    return ticks, labels


def _load_source_panel(source_dir: Path, xi: int, displacement_mev: float) -> tuple[PathBandsResult, tuple[int, ...], tuple[int, int], float]:
    tag = f"paper_fig2_r5g_xi{xi}_{_v_tag(displacement_mev)}meV"
    source_path = source_dir / f"{tag}_bands_path.npz"
    if not source_path.exists():
        raise FileNotFoundError(f"Expected source panel data at {source_path}")
    data = np.load(source_path, allow_pickle=True)
    energies = np.asarray(data["energies_mev"], dtype=float)
    band_min = int(data["band_min_index"])
    band_max = int(data["band_max_index"])
    flat_pair = (int(data["flat_valence_band_index"]), int(data["flat_conduction_band_index"]))
    selected_indices = tuple(range(band_min, band_max + 1))
    local_valence = flat_pair[0] - band_min
    local_conduction = flat_pair[1] - band_min
    if local_valence < 0 or local_conduction >= energies.shape[1]:
        raise ValueError(f"Source data {source_path} does not contain central flat bands {flat_pair}")
    energy_zero = 0.5 * (float(np.max(energies[:, local_valence])) + float(np.min(energies[:, local_conduction])))
    path = KPath(
        kvec=np.asarray(data["kvec"], dtype=np.complex128),
        kdist=np.asarray(data["kdist"], dtype=float),
        labels=tuple(str(label) for label in data["labels"]),
        node_indices=tuple(int(index) for index in data["node_indices"]),
    )
    return PathBandsResult(path=path, energies=energies, eigenvectors=None), selected_indices, flat_pair, energy_zero


def _write_summary_figure(output_dir: Path, panels: dict[tuple[int, float], dict[str, object]]) -> dict[str, Path]:
    plt = _load_plot_backend()
    fig, axes = plt.subplots(2, 5, figsize=(14.2, 4.9), sharey=True)
    for row, xi in enumerate((0, 1)):
        for col, displacement_mev in enumerate(DEFAULT_V_VALUES):
            panel = panels[(xi, displacement_mev)]
            path_result = panel["path_result"]
            assert isinstance(path_result, PathBandsResult)
            energies = np.asarray(panel["selected_energies_mev"], dtype=float)
            energy_zero = float(panel["energy_zero_mev"])
            ax = axes[row, col]
            for band_index in range(energies.shape[1]):
                ax.plot(path_result.path.kdist, energies[:, band_index] - energy_zero, color="red", linewidth=0.55)
            ticks, labels = _node_ticks(path_result)
            for xpos in ticks:
                ax.axvline(x=xpos, color="#cfcfcf", linewidth=0.45)
            ax.axhline(y=0.0, color="#9a9a9a", linewidth=0.45)
            ax.set_xticks(ticks, labels)
            ax.set_xlim(ticks[0], ticks[-1])
            ax.set_ylim(-150.0, 150.0)
            ax.set_title(_title(xi, displacement_mev), fontsize=9, pad=3)
            ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.4)
            if col == 0:
                ax.set_ylabel("Energy (meV)", fontsize=8)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.13, wspace=0.18, hspace=0.32)
    png_path = output_dir / "paper_fig2_r5g_hbn_band_structure_neutrality.png"
    pdf_path = output_dir / "paper_fig2_r5g_hbn_band_structure_neutrality.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"summary_png": png_path, "summary_pdf": pdf_path}


def main() -> None:
    args = _parse_args()
    start_time = datetime.now().isoformat(timespec="seconds")
    start = perf_counter()
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panels: dict[tuple[int, float], dict[str, object]] = {}
    metadata: dict[str, object] = {
        "paper_reference": str(PAPER_REFERENCE),
        "target": "2312.11617v1 Fig. 2",
        "theta_deg": float(args.theta_deg),
        "shell_count": int(args.shell_count),
        "points_per_segment": int(args.points_per_segment),
        "bands_per_side": int(args.bands_per_side),
        "layer_count": int(args.layer_count),
        "valley": int(args.valley),
        "energy_reference": "path central-gap midpoint: 0.5 * (max E_v + min E_c)",
        "y_window_mev": [-150.0, 150.0],
        "panels": [],
    }
    report_lines = [
        "# R5G/hBN Paper Fig. 2 Band Plot",
        "",
        f"- generated_at: `{start_time}`",
        f"- paper_reference: `{PAPER_REFERENCE}`",
        "- target: `2312.11617v1 Fig. 2`",
        f"- output_dir: `{output_dir}`",
        f"- theta_deg: `{args.theta_deg}`",
        f"- shell_count: `{args.shell_count}`",
        f"- layer_count: `{args.layer_count}`",
        f"- path: `Gamma_M -> K_M -> M_M -> Gamma_M -> Kprime_M`",
        "- V columns: `-48, -24, 0, 24, 48 meV`",
        "- rows: `xi=0`, `xi=1`",
        "- energy_reference: `path central-gap midpoint`",
        "",
        "## Debug Note",
        "",
        (
            "The previous plot used raw eigenvalues.  Eq. (A14)/(A20) has an arbitrary fitted onsite "
            "offset, while the paper figure is plotted relative to charge neutrality.  Each panel here "
            "subtracts the midpoint between the top central valence band and bottom central conduction band "
            "along the plotted path."
        ),
        (
            "The plotted moire path uses the single-valley K-sector mBZ edge "
            "Gamma -> K -> M -> Gamma -> Kprime with M=(g_m1+g_m2)/2 and adjacent corners "
            "(2*g_m1+g_m2)/3 and (g_m1+2*g_m2)/3.  The opposite M orbit is not equivalent "
            "before applying time reversal to the opposite valley; this is the branch that "
            "matches the remote-band shape in the paper's xi=0, V=48 meV panel."
        ),
        "",
        "## Panels",
        "",
        "| xi | V_meV | E_neutral_meV | band_window | flat_valence | flat_conduction | png | npz |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]

    for xi in (0, 1):
        for displacement_mev in DEFAULT_V_VALUES:
            model = RLGhBNModel.from_config(
                layer_count=int(args.layer_count),
                xi=xi,
                theta_deg=float(args.theta_deg),
                displacement_field_mev=float(displacement_mev),
                shell_count=int(args.shell_count),
            )
            if args.source_dir is not None:
                print(f"[panel] xi={xi} V={displacement_mev:g} meV replot source", flush=True)
                path_result, selected_indices, flat_pair, energy_zero = _load_source_panel(Path(args.source_dir), xi, displacement_mev)
                selected_energies = np.asarray(path_result.energies, dtype=float)
            else:
                path = _paper_fig2_path(model, int(args.points_per_segment))
                flat_pair = model.flat_band_indices
                selected_indices = _select_band_window(model.matrix_dim, flat_pair, int(args.bands_per_side))
                print(f"[panel] xi={xi} V={displacement_mev:g} meV bands=0..{selected_indices[-1]}", flush=True)
                path_result = model.bands_along_path(path, valley=int(args.valley), n_bands=selected_indices[-1] + 1)
                selected_energies = np.asarray(path_result.energies[:, selected_indices], dtype=float)
                energy_zero = neutrality_energy_mev(path_result, model.lattice, model.params)

            tag = f"paper_fig2_r5g_xi{xi}_{_v_tag(displacement_mev)}meV"
            selected_result = PathBandsResult(path=path_result.path, energies=selected_energies, eigenvectors=None)
            plot_paths = write_rlg_hbn_path_band_plot(
                output_dir,
                (RLGhBNPathPlotTrace(label=tag, path_result=selected_result, color="red", linewidth=0.55, energy_shift_mev=energy_zero),),
                stem=f"{tag}_neutrality",
                title=_title(xi, displacement_mev),
                ylim=(-150.0, 150.0),
            )
            npz_path = output_dir / f"{tag}_bands_path_neutrality.npz"
            np.savez_compressed(
                npz_path,
                kvec=np.asarray(path_result.path.kvec, dtype=np.complex128),
                kdist=np.asarray(path_result.path.kdist, dtype=float),
                labels=np.asarray(path_result.path.labels, dtype=object),
                node_indices=np.asarray(path_result.path.node_indices, dtype=int),
                band_indices=np.asarray(selected_indices, dtype=int),
                energies_mev_raw=selected_energies,
                energies_mev_shifted=selected_energies - energy_zero,
                energy_zero_mev=np.asarray(energy_zero, dtype=float),
                flat_valence_band_index=np.asarray(flat_pair[0], dtype=int),
                flat_conduction_band_index=np.asarray(flat_pair[1], dtype=int),
            )
            info_path = output_dir / f"{tag}_info.json"
            info = {
                "xi": xi,
                "displacement_field_mev": displacement_mev,
                "energy_zero_mev": energy_zero,
                "band_indices": list(selected_indices),
                "flat_valence_band_index": int(flat_pair[0]),
                "flat_conduction_band_index": int(flat_pair[1]),
                "bands_npz": str(npz_path),
                "band_plot_png": str(plot_paths["band_plot_png"]),
                "band_plot_pdf": str(plot_paths["band_plot_pdf"]),
            }
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

            panels[(xi, displacement_mev)] = {
                "path_result": path_result,
                "selected_energies_mev": selected_energies,
                "selected_indices": selected_indices,
                "energy_zero_mev": energy_zero,
            }
            metadata["panels"].append(info)
            report_lines.append(
                f"| {xi} | {displacement_mev:.1f} | {energy_zero:.6f} | "
                f"{selected_indices[0]}..{selected_indices[-1]} | {flat_pair[0]} | {flat_pair[1]} | "
                f"`{Path(plot_paths['band_plot_png']).name}` | `{npz_path.name}` |"
            )

    summary_paths = _write_summary_figure(output_dir, panels)
    validation = validate_physics(
        RLGhBNModel.from_config(
            layer_count=int(args.layer_count),
            xi=1,
            theta_deg=float(args.theta_deg),
            displacement_field_mev=24.0,
            shell_count=int(args.shell_count),
        )
    )
    validation_path = output_dir / "validation_report.md"
    validation_path.write_text(validation.to_markdown(), encoding="utf-8")

    elapsed = perf_counter() - start
    report_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary_png: `{summary_paths['summary_png']}`",
            f"- summary_pdf: `{summary_paths['summary_pdf']}`",
            f"- validation_report: `{validation_path}`",
            f"- metadata_json: `{output_dir / 'run_metadata.json'}`",
            "",
            "## Runtime",
            "",
            f"- elapsed_sec: `{elapsed:.6f}`",
            "",
        ]
    )
    report_path = output_dir / "paper_fig2_neutrality_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metadata["artifacts"] = {
        "summary_png": str(summary_paths["summary_png"]),
        "summary_pdf": str(summary_paths["summary_pdf"]),
        "report_md": str(report_path),
        "validation_report_md": str(validation_path),
    }
    metadata["runtime"] = {"elapsed_sec": elapsed, "start_time": start_time}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPO_ROOT / "results" / "RnG_hBN" / "LATEST_PAPER_FIG2_NEUTRALITY.txt").write_text(str(output_dir) + "\n", encoding="utf-8")

    print(f"[done] output_dir={output_dir}")
    print(f"summary_png={summary_paths['summary_png']}")
    print(f"summary_pdf={summary_paths['summary_pdf']}")
    print(f"report_md={report_path}")


if __name__ == "__main__":
    main()
