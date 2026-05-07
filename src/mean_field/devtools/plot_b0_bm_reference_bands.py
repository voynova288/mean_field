#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from mean_field import load_bm_unstrained_references
from mean_field.systems.tbg.zero_field import run_bm_unstrained

REFERENCE_COLOR = "#1f4e79"
MODEL_COLOR = "#d95d39"
PANEL_COLORS = (
    "#16324f",
    "#235789",
    "#4c8cbf",
    "#8cc5f4",
    "#8b3a3a",
    "#bf5f51",
    "#e08e79",
    "#f2c9b6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot packaged B0 BM reference bands and overlay the current Python reproduction.")
    parser.add_argument(
        "--theta",
        type=float,
        nargs="*",
        default=[1.20, 1.28],
        help="Twist angles to plot. Defaults to the packaged benchmark angles.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "bm_reference_bands"),
        help="Directory where the figure will be written.",
    )
    parser.add_argument(
        "--overlay-python",
        action="store_true",
        help="Also run the current Python BM solver and overlay the reproduced bands.",
    )
    return parser.parse_args()


def load_reference_map() -> dict[float, object]:
    return {round(ref.theta_deg, 2): ref for ref in load_bm_unstrained_references()}


def plot_case(ax: plt.Axes, theta: float, reference: object, *, overlay_python: bool) -> float | None:
    summary = reference.load_summary()
    ref_kdist, ref_rows = reference.load_path_data()
    ref_kdist_array = np.asarray(ref_kdist, dtype=float)
    ref_energy_array = np.asarray(ref_rows, dtype=float)
    max_err: float | None = None
    k_gap = float(summary["K_middle_gap_meV"])

    for band_index in range(ref_energy_array.shape[1]):
        color = PANEL_COLORS[band_index % len(PANEL_COLORS)]
        ax.plot(ref_kdist_array, ref_energy_array[:, band_index], color=color, lw=1.4, alpha=0.95)

    if overlay_python:
        points_per_segment = int(summary["points_per_segment"])
        lg = int(summary["lg"])
        run = run_bm_unstrained(theta, points_per_segment=points_per_segment, lg=lg, grid_lk=0)
        model_energy_array = run.path_solution.flattened_energies().T
        max_err = float(np.max(np.abs(model_energy_array - ref_energy_array)))
        k_gap = run.k_middle_gap_mev
        for band_index in range(model_energy_array.shape[1]):
            ax.plot(ref_kdist_array, model_energy_array[:, band_index], color=MODEL_COLOR, lw=0.9, ls="--", alpha=0.55)

    ref_nodes = reference.load_path_nodes()
    node_x = [node.k_dist for node in ref_nodes]
    node_labels = [node.label for node in ref_nodes]
    for xpos in node_x:
        ax.axvline(x=xpos, color="#999999", ls=":", lw=0.8, zorder=0)

    ax.set_xticks(node_x)
    ax.set_xticklabels(node_labels)
    ax.set_xlim(ref_kdist_array[0], ref_kdist_array[-1])
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (meV)")
    title = f"theta = {theta:.2f} deg | K-gap = {k_gap:.4f} meV"
    if max_err is not None:
        title += f" | max |Delta E| = {max_err:.2e} meV"
    ax.set_title(title, fontsize=10, pad=10)
    return max_err


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_map = load_reference_map()
    thetas = [round(theta, 2) for theta in args.theta]
    missing = [theta for theta in thetas if theta not in reference_map]
    if missing:
        raise SystemExit(f"Missing packaged BM references for theta={missing}")

    fig, axes = plt.subplots(1, len(thetas), figsize=(6.4 * len(thetas), 4.8), constrained_layout=True)
    if len(thetas) == 1:
        axes = [axes]

    max_errs: list[float] = []
    for ax, theta in zip(axes, thetas, strict=True):
        err = plot_case(ax, theta, reference_map[theta], overlay_python=args.overlay_python)
        if err is not None:
            max_errs.append(err)

    legend_handles = [Line2D([0], [0], color=REFERENCE_COLOR, lw=1.6, label="Packaged B0 reference")]
    if args.overlay_python:
        legend_handles.append(Line2D([0], [0], color=MODEL_COLOR, lw=1.0, ls="--", label="Current Python reproduction"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    if max_errs:
        title = f"B0 BM Reference Bands | worst max |Delta E| = {max(max_errs):.2e} meV"
    else:
        title = "B0 BM Reference Bands"
    fig.suptitle(title, fontsize=13, y=1.08)

    png_path = output_dir / "b0_bm_reference_bands.png"
    pdf_path = output_dir / "b0_bm_reference_bands.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
