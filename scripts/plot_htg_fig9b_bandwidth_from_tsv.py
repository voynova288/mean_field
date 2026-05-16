#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_THETA_GRID_DEG = [1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95]
EXPECTED_WAA_GRID_MEV = [40.0, 47.5, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _edges(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        step = 1.0
        return np.asarray([arr[0] - step / 2.0, arr[0] + step / 2.0])
    mids = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _pivot(rows: list[dict[str, str]]) -> tuple[list[float], list[float], np.ndarray]:
    theta_values = sorted({float(row["theta_deg"]) for row in rows})
    waa_values = sorted({float(row["wAA_mev"]) for row in rows})
    theta_index = {value: idx for idx, value in enumerate(theta_values)}
    waa_index = {value: idx for idx, value in enumerate(waa_values)}
    data = np.full((len(waa_values), len(theta_values)), np.nan, dtype=float)
    for row in rows:
        theta = float(row["theta_deg"])
        waa = float(row["wAA_mev"])
        value = row.get("wcond_mev", "")
        if value:
            data[waa_index[waa], theta_index[theta]] = float(value)
    return theta_values, waa_values, data


def _same_grid(left: list[float], right: list[float], tol: float = 1.0e-9) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


def _compact_label(row: dict[str, str]) -> str:
    label = row.get("class_compact_label", "").strip()
    if label:
        return label
    return row.get("class_label", "").strip().strip("[]").replace(" ", "")


def _phase_grid(rows: list[dict[str, str]], theta: list[float], waa: list[float]) -> np.ndarray:
    theta_index = {value: idx for idx, value in enumerate(theta)}
    waa_index = {value: idx for idx, value in enumerate(waa)}
    labels = np.full((len(waa), len(theta)), "", dtype=object)
    for row in rows:
        label = _compact_label(row)
        if not label:
            continue
        labels[waa_index[float(row["wAA_mev"])], theta_index[float(row["theta_deg"])]] = label
    return labels


def _draw_mask_outline(
    ax: plt.Axes,
    theta: list[float],
    waa: list[float],
    mask: np.ndarray,
    *,
    color: str = "0.20",
) -> None:
    x_edges = _edges(theta)
    y_edges = _edges(waa)
    n_y, n_x = mask.shape
    for i_y in range(n_y):
        for i_x in range(n_x):
            if not mask[i_y, i_x]:
                continue
            x0, x1 = x_edges[i_x], x_edges[i_x + 1]
            y0, y1 = y_edges[i_y], y_edges[i_y + 1]
            if i_x == 0 or not mask[i_y, i_x - 1]:
                ax.plot([x0, x0], [y0, y1], color=color, linestyle="--", linewidth=1.8)
            if i_x == n_x - 1 or not mask[i_y, i_x + 1]:
                ax.plot([x1, x1], [y0, y1], color=color, linestyle="--", linewidth=1.8)
            if i_y == 0 or not mask[i_y - 1, i_x]:
                ax.plot([x0, x1], [y0, y0], color=color, linestyle="--", linewidth=1.8)
            if i_y == n_y - 1 or not mask[i_y + 1, i_x]:
                ax.plot([x0, x1], [y1, y1], color=color, linestyle="--", linewidth=1.8)


def _mask_center(theta: list[float], waa: list[float], mask: np.ndarray) -> tuple[float, float] | None:
    if not np.any(mask):
        return None
    x_grid, y_grid = np.meshgrid(np.asarray(theta, dtype=float), np.asarray(waa, dtype=float))
    return float(np.mean(x_grid[mask])), float(np.mean(y_grid[mask]))


def _add_phase_annotations(
    ax: plt.Axes,
    theta: list[float],
    waa: list[float],
    rows: list[dict[str, str]],
) -> None:
    labels = _phase_grid(rows, theta, waa)
    d3a_mask = labels == "D3A"
    d3b_mask = labels == "D3B"
    if np.any(d3a_mask):
        _draw_mask_outline(ax, theta, waa, d3a_mask)

    bbox = {
        "boxstyle": "square,pad=0.12",
        "facecolor": "white",
        "edgecolor": "0.35",
        "linewidth": 0.8,
        "alpha": 0.72,
    }

    d3a_center = _mask_center(theta, waa, d3a_mask)
    if d3a_center is not None:
        ax.text(*d3a_center, r"$[D_3A]$", ha="center", va="center", fontsize=18, bbox=bbox)

    lower_half = np.asarray(waa, dtype=float)[:, None] <= float(np.median(waa))
    d3b_center = _mask_center(theta, waa, d3b_mask & lower_half)
    if d3b_center is not None:
        ax.text(*d3b_center, r"$[D_3B]$", ha="center", va="center", fontsize=18, bbox=bbox)

    fb_points = [
        (float(row["theta_deg"]), float(row["wAA_mev"]))
        for row in rows
        if row.get("family", "").strip() == "FB"
    ]
    if fb_points:
        ax.text(
            float(np.mean([point[0] for point in fb_points])),
            float(np.mean([point[1] for point in fb_points])),
            "FB",
            ha="center",
            va="center",
            fontsize=40,
            bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.50},
        )


def _save_paper_blocks(
    theta: list[float],
    waa: list[float],
    data: np.ndarray,
    rows: list[dict[str, str]],
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    mesh = ax.pcolormesh(
        _edges(theta),
        _edges(waa),
        data,
        cmap="viridis",
        vmin=0.0,
        vmax=35.0,
        shading="flat",
    )
    _add_phase_annotations(ax, theta, waa, rows)
    ax.set_title(r"$\nu=3$", fontsize=30, pad=12)
    ax.set_xlabel(r"$\theta\ (^\circ)$", fontsize=30, labelpad=12)
    ax.set_ylabel(r"$w_{AA}\ (\mathrm{meV})$", fontsize=30, labelpad=16)
    ax.set_xticks([1.6, 1.7, 1.8, 1.9])
    ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.tick_params(axis="both", labelsize=24, width=2.2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.052, pad=0.035)
    cbar.ax.set_title(r"$W_{\mathrm{cond.}}\ (\mathrm{meV})$", fontsize=22, pad=12)
    cbar.set_ticks([0, 10, 20, 30])
    cbar.ax.tick_params(labelsize=24, width=2.0, length=7)
    cbar.outline.set_linewidth(2.0)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _save_continuous(
    theta: list[float],
    waa: list[float],
    data: np.ndarray,
    rows: list[dict[str, str]],
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    mesh = ax.pcolormesh(
        np.asarray(theta, dtype=float),
        np.asarray(waa, dtype=float),
        data,
        cmap="viridis",
        vmin=0.0,
        vmax=35.0,
        shading="gouraud",
    )
    _add_phase_annotations(ax, theta, waa, rows)
    ax.set_title(r"$\nu=3$", fontsize=30, pad=12)
    ax.set_xlabel(r"$\theta\ (^\circ)$", fontsize=30, labelpad=12)
    ax.set_ylabel(r"$w_{AA}\ (\mathrm{meV})$", fontsize=30, labelpad=16)
    ax.set_xticks([1.6, 1.7, 1.8, 1.9])
    ax.set_yticks([40, 50, 60, 70, 80, 90])
    ax.tick_params(axis="both", labelsize=24, width=2.2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.052, pad=0.035)
    cbar.ax.set_title(r"$W_{\mathrm{cond.}}\ (\mathrm{meV})$", fontsize=22, pad=12)
    cbar.set_ticks([0, 10, 20, 30])
    cbar.ax.tick_params(labelsize=24, width=2.0, length=7)
    cbar.outline.set_linewidth(2.0)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HTG Fig. 9b Wcond heatmaps from a scan TSV.")
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--require-fig9b-8x10",
        action="store_true",
        help="Require the corrected 8x10 Fig. 9b center grid before plotting.",
    )
    args = parser.parse_args()

    tsv = args.tsv.resolve()
    output_dir = (args.output_dir or tsv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or tsv.stem

    rows = _read_rows(tsv)
    theta, waa, data = _pivot(rows)
    if args.require_fig9b_8x10:
        if not _same_grid(theta, EXPECTED_THETA_GRID_DEG) or not _same_grid(waa, EXPECTED_WAA_GRID_MEV):
            raise SystemExit(
                "scan TSV is not on the corrected 8x10 Fig. 9b center grid: "
                f"theta={theta}, wAA={waa}"
            )
        if list(data.shape) != [len(EXPECTED_WAA_GRID_MEV), len(EXPECTED_THETA_GRID_DEG)]:
            raise SystemExit(f"Wcond matrix shape is {data.shape}, expected (10, 8)")
    _save_paper_blocks(theta, waa, data, rows, output_dir / f"{prefix}_paper_blocks")
    _save_continuous(theta, waa, data, rows, output_dir / prefix)

    print(f"wrote {output_dir / f'{prefix}_paper_blocks.png'}")
    print(f"wrote {output_dir / f'{prefix}_paper_blocks.pdf'}")
    print(f"wrote {output_dir / f'{prefix}.png'}")
    print(f"wrote {output_dir / f'{prefix}.pdf'}")


if __name__ == "__main__":
    main()
