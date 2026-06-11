from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from ....core.plotting.bands import load_plot_backend
from ....core.lattice import KPath
from .hf_runners import HFPathResult, HFSCFPathPlotResult
from .model import BMSolution


HF_COLOR_MAP = {
    "K_up": "#1f77b4",
    "K_down": "#d62728",
    "Kprime_up": "#ff7f0e",
    "Kprime_down": "#2ca02c",
}
HF_FLAVOR_LABELS = {
    "K_up": "K↑",
    "K_down": "K↓",
    "Kprime_up": "K'↑",
    "Kprime_down": "K'↓",
}
HF_LEGEND_ORDER = ("K_up", "K_down", "Kprime_up", "Kprime_down")
DEFAULT_BAND_COLOR = "#222222"
DEFAULT_LINE_WIDTH = 1.1
DEFAULT_MARKER_SIZE = 2.2
DEFAULT_MARKER_EDGE_WIDTH = 0.25
DEFAULT_MARKER_EDGE_COLOR = "#ffffff"


def _load_plot_backend():
    return load_plot_backend(include_line2d=True)
def _display_node_label(label: str) -> str:
    return {"Gamma": "Γ", "M": "M", "K": "K", "Kprime": "K'"}.get(label, label)


def _flavor_from_band_label(band_label: str) -> str | None:
    parts = band_label.split("_")
    if len(parts) < 2:
        return None
    flavor = "_".join(parts[:2])
    return flavor if flavor in HF_COLOR_MAP else None


def _legend_handles(Line2D):
    return [
        Line2D(
            [0],
            [0],
            color=HF_COLOR_MAP[flavor],
            lw=1.8,
            marker="o",
            markersize=4.2,
            markerfacecolor=HF_COLOR_MAP[flavor],
            markeredgecolor=DEFAULT_MARKER_EDGE_COLOR,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
            label=HF_FLAVOR_LABELS[flavor],
        )
        for flavor in HF_LEGEND_ORDER
    ]


def write_path_band_plot(
    output_dir: Path | str,
    *,
    stem: str,
    kdist: np.ndarray,
    energies: np.ndarray,
    path: KPath,
    band_labels: tuple[str, ...] | None = None,
    mu: float | None = None,
    title: str | None = None,
) -> dict[str, Path]:
    plt, Line2D = _load_plot_backend()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    kdist = np.asarray(kdist, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2:
        raise ValueError(f"Expected a 2D energy array, got {energies.shape}")
    if energies.shape[0] != kdist.size:
        raise ValueError(f"Expected {kdist.size} path points, got {energies.shape[0]}")
    if mu is not None:
        energies = energies - float(mu)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    recognized_flavors: set[str] = set()
    for ib in range(energies.shape[1]):
        band_label = band_labels[ib] if band_labels is not None else ""
        flavor = _flavor_from_band_label(band_label)
        if flavor is not None:
            recognized_flavors.add(flavor)
        color = HF_COLOR_MAP.get(flavor, DEFAULT_BAND_COLOR)
        ax.plot(
            kdist,
            energies[:, ib],
            color=color,
            lw=DEFAULT_LINE_WIDTH,
            alpha=0.95,
            marker="o",
            markersize=DEFAULT_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=DEFAULT_MARKER_EDGE_COLOR,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
        )

    node_x = [float(node.k_dist) for node in path.nodes]
    node_labels = [_display_node_label(node.label) for node in path.nodes]
    for xpos in node_x:
        ax.axvline(x=xpos, color="#999999", ls=":", lw=0.8)
    if mu is not None:
        ax.axhline(y=0.0, color="#444444", ls="--", lw=0.9)

    ax.set_xticks(node_x)
    ax.set_xticklabels(node_labels)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    ax.set_ylabel("Energy - E_F (meV)" if mu is not None else "Energy (meV)")
    ax.set_xlabel("k-path")
    if title:
        title_pad = 26 if recognized_flavors else 12
        ax.set_title(title, fontsize=10, pad=title_pad)

    if recognized_flavors:
        ax.legend(
            handles=_legend_handles(Line2D),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=4,
            frameon=False,
            fontsize=8,
            handlelength=2.0,
            columnspacing=1.2,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}


def write_hf_band_plot(output_dir: Path | str, result: HFPathResult, *, stem: str = "band_plot") -> dict[str, Path]:
    title = (
        f"theta={result.params.dtheta_rad * 180.0 / np.pi:.2f}°, "
        f"nu={result.nu:g}, "
        f"init={result.init_mode}, "
        f"seed={result.seed}"
    )
    return write_path_band_plot(
        output_dir,
        stem=stem,
        kdist=np.asarray(result.path.kdist, dtype=float),
        energies=np.asarray(result.band_data.energies, dtype=float).T,
        path=result.path,
        band_labels=result.band_data.band_labels,
        mu=result.mu,
        title=title,
    )


def write_hf_scf_band_plot(
    output_dir: Path | str,
    result: HFSCFPathPlotResult,
    *,
    stem: str = "band_plot",
) -> dict[str, Path]:
    title = (
        f"theta={result.params.dtheta_rad * 180.0 / np.pi:.2f}°, "
        f"nu={result.nu:g}, "
        f"init={result.init_mode}, "
        f"seed={result.seed}, "
        "SCF grid points on path only"
    )
    return write_path_band_plot(
        output_dir,
        stem=stem,
        kdist=np.asarray(result.kdist, dtype=float),
        energies=np.asarray(result.band_data.energies, dtype=float).T,
        path=result.path,
        band_labels=result.band_data.band_labels,
        mu=result.mu,
        title=title,
    )


def write_bm_band_plot(
    output_dir: Path | str,
    *,
    theta_deg: float,
    path: KPath,
    path_solution: BMSolution,
    stem: str = "band_plot",
) -> dict[str, Path]:
    return write_path_band_plot(
        output_dir,
        stem=stem,
        kdist=np.asarray(path.kdist, dtype=float),
        energies=np.asarray(path_solution.flattened_energies(), dtype=float).T,
        path=path,
        title=f"theta={theta_deg:.2f}°, unstrained BM path bands",
    )
