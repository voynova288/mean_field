from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.plotting.bands import format_kpath_axis, load_plot_backend, plot_band_columns, save_figure_pair
from .bands import PathBandsResult


@dataclass(frozen=True)
class TDBGPathPlotTrace:
    label: str
    path_result: PathBandsResult
    color: str = "#1f1f1f"
    linestyle: str = "-"
    linewidth: float = 0.8
    alpha: float = 0.85


def _display_node_label(label: str) -> str:
    return {"Gamma": "Γ", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(label, label)


def write_tdbg_path_band_plot(
    output_dir: Path | str,
    traces: tuple[TDBGPathPlotTrace, ...],
    *,
    stem: str = "bands_path",
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    if not traces:
        raise ValueError("Expected at least one trace to plot.")

    plt = load_plot_backend()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for trace in traces:
        plot_band_columns(
            ax,
            trace.path_result.path.kdist,
            trace.path_result.energies,
            color=trace.color,
            linestyle=trace.linestyle,
            linewidth=trace.linewidth,
            alpha=trace.alpha,
        )

    reference_path = traces[0].path_result.path
    format_kpath_axis(ax, reference_path, label_formatter=_display_node_label, xlabel=None)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    paths = save_figure_pair(fig, output_dir, stem, key_prefix="band_plot")
    plt.close(fig)
    return paths
