from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

from ...plotting import load_plot_backend
from .bands import PathBandsResult


@dataclass(frozen=True)
class TDBGPathPlotTrace:
    label: str
    path_result: PathBandsResult
    color: str = "#1f1f1f"
    linestyle: str = "-"
    linewidth: float = 0.8
    alpha: float = 0.85


def _load_plot_backend():
    return load_plot_backend()
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

    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for trace in traces:
        energies = np.asarray(trace.path_result.energies, dtype=float)
        for band_index in range(energies.shape[1]):
            ax.plot(
                trace.path_result.path.kdist,
                energies[:, band_index],
                color=trace.color,
                linestyle=trace.linestyle,
                linewidth=trace.linewidth,
                alpha=trace.alpha,
            )

    reference_path = traces[0].path_result.path
    node_x = [float(node.k_dist) for node in reference_path.nodes]
    node_labels = [_display_node_label(node.label) for node in reference_path.nodes]
    for xpos in node_x:
        ax.axvline(x=xpos, color="#999999", linestyle=":", linewidth=0.8)

    ax.set_xticks(node_x, node_labels)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}
