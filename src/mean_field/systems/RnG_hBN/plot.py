from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

from ...plotting import load_plot_backend
from .bands import PathBandsResult
from .hamiltonian import flat_band_indices
from .lattice import RLGhBNLattice
from .params import RLGhBNParams


@dataclass(frozen=True)
class RLGhBNPathPlotTrace:
    label: str
    path_result: PathBandsResult
    color: str = "#1f1f1f"
    linestyle: str = "-"
    linewidth: float = 0.8
    alpha: float = 0.85
    energy_shift_mev: float = 0.0


def _load_plot_backend():
    return load_plot_backend()
def _display_node_label(label: str) -> str:
    return {"Gamma": "Gamma", "M": "M", "K": "K", "Kprime": "K'"}.get(label, label)


def path_bandwidth_mev(path_result: PathBandsResult, lattice: RLGhBNLattice, params: RLGhBNParams, *, band_index: int | None = None) -> float:
    valence, conduction = flat_band_indices(lattice, params)
    resolved_band = conduction if band_index is None else int(band_index)
    energies = np.asarray(path_result.energies, dtype=float)
    if resolved_band >= energies.shape[1]:
        raise ValueError(f"band_index={resolved_band} exceeds available bands {energies.shape[1]}")
    return float(np.max(energies[:, resolved_band]) - np.min(energies[:, resolved_band]))


def write_rlg_hbn_path_band_plot(
    output_dir: Path | str,
    traces: tuple[RLGhBNPathPlotTrace, ...],
    *,
    lattice: RLGhBNLattice | None = None,
    params: RLGhBNParams | None = None,
    stem: str = "bands_path",
    title: str | None = None,
    ylim: tuple[float, float] | None = (-100.0, 100.0),
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
                energies[:, band_index] - float(trace.energy_shift_mev),
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
    ylabel = "Energy (meV)"
    if any(abs(float(trace.energy_shift_mev)) > 0.0 for trace in traces):
        ylabel = "Energy - E_neutral (meV)"
    ax.set_ylabel(ylabel)
    resolved_title = title
    if resolved_title is None and lattice is not None and params is not None:
        bandwidth = path_bandwidth_mev(traces[0].path_result, lattice, params)
        resolved_title = (
            f"R{params.layer_count}G/hBN, V={params.displacement_field_mev:.3g} meV, "
            f"xi={params.xi}, theta={lattice.theta_deg:.3g} deg, bandwidth={bandwidth:.3g} meV"
        )
    if resolved_title is not None:
        ax.set_title(resolved_title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}


__all__ = ["RLGhBNPathPlotTrace", "path_bandwidth_mev", "write_rlg_hbn_path_band_plot"]
