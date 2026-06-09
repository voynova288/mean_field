from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...plotting import format_kpath_axis, load_plot_backend, plot_band_columns, save_figure_pair
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

    plt = load_plot_backend()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for trace in traces:
        plot_band_columns(
            ax,
            trace.path_result.path.kdist,
            trace.path_result.energies,
            energy_shift=float(trace.energy_shift_mev),
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
    paths = save_figure_pair(fig, output_dir, stem, key_prefix="band_plot")
    plt.close(fig)
    return paths


__all__ = ["RLGhBNPathPlotTrace", "path_bandwidth_mev", "write_rlg_hbn_path_band_plot"]
