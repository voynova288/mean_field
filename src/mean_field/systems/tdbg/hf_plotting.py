from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...plotting import (
    format_kpath_axis,
    load_plot_backend,
    save_figure_pair,
    write_kpath_band_tsv,
    write_kpath_nodes_tsv,
)
from ...core.lattice import KPath
from .projected_hf import TDBGProjectedHFData

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
DEFAULT_LINE_WIDTH = 1.0
DEFAULT_MARKER_SIZE = 1.8
DEFAULT_MARKER_EDGE_WIDTH = 0.2
DEFAULT_MARKER_EDGE_COLOR = "#ffffff"

@dataclass(frozen=True)
class TDBGHFBandData:
    """Flavor-resolved HF eigenvalues for TDBG projected-HF plots."""

    band_labels: tuple[str, ...]
    energies_ev: np.ndarray  # (n_band_total, n_k)
    mean_weights: np.ndarray  # (n_band_total, n_flavor)

@dataclass(frozen=True)
class TDBGHFPlotTrace:
    """One labeled HF band trace, optionally referenced to a source-grid mu."""

    label: str
    band_data: TDBGHFBandData
    mu_ev: float | None = None



def _display_node_label(label: str) -> str:
    return {"Gamma": "Γ", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(label, label)


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
            markersize=4.0,
            markerfacecolor=HF_COLOR_MAP[flavor],
            markeredgecolor=DEFAULT_MARKER_EDGE_COLOR,
            markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
            label=HF_FLAVOR_LABELS[flavor],
        )
        for flavor in HF_LEGEND_ORDER
    ]


def _flavor_sector_indices(data: TDBGProjectedHFData) -> tuple[tuple[str, tuple[int, ...]], ...]:
    sectors: list[tuple[str, tuple[int, ...]]] = []
    for valley_label in ("K", "Kprime"):
        for spin in ("up", "down"):
            indices = tuple(
                int(label.index)
                for label in data.labels
                if label.valley_label == valley_label and label.spin == spin
            )
            if indices:
                sectors.append((f"{valley_label}_{spin}", indices))
    return tuple(sectors)


def tdbg_hf_band_data(data: TDBGProjectedHFData, hamiltonian: np.ndarray) -> TDBGHFBandData:
    """Diagonalize TDBG HF Hamiltonians and label bands by dominant flavor.

    The labels follow the TBG B=0 benchmark plotting convention (`K_up`,
    `K_down`, `Kprime_up`, `Kprime_down`) but derive sectors from the TDBG
    system labels rather than assuming the TBG flattened index order.
    """

    h = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {h.shape}")
    if nt != data.nt:
        raise ValueError(f"Hamiltonian dimension {nt} incompatible with TDBG projected data nt={data.nt}")

    sectors = _flavor_sector_indices(data)
    energies = np.zeros((nt, nk), dtype=float)
    mean_weights = np.zeros((nt, len(sectors)), dtype=float)
    for ik in range(nk):
        h_k = 0.5 * (h[:, :, ik] + h[:, :, ik].conjugate().T)
        evals, evecs = np.linalg.eigh(h_k)
        energies[:, ik] = evals
        for ib in range(nt):
            for ifl, (_flavor, indices) in enumerate(sectors):
                sector_idx = np.asarray(indices, dtype=int)
                mean_weights[ib, ifl] += float(np.sum(np.abs(evecs[sector_idx, ib]) ** 2))
    if nk > 0:
        mean_weights /= float(nk)
    band_labels: list[str] = []
    for ib in range(nt):
        if sectors:
            dominant = int(np.argmax(mean_weights[ib, :]))
            band_labels.append(f"{sectors[dominant][0]}_b{ib + 1}")
        else:
            band_labels.append(f"b{ib + 1}")
    return TDBGHFBandData(band_labels=tuple(band_labels), energies_ev=energies, mean_weights=mean_weights)


def _energy_mev(trace: TDBGHFPlotTrace) -> tuple[np.ndarray, bool]:
    energies = np.asarray(trace.band_data.energies_ev, dtype=float)
    if trace.mu_ev is None or not np.isfinite(float(trace.mu_ev)):
        return 1.0e3 * energies, False
    return 1.0e3 * (energies - float(trace.mu_ev)), True


def write_tdbg_hf_path_band_plot(
    output_dir: Path | str,
    *,
    path: KPath,
    traces: Mapping[str, TDBGHFPlotTrace],
    stem: str = "hf_path_bands_by_state",
    title: str | None = None,
) -> dict[str, Path]:
    """Write benchmark-style flavor-colored TDBG HF path-band plots.

    Energies are referenced to each source-grid chemical potential when
    `trace.mu_ev` is provided, matching the existing TBG HF benchmark display
    convention `Energy - E_F (meV)`.
    """

    if not traces:
        return {}
    plt, Line2D = load_plot_backend(include_line2d=True)

    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, max(3.2, 2.9 * n)), squeeze=False)
    any_referenced = False
    for ax, trace in zip(axes[:, 0], traces.values(), strict=False):
        energies_mev, referenced = _energy_mev(trace)
        any_referenced = any_referenced or referenced
        for band in range(energies_mev.shape[0]):
            band_label = trace.band_data.band_labels[band]
            flavor = _flavor_from_band_label(band_label)
            color = HF_COLOR_MAP.get(flavor, DEFAULT_BAND_COLOR)
            ax.plot(
                path.kdist,
                energies_mev[band],
                color=color,
                lw=DEFAULT_LINE_WIDTH,
                alpha=0.95,
                marker="o",
                markersize=DEFAULT_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=DEFAULT_MARKER_EDGE_COLOR,
                markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH,
            )
        format_kpath_axis(
            ax,
            path,
            label_formatter=_display_node_label,
            vertical_line_kwargs={"color": "#999999", "linestyle": ":", "linewidth": 0.8},
            xlabel=None,
        )
        if referenced:
            ax.axhline(y=0.0, color="#444444", ls="--", lw=0.9)
        ax.set_title(trace.label, fontsize=10)
        ax.set_ylabel("Energy - E_F (meV)" if referenced else "Energy (meV)")
        ax.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("k-path")
    if title:
        fig.suptitle(title, fontsize=11)
    axes[0, 0].legend(
        handles=_legend_handles(Line2D),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.2,
    )
    rect_top = 0.91 if title or any_referenced else 0.94
    fig.tight_layout(rect=(0.0, 0.0, 1.0, rect_top))
    paths = save_figure_pair(fig, output_dir, stem, key_prefix="band_plot")
    plt.close(fig)
    return paths


def write_tdbg_hf_grid_band_plot(
    output_dir: Path | str,
    *,
    traces: Mapping[str, TDBGHFPlotTrace],
    stem: str = "hf_grid_bands_by_seed",
    title: str | None = None,
) -> dict[str, Path]:
    """Write flavor-colored SCF-grid HF bands versus grid index."""

    if not traces:
        return {}
    plt, Line2D = load_plot_backend(include_line2d=True)

    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, max(3.0, 2.7 * n)), squeeze=False)
    for ax, trace in zip(axes[:, 0], traces.values(), strict=False):
        energies_mev, referenced = _energy_mev(trace)
        x = np.arange(energies_mev.shape[1], dtype=int)
        for band in range(energies_mev.shape[0]):
            band_label = trace.band_data.band_labels[band]
            flavor = _flavor_from_band_label(band_label)
            color = HF_COLOR_MAP.get(flavor, DEFAULT_BAND_COLOR)
            ax.plot(x, energies_mev[band], color=color, lw=0.85, alpha=0.95)
        if referenced:
            ax.axhline(y=0.0, color="#444444", ls="--", lw=0.9)
        ax.set_title(trace.label, fontsize=10)
        ax.set_xlabel("SCF k-grid point index")
        ax.set_ylabel("Energy - E_F (meV)" if referenced else "Energy (meV)")
        ax.grid(alpha=0.22)
    if title:
        fig.suptitle(title, fontsize=11)
    axes[0, 0].legend(
        handles=_legend_handles(Line2D),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if title else 0.94))
    paths = save_figure_pair(fig, output_dir, stem, key_prefix="band_plot")
    plt.close(fig)
    return paths

def write_tdbg_hf_path_tsv(path: Path | str, *, kpath: KPath, trace: TDBGHFPlotTrace) -> None:
    """Write raw HF path eigenvalues in benchmark-style TSV format."""

    write_kpath_band_tsv(
        path,
        kdist=kpath.kdist,
        energies=trace.band_data.energies_ev,
        band_labels=trace.band_data.band_labels,
        bands_axis="rows",
    )


def write_tdbg_hf_path_nodes_tsv(path: Path | str, kpath: KPath) -> None:
    """Write high-symmetry path node metadata in benchmark-style TSV format."""

    write_kpath_nodes_tsv(path, kpath)


def write_tdbg_hf_scf_path_tsv(
    path: Path | str,
    *,
    data: TDBGProjectedHFData,
    kpath: KPath,
    hamiltonian: np.ndarray,
    path_tolerance: float = 1.0e-12,
) -> int:
    """Write exact SCF-grid points that lie on a requested path.

    This mirrors the TBG B=0 diagnostic separation: dense/off-grid path bands
    are not the same artifact as exact SCF-grid points projected onto that
    path. The function returns the number of exact path samples written.
    """

    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.shape != (data.nt, data.nt, data.nk):
        raise ValueError(f"Expected Hamiltonian shape {(data.nt, data.nt, data.nk)}, got {h.shape}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_kvec = np.asarray(data.kvec, dtype=np.complex128)
    path_kvec = np.asarray(kpath.kvec, dtype=np.complex128)
    if path_kvec.size == 0:
        raise ValueError("At least two path nodes are required to build an SCF path diagnostic.")
    distance_matrix = np.abs(path_kvec[:, None] - grid_kvec[None, :])
    nearest_grid_indices = np.argmin(distance_matrix, axis=1).astype(int)
    nearest_grid_distances = distance_matrix[np.arange(path_kvec.size), nearest_grid_indices]
    path_indices = np.flatnonzero(nearest_grid_distances <= float(path_tolerance)).astype(int)
    selected_grid_indices = nearest_grid_indices[path_indices].astype(int)
    selected_hamiltonian = h[:, :, selected_grid_indices]
    band_data = tdbg_hf_band_data(data, selected_hamiltonian)
    energies = np.asarray(band_data.energies_ev, dtype=float)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "path_index",
                    "path_k_dist",
                    "k_dist",
                    "distance_to_path",
                    "path_kx",
                    "path_ky",
                    "grid_index",
                    "grid_kx",
                    "grid_ky",
                    *band_data.band_labels,
                ]
            )
            + "\n"
        )
        for local_index, path_index in enumerate(path_indices):
            grid_index = int(selected_grid_indices[local_index])
            row = [
                str(int(path_index) + 1),
                f"{float(kpath.kdist[path_index]):.16f}",
                f"{float(kpath.kdist[path_index]):.16f}",
                f"{float(nearest_grid_distances[path_index]):.16f}",
                f"{float(path_kvec[path_index].real):.16f}",
                f"{float(path_kvec[path_index].imag):.16f}",
                str(grid_index + 1),
                f"{float(grid_kvec[grid_index].real):.16f}",
                f"{float(grid_kvec[grid_index].imag):.16f}",
            ]
            row.extend(f"{float(energies[ib, local_index]):.16f}" for ib in range(energies.shape[0]))
            handle.write("\t".join(row) + "\n")
    return int(path_indices.size)
