from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.plotting.bands import load_plot_backend
from .bands import PathBandsResult

@dataclass(frozen=True)
class HTGPathPlotTrace:
    label: str
    path_result: PathBandsResult
    color: str = "#1f1f1f"
    linestyle: str = "-"
    linewidth: float = 0.75
    alpha: float = 0.9
    energy_scale: float = 1.0


def _load_plot_backend():
    return load_plot_backend()


def _display_node_label(label: str) -> str:
    mapping = {
        "Gamma": r"$\Gamma$",
        "kappa": r"$\kappa$",
        "kappa_prime": r"$\kappa'$",
        "M": "m",
    }
    return mapping.get(label, label)


def write_htg_path_band_plot(
    output_dir: Path | str,
    traces: tuple[HTGPathPlotTrace, ...],
    *,
    stem: str,
    title: str | None = None,
    ylabel: str = "Energy (eV)",
    ylim: tuple[float, float] | None = None,
    annotate: str | None = None,
) -> dict[str, Path]:
    if not traces:
        raise ValueError("Expected at least one trace to plot.")
    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    reference_path = traces[0].path_result.path
    node_x = [float(node.k_dist) for node in reference_path.nodes]
    node_labels = [_display_node_label(node.label) for node in reference_path.nodes]
    for xpos in node_x:
        ax.axvline(x=xpos, color="#9a9a9a", linestyle=":", linewidth=0.8, zorder=0)
    ax.axhline(y=0.0, color="#777777", linestyle="-", linewidth=0.45, alpha=0.55, zorder=0)

    for trace in traces:
        energies = np.asarray(trace.path_result.energies, dtype=float) * float(trace.energy_scale)
        for band_index in range(energies.shape[1]):
            ax.plot(
                trace.path_result.path.kdist,
                energies[:, band_index],
                color=trace.color,
                linestyle=trace.linestyle,
                linewidth=trace.linewidth,
                alpha=trace.alpha,
                zorder=2,
            )
    ax.set_xticks(node_x, node_labels)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("k-path")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if annotate:
        ax.text(
            0.02,
            0.96,
            annotate,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.82, "pad": 4},
        )
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}


def write_htg_hf_path_band_plot(
    output_dir: Path | str,
    hf_path_result,
    *,
    stem: str = "hf_bands_path",
    title: str | None = None,
    energy_scale: float = 1000.0,
    energy_reference_ev: float | None = None,
    ylabel: str = "Energy - mu (meV)",
    ylim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    path = hf_path_result.path
    energies = _hf_path_plot_energy_values(
        hf_path_result,
        energy_scale=energy_scale,
        energy_reference_ev=energy_reference_ev,
    )
    sigma_z = np.asarray(hf_path_result.sigma_z_expectation, dtype=float)
    if energies.shape != sigma_z.shape:
        raise ValueError(f"Expected energies and sigma_z to share shape, got {energies.shape} and {sigma_z.shape}")

    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    node_x = [float(node.k_dist) for node in path.nodes]
    node_labels = [_display_node_label(node.label) for node in path.nodes]
    for xpos in node_x:
        ax.axvline(x=xpos, color="#9a9a9a", linestyle=":", linewidth=0.8, zorder=0)
    ax.axhline(y=0.0, color="#777777", linestyle="-", linewidth=0.45, alpha=0.55, zorder=0)

    scatter = None
    for band_index in range(energies.shape[1]):
        ax.plot(path.kdist, energies[:, band_index], color="#4f4f4f", linewidth=0.36, alpha=0.42, zorder=1)
        scatter = ax.scatter(
            path.kdist,
            energies[:, band_index],
            c=sigma_z[:, band_index],
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            s=7.5,
            linewidths=0.0,
            alpha=0.94,
            zorder=2,
        )

    ax.set_xticks(node_x, node_labels)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("k-path")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=10)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.018)
        cbar.set_label(r"$\langle \tilde{\sigma}_z \rangle$", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}


def _hf_path_plot_energy_values(
    hf_path_result,
    *,
    energy_scale: float = 1000.0,
    energy_reference_ev: float | None = None,
) -> np.ndarray:
    energies = np.asarray(hf_path_result.energies, dtype=float)
    reference_ev = energy_reference_ev
    if reference_ev is None:
        reference_ev = getattr(hf_path_result, "mu", 0.0)
    return (energies - float(reference_ev)) * float(energy_scale)


__all__ = [
    "HTGPathPlotTrace",
    "write_htg_hf_path_band_plot",
    "write_htg_path_band_plot",
]
