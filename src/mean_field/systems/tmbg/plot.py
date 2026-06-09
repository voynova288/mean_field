from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...plotting import format_kpath_axis, load_plot_backend, plot_band_columns, save_figure_pair
from .bands import PathBandsResult
from .lattice import TMBGLattice
from .topology import TopologyResult


@dataclass(frozen=True)
class TMBGBandPlotPanel:
    label: str
    path_result: PathBandsResult
    band_indices: tuple[int, ...] | None = None
    flat_band_indices: tuple[int, int] | None = None
    annotation: str | None = None
    primary_label: str | None = None
    overlay_path_results: tuple[PathBandsResult, ...] = ()
    overlay_label: str | None = None


def _load_plot_backend():
    return load_plot_backend()
def _display_node_label(label: str) -> str:
    return {
        "Gamma": r"$\tilde{\Gamma}$",
        "GammaPrime": r"$\tilde{\Gamma}'$",
        "M": r"$\tilde{M}$",
        "K": r"$\tilde{K}$",
        "Kprime": r"$\tilde{K}'$",
        "KPrime": r"$\tilde{K}'$",
    }.get(label, label)


def infer_flat_band_indices(
    energies: np.ndarray,
    *,
    energy_window: float = 0.030,
    search_half_width: int = 10,
) -> tuple[int, int]:
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError(f"Expected energies with shape (nk, nb>=2), got {energies.shape}")

    widths = np.ptp(energies, axis=0)
    centers = np.mean(energies, axis=0)
    n_bands = int(energies.shape[1])
    mid = n_bands // 2
    lower_bound = max(0, mid - int(search_half_width))
    upper_bound = min(n_bands - 1, mid + int(search_half_width))

    candidates: list[tuple[tuple[float, float, float, float, int], tuple[int, int]]] = []
    for ib in range(lower_bound, upper_bound):
        lower = energies[:, ib]
        upper = energies[:, ib + 1]
        if np.max(np.abs(lower)) >= energy_window or np.max(np.abs(upper)) >= energy_window:
            continue
        width_sum = float(widths[ib] + widths[ib + 1])
        max_abs_energy = float(max(np.max(np.abs(lower)), np.max(np.abs(upper))))
        center_abs = abs(float(centers[ib])) + abs(float(centers[ib + 1]))
        touching_gap = float(np.min(upper - lower))
        score = (width_sum, max_abs_energy, center_abs, -touching_gap, ib)
        candidates.append((score, (ib, ib + 1)))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    best_pair = (0, 1)
    best_score: tuple[float, float, float, float, int] | None = None
    for ib in range(energies.shape[1] - 1):
        lower = energies[:, ib]
        upper = energies[:, ib + 1]
        center_abs = abs(float(centers[ib])) + abs(float(centers[ib + 1]))
        width_sum = float(widths[ib] + widths[ib + 1])
        touching_gap = float(np.min(upper - lower))
        outer_gap_penalty = 0.0
        if ib > 0:
            outer_gap_penalty -= float(np.min(lower - energies[:, ib - 1]))
        if ib + 2 < energies.shape[1]:
            outer_gap_penalty -= float(np.min(energies[:, ib + 2] - upper))
        score = (center_abs, width_sum, -touching_gap, outer_gap_penalty, ib)
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (ib, ib + 1)
    return best_pair


def _resolve_flat_band_local_indices(panel: TMBGBandPlotPanel, selected_indices: np.ndarray) -> tuple[int | None, int | None]:
    if panel.flat_band_indices is None:
        if selected_indices.size < 2:
            return None, None
        local_valence, local_conduction = infer_flat_band_indices(panel.path_result.energies[:, selected_indices])
        return int(local_valence), int(local_conduction)

    selected_lookup = {int(index): ilocal for ilocal, index in enumerate(selected_indices.tolist())}
    valence_abs, conduction_abs = (int(panel.flat_band_indices[0]), int(panel.flat_band_indices[1]))
    return selected_lookup.get(valence_abs), selected_lookup.get(conduction_abs)


def write_tmbg_band_plot(
    output_dir: Path | str,
    result: PathBandsResult,
    *,
    stem: str = "bands_path",
    title: str | None = None,
) -> dict[str, Path]:
    plt = _load_plot_backend()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plot_band_columns(
        ax,
        result.path.kdist,
        result.energies,
        color="#1f1f1f",
        lw=1.0,
        marker="o",
        markersize=1.9,
        markerfacecolor="#1f1f1f",
        markeredgecolor="#ffffff",
        markeredgewidth=0.2,
    )

    format_kpath_axis(ax, result.path, label_formatter=_display_node_label)
    ax.set_ylabel("Energy (eV)")
    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    paths = save_figure_pair(fig, output_dir, stem, key_prefix="band_plot")
    plt.close(fig)
    return paths


def write_tmbg_paper_band_figure(
    output_dir: Path | str,
    panels: tuple[TMBGBandPlotPanel, ...],
    *,
    stem: str = "fig2_like_bands",
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> dict[str, Path]:
    if not panels:
        raise ValueError("Expected at least one panel to plot.")

    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    all_selected: list[np.ndarray] = []
    resolved_panel_data: list[tuple[TMBGBandPlotPanel, np.ndarray]] = []
    for panel in panels:
        energies = np.asarray(panel.path_result.energies, dtype=float)
        if panel.band_indices is None:
            selected = energies
        else:
            selected = energies[:, panel.band_indices]
        resolved_panel_data.append((panel, selected))
        all_selected.append(selected)

    if ylim is None:
        if len(panels) == 3:
            ylim = (-0.100, 0.100)
        else:
            stacked = np.concatenate([data.ravel() for data in all_selected])
            ymin = float(np.min(stacked))
            ymax = float(np.max(stacked))
            pad = 0.05 * max(ymax - ymin, 1.0e-3)
            ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (panel, selected) in zip(axes, resolved_panel_data, strict=True):
        selected_indices = (
            np.arange(panel.path_result.energies.shape[1], dtype=int)
            if panel.band_indices is None
            else np.asarray(panel.band_indices, dtype=int)
        )
        n_selected = selected.shape[1]
        valence_idx, conduction_idx = _resolve_flat_band_local_indices(panel, selected_indices)

        overlay_label_used = False
        for overlay_result in panel.overlay_path_results:
            overlay_energies = np.asarray(overlay_result.energies, dtype=float)
            overlay_selected = overlay_energies if panel.band_indices is None else overlay_energies[:, panel.band_indices]
            for ib in range(overlay_selected.shape[1]):
                ax.plot(
                    overlay_result.path.kdist,
                    overlay_selected[:, ib],
                    color="#58a6d6",
                    lw=0.85,
                    ls=(0, (4, 3)),
                    alpha=0.95,
                    zorder=1.8,
                    label=panel.overlay_label if panel.overlay_label and not overlay_label_used and ib == 0 else None,
                )
            overlay_label_used = True

        for ib in range(n_selected):
            color = "#6b6b6b"
            lw = 1.0
            zorder = 2
            if valence_idx is not None and ib == valence_idx:
                color = "#c73e1d"
                lw = 1.4
                zorder = 3
            if conduction_idx is not None and ib == conduction_idx:
                color = "#1d4e89"
                lw = 1.4
                zorder = 3
            ax.plot(
                panel.path_result.path.kdist,
                selected[:, ib],
                color=color,
                lw=lw,
                marker="o",
                markersize=1.6,
                markerfacecolor=color,
                markeredgecolor="#ffffff",
                markeredgewidth=0.18,
                zorder=zorder,
                label=panel.primary_label if panel.primary_label and ib == 0 else None,
            )

        node_x = [float(node.k_dist) for node in panel.path_result.path.nodes]
        node_labels = [_display_node_label(node.label) for node in panel.path_result.path.nodes]
        for xpos in node_x:
            ax.axvline(x=xpos, color="#b8b8b8", ls=":", lw=0.8, zorder=1)
        ax.set_xticks(node_x)
        ax.set_xticklabels(node_labels)
        ax.set_xlim(float(node_x[0]), float(node_x[-1]))
        ax.set_ylim(*ylim)
        ax.set_xlabel("k-path")
        ax.set_title(panel.label, fontsize=10)
        if panel.annotation is not None:
            ax.text(
                0.03,
                0.97,
                panel.annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.4,
                color="#1f1f1f",
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": (1.0, 1.0, 1.0, 0.82),
                    "edgecolor": "#d0d0d0",
                    "linewidth": 0.6,
                },
                zorder=4,
            )
        if panel.overlay_label is not None:
            ax.legend(loc="lower right", fontsize=7.2, frameon=False, handlelength=2.6)

    axes[0].set_ylabel("Energy (eV)")
    if title is not None:
        fig.suptitle(title, fontsize=11, y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"paper_band_plot_png": png_path, "paper_band_plot_pdf": pdf_path}


def _hexagon_vertices(lattice: TMBGLattice) -> np.ndarray:
    vertices = np.asarray(
        [
            (lattice.g_m1 + lattice.g_m2) / 3.0,
            (2.0 * lattice.g_m1 - lattice.g_m2) / 3.0,
            (lattice.g_m1 - 2.0 * lattice.g_m2) / 3.0,
            -(lattice.g_m1 + lattice.g_m2) / 3.0,
            -(2.0 * lattice.g_m1 - lattice.g_m2) / 3.0,
            -(lattice.g_m1 - 2.0 * lattice.g_m2) / 3.0,
        ],
        dtype=np.complex128,
    )
    angles = np.angle(vertices)
    ordered = vertices[np.argsort(angles)]
    return np.append(ordered, ordered[:1])


def write_tmbg_lattice_plot(
    output_dir: Path | str,
    lattice: TMBGLattice,
    *,
    stem: str = "lattice_plot",
    title: str | None = None,
) -> dict[str, Path]:
    plt = _load_plot_backend()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    g_vectors = np.asarray(lattice.g_vectors, dtype=np.complex128)
    ax.scatter(g_vectors.real, g_vectors.imag, s=18, color="#7a7a7a", alpha=0.85, label="G vectors", zorder=2)

    boundary = _hexagon_vertices(lattice)
    ax.plot(boundary.real, boundary.imag, color="#1f1f1f", lw=1.2, label="mBZ", zorder=3)

    q_style = {
        "Q0": (lattice.q0, "#c73e1d"),
        "Q+": (lattice.q_plus, "#1d4e89"),
        "Q-": (lattice.q_minus, "#2e8b57"),
    }
    for label, (qvec, color) in q_style.items():
        ax.annotate(
            "",
            xy=(qvec.real, qvec.imag),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": color},
            zorder=4,
        )
        ax.text(qvec.real, qvec.imag, f" {label}", color=color, fontsize=9, ha="left", va="bottom")

    ax.scatter([0.0], [0.0], s=28, color="#1f1f1f", zorder=5)
    ax.text(0.0, 0.0, " Γ", fontsize=9, ha="left", va="bottom", color="#1f1f1f")

    for label, kvec in (("K", lattice.k_m), ("K'", lattice.kprime_m), ("M", lattice.m_m)):
        ax.scatter([kvec.real], [kvec.imag], s=24, color="#111111", zorder=5)
        ax.text(kvec.real, kvec.imag, f" {label}", fontsize=9, ha="left", va="bottom", color="#111111")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$ (nm$^{-1}$)")
    ax.set_ylabel(r"$k_y$ (nm$^{-1}$)")
    ax.grid(True, color="#e1e1e1", lw=0.6, alpha=0.8)
    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"lattice_plot_png": png_path, "lattice_plot_pdf": pdf_path}


def write_tmbg_berry_curvature_plot(
    output_dir: Path | str,
    result: TopologyResult,
    *,
    stem: str = "berry_curvature",
    title: str | None = None,
) -> dict[str, Path]:
    plt = _load_plot_backend()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    field = np.asarray(result.berry_curvature, dtype=float)
    vmax = float(np.max(np.abs(field)))
    if vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(
        field.T,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel("mBZ fractional $k_1$")
    ax.set_ylabel("mBZ fractional $k_2$")
    resolved_title = title
    if resolved_title is None:
        bands = ",".join(str(index) for index in result.band_indices)
        resolved_title = f"valley={result.valley}, bands={bands}, C={result.chern_number:.4f}"
    ax.set_title(resolved_title, fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Berry flux per plaquette")
    fig.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"berry_curvature_png": png_path, "berry_curvature_pdf": pdf_path}
