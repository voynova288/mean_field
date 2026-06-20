from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.plotting.bands import format_kpath_axis, load_plot_backend, plot_band_columns, save_figure_pair
from .bands import PathBandsResult
from .lattice import TMBGLattice
from .topology import TopologyResult


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
