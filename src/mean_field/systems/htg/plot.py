from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np

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
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    return plt


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
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label(r"$\langle \tilde{\sigma}_z \rangle$", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}


def _spin_sector_path_bands(
    hamiltonian: np.ndarray,
    sigma_z_operator: np.ndarray,
    *,
    spin_index: int,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    sigma_z_operator = np.asarray(sigma_z_operator, dtype=np.complex128)
    nt, _, nk = hamiltonian.shape
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    if sigma_z_operator.shape != hamiltonian.shape:
        raise ValueError(f"Expected sigma_z_operator shape {hamiltonian.shape}, got {sigma_z_operator.shape}")
    if not (0 <= int(spin_index) < int(n_spin)):
        raise ValueError(f"spin_index={spin_index} is out of range for n_spin={n_spin}")

    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    block_indices = np.asarray(idx[int(spin_index), :, :].reshape(-1, order="C"), dtype=int)
    energies = np.zeros((nk, block_indices.size), dtype=float)
    sigma_z = np.zeros_like(energies)
    for ik in range(nk):
        h_block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
        sigma_block = sigma_z_operator[:, :, ik][np.ix_(block_indices, block_indices)]
        eigvals, eigvecs = np.linalg.eigh(h_block)
        energies[ik, :] = eigvals
        sigma_z[ik, :] = np.real(np.diag(eigvecs.conjugate().T @ sigma_block @ eigvecs))
    return energies, sigma_z


def write_htg_fig7_spin_resolved_plot(
    output_dir: Path | str,
    hf_path_result,
    *,
    stem: str = "fig7_spin_resolved_bands",
    title: str | None = None,
    energy_scale: float = 1000.0,
    energy_reference_ev: float | None = None,
    ylim: tuple[float, float] = (-60.0, 40.0),
) -> dict[str, Path]:
    """Write the spin-resolved Fig. 7-style HF path-band plot.

    Each panel highlights one spin sector with Chern-sublattice coloring and
    shows the opposite spin sector as thin dotted black lines, matching the
    paper's Fig. 7 visual convention more closely than the all-band overview.
    """

    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    path = hf_path_result.path
    hamiltonian = np.asarray(hf_path_result.hamiltonian, dtype=np.complex128)
    sigma_z_operator = np.asarray(hf_path_result.sigma_z_operator, dtype=np.complex128)
    reference_ev = energy_reference_ev
    if reference_ev is None:
        reference_ev = getattr(hf_path_result, "mu", 0.0)

    spin_energies: list[np.ndarray] = []
    spin_sigma: list[np.ndarray] = []
    for spin_index in range(2):
        energies_ev, sigma_values = _spin_sector_path_bands(
            hamiltonian,
            sigma_z_operator,
            spin_index=spin_index,
        )
        spin_energies.append((energies_ev - float(reference_ev)) * float(energy_scale))
        spin_sigma.append(sigma_values)

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.75), sharex=True, sharey=True, constrained_layout=True)
    node_x = [float(node.k_dist) for node in path.nodes]
    node_labels = [_display_node_label(node.label) for node in path.nodes]
    spin_titles = (r"spin $\uparrow$", r"spin $\downarrow$")
    scatter = None
    for spin_index, ax in enumerate(axes):
        other_spin = 1 - spin_index
        for xpos in node_x:
            ax.axvline(x=xpos, color="#9a9a9a", linestyle="-", linewidth=0.45, alpha=0.75, zorder=0)
        ax.axhline(y=0.0, color="#777777", linestyle="-", linewidth=0.35, alpha=0.45, zorder=0)
        for band_index in range(spin_energies[other_spin].shape[1]):
            ax.plot(
                path.kdist,
                spin_energies[other_spin][:, band_index],
                color="#1f1f1f",
                linestyle=":",
                linewidth=0.42,
                alpha=0.58,
                zorder=1,
            )
        for band_index in range(spin_energies[spin_index].shape[1]):
            ax.plot(
                path.kdist,
                spin_energies[spin_index][:, band_index],
                color="#303030",
                linewidth=0.32,
                alpha=0.55,
                zorder=2,
            )
            scatter = ax.scatter(
                path.kdist,
                spin_energies[spin_index][:, band_index],
                c=spin_sigma[spin_index][:, band_index],
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                s=7.0,
                linewidths=0.0,
                alpha=0.96,
                zorder=3,
            )
        ax.text(0.5, 1.02, spin_titles[spin_index], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2)
        ax.set_ylabel(r"$E$ (meV)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylim(*ylim)
        ax.set_xlim(float(node_x[0]), float(node_x[-1]))

    axes[-1].set_xticks(node_x, node_labels)
    if title is not None:
        axes[0].set_title(title, fontsize=8.8, pad=16)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, fraction=0.045, pad=0.025)
        cbar.set_label(r"$\langle \tilde{\sigma}_z \rangle$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"fig7_plot_png": png_path, "fig7_plot_pdf": pdf_path}


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


def _select_hartree_reference(values_ev: np.ndarray, mode: Literal["none", "global_mean", "global_min"]) -> float:
    if mode == "none":
        return 0.0
    if mode == "global_mean":
        return float(np.mean(values_ev))
    if mode == "global_min":
        return float(np.min(values_ev))
    raise ValueError(f"Unsupported Hartree reference mode: {mode}")


def write_htg_fig8a_potential_plot(
    output_dir: Path | str,
    interaction_path_result,
    *,
    stem: str = "fig8a_hartree_fock_potentials",
    title: str | None = r"$\nu=+4$",
    spin_index: int = 0,
    valley_index: int = 0,
    hartree_reference: Literal["none", "global_mean", "global_min"] = "global_min",
) -> dict[str, Path]:
    """Write a Fig. 8a-style path plot of Chern-sublattice diagonal potentials."""

    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    path = interaction_path_result.path
    hartree_diag = np.asarray(interaction_path_result.hartree_diagonal_ev, dtype=float)
    fock_diag = np.asarray(interaction_path_result.fock_diagonal_ev, dtype=float)
    if hartree_diag.shape != fock_diag.shape or hartree_diag.ndim != 4:
        raise ValueError(f"Expected diagonal arrays with matching shape (n_spin,n_eta,n_band,nk), got {hartree_diag.shape} and {fock_diag.shape}")
    if not (0 <= spin_index < hartree_diag.shape[0]):
        raise ValueError(f"spin_index={spin_index} is out of range for shape {hartree_diag.shape}")
    if not (0 <= valley_index < hartree_diag.shape[1]):
        raise ValueError(f"valley_index={valley_index} is out of range for shape {hartree_diag.shape}")
    if hartree_diag.shape[2] < 2:
        raise ValueError("Fig. 8a-style potential plot expects at least two Chern-sublattice bands.")

    hartree_selected = hartree_diag[spin_index, valley_index, :2, :]
    fock_selected = fock_diag[spin_index, valley_index, :2, :]
    hartree_reference_ev = _select_hartree_reference(hartree_selected, hartree_reference)
    hartree_mev = (hartree_selected - hartree_reference_ev) * 250.0
    fock_mev = fock_selected * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 3.2), sharex=True)
    node_x = [float(node.k_dist) for node in path.nodes]
    node_labels = [_display_node_label(node.label) for node in path.nodes]
    colors = ("#d62728", "#1f5eff")
    labels = (r"$\tau=K,\tilde{\sigma}=A$", r"$\tau=K,\tilde{\sigma}=B$")

    for ax in axes:
        for xpos in node_x:
            ax.axvline(x=xpos, color="#a8a8a8", linestyle="-", linewidth=0.45, alpha=0.65, zorder=0)
        ax.set_xlim(float(node_x[0]), float(node_x[-1]))

    for band_index, (color, label) in enumerate(zip(colors, labels, strict=True)):
        axes[0].plot(path.kdist, hartree_mev[band_index], color=color, linewidth=0.8, label=label)
        axes[1].plot(path.kdist, fock_mev[band_index], color=color, linewidth=0.8, label=label)

    axes[0].set_ylabel(r"$\frac{1}{4} E_{\mathrm{Hartree}}$ (meV)")
    axes[1].set_ylabel(r"$E_{\mathrm{Fock}}$ (meV)")
    axes[1].set_xticks(node_x, node_labels)
    if title is not None:
        axes[0].set_title(title, fontsize=9)
    axes[0].legend(loc="upper left", fontsize=6.8, frameon=False)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"potential_plot_png": png_path, "potential_plot_pdf": pdf_path}


def write_htg_fig3b_plot(
    output_dir: Path | str,
    panels: tuple[tuple[str, PathBandsResult, float], ...],
    *,
    stem: str = "fig3b_chiral_bands",
    ylim: tuple[float, float] = (-0.7, 0.7),
) -> dict[str, Path]:
    if not panels:
        raise ValueError("Expected at least one panel.")
    plt = _load_plot_backend()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.2), sharey=True)
    if len(panels) == 1:
        axes = np.asarray([axes])

    for ax, (title, result, energy_scale) in zip(axes, panels, strict=True):
        energies = np.asarray(result.energies, dtype=float) * float(energy_scale)
        node_x = [float(node.k_dist) for node in result.path.nodes]
        node_labels = [_display_node_label(node.label) for node in result.path.nodes]
        for xpos in node_x:
            ax.axvline(x=xpos, color="#9a9a9a", linestyle=":", linewidth=0.75, zorder=0)
        ax.axhline(y=0.0, color="#777777", linestyle="-", linewidth=0.45, alpha=0.55, zorder=0)
        for band_index in range(energies.shape[1]):
            ax.plot(result.path.kdist, energies[:, band_index], color="#1f1f1f", linewidth=0.62, alpha=0.9, zorder=2)
        ax.set_xticks(node_x, node_labels)
        ax.set_xlim(float(node_x[0]), float(node_x[-1]))
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k-path")
    axes[0].set_ylabel(r"$E / v k_\theta$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"band_plot_png": png_path, "band_plot_pdf": pdf_path}
