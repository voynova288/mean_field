from __future__ import annotations

from pathlib import Path

import numpy as np

from .bands import PathBandsResult
from .lattice import HTQGLattice


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_lattice(lattice: HTQGLattice, output_path: str | Path) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    g = np.asarray(lattice.g_vectors)
    ax.scatter(g.real, g.imag, s=10, c="0.75", label="G shell")
    q = np.asarray(lattice.q_vectors)
    ax.scatter(q.real, q.imag, s=50, c=["tab:red", "tab:green", "tab:blue"], label="q_j")
    for idx, qv in enumerate(q):
        ax.text(qv.real, qv.imag, f" q{idx}")
    special = {
        "Γ": lattice.gamma,
        "κ(path)": lattice.kappa_path,
        "κ'(path)": lattice.kappap_path,
        "M": lattice.m_path,
    }
    for label, kval in special.items():
        ax.scatter([kval.real], [kval.imag], s=70, marker="x", c="black")
        ax.text(kval.real, kval.imag, f" {label}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k_x$ [nm$^{-1}$]")
    ax.set_ylabel(r"$k_y$ [nm$^{-1}$]")
    ax.set_title(f"HTQG moire geometry, theta={lattice.theta_deg:.3f} deg")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_path_bands(
    result: PathBandsResult,
    output_path: str | Path,
    *,
    energy_window_ev: tuple[float, float] | None = (-0.1, 0.1),
    energy_unit: str = "meV",
) -> Path:
    plt = _load_pyplot()
    scale = 1000.0 if energy_unit == "meV" else 1.0
    ylabel = "Energy [meV]" if energy_unit == "meV" else "Energy [eV]"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    energies = np.asarray(result.energies, dtype=float) * scale
    for ib in range(energies.shape[1]):
        ax.plot(result.path.kdist, energies[:, ib], color="black", lw=0.8)
    for node in result.path.nodes:
        ax.axvline(node.k_dist, color="0.85", lw=0.6)
    ax.set_xticks([node.k_dist for node in result.path.nodes])
    ax.set_xticklabels([node.label for node in result.path.nodes])
    ax.set_ylabel(ylabel)
    if energy_window_ev is not None:
        ax.set_ylim(float(energy_window_ev[0]) * scale, float(energy_window_ev[1]) * scale)
    ax.set_title("HTQG continuum bands")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


__all__ = ["plot_lattice", "plot_path_bands"]
