#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_polshyn_s1abc_final")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    return plt


def _label(label: str) -> str:
    return {
        "Gamma": r"$\Gamma$",
        "Kminus": r"$K_-^M$",
        "Kplus": r"$K_+^M$",
        "M": r"$M$",
    }.get(str(label), str(label))


def _sector_color(ispin: int, ieta: int) -> str:
    if ispin == 0 and ieta == 0:
        return "#ffc61e"  # up K+
    if ispin == 1 and ieta == 0:
        return "#4a24c2"  # down K+
    return "#d24a7c"  # K- sectors


def plot_s1a(ax, npz_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=False)
    kdist = np.asarray(data["kdist"], dtype=float)
    energies = 1000.0 * np.asarray(data["energies_ev"], dtype=float)
    selected = np.asarray(data["selected_indices"], dtype=int)
    target = int(np.asarray(data["target_band_index"]).reshape(-1)[0])
    labels = ("Gamma", "Kminus", "M", "Kplus", "Gamma", "M")
    # Reconstruct node positions for the standard S1a path from equal-length segments.
    node_indices = np.linspace(0, len(kdist) - 1, 6, dtype=int)
    for band in selected:
        color = "#d62728" if int(band) == target else "#1f77b4"
        if int(band) == target - 1:
            color = "#2ca02c"
        if int(band) == target - 2:
            color = "#ff7f0e"
        ax.plot(kdist, energies[:, int(band)], lw=1.2, color=color)
    for idx in node_indices:
        ax.axvline(float(kdist[idx]), color="#bdbdbd", lw=0.5, zorder=0)
    ax.set_xticks([float(kdist[idx]) for idx in node_indices])
    ax.set_xticklabels([_label(v) for v in labels], fontsize=8)
    ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    ax.set_ylim(-80, 80)
    ax.set_ylabel("E (meV)")
    ax.text(0.46, 0.73, r"$C=2$", color="#d62728", transform=ax.transAxes, fontsize=10)
    ax.text(0.45, 0.45, r"$C=-1$", color="#2ca02c", transform=ax.transAxes, fontsize=10)


def plot_hf(ax, npz_path: Path, *, panel: str) -> None:
    data = np.load(npz_path, allow_pickle=False)
    x = np.asarray(data["x_ky"], dtype=float)
    energies = 1000.0 * np.asarray(data["grid_energies_ev_shifted"], dtype=float)
    for ispin in range(energies.shape[0]):
        for ieta in range(energies.shape[1]):
            color = _sector_color(ispin, ieta)
            alpha = 1.0 if (ispin, ieta) in {(0, 0), (1, 0)} else 0.9
            for ib in range(energies.shape[2]):
                ax.plot(x, energies[ispin, ieta, ib], "-o", color=color, lw=0.65, ms=2.0, alpha=alpha)
    ax.axhline(0.0, color="#1f77b4", lw=0.7, ls="--")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-20, 20)
    ax.set_xticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$k_y a_M$")
    ax.set_ylabel("E (meV)")
    if panel == "b":
        # Paper-style compact legend.
        ax.plot([], [], "-o", color="#ffc61e", lw=0.7, ms=2, label=r"$\uparrow K_+$")
        ax.plot([], [], "-o", color="#4a24c2", lw=0.7, ms=2, label=r"$\downarrow K_+$")
        ax.plot([], [], "-o", color="#d24a7c", lw=0.7, ms=2, label=r"$\uparrow\downarrow K_-$")
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, handlelength=1.6, columnspacing=0.8)
    else:
        ax.text(0.06, 0.88, r"$k_x=0$", transform=ax.transAxes, fontsize=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble paper-style Polshyn Fig. S1(a-c) from SCF-grid npz artifacts.")
    parser.add_argument("--s1a-npz", type=Path, required=True)
    parser.add_argument("--s1b-npz", type=Path, required=True)
    parser.add_argument("--s1c-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt = _load_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 2.55))
    plot_s1a(axes[0], args.s1a_npz)
    plot_hf(axes[1], args.s1b_npz, panel="b")
    plot_hf(axes[2], args.s1c_npz, panel="c")
    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(-0.22, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    fig.tight_layout(w_pad=1.0, pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=350, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
