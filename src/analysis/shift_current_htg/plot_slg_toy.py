from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SLG toy spectra from run_slg_toy.py")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--components", nargs="+", default=["x;xy", "y;yy"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xmax", type=float, default=8.0)
    return parser.parse_args()


def key(component: str) -> str:
    return "sigma_" + component.replace(";", "_")


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    photon = np.asarray(data["photon_energies_ev"], dtype=float)
    colors = {"x;xy": "#0047ff", "y;yy": "#e41a1c", "x;yy": "#0047ff", "y;xx": "#e41a1c"}
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    for component in args.components:
        ax.plot(photon, np.asarray(data[key(component)], dtype=float), lw=1.8, color=colors.get(component), label=rf"$\sigma^{{{component}}}$")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(float(np.min(photon)), float(args.xmax))
    ax.set_xlabel(r"photon energy $E_\gamma$ [eV]")
    ax.set_ylabel(r"$\sigma$ [$\mu$A nm V$^{-2}$]")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.2, lw=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
