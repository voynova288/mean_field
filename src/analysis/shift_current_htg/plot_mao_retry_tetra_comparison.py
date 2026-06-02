from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def safe(text: str) -> str:
    return (
        str(text)
        .replace(";", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace("|", "_")
        .replace(".", "p")
    )


def comp_tuple(component: str) -> tuple[int, int, int]:
    left, right = component.split(";", 1)
    labels = {"x": 0, "y": 1}
    return labels[left], labels[right[0]], labels[right[1]]


def load_component(path: Path, group: str, eta_mev: float, component: str, *, rotation_deg: float, reflect_y: bool) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    photon = np.asarray(data["photon_energies_ev"], dtype=float)
    eta = f"eta_{float(eta_mev):g}meV".replace(".", "p")
    labels = ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")
    tensor = np.zeros((2, 2, 2, photon.size), dtype=float)
    for label in labels:
        key = "sigma_" + safe(f"{group}|{label}|{eta}")
        tensor[comp_tuple(label)] = np.asarray(data[key], dtype=float)
    angle = np.deg2rad(float(rotation_deg))
    rot = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)
    reflection = np.diag([1.0, -1.0]) if reflect_y else np.eye(2)
    transform = rot @ reflection
    a, b, c = comp_tuple(component)
    values = np.einsum("i,j,k,ijkw->w", transform[a], transform[b], transform[c], tensor)
    return photon, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/shift_current_htg/mao_retry_tetra_comparison.png"))
    parser.add_argument("--fig1-crop", type=Path, default=Path("tmp/pdfs/render/fig1_crop.png"))
    parser.add_argument("--fig2-crop", type=Path, default=Path("tmp/pdfs/render/fig2_crop.png"))
    parser.add_argument("--central-flat", type=Path, default=Path("results/shift_current_htg/mao_retry_tetra_central_flat_shell5_m28/htg_bandpair_spectra.npz"))
    parser.add_argument("--active24", type=Path, default=Path("results/shift_current_htg/mao_retry_tetra_active24_shell3_m12/htg_bandpair_spectra.npz"))
    parser.add_argument("--rotation-deg", type=float, default=4.8)
    parser.add_argument("--eta-mev", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for ax, path, title in ((axes[0, 0], args.fig2_crop, "Mao Fig. 2 crop"), (axes[1, 0], args.fig1_crop, "Mao Fig. 1 crop")):
        if path.exists():
            ax.imshow(Image.open(path))
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"missing {path}", ha="center", va="center")
            ax.axis("off")
        ax.set_title(title)

    panels = [
        (axes[0, 1], args.central_flat, "central_flat", "Retry Fig. 2 central flat"),
        (axes[1, 1], args.active24, "active24", "Retry Fig. 1(b) active24"),
    ]
    for ax, npz, group, title_prefix in panels:
        photon, xyy = load_component(npz, group, args.eta_mev, "x;yy", rotation_deg=args.rotation_deg, reflect_y=True)
        _, yxx = load_component(npz, group, args.eta_mev, "y;xx", rotation_deg=args.rotation_deg, reflect_y=True)
        ax.plot(photon, xyy, color="#0047ff", lw=2.0, label=r"$\sigma^{x;yy}$")
        ax.plot(photon, yxx, color="#e41a1c", lw=2.0, label=r"$\sigma^{y;xx}$")
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_xlim(0.0, 0.12)
        ax.set_xlabel(r"$\hbar\omega$ [eV]")
        ax.set_ylabel(r"$\sigma$ [$\mu$A nm V$^{-2}$]")
        summary_path = npz.with_name("summary.json")
        config_label = ""
        if summary_path.exists():
            config = json.loads(summary_path.read_text()).get("config", {})
            config_label = f", shells={config.get('n_shells', '?')}, Nk={config.get('mesh_size', '?')}"
        ax.set_title(f"{title_prefix}{config_label}\neta={args.eta_mev:g} meV, reflect_y + rot={args.rotation_deg:g}°")
        ax.grid(True, alpha=0.22, lw=0.6)
        ax.legend(frameon=True, fontsize=8)
        ax.text(
            0.02,
            0.03,
            "rotation is empirical / not claimed as final convention",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(args.output)


if __name__ == "__main__":
    main()
