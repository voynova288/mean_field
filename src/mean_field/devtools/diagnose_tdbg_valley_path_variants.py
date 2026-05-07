from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mean_field.core.lattice import KPath
from mean_field.systems.tdbg import TDBGModel, TDBGParameters


def _same_axis_path(reference: KPath, kvec: np.ndarray) -> KPath:
    return KPath(
        kvec=np.asarray(kvec, dtype=np.complex128),
        kdist=np.asarray(reference.kdist, dtype=float),
        labels=reference.labels,
        node_indices=reference.node_indices,
    )


def _window_indices(energies: np.ndarray, center: int, half_width: int = 3) -> range:
    return range(max(0, center - half_width), min(energies.shape[1], center + half_width + 1))


def main() -> None:
    output_dir = Path("results/TDBG/tdbg_valley_path_diagnostic_20260425").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = 1.33
    resolution = 16
    deltas = (0.0, 0.005)
    variants: list[tuple[str, str]] = [
        ("same_path", "same physical K-Gamma-M-Kprime path"),
        ("opposite_corner_path", "Kprime-Gamma-M-K path drawn on same axis"),
        ("rotate_about_gamma", "180-degree rotated path about Gamma drawn on same axis"),
        ("negate_about_origin", "negated k path drawn on same axis"),
    ]

    fig, axes = plt.subplots(len(deltas), len(variants), figsize=(4.0 * len(variants), 3.6 * len(deltas)), sharey=True)
    axes = np.atleast_2d(axes)

    for row, delta_ev in enumerate(deltas):
        params = TDBGParameters.full(stacking="AB-AB", Delta=delta_ev)
        model = TDBGModel.from_config(theta_deg, cut=4.0, params=params)
        lattice = model.lattice
        base_path = model.standard_kpath(resolution=resolution)
        plus = model.bands_along_path(base_path, valley=1, n_bands=model.matrix_dim)
        center = model.matrix_dim // 2

        minus_paths = {
            "same_path": base_path,
            "opposite_corner_path": model.build_kpath(
                (lattice.kprime_m, lattice.gamma_m, lattice.m_m, lattice.k_m),
                ("K", "Gamma", "M", "Kprime"),
                segment_point_counts=(resolution, int(np.sqrt(3.0) * resolution / 2.0), int(resolution / 2.0)),
                duplicate_nodes=True,
            ),
            "rotate_about_gamma": _same_axis_path(base_path, 2.0 * lattice.gamma_m - base_path.kvec),
            "negate_about_origin": _same_axis_path(base_path, -base_path.kvec),
        }

        for col, (variant_key, variant_label) in enumerate(variants):
            axis = axes[row, col]
            minus = model.bands_along_path(minus_paths[variant_key], valley=-1, n_bands=model.matrix_dim)
            for idx in _window_indices(plus.energies, center):
                axis.plot(base_path.kdist, plus.energies[:, idx] * 1000.0, color="black", linewidth=0.9)
                axis.plot(base_path.kdist, minus.energies[:, idx] * 1000.0, color="red", linestyle=(0, (3, 2)), linewidth=0.8)
            for node in base_path.nodes:
                axis.axvline(node.k_dist, color="0.7", linewidth=0.6)
            axis.set_xticks([node.k_dist for node in base_path.nodes], ["K", r"$\Gamma$", "M", "K'"])
            axis.set_ylim(-45, 45)
            axis.set_title(variant_key, fontsize=9)
            if col == 0:
                axis.set_ylabel(f"Delta={delta_ev * 1000:.0f} meV\nEnergy (meV)")
            axis.text(0.02, 0.96, variant_label, transform=axis.transAxes, va="top", ha="left", fontsize=7)

    fig.tight_layout()
    png_path = output_dir / "tdbg_valley_path_variants.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
