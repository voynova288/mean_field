from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from mean_field.systems.htg.lattice import build_kpath_from_nodes, build_moire_k_grid

from .htg_adapter import MaoHTGConfig, build_mao_hamiltonian, make_mao_model, stacking_displacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot Mao hTTG band structure plus DOS for Fig. 1(a)-style checks.")
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=2)
    parser.add_argument("--points-per-segment", type=int, default=36)
    parser.add_argument("--central-band-count", type=int, default=24)
    parser.add_argument("--dos-mesh", type=int, default=10)
    parser.add_argument("--dos-sigma-mev", type=float, default=5.0)
    parser.add_argument("--energy-window", type=float, default=0.5, help="plot window +/- value in eV")
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_htg_bands_dos"))
    return parser.parse_args()


def central_indices(matrix_dim: int, count: int) -> np.ndarray:
    count = min(int(count), int(matrix_dim))
    center = int(matrix_dim) // 2
    lo = max(0, center - count // 2)
    hi = min(int(matrix_dim), lo + count)
    lo = max(0, hi - count)
    return np.arange(lo, hi, dtype=int)


def gaussian_dos(energies: np.ndarray, grid: np.ndarray, sigma_ev: float, weight: float) -> np.ndarray:
    diff = grid[:, None] - np.ravel(energies)[None, :]
    pref = float(weight) / (float(sigma_ev) * np.sqrt(2.0 * np.pi))
    return pref * np.sum(np.exp(-0.5 * (diff / float(sigma_ev)) ** 2), axis=1)


def main() -> None:
    args = parse_args()
    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=float(args.mass_mev) * 1.0e-3,
        domain=str(args.domain),
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(
        model.lattice,
        config.stacking,
        valley=config.valley,
        domain=config.domain,
    )
    lattice = model.lattice

    # Fig. 1(a)-style path: M - Gamma - M - K' - K - Gamma.
    # Important convention: the K' following the second M is the adjacent
    # extended-zone corner on the same mBZ edge, not the central-zone K'
    # point opposite K.  Using the central-zone K' cuts across the Brillouin
    # zone and visibly gives the wrong band path.
    kprime_edge = lattice.kappa_prime_m + lattice.b_m1
    nodes = (-lattice.m_m, lattice.gamma_m, lattice.m_m, kprime_edge, lattice.kappa_m, lattice.gamma_m)
    labels = ("M", r"$\Gamma$", "M", "K'", "K", r"$\Gamma$")
    path = build_kpath_from_nodes(nodes, labels, points_per_segment=int(args.points_per_segment))
    selected = central_indices(model.matrix_dim, int(args.central_band_count))

    path_bands = np.empty((path.kvec.size, selected.size), dtype=float)
    for ik, k_tilde in enumerate(path.kvec):
        evals = eigh(build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot), eigvals_only=True)
        path_bands[ik] = np.asarray(evals, dtype=float)[selected]

    _, k_grid = build_moire_k_grid(lattice, int(args.dos_mesh), endpoint=False, frac_shift=(0.5, 0.5))
    k_points = np.asarray(k_grid, dtype=np.complex128).reshape(-1)
    dos_evals = []
    for k_tilde in k_points:
        evals = eigh(build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot), eigvals_only=True)
        dos_evals.append(np.asarray(evals, dtype=float)[selected])
    dos_evals_array = np.asarray(dos_evals, dtype=float)
    energy_grid = np.linspace(-float(args.energy_window), float(args.energy_window), 600)
    dos = gaussian_dos(dos_evals_array, energy_grid, float(args.dos_sigma_mev) * 1.0e-3, weight=1.0 / float(k_points.size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "htg_bands_dos.npz"
    np.savez(
        npz_path,
        path_kdist=path.kdist,
        path_bands_ev=path_bands,
        node_indices=np.asarray(path.node_indices, dtype=int),
        node_labels=np.asarray(labels, dtype=object),
        dos_energy_ev=energy_grid,
        dos=dos,
        selected_band_indices=selected,
    )

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.2),
        gridspec_kw={"width_ratios": [3.2, 1.1], "wspace": 0.04},
        constrained_layout=True,
    )
    for ib in range(path_bands.shape[1]):
        ax_band.plot(path.kdist, path_bands[:, ib], color="black", lw=1.0)
    for node in path.nodes:
        ax_band.axvline(node.k_dist, color="0.75", lw=0.6)
    ax_band.axhline(0.0, color="0.5", lw=0.6)
    ax_band.set_xlim(float(path.kdist[0]), float(path.kdist[-1]))
    ax_band.set_ylim(-float(args.energy_window), float(args.energy_window))
    ax_band.set_ylabel("E [eV]")
    ax_band.set_xticks([node.k_dist for node in path.nodes])
    ax_band.set_xticklabels(labels)
    ax_band.set_title(f"{config.stacking} hTTG, theta={config.theta_deg:g} deg, r={config.corrugation_r:g}")

    ax_dos.fill_betweenx(energy_grid, 0.0, dos, color="#6baed6", alpha=0.85, lw=0.0)
    ax_dos.plot(dos, energy_grid, color="black", lw=0.8)
    ax_dos.axhline(0.0, color="0.5", lw=0.6)
    ax_dos.set_ylim(ax_band.get_ylim())
    ax_dos.set_xlabel("DOS [arb.]", fontsize=9)
    ax_dos.set_yticklabels([])
    ax_dos.tick_params(axis="y", length=0)

    png_path = args.output_dir / "htg_bands_dos.png"
    fig.savefig(png_path, dpi=180)
    fig.savefig(args.output_dir / "htg_bands_dos.pdf")

    summary = {
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "matrix_dim": model.matrix_dim,
            "central_band_count": int(selected.size),
            "dos_mesh": int(args.dos_mesh),
            "points_per_segment": int(args.points_per_segment),
            "dos_sigma_mev": float(args.dos_sigma_mev),
        },
        "outputs": {"npz": str(npz_path), "png": str(png_path)},
        "central_gap_at_path_min_abs_ev": float(np.min(np.abs(path_bands))),
        "central_band_minmax_ev": [float(np.min(path_bands)), float(np.max(path_bands))],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
