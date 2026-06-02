from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    b0_fig2_kpath,
    build_chau_b0_hamiltonian,
    config_summary,
    make_b0_parameters,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Chaudhary 2021 Fig. 2(a)-style bands using the previous b0 BM model.")
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=9)
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--points-per-segment", type=int, default=80)
    parser.add_argument("--band-window", type=int, default=8)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_b0_bands"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ChaudharyTBGConfig(
        theta_deg=float(args.theta_deg),
        n_shells=0,
        kinetic_ev=float(args.kinetic_ev),
        w_ab_ev=float(args.w_ab_mev) * 1.0e-3,
        w_aa_ratio=float(args.w_aa_ratio),
        delta1_ev=float(args.delta1_mev) * 1.0e-3,
        delta2_ev=float(args.delta2_mev) * 1.0e-3,
        valley=int(args.valley),
        dirac_sign=-1.0,
    )
    params = make_b0_parameters(config)
    path = b0_fig2_kpath(params, int(args.points_per_segment))
    lg = int(args.lg)
    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if bool(args.periodic_g_grid) else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, lg, int(args.valley))
    dim = 4 * lg * lg
    center = dim // 2
    lo = max(0, center - int(args.band_window))
    hi = min(dim, center + int(args.band_window))
    bands = []
    for k in path.kvec:
        evals = eigvalsh(
            build_chau_b0_hamiltonian(
                complex(k),
                params,
                config,
                lg=lg,
                sigma_rotation=not bool(args.no_sigma_rotation),
                periodic_g_grid=bool(args.periodic_g_grid),
                gvec=gvec,
                tunnel=tunnel,
            )
        )
        bands.append(evals[lo:hi])
    bands = np.asarray(bands, dtype=float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for ib in range(bands.shape[1]):
        ax.plot(path.kdist, 1.0e3 * bands[:, ib], color="black", lw=0.9)
    for idx in path.node_indices:
        ax.axvline(path.kdist[idx - 1], color="0.82", lw=0.8)
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.set_xticks([path.kdist[idx - 1] for idx in path.node_indices])
    ax.set_xticklabels(path.labels)
    ax.set_ylabel("E [meV]")
    ax.set_ylim(-65, 65)
    ax.set_title(
        f"Chaudhary 2021 b0 bands: theta={args.theta_deg}°, "
        f"Delta=({args.delta1_mev:g},{args.delta2_mev:g}) meV, lg={lg}"
    )
    fig.savefig(args.output_dir / "chaudhary2021_b0_bands.png", dpi=220)
    fig.savefig(args.output_dir / "chaudhary2021_b0_bands.pdf")
    plt.close(fig)

    summary = {
        "status": "band-only correction using previous b0 noninteracting model",
        "config": config_summary(config, b0_params=params, lg=lg),
        "run": {
            "points_per_segment": int(args.points_per_segment),
            "band_window": int(args.band_window),
            "sigma_rotation": not bool(args.no_sigma_rotation),
            "periodic_g_grid": bool(args.periodic_g_grid),
            "path_labels": list(path.labels),
            "band_slice": [int(lo), int(hi)],
        },
        "central_band_minmax_mev": [float(1.0e3 * np.min(bands[:, bands.shape[1] // 2 - 1 : bands.shape[1] // 2 + 1])), float(1.0e3 * np.max(bands[:, bands.shape[1] // 2 - 1 : bands.shape[1] // 2 + 1]))],
    }
    np.savez(args.output_dir / "chaudhary2021_b0_bands.npz", path_kdist=path.kdist, bands_ev=bands)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
