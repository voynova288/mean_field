#!/usr/bin/env python3
"""Reproduce the band-structure panels Fig. 2B and Fig. 3B of HTG.

Run from the repository root after installing the package, or with
PYTHONPATH pointing to the repository root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. python examples/reproduce_fig2_fig3.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

import numpy as np

import htg


def _central_metrics_from_grid(model: htg.HTGModel, *, mesh_size: int, central_band_count: int = 4) -> dict[str, float]:
    grid = model.bands_on_grid(mesh_size, central_band_count=central_band_count, return_eigenvectors=False)
    energies = grid.energies
    # central_band_count=4 gives [remote valence, valence, conduction, remote conduction].
    central = energies[:, :, 1:3]
    bandwidth_ev = 0.5 * float(np.max(central) - np.min(central))
    remote_gap_ev = min(
        float(np.min(central[:, :, 0] - energies[:, :, 0])),
        float(np.min(energies[:, :, 3] - central[:, :, 1])),
    )
    return {
        "central_bandwidth_meV": 1000.0 * bandwidth_ev,
        "remote_gap_meV": 1000.0 * remote_gap_ev,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced_outputs"))
    parser.add_argument("--points-per-segment", type=int, default=100)
    parser.add_argument("--n-shells-fig2", type=int, default=4)
    parser.add_argument("--n-shells-fig3", type=int, default=4)
    parser.add_argument("--metric-mesh", type=int, default=15)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fig. 2B: h-HTG continuum model at theta=1.5 deg, kappa=0.7.
    fig2_params = htg.HTGParams.default(kappa=0.7)
    fig2_model = htg.HTGModel.from_config(1.5, n_shells=args.n_shells_fig2, params=fig2_params)
    fig2_path = fig2_model.standard_kpath(points_per_segment=args.points_per_segment)
    fig2_bands = fig2_model.bands_along_path(fig2_path, central_band_count=8, return_eigenvectors=False)
    fig2_files = htg.write_htg_path_band_plot(
        output_dir,
        (htg.HTGPathPlotTrace("h-HTG", fig2_bands, energy_scale=1000.0),),
        stem="fig2b_htg_bands",
        title=r"h-HTG, $\theta=1.5^\circ$, $\kappa=0.7$",
        ylabel="E (meV)",
        ylim=(-200.0, 200.0),
    )

    # A small grid check reproduces the paper's W~15 meV and Egap~85 meV.
    metric_model = htg.HTGModel.from_config(1.5, n_shells=min(args.n_shells_fig2, 4), params=fig2_params)
    fig2_metrics = _central_metrics_from_grid(metric_model, mesh_size=args.metric_mesh)

    # Fig. 3B: chiral model.  The arXiv v1 Table I has a typo in alpha_2;
    # the published value is 1.197, and this is the one that gives the four flat bands.
    chiral_params = htg.HTGParams.chiral(zeta_rad=0.0)
    panels = []
    fig3_metrics: dict[str, dict[str, float]] = {}
    for i_alpha, alpha in enumerate(htg.MAGIC_ALPHA_ZETA0[:2], start=1):
        lattice = htg.build_chiral_lattice_from_alpha(alpha, n_shells=args.n_shells_fig3, params=chiral_params)
        model = htg.HTGModel(lattice=lattice, params=chiral_params)
        path = model.standard_kpath(points_per_segment=args.points_per_segment)
        bands = model.bands_along_path(path, central_band_count=12, return_eigenvectors=False)
        vk_theta = chiral_params.vk_theta_ev(lattice.k_theta)
        panels.append((rf"$\alpha_{i_alpha}={alpha:.3f}$", bands, 1.0 / vk_theta))
        normalized = bands.energies / vk_theta
        center = normalized.shape[1] // 2
        if i_alpha == 1:
            flat = normalized[:, center - 1 : center + 1]
        else:
            flat = normalized[:, center - 2 : center + 2]
        fig3_metrics[f"alpha_{i_alpha}"] = {
            "alpha": float(alpha),
            "theta_deg": float(lattice.theta_deg),
            "flat_manifold_half_span_E_over_vk": 0.5 * float(np.max(flat) - np.min(flat)),
        }

    fig3_files = htg.write_htg_fig3b_plot(
        output_dir,
        tuple(panels),
        stem="fig3b_chiral_bands",
        ylim=(-1.0, 1.0),
    )

    summary = {
        "fig2_files": {key: str(value) for key, value in fig2_files.items()},
        "fig2_grid_metrics": fig2_metrics,
        "fig3_files": {key: str(value) for key, value in fig3_files.items()},
        "fig3_metrics": fig3_metrics,
        "path": "Gamma-kappa-(kappa_prime+b1)-Gamma-M",
        "path_note": "The kappa_prime point is shifted by one moire reciprocal vector b1 so that the kappa->kappa_prime segment is an mBZ edge and does not pass through Gamma.",
        "n_shells_fig2": args.n_shells_fig2,
        "n_shells_fig3": args.n_shells_fig3,
        "points_per_segment": args.points_per_segment,
    }
    summary_path = output_dir / "reproduction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
