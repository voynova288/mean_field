from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .hipolito2016 import eq25_prefactor_scale, hipolito_eq25b_spectrum_energy_intervals
from .run_hipolito2016_benchmark_suite import crop_reference_fig5
from .slg_toy import GappedSLGParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hipolito 2016 Fig. 5(a) low-energy gap-series benchmark.  This focuses on the "
            "K/K' threshold region and Eq.(31) scaling, not the high-energy M/UV van-Hove peaks."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/hipolito2016_fig5a_gap_series"))
    parser.add_argument("--deltas-ev", default="0.1,0.2,0.5,1.0,2.0")
    parser.add_argument("--calibration-delta-ev", type=float, default=0.2)
    parser.add_argument("--hopping-ev", type=float, default=3.0)
    parser.add_argument("--gamma-mev", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--emin", type=float, default=0.02)
    parser.add_argument("--emax", type=float, default=2.5)
    parser.add_argument("--n-photon", type=int, default=401)
    parser.add_argument("--theta-count", type=int, default=48)
    parser.add_argument("--transition-energy-intervals", type=int, default=360)
    parser.add_argument("--transition-margin-ev", type=float, default=0.4)
    parser.add_argument("--patch-radius-nm-inv", type=float, default=3.5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--reference-offset-ev", type=float, default=0.03)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def eq31_target(delta_ev: float, hopping_ev: float) -> float:
    return -1.0 / (4.0 * (float(delta_ev) / float(hopping_ev)))


def compute_raw(delta_ev: float, args: argparse.Namespace, photon: np.ndarray) -> tuple[np.ndarray, dict[str, object], GappedSLGParams]:
    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * float(delta_ev))
    transition_emax = max(float(args.emax) + float(args.transition_margin_ev), float(delta_ev) + 0.2)
    raw, meta = hipolito_eq25b_spectrum_energy_intervals(
        params,
        photon,
        transition_emax_ev=transition_emax,
        theta_count=int(args.theta_count),
        transition_energy_intervals=int(args.transition_energy_intervals),
        patch_radius_nm_inv=float(args.patch_radius_nm_inv),
        gamma_ev=float(args.gamma_mev) * 1.0e-3,
        mu_ev=0.0,
        temperature_k=float(args.temperature_k),
    )
    return raw, meta, params


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = parse_float_list(args.deltas_ev)
    if float(args.calibration_delta_ev) not in deltas:
        deltas = sorted(set(deltas + [float(args.calibration_delta_ev)]))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_photon), dtype=float)

    raw_by_delta: dict[float, np.ndarray] = {}
    meta_by_delta: dict[float, dict[str, object]] = {}
    params_by_delta: dict[float, GappedSLGParams] = {}
    for delta in deltas:
        raw, meta, params = compute_raw(delta, args, photon)
        raw_by_delta[float(delta)] = raw
        meta_by_delta[float(delta)] = meta
        params_by_delta[float(delta)] = params

    # One global convention factor, calibrated on Delta=0.2 eV by Eq.(31), then
    # reused for all gaps.  This avoids per-curve visual normalization.
    cal_delta = float(args.calibration_delta_ev)
    cal_params = params_by_delta[cal_delta]
    cal_raw = raw_by_delta[cal_delta]
    cal_direct = eq25_prefactor_scale(cal_params) * cal_raw
    cal_idx = int(np.argmin(np.abs(photon - (cal_delta + float(args.reference_offset_ev)))))
    cal_target = eq31_target(cal_delta, float(args.hopping_ev))
    global_scale = eq25_prefactor_scale(cal_params) * cal_target / float(cal_direct[cal_idx].real)

    sigma_by_delta = {delta: global_scale * raw for delta, raw in raw_by_delta.items()}
    np.savez(
        out_dir / "hipolito2016_fig5a_gap_series_data.npz",
        photon_energies_ev=photon,
        deltas_ev=np.asarray(deltas, dtype=float),
        **{f"sigma_delta_{str(delta).replace('.', 'p')}": sigma_by_delta[delta] for delta in deltas},
    )

    crop_path = out_dir / "hipolito2016_fig5_reference_crop.png"
    have_crop = crop_reference_fig5(args.reference_page, crop_path)

    fig, axes = plt.subplots(1, 2 if have_crop else 1, figsize=(10.5 if have_crop else 5.3, 4.2), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for idx, delta in enumerate(deltas):
        y = np.maximum(-sigma_by_delta[delta].real, 1.0e-5)
        ax.semilogy(photon, y, lw=1.5, color=cmap(idx % 10), label=f"{delta:g}")
        target = -eq31_target(delta, float(args.hopping_ev))
        x0 = float(delta)
        x1 = min(float(args.emax), x0 + 0.25)
        if x0 < float(args.emax):
            ax.semilogy([x0, x1], [target, target], ls="--", lw=0.8, color=cmap(idx % 10), alpha=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_ylim(4.0e-2, max(10.0, 1.2 * max(-np.min(s.real) for s in sigma_by_delta.values())))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$-\mathrm{Re}\,\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Hipolito Fig. 5(a): gap thresholds")
    ax.grid(True, which="both", alpha=0.2, lw=0.5)
    ax.legend(title=r"$\Delta$ [eV]", frameon=False, fontsize=8, ncol=2)
    if have_crop:
        axes[1].imshow(Image.open(crop_path))
        axes[1].axis("off")
        axes[1].set_title("Hipolito Fig. 5 crop")
    fig.savefig(out_dir / "hipolito2016_fig5a_gap_series.png", dpi=180)
    fig.savefig(out_dir / "hipolito2016_fig5a_gap_series.pdf")
    plt.close(fig)

    metrics = {}
    for delta in deltas:
        idx = int(np.argmin(np.abs(photon - (float(delta) + float(args.reference_offset_ev)))))
        metrics[f"Delta_{delta:g}"] = {
            "Re_at_Delta_plus_offset": float(sigma_by_delta[delta].real[idx]),
            "Eq31_target": eq31_target(delta, float(args.hopping_ev)),
            "abs_error": abs(float(sigma_by_delta[delta].real[idx]) - eq31_target(delta, float(args.hopping_ev))),
            **meta_by_delta[delta],
        }
    summary = {
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Fig. 5(a) low-energy threshold series",
        "scope_note": "This benchmark reproduces the K/K' threshold/gap-dependence part of Fig. 5(a). It does not attempt the full 0-9 eV high-energy M/UV van-Hove structure.",
        "method": "Eq.(25b) with analytic generalized derivatives and transition-energy intervals whose resonant denominators are integrated analytically; one global convention factor calibrated by Eq.(31) at Delta=0.2 eV.",
        "parameters": {
            "deltas_ev": deltas,
            "calibration_delta_ev": float(args.calibration_delta_ev),
            "gamma0_ev": float(args.hopping_ev),
            "Gamma_mev": float(args.gamma_mev),
            "theta_count": int(args.theta_count),
            "transition_energy_intervals": int(args.transition_energy_intervals),
            "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
        },
        "global_scale": float(global_scale),
        "metrics": metrics,
        "outputs": {
            "png": str(out_dir / "hipolito2016_fig5a_gap_series.png"),
            "pdf": str(out_dir / "hipolito2016_fig5a_gap_series.pdf"),
            "data": str(out_dir / "hipolito2016_fig5a_gap_series_data.npz"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "hipolito2016_fig5a_gap_series.png")


if __name__ == "__main__":
    main()
