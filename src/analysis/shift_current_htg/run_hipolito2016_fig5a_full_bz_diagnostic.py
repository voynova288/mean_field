from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .hipolito2016 import eq25_prefactor_scale, hipolito_eq25b_spectrum_fixed_grid
from .run_hipolito2016_benchmark_suite import crop_reference_fig5
from .slg_toy import GappedSLGParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-BZ Hipolito Fig. 5(a) diagnostic with finite broadening. "
            "This captures both K/K' thresholds and M-point van-Hove features, "
            "but uses a broader Gamma by default to avoid fixed-grid shell wiggles."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/hipolito2016_fig5a_full_bz_diagnostic"))
    parser.add_argument("--deltas-ev", default="0.1,0.2,0.5,1,2,3,4,5,6,7")
    parser.add_argument("--calibration-delta-ev", type=float, default=0.2)
    parser.add_argument("--hopping-ev", type=float, default=3.0)
    parser.add_argument("--gamma-mev", type=float, default=30.0)
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--mesh-size", type=int, default=180)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=9.5)
    parser.add_argument("--n-photon", type=int, default=951)
    parser.add_argument("--reference-offset-ev", type=float, default=0.05)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def eq31_target(delta_ev: float, hopping_ev: float) -> float:
    return -1.0 / (4.0 * (float(delta_ev) / float(hopping_ev)))


def m_transition_ev(delta_ev: float, hopping_ev: float) -> float:
    return 2.0 * math.sqrt(float(hopping_ev) ** 2 + (0.5 * float(delta_ev)) ** 2)


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
    for delta in deltas:
        params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * float(delta))
        raw, meta = hipolito_eq25b_spectrum_fixed_grid(
            params,
            photon,
            gamma_ev=float(args.gamma_mev) * 1.0e-3,
            temperature_k=float(args.temperature_k),
            mesh_size=int(args.mesh_size),
        )
        raw_by_delta[float(delta)] = raw
        meta_by_delta[float(delta)] = meta

    cal_delta = float(args.calibration_delta_ev)
    cal_params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * cal_delta)
    cal_direct = eq25_prefactor_scale(cal_params) * raw_by_delta[cal_delta]
    idx = int(np.argmin(np.abs(photon - (cal_delta + float(args.reference_offset_ev)))))
    target = eq31_target(cal_delta, float(args.hopping_ev))
    scale = eq25_prefactor_scale(cal_params) * target / float(cal_direct[idx].real)
    spectra = {delta: scale * raw for delta, raw in raw_by_delta.items()}

    np.savez(
        out_dir / "hipolito2016_fig5a_full_bz_diagnostic_data.npz",
        photon_energies_ev=photon,
        deltas_ev=np.asarray(deltas, dtype=float),
        **{f"sigma_delta_{str(delta).replace('.', 'p')}": spectra[delta] for delta in deltas},
    )

    crop_path = out_dir / "hipolito2016_fig5_reference_crop.png"
    have_crop = crop_reference_fig5(args.reference_page, crop_path)

    fig, axes = plt.subplots(1, 2 if have_crop else 1, figsize=(11.0 if have_crop else 6.0, 4.4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for i, delta in enumerate(deltas):
        y = np.maximum(-spectra[delta].real, 2.0e-3)
        ax.semilogy(photon, y, lw=1.2, color=cmap(i % 10), label=f"{delta:g}")
        if delta <= float(args.emax):
            ax.axvline(delta, color=cmap(i % 10), lw=0.4, alpha=0.25)
        mt = m_transition_ev(delta, float(args.hopping_ev))
        if mt <= float(args.emax):
            ax.axvline(mt, color=cmap(i % 10), lw=0.6, ls="--", alpha=0.35)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_ylim(2.0e-2, max(10.0, 1.1 * max(-np.min(s.real) for s in spectra.values())))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$-\mathrm{Re}\,\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title(f"Full-BZ Fig. 5(a) diagnostic, $\\Gamma$={float(args.gamma_mev):g} meV")
    ax.grid(True, which="both", alpha=0.2, lw=0.5)
    ax.legend(title=r"$\Delta$ [eV]", frameon=False, fontsize=7, ncol=2)
    if have_crop:
        axes[1].imshow(Image.open(crop_path))
        axes[1].axis("off")
        axes[1].set_title("Hipolito Fig. 5 crop")
    fig.savefig(out_dir / "hipolito2016_fig5a_full_bz_diagnostic.png", dpi=180)
    fig.savefig(out_dir / "hipolito2016_fig5a_full_bz_diagnostic.pdf")
    plt.close(fig)

    metrics = {}
    for delta in deltas:
        mt = m_transition_ev(delta, float(args.hopping_ev))
        idx_m = int(np.argmin(np.abs(photon - mt)))
        idx_k = int(np.argmin(np.abs(photon - (float(delta) + float(args.reference_offset_ev)))))
        metrics[f"Delta_{delta:g}"] = {
            "K_threshold_ev": float(delta),
            "M_transition_ev": float(mt),
            "minus_Re_near_K_threshold": float(-spectra[delta].real[idx_k]),
            "minus_Re_near_M_transition": float(-spectra[delta].real[idx_m]),
            **meta_by_delta[delta],
        }
    summary = {
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Fig. 5(a) full-BZ diagnostic",
        "scope_note": "Uses full BZ and includes M transitions, but default Gamma=30 meV is broader than the published Gamma=1 meV to make fixed-grid quadrature stable. Use low-energy energy-quadrature benchmarks for strict Gamma=1 meV threshold tests.",
        "parameters": {
            "deltas_ev": deltas,
            "gamma0_ev": float(args.hopping_ev),
            "Gamma_mev": float(args.gamma_mev),
            "mesh_size": int(args.mesh_size),
            "n_photon": int(args.n_photon),
        },
        "global_scale": float(scale),
        "metrics": metrics,
        "outputs": {
            "png": str(out_dir / "hipolito2016_fig5a_full_bz_diagnostic.png"),
            "pdf": str(out_dir / "hipolito2016_fig5a_full_bz_diagnostic.pdf"),
            "data": str(out_dir / "hipolito2016_fig5a_full_bz_diagnostic_data.npz"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "hipolito2016_fig5a_full_bz_diagnostic.png")


if __name__ == "__main__":
    main()
