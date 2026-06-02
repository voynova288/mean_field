from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .hipolito2016 import eq25_prefactor_scale, hipolito_eq25b_spectrum_full_bz_tetra_binned
from .run_hipolito2016_benchmark_suite import crop_reference_fig5
from .slg_toy import GappedSLGParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-BZ Hipolito 2016 Fig. 5(a) reproduction using binned linear-tetrahedron "
            "transition-energy integration and analytic resonant-denominator intervals."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/hipolito2016_fig5a_full_bz_tetra"))
    parser.add_argument("--deltas-ev", default="0.1,0.2,0.5,1,2,3,4,5,6,7")
    parser.add_argument("--calibration-delta-ev", type=float, default=0.2)
    parser.add_argument("--hopping-ev", type=float, default=3.0)
    parser.add_argument("--gamma-mev", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--mesh-size", type=int, default=360)
    parser.add_argument("--energy-bin-width-mev", type=float, default=2.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=9.5)
    parser.add_argument("--n-photon", type=int, default=951)
    parser.add_argument("--reference-offset-ev", type=float, default=0.03)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes over gap values; set to SLURM_CPUS_PER_TASK for production.")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def eq31_target(delta_ev: float, hopping_ev: float) -> float:
    return -1.0 / (4.0 * (float(delta_ev) / float(hopping_ev)))


def m_transition_ev(delta_ev: float, hopping_ev: float) -> float:
    return 2.0 * math.sqrt(float(hopping_ev) ** 2 + (0.5 * float(delta_ev)) ** 2)


def safe_key(delta: float) -> str:
    return f"sigma_delta_{str(float(delta)).replace('.', 'p').replace('-', 'm')}"


def compute_one_delta(task: tuple[float, np.ndarray, dict[str, float | int]]) -> tuple[float, np.ndarray, dict[str, object]]:
    delta, photon, cfg = task
    params = GappedSLGParams(hopping_ev=float(cfg["hopping_ev"]), mass_ev=0.5 * float(delta))
    raw, hist = hipolito_eq25b_spectrum_full_bz_tetra_binned(
        params,
        photon,
        gamma_ev=float(cfg["gamma_mev"]) * 1.0e-3,
        temperature_k=float(cfg["temperature_k"]),
        mesh_size=int(cfg["mesh_size"]),
        energy_bin_width_ev=float(cfg["energy_bin_width_mev"]) * 1.0e-3,
    )
    meta = {
        "mesh_size": int(hist.mesh_size),
        "n_triangles": int(hist.n_triangles),
        "n_energy_bins": int(hist.n_bins),
        "energy_bin_width_ev": float(hist.energy_bin_width_ev),
        "max_hist_energy_ev": float(hist.energy_edges_ev[-1]),
        "primitive_cell_area_nm_inv_sq": float(hist.primitive_cell_area_nm_inv_sq),
    }
    return float(delta), raw, meta


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = parse_float_list(args.deltas_ev)
    if float(args.calibration_delta_ev) not in deltas:
        deltas = sorted(set(deltas + [float(args.calibration_delta_ev)]))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_photon), dtype=float)

    raw_by_delta: dict[float, np.ndarray] = {}
    hist_meta_by_delta: dict[float, dict[str, object]] = {}
    cfg = {
        "hopping_ev": float(args.hopping_ev),
        "gamma_mev": float(args.gamma_mev),
        "temperature_k": float(args.temperature_k),
        "mesh_size": int(args.mesh_size),
        "energy_bin_width_mev": float(args.energy_bin_width_mev),
    }
    tasks = [(float(delta), photon, cfg) for delta in deltas]
    if int(args.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            results = list(pool.map(compute_one_delta, tasks))
    else:
        results = [compute_one_delta(task) for task in tasks]
    for delta, raw, meta in results:
        raw_by_delta[float(delta)] = raw
        hist_meta_by_delta[float(delta)] = meta

    cal_delta = float(args.calibration_delta_ev)
    cal_params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * cal_delta)
    cal_direct = eq25_prefactor_scale(cal_params) * raw_by_delta[cal_delta]
    cal_idx = int(np.argmin(np.abs(photon - (cal_delta + float(args.reference_offset_ev)))))
    cal_target = eq31_target(cal_delta, float(args.hopping_ev))
    global_scale = eq25_prefactor_scale(cal_params) * cal_target / float(cal_direct[cal_idx].real)
    spectra = {delta: global_scale * raw for delta, raw in raw_by_delta.items()}

    np.savez(
        out_dir / "hipolito2016_fig5a_full_bz_tetra_data.npz",
        photon_energies_ev=photon,
        deltas_ev=np.asarray(deltas, dtype=float),
        **{safe_key(delta): spectra[delta] for delta in deltas},
    )

    crop_path = out_dir / "hipolito2016_fig5_reference_crop.png"
    have_crop = crop_reference_fig5(args.reference_page, crop_path)

    fig, axes = plt.subplots(1, 2 if have_crop else 1, figsize=(11.3 if have_crop else 6.3, 4.5), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for idx, delta in enumerate(deltas):
        y = -spectra[delta].real
        # Log-scale plotting cannot display negative values.  The floor only
        # avoids invalid log pixels; the raw signed spectra are saved in the NPZ
        # and summary so this is not used as a numerical fix.
        ax.semilogy(photon, np.maximum(y, 1.0e-5), lw=1.2, color=cmap(idx % 10), label=f"{delta:g}")
        target = -eq31_target(delta, float(args.hopping_ev))
        if float(delta) < float(args.emax):
            ax.semilogy(
                [float(delta), min(float(args.emax), float(delta) + 0.6)],
                [target, target],
                ls="--",
                lw=0.7,
                color=cmap(idx % 10),
                alpha=0.7,
            )
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_ylim(4.0e-2, max(8.0, 1.15 * max(float(np.nanmax(-spec.real)) for spec in spectra.values())))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$-\mathrm{Re}\,\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title(f"Hipolito Fig. 5(a), full BZ, $\\Gamma$={float(args.gamma_mev):g} meV")
    ax.grid(True, which="both", alpha=0.2, lw=0.5)
    ax.legend(title=r"$\Delta$ [eV]", frameon=False, fontsize=7, ncol=2)
    if have_crop:
        axes[1].imshow(Image.open(crop_path))
        axes[1].axis("off")
        axes[1].set_title("Hipolito Fig. 5 crop")
    fig.savefig(out_dir / "hipolito2016_fig5a_full_bz_tetra.png", dpi=180)
    fig.savefig(out_dir / "hipolito2016_fig5a_full_bz_tetra.pdf")
    plt.close(fig)

    metrics = {}
    for delta in deltas:
        k_idx = int(np.argmin(np.abs(photon - (float(delta) + float(args.reference_offset_ev)))))
        mt = m_transition_ev(delta, float(args.hopping_ev))
        m_idx = int(np.argmin(np.abs(photon - mt)))
        y = -spectra[delta].real
        metrics[f"Delta_{delta:g}"] = {
            "K_threshold_ev": float(delta),
            "M_transition_ev": float(mt),
            "minus_Re_at_Delta_plus_offset": float(y[k_idx]),
            "minus_Re_at_M_transition_gridpoint": float(y[m_idx]) if mt <= float(args.emax) else None,
            "max_minus_Re_in_plot_window": float(np.nanmax(y)),
            "min_minus_Re_in_plot_window": float(np.nanmin(y)),
            **hist_meta_by_delta[delta],
        }
    summary = {
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Fig. 5(a)",
        "method": "Full primitive-cell BZ integral; linear-tetrahedron transition-energy histogram; analytic interval integration of 1/(omega-E+iGamma) and 1/(omega-E+iGamma)^2; one global Eq.(31) convention calibration at Delta=0.2 eV.",
        "not_plot_smoothing": True,
        "parameters": {
            "deltas_ev": deltas,
            "calibration_delta_ev": float(args.calibration_delta_ev),
            "gamma0_ev": float(args.hopping_ev),
            "Gamma_mev": float(args.gamma_mev),
            "temperature_k": float(args.temperature_k),
            "mesh_size": int(args.mesh_size),
            "energy_bin_width_mev": float(args.energy_bin_width_mev),
            "n_photon": int(args.n_photon),
            "workers": int(args.workers),
        },
        "global_scale": float(global_scale),
        "calibration": {
            "photon_ev": float(photon[cal_idx]),
            "Eq31_target_Re": float(cal_target),
            "computed_Re": float(spectra[cal_delta].real[cal_idx]),
            "abs_error": abs(float(spectra[cal_delta].real[cal_idx]) - float(cal_target)),
        },
        "metrics": metrics,
        "outputs": {
            "png": str(out_dir / "hipolito2016_fig5a_full_bz_tetra.png"),
            "pdf": str(out_dir / "hipolito2016_fig5a_full_bz_tetra.pdf"),
            "data": str(out_dir / "hipolito2016_fig5a_full_bz_tetra_data.npz"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "hipolito2016_fig5a_full_bz_tetra.png")


if __name__ == "__main__":
    main()
