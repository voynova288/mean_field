from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .hipolito2016 import (
    hipolito_eq25b_spectrum_energy_intervals,
    normalize_by_eq31,
)
from .run_slg_toy_hipolito_fig4 import crop_reference_fig4
from .slg_toy import GappedSLGParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hipolito 2016 gapped-graphene benchmark figures for the shift-current code."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/hipolito2016_benchmark_suite"))
    parser.add_argument("--delta-ev", type=float, default=0.2)
    parser.add_argument("--hopping-ev", type=float, default=3.0)
    parser.add_argument("--gamma-mev", type=float, default=1.0)
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.8)
    parser.add_argument("--n-photon", type=int, default=321)
    parser.add_argument("--theta-count", type=int, default=72)
    parser.add_argument("--transition-energy-intervals", type=int, default=900)
    parser.add_argument("--transition-emax", type=float, default=1.2)
    parser.add_argument("--patch-radius-nm-inv", type=float, default=1.4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    parser.add_argument("--mu-values", default="0,0.125,0.15,0.175,0.2,0.225,0.25,0.275")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def crop_reference_fig5(reference_page: Path, output: Path) -> bool:
    if not reference_page.exists():
        return False
    image = Image.open(reference_page)
    width, height = image.size
    left = int(0.53 * width)
    upper = int(0.07 * height)
    right = int(0.98 * width)
    lower = int(0.68 * height)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, upper, right, lower)).save(output)
    return True


def compute_raw_for_mu(task: tuple[float, np.ndarray, dict[str, float | int]]) -> tuple[float, np.ndarray, dict[str, object]]:
    mu, photon, cfg = task
    params = GappedSLGParams(hopping_ev=float(cfg["hopping_ev"]), mass_ev=0.5 * float(cfg["delta_ev"]))
    raw, meta = hipolito_eq25b_spectrum_energy_intervals(
        params,
        photon,
        transition_emax_ev=float(cfg["transition_emax"]),
        theta_count=int(cfg["theta_count"]),
        transition_energy_intervals=int(cfg["transition_energy_intervals"]),
        patch_radius_nm_inv=float(cfg["patch_radius_nm_inv"]),
        gamma_ev=float(cfg["gamma_mev"]) * 1.0e-3,
        mu_ev=float(mu),
        temperature_k=float(cfg["temperature_k"]),
    )
    return float(mu), raw, meta


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * float(args.delta_ev))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_photon), dtype=float)
    gamma_ev = float(args.gamma_mev) * 1.0e-3
    mu_values = parse_float_list(args.mu_values)

    cfg = {
        "hopping_ev": float(args.hopping_ev),
        "delta_ev": float(args.delta_ev),
        "transition_emax": float(args.transition_emax),
        "theta_count": int(args.theta_count),
        "transition_energy_intervals": int(args.transition_energy_intervals),
        "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
        "gamma_mev": float(args.gamma_mev),
        "temperature_k": float(args.temperature_k),
    }
    tasks = [(float(mu), photon, cfg) for mu in mu_values]
    if int(args.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            mu_results = list(pool.map(compute_raw_for_mu, tasks))
    else:
        mu_results = [compute_raw_for_mu(task) for task in tasks]
    raw_by_mu = {mu: raw for mu, raw, _ in mu_results}
    interval_meta = mu_results[0][2]
    raw_mu0 = raw_by_mu.get(0.0)
    if raw_mu0 is None:
        raise ValueError("mu-values must include 0 for Eq.(31) calibration")
    sigma_mu0, scale, eq31_target = normalize_by_eq31(raw_mu0, photon, params=params)

    # Reuse the same Eq.(31)-fixed scale for all chemical potentials; do not
    # renormalize individual curves.
    spectra_by_mu: dict[float, np.ndarray] = {0.0: sigma_mu0}
    for mu in mu_values:
        if abs(mu) < 1.0e-15:
            continue
        spectra_by_mu[float(mu)] = scale * raw_by_mu[float(mu)]

    np.savez(
        out_dir / "hipolito2016_benchmark_data.npz",
        photon_energies_ev=photon,
        mu_values_ev=np.asarray(sorted(spectra_by_mu), dtype=float),
        sigma_mu0=sigma_mu0,
        **{f"sigma_mu_{str(mu).replace('.', 'p')}": spectra_by_mu[mu] for mu in sorted(spectra_by_mu)},
    )

    fig4_crop = out_dir / "hipolito2016_fig4_reference_crop.png"
    fig5_crop = out_dir / "hipolito2016_fig5_reference_crop.png"
    have_fig4 = crop_reference_fig4(args.reference_page, fig4_crop)
    have_fig5 = crop_reference_fig5(args.reference_page, fig5_crop)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(photon, sigma_mu0.real, color="#003c4c", lw=1.8, label="Re")
    ax.plot(photon, sigma_mu0.imag, color="#e41a1c", lw=1.4, label="Im")
    ax.plot(
        [float(args.delta_ev), min(float(args.emax), float(args.delta_ev) + 0.18)],
        [eq31_target, eq31_target],
        color="#003c4c",
        ls="--",
        lw=1.0,
        label="Eq.(31) threshold",
    )
    ax.axvline(float(args.delta_ev), color="0.35", ls="--", lw=1.0, label=r"$\Delta$")
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Benchmark A: Hipolito Fig. 4")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    if have_fig4:
        ax.imshow(Image.open(fig4_crop))
        ax.axis("off")
        ax.set_title("Hipolito Fig. 4 crop")
    else:
        ax.axis("off")

    ax = axes[1, 0]
    cmap = plt.get_cmap("tab10")
    for idx, mu in enumerate(sorted(spectra_by_mu)):
        ax.plot(photon, spectra_by_mu[mu].real, lw=1.4, color=cmap(idx % 10), label=f"{mu:g}")
        if mu > 0:
            ax.axvline(2.0 * mu, color=cmap(idx % 10), lw=0.6, alpha=0.35)
    ax.axvline(float(args.delta_ev), color="0.35", ls="--", lw=0.9)
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"Re $\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Benchmark B: Hipolito Fig. 5(b) Pauli blocking")
    ax.legend(title=r"$\mu$ [eV]", frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, lw=0.5)

    ax = axes[1, 1]
    if have_fig5:
        ax.imshow(Image.open(fig5_crop))
        ax.axis("off")
        ax.set_title("Hipolito Fig. 5 crop")
    else:
        ax.axis("off")

    fig.savefig(out_dir / "hipolito2016_benchmark_suite.png", dpi=180)
    fig.savefig(out_dir / "hipolito2016_benchmark_suite.pdf")
    plt.close(fig)

    # Simple acceptance metrics for future regression tests.
    idx_ref = int(np.argmin(np.abs(photon - (float(args.delta_ev) + 0.03))))
    fig4_ref_value = float(sigma_mu0.real[idx_ref])
    pauli_thresholds = {f"mu_{mu:g}": 2.0 * mu for mu in sorted(spectra_by_mu) if mu > 0}
    summary = {
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Figs. 4 and 5(b)",
        "purpose": "Benchmark suite for gapped-graphene photoconductivity / shift-current code paths.",
        "method": "Eq.(25b) with analytic generalized derivatives and transition-energy intervals around K/K' valleys; resonant denominators are integrated analytically over intervals; global convention fixed once by Eq.(31) for mu=0 and reused for all mu.",
        "parameters": {
            "gamma0_ev": float(params.hopping_ev),
            "Delta_ev": float(args.delta_ev),
            "mass_ev": float(params.mass_ev),
            "Gamma_mev": float(args.gamma_mev),
            "temperature_k": float(args.temperature_k),
            "theta_count": int(args.theta_count),
            "transition_energy_intervals": int(args.transition_energy_intervals),
            "transition_emax_ev": float(args.transition_emax),
            "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
            "workers": int(args.workers),
            **interval_meta,
        },
        "acceptance_metrics": {
            "eq31_target_sigma_over_sigma2": float(eq31_target),
            "fig4_Re_at_Delta_plus_0p03_ev": fig4_ref_value,
            "fig4_abs_error_to_eq31_target": abs(fig4_ref_value - float(eq31_target)),
            "fig5b_pauli_blocking_vertical_thresholds_ev": pauli_thresholds,
        },
        "outputs": {
            "png": str(out_dir / "hipolito2016_benchmark_suite.png"),
            "pdf": str(out_dir / "hipolito2016_benchmark_suite.pdf"),
            "data": str(out_dir / "hipolito2016_benchmark_data.npz"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "hipolito2016_benchmark_suite.png")


if __name__ == "__main__":
    main()
