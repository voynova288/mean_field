from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.htqg.bands import compute_bands_along_path, estimate_central_band_metrics
from mean_field.systems.htqg.domains import domain_displacements
from mean_field.systems.htqg.lattice import build_htqg_lattice, build_standard_kpath
from mean_field.systems.htqg.params import DEFAULT_THETA_DEG, HTQGParams

REPRESENTATIVE_DOMAINS = ("alpha_beta_alpha", "alpha_beta_gamma")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _domain_title(key: str) -> str:
    if key == "alpha_beta_alpha":
        return "Type-I αβα"
    if key == "alpha_beta_gamma":
        return "Type-II αβγ"
    return key


def _serialise_complex_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    return np.stack([arr.real, arr.imag], axis=-1)


def _save_domain_npz(path: Path, result, metrics: dict[str, float | None], *, domain_key: str) -> None:
    np.savez_compressed(
        path,
        domain=np.asarray(domain_key),
        kvec_complex_pairs=_serialise_complex_grid(result.path.kvec),
        kdist=np.asarray(result.path.kdist, dtype=float),
        energies_ev=np.asarray(result.energies, dtype=float),
        band_indices=np.asarray(result.band_indices, dtype=int),
        node_indices=np.asarray(result.path.node_indices, dtype=int),
        node_labels=np.asarray(result.path.labels),
        metrics_keys=np.asarray(tuple(metrics.keys())),
        metrics_values=np.asarray([np.nan if value is None else float(value) for value in metrics.values()], dtype=float),
    )


def _plot_combined(domain_payloads: dict[str, dict[str, Any]], output: Path, *, energy_window_ev: tuple[float, float]) -> None:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, len(domain_payloads), figsize=(10.5, 4.0), sharey=True)
    if len(domain_payloads) == 1:
        axes = [axes]
    scale = 1000.0
    for ax, (domain_key, payload) in zip(axes, domain_payloads.items(), strict=True):
        result = payload["result"]
        energies = np.asarray(result.energies, dtype=float) * scale
        for ib in range(energies.shape[1]):
            band_index = int(result.band_indices[ib]) if result.band_indices else ib
            is_central = band_index in {payload["matrix_dim"] // 2 - 1, payload["matrix_dim"] // 2}
            ax.plot(result.path.kdist, energies[:, ib], color="tab:red" if is_central else "black", lw=1.15 if is_central else 0.65)
        for node in result.path.nodes:
            ax.axvline(node.k_dist, color="0.86", lw=0.6, zorder=0)
        ax.axhline(0.0, color="0.82", lw=0.6, zorder=0)
        ax.set_xticks([node.k_dist for node in result.path.nodes])
        ax.set_xticklabels([node.label.replace("Gamma", "Γ").replace("kappa_prime", "κ'").replace("kappa", "κ") for node in result.path.nodes])
        ax.set_title(_domain_title(domain_key))
        ax.set_xlim(float(result.path.kdist[0]), float(result.path.kdist[-1]))
        ax.set_ylim(energy_window_ev[0] * scale, energy_window_ev[1] * scale)
        metrics = payload["metrics"]
        text_lines = []
        for key, label in (
            ("mean_flat_bandwidth_ev", "mean W"),
            ("central_gap_ev", "neutral gap"),
            ("remote_gap_ev", "remote gap"),
        ):
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                text_lines.append(f"{label}={1000.0 * float(value):.1f} meV")
        if text_lines:
            ax.text(0.03, 0.97, "\n".join(text_lines), transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    axes[0].set_ylabel("Energy [meV]")
    fig.suptitle("HTQG Fujimoto et al. Fig. 1 band targets - computed continuum bands", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _plot_single(domain_key: str, payload: dict[str, Any], output: Path, *, energy_window_ev: tuple[float, float]) -> None:
    _plot_combined({domain_key: payload}, output, energy_window_ev=energy_window_ev)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute HTQG Fujimoto-2025 Fig. 1(d,e) band panels.")
    parser.add_argument("--theta-deg", type=float, default=DEFAULT_THETA_DEG, help="Twist angle in degrees; paper default 2.25.")
    parser.add_argument("--n-shells", type=int, default=5, help="Hexagonal G-shell cutoff. Fig. 1 run should use at least 5.")
    parser.add_argument("--points-per-segment", type=int, default=60, help="K-path samples per segment.")
    parser.add_argument("--central-band-count", type=int, default=24, help="Number of central bands to compute/plot.")
    parser.add_argument("--energy-window-mev", type=float, default=150.0, help="Symmetric plotted energy window in meV.")
    parser.add_argument("--kappa", type=float, default=0.6, help="AA/AB tunneling ratio κ; paper Fig. 1 uses 0.6.")
    parser.add_argument("--include-realistic-ph-breaking", action="store_true", help="Enable Dirac rotations and paper MDT. Default keeps first-pass convention-locked PH-symmetric model.")
    parser.add_argument("--domains", nargs="+", default=list(REPRESENTATIVE_DOMAINS), help="Domain keys/aliases to compute. Default: αβα and αβγ representatives.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults under results/HTQG_Fujimoto2025_fig1/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if int(args.n_shells) < 5:
        raise SystemExit("Fig. 1 HTQG runs in this workflow require --n-shells >= 5.")
    if int(args.points_per_segment) <= 0:
        raise SystemExit("--points-per-segment must be positive.")
    if int(args.central_band_count) < 4:
        raise SystemExit("--central-band-count must be at least 4.")

    ensure_not_running_compute_on_login_node("HTQG Fig. 1 band diagonalization")

    params = (
        HTQGParams.realistic(kappa=float(args.kappa))
        if bool(args.include_realistic_ph_breaking)
        else HTQGParams.default(kappa=float(args.kappa), lambda_mdt_nm=0.0, include_dirac_rotation=False)
    )
    lattice = build_htqg_lattice(
        float(args.theta_deg),
        n_shells=int(args.n_shells),
        graphene_lattice_constant_nm=params.graphene_lattice_constant_nm,
    )
    path = build_standard_kpath(lattice, points_per_segment=int(args.points_per_segment))
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "HTQG_Fujimoto2025_fig1" / f"{_timestamp()}_theta{args.theta_deg:.4g}_kappa{args.kappa:.3g}_shell{args.n_shells}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[str, dict[str, Any]] = {}
    for requested_domain in args.domains:
        domain = domain_displacements(lattice, requested_domain)
        print(f"[htqg-fig1] computing domain={domain.key} label={domain.label} shell={lattice.n_shells} dim={lattice.matrix_dim}", flush=True)
        result = compute_bands_along_path(
            path,
            lattice,
            params,
            domain=domain,
            valley=1,
            central_band_count=int(args.central_band_count),
            return_eigenvectors=False,
        )
        metrics = estimate_central_band_metrics(result, lattice.matrix_dim)
        npz_path = output_dir / f"bands_{domain.key}.npz"
        _save_domain_npz(npz_path, result, metrics, domain_key=domain.key)
        payloads[domain.key] = {"result": result, "metrics": metrics, "matrix_dim": lattice.matrix_dim, "npz_path": str(npz_path), "domain": domain.to_dict()}
        _plot_single(domain.key, payloads[domain.key], output_dir / f"bands_{domain.key}.pdf", energy_window_ev=(-float(args.energy_window_mev) / 1000.0, float(args.energy_window_mev) / 1000.0))
        print(f"[htqg-fig1] metrics domain={domain.key}: {metrics}", flush=True)

    combined_pdf = output_dir / "fig1_de_htqg_bands_combined.pdf"
    _plot_combined(payloads, combined_pdf, energy_window_ev=(-float(args.energy_window_mev) / 1000.0, float(args.energy_window_mev) / 1000.0))

    summary = {
        "workflow": "run_htqg_fig1_bands",
        "status": "computed_first_pass",
        "caveat": "These are computed continuum bands for Fig. 1(d,e) targets. Paper-level reproduction still requires convention Gate-A and cutoff/path convergence checks.",
        "theta_deg": float(args.theta_deg),
        "kappa": float(args.kappa),
        "n_shells": int(args.n_shells),
        "N_G": int(lattice.n_g),
        "matrix_dim": int(lattice.matrix_dim),
        "points_per_segment": int(args.points_per_segment),
        "central_band_count": int(args.central_band_count),
        "energy_window_mev": float(args.energy_window_mev),
        "params": params.to_dict(),
        "lattice": lattice.to_summary_dict(),
        "path_labels": list(path.labels),
        "path_node_indices": [int(x) for x in path.node_indices],
        "outputs": {
            "combined_pdf": str(combined_pdf),
            "combined_png": str(combined_pdf.with_suffix(".png")),
        },
        "domains": {
            key: {
                "domain": payload["domain"],
                "metrics": payload["metrics"],
                "npz_path": payload["npz_path"],
                "pdf_path": str(output_dir / f"bands_{key}.pdf"),
                "png_path": str(output_dir / f"bands_{key}.png"),
            }
            for key, payload in payloads.items()
        },
        "paper_targets": {
            "alpha_beta_alpha": {"mean_flat_bandwidth_mev": 15.0, "remote_gap_mev": 70.0, "panel": "Fig. 1d"},
            "alpha_beta_gamma": {"conduction_bandwidth_mev": 12.0, "panel": "Fig. 1e"},
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[htqg-fig1] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
