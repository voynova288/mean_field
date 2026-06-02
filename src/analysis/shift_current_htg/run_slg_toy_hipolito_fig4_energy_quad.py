from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from PIL import Image

from .response import precompute_response_tensors
from .slg_toy import GappedSLGParams, _nearest_reciprocal_vectors, d2hdk, dhdk, diagonalize, hex_bz_vertices
from .run_slg_toy_hipolito_fig4 import crop_reference_fig4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resonance-resolved Hipolito Fig. 4 calculation.  This integrates the radial "
            "direction in transition-energy variables instead of using a fixed k mesh, "
            "eliminating narrow-Gamma quadrature wiggles."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/crossref_hipolito2016_fig4_energy_quad"))
    parser.add_argument("--gamma-mev", type=float, default=1.0)
    parser.add_argument("--delta-ev", type=float, default=0.2)
    parser.add_argument("--hopping-ev", type=float, default=3.0)
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--mu-ev", type=float, default=0.0)  # kept for provenance; the implemented gap case uses T~0 occupations
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.8)
    parser.add_argument("--n-photon", type=int, default=321)
    parser.add_argument("--theta-count", type=int, default=72)
    parser.add_argument("--transition-energy-nodes", type=int, default=900)
    parser.add_argument("--transition-emax", type=float, default=1.2, help="Upper transition energy included; keep above photon emax to avoid endpoint artifacts")
    parser.add_argument("--patch-radius-nm-inv", type=float, default=1.4)
    parser.add_argument("--reference-offset-ev", type=float, default=0.03)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    return parser.parse_args()


def inside_hex(k_xy: np.ndarray, nearest_g: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(np.all(np.asarray(k_xy, dtype=float) @ nearest_g.T <= bounds + 1.0e-10))


def transition_energy(k_xy: np.ndarray, params: GappedSLGParams) -> float:
    evals, _ = diagonalize(k_xy, params)
    return float(evals[1] - evals[0])


def ray_rmax(vertex: np.ndarray, direction: np.ndarray, *, params: GappedSLGParams, nearest_g: np.ndarray, bounds: np.ndarray, patch_radius: float) -> float:
    if not inside_hex(vertex + 1.0e-8 * direction, nearest_g, bounds):
        return 0.0
    hi = float(patch_radius)
    if inside_hex(vertex + hi * direction, nearest_g, bounds):
        return hi
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if inside_hex(vertex + mid * direction, nearest_g, bounds):
            lo = mid
        else:
            hi = mid
    return lo


def radius_for_transition_energy(vertex: np.ndarray, direction: np.ndarray, *, params: GappedSLGParams, rmax: float, target_ev: float) -> float:
    lo = 0.0
    hi = float(rmax)
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if transition_energy(vertex + mid * direction, params) < target_ev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eq25b_yyy_energy_quadrature(
    photon_ev: np.ndarray,
    *,
    params: GappedSLGParams,
    gamma_ev: float,
    theta_count: int,
    transition_energy_nodes: int,
    patch_radius: float,
    transition_energy_max_ev: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    vertices = hex_bz_vertices(params)
    nearest_g = _nearest_reciprocal_vectors(params)
    bounds = 0.5 * np.sum(nearest_g * nearest_g, axis=1)
    xg, wg = leggauss(int(transition_energy_nodes))
    delta_ev = 2.0 * float(params.mass_ev)

    raw = np.zeros_like(photon_ev, dtype=np.complex128)
    count = 0
    energy_area = 0.0
    for vertex in vertices:
        for itheta in range(int(theta_count)):
            theta = (itheta + 0.5) * 2.0 * math.pi / float(theta_count)
            dtheta = 2.0 * math.pi / float(theta_count)
            direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
            rmax = ray_rmax(vertex, direction, params=params, nearest_g=nearest_g, bounds=bounds, patch_radius=patch_radius)
            if rmax <= 0.0:
                continue
            e_hi = min(float(transition_energy_max_ev), transition_energy(vertex + rmax * direction, params))
            if e_hi <= delta_ev + 1.0e-10:
                continue
            e_nodes = 0.5 * (e_hi - delta_ev) * xg + 0.5 * (e_hi + delta_ev)
            e_weights = 0.5 * (e_hi - delta_ev) * wg
            for transition_ev, transition_weight in zip(e_nodes, e_weights, strict=True):
                radius = radius_for_transition_energy(
                    vertex,
                    direction,
                    params=params,
                    rmax=rmax,
                    target_ev=float(transition_ev),
                )
                k_xy = vertex + radius * direction
                evals, evecs = diagonalize(k_xy, params)
                tensors = precompute_response_tensors(
                    evals,
                    evecs,
                    dhdk(k_xy, params),
                    d2hdk=d2hdk(k_xy, params),
                    temperature_k=1.0,
                    denominator_cutoff_ev=1.0e-10,
                )
                D = tensors.D
                r = tensors.r
                gd = tensors.r_covariant
                occ = tensors.occupations
                radial_D = direction[0] * D[0] + direction[1] * D[1]
                d_transition_dr = float(np.real(radial_D[1, 1] - radial_D[0, 0]))
                if d_transition_dr <= 0.0:
                    continue
                # d^2 k = r dr dtheta = r |dE/dr|^{-1} dE dtheta.
                k_weight = float(radius / d_transition_dr * transition_weight * dtheta)
                energy_area += k_weight

                # Component sigma^{y;yy}; Hipolito's nonzero partners follow by C3v.
                lam = alpha = beta = 1
                for m, n in ((1, 0), (0, 1)):
                    delta_mn = float(evals[m] - evals[n])
                    f_nm = float(occ[n] - occ[m])
                    if abs(f_nm) < 1.0e-14:
                        continue
                    den = photon_ev - delta_mn + 1.0j * float(gamma_ev)
                    partial_beta_delta = D[beta, m, m] - D[beta, n, n]
                    cov_derivative = f_nm * (
                        1.0j * gd[beta, alpha, m, n] / den
                        + 1.0j * r[alpha, m, n] * partial_beta_delta / (den * den)
                    )
                    raw += (
                        k_weight
                        / (2.0 * math.pi) ** 2
                        * (-D[lam, n, m])
                        / (-delta_mn + 2.0j * float(gamma_ev))
                        * cov_derivative
                    )
                count += 1
    meta = {"quadrature_points": int(count), "energy_variable_area_nm_inv_sq": float(energy_area)}
    return raw, meta


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * float(args.delta_ev))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_photon), dtype=float)
    raw, meta = eq25b_yyy_energy_quadrature(
        photon,
        params=params,
        gamma_ev=float(args.gamma_mev) * 1.0e-3,
        theta_count=int(args.theta_count),
        transition_energy_nodes=int(args.transition_energy_nodes),
        patch_radius=float(args.patch_radius_nm_inv),
        transition_energy_max_ev=float(args.transition_emax),
    )

    # Direct Eq.(25b) prefactor in our D=<dH/dk> units; remaining global
    # convention fixed by Hipolito Eq.(31), not by visual matching.
    eq25_scale = -2.0 * float(params.hopping_ev) / float(params.bond_nm)
    delta_dimensionless = float(args.delta_ev) / float(params.hopping_ev)
    target = -1.0 / (4.0 * delta_dimensionless)
    idx = int(np.argmin(np.abs(photon - (float(args.delta_ev) + float(args.reference_offset_ev)))))
    scale = eq25_scale * target / float((eq25_scale * raw)[idx].real)
    sigma_over_sigma2 = scale * raw

    np.savez(
        output_dir / "hipolito2016_fig4_energy_quad_data.npz",
        photon_energies_ev=photon,
        sigma_over_sigma2=sigma_over_sigma2,
        raw=raw,
    )
    crop_path = output_dir / "hipolito2016_fig4_reference_crop.png"
    have_crop = crop_reference_fig4(args.reference_page, crop_path)

    fig, axes = plt.subplots(1, 2 if have_crop else 1, figsize=(10.5 if have_crop else 5.2, 4.0), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    ax.plot(photon, sigma_over_sigma2.real, lw=1.8, color="#003c4c", label="Re")
    ax.plot(photon, sigma_over_sigma2.imag, lw=1.4, color="#e41a1c", label="Im")
    ax.plot(
        [float(args.delta_ev), min(float(args.emax), float(args.delta_ev) + 0.18)],
        [target, target],
        ls="--",
        lw=1.0,
        color="#003c4c",
        label="Eq.(31) threshold",
    )
    ax.axvline(float(args.delta_ev), color="0.35", ls="--", lw=1.0, label=r"$\Delta$")
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Hipolito Fig. 4, energy-quadrature")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(frameon=False, fontsize=7)
    if have_crop:
        axes[1].imshow(Image.open(crop_path))
        axes[1].axis("off")
        axes[1].set_title("Hipolito 2016 Fig. 4 crop")
    fig.savefig(output_dir / "hipolito2016_fig4_energy_quad.png", dpi=180)
    fig.savefig(output_dir / "hipolito2016_fig4_energy_quad.pdf")
    plt.close(fig)

    summary = {
        "purpose": "Remove narrow-Gamma k-grid wiggles by integrating radial direction in transition-energy variables.",
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Fig. 4",
        "parameters": {
            "gamma0_ev": float(params.hopping_ev),
            "Delta_ev": float(args.delta_ev),
            "Gamma_mev": float(args.gamma_mev),
            "theta_count": int(args.theta_count),
            "transition_energy_nodes": int(args.transition_energy_nodes),
            "transition_emax_ev": float(args.transition_emax),
            "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
        },
        "eq31_target_sigma_over_sigma2": float(target),
        "scale_applied_to_raw_integral": float(scale),
        **meta,
        "outputs": {
            "png": str(output_dir / "hipolito2016_fig4_energy_quad.png"),
            "pdf": str(output_dir / "hipolito2016_fig4_energy_quad.pdf"),
            "data": str(output_dir / "hipolito2016_fig4_energy_quad_data.npz"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_dir / "hipolito2016_fig4_energy_quad.png")


if __name__ == "__main__":
    main()
