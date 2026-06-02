from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import E_CHARGE_C, HBAR_J_S
from .response import (
    add_transitions_to_integral,
    parse_component,
    positive_transition_terms,
    precompute_response_tensors,
    sigma_from_integral,
)
from .slg_toy import (
    GappedSLGParams,
    _nearest_reciprocal_vectors,
    d2hdk,
    dhdk,
    diagonalize,
    hex_bz_vertices,
)

HIPOLITO_COMPONENTS = ("y;yy", "y;xx", "x;xy", "x;yx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the low-energy gapped-graphene benchmark behind "
            "Hipolito/Pedersen/Pereira PRB 94, 045434 Fig. 4/5 and compare it "
            "with Mao Appendix-A Fig. 10. This is a cross-reference evidence plot, "
            "not a visual fit."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/crossref_hipolito2016_fig4_evidence"))
    parser.add_argument("--mesh-radial", type=int, default=60, help="radial points per K/K' corner patch")
    parser.add_argument("--mesh-angular", type=int, default=120, help="angular points per K/K' corner patch")
    parser.add_argument("--patch-radius-nm-inv", type=float, default=1.2)
    parser.add_argument("--eta-mev", type=float, default=20.0, help="Smoothed broadening for the finite-cost K-patch evidence plot")
    parser.add_argument("--emin", type=float, default=0.05)
    parser.add_argument("--emax", type=float, default=0.8)
    parser.add_argument("--n-energy", type=int, default=301)
    parser.add_argument("--hipolito-delta-ev", type=float, default=0.2, help="Gap Delta in Hipolito notation; mass=Delta/2")
    parser.add_argument("--hipolito-hopping-ev", type=float, default=3.0)
    parser.add_argument(
        "--mao-data",
        type=Path,
        default=Path("results/shift_current_htg/slg_toy_fig10_reproduction_audit_m150/slg_toy_fig10_audit_data.npz"),
        help="Existing Mao Appendix-A audit data to include as the right panel.",
    )
    return parser.parse_args()


def sigma2_ua_nm_per_v2(params: GappedSLGParams) -> float:
    """Hipolito's sigma_2 = e^3 a/(4 gamma0 hbar), converted to uA nm/V^2."""

    gamma_j = float(params.hopping_ev) * E_CHARGE_C
    a_m = float(params.bond_nm) * 1.0e-9
    sigma2_si = E_CHARGE_C**3 * a_m / (4.0 * gamma_j * HBAR_J_S)  # S m / V
    return float(sigma2_si * 1.0e15)


def in_first_hex_bz(k_xy: np.ndarray, nearest_g: np.ndarray) -> bool:
    bounds = 0.5 * np.sum(nearest_g * nearest_g, axis=1)
    return bool(np.all(np.asarray(k_xy, dtype=float) @ nearest_g.T <= bounds + 1.0e-10))


def k_corner_patch_grid(params: GappedSLGParams, *, radius: float, n_radial: int, n_angular: int) -> tuple[np.ndarray, np.ndarray]:
    """Patch grid around all six hexagonal BZ corners.

    Each BZ corner contributes its inside-hexagon 120-degree sector.  The six
    sectors together represent two full Dirac valleys.  This resolves the
    K/K' direct-gap onset much better than a uniform rectangular midpoint grid
    at the same cost.
    """

    if int(n_radial) <= 0 or int(n_angular) <= 0:
        raise ValueError("n_radial and n_angular must be positive")
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    vertices = hex_bz_vertices(params)
    nearest_g = _nearest_reciprocal_vectors(params)
    points: list[np.ndarray] = []
    weights: list[float] = []
    dr = radius / float(n_radial)
    dtheta = 2.0 * math.pi / float(n_angular)
    for vertex in vertices:
        for ir in range(int(n_radial)):
            rr = (ir + 0.5) * dr
            for it in range(int(n_angular)):
                theta = (it + 0.5) * dtheta
                k_xy = vertex + rr * np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
                if not in_first_hex_bz(k_xy, nearest_g):
                    continue
                points.append(np.asarray(k_xy, dtype=float))
                weights.append(float(rr * dr * dtheta))
    if not points:
        raise RuntimeError("K-corner patch grid selected no points")
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def compute_patch_spectra(
    photon_energies: np.ndarray,
    *,
    params: GappedSLGParams,
    eta_ev: float,
    radius: float,
    n_radial: int,
    n_angular: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parsed = {name: parse_component(name) for name in HIPOLITO_COMPONENTS}
    integrals = {name: np.zeros_like(photon_energies, dtype=np.complex128) for name in parsed}
    k_points, k_weights = k_corner_patch_grid(params, radius=radius, n_radial=n_radial, n_angular=n_angular)
    skipped = 0
    for k_xy, k_weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            d2hdk=d2hdk(k_xy, params),
            denominator_cutoff_ev=1.0e-10,
        )
        skipped += int(tensors.skipped_small_denominators)
        for name, component in parsed.items():
            transitions, weights = positive_transition_terms(tensors, component)
            add_transitions_to_integral(
                integrals[name],
                photon_energies,
                transitions,
                weights,
                k_weight_nm_inv_sq=float(k_weight),
                eta_ev=float(eta_ev),
            )
    spectra = {name: sigma_from_integral(integral) for name, integral in integrals.items()}
    meta = {
        "n_k_points": int(k_points.shape[0]),
        "patch_area_nm_inv_sq": float(np.sum(k_weights)),
        "skipped_small_denominators": int(skipped),
    }
    return spectra, meta


def main() -> None:
    args = parse_args()
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = float(args.eta_mev) * 1.0e-3
    hip_params = GappedSLGParams(
        hopping_ev=float(args.hipolito_hopping_ev),
        mass_ev=0.5 * float(args.hipolito_delta_ev),
    )
    spectra, patch_meta = compute_patch_spectra(
        photon,
        params=hip_params,
        eta_ev=eta_ev,
        radius=float(args.patch_radius_nm_inv),
        n_radial=int(args.mesh_radial),
        n_angular=int(args.mesh_angular),
    )
    sigma2 = sigma2_ua_nm_per_v2(hip_params)
    normalized = {name: values / sigma2 for name, values in spectra.items()}

    # Symmetry-related curves that should overlap Hipolito's Fig. 4 real-part convention.
    overlap_curves = {
        "sigma^{y;yy}": normalized["y;yy"],
        "-sigma^{y;xx}": -normalized["y;xx"],
        "-sigma^{x;xy}": -normalized["x;xy"],
        "-sigma^{x;yx}": -normalized["x;yx"],
    }
    ref = overlap_curves["sigma^{y;yy}"]
    overlap_error = float(max(np.max(np.abs(values - ref)) for values in overlap_curves.values()))

    peaks = {
        name: {
            "min_normalized_sigma_over_sigma2": float(np.min(values)),
            "max_abs_normalized_sigma_over_sigma2": float(np.max(np.abs(values))),
            "energy_at_max_abs_ev": float(photon[int(np.argmax(np.abs(values)))]),
        }
        for name, values in normalized.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "hipolito2016_fig4_crossref_data.npz",
        photon_energies_ev=photon,
        sigma2_uA_nm_per_V2=sigma2,
        **{f"sigma_{name.replace(';', '_')}": values for name, values in spectra.items()},
        **{f"normalized_{name.replace(';', '_')}": values for name, values in normalized.items()},
    )

    summary: dict[str, object] = {
        "purpose": "Cross-reference evidence: formal W-corrected gapped graphene has a K/K' direct-gap onset as in Hipolito 2016 Fig. 4/5, unlike Mao Fig. 10's M-only-looking benchmark.",
        "hipolito_params": {
            "hopping_ev": float(hip_params.hopping_ev),
            "delta_ev": float(args.hipolito_delta_ev),
            "mass_ev": float(hip_params.mass_ev),
            "eta_mev": float(args.eta_mev),
            "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
            "mesh_radial": int(args.mesh_radial),
            "mesh_angular": int(args.mesh_angular),
            "sigma2_uA_nm_per_V2": float(sigma2),
        },
        "patch_grid": patch_meta,
        "overlap_error_for_Hipolito_Fig4_real_component_relations_sigma_over_sigma2": overlap_error,
        "peaks": peaks,
    }

    mao_data_exists = args.mao_data.exists()
    if mao_data_exists:
        mao = np.load(args.mao_data)
        mao_e = np.asarray(mao["photon_energies_ev"], dtype=float)
        mao_formal = np.asarray(mao["formal_x_xy"], dtype=float)
        mao_printed = np.asarray(mao["printed_eq4_primitive_x_xy"], dtype=float)
        summary["mao_audit_panel"] = {
            "source": str(args.mao_data),
            "formal_peak_energy_ev": float(mao_e[int(np.argmax(np.abs(mao_formal)))]),
            "formal_peak_uA_nm_per_V2": float(mao_formal[int(np.argmax(np.abs(mao_formal)))]),
            "printed_eq4_primitive_peak_energy_ev": float(mao_e[int(np.argmax(np.abs(mao_printed)))]),
            "printed_eq4_primitive_peak_uA_nm_per_V2": float(mao_printed[int(np.argmax(np.abs(mao_printed)))]),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2 if mao_data_exists else 1, figsize=(10.0 if mao_data_exists else 5.0, 3.8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    colors = ["#111111", "#e41a1c", "#377eb8", "#4daf4a"]
    for (label, values), color in zip(overlap_curves.items(), colors, strict=True):
        ax.plot(photon, values, lw=1.4, label=label, color=color)
    ax.axvline(float(args.hipolito_delta_ev), color="0.35", ls="--", lw=1.0, label=r"$\Delta$")
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"Re $\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Hipolito 2016 Fig.4/5-style\nK/K' direct-gap onset")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.5)

    if mao_data_exists:
        ax = axes[1]
        mao = np.load(args.mao_data)
        mao_e = np.asarray(mao["photon_energies_ev"], dtype=float)
        mao_formal = np.asarray(mao["formal_x_xy"], dtype=float)
        mao_printed = np.asarray(mao["printed_eq4_primitive_x_xy"], dtype=float)
        mao_t = 2.73
        mao_m = 1.5
        mao_delta = 2.0 * mao_m
        mao_m_transition = 2.0 * math.sqrt(mao_t * mao_t + mao_m * mao_m)
        ax.plot(mao_e, mao_formal, color="#0047ff", lw=1.7, label="formal W-corrected")
        ax.plot(mao_e, mao_printed, color="#888888", lw=1.3, ls="--", label="printed Eq.(4)-only diagnostic")
        ax.axvline(mao_delta, color="#0047ff", ls=":", lw=1.1, label=r"$2m=3$ eV")
        ax.axvline(mao_m_transition, color="#888888", ls=":", lw=1.1, label="M transition")
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.set_xlim(0.0, 8.0)
        ax.set_xlabel(r"$\hbar\omega$ [eV]")
        ax.set_ylabel(r"$\sigma^{x;xy}$ [$\mu$A nm V$^{-2}$]")
        ax.set_title("Mao Appendix-A toy audit\nformal result peaks at K/K'")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2, lw=0.5)

    fig.savefig(args.output_dir / "hipolito2016_crossref_evidence.png", dpi=180)
    fig.savefig(args.output_dir / "hipolito2016_crossref_evidence.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_dir / "hipolito2016_crossref_evidence.png")


if __name__ == "__main__":
    main()
