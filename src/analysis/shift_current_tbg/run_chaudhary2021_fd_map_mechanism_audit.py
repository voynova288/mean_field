from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair,
    hamiltonian_gauge_data,
    shift_integrand_from_pair_generalized_derivative,
)
from analysis.shift_current_htg.constants import eta_mev_to_ev
from analysis.shift_current_htg.response import fermi_occupation, parse_component
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    build_chau_b0_hamiltonian,
    centered_flat_indices,
    config_summary,
    finite_difference_b0_dhdk,
    make_b0_parameters,
)
from .run_chaudhary2021_integrand_maps import _ws_hex_points
from .run_chaudhary2021_noninteracting import _parse_float_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanism audit for Chaudhary Fig. 2(e) FD k-space pocket: decompose the "
            "map into Pauli mask, transition-energy resonance, and quantum-geometry R."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=7)
    parser.add_argument("--mesh-size", type=int, default=55)
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--component", default="y;xx")
    parser.add_argument("--mu-mev", type=_parse_float_csv, default=(-30.0, 30.0))
    parser.add_argument("--target-energy-mev", type=float, default=21.6)
    parser.add_argument("--eta-mev-list", type=_parse_float_csv, default=(2.0, 5.0, 10.0, 20.0))
    parser.add_argument("--plot-eta-mev", type=float, default=10.0)
    parser.add_argument(
        "--principal-value-eta-mev",
        type=float,
        default=0.0,
        help="WannierBerri/Wannier90 sc_eta principal-value regularizer for intermediate denominators; 0 uses exact denominators.",
    )
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_fd_map_mechanism_audit"))
    return parser.parse_args()


def _safe_key(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _plot_scatter_grid(out: Path, kpts: np.ndarray, params, data: dict[str, np.ndarray], *, title: str) -> None:
    gscale = abs(complex(params.g1))
    panels = [
        ("hole_occ_mu_m30", r"hole Pauli mask $f_{Dv}-f_{Fv}$"),
        ("hole_transition_mev", r"hole transition $E_{Fv}-E_{Dv}$ (meV)"),
        ("hole_R_nm3", r"hole $R$ (nm$^3$)"),
        ("hole_R_delta_plot", r"hole $R\delta_\eta$"),
        ("electron_occ_mu_p30", r"electron Pauli mask $f_{Fc}-f_{Dc}$"),
        ("electron_transition_mev", r"electron transition $E_{Dc}-E_{Fc}$ (meV)"),
        ("electron_R_nm3", r"electron $R$ (nm$^3$)"),
        ("electron_R_delta_plot", r"electron $R\delta_\eta$"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), panels, strict=True):
        vals = np.asarray(data[key], dtype=float)
        finite = vals[np.isfinite(vals)]
        if "transition" in key:
            vmin = float(np.nanpercentile(finite, 2.0)) if finite.size else None
            vmax = float(np.nanpercentile(finite, 98.0)) if finite.size else None
            cmap = "viridis"
        elif "occ" in key:
            vmin, vmax, cmap = -0.05, 1.05, "gray_r"
        else:
            vmax_abs = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
            if not np.isfinite(vmax_abs) or vmax_abs <= 0.0:
                vmax_abs = 1.0
            vmin, vmax, cmap = -vmax_abs, vmax_abs, "coolwarm"
        sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=vals, s=9, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x/G$")
        ax.set_ylabel(r"$k_y/G$")
        ax.set_title(label, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.78)
    fig.suptitle(title)
    fig.savefig(out / "fd_map_mechanism.png", dpi=220)
    fig.savefig(out / "fd_map_mechanism.pdf")
    plt.close(fig)


def _lorentzian_at_transition(target_ev: float, transition_ev: np.ndarray, eta_ev: float) -> np.ndarray:
    transition = np.asarray(transition_ev, dtype=float)
    eta = float(eta_ev)
    return (eta / np.pi) / ((float(target_ev) - transition) ** 2 + eta * eta)


def _area_metrics(values: np.ndarray, kpts: np.ndarray, threshold_fraction: float) -> dict[str, object]:
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return {"n_points": 0, "fraction": 0.0, "centroid_dimless": [float("nan"), float("nan")], "rms_radius_dimless": float("nan")}
    abs_vals = np.abs(vals)
    vmax = float(np.nanmax(abs_vals[finite]))
    mask = finite & (abs_vals >= float(threshold_fraction) * vmax) & (abs_vals > 0.0)
    if not np.any(mask):
        return {"n_points": 0, "fraction": 0.0, "centroid_dimless": [float("nan"), float("nan")], "rms_radius_dimless": float("nan")}
    pts = np.asarray(kpts[mask], dtype=np.complex128)
    centroid = complex(np.mean(pts))
    radius = float(np.sqrt(np.mean(np.abs(pts - centroid) ** 2)))
    return {
        "n_points": int(np.count_nonzero(mask)),
        "fraction": float(np.count_nonzero(mask) / vals.size),
        "centroid_dimless": [float(centroid.real), float(centroid.imag)],
        "rms_radius_dimless": radius,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sigma_rotation = not bool(args.no_sigma_rotation)
    periodic_g_grid = bool(args.periodic_g_grid)
    config = ChaudharyTBGConfig(
        theta_deg=float(args.theta_deg),
        kinetic_ev=float(args.kinetic_ev),
        w_ab_ev=float(args.w_ab_mev) * 1.0e-3,
        w_aa_ratio=float(args.w_aa_ratio),
        delta1_ev=float(args.delta1_mev) * 1.0e-3,
        delta2_ev=float(args.delta2_mev) * 1.0e-3,
        valley=int(args.valley),
    )
    params = make_b0_parameters(config)
    lg = int(args.lg)
    dim = 4 * lg * lg
    v_flat, c_flat = centered_flat_indices(dim)
    hole_pair = (v_flat - 1, v_flat)  # Dv -> Fv
    electron_pair = (c_flat, c_flat + 1)  # Fc -> Dc
    component = parse_component(str(args.component))
    deriv_axis, optical_axis, optical_axis_2 = component
    if optical_axis != optical_axis_2:
        raise ValueError("This audit expects a linearly polarized component with b=c")

    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if periodic_g_grid else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, lg, int(config.valley))
    dhdk = finite_difference_b0_dhdk(
        params,
        config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        step_dimless=float(args.finite_step_dimless),
    )
    dhdk_stack = np.stack(dhdk, axis=0)
    kpts = _ws_hex_points(params, int(args.mesh_size))
    n_k = int(kpts.size)
    target_ev = float(args.target_energy_mev) * 1.0e-3
    plot_eta_ev = eta_mev_to_ev(float(args.plot_eta_mev))
    pv_eta_ev = eta_mev_to_ev(float(args.principal_value_eta_mev)) if float(args.principal_value_eta_mev) > 0.0 else None
    mu_values_ev = tuple(float(x) * 1.0e-3 for x in args.mu_mev)

    hole_transition = np.full(n_k, np.nan, dtype=float)
    electron_transition = np.full(n_k, np.nan, dtype=float)
    hole_R = np.full(n_k, np.nan, dtype=float)
    electron_R = np.full(n_k, np.nan, dtype=float)
    hole_fv_energy = np.full(n_k, np.nan, dtype=float)
    electron_fc_energy = np.full(n_k, np.nan, dtype=float)
    occ_by_mu = {f"{mu_mev:g}": np.zeros((2, n_k), dtype=float) for mu_mev in args.mu_mev}  # [hole,electron]

    skipped = 0
    for ik, k in enumerate(kpts):
        hmat = build_chau_b0_hamiltonian(
            complex(k),
            params,
            config,
            lg=lg,
            sigma_rotation=sigma_rotation,
            periodic_g_grid=periodic_g_grid,
            gvec=gvec,
            tunnel=tunnel,
        )
        evals, evecs = eigh(hmat)
        gauge_data = hamiltonian_gauge_data(evals, evecs, dhdk_stack, denominator_cutoff=float(args.denominator_cutoff_ev))
        for ipair, (pair, trans_out, R_out) in enumerate(((hole_pair, hole_transition, hole_R), (electron_pair, electron_transition, electron_R))):
            n_abs, m_abs = pair
            trans_out[ik] = float(evals[m_abs] - evals[n_abs])
            if trans_out[ik] > 0.0:
                gd = berry_connection_generalized_derivative_pair(
                    gauge_data.velocity_h,
                    gauge_data.energies,
                    n_abs,
                    m_abs,
                    denominator_cutoff=float(args.denominator_cutoff_ev),
                    principal_value_eta=pv_eta_ev,
                )
                skipped += int(gd.skipped_small_denominators)
                R_out[ik] = shift_integrand_from_pair_generalized_derivative(
                    gauge_data.berry_connection,
                    gd.values,
                    initial_band=n_abs,
                    final_band=m_abs,
                    deriv_axis=deriv_axis,
                    optical_axis=optical_axis,
                )
        hole_fv_energy[ik] = float(evals[v_flat])
        electron_fc_energy[ik] = float(evals[c_flat])
        for mu_mev, mu_ev in zip(args.mu_mev, mu_values_ev, strict=True):
            occ = fermi_occupation(evals, mu_ev=float(mu_ev), temperature_k=0.0)
            occ_by_mu[f"{mu_mev:g}"][0, ik] = float(occ[hole_pair[0]] - occ[hole_pair[1]])
            occ_by_mu[f"{mu_mev:g}"][1, ik] = float(occ[electron_pair[0]] - occ[electron_pair[1]])

    hole_delta_plot = _lorentzian_at_transition(target_ev, hole_transition, plot_eta_ev)
    electron_delta_plot = _lorentzian_at_transition(target_ev, electron_transition, plot_eta_ev)
    mu_hole_key = f"{float(args.mu_mev[0]):g}"
    mu_electron_key = f"{float(args.mu_mev[-1]):g}"
    hole_map_plot = occ_by_mu[mu_hole_key][0] * hole_R * hole_delta_plot
    electron_map_plot = occ_by_mu[mu_electron_key][1] * electron_R * electron_delta_plot

    arrays = {
        "kpts_dimless": kpts,
        "hole_transition_mev": 1.0e3 * hole_transition,
        "electron_transition_mev": 1.0e3 * electron_transition,
        "hole_R_nm3": hole_R,
        "electron_R_nm3": electron_R,
        "hole_fv_energy_mev": 1.0e3 * hole_fv_energy,
        "electron_fc_energy_mev": 1.0e3 * electron_fc_energy,
        "hole_R_delta_plot": hole_map_plot,
        "electron_R_delta_plot": electron_map_plot,
        "hole_occ_mu_m30": occ_by_mu[mu_hole_key][0],
        "electron_occ_mu_p30": occ_by_mu[mu_electron_key][1],
    }
    for mu_mev in args.mu_mev:
        key = f"{float(mu_mev):g}"
        arrays[f"hole_occ_mu_{_safe_key(float(mu_mev))}"] = occ_by_mu[key][0]
        arrays[f"electron_occ_mu_{_safe_key(float(mu_mev))}"] = occ_by_mu[key][1]
    np.savez_compressed(out / "fd_map_mechanism.npz", **arrays)

    eta_rows = []
    for eta_mev in tuple(float(x) for x in args.eta_mev_list):
        eta_ev = eta_mev_to_ev(eta_mev)
        hole_map = occ_by_mu[mu_hole_key][0] * hole_R * _lorentzian_at_transition(target_ev, hole_transition, eta_ev)
        electron_map = occ_by_mu[mu_electron_key][1] * electron_R * _lorentzian_at_transition(target_ev, electron_transition, eta_ev)
        for name, vals in (("hole", hole_map), ("electron", electron_map)):
            idx = int(np.nanargmax(np.abs(vals))) if vals.size else 0
            eta_rows.append(
                {
                    "eta_mev": float(eta_mev),
                    "channel": name,
                    "max_value_nm3_per_ev": float(vals[idx]),
                    "max_abs_k_dimless": [float(kpts[idx].real), float(kpts[idx].imag)],
                    "halfmax_area": _area_metrics(vals, kpts, 0.5),
                    "tenthmax_area": _area_metrics(vals, kpts, 0.1),
                }
            )

    def transition_stats(name: str, transition_mev: np.ndarray, R: np.ndarray, occ: np.ndarray) -> dict[str, object]:
        active = np.asarray(occ, dtype=float) > 0.5
        resonant = np.abs(np.asarray(transition_mev, dtype=float) - float(args.target_energy_mev)) <= float(args.plot_eta_mev)
        both = active & resonant
        return {
            "channel": name,
            "transition_mev_min_p05_p50_p95_max": [float(x) for x in np.nanpercentile(transition_mev, [0, 5, 50, 95, 100])],
            "R_nm3_p05_p50_p95": [float(x) for x in np.nanpercentile(R[np.isfinite(R)], [5, 50, 95])],
            "active_fraction": float(np.count_nonzero(active) / n_k),
            "resonant_fraction_within_plot_eta": float(np.count_nonzero(resonant) / n_k),
            "active_and_resonant_fraction": float(np.count_nonzero(both) / n_k),
        }

    summary = {
        "status": "FD Fig. 2(e) mechanism audit: separates Pauli mask, transition-energy resonance, and R quantum geometry for nearest same-side FD pairs.",
        "config": config_summary(config, b0_params=params, lg=lg),
        "run": {
            "component": str(args.component),
            "component_indices": [int(x) for x in component],
            "lg": int(lg),
            "mesh_size": int(args.mesh_size),
            "n_hex_points": int(n_k),
            "hole_pair_indices": [int(x) for x in hole_pair],
            "electron_pair_indices": [int(x) for x in electron_pair],
            "mu_mev": [float(x) for x in args.mu_mev],
            "target_energy_mev": float(args.target_energy_mev),
            "plot_eta_mev": float(args.plot_eta_mev),
            "principal_value_eta_mev": float(args.principal_value_eta_mev),
            "eta_mev_list": [float(x) for x in args.eta_mev_list],
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
            "skipped_denominators": int(skipped),
        },
        "transition_stats": [
            transition_stats("hole_Dv_to_Fv", 1.0e3 * hole_transition, hole_R, occ_by_mu[mu_hole_key][0]),
            transition_stats("electron_Fc_to_Dc", 1.0e3 * electron_transition, electron_R, occ_by_mu[mu_electron_key][1]),
        ],
        "eta_area_rows": eta_rows,
        "interpretation_hint": (
            "If the Pauli mask and transition resonance are broad but R*delta halfmax remains tiny, the pocket-size mismatch is dominated by quantum-geometry/R localization. "
            "If the active_and_resonant_fraction is tiny, it is instead a band-energy or chemical-potential convention issue."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_scatter_grid(
        out,
        kpts,
        params,
        arrays,
        title=(
            f"Chaudhary FD map mechanism, valley={config.valley}, target={args.target_energy_mev:g} meV, "
            f"eta={args.plot_eta_mev:g} meV, PV eta={args.principal_value_eta_mev:g} meV"
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
