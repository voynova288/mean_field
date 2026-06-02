from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import hamiltonian_gauge_data, wannierberri_shift_current_internal_imn
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WannierBerri ShiftCurrentFormula port audit for Chaudhary Fig. 2(e) FD maps."
    )
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=5)
    parser.add_argument("--mesh-size", type=int, default=35)
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--component", default="y;xx")
    parser.add_argument("--target-energy-mev", type=float, default=21.6)
    parser.add_argument("--eta-mev", type=float, default=10.0, help="spectral Lorentzian broadening")
    parser.add_argument("--sc-eta-mev", type=float, default=40.0, help="WannierBerri/Wannier90 shift-current principal-value eta")
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-10)
    parser.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_wannierberri_fd_group_audit"))
    return parser.parse_args()


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
    return {
        "n_points": int(np.count_nonzero(mask)),
        "fraction": float(np.count_nonzero(mask) / vals.size),
        "centroid_dimless": [float(centroid.real), float(centroid.imag)],
        "rms_radius_dimless": float(np.sqrt(np.mean(np.abs(pts - centroid) ** 2))),
    }


def _plot(out: Path, kpts: np.ndarray, params, maps: dict[str, np.ndarray], *, title: str) -> None:
    gscale = abs(complex(params.g1))
    panels = [
        ("hole_pair1", "hole D-1->Fv"),
        ("hole_group2", "hole {D-2,D-1}->Fv"),
        ("electron_pair1", "electron Fc->D+1"),
        ("electron_group2", "electron Fc->{D+1,D+2}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8), constrained_layout=True)
    for ax, (key, label) in zip(axes, panels, strict=True):
        vals = np.asarray(maps[key], dtype=float)
        finite = vals[np.isfinite(vals)]
        vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=vals, s=10, cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0.0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x/G$")
        ax.set_ylabel(r"$k_y/G$")
        ax.set_title(label, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.75)
    fig.suptitle(title)
    fig.savefig(out / "wannierberri_fd_group_maps.png", dpi=220)
    fig.savefig(out / "wannierberri_fd_group_maps.pdf")
    plt.close(fig)


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
    component = parse_component(str(args.component))
    a, b, c = component
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
    kpts = _ws_hex_points(params, int(args.mesh_size))
    target_ev = float(args.target_energy_mev) * 1.0e-3
    eta_ev = eta_mev_to_ev(float(args.eta_mev))
    sc_eta_ev = eta_mev_to_ev(float(args.sc_eta_mev))

    pair_defs = {
        "hole_pair1": ((v_flat - 1, v_flat), -30.0),
        "hole_group2": ((v_flat - 2, v_flat), (v_flat - 1, v_flat), -30.0),
        "electron_pair1": ((c_flat, c_flat + 1), 30.0),
        "electron_group2": ((c_flat, c_flat + 1), (c_flat, c_flat + 2), 30.0),
    }
    maps = {key: np.zeros(kpts.size, dtype=float) for key in pair_defs}
    transitions = {key: np.full(kpts.size, np.nan, dtype=float) for key in pair_defs}

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
        data = hamiltonian_gauge_data(evals, evecs, np.stack(dhdk, axis=0), denominator_cutoff=float(args.denominator_cutoff_ev))
        imn = wannierberri_shift_current_internal_imn(
            data.velocity_h,
            data.energies,
            sc_eta=sc_eta_ev,
            denominator_cutoff=float(args.denominator_cutoff_ev),
        )
        for key, definition in pair_defs.items():
            *pairs, mu_mev = definition
            mu_ev = float(mu_mev) * 1.0e-3
            occ = fermi_occupation(evals, mu_ev=mu_ev, temperature_k=0.0)
            val = 0.0
            trans_for_key = []
            for n_abs, m_abs in pairs:
                transition_ev = float(evals[m_abs] - evals[n_abs])
                trans_for_key.append(transition_ev)
                if transition_ev <= 0.0:
                    continue
                fnm = float(occ[n_abs] - occ[m_abs])
                if abs(fnm) < 1.0e-14:
                    continue
                val += fnm * imn[n_abs, m_abs, a, b, c] * float(_lorentzian_at_transition(target_ev, np.asarray([transition_ev]), eta_ev)[0])
            maps[key][ik] = float(val)
            transitions[key][ik] = float(np.mean(trans_for_key)) if trans_for_key else float("nan")

    np.savez_compressed(
        out / "wannierberri_fd_group_audit.npz",
        kpts_dimless=kpts,
        **{f"map_{k}": v for k, v in maps.items()},
        **{f"transition_mev_{k}": 1.0e3 * v for k, v in transitions.items()},
    )
    rows = []
    for key, vals in maps.items():
        idx = int(np.nanargmax(np.abs(vals))) if vals.size else 0
        rows.append(
            {
                "channel": key,
                "max_value": float(vals[idx]),
                "max_abs_k_dimless": [float(kpts[idx].real), float(kpts[idx].imag)],
                "halfmax_area": _area_metrics(vals, kpts, 0.5),
                "tenthmax_area": _area_metrics(vals, kpts, 0.1),
                "transition_mev_min_p50_max": [float(x) for x in np.nanpercentile(1.0e3 * transitions[key], [0, 50, 100])],
            }
        )
    summary = {
        "status": "WannierBerri ShiftCurrentFormula port audit for FD maps; values are WB Imn * f * Lorentzian, not conductivity prefactor.",
        "reference": "reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula external_terms=False",
        "config": config_summary(config, b0_params=params, lg=lg),
        "run": {
            "component": str(args.component),
            "component_indices": [int(x) for x in component],
            "lg": int(lg),
            "mesh_size": int(args.mesh_size),
            "n_hex_points": int(kpts.size),
            "target_energy_mev": float(args.target_energy_mev),
            "eta_mev": float(args.eta_mev),
            "sc_eta_mev": float(args.sc_eta_mev),
            "pair_defs": {k: str(v) for k, v in pair_defs.items()},
        },
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(out, kpts, params, maps, title=f"WB FD group audit, {args.component}, eta={args.eta_mev:g} meV, sc_eta={args.sc_eta_mev:g} meV")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
