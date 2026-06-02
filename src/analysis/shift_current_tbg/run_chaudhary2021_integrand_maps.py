from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, eigvalsh

from analysis.shift_current_htg.constants import SHIFT_CURRENT_PREFAC_UA_NM_PER_V2, eta_mev_to_ev
from analysis.shift_current_htg.response import (
    berry_connection_pair_from_D,
    fermi_occupation,
    generalized_derivative_pair_from_D,
    parse_component,
    velocity_matrices,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    build_chau_b0_hamiltonian,
    centered_flat_indices,
    fd_transition_pairs,
    finite_difference_b0_dhdk,
    flat_filling_to_mu,
    make_b0_parameters,
)
from .run_chaudhary2021_b0_noninteracting import _sample_flat_energies_b0
from .run_chaudhary2021_noninteracting import _parse_float_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chaudhary 2021 Fig. 2(d,e)-style k-space integrand maps.")
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=7)
    parser.add_argument("--mesh-size", type=int, default=65, help="square sampling before Wigner-Seitz hex mask")
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--component", default="y;xx")
    parser.add_argument("--group", choices=("FF", "FD"), default="FF")
    parser.add_argument("--labels", type=_parse_float_csv, default=(-2.0, 0.0, 2.0), help="filling labels or explicit mu labels")
    parser.add_argument("--mu-mev", type=_parse_float_csv, default=None, help="explicit chemical potentials in meV; one per label")
    parser.add_argument("--degeneracy", type=float, default=1.0, help="response flavor degeneracy multiplier")
    parser.add_argument("--filling-degeneracy", type=float, default=4.0, help="spin/valley degeneracy used to convert total TBG filling nu to chemical potential")
    parser.add_argument("--mu-mesh-size", type=int, default=12)
    parser.add_argument("--target-energy-mev", type=float, default=16.0)
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--fd-bands", type=int, default=1)
    parser.add_argument("--fd-mode", choices=("same_side", "cross_gap", "all"), default="same_side")
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_integrand_maps"))
    return parser.parse_args()


def _ws_hex_points(params, mesh_size: int) -> np.ndarray:
    g1, g2 = complex(params.g1), complex(params.g2)
    gscale = abs(g1)
    # Cover a little more than one hexagon and keep the Wigner-Seitz cell.
    vals = np.linspace(-0.75 * gscale, 0.75 * gscale, int(mesh_size), dtype=float)
    pts = []
    neighbors = [m * g1 + n * g2 for m in range(-1, 2) for n in range(-1, 2) if not (m == 0 and n == 0)]
    for x in vals:
        for y in vals:
            k = complex(x, y)
            if all(abs(k) <= abs(k - G) + 1.0e-12 for G in neighbors):
                pts.append(k)
    return np.asarray(pts, dtype=np.complex128)


def _component_weight_for_pairs(D, evals, occupations, pairs, component, cutoff):
    comp_a, comp_b, comp_c = component
    total = 0.0 + 0.0j
    skipped = 0
    for n_abs, m_abs in pairs:
        transition_ev = float(evals[m_abs] - evals[n_abs])
        if transition_ev <= 0.0:
            continue
        fnm = float(occupations[n_abs] - occupations[m_abs])
        if abs(fnm) < 1.0e-14:
            continue
        r_mn = berry_connection_pair_from_D(D, evals, m_abs, n_abs, denominator_cutoff_ev=cutoff)
        gd_nm = generalized_derivative_pair_from_D(D, evals, n_abs, m_abs, denominator_cutoff_ev=cutoff)
        skipped += int(gd_nm.skipped_small_denominators)
        weight = fnm * (r_mn[comp_b] * gd_nm.values[comp_a, comp_c] + r_mn[comp_c] * gd_nm.values[comp_a, comp_b])
        total += weight
    return total, skipped


def main() -> None:
    args = parse_args()
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
    sigma_rotation = not bool(args.no_sigma_rotation)
    periodic_g_grid = bool(args.periodic_g_grid)
    dim = 4 * lg * lg
    v_flat, c_flat = centered_flat_indices(dim)
    if str(args.group) == "FF":
        pairs = ((v_flat, c_flat),)
    else:
        pairs = fd_transition_pairs(dim, n_fd_bands_each_side=int(args.fd_bands), mode=str(args.fd_mode))
    component = parse_component(str(args.component))

    labels = tuple(float(x) for x in args.labels)
    if args.mu_mev is None:
        flat_energies = _sample_flat_energies_b0(
            params,
            config,
            lg=lg,
            mesh_size=int(args.mu_mesh_size),
            sigma_rotation=sigma_rotation,
            periodic_g_grid=periodic_g_grid,
        )
        mu_by_label = {f"{label:g}": flat_filling_to_mu(flat_energies, label, degeneracy=float(args.filling_degeneracy)) for label in labels}
        label_kind = "filling"
    else:
        mu_values = tuple(float(x) for x in args.mu_mev)
        if len(mu_values) != len(labels):
            raise ValueError("--mu-mev must have one value per label")
        mu_by_label = {f"{label:g}": mu * 1.0e-3 for label, mu in zip(labels, mu_values, strict=True)}
        label_kind = "mu_mev"

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
    gscale = abs(params.g1)

    values = np.zeros((len(labels), kpts.size), dtype=float)
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
        D = velocity_matrices(evecs, dhdk)
        for ilabel, label in enumerate(labels):
            occ = fermi_occupation(evals, mu_ev=float(mu_by_label[f"{label:g}"]), temperature_k=0.0)
            # Local Lorentzian-selected contribution.  The common BZ integration
            # weight is intentionally omitted; this is a map of the integrand.
            total = 0.0 + 0.0j
            comp_a, comp_b, comp_c = component
            for n_abs, m_abs in pairs:
                transition_ev = float(evals[m_abs] - evals[n_abs])
                if transition_ev <= 0.0:
                    continue
                fnm = float(occ[n_abs] - occ[m_abs])
                if abs(fnm) < 1.0e-14:
                    continue
                r_mn = berry_connection_pair_from_D(D, evals, m_abs, n_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                gd_nm = generalized_derivative_pair_from_D(D, evals, n_abs, m_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                skipped += int(gd_nm.skipped_small_denominators)
                local_weight = fnm * (r_mn[comp_b] * gd_nm.values[comp_a, comp_c] + r_mn[comp_c] * gd_nm.values[comp_a, comp_b])
                d = (eta_ev / np.pi) / ((target_ev - transition_ev) ** 2 + eta_ev * eta_ev)
                total += local_weight * d
            values[ilabel, ik] = float(np.real(-1.0j * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 * float(args.degeneracy) * total))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "integrand_maps.npz",
        kpts_dimless=kpts,
        values=values,
        labels=np.asarray(labels, dtype=float),
    )
    summary = {
        "status": "diagnostic k-space Lorentzian-selected integrand map; not BZ-integrated conductivity",
        "group": str(args.group),
        "component": str(args.component),
        "labels": [float(x) for x in labels],
        "label_kind": label_kind,
        "mu_by_label_ev": {k: float(v) for k, v in mu_by_label.items()},
        "target_energy_mev": float(args.target_energy_mev),
        "eta_mev": float(args.eta_mev),
        "response_degeneracy": float(args.degeneracy),
        "filling_degeneracy_for_mu": float(args.filling_degeneracy),
        "lg": int(lg),
        "mesh_size": int(args.mesh_size),
        "n_hex_points": int(kpts.size),
        "fd_bands": int(args.fd_bands),
        "fd_mode": str(args.fd_mode),
        "skipped_small_denominators": int(skipped),
        "value_minmax": [[float(np.min(v)), float(np.max(v))] for v in values],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.6), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([axes])
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    for ax, label, vals in zip(axes, labels, values, strict=True):
        sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=vals, s=10, cmap="magma", vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x/G$")
        ax.set_ylabel(r"$k_y/G$")
        label_text = (rf"$\mu$={label:g} meV" if label_kind == "mu_mev" else rf"$\nu$={label:g}")
        ax.set_title(label_text)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="local contribution [arb./physical prefactor]")
    fig.suptitle(f"{args.group} {args.component}, target={args.target_energy_mev:g} meV, eta={args.eta_mev:g} meV")
    fig.savefig(args.output_dir / "integrand_maps.png", dpi=220)
    fig.savefig(args.output_dir / "integrand_maps.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
