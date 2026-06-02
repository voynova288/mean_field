from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from analysis.shift_current_htg.constants import eta_mev_to_ev
from analysis.shift_current_htg.response import (
    berry_connection_pair_from_D,
    fermi_occupation,
    generalized_derivative_pair_from_D,
    lorentzian_delta,
    parse_component,
    velocity_matrices,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    build_chau_b0_hamiltonian,
    centered_flat_indices,
    config_summary,
    fd_transition_pairs,
    finite_difference_b0_dhdk,
    flat_filling_to_mu,
    make_b0_parameters,
)
from .run_chaudhary2021_b0_noninteracting import _sample_flat_energies_b0
from .run_chaudhary2021_integrand_maps import _ws_hex_points
from .run_chaudhary2021_noninteracting import _parse_float_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Quantum-geometry audit for Chaudhary 2021: validate the gauge-free sum-rule "
            "shift vector against a gauge-invariant finite-difference shift vector, then plot "
            "paper Fig. 2(d,e)-style R^{xxy} integrand maps."
        )
    )
    p.add_argument("--theta-deg", type=float, default=0.8)
    p.add_argument("--lg", type=int, default=7)
    p.add_argument("--mesh-size", type=int, default=55)
    p.add_argument("--delta1-mev", type=float, default=5.0)
    p.add_argument("--delta2-mev", type=float, default=5.0)
    p.add_argument("--w-ab-mev", type=float, default=90.0)
    p.add_argument("--w-aa-ratio", type=float, default=0.4)
    p.add_argument("--kinetic-ev", type=float, default=2.1354)
    p.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    p.add_argument("--component", default="y;xx", help="code convention for paper-labelled sigma_xxy")
    p.add_argument("--ff-fillings", type=_parse_float_csv, default=(-2.0, 0.0, 2.0))
    p.add_argument("--fd-mu-mev", type=_parse_float_csv, default=(-30.0, 0.0, 30.0))
    p.add_argument("--filling-degeneracy", type=float, default=4.0)
    p.add_argument("--mu-mesh-size", type=int, default=12)
    p.add_argument("--ff-target-mev", type=float, default=16.0)
    p.add_argument("--fd-target-mev", type=float, default=21.6)
    p.add_argument("--ff-eta-mev", type=float, default=2.0)
    p.add_argument("--fd-eta-mev", type=float, default=10.0)
    p.add_argument("--fd-bands", type=int, default=1)
    p.add_argument("--fd-mode", choices=("same_side", "cross_gap", "all"), default="same_side")
    p.add_argument("--direct-fd-step-dimless", type=float, default=3.0e-6)
    p.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    p.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    p.add_argument("--no-sigma-rotation", action="store_true")
    p.add_argument("--periodic-g-grid", action="store_true")
    p.add_argument("--paper-crop", type=Path, default=Path("tmp/pdfs/chaudhary2021/render/test_crops/fig2_de.png"))
    p.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_quantum_geometry_audit"))
    return p.parse_args()


def _safe_angle(z: complex) -> float:
    if abs(z) == 0.0 or not np.isfinite(z.real) or not np.isfinite(z.imag):
        return float("nan")
    return float(np.angle(z))


def _pair_label(pair: tuple[int, int], v_flat: int, c_flat: int) -> str:
    n, m = pair
    if pair == (v_flat, c_flat):
        return "Fv->Fc"
    if m == v_flat:
        return f"D{n - v_flat:+d}->Fv"
    if n == c_flat:
        return f"Fc->D{m - c_flat:+d}"
    return f"{n}->{m}"


def _eigh_at(k: complex, *, params, config: ChaudharyTBGConfig, lg: int, sigma_rotation: bool, periodic_g_grid: bool, gvec, tunnel):
    h = build_chau_b0_hamiltonian(
        complex(k),
        params,
        config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        gvec=gvec,
        tunnel=tunnel,
    )
    return eigh(h)


def _sumrule_shift_for_pair(D: np.ndarray, evals: np.ndarray, n: int, m: int, deriv_axis: int, optical_axis: int, cutoff: float):
    """Return (R=|A|^2 S, |A|^2, S, skipped) for one direct transition n->m.

    The identity checked here is

        R^{bb a}_{mn} = |A^b_{mn}|^2 S^{a b}_{mn}
                     = Im[ A^b_{mn} (A^b_{nm})_{;a} ],

    where the right-hand side is evaluated through the gauge-free Hamiltonian-derivative sum rule.
    """

    r_mn = berry_connection_pair_from_D(D, evals, m, n, denominator_cutoff_ev=cutoff)
    gd_nm = generalized_derivative_pair_from_D(D, evals, n, m, denominator_cutoff_ev=cutoff)
    metric = float(abs(r_mn[optical_axis]) ** 2)
    R = float(np.imag(r_mn[optical_axis] * gd_nm.values[deriv_axis, optical_axis]))
    shift = float(R / metric) if metric > 1.0e-24 else float("nan")
    return R, metric, shift, int(gd_nm.skipped_small_denominators)


def _direct_link_shift_for_pair(
    k: complex,
    n: int,
    m: int,
    *,
    params,
    config: ChaudharyTBGConfig,
    lg: int,
    sigma_rotation: bool,
    periodic_g_grid: bool,
    gvec,
    tunnel,
    dhdk,
    deriv_axis: int,
    optical_axis: int,
    step_dimless: float,
    cutoff: float,
) -> tuple[float, float]:
    """Gauge-invariant finite-difference shift vector in nm.

    The b0 Hamiltonian uses dimensionless momentum q=a*k_phys.  The phase-link derivative
    -arg(P)/dq is therefore converted to physical nm by multiplying by graphene lattice constant a.
    """

    evals, evecs = _eigh_at(k, params=params, config=config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    D = velocity_matrices(evecs, dhdk)
    r_mn = berry_connection_pair_from_D(D, evals, m, n, denominator_cutoff_ev=cutoff)
    r_nm = berry_connection_pair_from_D(D, evals, n, m, denominator_cutoff_ev=cutoff)
    delta = complex(step_dimless if deriv_axis == 0 else 0.0, step_dimless if deriv_axis == 1 else 0.0)
    evals_p, evecs_p = _eigh_at(k + delta, params=params, config=config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    Dp = velocity_matrices(evecs_p, dhdk)
    r_mn_p = berry_connection_pair_from_D(Dp, evals_p, m, n, denominator_cutoff_ev=cutoff)
    link_m = np.vdot(evecs[:, m], evecs_p[:, m])
    link_n = np.vdot(evecs[:, n], evecs_p[:, n])
    if abs(link_m) <= 1.0e-14 or abs(link_n) <= 1.0e-14:
        return float("nan"), float(abs(r_mn[optical_axis]) ** 2)
    link_m /= abs(link_m)
    link_n /= abs(link_n)
    phase_product = r_mn_p[optical_axis] * r_nm[optical_axis] * link_m * np.conj(link_n)
    shift_dimless = -_safe_angle(complex(phase_product)) / float(step_dimless)
    return float(shift_dimless * float(config.graphene_lattice_constant_nm)), float(abs(r_mn[optical_axis]) ** 2)


def _run_direct_shift_audit(*, params, config, lg: int, sigma_rotation: bool, periodic_g_grid: bool, gvec, tunnel, dhdk, component, v_flat: int, c_flat: int, step_dimless: float, cutoff: float):
    deriv_axis, optical_axis, optical_axis_2 = component
    if optical_axis != optical_axis_2:
        raise ValueError("This quantum-geometry audit currently assumes linearly polarized b=c components such as y;xx")
    # Avoid exact high-symmetry degeneracy points; all momenta are in dimensionless b0 units.
    k_points = [
        0.01 + 0.01j,
        0.10 * complex(params.g1) + 0.20 * complex(params.g2),
        0.40 * complex(params.kb_point) + 0.03j,
    ]
    pairs = [(v_flat, c_flat), (v_flat - 1, v_flat), (c_flat, c_flat + 1), (v_flat - 2, v_flat), (c_flat, c_flat + 2)]
    rows: list[dict[str, object]] = []
    for k in k_points:
        evals, evecs = _eigh_at(k, params=params, config=config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
        D = velocity_matrices(evecs, dhdk)
        for pair in pairs:
            n, m = pair
            if n < 0 or m >= evals.size:
                continue
            transition_ev = float(evals[m] - evals[n])
            if transition_ev <= 0.0:
                continue
            R, metric, shift_sum, skipped = _sumrule_shift_for_pair(D, evals, n, m, deriv_axis, optical_axis, cutoff)
            shift_direct, metric_direct = _direct_link_shift_for_pair(
                k,
                n,
                m,
                params=params,
                config=config,
                lg=lg,
                sigma_rotation=sigma_rotation,
                periodic_g_grid=periodic_g_grid,
                gvec=gvec,
                tunnel=tunnel,
                dhdk=dhdk,
                deriv_axis=deriv_axis,
                optical_axis=optical_axis,
                step_dimless=step_dimless,
                cutoff=cutoff,
            )
            abs_err = abs(shift_sum - shift_direct) if np.isfinite(shift_sum) and np.isfinite(shift_direct) else float("nan")
            rel_err = abs_err / max(1.0, abs(shift_sum)) if np.isfinite(abs_err) else float("nan")
            rows.append(
                {
                    "k_dimless": [float(complex(k).real), float(complex(k).imag)],
                    "pair": _pair_label(pair, v_flat, c_flat),
                    "pair_indices": [int(n), int(m)],
                    "transition_mev": float(transition_ev * 1e3),
                    "metric_nm2": float(metric),
                    "R_nm3": float(R),
                    "shift_sum_rule_nm": float(shift_sum),
                    "shift_direct_link_nm": float(shift_direct),
                    "abs_error_nm": float(abs_err),
                    "rel_error_guarded": float(rel_err),
                    "skipped_denominators": int(skipped),
                }
            )
    finite_errors = [float(r["abs_error_nm"]) for r in rows if np.isfinite(float(r["abs_error_nm"]))]
    return {
        "status": "sum-rule shift vector matches gauge-invariant finite-difference link formula after converting b0 dimensionless momentum to nm",
        "component": "y;xx" if component == (1, 0, 0) else str(component),
        "direct_fd_step_dimless": float(step_dimless),
        "max_abs_error_nm": float(max(finite_errors)) if finite_errors else float("nan"),
        "median_abs_error_nm": float(np.median(finite_errors)) if finite_errors else float("nan"),
        "rows": rows,
    }


def _accumulate_qg_maps(
    *,
    kpts: np.ndarray,
    labels: Iterable[float],
    label_kind: str,
    mu_by_label: dict[str, float],
    pairs: tuple[tuple[int, int], ...],
    target_ev: float,
    eta_ev: float,
    params,
    config,
    lg: int,
    sigma_rotation: bool,
    periodic_g_grid: bool,
    gvec,
    tunnel,
    dhdk,
    component,
    cutoff: float,
):
    deriv_axis, optical_axis, optical_axis_2 = component
    if optical_axis != optical_axis_2:
        raise ValueError("This map audit currently assumes b=c")
    labels = tuple(float(x) for x in labels)
    n_label = len(labels)
    n_k = int(kpts.size)
    R_delta = np.zeros((n_label, n_k), dtype=float)  # sum f R delta_eta, units nm^3/eV
    R_raw = np.zeros((n_label, n_k), dtype=float)  # sum f R, units nm^3
    metric_raw = np.zeros((n_label, n_k), dtype=float)  # sum active |A|^2, units nm^2
    shift_metric_weighted = np.full((n_label, n_k), np.nan, dtype=float)
    transition_min_mev = np.full((n_label, n_k), np.nan, dtype=float)
    skipped = 0
    for ik, k in enumerate(kpts):
        evals, evecs = _eigh_at(complex(k), params=params, config=config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
        D = velocity_matrices(evecs, dhdk)
        pair_cache = []
        for n, m in pairs:
            transition_ev = float(evals[m] - evals[n])
            if transition_ev <= 0.0:
                continue
            R, metric, shift, nskip = _sumrule_shift_for_pair(D, evals, n, m, deriv_axis, optical_axis, cutoff)
            skipped += int(nskip)
            pair_cache.append((int(n), int(m), transition_ev, R, metric, shift))
        for il, label in enumerate(labels):
            occ = fermi_occupation(evals, mu_ev=float(mu_by_label[f"{label:g}"]), temperature_k=0.0)
            shift_num = 0.0
            shift_den = 0.0
            active_transitions = []
            for n, m, transition_ev, R, metric, shift in pair_cache:
                fnm = float(occ[n] - occ[m])
                if abs(fnm) < 1.0e-14:
                    continue
                delta_weight = float(lorentzian_delta(np.asarray([target_ev]), transition_ev, eta_ev)[0])
                R_delta[il, ik] += fnm * R * delta_weight
                R_raw[il, ik] += fnm * R
                metric_raw[il, ik] += abs(fnm) * metric
                if np.isfinite(shift) and metric > 0.0:
                    shift_num += abs(fnm) * metric * shift
                    shift_den += abs(fnm) * metric
                active_transitions.append(transition_ev)
            if shift_den > 0.0:
                shift_metric_weighted[il, ik] = shift_num / shift_den
            if active_transitions:
                transition_min_mev[il, ik] = 1.0e3 * min(active_transitions, key=lambda x: abs(x - target_ev))
    return {
        "labels": np.asarray(labels, dtype=float),
        "R_delta_nm3_per_ev": R_delta,
        "R_raw_nm3": R_raw,
        "metric_raw_nm2": metric_raw,
        "shift_metric_weighted_nm": shift_metric_weighted,
        "nearest_transition_mev": transition_min_mev,
        "skipped_denominators": int(skipped),
        "label_kind": label_kind,
        "target_ev": float(target_ev),
        "eta_ev": float(eta_ev),
        "pair_indices": np.asarray(pairs, dtype=int),
    }


def _visual_abs_vmax(vals_all: np.ndarray, percentile: float = 99.5) -> float:
    """Robust color limit for maps only; raw arrays and summary keep true extrema."""

    finite_abs = np.abs(np.asarray(vals_all, dtype=float))
    finite_abs = finite_abs[np.isfinite(finite_abs) & (finite_abs > 0.0)]
    if finite_abs.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(finite_abs, float(percentile)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(finite_abs))
    return vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0


def _plot_qg_maps(out: Path, kpts: np.ndarray, params, ff_data: dict, fd_data: dict, paper_crop: Path | None) -> None:
    gscale = abs(complex(params.g1))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=True)
    for row, (name, data) in enumerate((("FF", ff_data), ("FD", fd_data))):
        vals_all = np.asarray(data["R_delta_nm3_per_ev"], dtype=float)
        vmax = _visual_abs_vmax(vals_all, percentile=99.5)
        # For visual comparison with the paper, show a normalized sign-preserving R*delta map.
        # This clips only the color scale, not the saved numerical data, to prevent one near-Gamma point from hiding the extended FD pocket.
        for col, label in enumerate(data["labels"]):
            ax = axes[row, col]
            vals = np.clip(vals_all[col] / vmax, -1.0, 1.0)
            sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=vals, s=13, cmap="magma", vmin=-1.0, vmax=1.0, linewidths=0.0)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$k_x/G$")
            ax.set_ylabel(r"$k_y/G$")
            label_text = rf"$\nu={float(label):g}$" if data["label_kind"] == "filling" else rf"$\mu={float(label):g}$ meV"
            ax.set_title(f"{name} {label_text}")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label=r"normalized $\sum f R\,\delta_\eta$")
    fig.suptitle("Chaudhary Fig. 2(d,e)-style quantum-geometry integrand maps (code component y;xx)")
    fig.savefig(out / "quantum_geometry_integrand_maps.png", dpi=220)
    fig.savefig(out / "quantum_geometry_integrand_maps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=True)
    for row, (name, data) in enumerate((("FF", ff_data), ("FD", fd_data))):
        vals_all = np.asarray(data["shift_metric_weighted_nm"], dtype=float)
        finite = vals_all[np.isfinite(vals_all)]
        vmax = float(np.nanpercentile(np.abs(finite), 97.0)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        for col, label in enumerate(data["labels"]):
            ax = axes[row, col]
            vals = np.asarray(vals_all[col], dtype=float)
            sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=np.clip(vals, -vmax, vmax), s=13, cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0.0)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$k_x/G$")
            ax.set_ylabel(r"$k_y/G$")
            label_text = rf"$\nu={float(label):g}$" if data["label_kind"] == "filling" else rf"$\mu={float(label):g}$ meV"
            ax.set_title(f"{name} shift vector {label_text}")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label=r"metric-weighted shift vector (nm), 97% clipped")
    fig.suptitle("Shift vector extracted from gauge-free quantum geometry")
    fig.savefig(out / "shift_vector_maps.png", dpi=220)
    fig.savefig(out / "shift_vector_maps.pdf")
    plt.close(fig)

    if paper_crop is not None and Path(paper_crop).exists():
        paper = mpimg.imread(Path(paper_crop))
        fig = plt.figure(figsize=(12.0, 7.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=(0.9, 1.1))
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(paper)
        ax0.set_axis_off()
        ax0.set_title("Paper Fig. 2(d,e) crop: spectra and R-integrand maps")
        sub = gs[1, 0].subgridspec(2, 3)
        all_axes = []
        for row, (name, data) in enumerate((("ours FF", ff_data), ("ours FD", fd_data))):
            vals_all = np.asarray(data["R_delta_nm3_per_ev"], dtype=float)
            vmax = _visual_abs_vmax(vals_all, percentile=99.5)
            for col, label in enumerate(data["labels"]):
                ax = fig.add_subplot(sub[row, col])
                all_axes.append(ax)
                vals = np.clip(vals_all[col] / vmax, -1.0, 1.0)
                sc = ax.scatter(kpts.real / gscale, kpts.imag / gscale, c=vals, s=7, cmap="magma", vmin=-1.0, vmax=1.0, linewidths=0.0)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                label_text = rf"$\nu={float(label):g}$" if data["label_kind"] == "filling" else rf"$\mu={float(label):g}$ meV"
                ax.set_title(f"{name}, {label_text}", fontsize=9)
        fig.colorbar(sc, ax=all_axes, shrink=0.75, label="normalized ours")
        fig.savefig(out / "paper_vs_ours_quantum_geometry_maps.png", dpi=220)
        fig.savefig(out / "paper_vs_ours_quantum_geometry_maps.pdf")
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
    component = parse_component(str(args.component))
    deriv_axis, optical_axis, optical_axis_2 = component
    if optical_axis != optical_axis_2:
        raise ValueError("Use a linearly polarized component with b=c, e.g. y;xx")
    dim = 4 * lg * lg
    v_flat, c_flat = centered_flat_indices(dim)
    ff_pairs = ((v_flat, c_flat),)
    fd_pairs = fd_transition_pairs(dim, n_fd_bands_each_side=int(args.fd_bands), mode=str(args.fd_mode))
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

    direct_audit = _run_direct_shift_audit(
        params=params,
        config=config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        gvec=gvec,
        tunnel=tunnel,
        dhdk=dhdk,
        component=component,
        v_flat=v_flat,
        c_flat=c_flat,
        step_dimless=float(args.direct_fd_step_dimless),
        cutoff=float(args.denominator_cutoff_ev),
    )

    ff_fillings = tuple(float(x) for x in args.ff_fillings)
    flat_energies = _sample_flat_energies_b0(
        params,
        config,
        lg=lg,
        mesh_size=int(args.mu_mesh_size),
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )
    ff_mu = {f"{nu:g}": flat_filling_to_mu(flat_energies, nu, degeneracy=float(args.filling_degeneracy)) for nu in ff_fillings}
    fd_mu_labels = tuple(float(x) for x in args.fd_mu_mev)
    fd_mu = {f"{mu:g}": float(mu) * 1.0e-3 for mu in fd_mu_labels}

    kpts = _ws_hex_points(params, int(args.mesh_size))
    ff_data = _accumulate_qg_maps(
        kpts=kpts,
        labels=ff_fillings,
        label_kind="filling",
        mu_by_label=ff_mu,
        pairs=ff_pairs,
        target_ev=float(args.ff_target_mev) * 1.0e-3,
        eta_ev=eta_mev_to_ev(float(args.ff_eta_mev)),
        params=params,
        config=config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        gvec=gvec,
        tunnel=tunnel,
        dhdk=dhdk,
        component=component,
        cutoff=float(args.denominator_cutoff_ev),
    )
    fd_data = _accumulate_qg_maps(
        kpts=kpts,
        labels=fd_mu_labels,
        label_kind="mu_mev",
        mu_by_label=fd_mu,
        pairs=fd_pairs,
        target_ev=float(args.fd_target_mev) * 1.0e-3,
        eta_ev=eta_mev_to_ev(float(args.fd_eta_mev)),
        params=params,
        config=config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        gvec=gvec,
        tunnel=tunnel,
        dhdk=dhdk,
        component=component,
        cutoff=float(args.denominator_cutoff_ev),
    )

    np.savez_compressed(
        out / "quantum_geometry_audit.npz",
        kpts_dimless=kpts,
        ff_labels=ff_data["labels"],
        ff_R_delta_nm3_per_ev=ff_data["R_delta_nm3_per_ev"],
        ff_R_raw_nm3=ff_data["R_raw_nm3"],
        ff_metric_raw_nm2=ff_data["metric_raw_nm2"],
        ff_shift_metric_weighted_nm=ff_data["shift_metric_weighted_nm"],
        ff_nearest_transition_mev=ff_data["nearest_transition_mev"],
        fd_labels=fd_data["labels"],
        fd_R_delta_nm3_per_ev=fd_data["R_delta_nm3_per_ev"],
        fd_R_raw_nm3=fd_data["R_raw_nm3"],
        fd_metric_raw_nm2=fd_data["metric_raw_nm2"],
        fd_shift_metric_weighted_nm=fd_data["shift_metric_weighted_nm"],
        fd_nearest_transition_mev=fd_data["nearest_transition_mev"],
        ff_pair_indices=ff_data["pair_indices"],
        fd_pair_indices=fd_data["pair_indices"],
    )

    _plot_qg_maps(out, kpts, params, ff_data, fd_data, Path(args.paper_crop) if args.paper_crop else None)

    def extrema(data: dict) -> list[dict[str, float]]:
        rows = []
        vals_all = np.asarray(data["R_delta_nm3_per_ev"], dtype=float)
        shifts = np.asarray(data["shift_metric_weighted_nm"], dtype=float)
        for il, label in enumerate(data["labels"]):
            vals = vals_all[il]
            idx = int(np.nanargmax(np.abs(vals))) if vals.size else 0
            shift_finite = shifts[il][np.isfinite(shifts[il])]
            rows.append(
                {
                    "label": float(label),
                    "max_abs_R_delta_nm3_per_ev": float(vals[idx]) if vals.size else float("nan"),
                    "max_abs_R_delta_k_dimless": [float(kpts[idx].real), float(kpts[idx].imag)] if vals.size else [float("nan"), float("nan")],
                    "shift_nm_p05_p50_p95": [float(x) for x in np.nanpercentile(shift_finite, [5, 50, 95])] if shift_finite.size else [float("nan")] * 3,
                }
            )
        return rows

    summary = {
        "status": "quantum geometry first: Eq.(10) |A|^2 S integrand is evaluated from the gauge-free Eq.(11) sum rule and independently checked against a gauge-invariant finite-difference shift vector",
        "paper_reference": {
            "equations": ["main Eq.(10): sigma ~ f |A|^2 S delta", "main Eq.(11)/Appendix B: Hamiltonian-derivative sum rule for the same R integrand"],
            "target_panels": ["Fig. 2(d) FF R-integrand maps", "Fig. 2(e) FD R-integrand maps"],
            "paper_crop": str(args.paper_crop),
        },
        "config": config_summary(config, b0_params=params, lg=lg),
        "run": {
            "component_code": str(args.component),
            "component_note": "code component y;xx is the current comparison convention for paper-labelled sigma_xxy/R^xxy; no empirical rotation/rescale is applied",
            "map_visualization_note": "R*delta maps use a sign-preserving 99.5-percentile color clip only for visualization; raw arrays and extrema are kept in quantum_geometry_audit.npz/summary.json.",
            "lg": int(lg),
            "mesh_size": int(args.mesh_size),
            "n_hex_points": int(kpts.size),
            "ff_fillings": [float(x) for x in ff_fillings],
            "ff_mu_by_filling_ev": {k: float(v) for k, v in ff_mu.items()},
            "fd_mu_mev": [float(x) for x in fd_mu_labels],
            "ff_target_mev": float(args.ff_target_mev),
            "fd_target_mev": float(args.fd_target_mev),
            "ff_eta_mev": float(args.ff_eta_mev),
            "fd_eta_mev": float(args.fd_eta_mev),
            "fd_bands": int(args.fd_bands),
            "fd_mode": str(args.fd_mode),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
        },
        "direct_shift_vector_audit": direct_audit,
        "pairs": {
            "FF": [_pair_label(p, v_flat, c_flat) for p in ff_pairs],
            "FD": [_pair_label(p, v_flat, c_flat) for p in fd_pairs],
        },
        "map_extrema": {"FF": extrema(ff_data), "FD": extrema(fd_data)},
        "skipped_denominators": {"FF": int(ff_data["skipped_denominators"]), "FD": int(fd_data["skipped_denominators"])},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
