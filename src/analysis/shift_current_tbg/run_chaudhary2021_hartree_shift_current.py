from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import hamiltonian_gauge_data, wannierberri_shift_current_internal_imn
from analysis.shift_current_htg.constants import eta_mev_to_ev
from analysis.shift_current_htg.response import (
    Component,
    berry_connection_pair_from_D,
    fermi_occupation,
    generalized_derivative_pair_from_D,
    parse_component,
    velocity_matrices,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    b0_mbz_area_nm_inv_sq,
    centered_flat_indices,
    config_summary,
    fd_transition_pairs,
    finite_difference_b0_dhdk,
    make_b0_parameters,
)
from .hartree import arrays_to_rho, build_hartree_b0_hamiltonian, build_hartree_matrix_from_rho
from .run_chaudhary2021_b0_noninteracting import _vertex_k_grid_b0
from .run_chaudhary2021_noninteracting import (
    _add_triangle_to_hist,
    _all_component_labels,
    _parse_float_csv,
    _safe_key,
    _spectrum_from_histogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chaudhary 2021 Hartree-corrected shift current from saved Hartree density states.")
    parser.add_argument("--hartree-dir", type=Path, required=True, help="Directory produced by run_chaudhary2021_hartree_bands.py")
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=7)
    parser.add_argument("--mesh-size", type=int, default=12)
    parser.add_argument("--c3-symmetrize-grid", action="store_true")
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--epsilon-r", type=float, default=15.0)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--degeneracy", type=float, default=1.0, help="response flavor degeneracy multiplier; use 1 for paper-like per-flavor plots")
    parser.add_argument("--filling", type=_parse_float_csv, default=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    parser.add_argument("--mu-mev", type=_parse_float_csv, default=None, help="optional explicit chemical potentials, one per filling label")
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--occupation-temperature-k", type=float, default=0.0, help="Fermi occupation temperature used in the response calculation")
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.16)
    parser.add_argument("--n-energy", type=int, default=641)
    parser.add_argument("--energy-bin-width-mev", type=float, default=0.25)
    parser.add_argument("--fd-bands", type=int, default=10)
    parser.add_argument("--fd-mode", choices=("same_side", "cross_gap", "all"), default="same_side")
    parser.add_argument("--component", action="append", default=None)
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument(
        "--response-formula",
        choices=("sum_rule", "wannierberri"),
        default="sum_rule",
        help="Response integrand formula. 'wannierberri' ports ShiftCurrentFormula with external_terms=False.",
    )
    parser.add_argument("--sc-eta-mev", type=float, default=40.0, help="WannierBerri/Wannier90 principal-value eta used only for --response-formula=wannierberri")
    parser.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_hartree_shift_current"))
    return parser.parse_args()


def _hartree_state_filename(filling: float) -> str:
    key = f"nu={float(filling):g}"
    return f"hartree_state_{key.replace('=', '_').replace('-', 'm').replace('.', 'p')}.npz"


def _load_rho_and_mu(hartree_dir: Path, filling: float) -> tuple[dict[tuple[int, int], complex], float]:
    path = Path(hartree_dir) / _hartree_state_filename(float(filling))
    if not path.exists():
        raise FileNotFoundError(f"Missing Hartree state for nu={filling:g}: {path}")
    data = np.load(path, allow_pickle=False)
    rho = arrays_to_rho(data["rho_shifts"], data["rho_values"])
    iter_mu = np.asarray(data["iter_mu_ev"], dtype=float)
    mu = float(iter_mu[-1]) if iter_mu.size else 0.0
    return rho, mu


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
    dim = 4 * lg * lg
    v_flat, c_flat = centered_flat_indices(dim)
    pair_groups = {
        "FF": ((v_flat, c_flat),),
        "FD": fd_transition_pairs(dim, n_fd_bands_each_side=int(args.fd_bands), mode=str(args.fd_mode)),
    }
    group_names = tuple(pair_groups)
    group_index = {name: i for i, name in enumerate(group_names)}
    component_text = tuple(args.component) if args.component is not None else _all_component_labels()
    components: dict[str, Component] = {text: parse_component(text) for text in component_text}
    component_names = tuple(components)
    component_index = {name: i for i, name in enumerate(component_names)}
    fillings = tuple(float(x) for x in args.filling)
    sigma_rotation = not bool(args.no_sigma_rotation)
    periodic_g_grid = bool(args.periodic_g_grid)

    rho_by_filling: dict[str, dict[tuple[int, int], complex]] = {}
    mu_by_filling: dict[str, float] = {}
    explicit_mu_mev = None if args.mu_mev is None else tuple(float(x) for x in args.mu_mev)
    if explicit_mu_mev is not None and len(explicit_mu_mev) != len(fillings):
        raise ValueError("--mu-mev must have one value per --filling label")
    for idx, filling in enumerate(fillings):
        rho, state_mu = _load_rho_and_mu(Path(args.hartree_dir), filling)
        rho_by_filling[f"{filling:g}"] = rho
        mu_by_filling[f"{filling:g}"] = (explicit_mu_mev[idx] * 1.0e-3 if explicit_mu_mev is not None else state_mu)

    dhdk = finite_difference_b0_dhdk(
        params,
        config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        step_dimless=float(args.finite_step_dimless),
    )
    gvec = _generate_gvec(params, lg)
    tunnel_builder = _generate_t12 if periodic_g_grid else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, lg, int(config.valley))
    hartree_matrix_by_filling = {
        key: build_hartree_matrix_from_rho(params, config, lg=lg, rho_q=rho, epsilon_r=float(args.epsilon_r))
        for key, rho in rho_by_filling.items()
    }

    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = eta_mev_to_ev(float(args.eta_mev))
    sc_eta_ev = eta_mev_to_ev(float(args.sc_eta_mev))
    bin_width_ev = float(args.energy_bin_width_mev) * 1.0e-3
    emax_hist = float(args.emax) + 20.0 * eta_ev + bin_width_ev
    n_bins = max(1, int(math.ceil(emax_hist / bin_width_ev)))
    energy_edges = np.arange(n_bins + 1, dtype=float) * bin_width_ev
    hist = np.zeros((len(fillings), len(group_names), len(component_names), n_bins), dtype=np.complex128)
    transition_measure = np.zeros((len(fillings), len(group_names), n_bins), dtype=np.complex128)

    n = int(args.mesh_size)
    rotations = (0, 1, 2) if bool(args.c3_symmetrize_grid) else (0,)
    triangle_area = b0_mbz_area_nm_inv_sq(params, config) / (2.0 * n * n * len(rotations))
    inv_2pi_sq = 1.0 / (2.0 * np.pi) ** 2
    tri_vertices = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1)))

    vertices_evaluated = 0
    skipped_small_denominators = 0
    triangles_added = 0
    transition_min = float("inf")
    transition_max = float("-inf")

    for rotation in rotations:
        k_grid = _vertex_k_grid_b0(params, n, rotation_multiple=int(rotation))
        # Hartree potential depends on filling, so cache vertex data per filling.
        vertex_transition = {
            group: np.full((len(fillings), n + 1, n + 1, len(pairs)), np.nan, dtype=float)
            for group, pairs in pair_groups.items()
        }
        vertex_weight = {
            group: np.zeros((len(fillings), n + 1, n + 1, len(pairs), len(component_names)), dtype=np.complex128)
            for group, pairs in pair_groups.items()
        }
        vertex_valid = {
            group: np.zeros((len(fillings), n + 1, n + 1, len(pairs)), dtype=bool)
            for group, pairs in pair_groups.items()
        }
        for fi, filling in enumerate(fillings):
            fill_key = f"{filling:g}"
            h_hartree = hartree_matrix_by_filling[fill_key]
            mu = float(mu_by_filling[fill_key])
            for i in range(n + 1):
                for j in range(n + 1):
                    hmat = build_hartree_b0_hamiltonian(
                        complex(k_grid[i, j]),
                        params,
                        config,
                        lg=lg,
                        rho_q=None,
                        epsilon_r=float(args.epsilon_r),
                        sigma_rotation=sigma_rotation,
                        periodic_g_grid=periodic_g_grid,
                        gvec=gvec,
                        tunnel=tunnel,
                        hartree_matrix=h_hartree,
                    )
                    evals, evecs = eigh(hmat)
                    D = velocity_matrices(evecs, dhdk)
                    wb_imn = None
                    if str(args.response_formula) == "wannierberri":
                        gauge_data = hamiltonian_gauge_data(
                            evals,
                            evecs,
                            np.stack(dhdk, axis=0),
                            denominator_cutoff=float(args.denominator_cutoff_ev),
                        )
                        wb_imn = wannierberri_shift_current_internal_imn(
                            gauge_data.velocity_h,
                            gauge_data.energies,
                            sc_eta=sc_eta_ev,
                            denominator_cutoff=float(args.denominator_cutoff_ev),
                        )
                    occupations = fermi_occupation(evals, mu_ev=mu, temperature_k=float(args.occupation_temperature_k))
                    vertices_evaluated += 1
                    for group_name, pairs in pair_groups.items():
                        for pair_idx, (n_abs, m_abs) in enumerate(pairs):
                            transition_ev = float(evals[m_abs] - evals[n_abs])
                            vertex_transition[group_name][fi, i, j, pair_idx] = transition_ev
                            transition_min = min(transition_min, transition_ev)
                            transition_max = max(transition_max, transition_ev)
                            if transition_ev <= 0.0:
                                continue
                            fnm = float(occupations[n_abs] - occupations[m_abs])
                            if abs(fnm) < 1.0e-14:
                                continue
                            vertex_valid[group_name][fi, i, j, pair_idx] = True
                            if str(args.response_formula) == "wannierberri":
                                if wb_imn is None:
                                    raise RuntimeError("Internal error: WannierBerri Imn was not precomputed")
                                for component_name, component in components.items():
                                    comp_a, comp_b, comp_c = component
                                    ci = component_index[component_name]
                                    # Existing histogram-to-sigma pipeline applies sigma=Re[-i*C*integral].
                                    # WannierBerri Imn is already the real integrand multiplying C, so store i*Imn.
                                    weight = float(args.degeneracy) * fnm * (1.0j * wb_imn[n_abs, m_abs, comp_a, comp_b, comp_c])
                                    if np.isfinite(weight.real) and np.isfinite(weight.imag):
                                        vertex_weight[group_name][fi, i, j, pair_idx, ci] = complex(weight)
                            else:
                                r_mn = berry_connection_pair_from_D(D, evals, m_abs, n_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                                gd_nm = generalized_derivative_pair_from_D(D, evals, n_abs, m_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                                skipped_small_denominators += int(gd_nm.skipped_small_denominators)
                                for component_name, component in components.items():
                                    comp_a, comp_b, comp_c = component
                                    ci = component_index[component_name]
                                    geom = r_mn[comp_b] * gd_nm.values[comp_a, comp_c] + r_mn[comp_c] * gd_nm.values[comp_a, comp_b]
                                    weight = float(args.degeneracy) * fnm * geom
                                    if np.isfinite(weight.real) and np.isfinite(weight.imag):
                                        vertex_weight[group_name][fi, i, j, pair_idx, ci] = complex(weight)

        for i in range(n):
            for j in range(n):
                for tri in tri_vertices:
                    coords = tuple((i + di, j + dj) for di, dj in tri)
                    for group_name, pairs in pair_groups.items():
                        gi = group_index[group_name]
                        for pair_idx in range(len(pairs)):
                            for fi in range(len(fillings)):
                                energies = np.asarray([vertex_transition[group_name][fi, ii, jj, pair_idx] for ii, jj in coords], dtype=float)
                                if np.nanmax(energies) < float(args.emin) - 10.0 * eta_ev:
                                    continue
                                if np.nanmin(energies) > float(args.emax) + 20.0 * eta_ev:
                                    continue
                                if not all(vertex_valid[group_name][fi, ii, jj, pair_idx] for ii, jj in coords):
                                    continue
                                coeff = (
                                    np.mean(np.asarray([vertex_weight[group_name][fi, ii, jj, pair_idx] for ii, jj in coords]), axis=0)
                                    * inv_2pi_sq
                                )
                                _add_triangle_to_hist(hist[fi, gi], energy_edges, energies=energies, coeff=coeff, triangle_area_nm_inv_sq=triangle_area)
                                _add_triangle_to_hist(
                                    transition_measure[fi, gi],
                                    energy_edges,
                                    energies=energies,
                                    coeff=np.asarray(1.0 + 0.0j),
                                    triangle_area_nm_inv_sq=triangle_area,
                                )
                                triangles_added += 1

    spectra: dict[str, np.ndarray] = {}
    peaks: dict[str, dict[str, float]] = {}
    for fi, filling in enumerate(fillings):
        for group_name in group_names:
            gi = group_index[group_name]
            for component_name in component_names:
                ci = component_index[component_name]
                key = f"nu={filling:g}|{group_name}|{component_name}"
                sigma = _spectrum_from_histogram(photon_energies, energy_edges, hist[fi, gi, ci], eta_ev=eta_ev)
                spectra[key] = sigma
                if sigma.size:
                    peak_index = int(np.argmax(np.abs(sigma)))
                    peaks[key] = {
                        "max_abs_uA_nm_per_V2": float(np.max(np.abs(sigma))),
                        "signed_value_at_max_abs_uA_nm_per_V2": float(sigma[peak_index]),
                        "energy_at_max_abs_ev": float(photon_energies[peak_index]),
                    }
        for group_name in group_names:
            key = f"nu={filling:g}|{group_name}"
            transition_measure_spectrum = _spectrum_from_histogram(photon_energies, energy_edges, transition_measure[fi, group_index[group_name]], eta_ev=eta_ev)
            spectra[f"transition_measure|{key}"] = transition_measure_spectrum

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "Hartree-corrected shift current using saved full-continuum Hartree potentials; diagnostic until convergence and epsilon convention are audited.",
        "config": config_summary(config, b0_params=params, lg=lg),
        "hartree_dir": str(Path(args.hartree_dir)),
        "run": {
            "lg": lg,
            "mesh_size": int(args.mesh_size),
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "fillings_nu": [float(x) for x in fillings],
            "mu_by_filling_ev": {k: float(v) for k, v in mu_by_filling.items()},
            "response_degeneracy_multiplier": float(args.degeneracy),
            "eta_mev": float(args.eta_mev),
            "occupation_temperature_k": float(args.occupation_temperature_k),
            "epsilon_r": float(args.epsilon_r),
            "fd_bands_each_side": int(args.fd_bands),
            "fd_mode": str(args.fd_mode),
            "response_formula": str(args.response_formula),
            "sc_eta_mev": float(args.sc_eta_mev),
            "component_names": list(component_names),
            "energy_bin_width_mev": float(args.energy_bin_width_mev),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
        },
        "diagnostics": {
            "vertices_evaluated": int(vertices_evaluated),
            "triangles_added": int(triangles_added),
            "skipped_small_denominators": int(skipped_small_denominators),
            "transition_min_ev": float(transition_min),
            "transition_max_ev": float(transition_max),
        },
        "peaks": peaks,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays = {
        "photon_energies_ev": photon_energies,
        "energy_edges_ev": energy_edges,
    }
    for key, values in spectra.items():
        arrays[f"spectrum_{_safe_key(key)}"] = values
    for fi, filling in enumerate(fillings):
        for group_name in group_names:
            gi = group_index[group_name]
            for component_name in component_names:
                ci = component_index[component_name]
                hist_key = f"nu={filling:g}_{group_name}_{component_name}"
                arrays[f"hist_{_safe_key(hist_key)}"] = hist[fi, gi, ci]
    np.savez_compressed(args.output_dir / "spectra_histograms.npz", **arrays)

    # Compact Fig. 1(c-f)-style diagnostic for y;xx if available.
    primary_component = "y;xx" if "y;xx" in component_names else component_names[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, group_name in zip(axes, group_names, strict=True):
        for filling in fillings:
            key = f"nu={filling:g}|{group_name}|{primary_component}"
            if key not in spectra:
                continue
            ax.plot(photon_energies * 1.0e3, spectra[key], lw=1.25, label=rf"$\nu={filling:g}$")
        ax.axhline(0.0, color="0.6", lw=0.7)
        ax.set_title(f"Hartree {group_name} {primary_component}")
        ax.set_xlabel("photon energy (meV)")
        ax.set_ylabel(r"$\sigma$ ($\mu$A nm V$^{-2}$)")
        ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.output_dir / "hartree_shift_current.png", dpi=220)
    fig.savefig(args.output_dir / "hartree_shift_current.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
