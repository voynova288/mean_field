from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, eigvalsh

from analysis.response_derivative_gauge import hamiltonian_gauge_data, wannierberri_shift_current_internal_imn
from analysis.shift_current_htg.constants import eta_mev_to_ev
from analysis.shift_current_htg.response import (
    Component,
    berry_connection_pair_from_D,
    component_label,
    fermi_occupation,
    generalized_derivative_pair_from_D,
    parse_component,
    sigma_from_integral,
    velocity_matrices,
)
from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    b0_fig2_kpath,
    b0_mbz_area_nm_inv_sq,
    build_chau_b0_hamiltonian,
    centered_flat_indices,
    config_summary,
    fd_transition_pairs,
    finite_difference_b0_dhdk,
    flat_filling_to_mu,
    make_b0_parameters,
)
from .run_chaudhary2021_noninteracting import (
    _add_triangle_to_hist,
    _all_component_labels,
    _lorentzian_interval_integral,
    _parse_float_csv,
    _plot_results,
    _safe_key,
    _spectrum_from_histogram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chaudhary 2021 noninteracting TBG shift current using the repository's previous b0 BM model. "
            "This replaces the experimental atmg-shell adapter whose bands do not match Fig. 2(a)."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=7, help="old b0 square G-grid side length; must be odd")
    parser.add_argument("--mesh-size", type=int, default=12)
    parser.add_argument("--c3-symmetrize-grid", action="store_true")
    parser.add_argument("--mu-mesh-size", type=int, default=None)
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--degeneracy", type=float, default=1.0, help="response flavor degeneracy multiplier; Chaudhary paper plots are most consistent with per-flavor degeneracy=1")
    parser.add_argument("--filling-degeneracy", type=float, default=4.0, help="spin/valley degeneracy used to convert total TBG filling nu to chemical potential")
    parser.add_argument("--filling", type=_parse_float_csv, default=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    parser.add_argument(
        "--mu-mev",
        type=_parse_float_csv,
        default=None,
        help=(
            "Optional explicit chemical potentials in meV, one per --filling label. "
            "This is useful for Chaudhary Fig. 2(e), which labels FD cuts by mu=+-30 meV rather than flat-band filling."
        ),
    )
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument(
        "--partial-occupation-triangles",
        action="store_true",
        help=(
            "Use zero-valued blocked vertices inside each transition-energy triangle instead of dropping the whole triangle "
            "whenever any vertex is Pauli blocked. This is important when a sharp Fermi pocket cuts through a small high-weight region."
        ),
    )
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.16)
    parser.add_argument("--n-energy", type=int, default=641)
    parser.add_argument("--energy-bin-width-mev", type=float, default=0.25)
    parser.add_argument("--fd-bands", type=int, default=10)
    parser.add_argument(
        "--fd-mode",
        choices=("same_side", "cross_gap", "all"),
        default="same_side",
        help="Direct FD transition set. 'same_side' is the paper Fig. 2 convention; 'all' is only a diagnostic.",
    )
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
    parser.add_argument("--path-points-per-segment", type=int, default=60)
    parser.add_argument("--band-window", type=int, default=8)
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_b0_nonint_fig2"))
    return parser.parse_args()


def _rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(math.cos(float(angle_rad)), math.sin(float(angle_rad)))


def _vertex_k_grid_b0(params, mesh_size: int, *, rotation_multiple: int = 0) -> np.ndarray:
    n = int(mesh_size)
    f = np.linspace(0.0, 1.0, n + 1, dtype=float)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    k = np.asarray(f1 * params.g1 + f2 * params.g2, dtype=np.complex128)
    if int(rotation_multiple) == 0:
        return k
    angle = 2.0 * np.pi * float(rotation_multiple) / 3.0
    return np.asarray([_rotate_complex(value, angle) for value in k.reshape(-1)], dtype=np.complex128).reshape(k.shape)


def _sample_flat_energies_b0(params, config: ChaudharyTBGConfig, *, lg: int, mesh_size: int, sigma_rotation: bool, periodic_g_grid: bool) -> np.ndarray:
    n = int(mesh_size)
    dim = 4 * int(lg) * int(lg)
    v_flat, c_flat = centered_flat_indices(dim)
    gvec = _generate_gvec(params, int(lg))
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, int(lg), int(config.valley))
    out = []
    f = np.arange(n, dtype=float) / float(n)
    for f1 in f:
        for f2 in f:
            k = complex(f1 * params.g1 + f2 * params.g2)
            evals = eigvalsh(
                build_chau_b0_hamiltonian(
                    k,
                    params,
                    config,
                    lg=int(lg),
                    sigma_rotation=bool(sigma_rotation),
                    periodic_g_grid=bool(periodic_g_grid),
                    gvec=gvec,
                    tunnel=tunnel,
                )
            )
            out.append([float(evals[v_flat]), float(evals[c_flat])])
    return np.asarray(out, dtype=float)


def _compute_path_bands_b0(params, config, *, lg: int, points_per_segment: int, band_window: int, sigma_rotation: bool, periodic_g_grid: bool):
    path = b0_fig2_kpath(params, int(points_per_segment))
    gvec = _generate_gvec(params, int(lg))
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, int(lg), int(config.valley))
    dim = 4 * int(lg) * int(lg)
    center = dim // 2
    lo = max(0, center - int(band_window))
    hi = min(dim, center + int(band_window))
    bands = []
    for k in path.kvec:
        evals = eigvalsh(
            build_chau_b0_hamiltonian(
                complex(k),
                params,
                config,
                lg=int(lg),
                sigma_rotation=bool(sigma_rotation),
                periodic_g_grid=bool(periodic_g_grid),
                gvec=gvec,
                tunnel=tunnel,
            )
        )
        bands.append(evals[lo:hi])
    return path, np.asarray(bands, dtype=float), (lo, hi)


def main() -> None:
    args = parse_args()
    if int(args.lg) % 2 != 1:
        raise ValueError("--lg must be odd for the previous b0 model")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ChaudharyTBGConfig(
        theta_deg=float(args.theta_deg),
        n_shells=0,
        kinetic_ev=float(args.kinetic_ev),
        w_ab_ev=float(args.w_ab_mev) * 1.0e-3,
        w_aa_ratio=float(args.w_aa_ratio),
        delta1_ev=float(args.delta1_mev) * 1.0e-3,
        delta2_ev=float(args.delta2_mev) * 1.0e-3,
        valley=int(args.valley),
        dirac_sign=-1.0,
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

    mu_mesh = int(args.mu_mesh_size) if args.mu_mesh_size is not None else max(8, min(int(args.mesh_size), 16))
    explicit_mu_mev = None if args.mu_mev is None else tuple(float(x) for x in args.mu_mev)
    if explicit_mu_mev is not None:
        if len(explicit_mu_mev) != len(fillings):
            raise ValueError(f"--mu-mev must have one value per --filling label: got {len(explicit_mu_mev)} vs {len(fillings)}")
        mu_by_filling = {f"{filling:g}": float(mu_mev) * 1.0e-3 for filling, mu_mev in zip(fillings, explicit_mu_mev, strict=True)}
    else:
        flat_energies = _sample_flat_energies_b0(
            params,
            config,
            lg=lg,
            mesh_size=mu_mesh,
            sigma_rotation=sigma_rotation,
            periodic_g_grid=periodic_g_grid,
        )
        mu_by_filling = {
            f"{filling:g}": flat_filling_to_mu(flat_energies, filling, degeneracy=float(args.filling_degeneracy))
            for filling in fillings
        }

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
    max_pair_count = max(len(pairs) for pairs in pair_groups.values())

    vertices_evaluated = 0
    attempted_weights = 0
    kept_weights = 0
    skipped_small_denominators = 0
    transition_min = float("inf")
    transition_max = float("-inf")
    triangles_added = 0
    tri_vertices = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1)))

    for rotation in rotations:
        k_grid = _vertex_k_grid_b0(params, n, rotation_multiple=int(rotation))
        vertex_transition = {
            group: np.full((n + 1, n + 1, len(pairs)), np.nan, dtype=float) for group, pairs in pair_groups.items()
        }
        vertex_weight = {
            group: np.zeros((len(fillings), n + 1, n + 1, len(pairs), len(component_names)), dtype=np.complex128)
            for group, pairs in pair_groups.items()
        }
        vertex_valid = {
            group: np.zeros((len(fillings), n + 1, n + 1, len(pairs)), dtype=bool) for group, pairs in pair_groups.items()
        }
        for i in range(n + 1):
            for j in range(n + 1):
                hmat = build_chau_b0_hamiltonian(
                    complex(k_grid[i, j]),
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
                occ_by_filling = [
                    fermi_occupation(evals, mu_ev=float(mu_by_filling[f"{filling:g}"]), temperature_k=0.0)
                    for filling in fillings
                ]
                vertices_evaluated += 1
                for group_name, pairs in pair_groups.items():
                    for pair_idx, (n_abs, m_abs) in enumerate(pairs):
                        transition_ev = float(evals[m_abs] - evals[n_abs])
                        vertex_transition[group_name][i, j, pair_idx] = transition_ev
                        transition_min = min(transition_min, transition_ev)
                        transition_max = max(transition_max, transition_ev)
                        if transition_ev <= 0.0:
                            continue
                        geom_by_component = np.zeros(len(component_names), dtype=np.complex128)
                        if str(args.response_formula) == "wannierberri":
                            if wb_imn is None:
                                raise RuntimeError("Internal error: WannierBerri Imn was not precomputed")
                            for component_name, component in components.items():
                                comp_a, comp_b, comp_c = component
                                # Existing histogram-to-sigma pipeline applies sigma=Re[-i*C*integral].
                                # WannierBerri Imn is already the real integrand multiplying C, so store i*Imn.
                                geom_by_component[component_index[component_name]] = 1.0j * wb_imn[n_abs, m_abs, comp_a, comp_b, comp_c]
                        else:
                            r_mn = berry_connection_pair_from_D(D, evals, m_abs, n_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                            gd_nm = generalized_derivative_pair_from_D(D, evals, n_abs, m_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                            skipped_small_denominators += int(gd_nm.skipped_small_denominators)
                            for component_name, component in components.items():
                                comp_a, comp_b, comp_c = component
                                geom_by_component[component_index[component_name]] = (
                                    r_mn[comp_b] * gd_nm.values[comp_a, comp_c]
                                    + r_mn[comp_c] * gd_nm.values[comp_a, comp_b]
                                )
                        for fi, occupations in enumerate(occ_by_filling):
                            fnm = float(occupations[n_abs] - occupations[m_abs])
                            if bool(args.partial_occupation_triangles):
                                vertex_valid[group_name][fi, i, j, pair_idx] = True
                            elif abs(fnm) >= 1.0e-14:
                                vertex_valid[group_name][fi, i, j, pair_idx] = True
                            if abs(fnm) < 1.0e-14:
                                continue
                            for component_name in component_names:
                                attempted_weights += 1
                                ci = component_index[component_name]
                                weight = float(args.degeneracy) * fnm * geom_by_component[ci]
                                if np.isfinite(weight.real) and np.isfinite(weight.imag):
                                    vertex_weight[group_name][fi, i, j, pair_idx, ci] = complex(weight)
                                    kept_weights += 1

        for i in range(n):
            for j in range(n):
                for tri in tri_vertices:
                    coords = tuple((i + di, j + dj) for di, dj in tri)
                    for group_name, pairs in pair_groups.items():
                        gi = group_index[group_name]
                        for pair_idx in range(len(pairs)):
                            energies = np.asarray([vertex_transition[group_name][ii, jj, pair_idx] for ii, jj in coords], dtype=float)
                            if np.nanmax(energies) < float(args.emin) - 10.0 * eta_ev:
                                continue
                            if np.nanmin(energies) > float(args.emax) + 20.0 * eta_ev:
                                continue
                            for fi in range(len(fillings)):
                                if not all(vertex_valid[group_name][fi, ii, jj, pair_idx] for ii, jj in coords):
                                    continue
                                coeff = (
                                    np.mean(np.asarray([vertex_weight[group_name][fi, ii, jj, pair_idx] for ii, jj in coords]), axis=0)
                                    * inv_2pi_sq
                                )
                                if np.all(np.abs(coeff) <= 1.0e-30):
                                    continue
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

    energy_midpoints = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    transition_energy_moments: dict[str, dict[str, float | None]] = {}
    for fi, filling in enumerate(fillings):
        for group_name in group_names:
            gi = group_index[group_name]
            measure = np.real(transition_measure[fi, gi])
            total = float(np.sum(measure))
            key = f"nu={filling:g}|{group_name}"
            if total > 0.0:
                mean_ev = float(np.sum(measure * energy_midpoints) / total)
                rms_ev = float(np.sqrt(np.sum(measure * energy_midpoints * energy_midpoints) / total))
                transition_energy_moments[key] = {"measure_nm_inv_sq": total, "mean_ev": mean_ev, "rms_ev": rms_ev}
            else:
                transition_energy_moments[key] = {"measure_nm_inv_sq": 0.0, "mean_ev": None, "rms_ev": None}

    path, bands, band_slice = _compute_path_bands_b0(
        params,
        config,
        lg=lg,
        points_per_segment=int(args.path_points_per_segment),
        band_window=int(args.band_window),
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )
    summary = {
        "status": "corrected noninteracting reproduction using previous b0 BM model; Hartree/interacting not included",
        "paper": "Chaudhary, Lewandowski, Refael, arXiv:2107.09090 / 2021",
        "method": "Previous b0 BM Hamiltonian + sublattice offsets; selected response formula; triangular transition-energy histogram; analytic Lorentzian interval integration.",
        "config": config_summary(config, b0_params=params, lg=lg),
        "run": {
            "mesh_size": int(args.mesh_size),
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "n_c3_rotations": int(len(rotations)),
            "mu_mesh_size": int(mu_mesh),
            "eta_mev": float(args.eta_mev),
            "partial_occupation_triangles": bool(args.partial_occupation_triangles),
            "energy_bin_width_mev": float(args.energy_bin_width_mev),
            "emin_ev": float(args.emin),
            "emax_ev": float(args.emax),
            "n_energy": int(args.n_energy),
            "fillings_nu_or_mu_labels": [float(x) for x in fillings],
            "explicit_mu_mev": None if explicit_mu_mev is None else [float(x) for x in explicit_mu_mev],
            "response_degeneracy_multiplier": float(args.degeneracy),
            "filling_degeneracy_for_mu": float(args.filling_degeneracy),
            "fd_bands_each_side": int(args.fd_bands),
            "fd_mode": str(args.fd_mode),
            "denominator_cutoff_ev": float(args.denominator_cutoff_ev),
            "response_formula": str(args.response_formula),
            "sc_eta_mev": float(args.sc_eta_mev),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
        },
        "mu_by_filling_ev": {key: float(value) for key, value in mu_by_filling.items()},
        "flat_indices": [int(v_flat), int(c_flat)],
        "pair_groups": {
            group_name: {"pair_count": int(len(pairs)), "pairs": [[int(n_abs), int(m_abs)] for n_abs, m_abs in pairs]}
            for group_name, pairs in pair_groups.items()
        },
        "components": {name: component_label(component) for name, component in components.items()},
        "stats": {
            "vertices_evaluated": int(vertices_evaluated),
            "triangles_added": int(triangles_added),
            "attempted_weights": int(attempted_weights),
            "kept_weights": int(kept_weights),
            "skipped_small_denominators": int(skipped_small_denominators),
            "transition_min_ev": float(transition_min) if np.isfinite(transition_min) else None,
            "transition_max_ev": float(transition_max) if np.isfinite(transition_max) else None,
            "max_pair_count": int(max_pair_count),
        },
        "transition_energy_moments": transition_energy_moments,
        "peaks": peaks,
    }

    arrays: dict[str, np.ndarray] = {
        "photon_energies_ev": photon_energies,
        "energy_edges_ev": energy_edges,
        "path_kdist": path.kdist,
        "path_bands_ev": bands,
    }
    for fi, filling in enumerate(fillings):
        for group_name in group_names:
            gi = group_index[group_name]
            for component_name in component_names:
                ci = component_index[component_name]
                arrays[f"hist_{_safe_key(f'nu={filling:g}_{group_name}_{component_name}')}" ] = hist[fi, gi, ci]
                key = f"nu={filling:g}|{group_name}|{component_name}"
                arrays[f"sigma_{_safe_key(key)}"] = spectra[key]
            arrays[f"transition_measure_{_safe_key(f'nu={filling:g}_{group_name}')}" ] = transition_measure[fi, gi]
    np.savez(output_dir / "chaudhary2021_b0_noninteracting.npz", **arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_results(
        output_dir,
        photon_energies,
        spectra,
        fillings=fillings,
        group_names=group_names,
        component_names=component_names,
        eta_mev=float(args.eta_mev),
        path=path,
        bands=bands,
        band_slice=band_slice,
        summary=summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
