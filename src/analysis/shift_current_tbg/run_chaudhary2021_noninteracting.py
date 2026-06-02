from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, eigvalsh

from mean_field.systems.atmg.lattice import build_kpath_from_nodes
from mean_field.systems.atmg.tbg import build_coupling_table

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

from .chaudhary2021 import (
    ChaudharyTBGConfig,
    build_chau_hamiltonian,
    centered_flat_indices,
    config_summary,
    fd_transition_pairs,
    flat_filling_to_mu,
    make_chau_lattice,
    sample_flat_energies_for_mu,
    validate_analytic_dhdk,
    analytic_dhdk,
)


def _parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def _all_component_labels() -> tuple[str, ...]:
    return ("y;xx",)


def _safe_key(text: str) -> str:
    return (
        str(text)
        .replace(";", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace("|", "_")
        .replace(".", "p")
    )


def _add_linear_density_segment_to_hist(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    e_left: float,
    e_right: float,
    slope: float,
    intercept: float,
    coeff: np.ndarray,
) -> None:
    if e_right <= e_left:
        return
    n_bins = int(edges.size - 1)
    start = max(0, int(np.searchsorted(edges, e_left, side="right") - 1))
    stop = min(n_bins - 1, int(np.searchsorted(edges, e_right, side="left")))
    for ibin in range(start, stop + 1):
        left = max(float(e_left), float(edges[ibin]))
        right = min(float(e_right), float(edges[ibin + 1]))
        if right <= left:
            continue
        density_integral = 0.5 * float(slope) * (right * right - left * left) + float(intercept) * (right - left)
        if density_integral != 0.0:
            hist[..., ibin] += coeff * density_integral


def _add_triangle_to_hist(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    energies: np.ndarray,
    coeff: np.ndarray,
    triangle_area_nm_inv_sq: float,
) -> None:
    e = np.sort(np.asarray(energies, dtype=float))
    e0, e1, e2 = float(e[0]), float(e[1]), float(e[2])
    area = float(triangle_area_nm_inv_sq)
    if e2 <= e0 + 1.0e-14:
        ibin = int(np.searchsorted(edges, 0.5 * (e0 + e2), side="right") - 1)
        if 0 <= ibin < hist.shape[-1]:
            hist[..., ibin] += coeff * area
        return
    if e1 > e0 + 1.0e-14:
        slope = 2.0 * area / ((e1 - e0) * (e2 - e0))
        _add_linear_density_segment_to_hist(
            hist,
            edges,
            e_left=e0,
            e_right=e1,
            slope=slope,
            intercept=-slope * e0,
            coeff=coeff,
        )
    if e2 > e1 + 1.0e-14:
        slope = -2.0 * area / ((e2 - e0) * (e2 - e1))
        _add_linear_density_segment_to_hist(
            hist,
            edges,
            e_left=e1,
            e_right=e2,
            slope=slope,
            intercept=-slope * e2,
            coeff=coeff,
        )


def _lorentzian_interval_integral(photon_energies_ev: np.ndarray, e_left: float, e_right: float, eta_ev: float) -> np.ndarray:
    photon = np.asarray(photon_energies_ev, dtype=float)
    eta = float(eta_ev)
    return (np.arctan((float(e_right) - photon) / eta) - np.arctan((float(e_left) - photon) / eta)) / np.pi


def _spectrum_from_histogram(
    photon_energies_ev: np.ndarray,
    energy_edges_ev: np.ndarray,
    coefficient_integrals: np.ndarray,
    *,
    eta_ev: float,
) -> np.ndarray:
    integral = np.zeros_like(photon_energies_ev, dtype=np.complex128)
    edges = np.asarray(energy_edges_ev, dtype=float)
    for ibin in range(edges.size - 1):
        width = float(edges[ibin + 1] - edges[ibin])
        if width <= 0.0:
            continue
        coeff_density = coefficient_integrals[ibin] / width
        if coeff_density == 0.0:
            continue
        integral += coeff_density * _lorentzian_interval_integral(
            photon_energies_ev,
            float(edges[ibin]),
            float(edges[ibin + 1]),
            float(eta_ev),
        )
    return sigma_from_integral(integral)


def _rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(math.cos(float(angle_rad)), math.sin(float(angle_rad)))


def _vertex_k_grid(lattice, mesh_size: int, *, rotation_multiple: int = 0) -> np.ndarray:
    n = int(mesh_size)
    f = np.linspace(0.0, 1.0, n + 1, dtype=float)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    k = np.asarray(f1 * lattice.g_m1 + f2 * lattice.g_m2, dtype=np.complex128)
    if int(rotation_multiple) == 0:
        return k
    angle = 2.0 * np.pi * float(rotation_multiple) / 3.0
    return np.asarray([_rotate_complex(value, angle) for value in k.reshape(-1)], dtype=np.complex128).reshape(k.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "First-pass reproduction of Chaudhary-Lewandowski-Refael 2021 noninteracting "
            "TBG shift-current results.  It uses the existing gauge-free shift-current core, "
            "linear transition-energy histograms, and analytic Lorentzian interval integration."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--mesh-size", type=int, default=18)
    parser.add_argument(
        "--c3-symmetrize-grid",
        action="store_true",
        help="Average three C3-rotated primitive-cell meshes. This is a quadrature symmetry restoration, not smoothing.",
    )
    parser.add_argument("--mu-mesh-size", type=int, default=None)
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354, help="paper hbar*v/a in eV")
    parser.add_argument("--dirac-sign", type=float, default=-1.0)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--degeneracy", type=float, default=1.0, help="response flavor degeneracy multiplier; Chaudhary paper plots are most consistent with per-flavor degeneracy=1")
    parser.add_argument("--filling-degeneracy", type=float, default=4.0, help="spin/valley degeneracy used to convert total TBG filling nu to chemical potential")
    parser.add_argument("--filling", type=_parse_float_csv, default=(-2.0, 0.0, 2.0, 3.0))
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.16)
    parser.add_argument("--n-energy", type=int, default=641)
    parser.add_argument("--energy-bin-width-mev", type=float, default=0.5)
    parser.add_argument("--fd-bands", type=int, default=10, help="dispersive bands on each side included in FD transitions")
    parser.add_argument(
        "--fd-mode",
        choices=("same_side", "cross_gap", "all"),
        default="same_side",
        help="Direct FD transition set; 'same_side' matches Chaudhary Fig. 2, 'all' is diagnostic only.",
    )
    parser.add_argument("--component", action="append", default=None, help="component like y;xx; repeatable")
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--finite-step", type=float, default=1.0e-6)
    parser.add_argument("--path-points-per-segment", type=int, default=80)
    parser.add_argument("--band-window", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_noninteracting_smoke"))
    return parser.parse_args()


def _compute_path_bands(lattice, config: ChaudharyTBGConfig, coupling_table, *, points_per_segment: int, band_window: int):
    nodes = (lattice.k_m, lattice.gamma_m, lattice.m_m, lattice.kprime_m)
    labels = ("K", "Gamma", "M", "K'")
    path = build_kpath_from_nodes(nodes, labels, int(points_per_segment))
    dim = 4 * int(lattice.n_g)
    center = dim // 2
    lo = max(0, center - int(band_window))
    hi = min(dim, center + int(band_window))
    bands = []
    for k in path.kvec:
        evals = eigvalsh(build_chau_hamiltonian(complex(k), lattice, config, coupling_table=coupling_table))
        bands.append(evals[lo:hi])
    return path, np.asarray(bands, dtype=float), (lo, hi)


def _plot_results(
    output_dir: Path,
    photon_energies: np.ndarray,
    spectra: dict[str, np.ndarray],
    *,
    fillings: tuple[float, ...],
    group_names: tuple[str, ...],
    component_names: tuple[str, ...],
    eta_mev: float,
    path,
    bands: np.ndarray,
    band_slice: tuple[int, int],
    summary: dict[str, object],
) -> None:
    fig, axes = plt.subplots(1 + len(group_names), 1, figsize=(8.0, 3.0 + 2.8 * len(group_names)), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax0 = axes[0]
    for ib in range(bands.shape[1]):
        ax0.plot(path.kdist, bands[:, ib], color="black", lw=0.9)
    for idx in path.node_indices:
        ax0.axvline(path.kdist[idx - 1], color="0.8", lw=0.8)
    ax0.axhline(0.0, color="0.55", lw=0.8)
    ax0.set_xticks([path.kdist[idx - 1] for idx in path.node_indices])
    ax0.set_xticklabels(path.labels)
    ax0.set_ylabel("E [eV]")
    ax0.set_title(
        "Chaudhary 2021 TBG noninteracting: "
        f"theta={summary['config']['theta_deg']}°, Delta=({summary['config']['delta1_ev']*1e3:.1f},"
        f"{summary['config']['delta2_ev']*1e3:.1f}) meV"
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fillings)))
    x_mev = 1.0e3 * photon_energies
    for ax, group in zip(axes[1:], group_names, strict=True):
        for component in component_names:
            for filling, color in zip(fillings, colors, strict=True):
                key = f"nu={filling:g}|{group}|{component}"
                if key not in spectra:
                    continue
                label = f"nu={filling:g}" if len(component_names) == 1 else f"{component}, nu={filling:g}"
                ax.plot(x_mev, spectra[key], lw=1.4, color=color, label=label)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel(r"$\sigma$ [$\mu$A nm V$^{-2}$]")
        ax.set_title(f"{group} contribution, eta={eta_mev:g} meV")
        ax.legend(fontsize=8, ncols=2)
    axes[-1].set_xlabel("photon energy [meV]")
    fig.savefig(output_dir / "chaudhary2021_noninteracting.png", dpi=220)
    fig.savefig(output_dir / "chaudhary2021_noninteracting.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ChaudharyTBGConfig(
        theta_deg=float(args.theta_deg),
        n_shells=int(args.n_shells),
        kinetic_ev=float(args.kinetic_ev),
        w_ab_ev=float(args.w_ab_mev) * 1.0e-3,
        w_aa_ratio=float(args.w_aa_ratio),
        delta1_ev=float(args.delta1_mev) * 1.0e-3,
        delta2_ev=float(args.delta2_mev) * 1.0e-3,
        valley=int(args.valley),
        dirac_sign=float(args.dirac_sign),
    )
    lattice = make_chau_lattice(config)
    coupling_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=config.valley)
    dhdk = analytic_dhdk(lattice, config)
    validation = validate_analytic_dhdk(
        0.0 + 0.0j,
        lattice,
        config,
        step_nm_inv=float(args.finite_step),
        coupling_table=coupling_table,
    )

    dim = 4 * int(lattice.n_g)
    v_flat, c_flat = centered_flat_indices(dim)
    pair_groups = {
        "FF": ((v_flat, c_flat),),
        "FD": fd_transition_pairs(dim, n_fd_bands_each_side=int(args.fd_bands), mode=str(args.fd_mode)),
    }
    group_names = tuple(pair_groups)
    component_text = tuple(args.component) if args.component is not None else _all_component_labels()
    components: dict[str, Component] = {text: parse_component(text) for text in component_text}
    component_names = tuple(components)
    component_index = {name: i for i, name in enumerate(component_names)}
    group_index = {name: i for i, name in enumerate(group_names)}
    fillings = tuple(float(x) for x in args.filling)

    mu_mesh = int(args.mu_mesh_size) if args.mu_mesh_size is not None else max(8, min(int(args.mesh_size), 18))
    flat_energies = sample_flat_energies_for_mu(lattice, config, mesh_size=mu_mesh, coupling_table=coupling_table)
    mu_by_filling = {
        f"{filling:g}": flat_filling_to_mu(flat_energies, filling, degeneracy=float(args.filling_degeneracy))
        for filling in fillings
    }

    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = eta_mev_to_ev(float(args.eta_mev))
    bin_width_ev = float(args.energy_bin_width_mev) * 1.0e-3
    emax_hist = float(args.emax) + 20.0 * eta_ev + bin_width_ev
    n_bins = max(1, int(math.ceil(emax_hist / bin_width_ev)))
    energy_edges = np.arange(n_bins + 1, dtype=float) * bin_width_ev
    hist = np.zeros((len(fillings), len(group_names), len(component_names), n_bins), dtype=np.complex128)
    transition_measure = np.zeros((len(fillings), len(group_names), n_bins), dtype=np.complex128)

    n = int(args.mesh_size)
    rotations = (0, 1, 2) if bool(args.c3_symmetrize_grid) else (0,)
    triangle_area = float(lattice.mbz_area) / (2.0 * n * n * len(rotations))
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
        k_grid = _vertex_k_grid(lattice, n, rotation_multiple=int(rotation))
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
                hmat = build_chau_hamiltonian(complex(k_grid[i, j]), lattice, config, coupling_table=coupling_table)
                evals, evecs = eigh(hmat)
                D = velocity_matrices(evecs, dhdk)
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
                        r_mn = berry_connection_pair_from_D(
                            D,
                            evals,
                            m_abs,
                            n_abs,
                            denominator_cutoff_ev=float(args.denominator_cutoff_ev),
                        )
                        gd_nm = generalized_derivative_pair_from_D(
                            D,
                            evals,
                            n_abs,
                            m_abs,
                            denominator_cutoff_ev=float(args.denominator_cutoff_ev),
                        )
                        skipped_small_denominators += int(gd_nm.skipped_small_denominators)
                        geom_by_component = np.zeros(len(component_names), dtype=np.complex128)
                        for component_name, component in components.items():
                            comp_a, comp_b, comp_c = component
                            geom_by_component[component_index[component_name]] = (
                                r_mn[comp_b] * gd_nm.values[comp_a, comp_c]
                                + r_mn[comp_c] * gd_nm.values[comp_a, comp_b]
                            )
                        for fi, occupations in enumerate(occ_by_filling):
                            fnm = float(occupations[n_abs] - occupations[m_abs])
                            if abs(fnm) < 1.0e-14:
                                continue
                            vertex_valid[group_name][fi, i, j, pair_idx] = True
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
                            energies = np.asarray(
                                [vertex_transition[group_name][ii, jj, pair_idx] for ii, jj in coords],
                                dtype=float,
                            )
                            if np.nanmax(energies) < float(args.emin) - 10.0 * eta_ev:
                                continue
                            if np.nanmin(energies) > float(args.emax) + 20.0 * eta_ev:
                                continue
                            for fi in range(len(fillings)):
                                if not all(vertex_valid[group_name][fi, ii, jj, pair_idx] for ii, jj in coords):
                                    continue
                                coeff = (
                                    np.mean(
                                        np.asarray(
                                            [vertex_weight[group_name][fi, ii, jj, pair_idx] for ii, jj in coords]
                                        ),
                                        axis=0,
                                    )
                                    * inv_2pi_sq
                                )
                                _add_triangle_to_hist(
                                    hist[fi, gi],
                                    energy_edges,
                                    energies=energies,
                                    coeff=coeff,
                                    triangle_area_nm_inv_sq=triangle_area,
                                )
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
                transition_energy_moments[key] = {
                    "measure_nm_inv_sq": total,
                    "mean_ev": mean_ev,
                    "rms_ev": rms_ev,
                }
            else:
                transition_energy_moments[key] = {
                    "measure_nm_inv_sq": 0.0,
                    "mean_ev": None,
                    "rms_ev": None,
                }

    path, bands, band_slice = _compute_path_bands(
        lattice,
        config,
        coupling_table,
        points_per_segment=int(args.path_points_per_segment),
        band_window=int(args.band_window),
    )

    summary = {
        "status": "first-pass noninteracting reproduction scaffold; Hartree/interacting figures are planned separately",
        "paper": "Chaudhary, Lewandowski, Refael, arXiv:2107.09090 / 2021",
        "method": "Gauge-free shift-current formula with linear transition-energy triangle histogram and analytic Lorentzian interval integration.",
        "config": config_summary(config, lattice),
        "run": {
            "mesh_size": int(args.mesh_size),
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "n_c3_rotations": int(len(rotations)),
            "mu_mesh_size": int(mu_mesh),
            "eta_mev": float(args.eta_mev),
            "energy_bin_width_mev": float(args.energy_bin_width_mev),
            "emin_ev": float(args.emin),
            "emax_ev": float(args.emax),
            "n_energy": int(args.n_energy),
            "fillings_nu": [float(x) for x in fillings],
            "response_degeneracy_multiplier": float(args.degeneracy),
            "filling_degeneracy_for_mu": float(args.filling_degeneracy),
            "fd_bands_each_side": int(args.fd_bands),
            "fd_mode": str(args.fd_mode),
            "denominator_cutoff_ev": float(args.denominator_cutoff_ev),
        },
        "mu_by_filling_ev": {key: float(value) for key, value in mu_by_filling.items()},
        "flat_indices": [int(v_flat), int(c_flat)],
        "pair_groups": {
            group_name: {
                "pair_count": int(len(pairs)),
                "pairs": [[int(n_abs), int(m_abs)] for n_abs, m_abs in pairs],
            }
            for group_name, pairs in pair_groups.items()
        },
        "components": {name: component_label(component) for name, component in components.items()},
        "dhdk_validation_at_gamma": {
            "max_abs_ev_nm": float(validation.max_abs_ev_nm),
            "max_abs_x_ev_nm": float(validation.max_abs_x_ev_nm),
            "max_abs_y_ev_nm": float(validation.max_abs_y_ev_nm),
            "passes_1e_minus_7": bool(validation.max_abs_ev_nm < 1.0e-7),
        },
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
    np.savez(output_dir / "chaudhary2021_noninteracting.npz", **arrays)
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
