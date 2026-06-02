from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .constants import eta_mev_to_ev
from .htg_adapter import (
    MaoHTGConfig,
    analytic_dhdk,
    build_mao_hamiltonian,
    make_mao_model,
    stacking_displacements,
    validate_analytic_dhdk,
)
from .response import (
    Component,
    berry_connection_pair_from_D,
    component_label,
    fermi_occupation,
    generalized_derivative_pair_from_D,
    parse_component,
    sigma_from_integral,
    velocity_matrices,
)
from .run_htg_bandpair_spectra import (
    PairSpec,
    PairWindowSpec,
    c3_symmetrize_k_points,
    parse_pair_spec,
    parse_pair_window_spec,
    resolve_relative_index,
    rotate_complex,
)


@dataclass(frozen=True)
class TriangleAccumulationStats:
    n_triangles: int
    n_vertices_evaluated: int
    attempted_weights: int
    kept_weights: int
    skipped_small_denominators: int
    transition_min_ev: float | None
    transition_max_ev: float | None


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


def _parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def _eta_tag(eta_mev: float) -> str:
    return f"eta_{float(eta_mev):g}meV".replace(".", "p")


def _all_component_labels() -> tuple[str, ...]:
    return ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "hTTG selected band-pair spectra using a binned linear-tetrahedron transition-energy "
            "histogram and analytic Lorentzian interval integration.  This removes fixed-grid "
            "resonant-shell wiggles without smoothing the plotted curves."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument(
        "--zeta-mode",
        choices=("mao", "layer"),
        default="mao",
        help=(
            "Dirac-block rotation convention. 'mao' uses zeta_rad=0 as printed in Mao Eq.(13); "
            "'layer' uses the code's layer-rotated Dirac cones zeta_rad=theta, a diagnostic for whether "
            "the numerical setup behind the paper used the standard BM layer rotations."
        ),
    )
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--mesh-size", type=int, default=18)
    parser.add_argument("--c3-symmetrize-grid", action="store_true")
    parser.add_argument("--eta-mev", type=_parse_float_csv, default=(1.0, 2.0))
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.12)
    parser.add_argument("--n-energy", type=int, default=481)
    parser.add_argument("--energy-bin-width-mev", type=float, default=0.25)
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--mu-ev", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=0.0)
    parser.add_argument("--component", action="append", default=None, help="component like x;yy; repeatable")
    parser.add_argument("--pair", action="append", type=parse_pair_spec, default=None)
    parser.add_argument("--pair-window", action="append", type=parse_pair_window_spec, default=None)
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--finite-step", type=float, default=1.0e-6)
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/htg_tetra_retry"))
    return parser.parse_args()


def _vertex_k_grid(lattice, mesh_size: int, rotation_multiple: int) -> np.ndarray:
    n = int(mesh_size)
    f = np.linspace(0.0, 1.0, n + 1, dtype=float)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    k = f1 * lattice.b_m1 + f2 * lattice.b_m2
    if int(rotation_multiple) == 0:
        return np.asarray(k, dtype=np.complex128)
    angle = 2.0 * np.pi * float(rotation_multiple) / 3.0
    return np.asarray([rotate_complex(value, angle) for value in k.reshape(-1)], dtype=np.complex128).reshape(k.shape)


def main() -> None:
    args = parse_args()
    component_text = tuple(args.component) if args.component is not None else _all_component_labels()
    components: dict[str, Component] = {text: parse_component(text) for text in component_text}
    pair_specs = tuple(args.pair) if args.pair is not None else tuple()
    window_specs = tuple(args.pair_window) if args.pair_window is not None else tuple()
    if not pair_specs and not window_specs:
        pair_specs = (PairSpec("central_flat", -1, 0),)
    pair_groups: dict[str, tuple[tuple[int, int], ...]] = {
        pair.name: ((int(pair.n_offset), int(pair.m_offset)),) for pair in pair_specs
    }
    for window in window_specs:
        pair_groups[window.name] = window.pair_offsets

    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=float(args.mass_mev) * 1.0e-3,
        domain=str(args.domain),
        zeta_rad=(None if str(args.zeta_mode) == "layer" else 0.0),
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(model.lattice, config.stacking, valley=config.valley, domain=config.domain)
    dhdk = analytic_dhdk(model, config)
    validation = validate_analytic_dhdk(
        0.0 + 0.0j,
        model,
        config,
        step_nm_inv=float(args.finite_step),
        d_top=d_top,
        d_bot=d_bot,
    )

    pair_absolute_indices = {
        group_name: tuple(
            (
                int(n_offset),
                int(m_offset),
                resolve_relative_index(model.matrix_dim, n_offset),
                resolve_relative_index(model.matrix_dim, m_offset),
            )
            for n_offset, m_offset in offsets
        )
        for group_name, offsets in pair_groups.items()
    }
    group_names = tuple(pair_groups)
    component_names = tuple(components)
    group_index = {name: i for i, name in enumerate(group_names)}
    component_index = {name: i for i, name in enumerate(component_names)}

    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    bin_width_ev = float(args.energy_bin_width_mev) * 1.0e-3
    emax_hist = float(args.emax) + 20.0 * max(eta_mev_to_ev(v) for v in args.eta_mev) + bin_width_ev
    n_bins = max(1, int(math.ceil(emax_hist / bin_width_ev)))
    energy_edges = np.arange(n_bins + 1, dtype=float) * bin_width_ev
    hist = np.zeros((len(group_names), len(component_names), n_bins), dtype=np.complex128)

    rotations = (0, 1, 2) if bool(args.c3_symmetrize_grid) else (0,)
    triangle_area = float(model.lattice.mbz_area) / (2.0 * int(args.mesh_size) * int(args.mesh_size) * len(rotations))
    inv_2pi_sq = 1.0 / (2.0 * np.pi) ** 2

    attempted = 0
    kept = 0
    skipped_total = 0
    transition_min = float("inf")
    transition_max = float("-inf")
    vertices_evaluated = 0
    triangles_added = 0

    # Per-rotation vertex arrays.  Store only selected transition energies and weights,
    # not full D matrices, so this stays manageable for shell-5 central-flat runs.
    n = int(args.mesh_size)
    for rotation in rotations:
        k_grid = _vertex_k_grid(model.lattice, n, rotation)
        vertex_transition = {
            group_name: np.full((n + 1, n + 1, len(group_pairs)), np.nan, dtype=float)
            for group_name, group_pairs in pair_absolute_indices.items()
        }
        vertex_weight = {
            group_name: np.zeros((n + 1, n + 1, len(group_pairs), len(component_names)), dtype=np.complex128)
            for group_name, group_pairs in pair_absolute_indices.items()
        }
        vertex_valid = {
            group_name: np.zeros((n + 1, n + 1, len(group_pairs)), dtype=bool)
            for group_name, group_pairs in pair_absolute_indices.items()
        }

        for i in range(n + 1):
            for j in range(n + 1):
                k_tilde = complex(k_grid[i, j])
                hmat = build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot)
                evals, evecs = eigh(hmat)
                occupations = fermi_occupation(evals, mu_ev=float(args.mu_ev), temperature_k=float(args.temperature_k))
                D = velocity_matrices(evecs, dhdk)
                vertices_evaluated += 1
                for group_name, group_pairs in pair_absolute_indices.items():
                    for pair_idx, (_n_offset, _m_offset, n_abs, m_abs) in enumerate(group_pairs):
                        transition_ev = float(evals[m_abs] - evals[n_abs])
                        transition_min = min(transition_min, transition_ev)
                        transition_max = max(transition_max, transition_ev)
                        vertex_transition[group_name][i, j, pair_idx] = transition_ev
                        if transition_ev <= 0.0:
                            continue
                        if abs(float(occupations[n_abs] - occupations[m_abs])) < 1.0e-14:
                            continue
                        vertex_valid[group_name][i, j, pair_idx] = True
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
                        skipped_total += int(gd_nm.skipped_small_denominators)
                        fnm = float(occupations[n_abs] - occupations[m_abs])
                        for component_name, component in components.items():
                            attempted += 1
                            comp_a, comp_b, comp_c = component
                            weight = fnm * (
                                r_mn[comp_b] * gd_nm.values[comp_a, comp_c]
                                + r_mn[comp_c] * gd_nm.values[comp_a, comp_b]
                            )
                            if np.isfinite(weight.real) and np.isfinite(weight.imag):
                                vertex_weight[group_name][i, j, pair_idx, component_index[component_name]] = complex(weight)
                                kept += 1

        tri_vertices = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1)))
        for i in range(n):
            for j in range(n):
                for tri in tri_vertices:
                    coords = tuple((i + di, j + dj) for di, dj in tri)
                    for group_name, group_pairs in pair_absolute_indices.items():
                        gi = group_index[group_name]
                        for pair_idx in range(len(group_pairs)):
                            if not all(vertex_valid[group_name][ii, jj, pair_idx] for ii, jj in coords):
                                continue
                            energies = np.asarray([vertex_transition[group_name][ii, jj, pair_idx] for ii, jj in coords], dtype=float)
                            if np.nanmax(energies) < float(args.emin) - 10.0 * max(eta_mev_to_ev(v) for v in args.eta_mev):
                                continue
                            if np.nanmin(energies) > float(args.emax) + 20.0 * max(eta_mev_to_ev(v) for v in args.eta_mev):
                                continue
                            coeff = (
                                np.mean(
                                    np.asarray([vertex_weight[group_name][ii, jj, pair_idx] for ii, jj in coords]),
                                    axis=0,
                                )
                                * inv_2pi_sq
                            )
                            _add_triangle_to_hist(
                                hist[gi],
                                energy_edges,
                                energies=energies,
                                coeff=coeff,
                                triangle_area_nm_inv_sq=triangle_area,
                            )
                            triangles_added += 1

    spectra: dict[str, np.ndarray] = {}
    peak_summary: dict[str, dict[str, float]] = {}
    for eta_mev in args.eta_mev:
        eta_ev = eta_mev_to_ev(float(eta_mev))
        eta_tag = _eta_tag(float(eta_mev))
        for group_name in group_names:
            gi = group_index[group_name]
            for component_name in component_names:
                ci = component_index[component_name]
                key = f"{group_name}|{component_name}|{eta_tag}"
                sigma = _spectrum_from_histogram(photon_energies, energy_edges, hist[gi, ci], eta_ev=eta_ev)
                spectra[key] = sigma
                if sigma.size:
                    peak_index = int(np.argmax(np.abs(sigma)))
                    peak_summary[key] = {
                        "max_abs_uA_nm_per_V2": float(np.max(np.abs(sigma))),
                        "energy_at_max_abs_ev": float(photon_energies[peak_index]),
                        "signed_value_at_max_abs_uA_nm_per_V2": float(sigma[peak_index]),
                    }

    transition_range = [float(transition_min), float(transition_max)] if np.isfinite(transition_min) else None
    summary = {
        "warning": "Selected-band-pair/window tetra-binned workflow. It removes resonant-shell wiggles but does not solve Mao axis/symmetry/component-label ambiguities.",
        "method": "Linear-tetrahedron transition-energy histogram over the primitive moire cell, optional C3-averaged rotated cells, and analytic Lorentzian interval integration.",
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "zeta_mode": str(args.zeta_mode),
            "zeta_rad": None if config.zeta_rad is None else float(config.zeta_rad),
            "mesh_size": int(args.mesh_size),
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "matrix_dim": int(model.matrix_dim),
            "eta_mev": [float(value) for value in args.eta_mev],
            "mass_ev": config.mass_ev,
            "mu_ev": float(args.mu_ev),
            "temperature_k": float(args.temperature_k),
            "denominator_cutoff_ev": float(args.denominator_cutoff_ev),
            "energy_bin_width_mev": float(args.energy_bin_width_mev),
            "n_energy": int(args.n_energy),
            "emin_ev": float(args.emin),
            "emax_ev": float(args.emax),
        },
        "dhdk_validation_at_gamma": {
            "max_abs_ev_nm": validation.max_abs_ev_nm,
            "passes_1e_minus_7": validation.max_abs_ev_nm < 1.0e-7,
        },
        "pair_groups": {
            group_name: {
                "pair_count": int(len(group_pairs)),
                "relative_offsets": [[int(n_offset), int(m_offset)] for n_offset, m_offset, _n_abs, _m_abs in group_pairs],
                "absolute_indices": [[int(n_abs), int(m_abs)] for _n_offset, _m_offset, n_abs, m_abs in group_pairs],
            }
            for group_name, group_pairs in pair_absolute_indices.items()
        },
        "components": {name: component_label(component) for name, component in components.items()},
        "stats": TriangleAccumulationStats(
            n_triangles=int(triangles_added),
            n_vertices_evaluated=int(vertices_evaluated),
            attempted_weights=int(attempted),
            kept_weights=int(kept),
            skipped_small_denominators=int(skipped_total),
            transition_min_ev=float(transition_min) if np.isfinite(transition_min) else None,
            transition_max_ev=float(transition_max) if np.isfinite(transition_max) else None,
        ).__dict__,
        "selected_transition_energy_range_ev": transition_range,
        "peaks": peak_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "photon_energies_ev": photon_energies,
        "energy_edges_ev": energy_edges,
    }
    for group_name in group_names:
        gi = group_index[group_name]
        for component_name in component_names:
            ci = component_index[component_name]
            arrays[f"hist_{_safe_key(f'{group_name}_{component_name}')}"] = hist[gi, ci]
    for key, sigma in spectra.items():
        arrays[f"sigma_{_safe_key(key)}"] = np.asarray(sigma, dtype=float)
    np.savez(args.output_dir / "htg_bandpair_spectra.npz", **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
