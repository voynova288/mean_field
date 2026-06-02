from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from analysis.shift_current_htg.constants import eta_mev_to_ev
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
    b0_mbz_area_nm_inv_sq,
    centered_flat_indices,
    config_summary,
    fd_transition_pairs,
    finite_difference_b0_dhdk,
    make_b0_parameters,
)
from .hartree import arrays_to_rho, build_hartree_b0_hamiltonian, build_hartree_matrix_from_rho
from .run_chaudhary2021_b0_noninteracting import _vertex_k_grid_b0
from .run_chaudhary2021_noninteracting import _add_triangle_to_hist, _parse_float_csv, _safe_key, _spectrum_from_histogram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hartree FD pair/region decomposition for Chaudhary 2021 Fig. 4 physics diagnostics.")
    p.add_argument("--hartree-dir", type=Path, required=True)
    p.add_argument("--theta-deg", type=float, default=0.8)
    p.add_argument("--lg", type=int, default=7)
    p.add_argument("--mesh-size", type=int, default=12)
    p.add_argument("--c3-symmetrize-grid", action="store_true")
    p.add_argument("--delta1-mev", type=float, default=5.0)
    p.add_argument("--delta2-mev", type=float, default=5.0)
    p.add_argument("--w-ab-mev", type=float, default=90.0)
    p.add_argument("--w-aa-ratio", type=float, default=0.4)
    p.add_argument("--kinetic-ev", type=float, default=2.1354)
    p.add_argument("--epsilon-r", type=float, default=10.0)
    p.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    p.add_argument("--degeneracy", type=float, default=1.0)
    p.add_argument("--filling", type=_parse_float_csv, default=(-3.95, -3.5, -2.0, 2.0, 3.5, 3.95))
    p.add_argument("--mu-mev", type=_parse_float_csv, default=None)
    p.add_argument("--occupation-temperature-k", type=float, default=15.0)
    p.add_argument("--eta-mev", type=float, default=2.0)
    p.add_argument("--emin", type=float, default=0.0)
    p.add_argument("--emax", type=float, default=0.16)
    p.add_argument("--n-energy", type=int, default=641)
    p.add_argument("--energy-bin-width-mev", type=float, default=0.25)
    p.add_argument("--fd-bands", type=int, default=10)
    p.add_argument("--fd-mode", choices=("same_side", "cross_gap", "all"), default="same_side")
    p.add_argument("--component", default="y;xx")
    p.add_argument("--gamma-radius-frac", type=float, default=0.23, help="region split radius as a fraction of |G| after reciprocal-lattice folding")
    p.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    p.add_argument("--finite-step-dimless", type=float, default=1.0e-6)
    p.add_argument("--no-sigma-rotation", action="store_true")
    p.add_argument("--periodic-g-grid", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_hartree_fd_decomposition"))
    return p.parse_args()


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


def _reciprocal_folded_gamma_distance(k: complex, g1: complex, g2: complex) -> float:
    return float(min(abs(complex(k) - (m * g1 + n * g2)) for m in (-1, 0, 1) for n in (-1, 0, 1)))


def _pair_label(pair: tuple[int, int], v_flat: int, c_flat: int) -> str:
    n_abs, m_abs = pair
    if m_abs == v_flat:
        return f"D{n_abs - v_flat:+d}->Fv"
    if n_abs == c_flat:
        return f"Fc->D{m_abs - c_flat:+d}"
    return f"{n_abs}->{m_abs}"


def main() -> None:
    args = parse_args()
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
    pairs = fd_transition_pairs(dim, n_fd_bands_each_side=int(args.fd_bands), mode=str(args.fd_mode))
    pair_labels = tuple(_pair_label(pair, v_flat, c_flat) for pair in pairs)
    component = parse_component(str(args.component))
    comp_a, comp_b, comp_c = component
    fillings = tuple(float(x) for x in args.filling)
    explicit_mu_mev = None if args.mu_mev is None else tuple(float(x) for x in args.mu_mev)
    if explicit_mu_mev is not None and len(explicit_mu_mev) != len(fillings):
        raise ValueError("--mu-mev must have one value per filling label")

    rho_by_filling = {}
    mu_by_filling = {}
    for i, filling in enumerate(fillings):
        rho, state_mu = _load_rho_and_mu(Path(args.hartree_dir), filling)
        key = f"{filling:g}"
        rho_by_filling[key] = rho
        mu_by_filling[key] = explicit_mu_mev[i] * 1.0e-3 if explicit_mu_mev is not None else state_mu

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
    bin_width_ev = float(args.energy_bin_width_mev) * 1.0e-3
    emax_hist = float(args.emax) + 20.0 * eta_ev + bin_width_ev
    n_bins = max(1, int(math.ceil(emax_hist / bin_width_ev)))
    energy_edges = np.arange(n_bins + 1, dtype=float) * bin_width_ev

    n_fill = len(fillings)
    n_pair = len(pairs)
    # hist_pair: filling, pair, bin.  hist_region: filling, region[gamma,outer], bin.
    hist_pair = np.zeros((n_fill, n_pair, n_bins), dtype=np.complex128)
    hist_region = np.zeros((n_fill, 2, n_bins), dtype=np.complex128)
    hist_pair_region = np.zeros((n_fill, n_pair, 2, n_bins), dtype=np.complex128)
    transition_measure_pair = np.zeros((n_fill, n_pair, n_bins), dtype=np.complex128)

    n = int(args.mesh_size)
    rotations = (0, 1, 2) if bool(args.c3_symmetrize_grid) else (0,)
    triangle_area = b0_mbz_area_nm_inv_sq(params, config) / (2.0 * n * n * len(rotations))
    inv_2pi_sq = 1.0 / (2.0 * np.pi) ** 2
    tri_vertices = (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1)))
    gamma_radius = float(args.gamma_radius_frac) * abs(complex(params.g1))

    vertices_evaluated = 0
    skipped_small_denominators = 0
    triangles_added = 0

    for rotation in rotations:
        k_grid = _vertex_k_grid_b0(params, n, rotation_multiple=int(rotation))
        vertex_transition = np.full((n_fill, n + 1, n + 1, n_pair), np.nan, dtype=float)
        vertex_weight = np.zeros((n_fill, n + 1, n + 1, n_pair), dtype=np.complex128)
        vertex_valid = np.zeros((n_fill, n + 1, n + 1, n_pair), dtype=bool)
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
                    occ = fermi_occupation(evals, mu_ev=mu, temperature_k=float(args.occupation_temperature_k))
                    vertices_evaluated += 1
                    for pair_idx, (n_abs, m_abs) in enumerate(pairs):
                        transition_ev = float(evals[m_abs] - evals[n_abs])
                        vertex_transition[fi, i, j, pair_idx] = transition_ev
                        if transition_ev <= 0.0:
                            continue
                        fnm = float(occ[n_abs] - occ[m_abs])
                        if abs(fnm) < 1.0e-14:
                            continue
                        r_mn = berry_connection_pair_from_D(D, evals, m_abs, n_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                        gd_nm = generalized_derivative_pair_from_D(D, evals, n_abs, m_abs, denominator_cutoff_ev=float(args.denominator_cutoff_ev))
                        skipped_small_denominators += int(gd_nm.skipped_small_denominators)
                        weight = float(args.degeneracy) * fnm * (r_mn[comp_b] * gd_nm.values[comp_a, comp_c] + r_mn[comp_c] * gd_nm.values[comp_a, comp_b])
                        if np.isfinite(weight.real) and np.isfinite(weight.imag):
                            vertex_valid[fi, i, j, pair_idx] = True
                            vertex_weight[fi, i, j, pair_idx] = complex(weight)

        for i in range(n):
            for j in range(n):
                for tri in tri_vertices:
                    coords = tuple((i + di, j + dj) for di, dj in tri)
                    centroid = sum((complex(k_grid[ii, jj]) for ii, jj in coords), 0.0j) / 3.0
                    region = 0 if _reciprocal_folded_gamma_distance(centroid, complex(params.g1), complex(params.g2)) <= gamma_radius else 1
                    for fi in range(n_fill):
                        for pair_idx in range(n_pair):
                            if not all(vertex_valid[fi, ii, jj, pair_idx] for ii, jj in coords):
                                continue
                            energies = np.asarray([vertex_transition[fi, ii, jj, pair_idx] for ii, jj in coords], dtype=float)
                            if np.nanmax(energies) < float(args.emin) - 10.0 * eta_ev:
                                continue
                            if np.nanmin(energies) > float(args.emax) + 20.0 * eta_ev:
                                continue
                            coeff = np.mean(np.asarray([vertex_weight[fi, ii, jj, pair_idx] for ii, jj in coords]), axis=0) * inv_2pi_sq
                            if abs(coeff) <= 1.0e-30:
                                continue
                            _add_triangle_to_hist(hist_pair[fi, pair_idx], energy_edges, energies=energies, coeff=coeff, triangle_area_nm_inv_sq=triangle_area)
                            _add_triangle_to_hist(hist_region[fi, region], energy_edges, energies=energies, coeff=coeff, triangle_area_nm_inv_sq=triangle_area)
                            _add_triangle_to_hist(hist_pair_region[fi, pair_idx, region], energy_edges, energies=energies, coeff=coeff, triangle_area_nm_inv_sq=triangle_area)
                            _add_triangle_to_hist(transition_measure_pair[fi, pair_idx], energy_edges, energies=energies, coeff=np.asarray(1.0 + 0.0j), triangle_area_nm_inv_sq=triangle_area)
                            triangles_added += 1

    spectra_pair = np.zeros((n_fill, n_pair, photon_energies.size), dtype=float)
    spectra_region = np.zeros((n_fill, 2, photon_energies.size), dtype=float)
    spectra_pair_region = np.zeros((n_fill, n_pair, 2, photon_energies.size), dtype=float)
    for fi in range(n_fill):
        for pi in range(n_pair):
            spectra_pair[fi, pi] = _spectrum_from_histogram(photon_energies, energy_edges, hist_pair[fi, pi], eta_ev=eta_ev)
            for ri in range(2):
                spectra_pair_region[fi, pi, ri] = _spectrum_from_histogram(photon_energies, energy_edges, hist_pair_region[fi, pi, ri], eta_ev=eta_ev)
        for ri in range(2):
            spectra_region[fi, ri] = _spectrum_from_histogram(photon_energies, energy_edges, hist_region[fi, ri], eta_ev=eta_ev)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "fd_decomposition.npz",
        photon_energies_ev=photon_energies,
        energy_edges_ev=energy_edges,
        fillings=np.asarray(fillings, dtype=float),
        pair_labels=np.asarray(pair_labels),
        pair_indices=np.asarray(pairs, dtype=int),
        hist_pair=hist_pair,
        hist_region=hist_region,
        hist_pair_region=hist_pair_region,
        transition_measure_pair=transition_measure_pair,
        spectra_pair=spectra_pair,
        spectra_region=spectra_region,
        spectra_pair_region=spectra_pair_region,
    )

    peaks = {}
    for fi, filling in enumerate(fillings):
        total = np.sum(spectra_pair[fi], axis=0)
        idx = int(np.argmax(np.abs(total)))
        peaks[f"nu={filling:g}|FD_total"] = {"energy_mev": float(photon_energies[idx] * 1e3), "value": float(total[idx]), "abs": float(abs(total[idx]))}
        for region_name, ri in (("gamma", 0), ("outer", 1)):
            arr = spectra_region[fi, ri]
            idx = int(np.argmax(np.abs(arr)))
            peaks[f"nu={filling:g}|{region_name}"] = {"energy_mev": float(photon_energies[idx] * 1e3), "value": float(arr[idx]), "abs": float(abs(arr[idx]))}
        pair_strength = []
        for pi, label in enumerate(pair_labels):
            arr = spectra_pair[fi, pi]
            idx = int(np.argmax(np.abs(arr)))
            pair_strength.append({"pair": str(label), "energy_mev": float(photon_energies[idx] * 1e3), "value": float(arr[idx]), "abs": float(abs(arr[idx]))})
        peaks[f"nu={filling:g}|top_pairs"] = sorted(pair_strength, key=lambda x: x["abs"], reverse=True)[:8]

    summary = {
        "status": "Hartree FD pair and gamma/outer region decomposition; diagnostic for Fig. 4 physics, not a new final reproduction.",
        "config": config_summary(config, b0_params=params, lg=lg),
        "hartree_dir": str(Path(args.hartree_dir)),
        "run": {
            "lg": lg,
            "mesh_size": int(args.mesh_size),
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "fillings": [float(x) for x in fillings],
            "mu_by_filling_ev": {k: float(v) for k, v in mu_by_filling.items()},
            "epsilon_r": float(args.epsilon_r),
            "eta_mev": float(args.eta_mev),
            "occupation_temperature_k": float(args.occupation_temperature_k),
            "fd_bands": int(args.fd_bands),
            "fd_mode": str(args.fd_mode),
            "component": str(args.component),
            "gamma_radius_frac": float(args.gamma_radius_frac),
            "gamma_radius_dimless": float(gamma_radius),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
        },
        "pair_labels": list(pair_labels),
        "pair_indices": [[int(a), int(b)] for a, b in pairs],
        "diagnostics": {
            "vertices_evaluated": int(vertices_evaluated),
            "triangles_added": int(triangles_added),
            "skipped_small_denominators": int(skipped_small_denominators),
        },
        "peaks": peaks,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Compact diagnostic plot: total/gamma/outer plus top pair heatmap-like curves for each filling.
    ncols = len(fillings)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.0), constrained_layout=True, squeeze=False)
    energy_mev = photon_energies * 1e3
    for fi, filling in enumerate(fillings):
        ax = axes[0, fi]
        total = np.sum(spectra_pair[fi], axis=0)
        ax.plot(energy_mev, total, color="black", lw=1.8, label="total")
        ax.plot(energy_mev, spectra_region[fi, 0], color="#2ca02c", lw=1.2, label="gamma")
        ax.plot(energy_mev, spectra_region[fi, 1], color="#ff7f0e", lw=1.2, label="outer")
        ax.axhline(0.0, color="0.65", lw=0.6)
        ax.set_title(rf"$\nu={filling:g}$")
        ax.set_xlim(float(args.emin) * 1e3, min(float(args.emax) * 1e3, 110.0))
        if fi == 0:
            ax.set_ylabel(r"$\sigma^{y;xx}$")
            ax.legend(fontsize=7)
        ax2 = axes[1, fi]
        order = np.argsort([np.max(np.abs(spectra_pair[fi, pi])) for pi in range(n_pair)])[::-1][:5]
        for pi in order:
            ax2.plot(energy_mev, spectra_pair[fi, pi], lw=1.1, label=str(pair_labels[pi]))
        ax2.axhline(0.0, color="0.65", lw=0.6)
        ax2.set_xlim(float(args.emin) * 1e3, min(float(args.emax) * 1e3, 110.0))
        ax2.set_xlabel(r"$\omega$ (meV)")
        if fi == 0:
            ax2.set_ylabel("top pair spectra")
        ax2.legend(fontsize=6)
    fig.suptitle(f"Hartree FD decomposition, eps={args.epsilon_r:g}, eta={args.eta_mev:g} meV")
    fig.savefig(args.output_dir / "fd_decomposition.png", dpi=220)
    fig.savefig(args.output_dir / "fd_decomposition.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
