from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

from mean_field.systems.tbg.zero_field.model import _generate_gvec, _generate_t12, _generate_t12_zero_fill

from .chaudhary2021 import ChaudharyTBGConfig, b0_fig2_kpath, config_summary, make_b0_parameters
from .hartree import (
    build_hartree_b0_hamiltonian,
    arrays_to_rho,
    build_hartree_matrix_from_rho,
    rho_to_arrays,
    run_flat_hartree_scf,
)
from .run_chaudhary2021_noninteracting import _parse_float_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chaudhary 2021 Hartree-only continuum-band diagnostic.")
    parser.add_argument("--theta-deg", type=float, default=0.8)
    parser.add_argument("--lg", type=int, default=7)
    parser.add_argument("--mesh-size", type=int, default=10, help="uniform parallelogram mesh for Hartree density")
    parser.add_argument("--path-points-per-segment", type=int, default=80)
    parser.add_argument("--nus", type=_parse_float_csv, default=(-3.0, -2.0, 0.0, 2.0, 3.0))
    parser.add_argument("--delta1-mev", type=float, default=5.0)
    parser.add_argument("--delta2-mev", type=float, default=5.0)
    parser.add_argument("--w-ab-mev", type=float, default=90.0)
    parser.add_argument("--w-aa-ratio", type=float, default=0.4)
    parser.add_argument("--kinetic-ev", type=float, default=2.1354)
    parser.add_argument("--epsilon-r", type=float, default=15.0)
    parser.add_argument("--temperature-k", type=float, default=0.0, help="Fermi-Dirac temperature for Hartree SCF occupations; 0 keeps sharp T=0 occupations")
    parser.add_argument("--density-mode", choices=("flat", "full_delta_occ", "full_fixed_cnp"), default="flat", help="Hartree source density: central-flat occupation difference; full-basis doped-carrier density with same-Hamiltonian CNP occupation subtracted; or full occupied density minus fixed noninteracting CNP reference for screening-convention diagnostics")
    parser.add_argument("--hartree-shift-mode", choices=("all", "first_star"), default="all", help="Fourier components kept in the Hartree potential. Ref. 65 keeps only the first moire reciprocal star; previous diagnostics used all shifts supported by the truncated G grid.")
    parser.add_argument("--mixing", type=float, default=0.35)
    parser.add_argument("--precision", type=float, default=1.0e-7)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--initial-state-dir", type=Path, default=None, help="optional directory with hartree_state_*.npz files used as initial rho guesses, e.g. a converged finite-T run")
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1, help="valley plotted after the flavor-summed Hartree SCF")
    parser.add_argument("--bands-each-side", type=int, default=5, help="number of bands below/above central gap to plot")
    parser.add_argument("--no-sigma-rotation", action="store_true")
    parser.add_argument("--periodic-g-grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_tbg/chaudhary2021_hartree_bands_smoke"))
    return parser.parse_args()


def _hartree_state_filename(filling: float) -> str:
    key = f"nu={float(filling):g}"
    return f"hartree_state_{key.replace('=', '_').replace('-', 'm').replace('.', 'p')}.npz"


def _load_initial_rho(initial_state_dir: Path | None, filling: float) -> dict[tuple[int, int], complex] | None:
    if initial_state_dir is None:
        return None
    path = Path(initial_state_dir) / _hartree_state_filename(float(filling))
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return arrays_to_rho(data["rho_shifts"], data["rho_values"])


def _uniform_k_grid(params, mesh_size: int) -> np.ndarray:
    frac = np.arange(int(mesh_size), dtype=float) / float(mesh_size)
    return np.ravel(frac[:, None] * params.g1 + frac[None, :] * params.g2, order="F").astype(np.complex128)


def _path_energies(
    params,
    config: ChaudharyTBGConfig,
    *,
    lg: int,
    path,
    rho_q,
    epsilon_r: float,
    sigma_rotation: bool,
    periodic_g_grid: bool,
    bands_each_side: int,
) -> np.ndarray:
    dim = 4 * int(lg) * int(lg)
    center = dim // 2
    lo = max(0, center - int(bands_each_side))
    hi = min(dim, center + int(bands_each_side))
    gvec = _generate_gvec(params, int(lg))
    tunnel_builder = _generate_t12 if bool(periodic_g_grid) else _generate_t12_zero_fill
    tunnel = tunnel_builder(params, int(lg), int(config.valley))
    h_hartree = None
    if rho_q:
        h_hartree = build_hartree_matrix_from_rho(params, config, lg=int(lg), rho_q=rho_q, epsilon_r=float(epsilon_r))
    out = np.zeros((path.kvec.size, hi - lo), dtype=float)
    for ik, k in enumerate(path.kvec):
        h = build_hartree_b0_hamiltonian(
            complex(k),
            params,
            config,
            lg=int(lg),
            rho_q=None,
            epsilon_r=float(epsilon_r),
            sigma_rotation=bool(sigma_rotation),
            periodic_g_grid=bool(periodic_g_grid),
            gvec=gvec,
            tunnel=tunnel,
            hartree_matrix=h_hartree,
        )
        out[ik, :] = eigvalsh(h, subset_by_index=[lo, hi - 1], driver="evr")
    return out


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
    k_grid = _uniform_k_grid(params, int(args.mesh_size))
    path = b0_fig2_kpath(params, int(args.path_points_per_segment))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nonint = _path_energies(
        params,
        config,
        lg=int(args.lg),
        path=path,
        rho_q={},
        epsilon_r=float(args.epsilon_r),
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        bands_each_side=int(args.bands_each_side),
    )

    results = {}
    path_arrays = {"noninteracting": nonint}
    for nu in tuple(float(x) for x in args.nus):
        if abs(nu) < 1.0e-14:
            rho_q = {}
            scf_summary = {
                "nu": float(nu),
                "iterations": 0,
                "converged": True,
                "final_error": 0.0,
                "mu_ev": 0.0,
                "density_mode": str(args.density_mode),
                "hartree_shift_mode": str(args.hartree_shift_mode),
                "note": "By construction the density is measured relative to charge neutrality, so Hartree potential is zero at nu=0.",
            }
            iter_error = np.asarray([], dtype=float)
            iter_mu = np.asarray([], dtype=float)
        else:
            initial_rho = _load_initial_rho(args.initial_state_dir, float(nu))
            scf = run_flat_hartree_scf(
                k_grid,
                params,
                config,
                lg=int(args.lg),
                nu=float(nu),
                epsilon_r=float(args.epsilon_r),
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                precision=float(args.precision),
                sigma_rotation=sigma_rotation,
                periodic_g_grid=periodic_g_grid,
                temperature_k=float(args.temperature_k),
                initial_rho_q=initial_rho,
                density_mode=str(args.density_mode),
                hartree_shift_mode=str(args.hartree_shift_mode),
            )
            rho_q = scf.rho_q
            scf_summary = {
                "nu": float(nu),
                "iterations": int(scf.iterations),
                "converged": bool(scf.converged),
                "final_error": float(scf.final_error),
                "mu_ev": float(scf.mu_ev),
                "temperature_k": float(scf.temperature_k),
                "density_mode": str(scf.density_mode),
                "hartree_shift_mode": str(scf.hartree_shift_mode),
                "initial_rho_q_count": int(len(initial_rho or {})),
            }
            iter_error = scf.iter_error
            iter_mu = scf.iter_mu_ev
        energies = _path_energies(
            params,
            config,
            lg=int(args.lg),
            path=path,
            rho_q=rho_q,
            epsilon_r=float(args.epsilon_r),
            sigma_rotation=sigma_rotation,
            periodic_g_grid=periodic_g_grid,
            bands_each_side=int(args.bands_each_side),
        )
        key = f"nu={nu:g}"
        path_arrays[key] = energies
        shifts, values = rho_to_arrays(rho_q)
        np.savez_compressed(
            args.output_dir / f"hartree_state_{key.replace('=', '_').replace('-', 'm').replace('.', 'p')}.npz",
            rho_shifts=shifts,
            rho_values=values,
            iter_error=iter_error,
            iter_mu_ev=iter_mu,
            density_mode=np.asarray(str(args.density_mode)),
            hartree_shift_mode=np.asarray(str(args.hartree_shift_mode)),
            path_kdist=path.kdist,
            path_energies_ev=energies,
            noninteracting_path_energies_ev=nonint,
        )
        # Band-flattening diagnostics for central two selected bands.
        center_col = nonint.shape[1] // 2
        flat_cols = [center_col - 1, center_col]
        results[key] = {
            **scf_summary,
            "central_bandwidths_mev": [float((np.max(energies[:, c]) - np.min(energies[:, c])) * 1.0e3) for c in flat_cols],
            "nonint_central_bandwidths_mev": [float((np.max(nonint[:, c]) - np.min(nonint[:, c])) * 1.0e3) for c in flat_cols],
            "rho_q_count": int(len(rho_q)),
            "max_abs_rho_q": float(max((abs(v) for v in rho_q.values()), default=0.0)),
        }

    summary = {
        "status": "Hartree-only full-continuum potential; density_mode records the source convention and hartree_shift_mode records whether all G shifts or only the first reciprocal star are kept. Diagnostic, not yet final Chaudhary reproduction.",
        "config": config_summary(config, b0_params=params, lg=int(args.lg)),
        "run": {
            "lg": int(args.lg),
            "mesh_size": int(args.mesh_size),
            "path_points_per_segment": int(args.path_points_per_segment),
            "nus": [float(x) for x in args.nus],
            "epsilon_r": float(args.epsilon_r),
            "temperature_k": float(args.temperature_k),
            "density_mode": str(args.density_mode),
            "hartree_shift_mode": str(args.hartree_shift_mode),
            "mixing": float(args.mixing),
            "precision": float(args.precision),
            "max_iter": int(args.max_iter),
            "initial_state_dir": None if args.initial_state_dir is None else str(args.initial_state_dir),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
            "plotted_valley": int(args.valley),
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "hartree_path_bands.npz",
        path_kdist=path.kdist,
        path_labels=np.asarray(path.labels),
        path_node_indices=np.asarray(path.node_indices, dtype=int),
        **{name.replace("=", "_").replace("-", "m").replace(".", "p"): arr for name, arr in path_arrays.items()},
    )

    n_panels = len(path_arrays)
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.2, max(2.4, 2.05 * n_panels)), sharex=True, constrained_layout=True)
    if n_panels == 1:
        axes = np.asarray([axes])
    for ax, (name, energies) in zip(axes, path_arrays.items(), strict=True):
        center_ref = 0.5 * (np.max(nonint[:, nonint.shape[1] // 2 - 1]) + np.min(nonint[:, nonint.shape[1] // 2]))
        for ib in range(energies.shape[1]):
            ax.plot(path.kdist, (energies[:, ib] - center_ref) * 1.0e3, color="black", lw=0.75)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_ylabel("E (meV)")
        ax.set_title(name)
        for idx in path.node_indices:
            safe_idx = min(int(idx), path.kdist.size - 1)
            ax.axvline(path.kdist[safe_idx], color="0.85", lw=0.5)
    axes[-1].set_xticks([path.kdist[min(int(i), path.kdist.size - 1)] for i in path.node_indices])
    axes[-1].set_xticklabels(path.labels)
    fig.suptitle(f"Chaudhary Hartree-only bands, lg={args.lg}, mesh={args.mesh_size}, epsilon={args.epsilon_r:g}")
    fig.savefig(args.output_dir / "hartree_bands.png", dpi=220)
    fig.savefig(args.output_dir / "hartree_bands.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
