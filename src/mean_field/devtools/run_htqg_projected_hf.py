from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.htqg.domains import domain_displacements
from mean_field.systems.htqg.hf import (
    HTQGInteractionSettings,
    HTQGProjectedHFConfig,
    build_htqg_overlap_blocks,
    build_htqg_projected_hf_data,
    gap_estimate,
    occupation_by_label,
    run_htqg_projected_hf,
)
from mean_field.systems.htqg.params import DEFAULT_THETA_DEG, HTQGParams


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _serialise_complex(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    return np.stack([arr.real, arr.imag], axis=-1)


def _safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _segment_integer_points(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    include_start: bool,
) -> list[tuple[int, int]]:
    di = int(end[0]) - int(start[0])
    dj = int(end[1]) - int(start[1])
    n_steps = math.gcd(abs(di), abs(dj))
    if n_steps <= 0:
        return [start] if include_start else []
    step = (di // n_steps, dj // n_steps)
    first = 0 if include_start else 1
    return [(int(start[0]) + t * step[0], int(start[1]) + t * step[1]) for t in range(first, n_steps + 1)]


def _build_exact_scf_path(data) -> dict[str, Any]:
    """Return the paper path as exact points on the saved SCF mesh.

    For the HTQG Fig.1 path convention used in this repository,
    ``Gamma -> q0 -> -q1 -> Gamma -> M(q0,-q1)`` is represented by the
    following unwrapped reciprocal-lattice fractional coordinates:

    - Gamma: ``(0, 0)``
    - q0: ``(-1/3, -1/3)``
    - -q1: ``(-2/3, 1/3)``
    - M: ``(-1/2, 0)``

    Therefore the path is exactly present on an unshifted mesh whose size is
    divisible by 6.  The returned flat indices point into the saved SCF-grid
    arrays; no off-grid target Hamiltonian is constructed here.
    """

    mesh = int(data.config.mesh_size)
    frac_shift = data.config.frac_shift
    if frac_shift is None:
        raise ValueError("Exact SCF-path plotting requires config.frac_shift=(0, 0), not the default shifted grid.")
    if abs(float(frac_shift[0])) > 1.0e-14 or abs(float(frac_shift[1])) > 1.0e-14:
        raise ValueError(
            "Exact SCF-path plotting requires an unshifted k mesh. "
            f"Got frac_shift={frac_shift!r}."
        )
    if mesh % 6 != 0:
        raise ValueError(f"Exact HTQG Gamma-kappa-kappa'-Gamma-M path requires mesh_size divisible by 6, got {mesh}.")

    nodes = [
        (0, 0),
        (-mesh // 3, -mesh // 3),
        (-2 * mesh // 3, mesh // 3),
        (0, 0),
        (-mesh // 2, 0),
    ]
    labels = ("Gamma", "kappa", "kappa_prime", "Gamma", "M")

    unwrapped: list[tuple[int, int]] = []
    node_indices: list[int] = [1]
    for iseg, (start, end) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        unwrapped.extend(_segment_integer_points(start, end, include_start=(iseg == 0)))
        node_indices.append(len(unwrapped))

    flat_indices = np.asarray([((i % mesh) * mesh + (j % mesh)) for i, j in unwrapped], dtype=int)
    mod_indices = np.asarray([((i % mesh), (j % mesh)) for i, j in unwrapped], dtype=int)
    unwrapped_indices = np.asarray(unwrapped, dtype=int)
    kvec = np.asarray(
        [(i / mesh) * data.lattice.b_m1 + (j / mesh) * data.lattice.b_m2 for i, j in unwrapped],
        dtype=np.complex128,
    )
    kdist = np.zeros((len(kvec),), dtype=float)
    if len(kvec) > 1:
        kdist[1:] = np.cumsum(np.abs(np.diff(kvec)))
    return {
        "source": "exact_saved_scf_grid_points",
        "mesh_size": mesh,
        "flat_indices": flat_indices,
        "mod_indices": mod_indices,
        "unwrapped_indices": unwrapped_indices,
        "kvec": kvec,
        "kdist": kdist,
        "node_indices": tuple(int(v) for v in node_indices),
        "node_distances": [float(kdist[int(v) - 1]) for v in node_indices],
        "labels": labels,
    }


def _save_run_npz(
    path: Path,
    *,
    result,
    path_info: dict[str, Any] | None,
    path_energies: np.ndarray,
    path_mu: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "density": np.asarray(result.state.density, dtype=np.complex128),
        "grid_energies_ev": np.asarray(result.state.energies, dtype=float),
        "grid_mu_ev": np.asarray(result.state.mu, dtype=float),
        "h0": np.asarray(result.data.h0, dtype=np.complex128),
        "k_grid_frac": np.asarray(result.data.k_grid_frac, dtype=float),
        "kvec_complex_pairs": _serialise_complex(result.data.kvec),
        "band_indices": np.asarray(result.data.band_indices, dtype=int),
        "label_json": np.asarray(json.dumps([label.to_dict() for label in result.data.labels], ensure_ascii=False)),
        "iter_err": np.asarray(result.run.iter_err, dtype=float),
        "iter_energy": np.asarray(result.run.iter_energy, dtype=float),
        "iter_oda": np.asarray(result.run.iter_oda, dtype=float),
        "path_energies_ev": np.asarray(path_energies, dtype=float),
        "path_mu_ev": np.asarray(path_mu, dtype=float),
    }
    if path_info is not None:
        arrays.update(
            {
                "path_source": np.asarray(str(path_info["source"])),
                "path_kdist": np.asarray(path_info["kdist"], dtype=float),
                "path_kvec_complex_pairs": _serialise_complex(path_info["kvec"]),
                "path_flat_indices": np.asarray(path_info["flat_indices"], dtype=int),
                "path_mod_indices": np.asarray(path_info["mod_indices"], dtype=int),
                "path_unwrapped_indices": np.asarray(path_info["unwrapped_indices"], dtype=int),
                "path_node_indices": np.asarray(path_info["node_indices"], dtype=int),
                "path_labels": np.asarray(path_info["labels"]),
            }
        )
    np.savez_compressed(path, **arrays)


def _plot_domain(domain_key: str, run_summaries: list[dict[str, Any]], output: Path, *, energy_window_mev: float) -> None:
    if not run_summaries or "path_kdist" not in run_summaries[0]:
        return
    plt = _load_pyplot()
    n = len(run_summaries)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, item in zip(axes, run_summaries, strict=True):
        kdist = np.asarray(item["path_kdist"], dtype=float)
        energies = (np.asarray(item["path_energies_ev"], dtype=float) - float(item["mu_ev"])) * 1000.0
        nocc = int(item["n_occupied_per_k"])
        for ib in range(energies.shape[0]):
            color = "tab:blue" if ib < nocc else "black"
            ax.plot(kdist, energies[ib], color=color, lw=0.8, marker=".", ms=2.0)
        for x in item["path_node_distances"]:
            ax.axvline(float(x), color="0.86", lw=0.5, zorder=0)
        ax.axhline(0.0, color="0.78", lw=0.6, zorder=0)
        ax.set_xticks([float(x) for x in item["path_node_distances"]])
        ax.set_xticklabels(
            [str(label).replace("Gamma", "Γ").replace("kappa_prime", "κ′").replace("kappa", "κ") for label in item["path_labels"]],
            fontsize=7,
        )
        ax.set_xlim(float(kdist[0]), float(kdist[-1]))
        ax.set_ylim(-float(energy_window_mev), float(energy_window_mev))
        gap = item.get("grid_gap_ev")
        gap_text = "metal" if gap is None or float(gap) <= 0.0 else f"grid gap={1000 * float(gap):.1f} meV"
        conv = "conv" if item.get("best_converged") else "not conv"
        ax.set_title(f"ν={item['filling']}\n{gap_text}", fontsize=9)
        ax.text(
            0.03,
            0.97,
            f"{item['best_init']} ({conv})\nE={item['energy_total_ev_per_cell']:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.68, "edgecolor": "none"},
        )
    axes[0].set_ylabel("HF energy - μ [meV]")
    fig.suptitle(f"HTQG {domain_key} projected HF: exact saved SCF-grid path", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HTQG central active-band spin/valley projected-HF filling scan.")
    parser.add_argument("--theta-deg", type=float, default=DEFAULT_THETA_DEG)
    parser.add_argument("--n-shells", type=int, default=6)
    parser.add_argument("--mesh-size", type=int, default=18)
    parser.add_argument(
        "--active-band-count",
        type=int,
        default=2,
        help="Number of contiguous continuum bands per valley/spin retained around charge neutrality. Use 4 for central flat pair plus nearest lower/upper bands.",
    )
    parser.add_argument("--g-shells", type=int, default=2)
    parser.add_argument("--epsilon-r", type=float, default=10.0)
    parser.add_argument("--d-sc-nm", type=float, default=25.0)
    parser.add_argument("--fillings", nargs="+", type=int, default=[-3, -2, -1, 0, 1, 2, 3])
    parser.add_argument("--domains", nargs="+", default=["alpha_beta_alpha", "alpha_beta_gamma"])
    parser.add_argument("--init-modes", nargs="+", default=["bm", "flavor", "valley_k", "valley_kprime"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--precision", type=float, default=1.0e-7)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--use-oda", action="store_true", help="Use the generic ODA machinery with the HTQG reference-aware delta kernel.")
    parser.add_argument(
        "--active-basis",
        choices=["auto", "energy", "sublattice_chern"],
        default="energy",
        help="Active two-band basis. Use energy for production; sublattice_chern is diagnostic until separately validated.",
    )
    parser.add_argument(
        "--frac-shift",
        nargs=2,
        type=float,
        metavar=("S1", "S2"),
        default=[0.0, 0.0],
        help="SCF mesh shift in build_moire_k_grid convention. Use 0 0 for exact high-symmetry SCF-path bands.",
    )
    parser.add_argument("--plot-mode", choices=["scf-path", "none"], default="scf-path")
    parser.add_argument("--energy-window-mev", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    ensure_not_running_compute_on_login_node("HTQG projected Hartree-Fock scan")

    params = HTQGParams.realistic(kappa=0.6)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "workflow": "run_htqg_projected_hf",
        "status": "projected_hf_scan_exact_scf_grid_bands",
        "important_conventions": {
            "active_space": "central active_band_count continuum bands in both valleys with spin degeneracy: active_band_count × 2 valleys × 2 spins states/k",
            "filling": "nu is relative to charge neutrality; n_occ_per_k = 2 valleys × 2 spins × (active_band_count/2) + nu",
            "hartree_reference": "charge_neutral",
            "fock_density": "absolute active-space density",
            "screening": "double-gate screened Coulomb V(q)=2π(e²/4πeps0)/(epsilon |q|) tanh(|q| d_sc)",
            "band_plot_rule": "HF bands are plotted only on exact saved SCF k-points; no off-grid target-path reconstruction is used.",
        },
        "theta_deg": float(args.theta_deg),
        "n_shells": int(args.n_shells),
        "mesh_size": int(args.mesh_size),
        "active_band_count": int(args.active_band_count),
        "g_shells": int(args.g_shells),
        "epsilon_r": float(args.epsilon_r),
        "d_sc_nm": float(args.d_sc_nm),
        "frac_shift": [float(v) for v in args.frac_shift],
        "use_oda": bool(args.use_oda),
        "active_basis": str(args.active_basis),
        "plot_mode": str(args.plot_mode),
        "fillings": [int(v) for v in args.fillings],
        "domains": {},
    }

    for domain_arg in args.domains:
        domain_runs: list[dict[str, Any]] = []
        for filling in args.fillings:
            config = HTQGProjectedHFConfig(
                theta_deg=float(args.theta_deg),
                n_shells=int(args.n_shells),
                mesh_size=int(args.mesh_size),
                active_band_count=int(args.active_band_count),
                domain=str(domain_arg),
                filling=int(filling),
                params=params,
                interaction=HTQGInteractionSettings(
                    epsilon_r=float(args.epsilon_r),
                    d_sc_nm=float(args.d_sc_nm),
                    g_shells=int(args.g_shells),
                ),
                precision=float(args.precision),
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                use_oda=bool(args.use_oda),
                active_basis=str(args.active_basis),
                frac_shift=(float(args.frac_shift[0]), float(args.frac_shift[1])),
            )
            data = build_htqg_projected_hf_data(config)
            domain_key = data.domain.key
            path_info = _build_exact_scf_path(data) if args.plot_mode == "scf-path" else None

            print(f"[htqg-hf] domain={domain_key} filling={filling} build overlaps", flush=True)
            overlap_blocks = build_htqg_overlap_blocks(data)
            candidates = []
            for init_mode in args.init_modes:
                print(f"[htqg-hf] domain={domain_key} filling={filling} init={init_mode}", flush=True)
                result = run_htqg_projected_hf(data, init_mode=str(init_mode), seed=int(args.seed), overlap_blocks=overlap_blocks)
                run_summary = result.to_summary_dict()
                run_summary["energy_total_ev_per_cell"] = float(result.energy_components["total"])
                candidates.append((float(result.energy_components["total"]), str(init_mode), result, run_summary))

            converged_candidates = [candidate for candidate in candidates if bool(candidate[3].get("converged"))]
            ranking_pool = converged_candidates if converged_candidates else candidates
            ranking_pool.sort(key=lambda item: item[0])
            _best_energy, best_init, best_result, best_summary = ranking_pool[0]

            if path_info is not None:
                path_energies = np.asarray(best_result.state.energies[:, np.asarray(path_info["flat_indices"], dtype=int)], dtype=float)
                path_mu = float(best_result.state.mu)
            else:
                path_energies = np.empty((best_result.data.nt, 0), dtype=float)
                path_mu = float(best_result.state.mu)

            npz_path = output_dir / domain_key / f"nu_{int(filling):+d}_best_{_safe_name(best_init)}.npz"
            _save_run_npz(npz_path, result=best_result, path_info=path_info, path_energies=path_energies, path_mu=path_mu)

            item = {
                "domain": domain_key,
                "filling": int(filling),
                "n_occupied_per_k": int(data.n_occupied_per_k),
                "best_init": best_init,
                "best_selected_from": "converged_candidates" if converged_candidates else "all_candidates_none_converged",
                "best_converged": bool(best_summary.get("converged")),
                "candidate_summaries": [candidate[3] for candidate in sorted(candidates, key=lambda item: item[0])],
                "best_summary": best_summary,
                "energy_total_ev_per_cell": float(best_result.energy_components["total"]),
                "energy_components_ev_per_cell": dict(best_result.energy_components),
                "grid_gap_ev": gap_estimate(best_result.state.energies, data.n_occupied_per_k),
                "mu_ev": float(best_result.state.mu),
                "occupation_by_label": occupation_by_label(data, best_result.state.density),
                "npz_path": str(npz_path),
            }
            if path_info is not None:
                item.update(
                    {
                        "path_source": str(path_info["source"]),
                        "path_kdist": np.asarray(path_info["kdist"], dtype=float).tolist(),
                        "path_energies_ev": np.asarray(path_energies, dtype=float).tolist(),
                        "path_labels": list(path_info["labels"]),
                        "path_node_indices": [int(v) for v in path_info["node_indices"]],
                        "path_node_distances": [float(v) for v in path_info["node_distances"]],
                        "path_flat_indices": np.asarray(path_info["flat_indices"], dtype=int).tolist(),
                        "path_mod_indices": np.asarray(path_info["mod_indices"], dtype=int).tolist(),
                        "path_unwrapped_indices": np.asarray(path_info["unwrapped_indices"], dtype=int).tolist(),
                    }
                )
            domain_runs.append(item)
            print(
                f"[htqg-hf] best domain={domain_key} filling={filling} init={best_init} "
                f"converged={item['best_converged']} selected_from={item['best_selected_from']} "
                f"E={item['energy_total_ev_per_cell']:.8f} grid_gap={item['grid_gap_ev']}",
                flush=True,
            )

        if domain_runs:
            domain_key = str(domain_runs[0]["domain"])
        else:
            lattice = build_htqg_projected_hf_data(HTQGProjectedHFConfig(domain=str(domain_arg), params=params)).lattice
            domain_key = domain_displacements(lattice, str(domain_arg)).key
        plot_path = output_dir / domain_key / "hf_scf_grid_path_bands"
        if args.plot_mode == "scf-path":
            _plot_domain(domain_key, domain_runs, plot_path, energy_window_mev=float(args.energy_window_mev))
        summary["domains"][domain_key] = {
            "runs": domain_runs,
            "plot_png": str(plot_path.with_suffix(".png")) if args.plot_mode == "scf-path" else None,
        }

    write_json(output_dir / "summary.json", summary)
    print(f"[htqg-hf] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
