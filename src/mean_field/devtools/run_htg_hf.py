from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.core.io import write_text_artifact
from mean_field.devtools._runtime import (
    complex_to_pairs as _complex_to_pairs,
    ensure_not_running_compute_on_login_node,
    parse_csv_ints as _parse_csv_ints,
    parse_csv_strings as _parse_csv_strings,
    write_json,
)
from mean_field.workflows import collect_slurm_metadata
from mean_field.systems.htg import (
    HTGModel,
    HTGParams,
    InteractionParams,
    KWAN_2023_FERMI_VELOCITY_M_PER_S,
    KWAN_2023_TUNNELING_EV,
    build_htg_interaction_components,
    classify_htg_strong_coupling_state,
    evaluate_htg_hf_path,
    evaluate_htg_interaction_path,
    htg_flavor_occupation_counts_for_init_mode,
    htg_occupied_state_count,
    htg_validation_report,
    scan_htg_ground_state,
    validate_hf_state,
    write_htg_fig7_spin_resolved_plot,
    write_htg_fig8a_potential_plot,
    write_htg_hf_path_band_plot,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "HTG"
KWAN_2023_WAA_EV = 0.075
KWAN_2023_KAPPA = KWAN_2023_WAA_EV / KWAN_2023_TUNNELING_EV


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the projected Hartree-Fock HTG adapter for one parameter point.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.80)
    parser.add_argument("--kappa", type=float, default=KWAN_2023_KAPPA)
    parser.add_argument("--w-ev", type=float, default=KWAN_2023_TUNNELING_EV)
    parser.add_argument(
        "--w-aa-mev",
        type=float,
        default=None,
        help="AA tunneling in meV. When set, this overrides --kappa via kappa = wAA / wAB.",
    )
    parser.add_argument("--fermi-velocity-m-per-s", type=float, default=KWAN_2023_FERMI_VELOCITY_M_PER_S)
    parser.add_argument(
        "--include-pauli-twist",
        action="store_true",
        help="Use layer-rotated Dirac matrices sigma_{+theta,0,-theta}. Kwan Fig. 7 Eq. (1) omits this by default.",
    )
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--nu", type=float, default=2.0)
    parser.add_argument("--epsilon-r", type=float, default=8.0)
    parser.add_argument("--d-sc-nm", type=float, default=25.0)
    parser.add_argument("--u-ev", type=float, default=0.0)
    parser.add_argument("--n-k", type=int, default=6)
    parser.add_argument("--g-shells", type=int, default=1)
    parser.add_argument(
        "--projected-band-count",
        type=int,
        default=2,
        help="Number of projected bands per spin/valley flavor. Use 4 for flat bands plus nearest lower/upper remote bands, or 6 for two remote bands on each side.",
    )
    parser.add_argument("--finite-zero-limit", action="store_true")
    parser.add_argument("--zero-cutoff-nm-inv", type=float, default=1.0e-12)
    parser.add_argument("--init-modes", type=_parse_csv_strings, default=("sublattice", "fb", "fi", "bm", "perturbed"))
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(1, 2))
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--oda-stall-threshold", type=float, default=1.0e-3)
    parser.add_argument("--path-points-per-segment", type=int, default=48)
    parser.add_argument("--hf-band-window-mev", type=float, default=80.0)
    parser.add_argument("--skip-path-bands", action="store_true")
    parser.add_argument("--skip-potentials", action="store_true")
    parser.add_argument("--skip-potential-path-plot", action="store_true")
    parser.add_argument("--disable-numba", action="store_true")
    return parser.parse_args()


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"htg_hf_{job_id}"
    else:
        stem = f"htg_hf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem



def _run_payload(run) -> dict[str, object]:
    return {
        "init_mode": run.init_mode,
        "seed": int(run.seed),
        "converged": bool(run.converged),
        "exit_reason": run.exit_reason,
        "iterations": int(run.iterations),
        "final_error": float(run.iter_err[-1]) if run.iter_err.size else None,
        "final_energy_ev": float(run.state.diagnostics.get("hf_energy", np.nan)),
        "hf_gap_ev": float(run.state.diagnostics.get("hf_gap", np.nan)),
        "sector_gap_ev": float(run.state.diagnostics.get("sector_gap", np.nan)),
        "filling": float(run.state.diagnostics.get("filling", np.nan)),
        "projector_idempotency_residual": float(run.state.diagnostics.get("projector_idempotency_residual", np.nan)),
        "hamiltonian_hermitian_residual": float(run.state.diagnostics.get("hamiltonian_hermitian_residual", np.nan)),
        "occupied_sigma_z_mean": float(run.state.diagnostics.get("occupied_sigma_z_mean", np.nan)),
    }


def _occupation_constraint_payload(mode: str, *, nu: float, seed: int, n_band: int) -> list[int] | None:
    counts = htg_flavor_occupation_counts_for_init_mode(
        mode,
        nu=nu,
        seed=int(seed),
        n_band=int(n_band),
    )
    return None if counts is None else [int(value) for value in counts]


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    ensure_not_running_compute_on_login_node("HTG projected Hartree-Fock")
    if args.disable_numba:
        os.environ["MEAN_FIELD_HF_DISABLE_NUMBA"] = "1"

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    w_aa_ev = float(args.w_ev) * float(args.kappa)
    if args.w_aa_mev is not None:
        w_aa_ev = float(args.w_aa_mev) / 1000.0
    kappa = float(w_aa_ev / float(args.w_ev))
    zeta_rad = None if args.include_pauli_twist else 0.0

    params = HTGParams(
        fermi_velocity_m_per_s=args.fermi_velocity_m_per_s,
        w_ev=args.w_ev,
        kappa=kappa,
        zeta_rad=zeta_rad,
        model_name="kwan2023_hf",
    )
    interaction = InteractionParams(
        epsilon_r=args.epsilon_r,
        d_sc_nm=args.d_sc_nm,
        U_ev=args.u_ev,
        n_k=args.n_k,
        g_shells=args.g_shells,
        finite_zero_limit=args.finite_zero_limit,
        zero_cutoff_nm_inv=args.zero_cutoff_nm_inv,
    )
    model = HTGModel.from_config(args.theta_deg, n_shells=args.n_shells, params=params)
    scan = scan_htg_ground_state(
        model,
        interaction,
        nu=args.nu,
        init_modes=args.init_modes,
        seeds=args.seeds,
        beta=args.beta,
        max_iter=args.max_iter,
        precision=args.precision,
        oda_stall_threshold=args.oda_stall_threshold,
        projected_band_count=args.projected_band_count,
        use_numba=False if args.disable_numba else None,
    )
    best = scan.best_run
    elapsed = perf_counter() - start

    hf_params = {
        "theta_deg": float(args.theta_deg),
        "fermi_velocity_m_per_s": float(args.fermi_velocity_m_per_s),
        "vf_ev_nm": float(params.vf_ev_nm),
        "w_ev": float(args.w_ev),
        "kappa": float(kappa),
        "wAA_ev": float(w_aa_ev),
        "wAA_mev": float(1000.0 * w_aa_ev),
        "include_pauli_twist": bool(args.include_pauli_twist),
        "zeta_rad": None if zeta_rad is None else float(zeta_rad),
        "n_shells": int(args.n_shells),
        "nu": float(args.nu),
        "epsilon_r": float(args.epsilon_r),
        "d_sc_nm": float(args.d_sc_nm),
        "U_ev": float(args.u_ev),
        "finite_zero_limit": bool(args.finite_zero_limit),
        "drop_q0_coulomb": bool(not args.finite_zero_limit),
        "zero_cutoff_nm_inv": float(args.zero_cutoff_nm_inv),
        "n_k": int(args.n_k),
        "g_shells": int(args.g_shells),
        "projected_band_count": int(best.state.n_band),
        "central_band_indices": [int(index) for index in best.basis_data.central_band_indices],
        "projected_band_indices": [int(index) for index in best.basis_data.projected_band_indices],
        "init_modes": list(args.init_modes),
        "seeds": [int(seed) for seed in args.seeds],
        "flavor_occupation_constraints": {
            f"{mode}:seed{int(seed)}": _occupation_constraint_payload(
                mode,
                nu=args.nu,
                seed=int(seed),
                n_band=best.state.n_band,
            )
            for mode in args.init_modes
            for seed in args.seeds
        },
        "best_flavor_occupation_constraint": (
            None if best.state.occupation_counts is None else list(best.state.occupation_counts)
        ),
        "max_iter": int(args.max_iter),
        "precision": float(args.precision),
        "beta": float(args.beta),
        "oda_stall_threshold": float(args.oda_stall_threshold),
        "path_points_per_segment": int(args.path_points_per_segment),
        "skip_path_bands": bool(args.skip_path_bands),
        "skip_potentials": bool(args.skip_potentials),
        "skip_potential_path_plot": bool(args.skip_potential_path_plot),
        "disable_numba": bool(args.disable_numba),
        "lattice": model.lattice_summary(),
        "moire_cell_area_nm2": float(best.basis_data.moire_cell_area_nm2),
    }
    write_json(output_dir / "hf_params.json", hf_params)
    slurm_metadata = collect_slurm_metadata()
    runtime_metadata: dict[str, object] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "elapsed_sec": float(elapsed),
    }
    if slurm_metadata:
        runtime_metadata["slurm"] = slurm_metadata
    write_json(
        output_dir / "hf_convergence.json",
        {
            "runtime": runtime_metadata,
            "runs": [_run_payload(run) for run in scan.runs],
            "best": _run_payload(best),
        },
    )

    np.savez_compressed(
        output_dir / "hf_ground_state.npz",
        density=np.asarray(best.state.density, dtype=np.complex128),
        hamiltonian=np.asarray(best.state.hamiltonian, dtype=np.complex128),
        h0=np.asarray(best.state.h0, dtype=np.complex128),
        energies_ev=np.asarray(best.state.energies, dtype=float),
        kvec_nm_inv=_complex_to_pairs(best.basis_data.kvec),
        k_grid_frac=np.asarray(best.basis_data.k_grid_frac, dtype=float),
        iter_energy_ev=np.asarray(best.iter_energy, dtype=float),
        iter_err=np.asarray(best.iter_err, dtype=float),
        iter_oda=np.asarray(best.iter_oda, dtype=float),
    )

    path_artifacts: dict[str, str] = {}
    path_result = None
    path_band_gap_ev = None
    if not args.skip_path_bands:
        path_result = evaluate_htg_hf_path(
            best,
            points_per_segment=args.path_points_per_segment,
            beta=args.beta,
            use_numba=False if args.disable_numba else None,
        )
        np.savez_compressed(
            output_dir / "hf_bands_path.npz",
            energies_ev=np.asarray(path_result.energies, dtype=float),
            sigma_z_expectation=np.asarray(path_result.sigma_z_expectation, dtype=float),
            hamiltonian=np.asarray(path_result.hamiltonian, dtype=np.complex128),
            kdist=np.asarray(path_result.path.kdist, dtype=float),
            kvec=np.stack([path_result.path.kvec.real, path_result.path.kvec.imag], axis=-1),
            labels=np.asarray(path_result.path.labels),
            node_indices=np.asarray(path_result.path.node_indices, dtype=int),
            mu=float(path_result.mu),
            nu=float(path_result.nu),
        )
        plot_paths = write_htg_hf_path_band_plot(
            output_dir,
            path_result,
            stem="hf_bands_path",
            title="h-HTG projected HF bands",
            ylim=(-abs(args.hf_band_window_mev), abs(args.hf_band_window_mev)),
        )
        fig7_plot_paths = write_htg_fig7_spin_resolved_plot(
            output_dir,
            path_result,
            stem="fig7_spin_resolved_bands",
            title=rf"$\nu={args.nu:+.0f}$",
            ylim=(-abs(args.hf_band_window_mev), abs(args.hf_band_window_mev)),
        )
        path_artifacts = {
            "hf_bands_path_npz": str(output_dir / "hf_bands_path.npz"),
            "hf_bands_path_png": str(plot_paths["band_plot_png"]),
            "hf_bands_path_pdf": str(plot_paths["band_plot_pdf"]),
            "fig7_spin_resolved_png": str(fig7_plot_paths["fig7_plot_png"]),
            "fig7_spin_resolved_pdf": str(fig7_plot_paths["fig7_plot_pdf"]),
        }
        sorted_path_energies = np.sort(path_result.energies, axis=None)
        path_occupied = htg_occupied_state_count(path_result.nu, path_result.energies.shape[1], path_result.energies.shape[0])
        if 0 < path_occupied < sorted_path_energies.size:
            path_band_gap_ev = float(sorted_path_energies[path_occupied] - sorted_path_energies[path_occupied - 1])

    potential_artifact = ""
    potential_path_artifacts: dict[str, str] = {}
    if not args.skip_potentials:
        components = build_htg_interaction_components(
            best.state.density,
            best.overlap_blocks,
            v0=best.state.v0,
            beta=args.beta,
            use_numba=False if args.disable_numba else None,
        )
        np.savez_compressed(
            output_dir / "hartree_fock_potentials.npz",
            hartree=np.asarray(components.hartree, dtype=np.complex128),
            fock=np.asarray(components.fock, dtype=np.complex128),
            total=np.asarray(components.total, dtype=np.complex128),
            hartree_eigenvalues_ev=np.asarray(components.hartree_eigenvalues, dtype=float),
            fock_eigenvalues_ev=np.asarray(components.fock_eigenvalues, dtype=float),
            kvec_nm_inv=np.stack([best.basis_data.kvec.real, best.basis_data.kvec.imag], axis=-1),
        )
        potential_artifact = str(output_dir / "hartree_fock_potentials.npz")
        if not args.skip_potential_path_plot:
            potential_path_result = evaluate_htg_interaction_path(
                best,
                points_per_segment=args.path_points_per_segment,
                beta=args.beta,
                use_numba=False if args.disable_numba else None,
            )
            np.savez_compressed(
                output_dir / "fig8a_potential_path.npz",
                hartree=np.asarray(potential_path_result.hartree, dtype=np.complex128),
                fock=np.asarray(potential_path_result.fock, dtype=np.complex128),
                total=np.asarray(potential_path_result.total, dtype=np.complex128),
                hartree_diagonal_ev=np.asarray(potential_path_result.hartree_diagonal_ev, dtype=float),
                fock_diagonal_ev=np.asarray(potential_path_result.fock_diagonal_ev, dtype=float),
                total_diagonal_ev=np.asarray(potential_path_result.total_diagonal_ev, dtype=float),
                kdist=np.asarray(potential_path_result.path.kdist, dtype=float),
                kvec=np.stack([potential_path_result.path.kvec.real, potential_path_result.path.kvec.imag], axis=-1),
                labels=np.asarray(potential_path_result.path.labels),
                node_indices=np.asarray(potential_path_result.path.node_indices, dtype=int),
                nu=float(potential_path_result.nu),
            )
            potential_plot_paths = write_htg_fig8a_potential_plot(
                output_dir,
                potential_path_result,
                stem="fig8a_hartree_fock_potentials",
                title=rf"$\nu={args.nu:+.0f}$",
            )
            potential_path_artifacts = {
                "fig8a_potential_path_npz": str(output_dir / "fig8a_potential_path.npz"),
                "fig8a_potential_png": str(potential_plot_paths["potential_plot_png"]),
                "fig8a_potential_pdf": str(potential_plot_paths["potential_plot_pdf"]),
            }

    strong_coupling = classify_htg_strong_coupling_state(
        best.state.density,
        n_spin=best.state.n_spin,
        n_eta=best.state.n_eta,
        n_band=best.state.n_band,
    )

    write_json(
        output_dir / "order_parameters.json",
        {
            "nu": float(args.nu),
            "hf_gap_ev": float(best.state.diagnostics.get("hf_gap", np.nan)),
            "sector_gap_ev": float(best.state.diagnostics.get("sector_gap", np.nan)),
            "occupied_sigma_z_mean": float(best.state.diagnostics.get("occupied_sigma_z_mean", np.nan)),
            "filling": float(best.state.diagnostics.get("filling", np.nan)),
            "projector_idempotency_residual": float(best.state.diagnostics.get("projector_idempotency_residual", np.nan)),
            "hamiltonian_hermitian_residual": float(best.state.diagnostics.get("hamiltonian_hermitian_residual", np.nan)),
            "best_init_mode": best.init_mode,
            "best_seed": int(best.seed),
            "best_exit_reason": best.exit_reason,
            "best_converged": bool(best.converged),
            "projected_band_count": int(best.state.n_band),
            "path_band_gap_ev": path_band_gap_ev,
            "strong_coupling": strong_coupling.to_dict(),
        },
    )

    validation_checks = validate_hf_state(best.state)
    validation = [check.to_dict() for check in validation_checks]
    write_json(output_dir / "validation_checks.json", validation)
    validation_report = htg_validation_report("HTG Projected Hartree-Fock Validation", validation_checks)
    write_json(output_dir / "validation_report.json", validation_report.to_dict())
    write_text_artifact(validation_report.to_markdown() + "\n", output_dir / "validation_report.md")
    report_lines = [
        "# HTG Projected Hartree-Fock Run",
        "",
        "## Runtime",
        "",
        f"- `hostname = {socket.gethostname()}`",
        f"- `slurm_job_id = {os.environ.get('SLURM_JOB_ID', '')}`",
        f"- `elapsed_sec = {elapsed:.3f}`",
        "",
        "## Best Seed",
        "",
        f"- `init_mode = {best.init_mode}`",
        f"- `seed = {best.seed}`",
        f"- `converged = {best.converged}`",
        f"- `exit_reason = {best.exit_reason}`",
        f"- `iterations = {best.iterations}`",
        f"- `final_energy_ev = {best.state.diagnostics.get('hf_energy', np.nan)}`",
        f"- `hf_gap_ev = {best.state.diagnostics.get('hf_gap', np.nan)}`",
        f"- `strong_coupling_family = {strong_coupling.family}`",
        f"- `strong_coupling_class = {strong_coupling.class_label}`",
        f"- `projected_band_count = {best.state.n_band}`",
        f"- `hf_bands_path_npz = {path_artifacts.get('hf_bands_path_npz', '')}`",
            f"- `hf_bands_path_png = {path_artifacts.get('hf_bands_path_png', '')}`",
            f"- `hartree_fock_potentials_npz = {potential_artifact}`",
            f"- `fig8a_potential_path_npz = {potential_path_artifacts.get('fig8a_potential_path_npz', '')}`",
            f"- `fig8a_potential_png = {potential_path_artifacts.get('fig8a_potential_png', '')}`",
            "",
            "## Validation",
            "",
    ]
    for check in validation:
        report_lines.append(f"- `{check['name']} = {check['passed']}` (`value = {check['value']}`)")
    report_lines.append("")
    write_text_artifact("\n".join(report_lines), output_dir / "validation_report.md")

    print(f"[done] output_dir={output_dir}")
    print(f"best_init_mode={best.init_mode}")
    print(f"best_seed={best.seed}")
    print(f"best_exit_reason={best.exit_reason}")
    print(f"hf_convergence_json={output_dir / 'hf_convergence.json'}")
    print(f"validation_report_md={output_dir / 'validation_report.md'}")
    if path_artifacts:
        print(f"hf_bands_path_png={path_artifacts['hf_bands_path_png']}")
    if potential_artifact:
        print(f"hartree_fock_potentials_npz={potential_artifact}")
    if potential_path_artifacts:
        print(f"fig8a_potential_png={potential_path_artifacts['fig8a_potential_png']}")


if __name__ == "__main__":
    main()
