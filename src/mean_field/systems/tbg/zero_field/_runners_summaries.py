from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403

def _compare_bm_unstrained_to_reference(reference: BMUnstrainedReference, run: BMUnstrainedRun) -> BMUnstrainedParity:
    reference_kdist, reference_energies = reference.load_path_data()
    generated_kdist = np.asarray(run.path.kdist, dtype=float)
    reference_kdist_array = np.asarray(reference_kdist, dtype=float)
    generated_energies = np.asarray(run.path_solution.flattened_energies(), dtype=float).T
    reference_energy_array = np.asarray(reference_energies, dtype=float)
    if generated_energies.shape != reference_energy_array.shape:
        raise ValueError(
            f"Reference path point count mismatch: {reference_energy_array.shape} vs {generated_energies.shape}"
        )

    diff_array = np.sort(reference_energy_array, axis=1) - np.sort(generated_energies, axis=1)
    summary = reference.load_summary()
    ref_valence = float(summary["valence_bandwidth_meV"])
    ref_conduction = float(summary["conduction_bandwidth_meV"])
    valence_diff = None if run.valence_bandwidth_mev is None else run.valence_bandwidth_mev - ref_valence
    conduction_diff = None if run.conduction_bandwidth_mev is None else run.conduction_bandwidth_mev - ref_conduction

    return BMUnstrainedParity(
        kdist_max_abs_diff=float(np.max(np.abs(reference_kdist_array - generated_kdist))),
        max_abs_band_diff_mev=float(np.max(np.abs(diff_array))),
        rms_band_diff_mev=float(np.sqrt(np.mean(diff_array**2))),
        mean_abs_band_diff_mev=float(np.mean(np.abs(diff_array))),
        k_middle_gap_diff_mev=run.k_middle_gap_mev - float(summary["K_middle_gap_meV"]),
        valence_bandwidth_diff_mev=valence_diff,
        conduction_bandwidth_diff_mev=conduction_diff,
    )


def _write_bm_path_tsv(path: Path, run: BMUnstrainedRun) -> Path:
    energies = np.asarray(run.path_solution.flattened_energies(), dtype=float)
    with path.open("w", encoding="utf-8") as handle:
        for ik, kdist in enumerate(run.path.kdist):
            row = [f"{float(kdist):.16f}"]
            row.extend(f"{float(energies[ib, ik]):.16f}" for ib in range(energies.shape[0]))
            handle.write("\t".join(row) + "\n")
    return path


def _write_bm_nodes_tsv(path: Path, run: BMUnstrainedRun) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("label\tindex\tk_dist\tkx\tky\n")
        for node in run.path.nodes:
            handle.write(
                f"{node.label}\t{node.index}\t{node.k_dist:.16f}\t{node.kx:.16f}\t{node.ky:.16f}\n"
            )
    return path


def _write_bm_summary(path: Path, run: BMUnstrainedRun) -> Path:
    entries = [
        ("implementation", "python_b0"),
        ("theta_deg", f"{run.params.dtheta_rad * 180.0 / np.pi:.2f}"),
        ("strain", str(run.params.strain)),
        ("Da_meV", str(run.params.deformation_potential)),
        ("vf_meV", str(run.params.vf)),
        ("w0_meV", str(run.params.w0)),
        ("w1_meV", str(run.params.w1)),
        ("path", "M-K-Gamma-M"),
        ("path_mode", "packaged_reference_nodes"),
        ("points_per_segment", str(run.path.node_indices[1] - run.path.node_indices[0])),
        ("lg", str(run.path_solution.lg)),
        ("grid_lk", "0" if run.grid_solution is None else str(int(round(np.sqrt(run.grid_solution.nk))) - 1)),
        ("selected_M_real", str(run.path.nodes[0].kx)),
        ("selected_M_imag", str(run.path.nodes[0].ky)),
        ("K_real", str(run.path.nodes[1].kx)),
        ("K_imag", str(run.path.nodes[1].ky)),
        ("K_middle_gap_meV", str(run.k_middle_gap_mev)),
        ("valence_bandwidth_meV", "" if run.valence_bandwidth_mev is None else str(run.valence_bandwidth_mev)),
        ("conduction_bandwidth_meV", "" if run.conduction_bandwidth_mev is None else str(run.conduction_bandwidth_mev)),
    ]
    return _write_key_value_summary(path, entries)


def _write_bm_runtime_summary(path: Path, run: BMUnstrainedRun) -> Path:
    env = run.runtime.environment
    entries = [
        ("implementation", "python_b0"),
        ("theta_deg", f"{run.params.dtheta_rad * 180.0 / np.pi:.2f}"),
        ("points_per_segment", str(run.path.node_indices[1] - run.path.node_indices[0])),
        ("lg", str(run.path_solution.lg)),
        ("grid_lk", "0" if run.grid_solution is None else str(int(round(np.sqrt(run.grid_solution.nk))) - 1)),
        ("start_time", run.runtime.start_time),
        ("end_time", run.runtime.end_time),
        ("path_elapsed_sec", str(run.runtime.path_elapsed_sec)),
        ("grid_elapsed_sec", str(run.runtime.grid_elapsed_sec)),
        ("total_elapsed_sec", str(run.runtime.total_elapsed_sec)),
        ("hostname", env.hostname),
        ("cpu_model", env.cpu_model),
        ("sys_cpu_threads", str(env.sys_cpu_threads)),
        ("blas_threads", str(env.blas_threads)),
        ("process_count", str(env.process_count)),
        ("jit_warmup_included", str(env.jit_warmup_included).lower()),
        ("slurm_partition", env.slurm_partition),
        ("slurm_nodelist", env.slurm_nodelist),
        ("slurm_cpus_per_task", str(env.slurm_cpus_per_task)),
        ("python_version", env.python_version),
        ("numpy_version", env.numpy_version),
    ]
    return _write_key_value_summary(path, entries)


def _write_bm_runtime_parity_summary(
    path: Path,
    result: BMUnstrainedBenchmarkRun,
) -> Path | None:
    if result.runtime_reference is None or result.runtime_parity is None:
        return None
    entries = [
        ("implementation", "python_b0"),
        ("reference_impl", "b0_reference"),
        ("reference_path_elapsed_sec", str(result.runtime_reference.path_elapsed_sec)),
        ("reference_grid_elapsed_sec", str(result.runtime_reference.grid_elapsed_sec)),
        ("reference_total_elapsed_sec", str(result.runtime_reference.total_elapsed_sec)),
        ("path_elapsed_sec_delta", str(result.runtime_parity.path_elapsed_sec_delta)),
        ("path_elapsed_sec_ratio", _format_ratio(result.runtime_parity.path_elapsed_sec_ratio)),
        ("grid_elapsed_sec_delta", str(result.runtime_parity.grid_elapsed_sec_delta)),
        ("grid_elapsed_sec_ratio", _format_ratio(result.runtime_parity.grid_elapsed_sec_ratio)),
        ("total_elapsed_sec_delta", str(result.runtime_parity.total_elapsed_sec_delta)),
        ("total_elapsed_sec_ratio", _format_ratio(result.runtime_parity.total_elapsed_sec_ratio)),
    ]
    return _write_key_value_summary(path, entries)


def _write_hf_runtime_summary(path: Path, result: B0HFBenchmarkRun) -> Path:
    env = result.runtime.environment
    entries = [
        ("benchmark_id", result.case.benchmark_id),
        ("implementation", "python_b0"),
        ("theta_deg", f"{result.case.theta_deg:.2f}"),
        ("nu", str(result.case.nu)),
        ("init_mode", result.path_result.init_mode),
        ("normalized_init_mode", result.path_result.normalized_init_mode),
        ("seed", str(result.path_result.seed)),
        ("lk", str(result.path_result.lk)),
        ("lg", str(result.path_result.lg)),
        ("points_per_segment", str(result.path_result.points_per_segment)),
        ("start_time", result.runtime.start_time),
        ("end_time", result.runtime.end_time),
        ("bm_elapsed_sec", str(result.runtime.bm_elapsed_sec)),
        ("hf_elapsed_sec", str(result.runtime.hf_elapsed_sec)),
        ("path_elapsed_sec", str(result.runtime.path_elapsed_sec)),
        ("total_elapsed_sec", str(result.runtime.total_elapsed_sec)),
        ("hostname", env.hostname),
        ("cpu_model", env.cpu_model),
        ("sys_cpu_threads", str(env.sys_cpu_threads)),
        ("blas_threads", str(env.blas_threads)),
        ("process_count", str(env.process_count)),
        ("jit_warmup_included", str(env.jit_warmup_included).lower()),
        ("slurm_partition", env.slurm_partition),
        ("slurm_nodelist", env.slurm_nodelist),
        ("slurm_cpus_per_task", str(env.slurm_cpus_per_task)),
        ("python_version", env.python_version),
        ("numpy_version", env.numpy_version),
        ("path_exit_reason", result.path_result.exit_reason),
        (
            "initial_density_override_path",
            "" if result.initial_density_override_path is None else str(result.initial_density_override_path),
        ),
    ]
    return _write_key_value_summary(path, entries)


def _write_hf_runtime_parity_summary(path: Path, result: B0HFBenchmarkRun) -> Path | None:
    if result.runtime_reference is None or result.runtime_parity is None:
        return None

    entries = [
        ("benchmark_id", result.case.benchmark_id),
        ("implementation", "python_b0"),
        ("reference_impl", "b0_reference"),
        ("reference_bm_elapsed_sec", str(result.runtime_reference.bm_elapsed_sec)),
        ("reference_hf_elapsed_sec", str(result.runtime_reference.hf_elapsed_sec)),
        ("reference_path_elapsed_sec", str(result.runtime_reference.path_elapsed_sec)),
        ("reference_total_elapsed_sec", str(result.runtime_reference.total_elapsed_sec)),
        ("bm_elapsed_sec_delta", str(result.runtime_parity.bm_elapsed_sec_delta)),
        ("bm_elapsed_sec_ratio", _format_ratio(result.runtime_parity.bm_elapsed_sec_ratio)),
        ("hf_elapsed_sec_delta", str(result.runtime_parity.hf_elapsed_sec_delta)),
        ("hf_elapsed_sec_ratio", _format_ratio(result.runtime_parity.hf_elapsed_sec_ratio)),
        ("path_elapsed_sec_delta", str(result.runtime_parity.path_elapsed_sec_delta)),
        ("path_elapsed_sec_ratio", _format_ratio(result.runtime_parity.path_elapsed_sec_ratio)),
        ("total_elapsed_sec_delta", str(result.runtime_parity.total_elapsed_sec_delta)),
        ("total_elapsed_sec_ratio", _format_ratio(result.runtime_parity.total_elapsed_sec_ratio)),
    ]
    return _write_key_value_summary(path, entries)

__all__ = [name for name in globals() if not name.startswith('__')]
