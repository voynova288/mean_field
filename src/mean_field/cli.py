from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmarks import (
    load_b0_parameter_references,
    load_b0_runtime_benchmarks,
    load_b0_suite,
    load_bm_unstrained_overlap_references,
    load_bm_unstrained_references,
    load_bm_unstrained_runtime_benchmarks,
)
from .runtime import collect_runtime_environment, current_timestamp, ensure_not_running_compute_on_login_node
from .systems.tmbg import diagnose_ktilde_symmetry, reproduce_paper_checkpoints
from .systems.tbg.zero_field import (
    export_overlap_diagnostics,
    run_b0_hf_benchmark_case,
    run_b0_hf_benchmark_suite,
    run_bm_unstrained,
    run_bm_unstrained_benchmark,
    write_b0_hf_benchmark_artifacts,
    write_b0_hf_suite_artifacts,
    write_bm_unstrained_benchmark_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mean-field")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench_parser = subparsers.add_parser("benchmarks", help="Inspect bundled benchmark suites.")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)
    bench_subparsers.add_parser("list", help="List zero-field benchmark cases.")
    nodes_parser = bench_subparsers.add_parser("nodes", help="Show benchmark path nodes for one case.")
    nodes_parser.add_argument("benchmark_id", help="Benchmark case identifier.")
    bench_subparsers.add_parser("parameters", help="List Julia-exported parameter reference rows.")
    bench_subparsers.add_parser("runtime-list", help="List bundled B0 runtime benchmark records.")
    bench_subparsers.add_parser("bm-list", help="List bundled BM unstrained references.")
    bench_subparsers.add_parser("bm-runtime-list", help="List bundled BM unstrained runtime benchmark records.")
    bench_subparsers.add_parser("bm-overlap-list", help="List bundled BM overlap diagnostics.")

    bm_parser = subparsers.add_parser("bm", help="Run the B=0 BM model.")
    bm_subparsers = bm_parser.add_subparsers(dest="bm_command", required=True)
    bm_compare = bm_subparsers.add_parser("compare-unstrained", help="Compare Python BM output with the unstrained Julia benchmark.")
    bm_compare.add_argument("theta_deg", type=float, help="Twist angle in degrees.")
    bm_benchmark = bm_subparsers.add_parser("benchmark-unstrained", help="Run the bundled BM unstrained benchmark with runtime and path-band outputs.")
    bm_benchmark.add_argument("theta_deg", type=float, help="Twist angle in degrees.")
    bm_benchmark.add_argument("--output-dir", type=Path, default=None, help="If set, write computed BM path outputs, runtime summaries, and band plots here.")
    bm_overlap = bm_subparsers.add_parser("compare-overlap", help="Compare compact BM overlap diagnostics with the Julia benchmark.")
    bm_overlap.add_argument("theta_deg", type=float, help="Twist angle in degrees.")
    bm_overlap.add_argument("lattice_kind", choices=("path", "grid"))
    bm_overlap.add_argument("m", type=int)
    bm_overlap.add_argument("n", type=int)

    hf_parser = subparsers.add_parser("hf", help="Run the B=0 HF benchmark workflow.")
    hf_subparsers = hf_parser.add_subparsers(dest="hf_command", required=True)
    hf_compare = hf_subparsers.add_parser("compare-case", help="Run one bundled B0 HF benchmark case and compare it with the reference path bands.")
    hf_compare.add_argument("benchmark_id", help="Benchmark case identifier.")
    hf_compare.add_argument("--lk", type=int, default=None, help="Override the BM/HF k-grid size.")
    hf_compare.add_argument("--lg", type=int, default=None, help="Override the reciprocal cutoff.")
    hf_compare.add_argument("--points-per-segment", type=int, default=None, help="Override the path resolution.")
    hf_compare.add_argument("--max-iter", type=int, default=300, help="Maximum restricted HF iterations.")
    hf_compare.add_argument("--precision", type=float, default=1e-5, help="Restricted HF convergence tolerance.")
    hf_compare.add_argument("--init-mode", default=None, help="Override the benchmark init mode.")
    hf_compare.add_argument("--seed", type=int, default=None, help="Override the benchmark seed.")
    hf_compare.add_argument("--overlap-lg", type=int, default=None, help="Override the overlap cutoff used inside HF.")
    hf_compare.add_argument(
        "--initial-density-path",
        type=Path,
        default=None,
        help="Explicit full-HF initial density TSV. Used for branch-continuation checks.",
    )
    hf_compare.add_argument("--output-dir", type=Path, default=None, help="If set, write computed HF path outputs and parity summaries here.")
    hf_suite = hf_subparsers.add_parser("compare-suite", help="Run the bundled B0 HF benchmark suite and summarize parity across cases.")
    hf_suite.add_argument("benchmark_ids", nargs="*", help="Optional benchmark ids. If omitted, run the full bundled suite.")
    hf_suite.add_argument("--lk", type=int, default=None, help="Override the BM/HF k-grid size.")
    hf_suite.add_argument("--lg", type=int, default=None, help="Override the reciprocal cutoff.")
    hf_suite.add_argument("--points-per-segment", type=int, default=None, help="Override the path resolution.")
    hf_suite.add_argument("--max-iter", type=int, default=300, help="Maximum restricted HF iterations.")
    hf_suite.add_argument("--precision", type=float, default=1e-5, help="Restricted HF convergence tolerance.")
    hf_suite.add_argument("--init-mode", default=None, help="Override the benchmark init mode.")
    hf_suite.add_argument("--seed", type=int, default=None, help="Override the benchmark seed.")
    hf_suite.add_argument("--overlap-lg", type=int, default=None, help="Override the overlap cutoff used inside HF.")
    hf_suite.add_argument("--output-dir", type=Path, default=None, help="If set, write per-case outputs and a suite summary here.")

    tmbg_parser = subparsers.add_parser("tmbg", help="Run the tMBG noninteracting workflow.")
    tmbg_subparsers = tmbg_parser.add_subparsers(dest="tmbg_command", required=True)
    tmbg_checkpoints = tmbg_subparsers.add_parser(
        "reproduce-checkpoints",
        help="Run the Park 2020 CP1-CP6 checkpoint orchestration.",
    )
    tmbg_checkpoints.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write the Fig. 2-like plots, markdown report, runtime summary, and JSON metadata here.",
    )
    tmbg_checkpoints.add_argument("--n-shells", type=int, default=5, help="Moire reciprocal-lattice shell cutoff.")
    tmbg_checkpoints.add_argument(
        "--points-per-segment",
        type=int,
        default=120,
        help="Path resolution per high-symmetry segment.",
    )
    tmbg_checkpoints.add_argument(
        "--path-n-bands",
        type=int,
        default=None,
        help="Optional override for the number of bands kept along the checkpoint path.",
    )
    tmbg_checkpoints.add_argument(
        "--topology-mesh-size",
        type=int,
        default=24,
        help="Uniform mesh size used by the topology checkpoints.",
    )
    tmbg_checkpoints.add_argument(
        "--topology-n-bands",
        type=int,
        default=None,
        help="Optional override for the number of bands retained in topology runs.",
    )
    tmbg_checkpoints.add_argument(
        "--bands-per-side",
        type=int,
        default=6,
        help="Number of bands kept on each side of the neutral flat-band pair in Fig. 2-like outputs.",
    )
    tmbg_checkpoints.add_argument(
        "--valley",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Valley label used for CP3/CP6 topology checks.",
    )
    tmbg_checkpoints.add_argument(
        "--skip-opposite-valley",
        action="store_true",
        help="Skip the K' sign-flip cross-check in CP3.",
    )
    tmbg_checkpoints.add_argument(
        "--cp4-delta-abs",
        type=float,
        default=0.06,
        help="Absolute interlayer-potential magnitude used by CP4.",
    )
    tmbg_checkpoints.add_argument(
        "--cp6-staggered-potential",
        dest="cp6_staggered_potentials",
        type=float,
        action="append",
        default=None,
        help="Repeat to override the sampled staggered potentials used by CP6.",
    )
    tmbg_diag = tmbg_subparsers.add_parser(
        "diagnose-ktilde-symmetry",
        help="Run the Ktilde chiral-limit and perturbation gap diagnostics.",
    )
    tmbg_diag.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write the markdown report, runtime summary, and JSON metadata here.",
    )
    tmbg_diag.add_argument("--theta-deg", type=float, default=1.21, help="Twist angle used by the Ktilde diagnostics.")
    tmbg_diag.add_argument("--n-shells", type=int, default=5, help="Moire reciprocal-lattice shell cutoff.")
    tmbg_diag.add_argument(
        "--valley",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Valley label used by the Ktilde diagnostics.",
    )
    return parser


def _write_key_value_summary(path: Path, entries: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def _ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    ensure_not_running_compute_on_login_node(workload_name)


def _tmbg_report_payload(report) -> dict[str, object]:
    return report.to_dict()


def cmd_benchmarks_list() -> int:
    suite = load_b0_suite()
    for case in suite.cases:
        print(
            f"{case.benchmark_id}\t"
            f"theta={case.theta_deg:.2f}\t"
            f"nu={case.nu}\t"
            f"state={case.state_label}\t"
            f"init={case.init_mode}\t"
            f"mu={case.mu_mev:.6f} meV"
        )
    return 0


def cmd_benchmarks_nodes(benchmark_id: str) -> int:
    suite = load_b0_suite()
    case = suite.get(benchmark_id)
    for node in case.load_reference_nodes():
        print(
            f"{node.label}\t"
            f"index={node.index}\t"
            f"k_dist={node.k_dist:.16f}\t"
            f"kx={node.kx:.16f}\t"
            f"ky={node.ky:.16f}"
        )
    return 0


def cmd_benchmarks_parameters() -> int:
    for row in load_b0_parameter_references():
        print(
            f"theta={row.theta_deg:.2f}\t"
            f"kb={row.kb:.16f}\t"
            f"g1=({row.g1.real:.16f},{row.g1.imag:.16f})\t"
            f"g2=({row.g2.real:.16f},{row.g2.imag:.16f})\t"
            f"Kt=({row.kt.real:.16f},{row.kt.imag:.16f})"
        )
    return 0


def cmd_benchmarks_runtime_list() -> int:
    for row in load_b0_runtime_benchmarks():
        print(
            f"{row.benchmark_id}\t"
            f"theta={row.theta_deg:.2f}\t"
            f"nu={row.nu}\t"
            f"init={row.init_mode}\t"
            f"lk={row.lk}\t"
            f"lg={row.lg}\t"
            f"total_elapsed_sec={row.total_elapsed_sec:.6f}\t"
            f"partition={row.slurm_partition}\t"
            f"cpus_per_task={row.slurm_cpus_per_task}"
        )
    return 0


def cmd_benchmarks_bm_list() -> int:
    for ref in load_bm_unstrained_references():
        summary = ref.load_summary()
        print(
            f"theta={ref.theta_deg:.2f}\t"
            f"path={summary.get('path', '?')}\t"
            f"points_per_segment={summary.get('points_per_segment', '?')}\t"
            f"grid_lk={summary.get('grid_lk', '?')}"
        )
    return 0


def cmd_benchmarks_bm_runtime_list() -> int:
    for row in load_bm_unstrained_runtime_benchmarks():
        print(
            f"theta={row.theta_deg:.2f}\t"
            f"points_per_segment={row.points_per_segment}\t"
            f"lg={row.lg}\t"
            f"grid_lk={row.grid_lk}\t"
            f"total_elapsed_sec={row.total_elapsed_sec:.6f}\t"
            f"partition={row.slurm_partition}\t"
            f"cpus_per_task={row.slurm_cpus_per_task}"
        )
    return 0


def cmd_benchmarks_bm_overlap_list() -> int:
    for row in load_bm_unstrained_overlap_references():
        print(
            f"theta={row.theta_deg:.2f}\t"
            f"lattice={row.lattice_kind}\t"
            f"valley={row.valley_label}\t"
            f"G=({row.m},{row.n})\t"
            f"fro_norm={row.fro_norm:.6e}"
        )
    return 0


def cmd_bm_compare_unstrained(theta_deg: float) -> int:
    refs = {ref.theta_deg: ref for ref in load_bm_unstrained_references()}
    rounded = round(theta_deg, 2)
    if rounded not in refs:
        raise SystemExit(f"No bundled BM unstrained reference for theta={theta_deg}")
    ref = refs[rounded]
    summary = ref.load_summary()
    points_per_segment = int(summary["points_per_segment"])
    lg = int(summary["lg"])
    run = run_bm_unstrained(rounded, points_per_segment=points_per_segment, lg=lg, grid_lk=0)
    ref_kdist, ref_energies = ref.load_path_data()
    ref_energy_array = np.asarray(ref_energies, dtype=float).T
    model_energy_array = run.path_solution.flattened_energies()
    max_err = float(np.max(np.abs(model_energy_array - ref_energy_array)))
    print(
        f"theta={rounded:.2f}\t"
        f"path_points={model_energy_array.shape[1]}\t"
        f"max_abs_path_energy_error_meV={max_err:.6e}\t"
        f"k_middle_gap_meV={run.k_middle_gap_mev:.12f}"
    )
    return 0


def cmd_bm_benchmark_unstrained(theta_deg: float, *, output_dir: Path | None = None) -> int:
    result = run_bm_unstrained_benchmark(theta_deg)
    output_suffix = ""
    if output_dir is not None:
        write_bm_unstrained_benchmark_artifacts(output_dir, result)
        output_suffix = f"\toutput_dir={output_dir}"

    total_ratio = None if result.runtime_parity is None else result.runtime_parity.total_elapsed_sec_ratio
    total_ratio_text = "n/a" if total_ratio is None else f"{total_ratio:.6e}"
    print(
        f"theta={result.reference.theta_deg:.2f}\t"
        f"path_points={result.run.path_solution.nk}\t"
        f"grid_points={0 if result.run.grid_solution is None else result.run.grid_solution.nk}\t"
        f"max_abs_path_energy_error_meV={result.parity.max_abs_band_diff_mev:.6e}\t"
        f"k_middle_gap_diff_meV={result.parity.k_middle_gap_diff_mev:.6e}\t"
        f"total_elapsed_sec={result.run.runtime.total_elapsed_sec:.6f}\t"
        f"total_elapsed_sec_ratio={total_ratio_text}"
        f"{output_suffix}"
    )
    return 0


def cmd_bm_compare_overlap(theta_deg: float, lattice_kind: str, m: int, n: int) -> int:
    rounded = round(theta_deg, 2)
    matches = [
        row
        for row in load_bm_unstrained_overlap_references()
        if round(row.theta_deg, 2) == rounded and row.lattice_kind == lattice_kind and row.m == m and row.n == n and row.valley_label == "K"
    ]
    if not matches:
        raise SystemExit(f"No overlap reference for theta={theta_deg}, lattice_kind={lattice_kind}, G=({m},{n})")
    ref = matches[0]
    diag = export_overlap_diagnostics(rounded, lattice_kind=lattice_kind, m=m, n=n, grid_lk=33 if lattice_kind == "grid" else 0)
    max_scalar_err = max(
        abs(diag.fro_norm - ref.fro_norm),
        abs(diag.max_abs - ref.max_abs),
        abs(diag.trace_real - ref.trace_real),
        abs(diag.trace_imag - ref.trace_imag),
        abs(diag.entry_11_real - ref.entry_11_real),
        abs(diag.entry_11_imag - ref.entry_11_imag),
        abs(diag.entry_mid_real - ref.entry_mid_real),
        abs(diag.entry_mid_imag - ref.entry_mid_imag),
    )
    print(
        f"theta={rounded:.2f}\t"
        f"lattice={lattice_kind}\t"
        f"G=({m},{n})\t"
        f"max_abs_scalar_error={max_scalar_err:.6e}"
    )
    return 0


def cmd_hf_compare_case(
    benchmark_id: str,
    *,
    lk: int | None = None,
    lg: int | None = None,
    points_per_segment: int | None = None,
    max_iter: int = 300,
    precision: float = 1e-5,
    init_mode: str | None = None,
    seed: int | None = None,
    overlap_lg: int | None = None,
    initial_density_path: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    case = load_b0_suite().get(benchmark_id)
    result = run_b0_hf_benchmark_case(
        case,
        lk=lk,
        lg=lg,
        points_per_segment=points_per_segment,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        overlap_lg=overlap_lg,
        precision=precision,
        initial_density_path=initial_density_path,
    )
    output_suffix = ""
    if output_dir is not None:
        write_b0_hf_benchmark_artifacts(output_dir, result)
        output_suffix = f"\toutput_dir={output_dir}"
    total_ratio = None if result.runtime_parity is None else result.runtime_parity.total_elapsed_sec_ratio
    total_ratio_text = "n/a" if total_ratio is None else f"{total_ratio:.6e}"
    print(
        f"benchmark_id={result.case.benchmark_id}\t"
        f"theta={result.case.theta_deg:.2f}\t"
        f"nu={result.case.nu}\t"
        f"nk={result.grid_solution.nk}\t"
        f"init={result.path_result.init_mode}\t"
        f"normalized_init={result.path_result.normalized_init_mode}\t"
        f"iterations={result.hf_run.iterations}\t"
        f"exit_reason={result.hf_run.exit_reason}\t"
        f"converged={result.hf_run.converged}\t"
        f"mu={result.path_result.mu:.12f}\t"
        f"total_elapsed_sec={result.runtime.total_elapsed_sec:.6f}\t"
        f"total_elapsed_sec_ratio={total_ratio_text}\t"
        f"kdist_max_abs_diff={result.parity.kdist_max_abs_diff:.6e}\t"
        f"max_abs_band_diff_meV={result.parity.max_abs_band_diff_mev:.6e}\t"
        f"rms_band_diff_meV={result.parity.rms_band_diff_mev:.6e}"
        f"{output_suffix}"
    )
    return 0


def cmd_hf_compare_suite(
    benchmark_ids: tuple[str, ...] | None = None,
    *,
    lk: int | None = None,
    lg: int | None = None,
    points_per_segment: int | None = None,
    max_iter: int = 300,
    precision: float = 1e-5,
    init_mode: str | None = None,
    seed: int | None = None,
    overlap_lg: int | None = None,
    output_dir: Path | None = None,
) -> int:
    suite_result = run_b0_hf_benchmark_suite(
        benchmark_ids=benchmark_ids,
        lk=lk,
        lg=lg,
        points_per_segment=points_per_segment,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        overlap_lg=overlap_lg,
        precision=precision,
    )
    output_suffix = ""
    if output_dir is not None:
        write_b0_hf_suite_artifacts(output_dir, suite_result)
        output_suffix = f"\toutput_dir={output_dir}"
    print(
        f"cases={len(suite_result.case_results)}\t"
        f"total_elapsed_sec={suite_result.total_elapsed_sec:.6f}\t"
        f"max_kdist_max_abs_diff={suite_result.max_kdist_max_abs_diff:.6e}\t"
        f"max_abs_band_diff_meV={suite_result.max_abs_band_diff_mev:.6e}"
        f"{output_suffix}"
    )
    return 0


def cmd_tmbg_reproduce_checkpoints(
    *,
    output_dir: Path | None = None,
    n_shells: int = 5,
    points_per_segment: int = 120,
    path_n_bands: int | None = None,
    topology_mesh_size: int = 24,
    topology_n_bands: int | None = None,
    bands_per_side: int = 6,
    valley: int = 1,
    verify_opposite_valley: bool = True,
    cp4_delta_abs: float = 0.06,
    cp6_staggered_potentials: tuple[float, ...] = (0.01, -0.01),
) -> int:
    _ensure_not_running_compute_on_login_node("tMBG checkpoints")

    start_time = current_timestamp()
    total_start = perf_counter()
    report = reproduce_paper_checkpoints(
        n_shells=n_shells,
        points_per_segment=points_per_segment,
        path_n_bands=path_n_bands,
        topology_mesh_size=topology_mesh_size,
        topology_n_bands=topology_n_bands,
        bands_per_side=bands_per_side,
        valley=valley,
        verify_opposite_valley=verify_opposite_valley,
        cp4_delta_abs=cp4_delta_abs,
        cp6_staggered_potentials=cp6_staggered_potentials,
        output_dir=output_dir,
    )
    total_elapsed = perf_counter() - total_start
    end_time = current_timestamp()
    env = collect_runtime_environment()

    output_suffix = ""
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        runtime_summary_path = resolved_output_dir / "runtime_summary.txt"
        metadata_path = resolved_output_dir / "run_metadata.json"
        report_path = resolved_output_dir / "paper_checkpoint_report.md"
        validation_report_path = resolved_output_dir / "validation_report.md"
        figure_png_path = resolved_output_dir / "fig2_like_bands.png"
        figure_pdf_path = resolved_output_dir / "fig2_like_bands.pdf"
        lattice_info_path = resolved_output_dir / "lattice_info.json"
        run_log_path = resolved_output_dir / "run.log"

        _write_key_value_summary(
            runtime_summary_path,
            [
                ("implementation", "python_tmbg"),
                ("runner_kind", "tmbg_paper_checkpoints"),
                ("n_shells", str(n_shells)),
                ("points_per_segment", str(points_per_segment)),
                ("path_n_bands", "" if path_n_bands is None else str(path_n_bands)),
                ("topology_mesh_size", str(topology_mesh_size)),
                ("topology_n_bands", "" if topology_n_bands is None else str(topology_n_bands)),
                ("bands_per_side", str(bands_per_side)),
                ("valley", str(valley)),
                ("verify_opposite_valley", str(bool(verify_opposite_valley)).lower()),
                ("cp4_delta_abs", str(cp4_delta_abs)),
                (
                    "cp6_staggered_potentials",
                    ",".join(f"{value:.16g}" for value in cp6_staggered_potentials),
                ),
                ("start_time", start_time),
                ("end_time", end_time),
                ("total_elapsed_sec", f"{total_elapsed:.16e}"),
                ("failure_count", str(report.failure_count)),
                ("skipped_count", str(report.skipped_count)),
                ("hostname", env.hostname),
                ("cpu_model", env.cpu_model),
                ("sys_cpu_threads", str(env.sys_cpu_threads)),
                ("blas_threads", str(env.blas_threads)),
                ("numba_threads", str(getattr(env, "numba_threads", ""))),
                ("backend_choice", str(getattr(env, "backend_choice", ""))),
                ("process_count", str(env.process_count)),
                ("jit_warmup_included", str(env.jit_warmup_included).lower()),
                ("slurm_partition", env.slurm_partition),
                ("slurm_nodelist", env.slurm_nodelist),
                ("slurm_cpus_per_task", str(env.slurm_cpus_per_task)),
                ("python_version", env.python_version),
                ("numpy_version", env.numpy_version),
                ("paper_checkpoint_report", str(report_path) if report_path.exists() else ""),
                ("validation_report", str(validation_report_path) if validation_report_path.exists() else ""),
                ("fig2_like_bands_png", str(figure_png_path) if figure_png_path.exists() else ""),
                ("fig2_like_bands_pdf", str(figure_pdf_path) if figure_pdf_path.exists() else ""),
                ("lattice_info_json", str(lattice_info_path) if lattice_info_path.exists() else ""),
                ("run_log", str(run_log_path) if run_log_path.exists() else ""),
            ],
        )
        metadata = {
            "implementation": "python_tmbg",
            "runner_kind": "tmbg_paper_checkpoints",
            "parameters": {
                "n_shells": n_shells,
                "points_per_segment": points_per_segment,
                "path_n_bands": path_n_bands,
                "topology_mesh_size": topology_mesh_size,
                "topology_n_bands": topology_n_bands,
                "bands_per_side": bands_per_side,
                "valley": valley,
                "verify_opposite_valley": bool(verify_opposite_valley),
                "cp4_delta_abs": cp4_delta_abs,
                "cp6_staggered_potentials": list(cp6_staggered_potentials),
            },
            "runtime": {
                "start_time": start_time,
                "end_time": end_time,
                "total_elapsed_sec": total_elapsed,
                "environment": {
                    "hostname": env.hostname,
                    "cpu_model": env.cpu_model,
                    "slurm_partition": env.slurm_partition,
                    "slurm_nodelist": env.slurm_nodelist,
                    "slurm_cpus_per_task": env.slurm_cpus_per_task,
                    "blas_threads": env.blas_threads,
                    "numba_threads": getattr(env, "numba_threads", None),
                    "sys_cpu_threads": env.sys_cpu_threads,
                    "process_count": env.process_count,
                    "backend_choice": getattr(env, "backend_choice", None),
                    "threadpoolctl_info": getattr(env, "threadpoolctl_info", ()),
                    "thread_env": getattr(env, "thread_env", {}),
                    "jit_warmup_included": env.jit_warmup_included,
                    "python_version": env.python_version,
                    "numpy_version": env.numpy_version,
                },
            },
            "artifacts": {
                "paper_checkpoint_report_md": str(report_path) if report_path.exists() else None,
                "validation_report_md": str(validation_report_path) if validation_report_path.exists() else None,
                "fig2_like_bands_png": str(figure_png_path) if figure_png_path.exists() else None,
                "fig2_like_bands_pdf": str(figure_pdf_path) if figure_pdf_path.exists() else None,
                "lattice_info_json": str(lattice_info_path) if lattice_info_path.exists() else None,
                "run_log": str(run_log_path) if run_log_path.exists() else None,
                "runtime_summary_txt": str(runtime_summary_path),
            },
            "report": _tmbg_report_payload(report),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        output_suffix = f"\toutput_dir={resolved_output_dir}"

    status = "fail" if report.has_failures else ("pass_with_skips" if report.has_skips else "pass")
    print(
        f"status={status}\t"
        f"checks={len(report.checks)}\t"
        f"failures={report.failure_count}\t"
        f"skipped={report.skipped_count}\t"
        f"total_elapsed_sec={total_elapsed:.6f}"
        f"{output_suffix}"
    )
    if report.has_failures:
        failed_checks = ",".join(check.name for check in report.checks if check.status == "fail")
        print(f"failed_checks={failed_checks}")
    return 0


def cmd_tmbg_diagnose_ktilde_symmetry(
    *,
    output_dir: Path | None = None,
    theta_deg: float = 1.21,
    n_shells: int = 5,
    valley: int = 1,
) -> int:
    _ensure_not_running_compute_on_login_node("tMBG Ktilde symmetry diagnostics")

    start_time = current_timestamp()
    total_start = perf_counter()
    report = diagnose_ktilde_symmetry(
        output_dir=output_dir,
        theta_deg=theta_deg,
        n_shells=n_shells,
        valley=valley,
    )
    total_elapsed = perf_counter() - total_start
    end_time = current_timestamp()
    env = collect_runtime_environment()

    output_suffix = ""
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        runtime_summary_path = resolved_output_dir / "runtime_summary.txt"
        metadata_path = resolved_output_dir / "run_metadata.json"
        report_path = resolved_output_dir / "ktilde_symmetry_report.md"

        _write_key_value_summary(
            runtime_summary_path,
            [
                ("implementation", "python_tmbg"),
                ("runner_kind", "tmbg_ktilde_symmetry_diagnostics"),
                ("theta_deg", f"{theta_deg:.16g}"),
                ("n_shells", str(n_shells)),
                ("valley", str(valley)),
                ("start_time", start_time),
                ("end_time", end_time),
                ("total_elapsed_sec", f"{total_elapsed:.16e}"),
                ("failure_count", str(report.failure_count)),
                ("skipped_count", str(report.skipped_count)),
                ("hostname", env.hostname),
                ("cpu_model", env.cpu_model),
                ("sys_cpu_threads", str(env.sys_cpu_threads)),
                ("blas_threads", str(env.blas_threads)),
                ("numba_threads", str(getattr(env, "numba_threads", ""))),
                ("backend_choice", str(getattr(env, "backend_choice", ""))),
                ("process_count", str(env.process_count)),
                ("jit_warmup_included", str(env.jit_warmup_included).lower()),
                ("slurm_partition", env.slurm_partition),
                ("slurm_nodelist", env.slurm_nodelist),
                ("slurm_cpus_per_task", str(env.slurm_cpus_per_task)),
                ("python_version", env.python_version),
                ("numpy_version", env.numpy_version),
                ("ktilde_symmetry_report", str(report_path) if report_path.exists() else ""),
            ],
        )
        metadata = {
            "implementation": "python_tmbg",
            "runner_kind": "tmbg_ktilde_symmetry_diagnostics",
            "parameters": {
                "theta_deg": theta_deg,
                "n_shells": n_shells,
                "valley": valley,
            },
            "runtime": {
                "start_time": start_time,
                "end_time": end_time,
                "total_elapsed_sec": total_elapsed,
                "environment": {
                    "hostname": env.hostname,
                    "cpu_model": env.cpu_model,
                    "slurm_partition": env.slurm_partition,
                    "slurm_nodelist": env.slurm_nodelist,
                    "slurm_cpus_per_task": env.slurm_cpus_per_task,
                    "blas_threads": env.blas_threads,
                    "numba_threads": getattr(env, "numba_threads", None),
                    "sys_cpu_threads": env.sys_cpu_threads,
                    "process_count": env.process_count,
                    "backend_choice": getattr(env, "backend_choice", None),
                    "threadpoolctl_info": getattr(env, "threadpoolctl_info", ()),
                    "thread_env": getattr(env, "thread_env", {}),
                    "jit_warmup_included": env.jit_warmup_included,
                    "python_version": env.python_version,
                    "numpy_version": env.numpy_version,
                },
            },
            "artifacts": {
                "ktilde_symmetry_report_md": str(report_path) if report_path.exists() else None,
                "runtime_summary_txt": str(runtime_summary_path),
            },
            "report": _tmbg_report_payload(report),
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        output_suffix = f"\toutput_dir={resolved_output_dir}"

    status = "fail" if report.has_failures else ("pass_with_skips" if report.has_skips else "pass")
    print(
        f"status={status}\t"
        f"checks={len(report.checks)}\t"
        f"failures={report.failure_count}\t"
        f"skipped={report.skipped_count}\t"
        f"total_elapsed_sec={total_elapsed:.6f}"
        f"{output_suffix}"
    )
    if report.has_failures:
        failed_checks = ",".join(check.name for check in report.checks if check.status == "fail")
        print(f"failed_checks={failed_checks}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmarks" and args.bench_command == "list":
        return cmd_benchmarks_list()
    if args.command == "benchmarks" and args.bench_command == "nodes":
        return cmd_benchmarks_nodes(args.benchmark_id)
    if args.command == "benchmarks" and args.bench_command == "parameters":
        return cmd_benchmarks_parameters()
    if args.command == "benchmarks" and args.bench_command == "runtime-list":
        return cmd_benchmarks_runtime_list()
    if args.command == "benchmarks" and args.bench_command == "bm-list":
        return cmd_benchmarks_bm_list()
    if args.command == "benchmarks" and args.bench_command == "bm-runtime-list":
        return cmd_benchmarks_bm_runtime_list()
    if args.command == "benchmarks" and args.bench_command == "bm-overlap-list":
        return cmd_benchmarks_bm_overlap_list()
    if args.command == "bm" and args.bm_command == "compare-unstrained":
        return cmd_bm_compare_unstrained(args.theta_deg)
    if args.command == "bm" and args.bm_command == "benchmark-unstrained":
        return cmd_bm_benchmark_unstrained(args.theta_deg, output_dir=args.output_dir)
    if args.command == "bm" and args.bm_command == "compare-overlap":
        return cmd_bm_compare_overlap(args.theta_deg, args.lattice_kind, args.m, args.n)
    if args.command == "hf" and args.hf_command == "compare-case":
        return cmd_hf_compare_case(
            args.benchmark_id,
            lk=args.lk,
            lg=args.lg,
            points_per_segment=args.points_per_segment,
            max_iter=args.max_iter,
            precision=args.precision,
            init_mode=args.init_mode,
            seed=args.seed,
            overlap_lg=args.overlap_lg,
            initial_density_path=args.initial_density_path,
            output_dir=args.output_dir,
        )
    if args.command == "hf" and args.hf_command == "compare-suite":
        return cmd_hf_compare_suite(
            benchmark_ids=tuple(args.benchmark_ids) if args.benchmark_ids else None,
            lk=args.lk,
            lg=args.lg,
            points_per_segment=args.points_per_segment,
            max_iter=args.max_iter,
            precision=args.precision,
            init_mode=args.init_mode,
            seed=args.seed,
            overlap_lg=args.overlap_lg,
            output_dir=args.output_dir,
        )
    if args.command == "tmbg" and args.tmbg_command == "reproduce-checkpoints":
        return cmd_tmbg_reproduce_checkpoints(
            output_dir=args.output_dir,
            n_shells=args.n_shells,
            points_per_segment=args.points_per_segment,
            path_n_bands=args.path_n_bands,
            topology_mesh_size=args.topology_mesh_size,
            topology_n_bands=args.topology_n_bands,
            bands_per_side=args.bands_per_side,
            valley=args.valley,
            verify_opposite_valley=not args.skip_opposite_valley,
            cp4_delta_abs=args.cp4_delta_abs,
            cp6_staggered_potentials=(
                (0.01, -0.01)
                if args.cp6_staggered_potentials is None
                else tuple(args.cp6_staggered_potentials)
            ),
        )
    if args.command == "tmbg" and args.tmbg_command == "diagnose-ktilde-symmetry":
        return cmd_tmbg_diagnose_ktilde_symmetry(
            output_dir=args.output_dir,
            theta_deg=args.theta_deg,
            n_shells=args.n_shells,
            valley=args.valley,
        )

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
