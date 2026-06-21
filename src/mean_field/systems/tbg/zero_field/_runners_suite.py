from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403
from ._runners_summaries import *  # noqa: F401,F403
from ._runners_b0 import run_b0_hf_benchmark_case
from ._runners_artifacts import write_b0_hf_benchmark_artifacts

def run_b0_hf_benchmark_suite(
    benchmark_ids: tuple[str, ...] | None = None,
    **kwargs: object,
) -> B0HFBenchmarkSuiteResult:
    suite = load_b0_suite()
    if benchmark_ids is None:
        selected_cases = suite.cases
    else:
        selected_cases = tuple(suite.get(benchmark_id) for benchmark_id in benchmark_ids)

    lk_override = kwargs.get("lk")
    lg_override = kwargs.get("lg")
    points_override = kwargs.get("points_per_segment")
    init_override = kwargs.get("init_mode")
    seed_override = kwargs.get("seed")
    beta = float(kwargs.get("beta", 1.0))
    max_iter = int(kwargs.get("max_iter", 300))
    overlap_lg_override = kwargs.get("overlap_lg")
    precision = float(kwargs.get("precision", 1e-5))

    grid_solution_cache: dict[tuple[float, int, int, str], tuple[TBGParameters, BMSolution]] = {}
    overlap_cache: dict[tuple[tuple[float, int, int, str], int], HFOverlapBlockSet] = {}
    case_results: list[B0HFBenchmarkRun] = []

    for case in selected_cases:
        grid_lk = case.lk if lk_override is None else int(lk_override)
        bm_lg = case.lg if lg_override is None else int(lg_override)
        path_points = case.points_per_segment if points_override is None else int(points_override)
        requested_init_mode = case.init_mode if init_override is None else str(init_override)
        requested_seed = case.seed if seed_override is None else int(seed_override)
        overlap_grid_lg = bm_lg if overlap_lg_override is None else int(overlap_lg_override)
        runner_kind = _benchmark_runner_kind(case)
        initial_density_override = None

        start_time = current_timestamp()
        grid_key = _benchmark_grid_cache_key(case, lk=grid_lk, lg=bm_lg)
        cached_grid = grid_solution_cache.get(grid_key)
        if cached_grid is None:
            params = build_b0_reference_parameters(case.theta_deg)
            bm_start = perf_counter()
            grid_solution = _build_benchmark_grid_solution(case, params, lk=grid_lk, lg=bm_lg)
            bm_elapsed = perf_counter() - bm_start
            grid_solution_cache[grid_key] = (params, grid_solution)
        else:
            params, grid_solution = cached_grid
            bm_elapsed = 0.0

        if runner_kind == "full" and not _full_init_mode_uses_stochastic_initialization(requested_init_mode):
            nt = int(grid_solution.n_spin * grid_solution.n_eta * grid_solution.nb)
            initial_density_override = _load_required_full_init_density_override(
                case,
                init_mode=requested_init_mode,
                seed=requested_seed,
                nt=nt,
                nk=int(grid_solution.nk),
                lk=grid_lk,
            )

        overlap_key = (grid_key, overlap_grid_lg)
        cached_overlap = overlap_cache.get(overlap_key)
        if cached_overlap is None:
            overlap_start = perf_counter()
            overlap_blocks = build_overlap_block_set(grid_solution, lg=overlap_grid_lg)
            overlap_elapsed = perf_counter() - overlap_start
            overlap_cache[overlap_key] = overlap_blocks
        else:
            overlap_blocks = cached_overlap
            overlap_elapsed = 0.0

        hf_start = perf_counter()
        hf_run = _run_benchmark_hf_from_grid_solution(
            case,
            grid_solution,
            overlap_blocks,
            grid_lk=grid_lk,
            requested_init_mode=requested_init_mode,
            requested_seed=requested_seed,
            beta=beta,
            max_iter=max_iter,
            precision=precision,
            overlap_lg=overlap_grid_lg,
            runner_kind=runner_kind,
            initial_density_override=initial_density_override,
        )
        hf_elapsed = overlap_elapsed + (perf_counter() - hf_start)

        path_start = perf_counter()
        path = _build_packaged_or_native_hf_benchmark_kpath(case, params, path_points)
        path_result = evaluate_restricted_hf_path(
            hf_run,
            grid_solution,
            points_per_segment=path_points,
            lg=bm_lg,
            overlap_lg=overlap_grid_lg,
            beta=beta,
            init_mode=requested_init_mode,
            path=path,
        )
        path_elapsed = perf_counter() - path_start

        total_elapsed = bm_elapsed + hf_elapsed + path_elapsed
        end_time = current_timestamp()
        runtime = B0HFBenchmarkRuntime(
            start_time=start_time,
            end_time=end_time,
            bm_elapsed_sec=float(bm_elapsed),
            hf_elapsed_sec=float(hf_elapsed),
            path_elapsed_sec=float(path_elapsed),
            total_elapsed_sec=float(total_elapsed),
            environment=collect_runtime_environment(),
        )

        runtime_reference = _b0_hf_runtime_reference_map().get(case.benchmark_id)
        runtime_parity = None if runtime_reference is None else _compare_hf_runtime_to_reference(runtime, runtime_reference)
        parity = compare_hf_path_to_reference(case.load_reference_path(), path_result)
        case_results.append(
            B0HFBenchmarkRun(
                case=case,
                params=params,
                path=path,
                grid_solution=grid_solution,
                hf_run=hf_run,
                path_result=path_result,
                parity=parity,
                runtime=runtime,
                runtime_reference=runtime_reference,
                runtime_parity=runtime_parity,
            )
        )

    return B0HFBenchmarkSuiteResult(case_results=tuple(case_results))


def write_b0_hf_suite_summary(path: Path | str, suite_result: B0HFBenchmarkSuiteResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "benchmark_id",
                    "theta_deg",
                    "nu",
                    "init_mode",
                    "normalized_init_mode",
                    "iterations",
                    "exit_reason",
                    "converged",
                    "mu_mev",
                    "bm_elapsed_sec",
                    "hf_elapsed_sec",
                    "path_elapsed_sec",
                    "total_elapsed_sec",
                    "total_elapsed_sec_ratio",
                    "kdist_max_abs_diff",
                    "max_abs_band_diff_mev",
                    "rms_band_diff_mev",
                    "mean_abs_band_diff_mev",
                ]
            )
            + "\n"
        )
        for result in suite_result.case_results:
            total_ratio = None if result.runtime_parity is None else result.runtime_parity.total_elapsed_sec_ratio
            handle.write(
                "\t".join(
                    [
                        result.case.benchmark_id,
                        f"{result.case.theta_deg:.2f}",
                        str(result.case.nu),
                        result.path_result.init_mode,
                        result.path_result.normalized_init_mode,
                        str(result.hf_run.iterations),
                        result.hf_run.exit_reason,
                        str(result.hf_run.converged),
                        f"{result.path_result.mu:.16f}",
                        f"{result.runtime.bm_elapsed_sec:.16e}",
                        f"{result.runtime.hf_elapsed_sec:.16e}",
                        f"{result.runtime.path_elapsed_sec:.16e}",
                        f"{result.runtime.total_elapsed_sec:.16e}",
                        "" if total_ratio is None else f"{total_ratio:.16e}",
                        f"{result.parity.kdist_max_abs_diff:.16e}",
                        f"{result.parity.max_abs_band_diff_mev:.16e}",
                        f"{result.parity.rms_band_diff_mev:.16e}",
                        f"{result.parity.mean_abs_band_diff_mev:.16e}",
                    ]
                )
                + "\n"
            )
    return path


def write_b0_hf_suite_artifacts(
    output_dir: Path | str,
    suite_result: B0HFBenchmarkSuiteResult,
    *,
    write_contract_sidecars: bool = True,
    overwrite_contract_sidecars: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if write_contract_sidecars:
        _ensure_tbg_zero_field_contract_sidecars_writable(
            output_dir,
            overwrite_contract_sidecars=overwrite_contract_sidecars,
        )
        for result in suite_result.case_results:
            _ensure_tbg_zero_field_contract_sidecars_writable(
                output_dir / result.case.benchmark_id,
                overwrite_contract_sidecars=overwrite_contract_sidecars,
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    case_roots: dict[str, Path] = {}
    for result in suite_result.case_results:
        case_root = output_dir / result.case.benchmark_id
        write_b0_hf_benchmark_artifacts(
            case_root,
            result,
            write_contract_sidecars=write_contract_sidecars,
            overwrite_contract_sidecars=overwrite_contract_sidecars,
        )
        case_roots[f"case_{result.case.benchmark_id}_root"] = case_root

    suite_summary_path = write_b0_hf_suite_summary(output_dir / "suite_summary.tsv", suite_result)
    artifacts = {"suite_summary_tsv": suite_summary_path, **case_roots}
    if write_contract_sidecars:
        write_b0_hf_suite_contract_sidecars(
            output_dir,
            suite_result,
            artifact_paths=artifacts,
            overwrite=overwrite_contract_sidecars,
        )
    return artifacts

__all__ = [name for name in globals() if not name.startswith('__')]
