from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403

def run_b0_hf_benchmark_case(
    case: BenchmarkCase | str,
    *,
    lk: int | None = None,
    lg: int | None = None,
    points_per_segment: int | None = None,
    init_mode: str | None = None,
    seed: int | None = None,
    beta: float = 1.0,
    max_iter: int = 300,
    overlap_lg: int | None = None,
    precision: float = 1e-5,
    initial_density_path: Path | str | None = None,
) -> B0HFBenchmarkRun:
    if isinstance(case, str):
        case = load_b0_suite().get(case)

    grid_lk = case.lk if lk is None else int(lk)
    bm_lg = case.lg if lg is None else int(lg)
    path_points = case.points_per_segment if points_per_segment is None else int(points_per_segment)
    requested_init_mode = case.init_mode if init_mode is None else str(init_mode)
    requested_seed = case.seed if seed is None else int(seed)
    overlap_grid_lg = bm_lg if overlap_lg is None else int(overlap_lg)
    runner_kind = _benchmark_runner_kind(case)
    initial_density_override = None
    initial_density_override_path = None if initial_density_path is None else Path(initial_density_path)

    start_time = current_timestamp()

    params = build_b0_reference_parameters(case.theta_deg)

    bm_start = perf_counter()
    grid_solution = _build_benchmark_grid_solution(case, params, lk=grid_lk, lg=bm_lg)
    bm_elapsed = perf_counter() - bm_start

    if initial_density_override_path is not None:
        if runner_kind != "full":
            raise ValueError("Explicit initial_density_path is only supported for full HF benchmark cases.")
        nt = int(grid_solution.n_spin * grid_solution.n_eta * grid_solution.nb)
        initial_density_override = _load_explicit_full_init_density_override(
            initial_density_override_path,
            nt=nt,
            nk=int(grid_solution.nk),
        )
    elif runner_kind == "full" and not _full_init_mode_uses_stochastic_initialization(requested_init_mode):
        nt = int(grid_solution.n_spin * grid_solution.n_eta * grid_solution.nb)
        initial_density_override = _load_required_full_init_density_override(
            case,
            init_mode=requested_init_mode,
            seed=requested_seed,
            nt=nt,
            nk=int(grid_solution.nk),
            lk=grid_lk,
        )

    overlap_start = perf_counter()
    overlap_blocks = build_overlap_block_set(grid_solution, lg=overlap_grid_lg)
    overlap_elapsed = perf_counter() - overlap_start

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
    return B0HFBenchmarkRun(
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
        initial_density_override_path=initial_density_override_path,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
