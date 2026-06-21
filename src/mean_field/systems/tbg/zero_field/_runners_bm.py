from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403
from ._runners_summaries import *  # noqa: F401,F403

def run_bm_unstrained(theta_deg: float, *, points_per_segment: int = 120, lg: int = 9, grid_lk: int = 33) -> BMUnstrainedRun:
    start_time = current_timestamp()
    total_start = perf_counter()

    params = build_b0_reference_parameters(theta_deg)

    path_start = perf_counter()
    path = _build_packaged_or_native_benchmark_kpath(params, theta_deg, points_per_segment)
    path_solution = solve_bm_model(params, path.kvec, lg=lg, sigma_rotation=True)
    path_elapsed = perf_counter() - path_start

    path_energies = np.sort(path_solution.flattened_energies()[:, path.node_indices[1] - 1])
    k_middle_gap = float(path_energies[4] - path_energies[3])
    if grid_lk > 0:
        grid_start = perf_counter()
        grid = build_b0_uniform_lattice(params, grid_lk)
        grid_solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
        grid_elapsed = perf_counter() - grid_start
        grid_energies = grid_solution.flattened_energies()
        valence_width = float(np.max(grid_energies[:4, :]) - np.min(grid_energies[:4, :]))
        conduction_width = float(np.max(grid_energies[4:, :]) - np.min(grid_energies[4:, :]))
    else:
        grid_solution = None
        grid_elapsed = 0.0
        valence_width = None
        conduction_width = None

    total_elapsed = perf_counter() - total_start
    end_time = current_timestamp()
    runtime = BMUnstrainedRuntime(
        start_time=start_time,
        end_time=end_time,
        path_elapsed_sec=float(path_elapsed),
        grid_elapsed_sec=float(grid_elapsed),
        total_elapsed_sec=float(total_elapsed),
        environment=collect_runtime_environment(),
    )

    return BMUnstrainedRun(
        params=params,
        path=path,
        path_solution=path_solution,
        grid_solution=grid_solution,
        k_middle_gap_mev=k_middle_gap,
        valence_bandwidth_mev=valence_width,
        conduction_bandwidth_mev=conduction_width,
        runtime=runtime,
    )


def run_bm_unstrained_benchmark(theta_deg: float) -> BMUnstrainedBenchmarkRun:
    reference = _bm_unstrained_reference_map().get(round(theta_deg, 2))
    if reference is None:
        raise KeyError(f"No bundled BM unstrained reference for theta={theta_deg}")

    summary = reference.load_summary()
    run = run_bm_unstrained(
        theta_deg,
        points_per_segment=int(summary["points_per_segment"]),
        lg=int(summary["lg"]),
        grid_lk=int(summary["grid_lk"]),
    )
    runtime_reference = _bm_unstrained_runtime_reference_map().get(round(theta_deg, 2))
    runtime_parity = None if runtime_reference is None else _compare_bm_runtime_to_reference(run.runtime, runtime_reference)
    return BMUnstrainedBenchmarkRun(
        reference=reference,
        run=run,
        parity=_compare_bm_unstrained_to_reference(reference, run),
        runtime_reference=runtime_reference,
        runtime_parity=runtime_parity,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
