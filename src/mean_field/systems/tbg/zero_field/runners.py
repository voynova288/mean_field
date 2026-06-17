from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from ....api.artifacts import required_artifact_files
from ....benchmarks import (
    BMRuntimeBenchmarkRecord,
    BMUnstrainedReference,
    BenchmarkCase,
    ParameterReference,
    RuntimeBenchmarkRecord,
    load_b0_parameter_references,
    load_b0_runtime_benchmarks,
    load_b0_suite,
    load_bm_unstrained_references,
    load_bm_unstrained_runtime_benchmarks,
    load_complex_tensor4_tsv,
)
from ....core.lattice import KPath
from ....runtime import RuntimeEnvironment, collect_runtime_environment, current_timestamp, safe_ratio
from ..params import TBGParameters
from .hf import (
    HFOverlapBlockSet,
    RestrictedHartreeFockState,
    RestrictedHartreeFockRun,
    build_overlap_block_set,
    normalize_full_init_mode,
    run_full_hartree_fock,
    run_restricted_hartree_fock,
)
from .artifacts import (
    write_b0_hf_benchmark_contract_sidecars,
    write_b0_hf_suite_contract_sidecars,
    write_bm_unstrained_benchmark_contract_sidecars,
)
from .hf_runners import (
    HFPathParity,
    HFPathResult,
    build_restricted_hf_scf_path_plot_result,
    compare_hf_path_to_reference,
    evaluate_restricted_hf_path,
    write_hf_path_nodes_tsv,
    write_hf_path_summary,
    write_hf_scf_path_tsv,
    write_hf_path_tsv,
)
from .model import BMSolution, build_b0_uniform_lattice, solve_bm_model
from .overlap import OverlapDiagnostics, calculate_overlap_compact, summarize_overlap
from .path import build_b0_benchmark_kpath, build_kpath_from_reference_nodes
from .path_advisor import rank_kpath_candidates_for_lk
from .plotting import write_bm_band_plot, write_hf_scf_band_plot
from .plotting import write_hf_band_plot


@dataclass(frozen=True)
class BMUnstrainedRuntime:
    start_time: str
    end_time: str
    path_elapsed_sec: float
    grid_elapsed_sec: float
    total_elapsed_sec: float
    environment: RuntimeEnvironment


@dataclass(frozen=True)
class BMUnstrainedParity:
    kdist_max_abs_diff: float
    max_abs_band_diff_mev: float
    rms_band_diff_mev: float
    mean_abs_band_diff_mev: float
    k_middle_gap_diff_mev: float
    valence_bandwidth_diff_mev: float | None
    conduction_bandwidth_diff_mev: float | None


@dataclass(frozen=True)
class BMUnstrainedRuntimeParity:
    path_elapsed_sec_delta: float
    path_elapsed_sec_ratio: float | None
    grid_elapsed_sec_delta: float
    grid_elapsed_sec_ratio: float | None
    total_elapsed_sec_delta: float
    total_elapsed_sec_ratio: float | None


@dataclass(frozen=True)
class BMUnstrainedRun:
    params: TBGParameters
    path: KPath
    path_solution: BMSolution
    grid_solution: BMSolution | None
    k_middle_gap_mev: float
    valence_bandwidth_mev: float | None
    conduction_bandwidth_mev: float | None
    runtime: BMUnstrainedRuntime


@dataclass(frozen=True)
class BMUnstrainedBenchmarkRun:
    reference: BMUnstrainedReference
    run: BMUnstrainedRun
    parity: BMUnstrainedParity
    runtime_reference: BMRuntimeBenchmarkRecord | None
    runtime_parity: BMUnstrainedRuntimeParity | None


@dataclass(frozen=True)
class B0HFBenchmarkRuntime:
    start_time: str
    end_time: str
    bm_elapsed_sec: float
    hf_elapsed_sec: float
    path_elapsed_sec: float
    total_elapsed_sec: float
    environment: RuntimeEnvironment


@dataclass(frozen=True)
class B0HFBenchmarkRuntimeParity:
    bm_elapsed_sec_delta: float
    bm_elapsed_sec_ratio: float | None
    hf_elapsed_sec_delta: float
    hf_elapsed_sec_ratio: float | None
    path_elapsed_sec_delta: float
    path_elapsed_sec_ratio: float | None
    total_elapsed_sec_delta: float
    total_elapsed_sec_ratio: float | None


@dataclass(frozen=True)
class B0HFBenchmarkRun:
    case: BenchmarkCase
    params: TBGParameters
    path: KPath
    grid_solution: BMSolution
    hf_run: RestrictedHartreeFockRun
    path_result: HFPathResult
    parity: HFPathParity
    runtime: B0HFBenchmarkRuntime
    runtime_reference: RuntimeBenchmarkRecord | None
    runtime_parity: B0HFBenchmarkRuntimeParity | None
    initial_density_override_path: Path | None = None


@dataclass(frozen=True)
class B0HFBenchmarkSuiteResult:
    case_results: tuple[B0HFBenchmarkRun, ...]

    @property
    def max_kdist_max_abs_diff(self) -> float:
        if not self.case_results:
            return 0.0
        return max(result.parity.kdist_max_abs_diff for result in self.case_results)

    @property
    def max_abs_band_diff_mev(self) -> float:
        if not self.case_results:
            return 0.0
        return max(result.parity.max_abs_band_diff_mev for result in self.case_results)

    @property
    def total_elapsed_sec(self) -> float:
        return float(sum(result.runtime.total_elapsed_sec for result in self.case_results))


@lru_cache(maxsize=1)
def _bm_unstrained_reference_map() -> dict[float, BMUnstrainedReference]:
    return {round(ref.theta_deg, 2): ref for ref in load_bm_unstrained_references()}


@lru_cache(maxsize=1)
def _bm_unstrained_runtime_reference_map() -> dict[float, BMRuntimeBenchmarkRecord]:
    return {round(ref.theta_deg, 2): ref for ref in load_bm_unstrained_runtime_benchmarks()}


@lru_cache(maxsize=1)
def _b0_parameter_reference_map() -> dict[float, ParameterReference]:
    return {round(ref.theta_deg, 2): ref for ref in load_b0_parameter_references()}


@lru_cache(maxsize=1)
def _b0_hf_runtime_reference_map() -> dict[str, RuntimeBenchmarkRecord]:
    return {row.benchmark_id: row for row in load_b0_runtime_benchmarks()}


def _ensure_tbg_zero_field_contract_sidecars_writable(
    root: Path | str,
    *,
    overwrite_contract_sidecars: bool,
) -> None:
    if overwrite_contract_sidecars:
        return
    result_root = Path(root)
    existing = [name for name in required_artifact_files() if (result_root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing TBG zero-field contract sidecars in {result_root}: {existing}. "
            "Pass overwrite_contract_sidecars=True only when intentionally replacing these sidecars."
        )


def _full_init_density_override_paths(case: BenchmarkCase, init_mode: str, seed: int, *, lk: int | None = None) -> tuple[Path, ...]:
    normalized = normalize_full_init_mode(init_mode)
    candidates: list[Path] = []
    if lk is not None:
        candidates.append(case.case_dir / f"initial_density_{normalized}_seed_{int(seed):03d}_lk{int(lk)}.tsv")
    candidates.append(case.case_dir / f"initial_density_{normalized}_seed_{int(seed):03d}.tsv")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _full_init_mode_uses_stochastic_initialization(init_mode: str) -> bool:
    normalized = normalize_full_init_mode(init_mode)
    return normalized in {"random", "diag_random", "flavor"}


def _load_full_init_density_override(path: Path, *, nt: int, nk: int) -> np.ndarray:
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, real_s, imag_s = stripped.split("\t")
            density[int(row_s), int(col_s), int(ik_s)] = complex(float(real_s), float(imag_s))
    return density


def _inspect_full_init_density_override(path: Path) -> tuple[int, int, int] | None:
    max_ik = -1
    max_row = -1
    max_col = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ik_s, row_s, col_s, _real_s, _imag_s = stripped.split("\t")
            max_ik = max(max_ik, int(ik_s))
            max_row = max(max_row, int(row_s))
            max_col = max(max_col, int(col_s))
    if max_ik < 0:
        return None
    return max_row + 1, max_col + 1, max_ik + 1


def _load_required_full_init_density_override(
    case: BenchmarkCase,
    *,
    init_mode: str,
    seed: int,
    nt: int,
    nk: int,
    lk: int | None = None,
) -> np.ndarray | None:
    if _full_init_mode_uses_stochastic_initialization(init_mode):
        return None

    override_candidates = _full_init_density_override_paths(case, init_mode, seed, lk=lk)
    override_path = next((path for path in override_candidates if path.is_file()), None)
    if override_path is None:
        raise FileNotFoundError(
            f"Full benchmark case {case.benchmark_id} requires Julia initial density "
            f"{', '.join(path.name for path in override_candidates)}, but none of them exist."
        )

    override_dims = _inspect_full_init_density_override(override_path)
    expected_dims = (nt, nt, nk)
    if override_dims is None:
        raise ValueError(
            f"Full benchmark case {case.benchmark_id} has an empty initial density override: {override_path}"
        )
    if override_dims != expected_dims:
        raise ValueError(
            f"Full benchmark case {case.benchmark_id} expects initial density shape {expected_dims}, "
            f"but {override_path.name} contains {override_dims}."
        )

    return _load_full_init_density_override(
        override_path,
        nt=nt,
        nk=nk,
    )


def _load_explicit_full_init_density_override(path: Path | str, *, nt: int, nk: int) -> np.ndarray:
    override_path = Path(path)
    if not override_path.is_file():
        raise FileNotFoundError(f"Explicit initial density override does not exist: {override_path}")

    override_dims = _inspect_full_init_density_override(override_path)
    expected_dims = (nt, nt, nk)
    if override_dims is None:
        raise ValueError(f"Explicit initial density override is empty: {override_path}")
    if override_dims != expected_dims:
        raise ValueError(
            f"Explicit initial density override expects shape {expected_dims}, "
            f"but {override_path} contains {override_dims}."
        )

    return _load_full_init_density_override(override_path, nt=nt, nk=nk)


def _benchmark_grid_reference_uk_path(case: BenchmarkCase, *, lk: int, lg: int) -> Path:
    reference_path_getter = getattr(case, "bm_grid_reference_uk_path", None)
    if reference_path_getter is None:
        return Path()
    return reference_path_getter(lk=lk, lg=lg)


def _apply_benchmark_grid_reference_uk(case: BenchmarkCase, solution: BMSolution, *, lk: int, lg: int) -> BMSolution:
    reference_path = _benchmark_grid_reference_uk_path(case, lk=lk, lg=lg)
    if not reference_path.is_file():
        return solution
    reference_uk = load_complex_tensor4_tsv(reference_path, shape=solution.uk.shape)
    return solution.with_reference_uk(reference_uk)


def _should_apply_benchmark_grid_reference_uk(case: BenchmarkCase, *, lk: int, lg: int) -> bool:
    return _benchmark_grid_reference_uk_path(case, lk=lk, lg=lg).is_file()


def _build_benchmark_grid_solution(case: BenchmarkCase, params: TBGParameters, *, lk: int, lg: int) -> BMSolution:
    grid = build_b0_uniform_lattice(params, lk)
    solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
    if not _should_apply_benchmark_grid_reference_uk(case, lk=lk, lg=lg):
        return solution
    return _apply_benchmark_grid_reference_uk(case, solution, lk=lk, lg=lg)


def _benchmark_grid_cache_key(case: BenchmarkCase, *, lk: int, lg: int) -> tuple[float, int, int, str]:
    reference_key = ""
    if _should_apply_benchmark_grid_reference_uk(case, lk=lk, lg=lg):
        reference_key = str(_benchmark_grid_reference_uk_path(case, lk=lk, lg=lg))
    return (round(case.theta_deg, 2), int(lk), int(lg), reference_key)


def _write_key_value_summary(path: Path, entries: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def _format_ratio(value: float | None) -> str:
    return "" if value is None else str(value)


def _compare_bm_runtime_to_reference(runtime: BMUnstrainedRuntime, reference: BMRuntimeBenchmarkRecord) -> BMUnstrainedRuntimeParity:
    return BMUnstrainedRuntimeParity(
        path_elapsed_sec_delta=runtime.path_elapsed_sec - reference.path_elapsed_sec,
        path_elapsed_sec_ratio=safe_ratio(runtime.path_elapsed_sec, reference.path_elapsed_sec),
        grid_elapsed_sec_delta=runtime.grid_elapsed_sec - reference.grid_elapsed_sec,
        grid_elapsed_sec_ratio=safe_ratio(runtime.grid_elapsed_sec, reference.grid_elapsed_sec),
        total_elapsed_sec_delta=runtime.total_elapsed_sec - reference.total_elapsed_sec,
        total_elapsed_sec_ratio=safe_ratio(runtime.total_elapsed_sec, reference.total_elapsed_sec),
    )


def _compare_hf_runtime_to_reference(runtime: B0HFBenchmarkRuntime, reference: RuntimeBenchmarkRecord) -> B0HFBenchmarkRuntimeParity:
    return B0HFBenchmarkRuntimeParity(
        bm_elapsed_sec_delta=runtime.bm_elapsed_sec - reference.bm_elapsed_sec,
        bm_elapsed_sec_ratio=safe_ratio(runtime.bm_elapsed_sec, reference.bm_elapsed_sec),
        hf_elapsed_sec_delta=runtime.hf_elapsed_sec - reference.hf_elapsed_sec,
        hf_elapsed_sec_ratio=safe_ratio(runtime.hf_elapsed_sec, reference.hf_elapsed_sec),
        path_elapsed_sec_delta=runtime.path_elapsed_sec - reference.path_elapsed_sec,
        path_elapsed_sec_ratio=safe_ratio(runtime.path_elapsed_sec, reference.path_elapsed_sec),
        total_elapsed_sec_delta=runtime.total_elapsed_sec - reference.total_elapsed_sec,
        total_elapsed_sec_ratio=safe_ratio(runtime.total_elapsed_sec, reference.total_elapsed_sec),
    )


def _benchmark_runner_kind(case: BenchmarkCase) -> str:
    return case.load_runtime_summary().entries.get("runner_kind", "restricted").strip().lower()


def _run_benchmark_hf_from_grid_solution(
    case: BenchmarkCase,
    grid_solution: BMSolution,
    overlap_blocks: HFOverlapBlockSet,
    *,
    grid_lk: int,
    requested_init_mode: str,
    requested_seed: int,
    beta: float,
    max_iter: int,
    precision: float,
    overlap_lg: int | None = None,
    runner_kind: str | None = None,
    initial_density_override: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    runner_kind = _benchmark_runner_kind(case) if runner_kind is None else str(runner_kind)
    if runner_kind == "full":
        initial_density = initial_density_override
        if initial_density is None and not _full_init_mode_uses_stochastic_initialization(requested_init_mode):
            nt = int(grid_solution.n_spin * grid_solution.n_eta * grid_solution.nb)
            initial_density = _load_required_full_init_density_override(
                case,
                init_mode=requested_init_mode,
                seed=requested_seed,
                nt=nt,
                nk=int(grid_solution.nk),
                lk=grid_lk,
            )
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=case.nu, precision=precision)
        if overlap_lg is not None:
            state.diagnostics["overlap_lg"] = float(overlap_lg)
        return run_full_hartree_fock(
            state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            grid_solution.params,
            init_mode=requested_init_mode,
            seed=requested_seed,
            beta=beta,
            max_iter=max_iter,
            initial_density=initial_density,
        )
    if runner_kind == "restricted":
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=case.nu, precision=precision)
        if overlap_lg is not None:
            state.diagnostics["overlap_lg"] = float(overlap_lg)
        return run_restricted_hartree_fock(
            state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            grid_solution.params,
            init_mode=requested_init_mode,
            seed=requested_seed,
            beta=beta,
            max_iter=max_iter,
        )
    raise ValueError(f"Unsupported benchmark runner_kind={runner_kind!r} for case {case.benchmark_id}")


def build_b0_reference_parameters(theta_deg: float) -> TBGParameters:
    reference = _b0_parameter_reference_map().get(round(theta_deg, 2))
    if reference is None:
        return TBGParameters.from_degrees(
            theta_deg,
            vf=2482.0,
            w0=77.0,
            w1=110.0,
            strain=0.0,
            alpha=0.5,
            deformation_potential=0.0,
        )

    return TBGParameters(
        dtheta_rad=reference.dtheta_rad,
        convention="b0",
        vf=reference.vf,
        w0=reference.w0,
        w1=reference.w1,
        strain=reference.strain,
        alpha=reference.alpha,
        deformation_potential=0.0,
    )


def _build_packaged_or_native_benchmark_kpath(params: TBGParameters, theta_deg: float, points_per_segment: int) -> KPath:
    reference = _bm_unstrained_reference_map().get(round(theta_deg, 2))
    if reference is None:
        return build_b0_benchmark_kpath(params, points_per_segment)

    summary = reference.load_summary()
    if int(summary["points_per_segment"]) != points_per_segment:
        return build_b0_benchmark_kpath(params, points_per_segment)

    return build_kpath_from_reference_nodes(reference.load_path_nodes())


def _build_packaged_or_native_hf_benchmark_kpath(case: BenchmarkCase, params: TBGParameters, points_per_segment: int) -> KPath:
    if points_per_segment != case.points_per_segment:
        return build_b0_benchmark_kpath(params, points_per_segment)
    return build_kpath_from_reference_nodes(case.load_reference_nodes())


def _build_advisor_hf_benchmark_kpath(
    benchmark_id: str,
    params: TBGParameters,
    *,
    lk: int,
    points_per_segment: int,
) -> tuple[KPath, object]:
    ranked = rank_kpath_candidates_for_lk(
        params,
        lk=lk,
        points_per_segment=points_per_segment,
    )
    if not ranked:
        raise ValueError(f"No advisor path candidates available for {benchmark_id}.")
    compatibility = ranked[0]
    return compatibility.candidate.path, compatibility


def _write_advisor_path_selection(path: Path, *, compatibility: object) -> Path:
    candidate = compatibility.candidate
    return _write_key_value_summary(
        path,
        [
            ("path_source", "advisor_ranked_candidate"),
            ("candidate_rank", "1"),
            ("candidate_name", str(candidate.name)),
            ("candidate_family", str(candidate.family)),
            ("lk", str(int(compatibility.lk))),
            ("exact_count", str(int(compatibility.exact_count))),
            ("exact_node_hit_count", str(int(compatibility.exact_node_hit_count))),
            ("exact_segment_counts", ",".join(str(value) for value in compatibility.exact_segment_counts)),
            ("mean_nearest_distance", f"{float(compatibility.mean_nearest_distance):.16e}"),
            ("max_nearest_distance", f"{float(compatibility.max_nearest_distance):.16e}"),
            ("m_real", f"{float(candidate.m_point.real):.16f}"),
            ("m_imag", f"{float(candidate.m_point.imag):.16f}"),
            ("k_real", f"{float(candidate.k_point.real):.16f}"),
            ("k_imag", f"{float(candidate.k_point.imag):.16f}"),
        ],
    )


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


def write_bm_unstrained_benchmark_artifacts(
    output_dir: Path | str,
    result: BMUnstrainedBenchmarkRun,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    path_tsv_path = _write_bm_path_tsv(output_dir / "computed_bm_path.tsv", result.run)
    nodes_tsv_path = _write_bm_nodes_tsv(output_dir / "computed_nodes.tsv", result.run)
    summary_path = _write_bm_summary(output_dir / "computed_summary.txt", result.run)
    parity_path = _write_key_value_summary(
        output_dir / "parity_to_reference_summary.txt",
        [
            ("implementation", "python_b0"),
            ("reference_impl", "b0_reference"),
            ("reference_path_tsv", str(result.reference.path_tsv_path)),
            ("kdist_max_abs_diff", str(result.parity.kdist_max_abs_diff)),
            ("max_abs_band_diff_mev", str(result.parity.max_abs_band_diff_mev)),
            ("rms_band_diff_mev", str(result.parity.rms_band_diff_mev)),
            ("mean_abs_band_diff_mev", str(result.parity.mean_abs_band_diff_mev)),
            ("k_middle_gap_diff_meV", str(result.parity.k_middle_gap_diff_mev)),
            (
                "valence_bandwidth_diff_meV",
                "" if result.parity.valence_bandwidth_diff_mev is None else str(result.parity.valence_bandwidth_diff_mev),
            ),
            (
                "conduction_bandwidth_diff_meV",
                ""
                if result.parity.conduction_bandwidth_diff_mev is None
                else str(result.parity.conduction_bandwidth_diff_mev),
            ),
        ],
    )
    runtime_summary_path = _write_bm_runtime_summary(output_dir / "runtime_summary.txt", result.run)
    runtime_parity_path = _write_bm_runtime_parity_summary(output_dir / "runtime_to_reference_summary.txt", result)
    plot_paths = write_bm_band_plot(
        output_dir,
        theta_deg=result.reference.theta_deg,
        path=result.run.path,
        path_solution=result.run.path_solution,
        stem="band_plot",
    )

    artifacts = {
        "path_tsv": path_tsv_path,
        "nodes_tsv": nodes_tsv_path,
        "summary_txt": summary_path,
        "parity_summary_txt": parity_path,
        "runtime_summary_txt": runtime_summary_path,
        **plot_paths,
    }
    if runtime_parity_path is not None:
        artifacts["runtime_parity_summary_txt"] = runtime_parity_path
    if write_contract_sidecars:
        write_bm_unstrained_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifacts,
            overwrite=overwrite_contract_sidecars,
        )
    return artifacts


def write_b0_hf_benchmark_artifacts(
    output_dir: Path | str,
    result: B0HFBenchmarkRun,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    path_tsv_path = output_dir / "computed_hf_path.tsv"
    scf_path_tsv_path = output_dir / "computed_hf_scf_path.tsv"
    nodes_tsv_path = output_dir / "computed_nodes.tsv"
    summary_path = output_dir / "computed_summary.txt"
    parity_path = output_dir / "parity_to_reference_summary.txt"
    runtime_summary_path = output_dir / "runtime_summary.txt"
    runtime_parity_path = output_dir / "runtime_to_reference_summary.txt"

    write_hf_path_tsv(path_tsv_path, result.path_result)
    write_hf_path_nodes_tsv(nodes_tsv_path, result.path_result)
    write_hf_path_summary(summary_path, result.path_result)
    scf_plot_result = build_restricted_hf_scf_path_plot_result(
        result.hf_run,
        result.grid_solution,
        path=result.path,
        init_mode=result.path_result.init_mode,
    )
    write_hf_scf_path_tsv(scf_path_tsv_path, scf_plot_result)

    _write_key_value_summary(
        parity_path,
        [
            ("benchmark_id", result.case.benchmark_id),
            ("implementation", "python_b0"),
            ("reference_impl", "b0_reference"),
            ("reference_path_tsv", str(result.case.reference_path_tsv_path)),
            ("kdist_max_abs_diff", str(result.parity.kdist_max_abs_diff)),
            ("max_abs_band_diff_mev", str(result.parity.max_abs_band_diff_mev)),
            ("rms_band_diff_mev", str(result.parity.rms_band_diff_mev)),
            ("mean_abs_band_diff_mev", str(result.parity.mean_abs_band_diff_mev)),
            ("energy_sorting", result.parity.energy_sorting),
        ],
    )
    _write_hf_runtime_summary(runtime_summary_path, result)
    runtime_parity_written = _write_hf_runtime_parity_summary(runtime_parity_path, result)
    plot_paths = write_hf_band_plot(output_dir, result.path_result, stem="band_plot")
    scf_plot_paths = write_hf_scf_band_plot(output_dir, scf_plot_result, stem="band_plot_scf_grid")

    advisor_path, advisor_compatibility = _build_advisor_hf_benchmark_kpath(
        result.case.benchmark_id,
        result.params,
        lk=result.path_result.lk,
        points_per_segment=result.path_result.points_per_segment,
    )
    advisor_path_result = evaluate_restricted_hf_path(
        result.hf_run,
        result.grid_solution,
        points_per_segment=result.path_result.points_per_segment,
        lg=result.path_result.lg,
        overlap_lg=result.path_result.overlap_lg,
        beta=result.path_result.beta,
        relative_permittivity=result.path_result.relative_permittivity,
        screening_lm=result.path_result.screening_lm,
        finite_zero_limit=result.path_result.finite_zero_limit,
        zero_cutoff=result.path_result.zero_cutoff,
        init_mode=result.path_result.init_mode,
        path=advisor_path,
    )
    advisor_scf_plot_result = build_restricted_hf_scf_path_plot_result(
        result.hf_run,
        result.grid_solution,
        path=advisor_path,
        init_mode=result.path_result.init_mode,
    )
    advisor_path_tsv_path = output_dir / "computed_hf_path_advisor.tsv"
    advisor_scf_path_tsv_path = output_dir / "computed_hf_scf_path_advisor.tsv"
    advisor_nodes_tsv_path = output_dir / "computed_nodes_advisor.tsv"
    advisor_summary_path = output_dir / "computed_summary_advisor.txt"
    advisor_selection_path = output_dir / "advisor_path_selection.txt"
    write_hf_path_tsv(advisor_path_tsv_path, advisor_path_result)
    write_hf_path_nodes_tsv(advisor_nodes_tsv_path, advisor_path_result)
    write_hf_path_summary(advisor_summary_path, advisor_path_result)
    write_hf_scf_path_tsv(advisor_scf_path_tsv_path, advisor_scf_plot_result)
    _write_advisor_path_selection(advisor_selection_path, compatibility=advisor_compatibility)
    advisor_plot_paths = write_hf_band_plot(output_dir, advisor_path_result, stem="band_plot_advisor")
    advisor_scf_plot_paths = write_hf_scf_band_plot(output_dir, advisor_scf_plot_result, stem="band_plot_scf_grid_advisor")

    artifacts = {
        "path_tsv": path_tsv_path,
        "scf_path_tsv": scf_path_tsv_path,
        "nodes_tsv": nodes_tsv_path,
        "summary_txt": summary_path,
        "parity_summary_txt": parity_path,
        "runtime_summary_txt": runtime_summary_path,
        **plot_paths,
        "scf_band_plot_png": scf_plot_paths["band_plot_png"],
        "scf_band_plot_pdf": scf_plot_paths["band_plot_pdf"],
        "advisor_path_tsv": advisor_path_tsv_path,
        "advisor_scf_path_tsv": advisor_scf_path_tsv_path,
        "advisor_nodes_tsv": advisor_nodes_tsv_path,
        "advisor_summary_txt": advisor_summary_path,
        "advisor_selection_txt": advisor_selection_path,
        "advisor_band_plot_png": advisor_plot_paths["band_plot_png"],
        "advisor_band_plot_pdf": advisor_plot_paths["band_plot_pdf"],
        "advisor_scf_band_plot_png": advisor_scf_plot_paths["band_plot_png"],
        "advisor_scf_band_plot_pdf": advisor_scf_plot_paths["band_plot_pdf"],
    }
    if runtime_parity_written is not None:
        artifacts["runtime_parity_summary_txt"] = runtime_parity_path
    if write_contract_sidecars:
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifacts,
            overwrite=overwrite_contract_sidecars,
        )
    return artifacts


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


def export_overlap_diagnostics(theta_deg: float, *, lattice_kind: str, m: int, n: int, points_per_segment: int = 120, lg: int = 9, grid_lk: int = 33) -> OverlapDiagnostics:
    run = run_bm_unstrained(theta_deg, points_per_segment=points_per_segment, lg=lg, grid_lk=grid_lk)
    if lattice_kind == "path":
        solution = run.path_solution
    else:
        if run.grid_solution is None:
            raise ValueError("Grid overlap requested but grid solution was not computed.")
        solution = run.grid_solution
    overlap = calculate_overlap_compact(solution, m, n, valley_index=0)
    return summarize_overlap(theta_deg, lattice_kind, overlap, m, n, valley_label="K")
