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

__all__ = [name for name in globals() if not name.startswith('__')]
