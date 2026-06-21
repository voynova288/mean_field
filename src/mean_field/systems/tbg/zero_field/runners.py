from __future__ import annotations

"""Compatibility facade for TBG zero-field benchmark runners."""

from ._runners_shared import (
    B0HFBenchmarkRun,
    B0HFBenchmarkRuntime,
    B0HFBenchmarkRuntimeParity,
    B0HFBenchmarkSuiteResult,
    BMUnstrainedBenchmarkRun,
    BMUnstrainedParity,
    BMUnstrainedRun,
    BMUnstrainedRuntime,
    BMUnstrainedRuntimeParity,
)
from ._runners_helpers import build_b0_reference_parameters
from ._runners_bm import run_bm_unstrained, run_bm_unstrained_benchmark
from ._runners_b0 import run_b0_hf_benchmark_case
from . import _runners_artifacts as _artifact_impl
from ._runners_suite import run_b0_hf_benchmark_suite, write_b0_hf_suite_artifacts, write_b0_hf_suite_summary
from ._runners_overlap import export_overlap_diagnostics
from .plotting import write_bm_band_plot, write_hf_band_plot, write_hf_scf_band_plot


def write_bm_unstrained_benchmark_artifacts(*args, **kwargs):
    _artifact_impl.write_bm_band_plot = write_bm_band_plot
    return _artifact_impl.write_bm_unstrained_benchmark_artifacts(*args, **kwargs)


def write_b0_hf_benchmark_artifacts(*args, **kwargs):
    _artifact_impl.write_hf_band_plot = write_hf_band_plot
    _artifact_impl.write_hf_scf_band_plot = write_hf_scf_band_plot
    return _artifact_impl.write_b0_hf_benchmark_artifacts(*args, **kwargs)

__all__ = [
    "B0HFBenchmarkRun",
    "B0HFBenchmarkRuntime",
    "B0HFBenchmarkRuntimeParity",
    "B0HFBenchmarkSuiteResult",
    "BMUnstrainedBenchmarkRun",
    "BMUnstrainedParity",
    "BMUnstrainedRun",
    "BMUnstrainedRuntime",
    "BMUnstrainedRuntimeParity",
    "build_b0_reference_parameters",
    "export_overlap_diagnostics",
    "run_b0_hf_benchmark_case",
    "run_b0_hf_benchmark_suite",
    "run_bm_unstrained",
    "run_bm_unstrained_benchmark",
    "write_b0_hf_benchmark_artifacts",
    "write_bm_band_plot",
    "write_hf_band_plot",
    "write_hf_scf_band_plot",
    "write_b0_hf_suite_artifacts",
    "write_b0_hf_suite_summary",
    "write_bm_unstrained_benchmark_artifacts",
]
