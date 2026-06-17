from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....api.artifacts import ModelRecord, write_contract_artifacts
from ..params import TBGParameters

_CONTRACT_FILENAMES = (
    "manifest.json",
    "model.json",
    "config.yaml",
    "conventions.json",
    "environment.json",
    "validation.json",
    "observables.json",
)


def complex_to_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _relative_artifact_files(root: Path, artifact_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    files: dict[str, str] = {}
    for key, raw_path in dict(artifact_paths or {}).items():
        path = Path(raw_path)
        if not path.is_absolute():
            files[str(key)] = path.as_posix()
            continue
        try:
            files[str(key)] = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            files[str(key)] = str(path)
    return files


def _ensure_contract_sidecars_absent(root: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [name for name in _CONTRACT_FILENAMES if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing TBG zero-field contract sidecars in {root}: {existing}. "
            "Pass overwrite=True only when replacing this workflow's sidecars intentionally."
        )


def _dataclass_payload(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _json_safe(value: object) -> object:
    if isinstance(value, complex):
        return complex_to_pair(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _runtime_environment_payload(runtime: object) -> dict[str, object]:
    environment = getattr(runtime, "environment", None)
    payload = {str(key): _json_safe(value) for key, value in _dataclass_payload(environment).items()}
    for key in (
        "start_time",
        "end_time",
        "bm_elapsed_sec",
        "hf_elapsed_sec",
        "path_elapsed_sec",
        "grid_elapsed_sec",
        "total_elapsed_sec",
    ):
        if hasattr(runtime, key):
            payload[key] = _json_safe(getattr(runtime, key))
    return payload


def _tbg_model_record(params: TBGParameters, *, system_name: str = "tbg", extra: Mapping[str, object] | None = None) -> ModelRecord:
    param_payload = {
        "dtheta_rad": float(params.dtheta_rad),
        "convention": str(params.convention),
        "vf": float(params.vf),
        "chemical_potential": float(params.chemical_potential),
        "w0": float(params.w0),
        "w1": float(params.w1),
        "strain": float(params.strain),
        "strain_angle_rad": float(params.strain_angle_rad),
        "poisson": float(params.poisson),
        "beta_g": float(params.beta_g),
        "alpha": float(params.alpha),
        "deformation_potential": float(params.deformation_potential),
    }
    param_payload.update(dict(extra or {}))
    lattice = {
        "g1_nm_inv_pair": complex_to_pair(params.g1),
        "g2_nm_inv_pair": complex_to_pair(params.g2),
        "a1_nm_pair": complex_to_pair(params.a1),
        "a2_nm_pair": complex_to_pair(params.a2),
        "theta12_rad": float(params.theta12),
        "kt_nm_inv_pair": complex_to_pair(params.kt),
        "kb_point_nm_inv_pair": complex_to_pair(params.kb_point),
    }
    return ModelRecord(system_name=system_name, params=param_payload, lattice=lattice)


def _bm_solution_summary(solution: object | None) -> dict[str, object] | None:
    if solution is None:
        return None
    spectrum = np.asarray(getattr(solution, "spectrum"), dtype=float)
    return {
        "lg": int(getattr(solution, "lg")),
        "nk": int(getattr(solution, "nk")),
        "nt": int(getattr(solution, "nt")),
        "n_eta": int(getattr(solution, "n_eta")),
        "n_spin": int(getattr(solution, "n_spin")),
        "nb": int(getattr(solution, "nb")),
        "periodic_g_grid": bool(getattr(solution, "periodic_g_grid", True)),
        "spectrum_shape": [int(value) for value in spectrum.shape],
        "energy_min_mev": float(np.min(spectrum)) if spectrum.size else float("nan"),
        "energy_max_mev": float(np.max(spectrum)) if spectrum.size else float("nan"),
    }


def _kpath_payload(path: object) -> dict[str, object]:
    kvec = np.asarray(getattr(path, "kvec"), dtype=np.complex128)
    kdist = np.asarray(getattr(path, "kdist"), dtype=float)
    return {
        "point_count": int(kvec.size),
        "labels": list(getattr(path, "labels")),
        "node_indices": [int(value) for value in getattr(path, "node_indices")],
        "kdist_min": float(np.min(kdist)) if kdist.size else 0.0,
        "kdist_max": float(np.max(kdist)) if kdist.size else 0.0,
    }


def _bm_unstrained_validation_payload(result: object) -> dict[str, object]:
    parity = getattr(result, "parity")
    runtime_parity = getattr(result, "runtime_parity", None)
    payload: dict[str, object] = {
        "status": "recorded",
        "parity": {
            "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
            "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
            "rms_band_diff_mev": float(parity.rms_band_diff_mev),
            "mean_abs_band_diff_mev": float(parity.mean_abs_band_diff_mev),
            "k_middle_gap_diff_mev": float(parity.k_middle_gap_diff_mev),
            "valence_bandwidth_diff_mev": _optional_float(parity.valence_bandwidth_diff_mev),
            "conduction_bandwidth_diff_mev": _optional_float(parity.conduction_bandwidth_diff_mev),
        },
    }
    if runtime_parity is not None:
        payload["runtime_parity"] = _json_safe(_dataclass_payload(runtime_parity))
    return payload


def _bm_unstrained_observables(result: object) -> dict[str, object]:
    run = getattr(result, "run")
    return {
        "theta_deg": float(getattr(getattr(result, "reference"), "theta_deg")),
        "k_middle_gap_mev": float(run.k_middle_gap_mev),
        "valence_bandwidth_mev": _optional_float(run.valence_bandwidth_mev),
        "conduction_bandwidth_mev": _optional_float(run.conduction_bandwidth_mev),
        "path": _kpath_payload(run.path),
        "path_solution": _bm_solution_summary(run.path_solution),
        "grid_solution": _bm_solution_summary(run.grid_solution),
    }


def _hf_state_shapes(state: object) -> dict[str, object]:
    return {
        "density": [int(value) for value in np.asarray(getattr(state, "density")).shape],
        "hamiltonian": [int(value) for value in np.asarray(getattr(state, "hamiltonian")).shape],
        "h0": [int(value) for value in np.asarray(getattr(state, "h0")).shape],
        "energies": [int(value) for value in np.asarray(getattr(state, "energies")).shape],
    }


def _diagnostics_payload(diagnostics: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in diagnostics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[str(key)] = _json_safe(value)
    return payload


def _b0_hf_validation_payload(result: object) -> dict[str, object]:
    parity = getattr(result, "parity")
    hf_run = getattr(result, "hf_run")
    runtime_parity = getattr(result, "runtime_parity", None)
    payload: dict[str, object] = {
        "status": "converged" if bool(hf_run.converged) else "not_converged",
        "converged": bool(hf_run.converged),
        "exit_reason": str(hf_run.exit_reason),
        "iterations": int(hf_run.iterations),
        "parity": {
            "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
            "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
            "rms_band_diff_mev": float(parity.rms_band_diff_mev),
            "mean_abs_band_diff_mev": float(parity.mean_abs_band_diff_mev),
            "energy_sorting": str(parity.energy_sorting),
        },
    }
    if runtime_parity is not None:
        payload["runtime_parity"] = _json_safe(_dataclass_payload(runtime_parity))
    return payload


def _b0_hf_observables(result: object) -> dict[str, object]:
    case = getattr(result, "case")
    hf_run = getattr(result, "hf_run")
    state = hf_run.state
    path_result = getattr(result, "path_result")
    band_data = getattr(path_result, "band_data")
    return {
        "benchmark_id": str(case.benchmark_id),
        "theta_deg": float(case.theta_deg),
        "nu": float(path_result.nu),
        "mu_mev": float(state.mu),
        "init_mode": str(path_result.init_mode),
        "normalized_init_mode": str(path_result.normalized_init_mode),
        "seed": int(path_result.seed),
        "iterations": int(hf_run.iterations),
        "exit_reason": str(hf_run.exit_reason),
        "converged": bool(hf_run.converged),
        "diagnostics": _diagnostics_payload(getattr(state, "diagnostics", {})),
        "path": _kpath_payload(path_result.path),
        "bands": {
            "labels": list(getattr(band_data, "band_labels")),
            "energy_shape": [int(value) for value in np.asarray(getattr(band_data, "energies")).shape],
            "mean_weights_shape": [int(value) for value in np.asarray(getattr(band_data, "mean_weights")).shape],
        },
        "state_shapes": _hf_state_shapes(state),
    }


def write_bm_unstrained_benchmark_contract_sidecars(
    output_dir: str | Path,
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field BM benchmark result.

    This is metadata-only: it references existing TSV/plot/text artifacts and
    summarizes in-memory result shapes/scalars, but never writes numerical
    arrays or reruns BM/HF computations.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    run = getattr(result, "run")
    reference = getattr(result, "reference")
    files = _relative_artifact_files(root, artifact_paths)
    model = _tbg_model_record(
        run.params,
        extra={
            "theta_deg": float(reference.theta_deg),
            "lg": int(getattr(run.path_solution, "lg")),
            "periodic_g_grid": bool(getattr(run.path_solution, "periodic_g_grid", True)),
        },
    )
    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.bm_unstrained_benchmark",
        system_name="tbg",
        model=model,
        config={
            "implementation": "python_b0",
            "runner_kind": "bm_unstrained_benchmark",
            "theta_deg": float(reference.theta_deg),
            "lg": int(getattr(run.path_solution, "lg")),
            "grid_lk": None if run.grid_solution is None else int(round(np.sqrt(int(getattr(run.grid_solution, "nk"))) - 1)),
            "points_per_segment": None,
            "reference_path_tsv": str(getattr(reference, "path_tsv_path", "")),
        },
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "nm^-1",
            "length_unit": "nm",
            "density_convention": "not_applicable",
            "wavefunction_axis_order": "basis_band_valley_k",
            "hamiltonian_axis_order": "basis_basis_valley_k",
            "gauge": "tbg_zero_field_bm_c2t_symmetrized_system_defined",
        },
        environment=_runtime_environment_payload(run.runtime),
        validation=_bm_unstrained_validation_payload(result),
        observables=_bm_unstrained_observables(result),
        files=files,
        metadata={
            "runner_kind": "bm_unstrained_benchmark",
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
        },
        array_files=(),
    )


def _b0_hf_suite_case_summary(result: object) -> dict[str, object]:
    case = getattr(result, "case")
    hf_run = getattr(result, "hf_run")
    parity = getattr(result, "parity")
    state = hf_run.state
    return {
        "benchmark_id": str(case.benchmark_id),
        "theta_deg": float(case.theta_deg),
        "nu": int(case.nu),
        "init_mode": str(getattr(result, "path_result").init_mode),
        "normalized_init_mode": str(getattr(result, "path_result").normalized_init_mode),
        "seed": int(getattr(result, "path_result").seed),
        "lk": int(getattr(result, "path_result").lk),
        "lg": int(getattr(result, "path_result").lg),
        "converged": bool(hf_run.converged),
        "exit_reason": str(hf_run.exit_reason),
        "iterations": int(hf_run.iterations),
        "mu_mev": float(state.mu),
        "max_abs_band_diff_mev": float(parity.max_abs_band_diff_mev),
        "kdist_max_abs_diff": float(parity.kdist_max_abs_diff),
        "runtime_total_elapsed_sec": float(getattr(result, "runtime").total_elapsed_sec),
    }


def _b0_hf_suite_validation_payload(suite_result: object) -> dict[str, object]:
    case_results = tuple(getattr(suite_result, "case_results"))
    converged_count = sum(1 for result in case_results if bool(getattr(result, "hf_run").converged))
    return {
        "status": "all_converged" if converged_count == len(case_results) else "not_all_converged",
        "case_count": int(len(case_results)),
        "converged_count": int(converged_count),
        "max_kdist_max_abs_diff": 0.0
        if not case_results
        else float(max(getattr(result, "parity").kdist_max_abs_diff for result in case_results)),
        "max_abs_band_diff_mev": 0.0
        if not case_results
        else float(max(getattr(result, "parity").max_abs_band_diff_mev for result in case_results)),
    }


def _b0_hf_suite_observables(suite_result: object) -> dict[str, object]:
    case_results = tuple(getattr(suite_result, "case_results"))
    return {
        "case_count": int(len(case_results)),
        "total_elapsed_sec": float(sum(getattr(result, "runtime").total_elapsed_sec for result in case_results)),
        "case_results": [_b0_hf_suite_case_summary(result) for result in case_results],
    }


def write_b0_hf_benchmark_contract_sidecars(
    output_dir: str | Path,
    result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field B0 HF benchmark result.

    This is metadata-only and references existing TSV/plot/text outputs.  It
    does not serialize density/Hamiltonian arrays and does not rerun BM/HF.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    case = getattr(result, "case")
    path_result = getattr(result, "path_result")
    hf_run = getattr(result, "hf_run")
    files = _relative_artifact_files(root, artifact_paths)
    model = _tbg_model_record(
        result.params,
        extra={
            "theta_deg": float(case.theta_deg),
            "nu": int(case.nu),
            "lk": int(path_result.lk),
            "lg": int(path_result.lg),
            "overlap_lg": None if path_result.overlap_lg is None else int(path_result.overlap_lg),
            "beta": float(path_result.beta),
        },
    )
    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.b0_hf_benchmark",
        system_name="tbg",
        model=model,
        config={
            "implementation": "python_b0",
            "runner_kind": "b0_hf_benchmark",
            "benchmark_id": str(case.benchmark_id),
            "theta_deg": float(case.theta_deg),
            "nu": int(case.nu),
            "init_mode": str(path_result.init_mode),
            "normalized_init_mode": str(path_result.normalized_init_mode),
            "seed": int(path_result.seed),
            "lk": int(path_result.lk),
            "lg": int(path_result.lg),
            "points_per_segment": int(path_result.points_per_segment),
            "overlap_lg": None if path_result.overlap_lg is None else int(path_result.overlap_lg),
            "beta": float(path_result.beta),
            "relative_permittivity": float(path_result.relative_permittivity),
            "screening_lm": _optional_float(path_result.screening_lm),
            "finite_zero_limit": bool(path_result.finite_zero_limit),
            "zero_cutoff": float(path_result.zero_cutoff),
            "include_interaction": bool(path_result.include_interaction),
            "precision": float(hf_run.state.precision),
            "initial_density_override_path": None
            if getattr(result, "initial_density_override_path", None) is None
            else str(result.initial_density_override_path),
        },
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "nm^-1",
            "length_unit": "nm",
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "wavefunction_axis_order": "basis_band_valley_k",
            "gauge": "tbg_zero_field_b0_projected_system_defined",
        },
        environment=_runtime_environment_payload(result.runtime),
        validation=_b0_hf_validation_payload(result),
        observables=_b0_hf_observables(result),
        files=files,
        metadata={
            "runner_kind": "b0_hf_benchmark",
            "benchmark_id": str(case.benchmark_id),
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
        },
        array_files=(),
    )


def write_b0_hf_suite_contract_sidecars(
    output_dir: str | Path,
    suite_result: object,
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write public contract sidecars for a zero-field B0 HF benchmark suite.

    The suite sidecar is metadata-only. It summarizes case-level scalar metrics
    and references existing suite/case artifacts without writing numerical
    arrays or rerunning BM/HF.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_contract_sidecars_absent(root, overwrite=overwrite)
    case_results = tuple(getattr(suite_result, "case_results"))
    config_cases = [
        {
            "benchmark_id": str(getattr(result, "case").benchmark_id),
            "theta_deg": float(getattr(result, "case").theta_deg),
            "nu": int(getattr(result, "case").nu),
            "init_mode": str(getattr(result, "path_result").init_mode),
            "seed": int(getattr(result, "path_result").seed),
            "lk": int(getattr(result, "path_result").lk),
            "lg": int(getattr(result, "path_result").lg),
        }
        for result in case_results
    ]
    return write_contract_artifacts(
        root,
        workflow="tbg.zero_field.b0_hf_suite",
        system_name="tbg",
        model=ModelRecord(system_name="tbg", params={"case_count": int(len(case_results))}),
        config={
            "implementation": "python_b0",
            "runner_kind": "b0_hf_suite",
            "benchmark_ids": [item["benchmark_id"] for item in config_cases],
            "cases": config_cases,
        },
        conventions={
            "energy_unit": "meV",
            "momentum_unit": "nm^-1",
            "length_unit": "nm",
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "wavefunction_axis_order": "basis_band_valley_k",
            "gauge": "tbg_zero_field_b0_projected_system_defined",
        },
        environment={},
        validation=_b0_hf_suite_validation_payload(suite_result),
        observables=_b0_hf_suite_observables(suite_result),
        files=_relative_artifact_files(root, artifact_paths),
        metadata={
            "runner_kind": "b0_hf_suite",
            "adapter": "mean_field.systems.tbg.zero_field.artifacts",
        },
        array_files=(),
    )


__all__ = [
    "complex_to_pair",
    "write_b0_hf_benchmark_contract_sidecars",
    "write_b0_hf_suite_contract_sidecars",
    "write_bm_unstrained_benchmark_contract_sidecars",
]
