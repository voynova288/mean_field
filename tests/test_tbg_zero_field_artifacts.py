from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api import (
    load_result,
    reconstruct_canonical_hf_run_result,
    required_artifact_files,
)
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_projected_basis_consistent,
)
from mean_field.benchmarks import BMUnstrainedReference, BenchmarkCase
from mean_field.core.hf import FlavorBandData, flavor_block_indices
from mean_field.core.lattice import KPath
from mean_field.runtime import RuntimeEnvironment
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    B0HFBenchmarkRun,
    B0HFBenchmarkRuntime,
    B0HFBenchmarkRuntimeParity,
    B0HFBenchmarkSuiteResult,
    BMUnstrainedBenchmarkRun,
    BMUnstrainedParity,
    BMUnstrainedRun,
    BMUnstrainedRuntime,
    BMUnstrainedRuntimeParity,
    BMSolution,
    HFPathParity,
    HFPathResult,
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    TBGZeroFieldHFRunProvenance,
    TBGZeroFieldHFSourceReceipt,
    TBGZeroFieldInteractionSpec,
    TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
    TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA,
    TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION,
    b0_hf_benchmark_run_to_hf_run_result,
    build_b0_uniform_lattice,
    build_fig6_kpath,
    build_restricted_hf_path_hamiltonian,
    build_restricted_hf_scf_path_plot_result,
    build_tbg_zero_field_half_open_torus_mesh,
    build_tbg_zero_field_screened_block_bundle,
    complex_to_pair,
    conventional_projector_to_stored_density,
    empty_overlap_block_set,
    load_tbg_zero_field_complete_hf_state_archive_npz,
    run_full_hartree_fock,
    run_restricted_hartree_fock,
    solve_bm_model_on_torus,
    tbg_zero_field_hf_run_to_hf_result,
    tbg_zero_field_lattice_kvec_sha256,
    tbg_zero_field_hf_run_to_hf_run_result,
    write_b0_hf_benchmark_artifacts,
    write_b0_hf_benchmark_contract_sidecars,
    write_b0_hf_suite_artifacts,
    write_b0_hf_suite_contract_sidecars,
    write_b0_hf_suite_summary,
    write_bm_unstrained_benchmark_artifacts,
    write_bm_unstrained_benchmark_contract_sidecars,
)


def test_tbg_parameters_from_degrees_preserves_independent_delta() -> None:
    params = TBGParameters.from_degrees(1.05, delta=1.25)

    assert params.delta == 1.25
    assert params.to_summary_dict()["delta"] == 1.25


def _runtime_environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        hostname="test-node",
        cpu_model="unit-test-cpu",
        slurm_partition="debug",
        slurm_nodelist="test001",
        slurm_cpus_per_task=1,
        blas_threads=1,
        numba_threads=1,
        sys_cpu_threads=1,
        process_count=1,
        backend_choice="numpy",
        threadpoolctl_info=(),
        thread_env={},
        jit_warmup_included=False,
        python_version="3.test",
        numpy_version=np.__version__,
    )


def _bm_solution(params: TBGParameters, *, nk: int = 2) -> BMSolution:
    lg = 1
    dim = 4 * lg * lg
    nb = 2
    n_eta = 2
    n_spin = 2
    nt = n_spin * n_eta * nb
    return BMSolution(
        params=params,
        lattice_kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.2j], dtype=np.complex128)[:nk],
        lg=lg,
        nlocal=4,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        hamiltonian=np.zeros((dim, dim, n_eta, nk), dtype=np.complex128),
        sigma_z=np.zeros((nt, nt, nk), dtype=np.complex128),
        uk=np.zeros((dim, nb, n_eta, nk), dtype=np.complex128),
        spectrum=np.asarray([[[-10.0, -9.0], [1.0, 2.0]], [[3.0, 4.0], [10.0, 11.0]]], dtype=float)[:, :, :nk],
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        sigma_rotation=True,
        periodic_g_grid=True,
    )


def _bm_benchmark_result(tmp_path: Path) -> BMUnstrainedBenchmarkRun:
    params = TBGParameters.from_degrees(1.05)
    path = KPath(
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 0.1], dtype=float),
        labels=("K", "G"),
        node_indices=(1, 2),
    )
    run = BMUnstrainedRun(
        params=params,
        path=path,
        path_solution=_bm_solution(params),
        grid_solution=_bm_solution(params),
        k_middle_gap_mev=1.25,
        valence_bandwidth_mev=2.5,
        conduction_bandwidth_mev=3.5,
        runtime=BMUnstrainedRuntime(
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:01",
            path_elapsed_sec=0.1,
            grid_elapsed_sec=0.2,
            total_elapsed_sec=0.3,
            environment=_runtime_environment(),
        ),
    )
    reference = BMUnstrainedReference(
        theta_deg=1.05,
        root=tmp_path / "reference",
        summary_path=tmp_path / "reference" / "summary.txt",
        path_nodes_path=tmp_path / "reference" / "nodes.tsv",
        path_tsv_path=tmp_path / "reference" / "path.tsv",
        grid_kvec_path=tmp_path / "reference" / "grid.tsv",
    )
    parity = BMUnstrainedParity(
        kdist_max_abs_diff=0.0,
        max_abs_band_diff_mev=0.01,
        rms_band_diff_mev=0.005,
        mean_abs_band_diff_mev=0.003,
        k_middle_gap_diff_mev=0.002,
        valence_bandwidth_diff_mev=0.004,
        conduction_bandwidth_diff_mev=0.006,
    )
    runtime_parity = BMUnstrainedRuntimeParity(
        path_elapsed_sec_delta=0.0,
        path_elapsed_sec_ratio=1.0,
        grid_elapsed_sec_delta=0.0,
        grid_elapsed_sec_ratio=1.0,
        total_elapsed_sec_delta=0.0,
        total_elapsed_sec_ratio=1.0,
    )
    return BMUnstrainedBenchmarkRun(
        reference=reference,
        run=run,
        parity=parity,
        runtime_reference=None,
        runtime_parity=runtime_parity,
    )


def test_tbg_zero_field_bm_unstrained_contract_sidecars_are_metadata_only(tmp_path: Path) -> None:
    result = _bm_benchmark_result(tmp_path)
    output_dir = tmp_path / "bm"
    output_dir.mkdir()
    path_tsv = output_dir / "path.tsv"
    plot_png = output_dir / "bands.png"
    path_tsv.write_text("0.0\t1.0\n", encoding="utf-8")
    plot_png.write_text("not a real png for metadata-only test\n", encoding="utf-8")

    paths = write_bm_unstrained_benchmark_contract_sidecars(
        output_dir,
        result,
        artifact_paths={"path_tsv": path_tsv, "band_plot_png": plot_png},
    )

    assert paths["manifest.json"] == output_dir / "manifest.json"
    assert {path.name for path in output_dir.iterdir()} >= set(required_artifact_files()) | {"path.tsv", "bands.png"}
    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.bm_unstrained_benchmark"
    assert loaded.manifest["metadata"]["runner_kind"] == "bm_unstrained_benchmark"
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert loaded.manifest["files"]["path_tsv"] == "path.tsv"
    assert loaded.manifest["files"]["band_plot_png"] == "bands.png"
    assert loaded.conventions is not None and loaded.conventions["density_convention"] == "not_applicable"
    assert loaded.conventions["energy_unit"] == "meV"
    assert loaded.validation is not None and loaded.validation["status"] == "recorded"
    assert loaded.observables is not None and loaded.observables["theta_deg"] == 1.05
    assert loaded.observables["path_solution"]["spectrum_shape"] == [2, 2, 2]
    assert loaded.model is not None
    assert loaded.model["lattice"]["g1_b0_code_pair"] == complex_to_pair(result.run.params.g1)
    assert loaded.model["lattice"]["g1_nm_inv_pair"] == complex_to_pair(result.run.params.g1 / 0.246)
    assert loaded.model["lattice"]["a1_b0_code_pair"] == complex_to_pair(result.run.params.a1)
    assert loaded.model["lattice"]["a1_nm_pair"] == complex_to_pair(result.run.params.a1 * 0.246)
    assert loaded.conventions is not None
    assert loaded.conventions["momentum_unit"] == "dimensionless_b0_code"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_bm_unstrained_benchmark_contract_sidecars(output_dir, result)


def _benchmark_case(tmp_path: Path, *, nu: int = 2, lg: int = 1) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id="unit_b0",
        theta_deg=1.05,
        nu=int(nu),
        state_label="unit",
        description="unit test",
        source_group="unit",
        source_path_tsv="reference.tsv",
        source_nodes_tsv="nodes.tsv",
        source_summary_txt="summary.txt",
        source_hf_jld2="hf.jld2",
        init_mode="sp",
        seed=7,
        lk=1,
        lg=int(lg),
        points_per_segment=1,
        mu_mev=0.12,
        exit_reason="converged",
        benchmark_case_dir=str(tmp_path / "case"),
    )


def _restricted_hf_run() -> RestrictedHartreeFockRun:
    nt = 8
    nk = 2
    state = RestrictedHartreeFockState(
        h0=np.zeros((nt, nt, nk), dtype=np.complex128),
        sigma_z=np.zeros((nt, nt, nk), dtype=np.complex128),
        density=np.zeros((nt, nt, nk), dtype=np.complex128),
        hamiltonian=np.zeros((nt, nt, nk), dtype=np.complex128),
        energies=np.asarray([[-1.0, -0.9], [-0.5, -0.4], [-0.2, -0.1], [0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.8, 0.9], [1.1, 1.2]], dtype=float),
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=2.0,
        v0=1.0,
        mu=0.12,
        precision=1.0e-5,
        n_spin=2,
        n_eta=2,
        n_band=2,
        diagnostics={"hf_energy": -1.5, "final_raw_norm": 0.0, "overlap_lg": 1.0, "beta": 1.0},
    )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=empty_overlap_block_set(),
        iter_energy=np.asarray([-2.0, -1.5], dtype=float),
        iter_err=np.asarray([1.0e-2, 1.0e-6], dtype=float),
        iter_oda=np.asarray([1.0, 0.7], dtype=float),
        init_mode="spindown",
        seed=7,
        converged=True,
        exit_reason="converged",
    )


def _b0_hf_benchmark_result(tmp_path: Path) -> B0HFBenchmarkRun:
    params = TBGParameters.from_degrees(1.05)
    path = KPath(
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 0.1], dtype=float),
        labels=("K", "G"),
        node_indices=(1, 2),
    )
    band_data = FlavorBandData(
        band_labels=tuple(f"b{i}" for i in range(8)),
        energies=np.zeros((8, 2), dtype=float),
        mean_weights=np.ones((8, 4), dtype=float),
    )
    hf_run = _restricted_hf_run()
    path_result = HFPathResult(
        params=params,
        path=path,
        hamiltonian=np.zeros((8, 8, 2), dtype=np.complex128),
        band_data=band_data,
        mu=0.12,
        nu=2.0,
        lk=1,
        lg=1,
        points_per_segment=1,
        init_mode="sp",
        normalized_init_mode="spindown",
        seed=7,
        exit_reason="converged",
        beta=1.0,
        overlap_lg=1,
        relative_permittivity=15.0,
        screening_lm=None,
        finite_zero_limit=False,
        zero_cutoff=1.0e-6,
        include_interaction=True,
    )
    return B0HFBenchmarkRun(
        case=_benchmark_case(tmp_path),
        params=params,
        path=path,
        grid_solution=_bm_solution(params),
        hf_run=hf_run,
        path_result=path_result,
        parity=HFPathParity(
            kdist_max_abs_diff=0.0,
            max_abs_band_diff_mev=0.02,
            rms_band_diff_mev=0.01,
            mean_abs_band_diff_mev=0.005,
            energy_sorting="ascending_per_k",
        ),
        runtime=B0HFBenchmarkRuntime(
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:02",
            bm_elapsed_sec=0.1,
            hf_elapsed_sec=0.2,
            path_elapsed_sec=0.3,
            total_elapsed_sec=0.6,
            environment=_runtime_environment(),
        ),
        runtime_reference=None,
        runtime_parity=B0HFBenchmarkRuntimeParity(
            bm_elapsed_sec_delta=0.0,
            bm_elapsed_sec_ratio=1.0,
            hf_elapsed_sec_delta=0.0,
            hf_elapsed_sec_ratio=1.0,
            path_elapsed_sec_delta=0.0,
            path_elapsed_sec_ratio=1.0,
            total_elapsed_sec_delta=0.0,
            total_elapsed_sec_ratio=1.0,
        ),
        initial_density_override_path=tmp_path / "initial_density.tsv",
    )


def test_tbg_zero_field_b0_hf_contract_sidecars_are_metadata_only(tmp_path: Path) -> None:
    result = _b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "b0_hf"
    output_dir.mkdir()
    path_tsv = output_dir / "hf_path.tsv"
    summary_txt = output_dir / "summary.txt"
    path_tsv.write_text("k_dist\tb0\n0.0\t0.0\n", encoding="utf-8")
    summary_txt.write_text("summary\n", encoding="utf-8")

    write_b0_hf_benchmark_contract_sidecars(
        output_dir,
        result,
        artifact_paths={"path_tsv": path_tsv, "summary_txt": summary_txt},
    )

    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.b0_hf_benchmark"
    assert loaded.manifest["metadata"]["benchmark_id"] == "unit_b0"
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert loaded.manifest["files"]["path_tsv"] == "hf_path.tsv"
    assert loaded.conventions is not None and loaded.conventions["density_convention"] == "stored_delta"
    assert loaded.validation is not None and loaded.validation["status"] == "converged"
    assert loaded.validation["iterations"] == 2
    assert loaded.observables is not None and loaded.observables["benchmark_id"] == "unit_b0"
    assert loaded.observables["mu_mev"] == 0.12
    assert loaded.observables["state_shapes"]["density"] == [8, 8, 2]
    assert loaded.config is not None and loaded.config["precision"] == 1.0e-5

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_b0_hf_benchmark_contract_sidecars(output_dir, result)


def test_tbg_zero_field_b0_hf_suite_contract_sidecars_summarize_cases_without_arrays(tmp_path: Path) -> None:
    result = _b0_hf_benchmark_result(tmp_path)
    suite_result = B0HFBenchmarkSuiteResult(case_results=(result,))
    output_dir = tmp_path / "b0_suite"
    output_dir.mkdir()
    suite_summary = output_dir / "suite_summary.tsv"
    suite_summary.write_text("benchmark_id\tconverged\nunit_b0\ttrue\n", encoding="utf-8")

    write_b0_hf_suite_contract_sidecars(
        output_dir,
        suite_result,
        artifact_paths={"suite_summary_tsv": suite_summary},
    )

    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.b0_hf_suite"
    assert loaded.manifest["metadata"]["runner_kind"] == "b0_hf_suite"
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert loaded.manifest["files"]["suite_summary_tsv"] == "suite_summary.tsv"
    assert loaded.validation is not None and loaded.validation["status"] == "all_converged"
    assert loaded.validation["case_count"] == 1
    assert loaded.validation["converged_count"] == 1
    assert loaded.observables is not None and loaded.observables["case_count"] == 1
    assert loaded.observables["case_results"][0]["benchmark_id"] == "unit_b0"
    assert loaded.config is not None and loaded.config["benchmark_ids"] == ["unit_b0"]

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_b0_hf_suite_contract_sidecars(output_dir, suite_result)


def _fake_plot_paths(output_dir: Path | str, *, stem: str = "band_plot") -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{stem}.png"
    pdf = root / f"{stem}.pdf"
    png.write_text("png", encoding="utf-8")
    pdf.write_text("pdf", encoding="utf-8")
    return {"band_plot_png": png, "band_plot_pdf": pdf}


def test_tbg_zero_field_bm_runner_writer_adds_contract_sidecars(monkeypatch, tmp_path: Path) -> None:
    import mean_field.systems.tbg.zero_field.runners as runner_module

    monkeypatch.setattr(
        runner_module,
        "write_bm_band_plot",
        lambda output_dir, **kwargs: _fake_plot_paths(output_dir, stem=str(kwargs.get("stem", "band_plot"))),
    )
    result = _bm_benchmark_result(tmp_path)
    output_dir = tmp_path / "bm_runner"

    artifact_paths = write_bm_unstrained_benchmark_artifacts(output_dir, result)

    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.bm_unstrained_benchmark"
    assert loaded.manifest["files"]["path_tsv"] == "computed_bm_path.tsv"
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert artifact_paths["path_tsv"].is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_bm_unstrained_benchmark_artifacts(output_dir, result)
    write_bm_unstrained_benchmark_artifacts(output_dir, result, overwrite_contract_sidecars=True)


def test_tbg_zero_field_b0_runner_writers_add_contract_sidecars(monkeypatch, tmp_path: Path) -> None:
    import mean_field.systems.tbg.zero_field.runners as runner_module

    monkeypatch.setattr(
        runner_module,
        "write_hf_band_plot",
        lambda output_dir, result, stem="band_plot": _fake_plot_paths(output_dir, stem=stem),
    )
    monkeypatch.setattr(
        runner_module,
        "write_hf_scf_band_plot",
        lambda output_dir, result, stem="band_plot_scf_grid": _fake_plot_paths(output_dir, stem=stem),
    )
    result = _b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "b0_runner"

    artifact_paths = write_b0_hf_benchmark_artifacts(output_dir, result)

    loaded = load_result(output_dir)
    assert loaded.manifest["metadata"]["workflow"] == "tbg.zero_field.b0_hf_benchmark"
    assert loaded.manifest["files"]["path_tsv"] == "computed_hf_path.tsv"
    assert loaded.validation is not None and loaded.validation["iterations"] == result.hf_run.iterations
    assert "array_summaries" not in loaded.manifest["metadata"]
    assert artifact_paths["path_tsv"].is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_b0_hf_benchmark_artifacts(output_dir, result)

    suite_output_dir = tmp_path / "b0_suite_runner"
    suite_artifacts = write_b0_hf_suite_artifacts(suite_output_dir, B0HFBenchmarkSuiteResult(case_results=(result,)))
    loaded_suite = load_result(suite_output_dir)
    assert loaded_suite.manifest["metadata"]["workflow"] == "tbg.zero_field.b0_hf_suite"
    assert loaded_suite.manifest["files"]["suite_summary_tsv"] == "suite_summary.tsv"
    assert loaded_suite.validation is not None and loaded_suite.validation["case_count"] == 1
    assert suite_artifacts["suite_summary_tsv"].is_file()
    assert load_result(suite_output_dir / result.case.benchmark_id).manifest["metadata"]["workflow"] == "tbg.zero_field.b0_hf_benchmark"

_TYPED_BM_GRID_CACHE: dict[tuple[str, int, int, bool, bool], BMSolution] = {}
_TYPED_BM_ARRAY_FIELDS = (
    "lattice_kvec",
    "hamiltonian",
    "sigma_z",
    "uk",
    "spectrum",
    "gvec",
)

def _bm_solution_on_grid(
    params: TBGParameters,
    *,
    lk: int = 1,
    lg: int = 7,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = True,
) -> BMSolution:
    key = (
        params.independent_fingerprint,
        int(lk),
        int(lg),
        bool(sigma_rotation),
        bool(periodic_g_grid),
    )
    if key not in _TYPED_BM_GRID_CACHE:
        base = solve_bm_model_on_torus(
            params,
            lk + 1,
            lg=lg,
            sigma_rotation=sigma_rotation,
            calculate_chern_operator=True,
            periodic_g_grid=periodic_g_grid,
        )
        for name in _TYPED_BM_ARRAY_FIELDS:
            np.asarray(getattr(base, name)).setflags(write=False)
        _TYPED_BM_GRID_CACHE[key] = base
    copied = deepcopy(_TYPED_BM_GRID_CACHE[key])
    for name in _TYPED_BM_ARRAY_FIELDS:
        object.__setattr__(copied, name, np.array(getattr(copied, name), copy=True))
    object.__setattr__(copied, "params", params)
    return copied

def _canonical_b0_hf_benchmark_result(
    tmp_path: Path,
    *,
    hf_mode: str = "restricted",
    params: TBGParameters | None = None,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = True,
) -> B0HFBenchmarkRun:
    params = TBGParameters.from_degrees(1.05) if params is None else params
    grid_solution = _bm_solution_on_grid(
        params,
        lk=1,
        lg=7,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )
    params = grid_solution.params
    state = RestrictedHartreeFockState.from_bm_solution(
        grid_solution,
        nu=0.0,
        precision=1.0e-6,
    )
    state.diagnostics["overlap_lg"] = 7.0
    interaction_spec = TBGZeroFieldInteractionSpec()
    screened_block_bundle = build_tbg_zero_field_screened_block_bundle(
        grid_solution,
        interaction_spec=interaction_spec,
        overlap_lg=7,
    )
    overlap_blocks = screened_block_bundle.screened_blocks
    runner = (
        run_full_hartree_fock
        if hf_mode == "full"
        else run_restricted_hartree_fock
    )
    resolved_init_mode = "educated" if hf_mode == "full" else "bm"
    hf_run = runner(
        state,
        overlap_blocks,
        grid_solution.lattice_kvec,
        grid_solution.params,
        init_mode=resolved_init_mode,
        seed=11,
        beta=0.0,
        max_iter=2,
        oda_stall_threshold=1.0e-3,
        interaction_spec=interaction_spec,
        source_solution=grid_solution,
        screened_block_bundle=screened_block_bundle,
    )
    assert hf_run.converged
    assert isinstance(hf_run.provenance, TBGZeroFieldHFRunProvenance)
    path = KPath(
        kvec=np.asarray([0.0 + 0.0j, params.g1], dtype=np.complex128),
        kdist=np.asarray([0.0, abs(params.g1)], dtype=float),
        labels=("G", "G1"),
        node_indices=(1, 2),
    )
    band_data = FlavorBandData(
        band_labels=tuple(f"b{i}" for i in range(state.nt)),
        energies=np.zeros((state.nt, 2), dtype=float),
        mean_weights=np.ones((state.nt, 4), dtype=float),
    )
    path_result = HFPathResult(
        params=params,
        path=path,
        hamiltonian=np.zeros((state.nt, state.nt, 2), dtype=np.complex128),
        band_data=band_data,
        mu=0.0,
        nu=0.0,
        lk=2,
        lg=grid_solution.lg,
        points_per_segment=1,
        init_mode=hf_run.init_mode,
        normalized_init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        beta=0.0,
        overlap_lg=7,
    )
    return B0HFBenchmarkRun(
        case=_benchmark_case(tmp_path, nu=0, lg=grid_solution.lg),
        params=params,
        path=path,
        grid_solution=grid_solution,
        hf_run=hf_run,
        path_result=path_result,
        parity=HFPathParity(
            kdist_max_abs_diff=0.0,
            max_abs_band_diff_mev=0.0,
            rms_band_diff_mev=0.0,
            mean_abs_band_diff_mev=0.0,
        ),
        runtime=B0HFBenchmarkRuntime(
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:01",
            bm_elapsed_sec=0.1,
            hf_elapsed_sec=0.2,
            path_elapsed_sec=0.3,
            total_elapsed_sec=0.6,
            environment=_runtime_environment(),
        ),
        runtime_reference=None,
        runtime_parity=None,
    )

@pytest.mark.parametrize("runner", [run_full_hartree_fock, run_restricted_hartree_fock])
@pytest.mark.parametrize("bad_seed", [2.9, True, np.bool_(False)])
def test_tbg_direct_full_and_restricted_runners_reject_float_and_bool_seeds(
    runner,
    bad_seed: object,
) -> None:
    diagnostic_run = _restricted_hf_run()
    params = TBGParameters.from_degrees(1.05)
    with pytest.raises(ValueError, match="seed must be a non-bool integer"):
        runner(
            diagnostic_run.state,
            diagnostic_run.overlap_blocks,
            np.asarray([0.0 + 0.0j, 0.1 + 0.2j], dtype=np.complex128),
            params,
            seed=bad_seed,  # type: ignore[arg-type]
            legacy_untyped=True,
        )

def test_tbg_zero_field_b0_hf_benchmark_run_wraps_canonical_hf_run_result(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)

    canonical = b0_hf_benchmark_run_to_hf_run_result(result, archive_manifest={"state": "b0_hf_state.npz"})

    assert isinstance(canonical, ContractHFRunResult)
    receipt = result.hf_run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    interaction_spec = result.hf_run.state.interaction_spec
    assert isinstance(interaction_spec, TBGZeroFieldInteractionSpec)
    assert canonical.archive_manifest == {
        "state": "b0_hf_state.npz",
        "hf_source_receipt": receipt.to_metadata(),
        "interaction_spec": interaction_spec.to_metadata(),
        "hf_run_provenance": result.hf_run.provenance.to_metadata(),
    }
    assert (
        TBGZeroFieldHFSourceReceipt.from_metadata(
            canonical.archive_manifest["hf_source_receipt"]
        )
        == receipt
    )
    assert canonical.best_seed == 11
    assert canonical.init_mode == "bm"
    assert canonical.converged is True
    assert canonical.exit_reason == "converged"
    assert len(canonical.iteration_history) == result.hf_run.iterations
    assert canonical.iteration_history[-1] == {
        "iteration": result.hf_run.iterations,
        "energy": float(result.hf_run.iter_energy[-1]),
        "error": float(result.hf_run.iter_err[-1]),
        "oda_lambda": float(result.hf_run.iter_oda[-1]),
    }

    final = canonical.final_state
    assert_projected_basis_consistent(final.basis)
    assert_density_state_consistent(final.density)
    assert_hamiltonian_parts_consistent(final.hamiltonian)
    np.testing.assert_allclose(final.basis.k_grid_frac, np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]))
    raw_kvec = np.asarray(result.grid_solution.torus_mesh.kvec, dtype=np.complex128)
    physical_kvec = raw_kvec / 0.246
    np.testing.assert_allclose(final.basis.kvec, physical_kvec)
    assert final.basis.metadata["kvec_unit"] == "nm^-1"
    assert final.basis.metadata["source_kvec_unit"] == "dimensionless_b0_code"
    assert (
        final.basis.metadata["source_kvec_b0_code_sha256"]
        == tbg_zero_field_lattice_kvec_sha256(raw_kvec)
    )
    assert (
        final.basis.metadata["kvec_nm_inv_sha256"]
        == tbg_zero_field_lattice_kvec_sha256(physical_kvec)
    )
    assert receipt.lattice_kvec_sha256 == tbg_zero_field_lattice_kvec_sha256(raw_kvec)
    assert receipt.lattice_kvec_sha256 != tbg_zero_field_lattice_kvec_sha256(physical_kvec)
    lattice = final.basis.physical_model.lattice
    assert lattice["g1_b0_code_pair"] == complex_to_pair(result.params.g1)
    assert lattice["g1_nm_inv_pair"] == complex_to_pair(result.params.g1 / 0.246)
    assert lattice["a1_b0_code_pair"] == complex_to_pair(result.params.a1)
    assert lattice["a1_nm_pair"] == complex_to_pair(result.params.a1 * 0.246)
    central_start = result.grid_solution.nlocal * result.grid_solution.lg**2 // 2 - 1
    assert final.basis.active_band_indices == (
        central_start,
        central_start,
        central_start,
        central_start,
        central_start + 1,
        central_start + 1,
        central_start + 1,
        central_start + 1,
    )
    assert final.basis.metadata["spin_degeneracy_implicit_in_micro_wavefunctions"] is True
    assert final.basis.metadata["supports_crpa"] is False
    np.testing.assert_allclose(final.density.density_delta, result.hf_run.state.density)
    assert final.density.reference.scheme == "average"
    assert final.density.reference.metadata["reference_diagonal"] == 0.5
    assert final.density.metadata["raw_density_convention"] == "stored_delta"
    assert final.density.n_occupied_total == 16
    np.testing.assert_allclose(final.hamiltonian.fixed, np.zeros_like(result.hf_run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.hartree, np.zeros_like(result.hf_run.state.h0))
    np.testing.assert_allclose(final.hamiltonian.fock, np.zeros_like(result.hf_run.state.h0))
    assert final.hamiltonian.metadata["component_resolution"] == "collapsed_total_minus_h0"
    assert final.hamiltonian.metadata["supports_crpa"] is False
    assert final.eigenvectors_active.size == 0
    assert final.observables["eigenvectors_active_available"] is False
    assert final.observables["grid_mesh_size"] == 2
    assert final.observables["torus_mesh_fingerprint"] == result.grid_solution.torus_mesh.fingerprint
    assert result.grid_solution.source_attestation is not None
    assert (
        final.observables["bm_source_attestation"]["fingerprint"]
        == result.grid_solution.source_attestation.fingerprint
    )
    assert final.observables["hf_source_receipt"] == receipt.to_metadata()
    assert final.observables["interaction_spec"] == interaction_spec.to_metadata()
    assert (
        TBGZeroFieldHFSourceReceipt.from_metadata(
            final.observables["hf_source_receipt"]
        )
        == receipt
    )

@pytest.mark.parametrize("forbidden_key", ["path_tsv", "parity_summary_txt"])
def test_typed_direct_sidecar_rejects_path_and_parity_artifact_keys(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    with pytest.raises(ValueError, match="reject off-grid/path/parity artifact keys"):
        write_b0_hf_benchmark_contract_sidecars(
            tmp_path / f"typed_direct_{forbidden_key}",
            result,
            artifact_paths={forbidden_key: tmp_path / f"{forbidden_key}.txt"},
        )

def test_typed_direct_sidecar_validates_live_typed_source(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    result.hf_run.state.v0 += 1.0e-6
    with pytest.raises(ValueError, match="final state hash does not match"):
        write_b0_hf_benchmark_contract_sidecars(
            tmp_path / "typed_direct_mutated_source",
            result,
        )


def test_typed_direct_sidecar_binds_recomputed_coverage_and_tsv_hash(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_direct_coverage"
    artifact_paths = write_b0_hf_benchmark_artifacts(
        output_dir,
        result,
        write_contract_sidecars=False,
    )

    write_b0_hf_benchmark_contract_sidecars(
        output_dir,
        result,
        artifact_paths=artifact_paths,
    )

    loaded = load_result(output_dir)
    assert loaded.validation is not None
    coverage = loaded.validation["exact_saved_scf_path"]
    assert coverage["exact_point_count"] == 1
    assert coverage["distinct_coordinate_count"] == 1
    assert coverage["represented_segment_count"] == 0
    assert coverage["coverage_gate"]["passed"] is False
    assert coverage["tsv_lineage"]["sha256"] == hashlib.sha256(
        artifact_paths["scf_path_tsv"].read_bytes()
    ).hexdigest()
    assert coverage["tsv_lineage"]["relative_path"] == "computed_hf_scf_path.tsv"


def test_typed_direct_sidecar_rejects_tsv_not_matching_saved_hf_state(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_direct_tsv_lineage"
    artifact_paths = write_b0_hf_benchmark_artifacts(
        output_dir,
        result,
        write_contract_sidecars=False,
    )
    rows = artifact_paths["scf_path_tsv"].read_text(encoding="utf-8").splitlines()
    fields = rows[1].split("\t")
    fields[-1] = f"{float(fields[-1]) + 1.0:.16f}"
    artifact_paths["scf_path_tsv"].write_text(
        "\n".join([rows[0], "\t".join(fields)]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="band data differs from saved HF state"):
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifact_paths,
        )


def test_typed_direct_sidecar_rejects_missing_referenced_file(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_direct_missing_file"
    artifact_paths = write_b0_hf_benchmark_artifacts(
        output_dir,
        result,
        write_contract_sidecars=False,
    )
    artifact_paths["advisor_selection_txt"].unlink()

    with pytest.raises(ValueError, match="existing file under output root"):
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifact_paths,
        )


def test_typed_direct_sidecar_rejects_referenced_file_outside_output_root(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_direct_root"
    artifact_paths = write_b0_hf_benchmark_artifacts(
        output_dir,
        result,
        write_contract_sidecars=False,
    )
    outside = tmp_path / "outside_runtime_summary.txt"
    outside.write_text("fabricated\n", encoding="utf-8")
    forged_paths = dict(artifact_paths)
    forged_paths["runtime_summary_txt"] = outside

    with pytest.raises(ValueError, match="existing file under output root"):
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=forged_paths,
        )


def test_typed_direct_sidecar_rejects_fabricated_plot_keys_for_10x10_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mean_field.systems.tbg.zero_field.artifacts as artifact_impl

    base_result = _canonical_b0_hf_benchmark_result(tmp_path)
    typed_source = artifact_impl._validated_typed_b0_hf_source(base_result)
    params = TBGParameters.from_degrees(1.05)
    path = build_fig6_kpath(params, 30)
    mesh = build_tbg_zero_field_half_open_torus_mesh(params, 10)
    result = replace(
        base_result,
        path=path,
        grid_solution=SimpleNamespace(lattice_kvec=mesh.kvec),
    )
    monkeypatch.setattr(
        artifact_impl,
        "_validated_typed_b0_hf_source",
        lambda _result: typed_source,
    )
    output_dir = tmp_path / "typed_direct_10x10_fabricated_plot"
    output_dir.mkdir()
    artifact_paths = {
        "advisor_selection_txt": output_dir / "advisor_path_selection.txt",
        TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY: (
            output_dir / "validated_complete_hf_state_archive.npz"
        ),
        "runtime_summary_txt": output_dir / "runtime_summary.txt",
        "scf_path_tsv": output_dir / "computed_hf_scf_path.tsv",
        "scf_band_plot_png": output_dir / "band_plot_scf_grid.png",
        "scf_band_plot_pdf": output_dir / "band_plot_scf_grid.pdf",
    }
    for path_value in artifact_paths.values():
        path_value.write_bytes(b"fabricated\n")

    with pytest.raises(ValueError, match="reject fabricated PNG/PDF plot keys"):
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifact_paths,
        )


def test_typed_suite_and_summary_writers_reject_path_parity_data(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    suite_result = B0HFBenchmarkSuiteResult(case_results=(result,))
    message = "suite/summary writers reject off-grid/path/parity data"

    with pytest.raises(ValueError, match=message):
        write_b0_hf_suite_summary(tmp_path / "typed_suite_summary.tsv", suite_result)
    with pytest.raises(ValueError, match=message):
        write_b0_hf_suite_contract_sidecars(
            tmp_path / "typed_suite_sidecars",
            suite_result,
            artifact_paths={"suite_summary_tsv": tmp_path / "suite_summary.tsv"},
        )
    with pytest.raises(ValueError, match=message):
        write_b0_hf_suite_artifacts(tmp_path / "typed_suite_artifacts", suite_result)

def test_typed_suite_rejection_validates_live_source_first(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    result.hf_run.state.v0 += 1.0e-6
    suite_result = B0HFBenchmarkSuiteResult(case_results=(result,))

    with pytest.raises(ValueError, match="final state hash does not match"):
        write_b0_hf_suite_summary(tmp_path / "invalid_typed_suite.tsv", suite_result)

def test_typed_artifact_writer_persists_complete_archive_and_refuses_insufficient_exact_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mean_field.systems.tbg.zero_field._runners_artifacts as artifact_impl
    import mean_field.systems.tbg.zero_field.runners as runner_module

    def _forbidden_off_grid(*_args, **_kwargs):
        raise AssertionError("typed artifact writer attempted off-grid reconstruction")

    def _forbidden_plot(*_args, **_kwargs):
        raise AssertionError("typed artifact writer plotted insufficient exact path coverage")

    monkeypatch.setattr(
        artifact_impl,
        "evaluate_restricted_hf_path",
        _forbidden_off_grid,
    )
    monkeypatch.setattr(runner_module, "write_hf_band_plot", _forbidden_plot)
    monkeypatch.setattr(runner_module, "write_hf_scf_band_plot", _forbidden_plot)
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_saved_grid_writer"

    artifact_paths = write_b0_hf_benchmark_artifacts(output_dir, result)

    assert set(artifact_paths) == {
        "scf_path_tsv",
        TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
        "path_limitation_txt",
        "advisor_selection_txt",
        "runtime_summary_txt",
    }
    limitation = artifact_paths["path_limitation_txt"].read_text(encoding="utf-8")
    assert "status=limited" in limitation
    assert "band_plot_written=false" in limitation
    assert "off_grid_reconstruction=forbidden" in limitation
    assert "paper_figure_claim=none" in limitation
    assert "status=unavailable" in artifact_paths["advisor_selection_txt"].read_text(
        encoding="utf-8"
    )
    assert artifact_paths["scf_path_tsv"].is_file()

    complete_archive = load_tbg_zero_field_complete_hf_state_archive_npz(
        artifact_paths[TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY]
    )
    assert complete_archive.receipt == result.hf_run.state.hf_source_receipt
    np.testing.assert_array_equal(complete_archive.state.h0, result.hf_run.state.h0)
    np.testing.assert_array_equal(complete_archive.state.sigma_z, result.hf_run.state.sigma_z)
    np.testing.assert_array_equal(complete_archive.state.density, result.hf_run.state.density)
    np.testing.assert_array_equal(complete_archive.state.hamiltonian, result.hf_run.state.hamiltonian)
    np.testing.assert_array_equal(complete_archive.state.energies, result.hf_run.state.energies)
    np.testing.assert_array_equal(complete_archive.state.sigma_ztauz, result.hf_run.state.sigma_ztauz)
    assert complete_archive.state.mu == result.hf_run.state.mu
    assert complete_archive.state.nu == result.hf_run.state.nu
    assert complete_archive.state.v0 == result.hf_run.state.v0
    np.testing.assert_array_equal(complete_archive.iter_energy, result.hf_run.iter_energy)
    np.testing.assert_array_equal(complete_archive.iter_err, result.hf_run.iter_err)
    np.testing.assert_array_equal(complete_archive.iter_oda, result.hf_run.iter_oda)
    np.testing.assert_array_equal(
        complete_archive.physical_kvec_nm_inv,
        result.grid_solution.torus_mesh.kvec / 0.246,
    )
    np.testing.assert_array_equal(complete_archive.bm_uk, result.grid_solution.uk)
    np.testing.assert_array_equal(complete_archive.bm_spectrum, result.grid_solution.spectrum)
    np.testing.assert_array_equal(complete_archive.bm_gvec, result.grid_solution.gvec)
    assert complete_archive.screened_blocks.shifts == result.hf_run.screened_block_bundle.screened_blocks.shifts
    np.testing.assert_array_equal(
        complete_archive.screened_blocks.gvecs,
        result.hf_run.screened_block_bundle.screened_blocks.gvecs,
    )
    assert complete_archive.params.delta == result.params.delta

    loaded = load_result(output_dir)
    assert "path_tsv" not in loaded.manifest["files"]
    assert "advisor_path_tsv" not in loaded.manifest["files"]
    assert loaded.manifest["files"][TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY] == "validated_complete_hf_state_archive.npz"
    assert loaded.manifest["metadata"]["validated_complete_hf_state_archive"] == {
        "artifact_key": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
        "schema": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA,
        "schema_version": TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_SCHEMA_VERSION,
        "description": "validated complete HF state archive/checkpoint",
        "validation": "strict_round_trip_rehash",
        "resume_authority": "none",
        "typed_resume_trajectory": "not_implemented_fail_closed",
        "evidence_paths": [
            "src/mean_field/systems/tbg/zero_field/model.py::BMSolution.source_attestation",
            "src/mean_field/systems/tbg/zero_field/_hf_basis_overlap.py::TBGZeroFieldHFSourceReceipt",
            "src/mean_field/systems/tbg/zero_field/_hf_basis_overlap.py::TBGZeroFieldScreenedBlockBundle",
        ],
        "uncertainty": {
            "companion_circular_total_q_cutoff_parity": "not_established",
            "off_grid_reconstruction": "not_included",
            "paper_figure_claim": "none",
        },
    }
    array_summary = loaded.manifest["metadata"]["array_summaries"][0]
    assert array_summary["path"] == str(output_dir / "validated_complete_hf_state_archive.npz")
    assert "state_density" in array_summary["keys"]
    assert "screened_overlaps" in array_summary["keys"]
    assert loaded.validation is not None
    assert loaded.validation["path_status"] == "limitation_report_only"
    assert loaded.validation["complete_hf_state_archive_status"] == "validated"
    assert loaded.validation["typed_resume_trajectory"] == "not_implemented_fail_closed"
    coverage_payload = loaded.validation["exact_saved_scf_path"]
    assert coverage_payload["exact_point_count"] == 1
    assert coverage_payload["distinct_coordinate_count"] == 1
    assert coverage_payload["represented_segment_count"] == 0
    assert coverage_payload["coverage_gate"]["passed"] is False
    assert coverage_payload["tsv_lineage"] == {
        "artifact_key": "scf_path_tsv",
        "relative_path": "computed_hf_scf_path.tsv",
        "sha256": hashlib.sha256(
            artifact_paths["scf_path_tsv"].read_bytes()
        ).hexdigest(),
        "byte_count": artifact_paths["scf_path_tsv"].stat().st_size,
        "row_count": 1,
    }
    assert loaded.validation["advisor_status"] == "unavailable"
    assert "parity" not in loaded.validation
    assert loaded.config is not None
    assert "points_per_segment" not in loaded.config
    assert loaded.environment is not None
    assert "path_elapsed_sec" not in loaded.environment
    assert "total_elapsed_sec" not in loaded.environment
    assert loaded.observables is not None
    assert loaded.observables["path"]["status"] == "limitation_report_only"
    assert loaded.observables["path"]["coverage"] == coverage_payload
    assert loaded.observables["complete_hf_state_archive"]["resume_authority"] == "none"
    assert (
        loaded.observables["complete_hf_state_archive"]["typed_resume_trajectory"]
        == "not_implemented_fail_closed"
    )
    assert loaded.observables["advisor"] == {"status": "unavailable"}
    assert loaded.model is not None
    assert loaded.model["params"]["delta"] == result.params.delta


def test_typed_complete_hf_state_archive_loader_rejects_tampered_final_state(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    output_dir = tmp_path / "typed_complete_archive_tamper"
    artifact_paths = write_b0_hf_benchmark_artifacts(
        output_dir,
        result,
        write_contract_sidecars=False,
    )
    source_path = artifact_paths[TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY]
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    arrays["state_density"].flat[0] += 1.0e-8
    tampered_path = output_dir / "tampered_complete_archive.npz"
    np.savez_compressed(tampered_path, **arrays)

    with pytest.raises(ValueError, match="final-state hash mismatch"):
        load_tbg_zero_field_complete_hf_state_archive_npz(tampered_path)


@pytest.mark.parametrize(
    ("array_key", "message"),
    [
        ("bm_uk", "Archived BM uk hash does not match source attestation"),
        ("screened_overlaps", "screened block inventory hash mismatch"),
    ],
)
def test_typed_complete_hf_state_archive_loader_rejects_tampered_source_and_bundle(
    tmp_path: Path,
    array_key: str,
    message: str,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    artifact_paths = write_b0_hf_benchmark_artifacts(
        tmp_path / f"typed_complete_archive_tamper_{array_key}",
        result,
        write_contract_sidecars=False,
    )
    source_path = artifact_paths[TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY]
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    arrays[array_key].flat[0] += 1.0e-8
    tampered_path = source_path.with_name(f"tampered_{array_key}.npz")
    np.savez_compressed(tampered_path, **arrays)

    with pytest.raises(ValueError, match=message):
        load_tbg_zero_field_complete_hf_state_archive_npz(tampered_path)


def test_exact_scf_plot_gate_accepts_two_spanned_segments() -> None:
    import mean_field.systems.tbg.zero_field.artifacts as artifact_impl

    path = KPath(
        kvec=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.complex128),
        kdist=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
        labels=("A", "B", "C"),
        node_indices=(1, 2, 4),
    )
    coverage = artifact_impl._exact_saved_scf_path_coverage(
        SimpleNamespace(
            path=path,
            grid_solution=SimpleNamespace(lattice_kvec=path.kvec),
        )
    )

    assert coverage.meaningful
    assert coverage.exact_point_count == 4
    assert coverage.distinct_coordinate_count == 4
    assert coverage.represented_segment_indices == (0, 1)


def test_exact_scf_plot_gate_rejects_10x10_mesh_without_k() -> None:
    import mean_field.systems.tbg.zero_field.artifacts as artifact_impl

    params = TBGParameters.from_degrees(1.05)
    path = build_fig6_kpath(params, 30)
    mesh = build_tbg_zero_field_half_open_torus_mesh(params, 10)
    coverage = artifact_impl._exact_saved_scf_path_coverage(
        SimpleNamespace(
            path=path,
            grid_solution=SimpleNamespace(lattice_kvec=mesh.kvec),
        )
    )
    k_node_index = int(path.node_indices[1]) - 1
    assert k_node_index not in set(int(value) for value in coverage.path_sample_indices)
    assert "K" in coverage.missing_interior_node_labels
    assert not coverage.meaningful

def test_typed_path_reconstruction_requires_diagnostic_override_and_preserves_source(
    tmp_path: Path,
) -> None:
    params = TBGParameters.from_degrees(1.05)
    result = _canonical_b0_hf_benchmark_result(
        tmp_path,
        params=params,
        sigma_rotation=False,
        periodic_g_grid=False,
    )

    exact = build_restricted_hf_scf_path_plot_result(
        result.hf_run,
        result.grid_solution,
        path=result.path,
    )
    assert exact.lk == result.grid_solution.torus_mesh.mesh_size == 2

    with pytest.raises(ValueError, match="forbids automatic off-grid reconstruction"):
        build_restricted_hf_path_hamiltonian(
            result.hf_run,
            result.grid_solution,
            path=result.path,
            include_interaction=False,
        )

    _path, path_solution, _hamiltonian = build_restricted_hf_path_hamiltonian(
        result.hf_run,
        result.grid_solution,
        path=result.path,
        include_interaction=False,
        allow_typed_off_grid_diagnostic=True,
    )
    assert path_solution.params is result.grid_solution.params
    assert path_solution.sigma_rotation is result.grid_solution.sigma_rotation is False
    assert path_solution.periodic_g_grid is result.grid_solution.periodic_g_grid is False

def test_tbg_typed_export_rejects_post_solver_complex_density_substitution(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    state = result.hf_run.state
    vector = np.asarray([1.0, 0.37 + 0.81j], dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    conventional_block = (vector[:, None] @ vector[None, :].conj())[:, :, None]
    stored_block = conventional_projector_to_stored_density(conventional_block)
    assert abs(stored_block[0, 1, 0].imag) > 1.0e-6
    for ik in range(state.nk):
        for sector in flavor_block_indices(
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        ):
            indices = np.asarray(sector, dtype=int)
            state.density[np.ix_(indices, indices, [ik])] = stored_block

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


def test_tbg_typed_export_rejects_state_v0_substitution(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    result.hf_run.state.v0 += 1.0e-6

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


def test_endpoint_inclusive_b0_is_refused_by_canonical_typed_export(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    endpoint_grid = build_b0_uniform_lattice(result.params, lk=1)
    endpoint_solution = replace(
        result.grid_solution,
        lattice_kvec=np.asarray(endpoint_grid.kvec, dtype=np.complex128),
        torus_mesh=None,
    )
    legacy_result = replace(result, grid_solution=endpoint_solution)
    with pytest.raises(ValueError, match="endpoint-inclusive B0"):
        b0_hf_benchmark_run_to_hf_run_result(legacy_result)


def test_tbg_zero_field_bare_restricted_hf_run_requires_grid_solution(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)

    with pytest.raises(ValueError, match="requires the matching BMSolution grid_solution"):
        tbg_zero_field_hf_run_to_hf_run_result(result.hf_run)


def test_tbg_typed_export_uses_actual_full_mode_metadata_and_rejects_mode_substitution(
    tmp_path: Path,
) -> None:
    full_result = _canonical_b0_hf_benchmark_result(tmp_path, hf_mode="full")
    public = tbg_zero_field_hf_run_to_hf_result(
        full_result.hf_run,
        grid_solution=full_result.grid_solution,
    )

    assert public.observables["hf_mode"] == "full"
    assert public.canonical_run_result.final_state.observables["hf_mode"] == "full"
    assert public.artifacts.metadata["workflow"] == "tbg.zero_field.full_hf.raw_run_result"
    assert public.artifacts.metadata["hf_mode"] == "full"

    restricted_result = _canonical_b0_hf_benchmark_result(tmp_path)
    with pytest.raises(ValueError, match="solver-issued"):
        replace(restricted_result.hf_run.provenance, hf_mode="full")


def test_tbg_typed_export_rejects_missing_run_provenance(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    with pytest.raises(ValueError, match="requires immutable typed run provenance"):
        tbg_zero_field_hf_run_to_hf_run_result(
            replace(
                result.hf_run,
                provenance=None,
                _production_issuer=None,
            ),
            grid_solution=result.grid_solution,
        )


def test_tbg_typed_export_rejects_mutated_live_solver_history(tmp_path: Path) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    changed = np.asarray(result.hf_run.iter_err, dtype=float).copy()
    changed[-1] += 1.0
    object.__setattr__(result.hf_run, "iter_err", changed)
    with pytest.raises(ValueError, match="iter_err hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )

@pytest.mark.parametrize(
    "field_name",
    ["density", "hamiltonian", "energies", "mu", "sigma_ztauz", "diagnostics"],
)
def test_tbg_typed_export_rejects_pre_export_final_state_mutation(
    tmp_path: Path,
    field_name: str,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    state = result.hf_run.state
    if field_name == "mu":
        state.mu += 1.0e-9
    elif field_name == "diagnostics":
        state.diagnostics["final_raw_norm"] += 1.0e-9
    else:
        getattr(state, field_name).flat[0] += 1.0e-9

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


def test_tbg_typed_export_rejects_beta_max_iter_seed_and_init_provenance_mismatch(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    provenance = result.hf_run.provenance
    assert isinstance(provenance, TBGZeroFieldHFRunProvenance)

    bad_cases = (
        ("beta", 2.0),
        ("nu", 0.25),
        ("precision", 2.0e-6),
        ("oda_stall_threshold", 2.0e-3),
        ("requested_max_iterations", 3),
        ("seed", 12),
        ("normalized_init_mode", "vp"),
        ("typed_receipt_fingerprint", "0" * 64),
        ("interaction_spec_fingerprint", "0" * 64),
        ("bm_generation_fingerprint", "0" * 64),
        ("mesh_fingerprint", "0" * 64),
    )
    for field_name, value in bad_cases:
        with pytest.raises(ValueError, match="solver-issued"):
            replace(provenance, **{field_name: value})

    public = tbg_zero_field_hf_run_to_hf_result(
        result.hf_run,
        grid_solution=result.grid_solution,
    )
    with pytest.raises(ValueError, match="HFConfig.max_iter"):
        tbg_zero_field_hf_run_to_hf_result(
            result.hf_run,
            grid_solution=result.grid_solution,
            config=replace(public.config, max_iter=provenance.requested_max_iterations + 1),
        )
    with pytest.raises(ValueError, match="exact chosen seed"):
        tbg_zero_field_hf_run_to_hf_result(
            result.hf_run,
            grid_solution=result.grid_solution,
            config=replace(public.config, seeds=("999",)),
        )
    for _field_name, forged_config, message in (
        (
            "filling",
            replace(public.config, filling=provenance.nu + 1.0e-5),
            "HFConfig.filling",
        ),
        (
            "precision",
            replace(public.config, precision=provenance.precision * 2.0),
            "HFConfig.precision",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            tbg_zero_field_hf_run_to_hf_result(
                result.hf_run,
                grid_solution=result.grid_solution,
                config=forged_config,
            )
    for metadata_key, forged_value in (
        ("beta", provenance.beta + 0.5),
        ("nu", provenance.nu + 1.0e-5),
        ("precision", provenance.precision * 2.0),
        ("oda_stall_threshold", provenance.oda_stall_threshold * 2.0),
        ("init_mode", "vp"),
        ("hf_mode", "full"),
    ):
        forged_metadata = dict(public.config.metadata)
        forged_metadata[metadata_key] = forged_value
        with pytest.raises(ValueError, match="conflicts with"):
            tbg_zero_field_hf_run_to_hf_result(
                result.hf_run,
                grid_solution=result.grid_solution,
                config=replace(public.config, metadata=forged_metadata),
            )


def test_tbg_density_export_rejects_substituted_nu_and_trace(tmp_path: Path) -> None:
    nu_result = _canonical_b0_hf_benchmark_result(tmp_path)
    nu_result.hf_run.state.nu = 0.25
    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            nu_result.hf_run,
            grid_solution=nu_result.grid_solution,
        )

    trace_result = _canonical_b0_hf_benchmark_result(tmp_path)
    trace_result.hf_run.state.density[0, 0, 0] -= 0.25
    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            trace_result.hf_run,
            grid_solution=trace_result.grid_solution,
        )


def test_tbg_typed_export_rejects_mutated_state_precision_and_oda_threshold(
    tmp_path: Path,
) -> None:
    precision_result = _canonical_b0_hf_benchmark_result(tmp_path)
    precision_result.hf_run.state.precision *= 2.0
    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            precision_result.hf_run,
            grid_solution=precision_result.grid_solution,
        )

    oda_result = _canonical_b0_hf_benchmark_result(tmp_path)
    oda_result.hf_run.state.diagnostics["oda_stall_threshold"] *= 2.0
    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            oda_result.hf_run,
            grid_solution=oda_result.grid_solution,
        )


def test_tbg_typed_export_rejects_post_solver_ensemble_density_substitution(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    state = result.hf_run.state
    for ik in range(state.nk):
        state.density[0, 0, ik] -= 0.25
        state.density[4, 4, ik] += 0.25

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


def test_tbg_density_export_rejects_traceless_hermitian_out_of_bounds_projector(
    tmp_path: Path,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    state = result.hf_run.state
    state.density[0, 0, 0] += 0.1
    state.density[1, 1, 0] -= 0.1

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "neg_inf"])
def test_tbg_density_export_rejects_nan_and_inf(
    tmp_path: Path,
    bad_value: float,
) -> None:
    result = _canonical_b0_hf_benchmark_result(tmp_path)
    result.hf_run.state.density[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="final state hash does not match"):
        tbg_zero_field_hf_run_to_hf_run_result(
            result.hf_run,
            grid_solution=result.grid_solution,
        )


def test_tbg_canonical_result_uses_actual_bm_params_and_generation_flags(
    tmp_path: Path,
) -> None:
    params = TBGParameters(
        dtheta_rad=1.07 * np.pi / 180.0,
        vf=2471.0,
        chemical_potential=3.5,
        w0=73.0,
        w1=108.0,
        delta=1.25,
        strain=2.0e-4,
        strain_angle_rad=0.37,
        poisson=0.19,
        beta_g=3.05,
        alpha=0.43,
        deformation_potential=2.75,
    )
    result = _canonical_b0_hf_benchmark_result(
        tmp_path,
        params=params,
        sigma_rotation=False,
        periodic_g_grid=False,
    )

    public = tbg_zero_field_hf_run_to_hf_result(
        result.hf_run,
        grid_solution=result.grid_solution,
    )

    assert public.model.params == params.to_summary_dict()
    assert public.model.lattice["sigma_rotation"] is False
    assert public.model.lattice["periodic_g_grid"] is False
    assert (
        public.observables["bm_generation_fingerprint"]
        == result.grid_solution.generation_fingerprint
    )
    assert (
        public.canonical_run_result.final_state.basis.physical_model.params["delta"]
        == params.delta
    )

    output_dir = tmp_path / "typed_nonzero_delta_artifacts"
    artifact_paths = write_b0_hf_benchmark_artifacts(output_dir, result)
    loaded = load_result(output_dir)
    assert loaded.model is not None
    assert loaded.model["params"]["delta"] == params.delta
    complete_archive = load_tbg_zero_field_complete_hf_state_archive_npz(
        artifact_paths[TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY]
    )
    assert complete_archive.params.delta == params.delta
