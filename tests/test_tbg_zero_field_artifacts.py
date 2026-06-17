from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mean_field.api import load_result, required_artifact_files
from mean_field.benchmarks import BMUnstrainedReference, BenchmarkCase
from mean_field.core.hf import FlavorBandData
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
    complex_to_pair,
    empty_overlap_block_set,
    write_b0_hf_benchmark_artifacts,
    write_b0_hf_benchmark_contract_sidecars,
    write_b0_hf_suite_artifacts,
    write_b0_hf_suite_contract_sidecars,
    write_bm_unstrained_benchmark_artifacts,
    write_bm_unstrained_benchmark_contract_sidecars,
)


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
    assert loaded.model is not None and loaded.model["lattice"]["g1_nm_inv_pair"] == complex_to_pair(result.run.params.g1)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_bm_unstrained_benchmark_contract_sidecars(output_dir, result)


def _benchmark_case(tmp_path: Path) -> BenchmarkCase:
    return BenchmarkCase(
        benchmark_id="unit_b0",
        theta_deg=1.05,
        nu=2,
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
        lg=1,
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
