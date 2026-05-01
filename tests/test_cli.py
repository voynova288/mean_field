from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mean_field import cli
from mean_field.systems.tmbg import ValidationCheck, ValidationReport


def test_cli_hf_compare_case_dispatches_and_writes_outputs(monkeypatch, capsys, tmp_path: Path) -> None:
    case = SimpleNamespace(benchmark_id="synthetic_case", theta_deg=1.2, nu=-2)
    result = SimpleNamespace(
        case=case,
        grid_solution=SimpleNamespace(nk=4),
        hf_run=SimpleNamespace(iterations=3, exit_reason="converged", converged=True),
        path_result=SimpleNamespace(init_mode="bm", normalized_init_mode="bm", mu=-1.25),
        runtime=SimpleNamespace(total_elapsed_sec=12.5),
        runtime_parity=SimpleNamespace(total_elapsed_sec_ratio=1.25),
        parity=SimpleNamespace(
            kdist_max_abs_diff=0.0,
            max_abs_band_diff_mev=1.0e-6,
            rms_band_diff_mev=2.0e-7,
        ),
    )

    class FakeSuite:
        def get(self, benchmark_id: str) -> object:
            assert benchmark_id == "synthetic_case"
            return case

    called: dict[str, object] = {}

    def fake_run(case_obj: object, **kwargs: object) -> object:
        called["case"] = case_obj
        called["kwargs"] = kwargs
        return result

    def fake_write(output_dir: Path, run_result: object) -> dict[str, Path]:
        called["output_dir"] = output_dir
        called["result"] = run_result
        return {}

    monkeypatch.setattr(cli, "load_b0_suite", lambda: FakeSuite())
    monkeypatch.setattr(cli, "run_b0_hf_benchmark_case", fake_run)
    monkeypatch.setattr(cli, "write_b0_hf_benchmark_artifacts", fake_write)

    rc = cli.main(["hf", "compare-case", "synthetic_case", "--lk", "1", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert called["case"] is case
    assert called["kwargs"]["lk"] == 1
    assert called["output_dir"] == tmp_path
    out = capsys.readouterr().out
    assert "benchmark_id=synthetic_case" in out
    assert "max_abs_band_diff_meV=1.000000e-06" in out
    assert "total_elapsed_sec=12.500000" in out


def test_cli_hf_compare_suite_dispatches_and_writes_outputs(monkeypatch, capsys, tmp_path: Path) -> None:
    suite_result = SimpleNamespace(
        case_results=(SimpleNamespace(), SimpleNamespace()),
        total_elapsed_sec=33.0,
        max_kdist_max_abs_diff=3.0e-7,
        max_abs_band_diff_mev=4.0e-6,
    )
    called: dict[str, object] = {}

    def fake_run_suite(**kwargs: object) -> object:
        called["kwargs"] = kwargs
        return suite_result

    def fake_write_suite(output_dir: Path, result: object) -> dict[str, Path]:
        called["output_dir"] = output_dir
        called["result"] = result
        return {}

    monkeypatch.setattr(cli, "run_b0_hf_benchmark_suite", fake_run_suite)
    monkeypatch.setattr(cli, "write_b0_hf_suite_artifacts", fake_write_suite)

    rc = cli.main(["hf", "compare-suite", "case_b", "case_a", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert called["kwargs"]["benchmark_ids"] == ("case_b", "case_a")
    assert called["output_dir"] == tmp_path
    out = capsys.readouterr().out
    assert "cases=2" in out
    assert "total_elapsed_sec=33.000000" in out
    assert "max_abs_band_diff_meV=4.000000e-06" in out


def test_cli_benchmarks_runtime_list_prints_rows(monkeypatch, capsys) -> None:
    rows = [
        SimpleNamespace(
            benchmark_id="case_a",
            theta_deg=1.2,
            nu=-1,
            init_mode="vp",
            lk=19,
            lg=9,
            total_elapsed_sec=12.5,
            slurm_partition="test",
            slurm_cpus_per_task=8,
        ),
    ]
    monkeypatch.setattr(cli, "load_b0_runtime_benchmarks", lambda: rows)

    rc = cli.main(["benchmarks", "runtime-list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "case_a" in out
    assert "partition=test" in out


def test_cli_bm_benchmark_unstrained_dispatches_and_writes_outputs(monkeypatch, capsys, tmp_path: Path) -> None:
    result = SimpleNamespace(
        reference=SimpleNamespace(theta_deg=1.2),
        run=SimpleNamespace(
            path_solution=SimpleNamespace(nk=361),
            grid_solution=SimpleNamespace(nk=1089),
            runtime=SimpleNamespace(total_elapsed_sec=45.0),
        ),
        parity=SimpleNamespace(max_abs_band_diff_mev=1.0e-6, k_middle_gap_diff_mev=2.0e-7),
        runtime_parity=SimpleNamespace(total_elapsed_sec_ratio=0.9),
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(cli, "run_bm_unstrained_benchmark", lambda theta_deg: result)
    monkeypatch.setattr(cli, "write_bm_unstrained_benchmark_artifacts", lambda output_dir, benchmark_result: called.update({"output_dir": output_dir, "result": benchmark_result}) or {})

    rc = cli.main(["bm", "benchmark-unstrained", "1.2", "--output-dir", str(tmp_path)])

    assert rc == 0
    assert called["output_dir"] == tmp_path
    out = capsys.readouterr().out
    assert "theta=1.20" in out
    assert "total_elapsed_sec=45.000000" in out
    assert "total_elapsed_sec_ratio=9.000000e-01" in out


def test_cli_benchmarks_bm_runtime_list_prints_rows(monkeypatch, capsys) -> None:
    rows = [
        SimpleNamespace(
            theta_deg=1.2,
            points_per_segment=120,
            lg=9,
            grid_lk=33,
            total_elapsed_sec=51.5,
            slurm_partition="regular",
            slurm_cpus_per_task=28,
        ),
    ]
    monkeypatch.setattr(cli, "load_bm_unstrained_runtime_benchmarks", lambda: rows)

    rc = cli.main(["benchmarks", "bm-runtime-list"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "theta=1.20" in out
    assert "partition=regular" in out


def test_cli_tmbg_reproduce_checkpoints_dispatches_and_writes_metadata(monkeypatch, capsys, tmp_path: Path) -> None:
    report = ValidationReport(
        title="tMBG Park 2020 Checkpoint Validation",
        checks=(
            ValidationCheck(name="CP1.minimal_magic_angle_bandwidth", status="pass", detail="ok", value=0.0012),
            ValidationCheck(name="CP3.delta_0_opposite_valley", status="skipped", detail="skipped"),
        ),
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_ensure_not_running_compute_on_login_node",
        lambda workload_name: called.update({"workload_name": workload_name}),
    )
    monkeypatch.setattr(
        cli,
        "reproduce_paper_checkpoints",
        lambda **kwargs: called.update({"kwargs": kwargs}) or report,
    )
    monkeypatch.setattr(
        cli,
        "collect_runtime_environment",
        lambda: SimpleNamespace(
            hostname="node012",
            cpu_model="Synthetic CPU",
            slurm_partition="regular",
            slurm_nodelist="node012",
            slurm_cpus_per_task=28,
            blas_threads=28,
            sys_cpu_threads=56,
            process_count=1,
            jit_warmup_included=False,
            python_version="3.13.0",
            numpy_version="2.3.0",
        ),
    )

    rc = cli.main(
        [
            "tmbg",
            "reproduce-checkpoints",
            "--output-dir",
            str(tmp_path),
            "--n-shells",
            "4",
            "--points-per-segment",
            "80",
            "--path-n-bands",
            "40",
            "--topology-mesh-size",
            "12",
            "--topology-n-bands",
            "24",
            "--valley",
            "-1",
            "--skip-opposite-valley",
            "--cp4-delta-abs",
            "0.05",
            "--cp6-staggered-potential",
            "0.02",
            "--cp6-staggered-potential",
            "-0.02",
        ]
    )

    assert rc == 0
    assert called["workload_name"] == "tMBG checkpoints"
    assert called["kwargs"]["output_dir"] == tmp_path
    assert called["kwargs"]["n_shells"] == 4
    assert called["kwargs"]["points_per_segment"] == 80
    assert called["kwargs"]["path_n_bands"] == 40
    assert called["kwargs"]["topology_mesh_size"] == 12
    assert called["kwargs"]["topology_n_bands"] == 24
    assert called["kwargs"]["valley"] == -1
    assert called["kwargs"]["verify_opposite_valley"] is False
    assert called["kwargs"]["cp4_delta_abs"] == 0.05
    assert called["kwargs"]["cp6_staggered_potentials"] == (0.02, -0.02)

    runtime_summary = (tmp_path / "runtime_summary.txt").read_text(encoding="utf-8")
    assert "runner_kind=tmbg_paper_checkpoints" in runtime_summary
    assert "verify_opposite_valley=false" in runtime_summary
    assert "cp6_staggered_potentials=0.02,-0.02" in runtime_summary

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parameters"]["n_shells"] == 4
    assert metadata["parameters"]["valley"] == -1
    assert metadata["report"]["failure_count"] == 0
    assert metadata["report"]["skipped_count"] == 1
    assert metadata["artifacts"]["runtime_summary_txt"] == str(tmp_path / "runtime_summary.txt")

    out = capsys.readouterr().out
    assert "status=pass_with_skips" in out
    assert f"output_dir={tmp_path}" in out


def test_cli_tmbg_ktilde_diagnostics_dispatches_and_writes_metadata(monkeypatch, capsys, tmp_path: Path) -> None:
    report = ValidationReport(
        title="tMBG Ktilde Symmetry Diagnostics",
        checks=(
            ValidationCheck(name="D1.chiral_limit_ktilde_touching", status="pass", detail="ok", value=0.0),
            ValidationCheck(name="D2.delta_p060_opens_ktilde_gap", status="pass", detail="open", value=0.12),
        ),
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_ensure_not_running_compute_on_login_node",
        lambda workload_name: called.update({"workload_name": workload_name}),
    )
    monkeypatch.setattr(
        cli,
        "diagnose_ktilde_symmetry",
        lambda **kwargs: called.update({"kwargs": kwargs}) or report,
    )
    monkeypatch.setattr(
        cli,
        "collect_runtime_environment",
        lambda: SimpleNamespace(
            hostname="node003",
            cpu_model="Synthetic CPU",
            slurm_partition="test",
            slurm_nodelist="node003",
            slurm_cpus_per_task=4,
            blas_threads=4,
            sys_cpu_threads=8,
            process_count=1,
            jit_warmup_included=False,
            python_version="3.13.0",
            numpy_version="2.3.0",
        ),
    )

    rc = cli.main(
        [
            "tmbg",
            "diagnose-ktilde-symmetry",
            "--output-dir",
            str(tmp_path),
            "--theta-deg",
            "1.23",
            "--n-shells",
            "4",
            "--valley",
            "-1",
        ]
    )

    assert rc == 0
    assert called["workload_name"] == "tMBG Ktilde symmetry diagnostics"
    assert called["kwargs"]["output_dir"] == tmp_path
    assert called["kwargs"]["theta_deg"] == 1.23
    assert called["kwargs"]["n_shells"] == 4
    assert called["kwargs"]["valley"] == -1

    runtime_summary = (tmp_path / "runtime_summary.txt").read_text(encoding="utf-8")
    assert "runner_kind=tmbg_ktilde_symmetry_diagnostics" in runtime_summary
    assert "theta_deg=1.23" in runtime_summary
    assert "valley=-1" in runtime_summary

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["parameters"]["theta_deg"] == 1.23
    assert metadata["parameters"]["n_shells"] == 4
    assert metadata["parameters"]["valley"] == -1
    assert metadata["report"]["failure_count"] == 0
    assert metadata["artifacts"]["runtime_summary_txt"] == str(tmp_path / "runtime_summary.txt")

    out = capsys.readouterr().out
    assert "status=pass" in out
    assert f"output_dir={tmp_path}" in out
