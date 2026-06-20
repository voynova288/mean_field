from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mean_field import cli
from mean_field.api import load_result, required_artifact_files
from mean_field.systems.tmbg import ValidationCheck, ValidationReport


def _fake_runtime_environment() -> SimpleNamespace:
    return SimpleNamespace(
        hostname="node012",
        cpu_model="Synthetic CPU",
        slurm_partition="regular",
        slurm_nodelist="node012",
        slurm_cpus_per_task=28,
        blas_threads=28,
        numba_threads=1,
        sys_cpu_threads=56,
        process_count=1,
        backend_choice="numpy",
        threadpoolctl_info=(),
        thread_env={},
        jit_warmup_included=False,
        python_version="3.13.0",
        numpy_version="2.3.0",
    )


def test_cli_tmbg_ktilde_diagnostics_writes_contract_sidecars(monkeypatch, tmp_path: Path) -> None:
    report = ValidationReport(
        title="tMBG Ktilde Symmetry Diagnostics",
        checks=(
            ValidationCheck(name="D1.chiral_limit_ktilde_touching", status="pass", detail="ok", value=0.0),
            ValidationCheck(name="D2.delta_p060_opens_ktilde_gap", status="pass", detail="open", value=0.12),
        ),
    )
    called: dict[str, object] = {}
    monkeypatch.setattr(cli, "_ensure_not_running_compute_on_login_node", lambda workload_name: None)
    monkeypatch.setattr(cli, "collect_runtime_environment", _fake_runtime_environment)
    monkeypatch.setattr(cli, "diagnose_ktilde_symmetry", lambda **kwargs: called.update(kwargs) or report)

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
    assert called["output_dir"] == tmp_path
    assert {path.name for path in tmp_path.iterdir()} >= set(required_artifact_files())
    result = load_result(tmp_path)
    assert result.manifest["metadata"]["workflow"] == "tmbg.ktilde_diagnostics"
    assert result.manifest["files"]["runtime_summary_txt"] == "runtime_summary.txt"
    assert result.config is not None and result.config["parameters"]["theta_deg"] == 1.23
    assert result.conventions is not None and result.conventions["topology_convention"]
    assert result.validation is not None and result.validation["status"] == "pass"
    assert result.observables is not None and result.observables["report"]["failure_count"] == 0
