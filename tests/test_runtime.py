from __future__ import annotations

from dataclasses import asdict

import pytest

import mean_field.runtime as runtime
from mean_field.runtime import collect_runtime_environment, configure_threading, ensure_not_running_compute_on_login_node


def test_configure_threading_sets_blas_default_and_numba_from_slurm(monkeypatch) -> None:
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setenv("SLURM_NTASKS", "3")

    config = configure_threading(backend_choice="numba-test")

    assert config.slurm_cpus_per_task == 7
    assert config.process_count == 3
    assert config.thread_env["OPENBLAS_NUM_THREADS"] == "1"
    assert config.thread_env["MKL_NUM_THREADS"] == "1"
    assert config.thread_env["NUMBA_NUM_THREADS"] == "7"
    assert config.blas_threads >= 1
    assert config.numba_threads >= 1
    assert config.backend_choice == "numba-test"
    assert isinstance(config.threadpoolctl_info, tuple)


def test_collect_runtime_environment_records_threading_and_backend_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    monkeypatch.setenv("MEAN_FIELD_HF_BACKEND", "numpy")

    env = collect_runtime_environment(process_count=2)
    payload = asdict(env)

    assert payload["slurm_cpus_per_task"] == 5
    assert payload["process_count"] >= 1
    assert payload["blas_threads"] >= 1
    assert payload["numba_threads"] >= 1
    assert payload["backend_choice"] == "numpy"
    assert "threadpoolctl_info" in payload
    assert payload["thread_env"]["NUMBA_NUM_THREADS"]


def test_compute_guard_rejects_login_nodes_without_slurm_allocation(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "login002")

    with pytest.raises(SystemExit, match="Refusing to run demo on login node login002"):
        ensure_not_running_compute_on_login_node("demo")


def test_compute_guard_allows_slurm_allocations_on_login_named_hosts(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "login002")

    ensure_not_running_compute_on_login_node("demo")
