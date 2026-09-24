from __future__ import annotations

import pytest

import mean_field.runtime as runtime
from mean_field.runtime import ensure_not_running_compute_on_login_node


def test_compute_guard_rejects_login_nodes_without_slurm_allocation(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "login002")

    with pytest.raises(SystemExit, match="Refusing to run demo on login node login002"):
        ensure_not_running_compute_on_login_node("demo")


def test_compute_guard_allows_slurm_allocations_on_login_named_hosts(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "login002")

    ensure_not_running_compute_on_login_node("demo")
