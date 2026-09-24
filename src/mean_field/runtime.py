"""Runtime safety guard for compute-heavy entrypoints."""

from __future__ import annotations

import os
import socket


def ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    """Guard compute-heavy entrypoints from accidental login-node execution."""

    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().strip().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit(
            f"Refusing to run {workload_name} on login node {hostname}; "
            "submit it through Slurm from login002."
        )


__all__ = ["ensure_not_running_compute_on_login_node"]
