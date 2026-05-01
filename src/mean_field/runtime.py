from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import platform
import socket
import sys

import numpy as np


_BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RuntimeEnvironment:
    hostname: str
    cpu_model: str
    slurm_partition: str
    slurm_nodelist: str
    slurm_cpus_per_task: int
    blas_threads: int
    sys_cpu_threads: int
    process_count: int
    jit_warmup_included: bool
    python_version: str
    numpy_version: str


def _read_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _detect_blas_threads(slurm_cpus_per_task: int) -> int:
    try:
        from threadpoolctl import threadpool_info  # type: ignore[import-not-found]

        counts = [int(info["num_threads"]) for info in threadpool_info() if int(info.get("num_threads", 0)) > 0]
        if counts:
            return max(counts)
    except Exception:
        pass

    for name in _BLAS_THREAD_ENV_VARS:
        value = _read_env_int(name)
        if value is not None:
            return value
    return max(1, slurm_cpus_per_task)


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def collect_runtime_environment(*, process_count: int | None = None, jit_warmup_included: bool = False) -> RuntimeEnvironment:
    sys_cpu_threads = max(1, os.cpu_count() or 1)
    slurm_cpus_per_task = _read_env_int("SLURM_CPUS_PER_TASK") or sys_cpu_threads
    ntasks = _read_env_int("SLURM_NTASKS")
    resolved_process_count = ntasks if ntasks is not None else (process_count if process_count is not None else 1)

    return RuntimeEnvironment(
        hostname=socket.gethostname(),
        cpu_model=_read_cpu_model(),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
        slurm_nodelist=os.environ.get("SLURM_JOB_NODELIST", ""),
        slurm_cpus_per_task=slurm_cpus_per_task,
        blas_threads=_detect_blas_threads(slurm_cpus_per_task),
        sys_cpu_threads=sys_cpu_threads,
        process_count=max(1, int(resolved_process_count)),
        jit_warmup_included=bool(jit_warmup_included),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
    )


def safe_ratio(measured: float, reference: float) -> float | None:
    if abs(reference) < 1e-15:
        return None
    return measured / reference
