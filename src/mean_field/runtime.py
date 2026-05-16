from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import platform
import socket
import sys


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
    numba_threads: int
    sys_cpu_threads: int
    process_count: int
    backend_choice: str
    threadpoolctl_info: tuple[dict[str, object], ...]
    thread_env: dict[str, str]
    jit_warmup_included: bool
    python_version: str
    numpy_version: str


@dataclass(frozen=True)
class ThreadingConfiguration:
    blas_threads: int
    numba_threads: int
    slurm_cpus_per_task: int
    process_count: int
    backend_choice: str
    threadpoolctl_info: tuple[dict[str, object], ...]
    thread_env: dict[str, str]


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


def _threadpool_info() -> tuple[dict[str, object], ...]:
    try:
        from threadpoolctl import threadpool_info  # type: ignore[import-not-found]
    except Exception:
        return ()

    entries: list[dict[str, object]] = []
    try:
        raw_entries = threadpool_info()
    except Exception:
        return ()
    for raw in raw_entries:
        normalized: dict[str, object] = {}
        for key, value in dict(raw).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[str(key)] = value
            else:
                normalized[str(key)] = str(value)
        entries.append(normalized)
    return tuple(entries)


def _set_threadpool_limits(blas_threads: int) -> None:
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]

        threadpool_limits(limits=int(blas_threads))
    except Exception:
        pass


def _configure_numba_threads(numba_threads: int) -> int:
    requested = max(1, int(numba_threads))
    try:
        import numba  # type: ignore[import-not-found]

        numba.set_num_threads(requested)
        return int(numba.get_num_threads())
    except Exception:
        return requested


def _default_backend_choice() -> str:
    explicit = os.environ.get("MEAN_FIELD_HF_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    disabled = os.environ.get("MEAN_FIELD_HF_DISABLE_NUMBA", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return "numpy"
    try:
        import numba  # noqa: F401  # type: ignore[import-not-found]
    except Exception:
        return "numpy"
    return "numba"


def _thread_env_snapshot() -> dict[str, str]:
    names = _BLAS_THREAD_ENV_VARS + ("NUMBA_NUM_THREADS",)
    return {name: os.environ.get(name, "") for name in names}


def configure_threading(
    *,
    blas_threads: int | None = None,
    numba_threads: int | None = None,
    process_count: int | None = None,
    backend_choice: str | None = None,
) -> ThreadingConfiguration:
    """Configure local compute threading for HF/cRPA runs.

    BLAS/OpenMP libraries default to one thread to avoid nested parallelism
    when HF kernels use Numba or Slurm process-level parallelism.  Numba uses
    ``SLURM_CPUS_PER_TASK`` by default so the explicitly parallel kernels can
    consume the requested CPU allocation.
    """

    sys_cpu_threads = max(1, os.cpu_count() or 1)
    slurm_cpus_per_task = _read_env_int("SLURM_CPUS_PER_TASK") or sys_cpu_threads
    ntasks = _read_env_int("SLURM_NTASKS")
    resolved_process_count = ntasks if ntasks is not None else (process_count if process_count is not None else 1)

    resolved_blas_threads = max(1, int(blas_threads if blas_threads is not None else 1))
    for name in _BLAS_THREAD_ENV_VARS:
        os.environ[name] = str(resolved_blas_threads)
    _set_threadpool_limits(resolved_blas_threads)

    resolved_numba_threads = max(1, int(numba_threads if numba_threads is not None else slurm_cpus_per_task))
    os.environ["NUMBA_NUM_THREADS"] = str(resolved_numba_threads)
    actual_numba_threads = _configure_numba_threads(_read_env_int("NUMBA_NUM_THREADS") or resolved_numba_threads)

    resolved_backend = str(backend_choice or _default_backend_choice())
    return ThreadingConfiguration(
        blas_threads=_detect_blas_threads(slurm_cpus_per_task),
        numba_threads=int(actual_numba_threads),
        slurm_cpus_per_task=int(slurm_cpus_per_task),
        process_count=max(1, int(resolved_process_count)),
        backend_choice=resolved_backend,
        threadpoolctl_info=_threadpool_info(),
        thread_env=_thread_env_snapshot(),
    )


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def collect_runtime_environment(
    *,
    process_count: int | None = None,
    jit_warmup_included: bool = False,
    backend_choice: str | None = None,
    configure_threads: bool = True,
) -> RuntimeEnvironment:
    import numpy as np

    sys_cpu_threads = max(1, os.cpu_count() or 1)
    if configure_threads:
        threading = configure_threading(process_count=process_count, backend_choice=backend_choice)
    else:
        slurm_cpus_per_task = _read_env_int("SLURM_CPUS_PER_TASK") or sys_cpu_threads
        ntasks = _read_env_int("SLURM_NTASKS")
        resolved_process_count = ntasks if ntasks is not None else (process_count if process_count is not None else 1)
        threading = ThreadingConfiguration(
            blas_threads=_detect_blas_threads(slurm_cpus_per_task),
            numba_threads=_read_env_int("NUMBA_NUM_THREADS") or 1,
            slurm_cpus_per_task=int(slurm_cpus_per_task),
            process_count=max(1, int(resolved_process_count)),
            backend_choice=str(backend_choice or _default_backend_choice()),
            threadpoolctl_info=_threadpool_info(),
            thread_env=_thread_env_snapshot(),
        )

    return RuntimeEnvironment(
        hostname=socket.gethostname(),
        cpu_model=_read_cpu_model(),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
        slurm_nodelist=os.environ.get("SLURM_JOB_NODELIST", ""),
        slurm_cpus_per_task=threading.slurm_cpus_per_task,
        blas_threads=threading.blas_threads,
        numba_threads=threading.numba_threads,
        sys_cpu_threads=sys_cpu_threads,
        process_count=threading.process_count,
        backend_choice=threading.backend_choice,
        threadpoolctl_info=threading.threadpoolctl_info,
        thread_env=threading.thread_env,
        jit_warmup_included=bool(jit_warmup_included),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
    )


def safe_ratio(measured: float, reference: float) -> float | None:
    if abs(reference) < 1e-15:
        return None
    return measured / reference
