from __future__ import annotations

import os
from pathlib import Path


def _find_package_root() -> Path:
    override = os.environ.get("MEAN_FIELD_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "benchmarks").is_dir() and ((parent / "pyproject.toml").is_file() or (parent / "src").is_dir()):
            return parent
    return current.parents[2]


PACKAGE_ROOT = _find_package_root()
BENCHMARKS_ROOT = PACKAGE_ROOT / "benchmarks"
B0_BENCHMARK_ROOT = BENCHMARKS_ROOT / "b0"
BM_UNSTRAINED_BENCHMARK_ROOT = B0_BENCHMARK_ROOT / "bm_inputs" / "unstrained_path"
