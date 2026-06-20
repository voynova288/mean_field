from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from mean_field.core.io import write_json_artifact
from mean_field.runtime import ensure_not_running_compute_on_login_node as _ensure_not_running_compute_on_login_node


def ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    """Guard devtool entrypoints that can launch BLAS/eigensolver/HF work."""

    _ensure_not_running_compute_on_login_node(workload_name)


def write_json(path: Path, payload: object, *, sort_keys: bool = True) -> None:
    write_json_artifact(payload, path, sort_keys=sort_keys)


def _parse_csv_values(text: str, converter, item_name: str):
    values = tuple(converter(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one comma-separated {item_name}.")
    return values


def parse_csv_floats(text: str) -> tuple[float, ...]:
    return _parse_csv_values(text, float, "float")


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return _parse_csv_values(text, int, "integer")


def parse_csv_strings(text: str, item_name: str = "mode") -> tuple[str, ...]:
    return _parse_csv_values(text, str, item_name)


def complex_to_pairs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    return np.stack([values.real, values.imag], axis=-1)


def complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected final axis of length 2 for complex pairs, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)
