from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Literal

import numpy as np


def ensure_not_running_compute_on_login_node(workload_name: str) -> None:
    """Guard devtool entrypoints that can launch BLAS/eigensolver/HF work."""
    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().strip().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit(
            f"Refusing to run {workload_name} on login node {hostname}; submit it through Slurm from login002."
        )


def write_json(path: Path, payload: object, *, sort_keys: bool = True) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


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


def select_flat_pair_window(
    total_bands: int,
    flat_pair: tuple[int, int],
    bands_per_side: int,
    *,
    mode: Literal["edges", "center"] = "edges",
) -> tuple[int, ...]:
    if mode == "center":
        center = (int(flat_pair[0]) + int(flat_pair[1])) // 2
        lower = max(0, center - int(bands_per_side))
        upper = min(int(total_bands), center + int(bands_per_side) + 2)
    else:
        lower = max(0, int(flat_pair[0]) - int(bands_per_side))
        upper = min(int(total_bands), int(flat_pair[1]) + int(bands_per_side) + 1)
    return tuple(range(lower, upper))


def select_energy_window_bands(
    energies: np.ndarray,
    *,
    emin: float,
    emax: float,
    fallback_each_side: int,
    fallback_include_center: bool = True,
) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    band_min = np.min(energies, axis=0)
    band_max = np.max(energies, axis=0)
    indices = np.nonzero((band_max >= float(emin)) & (band_min <= float(emax)))[0]
    if indices.size > 0:
        return indices
    center = energies.shape[1] // 2
    start = max(0, center - int(fallback_each_side))
    center_padding = 1 if fallback_include_center else 0
    stop = min(energies.shape[1], center + int(fallback_each_side) + center_padding)
    return np.arange(start, stop, dtype=int)
