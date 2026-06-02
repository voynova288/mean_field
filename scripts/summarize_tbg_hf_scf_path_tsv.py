#!/usr/bin/env python3
"""Summarize direct-gap and bandwidth metrics from a TBG SCF-grid path TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mean_field.systems.tbg.zero_field.hf import restricted_occupied_bands_per_k


METADATA_COLUMNS = {
    "path_index",
    "path_k_dist",
    "k_dist",
    "distance_to_path",
    "path_kx",
    "path_ky",
    "projected_kx",
    "projected_ky",
    "grid_index",
    "grid_kx",
    "grid_ky",
}


def _read_scf_path(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        band_columns = [name for name in reader.fieldnames if name not in METADATA_COLUMNS]
        kdist: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            kdist.append(float(row["k_dist"]))
            rows.append([float(row[name]) for name in band_columns])
    if not rows:
        raise ValueError(f"No path rows found in {path}")
    return band_columns, np.asarray(kdist, dtype=float), np.asarray(rows, dtype=float)


def summarize(tsv_path: Path, *, nu: float) -> dict[str, object]:
    band_columns, kdist, energies_by_path = _read_scf_path(tsv_path)
    n_path, nt = energies_by_path.shape
    occ = restricted_occupied_bands_per_k(float(nu), int(nt))
    if occ <= 0 or occ >= nt:
        raise ValueError(f"Cannot define central gap for nu={nu} with nt={nt}, occ={occ}")

    sorted_per_path = np.sort(energies_by_path, axis=1)
    valence = sorted_per_path[:, occ - 1]
    conduction = sorted_per_path[:, occ]
    direct_gap = conduction - valence
    direct_index = int(np.argmin(direct_gap))
    indirect_gap = float(np.min(conduction) - np.max(valence))

    band_widths = np.max(sorted_per_path, axis=0) - np.min(sorted_per_path, axis=0)
    conduction_widths = band_widths[occ:]

    return {
        "source_tsv": str(tsv_path),
        "nu": float(nu),
        "path_point_count": int(n_path),
        "band_count": int(nt),
        "occupied_bands_per_k": int(occ),
        "valence_band_count": int(occ),
        "conduction_band_count": int(nt - occ),
        "direct_gap_mev": float(np.min(direct_gap)),
        "direct_gap_path_index": int(direct_index),
        "direct_gap_k_dist": float(kdist[direct_index]),
        "indirect_gap_mev": indirect_gap,
        "valence_top_max_mev": float(np.max(valence)),
        "valence_top_min_mev": float(np.min(valence)),
        "first_conduction_min_mev": float(np.min(conduction)),
        "first_conduction_max_mev": float(np.max(conduction)),
        "top_valence_width_mev": float(band_widths[occ - 1]),
        "first_conduction_width_mev": float(band_widths[occ]),
        "conduction_widths_mev": [float(value) for value in conduction_widths],
        "max_conduction_width_mev": float(np.max(conduction_widths)),
        "mean_conduction_width_mev": float(np.mean(conduction_widths)),
        "band_columns": band_columns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = summarize(args.tsv, nu=float(args.nu))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
