#!/usr/bin/env python3
"""Report TBG HF bandwidths from converged SCF-grid data only."""

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
        raise ValueError(f"No SCF path rows found in {path}")
    return band_columns, np.asarray(kdist, dtype=float), np.asarray(rows, dtype=float)


def _sorted_band_report(energies_by_k: np.ndarray, *, nu: float) -> dict[str, object]:
    energies = np.asarray(energies_by_k, dtype=float)
    if energies.ndim != 2:
        raise ValueError(f"Expected a 2D energy array, got {energies.shape}")
    nk, nt = energies.shape
    occ = restricted_occupied_bands_per_k(float(nu), int(nt))
    if occ <= 0 or occ >= nt:
        raise ValueError(f"Cannot define central gap for nu={nu} with nt={nt}, occ={occ}")

    sorted_per_k = np.sort(energies, axis=1)
    valence = sorted_per_k[:, occ - 1]
    conduction = sorted_per_k[:, occ]
    direct_gap = conduction - valence
    band_widths = np.max(sorted_per_k, axis=0) - np.min(sorted_per_k, axis=0)
    conduction_widths = band_widths[occ:]

    return {
        "point_count": int(nk),
        "band_count": int(nt),
        "occupied_bands_per_k": int(occ),
        "valence_band_count": int(occ),
        "conduction_band_count": int(nt - occ),
        "direct_gap_mev": float(np.min(direct_gap)),
        "indirect_gap_mev": float(np.min(conduction) - np.max(valence)),
        "top_valence_width_mev": float(band_widths[occ - 1]),
        "first_conduction_width_mev": float(band_widths[occ]),
        "conduction_widths_mev": [float(value) for value in conduction_widths],
        "max_conduction_width_mev": float(np.max(conduction_widths)),
        "mean_conduction_width_mev": float(np.mean(conduction_widths)),
        "all_sorted_band_widths_mev": [float(value) for value in band_widths],
        "valence_top_min_mev": float(np.min(valence)),
        "valence_top_max_mev": float(np.max(valence)),
        "first_conduction_min_mev": float(np.min(conduction)),
        "first_conduction_max_mev": float(np.max(conduction)),
    }


def report_bandwidth(*, state_path: Path, scf_tsv: Path, nu: float) -> dict[str, object]:
    with np.load(state_path, allow_pickle=False) as data:
        energies = np.asarray(data["energies"], dtype=float)
        hamiltonian = np.asarray(data["hamiltonian"])
        nt = int(hamiltonian.shape[0])
        if energies.shape[0] == nt:
            grid_energies_by_k = energies.T
        elif energies.shape[1] == nt:
            grid_energies_by_k = energies
        else:
            raise ValueError(
                f"Cannot align energies shape {energies.shape} with Hamiltonian flavor dimension nt={nt}"
            )
        state_nu = float(np.asarray(data["nu"]).reshape(-1)[0]) if "nu" in data else float(nu)
        converged = bool(np.asarray(data["converged"]).reshape(-1)[0]) if "converged" in data else None
        if "iterations" in data:
            iterations = int(np.asarray(data["iterations"]).reshape(-1)[0])
        elif "iter_err" in data:
            iterations = int(np.asarray(data["iter_err"]).size)
        else:
            iterations = None
        exit_reason = str(np.asarray(data["exit_reason"]).reshape(-1)[0]) if "exit_reason" in data else ""

    _band_columns, _kdist, path_energies_by_k = _read_scf_path(scf_tsv)
    path_report = _sorted_band_report(path_energies_by_k, nu=nu)
    grid_report = _sorted_band_report(grid_energies_by_k, nu=nu)

    return {
        "source_state": str(state_path),
        "source_scf_tsv": str(scf_tsv),
        "nu": float(nu),
        "state_nu": float(state_nu),
        "state_converged": converged,
        "state_iterations": iterations,
        "state_exit_reason": exit_reason,
        "scf_path": path_report,
        "full_scf_grid": grid_report,
        "paper_reference_note": (
            "Zhang Table III gives the nu=-3 HF+cRPA gap as 4.4 meV. "
            "The local SCF-path central gap and full-grid indirect gap should be compared separately."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scf-tsv", type=Path, required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = report_bandwidth(state_path=args.state, scf_tsv=args.scf_tsv, nu=float(args.nu))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
