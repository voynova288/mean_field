#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import RLGhBNModel, diagonalize_hamiltonian


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected complex pairs on final axis, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)


def _model_from_manifest(entry: dict[str, object]) -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=int(entry["layer_count"]),
        xi=int(entry["xi"]),
        theta_deg=float(entry["theta_deg"]),
        displacement_field_mev=float(entry["displacement_field_mev"]),
        shell_count=int(entry["shell_count"]),
    )


def _segment_bounds(path_archive: np.lib.npyio.NpzFile, segment: str) -> tuple[int, int]:
    node_indices = np.asarray(path_archive["node_indices"], dtype=int)
    labels = [str(value) for value in path_archive["labels"]]
    normalized = [label.replace("$", "").replace("\\", "") for label in labels]
    if segment == "all":
        return 0, int(path_archive["kdist"].size) - 1
    if segment == "gamma_mprime":
        gamma_positions = [idx for idx, label in enumerate(normalized) if "Gamma" in label]
        mprime_positions = [idx for idx, label in enumerate(normalized) if "M'_M" in label or "M'" in label]
        if not gamma_positions or not mprime_positions:
            raise ValueError(f"Could not locate Gamma/Mprime in labels {labels}")
        # Fig. 6 path is Gamma-K-Kprime-Gamma-Mprime-M-Gamma.
        label_start = gamma_positions[-2]
        label_stop = mprime_positions[0]
        return int(node_indices[label_start]) - 1, int(node_indices[label_stop]) - 1
    raise ValueError(f"Unknown segment {segment!r}")


def _diagonalize_path(model: RLGhBNModel, kvec: np.ndarray, valley: int) -> np.ndarray:
    values = np.zeros((kvec.size, model.matrix_dim), dtype=float)
    for idx, kval in enumerate(np.asarray(kvec, dtype=np.complex128).reshape(-1)):
        eigvals, _ = diagonalize_hamiltonian(
            complex(kval),
            model.lattice,
            model.params,
            valley=int(valley),
            n_bands=None,
        )
        values[idx, :] = eigvals
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose RLG/hBN active-window isolation on saved exact SCF path points."
    )
    parser.add_argument("--basis-cache", type=Path, required=True)
    parser.add_argument("--path-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment", choices=("gamma_mprime", "all"), default="gamma_mprime")
    args = parser.parse_args()

    ensure_not_running_compute_on_login_node("RLG/hBN full single-particle active-window diagnostic")

    manifest = _read_json(args.basis_cache / "manifest.json")
    extra = manifest["extra"]
    if not isinstance(extra, dict):
        raise TypeError("manifest['extra'] must be a dictionary")
    active_indices = np.asarray(extra["active_band_indices"], dtype=int)
    if active_indices.ndim != 1 or active_indices.size == 0:
        raise ValueError(f"Invalid active indices {active_indices}")
    lower_neighbor = int(active_indices[0]) - 1
    upper_neighbor = int(active_indices[-1]) + 1
    if lower_neighbor < 0 or upper_neighbor >= int(manifest["model"]["lattice"]["matrix_dim"]):  # type: ignore[index]
        raise ValueError("Active window has no lower or upper neighbor in the full spectrum")

    physical_model = _model_from_manifest(manifest["model"])  # type: ignore[arg-type]
    basis_model = _model_from_manifest(extra["basis_model"])  # type: ignore[arg-type]

    archive = np.load(args.path_npz, allow_pickle=True)
    start, stop = _segment_bounds(archive, args.segment)
    kvec = _complex_from_pairs(archive["kvec_lookup_nm_inv"])[start : stop + 1]
    frac = np.asarray(archive["frac_lookup"], dtype=float)[start : stop + 1]
    mesh_indices = np.asarray(archive["mesh_indices"], dtype=int)[start : stop + 1]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "basis_cache": str(args.basis_cache),
        "path_npz": str(args.path_npz),
        "segment": str(args.segment),
        "active_band_indices": [int(value) for value in active_indices],
        "lower_neighbor": int(lower_neighbor),
        "upper_neighbor": int(upper_neighbor),
        "physical_v_mev": float(physical_model.params.displacement_field_mev),
        "screened_u_mev": float(basis_model.params.displacement_field_mev),
    }

    for valley in (1, -1):
        eig_u = _diagonalize_path(basis_model, kvec, valley=valley)
        eig_v = _diagonalize_path(physical_model, kvec, valley=valley)
        lower_gap_u = eig_u[:, active_indices[0]] - eig_u[:, lower_neighbor]
        upper_gap_u = eig_u[:, upper_neighbor] - eig_u[:, active_indices[-1]]
        lower_gap_v = eig_v[:, active_indices[0]] - eig_v[:, lower_neighbor]
        upper_gap_v = eig_v[:, upper_neighbor] - eig_v[:, active_indices[-1]]

        valley_key = "K" if valley == 1 else "Kprime"
        summary[valley_key] = {
            "min_lower_gap_u_mev": float(np.min(lower_gap_u)),
            "min_upper_gap_u_mev": float(np.min(upper_gap_u)),
            "min_lower_gap_v_mev": float(np.min(lower_gap_v)),
            "min_upper_gap_v_mev": float(np.min(upper_gap_v)),
            "min_upper_gap_u_path_position": int(np.argmin(upper_gap_u)),
            "min_upper_gap_v_path_position": int(np.argmin(upper_gap_v)),
        }
        for ipos in range(kvec.size):
            row: dict[str, object] = {
                "valley": valley_key,
                "path_position": int(ipos),
                "mesh_index": int(mesh_indices[ipos]),
                "frac_u": float(frac[ipos, 0]),
                "frac_v": float(frac[ipos, 1]),
                "lower_gap_u_mev": float(lower_gap_u[ipos]),
                "upper_gap_u_mev": float(upper_gap_u[ipos]),
                "lower_gap_v_mev": float(lower_gap_v[ipos]),
                "upper_gap_v_mev": float(upper_gap_v[ipos]),
            }
            for band_index in range(lower_neighbor, upper_neighbor + 1):
                row[f"U_band_{band_index}_mev"] = float(eig_u[ipos, band_index])
                row[f"V_band_{band_index}_mev"] = float(eig_v[ipos, band_index])
            rows.append(row)

    tsv_path = output_dir / "active_window_gamma_mprime.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary["output_tsv"] = str(tsv_path)
    summary_path = output_dir / "active_window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
