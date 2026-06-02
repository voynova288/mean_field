#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import rlg_hbn_reference_density


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _sector_index_table(n_spin: int, n_eta: int, n_band: int) -> np.ndarray:
    return np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )


def _panel_from_checkpoint(path: Path) -> str:
    for parent in path.parents:
        if PANEL_RE.match(parent.name):
            return parent.name
    return ""


def _label_from_checkpoint(path: Path) -> str:
    panel = _panel_from_checkpoint(path)
    run = ""
    for parent in path.parents:
        if parent.parent.name == "runs":
            run = parent.name
            break
    if panel and run:
        return f"{panel}/{run}"
    return path.stem


def _config_for_checkpoint(path: Path) -> Path | None:
    for parent in path.parents:
        candidate = parent / "paper_hf_config.json"
        if candidate.exists():
            return candidate
    return None


def _projection_occupations(
    occupation_projector: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> np.ndarray:
    projector = np.asarray(occupation_projector, dtype=np.complex128)
    table = _sector_index_table(n_spin, n_eta, n_band)
    out = np.zeros((int(n_spin), int(n_eta), int(n_band), projector.shape[2]), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            for iband in range(int(n_band)):
                idx = int(table[ispin, ieta, iband])
                out[ispin, ieta, iband, :] = projector[idx, idx, :].real
    return out


def _process_checkpoint(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    archive = np.load(path)
    density_delta = np.asarray(archive["density"], dtype=np.complex128)
    active_band_indices = np.asarray(archive["active_band_indices"], dtype=int).reshape(-1)
    nt, nt_rhs, nk = density_delta.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks in {path}, got {density_delta.shape}")

    n_spin = 2
    n_eta = 2
    if nt % (n_spin * n_eta) != 0:
        raise ValueError(f"Cannot infer n_band from nt={nt}")
    n_band = nt // (n_spin * n_eta)
    if active_band_indices.size != n_band:
        raise ValueError(
            f"Checkpoint active_band_indices has length {active_band_indices.size}, inferred n_band={n_band}"
        )

    config_path = _config_for_checkpoint(path)
    if config_path is not None:
        config = _read_json(config_path)
        scheme = str(config.get("interaction_scheme", config.get("scheme", "average")))
        active_valence = int(config.get("active_valence_bands", n_band // 2))
    else:
        config = {}
        scheme = "average"
        active_valence = n_band // 2

    reference_density = rlg_hbn_reference_density(
        nt,
        nk,
        scheme=scheme,
        active_valence_bands=active_valence,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    projector = density_delta + reference_density
    occupations = _projection_occupations(projector, n_spin=n_spin, n_eta=n_eta, n_band=n_band)

    rows: list[dict[str, object]] = []
    idx_table = _sector_index_table(n_spin, n_eta, n_band)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                vals = occupations[ispin, ieta, iband, :]
                row: dict[str, object] = {
                    "checkpoint_label": _label_from_checkpoint(path),
                    "checkpoint": str(path),
                    "spin": int(ispin),
                    "eta": int(ieta),
                    "active_band": int(iband),
                    "full_band_index": int(active_band_indices[iband]),
                    "basis_index": int(idx_table[ispin, ieta, iband]),
                    **_stats(vals),
                }
                rows.append(row)

    valence_edge_band = max(0, int(active_valence) - 1)
    conduction_edge_band = min(n_band - 1, int(active_valence + (n_band - active_valence) - 1))
    lowest_active_valence_band = 0
    highest_active_conduction_band = n_band - 1

    payload: dict[str, object] = {
        "checkpoint_label": _label_from_checkpoint(path),
        "checkpoint": str(path),
        "config": None if config_path is None else str(config_path),
        "iteration": int(np.asarray(archive["iteration"], dtype=int).reshape(-1)[-1])
        if "iteration" in archive.files
        else None,
        "energy_mev": float(np.asarray(archive["iter_energy_mev"], dtype=float).reshape(-1)[-1])
        if "iter_energy_mev" in archive.files and np.asarray(archive["iter_energy_mev"]).size
        else None,
        "err": float(np.asarray(archive["iter_err"], dtype=float).reshape(-1)[-1])
        if "iter_err" in archive.files and np.asarray(archive["iter_err"]).size
        else None,
        "active_band_indices": [int(value) for value in active_band_indices],
        "scheme": scheme,
        "active_valence_bands": int(active_valence),
        "n_band": int(n_band),
        "n_k": int(nk),
        "occupation_convention": "projection-band diagonal of P=density_delta+reference_density, P_ab=<c_a^dagger c_b>",
        "lowest_active_valence_band_stats": _stats(occupations[:, :, lowest_active_valence_band, :]),
        "valence_edge_band_stats": _stats(occupations[:, :, valence_edge_band, :]),
        "highest_active_conduction_band_stats": _stats(occupations[:, :, highest_active_conduction_band, :]),
        "conduction_edge_band_stats": _stats(occupations[:, :, conduction_edge_band, :]),
        "lowest_active_valence_full_band_index": int(active_band_indices[lowest_active_valence_band]),
        "valence_edge_full_band_index": int(active_band_indices[valence_edge_band]),
        "highest_active_conduction_full_band_index": int(active_band_indices[highest_active_conduction_band]),
        "conduction_edge_full_band_index": int(active_band_indices[conduction_edge_band]),
        "flavor_band_mean_occupations": [
            {
                "spin": int(row["spin"]),
                "eta": int(row["eta"]),
                "active_band": int(row["active_band"]),
                "full_band_index": int(row["full_band_index"]),
                "mean": float(row["mean"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
            }
            for row in rows
        ],
    }
    return payload, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RLG/hBN projected-band occupations from saved SCF checkpoints.")
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="projection_occupations")
    args = parser.parse_args()

    ensure_not_running_compute_on_login_node("RLG/hBN projection-band occupation diagnostic")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for checkpoint in args.checkpoint:
        payload, rows = _process_checkpoint(checkpoint)
        summaries.append(payload)
        all_rows.extend(rows)

    summary_path = output_dir / f"{args.label}_summary.json"
    tsv_path = output_dir / f"{args.label}_by_flavor_band.tsv"
    _write_json(
        summary_path,
        {
            "diagnostic": "RLG/hBN projected active-band occupation check for truncation health",
            "checkpoints": summaries,
            "output_tsv": str(tsv_path),
        },
    )
    if all_rows:
        with tsv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
