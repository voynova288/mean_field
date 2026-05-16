#!/usr/bin/env python3
"""Merge packed HTG Fig. 9b shard artifacts into the standard scan surface."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from pathlib import Path
from typing import Any


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _grid_edges(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [values[0] - 0.5, values[0] + 0.5]
    mids = [0.5 * (left + right) for left, right in zip(values[:-1], values[1:])]
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return [float(first), *[float(value) for value in mids], float(last)]


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (float(row["wAA_mev"]), float(row["theta_deg"]), row.get("case_label", "")))


def _metadata(
    *,
    selected_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    parameters: dict[str, Any],
    shard_jsons: list[dict[str, Any]],
) -> dict[str, Any]:
    theta_values = sorted({float(row["theta_deg"]) for row in selected_rows})
    waa_values = sorted({float(row["wAA_mev"]) for row in selected_rows})
    init_modes = [str(mode) for mode in parameters.get("init_modes", [])]
    seeds = [int(seed) for seed in parameters.get("seeds", [])]
    n_hf_candidates_per_grid = len(init_modes) * len(seeds)
    runtime = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "packed_shards": [
            {
                "runtime": shard.get("runtime", {}),
                "n_selected_rows": len(shard.get("rows", [])),
                "n_run_detail_rows": len(shard.get("run_details", [])),
            }
            for shard in shard_jsons
        ],
    }
    return {
        "figure": "Kwan Fig. 9(b)",
        "reproduction_mode": "paper-level exhaustive seed mode" if n_hf_candidates_per_grid >= 300 else "qualitative reproduction mode",
        "nu": float(parameters.get("nu", 3.0)),
        "theta_grid_deg": theta_values,
        "wAA_grid_meV": waa_values,
        "theta_values_deg": theta_values,
        "wAA_values_meV": waa_values,
        "theta_center_values_deg": theta_values,
        "wAA_center_values_meV": waa_values,
        "theta_edge_values_deg": _grid_edges(theta_values),
        "wAA_edge_values_meV": _grid_edges(waa_values),
        "n_theta": len(theta_values),
        "n_wAA": len(waa_values),
        "n_parameter_points": len(selected_rows),
        "wcond_array_shape": [len(waa_values), len(theta_values)],
        "wcond_array_axis_order": ["wAA_grid_meV rows ascending", "theta_grid_deg columns ascending"],
        "wcond_array_rows": "wAA_grid_meV",
        "wcond_array_columns": "theta_grid_deg",
        "mesh_note": (
            "Calculation points are the explicit center arrays theta_grid_deg and wAA_grid_meV. "
            "Major tick labels are plotting ticks only and cell edges are derived from adjacent centers."
        ),
        "calculation_points_are_cell_centers": True,
        "cell_edges_are_derived_for_plotting_only": True,
        "cell_edges_are_calculation_points": False,
        "parameter_grid_formula": "n_parameter_points = n_theta * n_wAA for this rectangular scan",
        "hf_initial_samples_per_grid_point": {
            "init_modes": init_modes,
            "normalized_init_modes": shard_jsons[0].get("grid_metadata", {})
            .get("hf_initial_samples_per_grid_point", {})
            .get("normalized_init_modes", init_modes),
            "init_mode_aliases": {"d3a": "fb", "d3b": "sublattice"},
            "n_init_modes": len(init_modes),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_hf_candidates_per_grid": n_hf_candidates_per_grid,
            "candidate_count_formula": "n_hf_candidates_per_grid = n_init_modes * n_seeds",
        },
        "n_total_hf_runs": len(selected_rows) * n_hf_candidates_per_grid,
        "observed_run_detail_rows": len(detail_rows),
        "total_hf_run_formula": "n_total_hf_runs = n_parameter_points * n_hf_candidates_per_grid",
        "paper_seed_note": (
            "Kwan et al. report >300 initial seeds of different types per parameter in their HF phase-diagram "
            "search. Runs with fewer candidates should be described as qualitative reproduction mode, not an "
            "independent exhaustive global-minimum search."
        ),
        "translation_symmetry": "primitive-cell translation-invariant HF scan; no doubled/tripled TSB sectors",
        "initial_state_scope": (
            "strong-coupling, BM, perturbed, and random primitive-cell seeds only; not the full paper TSB search space"
        ),
        "kwan_parameters": {
            "vF_m_per_s": float(parameters.get("fermi_velocity_m_per_s", 8.8e5)),
            "wAB_meV": 1000.0 * float(parameters.get("w_ev", 0.11)),
            "wAA_meV": "scanned",
            "epsilon_r": float(parameters.get("epsilon_r", 8.0)),
            "d_sc_nm": float(parameters.get("d_sc_nm", 25.0)),
            "U_ev": float(parameters.get("u_ev", parameters.get("U_ev", 0.0))),
            "interaction_scheme": "average",
            "pauli_twist": False,
            "system_size_for_phase_map": f"{int(parameters.get('n_k', 12))}x{int(parameters.get('n_k', 12))}",
            "fig9a_validation_size": "18x18",
            "drop_q0_coulomb": bool(parameters.get("drop_q0_coulomb", True)),
        },
        "observable": {
            "name": "Wcond",
            "definition": "max_k E_lowest_unoccupied(k) - min_k E_lowest_unoccupied(k)",
            "units": "meV",
            "computed_over": "full 2D self-consistent HF mBZ grid",
            "not_computed_from": "high-symmetry path bands or path_band_gap_ev",
            "implementation_note": parameters.get("bandwidth_definition", ""),
        },
        "runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="fig9b_conduction_bandwidth_scan_8x10_paper_level")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    selected_fields: list[str] | None = None
    detail_fields: list[str] | None = None
    shard_jsons: list[dict[str, Any]] = []
    for shard_dir in args.shard_dirs:
        selected_path = shard_dir / f"{args.prefix}.tsv"
        detail_path = shard_dir / f"{args.prefix}_run_details.tsv"
        json_path = shard_dir / f"{args.prefix}.json"
        fields, rows = _read_tsv(selected_path)
        detail_fieldnames, details = _read_tsv(detail_path)
        if selected_fields is None:
            selected_fields = fields
        elif selected_fields != fields:
            raise SystemExit(f"selected TSV fields differ in {selected_path}")
        if detail_fields is None:
            detail_fields = detail_fieldnames
        elif detail_fields != detail_fieldnames:
            raise SystemExit(f"detail TSV fields differ in {detail_path}")
        selected_rows.extend(rows)
        detail_rows.extend(details)
        shard_jsons.append(_read_json(json_path))

    if selected_fields is None or detail_fields is None or not shard_jsons:
        raise SystemExit("no shard artifacts found")
    selected_rows = _sort_rows(selected_rows)
    detail_rows = _sort_rows(detail_rows)
    parameters = dict(shard_jsons[0]["parameters"])
    parameters["cases"] = [
        {"theta_deg": float(row["theta_deg"]), "wAA_mev": float(row["wAA_mev"]), "label": row["case_label"]}
        for row in selected_rows
    ]
    metadata = _metadata(
        selected_rows=selected_rows,
        detail_rows=detail_rows,
        parameters=parameters,
        shard_jsons=shard_jsons,
    )

    selected_out = args.output_dir / f"{args.prefix}.tsv"
    detail_out = args.output_dir / f"{args.prefix}_run_details.tsv"
    json_out = args.output_dir / f"{args.prefix}.json"
    metadata_out = args.output_dir / "grid_metadata.json"
    _write_tsv(selected_out, selected_fields, selected_rows)
    _write_tsv(detail_out, detail_fields, detail_rows)
    _write_json(metadata_out, metadata)
    _write_json(
        json_out,
        {
            "artifacts": {
                "tsv": str(selected_out),
                "run_details_tsv": str(detail_out),
                "grid_metadata": str(metadata_out),
            },
            "grid_metadata": metadata,
            "parameters": parameters,
            "runtime": metadata["runtime"],
            "rows": selected_rows,
            "run_details": detail_rows,
        },
    )
    print(f"merged_selected_rows={len(selected_rows)}")
    print(f"merged_run_detail_rows={len(detail_rows)}")
    print(f"selected_tsv={selected_out}")
    print(f"run_details_tsv={detail_out}")
    print(f"grid_metadata={metadata_out}")


if __name__ == "__main__":
    main()
