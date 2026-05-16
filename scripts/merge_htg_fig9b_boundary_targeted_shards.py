#!/usr/bin/env python3
"""Merge targeted HTG Fig. 9b D3A/D3B boundary diagnostic shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _float_or_inf(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("inf")


def _sort_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (_float_or_inf(row, "wAA_meV"), _float_or_inf(row, "theta_deg")))


def _sort_details(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _float_or_inf(row, "wAA_meV"),
            _float_or_inf(row, "theta_deg"),
            row.get("requested_class", ""),
            _float_or_inf(row, "partial_flavor_index"),
            _float_or_inf(row, "seed_index"),
        ),
    )


def _metadata(summary_rows: list[dict[str, str]], detail_rows: list[dict[str, str]], shard_jsons: list[dict[str, Any]]) -> dict[str, Any]:
    if not shard_jsons:
        raise SystemExit("no shard metadata found")
    first = dict(shard_jsons[0].get("grid_metadata", {}))
    theta_values = sorted({float(row["theta_deg"]) for row in summary_rows})
    waa_values = sorted({float(row["wAA_meV"]) for row in summary_rows})
    first["theta_values_deg"] = theta_values
    first["wAA_values_meV"] = waa_values
    first["n_parameter_points"] = len(summary_rows)
    first["observed_run_detail_rows"] = len(detail_rows)
    first["runtime"] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "packed_shards": [
            {
                "runtime": shard.get("grid_metadata", {}).get("runtime", {}),
                "n_summary_rows": len(shard.get("rows", [])),
                "n_run_detail_rows": len(shard.get("run_details", [])),
            }
            for shard in shard_jsons
        ],
    }
    if "seed_protocol" in first:
        seed_protocol = dict(first["seed_protocol"])
        first["n_total_hf_runs"] = len(summary_rows) * int(seed_protocol.get("hf_candidates_per_parameter", 0))
    return first


def _value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return value if value != "" else ""


def _write_report(path: Path, summary_rows: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    lines = [
        "# HTG Fig. 9b Targeted D3A/D3B Boundary Diagnostic",
        "",
        "This merged diagnostic keeps the accepted global 8x10 Fig. 9b mesh unchanged and probes only the requested top-left 4x4 window.",
        "",
        "## Protocol",
        "",
        f"- HF mesh: {metadata.get('kwan_parameters', {}).get('system_size_for_phase_map', '')}",
        f"- epsilon_r: {metadata.get('kwan_parameters', {}).get('epsilon_r', '')}",
        f"- d_sc_nm: {metadata.get('kwan_parameters', {}).get('d_sc_nm', '')}",
        f"- wAB_meV: {metadata.get('kwan_parameters', {}).get('wAB_meV', '')}",
        f"- vF_m_per_s: {metadata.get('kwan_parameters', {}).get('vF_m_per_s', '')}",
        f"- U_ev: {metadata.get('kwan_parameters', {}).get('U_ev', '')}",
        f"- candidates per parameter: {metadata.get('seed_protocol', {}).get('hf_candidates_per_parameter', '')}",
        f"- ambiguity threshold: {metadata.get('boundary_policy', {}).get('ambiguity_threshold_meV', '')} meV per moire cell",
        "",
        "## Summary",
        "",
        "| theta | wAA | E_D3A eV | E_D3B eV | DeltaE meV | chosen | Wcond D3A | Wcond D3B | gap D3A | gap D3B | residual |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _sort_summary(summary_rows):
        lines.append(
            "| {theta} | {waa} | {ea} | {eb} | {de} | {chosen} | {wa} | {wb} | {ga} | {gb} | {res} |".format(
                theta=_value(row, "theta_deg"),
                waa=_value(row, "wAA_meV"),
                ea=_value(row, "E_D3A_min"),
                eb=_value(row, "E_D3B_min"),
                de=_value(row, "DeltaE"),
                chosen=_value(row, "chosen_class"),
                wa=_value(row, "Wcond_D3A"),
                wb=_value(row, "Wcond_D3B"),
                ga=_value(row, "HF_gap_D3A"),
                gb=_value(row, "HF_gap_D3B"),
                res=_value(row, "convergence_residual"),
            )
        )
    lines.extend(
        [
            "",
            "Cells marked `AMBIGUOUS_*` are not manual relabels; they mean the D3A/D3B energy splitting is below the configured threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="fig9b_d3_boundary_targeted")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    summary_fields: list[str] | None = None
    detail_fields: list[str] | None = None
    shard_jsons: list[dict[str, Any]] = []
    for shard_dir in args.shard_dirs:
        summary_path = shard_dir / f"{args.prefix}.csv"
        detail_path = shard_dir / f"{args.prefix}_run_details.csv"
        json_path = shard_dir / f"{args.prefix}.json"
        fields, rows = _read_csv(summary_path)
        detail_fieldnames, details = _read_csv(detail_path)
        if summary_fields is None:
            summary_fields = fields
        elif summary_fields != fields:
            raise SystemExit(f"summary CSV fields differ in {summary_path}")
        if detail_fields is None:
            detail_fields = detail_fieldnames
        elif detail_fields != detail_fieldnames:
            raise SystemExit(f"detail CSV fields differ in {detail_path}")
        summary_rows.extend(rows)
        detail_rows.extend(details)
        shard_jsons.append(_read_json(json_path))

    if summary_fields is None or detail_fields is None:
        raise SystemExit("no shard CSV artifacts found")
    summary_rows = _sort_summary(summary_rows)
    detail_rows = _sort_details(detail_rows)
    metadata = _metadata(summary_rows, detail_rows, shard_jsons)

    summary_out = args.output_dir / f"{args.prefix}.csv"
    detail_out = args.output_dir / f"{args.prefix}_run_details.csv"
    metadata_out = args.output_dir / "grid_metadata.json"
    json_out = args.output_dir / f"{args.prefix}.json"
    report_out = args.output_dir / f"{args.prefix}_report.md"
    _write_csv(summary_out, summary_fields, summary_rows)
    _write_csv(detail_out, detail_fields, detail_rows)
    _write_json(metadata_out, metadata)
    _write_report(report_out, summary_rows, metadata)
    _write_json(
        json_out,
        {
            "artifacts": {
                "csv": str(summary_out),
                "run_details_csv": str(detail_out),
                "grid_metadata": str(metadata_out),
                "report": str(report_out),
            },
            "grid_metadata": metadata,
            "rows": summary_rows,
            "run_details": detail_rows,
        },
    )
    print(f"merged_summary_rows={len(summary_rows)}")
    print(f"merged_run_detail_rows={len(detail_rows)}")
    print(f"summary_csv={summary_out}")
    print(f"run_details_csv={detail_out}")
    print(f"grid_metadata={metadata_out}")


if __name__ == "__main__":
    main()
