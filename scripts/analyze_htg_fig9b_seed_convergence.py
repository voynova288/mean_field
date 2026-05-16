#!/usr/bin/env python3
"""Analyze Fig. 9b seed-budget convergence from saved multi-init run details."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_THETA_GRID_DEG = [1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95]
EXPECTED_WAA_GRID_MEV = [40.0, 47.5, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
ENERGY_TOLERANCE_EV = 1.0e-10
WCOND_STABLE_TOLERANCE_MEV = 0.25


@dataclass(frozen=True)
class Stage:
    label: str
    init_modes: tuple[str, ...] | None
    max_seed: int | None
    target_candidates: int | None


STAGES = (
    Stage("4", ("d3b", "d3a", "bm", "fi"), 1, 4),
    Stage("16", ("d3b", "d3a", "bm", "fi", "flavor", "vp", "sp", "chern"), 2, 16),
    Stage("64", ("d3b", "d3a", "bm", "fi", "flavor", "vp", "sp", "chern"), 8, 64),
    Stage("300", ("d3b", "d3a", "bm", "fi", "flavor", "vp", "sp", "chern", "perturbed", "random"), 30, 300),
    Stage("full", None, None, None),
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _same_grid(left: list[float], right: list[float], tol: float = 1.0e-9) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _float_or_none(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


def _compact_class(row: dict[str, str]) -> str:
    value = row.get("class_compact_label", "")
    if value:
        return value
    return row.get("class_label", "").strip().strip("[]").replace(" ", "")


def _requested_mode(row: dict[str, str]) -> str:
    return row.get("requested_init_mode") or row.get("init_mode", "")


def _stage_rows(rows: list[dict[str, str]], stage: Stage) -> list[dict[str, str]]:
    if stage.init_modes is None:
        return rows
    allowed_modes = set(stage.init_modes)
    return [
        row
        for row in rows
        if _requested_mode(row) in allowed_modes and int(row["seed"]) <= int(stage.max_seed or 0)
    ]


def _select_best(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("cannot select from an empty candidate set")
    converged = [row for row in rows if _bool(row, "converged")]
    pool = converged or rows
    min_energy = min(_float(row, "final_energy_ev") for row in pool)
    degenerate = [row for row in pool if abs(_float(row, "final_energy_ev") - min_energy) <= ENERGY_TOLERANCE_EV]
    with_wcond = [row for row in degenerate if row.get("wcond_mev", "") != ""]
    if with_wcond:
        return min(with_wcond, key=lambda row: _float(row, "final_energy_ev"))
    return min(degenerate, key=lambda row: _float(row, "final_energy_ev"))


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def _compare_stage(
    left: dict[str, dict[str, object]],
    right: dict[str, dict[str, object]],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, object]:
    shared = sorted(set(left) & set(right))
    wcond_deltas = []
    class_changes = 0
    selected_changes = 0
    for case in shared:
        left_row = left[case]
        right_row = right[case]
        if left_row["class_compact_label"] != right_row["class_compact_label"]:
            class_changes += 1
        if (left_row["requested_init_mode"], left_row["seed"]) != (right_row["requested_init_mode"], right_row["seed"]):
            selected_changes += 1
        left_wcond = left_row.get("wcond_mev")
        right_wcond = right_row.get("wcond_mev")
        if isinstance(left_wcond, float) and isinstance(right_wcond, float):
            wcond_deltas.append(abs(right_wcond - left_wcond))
    max_delta = max(wcond_deltas) if wcond_deltas else None
    changed_wcond = sum(delta > WCOND_STABLE_TOLERANCE_MEV for delta in wcond_deltas)
    return {
        "from_stage": left_label,
        "to_stage": right_label,
        "shared_points": len(shared),
        "class_or_boundary_changes": class_changes,
        "selected_basin_changes": selected_changes,
        "wcond_pairs": len(wcond_deltas),
        "max_abs_delta_wcond_mev": max_delta,
        f"wcond_delta_gt_{WCOND_STABLE_TOLERANCE_MEV:g}mev": changed_wcond,
    }


def _make_report(
    path: Path,
    *,
    selected_tsv: Path,
    details_tsv: Path,
    theta_values: list[float],
    waa_values: list[float],
    stage_rows: list[dict[str, object]],
    comparisons: list[dict[str, object]],
) -> None:
    final_comparison = comparisons[-1] if comparisons else {}
    final_stable = (
        final_comparison.get("class_or_boundary_changes") == 0
        and final_comparison.get(f"wcond_delta_gt_{WCOND_STABLE_TOLERANCE_MEV:g}mev") == 0
    )
    lines = [
        "# HTG Fig. 9b Seed-Convergence Check",
        "",
        "## Source",
        "",
        f"- Selected TSV: `{selected_tsv}`",
        f"- Run details TSV: `{details_tsv}`",
        f"- theta_grid_deg: `{theta_values}`",
        f"- wAA_grid_meV: `{waa_values}`",
        f"- Wcond array shape: `({len(waa_values)}, {len(theta_values)})`, rows=wAA_grid_meV, columns=theta_grid_deg",
        "",
        "## Stage Definitions",
        "",
        "The stages are reconstructed from one comprehensive run by filtering candidate initial states; they are not separate HF reruns.",
        "",
        "| Stage | Init modes | Max seed | Target candidates/grid |",
        "| --- | --- | ---: | ---: |",
    ]
    for stage in STAGES:
        lines.append(
            "| {label} | {modes} | {max_seed} | {target} |".format(
                label=stage.label,
                modes="all available" if stage.init_modes is None else ",".join(stage.init_modes),
                max_seed="" if stage.max_seed is None else stage.max_seed,
                target="" if stage.target_candidates is None else stage.target_candidates,
            )
        )
    lines.extend(
        [
            "",
            "## Adjacent-Stage Changes",
            "",
            "| From | To | Shared points | Class/boundary changes | Selected-basin changes | Max abs delta Wcond meV | Wcond changes > 0.25 meV |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {from_stage} | {to_stage} | {shared_points} | {class_or_boundary_changes} | "
            "{selected_basin_changes} | {max_delta} | {changed_wcond} |".format(
                from_stage=row["from_stage"],
                to_stage=row["to_stage"],
                shared_points=row["shared_points"],
                class_or_boundary_changes=row["class_or_boundary_changes"],
                selected_basin_changes=row["selected_basin_changes"],
                max_delta=_fmt(row.get("max_abs_delta_wcond_mev")),
                changed_wcond=row[f"wcond_delta_gt_{WCOND_STABLE_TOLERANCE_MEV:g}mev"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if final_stable:
        lines.append(
            "- The largest tested stage transition shows no D3A/D3B class changes and no Wcond changes above 0.25 meV."
        )
    else:
        lines.append(
            "- The largest tested stage transition still changes either the selected class/boundary or Wcond above 0.25 meV; do not claim seed convergence."
        )
    lines.extend(
        [
            "- This report only addresses seed-budget stability on the corrected 8x10 mesh. It does not digitize the paper color map.",
            "",
            "## Per-Point Stage Selections",
            "",
            f"- TSV: `{path.with_suffix('.tsv')}`",
            f"- Rows: `{len(stage_rows)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected_tsv", type=Path)
    parser.add_argument("details_tsv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="fig9b_seed_convergence")
    args = parser.parse_args()

    selected_rows = _read_tsv(args.selected_tsv)
    detail_rows = _read_tsv(args.details_tsv)
    output_dir = (args.output_dir or args.selected_tsv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    theta_values = sorted({float(row["theta_deg"]) for row in selected_rows})
    waa_values = sorted({float(row["wAA_mev"]) for row in selected_rows})
    if not _same_grid(theta_values, EXPECTED_THETA_GRID_DEG) or not _same_grid(waa_values, EXPECTED_WAA_GRID_MEV):
        raise SystemExit(f"not the corrected 8x10 Fig. 9b grid: theta={theta_values}, wAA={waa_values}")

    detail_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in detail_rows:
        detail_by_case[row["case_label"]].append(row)

    stage_selection_rows: list[dict[str, object]] = []
    selections_by_stage: dict[str, dict[str, dict[str, object]]] = {}
    for stage in STAGES:
        stage_selection: dict[str, dict[str, object]] = {}
        for selected in selected_rows:
            case_label = selected["case_label"]
            candidates = _stage_rows(detail_by_case[case_label], stage)
            if not candidates:
                raise SystemExit(f"stage {stage.label} has no candidates for {case_label}")
            best = _select_best(candidates)
            payload = {
                "stage": stage.label,
                "case_label": case_label,
                "theta_deg": float(best["theta_deg"]),
                "wAA_mev": float(best["wAA_mev"]),
                "target_candidates_per_grid": "" if stage.target_candidates is None else stage.target_candidates,
                "observed_candidates_per_grid": len(candidates),
                "requested_init_mode": _requested_mode(best),
                "normalized_init_mode": best.get("init_mode", ""),
                "seed": int(best["seed"]),
                "converged": best.get("converged", ""),
                "final_energy_ev": _float(best, "final_energy_ev"),
                "final_error": _float_or_none(best, "final_error"),
                "class_label": best.get("class_label", ""),
                "class_compact_label": _compact_class(best),
                "family": best.get("family", ""),
                "wcond_mev": _float_or_none(best, "wcond_mev"),
                "hf_gap_mev": _float_or_none(best, "hf_gap_mev"),
            }
            stage_selection[case_label] = payload
            stage_selection_rows.append(payload)
        selections_by_stage[stage.label] = stage_selection

    comparisons = [
        _compare_stage(
            selections_by_stage[left.label],
            selections_by_stage[right.label],
            left_label=left.label,
            right_label=right.label,
        )
        for left, right in zip(STAGES[:-1], STAGES[1:])
    ]

    stage_tsv = output_dir / f"{args.prefix}.tsv"
    summary_tsv = output_dir / f"{args.prefix}_summary.tsv"
    report_path = output_dir / f"{args.prefix}.md"
    json_path = output_dir / f"{args.prefix}.json"
    _write_tsv(
        stage_tsv,
        stage_selection_rows,
        (
            "stage",
            "case_label",
            "theta_deg",
            "wAA_mev",
            "target_candidates_per_grid",
            "observed_candidates_per_grid",
            "requested_init_mode",
            "normalized_init_mode",
            "seed",
            "converged",
            "final_energy_ev",
            "final_error",
            "class_label",
            "class_compact_label",
            "family",
            "wcond_mev",
            "hf_gap_mev",
        ),
    )
    _write_tsv(
        summary_tsv,
        comparisons,
        (
            "from_stage",
            "to_stage",
            "shared_points",
            "class_or_boundary_changes",
            "selected_basin_changes",
            "wcond_pairs",
            "max_abs_delta_wcond_mev",
            f"wcond_delta_gt_{WCOND_STABLE_TOLERANCE_MEV:g}mev",
        ),
    )
    _write_json(
        json_path,
        {
            "theta_grid_deg": theta_values,
            "wAA_grid_meV": waa_values,
            "wcond_array_shape": [len(waa_values), len(theta_values)],
            "stages": [stage.__dict__ for stage in STAGES],
            "comparisons": comparisons,
            "artifacts": {
                "stage_tsv": str(stage_tsv),
                "summary_tsv": str(summary_tsv),
                "report": str(report_path),
            },
        },
    )
    _make_report(
        report_path,
        selected_tsv=args.selected_tsv,
        details_tsv=args.details_tsv,
        theta_values=theta_values,
        waa_values=waa_values,
        stage_rows=stage_selection_rows,
        comparisons=comparisons,
    )
    print(f"wrote {stage_tsv}")
    print(f"wrote {summary_tsv}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
