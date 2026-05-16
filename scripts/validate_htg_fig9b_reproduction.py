#!/usr/bin/env python3
"""Validate the local HTG Fig. 9b reproduction artifacts.

This script is intentionally read-only with respect to HF numerics: it audits the
saved multi-init scan and companion 18x18 band-structure run, then writes a
compact report and an anchor-point TSV next to the scan artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SCAN_DIR = Path("results/HTG/htg_fig9b_bandwidth_scan_8x10_paper_level_20260509_001")
DEFAULT_FIG9A_DIR = Path("results/HTG/htg_fig9_fig8b_d3b_theta180_w75_w110_nk18_g2_q0drop_20260503_001")
DEFAULT_EXACT_ANCHOR_DIR = Path("results/HTG/htg_fig9b_exact_anchor_scan_20260508_001")
DEFAULT_SCAN_PREFIX = "fig9b_conduction_bandwidth_scan_8x10_paper_level"
EXPECTED_THETA_GRID_DEG = [1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95]
EXPECTED_WAA_GRID_MEV = [40.0, 47.5, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
EXPECTED_WCOND_SHAPE = [len(EXPECTED_WAA_GRID_MEV), len(EXPECTED_THETA_GRID_DEG)]
PAPER_VISUAL_D3A_CELLS = {
    (1.60, 75.0),
    (1.60, 80.0),
    (1.65, 80.0),
    (1.60, 85.0),
    (1.65, 85.0),
    (1.70, 85.0),
    (1.60, 90.0),
    (1.65, 90.0),
    (1.70, 90.0),
}

ANCHORS = (
    {
        "anchor_name": "paper_point_bracket",
        "theta_deg": 1.80,
        "wAA_mev": 75.0,
        "expected_region": "FB / D3B-like low-bandwidth region; exact 75 meV is an 8x10 grid center",
    },
    {
        "anchor_name": "upper_left_d3a_bracket",
        "theta_deg": 1.60,
        "wAA_mev": 85.0,
        "expected_region": "upper-left D3A high-bandwidth pocket; exact 85 meV is an 8x10 grid center",
    },
    {
        "anchor_name": "lower_middle_d3b",
        "theta_deg": 1.80,
        "wAA_mev": 50.0,
        "expected_region": "lower-middle D3B / low-bandwidth point",
    },
    {
        "anchor_name": "lower_right_edge",
        "theta_deg": 1.95,
        "wAA_mev": 40.0,
        "expected_region": "lower-right edge with larger Wcond than central minimum",
    },
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _same_float(left: float, right: float, tol: float = 1.0e-9) -> bool:
    return abs(left - right) <= tol


def _same_float_list(left: list[float], right: list[float], tol: float = 1.0e-9) -> bool:
    return len(left) == len(right) and all(_same_float(a, b, tol=tol) for a, b in zip(left, right))


def _grid_edges(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [values[0] - 0.5, values[0] + 0.5]
    mids = [0.5 * (left + right) for left, right in zip(values[:-1], values[1:])]
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return [float(first), *[float(value) for value in mids], float(last)]


def _row_for(rows: list[dict[str, str]], theta: float, waa: float) -> dict[str, str]:
    for row in rows:
        if _same_float(float(row["theta_deg"]), theta) and _same_float(float(row["wAA_mev"]), waa):
            return row
    raise KeyError(f"no scan row for theta={theta}, wAA={waa}")


def _anchor_matches(rows: list[dict[str, str]], theta: float, waa: float) -> list[tuple[str, dict[str, str]]]:
    theta_values = sorted({float(row["theta_deg"]) for row in rows})
    matched_theta = min(theta_values, key=lambda value: abs(value - theta))
    theta_match = "exact_theta" if _same_float(matched_theta, theta) else "nearest_theta"

    waa_values = sorted({float(row["wAA_mev"]) for row in rows if _same_float(float(row["theta_deg"]), matched_theta)})
    exact_waa = [value for value in waa_values if _same_float(value, waa)]
    if exact_waa:
        return [(f"exact/{theta_match}", _row_for(rows, matched_theta, exact_waa[0]))]

    lower = max((value for value in waa_values if value < waa), default=None)
    upper = min((value for value in waa_values if value > waa), default=None)
    if lower is not None and upper is not None:
        return [
            (f"bracket_lower/{theta_match}", _row_for(rows, matched_theta, lower)),
            (f"bracket_upper/{theta_match}", _row_for(rows, matched_theta, upper)),
        ]

    nearest_waa = min(waa_values, key=lambda value: abs(value - waa))
    return [(f"nearest_wAA/{theta_match}", _row_for(rows, matched_theta, nearest_waa))]


def _write_anchor_tsv(path: Path, scan_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    has_final_error = "final_error" in scan_rows[0] if scan_rows else False
    for anchor in ANCHORS:
        for match_type, row in _anchor_matches(scan_rows, float(anchor["theta_deg"]), float(anchor["wAA_mev"])):
            output_rows.append(
                {
                    "anchor_name": anchor["anchor_name"],
                    "requested_theta_deg": _fmt(anchor["theta_deg"]),
                    "requested_wAA_mev": _fmt(anchor["wAA_mev"]),
                    "matched_theta_deg": row["theta_deg"],
                    "matched_wAA_mev": row["wAA_mev"],
                    "match_type": match_type,
                    "expected_region": anchor["expected_region"],
                    "best_init_mode": row.get("best_init_mode", ""),
                    "best_seed": row.get("best_seed", ""),
                    "class_label": row.get("class_label", ""),
                    "family_label": row.get("family", ""),
                    "hf_gap_mev": row.get("hf_gap_mev", ""),
                    "Wcond_mev": row.get("wcond_mev", ""),
                    "iterations": row.get("iterations", ""),
                    "final_error": row.get("final_error", ""),
                    "final_error_note": "" if has_final_error else "not_recorded_in_scan_tsv",
                    "converged": row.get("converged", ""),
                    "exit_reason": row.get("exit_reason", ""),
                    "selected_from_converged_pool": row.get("selected_from_converged_pool", ""),
                    "scan_run_count": row.get("scan_run_count", ""),
                    "converged_run_count": row.get("converged_run_count", ""),
                    "selected_with_wcond_tiebreak": row.get("selected_with_wcond_tiebreak", ""),
                }
            )

    fieldnames = list(output_rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def _status(ok: bool, *, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def _comparison_summary(comparison_path: Path) -> dict[str, str]:
    if not comparison_path.exists():
        return {}
    rows = _read_tsv(comparison_path)
    if not rows:
        return {}
    delta_key = "delta_wcond_mev" if "delta_wcond_mev" in rows[0] else "delta_mev"
    deltas = [float(row[delta_key]) for row in rows if row.get(delta_key, "") != ""]
    tolerance_mev = 1.0e-6
    return {
        "shared_points": str(len(rows)),
        "decreased": str(sum(delta < -tolerance_mev for delta in deltas)),
        "unchanged": str(sum(abs(delta) <= tolerance_mev for delta in deltas)),
        "increased": str(sum(delta > tolerance_mev for delta in deltas)),
    }


def _read_exact_anchor_rows(exact_anchor_dir: Path | None) -> list[dict[str, str]]:
    if exact_anchor_dir is None:
        return []
    path = exact_anchor_dir / "fig9b_exact_anchor_scan.tsv"
    if not path.exists():
        return []
    return _read_tsv(path)


def _mode_label(n_hf_candidates_per_grid: int) -> str:
    if int(n_hf_candidates_per_grid) >= 300:
        return "paper-level exhaustive seed mode"
    return "qualitative reproduction mode"


def _grid_metadata(
    *,
    scan_json: dict[str, Any],
    scan_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
) -> dict[str, Any]:
    params = scan_json["parameters"]
    runtime = scan_json.get("runtime", {})
    theta_values = sorted({float(row["theta_deg"]) for row in scan_rows})
    waa_values = sorted({float(row["wAA_mev"]) for row in scan_rows})
    theta_edges = _grid_edges(theta_values)
    waa_edges = _grid_edges(waa_values)
    init_modes = [str(mode) for mode in params.get("init_modes", [])]
    seeds = [int(seed) for seed in params.get("seeds", [])]
    n_hf_candidates_per_grid = len(init_modes) * len(seeds)
    n_parameter_points = len(scan_rows)
    metadata = scan_json.get("grid_metadata", {})
    wcond_array_shape = [len(waa_values), len(theta_values)]
    return {
        "figure": "Kwan Fig. 9(b)",
        "reproduction_mode": metadata.get("reproduction_mode", _mode_label(n_hf_candidates_per_grid)),
        "nu": float(params.get("nu", 3.0)),
        "theta_grid_deg": theta_values,
        "wAA_grid_meV": waa_values,
        "theta_values_deg": theta_values,
        "wAA_values_meV": waa_values,
        "theta_center_values_deg": theta_values,
        "wAA_center_values_meV": waa_values,
        "theta_edge_values_deg": theta_edges,
        "wAA_edge_values_meV": waa_edges,
        "n_theta": len(theta_values),
        "n_wAA": len(waa_values),
        "n_parameter_points": n_parameter_points,
        "wcond_array_shape": wcond_array_shape,
        "wcond_array_axis_order": ["wAA_grid_meV rows ascending", "theta_grid_deg columns ascending"],
        "wcond_array_rows": "wAA_grid_meV",
        "wcond_array_columns": "theta_grid_deg",
        "mesh_note": (
            "Calculation points are the explicit 8x10 center arrays theta_grid_deg and wAA_grid_meV. "
            "Major tick labels are plotting ticks only and cell edges are derived from adjacent centers."
        ),
        "calculation_points_are_cell_centers": True,
        "cell_edges_are_derived_for_plotting_only": True,
        "cell_edges_are_calculation_points": False,
        "parameter_grid_formula": "n_parameter_points = n_theta * n_wAA for this rectangular scan",
        "hf_initial_samples_per_grid_point": {
            "init_modes": init_modes,
            "normalized_init_modes": metadata.get("hf_initial_samples_per_grid_point", {}).get(
                "normalized_init_modes", init_modes
            ),
            "init_mode_aliases": {"d3a": "fb", "d3b": "sublattice"},
            "n_init_modes": len(init_modes),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_hf_candidates_per_grid": n_hf_candidates_per_grid,
            "candidate_count_formula": "n_hf_candidates_per_grid = n_init_modes * n_seeds",
        },
        "n_total_hf_runs": n_parameter_points * n_hf_candidates_per_grid,
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
            "vF_m_per_s": float(params.get("fermi_velocity_m_per_s", 8.8e5)),
            "wAB_meV": 1000.0 * float(params.get("w_ev", 0.11)),
            "wAA_meV": "scanned",
            "epsilon_r": float(params.get("epsilon_r", 8.0)),
            "d_sc_nm": float(params.get("d_sc_nm", 25.0)),
            "U_ev": float(params.get("u_ev", params.get("U_ev", 0.0))),
            "interaction_scheme": "average",
            "pauli_twist": False,
            "system_size_for_phase_map": f"{int(params.get('n_k', 12))}x{int(params.get('n_k', 12))}",
            "fig9a_validation_size": "18x18",
            "drop_q0_coulomb": bool(params.get("drop_q0_coulomb", True)),
        },
        "observable": {
            "name": "Wcond",
            "definition": "max_k E_lowest_unoccupied(k) - min_k E_lowest_unoccupied(k)",
            "units": "meV",
            "computed_over": "full 2D self-consistent HF mBZ grid",
            "not_computed_from": "high-symmetry path bands or path_band_gap_ev",
            "implementation_note": params.get("bandwidth_definition", ""),
        },
        "runtime": runtime,
    }


def _make_report(
    path: Path,
    *,
    scan_dir: Path,
    scan_prefix: str,
    fig9a_dir: Path,
    exact_anchor_dir: Path | None,
    scan_json: dict[str, Any],
    scan_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]],
    exact_anchor_rows: list[dict[str, str]],
    metadata: dict[str, Any],
) -> None:
    params = scan_json["parameters"]
    runtime = scan_json.get("runtime", {})
    comparison = _comparison_summary(scan_dir / "fig9b_conduction_bandwidth_scan_comparison.tsv")

    theta_values = sorted({float(row["theta_deg"]) for row in scan_rows})
    waa_values = sorted({float(row["wAA_mev"]) for row in scan_rows})
    expected_grid = _same_float_list(theta_values, EXPECTED_THETA_GRID_DEG) and _same_float_list(
        waa_values, EXPECTED_WAA_GRID_MEV
    )
    expected_shape = metadata.get("wcond_array_shape") == EXPECTED_WCOND_SHAPE
    wconds = [float(row["wcond_mev"]) for row in scan_rows if row.get("wcond_mev", "") != ""]
    class_counter = Counter(row["class_label"] for row in scan_rows)
    family_counter = Counter(row["family"] for row in scan_rows)
    detail_exit_counter = Counter(row["exit_reason"] for row in detail_rows)
    selected_converged = sum(_bool(row, "converged") for row in scan_rows)
    selected_from_converged = sum(_bool(row, "selected_from_converged_pool") for row in scan_rows)
    detail_converged = sum(_bool(row, "converged") for row in detail_rows)
    has_selected_final_error = bool(scan_rows) and "final_error" in scan_rows[0]
    samples = metadata["hf_initial_samples_per_grid_point"]
    selected_final_errors = [
        float(row["final_error"])
        for row in scan_rows
        if row.get("final_error", "") != ""
    ]
    d3a_rows = [row for row in scan_rows if row["class_label"] == "[D3 A]"]
    d3a_upper_left = all(float(row["theta_deg"]) <= 1.70 and float(row["wAA_mev"]) >= 70.0 for row in d3a_rows)
    d3a_cells = {(round(float(row["theta_deg"]), 2), round(float(row["wAA_mev"]), 1)) for row in d3a_rows}
    missing_paper_d3a_cells = sorted(PAPER_VISUAL_D3A_CELLS - d3a_cells)
    extra_paper_d3a_cells = sorted(d3a_cells - PAPER_VISUAL_D3A_CELLS)
    exact_anchor_cases = {(float(row["theta_deg"]), float(row["wAA_mev"])) for row in exact_anchor_rows}
    exact_anchor_errors = [
        float(row["final_error"])
        for row in exact_anchor_rows
        if row.get("final_error", "") != ""
    ]
    exact_anchor_wcond = [
        float(row["wcond_mev"])
        for row in exact_anchor_rows
        if row.get("wcond_mev", "") != ""
    ]

    fig9a_summary: dict[str, Any] = {}
    fig9a_params_path = fig9a_dir / "hf_params.json"
    fig9a_conv_path = fig9a_dir / "hf_convergence.json"
    fig9a_order_path = fig9a_dir / "order_parameters.json"
    fig9a_metrics_path = fig9a_dir / "fig9_fig8b_metrics.json"
    if fig9a_params_path.exists() and fig9a_conv_path.exists() and fig9a_order_path.exists() and fig9a_metrics_path.exists():
        fig9a_params = _read_json(fig9a_params_path)
        fig9a_conv = _read_json(fig9a_conv_path)
        fig9a_order = _read_json(fig9a_order_path)
        fig9a_metrics = _read_json(fig9a_metrics_path)
        fig9a_summary = {
            "nu": fig9a_params.get("nu"),
            "theta_deg": fig9a_params.get("theta_deg"),
            "wAA_mev": fig9a_params.get("wAA_mev"),
            "wAB_mev": 1000.0 * float(fig9a_params.get("w_ev", 0.0)),
            "n_k": fig9a_params.get("n_k"),
            "g_shells": fig9a_params.get("g_shells"),
            "epsilon_r": fig9a_params.get("epsilon_r"),
            "U_ev": fig9a_params.get("U_ev"),
            "converged": fig9a_conv.get("best", {}).get("converged"),
            "final_error": fig9a_conv.get("best", {}).get("final_error"),
            "iterations": fig9a_conv.get("best", {}).get("iterations"),
            "hf_gap_mev": 1000.0 * float(fig9a_conv.get("best", {}).get("hf_gap_ev", 0.0)),
            "class_label": fig9a_order.get("strong_coupling", {}).get("class_label"),
            "family": fig9a_order.get("strong_coupling", {}).get("family"),
            "path_conduction_bandwidth_mev": fig9a_metrics.get("conduction_bandwidth", {}).get("max_bandwidth_meV"),
        }

    checks = [
        (
            "Kwan Fig. 9b map parameters",
            _status(
                params.get("n_k") == 12
                and params.get("projected_band_count") == 2
                and abs(float(params.get("epsilon_r", 0.0)) - 8.0) < 1.0e-12
                and abs(float(params.get("w_ev", 0.0)) - 0.11) < 1.0e-12
                and abs(float(params.get("d_sc_nm", 0.0)) - 25.0) < 1.0e-12
            ),
            "n_k=12, projected_band_count=2, epsilon_r=8, wAB=110 meV, d_sc=25 nm.",
        ),
        (
            "Explicit 8x10 Fig. 9b mesh",
            _status(expected_grid and len(scan_rows) == 80),
            (
                f"theta_grid_deg={theta_values}; wAA_grid_meV={waa_values}; "
                f"rows={len(scan_rows)}. Grid centers are not inferred from major ticks."
            ),
        ),
        (
            "Wcond matrix shape",
            _status(expected_shape),
            (
                f"Wcond array shape={metadata.get('wcond_array_shape')} with rows=wAA_grid_meV and "
                f"columns=theta_grid_deg; expected {EXPECTED_WCOND_SHAPE}."
            ),
        ),
        (
            "HF sample accounting",
            _status(int(metadata["n_total_hf_runs"]) == len(detail_rows)),
            (
                f"{metadata['n_parameter_points']} parameter points x "
                f"{samples['n_hf_candidates_per_grid']} HF candidates/grid = "
                f"{metadata['n_total_hf_runs']} total HF runs; observed run-detail rows={len(detail_rows)}."
            ),
        ),
        (
            "Selected rows converged",
            _status(selected_converged == len(scan_rows) and selected_from_converged == len(scan_rows)),
            f"{selected_converged}/{len(scan_rows)} selected rows converged; {selected_from_converged}/{len(scan_rows)} selected from converged pool.",
        ),
        (
            "No blank Wcond",
            _status(len(wconds) == len(scan_rows)),
            f"{len(wconds)}/{len(scan_rows)} selected rows have finite Wcond.",
        ),
        (
            "Observable metadata",
            _status("first unoccupied self-consistent HF eigenband" in params.get("bandwidth_definition", "")),
            params.get("bandwidth_definition", ""),
        ),
        (
            "D3A paper-mask comparison",
            _status(not missing_paper_d3a_cells and not extra_paper_d3a_cells, warn=True),
            (
                f"selected D3A cells={sorted(d3a_cells)}; visual paper target={sorted(PAPER_VISUAL_D3A_CELLS)}; "
                f"missing={missing_paper_d3a_cells}; extra={extra_paper_d3a_cells}. "
                f"All selected D3A cells lie in the upper-left pocket: {d3a_upper_left}."
            ),
        ),
        (
            "Dominant FB family",
            _status(family_counter.get("FB", 0) == len(scan_rows)),
            f"family counts: {dict(family_counter)}.",
        ),
        (
            "Final-error audit field",
            _status(has_selected_final_error and len(selected_final_errors) == len(scan_rows), warn=True),
            (
                f"{len(selected_final_errors)}/{len(scan_rows)} selected 12x12 map rows carry final_error."
                if has_selected_final_error
                else "The 80-parameter-point 12x12 reproduction TSV records convergence/exit_reason but not final_error per selected point."
            ),
        ),
        (
            "Digitized-paper quantitative comparison",
            _status(False, warn=True),
            "This report audits local artifacts; it does not compare against digitized Fig. 9b color values.",
        ),
    ]

    if fig9a_summary:
        checks.append(
            (
                "18x18 Fig. 9a companion point",
                _status(
                    fig9a_summary["converged"] is True
                    and abs(float(fig9a_summary["theta_deg"]) - 1.80) < 1.0e-12
                    and abs(float(fig9a_summary["wAA_mev"]) - 75.0) < 1.0e-12
                    and fig9a_summary["n_k"] == 18
                    and fig9a_summary["g_shells"] == 2
                    and fig9a_summary["class_label"] == "[D3 B]"
                ),
                "Exact theta=1.80, wAA=75 meV, nu=3, n_k=18, g_shells=2 companion run is converged and D3B/FB.",
            )
        )
    if exact_anchor_rows:
        checks.append(
            (
                "Exact-anchor final_error audit",
                _status(
                    len(exact_anchor_errors) == len(exact_anchor_rows)
                    and all(row.get("converged") == "True" for row in exact_anchor_rows)
                ),
                f"{len(exact_anchor_errors)}/{len(exact_anchor_rows)} exact-anchor rows carry final_error and all selected rows converged.",
            )
        )
        checks.append(
            (
                "Exact wAA anchors present",
                _status((1.80, 75.0) in exact_anchor_cases and (1.60, 85.0) in exact_anchor_cases),
                "Exact theta=1.80,wAA=75 and theta=1.60,wAA=85 rows are present in the focused anchor scan.",
            )
        )

    lines: list[str] = []
    lines.append("# HTG Fig. 9b Reproduction Validation")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "Accept the reviewer qualification: the current artifacts support a qualitative Fig. 9b reproduction with "
        "auditable multi-init selection, but not yet an unqualified pointwise quantitative reproduction claim."
    )
    lines.append("")
    lines.append("## Source Artifacts")
    lines.append("")
    lines.append(f"- Scan directory: `{scan_dir}`")
    lines.append(f"- Scan artifact prefix: `{scan_prefix}`")
    lines.append(f"- Companion 18x18 band-structure directory: `{fig9a_dir}`")
    if exact_anchor_rows and exact_anchor_dir is not None:
        lines.append(f"- Exact-anchor scan directory: `{exact_anchor_dir}`")
    lines.append(f"- Slurm job: `{runtime.get('slurm_job_id', '')}` on `{runtime.get('hostname', '')}`")
    lines.append(f"- Runtime: `{_fmt(runtime.get('elapsed_sec'))} s`")
    lines.append(f"- Grid metadata: `{scan_dir / 'grid_metadata.json'}`")
    lines.append("")
    lines.append("## Checklist")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for name, status, detail in checks:
        lines.append(f"| {name} | {status} | {detail} |")
    lines.append("")
    lines.append("## Scan Summary")
    lines.append("")
    lines.append(f"- Grid: theta={theta_values}, wAA={waa_values} meV")
    lines.append(f"- Derived theta cell edges: {metadata['theta_edge_values_deg']}")
    lines.append(f"- Derived wAA cell edges: {metadata['wAA_edge_values_meV']} meV")
    lines.append(f"- Wcond array shape: {metadata['wcond_array_shape']} with rows=wAA_grid_meV and columns=theta_grid_deg")
    lines.append(f"- Mesh note: {metadata['mesh_note']}")
    lines.append(f"- Reproduction mode: {metadata['reproduction_mode']}")
    lines.append(f"- Parameter grid samples: {metadata['n_theta']} * {metadata['n_wAA']} = {metadata['n_parameter_points']}")
    lines.append(
        "- HF initial samples per grid point: "
        f"{samples['n_init_modes']} init modes * {samples['n_seeds']} seeds = {samples['n_hf_candidates_per_grid']}"
    )
    lines.append(f"- Total HF runs: {metadata['n_parameter_points']} * {samples['n_hf_candidates_per_grid']} = {metadata['n_total_hf_runs']}")
    lines.append(f"- Paper seed note: {metadata['paper_seed_note']}")
    lines.append(f"- Translation-symmetry scope: {metadata['translation_symmetry']}")
    lines.append(f"- Initial-state scope: {metadata['initial_state_scope']}")
    lines.append(f"- Selected rows: {len(scan_rows)}")
    lines.append(f"- Per-run rows: {len(detail_rows)}; converged={detail_converged}; exit reasons={dict(detail_exit_counter)}")
    lines.append(f"- Observable: `{metadata['observable']['definition']}` over `{metadata['observable']['computed_over']}`")
    lines.append(f"- Observable not computed from: `{metadata['observable']['not_computed_from']}`")
    lines.append(f"- Wcond min/max: {_fmt(min(wconds))} / {_fmt(max(wconds))} meV")
    lines.append(f"- Wcond <= 15 meV: {sum(value <= 15.0 for value in wconds)}/{len(wconds)}")
    if has_selected_final_error:
        lines.append(
            f"- Selected-row final_error: {len(selected_final_errors)}/{len(scan_rows)} rows; "
            f"max={_fmt(max(selected_final_errors)) if selected_final_errors else ''}"
        )
    lines.append(f"- Class counts: {dict(class_counter)}")
    lines.append(f"- Selected D3A cells: {sorted(d3a_cells)}")
    lines.append(f"- Paper-visual D3A target cells from local Fig. 5/Fig. 9b inspection: {sorted(PAPER_VISUAL_D3A_CELLS)}")
    lines.append(f"- Missing paper-visual D3A cells: {missing_paper_d3a_cells}")
    lines.append(f"- Extra D3A cells relative to paper-visual target: {extra_paper_d3a_cells}")
    if comparison:
        lines.append(
            "- Multi-init comparison vs old scan: "
            f"{comparison['shared_points']} shared points; decreased={comparison['decreased']}, "
            f"unchanged={comparison['unchanged']}, increased={comparison['increased']}."
        )
    lines.append("")
    lines.append("## Anchor Points")
    lines.append("")
    lines.append(
        "The corrected 8x10 reproduction scan uses explicit center values, including wAA=75 and 85 meV. "
        "Anchor matches are exact when those centers are present; plotted cell edges are not treated as calculation points."
    )
    lines.append("")
    lines.append(
        "| Anchor | Requested theta | Requested wAA | Matched theta | Matched wAA | Match | Class | Family | Wcond meV | HF gap meV | Init | Converged |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- |")
    for row in anchor_rows:
        lines.append(
            "| {anchor_name} | {requested_theta_deg} | {requested_wAA_mev} | {matched_theta_deg} | "
            "{matched_wAA_mev} | {match_type} | {class_label} | {family_label} | {Wcond_mev} | "
            "{hf_gap_mev} | {best_init_mode}:{best_seed} | {converged} |".format(**row)
        )
    lines.append("")
    if fig9a_summary:
        lines.append("## Companion 18x18 Point")
        lines.append("")
        lines.append(
            "- theta={theta_deg}, wAA={wAA_mev} meV, wAB={wAB_mev:.6g} meV, nu={nu}, "
            "n_k={n_k}, g_shells={g_shells}, epsilon_r={epsilon_r}, U_ev={U_ev}".format(**fig9a_summary)
        )
        lines.append(
            "- converged={converged}, iterations={iterations}, final_error={final_error}, "
            "HF gap={hf_gap_mev:.6g} meV, class={class_label}, family={family}".format(**fig9a_summary)
        )
        lines.append(
            "- path conduction bandwidth from the saved Fig. 9a-style band artifact: "
            f"{_fmt(fig9a_summary['path_conduction_bandwidth_mev'])} meV"
        )
        lines.append("")
    if exact_anchor_rows:
        lines.append("## Exact-Anchor Scan")
        lines.append("")
        lines.append(
            "These rows rerun selected anchor points directly as cross-checks against the corrected 8x10 reproduction map."
        )
        lines.append("")
        lines.append(
            f"- Selected rows: {len(exact_anchor_rows)}; finite Wcond rows: {len(exact_anchor_wcond)}; "
            f"final_error rows: {len(exact_anchor_errors)}"
        )
        if exact_anchor_wcond:
            lines.append(f"- Wcond min/max: {_fmt(min(exact_anchor_wcond))} / {_fmt(max(exact_anchor_wcond))} meV")
        lines.append("")
        lines.append("| case | theta | wAA | class | family | Wcond meV | HF gap meV | final error | init | converged |")
        lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
        for row in exact_anchor_rows:
            lines.append(
                "| {case_label} | {theta_deg} | {wAA_mev} | {class_label} | {family} | {wcond_mev} | "
                "{hf_gap_mev} | {final_error} | {best_init_mode}:{best_seed} | {converged} |".format(**row)
            )
        lines.append("")
    lines.append("## Remaining Qualification")
    lines.append("")
    if has_selected_final_error and len(selected_final_errors) == len(scan_rows):
        lines.append("- The selected 80-parameter-point 12x12 reproduction map rows now store `final_error` for the full scan.")
    else:
        lines.append("- The selected 80-parameter-point 12x12 reproduction map rows do not store `final_error`; they store `converged`, `exit_reason`, and iteration count.")
        if exact_anchor_rows:
            lines.append("- The exact-anchor scan does store `final_error`, but only for the focused anchor points, not for the full map.")
    lines.append("- The scan report validates local metadata and TSVs, but it does not digitize the paper color map for pointwise residuals.")
    if missing_paper_d3a_cells or extra_paper_d3a_cells:
        lines.append(
            "- Paper-visual D3A mask differences are reported as a notice only; selected TSV labels still follow "
            "the lowest-energy converged HF run."
        )
    lines.append("- This is a qualitative reproduction-mode seed budget unless `n_hf_candidates_per_grid >= 300`.")
    if not exact_anchor_rows:
        lines.append("- Exact wAA=75 and 85 meV are grid centers in the corrected 8x10 map.")
    else:
        lines.append("- Exact wAA=75 and 85 meV are covered by both the corrected 8x10 map and the focused anchor scan.")
    lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--scan-prefix", default=DEFAULT_SCAN_PREFIX)
    parser.add_argument("--fig9a-dir", type=Path, default=DEFAULT_FIG9A_DIR)
    parser.add_argument(
        "--exact-anchor-dir",
        type=Path,
        default=DEFAULT_EXACT_ANCHOR_DIR,
        help="Optional focused exact-anchor scan directory. Missing directories are ignored.",
    )
    args = parser.parse_args()

    scan_prefix = str(args.scan_prefix).strip()
    if not scan_prefix:
        raise SystemExit("--scan-prefix must not be empty")

    scan_json = _read_json(args.scan_dir / f"{scan_prefix}.json")
    scan_rows = _read_tsv(args.scan_dir / f"{scan_prefix}.tsv")
    detail_rows = _read_tsv(args.scan_dir / f"{scan_prefix}_run_details.tsv")
    exact_anchor_rows = _read_exact_anchor_rows(args.exact_anchor_dir)
    metadata = _grid_metadata(scan_json=scan_json, scan_rows=scan_rows, detail_rows=detail_rows)

    anchor_path = args.scan_dir / "fig9b_reproduction_anchor_points.tsv"
    report_path = args.scan_dir / "fig9b_reproduction_validation.md"
    metadata_path = args.scan_dir / "grid_metadata.json"
    _write_json(metadata_path, metadata)
    anchor_rows = _write_anchor_tsv(anchor_path, scan_rows)
    _make_report(
        report_path,
        scan_dir=args.scan_dir,
        scan_prefix=scan_prefix,
        fig9a_dir=args.fig9a_dir,
        exact_anchor_dir=args.exact_anchor_dir,
        scan_json=scan_json,
        scan_rows=scan_rows,
        detail_rows=detail_rows,
        anchor_rows=anchor_rows,
        exact_anchor_rows=exact_anchor_rows,
        metadata=metadata,
    )
    print(f"wrote {metadata_path}")
    print(f"wrote {anchor_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
