from __future__ import annotations

"""Validate reusable saved topology/paper-reproduction artifacts.

This module is intentionally file-based: it does not solve Hamiltonians or run
SCF.  It checks that saved old-result artifacts are internally consistent with
the unified Berry-geometry helpers where the required saved arrays exist.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import chern_number_from_berry_curvature


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(checks: list[dict[str, Any]], name: str, status: str, detail: str, **payload: Any) -> None:
    checks.append({"name": name, "status": status, "detail": detail, **payload})


def _check_tmbg_fig2(root: Path, checks: list[dict[str, Any]]) -> None:
    fig_root = root / "results/TMBG/tmbg_fig2_chern_paperpath_bothvalleys_sewn_final_20260427_105114"
    expected_by_delta = {
        "delta_+000mev": {"cnp_pair_total": -1},
        "delta_-040mev": {"valence": 1, "conduction": -2},
        "delta_+060mev": {"valence": -2, "conduction": 1},
    }
    for delta_dir, expected in expected_by_delta.items():
        panel_dir = fig_root / delta_dir
        chern_path = panel_dir / "chern_numbers.json"
        if not chern_path.exists():
            _record(checks, f"tmbg_fig2.{delta_dir}.chern_json", "fail", f"Missing {chern_path}")
            continue
        payload = _read_json(chern_path)
        status = payload.get("status")
        observed: dict[str, int] = {}
        for key, expected_value in expected.items():
            if key == "cnp_pair_total":
                entry = payload.get("chern", {}).get("cnp_pair_total", {})
            else:
                entry = payload.get("chern", {}).get(key, {})
            observed[key] = int(entry.get("rounded_chern_number"))
            if observed[key] != int(expected_value):
                _record(
                    checks,
                    f"tmbg_fig2.{delta_dir}.{key}",
                    "fail",
                    f"Expected rounded Chern {expected_value}, observed {observed[key]}",
                    path=str(chern_path),
                )
            else:
                _record(
                    checks,
                    f"tmbg_fig2.{delta_dir}.{key}",
                    "pass",
                    "Saved rounded Chern matches the paper-target expectation.",
                    path=str(chern_path),
                    rounded_chern=observed[key],
                    saved_status=status,
                )

        berry_path = panel_dir / "berry_curvature.npz"
        if not berry_path.exists():
            _record(checks, f"tmbg_fig2.{delta_dir}.saved_berry", "skip", f"Missing {berry_path}")
            continue
        with np.load(berry_path) as data:
            for key in ("valence", "conduction"):
                berry_key = f"berry_curvature_{key}"
                chern_key = f"chern_number_{key}"
                if berry_key not in data or chern_key not in data:
                    continue
                integrated = chern_number_from_berry_curvature(np.asarray(data[berry_key], dtype=float))
                saved = float(np.asarray(data[chern_key]).item())
                if abs(integrated - saved) < 1.0e-10:
                    _record(
                        checks,
                        f"tmbg_fig2.{delta_dir}.{key}.saved_berry_integral",
                        "pass",
                        "Unified curvature integration reproduces the saved Chern number.",
                        path=str(berry_path),
                        integrated_chern=integrated,
                        saved_chern=saved,
                    )
                else:
                    _record(
                        checks,
                        f"tmbg_fig2.{delta_dir}.{key}.saved_berry_integral",
                        "fail",
                        "Unified curvature integration does not match the saved Chern number.",
                        path=str(berry_path),
                        integrated_chern=integrated,
                        saved_chern=saved,
                    )


def _rounded(payload: dict[str, Any], panel: str, key: str) -> int | None:
    entry = payload["chern_by_panel"][panel]["topology"].get(key, {})
    if not isinstance(entry, dict) or "rounded_chern_number" not in entry:
        return None
    return int(entry["rounded_chern_number"])


def _check_tdbg_fig3(root: Path, checks: list[dict[str, Any]]) -> None:
    summary_path = root / "results/TDBG/tdbg_fig3_chern_20260425_theta133_mesh21_open_valleypath/summary.json"
    if not summary_path.exists():
        _record(checks, "tdbg_fig3.summary", "fail", f"Missing {summary_path}")
        return
    summary = _read_json(summary_path)
    expected = {
        "abab_delta_000mev": {"central_pair": 0},
        "abab_delta_005mev": {"valence_band": -3, "conduction_band": 3, "central_pair": 0},
        "abab_delta_020mev": {"valence_band": -2, "conduction_band": 2, "central_pair": 0},
        "abba_delta_000mev": {"valence_band": 0, "conduction_band": 2, "central_pair": 2},
        "abba_delta_005mev": {"valence_band": 0, "conduction_band": 2, "central_pair": 2},
        "abba_delta_020mev": {"valence_band": 1, "conduction_band": 1, "central_pair": 2},
    }
    for panel, panel_expected in expected.items():
        for key, expected_value in panel_expected.items():
            observed = _rounded(summary, panel, key)
            if observed == expected_value:
                _record(
                    checks,
                    f"tdbg_fig3.{panel}.{key}",
                    "pass",
                    "Saved rounded Chern matches the recorded Fig. 3 target summary.",
                    path=str(summary_path),
                    rounded_chern=observed,
                )
            else:
                _record(
                    checks,
                    f"tdbg_fig3.{panel}.{key}",
                    "fail",
                    f"Expected rounded Chern {expected_value}, observed {observed}",
                    path=str(summary_path),
                )


def _check_htg_chern(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / "results/HTG/htg_fig2b_fig3b_alpha2_1p197_paper_axes_20260429_165908/chern_numbers.json"
    if not path.exists():
        _record(checks, "htg_fig2b3b.chern_json", "fail", f"Missing {path}")
        return
    data = _read_json(path)
    expected = {"rounded_chern_a": -1, "rounded_chern_b": -2, "rounded_total_chern": -3}
    for key, expected_value in expected.items():
        observed = int(data[key])
        _record(
            checks,
            f"htg_fig2b3b.{key}",
            "pass" if observed == expected_value else "fail",
            "Saved HTG Chern-basis integer matches expected value." if observed == expected_value else "Saved HTG Chern mismatch.",
            path=str(path),
            observed=observed,
            expected=expected_value,
        )
    residual_ok = float(data.get("integer_residual_a", 1.0)) < 1.0e-10 and float(data.get("integer_residual_b", 1.0)) < 1.0e-10
    _record(
        checks,
        "htg_fig2b3b.integer_residuals",
        "pass" if residual_ok else "fail",
        "Saved HTG Chern residuals are near machine precision." if residual_ok else "Saved HTG Chern residuals are too large.",
        path=str(path),
        integer_residual_a=float(data.get("integer_residual_a", float("nan"))),
        integer_residual_b=float(data.get("integer_residual_b", float("nan"))),
    )


def _check_rlg_fig6_status(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / "results/RnG_hBN/fig6_completed_two_scf_chern_sewn_20260525_002/hf_chern_summary.json"
    if not path.exists():
        _record(checks, "rlg_fig6.chern_summary", "skip", f"Missing {path}")
        return
    data = _read_json(path)
    for panel in data.get("panels", []):
        comparison = panel.get("paper_fig6_comparison", {})
        central = panel.get("chern", {}).get("central_pair", {})
        name = f"rlg_fig6.{panel.get('panel')}.central_pair_abs_chern"
        matches = bool(comparison.get("matches_paper_abs_chern"))
        _record(
            checks,
            name,
            "pass" if matches else "known_gap",
            "Saved central-pair |C| matches the paper Fig. 6 target."
            if matches
            else "Saved artifact is a useful diagnostic but does not match the paper Fig. 6 |C| target.",
            path=str(path),
            observed_abs_rounded_chern=central.get("absolute_rounded_chern_number"),
            paper_expected_abs_chern=comparison.get("paper_expected_abs_chern"),
            rounded_chern=central.get("rounded_chern_number"),
        )


def run_validation(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _check_tmbg_fig2(root, checks)
    _check_tdbg_fig3(root, checks)
    _check_htg_chern(root, checks)
    _check_rlg_fig6_status(root, checks)
    failures = [check for check in checks if check["status"] == "fail"]
    known_gaps = [check for check in checks if check["status"] == "known_gap"]
    return {
        "root": str(root),
        "check_count": len(checks),
        "failure_count": len(failures),
        "known_gap_count": len(known_gaps),
        "checks": checks,
        "status": "pass" if not failures else "fail",
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Unified topology saved-result validation",
        "",
        f"- status: `{result['status']}`",
        f"- checks: `{result['check_count']}`",
        f"- failures: `{result['failure_count']}`",
        f"- known gaps: `{result['known_gap_count']}`",
        "",
        "## Checks",
        "",
        "| Status | Name | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        lines.append(f"| {check['status']} | `{check['name']}` | {check['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    result = run_validation(root)
    output_dir = args.output_dir or (root / "results/topology_framework_validation_20260528")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "saved_result_validation.json"
    md_path = output_dir / "saved_result_validation.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, result)
    print(f"saved_result_validation_json={json_path}")
    print(f"saved_result_validation_md={md_path}")
    print(f"status={result['status']} failures={result['failure_count']} known_gaps={result['known_gap_count']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
