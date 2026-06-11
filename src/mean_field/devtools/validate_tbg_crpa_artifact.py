from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mean_field.crpa import load_crpa_result, validate_hf_compatible_crpa
from mean_field.crpa.validation import (
    DEFAULT_FIG1E_PAPER_POINTS,
    compare_fig1e_window_to_reference,
    compare_fig1e_window_to_paper_points,
    crpa_convention_family,
    dielectric_algebra_summary,
    fig1e_gate_failures,
    fig1e_paper_point_gate_failures,
    hermitian_positivity_summary,
    validation_summary,
)
from mean_field.core.io import write_text_artifact
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.tbg import TBGParameters


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "TBG_HF_cRPA" / "crpa_validation"


def _params_from_result(result) -> TBGParameters:
    metadata = dict(result.metadata)
    return TBGParameters.from_degrees(
        float(result.theta_deg),
        vf=float(metadata.get("vf", 2135.4)),
        w0=float(metadata.get("w0", 79.7)),
        w1=float(metadata.get("w1", 97.4)),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TBG cRPA Artifact Validation",
        "",
        f"status: {payload['status']}",
        f"crpa_dir: {payload['crpa_dir']}",
        f"reference_crpa_dir: {payload.get('reference_crpa_dir', '')}",
        "",
        "## Checks",
        "",
    ]
    for group_name in ("metadata", "basic", "dielectric_algebra", "positivity", "fig1e_paper_points", "fig1e_reference"):
        group = payload.get(group_name)
        if not isinstance(group, dict):
            continue
        lines.extend([f"### {group_name}", ""])
        for key, value in sorted(group.items()):
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.16e}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    failures = payload.get("failures", [])
    lines.extend(["## Failures", ""])
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- none")
    write_text_artifact("\n".join(lines).rstrip() + "\n", path)


def _load_reference_points(path: Path) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pieces = stripped.replace(",", "\t").split()
            if len(pieces) < 2:
                continue
            try:
                q_value = float(pieces[0])
                eps_value = float(pieces[1])
            except ValueError:
                continue
            points.append((q_value, eps_value))
    if not points:
        raise ValueError(f"No two-column numeric Fig. 1(e) reference points found in {path}")
    return tuple(points)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a TBG cRPA artifact with metadata and Fig. 1(e) physics gates.")
    parser.add_argument("--crpa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reference-crpa-dir",
        type=Path,
        default=None,
        help="Optional known-good cRPA artifact used for an artifact-vs-artifact window-curve comparison.",
    )
    parser.add_argument(
        "--paper-points-tsv",
        type=Path,
        default=None,
        help="Optional two-column q_nm_inv, epsilon_times_bn table for the corrected Zhang Fig. 1(e) anchors.",
    )
    parser.add_argument("--disable-paper-point-gate", action="store_true")
    parser.add_argument("--require-hf-compatible", action="store_true")
    parser.add_argument("--theta-deg", type=float, default=None)
    parser.add_argument("--overlap-lg", type=int, default=9)
    parser.add_argument("--max-fig1e-rmse", type=float, default=2.5)
    parser.add_argument("--max-fig1e-max-abs", type=float, default=8.0)
    parser.add_argument("--max-fig1e-peak-abs", type=float, default=4.0)
    parser.add_argument("--max-fig1e-q-peak-abs-nm-inv", type=float, default=0.08)
    parser.add_argument("--max-fig1e-checkpoint-abs", type=float, default=5.0)
    parser.add_argument("--min-fig1e-points", type=int, default=20)
    parser.add_argument("--max-paper-point-rmse", type=float, default=0.8)
    parser.add_argument("--max-paper-point-max-abs", type=float, default=1.5)
    parser.add_argument("--max-paper-point-mean-abs", type=float, default=0.7)
    parser.add_argument("--min-paper-points", type=int, default=5)
    parser.add_argument("--epsilon-min-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hermitian-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--dielectric-identity-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--screened-v-hermitian-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--allow-login", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not bool(args.allow_login):
        ensure_not_running_compute_on_login_node("TBG cRPA artifact validation")

    result = load_crpa_result(args.crpa_dir)
    params = _params_from_result(result)
    theta_deg = float(result.theta_deg if args.theta_deg is None else args.theta_deg)
    failures: list[str] = []
    metadata_payload = dict(result.metadata)
    convention_family = crpa_convention_family(result)
    metadata_payload["convention_family"] = convention_family

    if bool(args.require_hf_compatible):
        try:
            validate_hf_compatible_crpa(result, params, theta_deg=theta_deg, overlap_lg=int(args.overlap_lg))
        except Exception as exc:
            failures.append(f"HF-compatible metadata gate failed: {exc}")

    basic = validation_summary(result)
    if float(basic["effective_epsilon_min"]) < 1.0 - float(args.epsilon_min_tolerance):
        failures.append(
            "effective_epsilon_min="
            f"{float(basic['effective_epsilon_min']):.6g} is below 1 within tolerance {float(args.epsilon_min_tolerance):.6g}"
        )

    dielectric_algebra = dielectric_algebra_summary(result, params)
    algebra_tolerance = float(args.dielectric_identity_tolerance)
    for key in (
        "epsilon_equals_I_plus_chi0V_max_abs",
        "epsilon_inv_left_residual_max_abs",
        "epsilon_inv_right_residual_max_abs",
        "screened_v_equals_V_epsilon_inv_max_abs",
        "effective_epsilon_equals_real_diag_max_abs",
        "epsilon_diag_imag_max_abs",
    ):
        value = float(dielectric_algebra[key])
        if value > algebra_tolerance:
            failures.append(f"{key}={value:.6g} exceeds {algebra_tolerance:.6g}")
    screened_v_antiherm = float(dielectric_algebra["screened_v_antihermitian_max_abs"])
    if screened_v_antiherm > float(args.screened_v_hermitian_tolerance):
        failures.append(
            "screened_v_antihermitian_max_abs="
            f"{screened_v_antiherm:.6g} exceeds {float(args.screened_v_hermitian_tolerance):.6g}"
        )

    positivity = hermitian_positivity_summary(result, params)
    if float(positivity["hermitian_similar_eig_min"]) < -float(args.positivity_tolerance):
        failures.append(
            "hermitian_similar_eig_min="
            f"{float(positivity['hermitian_similar_eig_min']):.6g} is below "
            f"-{float(args.positivity_tolerance):.6g}"
        )
    if float(positivity["hermitian_similar_antihermitian_max_abs"]) > float(args.hermitian_tolerance):
        failures.append(
            "hermitian_similar_antihermitian_max_abs="
            f"{float(positivity['hermitian_similar_antihermitian_max_abs']):.6g} exceeds "
            f"{float(args.hermitian_tolerance):.6g}"
        )

    paper_point_payload: dict[str, object] = {}
    if not bool(args.disable_paper_point_gate):
        reference_points = (
            _load_reference_points(Path(args.paper_points_tsv))
            if args.paper_points_tsv is not None
            else DEFAULT_FIG1E_PAPER_POINTS
        )
        paper_point_payload = compare_fig1e_window_to_paper_points(result, reference_points)
        if convention_family == "zhang_paper_reference":
            failures.extend(
                fig1e_paper_point_gate_failures(
                    paper_point_payload,
                    max_rmse=float(args.max_paper_point_rmse),
                    max_abs=float(args.max_paper_point_max_abs),
                    max_mean_abs=float(args.max_paper_point_mean_abs),
                    min_points=int(args.min_paper_points),
                )
            )
        else:
            paper_point_payload["fig1e_paper_gate_hard_fail_enabled"] = 0
            paper_point_payload["fig1e_paper_gate_skip_reason"] = (
                "artifact convention is not zhang_paper_reference"
            )

    fig1e_payload: dict[str, object] = {}
    reference_dir = None if args.reference_crpa_dir is None else Path(args.reference_crpa_dir)
    if reference_dir is not None:
        if not (reference_dir / "crpa_params.json").exists():
            failures.append(f"Fig. 1(e) reference cRPA artifact not found: {reference_dir}")
        else:
            reference = load_crpa_result(reference_dir)
            reference_convention_family = crpa_convention_family(reference)
            fig1e_payload = compare_fig1e_window_to_reference(result, reference)
            fig1e_payload["reference_convention_family"] = reference_convention_family
            if convention_family == reference_convention_family:
                failures.extend(
                    fig1e_gate_failures(
                        fig1e_payload,
                        max_rmse=float(args.max_fig1e_rmse),
                        max_abs=float(args.max_fig1e_max_abs),
                        max_peak_abs=float(args.max_fig1e_peak_abs),
                        max_q_peak_abs_nm_inv=float(args.max_fig1e_q_peak_abs_nm_inv),
                        max_checkpoint_abs=float(args.max_fig1e_checkpoint_abs),
                        min_points=int(args.min_fig1e_points),
                    )
                )
            else:
                fig1e_payload["fig1e_reference_gate_hard_fail_enabled"] = 0
                fig1e_payload["fig1e_reference_gate_skip_reason"] = "artifact/reference convention mismatch"

    status = "pass" if not failures else "fail"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_name = Path(args.crpa_dir).name
    payload: dict[str, object] = {
        "status": status,
        "crpa_dir": str(args.crpa_dir),
        "reference_crpa_dir": "" if reference_dir is None else str(reference_dir),
        "metadata": metadata_payload,
        "basic": basic,
        "dielectric_algebra": dielectric_algebra,
        "positivity": positivity,
        "fig1e_paper_points": paper_point_payload,
        "fig1e_reference": fig1e_payload,
        "failures": failures,
    }
    json_path = out / f"{safe_name}_validation.json"
    report_path = out / f"{safe_name}_validation.md"
    write_json(json_path, payload)
    _write_report(report_path, payload)
    print(f"[crpa-artifact-validation] status={status} json={json_path} report={report_path}", flush=True)
    if failures:
        for item in failures:
            print(f"[crpa-artifact-validation] failure: {item}", flush=True)
    else:
        print(
            "[crpa-artifact-validation] fig1e_paper_points "
            f"rmse={float(paper_point_payload.get('fig1e_paper_rmse', np.nan)):.6g} "
            f"max_abs={float(paper_point_payload.get('fig1e_paper_max_abs', np.nan)):.6g} "
            f"mean_abs={float(paper_point_payload.get('fig1e_paper_mean_abs', np.nan)):.6g}",
            flush=True,
        )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
