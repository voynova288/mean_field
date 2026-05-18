from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..systems.tbg.params import TBGParameters
from .band_classifier import classify_flat_bands
from .bm import solve_all_band_bm_model
from .coulomb import coulomb_potential_table_mev
from .diagnostics import representative_fig1e_window_curve
from .grid import build_q_shift_table, build_uniform_crpa_grid
from .susceptibility import compute_constrained_chi0, compute_constrained_chi0_by_subtraction
from .workflow import CRPAResult


DEFAULT_FIG1E_PAPER_POINTS: tuple[tuple[float, float], ...] = (
    # Corrected Zhang Fig. 1(e) digitization anchors from the local
    # 2026-05-05 PRL-PDF extraction, in units (q_nm_inv, epsilon * epsilon_BN).
    (0.4, 17.252347286782747),
    (0.5, 15.653099169877715),
    (0.540481454, 15.061710543312188),
    (0.6, 14.48698075129859),
    (0.8, 13.037662145353465),
    (1.0, 11.954837899532134),
    (1.08, 11.645816518609129),
)


def validation_summary(result: CRPAResult) -> dict[str, float]:
    chi_herm = 0.0
    eps_min = float(np.min(np.real(result.effective_epsilon)))
    eps_max = float(np.max(np.real(result.effective_epsilon)))
    for chi in result.chi0:
        chi_herm = max(chi_herm, float(np.max(np.abs(chi - chi.conjugate().T))))
    return {
        "chi0_hermiticity_max_abs": chi_herm,
        "effective_epsilon_min": eps_min,
        "effective_epsilon_max": eps_max,
        "effective_epsilon_times_bn_min": eps_min * float(result.coulomb_params.epsilon_bn),
        "effective_epsilon_times_bn_max": eps_max * float(result.coulomb_params.epsilon_bn),
    }


def _params_from_result(result: CRPAResult) -> TBGParameters:
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


def hermitian_positivity_summary(
    result: CRPAResult,
    params: TBGParameters | None = None,
) -> dict[str, float]:
    """Check eigenvalues of I + V^(1/2) chi0 V^(1/2).

    The raw dielectric matrix is stored as I + chi0 V and is generally not the
    Hermitian-similar representation.  This check implements the positivity
    diagnostic required by the cRPA work plan.
    """

    resolved_params = _params_from_result(result) if params is None else params
    eig_min = float("inf")
    eig_max = float("-inf")
    antihermitian = 0.0
    v_min = float("inf")
    v_max = float("-inf")
    for iq, q_tilde in enumerate(np.asarray(result.q_tilde, dtype=np.complex128)):
        v_q = np.asarray(
            coulomb_potential_table_mev(
                complex(q_tilde),
                np.asarray(result.q_vectors, dtype=np.complex128),
                resolved_params,
                result.coulomb_params,
            ),
            dtype=float,
        )
        v_min = min(v_min, float(np.min(v_q)))
        v_max = max(v_max, float(np.max(v_q)))
        sqrt_v = np.sqrt(np.maximum(v_q, 0.0))
        chi = np.asarray(result.chi0[int(iq)], dtype=np.complex128)
        hermitian_similar = np.eye(chi.shape[0], dtype=np.complex128)
        hermitian_similar += sqrt_v[:, None] * chi * sqrt_v[None, :]
        antihermitian = max(
            antihermitian,
            float(np.max(np.abs(hermitian_similar - hermitian_similar.conjugate().T))),
        )
        sym = 0.5 * (hermitian_similar + hermitian_similar.conjugate().T)
        eigvals = np.linalg.eigvalsh(sym)
        eig_min = min(eig_min, float(np.min(eigvals)))
        eig_max = max(eig_max, float(np.max(eigvals)))
    if not np.isfinite(eig_min):
        eig_min = float("nan")
        eig_max = float("nan")
    if not np.isfinite(v_min):
        v_min = float("nan")
        v_max = float("nan")
    return {
        "hermitian_similar_eig_min": eig_min,
        "hermitian_similar_eig_max": eig_max,
        "hermitian_similar_antihermitian_max_abs": antihermitian,
        "coulomb_table_min_mev": v_min,
        "coulomb_table_max_mev": v_max,
    }


def dielectric_algebra_summary(
    result: CRPAResult,
    params: TBGParameters | None = None,
) -> dict[str, float]:
    """Check the stored dielectric matrices against the Zhang cRPA identities."""

    resolved_params = _params_from_result(result) if params is None else params
    epsilon_formula = 0.0
    epsilon_inv_left = 0.0
    epsilon_inv_right = 0.0
    screened_v_formula = 0.0
    effective_epsilon_formula = 0.0
    screened_v_antihermitian = 0.0
    eps_diag_imag = 0.0
    for iq, q_tilde in enumerate(np.asarray(result.q_tilde, dtype=np.complex128)):
        chi = np.asarray(result.chi0[int(iq)], dtype=np.complex128)
        eps = np.asarray(result.dielectric_matrix[int(iq)], dtype=np.complex128)
        eps_inv = np.asarray(result.epsilon_inv[int(iq)], dtype=np.complex128)
        screened_v = np.asarray(result.screened_v[int(iq)], dtype=np.complex128)
        effective = np.asarray(result.effective_epsilon[int(iq)], dtype=float)
        v_q = np.asarray(
            coulomb_potential_table_mev(
                complex(q_tilde),
                np.asarray(result.q_vectors, dtype=np.complex128),
                resolved_params,
                result.coulomb_params,
            ),
            dtype=float,
        )
        eye = np.eye(chi.shape[0], dtype=np.complex128)
        eps_expected = eye + chi * v_q[None, :]
        screened_expected = v_q[:, None] * eps_inv
        epsilon_formula = max(epsilon_formula, float(np.max(np.abs(eps - eps_expected))))
        epsilon_inv_left = max(epsilon_inv_left, float(np.max(np.abs(eps_inv @ eps - eye))))
        epsilon_inv_right = max(epsilon_inv_right, float(np.max(np.abs(eps @ eps_inv - eye))))
        screened_v_formula = max(screened_v_formula, float(np.max(np.abs(screened_v - screened_expected))))
        effective_epsilon_formula = max(
            effective_epsilon_formula,
            float(np.max(np.abs(effective - np.real(np.diag(eps))))),
        )
        screened_v_antihermitian = max(
            screened_v_antihermitian,
            float(np.max(np.abs(screened_v - screened_v.conjugate().T))),
        )
        eps_diag_imag = max(eps_diag_imag, float(np.max(np.abs(np.imag(np.diag(eps))))))
    return {
        "epsilon_equals_I_plus_chi0V_max_abs": epsilon_formula,
        "epsilon_inv_left_residual_max_abs": epsilon_inv_left,
        "epsilon_inv_right_residual_max_abs": epsilon_inv_right,
        "screened_v_equals_V_epsilon_inv_max_abs": screened_v_formula,
        "effective_epsilon_equals_real_diag_max_abs": effective_epsilon_formula,
        "screened_v_antihermitian_max_abs": screened_v_antihermitian,
        "epsilon_diag_imag_max_abs": eps_diag_imag,
    }


def compare_fig1e_window_to_reference(
    result: CRPAResult,
    reference: CRPAResult,
    *,
    x_max_nm_inv: float = 1.2,
    bin_width_nm_inv: float = 0.0125,
) -> dict[str, float | int]:
    xs, ys, counts = representative_fig1e_window_curve(
        result,
        x_max_nm_inv=float(x_max_nm_inv),
        bin_width_nm_inv=float(bin_width_nm_inv),
    )
    ref_xs, ref_ys, ref_counts = representative_fig1e_window_curve(
        reference,
        x_max_nm_inv=float(x_max_nm_inv),
        bin_width_nm_inv=float(bin_width_nm_inv),
    )
    if xs.size < 2 or ref_xs.size < 2:
        return {
            "fig1e_curve_points": int(xs.size),
            "fig1e_reference_curve_points": int(ref_xs.size),
            "fig1e_rmse": float("inf"),
            "fig1e_mean_abs": float("inf"),
            "fig1e_max_abs": float("inf"),
            "fig1e_peak_abs_diff": float("inf"),
            "fig1e_q_peak_abs_diff_nm_inv": float("inf"),
            "fig1e_checkpoint_max_abs": float("inf"),
            "fig1e_q04_abs_diff": float("inf"),
            "fig1e_q08_abs_diff": float("inf"),
            "fig1e_q12_abs_diff": float("inf"),
            "fig1e_min_bin_count": 0,
            "fig1e_reference_min_bin_count": 0,
        }

    lo = max(float(np.min(xs)), float(np.min(ref_xs)))
    hi = min(float(np.max(xs)), float(np.max(ref_xs)), float(x_max_nm_inv))
    sample_x = ref_xs[(ref_xs >= lo) & (ref_xs <= hi)]
    if sample_x.size == 0:
        sample_x = np.linspace(lo, hi, num=64, dtype=float)
    y_interp = np.interp(sample_x, xs, ys)
    ref_interp = np.interp(sample_x, ref_xs, ref_ys)
    diff = y_interp - ref_interp

    peak_idx = int(np.argmax(ys))
    ref_peak_idx = int(np.argmax(ref_ys))

    checkpoint_diffs = {}
    for q in (0.4, 0.8, 1.2):
        value = float(np.interp(q, xs, ys))
        ref_value = float(np.interp(q, ref_xs, ref_ys))
        checkpoint_diffs[f"fig1e_q{str(q).replace('.', '')}_abs_diff"] = abs(value - ref_value)

    return {
        "fig1e_curve_points": int(xs.size),
        "fig1e_reference_curve_points": int(ref_xs.size),
        "fig1e_rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        "fig1e_mean_abs": float(np.mean(np.abs(diff))),
        "fig1e_max_abs": float(np.max(np.abs(diff))),
        "fig1e_peak_abs_diff": float(abs(float(ys[peak_idx]) - float(ref_ys[ref_peak_idx]))),
        "fig1e_q_peak_abs_diff_nm_inv": float(abs(float(xs[peak_idx]) - float(ref_xs[ref_peak_idx]))),
        "fig1e_checkpoint_max_abs": float(max(checkpoint_diffs.values())),
        "fig1e_q04_abs_diff": float(checkpoint_diffs["fig1e_q04_abs_diff"]),
        "fig1e_q08_abs_diff": float(checkpoint_diffs["fig1e_q08_abs_diff"]),
        "fig1e_q12_abs_diff": float(checkpoint_diffs["fig1e_q12_abs_diff"]),
        "fig1e_min_bin_count": int(np.min(counts)) if counts.size else 0,
        "fig1e_reference_min_bin_count": int(np.min(ref_counts)) if ref_counts.size else 0,
    }


def compare_fig1e_window_to_paper_points(
    result: CRPAResult,
    reference_points: tuple[tuple[float, float], ...] | np.ndarray = DEFAULT_FIG1E_PAPER_POINTS,
    *,
    x_max_nm_inv: float = 1.2,
    bin_width_nm_inv: float = 0.0125,
) -> dict[str, float | int]:
    """Compare the representative epsilon curve to corrected Fig. 1(e) anchors."""

    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("reference_points must have shape (n, 2).")
    xs, ys, counts = representative_fig1e_window_curve(
        result,
        x_max_nm_inv=float(x_max_nm_inv),
        bin_width_nm_inv=float(bin_width_nm_inv),
    )
    if xs.size < 2:
        return {
            "fig1e_paper_reference_points": int(points.shape[0]),
            "fig1e_paper_curve_points": int(xs.size),
            "fig1e_paper_rmse": float("inf"),
            "fig1e_paper_mean_abs": float("inf"),
            "fig1e_paper_max_abs": float("inf"),
            "fig1e_paper_min_bin_count": 0,
        }
    q_ref = points[:, 0]
    eps_ref = points[:, 1]
    in_range = (q_ref >= float(np.min(xs))) & (q_ref <= float(np.max(xs)))
    if not np.any(in_range):
        return {
            "fig1e_paper_reference_points": int(points.shape[0]),
            "fig1e_paper_curve_points": int(xs.size),
            "fig1e_paper_rmse": float("inf"),
            "fig1e_paper_mean_abs": float("inf"),
            "fig1e_paper_max_abs": float("inf"),
            "fig1e_paper_min_bin_count": int(np.min(counts)) if counts.size else 0,
        }
    computed = np.interp(q_ref[in_range], xs, ys)
    diff = computed - eps_ref[in_range]
    out: dict[str, float | int] = {
        "fig1e_paper_reference_points": int(points.shape[0]),
        "fig1e_paper_points_in_range": int(np.count_nonzero(in_range)),
        "fig1e_paper_curve_points": int(xs.size),
        "fig1e_paper_rmse": float(np.sqrt(np.mean(diff * diff))),
        "fig1e_paper_mean_abs": float(np.mean(np.abs(diff))),
        "fig1e_paper_max_abs": float(np.max(np.abs(diff))),
        "fig1e_paper_min_bin_count": int(np.min(counts)) if counts.size else 0,
    }
    for q_value, computed_value, reference_value in zip(q_ref[in_range], computed, eps_ref[in_range], strict=True):
        key = f"fig1e_paper_q{int(round(float(q_value) * 1000)):04d}_diff"
        out[key] = float(computed_value - reference_value)
    return out


def fig1e_gate_failures(
    comparison: dict[str, float | int],
    *,
    max_rmse: float = 2.5,
    max_abs: float = 8.0,
    max_peak_abs: float = 4.0,
    max_q_peak_abs_nm_inv: float = 0.08,
    max_checkpoint_abs: float = 5.0,
    min_points: int = 20,
) -> list[str]:
    failures: list[str] = []
    if int(comparison.get("fig1e_curve_points", 0)) < int(min_points):
        failures.append(
            f"Fig. 1(e) curve has too few points: {comparison.get('fig1e_curve_points')} < {int(min_points)}"
        )
    checks = (
        ("fig1e_rmse", float(max_rmse)),
        ("fig1e_max_abs", float(max_abs)),
        ("fig1e_peak_abs_diff", float(max_peak_abs)),
        ("fig1e_q_peak_abs_diff_nm_inv", float(max_q_peak_abs_nm_inv)),
        ("fig1e_checkpoint_max_abs", float(max_checkpoint_abs)),
    )
    for key, limit in checks:
        value = float(comparison.get(key, float("inf")))
        if not np.isfinite(value) or value > limit:
            failures.append(f"{key}={value:.6g} exceeds {limit:.6g}")
    return failures


def fig1e_paper_point_gate_failures(
    comparison: dict[str, float | int],
    *,
    max_rmse: float = 0.8,
    max_abs: float = 1.5,
    max_mean_abs: float = 0.7,
    min_points: int = 5,
) -> list[str]:
    failures: list[str] = []
    if int(comparison.get("fig1e_paper_points_in_range", 0)) < int(min_points):
        failures.append(
            "Fig. 1(e) paper-point comparison has too few in-range anchors: "
            f"{comparison.get('fig1e_paper_points_in_range')} < {int(min_points)}"
        )
    checks = (
        ("fig1e_paper_rmse", float(max_rmse)),
        ("fig1e_paper_max_abs", float(max_abs)),
        ("fig1e_paper_mean_abs", float(max_mean_abs)),
    )
    for key, limit in checks:
        value = float(comparison.get(key, float("inf")))
        if not np.isfinite(value) or value > limit:
            failures.append(f"{key}={value:.6g} exceeds {limit:.6g}")
    return failures


def write_validation_report(
    result: CRPAResult,
    output_path: Path | str,
    *,
    extra_checks: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = validation_summary(result)
    lines = [
        "# cRPA validation report",
        "",
        "## Parameters",
        "",
        f"- theta_deg: {result.theta_deg:.12g}",
        f"- lk: {result.lk}",
        f"- lg: {result.lg}",
        f"- q_lg: {result.q_lg}",
        f"- epsilon_bn: {result.coulomb_params.epsilon_bn:.12g}",
        f"- ds_angstrom: {result.coulomb_params.ds_angstrom:.12g}",
        f"- eta_mev: {result.eta_mev:.12g}",
        f"- bands_per_valley: {result.bands_per_valley}",
    ]
    if result.metadata:
        lines.extend(["", "## Convention Metadata", ""])
        for key in sorted(result.metadata):
            lines.append(f"- {key}: {result.metadata[key]}")
    lines.extend(["", "## Checks", ""])
    for key, value in summary.items():
        lines.append(f"- {key}: {value:.16g}")
    if extra_checks:
        for key, value in extra_checks.items():
            if isinstance(value, (int, float, np.floating)):
                lines.append(f"- {key}: {float(value):.16g}")
            else:
                lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Status",
            "",
            "- C1 restricted-sum cross-check: "
            + ("see checks above." if extra_checks and "c1_direct_minus_subtraction_max_abs" in extra_checks else "not run in this report."),
            "- CP-cRPA1 Fig. 1(e): compare `epsilon_vs_q.pdf` against the paper target.",
            "- CP-cRPA2/3 HF+cRPA: requires feeding this screening table into HF runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compute_c1_cross_check(
    params: TBGParameters,
    *,
    lk: int,
    lg: int,
    q_lg: int,
    bands_per_valley: int | None,
    q_index: tuple[int, int] = (1, 0),
    eta_mev: float = 1.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    form_factor_mode: str = "zhang_zero_fill",
    occupation_mode: str = "cnp_index",
) -> dict[str, float | int | list[int]]:
    """Check constrained direct summing against full minus flat-flat."""

    grid = build_uniform_crpa_grid(params, lk)
    solution = solve_all_band_bm_model(
        params,
        grid.kvec,
        lg=lg,
        bands_per_valley=bands_per_valley,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )
    classification = classify_flat_bands(solution.spectrum, method="center")
    q_shifts, _ = build_q_shift_table(q_lg)
    q_index = (int(q_index[0]) % grid.lk, int(q_index[1]) % grid.lk)
    direct = compute_constrained_chi0(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        eta_mev=eta_mev,
        form_factor_mode=form_factor_mode,
        occupation_mode=occupation_mode,
    )
    subtraction = compute_constrained_chi0_by_subtraction(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        eta_mev=eta_mev,
        form_factor_mode=form_factor_mode,
        occupation_mode=occupation_mode,
    )
    diff = direct - subtraction
    return {
        "c1_q_index": [int(q_index[0]), int(q_index[1])],
        "c1_direct_minus_subtraction_max_abs": float(np.max(np.abs(diff))),
        "c1_direct_norm_inf": float(np.max(np.abs(direct))),
        "c1_subtraction_norm_inf": float(np.max(np.abs(subtraction))),
        "c1_lk": int(lk),
        "c1_lg": int(lg),
        "c1_q_lg": int(q_lg),
        "c1_bands_per_valley": -1 if bands_per_valley is None else int(bands_per_valley),
        "c1_periodic_g_grid": int(bool(periodic_g_grid)),
        "c1_form_factor_mode": str(form_factor_mode),
        "c1_occupation_mode": str(occupation_mode),
    }
