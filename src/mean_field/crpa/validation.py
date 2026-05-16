from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..systems.tbg.params import TBGParameters
from .band_classifier import classify_flat_bands
from .bm import solve_all_band_bm_model
from .grid import build_q_shift_table, build_uniform_crpa_grid
from .susceptibility import compute_constrained_chi0, compute_constrained_chi0_by_subtraction
from .workflow import CRPAResult


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
