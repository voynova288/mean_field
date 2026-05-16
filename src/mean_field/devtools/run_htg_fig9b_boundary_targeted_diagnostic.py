#!/usr/bin/env python3
"""Targeted HTG Fig. 9b D3A/D3B boundary diagnostic.

This script intentionally does not define or alter the global Fig. 9b plotting
mesh.  It probes only the top-left D3A/D3B boundary cells with explicit
strong-coupling projectors and all four choices of the partially filled flavor.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Iterable

import numpy as np

from mean_field.core.hf import compute_hf_energy, run_projected_hartree_fock
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.htg import (
    HTGDensityBuilder,
    HTGInitializer,
    HTGModel,
    HTGHartreeFockRun,
    HTGHartreeFockState,
    HTGParams,
    InteractionParams,
    KWAN_2023_FERMI_VELOCITY_M_PER_S,
    KWAN_2023_TUNNELING_EV,
    build_htg_overlap_blocks,
    build_htg_projected_basis,
    classify_htg_strong_coupling_state,
)
from mean_field.systems.htg.mean_field_adapter import (
    _apply_random_rotation,
    _central_projected_band_indices,
    _htg_reference_density_blocks,
    _update_htg_hf_density_update_state,
    _update_htg_hf_step_state,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "HTG"
DEFAULT_THETA_VALUES = (1.60, 1.65, 1.70, 1.75)
DEFAULT_WAA_VALUES = (75.0, 80.0, 85.0, 90.0)
PHASES = ("D3A", "D3B")


@dataclass(frozen=True)
class BoundaryCase:
    index: int
    theta_deg: float
    wAA_mev: float
    label: str


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"htg_fig9b_d3_boundary_targeted_{job_id}"
    else:
        stem = f"htg_fig9b_d3_boundary_targeted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _cases(theta_values: Iterable[float], waa_values: Iterable[float]) -> tuple[BoundaryCase, ...]:
    cases: list[BoundaryCase] = []
    index = 0
    for waa in waa_values:
        for theta in theta_values:
            cases.append(
                BoundaryCase(
                    index=index,
                    theta_deg=float(theta),
                    wAA_mev=float(waa),
                    label=f"theta{float(theta):.2f}_wAA{float(waa):.1f}",
                )
            )
            index += 1
    return tuple(cases)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-prefix", default="fig9b_d3_boundary_targeted")
    parser.add_argument("--theta-values", type=_parse_csv_floats, default=DEFAULT_THETA_VALUES)
    parser.add_argument("--wAA-values", type=_parse_csv_floats, default=DEFAULT_WAA_VALUES)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--w-ev", type=float, default=KWAN_2023_TUNNELING_EV)
    parser.add_argument("--fermi-velocity-m-per-s", type=float, default=KWAN_2023_FERMI_VELOCITY_M_PER_S)
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--nu", type=float, default=3.0)
    parser.add_argument("--epsilon-r", type=float, default=8.0)
    parser.add_argument("--d-sc-nm", type=float, default=25.0)
    parser.add_argument("--u-ev", type=float, default=0.0)
    parser.add_argument("--n-k", type=int, default=12)
    parser.add_argument("--g-shells", type=int, default=1)
    parser.add_argument("--projected-band-count", type=int, default=2)
    parser.add_argument("--finite-zero-limit", action="store_true")
    parser.add_argument("--zero-cutoff-nm-inv", type=float, default=1.0e-12)
    parser.add_argument(
        "--seeds-per-flavor-class",
        type=int,
        default=76,
        help=(
            "Number of explicit seeds for each requested class and each partial-flavor permutation. "
            "The default gives 2 * 4 * 76 = 608 HF candidates per parameter point."
        ),
    )
    parser.add_argument("--perturbation-alpha", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--oda-stall-threshold", type=float, default=0.0)
    parser.add_argument("--ambiguity-threshold-mev", type=float, default=0.10)
    parser.add_argument("--warning-threshold-mev", type=float, default=0.05)
    parser.add_argument("--disable-numba", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.16g}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _format_csv_value(row.get(name)) for name in fieldnames})


def _compact_class_label(label: object) -> str:
    compact = str(label).strip().strip("[]").replace(" ", "")
    return compact or str(label)


def _run_final_error(run: HTGHartreeFockRun) -> float | None:
    if run.iter_err.size == 0:
        return None
    return float(run.iter_err[-1])


def _run_energy(run: HTGHartreeFockRun) -> float:
    return float(run.state.diagnostics.get("hf_energy", np.nan))


def _run_gap_mev(run: HTGHartreeFockRun) -> float:
    return 1000.0 * float(run.state.diagnostics.get("hf_gap", np.nan))


def _lower_remote_count(n_band: int) -> int:
    n_band = int(n_band)
    if n_band < 2 or n_band % 2 != 0:
        raise ValueError(f"projected band count must be an even integer >=2, got {n_band}")
    return (n_band - 2) // 2


def _conduction_bandwidth_mev(run: HTGHartreeFockRun) -> tuple[float | None, int | None]:
    counts = run.state.occupation_counts
    if counts is None:
        return None, None
    n_spin = int(run.state.n_spin)
    n_eta = int(run.state.n_eta)
    n_band = int(run.state.n_band)
    single_central_count = _lower_remote_count(n_band) + 1
    flavor_indices = [index for index, count in enumerate(counts) if int(count) == single_central_count]
    if len(flavor_indices) != 1:
        return None, None
    flavor_index = int(flavor_indices[0])
    ispin, ieta = np.unravel_index(flavor_index, (n_spin, n_eta), order="C")
    idx = np.arange(n_spin * n_eta * n_band, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    local_band = int(counts[flavor_index])
    if local_band < 0 or local_band >= n_band:
        return None, flavor_index
    band_index = int(idx[int(ispin), int(ieta), local_band])
    band = np.asarray(run.state.energies[band_index, :], dtype=float)
    return 1000.0 * float(np.max(band) - np.min(band)), flavor_index


def _flavor_label(flavor_index: int, *, n_spin: int, n_eta: int) -> str:
    ispin, ieta = np.unravel_index(int(flavor_index), (int(n_spin), int(n_eta)), order="C")
    return f"spin{int(ispin)}_eta{int(ieta)}"


def _strong_coupling_density(
    basis_data,
    *,
    phase: str,
    partial_flavor_index: int,
    perturbation_seed: int | None,
    perturbation_alpha: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    phase = str(phase).upper()
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase}")
    n_spin = int(basis_data.basis.n_spin)
    n_eta = int(basis_data.basis.n_flavor)
    n_band = int(basis_data.basis.n_band)
    n_flavor = n_spin * n_eta
    if not 0 <= int(partial_flavor_index) < n_flavor:
        raise ValueError(f"partial flavor index must be in [0, {n_flavor}), got {partial_flavor_index}")

    central_a, central_b = _central_projected_band_indices(n_band)
    lower_count = _lower_remote_count(n_band)
    idx = np.arange(basis_data.nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    projector = np.zeros((basis_data.nt, basis_data.nt, basis_data.nk), dtype=np.complex128)
    occupation_counts: list[int] = []
    partial_band = central_a if phase == "D3A" else central_b

    for ispin in range(n_spin):
        for ieta in range(n_eta):
            flavor_index = int(np.ravel_multi_index((ispin, ieta), (n_spin, n_eta), order="C"))
            local_bands = list(range(lower_count))
            if flavor_index == int(partial_flavor_index):
                local_bands.append(int(partial_band))
            else:
                local_bands.extend([int(central_a), int(central_b)])
            occupation_counts.append(len(local_bands))
            for iband in local_bands:
                state_index = int(idx[ispin, ieta, int(iband)])
                projector[state_index, state_index, :] = 1.0

    reference = _htg_reference_density_blocks(basis_data.nt, basis_data.nk, n_spin=n_spin, n_eta=n_eta)
    density = projector - reference
    if perturbation_seed is not None and float(perturbation_alpha) > 0.0:
        _apply_random_rotation(
            density,
            reference_density=reference,
            alpha=float(perturbation_alpha),
            seed=int(perturbation_seed),
        )
    return density, tuple(int(value) for value in occupation_counts)


def _actual_seed(case: BoundaryCase, *, phase: str, partial_flavor_index: int, seed_index: int) -> int:
    phase_offset = 0 if phase == "D3A" else 100_000_000
    theta_code = int(round(1000.0 * float(case.theta_deg)))
    waa_code = int(round(10.0 * float(case.wAA_mev)))
    return int(phase_offset + case.index * 100_000 + theta_code * 100 + waa_code + partial_flavor_index * 1000 + seed_index)


def _run_explicit_seed(
    *,
    case: BoundaryCase,
    basis_data,
    overlap_blocks,
    phase: str,
    partial_flavor_index: int,
    seed_index: int,
    args: argparse.Namespace,
) -> HTGHartreeFockRun:
    seed = _actual_seed(case, phase=phase, partial_flavor_index=partial_flavor_index, seed_index=seed_index)
    perturb_seed = None if int(seed_index) == 0 else seed
    perturb_alpha = 0.0 if int(seed_index) == 0 else float(args.perturbation_alpha)
    initial_density, occupation_counts = _strong_coupling_density(
        basis_data,
        phase=phase,
        partial_flavor_index=partial_flavor_index,
        perturbation_seed=perturb_seed,
        perturbation_alpha=perturb_alpha,
    )
    state = HTGHartreeFockState.from_projected_basis(
        basis_data,
        nu=float(args.nu),
        precision=float(args.precision),
        occupation_counts=occupation_counts,
    )
    init_mode = f"{phase.lower()}_partial{partial_flavor_index}_seed{seed_index}"
    base_run = run_projected_hartree_fock(
        state,
        initializer=HTGInitializer(initial_density=initial_density),
        density_builder=HTGDensityBuilder(
            float(args.nu),
            sigma_z=state.sigma_z,
            occupation_counts=occupation_counts,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        ),
        overlap_blocks=overlap_blocks,
        init_mode=init_mode,
        seed=seed,
        v0=state.v0,
        beta=float(args.beta),
        energy_functional=compute_hf_energy,
        oda_parameterizer="default",
        step_callback=_update_htg_hf_step_state,
        final_state_callback=_update_htg_hf_density_update_state,
        convergence_rule="raw",
        max_iter=int(args.max_iter),
        oda_stall_threshold=float(args.oda_stall_threshold),
        use_numba=False if bool(args.disable_numba) else None,
    )
    return HTGHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def _run_payload(
    *,
    case: BoundaryCase,
    run: HTGHartreeFockRun,
    requested_class: str,
    partial_flavor_index: int,
    seed_index: int,
) -> dict[str, object]:
    classification = classify_htg_strong_coupling_state(
        run.state.density,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
    ).to_dict()
    class_label = str(classification["class_label"])
    wcond, wcond_flavor = _conduction_bandwidth_mev(run)
    counts = run.state.occupation_counts
    return {
        "case_label": case.label,
        "theta_deg": float(case.theta_deg),
        "wAA_meV": float(case.wAA_mev),
        "requested_class": requested_class,
        "partial_flavor_index": int(partial_flavor_index),
        "partial_flavor_label": _flavor_label(
            partial_flavor_index,
            n_spin=run.state.n_spin,
            n_eta=run.state.n_eta,
        ),
        "seed_index": int(seed_index),
        "seed": int(run.seed),
        "init_mode": run.init_mode,
        "converged": bool(run.converged),
        "exit_reason": run.exit_reason,
        "iterations": int(run.iterations),
        "final_error": _run_final_error(run),
        "final_energy_ev": _run_energy(run),
        "hf_gap_mev": _run_gap_mev(run),
        "class_label": class_label,
        "class_compact_label": _compact_class_label(class_label),
        "family": str(classification["family"]),
        "nu_z": float(classification["nu_z"]),
        "wcond_mev": wcond,
        "wcond_flavor_index": wcond_flavor,
        "occupied_flavor_counts": "" if counts is None else str([int(value) for value in counts]),
    }


def _finite_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _best_for_final_class(rows: list[dict[str, object]], final_class: str) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("class_compact_label")) == final_class and _finite_float(row, "final_energy_ev") is not None
    ]
    if not candidates:
        return None
    converged = [row for row in candidates if bool(row.get("converged"))]
    pool = converged if converged else candidates
    return min(pool, key=lambda row: float(row["final_energy_ev"]))


def _summary_for_case(
    *,
    case: BoundaryCase,
    rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    best_a = _best_for_final_class(rows, "D3A")
    best_b = _best_for_final_class(rows, "D3B")
    energy_a = _finite_float(best_a, "final_energy_ev") if best_a is not None else None
    energy_b = _finite_float(best_b, "final_energy_ev") if best_b is not None else None
    delta_mev = None if energy_a is None or energy_b is None else 1000.0 * (float(energy_a) - float(energy_b))

    if delta_mev is None:
        chosen = "insufficient_D3A_or_D3B"
        lower_class = ""
        ambiguous = ""
    else:
        if abs(delta_mev) <= float(args.ambiguity_threshold_mev):
            if delta_mev < 0.0:
                chosen = "AMBIGUOUS_D3A_LOWER"
                lower_class = "D3A"
            elif delta_mev > 0.0:
                chosen = "AMBIGUOUS_D3B_LOWER"
                lower_class = "D3B"
            else:
                chosen = "AMBIGUOUS_DEGENERATE"
                lower_class = "degenerate"
            ambiguous = "true"
        elif delta_mev < 0.0:
            chosen = "D3A"
            lower_class = "D3A"
            ambiguous = "false"
        else:
            chosen = "D3B"
            lower_class = "D3B"
            ambiguous = "false"

    err_a = _finite_float(best_a, "final_error") if best_a is not None else None
    err_b = _finite_float(best_b, "final_error") if best_b is not None else None
    residuals = [value for value in (err_a, err_b) if value is not None]
    convergence_residual = max(residuals) if residuals else None
    n_requested_a = sum(1 for row in rows if row.get("requested_class") == "D3A")
    n_requested_b = sum(1 for row in rows if row.get("requested_class") == "D3B")
    n_final_a = sum(1 for row in rows if row.get("class_compact_label") == "D3A")
    n_final_b = sum(1 for row in rows if row.get("class_compact_label") == "D3B")

    return {
        "theta_deg": float(case.theta_deg),
        "wAA_meV": float(case.wAA_mev),
        "E_D3A_min": energy_a,
        "E_D3B_min": energy_b,
        "DeltaE": delta_mev,
        "chosen_class": chosen,
        "Wcond_D3A": _finite_float(best_a, "wcond_mev") if best_a is not None else None,
        "Wcond_D3B": _finite_float(best_b, "wcond_mev") if best_b is not None else None,
        "HF_gap_D3A": _finite_float(best_a, "hf_gap_mev") if best_a is not None else None,
        "HF_gap_D3B": _finite_float(best_b, "hf_gap_mev") if best_b is not None else None,
        "seed_count": int(len(rows)),
        "convergence_residual": convergence_residual,
        "case_label": case.label,
        "lower_energy_class": lower_class,
        "ambiguous": ambiguous,
        "ambiguity_threshold_meV": float(args.ambiguity_threshold_mev),
        "warning_threshold_meV": float(args.warning_threshold_mev),
        "near_warning_threshold": "" if delta_mev is None else str(abs(delta_mev) <= float(args.warning_threshold_mev)).lower(),
        "D3A_requested_seed_count": int(n_requested_a),
        "D3B_requested_seed_count": int(n_requested_b),
        "D3A_final_class_count": int(n_final_a),
        "D3B_final_class_count": int(n_final_b),
        "D3A_best_requested_class": "" if best_a is None else best_a.get("requested_class", ""),
        "D3B_best_requested_class": "" if best_b is None else best_b.get("requested_class", ""),
        "D3A_best_partial_flavor": "" if best_a is None else best_a.get("partial_flavor_label", ""),
        "D3B_best_partial_flavor": "" if best_b is None else best_b.get("partial_flavor_label", ""),
        "D3A_best_seed_index": "" if best_a is None else best_a.get("seed_index", ""),
        "D3B_best_seed_index": "" if best_b is None else best_b.get("seed_index", ""),
        "D3A_convergence_residual": err_a,
        "D3B_convergence_residual": err_b,
        "D3A_converged": "" if best_a is None else bool(best_a.get("converged")),
        "D3B_converged": "" if best_b is None else bool(best_b.get("converged")),
    }


SUMMARY_FIELDS = (
    "theta_deg",
    "wAA_meV",
    "E_D3A_min",
    "E_D3B_min",
    "DeltaE",
    "chosen_class",
    "Wcond_D3A",
    "Wcond_D3B",
    "HF_gap_D3A",
    "HF_gap_D3B",
    "seed_count",
    "convergence_residual",
    "case_label",
    "lower_energy_class",
    "ambiguous",
    "ambiguity_threshold_meV",
    "warning_threshold_meV",
    "near_warning_threshold",
    "D3A_requested_seed_count",
    "D3B_requested_seed_count",
    "D3A_final_class_count",
    "D3B_final_class_count",
    "D3A_best_requested_class",
    "D3B_best_requested_class",
    "D3A_best_partial_flavor",
    "D3B_best_partial_flavor",
    "D3A_best_seed_index",
    "D3B_best_seed_index",
    "D3A_convergence_residual",
    "D3B_convergence_residual",
    "D3A_converged",
    "D3B_converged",
)


DETAIL_FIELDS = (
    "case_label",
    "theta_deg",
    "wAA_meV",
    "requested_class",
    "partial_flavor_index",
    "partial_flavor_label",
    "seed_index",
    "seed",
    "init_mode",
    "converged",
    "exit_reason",
    "iterations",
    "final_error",
    "final_energy_ev",
    "hf_gap_mev",
    "class_label",
    "class_compact_label",
    "family",
    "nu_z",
    "wcond_mev",
    "wcond_flavor_index",
    "occupied_flavor_counts",
)


def _metadata(
    *,
    cases: tuple[BoundaryCase, ...],
    args: argparse.Namespace,
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    n_flavor_permutations = 4
    seed_count = len(PHASES) * n_flavor_permutations * int(args.seeds_per_flavor_class)
    return {
        "figure": "Kwan Fig. 9(b)",
        "diagnostic": "targeted D3A/D3B top-left boundary energy comparison",
        "global_mesh_status": "unchanged; this script only probes the requested 4x4 diagnostic window",
        "nu": float(args.nu),
        "theta_values_deg": sorted({float(case.theta_deg) for case in cases}),
        "wAA_values_meV": sorted({float(case.wAA_mev) for case in cases}),
        "n_parameter_points": int(len(cases)),
        "kwan_parameters": {
            "vF_m_per_s": float(args.fermi_velocity_m_per_s),
            "wAB_meV": 1000.0 * float(args.w_ev),
            "wAA_meV": "targeted scan",
            "epsilon_r": float(args.epsilon_r),
            "d_sc_nm": float(args.d_sc_nm),
            "U_ev": float(args.u_ev),
            "interaction_scheme": "average",
            "system_size_for_phase_map": f"{int(args.n_k)}x{int(args.n_k)}",
            "drop_q0_coulomb": bool(not args.finite_zero_limit),
            "zero_cutoff_nm_inv": float(args.zero_cutoff_nm_inv),
        },
        "seed_protocol": {
            "classes": list(PHASES),
            "D3A_definition": "three fully filled flavors plus one A-only occupied flavor",
            "D3B_definition": "three fully filled flavors plus one B-only occupied flavor",
            "partial_flavor_permutations": n_flavor_permutations,
            "seeds_per_flavor_class": int(args.seeds_per_flavor_class),
            "hf_candidates_per_parameter": seed_count,
            "candidate_count_formula": "2 requested classes * 4 partial-flavor permutations * seeds_per_flavor_class",
            "base_seed": "seed_index=0 is the unperturbed strong-coupling projector",
            "perturbed_seeds": "seed_index>0 applies a small random unitary perturbation to the explicit projector",
            "perturbation_alpha": float(args.perturbation_alpha),
        },
        "energy_columns": {
            "E_D3A_min": "eV per moire cell",
            "E_D3B_min": "eV per moire cell",
            "DeltaE": "meV per moire cell, E_D3A_min - E_D3B_min",
            "Wcond": "meV",
            "HF_gap": "meV",
        },
        "boundary_policy": {
            "manual_relabeling": False,
            "chosen_class": "lowest converged final D3A/D3B class unless abs(DeltaE) is below ambiguity_threshold_meV",
            "ambiguity_threshold_meV": float(args.ambiguity_threshold_mev),
            "warning_threshold_meV": float(args.warning_threshold_mev),
        },
        "translation_symmetry": "primitive-cell translation-invariant HF scan; no doubled/tripled TSB sectors",
        "initial_state_scope": "explicit primitive-cell D3A/D3B strong-coupling seeds and their perturbations",
        "runtime": runtime or {},
    }


def _write_report(path: Path, summary_rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    lines = [
        "# HTG Fig. 9b Targeted D3A/D3B Boundary Diagnostic",
        "",
        "This diagnostic keeps the accepted global 8x10 Fig. 9b mesh unchanged and probes only the requested top-left 4x4 window.",
        "",
        "## Protocol",
        "",
        f"- HF mesh: {metadata['kwan_parameters']['system_size_for_phase_map']}",
        f"- epsilon_r: {metadata['kwan_parameters']['epsilon_r']}",
        f"- d_sc_nm: {metadata['kwan_parameters']['d_sc_nm']}",
        f"- wAB_meV: {metadata['kwan_parameters']['wAB_meV']}",
        f"- vF_m_per_s: {metadata['kwan_parameters']['vF_m_per_s']}",
        f"- U_ev: {metadata['kwan_parameters']['U_ev']}",
        f"- candidates per parameter: {metadata['seed_protocol']['hf_candidates_per_parameter']}",
        f"- ambiguity threshold: {metadata['boundary_policy']['ambiguity_threshold_meV']} meV per moire cell",
        "",
        "## Summary",
        "",
        "| theta | wAA | E_D3A eV | E_D3B eV | DeltaE meV | chosen | Wcond D3A | Wcond D3B | gap D3A | gap D3B | residual |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (float(item["wAA_meV"]), float(item["theta_deg"]))):
        lines.append(
            "| {theta:.6g} | {waa:.6g} | {ea} | {eb} | {de} | {chosen} | {wa} | {wb} | {ga} | {gb} | {res} |".format(
                theta=float(row["theta_deg"]),
                waa=float(row["wAA_meV"]),
                ea=_format_csv_value(row.get("E_D3A_min")),
                eb=_format_csv_value(row.get("E_D3B_min")),
                de=_format_csv_value(row.get("DeltaE")),
                chosen=row.get("chosen_class", ""),
                wa=_format_csv_value(row.get("Wcond_D3A")),
                wb=_format_csv_value(row.get("Wcond_D3B")),
                ga=_format_csv_value(row.get("HF_gap_D3A")),
                gb=_format_csv_value(row.get("HF_gap_D3B")),
                res=_format_csv_value(row.get("convergence_residual")),
            )
        )
    lines.extend(
        [
            "",
            "Cells with `AMBIGUOUS_*` labels are not manually relabeled; they indicate that the computed D3A/D3B energy splitting is below the configured threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    artifact_prefix: str,
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{artifact_prefix}.csv"
    detail_path = output_dir / f"{artifact_prefix}_run_details.csv"
    metadata_path = output_dir / "grid_metadata.json"
    json_path = output_dir / f"{artifact_prefix}.json"
    report_path = output_dir / f"{artifact_prefix}_report.md"
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(detail_path, detail_rows, DETAIL_FIELDS)
    write_json(metadata_path, metadata)
    write_json(
        json_path,
        {
            "artifacts": {
                "csv": str(summary_path),
                "run_details_csv": str(detail_path),
                "grid_metadata": str(metadata_path),
                "report": str(report_path),
            },
            "grid_metadata": metadata,
            "rows": summary_rows,
            "run_details": detail_rows,
        },
    )
    _write_report(report_path, summary_rows, metadata)


def main() -> None:
    args = _parse_args()
    if int(args.shard_count) < 1:
        raise SystemExit("--shard-count must be positive")
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise SystemExit("--shard-index must satisfy 0 <= index < count")
    if int(args.seeds_per_flavor_class) < 1:
        raise SystemExit("--seeds-per-flavor-class must be positive")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    artifact_prefix = str(args.artifact_prefix).strip()
    if not artifact_prefix:
        raise SystemExit("--artifact-prefix must not be empty")

    all_cases = _cases(args.theta_values, args.wAA_values)
    shard_cases = tuple(case for case in all_cases if int(case.index) % int(args.shard_count) == int(args.shard_index))
    candidates_per_parameter = len(PHASES) * 4 * int(args.seeds_per_flavor_class)
    if args.dry_run:
        print(f"output_dir={output_dir}")
        print(f"artifact_prefix={artifact_prefix}")
        print(f"shard={args.shard_index}/{args.shard_count}")
        print(f"cases={len(shard_cases)} total_cases={len(all_cases)}")
        print(f"hf_candidates_per_parameter={candidates_per_parameter}")
        for case in shard_cases:
            print(f"case={case.index}:{case.theta_deg}:{case.wAA_mev}:{case.label}")
        return

    ensure_not_running_compute_on_login_node("HTG Fig. 9b targeted D3A/D3B boundary diagnostic")
    if args.disable_numba:
        os.environ["MEAN_FIELD_HF_DISABLE_NUMBA"] = "1"

    print(
        f"[boundary] shard={args.shard_index}/{args.shard_count} cases={len(shard_cases)} "
        f"candidates_per_parameter={candidates_per_parameter}",
        flush=True,
    )
    start = perf_counter()
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for case in shard_cases:
        print(f"[case] index={case.index} theta={case.theta_deg:g} wAA_meV={case.wAA_mev:g}", flush=True)
        w_aa_ev = float(case.wAA_mev) / 1000.0
        kappa = float(w_aa_ev / float(args.w_ev))
        params = HTGParams(
            fermi_velocity_m_per_s=float(args.fermi_velocity_m_per_s),
            w_ev=float(args.w_ev),
            kappa=kappa,
            zeta_rad=0.0,
            model_name="kwan2023_hf",
        )
        interaction = InteractionParams(
            epsilon_r=float(args.epsilon_r),
            d_sc_nm=float(args.d_sc_nm),
            U_ev=float(args.u_ev),
            n_k=int(args.n_k),
            g_shells=int(args.g_shells),
            finite_zero_limit=bool(args.finite_zero_limit),
            zero_cutoff_nm_inv=float(args.zero_cutoff_nm_inv),
        )
        model = HTGModel.from_config(case.theta_deg, n_shells=int(args.n_shells), params=params)
        basis_data = build_htg_projected_basis(
            model,
            interaction,
            mesh_size=int(args.n_k),
            projected_band_count=int(args.projected_band_count),
        )
        overlap_blocks = build_htg_overlap_blocks(basis_data, g_shells=int(args.g_shells))
        case_rows: list[dict[str, object]] = []
        n_flavors = int(basis_data.basis.n_spin) * int(basis_data.basis.n_flavor)
        for phase in PHASES:
            for partial_flavor_index in range(n_flavors):
                for seed_index in range(int(args.seeds_per_flavor_class)):
                    run = _run_explicit_seed(
                        case=case,
                        basis_data=basis_data,
                        overlap_blocks=overlap_blocks,
                        phase=phase,
                        partial_flavor_index=partial_flavor_index,
                        seed_index=seed_index,
                        args=args,
                    )
                    row = _run_payload(
                        case=case,
                        run=run,
                        requested_class=phase,
                        partial_flavor_index=partial_flavor_index,
                        seed_index=seed_index,
                    )
                    case_rows.append(row)
        detail_rows.extend(case_rows)
        summary = _summary_for_case(case=case, rows=case_rows, args=args)
        summary_rows.append(summary)
        print(
            "[case-done] theta={theta:g} wAA={waa:g} chosen={chosen} DeltaE_meV={delta}".format(
                theta=case.theta_deg,
                waa=case.wAA_mev,
                chosen=summary["chosen_class"],
                delta=_format_csv_value(summary.get("DeltaE")),
            ),
            flush=True,
        )
        runtime = {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            "elapsed_sec": float(perf_counter() - start),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
        }
        metadata = _metadata(cases=shard_cases, args=args, runtime=runtime)
        _write_outputs(
            output_dir=output_dir,
            artifact_prefix=artifact_prefix,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata=metadata,
        )

    runtime = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "elapsed_sec": float(perf_counter() - start),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    metadata = _metadata(cases=shard_cases, args=args, runtime=runtime)
    _write_outputs(
        output_dir=output_dir,
        artifact_prefix=artifact_prefix,
        summary_rows=summary_rows,
        detail_rows=detail_rows,
        metadata=metadata,
    )
    print(f"summary_csv={output_dir / f'{artifact_prefix}.csv'}")
    print(f"run_details_csv={output_dir / f'{artifact_prefix}_run_details.csv'}")
    print(f"grid_metadata={output_dir / 'grid_metadata.json'}")


if __name__ == "__main__":
    main()
