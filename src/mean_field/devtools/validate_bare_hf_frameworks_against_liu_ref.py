#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mean_field.core.hf import (
    build_flavor_band_data,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_norm_convergence,
    compute_oda_parameter,
)
from mean_field.crpa import (
    half_reference_delta_like,
    physical_projector_from_delta,
    run_bare_split_full_hartree_fock,
    split_oda_parameter,
)
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    build_b0_uniform_lattice,
    build_gamma_m_k_gamma_kprime_kpath,
    build_h0_from_bm,
    build_overlap_block_set,
    solve_bm_model,
)
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    build_full_density_from_hamiltonian,
    coulomb_unit,
    run_full_hartree_fock,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "benchmarks" / "Liu_reproduce_ref"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "TBG_HF_cRPA" / "hf_framework_validation"


@dataclass(frozen=True)
class ReferenceCase:
    case_dir: Path
    summary_row: dict[str, str]
    state_path: Path
    path_tsv: Path | None
    theta_deg: float
    nu: float
    init_mode: str
    seed: int
    lk: int
    lg: int
    overlap_lg: int
    w0_mev: float
    w1_mev: float
    vf_mev: float
    epsilon_r: float
    screening_lm: float
    tanh_argument_scale_a: float
    q_zero_limit: bool


def _scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _case_float(data, key: str, default: float | None = None) -> float:
    value = _scalar(data, key, default)
    if value is None:
        raise ValueError(f"Missing scalar {key}")
    return float(value)


def _case_int(data, key: str, default: int | None = None) -> int:
    value = _scalar(data, key, default)
    if value is None:
        raise ValueError(f"Missing scalar {key}")
    return int(value)


def _case_bool(data, key: str, default: bool = False) -> bool:
    value = _scalar(data, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _read_summary_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}, got {len(rows)}")
    return rows[0]


def _discover_cases(reference_root: Path, requested_nu: set[float] | None) -> list[ReferenceCase]:
    cases: list[ReferenceCase] = []
    for summary_path in sorted(reference_root.glob("*/summary.tsv")):
        case_dir = summary_path.parent
        row = _read_summary_row(summary_path)
        state_candidates = sorted((case_dir / "states").glob("*.npz"))
        if len(state_candidates) != 1:
            raise ValueError(f"Expected exactly one state file under {case_dir / 'states'}, got {len(state_candidates)}")
        path_candidates = sorted(
            p
            for p in (case_dir / "path_bands").glob("*_hf_path.tsv")
            if not p.name.endswith("_hf_path_nodes.tsv")
        )
        path_tsv = path_candidates[0] if path_candidates else None
        with np.load(state_candidates[0], allow_pickle=False) as data:
            nu = _case_float(data, "nu")
            if requested_nu is not None and not any(abs(nu - wanted) < 1.0e-9 for wanted in requested_nu):
                continue
            cases.append(
                ReferenceCase(
                    case_dir=case_dir,
                    summary_row=row,
                    state_path=state_candidates[0],
                    path_tsv=path_tsv,
                    theta_deg=_case_float(data, "theta_deg"),
                    nu=nu,
                    init_mode=str(_scalar(data, "init_mode", row.get("init_mode", ""))),
                    seed=_case_int(data, "seed", int(row.get("seed", "1"))),
                    lk=_case_int(data, "lk"),
                    lg=_case_int(data, "lg"),
                    overlap_lg=_case_int(data, "overlap_lg", _case_int(data, "lg")),
                    w0_mev=_case_float(data, "w0_mev", 79.7),
                    w1_mev=_case_float(data, "w1_mev", 97.4),
                    vf_mev=_case_float(data, "vf_mev", 2135.4),
                    epsilon_r=_case_float(data, "epsilon_r", 10.0),
                    screening_lm=_case_float(data, "screening_lm", 400.0 / 2.46 / 2.0),
                    tanh_argument_scale_a=_case_float(data, "tanh_argument_scale_a", 400.0 / 2.46),
                    q_zero_limit=_case_bool(data, "q_zero_limit", True),
                )
            )
    if not cases:
        raise ValueError(f"No Liu reference cases found under {reference_root}")
    return sorted(cases, key=lambda case: case.nu)


def _require_common_parameters(cases: list[ReferenceCase]) -> ReferenceCase:
    first = cases[0]
    fields = (
        "theta_deg",
        "lk",
        "lg",
        "overlap_lg",
        "w0_mev",
        "w1_mev",
        "vf_mev",
        "epsilon_r",
        "screening_lm",
        "tanh_argument_scale_a",
        "q_zero_limit",
    )
    for case in cases[1:]:
        for field in fields:
            lhs = getattr(first, field)
            rhs = getattr(case, field)
            if isinstance(lhs, float):
                if not math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(f"Reference cases do not share {field}: {lhs} vs {rhs}")
            elif lhs != rhs:
                raise ValueError(f"Reference cases do not share {field}: {lhs} vs {rhs}")
    return first


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))


def _path_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if len(header) < 2 or header[0] != "k_dist":
        raise ValueError(f"Unexpected HF path header in {path}: {header}")
    table = np.loadtxt(path, delimiter="\t", skiprows=1)
    if table.ndim == 1:
        table = table[None, :]
    return np.asarray(table[:, 0], dtype=float), np.asarray(table[:, 1:], dtype=float)


def _safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def _split_step_equivalence(
    *,
    density_initial: np.ndarray,
    h0: np.ndarray,
    sigma_z: np.ndarray,
    nu: float,
    overlap_blocks,
    v0: float,
    max_steps: int,
    use_numba: bool | None,
) -> dict[str, object]:
    density_wang = np.asarray(density_initial, dtype=np.complex128).copy()
    density_split = np.asarray(density_initial, dtype=np.complex128).copy()
    remote_bare = build_projected_interaction_hamiltonian(
        half_reference_delta_like(density_initial),
        overlap_blocks,
        v0=v0,
        use_numba=use_numba,
    )
    h0_split = np.asarray(h0, dtype=np.complex128) + remote_bare
    rows: list[dict[str, float | int]] = []

    for iteration in range(1, int(max_steps) + 1):
        wang_interaction = build_projected_interaction_hamiltonian(
            density_wang,
            overlap_blocks,
            v0=v0,
            use_numba=use_numba,
        )
        wang_hamiltonian = h0 + wang_interaction

        active_projector = physical_projector_from_delta(density_split)
        split_interaction = build_projected_interaction_hamiltonian(
            active_projector,
            overlap_blocks,
            v0=v0,
            use_numba=use_numba,
        )
        split_hamiltonian = h0_split + split_interaction

        density_update_wang, energies_wang, _sigma_wang, mu_wang = build_full_density_from_hamiltonian(
            wang_hamiltonian,
            sigma_z,
            nu=nu,
        )
        density_update_split, energies_split, _sigma_split, mu_split = build_full_density_from_hamiltonian(
            split_hamiltonian,
            sigma_z,
            nu=nu,
        )
        delta_wang = density_update_wang - density_wang
        delta_split = density_update_split - density_split
        delta_h_wang = build_projected_interaction_hamiltonian(
            delta_wang,
            overlap_blocks,
            v0=v0,
            use_numba=use_numba,
        )
        delta_h_split = build_projected_interaction_hamiltonian(
            delta_split,
            overlap_blocks,
            v0=v0,
            use_numba=use_numba,
        )

        state_wang = SimpleNamespace(
            h0=h0,
            density=density_wang,
            hamiltonian=wang_hamiltonian,
            nk=int(density_wang.shape[2]),
        )
        state_split = SimpleNamespace(
            h0=h0_split,
            density=density_split,
            hamiltonian=split_hamiltonian,
            nk=int(density_split.shape[2]),
        )
        lambda_wang = compute_oda_parameter(
            state_wang,
            delta_wang,
            delta_h=delta_h_wang,
            interaction_h=wang_interaction,
        )
        lambda_split = split_oda_parameter(
            state_split,
            delta_split,
            delta_h=delta_h_split,
            interaction_h=split_interaction,
        )
        mixed_wang = lambda_wang * density_update_wang + (1.0 - lambda_wang) * density_wang
        mixed_split = lambda_split * density_update_split + (1.0 - lambda_split) * density_split

        rows.append(
            {
                "iteration": iteration,
                "density_before_max_abs": _max_abs(density_wang, density_split),
                "hamiltonian_max_abs": _max_abs(wang_hamiltonian, split_hamiltonian),
                "raw_density_update_max_abs": _max_abs(density_update_wang, density_update_split),
                "energies_max_abs": _max_abs(energies_wang, energies_split),
                "mu_abs_diff": float(abs(float(mu_wang) - float(mu_split))),
                "delta_interaction_max_abs": _max_abs(delta_h_wang, delta_h_split),
                "oda_lambda_abs_diff": float(abs(float(lambda_wang) - float(lambda_split))),
                "mixed_density_max_abs": _max_abs(mixed_wang, mixed_split),
                "wang_oda": float(lambda_wang),
                "split_oda": float(lambda_split),
            }
        )
        density_wang = np.asarray(mixed_wang, dtype=np.complex128)
        density_split = np.asarray(mixed_split, dtype=np.complex128)

    return {
        "remote_bare_max_abs": float(np.max(np.abs(remote_bare))),
        "remote_bare_fro_norm": float(np.linalg.norm(remote_bare)),
        "final_density_frame_max_abs": _max_abs(density_wang, density_split),
        "steps": rows,
    }


def _split_target_hamiltonian(
    base_hamiltonian: np.ndarray,
    density_delta: np.ndarray,
    *,
    source_overlap_blocks,
    target_overlap_blocks,
    target_source_overlap_blocks,
    v0: float,
    use_numba: bool | None,
) -> np.ndarray:
    remote_target = build_projected_target_hamiltonian(
        np.zeros_like(base_hamiltonian),
        half_reference_delta_like(density_delta),
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=v0,
        use_numba=use_numba,
    )
    return build_projected_target_hamiltonian(
        base_hamiltonian + remote_target,
        physical_projector_from_delta(density_delta),
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=v0,
        use_numba=use_numba,
    )


def _trace_max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    if lhs.size == 0 and rhs.size == 0:
        return 0.0
    return _max_abs(lhs, rhs)


def _run_converged_equivalence(
    *,
    density_initial: np.ndarray,
    grid_solution,
    params: TBGParameters,
    nu: float,
    init_mode: str,
    seed: int,
    overlap_blocks,
    max_iter: int,
    precision: float,
    oda_stall_threshold: float,
    use_numba: bool | None,
    path=None,
    path_h0: np.ndarray | None = None,
    path_overlap=None,
    path_grid_overlap=None,
    reference_path_tsv: Path | None = None,
) -> dict[str, object]:
    wang_state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=nu, precision=precision)
    split_state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=nu, precision=precision)

    wang_run = run_full_hartree_fock(
        wang_state,
        overlap_blocks,
        grid_solution.lattice_kvec,
        params,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        initial_density=density_initial,
    )
    split_run = run_bare_split_full_hartree_fock(
        split_state,
        overlap_blocks,
        grid_solution.lattice_kvec,
        params,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        initial_density=density_initial,
        use_numba=use_numba,
    )

    wang_energy_trace = np.asarray(wang_run.iter_energy, dtype=float)
    split_energy_trace = np.asarray(split_run.iter_energy, dtype=float)
    common_energy_len = min(int(wang_energy_trace.size), int(split_energy_trace.size))
    energy_offset = split_energy_trace[:common_energy_len] - wang_energy_trace[:common_energy_len]
    if common_energy_len:
        energy_offset_spread = float(np.max(energy_offset) - np.min(energy_offset))
        energy_offset_first = float(energy_offset[0])
    else:
        energy_offset_spread = 0.0
        energy_offset_first = float("nan")

    path_payload: dict[str, float | str] | None = None
    if path is not None and path_h0 is not None and path_overlap is not None and path_grid_overlap is not None:
        h_path_wang = build_projected_target_hamiltonian(
            path_h0,
            wang_run.state.density,
            source_overlap_blocks=overlap_blocks,
            target_overlap_blocks=path_overlap,
            target_source_overlap_blocks=path_grid_overlap,
            v0=wang_run.state.v0,
            use_numba=use_numba,
        )
        h_path_split = _split_target_hamiltonian(
            path_h0,
            split_run.state.density,
            source_overlap_blocks=overlap_blocks,
            target_overlap_blocks=path_overlap,
            target_source_overlap_blocks=path_grid_overlap,
            v0=split_run.state.v0,
            use_numba=use_numba,
        )
        bands_wang = build_flavor_band_data(h_path_wang).energies.T
        bands_split = build_flavor_band_data(h_path_split).energies.T
        path_payload = {
            "wang_split_hamiltonian_max_abs": _max_abs(h_path_wang, h_path_split),
            "wang_split_sorted_band_max_abs": _max_abs(np.sort(bands_wang, axis=1), np.sort(bands_split, axis=1)),
            "wang_split_direct_band_max_abs": _max_abs(bands_wang, bands_split),
        }
        if reference_path_tsv is not None:
            ref_kdist, ref_bands = _path_table(reference_path_tsv)
            path_payload.update(
                {
                    "reference_path_tsv": str(reference_path_tsv),
                    "kdist_reference_max_abs": _max_abs(path.kdist, ref_kdist),
                    "wang_reference_sorted_band_max_abs": _max_abs(
                        np.sort(bands_wang, axis=1),
                        np.sort(ref_bands, axis=1),
                    ),
                    "split_reference_sorted_band_max_abs": _max_abs(
                        np.sort(bands_split, axis=1),
                        np.sort(ref_bands, axis=1),
                    ),
                }
            )

    payload: dict[str, object] = {
        "wang_iterations": int(wang_run.iterations),
        "split_iterations": int(split_run.iterations),
        "wang_exit_reason": str(wang_run.exit_reason),
        "split_exit_reason": str(split_run.exit_reason),
        "wang_converged": bool(wang_run.converged),
        "split_converged": bool(split_run.converged),
        "density_max_abs": _max_abs(wang_run.state.density, split_run.state.density),
        "hamiltonian_max_abs": _max_abs(wang_run.state.hamiltonian, split_run.state.hamiltonian),
        "energies_max_abs": _max_abs(wang_run.state.energies, split_run.state.energies),
        "mu_abs_diff": float(abs(float(wang_run.state.mu) - float(split_run.state.mu))),
        "iter_err_max_abs": _trace_max_abs(wang_run.iter_err, split_run.iter_err),
        "iter_oda_max_abs": _trace_max_abs(wang_run.iter_oda, split_run.iter_oda),
        "iter_trace_length_abs_diff": int(abs(int(wang_run.iterations) - int(split_run.iterations))),
        "energy_offset_first": energy_offset_first,
        "energy_offset_spread": energy_offset_spread,
        "wang_final_raw_norm": float(wang_run.state.diagnostics.get("final_raw_norm", float("nan"))),
        "split_final_raw_norm": float(split_run.state.diagnostics.get("final_raw_norm", float("nan"))),
    }
    if path_payload is not None:
        payload["path_bands"] = path_payload
    return payload


def _write_report(path: Path, payload: dict[str, object]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# Bare HF framework validation against Liu_reproduce_ref",
        "",
        f"status: {aggregate['status']}",
        f"reference_root: {payload['reference_root']}",
        f"case_count: {aggregate['case_count']}",
        "",
        "## Aggregate maxima",
        "",
    ]
    for key, value in aggregate["maxima"].items():
        lines.append(f"- {key}: {value:.16e}")
    lines.extend(["", "## Cases", ""])
    for case in payload["cases"]:
        lines.extend(
            [
                f"### nu={case['nu']:+.6g}",
                "",
                f"- state: {case['state_path']}",
                f"- Wang current vs Liu h0 max abs: {case['wang_reference']['h0_max_abs']:.16e}",
                f"- Wang current vs Liu Hamiltonian max abs: {case['wang_reference']['hamiltonian_max_abs']:.16e}",
                f"- Wang fixed-point raw norm: {case['wang_reference']['fixed_point_norm']:.16e}",
                f"- Zhang/Wang step Hamiltonian max abs: {case['zhang_wang_equivalence']['max_hamiltonian_abs']:.16e}",
                f"- Zhang/Wang step density max abs: {case['zhang_wang_equivalence']['max_raw_density_update_abs']:.16e}",
                f"- Zhang/Wang ODA lambda max abs: {case['zhang_wang_equivalence']['max_oda_lambda_abs']:.16e}",
            ]
        )
        path_metrics = case.get("path_bands")
        if path_metrics is not None:
            lines.extend(
                [
                    f"- Path k_dist max abs: {path_metrics['kdist_max_abs']:.16e}",
                    f"- Path sorted band max abs: {path_metrics['sorted_band_max_abs']:.16e}",
                ]
            )
        converged_metrics = case.get("converged_equivalence")
        if converged_metrics is not None:
            lines.extend(
                [
                    f"- Converged Wang iterations: {converged_metrics['wang_iterations']}",
                    f"- Converged Zhang split iterations: {converged_metrics['split_iterations']}",
                    f"- Converged Hamiltonian max abs: {converged_metrics['hamiltonian_max_abs']:.16e}",
                    f"- Converged density max abs: {converged_metrics['density_max_abs']:.16e}",
                    f"- Converged energies max abs: {converged_metrics['energies_max_abs']:.16e}",
                    f"- Converged ODA trace max abs: {converged_metrics['iter_oda_max_abs']:.16e}",
                ]
            )
            converged_path = converged_metrics.get("path_bands")
            if converged_path is not None:
                lines.append(
                    f"- Converged path sorted band max abs: {converged_path['wang_split_sorted_band_max_abs']:.16e}"
                )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the no-cRPA Wang/Xiaoyu framework against Liu_reproduce_ref and "
            "the no-cRPA Zhang bare-split framework against Wang step by step."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nu", type=float, action="append", default=None, help="Optional filling filter; repeatable.")
    parser.add_argument("--steps", type=int, default=3, help="Number of Zhang/Wang HF map steps to compare.")
    parser.add_argument("--check-path", action="store_true", help="Also rebuild and compare the saved HF path-band TSVs.")
    parser.add_argument(
        "--run-converged",
        action="store_true",
        help="Run both Wang no-cRPA and Zhang bare-split no-cRPA HF from each Liu reference density to convergence.",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--precision", type=float, default=1.0e-5)
    parser.add_argument("--oda-stall-threshold", type=float, default=1.0e-3)
    parser.add_argument("--path-kind", choices=("gamma-m-k-gamma-kprime",), default="gamma-m-k-gamma-kprime")
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=1.0e-7)
    parser.add_argument("--fixed-point-threshold", type=float, default=5.0e-5)
    parser.add_argument("--use-numba", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--allow-login", action="store_true", help="Bypass the login-node compute guard for tiny dry runs only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_login:
        ensure_not_running_compute_on_login_node("bare HF framework validation")

    requested_nu = None if args.nu is None else {float(value) for value in args.nu}
    cases = _discover_cases(args.reference_root, requested_nu)
    common = _require_common_parameters(cases)
    use_numba: bool | None
    if args.use_numba == "auto":
        use_numba = None
    else:
        use_numba = args.use_numba == "true"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = TBGParameters.from_degrees(
        common.theta_deg,
        vf=common.vf_mev,
        w0=common.w0_mev,
        w1=common.w1_mev,
    )
    grid = build_b0_uniform_lattice(params, common.lk)
    grid_solution = solve_bm_model(params, grid.kvec, lg=common.lg, sigma_rotation=True)
    grid_h0 = build_h0_from_bm(grid_solution)
    overlap_blocks = build_overlap_block_set(
        grid_solution,
        lg=common.overlap_lg,
        relative_permittivity=common.epsilon_r,
        screening_lm=common.screening_lm,
        finite_zero_limit=common.q_zero_limit,
    )
    v0 = coulomb_unit(params)

    path = None
    path_h0 = None
    path_overlap = None
    path_grid_overlap = None
    if args.check_path or args.run_converged:
        path = build_gamma_m_k_gamma_kprime_kpath(params, int(args.points_per_segment))
        path_solution = solve_bm_model(params, path.kvec, lg=common.lg, sigma_rotation=True)
        path_h0 = build_h0_from_bm(path_solution)
        path_overlap = build_overlap_block_set(
            path_solution,
            lg=common.overlap_lg,
            relative_permittivity=common.epsilon_r,
            screening_lm=common.screening_lm,
            finite_zero_limit=common.q_zero_limit,
        )
        path_grid_overlap = build_overlap_block_set(
            path_solution,
            source_solution=grid_solution,
            lg=common.overlap_lg,
            relative_permittivity=common.epsilon_r,
            screening_lm=common.screening_lm,
            finite_zero_limit=common.q_zero_limit,
        )

    case_payloads: list[dict[str, object]] = []
    maxima = {
        "wang_reference_h0_max_abs": 0.0,
        "wang_reference_sigma_z_max_abs": 0.0,
        "wang_reference_hamiltonian_max_abs": 0.0,
        "wang_reference_energies_max_abs": 0.0,
        "wang_reference_fixed_point_norm": 0.0,
        "zhang_wang_hamiltonian_max_abs": 0.0,
        "zhang_wang_raw_density_update_max_abs": 0.0,
        "zhang_wang_mixed_density_max_abs": 0.0,
        "zhang_wang_oda_lambda_abs": 0.0,
        "path_sorted_band_max_abs": 0.0,
        "converged_hamiltonian_max_abs": 0.0,
        "converged_density_max_abs": 0.0,
        "converged_energies_max_abs": 0.0,
        "converged_oda_trace_max_abs": 0.0,
        "converged_iter_trace_length_abs_diff": 0.0,
        "converged_path_sorted_band_max_abs": 0.0,
        "converged_wang_reference_path_sorted_band_max_abs": 0.0,
        "converged_split_reference_path_sorted_band_max_abs": 0.0,
    }

    for case in cases:
        print(f"[case] nu={case.nu:+.6g} state={case.state_path}", flush=True)
        with np.load(case.state_path, allow_pickle=False) as data:
            density_ref = np.asarray(data["density"], dtype=np.complex128)
            h0_ref = np.asarray(data["h0"], dtype=np.complex128)
            hamiltonian_ref = np.asarray(data["hamiltonian"], dtype=np.complex128)
            energies_ref = np.asarray(data["energies"], dtype=float)
            sigma_z_ref = np.asarray(data["sigma_z"], dtype=np.complex128)
            iter_err_ref = np.asarray(data["iter_err"], dtype=float)

        wang_interaction = build_projected_interaction_hamiltonian(
            density_ref,
            overlap_blocks,
            v0=v0,
            use_numba=use_numba,
        )
        wang_hamiltonian = grid_h0 + wang_interaction
        density_update, energies_update, _sigma_update, mu_update = build_full_density_from_hamiltonian(
            wang_hamiltonian,
            grid_solution.sigma_z,
            nu=case.nu,
        )
        fixed_point_norm = float(calculate_norm_convergence(density_update, density_ref))
        fixed_point_max_abs = _max_abs(density_update, density_ref)
        wang_reference = {
            "h0_max_abs": _max_abs(grid_h0, h0_ref),
            "sigma_z_max_abs": _max_abs(grid_solution.sigma_z, sigma_z_ref),
            "hamiltonian_max_abs": _max_abs(wang_hamiltonian, hamiltonian_ref),
            "energies_max_abs": _max_abs(energies_update, energies_ref),
            "fixed_point_norm": fixed_point_norm,
            "fixed_point_max_abs": fixed_point_max_abs,
            "mu_mev": float(mu_update),
            "summary_final_error": _safe_float(case.summary_row, "final_error"),
            "state_last_iter_error": float(iter_err_ref[-1]) if iter_err_ref.size else float("nan"),
        }

        equivalence = _split_step_equivalence(
            density_initial=density_ref,
            h0=grid_h0,
            sigma_z=grid_solution.sigma_z,
            nu=case.nu,
            overlap_blocks=overlap_blocks,
            v0=v0,
            max_steps=args.steps,
            use_numba=use_numba,
        )
        step_rows = equivalence["steps"]
        zhang_wang_equivalence = {
            "remote_bare_max_abs": equivalence["remote_bare_max_abs"],
            "remote_bare_fro_norm": equivalence["remote_bare_fro_norm"],
            "final_density_frame_max_abs": equivalence["final_density_frame_max_abs"],
            "max_hamiltonian_abs": max(float(row["hamiltonian_max_abs"]) for row in step_rows),
            "max_raw_density_update_abs": max(float(row["raw_density_update_max_abs"]) for row in step_rows),
            "max_mixed_density_abs": max(float(row["mixed_density_max_abs"]) for row in step_rows),
            "max_oda_lambda_abs": max(float(row["oda_lambda_abs_diff"]) for row in step_rows),
            "steps": step_rows,
        }

        path_payload = None
        if args.check_path:
            if case.path_tsv is None:
                raise ValueError(f"Requested path check but no path TSV exists for {case.case_dir}")
            assert path is not None
            assert path_h0 is not None
            assert path_overlap is not None
            assert path_grid_overlap is not None
            h_path = build_projected_target_hamiltonian(
                path_h0,
                density_ref,
                source_overlap_blocks=overlap_blocks,
                target_overlap_blocks=path_overlap,
                target_source_overlap_blocks=path_grid_overlap,
                v0=v0,
                use_numba=use_numba,
            )
            generated_bands = build_flavor_band_data(h_path).energies.T
            ref_kdist, ref_bands = _path_table(case.path_tsv)
            if generated_bands.shape != ref_bands.shape:
                raise ValueError(f"Path band shape mismatch for {case.path_tsv}: {generated_bands.shape} vs {ref_bands.shape}")
            path_payload = {
                "path_tsv": str(case.path_tsv),
                "kdist_max_abs": _max_abs(path.kdist, ref_kdist),
                "sorted_band_max_abs": _max_abs(np.sort(generated_bands, axis=1), np.sort(ref_bands, axis=1)),
                "direct_column_band_max_abs": _max_abs(generated_bands, ref_bands),
            }

        converged_payload = None
        if args.run_converged:
            converged_payload = _run_converged_equivalence(
                density_initial=density_ref,
                grid_solution=grid_solution,
                params=params,
                nu=case.nu,
                init_mode=case.init_mode,
                seed=case.seed,
                overlap_blocks=overlap_blocks,
                max_iter=int(args.max_iter),
                precision=float(args.precision),
                oda_stall_threshold=float(args.oda_stall_threshold),
                use_numba=use_numba,
                path=path,
                path_h0=path_h0,
                path_overlap=path_overlap,
                path_grid_overlap=path_grid_overlap,
                reference_path_tsv=case.path_tsv,
            )

        maxima["wang_reference_h0_max_abs"] = max(maxima["wang_reference_h0_max_abs"], wang_reference["h0_max_abs"])
        maxima["wang_reference_sigma_z_max_abs"] = max(
            maxima["wang_reference_sigma_z_max_abs"],
            wang_reference["sigma_z_max_abs"],
        )
        maxima["wang_reference_hamiltonian_max_abs"] = max(
            maxima["wang_reference_hamiltonian_max_abs"],
            wang_reference["hamiltonian_max_abs"],
        )
        maxima["wang_reference_energies_max_abs"] = max(
            maxima["wang_reference_energies_max_abs"],
            wang_reference["energies_max_abs"],
        )
        maxima["wang_reference_fixed_point_norm"] = max(
            maxima["wang_reference_fixed_point_norm"],
            wang_reference["fixed_point_norm"],
        )
        maxima["zhang_wang_hamiltonian_max_abs"] = max(
            maxima["zhang_wang_hamiltonian_max_abs"],
            zhang_wang_equivalence["max_hamiltonian_abs"],
        )
        maxima["zhang_wang_raw_density_update_max_abs"] = max(
            maxima["zhang_wang_raw_density_update_max_abs"],
            zhang_wang_equivalence["max_raw_density_update_abs"],
        )
        maxima["zhang_wang_mixed_density_max_abs"] = max(
            maxima["zhang_wang_mixed_density_max_abs"],
            zhang_wang_equivalence["max_mixed_density_abs"],
        )
        maxima["zhang_wang_oda_lambda_abs"] = max(
            maxima["zhang_wang_oda_lambda_abs"],
            zhang_wang_equivalence["max_oda_lambda_abs"],
        )
        if path_payload is not None:
            maxima["path_sorted_band_max_abs"] = max(
                maxima["path_sorted_band_max_abs"],
                path_payload["sorted_band_max_abs"],
            )
        if converged_payload is not None:
            maxima["converged_hamiltonian_max_abs"] = max(
                maxima["converged_hamiltonian_max_abs"],
                converged_payload["hamiltonian_max_abs"],
            )
            maxima["converged_density_max_abs"] = max(
                maxima["converged_density_max_abs"],
                converged_payload["density_max_abs"],
            )
            maxima["converged_energies_max_abs"] = max(
                maxima["converged_energies_max_abs"],
                converged_payload["energies_max_abs"],
            )
            maxima["converged_oda_trace_max_abs"] = max(
                maxima["converged_oda_trace_max_abs"],
                converged_payload["iter_oda_max_abs"],
            )
            maxima["converged_iter_trace_length_abs_diff"] = max(
                maxima["converged_iter_trace_length_abs_diff"],
                float(converged_payload["iter_trace_length_abs_diff"]),
            )
            converged_path = converged_payload.get("path_bands")
            if converged_path is not None:
                maxima["converged_path_sorted_band_max_abs"] = max(
                    maxima["converged_path_sorted_band_max_abs"],
                    converged_path["wang_split_sorted_band_max_abs"],
                )
                maxima["converged_wang_reference_path_sorted_band_max_abs"] = max(
                    maxima["converged_wang_reference_path_sorted_band_max_abs"],
                    converged_path.get("wang_reference_sorted_band_max_abs", 0.0),
                )
                maxima["converged_split_reference_path_sorted_band_max_abs"] = max(
                    maxima["converged_split_reference_path_sorted_band_max_abs"],
                    converged_path.get("split_reference_sorted_band_max_abs", 0.0),
                )

        payload_case: dict[str, object] = {
            "case_dir": str(case.case_dir),
            "state_path": str(case.state_path),
            "nu": case.nu,
            "init_mode": case.init_mode,
            "seed": case.seed,
            "wang_reference": wang_reference,
            "zhang_wang_equivalence": zhang_wang_equivalence,
        }
        if path_payload is not None:
            payload_case["path_bands"] = path_payload
        if converged_payload is not None:
            payload_case["converged_equivalence"] = converged_payload
        case_payloads.append(payload_case)

    threshold = float(args.threshold)
    fixed_threshold = float(args.fixed_point_threshold)
    passed = (
        maxima["wang_reference_h0_max_abs"] <= threshold
        and maxima["wang_reference_sigma_z_max_abs"] <= threshold
        and maxima["wang_reference_hamiltonian_max_abs"] <= threshold
        and maxima["wang_reference_energies_max_abs"] <= threshold
        and maxima["wang_reference_fixed_point_norm"] <= fixed_threshold
        and maxima["zhang_wang_hamiltonian_max_abs"] <= threshold
        and maxima["zhang_wang_raw_density_update_max_abs"] <= threshold
        and maxima["zhang_wang_mixed_density_max_abs"] <= threshold
        and maxima["zhang_wang_oda_lambda_abs"] <= threshold
        and (not args.check_path or maxima["path_sorted_band_max_abs"] <= threshold)
        and (
            not args.run_converged
            or (
                maxima["converged_hamiltonian_max_abs"] <= threshold
                and maxima["converged_density_max_abs"] <= threshold
                and maxima["converged_energies_max_abs"] <= threshold
                and maxima["converged_oda_trace_max_abs"] <= threshold
                and maxima["converged_iter_trace_length_abs_diff"] == 0.0
                and maxima["converged_path_sorted_band_max_abs"] <= threshold
                and all(
                    bool(case_payload.get("converged_equivalence", {}).get("wang_converged", False))
                    and bool(case_payload.get("converged_equivalence", {}).get("split_converged", False))
                    for case_payload in case_payloads
                )
            )
        )
    )

    payload: dict[str, object] = {
        "reference_root": str(args.reference_root),
        "output_dir": str(output_dir),
        "pid": int(os.getpid()),
        "cases": case_payloads,
        "aggregate": {
            "status": "pass" if passed else "fail",
            "case_count": len(cases),
            "threshold": threshold,
            "fixed_point_threshold": fixed_threshold,
            "maxima": maxima,
        },
    }
    json_path = output_dir / "bare_hf_framework_validation.json"
    report_path = output_dir / "bare_hf_framework_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    print(f"[done] status={payload['aggregate']['status']} json={json_path} report={report_path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
