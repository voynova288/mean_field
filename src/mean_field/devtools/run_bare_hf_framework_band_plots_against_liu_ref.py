#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mean_field.core.hf import build_flavor_band_data, build_projected_target_hamiltonian
from mean_field.crpa import run_bare_split_full_hartree_fock
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.validate_bare_hf_frameworks_against_liu_ref import (
    DEFAULT_REFERENCE_ROOT,
    _discover_cases,
    _max_abs,
    _path_table,
    _require_common_parameters,
    _split_target_hamiltonian,
)
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    HFPathResult,
    RestrictedHartreeFockState,
    build_b0_uniform_lattice,
    build_gamma_m_k_gamma_kprime_kpath,
    build_h0_from_bm,
    build_overlap_block_set,
    run_full_hartree_fock,
    solve_bm_model,
    write_hf_band_plot,
    write_hf_path_nodes_tsv,
    write_hf_path_summary,
    write_hf_path_tsv,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "TBG_HF_cRPA"
    / "hf_framework_bands"
    / "liu_ref_lk24_20260516_wang_zhang_bare_converged"
)


def _nu_label(nu: float) -> str:
    value = int(round(float(nu) * 1000.0))
    sign = "+" if value >= 0 else "-"
    return f"nu_{sign}{abs(value):04d}"


def _make_path_result(
    *,
    params: TBGParameters,
    path,
    hamiltonian: np.ndarray,
    state,
    exit_reason: str,
    case,
    points_per_segment: int,
    framework_label: str,
    common,
) -> HFPathResult:
    band_data = build_flavor_band_data(
        hamiltonian,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    return HFPathResult(
        params=params,
        path=path,
        hamiltonian=np.asarray(hamiltonian, dtype=np.complex128),
        band_data=band_data,
        mu=float(state.mu),
        nu=float(case.nu),
        lk=int(case.lk),
        lg=int(common.lg),
        points_per_segment=int(points_per_segment),
        init_mode=f"{case.init_mode} ({framework_label})",
        normalized_init_mode=str(case.init_mode),
        seed=int(case.seed),
        exit_reason=str(exit_reason),
        beta=1.0,
        overlap_lg=int(common.overlap_lg),
        relative_permittivity=float(common.epsilon_r),
        screening_lm=float(common.screening_lm),
        finite_zero_limit=bool(common.q_zero_limit),
    )


def _write_framework_outputs(
    *,
    output_dir: Path,
    result: HFPathResult,
    stem: str,
    state_path: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_tsv = output_dir / f"{stem}_hf_path.tsv"
    nodes_tsv = output_dir / f"{stem}_hf_path_nodes.tsv"
    summary_txt = output_dir / f"{stem}_hf_path_summary.txt"
    write_hf_path_tsv(path_tsv, result)
    write_hf_path_nodes_tsv(nodes_tsv, result)
    write_hf_path_summary(summary_txt, result, hf_state_path=str(state_path))
    plot_paths = write_hf_band_plot(output_dir, result, stem=f"{stem}_band_plot")
    return {
        "path_tsv": str(path_tsv),
        "nodes_tsv": str(nodes_tsv),
        "summary_txt": str(summary_txt),
        "band_plot_png": str(plot_paths["band_plot_png"]),
        "band_plot_pdf": str(plot_paths["band_plot_pdf"]),
    }


def _case_summary_path(output_dir: Path, case_dir_name: str) -> Path:
    return output_dir / case_dir_name / "framework_band_plot_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-cRPA Wang and Zhang bare-split HF frameworks from "
            "Liu_reproduce_ref states, then write converged path-band plots."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nu", type=float, action="append", default=None, help="Optional filling filter; repeatable.")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--precision", type=float, default=1.0e-5)
    parser.add_argument("--oda-stall-threshold", type=float, default=1.0e-3)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=1.0e-7)
    parser.add_argument("--use-numba", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--allow-login", action="store_true", help="Bypass login-node compute guard for tiny dry runs only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_login:
        ensure_not_running_compute_on_login_node("bare HF framework band plotting")

    requested_nu = None if args.nu is None else {float(value) for value in args.nu}
    cases = _discover_cases(args.reference_root, requested_nu)
    common = _require_common_parameters(cases)
    if args.use_numba == "auto":
        use_numba: bool | None = None
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
    overlap_blocks = build_overlap_block_set(
        grid_solution,
        lg=common.overlap_lg,
        relative_permittivity=common.epsilon_r,
        screening_lm=common.screening_lm,
        finite_zero_limit=common.q_zero_limit,
    )
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

    payload_cases: list[dict[str, object]] = []
    for case in cases:
        print(f"[case] nu={case.nu:+.6g} state={case.state_path}", flush=True)
        with np.load(case.state_path, allow_pickle=False) as data:
            density_initial = np.asarray(data["density"], dtype=np.complex128)

        wang_state = RestrictedHartreeFockState.from_bm_solution(
            grid_solution,
            nu=case.nu,
            precision=float(args.precision),
        )
        zhang_state = RestrictedHartreeFockState.from_bm_solution(
            grid_solution,
            nu=case.nu,
            precision=float(args.precision),
        )
        wang_run = run_full_hartree_fock(
            wang_state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            params,
            init_mode=case.init_mode,
            seed=case.seed,
            max_iter=int(args.max_iter),
            oda_stall_threshold=float(args.oda_stall_threshold),
            initial_density=density_initial,
        )
        zhang_run = run_bare_split_full_hartree_fock(
            zhang_state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            params,
            init_mode=case.init_mode,
            seed=case.seed,
            max_iter=int(args.max_iter),
            oda_stall_threshold=float(args.oda_stall_threshold),
            initial_density=density_initial,
            use_numba=use_numba,
        )

        wang_path_h = build_projected_target_hamiltonian(
            path_h0,
            wang_run.state.density,
            source_overlap_blocks=overlap_blocks,
            target_overlap_blocks=path_overlap,
            target_source_overlap_blocks=path_grid_overlap,
            v0=wang_run.state.v0,
            use_numba=use_numba,
        )
        zhang_path_h = _split_target_hamiltonian(
            path_h0,
            zhang_run.state.density,
            source_overlap_blocks=overlap_blocks,
            target_overlap_blocks=path_overlap,
            target_source_overlap_blocks=path_grid_overlap,
            v0=zhang_run.state.v0,
            use_numba=use_numba,
        )
        wang_result = _make_path_result(
            params=params,
            path=path,
            hamiltonian=wang_path_h,
            state=wang_run.state,
            exit_reason=wang_run.exit_reason,
            case=case,
            points_per_segment=int(args.points_per_segment),
            framework_label="Wang no-cRPA",
            common=common,
        )
        zhang_result = _make_path_result(
            params=params,
            path=path,
            hamiltonian=zhang_path_h,
            state=zhang_run.state,
            exit_reason=zhang_run.exit_reason,
            case=case,
            points_per_segment=int(args.points_per_segment),
            framework_label="Zhang bare-split no-cRPA",
            common=common,
        )

        case_root = output_dir / case.case_dir.name
        stem_base = case.state_path.stem
        wang_files = _write_framework_outputs(
            output_dir=case_root / "wang_no_crpa" / "path_bands",
            result=wang_result,
            stem=f"{stem_base}_wang_no_crpa",
            state_path=case.state_path,
        )
        zhang_files = _write_framework_outputs(
            output_dir=case_root / "zhang_bare_split_no_crpa" / "path_bands",
            result=zhang_result,
            stem=f"{stem_base}_zhang_bare_split_no_crpa",
            state_path=case.state_path,
        )

        bands_wang = wang_result.band_data.energies.T
        bands_zhang = zhang_result.band_data.energies.T
        metrics: dict[str, float | str | None] = {
            "wang_zhang_path_hamiltonian_max_abs": _max_abs(wang_path_h, zhang_path_h),
            "wang_zhang_path_sorted_band_max_abs": _max_abs(np.sort(bands_wang, axis=1), np.sort(bands_zhang, axis=1)),
            "wang_zhang_path_direct_band_max_abs": _max_abs(bands_wang, bands_zhang),
            "wang_converged": bool(wang_run.converged),
            "zhang_converged": bool(zhang_run.converged),
            "wang_iterations": int(wang_run.iterations),
            "zhang_iterations": int(zhang_run.iterations),
            "wang_exit_reason": str(wang_run.exit_reason),
            "zhang_exit_reason": str(zhang_run.exit_reason),
        }
        if case.path_tsv is not None:
            ref_kdist, ref_bands = _path_table(case.path_tsv)
            metrics.update(
                {
                    "reference_path_tsv": str(case.path_tsv),
                    "kdist_reference_max_abs": _max_abs(path.kdist, ref_kdist),
                    "wang_reference_sorted_band_max_abs": _max_abs(np.sort(bands_wang, axis=1), np.sort(ref_bands, axis=1)),
                    "zhang_reference_sorted_band_max_abs": _max_abs(
                        np.sort(bands_zhang, axis=1),
                        np.sort(ref_bands, axis=1),
                    ),
                }
            )

        case_payload: dict[str, object] = {
            "case_dir": str(case.case_dir),
            "nu": float(case.nu),
            "nu_label": _nu_label(case.nu),
            "state_path": str(case.state_path),
            "output_dir": str(case_root),
            "wang": wang_files,
            "zhang_bare_split": zhang_files,
            "metrics": metrics,
        }
        _case_summary_path(output_dir, case.case_dir.name).write_text(
            json.dumps(case_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload_cases.append(case_payload)
        print(f"[case-done] nu={case.nu:+.6g} output={case_root}", flush=True)

    summary = {
        "status": "pass"
        if all(
            c["metrics"]["wang_zhang_path_sorted_band_max_abs"] <= float(args.threshold)
            and c["metrics"]["wang_zhang_path_hamiltonian_max_abs"] <= float(args.threshold)
            for c in payload_cases
        )
        else "fail",
        "reference_root": str(args.reference_root),
        "output_dir": str(output_dir),
        "case_count": len(payload_cases),
        "cases": payload_cases,
    }
    suffix = "all" if requested_nu is None else "_".join(_nu_label(value) for value in sorted(requested_nu))
    summary_path = output_dir / f"framework_band_plot_summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] status={summary['status']} summary={summary_path}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
