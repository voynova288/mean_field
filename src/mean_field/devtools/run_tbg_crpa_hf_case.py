from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.crpa import (
    CRPAScreenedCoulomb,
    load_crpa_result,
    run_full_crpa_hartree_fock,
    validate_hf_compatible_crpa,
)
from mean_field.crpa.validation import (
    compare_fig1e_window_to_paper_points,
    crpa_convention_family,
    fig1e_paper_point_gate_failures,
)
from mean_field.devtools.resample_b0_density_stack import resample_density_stack
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    RestrictedHartreeFockState,
    build_b0_uniform_lattice,
    build_overlap_block_set,
    solve_bm_model,
)


DEFAULT_OUTPUT_ROOT = Path("results") / "TBG_HF_cRPA" / "hf_crpa_runs"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CRPA_PHYSICS_REFERENCE_DIR = (
    REPO_ROOT / "results" / "TBG_HF_cRPA" / "crpa_lk24_lg9_q11_zhang_appendix_fig4_merged"
)


@dataclass(frozen=True)
class InitSpec:
    init_mode: str
    seed: int
    label: str
    initial_state_path: Path | None = None


def _parse_init_spec(raw: str) -> InitSpec:
    if ":" in raw:
        mode, seed = raw.split(":", 1)
        normalized_mode = mode.strip()
        return InitSpec(normalized_mode, int(seed), normalized_mode)
    normalized_mode = raw.strip()
    return InitSpec(normalized_mode, 1, normalized_mode)


def _safe_label(raw: str) -> str:
    return "".join(char if char.isascii() and (char.isalnum() or char in "._+-") else "_" for char in raw.strip())


def _parse_init_state_spec(raw: str) -> InitSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
        safe_label = _safe_label(label)
        state_path = Path(path)
    else:
        state_path = Path(raw)
        safe_label = _safe_label(f"restart_{state_path.stem}")
    if not safe_label:
        raise ValueError(f"Empty restart label in --init-state {raw!r}")
    return InitSpec("bm", 0, safe_label, state_path)


def _infer_inclusive_lk_from_nk(nk: int) -> int:
    side = int(round(np.sqrt(int(nk))))
    if side * side != int(nk) or side < 2:
        raise ValueError(f"Cannot infer inclusive square-grid lk from nk={nk}")
    return side - 1


def _load_initial_density(
    path: Path,
    *,
    target_lk: int,
    resample_method: str,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    with np.load(path) as data:
        if "density" not in data:
            raise ValueError(f"Restart state {path} does not contain a density array")
        density = np.asarray(data["density"], dtype=np.complex128)
        metadata: list[tuple[str, str]] = [("initial_state_path", str(path))]
        source_lk = int(np.asarray(data["lk"]).reshape(-1)[0]) if "lk" in data else _infer_inclusive_lk_from_nk(density.shape[2])
        target_lk = int(target_lk)
        target_nk = (target_lk + 1) * (target_lk + 1)
        if density.shape[2] != target_nk:
            if resample_method == "none":
                raise ValueError(
                    f"Restart state density nk={density.shape[2]} does not match target lk={target_lk} "
                    f"(nk={target_nk}). Use --initial-state-resample bilinear or nearest to continue across lk."
                )
            density = resample_density_stack(
                density,
                source_lk=source_lk,
                target_lk=target_lk,
                method=resample_method,
                hermitize=True,
            )
            metadata.extend(
                [
                    ("initial_state_resampled", "true"),
                    ("initial_state_source_lk", str(source_lk)),
                    ("initial_state_target_lk", str(target_lk)),
                    ("initial_state_resample_method", str(resample_method)),
                ]
            )
        else:
            metadata.extend(
                [
                    ("initial_state_resampled", "false"),
                    ("initial_state_source_lk", str(source_lk)),
                    ("initial_state_target_lk", str(target_lk)),
                    ("initial_state_resample_method", "none"),
                ]
            )
        return density, metadata


def _theta_tag(theta_deg: float) -> str:
    return f"{round(theta_deg * 100):03d}"


def _nu_tag(nu: float) -> str:
    return f"{round(nu * 1000):+05d}"


def _case_tag(theta_deg: float, nu: float, init_mode: str, seed: int, lk: int, lg: int) -> str:
    return f"theta_{_theta_tag(theta_deg)}_nu_{_nu_tag(nu)}_init_{init_mode}_seed_{seed:03d}_lk{lk}_lg{lg}"


def _write_key_value_file(path: Path, entries: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "theta_deg",
        "nu",
        "init_mode",
        "seed",
        "iterations",
        "exit_reason",
        "converged",
        "mu_mev",
        "final_energy",
        "final_error",
        "final_oda",
        "hf_elapsed_sec",
        "state_path",
        "initial_state_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(row.get(column, "") for column in columns) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a zero-field TBG full HF case with Zhang cRPA screened Coulomb.")
    parser.add_argument("--crpa-dir", type=Path, required=True, help="cRPA artifact directory containing screened_coulomb.npz.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--lk", type=int, default=6)
    parser.add_argument("--lg", type=int, default=9)
    parser.add_argument("--overlap-lg", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--precision", type=float, default=1.0e-5)
    parser.add_argument("--oda-stall-threshold", type=float, default=1.0e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.4)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument(
        "--fock-interpolation",
        choices=("matrix_diagonal", "linear", "nearest"),
        default="matrix_diagonal",
        help="cRPA Fock lookup. Use matrix_diagonal for production HF.",
    )
    parser.add_argument(
        "--initial-state-resample",
        choices=("none", "bilinear", "nearest"),
        default="none",
        help="Resample --init-state density when its inclusive lk differs from --lk.",
    )
    parser.add_argument(
        "--allow-incompatible-crpa",
        action="store_true",
        help="Bypass HF-compatible cRPA metadata checks. Intended only for diagnostic old-artifact runs.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Required with --allow-incompatible-crpa; marks the run as non-production diagnostics.",
    )
    parser.add_argument("--crpa-physics-reference-dir", type=Path, default=DEFAULT_CRPA_PHYSICS_REFERENCE_DIR)
    parser.add_argument(
        "--skip-crpa-physics-gate",
        action="store_true",
        help=(
            "Skip the convention-aware cRPA gate. Fig. 1(e) is a hard gate only for "
            "zhang_zero_fill paper-reference artifacts, not for HF-compatible artifacts."
        ),
    )
    parser.add_argument("--crpa-fig1e-max-rmse", type=float, default=0.8)
    parser.add_argument("--crpa-fig1e-max-abs", type=float, default=1.5)
    parser.add_argument("--crpa-fig1e-max-mean-abs", type=float, default=0.7)
    parser.add_argument("--crpa-fig1e-min-paper-points", type=int, default=5)
    parser.add_argument(
        "--init",
        dest="init_specs",
        action="append",
        default=None,
        help="Initialization as mode or mode:seed. Repeat for multiple cases. Default: flavor:1.",
    )
    parser.add_argument(
        "--init-state",
        dest="init_state_specs",
        action="append",
        default=None,
        help=(
            "Restart from a saved .npz state containing density. Use label=path to control the output tag; "
            "repeat for multiple restart states."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.allow_incompatible_crpa) and not bool(args.diagnostic_only):
        raise SystemExit(
            "--allow-incompatible-crpa now requires --diagnostic-only. "
            "Production HF bands must use HF-compatible cRPA metadata."
        )
    theta_deg = float(args.theta_deg)
    nu = float(args.nu)
    lk = int(args.lk)
    lg = int(args.lg)
    overlap_lg = int(args.overlap_lg)
    run_tag = args.run_tag or f"zhang_crpa_lg9_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = Path(args.output_root) / f"theta_{_theta_tag(theta_deg)}_nu_{_nu_tag(nu)}_{run_tag}"
    state_dir = root / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    init_specs_list = [_parse_init_spec(item) for item in (args.init_specs or [])]
    init_specs_list.extend(_parse_init_state_spec(item) for item in (args.init_state_specs or []))
    if not init_specs_list:
        init_specs_list.append(_parse_init_spec("flavor:1"))
    init_specs = tuple(init_specs_list)
    started = datetime.now()
    total_start = perf_counter()

    params = TBGParameters.from_degrees(
        theta_deg,
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )
    crpa_result = load_crpa_result(args.crpa_dir)
    if not bool(args.allow_incompatible_crpa):
        validate_hf_compatible_crpa(crpa_result, params, theta_deg=theta_deg, overlap_lg=overlap_lg)
    crpa_physics_gate: dict[str, float | int | str] = {}
    crpa_convention = crpa_convention_family(crpa_result)
    if not bool(args.skip_crpa_physics_gate):
        if crpa_convention == "zhang_paper_reference":
            comparison = compare_fig1e_window_to_paper_points(crpa_result)
            failures = fig1e_paper_point_gate_failures(
                comparison,
                max_rmse=float(args.crpa_fig1e_max_rmse),
                max_abs=float(args.crpa_fig1e_max_abs),
                max_mean_abs=float(args.crpa_fig1e_max_mean_abs),
                min_points=int(args.crpa_fig1e_min_paper_points),
            )
            crpa_physics_gate = {
                key: float(value) if isinstance(value, float) else int(value) for key, value in comparison.items()
            }
            crpa_physics_gate["gate_type"] = "corrected_fig1e_paper_points"
            crpa_physics_gate["convention_family"] = crpa_convention
            if failures:
                raise SystemExit(f"cRPA Fig. 1(e) physics gate failed for {args.crpa_dir}: {'; '.join(failures)}")
            print(
                "[stage] crpa_physics_gate "
                "reference=corrected_fig1e_paper_points "
                f"fig1e_paper_rmse={float(comparison['fig1e_paper_rmse']):.6g} "
                f"fig1e_paper_max_abs={float(comparison['fig1e_paper_max_abs']):.6g} "
                f"fig1e_paper_mean_abs={float(comparison['fig1e_paper_mean_abs']):.6g}",
                flush=True,
            )
        else:
            crpa_physics_gate = {
                "gate_type": "hf_compatible_convention",
                "convention_family": crpa_convention,
                "fig1e_paper_gate": "diagnostic_only_not_a_hard_gate_for_hf",
            }
            print(
                "[stage] crpa_physics_gate "
                f"convention_family={crpa_convention} "
                "fig1e_paper_gate=diagnostic_only_not_a_hard_gate_for_hf",
                flush=True,
            )
    crpa_screening = CRPAScreenedCoulomb(crpa_result)

    print(
        "[stage] setup "
        f"theta={theta_deg} nu={nu} lk={lk} lg={lg} overlap_lg={overlap_lg} "
        f"crpa_lk={crpa_result.lk} crpa_lg={crpa_result.lg} crpa_q_lg={crpa_result.q_lg}",
        flush=True,
    )
    print(f"[stage] crpa_dir={args.crpa_dir}", flush=True)

    bm_start = perf_counter()
    grid = build_b0_uniform_lattice(params, lk)
    grid_solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
    bm_elapsed = perf_counter() - bm_start
    print(f"[stage] bm_grid done elapsed_sec={bm_elapsed:.3f} nk={grid_solution.nk}", flush=True)

    overlap_start = perf_counter()
    overlap_blocks = build_overlap_block_set(
        grid_solution,
        lg=overlap_lg,
        relative_permittivity=float(crpa_result.coulomb_params.epsilon_bn),
        screening_lm=float(crpa_result.coulomb_params.screening_lm),
        finite_zero_limit=bool(crpa_result.coulomb_params.finite_zero_limit),
        zero_cutoff=float(crpa_result.coulomb_params.zero_cutoff),
    )
    overlap_elapsed = perf_counter() - overlap_start
    print(f"[stage] overlap done elapsed_sec={overlap_elapsed:.3f} shifts={len(overlap_blocks.shifts)}", flush=True)

    rows: list[dict[str, str]] = []
    for spec in init_specs:
        tag = _case_tag(theta_deg, nu, spec.label, spec.seed, lk, lg)
        state_path = state_dir / f"{tag}.npz"
        initial_density = None
        initial_density_metadata: list[tuple[str, str]] = []
        if spec.initial_state_path is not None:
            initial_density, initial_density_metadata = _load_initial_density(
                spec.initial_state_path,
                target_lk=lk,
                resample_method=str(args.initial_state_resample),
            )
        restart_suffix = "" if spec.initial_state_path is None else f" restart_from={spec.initial_state_path}"
        if any(key == "initial_state_resampled" and value == "true" for key, value in initial_density_metadata):
            restart_suffix += f" resample={args.initial_state_resample}"
        print(f"[stage] hf:start init={spec.label} seed={spec.seed}{restart_suffix}", flush=True)
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=nu, precision=float(args.precision))
        state.diagnostics["overlap_lg"] = float(overlap_lg)
        hf_start = perf_counter()
        hf_run = run_full_crpa_hartree_fock(
            state,
            overlap_blocks,
            grid_solution.lattice_kvec,
            params,
            crpa_screening=crpa_screening,
            init_mode=spec.init_mode,
            seed=spec.seed,
            initial_density=initial_density,
            beta=float(args.beta),
            max_iter=int(args.max_iter),
            oda_stall_threshold=float(args.oda_stall_threshold),
            fock_interpolation=str(args.fock_interpolation),
        )
        hf_elapsed = perf_counter() - hf_start
        print(
            "[stage] hf:done "
            f"init={spec.label} seed={spec.seed} elapsed_sec={hf_elapsed:.3f} "
            f"iterations={hf_run.iterations} converged={str(hf_run.converged).lower()} "
            f"exit_reason={hf_run.exit_reason}",
            flush=True,
        )
        np.savez_compressed(
            state_path,
            density=hf_run.state.density,
            hamiltonian=hf_run.state.hamiltonian,
            h0=hf_run.state.h0,
            energies=hf_run.state.energies,
            sigma_ztauz=hf_run.state.sigma_ztauz,
            sigma_z=hf_run.state.sigma_z,
            mu=np.asarray([hf_run.state.mu], dtype=float),
            iter_energy=hf_run.iter_energy,
            iter_err=hf_run.iter_err,
            iter_oda=hf_run.iter_oda,
            nu=np.asarray([nu], dtype=float),
            init_mode=np.asarray([spec.label]),
            normalized_init_mode=np.asarray([hf_run.init_mode]),
            seed=np.asarray([spec.seed], dtype=int),
            initial_state_path=np.asarray(["" if spec.initial_state_path is None else str(spec.initial_state_path)]),
            initial_state_resampled=np.asarray(
                [
                    any(
                        key == "initial_state_resampled" and value == "true"
                        for key, value in initial_density_metadata
                    )
                ]
            ),
            initial_state_resample_method=np.asarray([str(args.initial_state_resample)]),
            initial_state_metadata_json=np.asarray([json.dumps(dict(initial_density_metadata), sort_keys=True)]),
            converged=np.asarray([hf_run.converged]),
            exit_reason=np.asarray([hf_run.exit_reason]),
            theta_deg=np.asarray([theta_deg], dtype=float),
            lk=np.asarray([lk], dtype=int),
            lg=np.asarray([lg], dtype=int),
            overlap_lg=np.asarray([overlap_lg], dtype=int),
            w0_mev=np.asarray([float(args.w0)], dtype=float),
            w1_mev=np.asarray([float(args.w1)], dtype=float),
            vf_mev=np.asarray([float(args.vf)], dtype=float),
            crpa_dir=np.asarray([str(args.crpa_dir)]),
            crpa_lk=np.asarray([crpa_result.lk], dtype=int),
            crpa_lg=np.asarray([crpa_result.lg], dtype=int),
            crpa_q_lg=np.asarray([crpa_result.q_lg], dtype=int),
            crpa_metadata_json=np.asarray([json.dumps(crpa_result.metadata, sort_keys=True)]),
            crpa_physics_gate_json=np.asarray([json.dumps(crpa_physics_gate, sort_keys=True)]),
            diagnostic_only=np.asarray([bool(args.diagnostic_only)]),
            fock_interpolation=np.asarray([str(args.fock_interpolation)]),
            epsilon_bn=np.asarray([float(crpa_result.coulomb_params.epsilon_bn)], dtype=float),
            ds_angstrom=np.asarray([float(crpa_result.coulomb_params.ds_angstrom)], dtype=float),
            max_iter=np.asarray([int(args.max_iter)], dtype=int),
        )
        rows.append(
            {
                "theta_deg": f"{theta_deg:.16g}",
                "nu": f"{nu:.16g}",
                "init_mode": spec.label,
                "seed": str(spec.seed),
                "iterations": str(hf_run.iterations),
                "exit_reason": hf_run.exit_reason,
                "converged": str(hf_run.converged).lower(),
                "mu_mev": f"{float(hf_run.state.mu):.16e}",
                "final_energy": "" if hf_run.iter_energy.size == 0 else f"{float(hf_run.iter_energy[-1]):.16e}",
                "final_error": "" if hf_run.iter_err.size == 0 else f"{float(hf_run.iter_err[-1]):.16e}",
                "final_oda": "" if hf_run.iter_oda.size == 0 else f"{float(hf_run.iter_oda[-1]):.16e}",
                "hf_elapsed_sec": f"{hf_elapsed:.16e}",
                "state_path": str(state_path),
                "initial_state_path": "" if spec.initial_state_path is None else str(spec.initial_state_path),
            }
        )
        _write_summary(root / "summary.tsv", rows)

    total_elapsed = perf_counter() - total_start
    _write_key_value_file(
        root / "run_info.txt",
        [
            ("theta_deg", f"{theta_deg:.16g}"),
            ("nu", f"{nu:.16g}"),
            ("run_tag", run_tag),
            ("lk", str(lk)),
            ("lg", str(lg)),
            ("overlap_lg", str(overlap_lg)),
            ("max_iter", str(int(args.max_iter))),
            ("precision", f"{float(args.precision):.16g}"),
            ("w0_meV", f"{float(args.w0):.16g}"),
            ("w1_meV", f"{float(args.w1):.16g}"),
            ("vf_meV", f"{float(args.vf):.16g}"),
            ("interaction_model", "zhang_crpa_screened"),
            ("fock_interpolation", str(args.fock_interpolation)),
            ("initial_state_resample", str(args.initial_state_resample)),
            ("allow_incompatible_crpa", str(bool(args.allow_incompatible_crpa)).lower()),
            ("diagnostic_only", str(bool(args.diagnostic_only)).lower()),
            ("crpa_dir", str(args.crpa_dir)),
            ("crpa_lk", str(crpa_result.lk)),
            ("crpa_lg", str(crpa_result.lg)),
            ("crpa_q_lg", str(crpa_result.q_lg)),
            ("crpa_metadata_json", json.dumps(crpa_result.metadata, sort_keys=True)),
            ("crpa_physics_gate_json", json.dumps(crpa_physics_gate, sort_keys=True)),
            ("epsilon_bn", f"{float(crpa_result.coulomb_params.epsilon_bn):.16g}"),
            ("ds_angstrom", f"{float(crpa_result.coulomb_params.ds_angstrom):.16g}"),
            (
                "init_specs",
                ",".join(
                    f"{spec.label}={spec.initial_state_path}" if spec.initial_state_path is not None else f"{spec.init_mode}:{spec.seed}"
                    for spec in init_specs
                ),
            ),
            ("start_time", started.strftime("%Y-%m-%dT%H:%M:%S")),
            ("end_time", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
            ("bm_elapsed_sec", f"{bm_elapsed:.16e}"),
            ("overlap_elapsed_sec", f"{overlap_elapsed:.16e}"),
            ("total_elapsed_sec", f"{total_elapsed:.16e}"),
            ("hostname", socket.gethostname()),
            ("output_dir", str(root)),
        ],
    )
    print(f"[stage] complete output_dir={root} total_elapsed_sec={total_elapsed:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
