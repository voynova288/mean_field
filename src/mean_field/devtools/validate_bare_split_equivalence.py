from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mean_field.core.hf import build_projected_interaction_hamiltonian
from mean_field.crpa import half_reference_delta_like, physical_projector_from_delta
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.validate_bare_hf_frameworks_against_liu_ref import _split_step_equivalence
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    RestrictedHartreeFockState,
    build_b0_uniform_lattice,
    build_h0_from_bm,
    build_overlap_block_set,
    initialize_full_state,
    solve_bm_model,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "TBG_HF_cRPA" / "hf_framework_validation" / "bare_split_smoke"


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny Zhang/Wang no-cRPA bare-split equivalence smoke test.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--lk", type=int, default=2)
    parser.add_argument("--lg", type=int, default=3)
    parser.add_argument("--overlap-lg", type=int, default=None)
    parser.add_argument("--epsilon-r", type=float, default=10.0)
    parser.add_argument("--screening-lm", type=float, default=400.0 / 2.46 / 2.0)
    parser.add_argument("--finite-zero-limit", action="store_true", default=True)
    parser.add_argument("--zero-cutoff", type=float, default=1.0e-6)
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.4)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument("--init-mode", default="random")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--use-numba", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--allow-login", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not bool(args.allow_login):
        ensure_not_running_compute_on_login_node("bare split equivalence validation")
    use_numba: bool | None
    if args.use_numba == "auto":
        use_numba = None
    else:
        use_numba = args.use_numba == "true"

    params = TBGParameters.from_degrees(
        float(args.theta_deg),
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )
    grid = build_b0_uniform_lattice(params, int(args.lk))
    solution = solve_bm_model(params, grid.kvec, lg=int(args.lg), sigma_rotation=True)
    h0 = build_h0_from_bm(solution)
    overlap_lg = int(args.lg if args.overlap_lg is None else args.overlap_lg)
    overlap_blocks = build_overlap_block_set(
        solution,
        lg=overlap_lg,
        relative_permittivity=float(args.epsilon_r),
        screening_lm=float(args.screening_lm),
        finite_zero_limit=bool(args.finite_zero_limit),
        zero_cutoff=float(args.zero_cutoff),
    )
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=float(args.nu))
    initialize_full_state(state, init_mode=str(args.init_mode), seed=int(args.seed))
    density_initial = state.density.copy()

    sigma_ref = build_projected_interaction_hamiltonian(
        half_reference_delta_like(density_initial),
        overlap_blocks,
        v0=state.v0,
        use_numba=use_numba,
    )
    sigma_projector = build_projected_interaction_hamiltonian(
        physical_projector_from_delta(density_initial),
        overlap_blocks,
        v0=state.v0,
        use_numba=use_numba,
    )
    sigma_delta = build_projected_interaction_hamiltonian(
        density_initial,
        overlap_blocks,
        v0=state.v0,
        use_numba=use_numba,
    )
    bare_split_identity_max_abs = _max_abs(sigma_ref + sigma_projector, sigma_delta)

    steps = _split_step_equivalence(
        density_initial=density_initial,
        h0=h0,
        sigma_z=solution.sigma_z,
        nu=float(args.nu),
        overlap_blocks=overlap_blocks,
        v0=state.v0,
        max_steps=int(args.steps),
        use_numba=use_numba,
    )
    step_rows = list(steps["steps"])
    maxima = {
        "bare_split_identity_max_abs": bare_split_identity_max_abs,
        "hamiltonian_max_abs": max(float(row["hamiltonian_max_abs"]) for row in step_rows),
        "raw_density_update_max_abs": max(float(row["raw_density_update_max_abs"]) for row in step_rows),
        "mixed_density_max_abs": max(float(row["mixed_density_max_abs"]) for row in step_rows),
        "oda_lambda_abs_diff": max(float(row["oda_lambda_abs_diff"]) for row in step_rows),
        "delta_interaction_max_abs": max(float(row["delta_interaction_max_abs"]) for row in step_rows),
    }
    passed = all(float(value) <= float(args.tolerance) for value in maxima.values())
    payload: dict[str, object] = {
        "status": "pass" if passed else "fail",
        "parameters": {
            "theta_deg": float(args.theta_deg),
            "nu": float(args.nu),
            "lk": int(args.lk),
            "lg": int(args.lg),
            "overlap_lg": overlap_lg,
            "epsilon_r": float(args.epsilon_r),
            "screening_lm": float(args.screening_lm),
            "steps": int(args.steps),
            "tolerance": float(args.tolerance),
        },
        "maxima": maxima,
        "steps": step_rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "bare_split_equivalence.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[bare-split-equivalence] "
        f"status={payload['status']} "
        f"identity={bare_split_identity_max_abs:.6e} "
        f"hamiltonian={maxima['hamiltonian_max_abs']:.6e} "
        f"density={maxima['raw_density_update_max_abs']:.6e} "
        f"json={json_path}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
