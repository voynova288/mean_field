from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from time import perf_counter

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
    load_or_solve_screening,
    screening_result_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "RnG_hBN" / "cache"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the R5G/hBN screened-U checkpoints required before Fig. 6 HF runs."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--screening-solver", choices=("grid", "fixed_point"), default="grid")
    parser.add_argument("--screening-u-min-mev", type=float, default=-100.0)
    parser.add_argument("--screening-u-max-mev", type=float, default=200.0)
    parser.add_argument("--screening-u-grid-points", type=int, default=121)
    parser.add_argument("--tolerance-mev", type=float, default=3.0)
    return parser.parse_args()


def validate_fig6_screening_checkpoints(
    *,
    cache_dir: Path,
    cache_policy: str = "reuse",
    screening_solver: str = "grid",
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    tolerance_mev: float = 3.0,
) -> dict[str, object]:
    start = perf_counter()
    checks: list[dict[str, object]] = []
    for epsilon_r, expected_u_mev in ((5.0, 28.3), (12.5, 38.8)):
        model = RLGhBNModel.from_config(
            layer_count=5,
            xi=0,
            theta_deg=0.77,
            displacement_field_mev=48.0,
            shell_count=4,
        )
        interaction = RLGhBNInteractionParams(
            epsilon_r=float(epsilon_r),
            gate_distance_nm=10.0,
            scheme="average",
            active_valence_bands=3,
            active_conduction_bands=3,
            k_mesh_size=18,
            interaction_cutoff_q1=3.0,
            use_screened_basis=True,
        )
        cache = load_or_solve_screening(
            model,
            interaction,
            cache_dir=cache_dir,
            cache_policy=cache_policy,
            solver=screening_solver,
            mesh_size=18,
            u_min_mev=screening_u_min_mev,
            u_max_mev=screening_u_max_mev,
            n_grid=screening_u_grid_points,
            root_tolerance_mev=1.0e-5,
        )
        result = cache.value
        actual_u_mev = float(result.screened_u_mev)  # type: ignore[attr-defined]
        error_mev = actual_u_mev - float(expected_u_mev)
        checks.append(
            {
                "layer_count": 5,
                "xi": 0,
                "v_mev": 48.0,
                "epsilon_r": float(epsilon_r),
                "expected_screened_u_mev": float(expected_u_mev),
                "screened_u_mev": actual_u_mev,
                "error_mev": float(error_mev),
                "abs_error_mev": abs(float(error_mev)),
                "tolerance_mev": float(tolerance_mev),
                "passed": bool(abs(float(error_mev)) <= float(tolerance_mev)),
                "screening": screening_result_to_dict(result),  # type: ignore[arg-type]
                "screening_cache_key": str(cache.key),
                "cache_hit": bool(cache.hit),
                "cache_path": "" if cache.path is None else str(cache.path),
            }
        )
    return {
        "paper_target": "fig6",
        "description": "R5G xi=0 V=48 meV screened-U prereq checkpoints from 2312.11617v1 Appendix B5.",
        "host": socket.gethostname(),
        "cache_dir": str(Path(cache_dir).resolve()),
        "cache_policy": str(cache_policy),
        "screening_solver": str(screening_solver),
        "elapsed_sec": float(perf_counter() - start),
        "checks": checks,
        "passed": all(bool(item["passed"]) for item in checks),
    }


def main() -> None:
    args = _parse_args()
    ensure_not_running_compute_on_login_node("RLG/hBN Fig. 6 screening prereq validation")
    payload = validate_fig6_screening_checkpoints(
        cache_dir=Path(args.cache_dir).resolve(),
        cache_policy=str(args.cache_policy),
        screening_solver=str(args.screening_solver),
        screening_u_min_mev=float(args.screening_u_min_mev),
        screening_u_max_mev=float(args.screening_u_max_mev),
        screening_u_grid_points=int(args.screening_u_grid_points),
        tolerance_mev=float(args.tolerance_mev),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not bool(payload["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
