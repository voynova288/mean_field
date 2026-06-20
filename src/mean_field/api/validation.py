from __future__ import annotations

from pathlib import Path
import socket
from time import perf_counter


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
    """Validate the R5G/hBN screened-U checkpoints used before Fig. 6 HF runs."""

    from mean_field.systems.RnG_hBN import (  # noqa: PLC0415
        RLGhBNInteractionParams,
        RLGhBNModel,
        load_or_solve_screening,
        screening_result_to_dict,
    )

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
        cache = load_or_solve_screening(
            model,
            RLGhBNInteractionParams(
                epsilon_r=float(epsilon_r),
                gate_distance_nm=10.0,
                scheme="average",
                active_valence_bands=3,
                active_conduction_bands=3,
                k_mesh_size=18,
                interaction_cutoff_q1=3.0,
                use_screened_basis=True,
            ),
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
                "error_mev": error_mev,
                "abs_error_mev": abs(error_mev),
                "tolerance_mev": float(tolerance_mev),
                "passed": bool(abs(error_mev) <= float(tolerance_mev)),
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


__all__ = ["validate_fig6_screening_checkpoints"]
