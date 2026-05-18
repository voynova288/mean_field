from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from time import perf_counter

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS
from mean_field.systems.RnG_hBN import (
    RLGhBNInteractionParams,
    RLGhBNModel,
    load_or_build_layer_overlap_blocks,
    load_or_build_projected_basis,
    load_or_solve_screening,
    screening_result_to_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "RnG_hBN" / "cache_fig6"


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm RLG/hBN Fig. 6 screening, basis, and overlap caches.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--screening-solver", choices=("grid", "fixed_point"), default="grid")
    parser.add_argument("--screening-u-min-mev", type=float, default=-100.0)
    parser.add_argument("--screening-u-max-mev", type=float, default=200.0)
    parser.add_argument("--screening-u-grid-points", type=int, default=121)
    parser.add_argument("--xi-values", type=_parse_csv_ints, default=(0, 1))
    parser.add_argument("--v-values-mev", type=_parse_csv_floats, default=(64.0,))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_not_running_compute_on_login_node("RLG/hBN Fig. 6 cache warmup")
    start = perf_counter()
    cache_dir = Path(args.cache_dir).resolve()
    if args.cache_policy != "off":
        cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = None if args.output_dir is None else Path(args.output_dir).resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = PAPER_CONFIGS["fig6"]
    rows: list[dict[str, object]] = []
    for xi in tuple(int(value) for value in args.xi_values):
        for v_mev in tuple(float(value) for value in args.v_values_mev):
            panel_start = perf_counter()
            panel = f"xi{xi}_V{int(round(v_mev)):03d}meV"
            print(f"[warm] start {panel}", flush=True)
            model = RLGhBNModel.from_config(
                layer_count=int(config["layer_count"]),
                xi=int(xi),
                theta_deg=float(config["theta_deg"]),
                displacement_field_mev=float(v_mev),
                shell_count=int(config["shell_count"]),
            )
            interaction = RLGhBNInteractionParams(
                epsilon_r=float(config["epsilon_r"]),
                gate_distance_nm=float(config["gate_distance_nm"]),
                scheme=str(config["scheme"]),
                active_valence_bands=int(config["active_valence_bands"]),
                active_conduction_bands=int(config["active_conduction_bands"]),
                k_mesh_size=int(config["k_mesh_size"]),
                interaction_cutoff_q1=float(config["interaction_cutoff_q1"]),
                use_screened_basis=True,
            )
            screening = load_or_solve_screening(
                model,
                interaction,
                cache_dir=cache_dir,
                cache_policy=str(args.cache_policy),
                solver=str(args.screening_solver),
                mesh_size=int(config["k_mesh_size"]),
                u_min_mev=float(args.screening_u_min_mev),
                u_max_mev=float(args.screening_u_max_mev),
                n_grid=int(args.screening_u_grid_points),
                root_tolerance_mev=1.0e-5,
            )
            basis = load_or_build_projected_basis(
                model,
                interaction,
                cache_dir=cache_dir,
                cache_policy=str(args.cache_policy),
                mesh_size=int(config["k_mesh_size"]),
                screening=screening.value,  # type: ignore[arg-type]
                screening_solver=str(args.screening_solver),
                screening_mesh_size=int(config["k_mesh_size"]),
                screening_u_min_mev=float(args.screening_u_min_mev),
                screening_u_max_mev=float(args.screening_u_max_mev),
                screening_u_grid_points=int(args.screening_u_grid_points),
            )
            overlap = load_or_build_layer_overlap_blocks(
                basis.value,  # type: ignore[arg-type]
                cache_dir=cache_dir,
                cache_policy=str(args.cache_policy),
                basis_cache_key=str(basis.key),
            )
            row = {
                "panel": panel,
                "xi": int(xi),
                "v_mev": float(v_mev),
                "elapsed_sec": float(perf_counter() - panel_start),
                "screening_cache_key": str(screening.key),
                "screening_cache_hit": bool(screening.hit),
                "screening": screening_result_to_dict(screening.value),  # type: ignore[arg-type]
                "basis_cache_key": str(basis.key),
                "basis_cache_hit": bool(basis.hit),
                "overlap_cache_key": str(overlap.key),
                "overlap_cache_hit": bool(overlap.hit),
            }
            rows.append(row)
            print(
                f"[warm] done {panel} screening_hit={screening.hit} "
                f"basis_hit={basis.hit} overlap_hit={overlap.hit}",
                flush=True,
            )

    payload = {
        "paper_target": "fig6",
        "host": socket.gethostname(),
        "cache_dir": str(cache_dir),
        "cache_policy": str(args.cache_policy),
        "screening_solver": str(args.screening_solver),
        "elapsed_sec": float(perf_counter() - start),
        "panels": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if output_dir is not None:
        (output_dir / "cache_warmup_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
