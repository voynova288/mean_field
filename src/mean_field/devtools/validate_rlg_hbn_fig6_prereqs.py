from __future__ import annotations

import argparse
import json
from pathlib import Path

from mean_field.api.validation import validate_fig6_screening_checkpoints
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "RnG_hBN" / "cache"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R5G/hBN screened-U checkpoints required before Fig. 6 HF runs.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--screening-solver", choices=("grid", "fixed_point"), default="grid")
    parser.add_argument("--screening-u-min-mev", type=float, default=-100.0)
    parser.add_argument("--screening-u-max-mev", type=float, default=200.0)
    parser.add_argument("--screening-u-grid-points", type=int, default=121)
    parser.add_argument("--tolerance-mev", type=float, default=3.0)
    return parser.parse_args()


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
