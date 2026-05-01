from __future__ import annotations

import argparse
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_DEFAULTS = {
    "b0": REPO_ROOT / "benchmarks" / "b0",
    "bm-unstrained": REPO_ROOT / "benchmarks" / "b0" / "bm_inputs" / "unstrained_path",
}


def sync_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy curated benchmark trees into this repository.")
    parser.add_argument(
        "target",
        choices=tuple(TARGET_DEFAULTS),
        help="Benchmark tree to refresh inside the repository.",
    )
    parser.add_argument("--source", required=True, help="Source benchmark directory.")
    parser.add_argument(
        "--destination",
        default=None,
        help="Optional destination override. Defaults to the standard path for the selected target.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    destination = Path(args.destination).resolve() if args.destination else TARGET_DEFAULTS[args.target]
    sync_tree(source, destination)
    print(f"target={args.target}")
    print(f"source={source}")
    print(f"destination={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

