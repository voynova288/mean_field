from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from mean_field.devtools._runtime import write_json
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select lowest-energy RnG/hBN HF panel results from parallel array task outputs."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--paper-target", choices=tuple(PAPER_CONFIGS), required=True)
    parser.add_argument("--tasks-subdir", default="tasks")
    return parser.parse_args()


def _task_panel_dirs(tasks_root: Path) -> list[Path]:
    return sorted(
        path
        for path in tasks_root.glob("*/xi*_V*meV")
        if (path / "hf_convergence.json").exists() and (path / "hf_ground_state.npz").exists()
    )


def _best_energy(panel_dir: Path) -> float:
    convergence = _read_json(panel_dir / "hf_convergence.json")
    best = convergence.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"Missing best run payload in {panel_dir / 'hf_convergence.json'}")
    return float(best["final_energy_mev"])


def _expected_panels(paper_target: str) -> list[str]:
    config = PAPER_CONFIGS[paper_target]
    panels: list[str] = []
    for xi in tuple(int(value) for value in config["xi_values"]):
        for v_mev in tuple(float(value) for value in config["v_values_mev"]):
            panels.append(f"xi{xi}_V{int(round(v_mev)):03d}meV")
    return panels


def _copy_selected_panel(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("panel_config.json", "hf_convergence.json", "hf_ground_state.npz"):
        shutil.copy2(source / filename, destination / filename)


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).resolve()
    tasks_root = source_root / str(args.tasks_subdir)
    if not tasks_root.exists():
        raise FileNotFoundError(tasks_root)

    panel_dirs = _task_panel_dirs(tasks_root)
    grouped: dict[str, list[Path]] = {}
    for panel_dir in panel_dirs:
        grouped.setdefault(panel_dir.name, []).append(panel_dir)

    expected = _expected_panels(str(args.paper_target))
    missing = [panel for panel in expected if panel not in grouped]
    if missing:
        raise RuntimeError(f"Missing completed panel tasks for {missing}; found panels={sorted(grouped)}")

    selected_rows: list[dict[str, object]] = []
    for panel in expected:
        candidates = grouped[panel]
        selected = min(candidates, key=_best_energy)
        selected_energy = _best_energy(selected)
        destination = source_root / panel
        _copy_selected_panel(selected, destination)
        convergence = _read_json(selected / "hf_convergence.json")
        best = convergence.get("best")
        if not isinstance(best, dict):
            best = {}
        write_json(
            destination / "selection_metadata.json",
            {
                "selected_from": str(selected),
                "candidate_count": len(candidates),
                "selected_final_energy_mev": selected_energy,
                "selected_init_mode": best.get("init_mode"),
                "selected_seed": best.get("seed"),
            },
        )
        selected_rows.append(
            {
                "panel": panel,
                "selected_from": str(selected),
                "candidate_count": len(candidates),
                "selected_final_energy_mev": selected_energy,
                "selected_init_mode": best.get("init_mode"),
                "selected_seed": best.get("seed"),
                "all_candidates": [
                    {
                        "path": str(candidate),
                        "final_energy_mev": _best_energy(candidate),
                    }
                    for candidate in sorted(candidates)
                ],
            }
        )

    first_config_path = next(path for path in tasks_root.glob("*/paper_hf_config.json"))
    first_config = _read_json(first_config_path)
    base = dict(PAPER_CONFIGS[str(args.paper_target)])
    merged_config = {
        **base,
        "paper_target": str(args.paper_target),
        "init_modes": ["flavor", "bm", "perturbed"],
        "seeds": [1],
        "max_iter": int(first_config.get("max_iter", 80)),
        "precision": float(first_config.get("precision", 1.0e-6)),
        "beta": float(first_config.get("beta", 1.0)),
        "oda_stall_threshold": float(first_config.get("oda_stall_threshold", 1.0e-3)),
        "screening_mesh_size": first_config.get("screening_mesh_size"),
        "parallel_selection": True,
        "tasks_root": str(tasks_root),
    }
    write_json(source_root / "paper_hf_config.json", merged_config)
    write_json(
        source_root / "parallel_selection_summary.json",
        {
            "source_root": str(source_root),
            "paper_target": str(args.paper_target),
            "tasks_root": str(tasks_root),
            "selected": selected_rows,
        },
    )
    print(f"[merge] selected {len(selected_rows)} panels under {source_root}", flush=True)


if __name__ == "__main__":
    main()
