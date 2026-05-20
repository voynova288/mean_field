from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from mean_field.devtools._runtime import write_json
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS, default_rlg_hbn_run_specs


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


def _best_run_spec(panel_dir: Path) -> tuple[str, int] | None:
    convergence = _read_json(panel_dir / "hf_convergence.json")
    best = convergence.get("best")
    if not isinstance(best, dict):
        return None
    init_mode = best.get("init_mode")
    seed = best.get("seed")
    if init_mode is None or seed is None:
        return None
    return str(init_mode), int(seed)


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
    for filename in ("screening_result.json",):
        source_path = source / filename
        if source_path.exists():
            shutil.copy2(source_path, destination / filename)


def _merge_cache_manifests(tasks_root: Path) -> dict[str, object]:
    merged: dict[str, object] = {"cache_dir": "", "entries": [], "summary": {}}
    entries = merged["entries"]
    summary = merged["summary"]
    if not isinstance(entries, list) or not isinstance(summary, dict):
        raise TypeError("Internal cache manifest accumulator has unexpected type.")

    for manifest_path in sorted(tasks_root.glob("*/cache_manifest.json")):
        payload = _read_json(manifest_path)
        if not merged["cache_dir"] and payload.get("cache_dir"):
            merged["cache_dir"] = str(payload["cache_dir"])
        task_name = manifest_path.parent.name
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            continue
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            entry["source_manifest"] = str(manifest_path)
            entry["source_task"] = task_name
            entries.append(entry)
            kind = str(entry.get("kind", "unknown"))
            kind_summary = summary.setdefault(kind, {"hit": 0, "miss": 0})
            if not isinstance(kind_summary, dict):
                kind_summary = {"hit": 0, "miss": 0}
                summary[kind] = kind_summary
            bucket = "hit" if bool(entry.get("hit", False)) else "miss"
            kind_summary[bucket] = int(kind_summary.get(bucket, 0)) + 1
    return merged


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).resolve()
    tasks_root = source_root / str(args.tasks_subdir)
    if not tasks_root.exists():
        raise FileNotFoundError(tasks_root)

    allowed_specs = set(default_rlg_hbn_run_specs(str(args.paper_target)))
    all_panel_dirs = _task_panel_dirs(tasks_root)
    ignored_panel_dirs: list[dict[str, object]] = []
    panel_dirs: list[Path] = []
    for panel_dir in all_panel_dirs:
        spec = _best_run_spec(panel_dir)
        if spec is not None and spec not in allowed_specs:
            ignored_panel_dirs.append(
                {
                    "path": str(panel_dir),
                    "init_mode": spec[0],
                    "seed": spec[1],
                    "reason": "not in deduplicated paper-target run specs",
                }
            )
            continue
        panel_dirs.append(panel_dir)
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
    run_specs = default_rlg_hbn_run_specs(str(args.paper_target))
    init_modes = list(dict.fromkeys(init_mode for init_mode, _ in run_specs))
    seeds = list(dict.fromkeys(int(seed) for _, seed in run_specs))
    merged_config = {
        **base,
        "paper_target": str(args.paper_target),
        "init_modes": init_modes,
        "seeds": seeds,
        "run_specs": [
            {"init_mode": str(init_mode), "seed": int(seed)}
            for init_mode, seed in run_specs
        ],
        "candidate_count": len(run_specs),
        "max_iter": int(first_config.get("max_iter", 80)),
        "precision": float(first_config.get("precision", 1.0e-6)),
        "beta": float(first_config.get("beta", 1.0)),
        "oda_stall_threshold": float(first_config.get("oda_stall_threshold", 1.0e-3)),
        "screening_mesh_size": first_config.get("screening_mesh_size"),
        "screening_solver": first_config.get("screening_solver"),
        "screening_u_min_mev": first_config.get("screening_u_min_mev"),
        "screening_u_max_mev": first_config.get("screening_u_max_mev"),
        "screening_u_grid_points": first_config.get("screening_u_grid_points"),
        "cache_dir": first_config.get("cache_dir"),
        "cache_policy": first_config.get("cache_policy"),
        "skip_screening_check": first_config.get("skip_screening_check", True),
        "use_screened_basis": first_config.get("use_screened_basis", base.get("use_screened_basis", True)),
        "parallel_selection": True,
        "tasks_root": str(tasks_root),
    }
    write_json(source_root / "paper_hf_config.json", merged_config)
    write_json(source_root / "cache_manifest.json", _merge_cache_manifests(tasks_root))
    write_json(
        source_root / "parallel_selection_summary.json",
        {
            "source_root": str(source_root),
            "paper_target": str(args.paper_target),
            "tasks_root": str(tasks_root),
            "allowed_run_specs": [
                {"init_mode": str(init_mode), "seed": int(seed)}
                for init_mode, seed in sorted(allowed_specs)
            ],
            "ignored_candidates": ignored_panel_dirs,
            "selected": selected_rows,
        },
    )
    print(f"[merge] selected {len(selected_rows)} panels under {source_root}", flush=True)


if __name__ == "__main__":
    main()
