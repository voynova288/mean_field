#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import socket
import sys

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import load_projected_basis_cache, rlg_hbn_reference_density

# Reuse the battle-tested HF Chern helpers without writing into source task dirs.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from compute_rlg_hbn_hf_chern import (  # noqa: E402
    _compute_projected_basis_topology,
    _paper_expected_abs_chern,
    _panel_values,
    _read_json,
    _sector_indices,
    _selected_hf_occupation_on_grid,
    _selected_hf_wavefunctions_on_grid,
    _stats_payload,
    _string_from_archive,
    _topology_payload,
)

TASK_RE = re.compile(r"^task_(?P<task_id>\d+)_(?P<panel>xi-?\d+_V\d+meV)_(?P<init>.+)_seed(?P<seed>\d+)$")


def _discover_task_dirs(source_root: Path, task_ids: set[int] | None) -> list[Path]:
    rows: list[tuple[int, Path]] = []
    for task in (source_root / "tasks").glob("task_*"):
        match = TASK_RE.match(task.name)
        if match is None:
            continue
        task_id = int(match.group("task_id"))
        if task_ids is not None and task_id not in task_ids:
            continue
        panel = match.group("panel")
        panel_dir = task / panel
        if (task / "paper_hf_config.json").is_file() and (panel_dir / "hf_ground_state.npz").is_file() and (panel_dir / "hf_convergence.json").is_file():
            rows.append((task_id, task))
    return [task for _, task in sorted(rows)]


def _task_metadata(task_dir: Path) -> dict[str, object]:
    match = TASK_RE.match(task_dir.name)
    if match is None:
        return {"task_id": -1, "panel_from_task": "", "init_mode": "", "seed": -1}
    return {
        "task_id": int(match.group("task_id")),
        "panel_from_task": str(match.group("panel")),
        "init_mode": str(match.group("init")),
        "seed": int(match.group("seed")),
    }


def _compute_task(task_dir: Path, *, cache_dir: Path) -> list[dict[str, object]]:
    config = _read_json(task_dir / "paper_hf_config.json")
    panel_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir() and (path / "hf_ground_state.npz").exists())
    rows: list[dict[str, object]] = []
    task_meta = _task_metadata(task_dir)
    for panel_dir in panel_dirs:
        xi, v_mev = _panel_values(panel_dir.name)
        state_path = panel_dir / "hf_ground_state.npz"
        convergence_path = panel_dir / "hf_convergence.json"
        if not convergence_path.exists():
            continue
        archive = np.load(state_path)
        convergence = _read_json(convergence_path)
        best = convergence.get("best") or {}
        if not isinstance(best, dict):
            best = {}
        basis_key = str(convergence.get("basis_cache_key") or _string_from_archive(archive, "cache_key_basis"))
        if not basis_key:
            raise ValueError(f"{state_path} does not record a basis cache key")
        basis_data = load_projected_basis_cache(cache_dir, basis_key)
        hamiltonian = np.asarray(archive["hamiltonian"], dtype=np.complex128)
        density_delta = np.asarray(archive["density"], dtype=np.complex128)
        mesh_size = int(config["k_mesh_size"])
        n_spin = 2
        n_eta = 2
        n_band = int(hamiltonian.shape[0]) // (n_spin * n_eta)
        active_valence = int(config["active_valence_bands"])
        reference_density = rlg_hbn_reference_density(
            int(hamiltonian.shape[0]),
            int(hamiltonian.shape[2]),
            scheme=str(config.get("scheme", "average")),
            active_valence_bands=int(active_valence),
            n_spin=n_spin,
            n_eta=n_eta,
        )
        occupation_projector = density_delta + reference_density
        target_band = (active_valence,)
        paper_abs = _paper_expected_abs_chern(int(xi))
        for spin_index in range(n_spin):
            for eta_index in range(n_eta):
                sector = _sector_indices(
                    n_spin=n_spin,
                    n_eta=n_eta,
                    n_band=n_band,
                    spin=spin_index,
                    eta=eta_index,
                )
                basis_valley = int(basis_data.valleys[int(eta_index)])
                wavefunctions = _selected_hf_wavefunctions_on_grid(
                    hamiltonian=hamiltonian,
                    basis_wavefunctions=basis_data.basis.wavefunctions,
                    k_grid_frac=np.asarray(archive["k_grid_frac"], dtype=float),
                    mesh_size=mesh_size,
                    sector=sector,
                    eta=int(eta_index),
                    band_indices=target_band,
                )
                occ = _selected_hf_occupation_on_grid(
                    hamiltonian=hamiltonian,
                    occupation_projector=occupation_projector,
                    k_grid_frac=np.asarray(archive["k_grid_frac"], dtype=float),
                    mesh_size=mesh_size,
                    sector=sector,
                    band_indices=target_band,
                )
                result = _compute_projected_basis_topology(
                    wavefunctions,
                    band_indices=(0,),
                    local_basis_size=int(basis_data.basis.local_basis_size),
                    grid_shape=tuple(int(value) for value in basis_data.basis.grid_shape),
                    valley=basis_valley,
                    boundary_mode=str(basis_data.basis.boundary_mode),
                    sew_boundaries=True,
                )
                topo = _topology_payload(result)
                occ_stats = _stats_payload(occ)
                observed_abs = int(abs(result.rounded_chern_number))
                rows.append(
                    {
                        **task_meta,
                        "task": task_dir.name,
                        "panel": panel_dir.name,
                        "xi": int(xi),
                        "v_mev": float(v_mev),
                        "spin": int(spin_index),
                        "eta": int(eta_index),
                        "basis_valley": int(basis_valley),
                        "hf_sector_band_index": int(active_valence),
                        "occupation_min": float(occ_stats["min"]),
                        "occupation_mean": float(occ_stats["mean"]),
                        "occupation_max": float(occ_stats["max"]),
                        "chern": float(result.chern_number),
                        "rounded": int(result.rounded_chern_number),
                        "abs_rounded": int(observed_abs),
                        "paper_abs": int(paper_abs),
                        "matches_paper_abs": bool(observed_abs == int(paper_abs)),
                        "min_link": float(topo["min_link_singular_value"]),
                        "min_link_location": ":".join(str(x) for x in topo["min_link_location"]),
                        "iterations": best.get("iterations"),
                        "final_error": best.get("final_error"),
                        "final_energy_mev": best.get("final_energy_mev"),
                        "exit_reason": best.get("exit_reason"),
                        "state": str(state_path),
                    }
                )
    return rows


def _parse_task_ids(text: str | None) -> set[int] | None:
    if text is None or not text.strip():
        return None
    return {int(piece.strip()) for piece in text.split(",") if piece.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute all spin/valley sector Chern table for completed RLG/hBN HF tasks.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=str, default=None, help="Optional comma-separated task ids.")
    args = parser.parse_args()

    ensure_not_running_compute_on_login_node("RLG/hBN all-sector HF Chern diagnostic")
    source_root = args.source_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = _discover_task_dirs(source_root, _parse_task_ids(args.task_ids))
    if not task_dirs:
        raise FileNotFoundError(f"No completed task dirs found under {source_root}/tasks")

    rows: list[dict[str, object]] = []
    for task_dir in task_dirs:
        print(f"[task] {task_dir}", flush=True)
        rows.extend(_compute_task(task_dir, cache_dir=cache_dir))

    rows.sort(key=lambda row: (int(row["task_id"]), int(row["spin"]), int(row["eta"])))
    tsv_path = output_dir / "all_sector_occupied_conduction_chern.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        occupied_like = [row for row in task_rows if float(row["occupation_mean"]) > 0.5]
        best = max(task_rows, key=lambda row: float(row["occupation_mean"]))
        summary_rows.append(
            {
                "task": task,
                "occupied_like_sector_count": len(occupied_like),
                "occupied_like_matches": [bool(row["matches_paper_abs"]) for row in occupied_like],
                "best_occupation_sector": {k: best[k] for k in ("spin", "eta", "basis_valley", "occupation_mean", "chern", "rounded", "abs_rounded", "paper_abs", "matches_paper_abs")},
            }
        )

    summary = {
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "hostname": socket.gethostname(),
        "task_count": len(task_dirs),
        "row_count": len(rows),
        "output_tsv": str(tsv_path),
        "tasks": summary_rows,
    }
    summary_path = output_dir / "all_sector_occupied_conduction_chern_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = output_dir / "all_sector_occupied_conduction_chern_report.md"
    lines = [
        "# RLG/hBN completed-task all-sector occupied-conduction Chern diagnostic",
        "",
        f"- TSV: `{tsv_path}`",
        "",
        "| task | occupied-like sectors | best sector (spin,eta) | best occ | best C | |C| | paper | match |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary_rows:
        best = item["best_occupation_sector"]
        assert isinstance(best, dict)
        lines.append(
            f"| {item['task']} | {item['occupied_like_sector_count']} | ({best['spin']},{best['eta']}) | {float(best['occupation_mean']):.6f} | {float(best['chern']):.8f} | {best['abs_rounded']} | {best['paper_abs']} | {best['matches_paper_abs']} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] report={report}", flush=True)


if __name__ == "__main__":
    main()
