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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from compute_rlg_hbn_hf_chern import (  # noqa: E402
    _compute_projected_basis_topology,
    _grid_lookup,
    _panel_values,
    _read_json,
    _sector_indices,
    _selected_hf_occupation_on_grid,
    _selected_hf_wavefunctions_on_grid,
    _stats_payload,
    _string_from_archive,
)

TASK_RE = re.compile(r"^task_(?P<task_id>\d+)_(?P<panel>xi-?\d+_V\d+meV)_(?P<init>.+)_seed(?P<seed>\d+)$")


def _parse_task_ids(text: str | None) -> set[int] | None:
    if text is None or not text.strip():
        return None
    return {int(piece.strip()) for piece in text.split(",") if piece.strip()}


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


def _sector_energies_on_grid(hamiltonian: np.ndarray, k_grid_frac: np.ndarray, mesh_size: int, sector: np.ndarray) -> np.ndarray:
    lookup = _grid_lookup(k_grid_frac, mesh_size)
    n_band = int(sector.size)
    energies = np.zeros((int(mesh_size), int(mesh_size), n_band), dtype=float)
    for ix in range(int(mesh_size)):
        for iy in range(int(mesh_size)):
            ik = lookup[(ix, iy)]
            block = hamiltonian[:, :, int(ik)][np.ix_(sector, sector)]
            energies[ix, iy, :] = np.linalg.eigvalsh(block)
    return energies


def _band_gap_stats(energies: np.ndarray, band: int) -> dict[str, float | None]:
    vals = np.asarray(energies[:, :, int(band)], dtype=float)
    n_band = int(energies.shape[2])
    lower = None
    upper = None
    if int(band) > 0:
        lower = float(np.min(vals - energies[:, :, int(band) - 1]))
    if int(band) < n_band - 1:
        upper = float(np.min(energies[:, :, int(band) + 1] - vals))
    return {
        "energy_min_mev": float(np.min(vals)),
        "energy_mean_mev": float(np.mean(vals)),
        "energy_max_mev": float(np.max(vals)),
        "bandwidth_mev": float(np.max(vals) - np.min(vals)),
        "min_lower_direct_gap_mev": lower,
        "min_upper_direct_gap_mev": upper,
    }


def _compute_task(task_dir: Path, *, cache_dir: Path, flat_threshold_mev: float) -> list[dict[str, object]]:
    config = _read_json(task_dir / "paper_hf_config.json")
    task_meta = _task_metadata(task_dir)
    rows: list[dict[str, object]] = []
    for panel_dir in sorted(path for path in task_dir.iterdir() if path.is_dir() and (path / "hf_ground_state.npz").exists()):
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
        basis_data = load_projected_basis_cache(cache_dir, basis_key)
        hamiltonian = np.asarray(archive["hamiltonian"], dtype=np.complex128)
        k_grid_frac = np.asarray(archive["k_grid_frac"], dtype=float)
        density_delta = np.asarray(archive["density"], dtype=np.complex128)
        mesh_size = int(config["k_mesh_size"])
        n_spin = 2
        n_eta = 2
        n_band = int(hamiltonian.shape[0]) // (n_spin * n_eta)
        active_valence = int(config["active_valence_bands"])
        reference_density = rlg_hbn_reference_density(
            int(hamiltonian.shape[0]),
            int(hamiltonian.shape[2]),
            scheme=str(config.get("scheme", config.get("interaction_scheme", "average"))),
            active_valence_bands=active_valence,
            n_spin=n_spin,
            n_eta=n_eta,
        )
        occupation_projector = density_delta + reference_density
        central_bands = {active_valence - 1, active_valence}
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
                sector_energies = _sector_energies_on_grid(hamiltonian, k_grid_frac, mesh_size, sector)
                for band in range(n_band):
                    energy_stats = _band_gap_stats(sector_energies, band)
                    wavefunctions = _selected_hf_wavefunctions_on_grid(
                        hamiltonian=hamiltonian,
                        basis_wavefunctions=basis_data.basis.wavefunctions,
                        k_grid_frac=k_grid_frac,
                        mesh_size=mesh_size,
                        sector=sector,
                        eta=int(eta_index),
                        band_indices=(band,),
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
                    occ = _selected_hf_occupation_on_grid(
                        hamiltonian=hamiltonian,
                        occupation_projector=occupation_projector,
                        k_grid_frac=k_grid_frac,
                        mesh_size=mesh_size,
                        sector=sector,
                        band_indices=(band,),
                    )
                    occ_stats = _stats_payload(occ)
                    is_remote = int(band) not in central_bands
                    is_flat = bool(float(energy_stats["bandwidth_mev"]) <= float(flat_threshold_mev))
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
                            "active_band": int(band),
                            "relative_band_label": int(band - active_valence),
                            "is_central_pair_band": bool(int(band) in central_bands),
                            "is_remote_band": bool(is_remote),
                            "remote_flat_candidate": bool(is_remote and is_flat),
                            **energy_stats,
                            "occupation_min": float(occ_stats["min"]),
                            "occupation_mean": float(occ_stats["mean"]),
                            "occupation_max": float(occ_stats["max"]),
                            "chern": float(result.chern_number),
                            "rounded": int(result.rounded_chern_number),
                            "abs_rounded": int(abs(result.rounded_chern_number)),
                            "integer_residual": float(result.integer_residual),
                            "min_link": float(result.min_link_singular_value),
                            "min_link_location": ":".join(str(x) for x in result.min_link_location),
                            "iterations": best.get("iterations"),
                            "final_error": best.get("final_error"),
                            "final_energy_mev": best.get("final_energy_mev"),
                            "exit_reason": best.get("exit_reason"),
                            "state": str(state_path),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Chern/bandwidth table for all active HF bands in completed RLG/hBN tasks.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--flat-threshold-mev", type=float, default=20.0)
    args = parser.parse_args()

    ensure_not_running_compute_on_login_node("RLG/hBN all-active-band Chern diagnostic")
    source_root = args.source_root.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = _discover_task_dirs(source_root, _parse_task_ids(args.task_ids))
    if not task_dirs:
        raise FileNotFoundError(f"No completed tasks found under {source_root}/tasks for task_ids={args.task_ids}")
    rows: list[dict[str, object]] = []
    for task_dir in task_dirs:
        print(f"[task] {task_dir}", flush=True)
        rows.extend(_compute_task(task_dir, cache_dir=cache_dir, flat_threshold_mev=float(args.flat_threshold_mev)))
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["spin"]), int(row["eta"]), int(row["active_band"])))

    tsv_path = output_dir / "active_band_chern_bandwidth.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    candidates = [row for row in rows if bool(row["remote_flat_candidate"])]
    # Prefer sectors with nonzero occupation, but report all remote flat candidates.
    summary_path = output_dir / "active_band_chern_bandwidth_summary.json"
    summary = {
        "source_root": str(source_root),
        "cache_dir": str(cache_dir),
        "hostname": socket.gethostname(),
        "task_ids": None if args.task_ids is None else sorted(_parse_task_ids(args.task_ids) or []),
        "flat_threshold_mev": float(args.flat_threshold_mev),
        "row_count": len(rows),
        "remote_flat_candidate_count": len(candidates),
        "output_tsv": str(tsv_path),
        "remote_flat_candidates": candidates,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = output_dir / "active_band_chern_bandwidth_report.md"
    lines = [
        "# RLG/hBN active-band Chern/bandwidth diagnostic",
        "",
        f"- TSV: `{tsv_path}`",
        f"- Remote-flat threshold: `{float(args.flat_threshold_mev):g}` meV",
        "",
        "## Remote flat candidates",
        "",
        "| task | panel | spin | eta | band | rel | bw meV | C | |C| | occ mean | lower gap | upper gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if candidates:
        for row in candidates:
            lower = row["min_lower_direct_gap_mev"]
            upper = row["min_upper_direct_gap_mev"]
            lines.append(
                f"| {row['task']} | {row['panel']} | {row['spin']} | {row['eta']} | {row['active_band']} | {row['relative_band_label']} | "
                f"{float(row['bandwidth_mev']):.6g} | {float(row['chern']):.8f} | {row['abs_rounded']} | {float(row['occupation_mean']):.6f} | "
                f"{'' if lower is None else f'{float(lower):.6g}'} | {'' if upper is None else f'{float(upper):.6g}'} |"
            )
    else:
        lines.append("| none | | | | | | | | | | | |")
    lines.extend([
        "",
        "## All active bands",
        "",
        "| task | panel | spin | eta | band | rel | bw meV | C | |C| | occ mean | remote? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['panel']} | {row['spin']} | {row['eta']} | {row['active_band']} | {row['relative_band_label']} | "
            f"{float(row['bandwidth_mev']):.6g} | {float(row['chern']):.8f} | {row['abs_rounded']} | {float(row['occupation_mean']):.6f} | {row['is_remote_band']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
