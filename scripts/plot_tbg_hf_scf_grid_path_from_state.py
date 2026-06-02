#!/usr/bin/env python3
"""Plot TBG HF bands using only SCF grid points that lie on the k-path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mean_field.devtools.run_custom_b0_hf_case import _build_path
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.hf_runners import (
    build_restricted_hf_scf_path_plot_result,
    write_hf_scf_path_tsv,
)
from mean_field.systems.tbg.zero_field.model import build_b0_uniform_lattice
from mean_field.systems.tbg.zero_field.plotting import write_hf_scf_band_plot


def _scalar(npz, key: str, default=None):
    if key not in npz.files:
        return default
    value = np.asarray(npz[key])
    if value.size == 0:
        return default
    return value.reshape(-1)[0].item()


def _read_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _state_paths_from_roots(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
            continue
        paths.extend(sorted(root.glob("**/states/*.npz")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _run_dir_from_state_path(state_path: Path) -> Path:
    if state_path.parent.name == "states":
        return state_path.parent.parent
    return state_path.parent


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_state(
    state_path: Path,
    *,
    output_dir: Path | None,
    path_kind_override: str | None,
    points_per_segment_override: int | None,
    path_tolerance: float,
) -> dict[str, object]:
    state_path = state_path.resolve()
    run_dir = _run_dir_from_state_path(state_path)
    run_info = _read_kv(run_dir / "run_info.txt")
    with np.load(state_path, allow_pickle=False) as npz:
        theta_deg = float(_scalar(npz, "theta_deg"))
        lk = int(_scalar(npz, "lk"))
        lg = int(_scalar(npz, "lg"))
        params = TBGParameters(
            dtheta_rad=float(np.deg2rad(theta_deg)),
            vf=float(_scalar(npz, "vf_mev")),
            w0=float(_scalar(npz, "w0_mev")),
            w1=float(_scalar(npz, "w1_mev")),
        )
        hamiltonian = np.asarray(npz["hamiltonian"], dtype=np.complex128)
        nt = int(hamiltonian.shape[0])
        if nt % 4 != 0:
            raise ValueError(f"Expected Hamiltonian dimension divisible by 4, got {nt} in {state_path}")
        state = SimpleNamespace(
            hamiltonian=hamiltonian,
            n_spin=2,
            n_eta=2,
            n_band=nt // 4,
            mu=float(_scalar(npz, "mu")),
            nu=float(_scalar(npz, "nu")),
        )
        hf_run = SimpleNamespace(
            state=state,
            init_mode=str(_scalar(npz, "normalized_init_mode", _scalar(npz, "init_mode", ""))),
            seed=int(_scalar(npz, "seed", 0)),
            exit_reason=str(_scalar(npz, "exit_reason", "")),
        )
        requested_init_mode = str(_scalar(npz, "init_mode", hf_run.init_mode))

    grid = build_b0_uniform_lattice(params, lk)
    grid_solution = SimpleNamespace(
        params=params,
        lattice_kvec=np.asarray(grid.kvec, dtype=np.complex128),
        nk=int(grid.nk),
        lg=int(lg),
    )
    path_kind = str(path_kind_override or run_info.get("path_kind", "gamma-m-k-gamma-kprime"))
    points_per_segment = int(points_per_segment_override or run_info.get("points_per_segment", "120"))
    path = _build_path(params, path_kind=path_kind, points_per_segment=points_per_segment)

    result = build_restricted_hf_scf_path_plot_result(
        hf_run,
        grid_solution,
        path=path,
        init_mode=requested_init_mode,
        path_tolerance=float(path_tolerance),
    )
    if result.kdist.size == 0:
        raise ValueError(f"No exact SCF grid points were found on {path_kind} for {state_path}")

    out = output_dir if output_dir is not None else run_dir / "path_bands"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{state_path.stem}_scf_grid_band_plot"
    tsv_path = out / f"{state_path.stem}_hf_scf_path.tsv"
    write_hf_scf_path_tsv(tsv_path, result)
    plot_paths = write_hf_scf_band_plot(out, result, stem=stem)

    summary = {
        "source_state": str(state_path),
        "run_dir": str(run_dir),
        "approximation": "SCF grid points on selected path only; no off-grid path Hamiltonian, nearest mapping, or interpolation",
        "path_kind": path_kind,
        "points_per_segment": points_per_segment,
        "path_tolerance": float(path_tolerance),
        "scf_path_point_count": int(result.kdist.size),
        "grid_point_count": int(grid.nk),
        "lk": int(lk),
        "lg": int(lg),
        "mu_mev": float(result.mu),
        "tsv": str(tsv_path),
        "png": str(plot_paths["band_plot_png"]),
        "pdf": str(plot_paths["band_plot_pdf"]),
    }
    summary_path = out / f"{state_path.stem}_scf_grid_band_plot_summary.json"
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=[], help="Result root or state npz to process.")
    parser.add_argument("--state", type=Path, action="append", default=[], help="Specific state npz to process.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory for all plots.")
    parser.add_argument("--path-kind", default=None, help="Override path kind. Defaults to run_info.txt or gamma-m-k-gamma-kprime.")
    parser.add_argument("--points-per-segment", type=int, default=None, help="Override path resolution.")
    parser.add_argument("--path-tolerance", type=float, default=1.0e-12, help="Exact-hit tolerance in code momentum units.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    states = list(args.state) + _state_paths_from_roots(tuple(args.root))
    if not states:
        raise SystemExit("Pass at least one --state or --root.")
    reports = []
    for state in states:
        report = plot_state(
            state,
            output_dir=args.output_dir,
            path_kind_override=args.path_kind,
            points_per_segment_override=args.points_per_segment,
            path_tolerance=float(args.path_tolerance),
        )
        reports.append(report)
        print(f"[scf-grid-plot] {report['png']}")
    print(json.dumps({"count": len(reports), "plots": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
