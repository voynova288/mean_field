from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.hf import FlavorBandData
from mean_field.systems.tbg.zero_field.hf_runners import HFSCFPathPlotResult, write_hf_scf_path_tsv
from mean_field.systems.tbg.zero_field.path import build_kpath_from_reference_nodes, project_kvec_onto_path
from mean_field.systems.tbg.zero_field.plotting import write_hf_scf_band_plot


def _load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        summary[key.strip()] = value.strip()
    return summary


def _load_path(path: Path):
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        nodes = [
            SimpleNamespace(
                label=row["label"],
                index=int(row["index"]),
                kvec=complex(float(row["kx"]), float(row["ky"])),
            )
            for row in reader
        ]
    return build_kpath_from_reference_nodes(nodes)


def _load_legacy_scf_path(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        if "grid_ky" not in fieldnames:
            raise ValueError(f"Expected a grid_ky column in {path}.")
        energy_start = fieldnames.index("grid_ky") + 1
        band_labels = fieldnames[energy_start:]

        path_indices: list[int] = []
        path_kvec: list[complex] = []
        grid_indices: list[int] = []
        grid_kvec: list[complex] = []
        energies: list[list[float]] = []

        for row in reader:
            path_indices.append(int(row["path_index"]) - 1)
            path_kvec.append(complex(float(row["path_kx"]), float(row["path_ky"])))
            grid_indices.append(int(row["grid_index"]) - 1)
            grid_kvec.append(complex(float(row["grid_kx"]), float(row["grid_ky"])))
            energies.append([float(row[label]) for label in band_labels])

    return (
        np.asarray(path_indices, dtype=int),
        np.asarray(path_kvec, dtype=np.complex128),
        np.asarray(grid_indices, dtype=int),
        np.asarray(grid_kvec, dtype=np.complex128),
        tuple(str(label) for label in band_labels),
        np.asarray(energies, dtype=float).T,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair legacy SCF-grid-only HF artifacts by keeping only exact on-path SCF points.")
    parser.add_argument("result_dir", type=Path, help="Benchmark leaf directory containing computed_nodes.tsv and computed_hf_scf_path.tsv")
    parser.add_argument(
        "--stem",
        default="band_plot_scf_grid",
        help="Plot stem to rewrite. Default keeps the reconstructed main band_plot untouched.",
    )
    parser.add_argument("--path-tolerance", type=float, default=1e-12, help="Maximum distance from path allowed for SCF points.")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    nodes_path = result_dir / "computed_nodes.tsv"
    summary_path = result_dir / "computed_summary.txt"
    scf_path_tsv = result_dir / "computed_hf_scf_path.tsv"

    path = _load_path(nodes_path)
    summary = _load_summary(summary_path)
    path_indices, path_kvec, grid_indices, grid_kvec, band_labels, energies = _load_legacy_scf_path(scf_path_tsv)
    projected_kdist_all, projected_kvec_all, distance_to_path_all = project_kvec_onto_path(path, grid_kvec)

    selected_positions: list[int] = []
    seen_grid_indices: set[int] = set()
    tolerance = float(args.path_tolerance)
    for pos in np.argsort(projected_kdist_all, kind="stable"):
        if float(distance_to_path_all[pos]) > tolerance:
            continue
        grid_index = int(grid_indices[pos])
        if grid_index in seen_grid_indices:
            continue
        seen_grid_indices.add(grid_index)
        selected_positions.append(int(pos))

    selected = np.asarray(selected_positions, dtype=int)
    projected_kdist = projected_kdist_all[selected]
    projected_kvec = projected_kvec_all[selected]
    distance_to_path = distance_to_path_all[selected]
    grid_indices = grid_indices[selected]
    grid_kvec = grid_kvec[selected]
    energies = energies[:, selected]
    if selected.size == 0:
        path_indices = np.asarray([], dtype=int)
        path_kvec = np.asarray([], dtype=np.complex128)
    else:
        distance_matrix = np.abs(path.kvec[:, None] - projected_kvec[None, :])
        path_indices = np.argmin(distance_matrix, axis=0).astype(int)
        path_kvec = np.asarray(projected_kvec, dtype=np.complex128)

    theta_deg = float(summary["theta_deg"])
    params = TBGParameters.from_degrees(theta_deg)
    plot_result = HFSCFPathPlotResult(
        params=params,
        path=path,
        kdist=np.asarray(projected_kdist, dtype=float),
        projected_kvec=np.asarray(projected_kvec, dtype=np.complex128),
        distance_to_path=np.asarray(distance_to_path, dtype=float),
        path_sample_indices=np.asarray(path_indices, dtype=int),
        path_kvec=np.asarray(path_kvec, dtype=np.complex128),
        grid_kvec=np.asarray(grid_kvec, dtype=np.complex128),
        grid_indices=np.asarray(grid_indices, dtype=int),
        band_data=FlavorBandData(
            band_labels=tuple(band_labels),
            energies=np.asarray(energies, dtype=float),
            mean_weights=np.zeros((len(band_labels), 0), dtype=float),
        ),
        mu=float(summary["mu"]),
        nu=float(summary["nu"]),
        lk=int(summary["lk"]),
        lg=int(summary["lg"]),
        init_mode=summary["init_mode"],
        normalized_init_mode=summary["normalized_init_mode"],
        seed=int(summary["seed"]),
        exit_reason=summary["exit_reason"],
    )

    write_hf_scf_path_tsv(scf_path_tsv, plot_result)
    artifacts = write_hf_scf_band_plot(result_dir, plot_result, stem=args.stem)
    print(f"[repair] rewrote {scf_path_tsv}")
    print(f"[repair] rewrote {artifacts['band_plot_png']}")
    print(f"[repair] rewrote {artifacts['band_plot_pdf']}")
    max_distance = float(np.max(distance_to_path)) if distance_to_path.size else float("nan")
    print(f"[repair] kept_points={selected.size}")
    print(f"[repair] max_distance_to_path={max_distance:.12f}")


if __name__ == "__main__":
    main()
