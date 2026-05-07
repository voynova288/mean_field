#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.hf import FlavorBandData
from mean_field.systems.tbg.zero_field.hf_runners import HFSCFPathPlotResult, write_hf_scf_path_tsv
from mean_field.systems.tbg.zero_field.path import project_kvec_onto_path
from mean_field.systems.tbg.zero_field.path_advisor import rank_kpath_candidates_for_lk
from mean_field.systems.tbg.zero_field.plotting import write_hf_scf_band_plot, write_path_band_plot


def _load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        summary[key.strip()] = value.strip()
    return summary


def _write_nodes_tsv(path: Path, *, labels: tuple[str, ...], node_indices: tuple[int, ...], kvec: np.ndarray, kdist: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("label\tindex\tk_dist\tkx\tky\n")
        for label, node_index in zip(labels, node_indices, strict=True):
            idx = int(node_index) - 1
            kval = complex(kvec[idx])
            handle.write(
                f"{label}\t{node_index}\t{float(kdist[idx]):.16f}\t{float(kval.real):.16f}\t{float(kval.imag):.16f}\n"
            )


def _load_hf_path(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    band_labels = tuple(str(value) for value in header[1:])
    kdist = np.asarray([float(row[0]) for row in rows], dtype=float)
    energies = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    return kdist, energies, band_labels


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


def _write_repair_metadata(path: Path, *, candidate_rank: int, compatibility: object) -> None:
    candidate = compatibility.candidate
    lines = [
        ("path_source", "advisor_ranked_candidate"),
        ("candidate_rank", str(int(candidate_rank))),
        ("candidate_name", str(candidate.name)),
        ("candidate_family", str(candidate.family)),
        ("lk", str(int(compatibility.lk))),
        ("exact_count", str(int(compatibility.exact_count))),
        ("exact_node_hit_count", str(int(compatibility.exact_node_hit_count))),
        ("exact_segment_counts", ",".join(str(value) for value in compatibility.exact_segment_counts)),
        ("mean_nearest_distance", f"{float(compatibility.mean_nearest_distance):.16e}"),
        ("max_nearest_distance", f"{float(compatibility.max_nearest_distance):.16e}"),
        ("m_real", f"{float(candidate.m_point.real):.16f}"),
        ("m_imag", f"{float(candidate.m_point.imag):.16f}"),
        ("k_real", f"{float(candidate.k_point.real):.16f}"),
        ("k_imag", f"{float(candidate.k_point.imag):.16f}"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for key, value in lines:
            handle.write(f"{key}={value}\n")


def repair_result_dir(result_dir: Path, *, candidate_rank: int, path_tolerance: float) -> None:
    result_dir = result_dir.resolve()
    summary = _load_summary(result_dir / "computed_summary.txt")
    theta_deg = float(summary["theta_deg"])
    lk = int(summary["lk"])
    points_per_segment = int(summary["points_per_segment"])
    params = TBGParameters.from_degrees(theta_deg)
    ranked = rank_kpath_candidates_for_lk(
        params,
        lk=lk,
        points_per_segment=points_per_segment,
    )
    if not ranked:
        raise ValueError(f"No advisor candidates available for theta={theta_deg:.2f}, lk={lk}.")
    if candidate_rank < 1 or candidate_rank > len(ranked):
        raise ValueError(f"candidate_rank={candidate_rank} out of range for {result_dir}.")

    compatibility = ranked[candidate_rank - 1]
    path = compatibility.candidate.path
    _write_nodes_tsv(
        result_dir / "computed_nodes.tsv",
        labels=tuple(path.labels),
        node_indices=tuple(path.node_indices),
        kvec=np.asarray(path.kvec, dtype=np.complex128),
        kdist=np.asarray(path.kdist, dtype=float),
    )
    _write_repair_metadata(result_dir / "advisor_path_selection.txt", candidate_rank=candidate_rank, compatibility=compatibility)

    title = (
        f"theta={float(summary['theta_deg']):.2f}°, "
        f"nu={float(summary['nu']):g}, "
        f"init={summary['init_mode']}, "
        f"seed={int(summary['seed'])}"
    )

    hf_path_tsv = result_dir / "computed_hf_path.tsv"
    if hf_path_tsv.is_file():
        kdist, energies, band_labels = _load_hf_path(hf_path_tsv)
        artifacts = write_path_band_plot(
            result_dir,
            stem="band_plot",
            kdist=kdist,
            energies=energies,
            path=path,
            band_labels=band_labels,
            mu=float(summary["mu"]),
            title=title,
        )
        print(f"[repair] rewrote {artifacts['band_plot_png']}")
        print(f"[repair] rewrote {artifacts['band_plot_pdf']}")

    scf_path_tsv = result_dir / "computed_hf_scf_path.tsv"
    if scf_path_tsv.is_file():
        _, _, grid_indices, grid_kvec, band_labels, energies = _load_legacy_scf_path(scf_path_tsv)
        projected_kdist_all, projected_kvec_all, distance_to_path_all = project_kvec_onto_path(path, grid_kvec)

        selected_positions: list[int] = []
        seen_grid_indices: set[int] = set()
        for pos in np.argsort(projected_kdist_all, kind="stable"):
            if float(distance_to_path_all[pos]) > path_tolerance:
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
        selected_grid_indices = grid_indices[selected]
        selected_grid_kvec = grid_kvec[selected]
        selected_energies = energies[:, selected]
        if selected.size == 0:
            path_indices = np.asarray([], dtype=int)
            path_kvec = np.asarray([], dtype=np.complex128)
        else:
            distance_matrix = np.abs(path.kvec[:, None] - projected_kvec[None, :])
            path_indices = np.argmin(distance_matrix, axis=0).astype(int)
            path_kvec = np.asarray(projected_kvec, dtype=np.complex128)

        plot_result = HFSCFPathPlotResult(
            params=params,
            path=path,
            kdist=np.asarray(projected_kdist, dtype=float),
            projected_kvec=np.asarray(projected_kvec, dtype=np.complex128),
            distance_to_path=np.asarray(distance_to_path, dtype=float),
            path_sample_indices=np.asarray(path_indices, dtype=int),
            path_kvec=np.asarray(path_kvec, dtype=np.complex128),
            grid_kvec=np.asarray(selected_grid_kvec, dtype=np.complex128),
            grid_indices=np.asarray(selected_grid_indices, dtype=int),
            band_data=FlavorBandData(
                band_labels=tuple(band_labels),
                energies=np.asarray(selected_energies, dtype=float),
                mean_weights=np.zeros((len(band_labels), 0), dtype=float),
            ),
            mu=float(summary["mu"]),
            nu=float(summary["nu"]),
            lk=lk,
            lg=int(summary["lg"]),
            init_mode=summary["init_mode"],
            normalized_init_mode=summary["normalized_init_mode"],
            seed=int(summary["seed"]),
            exit_reason=summary["exit_reason"],
        )

        write_hf_scf_path_tsv(scf_path_tsv, plot_result)
        artifacts = write_hf_scf_band_plot(result_dir, plot_result, stem="band_plot_scf_grid")
        print(f"[repair] rewrote {scf_path_tsv}")
        print(f"[repair] rewrote {artifacts['band_plot_png']}")
        print(f"[repair] rewrote {artifacts['band_plot_pdf']}")

    print(
        "[repair] "
        f"result_dir={result_dir} "
        f"candidate={compatibility.candidate.name} "
        f"exact_count={compatibility.exact_count} "
        f"exact_segments={compatibility.exact_segment_counts}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite B0 HF artifact nodes/plots using the advisor-ranked right-triangle path."
    )
    parser.add_argument("result_dirs", nargs="+", type=Path, help="Leaf result directories containing computed_summary.txt.")
    parser.add_argument("--candidate-rank", type=int, default=1, help="1-based advisor candidate rank to use.")
    parser.add_argument("--path-tolerance", type=float, default=1e-12, help="Distance tolerance for SCF grid points.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for result_dir in args.result_dirs:
        repair_result_dir(result_dir, candidate_rank=int(args.candidate_rank), path_tolerance=float(args.path_tolerance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
