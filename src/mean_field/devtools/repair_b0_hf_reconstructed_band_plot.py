from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.systems.tbg.zero_field.plotting import write_path_band_plot


def _load_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        summary[key.strip()] = value.strip()
    return summary


def _load_path(nodes_path: Path) -> KPath:
    with nodes_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    labels = tuple(str(row["label"]) for row in rows)
    node_indices = tuple(int(row["index"]) for row in rows)
    kvec = np.asarray([complex(float(row["kx"]), float(row["ky"])) for row in rows], dtype=np.complex128)
    kdist_nodes = np.asarray([float(row["k_dist"]) for row in rows], dtype=float)
    if len(labels) < 2:
        raise ValueError(f"Expected at least two path nodes in {nodes_path}.")
    n_total = int(node_indices[-1])
    kdist = np.zeros(n_total, dtype=float)
    path_kvec = np.zeros(n_total, dtype=np.complex128)
    for iseg in range(len(node_indices) - 1):
        start = node_indices[iseg] - 1
        stop = node_indices[iseg + 1] - 1
        n_steps = stop - start
        path_kvec[start] = kvec[iseg]
        kdist[start] = kdist_nodes[iseg]
        if n_steps <= 0:
            continue
        dk = (kvec[iseg + 1] - kvec[iseg]) / n_steps
        dd = (kdist_nodes[iseg + 1] - kdist_nodes[iseg]) / n_steps
        for istep in range(1, n_steps + 1):
            path_kvec[start + istep] = kvec[iseg] + istep * dk
            kdist[start + istep] = kdist_nodes[iseg] + istep * dd
    return KPath(
        kvec=path_kvec,
        kdist=kdist,
        labels=labels,
        node_indices=node_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the reconstructed HF band plot from computed_hf_path.tsv.")
    parser.add_argument("result_dir", type=Path, help="Benchmark leaf directory containing computed_hf_path.tsv and computed_nodes.tsv")
    parser.add_argument("--stem", default="band_plot", help="Plot stem to rewrite.")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    summary = _load_summary(result_dir / "computed_summary.txt")
    path = _load_path(result_dir / "computed_nodes.tsv")

    with (result_dir / "computed_hf_path.tsv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    band_labels = tuple(str(value) for value in header[1:])
    kdist = np.asarray([float(row[0]) for row in rows], dtype=float)
    energies = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    title = (
        f"theta={float(summary['theta_deg']):.2f}°, "
        f"nu={float(summary['nu']):g}, "
        f"init={summary['init_mode']}, "
        f"seed={int(summary['seed'])}"
    )
    artifacts = write_path_band_plot(
        result_dir,
        stem=args.stem,
        kdist=kdist,
        energies=energies,
        path=path,
        band_labels=band_labels,
        mu=float(summary["mu"]),
        title=title,
    )
    print(f"[repair] rewrote {artifacts['band_plot_png']}")
    print(f"[repair] rewrote {artifacts['band_plot_pdf']}")


if __name__ == "__main__":
    main()
