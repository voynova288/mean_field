#!/usr/bin/env python3
"""Compare saved no-remote and remote-projector cRPA HF states.

The script is intentionally read-only with respect to the HF states.  It uses
the existing SCF-grid path TSV, so all plotted points are exact SCF grid points
and no off-grid Hamiltonian is reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _read_path_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    kdist: list[float] = []
    indices: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kdist.append(float(row["k_dist"]))
            indices.append(int(row["grid_index"]) - 1)
    if not indices:
        raise ValueError(f"No path points found in {path}")
    return np.asarray(kdist, dtype=float), np.asarray(indices, dtype=int)


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conjugate().swapaxes(0, 1))


def _traceless(matrix: np.ndarray) -> np.ndarray:
    nt = matrix.shape[0]
    out = matrix.copy()
    eye = np.eye(nt, dtype=np.complex128)[:, :, None]
    traces = np.trace(out, axis1=0, axis2=1) / float(nt)
    out -= eye * traces[None, None, :]
    return out


def _eigvals_on_indices(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = _hermitian_part(matrix[:, :, indices])
    eigs = [np.linalg.eigvalsh(selected[:, :, ik]) for ik in range(selected.shape[2])]
    return np.asarray(eigs, dtype=float).T


def _spectral_width(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    eigs = _eigvals_on_indices(matrix, indices)
    return np.max(eigs, axis=0) - np.min(eigs, axis=0)


def _matrix_summary(matrix: np.ndarray, path_indices: np.ndarray) -> dict[str, float]:
    herm = _hermitian_part(matrix)
    anti = matrix - matrix.conjugate().swapaxes(0, 1)
    all_eigs = np.asarray([np.linalg.eigvalsh(herm[:, :, ik]) for ik in range(herm.shape[2])], dtype=float)
    path_eigs = _eigvals_on_indices(matrix, path_indices).T
    all_width = np.max(all_eigs, axis=1) - np.min(all_eigs, axis=1)
    path_width = np.max(path_eigs, axis=1) - np.min(path_eigs, axis=1)
    return {
        "fro_norm": float(np.linalg.norm(matrix)),
        "max_abs": float(np.max(np.abs(matrix))),
        "antihermitian_max_abs": float(np.max(np.abs(anti))),
        "grid_eig_min_mev": float(np.min(all_eigs)),
        "grid_eig_max_mev": float(np.max(all_eigs)),
        "grid_spectral_width_min_mev": float(np.min(all_width)),
        "grid_spectral_width_mean_mev": float(np.mean(all_width)),
        "grid_spectral_width_max_mev": float(np.max(all_width)),
        "path_eig_min_mev": float(np.min(path_eigs)),
        "path_eig_max_mev": float(np.max(path_eigs)),
        "path_spectral_width_min_mev": float(np.min(path_width)),
        "path_spectral_width_mean_mev": float(np.mean(path_width)),
        "path_spectral_width_max_mev": float(np.max(path_width)),
    }


def _plot_eigs(path: Path, kdist: np.ndarray, eigs: np.ndarray, *, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for ib in range(eigs.shape[0]):
        ax.plot(kdist, eigs[ib], marker="o", ms=2.5, lw=1.1)
    ax.axhline(0.0, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel("SCF-grid k-path distance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.88", lw=0.7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_widths(path: Path, kdist: np.ndarray, widths: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for label, values in widths.items():
        ax.plot(kdist, values, marker="o", ms=2.7, lw=1.3, label=label)
    ax.set_xlabel("SCF-grid k-path distance")
    ax.set_ylabel("spectral width (meV)")
    ax.set_title("SCF-grid spectral widths of saved Hamiltonian pieces")
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_width_tsv(path: Path, kdist: np.ndarray, indices: np.ndarray, widths: dict[str, np.ndarray]) -> None:
    labels = list(widths)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(["path_i", "grid_index_1based", "k_dist", *labels]) + "\n")
        for i, kd in enumerate(kdist):
            row = [str(i + 1), str(int(indices[i]) + 1), f"{float(kd):.16e}"]
            row.extend(f"{float(widths[label][i]):.16e}" for label in labels)
            handle.write("\t".join(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-remote-state", type=Path, required=True)
    parser.add_argument("--remote-state", type=Path, required=True)
    parser.add_argument("--path-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    no_remote = _load_state(args.no_remote_state)
    remote = _load_state(args.remote_state)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for key in ("h0", "hamiltonian"):
        if no_remote[key].shape != remote[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {no_remote[key].shape} != {remote[key].shape}")

    kdist, path_indices = _read_path_tsv(args.path_tsv)

    bm_h0 = np.asarray(no_remote["h0"], dtype=np.complex128)
    remote_bare = np.asarray(remote["h0"], dtype=np.complex128) - bm_h0
    remote_bare_traceless = _traceless(remote_bare)
    no_remote_active = np.asarray(no_remote["hamiltonian"], dtype=np.complex128) - np.asarray(no_remote["h0"], dtype=np.complex128)
    remote_active = np.asarray(remote["hamiltonian"], dtype=np.complex128) - np.asarray(remote["h0"], dtype=np.complex128)
    final_difference = np.asarray(remote["hamiltonian"], dtype=np.complex128) - np.asarray(no_remote["hamiltonian"], dtype=np.complex128)

    matrices = {
        "bm_h0": bm_h0,
        "remote_bare": remote_bare,
        "remote_bare_traceless": remote_bare_traceless,
        "no_remote_active": no_remote_active,
        "remote_active": remote_active,
        "final_remote_minus_no_remote": final_difference,
    }
    summary = {
        "approximation": "saved-state diagnostic on exact SCF-grid path points; no off-grid Hamiltonian",
        "no_remote_state": str(args.no_remote_state),
        "remote_state": str(args.remote_state),
        "path_tsv": str(args.path_tsv),
        "path_point_count": int(path_indices.size),
        "matrix_summaries": {name: _matrix_summary(matrix, path_indices) for name, matrix in matrices.items()},
    }

    raw_plot = out / "remote_bare_path_eigs.png"
    traceless_plot = out / "remote_bare_traceless_path_eigs.png"
    width_plot = out / "hamiltonian_piece_path_widths.png"
    width_tsv = out / "hamiltonian_piece_path_widths.tsv"
    summary_path = out / "remote_bare_state_diagnostic_summary.json"

    _plot_eigs(
        raw_plot,
        kdist,
        _eigvals_on_indices(remote_bare, path_indices),
        title="Remote bare h0 difference on SCF-grid path",
        ylabel="remote bare eigenvalue (meV)",
    )
    _plot_eigs(
        traceless_plot,
        kdist,
        _eigvals_on_indices(remote_bare_traceless, path_indices),
        title="Traceless remote bare h0 difference on SCF-grid path",
        ylabel="traceless eigenvalue (meV)",
    )
    widths = {
        "remote_bare_traceless": _spectral_width(remote_bare_traceless, path_indices),
        "no_remote_active": _spectral_width(no_remote_active, path_indices),
        "remote_active": _spectral_width(remote_active, path_indices),
        "final_remote_minus_no_remote": _spectral_width(final_difference, path_indices),
    }
    _plot_widths(width_plot, kdist, widths)
    _write_width_tsv(width_tsv, kdist, path_indices, widths)

    summary["plots"] = {
        "remote_bare_path_eigs_png": str(raw_plot),
        "remote_bare_traceless_path_eigs_png": str(traceless_plot),
        "hamiltonian_piece_path_widths_png": str(width_plot),
        "hamiltonian_piece_path_widths_tsv": str(width_tsv),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "plots": summary["plots"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
