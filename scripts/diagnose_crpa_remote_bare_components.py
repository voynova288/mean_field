#!/usr/bin/env python3
"""Reconstruct and split the saved cRPA remote-bare one-body term."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mean_field.core.hf.overlap import compute_density_overlap_trace_from_diagonal, contract_fock_term_from_overlap
from mean_field.crpa.hf_interface import half_reference_delta_like
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    build_overlap_block_set,
)
from mean_field.systems.tbg.zero_field.model import build_b0_uniform_lattice, solve_bm_model


def _scalar(npz: dict[str, np.ndarray], key: str, default=None):
    if key not in npz:
        return default
    value = np.asarray(npz[key])
    if value.size == 0:
        return default
    return value.reshape(-1)[0].item()


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


def _eigvals_on_indices(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = _hermitian_part(matrix[:, :, indices])
    eigs = [np.linalg.eigvalsh(selected[:, :, ik]) for ik in range(selected.shape[2])]
    return np.asarray(eigs, dtype=float).T


def _spectral_width(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    eigs = _eigvals_on_indices(matrix, indices)
    return np.max(eigs, axis=0) - np.min(eigs, axis=0)


def _traceless(matrix: np.ndarray) -> np.ndarray:
    nt = matrix.shape[0]
    out = matrix.copy()
    eye = np.eye(nt, dtype=np.complex128)[:, :, None]
    traces = np.trace(out, axis1=0, axis2=1) / float(nt)
    out -= eye * traces[None, None, :]
    return out


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
        "grid_spectral_width_mean_mev": float(np.mean(all_width)),
        "grid_spectral_width_max_mev": float(np.max(all_width)),
        "path_eig_min_mev": float(np.min(path_eigs)),
        "path_eig_max_mev": float(np.max(path_eigs)),
        "path_spectral_width_mean_mev": float(np.mean(path_width)),
        "path_spectral_width_max_mev": float(np.max(path_width)),
    }


def _build_bare_components(
    density: np.ndarray,
    overlap_blocks,
    *,
    v0: float,
    use_numba: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density, got {density.shape}")
    hartree = np.zeros_like(density, dtype=np.complex128)
    fock = np.zeros_like(density, dtype=np.complex128)
    scale = float(v0) / float(nk)
    for shift in overlap_blocks.shifts:
        diagonal = overlap_blocks.diagonal_overlaps.get(shift)
        hartree_kernel = overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None and diagonal is not None:
            trace = compute_density_overlap_trace_from_diagonal(density, diagonal, use_numba=use_numba)
            hartree += scale * float(hartree_kernel) * trace * diagonal
        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            fock -= contract_fock_term_from_overlap(
                overlap_blocks.overlaps[shift],
                density,
                scale * fock_kernel,
                use_numba=use_numba,
            )
    return hartree, fock


def _plot_widths(path: Path, kdist: np.ndarray, widths: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for label, values in widths.items():
        ax.plot(kdist, values, marker="o", ms=2.7, lw=1.3, label=label)
    ax.set_xlabel("SCF-grid k-path distance")
    ax.set_ylabel("spectral width (meV)")
    ax.set_title("Remote-bare Hartree/Fock component widths")
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_eigs(path: Path, kdist: np.ndarray, eigs_by_name: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(len(eigs_by_name), 1, figsize=(9.0, 3.2 * len(eigs_by_name)), sharex=True, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (label, eigs) in zip(axes_arr, eigs_by_name.items(), strict=True):
        for ib in range(eigs.shape[0]):
            ax.plot(kdist, eigs[ib], marker="o", ms=2.2, lw=1.0)
        ax.axhline(0.0, color="0.35", lw=0.9, ls="--")
        ax.set_ylabel("meV")
        ax.set_title(label)
        ax.grid(True, color="0.88", lw=0.7)
    axes_arr[-1].set_xlabel("SCF-grid k-path distance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


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

    theta_deg = float(_scalar(remote, "theta_deg"))
    params = TBGParameters(
        dtheta_rad=float(np.deg2rad(theta_deg)),
        vf=float(_scalar(remote, "vf_mev")),
        w0=float(_scalar(remote, "w0_mev")),
        w1=float(_scalar(remote, "w1_mev")),
    )
    lk = int(_scalar(remote, "lk"))
    lg = int(_scalar(remote, "lg"))
    overlap_lg = int(_scalar(remote, "overlap_lg", lg))
    screening_kwargs = {
        "relative_permittivity": float(_scalar(remote, "effective_relative_permittivity", _scalar(remote, "epsilon_r", 4.0))),
        "screening_lm": float(_scalar(remote, "screening_lm")),
        "finite_zero_limit": bool(_scalar(remote, "q_zero_limit", True)),
    }

    grid = build_b0_uniform_lattice(params, lk)
    solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
    overlap_blocks = build_overlap_block_set(solution, lg=overlap_lg, **screening_kwargs)
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=float(_scalar(remote, "nu")))
    density_ref = half_reference_delta_like(np.asarray(remote["density"], dtype=np.complex128))
    hartree, fock = _build_bare_components(density_ref, overlap_blocks, v0=float(state.v0), use_numba=False)
    reconstructed = hartree + fock
    saved_remote_bare = np.asarray(remote["h0"], dtype=np.complex128) - np.asarray(no_remote["h0"], dtype=np.complex128)

    kdist, path_indices = _read_path_tsv(args.path_tsv)
    pieces = {
        "saved_remote_bare": saved_remote_bare,
        "reconstructed_remote_bare": reconstructed,
        "remote_bare_hartree": hartree,
        "remote_bare_fock": fock,
        "remote_bare_hartree_traceless": _traceless(hartree),
        "remote_bare_fock_traceless": _traceless(fock),
    }
    summary = {
        "approximation": "reconstructed remote bare components on exact SCF-grid path points",
        "no_remote_state": str(args.no_remote_state),
        "remote_state": str(args.remote_state),
        "path_tsv": str(args.path_tsv),
        "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed - saved_remote_bare))),
        "screening_kwargs": screening_kwargs,
        "matrix_summaries": {name: _matrix_summary(matrix, path_indices) for name, matrix in pieces.items()},
    }

    width_plot = out / "remote_bare_component_widths.png"
    eig_plot = out / "remote_bare_component_traceless_eigs.png"
    summary_path = out / "remote_bare_component_summary.json"
    widths = {
        "hartree_traceless": _spectral_width(_traceless(hartree), path_indices),
        "fock_traceless": _spectral_width(_traceless(fock), path_indices),
        "hartree_plus_fock_traceless": _spectral_width(_traceless(reconstructed), path_indices),
        "saved_remote_bare_traceless": _spectral_width(_traceless(saved_remote_bare), path_indices),
    }
    _plot_widths(width_plot, kdist, widths)
    _plot_eigs(
        eig_plot,
        kdist,
        {
            "remote bare Hartree, trace removed": _eigvals_on_indices(_traceless(hartree), path_indices),
            "remote bare Fock, trace removed": _eigvals_on_indices(_traceless(fock), path_indices),
            "remote bare total, trace removed": _eigvals_on_indices(_traceless(reconstructed), path_indices),
        },
    )
    summary["plots"] = {
        "remote_bare_component_widths_png": str(width_plot),
        "remote_bare_component_traceless_eigs_png": str(eig_plot),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "plots": summary["plots"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
