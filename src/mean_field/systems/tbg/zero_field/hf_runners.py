from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ....benchmarks import HFPathReference
from ....core.hf import FlavorBandData, build_flavor_band_data, build_projected_target_hamiltonian
from ....core.lattice import KPath
from ..params import TBGParameters
from .hf import (
    RestrictedHartreeFockRun,
    build_h0_from_bm,
    build_overlap_block_set,
)
from .model import BMSolution, solve_bm_model
from .path import build_fig6_kpath


@dataclass(frozen=True)
class HFPathResult:
    params: TBGParameters
    path: KPath
    hamiltonian: np.ndarray
    band_data: FlavorBandData
    mu: float
    nu: float
    lk: int
    lg: int
    points_per_segment: int
    init_mode: str
    normalized_init_mode: str
    seed: int
    exit_reason: str
    beta: float = 1.0
    overlap_lg: int | None = None
    relative_permittivity: float = 15.0
    screening_lm: float | None = None
    finite_zero_limit: bool = False
    zero_cutoff: float = 1e-6


@dataclass(frozen=True)
class HFPathParity:
    kdist_max_abs_diff: float
    max_abs_band_diff_mev: float
    rms_band_diff_mev: float
    mean_abs_band_diff_mev: float
    energy_sorting: str = "ascending_per_k"


@dataclass(frozen=True)
class HFSCFPathPlotResult:
    params: TBGParameters
    path: KPath
    kdist: np.ndarray
    projected_kvec: np.ndarray
    distance_to_path: np.ndarray
    path_sample_indices: np.ndarray
    path_kvec: np.ndarray
    grid_kvec: np.ndarray
    grid_indices: np.ndarray
    band_data: FlavorBandData
    mu: float
    nu: float
    lk: int
    lg: int
    init_mode: str
    normalized_init_mode: str
    seed: int
    exit_reason: str


def build_restricted_hf_scf_path_plot_result(
    hf_run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
    *,
    path: KPath,
    init_mode: str | None = None,
    path_tolerance: float = 1e-12,
) -> HFSCFPathPlotResult:
    # Diagnostic-only view: keep only path samples that are exact SCF grid
    # points. This preserves repeated path coordinates, such as Gamma appearing
    # at both the start and middle of a high-symmetry path, without recomputing
    # any off-grid Hamiltonians.
    grid_kvec = np.asarray(grid_solution.lattice_kvec, dtype=np.complex128)
    if path.kvec.size == 0:
        raise ValueError("At least two path nodes are required to build an SCF path plot.")

    distance_matrix = np.abs(np.asarray(path.kvec, dtype=np.complex128)[:, None] - grid_kvec[None, :])
    nearest_grid_indices = np.argmin(distance_matrix, axis=1).astype(int)
    nearest_grid_distances = distance_matrix[np.arange(path.kvec.size), nearest_grid_indices]
    path_indices = np.flatnonzero(nearest_grid_distances <= float(path_tolerance)).astype(int)
    indices = nearest_grid_indices[path_indices].astype(int)
    path_kvec = np.asarray(path.kvec[path_indices], dtype=np.complex128)
    distance_to_path = np.asarray(nearest_grid_distances[path_indices], dtype=float)
    selected_hamiltonian = np.asarray(hf_run.state.hamiltonian[:, :, indices], dtype=np.complex128)
    band_data = build_flavor_band_data(
        selected_hamiltonian,
        n_spin=hf_run.state.n_spin,
        n_eta=hf_run.state.n_eta,
        n_band=hf_run.state.n_band,
    )
    requested_init_mode = hf_run.init_mode if init_mode is None else str(init_mode)
    return HFSCFPathPlotResult(
        params=grid_solution.params,
        path=path,
        kdist=np.asarray(path.kdist[path_indices], dtype=float),
        projected_kvec=np.asarray(path_kvec, dtype=np.complex128),
        distance_to_path=np.asarray(distance_to_path, dtype=float),
        path_sample_indices=path_indices,
        path_kvec=np.asarray(path_kvec, dtype=np.complex128),
        grid_kvec=np.asarray(grid_kvec[indices], dtype=np.complex128),
        grid_indices=indices,
        band_data=band_data,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        lk=int(round(np.sqrt(grid_solution.nk))) - 1,
        lg=grid_solution.lg,
        init_mode=requested_init_mode,
        normalized_init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
    )


def build_restricted_hf_path_hamiltonian(
    hf_run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
    *,
    points_per_segment: int = 120,
    lg: int | None = None,
    overlap_lg: int | None = None,
    beta: float | None = None,
    path: KPath | None = None,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> tuple[KPath, BMSolution, np.ndarray]:
    state = hf_run.state
    params = grid_solution.params
    path = build_fig6_kpath(params, points_per_segment) if path is None else path
    bm_lg = grid_solution.lg if lg is None else int(lg)
    resolved_overlap_lg = (
        int(overlap_lg)
        if overlap_lg is not None
        else int(hf_run.state.diagnostics.get("overlap_lg", float(bm_lg)))
    )
    resolved_beta = float(beta) if beta is not None else float(hf_run.state.diagnostics.get("beta", 1.0))
    path_solution = solve_bm_model(params, path.kvec, lg=bm_lg, sigma_rotation=True)
    h_path = build_h0_from_bm(path_solution)

    screening_kwargs = {
        "relative_permittivity": float(relative_permittivity),
        "screening_lm": screening_lm,
        "finite_zero_limit": bool(finite_zero_limit),
        "zero_cutoff": float(zero_cutoff),
    }
    grid_overlap = build_overlap_block_set(grid_solution, lg=resolved_overlap_lg, **screening_kwargs)
    path_overlap = build_overlap_block_set(path_solution, lg=resolved_overlap_lg, **screening_kwargs)
    path_grid_overlap = build_overlap_block_set(
        path_solution,
        source_solution=grid_solution,
        lg=resolved_overlap_lg,
        **screening_kwargs,
    )
    h_path = build_projected_target_hamiltonian(
        h_path,
        state.density,
        source_overlap_blocks=grid_overlap,
        target_overlap_blocks=path_overlap,
        target_source_overlap_blocks=path_grid_overlap,
        v0=state.v0,
        beta=resolved_beta,
    )

    return path, path_solution, h_path


def evaluate_restricted_hf_path(
    hf_run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
    *,
    points_per_segment: int = 120,
    lg: int | None = None,
    overlap_lg: int | None = None,
    beta: float | None = None,
    init_mode: str | None = None,
    path: KPath | None = None,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> HFPathResult:
    path, _, h_path = build_restricted_hf_path_hamiltonian(
        hf_run,
        grid_solution,
        points_per_segment=points_per_segment,
        lg=lg,
        overlap_lg=overlap_lg,
        beta=beta,
        path=path,
        relative_permittivity=relative_permittivity,
        screening_lm=screening_lm,
        finite_zero_limit=finite_zero_limit,
        zero_cutoff=zero_cutoff,
    )
    band_data = build_flavor_band_data(
        h_path,
        n_spin=hf_run.state.n_spin,
        n_eta=hf_run.state.n_eta,
        n_band=hf_run.state.n_band,
    )
    requested_init_mode = hf_run.init_mode if init_mode is None else str(init_mode)
    bm_lg = grid_solution.lg if lg is None else int(lg)
    resolved_overlap_lg = (
        int(overlap_lg)
        if overlap_lg is not None
        else int(hf_run.state.diagnostics.get("overlap_lg", float(bm_lg)))
    )
    resolved_beta = float(beta) if beta is not None else float(hf_run.state.diagnostics.get("beta", 1.0))
    return HFPathResult(
        params=grid_solution.params,
        path=path,
        hamiltonian=h_path,
        band_data=band_data,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        lk=int(round(np.sqrt(grid_solution.nk))) - 1,
        lg=bm_lg,
        points_per_segment=points_per_segment,
        init_mode=requested_init_mode,
        normalized_init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        beta=resolved_beta,
        overlap_lg=resolved_overlap_lg,
        relative_permittivity=float(relative_permittivity),
        screening_lm=screening_lm,
        finite_zero_limit=bool(finite_zero_limit),
        zero_cutoff=float(zero_cutoff),
    )


def compare_hf_path_to_reference(reference: HFPathReference, result: HFPathResult) -> HFPathParity:
    generated_kdist = np.asarray(result.path.kdist, dtype=float)
    reference_kdist = np.asarray(reference.kdist, dtype=float)
    if generated_kdist.shape != reference_kdist.shape:
        raise ValueError(f"Reference path point count mismatch: {reference_kdist.shape} vs {generated_kdist.shape}")
    if result.band_data.energies.T.shape != reference.energies.shape:
        raise ValueError(
            f"Reference band array shape mismatch: {reference.energies.shape} vs {result.band_data.energies.T.shape}"
        )

    diffs: list[float] = []
    generated_rows = result.band_data.energies.T
    for ref_row, gen_row in zip(reference.energies, generated_rows, strict=True):
        ref_sorted = np.sort(ref_row)
        gen_sorted = np.sort(gen_row)
        diffs.extend((ref_sorted - gen_sorted).tolist())

    diff_array = np.asarray(diffs, dtype=float)
    abs_diff = np.abs(diff_array)
    return HFPathParity(
        kdist_max_abs_diff=float(np.max(np.abs(reference_kdist - generated_kdist))),
        max_abs_band_diff_mev=float(np.max(abs_diff)),
        rms_band_diff_mev=float(np.sqrt(np.mean(diff_array**2))),
        mean_abs_band_diff_mev=float(np.mean(abs_diff)),
    )


def write_hf_path_tsv(path: Path, result: HFPathResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(["k_dist", *result.band_data.band_labels]) + "\n")
        for ik, kdist in enumerate(result.path.kdist):
            row = [f"{float(kdist):.16f}"]
            row.extend(f"{float(result.band_data.energies[ib, ik]):.16f}" for ib in range(result.band_data.energies.shape[0]))
            handle.write("\t".join(row) + "\n")


def write_hf_path_nodes_tsv(path: Path, result: HFPathResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("label\tindex\tk_dist\tkx\tky\n")
        for node in result.path.nodes:
            handle.write(
                f"{node.label}\t{node.index}\t{node.k_dist:.16f}\t{node.kx:.16f}\t{node.ky:.16f}\n"
            )


def write_hf_scf_path_tsv(path: Path, result: HFSCFPathPlotResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "path_index",
                    "path_k_dist",
                    "k_dist",
                    "distance_to_path",
                    "path_kx",
                    "path_ky",
                    "projected_kx",
                    "projected_ky",
                    "grid_index",
                    "grid_kx",
                    "grid_ky",
                    *result.band_data.band_labels,
                ]
            )
            + "\n"
        )
        for ik, kdist in enumerate(result.kdist):
            row = [
                str(int(result.path_sample_indices[ik]) + 1),
                f"{float(result.path.kdist[result.path_sample_indices[ik]]):.16f}",
                f"{float(kdist):.16f}",
                f"{float(result.distance_to_path[ik]):.16f}",
                f"{float(result.path_kvec[ik].real):.16f}",
                f"{float(result.path_kvec[ik].imag):.16f}",
                f"{float(result.projected_kvec[ik].real):.16f}",
                f"{float(result.projected_kvec[ik].imag):.16f}",
                str(int(result.grid_indices[ik]) + 1),
                f"{float(result.grid_kvec[ik].real):.16f}",
                f"{float(result.grid_kvec[ik].imag):.16f}",
            ]
            row.extend(f"{float(result.band_data.energies[ib, ik]):.16f}" for ib in range(result.band_data.energies.shape[0]))
            handle.write("\t".join(row) + "\n")


def write_hf_path_summary(path: Path, result: HFPathResult, *, hf_state_path: str = "") -> None:
    path_label = "-".join(result.path.labels)
    entries = [
        ("hf_path", hf_state_path),
        ("theta_deg", f"{result.params.dtheta_rad * 180.0 / np.pi:.2f}"),
        ("nu", f"{result.nu}"),
        ("init_mode", result.init_mode),
        ("normalized_init_mode", result.normalized_init_mode),
        ("seed", str(result.seed)),
        ("lk", str(result.lk)),
        ("lg", str(result.lg)),
        ("overlap_lg", "" if result.overlap_lg is None else str(result.overlap_lg)),
        ("beta", f"{result.beta}"),
        ("relative_permittivity", f"{result.relative_permittivity}"),
        ("screening_lm", "" if result.screening_lm is None else f"{result.screening_lm}"),
        ("finite_zero_limit", str(result.finite_zero_limit).lower()),
        ("drop_q0_coulomb", str(not result.finite_zero_limit).lower()),
        ("zero_cutoff", f"{result.zero_cutoff}"),
        ("points_per_segment", str(result.points_per_segment)),
        ("mu", f"{result.mu}"),
        ("exit_reason", result.exit_reason),
        ("path", path_label),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
