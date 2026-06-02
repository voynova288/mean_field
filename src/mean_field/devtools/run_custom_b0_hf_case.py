#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
from time import perf_counter

import numpy as np

from mean_field.core.hf import build_flavor_band_data, build_projected_target_hamiltonian
from mean_field.crpa import (
    CRPAScreenedCoulomb,
    active_lower_flat_projector_like,
    build_bare_projected_target_components,
    build_crpa_hartree_delta_fock_projector_components,
    build_crpa_projected_interaction_components,
    build_crpa_projected_target_components,
    build_crpa_projected_target_components_from_densities,
    build_crpa_projected_target_hamiltonian,
    build_fock_screened_overlap_blocks,
    crpa_active_density_from_delta,
    crpa_hartree_delta_fock_projector_energy_components,
    crpa_remote_bare_scale,
    crpa_split_mode,
    crpa_split_uses_active_cnp_reference,
    crpa_split_uses_hartree_delta_fock_projector,
    crpa_split_uses_remote_bare,
    crpa_hf_energy_components,
    half_reference_delta_like,
    load_crpa_result,
    physical_projector_from_delta,
    run_full_crpa_hartree_fock,
    select_active_cnp_reference_components,
    select_remote_reference_components,
    validate_hf_compatible_crpa,
)
from mean_field.crpa.validation import (
    compare_fig1e_window_to_paper_points,
    crpa_convention_family,
    fig1e_paper_point_gate_failures,
)
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.resample_b0_density_stack import resample_density_stack
from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import (
    HFPathResult,
    RestrictedHartreeFockState,
    build_b0_uniform_lattice,
    build_fig6_kpath,
    build_gamma_m_k_gamma_kprime_kpath,
    build_h0_from_bm,
    build_overlap_block_set,
    build_restricted_hf_scf_path_plot_result,
    moire_bz_vertices,
    restricted_gap_estimate,
    restricted_occupied_bands_per_k,
    run_full_hartree_fock,
    sampled_cell_vertices,
    solve_bm_model,
    write_hf_band_plot,
    write_hf_path_nodes_tsv,
    write_hf_path_summary,
    write_hf_path_tsv,
    write_hf_scf_band_plot,
    write_hf_scf_path_tsv,
)
from mean_field.systems.tbg.zero_field.path import project_kvec_onto_path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "TBG_HF" / "custom_b0_hf_targeted_runs"
DEFAULT_RUN_TAG = "eps4_gate400a_ds400over2p46_q0limit_w0_79p7_w1_97p5_vf_2135p4_20260425_meanfield"
DEFAULT_CRPA_PHYSICS_REFERENCE_DIR = (
    REPO_ROOT / "results" / "TBG_HF_cRPA" / "crpa_lk24_lg9_q11_zhang_appendix_fig4_merged"
)
GRAPHENE_LATTICE_A_ANGSTROM = 2.46


@dataclass(frozen=True)
class InitSpec:
    init_mode: str
    seed: int


def _default_init_specs() -> tuple[InitSpec, ...]:
    deterministic = tuple(InitSpec(init, 1) for init in ("vp", "sp", "chern", "tivc", "kivc", "flavor", "bm"))
    random = tuple(InitSpec("random", seed) for seed in range(1, 7))
    return (*deterministic, *random)


def _ensure_not_running_on_login_node() -> None:
    ensure_not_running_compute_on_login_node("custom B0 HF")


def _theta_tag(theta_deg: float) -> str:
    return f"{round(theta_deg * 100):03d}"


def _nu_tag(nu: float) -> str:
    return f"{round(nu * 1000):+05d}"


def _case_tag(theta_deg: float, nu: float, init_mode: str, seed: int, lk: int, lg: int) -> str:
    return f"theta_{_theta_tag(theta_deg)}_nu_{_nu_tag(nu)}_init_{init_mode}_seed_{seed:03d}_lk{lk}_lg{lg}"


def _nu_file_label(nu: float) -> str:
    rounded = int(round(float(nu)))
    if abs(float(nu) - rounded) < 1.0e-12:
        return f"{rounded:+d}" if rounded else "0"
    return f"{float(nu):+.6g}".replace(".", "p")


def _write_key_value_file(path: Path, entries: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def _append_key_value_file(path: Path, entries: list[tuple[str, str]]) -> Path:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in entries:
            handle.write(f"{key}={value}\n")
    return path


def _write_summary_table(path: Path, rows: list[dict[str, str]]) -> Path:
    columns = _summary_columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(row.get(column, "") for column in columns) + "\n")
    return path


def _summary_columns() -> list[str]:
    return [
        "theta_deg",
        "nu",
        "init_mode",
        "seed",
        "iterations",
        "exit_reason",
        "converged",
        "mu_mev",
        "final_energy",
        "final_error",
        "final_oda",
        "hf_elapsed_sec",
        "path_elapsed_sec",
        "gap_source",
        "indirect_gap_mev",
        "direct_gap_mev",
        "full_scf_grid_indirect_gap_mev",
        "full_scf_grid_direct_gap_mev",
        "scf_path_indirect_gap_mev",
        "scf_path_direct_gap_mev",
        "E_band",
        "E_Hartree",
        "E_Fock",
        "E_total",
        "valley_polarization_tau_z",
        "spin_polarization",
        "kivc_amplitude",
        "svp_amplitude",
        "chern_proxy_sigma_ztauz",
        "q_lookup_failures",
        "q_lookup_fallbacks",
        "max_q_reconstruction_residual_nm_inv",
        "eps_crpa_min",
        "eps_crpa_mean",
        "eps_crpa_max",
        "state_path",
        "path_tsv",
    ]


def _gap_summary(energies: np.ndarray, nu: float) -> dict[str, float]:
    energy_array = np.asarray(energies, dtype=float)
    nt, nk = energy_array.shape
    occ = restricted_occupied_bands_per_k(float(nu), nt)
    if occ <= 0 or occ >= nt:
        return {
            "indirect_gap_mev": float("nan"),
            "direct_gap_mev": float("nan"),
        }
    sorted_per_k = np.sort(energy_array, axis=0)
    valence = sorted_per_k[occ - 1, :]
    conduction = sorted_per_k[occ, :]
    return {
        "indirect_gap_mev": float(np.min(conduction) - np.max(valence)),
        "direct_gap_mev": float(np.min(conduction - valence)),
    }


def _order_parameter_summary(density: np.ndarray, energies: np.ndarray, sigma_ztauz: np.ndarray, nu: float) -> dict[str, float]:
    rho = np.asarray(density, dtype=np.complex128)
    nt, _, nk = rho.shape
    n_spin, n_eta, n_band = 2, 2, nt // 4
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    diagonal = np.real(np.diagonal(rho, axis1=0, axis2=1).T)
    # The stored projector subtracts 1/2 from every state.  Add it back for
    # polarization diagnostics so that empty/filled weights are in [0, 1].
    occupation = diagonal + 0.5
    total = float(np.sum(occupation))
    if abs(total) < 1.0e-14:
        total = float("nan")

    spin_sign = np.asarray([1.0, -1.0], dtype=float)
    valley_sign = np.asarray([1.0, -1.0], dtype=float)
    spin_weight = 0.0
    valley_weight = 0.0
    svp_weight = 0.0
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            sector = idx[ispin, ieta, :].reshape(-1)
            occ_sector = float(np.sum(occupation[sector, :]))
            spin_weight += spin_sign[ispin] * occ_sector
            valley_weight += valley_sign[ieta] * occ_sector
            svp_weight += spin_sign[ispin] * valley_sign[ieta] * occ_sector

    kivc_norm_sq = 0.0
    for ispin in range(n_spin):
        left = idx[ispin, 0, :].reshape(-1)
        right = idx[ispin, 1, :].reshape(-1)
        block = rho[np.ix_(left, right, np.arange(nk))]
        kivc_norm_sq += float(np.sum(np.abs(block) ** 2))

    occ_count = int(round((float(nu) + 4.0) / 8.0 * nt * nk))
    sorted_flat = np.sort(np.asarray(energies, dtype=float), axis=None)
    if occ_count <= 0 or occ_count >= sorted_flat.size:
        chern_proxy = float("nan")
    else:
        flat_energies = np.ravel(np.asarray(energies, dtype=float), order="F")
        order = np.argsort(flat_energies, kind="stable")[:occ_count]
        chern_proxy = float(np.sum(np.ravel(sigma_ztauz, order="F")[order]) / float(nk))

    return {
        "valley_polarization_tau_z": float(valley_weight / total),
        "spin_polarization": float(spin_weight / total),
        "svp_amplitude": float(abs(svp_weight / total)),
        "kivc_amplitude": float(np.sqrt(kivc_norm_sq / max(1, nk))),
        "chern_proxy_sigma_ztauz": chern_proxy,
    }


def _lookup_diagnostics_for_scf_grid(
    *,
    crpa_screening: CRPAScreenedCoulomb | None,
    overlap_blocks,
    lattice_kvec: np.ndarray,
    method: str,
) -> dict[str, float | int | str]:
    if crpa_screening is None:
        return {
            "method": "none",
            "q_lookup_failures": 0,
            "q_lookup_fallbacks": 0,
            "q_count": 0,
            "eps_crpa_min": 1.0,
            "eps_crpa_mean": 1.0,
            "eps_crpa_max": 1.0,
            "max_q_reconstruction_residual": 0.0,
            "max_q_reconstruction_residual_nm_inv": 0.0,
        }

    eps_min = float("inf")
    eps_max = float("-inf")
    eps_sum = 0.0
    q_count = 0
    failures = 0
    fallbacks = 0
    max_residual = 0.0
    max_residual_nm = 0.0
    kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    for _shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        qvals = kvec[None, :] - kvec[:, None] + complex(gvec)
        diag = crpa_screening.fock_lookup_diagnostics(qvals, method=method)
        info = diag.as_dict()
        count = int(info["q_count"])
        q_count += count
        failures += int(info["q_lookup_failures"])
        fallbacks += int(info["q_lookup_fallbacks"])
        eps_sum += float(info["eps_crpa_mean"]) * count
        eps_min = min(eps_min, float(info["eps_crpa_min"]))
        eps_max = max(eps_max, float(info["eps_crpa_max"]))
        max_residual = max(max_residual, float(info["max_q_reconstruction_residual"]))
        max_residual_nm = max(max_residual_nm, float(info["max_q_reconstruction_residual_nm_inv"]))

    eps_mean = float(eps_sum / q_count) if q_count else float("nan")
    return {
        "method": str(method),
        "q_lookup_failures": int(failures),
        "q_lookup_fallbacks": int(fallbacks),
        "q_count": int(q_count),
        "eps_crpa_min": float(eps_min),
        "eps_crpa_mean": eps_mean,
        "eps_crpa_max": float(eps_max),
        "max_q_reconstruction_residual": float(max_residual),
        "max_q_reconstruction_residual_nm_inv": float(max_residual_nm),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _combine_summary_parts(root_dir: Path) -> int:
    part_dir = root_dir / "summary_parts"
    if not part_dir.exists():
        raise SystemExit(f"Missing summary part directory: {part_dir}")
    columns = _summary_columns()
    rows: list[dict[str, str]] = []
    for part_path in sorted(part_dir.glob("*.tsv")):
        with part_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            if header != columns:
                raise SystemExit(f"Unexpected summary columns in {part_path}: {header}")
            for line in handle:
                values = line.rstrip("\n").split("\t")
                if len(values) != len(columns):
                    raise SystemExit(f"Unexpected column count in {part_path}: {line.rstrip()}")
                rows.append(dict(zip(columns, values, strict=True)))
    if not rows:
        raise SystemExit(f"No summary part rows found under {part_dir}")
    _write_summary_table(root_dir / "summary.tsv", rows)
    _write_key_value_file(
        root_dir / "summary_info.txt",
        [
            ("output_dir", str(root_dir)),
            ("summary_parts", str(part_dir)),
            ("part_count", str(len(rows))),
            ("combined_at", datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
        ],
    )
    return len(rows)


def _infer_lk_from_density_nk(nk: int) -> int:
    side = int(round(np.sqrt(nk)))
    if side * side != nk or side < 2:
        raise ValueError(f"Cannot infer an inclusive B0 square-grid lk from nk={nk}")
    return side - 1


def _load_initial_density(
    path: Path | None,
    *,
    target_lk: int,
    resample_method: str,
) -> tuple[np.ndarray | None, list[tuple[str, str]]]:
    if path is None:
        return None, []
    if not path.exists():
        raise SystemExit(f"Initial state does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "density" not in data:
            raise SystemExit(f"Initial state is missing density array: {path}")
        density = np.asarray(data["density"], dtype=np.complex128)
        metadata: list[tuple[str, str]] = [("initial_state", str(path))]
        for key in ("theta_deg", "nu", "lk", "lg", "iterations", "exit_reason", "converged"):
            if key not in data:
                continue
            value = np.asarray(data[key]).reshape(-1)[0]
            metadata.append((f"initial_state_{key}", str(value)))
        target_nk = (int(target_lk) + 1) ** 2
        if density.shape[2] != target_nk:
            if resample_method == "none":
                raise SystemExit(
                    f"Initial state density nk={density.shape[2]} does not match target lk={target_lk} "
                    f"(nk={target_nk}). Pass --initial-state-resample bilinear or nearest to continue branches across lk."
                )
            if "lk" in data:
                source_lk = int(np.asarray(data["lk"]).reshape(-1)[0])
            else:
                source_lk = _infer_lk_from_density_nk(int(density.shape[2]))
            density = resample_density_stack(
                density,
                source_lk=source_lk,
                target_lk=int(target_lk),
                method=resample_method,
                hermitize=True,
            )
            metadata.extend(
                [
                    ("initial_state_resampled", "true"),
                    ("initial_state_source_lk", str(source_lk)),
                    ("initial_state_target_lk", str(int(target_lk))),
                    ("initial_state_resample_method", resample_method),
                ]
            )
        else:
            metadata.append(("initial_state_resampled", "false"))
    return density, metadata


def _build_path(params: TBGParameters, *, path_kind: str, points_per_segment: int):
    if path_kind == "fig6":
        return build_fig6_kpath(params, points_per_segment)
    if path_kind == "gamma-m-k-gamma-kprime":
        return build_gamma_m_k_gamma_kprime_kpath(params, points_per_segment)
    raise ValueError(f"Unsupported path kind: {path_kind}")


def _segment_counts_for_exact_path_points(path, exact_kdist: np.ndarray) -> tuple[int, ...]:
    node_edges = np.asarray([float(node.k_dist) for node in path.nodes], dtype=float)
    segment_counts = np.zeros(max(node_edges.size - 1, 0), dtype=int)
    if exact_kdist.size > 0 and segment_counts.size > 0:
        segment_indices = np.searchsorted(node_edges[1:], exact_kdist, side="right")
        segment_indices = np.clip(segment_indices, 0, segment_counts.size - 1)
        for iseg in segment_indices:
            segment_counts[int(iseg)] += 1
    return tuple(int(value) for value in segment_counts)


def _write_kmesh_path_overlay(output_dir: Path, *, params: TBGParameters, grid, path, path_kind: str, exact_tolerance: float = 1e-12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    projected_kdist, _, distance_to_path = project_kvec_onto_path(path, grid.kvec)
    exact_mask = distance_to_path <= float(exact_tolerance)
    exact_indices = np.flatnonzero(exact_mask)
    if exact_indices.size > 0:
        order = np.argsort(projected_kdist[exact_indices], kind="stable")
        exact_indices = exact_indices[order]
    exact_kdist = np.asarray(projected_kdist[exact_indices], dtype=float)
    exact_grid_kvec = np.asarray(grid.kvec[exact_indices], dtype=np.complex128)
    node_min_distances = np.asarray(
        [float(np.min(np.abs(grid.kvec - complex(node.kvec)))) for node in path.nodes],
        dtype=float,
    )
    segment_counts = _segment_counts_for_exact_path_points(path, exact_kdist)

    stem = f"kmesh_path_overlay_{path_kind.replace('-', '_')}_lk{int(grid.lk)}"
    summary_path = output_dir / f"{stem}_summary.txt"
    _write_key_value_file(
        summary_path,
        [
            ("path_kind", path_kind),
            ("path_labels", "-".join(path.labels)),
            ("lk", str(int(grid.lk))),
            ("nk", str(int(grid.nk))),
            ("exact_tolerance", f"{float(exact_tolerance):.16e}"),
            ("exact_count", str(int(exact_indices.size))),
            ("exact_segment_counts", ",".join(str(value) for value in segment_counts)),
            ("exact_node_hit_count", str(int(np.count_nonzero(node_min_distances <= exact_tolerance)))),
            ("node_min_distances", ",".join(f"{value:.16e}" for value in node_min_distances)),
        ],
    )

    nodes_tsv = output_dir / f"{stem}_nodes.tsv"
    with nodes_tsv.open("w", encoding="utf-8") as handle:
        handle.write("label\tindex\tk_dist\tkx\tky\tnearest_grid_distance\n")
        for node, distance in zip(path.nodes, node_min_distances, strict=True):
            handle.write(
                f"{node.label}\t{node.index}\t{node.k_dist:.16f}\t{node.kx:.16f}\t{node.ky:.16f}\t{distance:.16e}\n"
            )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_mean_field")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    grid_kvec = np.asarray(grid.kvec, dtype=np.complex128)
    path_kvec = np.asarray(path.kvec, dtype=np.complex128)
    node_kvec = np.asarray([complex(node.kvec) for node in path.nodes], dtype=np.complex128)
    cell_vertices = np.asarray(sampled_cell_vertices(params), dtype=np.complex128)
    cell_loop = np.concatenate([cell_vertices, cell_vertices[:1]])
    bz_vertices = np.asarray(moire_bz_vertices(params), dtype=np.complex128)
    bz_loop = np.concatenate([bz_vertices, bz_vertices[:1]])

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(grid_kvec.real, grid_kvec.imag, s=10, color="#c7c7c7", alpha=0.82, linewidths=0.0, label="kmesh")
    ax.plot(cell_loop.real, cell_loop.imag, color="#7f7f7f", lw=1.0, ls="--", label="sampled cell")
    ax.plot(bz_loop.real, bz_loop.imag, color="#9467bd", lw=1.25, alpha=0.95, label="moire BZ")
    ax.plot(path_kvec.real, path_kvec.imag, color="#1f77b4", lw=1.7, label="path")
    if exact_grid_kvec.size > 0:
        ax.scatter(
            exact_grid_kvec.real,
            exact_grid_kvec.imag,
            s=24,
            color="#d62728",
            edgecolors="#ffffff",
            linewidths=0.35,
            zorder=3,
            label="grid points on path",
        )
    ax.scatter(node_kvec.real, node_kvec.imag, s=28, color="#111111", zorder=4, label="path nodes")
    label_map = {"Gamma": "Γ", "Kprime": "K'"}
    for node in path.nodes:
        ax.text(node.kx, node.ky, f" {label_map.get(node.label, node.label)}", fontsize=8, va="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title(
        f"lk={int(grid.lk)} {path_kind}\n"
        f"exact={int(exact_indices.size)}, segments={segment_counts}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a custom full B=0 TBG Hartree-Fock case with parameterized screened Coulomb interaction."
    )
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--lk", type=int, default=19)
    parser.add_argument("--lg", type=int, default=9)
    parser.add_argument(
        "--periodic-g-grid",
        dest="periodic_g_grid",
        action="store_true",
        default=True,
        help="Use periodic reciprocal-grid wrapping in the BM tunneling and HF overlaps. This is the default benchmark convention.",
    )
    parser.add_argument(
        "--zero-fill-g-grid",
        dest="periodic_g_grid",
        action="store_false",
        help="Use finite-cutoff zero-fill reciprocal-grid shifts for Zhang-convention diagnostic runs.",
    )
    parser.add_argument("--overlap-lg", type=int, default=None)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--precision", type=float, default=1e-5)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="Optional prior .npz state to warm-start from its density array.",
    )
    parser.add_argument(
        "--initial-state-resample",
        choices=("none", "bilinear", "nearest"),
        default="none",
        help="Resample initial-state density if its lk differs from --lk.",
    )
    parser.add_argument(
        "--path-kind",
        choices=("fig6", "gamma-m-k-gamma-kprime"),
        default="fig6",
        help="High-symmetry path to use for reconstructed and SCF-grid band artifacts.",
    )
    parser.add_argument(
        "--write-scf-path",
        dest="write_scf_path",
        action="store_true",
        default=True,
        help="Write unreconstructed SCF-grid-only path TSV and line plot for grid points on the chosen path. This is the default.",
    )
    parser.add_argument(
        "--skip-scf-path",
        dest="write_scf_path",
        action="store_false",
        help="Do not write the SCF-grid-only path TSV/line plot.",
    )
    parser.add_argument(
        "--write-reconstructed-path",
        action="store_true",
        help=(
            "Also build the off-grid reconstructed path Hamiltonian and dense path-band plot. "
            "This is off by default because cRPA/HF diagnostics should use SCF-grid path points."
        ),
    )
    parser.add_argument(
        "--summary-mode",
        choices=("root", "parts", "both"),
        default="root",
        help="Write the aggregate summary.tsv, per-init summary_parts, or both. Use parts for parallel per-init jobs.",
    )
    parser.add_argument(
        "--combine-summary-only",
        action="store_true",
        help="Only combine summary_parts/*.tsv for this theta/nu/run-tag into summary.tsv.",
    )
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.5)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument("--epsilon-r", type=float, default=4.0)
    parser.add_argument(
        "--crpa-dir",
        type=Path,
        default=None,
        help="Optional Zhang cRPA artifact directory. When set, keep the production full-HF workflow and replace only the Coulomb kernel with fixed cRPA screening.",
    )
    parser.add_argument(
        "--allow-incompatible-crpa",
        action="store_true",
        help=(
            "Bypass HF-compatible cRPA metadata checks. This is only for diagnostics: production HF+cRPA "
            "requires the cRPA artifact to use the same periodic-G/form-factor convention as the HF overlaps."
        ),
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Required with --allow-incompatible-crpa; marks the run as non-production diagnostics.",
    )
    parser.add_argument(
        "--strict-hf-compatible-crpa",
        action="store_true",
        help="Deprecated compatibility flag; HF-compatible cRPA validation is now the default unless --allow-incompatible-crpa is set.",
    )
    parser.add_argument(
        "--crpa-physics-reference-dir",
        type=Path,
        default=DEFAULT_CRPA_PHYSICS_REFERENCE_DIR,
        help="Deprecated: retained for CLI compatibility. The production physics gate uses corrected Fig. 1(e) paper anchors.",
    )
    parser.add_argument(
        "--skip-crpa-physics-gate",
        action="store_true",
        help=(
            "Skip the convention-aware cRPA gate. Fig. 1(e) is a hard gate only for "
            "zhang_zero_fill paper-reference artifacts, not for HF-compatible artifacts."
        ),
    )
    parser.add_argument("--crpa-fig1e-max-rmse", type=float, default=0.8)
    parser.add_argument("--crpa-fig1e-max-abs", type=float, default=1.5)
    parser.add_argument("--crpa-fig1e-max-mean-abs", type=float, default=0.7)
    parser.add_argument("--crpa-fig1e-min-paper-points", type=int, default=5)
    parser.add_argument(
        "--fock-interpolation",
        choices=("matrix_diagonal", "linear", "nearest"),
        default="matrix_diagonal",
        help="cRPA Fock lookup for the self-consistent HF grid. Use matrix_diagonal for production.",
    )
    parser.add_argument(
        "--path-fock-interpolation",
        choices=("linear", "nearest", "matrix_diagonal"),
        default="linear",
        help="cRPA Fock lookup for off-grid path-band evaluation. Path meshes normally need linear interpolation.",
    )
    parser.add_argument(
        "--tanh-argument-scale-a",
        type=float,
        default=400.0 / GRAPHENE_LATTICE_A_ANGSTROM,
        help="Dimensionless scale ds_code in tanh(|q| * ds_code). For ds=400 Angstrom use 400/2.46.",
    )
    parser.add_argument(
        "--zero-limit",
        choices=("finite", "zero"),
        default="finite",
        help="How to treat the screened Coulomb kernel for |q| below the cutoff.",
    )
    parser.add_argument("--zero-cutoff", type=float, default=1e-6)
    parser.add_argument(
        "--init",
        dest="init_specs",
        action="append",
        default=None,
        help="Optional init spec as mode or mode:seed. Repeat to override the default targeted set.",
    )
    return parser.parse_args()


def _parse_init_specs(raw_specs: list[str] | None) -> tuple[InitSpec, ...]:
    if not raw_specs:
        return _default_init_specs()
    parsed: list[InitSpec] = []
    for raw in raw_specs:
        if ":" in raw:
            mode, seed_text = raw.split(":", 1)
            parsed.append(InitSpec(mode.strip(), int(seed_text)))
        else:
            parsed.append(InitSpec(raw.strip(), 1))
    return tuple(parsed)


def main() -> int:
    args = parse_args()

    theta_deg = float(args.theta_deg)
    nu = float(args.nu)
    lk = int(args.lk)
    lg = int(args.lg)
    root_dir = Path(args.output_root) / f"theta_{_theta_tag(theta_deg)}_nu_{_nu_tag(nu)}_{args.run_tag}"
    if args.combine_summary_only:
        row_count = _combine_summary_parts(root_dir)
        print(f"[stage] combine_summary done output_dir={root_dir} rows={row_count}", flush=True)
        return 0
    if bool(args.allow_incompatible_crpa) and not bool(args.diagnostic_only):
        raise SystemExit(
            "--allow-incompatible-crpa now requires --diagnostic-only. "
            "Production HF bands must use HF-compatible cRPA metadata."
        )

    _ensure_not_running_on_login_node()

    overlap_lg = lg if args.overlap_lg is None else int(args.overlap_lg)
    periodic_g_grid = bool(args.periodic_g_grid)
    g_boundary_mode = "periodic" if periodic_g_grid else "zero_fill"
    tanh_argument_scale_a = float(args.tanh_argument_scale_a)
    screening_lm = tanh_argument_scale_a / 2.0
    finite_zero_limit = args.zero_limit == "finite"
    init_specs = _parse_init_specs(args.init_specs)
    initial_density, initial_state_entries = _load_initial_density(
        args.initial_state,
        target_lk=lk,
        resample_method=str(args.initial_state_resample),
    )

    state_dir = root_dir / "states"
    path_dir = root_dir / "path_bands"
    state_dir.mkdir(parents=True, exist_ok=True)
    path_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    total_start = perf_counter()
    print(
        f"[stage] setup theta={theta_deg} nu={nu} lk={lk} lg={lg} overlap_lg={overlap_lg} "
        f"periodic_g_grid={str(periodic_g_grid).lower()} g_boundary_mode={g_boundary_mode}",
        flush=True,
    )
    print(
        "[stage] interaction_input "
        f"requested_epsilon_r={args.epsilon_r} tanh_argument_scale_a={tanh_argument_scale_a:.16g} "
        f"screening_lm={screening_lm:.16g} finite_zero_limit={str(finite_zero_limit).lower()}",
        flush=True,
    )
    if initial_density is not None:
        print(f"[stage] resume initial_state={args.initial_state}", flush=True)
        if any(key == "initial_state_resampled" and value == "true" for key, value in initial_state_entries):
            print(f"[stage] resume initial_state_resample={args.initial_state_resample} target_lk={lk}", flush=True)

    params = TBGParameters.from_degrees(
        theta_deg,
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )
    crpa_result = None
    crpa_screening = None
    crpa_physics_gate: dict[str, float | int | str] = {}
    crpa_split_mode_value = ""
    crpa_remote_bare_scale_value = 0.0
    if args.crpa_dir is not None:
        crpa_split_mode_value = crpa_split_mode()
        crpa_remote_bare_scale_value = crpa_remote_bare_scale()
        crpa_result = load_crpa_result(args.crpa_dir)
        crpa_periodic_g_grid = bool(crpa_result.metadata.get("periodic_g_grid", False))
        if crpa_periodic_g_grid != periodic_g_grid and not bool(args.allow_incompatible_crpa):
            raise SystemExit(
                "cRPA/HF reciprocal-grid boundary mismatch: "
                f"HF periodic_g_grid={periodic_g_grid}, cRPA metadata periodic_g_grid={crpa_periodic_g_grid}. "
                "Use matching artifacts for production, or --allow-incompatible-crpa --diagnostic-only for an explicit diagnostic."
            )
        if not bool(args.allow_incompatible_crpa):
            validate_hf_compatible_crpa(crpa_result, params, theta_deg=theta_deg, overlap_lg=overlap_lg)
        crpa_convention = crpa_convention_family(crpa_result)
        if not bool(args.skip_crpa_physics_gate):
            if crpa_convention == "zhang_paper_reference":
                comparison = compare_fig1e_window_to_paper_points(crpa_result)
                failures = fig1e_paper_point_gate_failures(
                    comparison,
                    max_rmse=float(args.crpa_fig1e_max_rmse),
                    max_abs=float(args.crpa_fig1e_max_abs),
                    max_mean_abs=float(args.crpa_fig1e_max_mean_abs),
                    min_points=int(args.crpa_fig1e_min_paper_points),
                )
                crpa_physics_gate = {
                    key: float(value) if isinstance(value, float) else int(value) for key, value in comparison.items()
                }
                crpa_physics_gate["gate_type"] = "corrected_fig1e_paper_points"
                crpa_physics_gate["convention_family"] = crpa_convention
                if failures:
                    detail = "; ".join(failures)
                    raise SystemExit(f"cRPA Fig. 1(e) physics gate failed for {args.crpa_dir}: {detail}")
                print(
                    "[stage] crpa_physics_gate "
                    "reference=corrected_fig1e_paper_points "
                    f"fig1e_paper_rmse={float(comparison['fig1e_paper_rmse']):.6g} "
                    f"fig1e_paper_max_abs={float(comparison['fig1e_paper_max_abs']):.6g} "
                    f"fig1e_paper_mean_abs={float(comparison['fig1e_paper_mean_abs']):.6g}",
                    flush=True,
                )
            else:
                crpa_physics_gate = {
                    "gate_type": "hf_compatible_convention",
                    "convention_family": crpa_convention,
                    "fig1e_paper_gate": "diagnostic_only_not_a_hard_gate_for_hf",
                }
                print(
                    "[stage] crpa_physics_gate "
                    f"convention_family={crpa_convention} "
                    "fig1e_paper_gate=diagnostic_only_not_a_hard_gate_for_hf",
                    flush=True,
                )
        if str(args.fock_interpolation).strip().lower() == "matrix_diagonal":
            required_q_lg = overlap_lg + 2 if overlap_lg % 2 == 1 else overlap_lg
            if int(crpa_result.q_lg) < int(required_q_lg):
                raise SystemExit(
                    "Exact crpa_matrix_diagonal HF requires a larger cRPA Q table: "
                    f"crpa q_lg={crpa_result.q_lg}, overlap_lg={overlap_lg}, required q_lg>={required_q_lg}. "
                    "For the current endpoint-including B0 HF grid, overlap_lg=9 needs q_lg=11. "
                    "Do not fall back to radial interpolation for an exact Fig. 4 benchmark."
                )
        crpa_screening = CRPAScreenedCoulomb(crpa_result)
        print(
            "[stage] crpa "
            f"dir={args.crpa_dir} lk={crpa_result.lk} lg={crpa_result.lg} q_lg={crpa_result.q_lg} "
            f"epsilon_bn={float(crpa_result.coulomb_params.epsilon_bn):.16g} "
            f"fock_interpolation={args.fock_interpolation} split_mode={crpa_split_mode_value} "
            f"remote_bare_scale={crpa_remote_bare_scale_value:.16g}",
            flush=True,
        )
    bare_interaction_model = "double_gate_tanh_q0limit" if finite_zero_limit else "double_gate_tanh_zero_q0"
    interaction_model = "zhang_crpa_screened" if crpa_result is not None else bare_interaction_model
    screening_kwargs = {
        "relative_permittivity": float(args.epsilon_r) if crpa_result is None else float(crpa_result.coulomb_params.epsilon_bn),
        "screening_lm": screening_lm if crpa_result is None else float(crpa_result.coulomb_params.screening_lm),
        "finite_zero_limit": finite_zero_limit if crpa_result is None else bool(crpa_result.coulomb_params.finite_zero_limit),
        "zero_cutoff": float(args.zero_cutoff) if crpa_result is None else float(crpa_result.coulomb_params.zero_cutoff),
    }
    print(
        "[stage] interaction_effective "
        f"model={interaction_model} relative_permittivity={float(screening_kwargs['relative_permittivity']):.16g} "
        f"screening_lm={float(screening_kwargs['screening_lm']):.16g} "
        f"finite_zero_limit={str(bool(screening_kwargs['finite_zero_limit'])).lower()}",
        flush=True,
    )

    bm_start = perf_counter()
    grid = build_b0_uniform_lattice(params, lk)
    grid_solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True, periodic_g_grid=periodic_g_grid)
    bm_elapsed = perf_counter() - bm_start
    print(f"[stage] bm_grid done elapsed_sec={bm_elapsed:.3f} nk={grid_solution.nk}", flush=True)

    overlap_start = perf_counter()
    grid_overlap = build_overlap_block_set(grid_solution, lg=overlap_lg, **screening_kwargs)
    overlap_elapsed = perf_counter() - overlap_start
    print(f"[stage] grid_overlap done elapsed_sec={overlap_elapsed:.3f} shifts={len(grid_overlap.shifts)}", flush=True)

    q_lookup_diagnostics = _lookup_diagnostics_for_scf_grid(
        crpa_screening=crpa_screening,
        overlap_blocks=grid_overlap,
        lattice_kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
        method=str(args.fock_interpolation),
    )
    if crpa_screening is not None:
        print(
            "[stage] crpa_q_lookup "
            f"method={q_lookup_diagnostics['method']} "
            f"failures={q_lookup_diagnostics['q_lookup_failures']} "
            f"fallbacks={q_lookup_diagnostics['q_lookup_fallbacks']} "
            f"max_residual_nm_inv={float(q_lookup_diagnostics['max_q_reconstruction_residual_nm_inv']):.3e} "
            f"eps_min={float(q_lookup_diagnostics['eps_crpa_min']):.6g} "
            f"eps_mean={float(q_lookup_diagnostics['eps_crpa_mean']):.6g} "
            f"eps_max={float(q_lookup_diagnostics['eps_crpa_max']):.6g}",
            flush=True,
        )

    path_start = perf_counter()
    path = _build_path(params, path_kind=str(args.path_kind), points_per_segment=int(args.points_per_segment))
    path_h0 = None
    path_overlap = None
    path_grid_overlap = None
    path_grid_overlap_bare = None
    if args.write_reconstructed_path:
        path_solution = solve_bm_model(params, path.kvec, lg=lg, sigma_rotation=True, periodic_g_grid=periodic_g_grid)
        path_h0 = build_h0_from_bm(path_solution)
        path_overlap = build_overlap_block_set(path_solution, lg=overlap_lg, **screening_kwargs)
        path_grid_overlap = build_overlap_block_set(
            path_solution,
            source_solution=grid_solution,
            lg=overlap_lg,
            **screening_kwargs,
        )
        path_grid_overlap_bare = path_grid_overlap
        if crpa_screening is not None:
            path_grid_overlap = build_fock_screened_overlap_blocks(
                path_grid_overlap_bare,
                target_kvec=np.asarray(path_solution.lattice_kvec, dtype=np.complex128),
                source_kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
                params=params,
                crpa_screening=crpa_screening,
                fock_interpolation=str(args.path_fock_interpolation),
                **screening_kwargs,
            )
    path_setup_elapsed = perf_counter() - path_start
    _write_kmesh_path_overlay(path_dir, params=params, grid=grid, path=path, path_kind=str(args.path_kind))
    print(
        f"[stage] path_setup done elapsed_sec={path_setup_elapsed:.3f} "
        f"path_kind={args.path_kind} path_points={path.kvec.size} "
        f"write_scf_path={str(bool(args.write_scf_path)).lower()} "
        f"write_reconstructed_path={str(bool(args.write_reconstructed_path)).lower()}",
        flush=True,
    )

    rows: list[dict[str, str]] = []
    for spec in init_specs:
        tag = _case_tag(theta_deg, nu, spec.init_mode, spec.seed, lk, lg)
        state_path = state_dir / f"{tag}.npz"
        path_tsv = path_dir / f"{tag}_hf_path.tsv"
        nodes_tsv = path_dir / f"{tag}_hf_path_nodes.tsv"
        path_summary = path_dir / f"{tag}_hf_path_summary.txt"
        scf_path_tsv = path_dir / f"{tag}_hf_scf_path.tsv"

        print(f"[stage] hf:start init={spec.init_mode} seed={spec.seed}", flush=True)
        hf_start = perf_counter()
        state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=nu, precision=float(args.precision))
        if crpa_screening is None:
            hf_run = run_full_hartree_fock(
                state,
                grid_overlap,
                grid_solution.lattice_kvec,
                params,
                init_mode=spec.init_mode,
                seed=spec.seed,
                max_iter=int(args.max_iter),
                initial_density=initial_density,
            )
        else:
            hf_run = run_full_crpa_hartree_fock(
                state,
                grid_overlap,
                grid_solution.lattice_kvec,
                params,
                crpa_screening=crpa_screening,
                init_mode=spec.init_mode,
                seed=spec.seed,
                max_iter=int(args.max_iter),
                initial_density=initial_density,
                fock_interpolation=str(args.fock_interpolation),
            )
        hf_elapsed = perf_counter() - hf_start
        print(
            "[stage] hf:done "
            f"init={spec.init_mode} seed={spec.seed} elapsed_sec={hf_elapsed:.3f} "
            f"iterations={hf_run.iterations} converged={str(hf_run.converged).lower()} "
            f"exit_reason={hf_run.exit_reason}",
            flush=True,
        )

        gap_summary = _gap_summary(hf_run.state.energies, nu)
        order_summary = _order_parameter_summary(
            hf_run.state.density,
            hf_run.state.energies,
            hf_run.state.sigma_ztauz,
            nu,
        )
        if crpa_screening is not None:
            energy_overlap = build_fock_screened_overlap_blocks(
                grid_overlap,
                lattice_kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
                params=params,
                crpa_screening=crpa_screening,
                fock_interpolation=str(args.fock_interpolation),
                **screening_kwargs,
            )
            if crpa_split_uses_hartree_delta_fock_projector(crpa_split_mode_value):
                hartree_h, fock_h = build_crpa_hartree_delta_fock_projector_components(
                    hf_run.state.density,
                    energy_overlap,
                    crpa_screening=crpa_screening,
                    params=params,
                )
                energy_summary = crpa_hartree_delta_fock_projector_energy_components(
                    hf_run.state.h0,
                    hf_run.state.density,
                    hartree_h,
                    fock_h,
                )
            else:
                hartree_h, fock_h = build_crpa_projected_interaction_components(
                    crpa_active_density_from_delta(hf_run.state.density, crpa_split_mode_value),
                    energy_overlap,
                    crpa_screening=crpa_screening,
                    params=params,
                )
                energy_summary = crpa_hf_energy_components(
                    hf_run.state.h0,
                    hf_run.state.density,
                    hartree_h,
                    fock_h,
                )
        else:
            nk_energy = int(hf_run.state.density.shape[2])
            e_band = np.einsum("abk,abk->", hf_run.state.h0, hf_run.state.density, optimize=True).real / float(
                nk_energy
            )
            e_total = float(hf_run.iter_energy[-1]) if hf_run.iter_energy.size else float("nan")
            energy_summary = {
                "E_band": float(e_band),
                "E_Hartree": float("nan"),
                "E_Fock": float("nan"),
                "E_total": e_total,
            }

        np.savez_compressed(
            state_path,
            density=hf_run.state.density,
            hamiltonian=hf_run.state.hamiltonian,
            h0=hf_run.state.h0,
            energies=hf_run.state.energies,
            sigma_ztauz=hf_run.state.sigma_ztauz,
            sigma_z=hf_run.state.sigma_z,
            mu=np.asarray([hf_run.state.mu], dtype=float),
            iter_energy=hf_run.iter_energy,
            iter_err=hf_run.iter_err,
            iter_oda=hf_run.iter_oda,
            nu=np.asarray([nu], dtype=float),
            init_mode=np.asarray([spec.init_mode]),
            normalized_init_mode=np.asarray([hf_run.init_mode]),
            seed=np.asarray([spec.seed], dtype=int),
            converged=np.asarray([hf_run.converged]),
            exit_reason=np.asarray([hf_run.exit_reason]),
            theta_deg=np.asarray([theta_deg], dtype=float),
            lk=np.asarray([lk], dtype=int),
            lg=np.asarray([lg], dtype=int),
            overlap_lg=np.asarray([overlap_lg], dtype=int),
            periodic_g_grid=np.asarray([periodic_g_grid]),
            g_boundary_mode=np.asarray([g_boundary_mode]),
            w0_mev=np.asarray([float(args.w0)], dtype=float),
            w1_mev=np.asarray([float(args.w1)], dtype=float),
            vf_mev=np.asarray([float(args.vf)], dtype=float),
            epsilon_r=np.asarray([float(args.epsilon_r)], dtype=float),
            effective_relative_permittivity=np.asarray([float(screening_kwargs["relative_permittivity"])], dtype=float),
            tanh_argument_scale_a=np.asarray([tanh_argument_scale_a], dtype=float),
            screening_lm=np.asarray([float(screening_kwargs["screening_lm"])], dtype=float),
            q_zero_limit=np.asarray([bool(screening_kwargs["finite_zero_limit"])]),
            interaction_model=np.asarray([interaction_model]),
            crpa_dir=np.asarray(["" if args.crpa_dir is None else str(args.crpa_dir)]),
            crpa_lk=np.asarray([-1 if crpa_result is None else int(crpa_result.lk)], dtype=int),
            crpa_lg=np.asarray([-1 if crpa_result is None else int(crpa_result.lg)], dtype=int),
            crpa_q_lg=np.asarray([-1 if crpa_result is None else int(crpa_result.q_lg)], dtype=int),
            crpa_metadata_json=np.asarray(["" if crpa_result is None else json.dumps(crpa_result.metadata, sort_keys=True)]),
            crpa_split_mode=np.asarray([crpa_split_mode_value]),
            crpa_remote_bare_scale=np.asarray([crpa_remote_bare_scale_value], dtype=float),
            fock_interpolation=np.asarray([str(args.fock_interpolation)]),
            q_lookup_diagnostics_json=np.asarray([json.dumps(q_lookup_diagnostics, sort_keys=True)]),
            crpa_physics_gate_json=np.asarray([json.dumps(crpa_physics_gate, sort_keys=True)]),
            diagnostic_only=np.asarray([bool(args.diagnostic_only)]),
            energy_summary_json=np.asarray([json.dumps(energy_summary, sort_keys=True)]),
            order_parameters_json=np.asarray([json.dumps(order_summary, sort_keys=True)]),
            indirect_gap_mev=np.asarray([float(gap_summary["indirect_gap_mev"])], dtype=float),
            direct_gap_mev=np.asarray([float(gap_summary["direct_gap_mev"])], dtype=float),
            max_iter=np.asarray([int(args.max_iter)], dtype=int),
            resumed_from_state=np.asarray(["" if args.initial_state is None else str(args.initial_state)]),
        )

        scf_path_elapsed = 0.0
        scf_plot_paths: dict[str, Path] | None = None
        scf_path_gap_summary = {
            "indirect_gap_mev": float("nan"),
            "direct_gap_mev": float("nan"),
        }
        if args.write_scf_path:
            scf_path_start = perf_counter()
            scf_path_result = build_restricted_hf_scf_path_plot_result(
                hf_run,
                grid_solution,
                path=path,
                init_mode=spec.init_mode,
            )
            scf_path_gap_summary = _gap_summary(scf_path_result.band_data.energies, nu)
            write_hf_scf_path_tsv(scf_path_tsv, scf_path_result)
            scf_plot_paths = write_hf_scf_band_plot(path_dir, scf_path_result, stem=f"{tag}_scf_grid_band_plot")
            scf_path_elapsed = perf_counter() - scf_path_start
            print(
                f"[stage] scf_path:done init={spec.init_mode} seed={spec.seed} "
                f"elapsed_sec={scf_path_elapsed:.3f} scf_path_tsv={scf_path_tsv}",
                flush=True,
            )

        path_elapsed = 0.0
        reconstructed_plot_paths: dict[str, Path] | None = None
        if args.write_reconstructed_path:
            if path_h0 is None or path_overlap is None or path_grid_overlap is None:
                raise RuntimeError("Reconstructed path setup was not initialized.")
            print(f"[stage] path:start init={spec.init_mode} seed={spec.seed}", flush=True)
            eval_start = perf_counter()
            if crpa_screening is None:
                h_path = build_projected_target_hamiltonian(
                    path_h0,
                    hf_run.state.density,
                    source_overlap_blocks=grid_overlap,
                    target_overlap_blocks=path_overlap,
                    target_source_overlap_blocks=path_grid_overlap,
                    v0=hf_run.state.v0,
                )
            else:
                if crpa_split_uses_remote_bare(crpa_split_mode_value):
                    if path_grid_overlap_bare is None:
                        raise RuntimeError("Bare path-grid overlap was not initialized.")
                    remote_path_hartree, remote_path_fock = build_bare_projected_target_components(
                        half_reference_delta_like(hf_run.state.density),
                        source_overlap_blocks=grid_overlap,
                        target_overlap_blocks=path_overlap,
                        target_source_overlap_blocks=path_grid_overlap_bare,
                        v0=hf_run.state.v0,
                    )
                    remote_path_bare = select_remote_reference_components(
                        remote_path_hartree,
                        remote_path_fock,
                        crpa_split_mode_value,
                    )
                    remote_path_bare *= crpa_remote_bare_scale_value
                else:
                    remote_path_bare = np.zeros_like(path_h0)
                if crpa_split_uses_active_cnp_reference(crpa_split_mode_value):
                    active_cnp_path_projector = active_lower_flat_projector_like(
                        hf_run.state.density,
                        n_spin=hf_run.state.n_spin,
                        n_eta=hf_run.state.n_eta,
                        n_band=hf_run.state.n_band,
                    )
                    active_cnp_path_hartree, active_cnp_path_fock = build_crpa_projected_target_components(
                        active_cnp_path_projector,
                        source_overlap_blocks=grid_overlap,
                        target_overlap_blocks=path_overlap,
                        target_source_overlap_blocks=path_grid_overlap,
                        crpa_screening=crpa_screening,
                        params=params,
                    )
                    active_cnp_path_reference = select_active_cnp_reference_components(
                        active_cnp_path_hartree,
                        active_cnp_path_fock,
                        crpa_split_mode_value,
                    )
                else:
                    active_cnp_path_reference = np.zeros_like(path_h0)
                path_base_hamiltonian = path_h0 + remote_path_bare + active_cnp_path_reference
                if crpa_split_uses_hartree_delta_fock_projector(crpa_split_mode_value):
                    path_hartree, path_fock = build_crpa_projected_target_components_from_densities(
                        hf_run.state.density,
                        physical_projector_from_delta(hf_run.state.density),
                        source_overlap_blocks=grid_overlap,
                        target_overlap_blocks=path_overlap,
                        target_source_overlap_blocks=path_grid_overlap,
                        crpa_screening=crpa_screening,
                        params=params,
                    )
                    h_path = path_base_hamiltonian + path_hartree + path_fock
                else:
                    h_path = build_crpa_projected_target_hamiltonian(
                        path_base_hamiltonian,
                        crpa_active_density_from_delta(hf_run.state.density, crpa_split_mode_value),
                        source_overlap_blocks=grid_overlap,
                        target_overlap_blocks=path_overlap,
                        target_source_overlap_blocks=path_grid_overlap,
                        crpa_screening=crpa_screening,
                        params=params,
                    )
            band_data = build_flavor_band_data(
                h_path,
                n_spin=hf_run.state.n_spin,
                n_eta=hf_run.state.n_eta,
                n_band=hf_run.state.n_band,
            )
            path_result = HFPathResult(
                params=params,
                path=path,
                hamiltonian=h_path,
                band_data=band_data,
                mu=hf_run.state.mu,
                nu=nu,
                lk=lk,
                lg=lg,
                points_per_segment=int(args.points_per_segment),
                init_mode=spec.init_mode,
                normalized_init_mode=hf_run.init_mode,
                seed=spec.seed,
                exit_reason=hf_run.exit_reason,
                overlap_lg=overlap_lg,
                relative_permittivity=float(screening_kwargs["relative_permittivity"]),
                screening_lm=float(screening_kwargs["screening_lm"]),
                finite_zero_limit=bool(screening_kwargs["finite_zero_limit"]),
                zero_cutoff=float(screening_kwargs["zero_cutoff"]),
            )
            path_elapsed = perf_counter() - eval_start
            write_hf_path_tsv(path_tsv, path_result)
            write_hf_path_nodes_tsv(nodes_tsv, path_result)
            write_hf_path_summary(path_summary, path_result, hf_state_path=str(state_path))
            _append_key_value_file(
                path_summary,
                [
                    ("w0_meV", f"{float(args.w0):.16g}"),
                    ("w1_meV", f"{float(args.w1):.16g}"),
                    ("vf_meV", f"{float(args.vf):.16g}"),
                    ("periodic_g_grid", str(periodic_g_grid).lower()),
                    ("g_boundary_mode", g_boundary_mode),
                    ("epsilon_r", f"{float(args.epsilon_r):.16g}"),
                    ("effective_relative_permittivity", f"{float(screening_kwargs['relative_permittivity']):.16g}"),
                    ("tanh_argument_scale_a", f"{tanh_argument_scale_a:.16g}"),
                    ("screening_lm", f"{float(screening_kwargs['screening_lm']):.16g}"),
                    ("interaction_model", interaction_model),
                    ("q_zero_limit", str(bool(screening_kwargs["finite_zero_limit"])).lower()),
                    ("crpa_dir", "" if args.crpa_dir is None else str(args.crpa_dir)),
                    ("crpa_lk", "" if crpa_result is None else str(crpa_result.lk)),
                    ("crpa_lg", "" if crpa_result is None else str(crpa_result.lg)),
                    ("crpa_q_lg", "" if crpa_result is None else str(crpa_result.q_lg)),
                    ("crpa_split_mode", crpa_split_mode_value),
                    ("crpa_remote_bare_scale", f"{crpa_remote_bare_scale_value:.16g}"),
                    ("crpa_physics_gate_json", json.dumps(crpa_physics_gate, sort_keys=True)),
                    ("fock_interpolation", str(args.fock_interpolation)),
                    ("path_fock_interpolation", str(args.path_fock_interpolation)),
                    ("diagnostic_only", str(bool(args.diagnostic_only)).lower()),
                    ("path_kind", str(args.path_kind)),
                    ("path_labels", "-".join(path.labels)),
                    ("write_scf_path", str(bool(args.write_scf_path)).lower()),
                    ("write_reconstructed_path", str(bool(args.write_reconstructed_path)).lower()),
                    ("scf_path_tsv", str(scf_path_tsv) if args.write_scf_path else ""),
                    ("scf_path_elapsed_sec", f"{scf_path_elapsed:.16e}"),
                    *initial_state_entries,
                ],
            )
            reconstructed_plot_paths = write_hf_band_plot(path_dir, path_result, stem=f"{tag}_band_plot")
            print(
                f"[stage] path:done init={spec.init_mode} seed={spec.seed} "
                f"elapsed_sec={path_elapsed:.3f} path_tsv={path_tsv}",
                flush=True,
            )
        elif not args.write_scf_path:
            raise ValueError("At least one of --write-scf-path or --write-reconstructed-path must be enabled.")
        else:
            print(
                f"[stage] path:skip_reconstructed init={spec.init_mode} seed={spec.seed} "
                f"scf_path_tsv={scf_path_tsv}",
                flush=True,
            )

        canonical_tsv = scf_path_tsv if args.write_scf_path else path_tsv
        canonical_summary = "" if args.write_scf_path else str(path_summary)
        canonical_plot_paths = scf_plot_paths if scf_plot_paths is not None else reconstructed_plot_paths
        if canonical_plot_paths is None:
            raise RuntimeError("No canonical band plot was generated.")
        screening_label = "crpa" if crpa_result is not None else "no_crpa"
        nu_label = _nu_file_label(nu)
        fig4_payload: dict[str, object] = {
            "theta_deg": theta_deg,
            "nu": nu,
            "init_mode": spec.init_mode,
            "seed": spec.seed,
            "screening_mode": "crpa_matrix_diagonal" if crpa_result is not None else "no_crpa",
            "interaction_model": interaction_model,
            "periodic_g_grid": periodic_g_grid,
            "g_boundary_mode": g_boundary_mode,
            "crpa_split_mode": crpa_split_mode_value,
            "crpa_remote_bare_scale": crpa_remote_bare_scale_value,
            "hartree_crpa_convention": "full_matrix_qe_0" if crpa_result is not None else "bare_bn_screened_scalar",
            "fock_convention": (
                "V_bare_with_BN / eps_crpa; eps_total is plotting only"
                if crpa_result is not None
                else "V_bare_with_BN without cRPA"
            ),
            "converged": bool(hf_run.converged),
            "exit_reason": hf_run.exit_reason,
            "iterations": int(hf_run.iterations),
            "convergence_residual": float(hf_run.iter_err[-1]) if hf_run.iter_err.size else float("nan"),
            "mu_mev": float(hf_run.state.mu),
            **energy_summary,
            **gap_summary,
            "chern_number": float(order_summary["chern_proxy_sigma_ztauz"]),
            "chern_number_note": "sigma_z_tau_z occupied-sum proxy; integer Chern post-check still required",
            **order_summary,
            **q_lookup_diagnostics,
            "state_path": str(state_path),
            "path_tsv": str(canonical_tsv),
            "path_summary": canonical_summary,
            "band_plot_png": str(canonical_plot_paths["band_plot_png"]),
            "band_plot_pdf": str(canonical_plot_paths["band_plot_pdf"]),
            "scf_path_tsv": str(scf_path_tsv) if args.write_scf_path else "",
            "scf_band_plot_png": (
                "" if scf_plot_paths is None else str(scf_plot_paths["band_plot_png"])
            ),
            "scf_band_plot_pdf": (
                "" if scf_plot_paths is None else str(scf_plot_paths["band_plot_pdf"])
            ),
            "reconstructed_path_tsv": str(path_tsv) if args.write_reconstructed_path else "",
            "reconstructed_path_summary": str(path_summary) if args.write_reconstructed_path else "",
            "reconstructed_band_plot_png": (
                "" if reconstructed_plot_paths is None else str(reconstructed_plot_paths["band_plot_png"])
            ),
            "reconstructed_band_plot_pdf": (
                "" if reconstructed_plot_paths is None else str(reconstructed_plot_paths["band_plot_pdf"])
            ),
        }
        _write_json(root_dir / f"hf_summary_nu_{nu_label}_{screening_label}.json", fig4_payload)
        _write_json(root_dir / f"order_parameters_nu_{nu_label}.json", order_summary)
        _copy_if_exists(canonical_tsv, root_dir / f"hf_band_nu_{nu_label}_{screening_label}.csv")
        _copy_if_exists(canonical_plot_paths["band_plot_png"], root_dir / f"hf_band_nu_{nu_label}_{screening_label}.png")
        _copy_if_exists(state_path, root_dir / f"density_matrix_final_nu_{nu_label}.npz")

        rows.append(
            {
                "theta_deg": f"{theta_deg:.16g}",
                "nu": f"{nu:.16g}",
                "init_mode": spec.init_mode,
                "seed": str(spec.seed),
                "iterations": str(hf_run.iterations),
                "exit_reason": hf_run.exit_reason,
                "converged": str(hf_run.converged).lower(),
                "mu_mev": f"{float(hf_run.state.mu):.16e}",
                "final_energy": "" if hf_run.iter_energy.size == 0 else f"{float(hf_run.iter_energy[-1]):.16e}",
                "final_error": "" if hf_run.iter_err.size == 0 else f"{float(hf_run.iter_err[-1]):.16e}",
                "final_oda": "" if hf_run.iter_oda.size == 0 else f"{float(hf_run.iter_oda[-1]):.16e}",
                "hf_elapsed_sec": f"{hf_elapsed:.16e}",
                "path_elapsed_sec": f"{path_elapsed:.16e}",
                "gap_source": "full_scf_grid",
                "indirect_gap_mev": f"{float(gap_summary['indirect_gap_mev']):.16e}",
                "direct_gap_mev": f"{float(gap_summary['direct_gap_mev']):.16e}",
                "full_scf_grid_indirect_gap_mev": f"{float(gap_summary['indirect_gap_mev']):.16e}",
                "full_scf_grid_direct_gap_mev": f"{float(gap_summary['direct_gap_mev']):.16e}",
                "scf_path_indirect_gap_mev": f"{float(scf_path_gap_summary['indirect_gap_mev']):.16e}",
                "scf_path_direct_gap_mev": f"{float(scf_path_gap_summary['direct_gap_mev']):.16e}",
                "E_band": f"{float(energy_summary['E_band']):.16e}",
                "E_Hartree": f"{float(energy_summary['E_Hartree']):.16e}",
                "E_Fock": f"{float(energy_summary['E_Fock']):.16e}",
                "E_total": f"{float(energy_summary['E_total']):.16e}",
                "valley_polarization_tau_z": f"{float(order_summary['valley_polarization_tau_z']):.16e}",
                "spin_polarization": f"{float(order_summary['spin_polarization']):.16e}",
                "kivc_amplitude": f"{float(order_summary['kivc_amplitude']):.16e}",
                "svp_amplitude": f"{float(order_summary['svp_amplitude']):.16e}",
                "chern_proxy_sigma_ztauz": f"{float(order_summary['chern_proxy_sigma_ztauz']):.16e}",
                "q_lookup_failures": str(int(q_lookup_diagnostics["q_lookup_failures"])),
                "q_lookup_fallbacks": str(int(q_lookup_diagnostics["q_lookup_fallbacks"])),
                "max_q_reconstruction_residual_nm_inv": f"{float(q_lookup_diagnostics['max_q_reconstruction_residual_nm_inv']):.16e}",
                "eps_crpa_min": f"{float(q_lookup_diagnostics['eps_crpa_min']):.16e}",
                "eps_crpa_mean": f"{float(q_lookup_diagnostics['eps_crpa_mean']):.16e}",
                "eps_crpa_max": f"{float(q_lookup_diagnostics['eps_crpa_max']):.16e}",
                "crpa_split_mode": crpa_split_mode_value,
                "state_path": str(state_path),
                "path_tsv": str(canonical_tsv),
            }
        )
        if args.summary_mode in ("parts", "both"):
            _write_summary_table(root_dir / "summary_parts" / f"{tag}.tsv", [rows[-1]])
        if args.summary_mode in ("root", "both"):
            _write_summary_table(root_dir / "summary.tsv", rows)

    total_elapsed = perf_counter() - total_start
    finished = datetime.now()
    run_info_entries = [
        ("theta_deg", f"{theta_deg:.16g}"),
        ("theta_tag", _theta_tag(theta_deg)),
        ("nu", f"{nu:.16g}"),
        ("nu_tag", _nu_tag(nu)),
        ("run_tag", str(args.run_tag)),
        ("lk", str(lk)),
        ("lg", str(lg)),
        ("overlap_lg", str(overlap_lg)),
        ("periodic_g_grid", str(periodic_g_grid).lower()),
        ("g_boundary_mode", g_boundary_mode),
        ("points_per_segment", str(int(args.points_per_segment))),
        ("path_kind", str(args.path_kind)),
        ("path_labels", "-".join(path.labels)),
        ("write_scf_path", str(bool(args.write_scf_path)).lower()),
        ("initial_state_resample", str(args.initial_state_resample)),
        ("max_iter", str(int(args.max_iter))),
        ("precision", f"{float(args.precision):.16g}"),
        ("w0_meV", f"{float(args.w0):.16g}"),
        ("w1_meV", f"{float(args.w1):.16g}"),
        ("vf_meV", f"{float(args.vf):.16g}"),
        ("epsilon_r", f"{float(args.epsilon_r):.16g}"),
        ("effective_relative_permittivity", f"{float(screening_kwargs['relative_permittivity']):.16g}"),
        ("tanh_argument_scale_a", f"{tanh_argument_scale_a:.16g}"),
        ("screening_lm", f"{float(screening_kwargs['screening_lm']):.16g}"),
        ("physical_ds_angstrom", f"{tanh_argument_scale_a * GRAPHENE_LATTICE_A_ANGSTROM:.16g}"),
        ("interaction_model", interaction_model),
        ("q_zero_limit", str(bool(screening_kwargs["finite_zero_limit"])).lower()),
        ("allow_incompatible_crpa", str(bool(args.allow_incompatible_crpa)).lower()),
        ("diagnostic_only", str(bool(args.diagnostic_only)).lower()),
        ("crpa_dir", "" if args.crpa_dir is None else str(args.crpa_dir)),
        ("crpa_lk", "" if crpa_result is None else str(crpa_result.lk)),
        ("crpa_lg", "" if crpa_result is None else str(crpa_result.lg)),
        ("crpa_q_lg", "" if crpa_result is None else str(crpa_result.q_lg)),
        ("crpa_split_mode", crpa_split_mode_value),
        ("crpa_remote_bare_scale", f"{crpa_remote_bare_scale_value:.16g}"),
        ("crpa_metadata_json", "" if crpa_result is None else json.dumps(crpa_result.metadata, sort_keys=True)),
        ("crpa_physics_gate_json", json.dumps(crpa_physics_gate, sort_keys=True)),
        ("fock_interpolation", str(args.fock_interpolation)),
        ("path_fock_interpolation", str(args.path_fock_interpolation)),
        ("init_specs", ",".join(f"{spec.init_mode}:{spec.seed}" for spec in init_specs)),
        *initial_state_entries,
        ("start_time", started.strftime("%Y-%m-%dT%H:%M:%S")),
        ("end_time", finished.strftime("%Y-%m-%dT%H:%M:%S")),
        ("bm_elapsed_sec", f"{bm_elapsed:.16e}"),
        ("grid_overlap_elapsed_sec", f"{overlap_elapsed:.16e}"),
        ("path_setup_elapsed_sec", f"{path_setup_elapsed:.16e}"),
        ("total_elapsed_sec", f"{total_elapsed:.16e}"),
        ("hostname", socket.gethostname()),
        ("output_dir", str(root_dir)),
    ]
    if args.summary_mode in ("root", "both"):
        _write_key_value_file(root_dir / "run_info.txt", run_info_entries)
    if args.summary_mode in ("parts", "both"):
        task_tag = _case_tag(theta_deg, nu, init_specs[0].init_mode, init_specs[0].seed, lk, lg) if len(init_specs) == 1 else "multi_init"
        _write_key_value_file(root_dir / "run_info_parts" / f"{task_tag}.txt", run_info_entries)
    print(f"[stage] complete output_dir={root_dir} total_elapsed_sec={total_elapsed:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
