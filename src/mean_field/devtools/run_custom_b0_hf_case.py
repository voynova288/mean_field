#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.core.hf import build_flavor_band_data, build_projected_target_hamiltonian
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
from mean_field.devtools.resample_b0_density_stack import resample_density_stack


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "TBG_HF" / "custom_b0_hf_targeted_runs"
DEFAULT_RUN_TAG = "eps4_gate400a_ds400over2p46_q0limit_w0_79p7_w1_97p5_vf_2135p4_20260425_meanfield"
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
    import os

    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().strip().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit(
            f"Refusing to run custom B0 HF on login node {hostname}; submit it through Slurm from login002."
        )


def _theta_tag(theta_deg: float) -> str:
    return f"{round(theta_deg * 100):03d}"


def _nu_tag(nu: float) -> str:
    return f"{round(nu * 1000):+05d}"


def _case_tag(theta_deg: float, nu: float, init_mode: str, seed: int, lk: int, lg: int) -> str:
    return f"theta_{_theta_tag(theta_deg)}_nu_{_nu_tag(nu)}_init_{init_mode}_seed_{seed:03d}_lk{lk}_lg{lg}"


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
        "state_path",
        "path_tsv",
    ]


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
        action="store_true",
        help="Also write unreconstructed SCF-grid-only path TSV and band plot for grid points on the chosen path.",
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

    _ensure_not_running_on_login_node()

    overlap_lg = lg if args.overlap_lg is None else int(args.overlap_lg)
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
    print(f"[stage] setup theta={theta_deg} nu={nu} lk={lk} lg={lg} overlap_lg={overlap_lg}", flush=True)
    print(
        "[stage] interaction "
        f"epsilon_r={args.epsilon_r} tanh_argument_scale_a={tanh_argument_scale_a:.16g} "
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
    screening_kwargs = {
        "relative_permittivity": float(args.epsilon_r),
        "screening_lm": screening_lm,
        "finite_zero_limit": finite_zero_limit,
        "zero_cutoff": float(args.zero_cutoff),
    }

    bm_start = perf_counter()
    grid = build_b0_uniform_lattice(params, lk)
    grid_solution = solve_bm_model(params, grid.kvec, lg=lg, sigma_rotation=True)
    bm_elapsed = perf_counter() - bm_start
    print(f"[stage] bm_grid done elapsed_sec={bm_elapsed:.3f} nk={grid_solution.nk}", flush=True)

    overlap_start = perf_counter()
    grid_overlap = build_overlap_block_set(grid_solution, lg=overlap_lg, **screening_kwargs)
    overlap_elapsed = perf_counter() - overlap_start
    print(f"[stage] grid_overlap done elapsed_sec={overlap_elapsed:.3f} shifts={len(grid_overlap.shifts)}", flush=True)

    path_start = perf_counter()
    path = _build_path(params, path_kind=str(args.path_kind), points_per_segment=int(args.points_per_segment))
    path_solution = solve_bm_model(params, path.kvec, lg=lg, sigma_rotation=True)
    path_h0 = build_h0_from_bm(path_solution)
    path_overlap = build_overlap_block_set(path_solution, lg=overlap_lg, **screening_kwargs)
    path_grid_overlap = build_overlap_block_set(path_solution, source_solution=grid_solution, lg=overlap_lg, **screening_kwargs)
    path_setup_elapsed = perf_counter() - path_start
    _write_kmesh_path_overlay(path_dir, params=params, grid=grid, path=path, path_kind=str(args.path_kind))
    print(
        f"[stage] path_setup done elapsed_sec={path_setup_elapsed:.3f} "
        f"path_kind={args.path_kind} path_points={path.kvec.size}",
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
        hf_elapsed = perf_counter() - hf_start
        print(
            "[stage] hf:done "
            f"init={spec.init_mode} seed={spec.seed} elapsed_sec={hf_elapsed:.3f} "
            f"iterations={hf_run.iterations} converged={str(hf_run.converged).lower()} "
            f"exit_reason={hf_run.exit_reason}",
            flush=True,
        )

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
            w0_mev=np.asarray([float(args.w0)], dtype=float),
            w1_mev=np.asarray([float(args.w1)], dtype=float),
            vf_mev=np.asarray([float(args.vf)], dtype=float),
            epsilon_r=np.asarray([float(args.epsilon_r)], dtype=float),
            tanh_argument_scale_a=np.asarray([tanh_argument_scale_a], dtype=float),
            screening_lm=np.asarray([screening_lm], dtype=float),
            q_zero_limit=np.asarray([finite_zero_limit]),
            max_iter=np.asarray([int(args.max_iter)], dtype=int),
            resumed_from_state=np.asarray(["" if args.initial_state is None else str(args.initial_state)]),
        )

        print(f"[stage] path:start init={spec.init_mode} seed={spec.seed}", flush=True)
        eval_start = perf_counter()
        h_path = build_projected_target_hamiltonian(
            path_h0,
            hf_run.state.density,
            source_overlap_blocks=grid_overlap,
            target_overlap_blocks=path_overlap,
            target_source_overlap_blocks=path_grid_overlap,
            v0=hf_run.state.v0,
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
            relative_permittivity=float(args.epsilon_r),
            screening_lm=screening_lm,
            finite_zero_limit=finite_zero_limit,
            zero_cutoff=float(args.zero_cutoff),
        )
        path_elapsed = perf_counter() - eval_start
        write_hf_path_tsv(path_tsv, path_result)
        write_hf_path_nodes_tsv(nodes_tsv, path_result)
        write_hf_path_summary(path_summary, path_result, hf_state_path=str(state_path))
        scf_path_elapsed = 0.0
        if args.write_scf_path:
            scf_path_start = perf_counter()
            scf_path_result = build_restricted_hf_scf_path_plot_result(
                hf_run,
                grid_solution,
                path=path,
                init_mode=spec.init_mode,
            )
            write_hf_scf_path_tsv(scf_path_tsv, scf_path_result)
            write_hf_scf_band_plot(path_dir, scf_path_result, stem=f"{tag}_scf_grid_band_plot")
            scf_path_elapsed = perf_counter() - scf_path_start
            print(
                f"[stage] scf_path:done init={spec.init_mode} seed={spec.seed} "
                f"elapsed_sec={scf_path_elapsed:.3f} scf_path_tsv={scf_path_tsv}",
                flush=True,
            )
        _append_key_value_file(
            path_summary,
            [
                ("w0_meV", f"{float(args.w0):.16g}"),
                ("w1_meV", f"{float(args.w1):.16g}"),
                ("vf_meV", f"{float(args.vf):.16g}"),
                ("epsilon_r", f"{float(args.epsilon_r):.16g}"),
                ("tanh_argument_scale_a", f"{tanh_argument_scale_a:.16g}"),
                ("screening_lm", f"{screening_lm:.16g}"),
                ("interaction_model", "double_gate_tanh_q0limit" if finite_zero_limit else "double_gate_tanh_zero_q0"),
                ("q_zero_limit", str(finite_zero_limit).lower()),
                ("path_kind", str(args.path_kind)),
                ("path_labels", "-".join(path.labels)),
                ("write_scf_path", str(bool(args.write_scf_path)).lower()),
                ("scf_path_tsv", str(scf_path_tsv) if args.write_scf_path else ""),
                ("scf_path_elapsed_sec", f"{scf_path_elapsed:.16e}"),
                *initial_state_entries,
            ],
        )
        write_hf_band_plot(path_dir, path_result, stem=f"{tag}_band_plot")
        print(f"[stage] path:done init={spec.init_mode} seed={spec.seed} elapsed_sec={path_elapsed:.3f} path_tsv={path_tsv}", flush=True)

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
                "state_path": str(state_path),
                "path_tsv": str(path_tsv),
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
        ("tanh_argument_scale_a", f"{tanh_argument_scale_a:.16g}"),
        ("screening_lm", f"{screening_lm:.16g}"),
        ("physical_ds_angstrom", f"{tanh_argument_scale_a * GRAPHENE_LATTICE_A_ANGSTROM:.16g}"),
        ("interaction_model", "double_gate_tanh_q0limit" if finite_zero_limit else "double_gate_tanh_zero_q0"),
        ("q_zero_limit", str(finite_zero_limit).lower()),
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
