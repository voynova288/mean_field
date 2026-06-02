from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.supercell import (
    build_hartree_blocks_from_diagonals,
    build_remote_interaction_blocks,
    build_supercell_uniform_lattice,
    diagonal_h0_blocks,
    extract_supercell_gamma_m_k_gamma_kprime_scf_grid_path,
    max_hermitian_error,
    path_sector_energies,
    precompute_diagonal_overlaps,
    screening_lm_from_ds_angstrom,
    sector_index_from_label,
    sector_labels,
    solve_supercell_bm_model,
    supercell_coulomb_unit,
    supercell_interaction_shifts,
    zero_reference_density_blocks,
    zhang_sqrt3_tripled_supercell,
)
from mean_field.devtools.run_tbg_zhang_fig10 import _energy_offset_mev, _write_scf_grid_plot, _write_tsv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess a Zhang Fig. 10 run by plotting the exact SCF-grid points on "
            "Gamma_s-M_s-K_s-Gamma_s-K'_s. This avoids treating dense post-SCF path "
            "reconstruction as the default benchmark artifact."
        )
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--plot-sector", default=None, help="Override summary selected sector; default uses summary selected_sector.")
    parser.add_argument(
        "--energy-reference",
        choices=("none", "max_selected", "mean_selected"),
        default=None,
        help="Override summary energy_reference.",
    )
    parser.add_argument("--ylim-ev", default=None, help="Plot y limits as min,max, auto, or omit to reuse no fixed limit.")
    return parser


def _parse_ylim(raw: str | None) -> tuple[float, float] | None:
    if raw is None or str(raw).strip().lower() in {"", "none", "auto"}:
        return None
    parts = [float(part.strip()) for part in str(raw).split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise ValueError(f"Expected --ylim-ev ymin,ymax, got {raw!r}")
    return float(parts[0]), float(parts[1])


def _summary_path(result_dir: Path) -> Path:
    path = result_dir / "zhang_fig10_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return path


def _data_path(result_dir: Path, summary: dict[str, object]) -> Path:
    raw = summary.get("data_npz")
    if isinstance(raw, str) and Path(raw).exists():
        return Path(raw)
    path = result_dir / "zhang_fig10_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    start = perf_counter()
    result_dir = args.result_dir.resolve()
    summary = json.loads(_summary_path(result_dir).read_text(encoding="utf-8"))
    data_npz = _data_path(result_dir, summary)
    data = np.load(data_npz)

    params = TBGParameters.from_degrees(
        float(summary["theta_deg"]),
        vf=float(summary["vf_mev"]),
        w0=float(summary["w0_mev"]),
        w1=float(summary["w1_mev"]),
    )
    supercell = zhang_sqrt3_tripled_supercell()
    kmesh = int(summary["kmesh"])
    lg = int(summary["lg"])
    interaction_lg = int(summary.get("interaction_lg", lg))
    grid = build_supercell_uniform_lattice(params, supercell, kmesh, endpoint=False)

    print(f"[scf-grid] result_dir={result_dir}", flush=True)
    print(f"[scf-grid] lg={lg} kmesh={kmesh} interaction_lg={interaction_lg}", flush=True)

    t0 = perf_counter()
    grid_solution = solve_supercell_bm_model(params, grid.kvec, supercell=supercell, lg=lg, calculate_chern_operator=False)
    print(f"[scf-grid] solved grid BM in {perf_counter() - t0:.2f} s", flush=True)
    shifts, gvecs = supercell_interaction_shifts(grid_solution, interaction_lg)
    t0 = perf_counter()
    grid_diags = precompute_diagonal_overlaps(grid_solution, shifts)
    print(f"[scf-grid] precomputed grid diagonal overlaps in {perf_counter() - t0:.2f} s", flush=True)

    screening_lm = screening_lm_from_ds_angstrom(float(summary["ds_angstrom"]))
    v0 = supercell_coulomb_unit(params, supercell)
    if str(summary.get("coulomb_area", "supercell")) == "primitive":
        v0 *= float(supercell.area_ratio)

    remote_ref = zero_reference_density_blocks(grid_solution)
    t0 = perf_counter()
    remote_grid = build_remote_interaction_blocks(
        grid_solution,
        grid_solution,
        remote_ref,
        source_diagonals=grid_diags,
        target_diagonals=grid_diags,
        shifts=shifts,
        gvecs=gvecs,
        v0=v0,
        screening_lm=screening_lm,
        relative_permittivity=float(summary["epsilon_r"]),
        beta=1.0,
        finite_zero_limit=True,
        include_hartree=bool(summary.get("remote_include_hartree", True)),
        include_fock=bool(summary.get("remote_include_fock", True)),
        progress_prefix="[scf-grid remote-grid]",
    )
    print(f"[scf-grid] built grid remote potential in {perf_counter() - t0:.2f} s", flush=True)

    if "grid_density_for_hartree" in data.files:
        density_for_hartree = np.asarray(data["grid_density_for_hartree"], dtype=np.complex128)
    elif "grid_density_centered" in data.files:
        density_for_hartree = np.asarray(data["grid_density_centered"], dtype=np.complex128).copy()
        if str(summary.get("hartree_density_reference", "occupation")) == "occupation":
            eye = np.eye(grid_solution.nb, dtype=np.complex128)
            density_for_hartree += 0.5 * eye[None, None, :, :, None]
    else:
        raise KeyError(f"{data_npz} contains neither grid_density_for_hartree nor grid_density_centered")

    hartree_grid = build_hartree_blocks_from_diagonals(
        density_for_hartree,
        source_diagonals=grid_diags,
        target_diagonals=grid_diags,
        shifts=shifts,
        gvecs=gvecs,
        target_nk=grid_solution.nk,
        v0=v0,
        screening_lm=screening_lm,
        relative_permittivity=float(summary["epsilon_r"]),
        beta=1.0,
        finite_zero_limit=True,
    )

    h_grid = diagonal_h0_blocks(grid_solution) + remote_grid + hartree_grid
    all_grid_energies_mev = path_sector_energies(h_grid)
    labels = sector_labels(grid_solution.n_spin, grid_solution.n_eta)
    selected_sector = str(args.plot_sector or summary.get("selected_sector", "partial"))
    ispin, ieta = sector_index_from_label(selected_sector, n_spin=grid_solution.n_spin, n_eta=grid_solution.n_eta)
    selected_label = labels[ispin * grid_solution.n_eta + ieta]

    scf_path = extract_supercell_gamma_m_k_gamma_kprime_scf_grid_path(
        grid,
        super_g1=grid_solution.super_g1,
        super_g2=grid_solution.super_g2,
    )
    selected_mev = all_grid_energies_mev[ispin, ieta][:, scf_path.grid_indices]
    energy_reference = str(args.energy_reference or summary.get("energy_reference", "max_selected"))
    offset_mev = _energy_offset_mev(selected_mev, energy_reference)
    selected_ev = (selected_mev - offset_mev) / 1000.0

    plot_paths = _write_scf_grid_plot(
        result_dir,
        kdist=np.asarray(scf_path.kdist, dtype=float),
        selected_energies_ev=selected_ev,
        segment_indices=np.asarray(scf_path.segment_indices, dtype=int),
        path_labels=scf_path.labels,
        node_kdist=np.asarray(scf_path.node_kdist, dtype=float),
        title=r"$\nu=8/3$, SCF-grid Hartree + remote, no cRPA",
        ylim_ev=_parse_ylim(args.ylim_ev),
    )

    tsv_path = result_dir / "zhang_fig10_selected_sector_scf_grid_bands.tsv"
    rows = np.column_stack(
        [
            scf_path.kdist,
            scf_path.grid_indices,
            scf_path.frac_coords,
            scf_path.segment_indices,
            selected_ev.T,
        ]
    )
    _write_tsv(
        tsv_path,
        [
            "kdist",
            "grid_index",
            "frac_g1",
            "frac_g2",
            "segment_index",
            *[f"band_{i + 1}_ev_shifted" for i in range(selected_ev.shape[0])],
        ],
        rows,
    )

    scf_npz = result_dir / "zhang_fig10_scf_grid_data.npz"
    np.savez_compressed(
        scf_npz,
        scf_grid_kdist=np.asarray(scf_path.kdist, dtype=float),
        scf_grid_indices=np.asarray(scf_path.grid_indices, dtype=int),
        scf_grid_frac_coords=np.asarray(scf_path.frac_coords, dtype=float),
        scf_grid_segment_indices=np.asarray(scf_path.segment_indices, dtype=int),
        scf_grid_node_kdist=np.asarray(scf_path.node_kdist, dtype=float),
        scf_grid_labels=np.asarray(scf_path.labels),
        selected_sector=np.asarray([selected_label]),
        selected_scf_grid_energies_mev_raw=np.asarray(selected_mev, dtype=float),
        selected_scf_grid_energies_ev_shifted=np.asarray(selected_ev, dtype=float),
        all_grid_energies_mev_raw=np.asarray(all_grid_energies_mev, dtype=float),
        scf_grid_energy_offset_mev=np.asarray([offset_mev], dtype=float),
    )

    scf_summary = {
        "task": "Zhang Fig. 10 SCF-grid point band postprocess",
        "source_result_dir": str(result_dir),
        "source_data_npz": str(data_npz),
        "policy_source": "docs/benchmark_strategy.md: exact SCF-grid quantities are default; reconstructed paths are auxiliary",
        "plot_paths": plot_paths,
        "selected_tsv": str(tsv_path),
        "data_npz": str(scf_npz),
        "selected_sector": selected_label,
        "energy_reference": energy_reference,
        "plot_energy_offset_mev": float(offset_mev),
        "scf_grid_path": {
            "n_rows_with_duplicate_segment_endpoints": int(scf_path.kdist.size),
            "unique_grid_count": int(scf_path.unique_grid_count),
            "segment_counts": list(scf_path.segment_counts),
            "exact_node_hit_count": int(scf_path.exact_node_hit_count),
            "exact_node_hit_mask": [bool(value) for value in scf_path.exact_node_hit_mask.tolist()],
            "exact_tolerance": float(scf_path.exact_tolerance),
        },
        "diagnostics": {
            "remote_grid_norm_mev": float(np.linalg.norm(remote_grid)),
            "hartree_grid_norm_mev": float(np.linalg.norm(hartree_grid)),
            "h_grid_hermitian_error_mev": max_hermitian_error(h_grid),
            "selected_scf_grid_energy_min_raw_mev": float(np.min(selected_mev)),
            "selected_scf_grid_energy_max_raw_mev": float(np.max(selected_mev)),
            "selected_scf_grid_energy_min_shifted_ev": float(np.min(selected_ev)),
            "selected_scf_grid_energy_max_shifted_ev": float(np.max(selected_ev)),
        },
        "elapsed_sec": float(perf_counter() - start),
    }
    scf_summary_path = result_dir / "zhang_fig10_scf_grid_summary.json"
    scf_summary_path.write_text(json.dumps(scf_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[scf-grid] wrote {plot_paths['png']}", flush=True)
    print(f"[scf-grid] wrote {scf_summary_path}", flush=True)
    print(f"[scf-grid] elapsed_sec={scf_summary['elapsed_sec']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
