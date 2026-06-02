from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.supercell import (
    build_hartree_blocks_from_diagonals,
    build_remote_interaction_blocks,
    build_supercell_gamma_m_k_gamma_kprime_path,
    build_supercell_uniform_lattice,
    complex_to_pair,
    diagonal_h0_blocks,
    extract_supercell_gamma_m_k_gamma_kprime_scf_grid_path,
    filling_from_occupation_counts,
    max_hermitian_error,
    occupation_counts_svp_8over3,
    path_sector_energies,
    precompute_diagonal_overlaps,
    random_density_blocks,
    reciprocal_shift_labels,
    screening_lm_from_ds_angstrom,
    sector_index_from_label,
    sector_labels,
    solve_supercell_bm_model,
    supercell_coulomb_unit,
    supercell_interaction_shifts,
    zero_reference_density_blocks,
    zhang_sqrt3_tripled_supercell,
    run_hartree_only_scf,
    run_hartree_fock_scf,
)


DEFAULT_OUTPUT_ROOT = Path("results") / "TBG_Zhang2022_fig10"


def _load_plot_backend():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    return plt


def _parse_ylim(raw: str | None) -> tuple[float, float] | None:
    if raw is None or str(raw).strip().lower() in {"", "none", "auto"}:
        return None
    parts = [float(part.strip()) for part in str(raw).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("ylim must be 'ymin,ymax' in eV")
    if parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError(f"Expected ymin < ymax, got {parts}")
    return float(parts[0]), float(parts[1])


def _display_label(label: str) -> str:
    return {
        "Gamma_s": r"$\Gamma_s$",
        "M_s": r"$M_s$",
        "K_s": r"$K_s$",
        "Kprime_s": r"$K'_s$",
    }.get(label, label)


def _write_tsv(path: Path, header: list[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(f"{float(value):.12g}" for value in row) + "\n")


def _write_plot(
    output_dir: Path,
    *,
    kdist: np.ndarray,
    selected_energies_ev: np.ndarray,
    path_labels: tuple[str, ...],
    node_indices: tuple[int, ...],
    title: str,
    ylim_ev: tuple[float, float] | None,
) -> dict[str, str]:
    plt = _load_plot_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "zhang_fig10_hartree_remote_bands_reconstructed_path.png"
    pdf_path = output_dir / "zhang_fig10_hartree_remote_bands_reconstructed_path.pdf"

    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    for ib in range(selected_energies_ev.shape[0]):
        ax.plot(kdist, selected_energies_ev[ib], color="#008000", lw=1.05)

    node_x = [float(kdist[index - 1]) for index in node_indices]
    for xpos in node_x:
        ax.axvline(xpos, color="#b0b0b0", lw=0.55, alpha=0.8)
    ax.set_xticks(node_x)
    ax.set_xticklabels([_display_label(label) for label in path_labels], fontsize=11)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if ylim_ev is not None:
        ax.set_ylim(*ylim_ev)
    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.35)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _energy_offset_mev(energies_mev: np.ndarray, reference: str) -> float:
    if reference == "none":
        return 0.0
    if reference == "max_selected":
        return float(np.max(energies_mev))
    if reference == "mean_selected":
        return float(np.mean(energies_mev))
    raise RuntimeError(f"Unhandled energy reference {reference}")


def _write_scf_grid_plot(
    output_dir: Path,
    *,
    kdist: np.ndarray,
    selected_energies_ev: np.ndarray,
    segment_indices: np.ndarray,
    path_labels: tuple[str, ...],
    node_kdist: np.ndarray,
    title: str,
    ylim_ev: tuple[float, float] | None,
) -> dict[str, str]:
    plt = _load_plot_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "zhang_fig10_hartree_remote_scf_grid_bands.png"
    pdf_path = output_dir / "zhang_fig10_hartree_remote_scf_grid_bands.pdf"

    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    segment_indices = np.asarray(segment_indices, dtype=int)
    for ib in range(selected_energies_ev.shape[0]):
        for iseg in range(max(len(path_labels) - 1, 0)):
            mask = segment_indices == iseg
            if not np.any(mask):
                continue
            ax.plot(
                kdist[mask],
                selected_energies_ev[ib, mask],
                color="#008000",
                lw=0.9,
                marker="o",
                ms=1.8,
                mec="#008000",
                mfc="#008000",
            )

    node_x = [float(value) for value in node_kdist]
    for xpos in node_x:
        ax.axvline(xpos, color="#b0b0b0", lw=0.55, alpha=0.8)
    ax.set_xticks(node_x)
    ax.set_xticklabels([_display_label(label) for label in path_labels], fontsize=11)
    ax.set_xlim(float(node_x[0]), float(node_x[-1]))
    if ylim_ev is not None:
        ax.set_ylim(*ylim_ev)
    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout(pad=0.35)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Zhang et al. 2022 Fig. 10: sqrt(3)xsqrt(3) tripled-cell "
            "TBG bands with bare double-gate Hartree and remote-band HF potentials only. "
            "No cRPA screened interaction is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--vf", type=float, default=2135.4, help="BM Fermi velocity in meV code units.")
    parser.add_argument("--w0", type=float, default=79.7, help="AA tunneling in meV.")
    parser.add_argument("--w1", type=float, default=97.4, help="AB tunneling in meV.")
    parser.add_argument("--epsilon-r", type=float, default=7.0, help="Fixed dielectric constant; Fig. 10 uses epsilon=7, not cRPA.")
    parser.add_argument("--ds-angstrom", type=float, default=400.0)
    parser.add_argument(
        "--coulomb-area",
        choices=("primitive", "supercell"),
        default="primitive",
        help=(
            "Coulomb prefactor area convention. The Zhang supercell HF formulas keep the primitive moire-area "
            "normalization for the flat-electron Hartree scale; use supercell only as a diagnostic."
        ),
    )
    parser.add_argument("--lg", type=int, default=11, help="Odd plane-wave grid side in the tripled reciprocal basis; paper uses 11.")
    parser.add_argument("--kmesh", type=int, default=12, help="Supercell BZ mesh side; paper uses 12x12.")
    parser.add_argument("--interaction-lg", type=int, default=None, help="Odd Q-shift grid side; defaults to --lg.")
    parser.add_argument("--path-points-per-segment", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--scf-interaction",
        choices=("hartree_fock", "hartree_only"),
        default="hartree_fock",
        help=(
            "Interaction used to generate the density. Fig. 10 should use the density-wave HF density "
            "(hartree_fock), then plot only Hartree + remote potentials."
        ),
    )
    parser.add_argument(
        "--init",
        choices=("bm", "random"),
        default="random",
        help="Initial density for the active HF density search; random is needed to allow primitive-translation breaking.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-scf", action="store_true", help="Use one Hartree build from the remote-corrected BM density without iterating.")
    parser.add_argument("--skip-remote-fock", action="store_true", help="Diagnostic only: omit the Fock part of Eq. 19 remote-band potential.")
    parser.add_argument("--skip-remote-hartree", action="store_true", help="Diagnostic only: omit the Hartree part of Eq. 19 remote-band potential.")
    parser.add_argument(
        "--hartree-density-reference",
        choices=("occupation", "centered"),
        default="occupation",
        help=(
            "Density entering the plotted active Hartree potential. Zhang Eq. (15) uses the occupation "
            "matrix P, while this code's SCF density is stored as centered Delta=P-1/2."
        ),
    )
    parser.add_argument(
        "--plot-sector",
        default="partial",
        help="Flavor sector to plot: partial (=Kprime_up), K_up, K_down, Kprime_up, or Kprime_down.",
    )
    parser.add_argument(
        "--energy-reference",
        choices=("none", "max_selected", "mean_selected"),
        default="max_selected",
        help="Scalar shift used only in the plotted/TSV energies. Raw meV data are saved in the NPZ.",
    )
    parser.add_argument("--ylim-ev", type=_parse_ylim, default=(-0.4, 0.1), help="Plot y-limits in eV as 'min,max', or 'auto'.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    start_time = perf_counter()

    if args.output_dir is None:
        run_tag = args.run_tag or datetime.now().strftime("fig10_%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / run_tag
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interaction_lg = int(args.lg if args.interaction_lg is None else args.interaction_lg)
    if interaction_lg <= 0 or interaction_lg % 2 == 0:
        raise SystemExit(f"--interaction-lg must be positive odd, got {interaction_lg}")

    print("[fig10] Zhang Fig. 10 reproduction: no cRPA; fixed epsilon interaction", flush=True)
    print(f"[fig10] output_dir={output_dir}", flush=True)

    params = TBGParameters.from_degrees(
        float(args.theta_deg),
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
    )
    supercell = zhang_sqrt3_tripled_supercell()
    grid = build_supercell_uniform_lattice(params, supercell, int(args.kmesh), endpoint=False)
    path = build_supercell_gamma_m_k_gamma_kprime_path(params, supercell, int(args.path_points_per_segment))

    print(
        f"[fig10] supercell={supercell.as_dict()} lg={args.lg} kmesh={args.kmesh} "
        f"path_points={path.kvec.size} interaction_lg={interaction_lg}",
        flush=True,
    )

    t0 = perf_counter()
    grid_solution = solve_supercell_bm_model(params, grid.kvec, supercell=supercell, lg=int(args.lg), calculate_chern_operator=False)
    print(f"[fig10] solved grid BM in {perf_counter() - t0:.2f} s", flush=True)
    t0 = perf_counter()
    path_solution = solve_supercell_bm_model(params, path.kvec, supercell=supercell, lg=int(args.lg), calculate_chern_operator=False)
    print(f"[fig10] solved path BM in {perf_counter() - t0:.2f} s", flush=True)

    shifts, gvecs = supercell_interaction_shifts(grid_solution, interaction_lg)
    print(f"[fig10] Q-shift count={len(shifts)} labels={reciprocal_shift_labels(interaction_lg)}", flush=True)

    t0 = perf_counter()
    grid_diags = precompute_diagonal_overlaps(grid_solution, shifts)
    print(f"[fig10] precomputed grid diagonal overlaps in {perf_counter() - t0:.2f} s", flush=True)
    t0 = perf_counter()
    path_diags = precompute_diagonal_overlaps(path_solution, shifts)
    print(f"[fig10] precomputed path diagonal overlaps in {perf_counter() - t0:.2f} s", flush=True)

    screening_lm = screening_lm_from_ds_angstrom(float(args.ds_angstrom))
    v0_super = supercell_coulomb_unit(params, supercell)
    if args.coulomb_area == "primitive":
        v0_super *= float(supercell.area_ratio)
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
        v0=v0_super,
        screening_lm=screening_lm,
        relative_permittivity=float(args.epsilon_r),
        beta=float(args.beta),
        finite_zero_limit=True,
        include_hartree=not bool(args.skip_remote_hartree),
        include_fock=not bool(args.skip_remote_fock),
        progress_prefix="[fig10 remote-grid]",
    )
    print(f"[fig10] built grid remote potential in {perf_counter() - t0:.2f} s", flush=True)

    h0_grid = diagonal_h0_blocks(grid_solution) + remote_grid
    occ = occupation_counts_svp_8over3(grid_solution.nb)
    primitive_nu = filling_from_occupation_counts(occ, nb=grid_solution.nb, area_ratio=supercell.area_ratio)
    print(f"[fig10] occupation_counts(spin,valley)={occ.tolist()} primitive_nu={primitive_nu:.12g}", flush=True)

    if bool(args.no_scf):
        from mean_field.systems.tbg.zero_field.supercell import density_from_fixed_sector_occupations

        density_grid, _ = density_from_fixed_sector_occupations(h0_grid, occ)
        hartree_grid = build_hartree_blocks_from_diagonals(
            density_grid,
            source_diagonals=grid_diags,
            target_diagonals=grid_diags,
            shifts=shifts,
            gvecs=gvecs,
            target_nk=grid_solution.nk,
            v0=v0_super,
            screening_lm=screening_lm,
            relative_permittivity=float(args.epsilon_r),
            beta=float(args.beta),
            finite_zero_limit=True,
        )
        scf_info: dict[str, float | int | bool | str] = {
            "mode": "no_scf",
            "iterations": 0,
            "converged": False,
            "final_raw_norm": float("nan"),
            "mixing": float(args.mixing),
            "precision": float(args.precision),
            "no_scf": True,
        }
    else:
        initial_density = None
        if args.init == "random":
            initial_density = random_density_blocks(
                n_spin=grid_solution.n_spin,
                n_eta=grid_solution.n_eta,
                nb=grid_solution.nb,
                nk=grid_solution.nk,
                occupation_counts=occ,
                seed=int(args.seed),
            )
        if args.scf_interaction == "hartree_only":
            density_grid, _active_interaction_grid, _grid_energies, scf_info = run_hartree_only_scf(
                h0_grid,
                occupation_counts=occ,
                source_diagonals=grid_diags,
                shifts=shifts,
                gvecs=gvecs,
                v0=v0_super,
                screening_lm=screening_lm,
                relative_permittivity=float(args.epsilon_r),
                beta=float(args.beta),
                finite_zero_limit=True,
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                precision=float(args.precision),
            )
            if initial_density is not None:
                scf_info["requested_init"] = "random_not_used_by_hartree_only"
        else:
            density_grid, _active_interaction_grid, _grid_energies, scf_info = run_hartree_fock_scf(
                grid_solution,
                h0_grid,
                occupation_counts=occ,
                source_diagonals=grid_diags,
                shifts=shifts,
                gvecs=gvecs,
                v0=v0_super,
                screening_lm=screening_lm,
                relative_permittivity=float(args.epsilon_r),
                beta=float(args.beta),
                finite_zero_limit=True,
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                precision=float(args.precision),
                initial_density_blocks=initial_density,
            )
            scf_info["requested_init"] = str(args.init)
            scf_info["seed"] = int(args.seed)
        hartree_grid = build_hartree_blocks_from_diagonals(
            density_grid,
            source_diagonals=grid_diags,
            target_diagonals=grid_diags,
            shifts=shifts,
            gvecs=gvecs,
            target_nk=grid_solution.nk,
            v0=v0_super,
            screening_lm=screening_lm,
            relative_permittivity=float(args.epsilon_r),
            beta=float(args.beta),
            finite_zero_limit=True,
        )

    density_for_hartree = np.asarray(density_grid, dtype=np.complex128).copy()
    if args.hartree_density_reference == "occupation":
        eye = np.eye(grid_solution.nb, dtype=np.complex128)
        density_for_hartree += 0.5 * eye[None, None, :, :, None]
    hartree_grid = build_hartree_blocks_from_diagonals(
        density_for_hartree,
        source_diagonals=grid_diags,
        target_diagonals=grid_diags,
        shifts=shifts,
        gvecs=gvecs,
        target_nk=grid_solution.nk,
        v0=v0_super,
        screening_lm=screening_lm,
        relative_permittivity=float(args.epsilon_r),
        beta=float(args.beta),
        finite_zero_limit=True,
    )

    labels = sector_labels(grid_solution.n_spin, grid_solution.n_eta)
    ispin, ieta = sector_index_from_label(args.plot_sector, n_spin=grid_solution.n_spin, n_eta=grid_solution.n_eta)
    selected_label = labels[ispin * grid_solution.n_eta + ieta]

    h_grid = h0_grid + hartree_grid
    all_grid_energies_mev = path_sector_energies(h_grid)
    scf_path = extract_supercell_gamma_m_k_gamma_kprime_scf_grid_path(
        grid,
        super_g1=grid_solution.super_g1,
        super_g2=grid_solution.super_g2,
    )
    selected_scf_grid_mev = all_grid_energies_mev[ispin, ieta][:, scf_path.grid_indices]
    scf_grid_offset_mev = _energy_offset_mev(selected_scf_grid_mev, str(args.energy_reference))
    selected_scf_grid_ev = (selected_scf_grid_mev - scf_grid_offset_mev) / 1000.0
    scf_grid_plot_paths = _write_scf_grid_plot(
        output_dir,
        kdist=np.asarray(scf_path.kdist, dtype=float),
        selected_energies_ev=selected_scf_grid_ev,
        segment_indices=np.asarray(scf_path.segment_indices, dtype=int),
        path_labels=scf_path.labels,
        node_kdist=np.asarray(scf_path.node_kdist, dtype=float),
        title=r"$\nu=8/3$, SCF-grid Hartree + remote, no cRPA",
        ylim_ev=args.ylim_ev,
    )
    scf_grid_tsv = output_dir / "zhang_fig10_selected_sector_scf_grid_bands.tsv"
    scf_grid_rows = np.column_stack(
        [
            scf_path.kdist,
            scf_path.grid_indices,
            scf_path.frac_coords,
            scf_path.segment_indices,
            selected_scf_grid_ev.T,
        ]
    )
    _write_tsv(
        scf_grid_tsv,
        [
            "kdist",
            "grid_index",
            "frac_g1",
            "frac_g2",
            "segment_index",
            *[f"band_{i + 1}_ev_shifted" for i in range(selected_scf_grid_ev.shape[0])],
        ],
        scf_grid_rows,
    )

    t0 = perf_counter()
    remote_path = build_remote_interaction_blocks(
        path_solution,
        grid_solution,
        remote_ref,
        source_diagonals=grid_diags,
        target_diagonals=path_diags,
        shifts=shifts,
        gvecs=gvecs,
        v0=v0_super,
        screening_lm=screening_lm,
        relative_permittivity=float(args.epsilon_r),
        beta=float(args.beta),
        finite_zero_limit=True,
        include_hartree=not bool(args.skip_remote_hartree),
        include_fock=not bool(args.skip_remote_fock),
        progress_prefix="[fig10 remote-path]",
    )
    print(f"[fig10] built path remote potential in {perf_counter() - t0:.2f} s", flush=True)

    hartree_path = build_hartree_blocks_from_diagonals(
        density_for_hartree,
        source_diagonals=grid_diags,
        target_diagonals=path_diags,
        shifts=shifts,
        gvecs=gvecs,
        target_nk=path_solution.nk,
        v0=v0_super,
        screening_lm=screening_lm,
        relative_permittivity=float(args.epsilon_r),
        beta=float(args.beta),
        finite_zero_limit=True,
    )
    h_path = diagonal_h0_blocks(path_solution) + remote_path + hartree_path
    all_sector_energies_mev = path_sector_energies(h_path)
    selected_mev = all_sector_energies_mev[ispin, ieta]

    offset_mev = _energy_offset_mev(selected_mev, str(args.energy_reference))
    selected_ev = (selected_mev - offset_mev) / 1000.0

    reconstructed_plot_paths = _write_plot(
        output_dir,
        kdist=np.asarray(path.kdist, dtype=float),
        selected_energies_ev=selected_ev,
        path_labels=path.labels,
        node_indices=path.node_indices,
        title=r"$\nu=8/3$, reconstructed path, no cRPA",
        ylim_ev=args.ylim_ev,
    )

    reconstructed_tsv = output_dir / "zhang_fig10_selected_sector_bands_reconstructed_path.tsv"
    rows = np.column_stack([path.kdist, selected_ev.T])
    _write_tsv(
        reconstructed_tsv,
        ["kdist", *[f"band_{i + 1}_ev_shifted" for i in range(selected_ev.shape[0])]],
        rows,
    )

    npz_path = output_dir / "zhang_fig10_data.npz"
    np.savez_compressed(
        npz_path,
        kdist=np.asarray(path.kdist, dtype=float),
        path_kvec=np.asarray(path.kvec, dtype=np.complex128),
        path_node_indices=np.asarray(path.node_indices, dtype=int),
        path_labels=np.asarray(path.labels),
        sector_labels=np.asarray(labels),
        selected_sector=np.asarray([selected_label]),
        selected_energies_mev_raw=np.asarray(selected_mev, dtype=float),
        selected_energies_ev_shifted=np.asarray(selected_ev, dtype=float),
        all_sector_energies_mev_raw=np.asarray(all_sector_energies_mev, dtype=float),
        energy_offset_mev=np.asarray([offset_mev], dtype=float),
        scf_grid_kdist=np.asarray(scf_path.kdist, dtype=float),
        scf_grid_indices=np.asarray(scf_path.grid_indices, dtype=int),
        scf_grid_frac_coords=np.asarray(scf_path.frac_coords, dtype=float),
        scf_grid_segment_indices=np.asarray(scf_path.segment_indices, dtype=int),
        scf_grid_node_kdist=np.asarray(scf_path.node_kdist, dtype=float),
        scf_grid_labels=np.asarray(scf_path.labels),
        selected_scf_grid_energies_mev_raw=np.asarray(selected_scf_grid_mev, dtype=float),
        selected_scf_grid_energies_ev_shifted=np.asarray(selected_scf_grid_ev, dtype=float),
        all_grid_energies_mev_raw=np.asarray(all_grid_energies_mev, dtype=float),
        scf_grid_energy_offset_mev=np.asarray([scf_grid_offset_mev], dtype=float),
        occupation_counts=np.asarray(occ, dtype=int),
        grid_density_centered=np.asarray(density_grid, dtype=np.complex128),
        grid_density_for_hartree=np.asarray(density_for_hartree, dtype=np.complex128),
        grid_hartree_norm_mev=np.asarray([float(np.linalg.norm(hartree_grid))]),
        path_hartree_norm_mev=np.asarray([float(np.linalg.norm(hartree_path))]),
        path_remote_norm_mev=np.asarray([float(np.linalg.norm(remote_path))]),
    )

    summary = {
        "task": "Zhang et al. 2022 Fig. 10 reproduction",
        "no_crpa": True,
        "interaction": "bare double-gate Coulomb with fixed epsilon; cRPA deliberately not used",
        "output_dir": str(output_dir),
        "plot_source": "scf_grid_exact_points",
        "plot_paths": scf_grid_plot_paths,
        "selected_tsv": str(scf_grid_tsv),
        "reconstructed_path_plot_paths": reconstructed_plot_paths,
        "reconstructed_path_selected_tsv": str(reconstructed_tsv),
        "data_npz": str(npz_path),
        "theta_deg": float(args.theta_deg),
        "vf_mev": float(args.vf),
        "w0_mev": float(args.w0),
        "w1_mev": float(args.w1),
        "epsilon_r": float(args.epsilon_r),
        "ds_angstrom": float(args.ds_angstrom),
        "coulomb_area": str(args.coulomb_area),
        "coulomb_v0_mev": float(v0_super),
        "screening_lm_code": float(screening_lm),
        "lg": int(args.lg),
        "kmesh": int(args.kmesh),
        "interaction_lg": int(interaction_lg),
        "supercell": supercell.as_dict(),
        "supercell_G1_internal": complex_to_pair(grid_solution.super_g1),
        "supercell_G2_internal": complex_to_pair(grid_solution.super_g2),
        "primitive_g1_internal": complex_to_pair(params.g1),
        "primitive_g2_internal": complex_to_pair(params.g2),
        "active_bands_per_spin_valley": int(grid_solution.nb),
        "primitive_nu_from_occupation_counts": float(primitive_nu),
        "occupation_counts_spin_valley": occ.tolist(),
        "occupation_count_labels": {
            "axis0_spin": ["up", "down"],
            "axis1_valley": ["K", "Kprime"],
        },
        "selected_sector": selected_label,
        "hartree_density_reference": str(args.hartree_density_reference),
        "energy_reference": str(args.energy_reference),
        "plot_energy_offset_mev": float(scf_grid_offset_mev),
        "reconstructed_path_energy_offset_mev": float(offset_mev),
        "scf_grid_path": {
            "policy_source": "docs/benchmark_strategy.md: use exact SCF-grid quantities by default; reconstructed paths are auxiliary",
            "n_rows_with_duplicate_segment_endpoints": int(scf_path.kdist.size),
            "unique_grid_count": int(scf_path.unique_grid_count),
            "segment_counts": list(scf_path.segment_counts),
            "exact_node_hit_count": int(scf_path.exact_node_hit_count),
            "exact_node_hit_mask": [bool(value) for value in scf_path.exact_node_hit_mask.tolist()],
            "exact_tolerance": float(scf_path.exact_tolerance),
        },
        "remote_include_hartree": not bool(args.skip_remote_hartree),
        "remote_include_fock": not bool(args.skip_remote_fock),
        "scf_interaction_for_density": str(args.scf_interaction),
        "init": str(args.init),
        "seed": int(args.seed),
        "scf": scf_info,
        "diagnostics": {
            "remote_grid_norm_mev": float(np.linalg.norm(remote_grid)),
            "remote_path_norm_mev": float(np.linalg.norm(remote_path)),
            "hartree_grid_norm_mev": float(np.linalg.norm(hartree_grid)),
            "hartree_path_norm_mev": float(np.linalg.norm(hartree_path)),
            "h_grid_hermitian_error_mev": max_hermitian_error(h_grid),
            "h_path_hermitian_error_mev": max_hermitian_error(h_path),
            "selected_scf_grid_energy_min_raw_mev": float(np.min(selected_scf_grid_mev)),
            "selected_scf_grid_energy_max_raw_mev": float(np.max(selected_scf_grid_mev)),
            "selected_scf_grid_energy_min_shifted_ev": float(np.min(selected_scf_grid_ev)),
            "selected_scf_grid_energy_max_shifted_ev": float(np.max(selected_scf_grid_ev)),
            "selected_energy_min_raw_mev": float(np.min(selected_mev)),
            "selected_energy_max_raw_mev": float(np.max(selected_mev)),
            "selected_energy_min_shifted_ev": float(np.min(selected_ev)),
            "selected_energy_max_shifted_ev": float(np.max(selected_ev)),
        },
        "elapsed_sec": float(perf_counter() - start_time),
    }
    summary_path = output_dir / "zhang_fig10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    note_path = output_dir / "README.md"
    note_path.write_text(
        "# Zhang 2022 Fig. 10 reproduction\n\n"
        "This run uses the Zhang supplementary-material supercell convention for the "
        "sqrt(3) x sqrt(3) tripled moire cell, fixed dielectric epsilon=7, and "
        "double-gate screening length ds=400 Angstrom.  It intentionally does **not** "
        "use any cRPA artifact or cRPA-screened plot as a reference.\n\n"
        "Following `docs/benchmark_strategy.md`, the default band artifact is the exact SCF-grid "
        "point line plot; the dense reconstructed path is written only as an auxiliary diagnostic.\n\n"
        f"Selected plotted sector: `{selected_label}`.  SCF-grid plot energy offset: `{scf_grid_offset_mev:.12g}` meV.\n\n"
        f"SCF info: `{scf_info}`.\n",
        encoding="utf-8",
    )

    print(f"[fig10] wrote {scf_grid_plot_paths['png']}", flush=True)
    print(f"[fig10] wrote auxiliary {reconstructed_plot_paths['png']}", flush=True)
    print(f"[fig10] wrote {summary_path}", flush=True)
    print(f"[fig10] elapsed_sec={summary['elapsed_sec']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
