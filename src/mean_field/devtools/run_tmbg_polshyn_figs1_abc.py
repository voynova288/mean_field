from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tmbg import TMBGModel, TMBGParameters, infer_flat_band_indices
from mean_field.systems.tmbg.bands import compute_bands_along_path
from mean_field.systems.tmbg.polshyn_supercell import (
    build_doubled_uniform_grid,
    build_full_p0_subtraction_h0_correction,
    build_interaction_blocks,
    build_polshyn_kx0_path,
    cdw_density_blocks,
    basis_with_h0_correction,
    build_polshyn_projected_basis,
    build_polshyn_s1a_path,
    build_wang_overlap_blocks,
    density_from_fixed_sector_occupations,
    estimate_fermi_level_from_sector_energies,
    max_hermitian_error,
    moire_cell_area_nm2,
    occupation_counts_nu_7over2,
    overlap_blocks_with_hartree_q0_zeroed,
    path_sector_energies,
    polshyn_doubled_cell,
    precompute_compact_overlaps,
    precompute_diagonal_overlaps,
    primitive_nu_from_counts,
    projected_p0_subtraction_density_blocks,
    random_density_blocks,
    reference_projector_blocks,
    run_projected_hf_scf,
    scaled_overlap_blocks,
    run_projected_hf_scf_wang,
    supercell_interaction_shifts,
    translation_order_parameters,
    wang_interaction_blocks_from_sector_density,
    wang_sector_density_blocks,
    wang_sector_energies_from_flat_hamiltonian,
    wang_sector_energy_blocks,
    wang_sector_hamiltonian_blocks,
    wang_target_hamiltonian,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "TMBG_Polshyn2021_figS1"


def _load_plot_backend():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    return plt


def _parse_band_index(raw: str | None) -> int | None:
    if raw is None or str(raw).strip().lower() in {"", "auto"}:
        return None
    return int(raw)


def _selected_indices_for_panel(target_band_index: int, *, lower_remote: int, upper_remote: int) -> tuple[int, ...]:
    start = int(target_band_index) - int(lower_remote)
    stop = int(target_band_index) + int(upper_remote)
    if start < 0:
        raise ValueError(f"Remote window starts at negative band index: {start}")
    return tuple(range(start, stop + 1))


def _display_s1a_label(label: str) -> str:
    return {
        "Gamma": r"$\Gamma$",
        "Kminus": r"$K_-^M$",
        "Kplus": r"$K_+^M$",
        "M": r"$M$",
    }.get(label, label)


def _s1a_band_metrics(path_result, *, target_band_index: int) -> dict[str, object]:
    energies = np.asarray(path_result.energies, dtype=float)
    target = energies[:, int(target_band_index)]
    below = energies[:, int(target_band_index) - 1] if int(target_band_index) > 0 else np.full_like(target, np.nan)
    above = energies[:, int(target_band_index) + 1] if int(target_band_index) + 1 < energies.shape[1] else np.full_like(target, np.nan)
    node_metrics: list[dict[str, float | int | str]] = []
    for inode, (label, idx1) in enumerate(zip(path_result.path.labels, path_result.path.node_indices, strict=True)):
        idx = int(idx1) - 1
        node_metrics.append(
            {
                "node": int(inode),
                "label": str(label),
                "target_mev": float(1000.0 * target[idx]),
                "below_mev": float(1000.0 * below[idx]),
                "above_mev": float(1000.0 * above[idx]),
                "gap_to_below_mev": float(1000.0 * (target[idx] - below[idx])),
                "gap_to_above_mev": float(1000.0 * (above[idx] - target[idx])),
            }
        )
    return {
        "target_bandwidth_mev": float(1000.0 * (np.max(target) - np.min(target))),
        "target_min_mev": float(1000.0 * np.min(target)),
        "target_max_mev": float(1000.0 * np.max(target)),
        "min_gap_to_below_mev": float(1000.0 * np.nanmin(target - below)),
        "min_gap_to_above_mev": float(1000.0 * np.nanmin(above - target)),
        "node_metrics": node_metrics,
    }


def _write_s1a_plot(output_dir: Path, path_result, *, target_band_index: int, selected_indices: tuple[int, ...]) -> dict[str, str]:
    plt = _load_plot_backend()
    png = output_dir / "polshyn_figS1a_noninteracting_bands.png"
    pdf = output_dir / "polshyn_figS1a_noninteracting_bands.pdf"
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    energies_mev = 1000.0 * np.asarray(path_result.energies[:, selected_indices], dtype=float)
    for ilocal, band_index in enumerate(selected_indices):
        color = "#d62728" if int(band_index) == int(target_band_index) else "#1f77b4"
        if int(band_index) == int(target_band_index) - 1:
            color = "#2ca02c"
        if int(band_index) == int(target_band_index) - 2:
            color = "#ff7f0e"
        ax.plot(path_result.path.kdist, energies_mev[:, ilocal], lw=1.2, color=color)
    node_x = [float(path_result.path.kdist[idx - 1]) for idx in path_result.path.node_indices]
    for xpos in node_x:
        ax.axvline(xpos, color="#c0c0c0", lw=0.5)
    ax.set_xticks(node_x)
    ax.set_xticklabels([_display_s1a_label(label) for label in path_result.path.labels], fontsize=9)
    ax.set_xlim(node_x[0], node_x[-1])
    ax.set_ylim(-80.0, 80.0)
    ax.set_ylabel("E (meV)")
    ax.set_title("Polshyn Fig. S1a anchor", fontsize=9)
    ax.text(0.53, 0.70, r"target $C=2$ band", color="#d62728", transform=ax.transAxes, fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _sector_color(ispin: int, ieta: int) -> str:
    if ispin == 0 and ieta == 0:
        return "#ffcc22"  # K+ up, yellow
    if ispin == 1 and ieta == 0:
        return "#4b1fb3"  # K+ down, purple
    return "#cc4778"  # K- sectors, pink


def _extract_scf_grid_kx0_line(
    frac_grid: np.ndarray | None,
    grid_energies: np.ndarray,
    *,
    f1_tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact SCF-grid energies on the kx=0 row, ordered by centered ky.

    This is a diagnostic guard against over-interpreting dense post-SCF line
    reconstructions: the HF fixed point is only self-consistent on the SCF mesh.
    """

    if frac_grid is None:
        raise ValueError("SCF-grid line extraction requires k_grid_frac metadata")
    frac = np.asarray(frac_grid, dtype=float).reshape(-1, 2)
    energies = np.asarray(grid_energies, dtype=float)
    mask = np.isclose(frac[:, 0], 0.0, atol=float(f1_tol))
    if not np.any(mask):
        raise ValueError("No f1=0 SCF-grid row found for kx=0 extraction")
    indices = np.nonzero(mask)[0]
    f2 = frac[indices, 1]
    f2_centered = ((f2 + 0.5) % 1.0) - 0.5
    x_ky = -2.0 * np.pi * f2_centered
    order = np.argsort(x_ky)
    selected = indices[order]
    return np.asarray(x_ky[order], dtype=float), energies[..., selected], np.asarray(selected, dtype=int)


def _write_hf_line_plot(
    output_dir: Path,
    *,
    stem: str,
    x_ky: np.ndarray,
    energies_ev: np.ndarray,
    occupation_counts: np.ndarray,
    title: str,
    ylim_mev: tuple[float, float] = (-20.0, 20.0),
) -> dict[str, str]:
    plt = _load_plot_backend()
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    energies_mev = 1000.0 * np.asarray(energies_ev, dtype=float)
    for ispin in range(energies_mev.shape[0]):
        for ieta in range(energies_mev.shape[1]):
            color = _sector_color(ispin, ieta)
            alpha = 1.0 if (ispin, ieta) in {(0, 0), (1, 0)} else 0.85
            for ib in range(energies_mev.shape[2]):
                ax.plot(x_ky, energies_mev[ispin, ieta, ib], color=color, lw=0.8, marker="o", ms=1.7, alpha=alpha)
    ax.axhline(0.0, color="#1f77b4", lw=0.7, ls="--", alpha=0.8)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(*ylim_mev)
    ax.set_xticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$k_y a_M$")
    ax.set_ylabel("E (meV)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.35)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_order_plot(
    output_dir: Path,
    *,
    stem: str,
    frac_grid: np.ndarray | None,
    order_x2: np.ndarray,
    title: str,
) -> dict[str, str]:
    plt = _load_plot_backend()
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    values = np.asarray(order_x2, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(2.45, 2.15))
    if frac_grid is not None:
        frac = np.asarray(frac_grid, dtype=float)
        f1 = frac[..., 0].reshape(-1)
        f2 = frac[..., 1].reshape(-1)
        mesh = int(round(np.sqrt(values.size)))
        if mesh * mesh == values.size and frac.shape[:2] == (mesh, mesh):
            img = values.reshape(mesh, mesh)
            handle = ax.imshow(img.T, origin="lower", extent=(0, 1, 0, 1), vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
        else:
            handle = ax.scatter(f1, f2, c=values, vmin=0.0, vmax=1.0, cmap="magma", s=18)
    else:
        handle = ax.plot(np.arange(values.size), values, marker="o", ms=2.0, lw=0.8)[0]
    if frac_grid is not None:
        ax.set_xlabel(r"$k_1/B_1$")
        ax.set_ylabel(r"$k_2/B_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04, label=r"$2O(k)$")
    else:
        ax.set_xlabel("grid index")
        ax.set_ylabel(r"$2O(k)$")
        ax.set_ylim(0, max(1.0, float(np.max(values)) * 1.05))
    ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_tsv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(f"{float(value):.12g}" for value in row) + "\n")


def _run_hf_panel(
    *,
    output_dir: Path,
    model: TMBGModel,
    target_band_index: int,
    lower_remote: int,
    upper_remote: int,
    kmesh: int,
    line_points: int,
    g_shells: int,
    epsilon_r: float,
    gate_distance_nm: float,
    coulomb_area: str,
    max_iter: int,
    mixing: float,
    precision: float,
    seed: int,
    init: str,
    label: str,
    hf_engine: str,
    oda_stall_threshold: float,
    h0_subtraction: str,
    p0_reference: str,
    skip_reconstructed_line: bool,
    hartree_scale: float,
    fock_scale: float,
    zero_hartree_q0: bool,
) -> dict[str, object]:
    supercell = polshyn_doubled_cell()
    projected_indices = _selected_indices_for_panel(target_band_index, lower_remote=lower_remote, upper_remote=upper_remote)
    frac_grid, grid_kvec = build_doubled_uniform_grid(model.lattice, kmesh, supercell=supercell, endpoint=False)
    grid_basis = build_polshyn_projected_basis(
        model,
        grid_kvec,
        projected_indices=projected_indices,
        target_band_index=target_band_index,
        supercell=supercell,
        k_grid_frac=frac_grid,
    )
    shifts, gvecs = supercell_interaction_shifts(grid_basis, g_shells)
    occ = occupation_counts_nu_7over2(projected_indices, target_band_index)
    primitive_nu = primitive_nu_from_counts(occ, grid_basis.reference_diagonal, area_ratio=supercell.area_ratio)
    area_ratio_for_v0 = 1 if coulomb_area == "primitive" else supercell.area_ratio
    v0 = 1.0 / moire_cell_area_nm2(model.lattice, area_ratio=area_ratio_for_v0)

    h0_subtraction_info: dict[str, float | str] = {"mode": str(h0_subtraction), "p0_reference": str(p0_reference)}
    precomputed_wang_overlap_blocks = None
    if h0_subtraction in {"full-p0", "minus-full-p0", "active-minus-full-p0"}:
        h0_correction, h0_subtraction_diagnostics = build_full_p0_subtraction_h0_correction(
            grid_basis,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            zero_hartree_q0=True,
            include_active_reference=True,
            p0_reference=str(p0_reference),
            hartree_scale=float(hartree_scale),
            fock_scale=float(fock_scale),
            progress_prefix=f"[polshyn {label} h0-subtraction]",
        )
        if h0_subtraction == "minus-full-p0":
            h0_correction = -h0_correction
            h0_subtraction_diagnostics["sign"] = -1.0
        elif h0_subtraction == "active-minus-full-p0":
            reference_density = reference_projector_blocks(grid_basis)
            trace_ref = np.trace(reference_density, axis1=2, axis2=3).real
            h0_subtraction_diagnostics["reference_density_trace_mean"] = float(np.mean(trace_ref))
            h0_subtraction_diagnostics["reference_density_trace_min"] = float(np.min(trace_ref))
            h0_subtraction_diagnostics["reference_density_trace_max"] = float(np.max(trace_ref))
            if float(np.linalg.norm(reference_density)) > 1.0e-14:
                precomputed_wang_overlap_blocks = build_wang_overlap_blocks(
                    grid_basis,
                    grid_basis,
                    shifts,
                    gvecs,
                    epsilon_r=epsilon_r,
                    d_sc_nm=gate_distance_nm,
                    include_hartree=True,
                    include_fock=True,
                    progress_prefix=f"[polshyn {label} h0-active-reference]",
                )
                h0_reference_blocks = precomputed_wang_overlap_blocks
                if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
                    h0_reference_blocks = scaled_overlap_blocks(
                        h0_reference_blocks,
                        hartree_scale=float(hartree_scale),
                        fock_scale=float(fock_scale),
                    )
                ref_correction = wang_interaction_blocks_from_sector_density(
                    reference_density,
                    grid_basis,
                    overlap_blocks_with_hartree_q0_zeroed(h0_reference_blocks),
                    v0=v0,
                )
            else:
                ref_correction = np.zeros_like(h0_correction)
            h0_correction = 2.0 * ref_correction - h0_correction
            h0_subtraction_diagnostics["sign"] = -1.0
            h0_subtraction_diagnostics["active_reference_sign"] = 1.0
            h0_subtraction_diagnostics["active_reference_correction_norm_ev"] = float(np.linalg.norm(ref_correction))
        else:
            h0_subtraction_diagnostics["sign"] = 1.0
        h0_subtraction_diagnostics["final_h0_correction_norm_ev"] = float(np.linalg.norm(h0_correction))
        h0_subtraction_diagnostics["final_h0_correction_max_abs_mev"] = float(1000.0 * np.max(np.abs(h0_correction)))
        grid_basis = basis_with_h0_correction(grid_basis, h0_correction)
        h0_subtraction_info.update({key: float(value) for key, value in h0_subtraction_diagnostics.items()})
    elif h0_subtraction == "active-reference":
        reference_density = reference_projector_blocks(grid_basis)
        trace_ref = np.trace(reference_density, axis1=2, axis2=3).real
        h0_subtraction_info["reference_density_trace_mean"] = float(np.mean(trace_ref))
        h0_subtraction_info["reference_density_trace_min"] = float(np.min(trace_ref))
        h0_subtraction_info["reference_density_trace_max"] = float(np.max(trace_ref))
        h0_subtraction_info["hartree_q0_zeroed"] = 1.0
        if float(np.linalg.norm(reference_density)) > 1.0e-14:
            precomputed_wang_overlap_blocks = build_wang_overlap_blocks(
                grid_basis,
                grid_basis,
                shifts,
                gvecs,
                epsilon_r=epsilon_r,
                d_sc_nm=gate_distance_nm,
                include_hartree=True,
                include_fock=True,
                progress_prefix=f"[polshyn {label} h0-reference]",
            )
            h0_reference_blocks = precomputed_wang_overlap_blocks
            if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
                h0_reference_blocks = scaled_overlap_blocks(
                    h0_reference_blocks,
                    hartree_scale=float(hartree_scale),
                    fock_scale=float(fock_scale),
                )
            h0_correction_overlap_blocks = overlap_blocks_with_hartree_q0_zeroed(h0_reference_blocks)
            h0_correction = wang_interaction_blocks_from_sector_density(
                reference_density,
                grid_basis,
                h0_correction_overlap_blocks,
                v0=v0,
            )
            grid_basis = basis_with_h0_correction(grid_basis, h0_correction)
            h0_subtraction_info["h0_correction_norm_ev"] = float(np.linalg.norm(h0_correction))
            h0_subtraction_info["h0_correction_max_abs_mev"] = float(1000.0 * np.max(np.abs(h0_correction)))
        else:
            h0_subtraction_info["h0_correction_norm_ev"] = 0.0
            h0_subtraction_info["h0_correction_max_abs_mev"] = 0.0
    elif h0_subtraction == "projected-p0":
        precomputed_wang_overlap_blocks = build_wang_overlap_blocks(
            grid_basis,
            grid_basis,
            shifts,
            gvecs,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            include_hartree=True,
            include_fock=True,
            progress_prefix=f"[polshyn {label} h0-subtraction]",
        )
        subtraction_density, h0_subtraction_diagnostics = projected_p0_subtraction_density_blocks(
            grid_basis,
            include_active_reference=True,
            p0_reference=str(p0_reference),
        )
        h0_subtraction_blocks = precomputed_wang_overlap_blocks
        if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
            h0_subtraction_blocks = scaled_overlap_blocks(
                h0_subtraction_blocks,
                hartree_scale=float(hartree_scale),
                fock_scale=float(fock_scale),
            )
        h0_correction_overlap_blocks = overlap_blocks_with_hartree_q0_zeroed(h0_subtraction_blocks)
        h0_correction = wang_interaction_blocks_from_sector_density(
            subtraction_density,
            grid_basis,
            h0_correction_overlap_blocks,
            v0=v0,
        )
        grid_basis = basis_with_h0_correction(grid_basis, h0_correction)
        h0_subtraction_info.update({key: float(value) for key, value in h0_subtraction_diagnostics.items()})
        h0_subtraction_info["hartree_q0_zeroed"] = 1.0
        h0_subtraction_info["h0_correction_norm_ev"] = float(np.linalg.norm(h0_correction))
        h0_subtraction_info["h0_correction_max_abs_mev"] = float(1000.0 * np.max(np.abs(h0_correction)))
    elif h0_subtraction != "none":
        raise ValueError(f"Unsupported h0_subtraction={h0_subtraction!r}")

    initial_density = None
    if init == "random":
        initial_density = random_density_blocks(
            n_spin=grid_basis.n_spin,
            n_eta=grid_basis.n_eta,
            nb=grid_basis.nb,
            nk=grid_basis.nk,
            occupation_counts=occ,
            reference_diagonal=grid_basis.reference_diagonal,
            seed=seed,
        )
    elif init == "cdw":
        initial_density = cdw_density_blocks(
            projected_indices=projected_indices,
            target_band_index=target_band_index,
            n_spin=grid_basis.n_spin,
            n_eta=grid_basis.n_eta,
            nb=grid_basis.nb,
            nk=grid_basis.nk,
            reference_diagonal=grid_basis.reference_diagonal,
        )

    wang_state = None
    wang_grid_overlap_blocks = None
    if hf_engine == "wang":
        wang_state, wang_grid_overlap_blocks, scf_info = run_projected_hf_scf_wang(
            grid_basis,
            occupation_counts=occ,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            max_iter=max_iter,
            precision=precision,
            initial_density_blocks=initial_density,
            oda_stall_threshold=oda_stall_threshold,
            progress_prefix=f"[polshyn {label} grid]",
            overlap_blocks=precomputed_wang_overlap_blocks,
            seed=seed,
            hartree_scale=hartree_scale,
            fock_scale=fock_scale,
            zero_hartree_q0=zero_hartree_q0,
        )
        density = wang_sector_density_blocks(wang_state, grid_basis)
        grid_total = wang_sector_hamiltonian_blocks(wang_state, grid_basis)
        grid_total_energies = wang_sector_energy_blocks(wang_state, grid_basis)
    else:
        grid_diags = precompute_diagonal_overlaps(grid_basis, shifts)
        grid_compact_overlaps = precompute_compact_overlaps(
            grid_basis,
            grid_basis,
            shifts,
            progress_prefix=f"[polshyn {label} grid]",
        )
        density, interaction_grid, _grid_energies, scf_info = run_projected_hf_scf(
            grid_basis,
            occupation_counts=occ,
            source_diagonals=grid_diags,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            max_iter=max_iter,
            mixing=mixing,
            precision=precision,
            initial_density_blocks=initial_density,
            compact_overlaps=grid_compact_overlaps,
        )
        grid_total = grid_basis.h0_blocks + interaction_grid
        grid_total_energies = path_sector_energies(grid_total)
    scf_info["hf_engine"] = str(hf_engine)
    scf_info["h0_subtraction"] = str(h0_subtraction)
    scf_info["hartree_scale"] = float(hartree_scale)
    scf_info["fock_scale"] = float(fock_scale)
    scf_info["zero_hartree_q0"] = bool(zero_hartree_q0)
    scf_info["requested_init"] = init
    scf_info["seed"] = int(seed)
    order = translation_order_parameters(
        density,
        projected_indices=projected_indices,
        target_band_index=target_band_index,
        spin_index=0,
        valley_index=0,
    )
    order_npz = output_dir / f"polshyn_figS1{label}_translation_order.npz"
    np.savez_compressed(
        order_npz,
        frac_grid=np.asarray(grid_basis.k_grid_frac, dtype=float) if grid_basis.k_grid_frac is not None else np.asarray([], dtype=float),
        target_raw=np.asarray(order["target_raw"], dtype=float),
        all_raw=np.asarray(order["all_raw"], dtype=float),
        target_x2=np.asarray(order["target_x2"], dtype=float),
        all_x2=np.asarray(order["all_x2"], dtype=float),
        projected_indices=np.asarray(projected_indices, dtype=int),
        target_band_index=np.asarray([target_band_index], dtype=int),
    )
    order_plot = _write_order_plot(
        output_dir,
        stem=f"polshyn_figS1{label}_translation_order",
        frac_grid=grid_basis.k_grid_frac,
        order_x2=np.asarray(order["target_x2"], dtype=float),
        title=f"S1{label} K+ up target CDW order",
    )
    mu = estimate_fermi_level_from_sector_energies(grid_total_energies, occ)
    grid_h0_energies = path_sector_energies(grid_basis.h0_blocks)
    grid_interaction_blocks = grid_total - grid_basis.h0_blocks
    grid_interaction_energies = path_sector_energies(grid_interaction_blocks)
    scf_x_ky, scf_grid_shifted, scf_grid_indices = _extract_scf_grid_kx0_line(
        grid_basis.k_grid_frac,
        grid_total_energies - float(mu),
    )
    scf_grid_plot = _write_hf_line_plot(
        output_dir,
        stem=f"polshyn_figS1{label}_hf_kx0_scf_grid",
        x_ky=scf_x_ky,
        energies_ev=scf_grid_shifted,
        occupation_counts=occ,
        title=f"Polshyn Fig. S1{label}, SCF-grid kx=0",
    )
    scf_grid_npz = output_dir / f"polshyn_figS1{label}_hf_kx0_scf_grid_data.npz"
    np.savez_compressed(
        scf_grid_npz,
        x_ky=np.asarray(scf_x_ky, dtype=float),
        grid_indices=np.asarray(scf_grid_indices, dtype=int),
        grid_energies_ev_shifted=np.asarray(scf_grid_shifted, dtype=float),
        grid_energies_ev_raw=np.asarray(grid_total_energies[..., scf_grid_indices], dtype=float),
        grid_h0_energies_ev=np.asarray(grid_h0_energies[..., scf_grid_indices], dtype=float),
        grid_interaction_eigenvalues_ev=np.asarray(grid_interaction_energies[..., scf_grid_indices], dtype=float),
        grid_interaction_blocks_ev=np.asarray(grid_interaction_blocks[..., scf_grid_indices], dtype=np.complex128),
        occupation_counts=np.asarray(occ, dtype=int),
        projected_indices=np.asarray(projected_indices, dtype=int),
        target_band_index=np.asarray([target_band_index], dtype=int),
        reference_diagonal=np.asarray(grid_basis.reference_diagonal, dtype=float),
        fermi_level_ev=np.asarray([mu], dtype=float),
    )

    if skip_reconstructed_line:
        return {
            "label": f"S1{label}",
            "projected_indices": list(projected_indices),
            "target_band_index": int(target_band_index),
            "lower_remote": int(lower_remote),
            "upper_remote": int(upper_remote),
            "occupation_counts": occ.tolist(),
            "primitive_nu_from_counts": float(primitive_nu),
            "reference_diagonal": grid_basis.reference_diagonal.tolist(),
            "coulomb_area": str(coulomb_area),
            "hartree_scale": float(hartree_scale),
            "fock_scale": float(fock_scale),
            "zero_hartree_q0": bool(zero_hartree_q0),
            "v0_nm_inv_sq": float(v0),
            "fermi_level_ev": float(mu),
            "plot_paths": scf_grid_plot,
            "scf_grid_plot_paths": scf_grid_plot,
            "scf_grid_data_npz": str(scf_grid_npz),
            "scf_grid_indices": scf_grid_indices.tolist(),
            "data_npz": str(scf_grid_npz),
            "selected_tsv": None,
            "scf": scf_info,
            "h0_subtraction_info": h0_subtraction_info,
            "translation_order_npz": str(order_npz),
            "translation_order_plot_paths": order_plot,
            "translation_order_diagnostics": {
                key: float(value)
                for key, value in order.items()
                if key.endswith("_min") or key.endswith("_mean") or key.endswith("_max")
            },
            "diagnostics": {
                "h_grid_hermitian_error_ev": max_hermitian_error(grid_total),
                "h_line_hermitian_error_ev": float("nan"),
                "line_energy_min_shifted_mev": float(1000.0 * np.min(scf_grid_shifted)),
                "line_energy_max_shifted_mev": float(1000.0 * np.max(scf_grid_shifted)),
            },
        }

    line_path = build_polshyn_kx0_path(model.lattice, line_points, supercell=supercell)
    line_basis = build_polshyn_projected_basis(
        model,
        line_path.kvec,
        projected_indices=projected_indices,
        target_band_index=target_band_index,
        supercell=supercell,
    )
    if hf_engine == "wang":
        if wang_state is None or wang_grid_overlap_blocks is None:
            raise RuntimeError("Internal error: missing Wang HF state/overlaps")
        line_h_flat, _line_overlap_blocks, _line_grid_overlap_blocks = wang_target_hamiltonian(
            line_basis,
            grid_basis,
            wang_state,
            wang_grid_overlap_blocks,
            shifts,
            gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            progress_prefix=f"[polshyn {label} line]",
        )
        line_h = None
        h_line_error = float(np.max(np.abs(line_h_flat - np.swapaxes(line_h_flat.conjugate(), 0, 1))))
        line_energies = wang_sector_energies_from_flat_hamiltonian(
            line_h_flat,
            n_spin=grid_basis.n_spin,
            n_eta=grid_basis.n_eta,
            nb=grid_basis.nb,
        )
    else:
        line_diags = precompute_diagonal_overlaps(line_basis, shifts)
        line_grid_compact_overlaps = precompute_compact_overlaps(
            line_basis,
            grid_basis,
            shifts,
            progress_prefix=f"[polshyn {label} line-grid]",
        )
        interaction_line = build_interaction_blocks(
            line_basis,
            grid_basis,
            density,
            source_diagonals=grid_diags,
            target_diagonals=line_diags,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=epsilon_r,
            d_sc_nm=gate_distance_nm,
            include_hartree=True,
            include_fock=True,
            compact_overlaps=line_grid_compact_overlaps,
            progress_prefix=f"[polshyn {label} line]",
        )
        line_h = line_basis.h0_blocks + interaction_line
        h_line_error = max_hermitian_error(line_h)
        line_energies = path_sector_energies(line_h)
    shifted = line_energies - float(mu)
    x_ky = np.linspace(-np.pi, np.pi, int(line_points), dtype=float)

    stem = f"polshyn_figS1{label}_hf_kx0"
    plot_paths = _write_hf_line_plot(
        output_dir,
        stem=stem,
        x_ky=x_ky,
        energies_ev=shifted,
        occupation_counts=occ,
        title=f"Polshyn Fig. S1{label}, {'with remote' if lower_remote else 'no remote'}",
    )
    npz_path = output_dir / f"{stem}_data.npz"
    np.savez_compressed(
        npz_path,
        x_ky=x_ky,
        line_kvec=np.asarray(line_path.kvec, dtype=np.complex128),
        line_energies_ev_shifted=np.asarray(shifted, dtype=float),
        line_energies_ev_raw=np.asarray(line_energies, dtype=float),
        grid_density=np.asarray(density, dtype=np.complex128),
        occupation_counts=np.asarray(occ, dtype=int),
        projected_indices=np.asarray(projected_indices, dtype=int),
        target_band_index=np.asarray([target_band_index], dtype=int),
        reference_diagonal=np.asarray(grid_basis.reference_diagonal, dtype=float),
        fermi_level_ev=np.asarray([mu], dtype=float),
    )
    tsv_path = output_dir / f"{stem}_selected.tsv"
    rows = [x_ky]
    headers = ["ky_aM"]
    for ispin in range(shifted.shape[0]):
        for ieta in range(shifted.shape[1]):
            for ib in range(shifted.shape[2]):
                rows.append(shifted[ispin, ieta, ib])
                headers.append(f"spin{ispin}_eta{ieta}_band{ib}_ev")
    _write_tsv(tsv_path, headers, np.column_stack(rows))

    return {
        "label": f"S1{label}",
        "projected_indices": list(projected_indices),
        "target_band_index": int(target_band_index),
        "lower_remote": int(lower_remote),
        "upper_remote": int(upper_remote),
        "occupation_counts": occ.tolist(),
        "primitive_nu_from_counts": float(primitive_nu),
        "reference_diagonal": grid_basis.reference_diagonal.tolist(),
        "coulomb_area": str(coulomb_area),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
        "zero_hartree_q0": bool(zero_hartree_q0),
        "v0_nm_inv_sq": float(v0),
        "fermi_level_ev": float(mu),
        "plot_paths": plot_paths,
        "scf_grid_plot_paths": scf_grid_plot,
        "scf_grid_data_npz": str(scf_grid_npz),
        "scf_grid_indices": scf_grid_indices.tolist(),
        "data_npz": str(npz_path),
        "selected_tsv": str(tsv_path),
        "scf": scf_info,
        "h0_subtraction_info": h0_subtraction_info,
        "translation_order_npz": str(order_npz),
        "translation_order_plot_paths": order_plot,
        "translation_order_diagnostics": {
            key: float(value)
            for key, value in order.items()
            if key.endswith("_min") or key.endswith("_mean") or key.endswith("_max")
        },
        "diagnostics": {
            "h_grid_hermitian_error_ev": max_hermitian_error(grid_total),
            "h_line_hermitian_error_ev": float(h_line_error),
            "line_energy_min_shifted_mev": float(1000.0 * np.min(shifted)),
            "line_energy_max_shifted_mev": float(1000.0 * np.max(shifted)),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Polshyn et al. 2021 Supplementary Fig. S1(a-c) for tMBG.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--theta-deg", type=float, default=1.29)
    parser.add_argument(
        "--parameter-set",
        choices=("polshyn2020", "park"),
        default="polshyn2020",
        help="Single-particle continuum convention. Polshyn S1 should use the Polshyn 2020 supplementary Hamiltonian, not the Park checkpoint convention.",
    )
    parser.add_argument("--interlayer-potential", type=float, default=-0.033, help="Code layer potential in eV; Polshyn D=+0.4 V/nm maps to about -0.033 eV in this code convention.")
    parser.add_argument("--staggered-potential", type=float, default=0.0)
    parser.add_argument(
        "--blg-stacking",
        choices=("AB", "BA"),
        default="BA",
        help="Bernal chirality of the untwisted bilayer. Polshyn's C=2 target-band configuration uses BA; AB is kept for historical Park-checkpoint diagnostics.",
    )
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--target-band-index", default="auto")
    parser.add_argument(
        "--auto-target-role",
        choices=("lower-flat", "upper-flat"),
        default="upper-flat",
        help="Which member of the inferred central flat-band pair to use when --target-band-index=auto. With the Polshyn 2020 convention, the labeled C=2 band is the upper/conduction flat-band member.",
    )
    parser.add_argument("--s1a-points-per-segment", type=int, default=80)
    parser.add_argument("--s1a-bands-per-side", type=int, default=5)
    parser.add_argument("--panels", choices=("a", "b", "c", "ab", "ac", "bc", "abc"), default="abc")
    parser.add_argument("--kmesh", type=int, default=9)
    parser.add_argument("--line-points", type=int, default=49)
    parser.add_argument("--g-shells", type=int, default=2)
    parser.add_argument("--epsilon-r", type=float, default=20.0)
    parser.add_argument("--gate-distance-nm", type=float, default=120.0)
    parser.add_argument("--coulomb-area", choices=("supercell", "primitive"), default="supercell")
    parser.add_argument(
        "--hf-engine",
        choices=("wang", "legacy"),
        default="wang",
        help="SCF engine. 'wang' uses the generic Wang/Xiaoyu stored-projector ODA framework; 'legacy' keeps the earlier fixed-mixing compact Polshyn prototype.",
    )
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--oda-stall-threshold", type=float, default=1e-4)
    parser.add_argument("--hartree-scale", type=float, default=1.0, help="Diagnostic multiplier for Hartree kernels; keep 1.0 for production comparisons.")
    parser.add_argument("--fock-scale", type=float, default=1.0, help="Diagnostic multiplier for Fock kernels; keep 1.0 for production comparisons.")
    parser.add_argument(
        "--zero-hartree-q0",
        action="store_true",
        help="Diagnostic/IBM-style convention that removes only the uniform q=0 Hartree term while keeping finite-G Hartree and Fock.",
    )
    parser.add_argument(
        "--h0-subtraction",
        choices=("none", "active-reference", "projected-p0", "full-p0", "minus-full-p0", "active-minus-full-p0"),
        default="active-reference",
        help="Static h0 correction. Default 'active-reference' adds HF[P_ref] so active remote filled bands enter the HF potential; 'active-minus-full-p0' keeps HF[P_ref] but flips the active-remote P0 core used by the current paper-matching diagnostic; 'full-p0' keeps active-remote off-block pieces of the CNP subtraction; 'projected-p0' is the older active-projected diagnostic.",
    )
    parser.add_argument(
        "--p0-reference",
        choices=("decoupled-layers", "bernal-bilayer"),
        default="decoupled-layers",
        help="CNP reference used by P0 subtraction diagnostics. 'bernal-bilayer' keeps the untwisted BLG block and layer potentials while removing moire tunnelling.",
    )
    parser.add_argument(
        "--skip-reconstructed-line",
        action="store_true",
        help="Only emit exact SCF-grid kx=0 data/plots; skip dense post-SCF line reconstruction, which is not benchmark-trustworthy for this Polshyn run.",
    )
    parser.add_argument("--precision", type=float, default=1e-6)
    parser.add_argument("--init", choices=("bm", "random", "cdw"), default="cdw")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ensure_not_running_compute_on_login_node("Polshyn 2021 Fig. S1(a-c) reproduction")
    start = perf_counter()
    if args.output_dir is None:
        tag = args.run_tag or datetime.now().strftime("polshyn_figS1_%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / tag
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.parameter_set) == "polshyn2020":
        params = TMBGParameters.polshyn2020(
            interlayer_potential=float(args.interlayer_potential),
            staggered_potential=float(args.staggered_potential),
            blg_stacking=str(args.blg_stacking),
        )
    else:
        params = TMBGParameters.full(
            interlayer_potential=float(args.interlayer_potential),
            staggered_potential=float(args.staggered_potential),
            blg_stacking=str(args.blg_stacking),
        )
    model = TMBGModel.from_config(float(args.theta_deg), n_shells=int(args.n_shells), params=params)
    s1a_path = build_polshyn_s1a_path(model.lattice, int(args.s1a_points_per_segment))
    s1a_result = compute_bands_along_path(
        s1a_path,
        model.lattice,
        model.params,
        valley=1,
        n_bands=model.lattice.matrix_dim,
        return_eigenvectors=False,
    )
    flat_pair = infer_flat_band_indices(s1a_result.energies)
    target_auto = int(flat_pair[0] if str(args.auto_target_role) == "lower-flat" else flat_pair[1])
    target_band = target_auto if _parse_band_index(args.target_band_index) is None else int(_parse_band_index(args.target_band_index))
    s1a_low = max(0, target_band - int(args.s1a_bands_per_side))
    s1a_high = min(model.lattice.matrix_dim - 1, target_band + int(args.s1a_bands_per_side))
    s1a_selected = tuple(range(s1a_low, s1a_high + 1))
    s1a_plot = _write_s1a_plot(output_dir, s1a_result, target_band_index=target_band, selected_indices=s1a_selected)
    np.savez_compressed(
        output_dir / "polshyn_figS1a_noninteracting_bands.npz",
        kdist=np.asarray(s1a_path.kdist, dtype=float),
        kvec=np.asarray(s1a_path.kvec, dtype=np.complex128),
        energies_ev=np.asarray(s1a_result.energies, dtype=float),
        flat_pair=np.asarray(flat_pair, dtype=int),
        target_band_index=np.asarray([target_band], dtype=int),
        selected_indices=np.asarray(s1a_selected, dtype=int),
    )

    panels: list[dict[str, object]] = [
        {
            "label": "S1a",
            "plot_paths": s1a_plot,
            "target_band_index": int(target_band),
            "auto_flat_pair": [int(flat_pair[0]), int(flat_pair[1])],
            "auto_target_role": str(args.auto_target_role),
            "selected_indices": list(s1a_selected),
            "metrics": _s1a_band_metrics(s1a_result, target_band_index=target_band),
        }
    ]

    if "b" in str(args.panels):
        panels.append(
            _run_hf_panel(
                output_dir=output_dir,
                model=model,
                target_band_index=target_band,
                lower_remote=0,
                upper_remote=0,
                kmesh=int(args.kmesh),
                line_points=int(args.line_points),
                g_shells=int(args.g_shells),
                epsilon_r=float(args.epsilon_r),
                gate_distance_nm=float(args.gate_distance_nm),
                coulomb_area=str(args.coulomb_area),
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                precision=float(args.precision),
                seed=int(args.seed),
                init=str(args.init),
                label="b",
                hf_engine=str(args.hf_engine),
                oda_stall_threshold=float(args.oda_stall_threshold),
                h0_subtraction=str(args.h0_subtraction),
                p0_reference=str(args.p0_reference),
                skip_reconstructed_line=bool(args.skip_reconstructed_line),
                hartree_scale=float(args.hartree_scale),
                fock_scale=float(args.fock_scale),
                zero_hartree_q0=bool(args.zero_hartree_q0),
            )
        )
    if "c" in str(args.panels):
        panels.append(
            _run_hf_panel(
                output_dir=output_dir,
                model=model,
                target_band_index=target_band,
                lower_remote=3,
                upper_remote=2,
                kmesh=int(args.kmesh),
                line_points=int(args.line_points),
                g_shells=int(args.g_shells),
                epsilon_r=float(args.epsilon_r),
                gate_distance_nm=float(args.gate_distance_nm),
                coulomb_area=str(args.coulomb_area),
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                precision=float(args.precision),
                seed=int(args.seed),
                init=str(args.init),
                label="c",
                hf_engine=str(args.hf_engine),
                oda_stall_threshold=float(args.oda_stall_threshold),
                h0_subtraction=str(args.h0_subtraction),
                p0_reference=str(args.p0_reference),
                skip_reconstructed_line=bool(args.skip_reconstructed_line),
                hartree_scale=float(args.hartree_scale),
                fock_scale=float(args.fock_scale),
                zero_hartree_q0=bool(args.zero_hartree_q0),
            )
        )

    summary = {
        "task": "Polshyn et al. 2021 Supplementary Fig. S1(a-c) reproduction",
        "paper_reference": "Topological charge density waves at half-integer filling of a moire superlattice, Supplementary Fig. S1",
        "output_dir": str(output_dir),
        "theta_deg": float(args.theta_deg),
        "parameter_set": str(args.parameter_set),
        "interlayer_potential_ev": float(args.interlayer_potential),
        "staggered_potential_ev": float(args.staggered_potential),
        "blg_stacking": str(args.blg_stacking),
        "bernal_convention": str(params.bernal_convention),
        "model_name": str(params.model_name),
        "n_shells": int(args.n_shells),
        "auto_target_role": str(args.auto_target_role),
        "matrix_dim": int(model.lattice.matrix_dim),
        "n_g": int(model.lattice.n_g),
        "supercell": polshyn_doubled_cell().as_dict(),
        "supercell_basis": {
            "B1": [float((model.lattice.g_m1 / 2.0).real), float((model.lattice.g_m1 / 2.0).imag)],
            "B2": [float((model.lattice.g_m2 - model.lattice.g_m1 / 2.0).real), float((model.lattice.g_m2 - model.lattice.g_m1 / 2.0).imag)],
            "Q_cdw": [float((model.lattice.g_m1 / 2.0).real), float((model.lattice.g_m1 / 2.0).imag)],
        },
        "kmesh": int(args.kmesh),
        "line_points": int(args.line_points),
        "g_shells": int(args.g_shells),
        "epsilon_r": float(args.epsilon_r),
        "gate_distance_nm": float(args.gate_distance_nm),
        "coulomb_area": str(args.coulomb_area),
        "hf_engine": str(args.hf_engine),
        "oda_stall_threshold": float(args.oda_stall_threshold),
        "hartree_scale": float(args.hartree_scale),
        "fock_scale": float(args.fock_scale),
        "zero_hartree_q0": bool(args.zero_hartree_q0),
        "h0_subtraction": str(args.h0_subtraction),
        "p0_reference": str(args.p0_reference),
        "skip_reconstructed_line": bool(args.skip_reconstructed_line),
        "panels_requested": str(args.panels),
        "panels": panels,
        "status_note": (
            "Generated artifacts must be compared against the paper panel before claiming success. "
            "If disagreement remains, classify it as parameter tolerance or physics/code error."
        ),
        "elapsed_sec": float(perf_counter() - start),
    }
    summary_path = output_dir / "polshyn_figS1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Polshyn 2021 Supplementary Fig. S1(a-c)\n\n"
        "This run uses a doubled rectangular tMBG supercell with B1=G_M1/2 and B2=G_M2-G_M1/2.\n"
        "Do not mark as successful until the generated panels have been compared directly with the paper figure.\n\n"
        f"Summary: `{summary_path}`\n",
        encoding="utf-8",
    )
    print(f"[polshyn] wrote {summary_path}", flush=True)
    print(f"[polshyn] elapsed_sec={summary['elapsed_sec']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
