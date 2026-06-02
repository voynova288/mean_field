#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from analysis.topology import WavefunctionIndex, compute_lattice_topology
from mean_field.systems.tmbg import TMBGModel, TMBGParameters
from mean_field.systems.tmbg.polshyn_supercell import (
    basis_with_h0_correction,
    build_doubled_uniform_grid,
    build_full_p0_subtraction_h0_correction,
    build_polshyn_projected_basis,
    build_wang_overlap_blocks,
    cdw_density_blocks,
    moire_cell_area_nm2,
    occupation_counts_nu_7over2,
    overlap_blocks_with_hartree_q0_zeroed,
    polshyn_doubled_cell,
    reference_projector_blocks,
    run_projected_hf_scf_wang,
    scaled_overlap_blocks,
    supercell_interaction_shifts,
    wang_interaction_blocks_from_sector_density,
    wang_sector_hamiltonian_blocks,
)


def _sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    return np.asarray([int(ispin) + int(n_spin) * (int(ieta) + int(n_eta) * ib) for ib in range(int(nb))], dtype=int)


def _shift_grid(values: np.ndarray, dm: int, dn: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    dm = int(dm)
    dn = int(dn)
    nx, ny = arr.shape[1], arr.shape[2]
    out = np.zeros_like(arr)
    if abs(dm) >= nx or abs(dn) >= ny:
        return out
    dst_x = slice(max(dm, 0), nx + min(dm, 0))
    src_x = slice(max(-dm, 0), nx - max(dm, 0))
    dst_y = slice(max(dn, 0), ny + min(dn, 0))
    src_y = slice(max(-dn, 0), ny - max(dn, 0))
    out[:, dst_x, dst_y, ...] = arr[:, src_x, src_y, ...]
    return out


def _supercell_sewing_transform(basis, dm: int, dn: int):
    nx, ny = basis.embedding_shape

    def sew(vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.complex128)
        original_shape = arr.shape
        if arr.ndim == 1:
            cols = 1
            working = arr[:, None]
        elif arr.ndim == 2:
            cols = arr.shape[1]
            working = arr
        else:
            raise ValueError(f"Expected vector/subspace rank 1 or 2, got {arr.shape}")
        grid = working.reshape(basis.local_basis_size, nx, ny, cols, order="F")
        shifted = _shift_grid(grid, int(dm), int(dn))
        out = shifted.reshape(working.shape, order="F")
        return out.reshape(original_shape) if arr.ndim == 1 else out

    return sew


def _compute_topology_result(wavefunctions_mesh: np.ndarray, basis, indices: tuple[int, ...], *, label: str):
    # The Polshyn doubled-cell embedding uses supercell reciprocal coordinates.
    # Boundary sewing k -> k+B_i requires shifting the embedding by -e_i.  This
    # convention reproduces the sewn noninteracting checks C_target=+2 and
    # C_lower_remote=-1 for the Polshyn 2020 parameter set.
    return compute_lattice_topology(
        wavefunctions_mesh,
        indices,
        index=WavefunctionIndex(
            indices=tuple(int(i) for i in indices),
            role="hf_band" if len(indices) == 1 else "hf_subspace",
            labels=(str(label),),
            system="tmbg_polshyn_doubled",
            valley=1,
        ),
        k_grid_frac=basis.k_grid_frac,
        sewing_transforms=(
            _supercell_sewing_transform(basis, -1, 0),
            _supercell_sewing_transform(basis, 0, -1),
        ),
        link_method="determinant",
    )


def _topology_payload(result, indices: tuple[int, ...], *, label: str) -> dict[str, object]:
    return {
        "label": str(label),
        "indices": [int(i) for i in indices],
        "chern_number": float(result.chern_number),
        "rounded_chern_number": int(result.rounded_chern_number),
        "integer_residual": float(result.integer_residual),
        "min_link_magnitude": float(result.min_link_magnitude),
        "max_abs_berry_flux_over_pi": float(np.max(np.abs(result.berry_curvature)) / np.pi),
    }


def _compute_topology(wavefunctions_mesh: np.ndarray, basis, indices: tuple[int, ...], *, label: str) -> dict[str, object]:
    result = _compute_topology_result(wavefunctions_mesh, basis, indices, label=label)
    return _topology_payload(result, indices, label=label)


def _curvature_density(result, *, divide_by: float = 1.0) -> np.ndarray:
    mesh_x, mesh_y = result.berry_curvature.shape
    plaquette_area = (2.0 * np.pi / np.sqrt(3.0) / float(mesh_x)) * (2.0 * np.pi / float(mesh_y))
    return np.asarray(result.berry_curvature, dtype=float) / plaquette_area / float(divide_by)


def _plot_s1c_berry_panels(output_dir: Path, *, conduction: np.ndarray, valence: np.ndarray, folded_average: np.ndarray) -> dict[str, str]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_polshyn_chern")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    average = 0.5 * (np.asarray(conduction) + np.asarray(valence))
    panels = (
        ("d", conduction, r"$\Omega_C(\mathbf{k})/a_M^2$"),
        ("e", valence, r"$\Omega_V(\mathbf{k})/a_M^2$"),
        ("f", average, r"$(\Omega_C+\Omega_V)/(2a_M^2)$"),
        ("g", folded_average, r"$\Omega_{BM}(\mathbf{k})/a_M^2$"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.35), constrained_layout=True)
    extent = (-np.pi / np.sqrt(3.0), np.pi / np.sqrt(3.0), -np.pi, np.pi)
    vmax = max(1.0, *(float(np.nanmax(field)) for _label, field, _title in panels))
    for ax, (label, field, title) in zip(axes, panels, strict=True):
        centered_field = np.fft.fftshift(np.asarray(field, dtype=float), axes=(0, 1))
        img = ax.imshow(
            centered_field.T,
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel(r"$k_x a_M$")
        ax.set_xticks([-np.pi / np.sqrt(3.0), 0.0, np.pi / np.sqrt(3.0)])
        ax.set_xticklabels([r"$-\frac{\pi}{\sqrt{3}}$", "0", r"$\frac{\pi}{\sqrt{3}}$"], fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel(r"$k_y a_M$")
            ax.set_yticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
            ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=7)
        else:
            ax.set_yticks([])
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(title, fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    png = output_dir / "polshyn_figS1c_berry_curvature_d_to_g.png"
    pdf = output_dir / "polshyn_figS1c_berry_curvature_d_to_g.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _sector_hf_wavefunctions(basis, h_blocks: np.ndarray, *, ispin: int, ieta: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = int(round(np.sqrt(basis.nk)))
    if mesh * mesh != basis.nk:
        raise ValueError(f"Expected square grid, got nk={basis.nk}")
    nb = int(basis.nb)
    basis_dim = int(basis.basis_dimension)
    sector_wave = np.zeros((mesh, mesh, basis_dim, nb), dtype=np.complex128)
    sector_eval = np.zeros((mesh, mesh, nb), dtype=float)
    for ik in range(basis.nk):
        ix = ik // mesh
        iy = ik % mesh
        h = np.asarray(h_blocks[int(ispin), int(ieta), :, :, ik], dtype=np.complex128)
        h = 0.5 * (h + h.conjugate().T)
        evals, evecs = np.linalg.eigh(h)
        sector_eval[ix, iy, :] = evals
        u0 = np.asarray(basis.wavefunctions[:, :, int(ieta), ik], dtype=np.complex128)
        sector_wave[ix, iy, :, :] = u0 @ evecs
    return sector_wave, sector_eval


def _folded_noninteracting_wave_and_basis(model, *, band_index: int, mesh: int):
    frac, kvec = build_doubled_uniform_grid(model.lattice, int(mesh), supercell=polshyn_doubled_cell(), endpoint=False)
    basis = build_polshyn_projected_basis(
        model,
        kvec,
        projected_indices=(int(band_index),),
        target_band_index=int(band_index),
        supercell=polshyn_doubled_cell(),
        k_grid_frac=frac,
    )
    wave = basis.wavefunctions[:, :, 0, :].transpose(2, 0, 1).reshape(int(mesh), int(mesh), basis.basis_dimension, basis.nb)
    return wave, basis


def _folded_noninteracting_chern(model, *, band_index: int, mesh: int) -> dict[str, object]:
    wave, basis = _folded_noninteracting_wave_and_basis(model, band_index=int(band_index), mesh=int(mesh))
    return _compute_topology(wave, basis, (0, 1), label=f"noninteracting_folded_primitive_band_{band_index}")


def _panel_payload(summary: dict[str, object], label: str) -> dict[str, object]:
    target = f"S1{label.lower()}"
    for panel in summary["panels"]:
        if panel.get("label") == target:
            return panel
    raise ValueError(f"Panel {target} not found in summary")


def _build_corrected_basis(summary: dict[str, object], panel: dict[str, object]):
    params = TMBGParameters.polshyn2020(
        interlayer_potential=float(summary["interlayer_potential_ev"]),
        staggered_potential=float(summary.get("staggered_potential_ev", 0.0)),
        blg_stacking=str(summary.get("blg_stacking", "BA")),
    )
    model = TMBGModel.from_config(float(summary["theta_deg"]), n_shells=int(summary["n_shells"]), params=params)
    supercell = polshyn_doubled_cell()
    kmesh = int(summary["kmesh"])
    frac, kvec = build_doubled_uniform_grid(model.lattice, kmesh, supercell=supercell, endpoint=False)
    projected_indices = tuple(int(x) for x in panel["projected_indices"])
    target_band_index = int(panel["target_band_index"])
    basis = build_polshyn_projected_basis(
        model,
        kvec,
        projected_indices=projected_indices,
        target_band_index=target_band_index,
        supercell=supercell,
        k_grid_frac=frac,
    )
    shifts, gvecs = supercell_interaction_shifts(basis, int(summary["g_shells"]))
    area_ratio_for_v0 = 1 if str(summary.get("coulomb_area", "supercell")) == "primitive" else supercell.area_ratio
    v0 = 1.0 / moire_cell_area_nm2(model.lattice, area_ratio=area_ratio_for_v0)
    eps = float(summary["epsilon_r"])
    dnm = float(summary["gate_distance_nm"])
    h0_subtraction = str(panel.get("scf", {}).get("h0_subtraction", summary.get("h0_subtraction", "active-reference")))
    hartree_scale = float(panel.get("hartree_scale", summary.get("hartree_scale", 1.0)))
    fock_scale = float(panel.get("fock_scale", summary.get("fock_scale", 1.0)))
    p0_reference = str(summary.get("p0_reference", "decoupled-layers"))
    precomputed = None

    if h0_subtraction in {"full-p0", "minus-full-p0", "active-minus-full-p0"}:
        corr, _diag = build_full_p0_subtraction_h0_correction(
            basis,
            shifts=shifts,
            gvecs=gvecs,
            v0=v0,
            epsilon_r=eps,
            d_sc_nm=dnm,
            zero_hartree_q0=True,
            include_active_reference=True,
            p0_reference=p0_reference,
            hartree_scale=hartree_scale,
            fock_scale=fock_scale,
            progress_prefix="[chern h0-subtraction]",
        )
        if h0_subtraction == "minus-full-p0":
            corr = -corr
        elif h0_subtraction == "active-minus-full-p0":
            ref_density = reference_projector_blocks(basis)
            precomputed = build_wang_overlap_blocks(
                basis,
                basis,
                shifts,
                gvecs,
                epsilon_r=eps,
                d_sc_nm=dnm,
                include_hartree=True,
                include_fock=True,
                progress_prefix="[chern h0-active-reference]",
            )
            ref_blocks = precomputed
            if hartree_scale != 1.0 or fock_scale != 1.0:
                ref_blocks = scaled_overlap_blocks(ref_blocks, hartree_scale=hartree_scale, fock_scale=fock_scale)
            ref_corr = wang_interaction_blocks_from_sector_density(
                ref_density,
                basis,
                overlap_blocks_with_hartree_q0_zeroed(ref_blocks),
                v0=v0,
            )
            corr = 2.0 * ref_corr - corr
        basis = basis_with_h0_correction(basis, corr)
    elif h0_subtraction == "active-reference":
        ref_density = reference_projector_blocks(basis)
        if float(np.linalg.norm(ref_density)) > 1.0e-14:
            precomputed = build_wang_overlap_blocks(
                basis,
                basis,
                shifts,
                gvecs,
                epsilon_r=eps,
                d_sc_nm=dnm,
                include_hartree=True,
                include_fock=True,
                progress_prefix="[chern h0-reference]",
            )
            ref_blocks = precomputed
            if hartree_scale != 1.0 or fock_scale != 1.0:
                ref_blocks = scaled_overlap_blocks(ref_blocks, hartree_scale=hartree_scale, fock_scale=fock_scale)
            corr = wang_interaction_blocks_from_sector_density(
                ref_density,
                basis,
                overlap_blocks_with_hartree_q0_zeroed(ref_blocks),
                v0=v0,
            )
            basis = basis_with_h0_correction(basis, corr)
    elif h0_subtraction != "none":
        raise ValueError(f"Unsupported h0_subtraction={h0_subtraction!r}")
    return model, basis, shifts, gvecs, v0, precomputed


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run Polshyn HF and compute FHS Chern numbers on the exact SCF grid.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--panel", choices=("b", "c"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--precision", type=float, default=None)
    args = parser.parse_args()
    start = perf_counter()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    panel = _panel_payload(summary, args.panel)
    model, basis, shifts, gvecs, v0, precomputed = _build_corrected_basis(summary, panel)
    occ = np.asarray(panel["occupation_counts"], dtype=int)
    init = cdw_density_blocks(
        projected_indices=tuple(int(x) for x in panel["projected_indices"]),
        target_band_index=int(panel["target_band_index"]),
        n_spin=basis.n_spin,
        n_eta=basis.n_eta,
        nb=basis.nb,
        nk=basis.nk,
        reference_diagonal=basis.reference_diagonal,
    )
    state, overlaps, scf_info = run_projected_hf_scf_wang(
        basis,
        occupation_counts=occ,
        shifts=shifts,
        gvecs=gvecs,
        v0=v0,
        epsilon_r=float(summary["epsilon_r"]),
        d_sc_nm=float(summary["gate_distance_nm"]),
        max_iter=int(args.max_iter or panel["scf"].get("iterations", summary.get("max_iter", 80)) + 5),
        precision=float(args.precision or panel["scf"].get("precision", 1e-6)),
        initial_density_blocks=init,
        oda_stall_threshold=float(summary.get("oda_stall_threshold", 1e-4)),
        progress_prefix="[chern grid]",
        overlap_blocks=precomputed,
        seed=int(panel["scf"].get("seed", 1)),
        hartree_scale=float(summary.get("hartree_scale", 1.0)),
        fock_scale=float(summary.get("fock_scale", 1.0)),
        zero_hartree_q0=bool(panel["scf"].get("zero_hartree_q0", False)),
    )
    h_blocks = wang_sector_hamiltonian_blocks(state, basis)

    sector_wave, sector_eval = _sector_hf_wavefunctions(basis, h_blocks, ispin=0, ieta=0)
    n_occ_target_sector = int(occ[0, 0])
    selected = []
    if 0 < n_occ_target_sector < basis.nb:
        selected.extend([
            ("Kplus_up_HF_valence_occupied_top", (n_occ_target_sector - 1,)),
            ("Kplus_up_HF_conduction_empty_bottom", (n_occ_target_sector,)),
            ("Kplus_up_HF_two_split_target_bands", (n_occ_target_sector - 1, n_occ_target_sector)),
        ])
    # Include individual K+ up bands and lower occupied remote subspace diagnostics for S1c.
    for ib in range(basis.nb):
        selected.append((f"Kplus_up_HF_band_{ib}", (ib,)))
    if n_occ_target_sector >= 2:
        selected.append(("Kplus_up_occupied_subspace", tuple(range(n_occ_target_sector))))
    cherns = []
    for label, indices in selected:
        try:
            cherns.append(_compute_topology(sector_wave, basis, tuple(indices), label=label))
        except Exception as exc:
            cherns.append({"label": label, "indices": [int(i) for i in indices], "error": str(exc)})

    ni = {
        "target_folded_subspace": _folded_noninteracting_chern(model, band_index=int(panel["target_band_index"]), mesh=int(summary["kmesh"])),
    }
    lower_remote = int(panel["target_band_index"]) - 1
    ni["lower_remote_folded_subspace"] = _folded_noninteracting_chern(model, band_index=lower_remote, mesh=int(summary["kmesh"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    berry_plot_paths: dict[str, str] | None = None
    berry_npz = args.output_dir / f"polshyn_figS1{args.panel}_berry_curvature_panels.npz"
    if args.panel == "c" and 0 < n_occ_target_sector < basis.nb:
        valence_result = _compute_topology_result(
            sector_wave,
            basis,
            (n_occ_target_sector - 1,),
            label="Kplus_up_HF_valence_occupied_top",
        )
        conduction_result = _compute_topology_result(
            sector_wave,
            basis,
            (n_occ_target_sector,),
            label="Kplus_up_HF_conduction_empty_bottom",
        )
        target_wave, target_basis = _folded_noninteracting_wave_and_basis(
            model,
            band_index=int(panel["target_band_index"]),
            mesh=int(summary["kmesh"]),
        )
        folded_result = _compute_topology_result(
            target_wave,
            target_basis,
            (0, 1),
            label="noninteracting_folded_target_subspace",
        )
        conduction_density = _curvature_density(conduction_result)
        valence_density = _curvature_density(valence_result)
        folded_average_density = _curvature_density(folded_result, divide_by=2.0)
        berry_plot_paths = _plot_s1c_berry_panels(
            args.output_dir,
            conduction=conduction_density,
            valence=valence_density,
            folded_average=folded_average_density,
        )
        np.savez_compressed(
            berry_npz,
            hf_conduction_density=np.asarray(conduction_density, dtype=float),
            hf_valence_density=np.asarray(valence_density, dtype=float),
            hf_average_density=np.asarray(0.5 * (conduction_density + valence_density), dtype=float),
            folded_noninteracting_average_density=np.asarray(folded_average_density, dtype=float),
            hf_conduction_chern=np.asarray([conduction_result.chern_number], dtype=float),
            hf_valence_chern=np.asarray([valence_result.chern_number], dtype=float),
            folded_noninteracting_chern=np.asarray([folded_result.chern_number], dtype=float),
        )

    payload = {
        "summary_json": str(args.summary_json),
        "panel": f"S1{args.panel}",
        "source_h0_subtraction": str(panel["scf"].get("h0_subtraction")),
        "kmesh": int(summary["kmesh"]),
        "projected_indices": [int(x) for x in panel["projected_indices"]],
        "target_band_index": int(panel["target_band_index"]),
        "occupation_counts": np.asarray(occ, dtype=int).tolist(),
        "scf_info": scf_info,
        "noninteracting_checks": ni,
        "chern": cherns,
        "berry_curvature_panels_npz": str(berry_npz) if berry_plot_paths is not None else None,
        "berry_curvature_plot_paths": berry_plot_paths,
        "expected_from_paper": {
            "target_primitive_C_Kplus": 2,
            "lower_remote_primitive_C_Kplus": -1,
            "HF_split_target_band_cherns_Kplus_up": [1, 1],
            "SBCI_hall_from_one_unfilled_C1_band": -1,
        },
        "elapsed_sec": float(perf_counter() - start),
    }
    out_json = args.output_dir / f"polshyn_figS1{args.panel}_hf_chern_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / f"polshyn_figS1{args.panel}_hf_chern_wavefunction_data.npz",
        sector_energies_ev=np.asarray(sector_eval, dtype=float),
    )
    report = [
        f"# Polshyn Fig. S1{args.panel} HF Chern check",
        "",
        f"source: `{args.summary_json}`",
        f"h0_subtraction: `{panel['scf'].get('h0_subtraction')}`",
        f"kmesh: `{summary['kmesh']}`",
        "",
        "## Noninteracting folded checks",
    ]
    for key, val in ni.items():
        report.append(f"- {key}: C = {val['chern_number']:.8f}, rounded = {val['rounded_chern_number']}, min_link = {val['min_link_magnitude']:.3e}")
    report.extend(["", "## HF Chern table", "", "| label | indices | C | rounded | min link |", "|---|---:|---:|---:|---:|"])
    for item in cherns:
        if "error" in item:
            report.append(f"| {item['label']} | {item['indices']} | ERROR |  |  |")
        else:
            report.append(f"| {item['label']} | {item['indices']} | {item['chern_number']:.8f} | {item['rounded_chern_number']} | {item['min_link_magnitude']:.3e} |")
    (args.output_dir / f"polshyn_figS1{args.panel}_hf_chern_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
