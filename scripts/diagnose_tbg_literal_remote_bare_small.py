#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    ProjectedWavefunctionBasis,
    build_projected_interaction_hamiltonian,
    compute_density_overlap_trace_from_diagonal,
    contract_fock_term_from_overlap,
    run_hartree_fock_problem,
)
from mean_field.crpa import (
    CRPAScreenedCoulomb,
    build_crpa_projected_interaction_components,
    build_crpa_projected_interaction_hamiltonian,
    build_fock_screened_overlap_blocks,
    classify_flat_bands,
    crpa_split_energy_functional,
    half_reference_delta_like,
    load_crpa_result,
    physical_projector_from_delta,
    solve_all_band_bm_model,
    split_oda_parameter,
)
from mean_field.crpa.coulomb import coulomb_potential_table_mev
from mean_field.crpa.dielectric import compute_dielectric
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.hf import (
    RestrictedHartreeFockState,
    build_full_density_from_hamiltonian,
    build_h0_from_bm,
    build_overlap_block_set,
    coulomb_unit,
    initialize_full_state,
    offdiag_flavor_norm,
    restricted_filling,
    restricted_gap_estimate,
)
from mean_field.systems.tbg.zero_field.model import build_b0_uniform_lattice, solve_bm_model
from mean_field.systems.tbg.zero_field.model import build_sigma_z_from_uk


DEFAULT_OUTPUT_DIR = Path("results") / "TBG_HF_cRPA" / "crpa_gap_smoke_20260523" / "literal_remote_small"
DEFAULT_CRPA_DIR = (
    Path("results")
    / "TBG_HF_cRPA"
    / "crpa_lk24_lg9_q11_hfcompatible_fig4_20260522_epsbn4_merged"
)


def _rectangular_overlap(
    target: ProjectedWavefunctionBasis,
    source: ProjectedWavefunctionBasis,
    m: int,
    n: int,
) -> np.ndarray:
    if target.local_basis_size != source.local_basis_size:
        raise ValueError(f"local_basis_size mismatch: {target.local_basis_size} != {source.local_basis_size}")
    if target.n_flavor != source.n_flavor:
        raise ValueError(f"n_flavor mismatch: {target.n_flavor} != {source.n_flavor}")
    if target.n_spin != source.n_spin:
        raise ValueError(f"n_spin mismatch: {target.n_spin} != {source.n_spin}")
    if target.grid_shape != source.grid_shape:
        raise ValueError(f"grid_shape mismatch: {target.grid_shape} != {source.grid_shape}")
    if target.basis_dimension != source.basis_dimension:
        raise ValueError(f"basis_dimension mismatch: {target.basis_dimension} != {source.basis_dimension}")

    nx, ny = target.grid_shape
    target_band_k = target.n_band * target.nk
    source_band_k = source.n_band * source.nk
    overlap_blocks = np.zeros(
        (
            target.n_spin,
            target.n_flavor,
            target_band_k,
            source.n_spin,
            source.n_flavor,
            source_band_k,
        ),
        dtype=np.complex128,
        order="F",
    )

    for iflavor in range(target.n_flavor):
        ul = target.wavefunctions[:, :, iflavor, :].reshape(target.basis_dimension, target_band_k, order="F")
        ur = source.wavefunctions[:, :, iflavor, :].reshape(source.basis_dimension, source_band_k, order="F")
        shifted = np.roll(
            ur.reshape(source.local_basis_size, nx, ny, source_band_k, order="F"),
            shift=(0, -int(m), -int(n), 0),
            axis=(0, 1, 2, 3),
        ).reshape(source.basis_dimension, source_band_k, order="F")
        lambda_kp = ul.conj().T @ shifted
        for ispin in range(target.n_spin):
            overlap_blocks[ispin, iflavor, :, ispin, iflavor, :] = lambda_kp

    return overlap_blocks.reshape((target.nt, target.nk, source.nt, source.nk), order="F")


def _same_basis_diagonal_overlap(
    basis: ProjectedWavefunctionBasis,
    m: int,
    n: int,
) -> np.ndarray:
    nx, ny = basis.grid_shape
    band_k = basis.n_band * basis.nk
    diagonal = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    state_idx = np.arange(basis.nt, dtype=int).reshape((basis.n_spin, basis.n_flavor, basis.n_band), order="F")

    for iflavor in range(basis.n_flavor):
        u = basis.wavefunctions[:, :, iflavor, :].reshape(basis.basis_dimension, band_k, order="F")
        shifted = np.roll(
            u.reshape(basis.local_basis_size, nx, ny, band_k, order="F"),
            shift=(0, -int(m), -int(n), 0),
            axis=(0, 1, 2, 3),
        ).reshape(basis.basis_dimension, band_k, order="F")
        lambda_kp = u.conj().T @ shifted
        for ik in range(basis.nk):
            band_slice = slice(ik * basis.n_band, (ik + 1) * basis.n_band)
            block = lambda_kp[band_slice, band_slice]
            for ispin in range(basis.n_spin):
                rows = np.asarray(state_idx[ispin, iflavor, :], dtype=int)
                diagonal[np.ix_(rows, rows, [ik])] = block[:, :, None]
    return diagonal


def _rectangular_fock_term(
    target_source_overlap: np.ndarray,
    source_density: np.ndarray,
    coeff_matrix: np.ndarray,
) -> np.ndarray:
    target_source_overlap = np.asarray(target_source_overlap, dtype=np.complex128)
    source_density = np.asarray(source_density, dtype=np.complex128)
    coeff_matrix = np.asarray(coeff_matrix)
    nt_target, nk_target, nt_source, nk_source = target_source_overlap.shape
    if source_density.shape != (nt_source, nt_source, nk_source):
        raise ValueError(f"Expected source density shape {(nt_source, nt_source, nk_source)}, got {source_density.shape}")
    if coeff_matrix.shape != (nk_target, nk_source):
        raise ValueError(f"Expected coeff_matrix shape {(nk_target, nk_source)}, got {coeff_matrix.shape}")

    lambda_blocks = np.transpose(target_source_overlap, (1, 3, 0, 2))
    density_t = np.transpose(source_density, (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", lambda_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(lambda_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def _remote_density(
    *,
    n_spin: int,
    n_flavor: int,
    remote_band_indices: np.ndarray,
    flat_lo: int,
    flat_hi: int,
    nk: int,
    mode: str,
) -> np.ndarray:
    n_remote = int(remote_band_indices.size)
    nt_source = int(n_spin * n_flavor * n_remote)
    density = np.zeros((nt_source, nt_source, nk), dtype=np.complex128)
    idx = np.arange(nt_source, dtype=int).reshape((n_spin, n_flavor, n_remote), order="F")

    weights = np.zeros(n_remote, dtype=float)
    below = remote_band_indices < int(flat_lo)
    above = remote_band_indices > int(flat_hi)
    if mode == "remote_delta":
        weights[below] = 0.5
        weights[above] = -0.5
    elif mode == "remote_projector":
        weights[below] = 1.0
        weights[above] = 0.0
    else:
        raise ValueError(f"Unsupported remote density mode: {mode}")

    for ispin in range(n_spin):
        for iflavor in range(n_flavor):
            for iband, weight in enumerate(weights):
                state = int(idx[ispin, iflavor, iband])
                density[state, state, :] = float(weight)
    return density


def _build_literal_remote_static_components(
    *,
    active_overlap_blocks: HFOverlapBlockSet,
    active_basis: ProjectedWavefunctionBasis,
    remote_basis: ProjectedWavefunctionBasis,
    remote_density: np.ndarray,
    v0: float,
    beta: float,
) -> np.ndarray:
    nk_source = int(remote_density.shape[2])
    if nk_source != active_basis.nk:
        raise ValueError(f"Expected matching target/source nk, got {active_basis.nk} and {nk_source}")
    scale = float(beta) * float(v0) / float(nk_source)
    hartree_h = np.zeros((active_basis.nt, active_basis.nt, active_basis.nk), dtype=np.complex128)
    fock_h = np.zeros_like(hartree_h)

    for shift in active_overlap_blocks.shifts:
        target_diagonal = active_overlap_blocks.diagonal_overlaps.get(shift)
        hartree_kernel = active_overlap_blocks.hartree_screening.get(shift)
        fock_kernel = active_overlap_blocks.fock_screening.get(shift)
        if target_diagonal is None and hartree_kernel is not None:
            raise ValueError(f"Missing target diagonal overlap for shift {shift}")

        if hartree_kernel is not None and target_diagonal is not None:
            source_diagonal = _same_basis_diagonal_overlap(remote_basis, int(shift[0]), int(shift[1]))
            tr_pg = compute_density_overlap_trace_from_diagonal(remote_density, source_diagonal)
            prefactor = scale * float(hartree_kernel)
            if prefactor != 0.0:
                hartree_h += prefactor * tr_pg * target_diagonal

        if fock_kernel is not None:
            target_source_overlap = _rectangular_overlap(active_basis, remote_basis, int(shift[0]), int(shift[1]))
            fock_h -= _rectangular_fock_term(target_source_overlap, remote_density, scale * fock_kernel)

    return hartree_h, fock_h


def _build_active_static_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square active density, got {density.shape}")
    scale = float(beta) * float(v0) / float(nk)
    hartree_h = np.zeros_like(density)
    fock_h = np.zeros_like(density)

    for shift in overlap_blocks.shifts:
        overlap = overlap_blocks.overlaps[shift]
        if overlap.shape != (nt, nk, nt, nk):
            raise ValueError(f"Expected active overlap shape {(nt, nk, nt, nk)}, got {overlap.shape}")
        diagonal_overlap = overlap_blocks.diagonal_overlaps.get(shift)
        hartree_kernel = overlap_blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            if diagonal_overlap is None:
                raise ValueError(f"Missing active diagonal overlap for shift {shift}")
            tr_pg = compute_density_overlap_trace_from_diagonal(density, diagonal_overlap)
            prefactor = scale * float(hartree_kernel)
            if prefactor != 0.0:
                hartree_h += prefactor * tr_pg * diagonal_overlap
        fock_kernel = overlap_blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            fock_h -= contract_fock_term_from_overlap(overlap, density, scale * fock_kernel)

    return hartree_h, fock_h


def _active_lower_flat_projector(solution) -> np.ndarray:
    if int(solution.nb) < 2:
        raise ValueError(f"Expected at least two active bands, got nb={solution.nb}")
    density = np.zeros((solution.nt, solution.nt, solution.nk), dtype=np.complex128)
    idx = np.arange(solution.nt, dtype=int).reshape(
        (solution.n_spin, solution.n_eta, solution.nb),
        order="F",
    )
    for ispin in range(solution.n_spin):
        for ieta in range(solution.n_eta):
            state = int(idx[ispin, ieta, 0])
            density[state, state, :] = 1.0
    return density


def _target_diagonal_block_set(
    *,
    target_basis: ProjectedWavefunctionBasis,
    reference_blocks: HFOverlapBlockSet,
) -> HFOverlapBlockSet:
    diagonal_overlaps = {
        shift: _same_basis_diagonal_overlap(target_basis, int(shift[0]), int(shift[1]))
        for shift in reference_blocks.shifts
    }
    return HFOverlapBlockSet(
        shifts=reference_blocks.shifts,
        gvecs=np.asarray(reference_blocks.gvecs, dtype=np.complex128),
        overlaps={},
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=dict(reference_blocks.hartree_screening),
        fock_screening=dict(reference_blocks.fock_screening),
    )


def _all_band_h0(all_band) -> np.ndarray:
    n_spin = 2
    nt = int(n_spin * all_band.n_eta * all_band.nb)
    h0 = np.zeros((nt, nt, all_band.nk), dtype=np.complex128)
    idx = np.arange(nt, dtype=int).reshape((n_spin, all_band.n_eta, all_band.nb), order="F")
    for ik in range(all_band.nk):
        for ieta in range(all_band.n_eta):
            for ispin in range(n_spin):
                rows = idx[ispin, ieta, :]
                h0[rows, rows, ik] = np.asarray(all_band.spectrum[:, ieta, ik], dtype=float)
    return h0


def _remote_renormalized_active_solution(
    *,
    reference_solution,
    all_band,
    active_overlap: HFOverlapBlockSet,
    remote_basis: ProjectedWavefunctionBasis,
    remote_density: np.ndarray,
    flat_indices: np.ndarray,
    params: TBGParameters,
    beta: float,
) :
    """Diagonalize h_BM + Sigma_remote in all-band space, then reselect flat bands."""

    all_basis = ProjectedWavefunctionBasis(
        all_band.uk,
        grid_shape=(int(all_band.lg), int(all_band.lg)),
        n_spin=reference_solution.n_spin,
        local_basis_size=reference_solution.nlocal,
        name="all_band_remote_target",
    )
    full_target_blocks = _target_diagonal_block_set(
        target_basis=all_basis,
        reference_blocks=active_overlap,
    )
    remote_hartree, remote_fock = _build_literal_remote_static_components(
        active_overlap_blocks=full_target_blocks,
        active_basis=all_basis,
        remote_basis=remote_basis,
        remote_density=remote_density,
        v0=coulomb_unit(params),
        beta=float(beta),
    )
    full_onebody = _all_band_h0(all_band) + remote_hartree + remote_fock
    state_idx = np.arange(full_onebody.shape[0], dtype=int).reshape(
        (reference_solution.n_spin, all_band.n_eta, all_band.nb),
        order="F",
    )
    flat_indices = np.asarray(flat_indices, dtype=int).reshape(-1)
    nb_active = int(reference_solution.nb)
    if nb_active != flat_indices.size:
        raise ValueError(f"Expected {nb_active} flat indices, got {flat_indices.size}")

    renorm_uk = np.zeros_like(reference_solution.uk, dtype=np.complex128)
    renorm_spectrum = np.zeros_like(reference_solution.spectrum, dtype=float)
    selected_indices = np.zeros((all_band.n_eta, all_band.nk, nb_active), dtype=int)
    selected_weights = np.zeros((all_band.n_eta, all_band.nk, nb_active), dtype=float)

    for ieta in range(all_band.n_eta):
        for ik in range(all_band.nk):
            rows = state_idx[0, ieta, :]
            h_block = np.asarray(full_onebody[np.ix_(rows, rows, [ik])][:, :, 0], dtype=np.complex128)
            h_block = 0.5 * (h_block + h_block.conj().T)
            evals, evecs = np.linalg.eigh(h_block)
            flat_weight = np.sum(np.abs(evecs[flat_indices, :]) ** 2, axis=0)
            chosen = np.argsort(-flat_weight, kind="stable")[:nb_active]
            chosen = chosen[np.argsort(evals[chosen], kind="stable")]
            selected_indices[ieta, ik, :] = chosen
            selected_weights[ieta, ik, :] = flat_weight[chosen]
            renorm_spectrum[:, ieta, ik] = evals[chosen]
            renorm_uk[:, :, ieta, ik] = np.asarray(all_band.uk[:, :, ieta, ik], dtype=np.complex128) @ evecs[:, chosen]

    renorm_sigma_z = build_sigma_z_from_uk(
        renorm_uk,
        lg=int(reference_solution.lg),
        n_spin=int(reference_solution.n_spin),
    )
    renorm_solution = replace(
        reference_solution,
        uk=renorm_uk,
        spectrum=renorm_spectrum,
        sigma_z=renorm_sigma_z,
    )
    diagnostics = {
        "remote_hartree_fro": float(np.linalg.norm(remote_hartree)),
        "remote_fock_fro": float(np.linalg.norm(remote_fock)),
        "remote_total_fro": float(np.linalg.norm(remote_hartree + remote_fock)),
        "remote_total_max_abs": float(np.max(np.abs(remote_hartree + remote_fock))),
        "min_selected_flat_weight": float(np.min(selected_weights)),
        "mean_selected_flat_weight": float(np.mean(selected_weights)),
        "max_selected_flat_weight": float(np.max(selected_weights)),
        "selected_indices": selected_indices.tolist(),
    }
    return renorm_solution, diagnostics


def _band_metrics(energies: np.ndarray, *, nu: float) -> dict[str, float]:
    sorted_energies = np.sort(np.asarray(energies, dtype=float), axis=0)
    nt, nk = sorted_energies.shape
    total_occ = int(round((float(nu) + 4.0) / 8.0 * nt * nk))
    per_k = total_occ / float(nk)
    metrics: dict[str, float] = {
        "energy_min": float(np.min(sorted_energies)),
        "energy_max": float(np.max(sorted_energies)),
        "energy_span": float(np.max(sorted_energies) - np.min(sorted_energies)),
        "occupied_states": float(total_occ),
        "occupied_per_k": float(per_k),
    }
    if abs(per_k - round(per_k)) > 1e-9 or per_k <= 0 or per_k >= nt:
        metrics.update(
            {
                "direct_gap": float("nan"),
                "indirect_gap": float("nan"),
                "top_valence_width": float("nan"),
                "bottom_conduction_width": float("nan"),
                "max_conduction_width": float("nan"),
            }
        )
        return metrics

    occ = int(round(per_k))
    valence = sorted_energies[occ - 1, :]
    bottom_conduction = sorted_energies[occ, :]
    conduction_widths = [
        float(np.max(sorted_energies[iband, :]) - np.min(sorted_energies[iband, :]))
        for iband in range(occ, nt)
    ]
    metrics.update(
        {
            "direct_gap": float(np.min(bottom_conduction - valence)),
            "indirect_gap": float(np.min(bottom_conduction) - np.max(valence)),
            "top_valence_width": float(np.max(valence) - np.min(valence)),
            "bottom_conduction_width": float(np.max(bottom_conduction) - np.min(bottom_conduction)),
            "max_conduction_width": float(max(conduction_widths) if conduction_widths else float("nan")),
            "mean_conduction_width": float(np.mean(conduction_widths) if conduction_widths else float("nan")),
        }
    )
    return metrics


def _exact_gamma_m_k_gamma_kprime_indices(lk: int, params: TBGParameters) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, int], ...]]:
    if int(lk) % 6 != 0:
        raise ValueError("The exact SCF high-symmetry path requires lk divisible by 6.")
    lk = int(lk)
    nodes = (
        ("Gamma", (0, 0)),
        ("M", (lk // 2, lk // 2)),
        ("K", (2 * lk // 3, lk // 3)),
        ("Gamma", (0, 0)),
        ("Kprime", (lk // 3, 2 * lk // 3)),
    )
    coords: list[tuple[int, int]] = [nodes[0][1]]
    node_positions: list[tuple[str, int]] = [(nodes[0][0], 0)]
    for label, end in nodes[1:]:
        start = coords[-1]
        di = int(end[0] - start[0])
        dj = int(end[1] - start[1])
        steps = int(np.gcd(abs(di), abs(dj)))
        if steps <= 0:
            raise ValueError(f"Repeated path node without segment: {label}")
        step_i = di // steps
        step_j = dj // steps
        for step in range(1, steps + 1):
            coords.append((int(start[0] + step * step_i), int(start[1] + step * step_j)))
        node_positions.append((label, len(coords) - 1))

    indices = np.asarray([i + (lk + 1) * j for i, j in coords], dtype=int)
    kvec = np.asarray([(i / float(lk)) * params.g1 + (j / float(lk)) * params.g2 for i, j in coords], dtype=np.complex128)
    kdist = np.zeros(indices.size, dtype=float)
    if indices.size > 1:
        kdist[1:] = np.cumsum(np.abs(np.diff(kvec)))
    return indices, kdist, tuple(node_positions)


def _write_path_tsv(
    path: Path,
    *,
    dynamic_mode: str,
    static_mode: str,
    energies: np.ndarray,
    mu: float,
    path_indices: np.ndarray,
    path_dist: np.ndarray,
) -> None:
    sorted_energies = np.sort(np.asarray(energies, dtype=float), axis=0)
    lines = ["dynamic_mode\tstatic_mode\tpath_order\tk_index\tk_dist\tband\tenergy_mev\tenergy_minus_mu_mev"]
    for order, (ik, kdist) in enumerate(zip(path_indices, path_dist, strict=True)):
        for iband in range(sorted_energies.shape[0]):
            value = float(sorted_energies[iband, int(ik)])
            lines.append(
                f"{dynamic_mode}\t{static_mode}\t{order}\t{int(ik)}\t{float(kdist):.16e}\t"
                f"{iband}\t{value:.16e}\t{value - float(mu):.16e}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metrics_tsv(path: Path, rows: list[dict[str, str | float | int | bool]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = tuple(rows[0].keys())
    lines = ["\t".join(keys)]
    for row in rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.16e}")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_float_list(raw: str) -> tuple[float, ...]:
    text = str(raw).strip()
    if not text:
        return ()
    values: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values)


def _scale_tag(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return text


def _scaled_crpa_screening(
    screening: CRPAScreenedCoulomb,
    *,
    params: TBGParameters,
    chi0_scale: float,
) -> CRPAScreenedCoulomb:
    scale = float(chi0_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"Expected non-negative finite chi0_scale, got {chi0_scale}")
    if abs(scale - 1.0) < 1.0e-15:
        return screening

    result = screening.result
    dielectric = np.zeros_like(result.dielectric_matrix, dtype=np.complex128)
    epsilon_inv = np.zeros_like(result.epsilon_inv, dtype=np.complex128)
    screened_v = np.zeros_like(result.screened_v, dtype=np.complex128)
    effective_epsilon = np.zeros_like(result.effective_epsilon, dtype=float)
    for iq, q_tilde in enumerate(np.asarray(result.q_tilde, dtype=np.complex128)):
        v_q = coulomb_potential_table_mev(
            complex(q_tilde),
            np.asarray(result.q_vectors, dtype=np.complex128),
            params,
            result.coulomb_params,
        )
        item = compute_dielectric(scale * np.asarray(result.chi0[iq], dtype=np.complex128), v_q)
        dielectric[iq, :, :] = item.epsilon
        epsilon_inv[iq, :, :] = item.epsilon_inv
        screened_v[iq, :, :] = item.screened_v
        effective_epsilon[iq, :] = item.effective_epsilon

    metadata = dict(result.metadata)
    metadata["diagnostic_chi0_scale"] = float(scale)
    scaled_result = replace(
        result,
        chi0=scale * np.asarray(result.chi0, dtype=np.complex128),
        dielectric_matrix=dielectric,
        epsilon_inv=epsilon_inv,
        screened_v=screened_v,
        effective_epsilon=effective_epsilon,
        metadata=metadata,
    )
    return CRPAScreenedCoulomb(scaled_result)


def _write_band_plot(
    path: Path,
    *,
    title: str,
    mode_results: list[dict[str, object]],
    nu: float,
    path_indices: np.ndarray,
    path_dist: np.ndarray,
    node_positions: tuple[tuple[str, int], ...],
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_tbg_literal_remote_small")
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    nrows = max(1, len(mode_results))
    fig, axes = plt.subplots(nrows, 1, figsize=(8.0, 2.5 * nrows), sharex=True, squeeze=False)
    node_x = [float(path_dist[pos]) for _label, pos in node_positions]
    node_labels = [label for label, _pos in node_positions]

    for ax, result in zip(axes[:, 0], mode_results, strict=True):
        energies = np.sort(np.asarray(result["energies"], dtype=float), axis=0)
        mu = float(result["mu"])
        nt, nk = energies.shape
        total_occ = int(round((float(nu) + 4.0) / 8.0 * nt * nk))
        occ = int(round(total_occ / float(nk)))
        path_energies = energies[:, path_indices] - mu
        for iband in range(nt):
            color = "#1f2933" if iband < occ else "#b91c1c"
            linewidth = 1.6 if iband < occ else 1.0
            ax.plot(path_dist, path_energies[iband, :], color=color, linewidth=linewidth)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        for xpos in node_x:
            ax.axvline(xpos, color="#999999", linestyle=":", linewidth=0.8)
        metrics = result["metrics"]
        ax.set_ylabel("E - mu (meV)")
        ax.set_title(
            f"{result['static_mode']}  direct={float(metrics['direct_gap']):.3g} "
            f"indirect={float(metrics['indirect_gap']):.3g} "
            f"wv={float(metrics['top_valence_width']):.3g}",
            fontsize=10,
        )
    axes[-1, 0].set_xticks(node_x, node_labels)
    axes[-1, 0].set_xlabel("SCF-grid path")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _state_update_callback(state: RestrictedHartreeFockState, density_update: DensityUpdateResult) -> None:
    sigma = np.asarray(density_update.observables["sigma_ztauz"], dtype=float)
    state.sigma_ztauz[:, :] = sigma
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(state.density)
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)


def _run_small_hf(
    *,
    dynamic_mode: str,
    static_mode: str,
    grid_solution,
    active_overlap: HFOverlapBlockSet,
    crpa_screening: CRPAScreenedCoulomb | None,
    params: TBGParameters,
    static_h: np.ndarray,
    nu: float,
    init_mode: str,
    seed: int,
    max_iter: int,
    precision: float,
    beta: float,
    fock_interpolation: str,
    screening_kwargs: dict[str, float | bool],
    iteration_rows: list[dict[str, str | float | int | bool]],
):
    state = RestrictedHartreeFockState.from_bm_solution(grid_solution, nu=nu, precision=precision)
    state.h0[:, :, :] += np.asarray(static_h, dtype=np.complex128)
    if dynamic_mode == "bare":
        dynamic_overlap = active_overlap

        def active_builder(density: np.ndarray) -> np.ndarray:
            return build_projected_interaction_hamiltonian(
                density,
                dynamic_overlap,
                v0=state.v0,
                beta=beta,
            )

    elif dynamic_mode == "crpa":
        if crpa_screening is None:
            raise ValueError("dynamic_mode=crpa requires --crpa-dir")
        dynamic_overlap = build_fock_screened_overlap_blocks(
            active_overlap,
            lattice_kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
            params=params,
            crpa_screening=crpa_screening,
            fock_interpolation=fock_interpolation,
            **screening_kwargs,
        )

        def active_builder(density: np.ndarray) -> np.ndarray:
            return build_crpa_projected_interaction_hamiltonian(
                density,
                dynamic_overlap,
                crpa_screening=crpa_screening,
                params=params,
                beta=beta,
            )

    else:
        raise ValueError(f"Unsupported dynamic mode: {dynamic_mode}")

    def interaction_builder(density_delta: np.ndarray) -> np.ndarray:
        return active_builder(physical_projector_from_delta(density_delta))

    def oda_delta_interaction_builder(delta_density: np.ndarray) -> np.ndarray:
        return active_builder(delta_density)

    def oda_parameterizer(state_obj, delta_density: np.ndarray) -> float:
        delta_h = oda_delta_interaction_builder(delta_density)
        return split_oda_parameter(
            state_obj,
            delta_density,
            delta_h=delta_h,
            interaction_h=state_obj.hamiltonian - state_obj.h0,
        )

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, sigma_ztauz, mu = build_full_density_from_hamiltonian(
            hamiltonian,
            state.sigma_z,
            nu=nu,
        )
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables={"sigma_ztauz": sigma_ztauz},
        )

    def step_callback(state_obj, step) -> None:
        _state_update_callback(state_obj, step.density_update)
        metrics = _band_metrics(step.density_update.energies, nu=nu)
        iteration_rows.append(
            {
                "dynamic_mode": dynamic_mode,
                "static_mode": static_mode,
                "iteration": int(step.iteration),
                "energy": float(step.energy),
                "norm_raw": float(step.norm_raw),
                "norm_mixed": float(step.norm_mixed),
                "oda_lambda": float(step.oda_lambda),
                **metrics,
            }
        )

    kernel = HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        energy_functional=crpa_split_energy_functional if dynamic_mode == "crpa" else crpa_split_energy_functional,
        oda_parameterizer=oda_parameterizer,
        step_callback=step_callback,
        final_state_callback=_state_update_callback,
        convergence_rule="mixed",
    )
    problem = HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_full_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
        ),
        kernel=kernel,
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
    )
    metrics = _band_metrics(run.state.energies, nu=nu)
    return {
        "dynamic_mode": dynamic_mode,
        "static_mode": static_mode,
        "iterations": int(run.iterations),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "mu": float(run.state.mu),
        "metrics": metrics,
        "energies": np.asarray(run.state.energies, dtype=float).copy(),
        "density_norm": float(np.linalg.norm(run.state.density)),
        "static_fro_norm": float(np.linalg.norm(static_h)),
        "static_max_abs": float(np.max(np.abs(static_h))) if static_h.size else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-system diagnostic for literal remote-band bare background in TBG HF+cRPA."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crpa-dir", type=Path, default=DEFAULT_CRPA_DIR)
    parser.add_argument("--dynamic", choices=("crpa", "bare", "both"), default="both")
    parser.add_argument("--lk", type=int, default=6)
    parser.add_argument("--lg", type=int, default=3)
    parser.add_argument("--overlap-lg", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--precision", type=float, default=1e-5)
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--nu", type=float, default=-3.0)
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.4)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument("--epsilon-r", type=float, default=4.0)
    parser.add_argument("--tanh-argument-scale-a", type=float, default=400.0 / 2.46)
    parser.add_argument("--zero-cutoff", type=float, default=1.0e-6)
    parser.add_argument("--finite-zero-limit", action="store_true", default=True)
    parser.add_argument("--fock-interpolation", choices=("matrix_diagonal", "nearest", "linear"), default="matrix_diagonal")
    parser.add_argument("--init", default="vp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--crpa-chi0-scale", type=float, default=1.0)
    parser.add_argument("--current-half-scales", default="0.05,0.1,0.2,0.5")
    parser.add_argument(
        "--include-remote-renormalized-basis",
        action="store_true",
        help="Also diagonalize h_BM plus the literal remote potential in all-band space, then run HF in the reselected flat basis.",
    )
    parser.add_argument(
        "--remote-band-window",
        type=int,
        default=0,
        help="If positive, keep only this many remote bands below and above the flat pair in literal-remote diagnostics.",
    )
    parser.add_argument("--allow-login", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not bool(args.allow_login):
        ensure_not_running_compute_on_login_node("TBG literal remote small diagnostic")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    params = TBGParameters.from_degrees(
        float(args.theta_deg),
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
    )
    screening_lm = float(args.tanh_argument_scale_a) * 0.246
    screening_kwargs: dict[str, float | bool] = {
        "relative_permittivity": float(args.epsilon_r),
        "screening_lm": float(screening_lm),
        "finite_zero_limit": bool(args.finite_zero_limit),
        "zero_cutoff": float(args.zero_cutoff),
    }

    print(
        "[literal-remote-small] setup "
        f"lk={args.lk} lg={args.lg} overlap_lg={args.overlap_lg} max_iter={args.max_iter} "
        f"dynamic={args.dynamic}",
        flush=True,
    )
    start = perf_counter()
    crpa_screening = None
    if args.dynamic in {"crpa", "both"}:
        crpa_result = load_crpa_result(args.crpa_dir)
        crpa_screening = CRPAScreenedCoulomb(crpa_result)
        if abs(float(args.crpa_chi0_scale) - 1.0) >= 1.0e-15:
            scale_start = perf_counter()
            crpa_screening = _scaled_crpa_screening(
                crpa_screening,
                params=params,
                chi0_scale=float(args.crpa_chi0_scale),
            )
            print(
                "[literal-remote-small] crpa_scaled "
                f"chi0_scale={float(args.crpa_chi0_scale):.6g} elapsed_sec={perf_counter() - scale_start:.3f}",
                flush=True,
            )
        screening_kwargs = {
            "relative_permittivity": float(crpa_result.coulomb_params.epsilon_bn),
            "screening_lm": float(crpa_result.coulomb_params.screening_lm),
            "finite_zero_limit": bool(crpa_result.coulomb_params.finite_zero_limit),
            "zero_cutoff": float(crpa_result.coulomb_params.zero_cutoff),
        }
        print(
            "[literal-remote-small] crpa "
            f"dir={args.crpa_dir} lk={crpa_result.lk} lg={crpa_result.lg} q_lg={crpa_result.q_lg} "
            f"chi0_scale={float(args.crpa_chi0_scale):.6g}",
            flush=True,
        )

    grid = build_b0_uniform_lattice(params, int(args.lk))
    grid_solution = solve_bm_model(params, grid.kvec, lg=int(args.lg), sigma_rotation=True)
    active_h0 = build_h0_from_bm(grid_solution)
    active_overlap = build_overlap_block_set(grid_solution, lg=int(args.overlap_lg), **screening_kwargs)
    active_basis = ProjectedWavefunctionBasis(
        grid_solution.uk,
        grid_shape=(int(args.lg), int(args.lg)),
        n_spin=grid_solution.n_spin,
        local_basis_size=grid_solution.nlocal,
        name="active_flat_c2t",
    )
    print(
        "[literal-remote-small] active "
        f"nk={grid_solution.nk} nt={grid_solution.nt} v0={coulomb_unit(params):.6g}",
        flush=True,
    )

    all_band = solve_all_band_bm_model(
        params,
        grid.kvec,
        lg=int(args.lg),
        periodic_g_grid=True,
        sigma_rotation=True,
    )
    classification = classify_flat_bands(all_band.spectrum, method="center")
    flat_indices = np.asarray(classification.flat_indices, dtype=int)
    first_flat = flat_indices[0, 0, :]
    if not np.all(flat_indices == first_flat[None, None, :]):
        raise RuntimeError("Small diagnostic currently expects k-independent center flat-band indices.")
    flat_lo = int(np.min(first_flat))
    flat_hi = int(np.max(first_flat))
    all_remote_indices = np.asarray(
        [iband for iband in range(all_band.nb) if iband < flat_lo or iband > flat_hi],
        dtype=int,
    )
    remote_window = int(args.remote_band_window)
    if remote_window > 0:
        lower_remote = np.arange(max(0, flat_lo - remote_window), flat_lo, dtype=int)
        upper_remote = np.arange(flat_hi + 1, min(all_band.nb, flat_hi + 1 + remote_window), dtype=int)
        remote_indices = np.concatenate([lower_remote, upper_remote]).astype(int)
    else:
        remote_indices = all_remote_indices
    remote_basis = ProjectedWavefunctionBasis(
        all_band.uk[:, remote_indices, :, :],
        grid_shape=(int(args.lg), int(args.lg)),
        n_spin=grid_solution.n_spin,
        local_basis_size=grid_solution.nlocal,
        name="remote_raw_all_band",
    )
    print(
        "[literal-remote-small] remote "
        f"all_nb={all_band.nb} flat=({flat_lo},{flat_hi}) "
        f"remote_bands={remote_indices.size} remote_band_window={remote_window}",
        flush=True,
    )

    zero_static = np.zeros_like(active_h0)
    current_hartree_static, current_fock_static = _build_active_static_components(
        half_reference_delta_like(zero_static),
        active_overlap,
        v0=coulomb_unit(params),
        beta=float(args.beta),
    )
    current_static = current_hartree_static + current_fock_static
    remote_delta_density = _remote_density(
        n_spin=grid_solution.n_spin,
        n_flavor=grid_solution.n_eta,
        remote_band_indices=remote_indices,
        flat_lo=flat_lo,
        flat_hi=flat_hi,
        nk=grid_solution.nk,
        mode="remote_delta",
    )
    remote_projector_density = _remote_density(
        n_spin=grid_solution.n_spin,
        n_flavor=grid_solution.n_eta,
        remote_band_indices=remote_indices,
        flat_lo=flat_lo,
        flat_hi=flat_hi,
        nk=grid_solution.nk,
        mode="remote_projector",
    )
    literal_delta_hartree, literal_delta_fock = _build_literal_remote_static_components(
        active_overlap_blocks=active_overlap,
        active_basis=active_basis,
        remote_basis=remote_basis,
        remote_density=remote_delta_density,
        v0=coulomb_unit(params),
        beta=float(args.beta),
    )
    literal_delta_static = literal_delta_hartree + literal_delta_fock
    base_static_modes = {
        "no_remote": zero_static,
        "current_active_half": current_static,
        "current_active_half_fock_only": current_fock_static,
        "literal_remote_delta": literal_delta_static,
        "literal_remote_delta_fock_only": literal_delta_fock,
        "minus_literal_remote_delta": -literal_delta_static,
        "minus_literal_remote_delta_fock_only": -literal_delta_fock,
    }
    for scale in _parse_float_list(str(args.current_half_scales)):
        if abs(float(scale)) < 1.0e-15 or abs(float(scale) - 1.0) < 1.0e-15:
            continue
        base_static_modes[f"current_active_half_x{_scale_tag(scale)}"] = float(scale) * current_static
    static_components = {
        "current_active_half": (current_hartree_static, current_fock_static),
        "literal_remote_delta": (literal_delta_hartree, literal_delta_fock),
    }
    crpa_extra_static_modes: dict[str, np.ndarray] = {}
    if crpa_screening is not None:
        screened_overlap_for_reference = build_fock_screened_overlap_blocks(
            active_overlap,
            lattice_kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
            params=params,
            crpa_screening=crpa_screening,
            fock_interpolation=str(args.fock_interpolation),
            **screening_kwargs,
        )
        crpa_reference_hartree, crpa_reference_fock = build_crpa_projected_interaction_components(
            half_reference_delta_like(zero_static),
            screened_overlap_for_reference,
            crpa_screening=crpa_screening,
            params=params,
            beta=float(args.beta),
        )
        crpa_extra_static_modes = {
            "current_crpa_half": crpa_reference_hartree + crpa_reference_fock,
            "current_crpa_half_fock_only": crpa_reference_fock,
        }
        static_components["current_crpa_half"] = (crpa_reference_hartree, crpa_reference_fock)

        cnp_hartree, cnp_fock = build_crpa_projected_interaction_components(
            _active_lower_flat_projector(grid_solution),
            screened_overlap_for_reference,
            crpa_screening=crpa_screening,
            params=params,
            beta=float(args.beta),
        )
        cnp_reference = cnp_hartree + cnp_fock
        crpa_extra_static_modes.update(
            {
                "minus_active_cnp_crpa": -cnp_reference,
                "minus_active_cnp_crpa_fock_only": -cnp_fock,
                "literal_remote_delta_minus_active_cnp_crpa": literal_delta_static - cnp_reference,
                "literal_remote_delta_fock_only_minus_active_cnp_crpa_fock_only": literal_delta_fock - cnp_fock,
            }
        )
        static_components["minus_active_cnp_crpa"] = (-cnp_hartree, -cnp_fock)
        static_components["minus_active_cnp_crpa_fock_only"] = (np.zeros_like(cnp_fock), -cnp_fock)
        static_components["literal_remote_delta_minus_active_cnp_crpa"] = (
            literal_delta_hartree - cnp_hartree,
            literal_delta_fock - cnp_fock,
        )
        static_components["literal_remote_delta_fock_only_minus_active_cnp_crpa_fock_only"] = (
            np.zeros_like(cnp_fock),
            literal_delta_fock - cnp_fock,
        )

    renormalized_basis_cases: dict[str, tuple[object, HFOverlapBlockSet, dict[str, object]]] = {}
    if bool(args.include_remote_renormalized_basis):
        for remote_mode, remote_density in (
            ("remote_delta", remote_delta_density),
            ("remote_projector", remote_projector_density),
        ):
            case_start = perf_counter()
            renorm_solution, renorm_diagnostics = _remote_renormalized_active_solution(
                reference_solution=grid_solution,
                all_band=all_band,
                active_overlap=active_overlap,
                remote_basis=remote_basis,
                remote_density=remote_density,
                flat_indices=first_flat,
                params=params,
                beta=float(args.beta),
            )
            renorm_overlap = build_overlap_block_set(
                renorm_solution,
                lg=int(args.overlap_lg),
                **screening_kwargs,
            )
            case_name = f"{remote_mode}_renormalized_basis"
            renorm_diagnostics["elapsed_sec"] = float(perf_counter() - case_start)
            renormalized_basis_cases[case_name] = (renorm_solution, renorm_overlap, renorm_diagnostics)
            print(
                "[literal-remote-small] renormalized_basis "
                f"mode={case_name} "
                f"remote_fro={float(renorm_diagnostics['remote_total_fro']):.6e} "
                f"remote_max={float(renorm_diagnostics['remote_total_max_abs']):.6e} "
                f"min_flat_weight={float(renorm_diagnostics['min_selected_flat_weight']):.6e} "
                f"elapsed_sec={float(renorm_diagnostics['elapsed_sec']):.3f}",
                flush=True,
            )

    for name, h_static in {**base_static_modes, **crpa_extra_static_modes}.items():
        component_text = ""
        if name in static_components:
            h_part, f_part = static_components[name]
            component_text = (
                f" hartree_fro={np.linalg.norm(h_part):.6e} fock_fro={np.linalg.norm(f_part):.6e}"
                f" hartree_max={np.max(np.abs(h_part)):.6e} fock_max={np.max(np.abs(f_part)):.6e}"
            )
        print(
            "[literal-remote-small] static "
            f"mode={name} fro={np.linalg.norm(h_static):.6e} max_abs={np.max(np.abs(h_static)):.6e}"
            f"{component_text}",
            flush=True,
        )

    dynamic_modes = ("bare", "crpa") if args.dynamic == "both" else (str(args.dynamic),)
    path_indices, path_dist, node_positions = _exact_gamma_m_k_gamma_kprime_indices(int(args.lk), params)
    iteration_rows: list[dict[str, str | float | int | bool]] = []
    metrics_rows: list[dict[str, str | float | int | bool]] = []
    plot_groups: dict[str, list[dict[str, object]]] = {mode: [] for mode in dynamic_modes}
    summary_results: list[dict[str, object]] = []

    for dynamic_mode in dynamic_modes:
        mode_static_modes = dict(base_static_modes)
        if dynamic_mode == "crpa":
            mode_static_modes.update(crpa_extra_static_modes)
        case_specs: list[tuple[str, object, HFOverlapBlockSet, np.ndarray]] = [
            (static_mode, grid_solution, active_overlap, static_h)
            for static_mode, static_h in mode_static_modes.items()
        ]
        case_specs.extend(
            (
                static_mode,
                renorm_solution,
                renorm_overlap,
                np.zeros_like(build_h0_from_bm(renorm_solution), dtype=np.complex128),
            )
            for static_mode, (renorm_solution, renorm_overlap, _diagnostics) in renormalized_basis_cases.items()
        )
        for static_mode, case_solution, case_overlap, static_h in case_specs:
            run_start = perf_counter()
            print(
                "[literal-remote-small] hf:start "
                f"dynamic={dynamic_mode} static={static_mode}",
                flush=True,
            )
            result = _run_small_hf(
                dynamic_mode=dynamic_mode,
                static_mode=static_mode,
                grid_solution=case_solution,
                active_overlap=case_overlap,
                crpa_screening=crpa_screening,
                params=params,
                static_h=static_h,
                nu=float(args.nu),
                init_mode=str(args.init),
                seed=int(args.seed),
                max_iter=int(args.max_iter),
                precision=float(args.precision),
                beta=float(args.beta),
                fock_interpolation=str(args.fock_interpolation),
                screening_kwargs=screening_kwargs,
                iteration_rows=iteration_rows,
            )
            elapsed = perf_counter() - run_start
            result["elapsed_sec"] = float(elapsed)
            plot_groups[dynamic_mode].append(result)
            path_tsv = output_dir / f"scf_path_{dynamic_mode}_{static_mode}.tsv"
            _write_path_tsv(
                path_tsv,
                dynamic_mode=dynamic_mode,
                static_mode=static_mode,
                energies=np.asarray(result["energies"], dtype=float),
                mu=float(result["mu"]),
                path_indices=path_indices,
                path_dist=path_dist,
            )
            metrics = result["metrics"]
            row = {
                "dynamic_mode": dynamic_mode,
                "static_mode": static_mode,
                "iterations": int(result["iterations"]),
                "converged": bool(result["converged"]),
                "exit_reason": str(result["exit_reason"]),
                "mu": float(result["mu"]),
                "elapsed_sec": float(elapsed),
                "static_fro_norm": float(result["static_fro_norm"]),
                "static_max_abs": float(result["static_max_abs"]),
                **metrics,
            }
            metrics_rows.append(row)
            summary_results.append({key: value for key, value in row.items()})
            print(
                "[literal-remote-small] hf:done "
                f"dynamic={dynamic_mode} static={static_mode} "
                f"direct={float(metrics['direct_gap']):.6g} indirect={float(metrics['indirect_gap']):.6g} "
                f"wv={float(metrics['top_valence_width']):.6g} wcmax={float(metrics['max_conduction_width']):.6g}",
                flush=True,
            )

    for dynamic_mode, mode_results in plot_groups.items():
        _write_band_plot(
            output_dir / f"small_scf_path_{dynamic_mode}.png",
            title=f"TBG small HF {dynamic_mode}, lk={args.lk}, lg={args.lg}, nu={args.nu}",
            mode_results=mode_results,
            nu=float(args.nu),
            path_indices=path_indices,
            path_dist=path_dist,
            node_positions=node_positions,
        )

    _write_metrics_tsv(output_dir / "final_metrics.tsv", metrics_rows)
    _write_metrics_tsv(output_dir / "iteration_metrics.tsv", iteration_rows)
    summary = {
        "elapsed_sec": float(perf_counter() - start),
        "output_dir": str(output_dir),
        "crpa_dir": str(args.crpa_dir),
        "dynamic": str(args.dynamic),
        "lk": int(args.lk),
        "lg": int(args.lg),
        "overlap_lg": int(args.overlap_lg),
        "max_iter": int(args.max_iter),
        "crpa_chi0_scale": float(args.crpa_chi0_scale),
        "theta_deg": float(args.theta_deg),
        "nu": float(args.nu),
        "w0": float(args.w0),
        "w1": float(args.w1),
        "vf": float(args.vf),
        "screening_kwargs": screening_kwargs,
        "flat_indices": [int(v) for v in first_flat],
        "remote_band_count": int(remote_indices.size),
        "remote_band_window": int(remote_window),
        "remote_band_indices": [int(v) for v in remote_indices.tolist()],
        "all_remote_band_count": int(all_remote_indices.size),
        "remote_renormalized_basis": {
            name: diagnostics for name, (_solution, _overlap, diagnostics) in renormalized_basis_cases.items()
        },
        "results": summary_results,
        "artifacts": {
            "final_metrics_tsv": str(output_dir / "final_metrics.tsv"),
            "iteration_metrics_tsv": str(output_dir / "iteration_metrics.tsv"),
            "bare_plot": str(output_dir / "small_scf_path_bare.png"),
            "crpa_plot": str(output_dir / "small_scf_path_crpa.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[literal-remote-small] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
