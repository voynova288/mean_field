from __future__ import annotations

import numpy as np

from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .model import TDBGModel
from .params import TDBGParameters
from .projected_hf_config import (
    SPIN_LABELS,
    TDBGProjectedHFConfig,
    VALLEY_SEQUENCE,
    tdbg_parameters_from_paper_ud_for_valley,
    validate_tdbg_projected_hf_config,
)
from .projected_hf_geometry import _shift_table, tdbg_band_window_indices, tdbg_moire_area_nm2
from .projected_hf_state import TDBGProjectedHFData, TDBGStateLabel

_EV_TO_J = 1.602176634e-19
_NM_TO_M = 1.0e-9
_ELECTRON_MASS_KG = 9.1093837015e-31
_HBAR_J_S = 1.054571817e-34
_MU_B_J_PER_T = 9.2740100783e-24

def _projected_orbital_g_matrix(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int,
    band_indices: tuple[int, ...],
    delta_k_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected-band energies, wavefunctions, and orbital-g matrix.

    This implements the Liu SI Eq. (8) structure in the zero-field continuum
    eigenbasis. The derivative matrices are finite differences of the full
    continuum Hamiltonian with respect to kx/ky in units of J*m, so
    `mu_B * g * B` is an energy. The selected subspace is not re-diagonalized
    here; callers add the resulting matrix to the projected one-body `h0`.
    """

    evals_ev, evecs = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=None)
    h_x_plus = build_hamiltonian(k_tilde + float(delta_k_nm_inv), lattice, params, valley=valley)
    h_x_minus = build_hamiltonian(k_tilde - float(delta_k_nm_inv), lattice, params, valley=valley)
    h_y_plus = build_hamiltonian(k_tilde + 1j * float(delta_k_nm_inv), lattice, params, valley=valley)
    h_y_minus = build_hamiltonian(k_tilde - 1j * float(delta_k_nm_inv), lattice, params, valley=valley)
    d_hx_j_m = ((h_x_plus - h_x_minus) / (2.0 * float(delta_k_nm_inv))) * _EV_TO_J * _NM_TO_M
    d_hy_j_m = ((h_y_plus - h_y_minus) / (2.0 * float(delta_k_nm_inv))) * _EV_TO_J * _NM_TO_M
    dx = evecs.conjugate().T @ d_hx_j_m @ evecs
    dy = evecs.conjugate().T @ d_hy_j_m @ evecs
    evals_j = np.asarray(evals_ev, dtype=float) * _EV_TO_J
    selected = np.asarray(band_indices, dtype=int)
    g_matrix = np.zeros((selected.size, selected.size), dtype=np.complex128)
    denom_cutoff = 1.0e-30
    prefactor = -1j * _ELECTRON_MASS_KG / (2.0 * _HBAR_J_S * _HBAR_J_S)
    all_indices = np.arange(evals_j.size, dtype=int)
    for ia, m in enumerate(selected):
        for ib, mp in enumerate(selected):
            terms = np.zeros(evals_j.size, dtype=np.complex128)
            denom_m = evals_j[int(m)] - evals_j
            denom_mp = evals_j[int(mp)] - evals_j
            valid = (np.abs(denom_m) > denom_cutoff) & (np.abs(denom_mp) > denom_cutoff)
            valid &= all_indices != int(m)
            valid &= all_indices != int(mp)
            if np.any(valid):
                berry_comm = dx[int(m), valid] * dy[valid, int(mp)] - dy[int(m), valid] * dx[valid, int(mp)]
                terms[valid] = (1.0 / denom_m[valid] + 1.0 / denom_mp[valid]) * berry_comm
                g_matrix[ia, ib] = prefactor * np.sum(terms[valid])
    g_matrix = 0.5 * (g_matrix + g_matrix.conjugate().T)
    return np.asarray(evals_ev[selected], dtype=float), evecs[:, selected], g_matrix

def _projected_onebody_and_wavefunctions(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int,
    band_indices: tuple[int, ...],
    orbital_zeeman_b_t: float,
    orbital_zeeman_delta_k_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(orbital_zeeman_b_t)) <= 0.0:
        evals, vec = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=max(band_indices) + 1)
        selected = np.asarray(band_indices, dtype=int)
        return np.diag(np.asarray(evals, dtype=float)[selected]).astype(np.complex128), vec[:, selected]
    evals, vec, g_matrix = _projected_orbital_g_matrix(
        k_tilde,
        lattice,
        params,
        valley=valley,
        band_indices=band_indices,
        delta_k_nm_inv=float(orbital_zeeman_delta_k_nm_inv),
    )
    zeeman_ev = (_MU_B_J_PER_T * float(orbital_zeeman_b_t) / _EV_TO_J) * g_matrix
    h0 = np.diag(evals).astype(np.complex128) + zeeman_ev
    h0 = 0.5 * (h0 + h0.conjugate().T)
    return h0, vec

def build_tdbg_projected_hf_data(config: TDBGProjectedHFConfig) -> TDBGProjectedHFData:
    validate_tdbg_projected_hf_config(config)
    valley_params = {
        int(valley): tdbg_parameters_from_paper_ud_for_valley(
            config.paper_ud_ev,
            stacking=config.stacking,
            valley=int(valley),
            convention=config.paper_ud_convention,
        )
        for valley in VALLEY_SEQUENCE
    }
    params = valley_params[VALLEY_SEQUENCE[0]]
    model = TDBGModel.from_config(config.theta_deg, cut=config.cut, params=params)
    band_indices = tdbg_band_window_indices(model.matrix_dim, config.window)
    n_band = len(band_indices)
    if n_band < 1:
        raise ValueError("Projected TDBG window must include at least one band")
    lower_count = 0 if n_band == 1 else n_band // 2
    if n_band != 1 and n_band % 2 != 0:
        raise ValueError(f"Projected TDBG multi-band window must be even, got {n_band}")

    mesh = int(config.mesh_size)
    frac_shift = config.frac_shift if config.frac_shift is not None else (0.5 / mesh, 0.5 / mesh)
    k_grid_frac, kvec_grid = build_moire_k_grid(model.lattice, mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid, dtype=np.complex128).reshape(-1)
    nk = int(kvec.size)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * n_band

    labels: list[TDBGStateLabel] = []
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band_index in enumerate(band_indices):
                idx = iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)
                labels.append(
                    TDBGStateLabel(
                        index=int(idx),
                        spin=spin,
                        valley=int(valley),
                        band_position=int(iband),
                        band_index=int(band_index),
                    )
                )

    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    wavefunctions = np.zeros((nt, nk, model.lattice.n_q, 4), dtype=np.complex128)
    for valley in VALLEY_SEQUENCE:
        valley_labels = [label for label in labels if int(label.valley) == int(valley)]
        for ik, kval in enumerate(kvec):
            h_proj, vec = _projected_onebody_and_wavefunctions(
                kval,
                model.lattice,
                valley_params[int(valley)],
                valley=int(valley),
                band_indices=band_indices,
                orbital_zeeman_b_t=float(config.orbital_zeeman_b_t),
                orbital_zeeman_delta_k_nm_inv=float(config.orbital_zeeman_delta_k_nm_inv),
            )
            for spin in SPIN_LABELS:
                spin_indices = [label.index for label in valley_labels if label.spin == spin]
                h0[np.ix_(spin_indices, spin_indices, [ik])] = h_proj[:, :, None]
            for label in valley_labels:
                wavefunctions[label.index, ik, :, :] = vec[:, label.band_position].reshape(model.lattice.n_q, 4)

    reference_density = np.zeros((nt, nt, nk), dtype=np.complex128)
    if lower_count > 0:
        for label in labels:
            if label.band_position < lower_count:
                reference_density[label.index, label.index, :] = 1.0

    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * lower_count
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(
            f"Invalid TDBG occupied count per k: neutral={neutral_occupied_per_k}, "
            f"filling={config.filling}, occupied={n_occupied_per_k}, nt={nt}"
        )
    shifts, gvecs, srcmaps = _shift_table(model.lattice, config.interaction.g_shells)
    return TDBGProjectedHFData(
        model=model,
        config=config,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec,
        band_indices=band_indices,
        labels=tuple(labels),
        h0=h0,
        wavefunctions=wavefunctions,
        reference_density=reference_density,
        n_occupied_per_k=int(n_occupied_per_k),
        lower_band_count=int(lower_count),
        moire_area_nm2=tdbg_moire_area_nm2(model.lattice),
        shifts=shifts,
        shift_gvecs=gvecs,
        shift_srcmaps=srcmaps,
        valley_params=valley_params,
    )

__all__ = [
    "_projected_onebody_and_wavefunctions",
    "_projected_orbital_g_matrix",
    "build_tdbg_projected_hf_data",
]
