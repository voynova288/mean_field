from __future__ import annotations

import numpy as np

from ...core.validation import ValidationCheck, ValidationReport, ValidationStatus, status_from_bool
from .hamiltonian import build_hamiltonian, flat_band_indices
from .lattice import rotate_complex
from .model import RLGhBNModel
from .params import (
    DEFAULT_FERMI_VELOCITY_MEV_NM,
    DEFAULT_REMOTE_VELOCITY_MEV_NM,
    DEFAULT_T1_MEV,
    DEFAULT_T2_MEV,
    MOIRE_PARAMETER_TABLE,
)



def _status_from_bool(condition: bool) -> ValidationStatus:
    return status_from_bool(condition)





def _moire_off_bottom_residual(model: RLGhBNModel, k_tilde: complex) -> float:
    full = model.build_hamiltonian(k_tilde, valley=1)
    no_moire_params = type(model.params)(
        layer_count=model.params.layer_count,
        xi=model.params.xi,
        displacement_field_mev=model.params.displacement_field_mev,
        graphene_lattice_constant_nm=model.params.graphene_lattice_constant_nm,
        hbn_lattice_mismatch=model.params.hbn_lattice_mismatch,
        fermi_velocity_mev_nm=model.params.fermi_velocity_mev_nm,
        v3_mev_nm=model.params.v3_mev_nm,
        v4_mev_nm=model.params.v4_mev_nm,
        t1_mev=model.params.t1_mev,
        t2_mev=model.params.t2_mev,
        isp_mev=model.params.isp_mev,
        layer_spacing_nm=model.params.layer_spacing_nm,
        moire_v0_mev=0.0,
        moire_v1_mev=0.0,
        moire_phase_deg=model.params.moire_phase_deg,
    )
    bare = build_hamiltonian(k_tilde, model.lattice, no_moire_params, valley=1)
    moire = full - bare
    mask = np.ones(moire.shape, dtype=bool)
    for row_g_index in range(model.lattice.n_g):
        row_slice = model.lattice.layer_slice(row_g_index, 0, layer_count=model.params.layer_count)
        for col_g_index in range(model.lattice.n_g):
            col_slice = model.lattice.layer_slice(col_g_index, 0, layer_count=model.params.layer_count)
            mask[row_slice, col_slice] = False
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(moire[mask])))


def validate_physics(model: RLGhBNModel) -> ValidationReport:
    generic_k = model.lattice.k_m / 7.0 + model.lattice.m_m / 11.0
    hamiltonian = model.build_hamiltonian(generic_k, valley=1)
    hermiticity_residual = float(np.max(np.abs(hamiltonian - hamiltonian.conjugate().T)))

    evals_k, _ = model.diagonalize(generic_k, valley=1, n_bands=model.matrix_dim)
    evals_kprime, _ = model.diagonalize(-generic_k, valley=-1, n_bands=model.matrix_dim)
    time_reversal_residual = float(np.max(np.abs(evals_k - evals_kprime)))

    c3_k = rotate_complex(generic_k, 2.0 * np.pi / 3.0)
    evals_c3, _ = model.diagonalize(c3_k, valley=1, n_bands=model.matrix_dim)
    c3_residual = float(np.max(np.abs(evals_k - evals_c3)))

    q_norms = np.linalg.norm(model.lattice.q_vectors, axis=1)
    g_norms = np.linalg.norm(model.lattice.g_vectors_basis, axis=1)
    q_norm_residual = float(np.max(np.abs(q_norms - q_norms[0])))
    g_norm_residual = float(np.max(np.abs(g_norms - g_norms[0])))
    moire_bottom_residual = _moire_off_bottom_residual(model, generic_k)
    table_params = MOIRE_PARAMETER_TABLE[(model.params.layer_count, model.params.xi)]
    flat_valence, flat_conduction = flat_band_indices(model.lattice, model.params)

    checks = [
        ValidationCheck(
            name="parameter_table",
            status=_status_from_bool(
                np.isclose(model.params.fermi_velocity_mev_nm, DEFAULT_FERMI_VELOCITY_MEV_NM)
                and np.isclose(model.params.v3_mev_nm, DEFAULT_REMOTE_VELOCITY_MEV_NM)
                and np.isclose(model.params.v4_mev_nm, DEFAULT_REMOTE_VELOCITY_MEV_NM)
                and np.isclose(model.params.t1_mev, DEFAULT_T1_MEV)
                and np.isclose(model.params.t2_mev, DEFAULT_T2_MEV)
                and np.allclose(
                    [model.params.moire_v0_mev, model.params.moire_v1_mev, model.params.moire_phase_deg],
                    table_params,
                )
            ),
            detail="Core RLG and hBN moire parameters match the work-document Table II defaults.",
        ),
        ValidationCheck(
            name="default_g_basis_size",
            status=_status_from_bool(model.lattice.shell_count != 4 or model.lattice.n_g == 19),
            detail="The paper default shell_count=4 keeps N_G=19 reciprocal vectors.",
            value=model.lattice.n_g,
        ),
        ValidationCheck(
            name="q_c3_norms",
            status=_status_from_bool(q_norm_residual < 1.0e-12 and g_norm_residual < 1.0e-12),
            detail="The three q vectors and three moire reciprocal vectors have equal norms.",
            value=max(q_norm_residual, g_norm_residual),
        ),
        ValidationCheck(
            name="hermiticity",
            status=_status_from_bool(hermiticity_residual < 1.0e-10),
            detail="The RLG/hBN single-particle Hamiltonian is Hermitian at a generic moire momentum.",
            value=hermiticity_residual,
        ),
        ValidationCheck(
            name="moire_bottom_layer_only",
            status=_status_from_bool(moire_bottom_residual < 1.0e-12),
            detail="Subtracting the no-moire Hamiltonian leaves moire terms only in layer l=0 blocks.",
            value=moire_bottom_residual,
        ),
        ValidationCheck(
            name="time_reversal",
            status=_status_from_bool(time_reversal_residual < 1.0e-10),
            detail="The K and Kprime spectra satisfy E_K(k)=E_Kprime(-k).",
            value=time_reversal_residual,
        ),
        ValidationCheck(
            name="c3_spectrum",
            status=_status_from_bool(c3_residual < 1.0e-8),
            detail="The finite reciprocal-vector shell preserves the single-valley C3 spectrum.",
            value=c3_residual,
        ),
        ValidationCheck(
            name="flat_band_indices",
            status=_status_from_bool(flat_valence == model.params.layer_count * model.lattice.n_g - 1 and flat_conduction == flat_valence + 1),
            detail="The central valence and conduction bands use the L*N_G valence-count convention.",
            value=f"{flat_valence},{flat_conduction}",
        ),
    ]

    return ValidationReport(title="RLG/hBN validation", checks=tuple(checks))


def reproduce_paper_checkpoints(model: RLGhBNModel) -> ValidationReport:
    checks = list(validate_physics(model).checks)
    checks.append(
        ValidationCheck(
            name="paper_checkpoints",
            status="skipped",
            detail=(
                "CP1-CP9 require external paper overlays or finite-grid topology scans. "
                "Use bands_along_standard_path, topology_on_grid, and valence_charge_background for those runs."
            ),
        )
    )
    return ValidationReport(title="RLG/hBN paper checkpoints", checks=tuple(checks))


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "reproduce_paper_checkpoints",
    "validate_physics",
]
