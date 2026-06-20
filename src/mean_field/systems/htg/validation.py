from __future__ import annotations

import math

import numpy as np

from mean_field.core.validation import ValidationCheck, ValidationReport, status_from_bool

from .hamiltonian import build_hamiltonian, moire_coupling_matrix
from .lattice import HTGLattice, dot_2d
from .params import HTGParams
from .mean_field_adapter import (
    HTGHartreeFockState,
    hermitian_residual,
    htg_filling_from_density,
    htg_gap_estimate,
    htg_occupied_state_count,
    projector_idempotency_residual,
)



def _detail(value: float | str, tolerance: float | None = None) -> str:
    detail = f"value={value}"
    if tolerance is not None:
        detail += f", tolerance={tolerance}"
    return detail


def _check(name: str, condition: bool, value: float | str, tolerance: float | None = None) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=status_from_bool(condition),
        detail=_detail(value, tolerance),
        value=value,
    )


def htg_validation_report(title: str, checks: tuple[ValidationCheck, ...]) -> ValidationReport:
    """Build the shared validation report for HTG checks."""

    return ValidationReport(title=str(title), checks=tuple(checks))


def validate_lattice(lattice: HTGLattice, *, atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    q_norms = np.asarray([abs(q) for q in lattice.q_vectors], dtype=float)
    q_angle_dot = dot_2d(lattice.q0, lattice.q1) / (abs(lattice.q0) * abs(lattice.q1))
    b_angle_dot = dot_2d(lattice.b_m1, lattice.b_m2) / (abs(lattice.b_m1) * abs(lattice.b_m2))
    l_m_from_b = 4.0 * math.pi / (math.sqrt(3.0) * abs(lattice.b_m1))
    has_zero_g = any(abs(complex(value)) <= atol for value in lattice.g_vectors)
    q_norm_error = float(np.max(np.abs(q_norms - lattice.k_theta)))
    q_sum_error = float(abs(np.sum(lattice.q_vectors)))
    b_norm_error = float(abs(abs(lattice.b_m1) - abs(lattice.b_m2)))
    l_m_error = float(abs(lattice.l_m - l_m_from_b))
    kappa_m_error = float(abs(lattice.kappa_m + lattice.q0))
    kappa_prime_error = float(abs(lattice.kappa_prime_m - lattice.q0))
    return (
        _check("q_norm_equal", q_norm_error < atol, q_norm_error, atol),
        _check("q_sum_zero", q_sum_error < atol, q_sum_error, atol),
        _check("q_angle_120deg", abs(q_angle_dot + 0.5) < atol, float(q_angle_dot), atol),
        _check("b_norm_equal", b_norm_error < atol, b_norm_error, atol),
        _check("b_angle_60deg", abs(b_angle_dot - 0.5) < atol, float(b_angle_dot), atol),
        _check("l_m_consistent", l_m_error < atol, l_m_error, atol),
        _check("g_contains_zero", has_zero_g, "yes" if has_zero_g else "no"),
        _check("kappa_folds_layer1", kappa_m_error < atol, kappa_m_error, atol),
        _check("kappa_prime_folds_layer3", kappa_prime_error < atol, kappa_prime_error, atol),
    )


def validate_static_hamiltonian(
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    atol: float = 1.0e-10,
) -> tuple[ValidationCheck, ...]:
    hmat = build_hamiltonian(lattice.gamma_m, lattice, params, valley=valley)
    hermitian_residual_value = float(np.max(np.abs(hmat - hmat.conjugate().T)))
    t0 = moire_coupling_matrix(0, params, valley=valley)
    t1 = moire_coupling_matrix(1, params, valley=valley)
    t2 = moire_coupling_matrix(2, params, valley=valley)
    t0_imag = float(np.max(np.abs(t0.imag)))
    t12_residual = float(np.max(np.abs(t1 - t2.conjugate())))
    return (
        _check("hamiltonian_hermitian_gamma", hermitian_residual_value < atol, hermitian_residual_value, atol),
        _check("t0_real", t0_imag < atol, t0_imag, atol),
        _check("t1_equals_t2_conjugate", t12_residual < atol, t12_residual, atol),
    )


def validate_hf_state(
    state: HTGHartreeFockState,
    *,
    hermitian_atol: float = 1.0e-10,
    projector_atol: float = 1.0e-8,
    filling_atol: float = 1.0e-9,
) -> tuple[ValidationCheck, ...]:
    """Validate the system-independent hard constraints for an HTG HF state."""

    h_residual = hermitian_residual(state.hamiltonian)
    p_residual = projector_idempotency_residual(state.density, n_spin=state.n_spin, n_eta=state.n_eta)
    filling = htg_filling_from_density(state.density, n_spin=state.n_spin, n_eta=state.n_eta)
    filling_error = abs(filling - float(state.nu))
    gap_ev = htg_gap_estimate(state.energies, state.nu)
    occupied_count = htg_occupied_state_count(state.nu, state.nt, state.nk, n_spin=state.n_spin, n_eta=state.n_eta)
    if occupied_count <= 0 or occupied_count >= state.nt * state.nk:
        gap_check = ValidationCheck(
            name="hf_gap_ev",
            status="pass",
            detail="No internal HF gap is defined at full or empty projected-space filling.",
            value="not_applicable_full_or_empty_projected_space",
        )
    else:
        gap_value: float | str = gap_ev if np.isfinite(gap_ev) else "nan"
        gap_check = _check("hf_gap_ev", bool(np.isfinite(gap_ev)), gap_value)
    return (
        _check("hf_hamiltonian_hermitian", h_residual < hermitian_atol, h_residual, hermitian_atol),
        _check("hf_projector_idempotent", p_residual < projector_atol, p_residual, projector_atol),
        _check("hf_filling", filling_error < filling_atol, filling_error, filling_atol),
        gap_check,
    )
