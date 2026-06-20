from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ...core.validation import ValidationCheck, ValidationReport, status_from_bool
from .chiral import chiral_symmetry_residual
from .domains import all_domains, domain_displacements
from .hamiltonian import build_coupling_table, build_hamiltonian, layer_k_offset, moire_coupling_matrix
from .lattice import HTQGLattice, dot_2d, rotate_complex
from .params import HTQGParams
from .symmetry import validate_internal_unitarity


def _check(name: str, condition: bool, value: float | int | str | None, detail: str, tolerance: float | None = None) -> ValidationCheck:
    return ValidationCheck(name=name, status=status_from_bool(condition), value=value, detail=detail, tolerance=tolerance)


def _g_closure_residual(lattice: HTQGLattice, angle_rad: float) -> float:
    points = np.asarray(lattice.g_vectors, dtype=np.complex128)
    max_distance = 0.0
    for gvec in points:
        rotated = rotate_complex(complex(gvec), angle_rad)
        distance = float(np.min(np.abs(points - rotated)))
        max_distance = max(max_distance, distance)
    return max_distance


def validate_lattice(lattice: HTQGLattice, *, atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    q_norms = np.asarray([abs(q) for q in lattice.q_vectors], dtype=float)
    q_angle_dot = dot_2d(lattice.q0, lattice.q1) / (abs(lattice.q0) * abs(lattice.q1))
    b_angle_dot = dot_2d(lattice.b_m1, lattice.b_m2) / (abs(lattice.b_m1) * abs(lattice.b_m2))
    expected_ng = 1 + 3 * lattice.n_shells * (lattice.n_shells + 1)
    has_zero_g = any(abs(complex(value)) <= atol for value in lattice.g_vectors)
    c3_residual = _g_closure_residual(lattice, 2.0 * math.pi / 3.0)
    c3m_residual = _g_closure_residual(lattice, -2.0 * math.pi / 3.0)
    return (
        _check("q_norm_equal", bool(np.max(np.abs(q_norms - lattice.k_theta)) < atol), float(np.max(np.abs(q_norms - lattice.k_theta))), "|q_n| = k_theta", atol),
        _check("q_sum_zero", bool(abs(np.sum(lattice.q_vectors)) < atol), float(abs(np.sum(lattice.q_vectors))), "sum_n q_n = 0", atol),
        _check("q_angle_120deg", bool(abs(q_angle_dot + 0.5) < atol), float(q_angle_dot), "q0 dot q1 / |q|^2 = -1/2", atol),
        _check("b_angle_60deg", bool(abs(b_angle_dot - 0.5) < atol), float(b_angle_dot), "b_M1 dot b_M2 / |b|^2 = +1/2", atol),
        _check("three_q0_is_reciprocal", bool(abs(3.0 * lattice.q0 + lattice.b_m1 + lattice.b_m2) < atol), float(abs(3.0 * lattice.q0 + lattice.b_m1 + lattice.b_m2)), "3 q0 = -(b1+b2)", atol),
        _check("kappap_class_is_minus_q0", bool(abs(lattice.kappap_class + lattice.q0) < atol), float(abs(lattice.kappap_class + lattice.q0)), "folding class kappa' = -q0", atol),
        _check("g_count_hex_shell", lattice.n_g == expected_ng, lattice.n_g, f"hex shell N_G should be {expected_ng}"),
        _check("g_contains_zero", bool(has_zero_g), "yes" if has_zero_g else "no", "G=0 belongs to cutoff"),
        _check("g_c3_closed_plus", c3_residual < atol, c3_residual, "G shell closed under +C3", atol),
        _check("g_c3_closed_minus", c3m_residual < atol, c3m_residual, "G shell closed under -C3", atol),
    )


def validate_params(params: HTQGParams) -> tuple[ValidationCheck, ...]:
    target = 0.553
    residual = abs(params.vf_ev_nm - target)
    return (
        _check("hbar_v_paper_scale", residual < 0.01 * target, float(params.vf_ev_nm), "hbar v ≈ 0.553 eV nm for v=8.4e5 m/s", 0.01 * target),
        _check("mdt_units_nm", abs(params.lambda_mdt_nm) < 1.0, float(params.lambda_mdt_nm), "lambda_MDT should be in nm, Kwan-Tan-Devakul value -0.23 nm"),
    )


def validate_domains(lattice: HTQGLattice, *, atol: float = 1.0e-12) -> tuple[ValidationCheck, ...]:
    aba = domain_displacements(lattice, "alpha_beta_alpha")
    bab = domain_displacements(lattice, "beta_alpha_beta")
    abg = domain_displacements(lattice, "alpha_beta_gamma")
    gba = domain_displacements(lattice, "gamma_beta_alpha")
    return (
        _check("alpha_beta_alpha_partner", abs(aba.d12 + bab.d12) < atol and abs(aba.d34 + bab.d34) < atol, float(abs(aba.d12 + bab.d12) + abs(aba.d34 + bab.d34)), "βαβ is d -> -d partner of αβα", atol),
        _check("alpha_beta_gamma_partner", abs(abg.d12 + gba.d12) < atol and abs(abg.d34 + gba.d34) < atol, float(abs(abg.d12 + gba.d12) + abs(abg.d34 + gba.d34)), "γβα is d -> -d partner of αβγ", atol),
        _check("type_i_displacements", abs(aba.d12 - lattice.d_ba) < atol and abs(aba.d34 + lattice.d_ba) < atol, aba.label, "αβα=(d_BA,-d_BA)", atol),
        _check("type_ii_displacements", abs(abg.d12 - lattice.d_ba) < atol and abs(abg.d34 - lattice.d_ba) < atol, abg.label, "αβγ=(d_BA,d_BA)", atol),
    )


def validate_hamiltonian_static(
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str = "alpha_beta_alpha",
    atol: float = 1.0e-10,
) -> tuple[ValidationCheck, ...]:
    hmat = build_hamiltonian(lattice.gamma, lattice, params, domain=domain, valley=1)
    hermitian_residual = float(np.max(np.abs(hmat - hmat.conjugate().T)))
    coupling_entries = build_coupling_table(lattice.g_vectors, lattice.q_vectors)
    t0 = moire_coupling_matrix(0, params)
    t1 = moire_coupling_matrix(1, params)
    t2 = moire_coupling_matrix(2, params)
    t0_imag = float(np.max(np.abs(t0.imag)))
    t12_residual = float(np.max(np.abs(t1 - t2.conjugate())))
    return (
        _check("hamiltonian_dimension", hmat.shape == (lattice.matrix_dim, lattice.matrix_dim), str(hmat.shape), "dim = 8*N_G"),
        _check("hamiltonian_hermitian_gamma", hermitian_residual < atol, hermitian_residual, "H=H† at Gamma", atol),
        _check("coupling_table_nonempty", len(coupling_entries) > 0, len(coupling_entries), "G_target=G_source+q_j-q0 edges exist"),
        _check("t0_real", t0_imag < atol, t0_imag, "T0 is real in the K-valley convention", atol),
        _check("t1_equals_t2_conjugate", t12_residual < atol, t12_residual, "T1 = T2*", atol),
    )


def validate_decoupled_dirac_limit(
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    k_tilde: complex | None = None,
    atol: float = 1.0e-10,
) -> tuple[ValidationCheck, ...]:
    kval = lattice.gamma if k_tilde is None else complex(k_tilde)
    zero_w = replace(params, w_ev=0.0, lambda_mdt_nm=0.0)
    hmat = build_hamiltonian(kval, lattice, zero_w, domain="alpha_beta_alpha", valley=1)
    evals = np.linalg.eigvalsh(hmat)
    expected: list[float] = []
    for gvec in lattice.g_vectors:
        for layer in (1, 2, 3, 4):
            momentum = complex(kval + gvec - layer_k_offset(lattice, layer))
            energy = zero_w.vf_ev_nm * abs(momentum)
            expected.extend([-energy, energy])
    residual = float(np.max(np.abs(np.sort(evals) - np.sort(np.asarray(expected, dtype=float)))))
    return (_check("w_zero_four_fold_dirac_limit", residual < atol, residual, "w->0 spectrum matches four folded monolayer Dirac cones", atol),)


def validate_time_reversal(
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    k_tilde: complex | None = None,
    domain: str = "alpha_beta_alpha",
    atol: float = 1.0e-10,
) -> tuple[ValidationCheck, ...]:
    kval = lattice.kappa_path if k_tilde is None else complex(k_tilde)
    evals_k = np.linalg.eigvalsh(build_hamiltonian(kval, lattice, params, domain=domain, valley=1))
    evals_kp = np.linalg.eigvalsh(build_hamiltonian(-kval, lattice, params, domain=domain, valley=-1))
    residual = float(np.max(np.abs(evals_k - evals_kp)))
    return (_check("time_reversal_spectrum", residual < atol, residual, "E_K(k)=E_K'(-k)", atol),)


def validate_chiral_limit(
    lattice: HTQGLattice,
    *,
    domain: str = "alpha_beta_alpha",
    atol: float = 1.0e-10,
) -> tuple[ValidationCheck, ...]:
    residual = chiral_symmetry_residual(lattice.gamma, lattice, HTQGParams.chiral(), domain=domain)
    return (_check("chiral_anticommutator", residual < atol, residual, "{H, sigma_z}=0 for kappa=0, no MDT/rotation", atol),)


def validate_internal_symmetry_matrices(*, domain: str = "alpha_beta_alpha", atol: float = 1.0e-12) -> tuple[ValidationCheck, ...]:
    residuals = validate_internal_unitarity(domain, atol=atol)
    checks = [
        _check(f"{name}_unitary_internal", residual < atol, residual, "Appendix-C internal matrix is unitary", atol)
        for name, residual in residuals.items()
    ]
    checks.append(
        ValidationCheck(
            name="gate_A_full_plane_wave_symmetry",
            status="skipped",
            value="not_run",
            detail="Full Gate-A symmetry requires momentum/G permutation residuals and is not claimed by internal unitarity checks.",
            tolerance=None,
        )
    )
    return tuple(checks)


def run_lightweight_validation(
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str = "alpha_beta_alpha",
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    checks.extend(validate_lattice(lattice))
    checks.extend(validate_params(params))
    checks.extend(validate_domains(lattice))
    checks.extend(validate_hamiltonian_static(lattice, params, domain=domain))
    checks.extend(validate_decoupled_dirac_limit(lattice, params))
    checks.extend(validate_time_reversal(lattice, params, domain=domain))
    checks.extend(validate_chiral_limit(lattice, domain=domain))
    checks.extend(validate_internal_symmetry_matrices(domain=domain))
    return ValidationReport(title=f"HTQG lightweight validation ({domain})", checks=tuple(checks))


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "run_lightweight_validation",
    "validate_chiral_limit",
    "validate_decoupled_dirac_limit",
    "validate_domains",
    "validate_hamiltonian_static",
    "validate_internal_symmetry_matrices",
    "validate_lattice",
    "validate_params",
    "validate_time_reversal",
]
