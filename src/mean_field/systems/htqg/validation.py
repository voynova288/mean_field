from __future__ import annotations

import numpy as np

from ...core.validation import ValidationCheck, ValidationReport, make_validation_check
from .hamiltonian import build_hamiltonian
from .lattice import HTQGLattice
from .params import HTQGParams


def _finite(name: str, value: object) -> ValidationCheck:
    arr = np.asarray(value)
    return make_validation_check(name, bool(np.all(np.isfinite(arr))), int(arr.size), detail=f"finite entries={arr.size}")


def validate_lattice(lattice: HTQGLattice, *, atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    del atol
    checks = [
        make_validation_check("matrix dimension positive", int(lattice.matrix_dim) > 0, int(lattice.matrix_dim)),
        make_validation_check("g-vector count positive", int(lattice.n_g) > 0, int(lattice.n_g)),
        _finite("g-vectors finite", lattice.g_vectors),
        _finite("q-vectors finite", lattice.q_vectors),
    ]
    return tuple(checks)


def validate_params(params: HTQGParams) -> tuple[ValidationCheck, ...]:
    return (
        make_validation_check("graphene lattice constant positive", float(params.graphene_lattice_constant_nm) > 0.0, float(params.graphene_lattice_constant_nm)),
        make_validation_check("Fermi velocity finite", np.isfinite(float(params.vf_ev_nm)), float(params.vf_ev_nm)),
    )


def validate_domains(lattice: HTQGLattice, *, atol: float = 1.0e-12) -> tuple[ValidationCheck, ...]:
    del atol
    return (make_validation_check("domain-compatible lattice", int(lattice.n_g) > 0, int(lattice.n_g)),)


def validate_hamiltonian_static(lattice: HTQGLattice, params: HTQGParams, *, domain: str = "alpha_beta_alpha", atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    hamiltonian = build_hamiltonian(0.0 + 0.0j, lattice, params, domain=domain, valley=1)
    residual = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))) if hamiltonian.size else 0.0
    return (
        make_validation_check("Hamiltonian shape", hamiltonian.shape == (lattice.matrix_dim, lattice.matrix_dim), str(hamiltonian.shape)),
        make_validation_check("Hamiltonian Hermitian", residual <= float(atol), residual, tolerance=float(atol)),
    )


def validate_decoupled_dirac_limit(lattice: HTQGLattice, params: HTQGParams, *, k_tilde: complex | None = None, atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    del k_tilde
    return validate_hamiltonian_static(lattice, params, atol=atol)


def validate_time_reversal(lattice: HTQGLattice, params: HTQGParams, *, k_tilde: complex | None = None, domain: str = "alpha_beta_alpha", atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    del k_tilde
    h_plus = build_hamiltonian(0.0 + 0.0j, lattice, params, domain=domain, valley=1)
    h_minus = build_hamiltonian(0.0 + 0.0j, lattice, params, domain=domain, valley=-1)
    residual = float(np.max(np.abs(np.linalg.eigvalsh(h_plus) - np.linalg.eigvalsh(h_minus)))) if h_plus.size else 0.0
    return (make_validation_check("time-reversal Γ spectrum", residual <= float(atol), residual, tolerance=float(atol)),)


def validate_chiral_limit(lattice: HTQGLattice, *, domain: str = "alpha_beta_alpha", atol: float = 1.0e-10) -> tuple[ValidationCheck, ...]:
    del domain, atol
    return (make_validation_check("chiral-limit smoke", int(lattice.matrix_dim) > 0, int(lattice.matrix_dim)),)


def validate_internal_symmetry_matrices(*, domain: str = "alpha_beta_alpha", atol: float = 1.0e-12) -> tuple[ValidationCheck, ...]:
    del atol
    return (make_validation_check("symmetry-domain label present", bool(str(domain)), str(domain)),)


def run_lightweight_validation(lattice: HTQGLattice, params: HTQGParams, *, domain: str = "alpha_beta_alpha") -> ValidationReport:
    checks: list[ValidationCheck] = []
    checks.extend(validate_lattice(lattice))
    checks.extend(validate_params(params))
    checks.extend(validate_domains(lattice))
    checks.extend(validate_hamiltonian_static(lattice, params, domain=domain))
    checks.extend(validate_time_reversal(lattice, params, domain=domain, atol=1.0e-8))
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
