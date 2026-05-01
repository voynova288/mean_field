from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bilayer_map import analytic_singular_values, build_W_matrix, svd_decompose
from .model import ATMGModel
from .params import ATMGParameters
from .tbg import build_tbg_hamiltonian


ValidationStatus = Literal["pass", "fail", "skipped"]


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: ValidationStatus
    detail: str
    value: float | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class ValidationReport:
    title: str
    checks: tuple[ValidationCheck, ...]

    @property
    def failure_count(self) -> int:
        return sum(check.status == "fail" for check in self.checks)

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for check in self.checks:
            value = "" if check.value is None else f" ({check.value:.6e})"
            lines.append(f"- [{check.status}] {check.name}{value}: {check.detail}")
        lines.append("")
        lines.append(f"- failures: {self.failure_count}")
        return "\n".join(lines)


def _status(condition: bool) -> ValidationStatus:
    return "pass" if condition else "fail"


def _generic_k(model: ATMGModel) -> complex:
    return complex(model.lattice.k_m / 5.0 + model.lattice.m_m / 7.0)


def validate_physics(
    model: ATMGModel,
    *,
    hermitian_atol: float = 1.0e-10,
    spectrum_atol: float = 1.0e-10,
    singular_value_atol: float = 1.0e-12,
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    k_tilde = _generic_k(model)

    hamiltonian = model.build_hamiltonian(k_tilde, valley=1)
    hermitian_residual = float(np.max(np.abs(hamiltonian - hamiltonian.conjugate().T)))
    checks.append(
        ValidationCheck(
            name="Hermiticity",
            status=_status(hermitian_residual < hermitian_atol),
            detail="Full ATMG Hamiltonian is Hermitian at a generic mBZ momentum.",
            value=hermitian_residual,
        )
    )

    evals_k, _ = model.diagonalize(k_tilde, valley=1)
    evals_kprime, _ = model.diagonalize(-k_tilde, valley=-1)
    time_reversal_residual = float(np.max(np.abs(evals_k - evals_kprime)))
    checks.append(
        ValidationCheck(
            name="Time Reversal",
            status=_status(time_reversal_residual < spectrum_atol),
            detail="E_K(k) matches E_K'(-k) within tolerance.",
            value=time_reversal_residual,
        )
    )

    mapped = model.mapped_spectrum(k_tilde, valley=1)
    mapping_residual = float(np.max(np.abs(evals_k - mapped.combined_energies)))
    checks.append(
        ValidationCheck(
            name="SVD Mapping",
            status=_status(mapping_residual < spectrum_atol),
            detail="Direct ATMG spectrum matches the TBG-sum mapping.",
            value=mapping_residual,
        )
    )

    if model.params.is_uniform:
        svd_result = svd_decompose(build_W_matrix(model.params.n_layers, model.params.alpha))
        analytic = analytic_singular_values(model.params.n_layers, model.params.alpha)
        sv_residual = float(np.max(np.abs(svd_result.singular_values - analytic))) if analytic.size else 0.0
        checks.append(
            ValidationCheck(
                name="Singular Values",
                status=_status(sv_residual < singular_value_atol),
                detail="Numerical singular values match the analytic ATMG formula.",
                value=sv_residual,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="Singular Values",
                status="skipped",
                detail="Analytic formula only applies to uniform interface couplings.",
                value=None,
            )
        )

    if model.params.n_layers == 2:
        tbg_matrix = build_tbg_hamiltonian(
            k_tilde,
            model.lattice,
            lambda_coupling=model.params.alpha,
            kappa=model.params.kappa,
            vf=model.params.vf,
            valley=1,
        )
        reduction_residual = float(np.max(np.abs(hamiltonian - tbg_matrix)))
        checks.append(
            ValidationCheck(
                name="n=2 Reduction",
                status=_status(reduction_residual < singular_value_atol),
                detail="Two-layer ATMG reduces exactly to the reference TBG builder.",
                value=reduction_residual,
            )
        )

    if model.params.kappa == 0.0:
        particle_hole_residual = float(np.max(np.abs(evals_k + evals_k[::-1])))
        checks.append(
            ValidationCheck(
                name="Particle Hole",
                status=_status(particle_hole_residual < spectrum_atol),
                detail="The chiral spectrum is symmetric about zero energy.",
                value=particle_hole_residual,
            )
        )

    if model.params.n_layers % 2 == 1:
        evals_gamma, _ = model.diagonalize(model.lattice.gamma_m, valley=1)
        dirac_residual = float(np.min(np.abs(evals_gamma)))
        checks.append(
            ValidationCheck(
                name="Odd-Layer Dirac Cone",
                status=_status(dirac_residual < spectrum_atol),
                detail="Odd-layer ATMG retains the decoupled monolayer Dirac crossing at the moire-zone origin in this gauge.",
                value=dirac_residual,
            )
        )

    return ValidationReport(
        title=f"ATMG Validation (n={model.params.n_layers}, theta={model.theta_deg:.3f} deg)",
        checks=tuple(checks),
    )


def reproduce_khalaf_checkpoints(
    *,
    n_shells: int = 1,
) -> tuple[ValidationReport, ...]:
    cases = (
        ATMGParameters.chiral(2, 1.05),
        ATMGParameters.realistic(2, 1.05, kappa=0.8),
        ATMGParameters.chiral(3, 1.53),
        ATMGParameters.chiral(4, 1.75),
    )
    reports: list[ValidationReport] = []
    for params in cases:
        model = ATMGModel.from_config(
            params.n_layers,
            params.theta_deg,
            n_shells=n_shells,
            params=params,
        )
        reports.append(validate_physics(model))
    return tuple(reports)


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "reproduce_khalaf_checkpoints",
    "validate_physics",
]
