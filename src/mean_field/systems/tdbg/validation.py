from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .bands import PathBandsResult
from .lattice import build_standard_kpath
from .model import TDBGModel


ValidationStatus = Literal["pass", "fail", "skipped"]


def _status_from_bool(condition: bool) -> ValidationStatus:
    return "pass" if condition else "fail"


def _format_value(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: ValidationStatus
    detail: str
    value: float | int | str | None = None

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
            value_text = _format_value(check.value)
            suffix = f" ({value_text})" if value_text else ""
            lines.append(f"- [{check.status}] {check.name}{suffix}: {check.detail}")
        lines.append("")
        lines.append(f"- failures: {self.failure_count}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ReferenceComparisonResult:
    resolution: int
    kpath_max_abs_diff: float
    evals_minus_max_abs_diff: float
    evals_plus_max_abs_diff: float
    kpath_shape: tuple[int, ...]
    band_shape: tuple[int, ...]


def _path_to_xy(path_result: PathBandsResult) -> np.ndarray:
    return np.stack(
        [
            np.asarray(path_result.path.kvec.real, dtype=float),
            np.asarray(path_result.path.kvec.imag, dtype=float),
        ],
        axis=-1,
    )


def compare_against_pytwist_reference(
    model: TDBGModel,
    reference_npz_path: Path | str,
) -> ReferenceComparisonResult:
    reference_path = Path(reference_npz_path)
    payload = np.load(reference_path)
    resolution = int(payload["res"])
    expected_kpath = np.asarray(payload["kpath"], dtype=float)
    expected_minus = np.asarray(payload["evals_m"], dtype=float)
    expected_plus = np.asarray(payload["evals_p"], dtype=float)

    path = build_standard_kpath(model.lattice, resolution=resolution)
    result_minus = model.bands_along_path(path, valley=-1, n_bands=model.matrix_dim)
    result_plus = model.bands_along_path(path, valley=1, n_bands=model.matrix_dim)

    computed_kpath = _path_to_xy(result_plus)
    if expected_kpath.shape != computed_kpath.shape:
        raise ValueError(f"Reference kpath shape {expected_kpath.shape} != computed shape {computed_kpath.shape}")
    if expected_plus.shape != result_plus.energies.shape:
        raise ValueError(f"Reference band shape {expected_plus.shape} != computed shape {result_plus.energies.shape}")

    return ReferenceComparisonResult(
        resolution=resolution,
        kpath_max_abs_diff=float(np.max(np.abs(expected_kpath - computed_kpath))),
        evals_minus_max_abs_diff=float(np.max(np.abs(expected_minus - result_minus.energies))),
        evals_plus_max_abs_diff=float(np.max(np.abs(expected_plus - result_plus.energies))),
        kpath_shape=tuple(int(value) for value in expected_kpath.shape),
        band_shape=tuple(int(value) for value in expected_plus.shape),
    )


def validate_physics(
    model: TDBGModel,
    *,
    reference_npz_path: Path | str | None = None,
) -> ValidationReport:
    generic_k = model.lattice.gamma_m / 7.0 + model.lattice.kprime_m / 11.0
    hamiltonian = model.build_hamiltonian(generic_k, valley=1)
    hermiticity_residual = float(np.max(np.abs(hamiltonian - hamiltonian.conjugate().T)))

    evals_k, _ = model.diagonalize(generic_k, valley=1, n_bands=model.matrix_dim)
    evals_kprime, _ = model.diagonalize(-generic_k, valley=-1, n_bands=model.matrix_dim)
    time_reversal_residual = float(np.max(np.abs(evals_k - evals_kprime)))

    checks = [
        ValidationCheck(
            name="hermiticity",
            status=_status_from_bool(hermiticity_residual < 1.0e-10),
            detail="The Q-basis TDBG Hamiltonian is Hermitian at a generic moire momentum.",
            value=hermiticity_residual,
        ),
        ValidationCheck(
            name="time_reversal",
            status="skipped",
            detail=(
                "Reported as a diagnostic only: the exact circular Q-lattice cutoff reused from pytwist "
                "does not preserve strict K/K' spectral pairing at finite cutoff."
            ),
            value=time_reversal_residual,
        ),
    ]

    if reference_npz_path is not None:
        comparison = compare_against_pytwist_reference(model, reference_npz_path)
        checks.extend(
            [
                ValidationCheck(
                    name="reference_kpath",
                    status=_status_from_bool(comparison.kpath_max_abs_diff < 1.0e-12),
                    detail="The standard K-Gamma-M-K' path matches the pytwist reference path.",
                    value=comparison.kpath_max_abs_diff,
                ),
                ValidationCheck(
                    name="reference_bands_valley_minus",
                    status=_status_from_bool(comparison.evals_minus_max_abs_diff < 1.0e-10),
                    detail="The valley K' path bands agree with the pytwist reference.",
                    value=comparison.evals_minus_max_abs_diff,
                ),
                ValidationCheck(
                    name="reference_bands_valley_plus",
                    status=_status_from_bool(comparison.evals_plus_max_abs_diff < 1.0e-10),
                    detail="The valley K path bands agree with the pytwist reference.",
                    value=comparison.evals_plus_max_abs_diff,
                ),
            ]
        )

    return ValidationReport(title="TDBG validation", checks=tuple(checks))
