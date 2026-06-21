from __future__ import annotations

import numpy as np

from ...core.validation import ValidationCheck, ValidationReport, make_validation_check
from .hamiltonian import build_hamiltonian, hamiltonian_dimension
from .model import RLGhBNModel


def _finite(name: str, value: object) -> ValidationCheck:
    arr = np.asarray(value)
    return make_validation_check(name, bool(np.all(np.isfinite(arr))), int(arr.size), detail=f"finite entries={arr.size}")


def validate_physics(model: RLGhBNModel) -> ValidationReport:
    """Cheap RLG/hBN structural validation.

    Historical paper checkpoint validation was retired to ignored local archives.
    This tracked surface intentionally avoids grids, HF, topology, or response
    recomputation and only checks model-shape and Γ-point Hamiltonian invariants.
    """

    lattice, params = model.lattice, model.params
    hamiltonian = build_hamiltonian(0.0 + 0.0j, lattice, params, valley=model.xi)
    expected_dim = hamiltonian_dimension(lattice, params)
    residual = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))) if hamiltonian.size else 0.0
    checks = (
        make_validation_check("matrix dimension positive", expected_dim > 0, expected_dim),
        make_validation_check("Hamiltonian shape", hamiltonian.shape == (expected_dim, expected_dim), str(hamiltonian.shape)),
        make_validation_check("Hamiltonian Hermitian", residual <= 1.0e-10, residual, tolerance=1.0e-10),
        make_validation_check("layer count positive", int(params.layer_count) > 0, int(params.layer_count)),
        make_validation_check("g-vector count positive", int(lattice.n_g) > 0, int(lattice.n_g)),
        _finite("g-vectors finite", lattice.g_vectors),
    )
    return ValidationReport(title="RLG/hBN validation", checks=checks)


def reproduce_paper_checkpoints(model: RLGhBNModel) -> ValidationReport:
    checks = list(validate_physics(model).checks)
    checks.append(
        ValidationCheck(
            "paper checkpoints retired",
            "skipped",
            "Archived paper-checkpoint validation is kept under ignored local_archive; tracked validation is smoke-only.",
            val="retired_surface",
        )
    )
    return ValidationReport(title="RLG/hBN paper checkpoints", checks=tuple(checks))


__all__ = ["ValidationCheck", "ValidationReport", "reproduce_paper_checkpoints", "validate_physics"]
