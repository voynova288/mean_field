from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mean_field.core.contracts import (
    DensityState,
    HFRunResult,
    HFState,
    HamiltonianParts,
    ProjectedBasis,
    ReferenceDensity,
    SingleParticleModel,
)
from mean_field.core.hf import (
    TDHFCanonicalOrbitals,
    canonical_tdhf_orbitals_from_hf_run_result,
    canonical_tdhf_orbitals_from_hf_state,
)


def _toy_model() -> SingleParticleModel:
    def hamiltonian_builder(kvec: np.ndarray) -> np.ndarray:
        kvec = np.asarray(kvec).reshape(-1)
        out = np.zeros((2, 2, kvec.size), dtype=np.complex128)
        out[0, 0, :] = -1.0
        out[1, 1, :] = 1.0
        return out

    def diagonalizer(kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hamiltonian = hamiltonian_builder(kvec)
        energies = np.zeros((2, hamiltonian.shape[2]), dtype=float)
        vectors = np.zeros((2, 2, hamiltonian.shape[2]), dtype=np.complex128)
        for ik in range(hamiltonian.shape[2]):
            energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(hamiltonian[:, :, ik])
        return energies, vectors

    return SingleParticleModel(
        system="toy",
        lattice=None,
        params={},
        hamiltonian_builder=hamiltonian_builder,
        diagonalizer=diagonalizer,
    )


def _toy_hf_state(
    hamiltonian: np.ndarray,
    projector: np.ndarray,
    *,
    k_grid_frac: np.ndarray | None = None,
    mu: float = np.nan,
    flavor_labels: tuple[str, ...] = ("a", "b"),
) -> HFState:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    projector = np.asarray(projector, dtype=np.complex128)
    nt, _nt_rhs, nk = hamiltonian.shape
    if k_grid_frac is None:
        k_grid_frac = np.column_stack((np.arange(nk, dtype=float) / float(nk), np.zeros(nk)))

    reference = ReferenceDensity(scheme="custom", reference=np.zeros_like(projector))
    density = DensityState(
        density_delta=projector,
        reference=reference,
        filling=0.0,
        n_occupied_total=int(round(float(np.trace(projector, axis1=0, axis2=1).real.sum()))),
    )
    parts = HamiltonianParts(
        h0=hamiltonian,
        fixed=np.zeros_like(hamiltonian),
        hartree=np.zeros_like(hamiltonian),
        fock=np.zeros_like(hamiltonian),
        total=hamiltonian,
        density_input_convention="delta",
    )
    model = _toy_model()
    basis = ProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.arange(nk, dtype=np.complex128),
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        h0=hamiltonian,
        basis_energies=np.zeros((nt, nk), dtype=float),
        active_band_indices=tuple(range(nt)),
        active_valence_bands=1,
        active_conduction_bands=max(0, nt - 1),
        micro_wavefunctions=np.ones((nk, nt, nt), dtype=np.complex128),
        flavor_labels=flavor_labels,
    )
    return HFState(
        basis=basis,
        density=density,
        hamiltonian=parts,
        energies=np.zeros((nt, nk), dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(mu),
    )


def test_canonical_tdhf_orbitals_diagonalize_hf_state_and_flatten_fortran_order() -> None:
    hamiltonian = np.zeros((2, 2, 2), dtype=np.complex128)
    expected_energies = np.asarray([[-1.0, 0.25], [2.0, 3.0]], dtype=float)
    for ik in range(2):
        hamiltonian[:, :, ik] = np.diag(expected_energies[:, ik])
    projector = np.zeros_like(hamiltonian)
    projector[0, 0, :] = 1.0

    state = _toy_hf_state(hamiltonian, projector, mu=0.125)
    orbitals = canonical_tdhf_orbitals_from_hf_state(state)

    assert isinstance(orbitals, TDHFCanonicalOrbitals)
    np.testing.assert_allclose(orbitals.energies, expected_energies)
    np.testing.assert_allclose(orbitals.global_energies, expected_energies.reshape(-1, order="F"))
    np.testing.assert_array_equal(orbitals.occupied_mask, np.asarray([[True, True], [False, False]]))
    np.testing.assert_array_equal(orbitals.occupied_global_indices, np.asarray([0, 2]))
    assert orbitals.mu == pytest.approx(0.125)
    assert orbitals.flavor_label(1) == "b"
    for ik in range(orbitals.nk):
        for local in range(orbitals.nt):
            assert orbitals.decode_global_index(orbitals.global_index(local, ik)) == (local, ik)


def test_canonical_tdhf_orbitals_from_hf_run_result_delegates_to_final_state() -> None:
    hamiltonian = np.zeros((2, 2, 1), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag([-2.0, 5.0])
    projector = np.zeros_like(hamiltonian)
    projector[0, 0, 0] = 1.0
    state = _toy_hf_state(hamiltonian, projector)
    result = HFRunResult(
        final_state=state,
        iteration_history=[],
        converged=True,
        exit_reason="toy-converged",
        best_seed=7,
        init_mode="toy",
    )

    orbitals = canonical_tdhf_orbitals_from_hf_run_result(result)

    np.testing.assert_allclose(orbitals.energies[:, 0], [-2.0, 5.0])
    assert orbitals.metadata["source"] == "HFRunResult.final_state"
    assert orbitals.metadata["hf_run_converged"] is True
    assert orbitals.metadata["hf_run_best_seed"] == 7


def test_projector_occupation_is_validated_after_nontrivial_hf_basis_rotation() -> None:
    theta = np.pi / 4.0
    unitary = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    hamiltonian = (unitary @ np.diag([-3.0, 4.0]) @ unitary.conjugate().T)[:, :, None]
    occupied_vector = unitary[:, 0]
    projector = np.outer(occupied_vector, occupied_vector.conjugate())[:, :, None]
    assert not np.allclose(np.diag(projector[:, :, 0]).real, [1.0, 0.0])

    state = _toy_hf_state(hamiltonian, projector)
    orbitals = canonical_tdhf_orbitals_from_hf_state(state)

    np.testing.assert_allclose(orbitals.energies[:, 0], [-3.0, 4.0])
    np.testing.assert_array_equal(orbitals.occupied_mask[:, 0], [True, False])
    assert orbitals.metadata["projector_hf_offdiag_residual"] <= 1.0e-12


def test_projector_policy_rejects_fractional_hf_basis_occupation_and_energy_sort_is_explicit() -> None:
    hamiltonian = np.zeros((2, 2, 1), dtype=np.complex128)
    hamiltonian[:, :, 0] = np.diag([-1.0, 1.0])
    projector = 0.5 * np.eye(2, dtype=np.complex128)[:, :, None]
    state = _toy_hf_state(hamiltonian, projector)

    with pytest.raises(ValueError, match="integer occupations"):
        canonical_tdhf_orbitals_from_hf_state(state)

    fallback = canonical_tdhf_orbitals_from_hf_state(state, occupation_policy="energy_sort")
    np.testing.assert_array_equal(fallback.occupied_mask[:, 0], [True, False])
    assert fallback.metadata["occupation_policy"] == "energy_sort"


def test_tdhf_boundary_module_does_not_import_systems_or_crpa() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "mean_field" / "core" / "hf" / "tdhf_boundary.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_prefixes = ("mean_field.systems", "mean_field.crpa")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_prefixes or alias.name.startswith(tuple(prefix + "." for prefix in forbidden_prefixes)):
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in forbidden_prefixes or module.startswith(tuple(prefix + "." for prefix in forbidden_prefixes)):
                offenders.append(f"from {module} import ...")

    assert offenders == []
