from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.contracts import (
    DensityState,
    HamiltonianParts,
    InteractionConfig,
    MFConfig,
    OutputConfig,
    ProjectedBasis,
    ProjectionConfig,
    ReferenceDensity,
    SingleParticleModel,
    SolverConfig,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_matrix_field_shape,
    assert_no_screened_diag_h0_for_RnG,
    assert_projector_field,
)


def _toy_model(system: str = "toy", displacement: float = 0.0) -> SingleParticleModel:
    def h_builder(kvec: np.ndarray) -> np.ndarray:
        kvec = np.asarray(kvec)
        out = np.zeros((2, 2, kvec.size), dtype=np.complex128)
        out[0, 0, :] = -1.0 + displacement
        out[1, 1, :] = 1.0 + displacement
        return out

    def diagonalizer(kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = h_builder(kvec)
        energies = np.zeros((2, kvec.size), dtype=float)
        vectors = np.zeros((2, 2, kvec.size), dtype=np.complex128)
        for ik in range(kvec.size):
            energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])
        return energies, vectors

    return SingleParticleModel(
        system=system,
        lattice=None,
        params={"displacement_mev": displacement},
        hamiltonian_builder=h_builder,
        diagonalizer=diagonalizer,
        metadata={"displacement_mev": displacement},
    )


def test_density_state_projector_relation() -> None:
    reference = ReferenceDensity(
        scheme="average",
        reference=np.stack([0.5 * np.eye(2), 0.5 * np.eye(2)], axis=2),
    )
    projector = np.zeros((2, 2, 2), dtype=np.complex128)
    projector[0, 0, :] = 1.0
    state = DensityState(
        density_delta=projector - reference.reference,
        reference=reference,
        filling=0.0,
        n_occupied_total=2,
    )

    np.testing.assert_allclose(state.projector, projector)
    np.testing.assert_allclose(state.density_delta, state.projector - state.reference.reference)
    assert_density_state_consistent(state)


def test_density_state_can_skip_projector_check_for_intermediate_mixed_state() -> None:
    reference = ReferenceDensity(scheme="average", reference=0.5 * np.eye(2, dtype=np.complex128)[:, :, None])
    mixed_projector = 0.5 * np.eye(2, dtype=np.complex128)[:, :, None]
    state = DensityState(
        density_delta=mixed_projector - reference.reference,
        reference=reference,
        filling=0.0,
        n_occupied_total=1,
    )

    assert_density_state_consistent(state, require_projector=False)
    with pytest.raises(ValueError, match="idempotent"):
        assert_projector_field(state.projector)


def test_hamiltonian_parts_sum_rule() -> None:
    h0 = np.zeros((2, 2, 1), dtype=np.complex128)
    h0[:, :, 0] = np.diag([-1.0, 1.0])
    fixed = np.zeros_like(h0)
    hartree = np.zeros_like(h0)
    hartree[0, 0, 0] = 0.2
    fock = np.zeros_like(h0)
    fock[1, 1, 0] = -0.1
    parts = HamiltonianParts(
        h0=h0,
        fixed=fixed,
        hartree=hartree,
        fock=fock,
        total=h0 + fixed + hartree + fock,
        density_input_convention="delta",
    )

    assert_hamiltonian_parts_consistent(parts)
    with pytest.raises(ValueError, match="sum residual"):
        HamiltonianParts(
            h0=h0,
            fixed=fixed,
            hartree=hartree,
            fock=fock,
            total=h0,
            density_input_convention="delta",
        )


def test_projected_basis_contract_and_rng_screened_h0_guard() -> None:
    physical = _toy_model("RnG_hBN", displacement=48.0)
    basis_model = _toy_model("RnG_hBN", displacement=28.3)
    kvec = np.asarray([0.0 + 0.0j, 0.1 + 0.0j])
    k_grid_frac = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=float)
    h0 = np.zeros((2, 2, 2), dtype=np.complex128)
    h0[:, :, 0] = np.asarray([[-0.7, 0.05], [0.05, 1.2]])
    h0[:, :, 1] = np.asarray([[-0.6, 0.03], [0.03, 1.1]])
    basis_energies = np.asarray([[-0.9, -0.8], [0.9, 0.8]], dtype=float)

    basis = ProjectedBasis(
        physical_model=physical,
        basis_model=basis_model,
        kvec=kvec,
        k_grid_frac=k_grid_frac,
        h0=h0,
        basis_energies=basis_energies,
        active_band_indices=(0, 1),
        active_valence_bands=1,
        active_conduction_bands=1,
        micro_wavefunctions=np.ones((2, 2, 2), dtype=np.complex128),
        band_labels=("v", "c"),
        metadata={
            "projection_mode": "screened",
            "physical_model_displacement_mev": 48.0,
            "basis_model_displacement_mev": 28.3,
            "h0_rule": "project_H_sp_V_into_H_sp_U_basis",
        },
    )
    assert basis.h0.shape == (2, 2, 2)
    assert_no_screened_diag_h0_for_RnG(basis)

    diag_h0 = np.zeros_like(h0)
    for ik in range(2):
        diag_h0[:, :, ik] = np.diag(basis_energies[:, ik])
    bad_basis = ProjectedBasis(
        physical_model=physical,
        basis_model=basis_model,
        kvec=kvec,
        k_grid_frac=k_grid_frac,
        h0=diag_h0,
        basis_energies=basis_energies,
        active_band_indices=(0, 1),
        active_valence_bands=1,
        active_conduction_bands=1,
        micro_wavefunctions=np.ones((2, 2, 2), dtype=np.complex128),
        metadata={
            "projection_mode": "screened",
            "physical_model_displacement_mev": 48.0,
            "basis_model_displacement_mev": 28.3,
            "h0_rule": "project_H_sp_V_into_H_sp_U_basis",
        },
    )
    with pytest.raises(ValueError, match=r"diag\(E\[H_sp\(U\)\]\)"):
        assert_no_screened_diag_h0_for_RnG(bad_basis)


def test_mf_config_contains_no_state_tensors() -> None:
    config = MFConfig(
        system="RnG_hBN",
        run_id="smoke",
        layer_count=5,
        xi=0,
        theta_deg=0.77,
        displacement_mev=48.0,
        k_mesh=3,
        g_shell=1,
        projection=ProjectionConfig(
            mode="screened",
            active_valence_bands=1,
            active_conduction_bands=1,
            basis_displacement_mev=28.3,
        ),
        interaction=InteractionConfig(scheme="average", kind="layered_3d"),
        solver=SolverConfig(max_iter=1, seeds=(1,), init_modes=("bm",)),
        output=OutputConfig(root=None),
    )

    assert config.projection.mode == "screened"
    assert not any(hasattr(config, name) for name in ("density", "hamiltonian", "wavefunctions"))


def test_matrix_field_shape_validator_rejects_external_archive_order() -> None:
    assert_matrix_field_shape(np.zeros((2, 2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="square"):
        assert_matrix_field_shape(np.zeros((3, 2, 2), dtype=np.complex128))
