from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api import HFConfig, HFResult, ModelRecord, WavefunctionBundle
from analysis.topology import compute_system_topology_from_bundle
from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HamiltonianParts as ContractHamiltonianParts,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    ProjectedBasis as ContractProjectedBasis,
    ReferenceDensity as ContractReferenceDensity,
    SingleParticleModel as ContractSingleParticleModel,
)


def _rotation(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.complex128,
    )


def _toy_model() -> ContractSingleParticleModel:
    return ContractSingleParticleModel(
        system="toy",
        lattice={"kind": "unit-test"},
        params={},
        hamiltonian_builder=lambda _k: np.zeros((2, 2), dtype=np.complex128),
        diagonalizer=lambda h: np.linalg.eigh(h),
        metadata={"evidence_path": "tests/test_api_hf_result_reconstruction.py"},
    )


def _toy_canonical_run_result() -> tuple[ContractHFRunResult, np.ndarray, np.ndarray]:
    n_active = 2
    n_k = 3
    micro_dim = 3
    kvec = np.asarray([0.0, 0.25 + 0.5j, 0.5 + 0.25j], dtype=np.complex128)
    k_grid_frac = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=float)
    h0 = np.zeros((n_active, n_active, n_k), dtype=np.complex128)
    micro_wavefunctions = np.zeros((n_k, micro_dim, n_active), dtype=np.complex128)
    for ik in range(n_k):
        micro_wavefunctions[ik] = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.1 * (ik + 1), 0.2j * (ik + 1)],
            ],
            dtype=np.complex128,
        )
    eigenvectors_active = np.empty((n_active, n_active, n_k), dtype=np.complex128)
    for ik in range(n_k):
        eigenvectors_active[:, :, ik] = _rotation(0.2 * ik)

    model = _toy_model()
    basis = ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=kvec,
        k_grid_frac=k_grid_frac,
        h0=h0,
        basis_energies=np.asarray([[-1.0, -0.9, -0.8], [0.4, 0.5, 0.6]], dtype=float),
        active_band_indices=(10, 11),
        active_valence_bands=1,
        active_conduction_bands=1,
        micro_wavefunctions=micro_wavefunctions,
        metadata={"system": "toy", "fixture": "canonical_dense_arrays", "wavefunctions_axis_order": "k,microscopic_basis,active_basis"},
    )
    reference = ContractReferenceDensity(
        scheme="custom",
        reference=np.zeros_like(h0),
        metadata={"fixture": "zero_reference"},
    )
    density = ContractDensityState(
        density_delta=np.zeros_like(h0),
        reference=reference,
        filling=0.0,
        n_occupied_total=0,
    )
    hamiltonian = ContractHamiltonianParts(
        h0=h0,
        fixed=np.zeros_like(h0),
        hartree=np.zeros_like(h0),
        fock=np.zeros_like(h0),
        total=h0,
        density_input_convention="delta",
    )
    final_state = ContractHFState(
        basis=basis,
        density=density,
        hamiltonian=hamiltonian,
        energies=np.asarray([[-1.0, -0.9, -0.8], [0.4, 0.5, 0.6]], dtype=float),
        eigenvectors_active=eigenvectors_active,
        mu=0.0,
    )
    return (
        ContractHFRunResult(
            final_state=final_state,
            iteration_history=[],
            converged=True,
            exit_reason="toy_converged",
            best_seed=1,
            init_mode="toy",
        ),
        micro_wavefunctions,
        eigenvectors_active,
    )


def _toy_hf_result(canonical_run_result: ContractHFRunResult | None) -> HFResult:
    return HFResult(
        model=ModelRecord(system_name="toy"),
        config=HFConfig(filling=0.0, mesh=(3, 1)),
        state=SimpleNamespace(),
        canonical_run_result=canonical_run_result,
    )


def test_hfresult_reconstruct_micro_wavefunctions_uses_canonical_dense_array_fallback() -> None:
    canonical_run_result, micro_wavefunctions, eigenvectors_active = _toy_canonical_run_result()

    bundle = _toy_hf_result(canonical_run_result).reconstruct_micro_wavefunctions()

    expected = np.einsum("kba,ahk->kbh", micro_wavefunctions, eigenvectors_active, optimize=True)
    assert isinstance(bundle, WavefunctionBundle)
    np.testing.assert_allclose(bundle.k, canonical_run_result.final_state.basis.kvec)
    np.testing.assert_allclose(bundle.wavefunctions, expected, atol=1.0e-14)
    assert bundle.metadata["hf_result_reconstruction"] == "canonical_dense_array_fallback"
    assert bundle.metadata["reconstruction_path"] == "HFResult.canonical_dense_array_fallback"
    assert bundle.metadata["source"] == "hf_reconstructed"
    assert bundle.metadata["psi_micro_axis_order"] == "k,microscopic_basis,hf_state"
    assert bundle.metadata["k_grid_frac_shape"] == [3, 2]
    assert bundle.metadata["active_eigenvectors_unitarity_residual"] < 1.0e-14
    assert bundle.metadata["selection_argument"] == "all"
    assert bundle.metadata["selected_hf_state_indices"] == [0, 1]
    assert "src/mean_field/api/_hf_result.py" in bundle.metadata["evidence_paths"]
    assert "system-specific sewing" in bundle.metadata["uncertainty"]
    assert bundle.metadata["topology_eligible"] is False
    assert "no system sewing/grid topology adapter" in bundle.metadata["topology_ineligible_reason"]
    with pytest.raises(ValueError, match="topology_eligible=False"):
        compute_system_topology_from_bundle(bundle, 0, system="toy")
    assert bundle.convention.wavefunction_axis_order == "k,microscopic_basis,hf_state"

def test_hfresult_reconstruct_micro_wavefunctions_supports_selected_canonical_fallback_and_guard() -> None:
    canonical_run_result, micro_wavefunctions, eigenvectors_active = _toy_canonical_run_result()
    result = _toy_hf_result(canonical_run_result)
    selected = (1,)
    full_elements = micro_wavefunctions.shape[0] * micro_wavefunctions.shape[1] * eigenvectors_active.shape[1]
    selected_elements = micro_wavefunctions.shape[0] * micro_wavefunctions.shape[1] * len(selected)

    with pytest.raises(ValueError, match="size guard"):
        result.reconstruct_micro_wavefunctions(max_dense_elements=full_elements - 1)

    bundle = result.reconstruct_micro_wavefunctions(state_indices=selected, max_dense_elements=selected_elements)

    expected = np.einsum("kba,ahk->kbh", micro_wavefunctions, eigenvectors_active[:, list(selected), :], optimize=True)
    np.testing.assert_allclose(bundle.wavefunctions, expected, atol=1.0e-14)
    assert bundle.wavefunctions.shape == (micro_wavefunctions.shape[0], micro_wavefunctions.shape[1], len(selected))
    assert bundle.metadata["selection_argument"] == "state_indices"
    assert bundle.metadata["selected_hf_state_indices"] == list(selected)
    assert bundle.metadata["selected_hf_band_indices"] == list(selected)
    assert bundle.metadata["dense_reconstruction_estimated_elements"] == selected_elements
    assert bundle.metadata["max_dense_elements"] == selected_elements
    assert bundle.metadata["n_reconstructed_states"] == len(selected)
    assert bundle.metadata["band_indices_argument_meaning"].startswith("HF eigenstate indices")

    alias = result.reconstruct_micro_wavefunctions(band_indices=0, max_dense_elements=selected_elements)
    np.testing.assert_allclose(alias.wavefunctions, np.einsum("kba,ahk->kbh", micro_wavefunctions, eigenvectors_active[:, [0], :], optimize=True))
    assert alias.metadata["selection_argument"] == "band_indices"
    assert alias.metadata["selected_hf_state_indices"] == [0]

    with pytest.raises(ValueError, match="only one of state_indices or band_indices"):
        result.reconstruct_micro_wavefunctions(state_indices=0, band_indices=0)
    with pytest.raises(ValueError, match="Duplicate"):
        result.reconstruct_micro_wavefunctions(state_indices=(0, 0))
    with pytest.raises(ValueError, match="outside"):
        result.reconstruct_micro_wavefunctions(state_indices=(eigenvectors_active.shape[1],))


def test_hfresult_reconstruct_micro_wavefunctions_requires_nonempty_canonical_arrays() -> None:
    canonical_run_result, _micro_wavefunctions, _eigenvectors_active = _toy_canonical_run_result()

    with pytest.raises(NotImplementedError, match="canonical dense arrays"):
        _toy_hf_result(None).reconstruct_micro_wavefunctions()

    empty_eigen_state = replace(
        canonical_run_result.final_state,
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
    )
    with pytest.raises(NotImplementedError, match="final_state.eigenvectors_active.*empty"):
        _toy_hf_result(replace(canonical_run_result, final_state=empty_eigen_state)).reconstruct_micro_wavefunctions()

    empty_basis = replace(
        canonical_run_result.final_state.basis,
        micro_wavefunctions=np.empty((0,), dtype=np.complex128),
    )
    empty_micro_state = replace(canonical_run_result.final_state, basis=empty_basis)
    with pytest.raises(NotImplementedError, match="basis.micro_wavefunctions.*empty"):
        _toy_hf_result(replace(canonical_run_result, final_state=empty_micro_state)).reconstruct_micro_wavefunctions()

    raw_basis = replace(canonical_run_result.final_state.basis, micro_wavefunctions=np.zeros((2, 3, 2, 1), dtype=np.complex128))
    raw_state = replace(canonical_run_result.final_state, basis=raw_basis)
    with pytest.raises(NotImplementedError, match="rank 3"):
        _toy_hf_result(replace(canonical_run_result, final_state=raw_state)).reconstruct_micro_wavefunctions()

    unlabelled_basis = replace(canonical_run_result.final_state.basis, metadata={"system": "toy"})
    unlabelled_state = replace(canonical_run_result.final_state, basis=unlabelled_basis)
    with pytest.raises(NotImplementedError, match="wavefunctions_axis_order"):
        _toy_hf_result(replace(canonical_run_result, final_state=unlabelled_state)).reconstruct_micro_wavefunctions()


def test_hfresult_reconstruct_micro_wavefunctions_keeps_state_adapter_precedence() -> None:
    sentinel = WavefunctionBundle(
        k=np.asarray([0.0], dtype=np.complex128),
        wavefunctions=np.ones((1, 1, 1), dtype=np.complex128),
        metadata={"source": "state_adapter"},
    )
    result = HFResult(
        model=ModelRecord(system_name="toy"),
        config=HFConfig(filling=0.0, mesh=(1, 1)),
        state=SimpleNamespace(reconstruct_micro_wavefunctions=lambda: sentinel),
        canonical_run_result=None,
    )

    assert result.reconstruct_micro_wavefunctions() is sentinel
    with pytest.raises(NotImplementedError, match="does not support.*state_indices"):
        result.reconstruct_micro_wavefunctions(state_indices=0)

def test_hfresult_reconstruct_micro_wavefunctions_forwards_selected_kwargs_to_state_adapter() -> None:
    sentinel = WavefunctionBundle(
        k=np.asarray([0.0], dtype=np.complex128),
        wavefunctions=np.ones((1, 1, 1), dtype=np.complex128),
        metadata={"source": "state_adapter"},
    )
    calls: list[dict[str, object]] = []

    class Adapter:
        def reconstruct_micro_wavefunctions(
            self,
            *,
            state_indices=None,
            band_indices=None,
            max_dense_elements=None,
        ):
            calls.append(
                {
                    "state_indices": state_indices,
                    "band_indices": band_indices,
                    "max_dense_elements": max_dense_elements,
                }
            )
            return sentinel

    result = HFResult(
        model=ModelRecord(system_name="toy"),
        config=HFConfig(filling=0.0, mesh=(1, 1)),
        state=Adapter(),
        canonical_run_result=None,
    )

    assert result.reconstruct_micro_wavefunctions(state_indices=(0, 2), max_dense_elements=123) is sentinel
    assert calls[-1] == {"state_indices": (0, 2), "band_indices": None, "max_dense_elements": 123}

    assert result.reconstruct_micro_wavefunctions(band_indices=1, max_dense_elements=None) is sentinel
    assert calls[-1] == {"state_indices": None, "band_indices": 1, "max_dense_elements": None}
