from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api import HFConfig, HFResult, ModelRecord, WavefunctionBundle
from mean_field.systems.tdbg.projected_hf_config import (
    SPIN_LABELS,
    TDBGInteractionSettings,
    TDBG_LOCAL_LABELS,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
    VALLEY_SEQUENCE,
)
from mean_field.systems.tdbg.projected_hf_contracts import tdbg_projected_hf_result_to_hf_run_result
from mean_field.systems.tdbg.projected_hf_state import (
    TDBGProjectedHFData,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    TDBGStateLabel,
    reconstruct_tdbg_projected_hf_micro_wavefunctions,
    tdbg_active_eigensystem_from_hamiltonian,
    tdbg_canonical_projected_micro_basis,
)


def _fake_tdbg_labels(band_indices: tuple[int, ...]) -> tuple[TDBGStateLabel, ...]:
    labels: list[TDBGStateLabel] = []
    n_band = len(band_indices)
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band_index in enumerate(band_indices):
                labels.append(
                    TDBGStateLabel(
                        index=int(iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)),
                        spin=str(spin),
                        valley=int(valley),
                        band_position=int(iband),
                        band_index=int(band_index),
                    )
                )
    return tuple(labels)


def _fake_tdbg_data(*, nk: int = 2, n_q: int = 2) -> TDBGProjectedHFData:
    band_indices = (10, 11)
    labels = _fake_tdbg_labels(band_indices)
    nt = len(labels)
    wavefunctions = np.zeros((nt, nk, n_q, 4), dtype=np.complex128)
    for state in range(nt):
        for ik in range(nk):
            for iq in range(n_q):
                for local in range(4):
                    wavefunctions[state, ik, iq, local] = (
                        state + 0.1 * ik + 0.01 * iq + 0.001 * local
                    ) + 1j * (10.0 * state + ik + 0.1 * iq + 0.01 * local)
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        h0[:, :, ik] = np.diag(np.linspace(-1.0, 1.0, nt) + 0.05 * ik)
    config = TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=1.0,
        mesh_size=1,
        window=TDBGProjectedWindow("two_flat"),
        filling=0,
        interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        max_iter=1,
    )
    return TDBGProjectedHFData(
        model=SimpleNamespace(lattice=SimpleNamespace(n_q=n_q), params={}),
        config=config,
        k_grid_frac=np.asarray([[float(ik) / max(nk, 1), 0.25] for ik in range(nk)], dtype=float),
        kvec=np.asarray([0.1 * ik + 0.2j * ik for ik in range(nk)], dtype=np.complex128),
        band_indices=band_indices,
        labels=labels,
        h0=h0,
        wavefunctions=wavefunctions,
        reference_density=np.zeros_like(h0),
        n_occupied_per_k=0,
        lower_band_count=0,
        moire_area_nm2=1.0,
        shifts=(),
        shift_gvecs=np.empty((0,), dtype=np.complex128),
        shift_srcmaps=(),
        valley_params=None,
    )


def _fake_tdbg_result(data: TDBGProjectedHFData) -> TDBGProjectedHFResult:
    nt = data.nt
    nk = data.nk
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        diag = np.linspace(-2.0, 2.0, nt) + 0.1 * ik
        block = np.diag(diag).astype(np.complex128)
        block[0, 1] = block[1, 0] = 0.17 + 0.03 * ik
        block[2, 5] = 0.11j
        block[5, 2] = -0.11j
        hamiltonian[:, :, ik] = block
    energies, _eigenvectors, _metadata = tdbg_active_eigensystem_from_hamiltonian(hamiltonian)
    state = TDBGProjectedHFState(
        h0=np.asarray(data.h0, dtype=np.complex128),
        density=np.zeros_like(data.h0),
        hamiltonian=hamiltonian,
        energies=energies,
        mu=0.0,
        diagnostics={"final_raw_norm": 0.0, "hf_energy": 0.0},
    )
    run = SimpleNamespace(
        state=state,
        iter_energy=np.asarray([], dtype=float),
        iter_err=np.asarray([], dtype=float),
        iter_oda=np.asarray([], dtype=float),
        init_mode="toy",
        seed=1,
        converged=True,
        exit_reason="toy_converged",
        iterations=0,
    )
    return TDBGProjectedHFResult(
        run=run,
        data=data,
        init_mode="toy",
        seed=1,
        order_parameters={},
        energy_components={},
    )


def _expected_tdbg_reconstruction(result: TDBGProjectedHFResult, selected: tuple[int, ...] | None = None) -> np.ndarray:
    basis = tdbg_canonical_projected_micro_basis(result.data)
    _energies, eigenvectors, _metadata = tdbg_active_eigensystem_from_hamiltonian(result.run.state.hamiltonian)
    full = np.einsum("kba,ahk->kbh", basis, eigenvectors, optimize=True)
    if selected is None:
        return full
    return full[:, :, list(selected)]

def test_tdbg_raw_state_k_q_local_expands_to_spin_valley_direct_sum_canonical_basis() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)

    canonical = tdbg_canonical_projected_micro_basis(data)

    sector_stride = data.model.lattice.n_q * len(TDBG_LOCAL_LABELS)
    assert canonical.shape == (data.nk, len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * sector_stride, data.nt)
    for label in data.labels:
        spin_index = SPIN_LABELS.index(label.spin)
        valley_index = VALLEY_SEQUENCE.index(label.valley)
        row0 = (spin_index * len(VALLEY_SEQUENCE) + valley_index) * sector_stride
        expected = data.wavefunctions[label.index].reshape(data.nk, sector_stride)
        np.testing.assert_allclose(canonical[:, row0 : row0 + sector_stride, label.index], expected)
        outside = canonical[:, :, label.index].copy()
        outside[:, row0 : row0 + sector_stride] = 0.0
        assert np.count_nonzero(outside) == 0


def test_tdbg_result_reconstructs_micro_wavefunctions_by_manual_contraction() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    result = _fake_tdbg_result(data)

    bundle = result.reconstruct_micro_wavefunctions()

    expected = _expected_tdbg_reconstruction(result)
    assert isinstance(bundle, WavefunctionBundle)
    np.testing.assert_allclose(bundle.k, data.kvec)
    np.testing.assert_allclose(bundle.wavefunctions, expected, atol=1.0e-14)
    assert bundle.metadata["raw_wavefunctions_axis_order"] == "state,k,q_site,local"
    assert bundle.metadata["micro_basis_axis_order"] == "k,microscopic_basis,active_basis"
    assert bundle.metadata["microscopic_basis_axis_order"] == "spin,valley,q_site,local"
    assert bundle.metadata["reconstruction_path"] == "TDBGProjectedHFResult.reconstruct_micro_wavefunctions"
    assert bundle.metadata["sewing_transforms_available"] is False
    assert bundle.metadata["sewing_policy"].startswith("unavailable")
    assert bundle.metadata["topology_eligible"] is False
    assert "sewing" in bundle.metadata["topology_ineligible_reason"]
    assert bundle.metadata["sewing_transforms_count"] == 0
    assert bundle.metadata["canonical_micro_basis_materialized"] is False
    assert bundle.metadata["selected_hf_state_indices"] == list(range(data.nt))
    assert bundle.metadata["active_hamiltonian_hermiticity_policy"] == "reject_without_symmetrization"
    assert "src/mean_field/systems/tdbg/projected_hf_state.py" in bundle.metadata["evidence_paths"]
    assert bundle.convention.wavefunction_axis_order == "k,microscopic_basis,hf_state"


def test_tdbg_eigensystem_rejects_nonhermitian_hamiltonian_without_symmetrizing() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    result = _fake_tdbg_result(data)
    bad = np.array(result.run.state.hamiltonian, copy=True)
    bad[0, 1, 0] += 1.0e-4

    with pytest.raises(ValueError, match="not Hermitian enough"):
        tdbg_active_eigensystem_from_hamiltonian(bad, hermiticity_atol=1.0e-8)

    result.run.state.hamiltonian[:, :, :] = bad
    with pytest.raises(ValueError, match="Refusing to symmetrize silently"):
        result.reconstruct_micro_wavefunctions(hermiticity_atol=1.0e-8)

def test_tdbg_selected_band_indices_reconstructs_without_full_output_allocation() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    result = _fake_tdbg_result(data)
    selected = (1, 3)
    micro_dim = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * data.model.lattice.n_q * len(TDBG_LOCAL_LABELS)
    full_elements = data.nk * micro_dim * data.nt
    selected_elements = data.nk * micro_dim * len(selected)
    assert selected_elements < full_elements

    with pytest.raises(ValueError, match="size guard"):
        result.reconstruct_micro_wavefunctions(max_dense_elements=full_elements - 1)

    bundle = result.reconstruct_micro_wavefunctions(band_indices=selected, max_dense_elements=selected_elements)

    np.testing.assert_allclose(bundle.wavefunctions, _expected_tdbg_reconstruction(result, selected), atol=1.0e-14)
    assert bundle.wavefunctions.shape == (data.nk, micro_dim, len(selected))
    assert bundle.metadata["selection_argument"] == "band_indices"
    assert bundle.metadata["selected_hf_state_indices"] == list(selected)
    assert bundle.metadata["selected_hf_band_indices"] == list(selected)
    assert bundle.metadata["dense_reconstruction_estimated_elements"] == selected_elements
    assert bundle.metadata["max_dense_elements"] == selected_elements
    assert bundle.metadata["n_reconstructed_states"] == len(selected)
    assert bundle.metadata["band_indices_argument_meaning"].startswith("HF eigenstate indices")

def test_tdbg_state_indices_alias_and_selection_validation() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    result = _fake_tdbg_result(data)

    bundle = result.reconstruct_micro_wavefunctions(state_indices=4, max_dense_elements=10_000)
    np.testing.assert_allclose(bundle.wavefunctions, _expected_tdbg_reconstruction(result, (4,)), atol=1.0e-14)
    assert bundle.metadata["selection_argument"] == "state_indices"
    assert bundle.metadata["selected_hf_state_indices"] == [4]

    with pytest.raises(ValueError, match="only one of state_indices or band_indices"):
        result.reconstruct_micro_wavefunctions(state_indices=0, band_indices=0)
    with pytest.raises(ValueError, match="Duplicate"):
        result.reconstruct_micro_wavefunctions(state_indices=(0, 0))
    with pytest.raises(ValueError, match="outside"):
        result.reconstruct_micro_wavefunctions(state_indices=(data.nt,))

def test_tdbg_core_reconstruction_bundle_keeps_topology_ineligible_sewing_metadata() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    result = _fake_tdbg_result(data)

    core_bundle = reconstruct_tdbg_projected_hf_micro_wavefunctions(result, state_indices=(0,), max_dense_elements=10_000)

    assert core_bundle.sewing_transforms == ()
    assert core_bundle.basis_metadata["sewing_transforms_available"] is False
    assert core_bundle.basis_metadata["topology_eligible"] is False
    assert "sewing" in core_bundle.basis_metadata["topology_ineligible_reason"]
    assert core_bundle.basis_metadata["sewing_transforms_count"] == 0
    assert core_bundle.basis_metadata["uncertainty"].startswith("Algebraic direct-sum")

def test_public_hfresult_tdbg_state_adapter_precedes_raw_noncanonical_contract_fallback() -> None:
    data = _fake_tdbg_data(nk=2, n_q=2)
    raw = _fake_tdbg_result(data)
    canonical = tdbg_projected_hf_result_to_hf_run_result(raw)
    _energies, eigenvectors, _metadata = tdbg_active_eigensystem_from_hamiltonian(raw.run.state.hamiltonian)
    nonempty_raw_contract_state = replace(canonical.final_state, eigenvectors_active=eigenvectors)
    nonempty_raw_contract = replace(canonical, final_state=nonempty_raw_contract_state)

    public_result = HFResult(
        model=ModelRecord(system_name="tdbg"),
        config=HFConfig(filling=0.0, mesh=(data.nk, 1), density_convention="projector"),
        state=raw,
        canonical_run_result=canonical,
    )
    bundle = public_result.reconstruct_micro_wavefunctions()
    expected = np.einsum(
        "kba,ahk->kbh",
        tdbg_canonical_projected_micro_basis(data),
        eigenvectors,
        optimize=True,
    )
    np.testing.assert_allclose(bundle.wavefunctions, expected, atol=1.0e-14)

    fallback_only = HFResult(
        model=ModelRecord(system_name="tdbg"),
        config=HFConfig(filling=0.0, mesh=(data.nk, 1), density_convention="projector"),
        state=SimpleNamespace(),
        canonical_run_result=nonempty_raw_contract,
    )
    with pytest.raises(NotImplementedError, match="rank 3"):
        fallback_only.reconstruct_micro_wavefunctions()
