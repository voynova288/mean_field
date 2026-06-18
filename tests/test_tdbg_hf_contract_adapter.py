from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api.hf import HFState as APIHFState
from mean_field.core.contracts import (
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    assert_density_state_consistent,
    assert_hamiltonian_parts_consistent,
    assert_projected_basis_consistent,
)
from mean_field.core.hf import HartreeFockRun
from mean_field.systems.tdbg.projected_hf import (
    SPIN_LABELS,
    VALLEY_SEQUENCE,
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedHFData,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    TDBGProjectedWindow,
    TDBGStateLabel,
    initialize_tdbg_density,
    tdbg_energy_components,
    tdbg_order_parameters,
    tdbg_projected_hf_result_to_hf_run_result,
)


def _toy_data(*, filling: int = 2) -> TDBGProjectedHFData:
    labels: list[TDBGStateLabel] = []
    band_indices = (10, 11)
    n_band = 2
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band in enumerate(band_indices):
                idx = iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)
                labels.append(TDBGStateLabel(idx, spin, valley, iband, band))
    nt = len(labels)
    nk = 1
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    reference = np.zeros_like(h0)
    for label in labels:
        h0[label.index, label.index, :] = 0.01 * float(label.index)
        if label.band_position == 0:
            reference[label.index, label.index, :] = 1.0
    lattice = SimpleNamespace(n_q=1)
    model = SimpleNamespace(lattice=lattice)
    return TDBGProjectedHFData(
        model=model,  # type: ignore[arg-type]
        config=TDBGProjectedHFConfig(
            window=TDBGProjectedWindow("two_flat"),
            filling=int(filling),
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
            max_iter=2,
        ),
        k_grid_frac=np.zeros((1, 1, 2)),
        kvec=np.zeros(1, dtype=np.complex128),
        band_indices=band_indices,
        labels=tuple(labels),
        h0=h0,
        wavefunctions=np.zeros((nt, nk, 1, 4), dtype=np.complex128),
        reference_density=reference,
        n_occupied_per_k=4 + int(filling),
        lower_band_count=1,
        moire_area_nm2=1.0,
        shifts=(),
        shift_gvecs=np.zeros(0, dtype=np.complex128),
        shift_srcmaps=(),
    )


def _toy_result(*, with_components: bool = True) -> TDBGProjectedHFResult:
    data = _toy_data()
    density = initialize_tdbg_density(data, init_mode="sp", seed=1)
    hartree = np.zeros_like(data.h0)
    fock = np.zeros_like(data.h0)
    onsite = np.zeros_like(data.h0)
    hartree[0, 0, 0] = 0.2
    fock[1, 1, 0] = -0.05
    onsite[2, 2, 0] = 0.03
    hamiltonian_components = {"hartree": hartree, "fock": fock, "onsite": onsite} if with_components else None
    interaction = hartree + fock + onsite if with_components else np.zeros_like(data.h0)
    state = TDBGProjectedHFState(
        h0=data.h0.copy(),
        density=density,
        hamiltonian=data.h0 + interaction,
        energies=np.zeros((data.nt, data.nk), dtype=float),
        mu=0.125,
        precision=1.0e-7,
        diagnostics={"hf_energy": 1.5, "final_raw_norm": 0.25},
    )
    run = HartreeFockRun(
        state=state,
        iter_energy=np.asarray([1.7, 1.5], dtype=float),
        iter_err=np.asarray([0.5, 0.25], dtype=float),
        iter_oda=np.asarray([1.0, 0.8], dtype=float),
        init_mode="sp",
        seed=3,
        converged=False,
        exit_reason="max_iter",
    )
    energy_components = tdbg_energy_components(
        data,
        density,
        interaction_components={} if hamiltonian_components is None else hamiltonian_components,
    )
    return TDBGProjectedHFResult(
        run=run,
        data=data,
        init_mode="sp",
        seed=3,
        order_parameters=tdbg_order_parameters(data, density),
        energy_components=energy_components,
        hamiltonian_components=hamiltonian_components,
    )


def test_tdbg_projected_hf_result_wraps_canonical_hf_run_result() -> None:
    raw = _toy_result()

    canonical = tdbg_projected_hf_result_to_hf_run_result(raw)

    assert isinstance(canonical, ContractHFRunResult)
    assert isinstance(canonical.final_state, ContractHFState)
    assert not isinstance(canonical.final_state, APIHFState)
    assert canonical.best_seed == 3
    assert canonical.init_mode == "sp"
    assert canonical.iteration_history == [
        {"iteration": 1, "energy": 1.7, "error": 0.5, "oda_lambda": 1.0},
        {"iteration": 2, "energy": 1.5, "error": 0.25, "oda_lambda": 0.8},
    ]

    np.testing.assert_allclose(canonical.final_state.density.projector, raw.run.state.density)
    np.testing.assert_allclose(
        canonical.final_state.density.density_delta,
        raw.run.state.density - raw.data.reference_density,
    )
    assert canonical.final_state.density.reference.scheme == "CN"
    assert canonical.final_state.density.n_occupied_total == raw.data.n_occupied_per_k * raw.data.nk
    assert_density_state_consistent(canonical.final_state.density)

    assert canonical.final_state.basis.k_grid_frac.shape == (raw.data.nk, 2)
    assert canonical.final_state.basis.active_band_indices == tuple(label.band_index for label in raw.data.labels)
    assert canonical.final_state.basis.metadata["wavefunctions_axis_order"] == "state,k,q_site,local"
    assert_projected_basis_consistent(canonical.final_state.basis)

    assert canonical.final_state.hamiltonian.metadata["fixed_component_names"] == ["onsite"]
    assert canonical.final_state.hamiltonian.metadata["supports_crpa"] is False
    assert_hamiltonian_parts_consistent(canonical.final_state.hamiltonian)
    np.testing.assert_allclose(canonical.final_state.hamiltonian.total, raw.run.state.hamiltonian)
    assert canonical.final_state.observables["eigenvectors_active_available"] is False
    assert canonical.final_state.eigenvectors_active.size == 0


def test_tdbg_contract_adapter_falls_back_to_collapsed_interaction_when_components_absent() -> None:
    raw = _toy_result(with_components=False)

    canonical = tdbg_projected_hf_result_to_hf_run_result(raw, archive_manifest={"path": "toy"})

    assert canonical.archive_manifest == {"path": "toy"}
    assert canonical.final_state.hamiltonian.metadata["component_resolution"] == "collapsed_interaction_minus_h0"
    assert canonical.final_state.hamiltonian.metadata["supports_crpa"] is False
    assert_hamiltonian_parts_consistent(canonical.final_state.hamiltonian)
    np.testing.assert_allclose(canonical.final_state.hamiltonian.total, raw.run.state.hamiltonian)


def test_tdbg_contract_adapter_rejects_inconsistent_components() -> None:
    raw = _toy_result()
    assert raw.hamiltonian_components is not None
    bad_components = dict(raw.hamiltonian_components)
    bad_components["hartree"] = bad_components["hartree"].copy()
    bad_components["hartree"][0, 0, 0] += 1.0
    bad = TDBGProjectedHFResult(
        run=raw.run,
        data=raw.data,
        init_mode=raw.init_mode,
        seed=raw.seed,
        order_parameters=raw.order_parameters,
        energy_components=raw.energy_components,
        hamiltonian_components=bad_components,
    )

    with pytest.raises(ValueError, match="sum residual"):
        tdbg_projected_hf_result_to_hf_run_result(bad)
