from __future__ import annotations

"""Canonical mean-field contract adapters for RnG/hBN HF runs.

The functions here are post-run I/O adapters.  They wrap arrays already produced
by the existing RnG/hBN Hartree-Fock path and do not change SCF, screening,
interaction contractions, topology, or cRPA behavior.
"""

from collections.abc import Mapping
from typing import Any
import math

import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from mean_field.core.hf.contracts_bridge import density_state_from_delta

from .hf import (
    RLGhBNHartreeFockRun,
    RLGhBNProjectedBasisData,
    rlg_hbn_filling_from_density,
    rlg_hbn_occupied_state_count,
)


def _unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "RnG/hBN contract records an already-built projected basis; "
        "use mean_field.systems.RnG_hBN model builders for fresh Hamiltonians."
    )


def _unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "RnG/hBN contract records post-run arrays; "
        "fresh diagonalization is not performed by the adapter."
    )


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _summary_dict(value: object) -> dict[str, object]:
    to_summary_dict = getattr(value, "to_summary_dict", None)
    if callable(to_summary_dict):
        return dict(to_summary_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {"repr": repr(value)}


def _model_metadata(data: RLGhBNProjectedBasisData, *, role: str) -> dict[str, object]:
    model = data.basis_model if role == "basis" else data.model
    params = getattr(model, "params", None)
    displacement = getattr(params, "displacement_field_mev", None)
    metadata: dict[str, object] = {
        "role": role,
        "source": "mean_field.systems.RnG_hBN.hf",
        "theta_deg": float(getattr(getattr(model, "lattice", None), "theta_deg", float("nan"))),
        "layer_count": int(getattr(params, "layer_count", 0)),
        "xi": int(getattr(params, "xi", 0)),
        "displacement_field_mev": None if displacement is None else float(displacement),
        "basis_displacement_field_mev": float(data.basis_model.params.displacement_field_mev),
        "physical_displacement_field_mev": float(data.model.params.displacement_field_mev),
        "interaction": data.interaction.to_summary_dict(),
        "screening_available": data.screening is not None,
        "supports_crpa": False,
    }
    return metadata


def _single_particle_model(data: RLGhBNProjectedBasisData, *, role: str) -> ContractSingleParticleModel:
    model = data.basis_model if role == "basis" else data.model
    return ContractSingleParticleModel(
        system="RnG_hBN",
        lattice=getattr(model, "lattice", None),
        params=_summary_dict(getattr(model, "params", None)),
        hamiltonian_builder=_unavailable_hamiltonian_builder,
        diagonalizer=_unavailable_diagonalizer,
        metadata=_model_metadata(data, role=role),
    )


def _flatten_k_grid_frac(data: RLGhBNProjectedBasisData) -> np.ndarray:
    k_grid_frac = np.asarray(data.k_grid_frac, dtype=float)
    if k_grid_frac.size != int(data.nk) * 2:
        raise ValueError(
            "RnG/hBN canonical ProjectedBasis requires k_grid_frac with 2 coordinates per k point; "
            f"got shape {k_grid_frac.shape} for nk={data.nk}"
        )
    return k_grid_frac.reshape((int(data.nk), 2))


def _state_index(data: RLGhBNProjectedBasisData) -> np.ndarray:
    return np.arange(int(data.basis.nt), dtype=int).reshape(
        (int(data.basis.n_spin), int(data.basis.n_flavor), int(data.basis.n_band)),
        order="F",
    )


def _active_band_indices(data: RLGhBNProjectedBasisData) -> tuple[int, ...]:
    active = tuple(int(index) for index in data.active_band_indices)
    if len(active) != int(data.basis.n_band):
        raise ValueError(
            "RnG/hBN active_band_indices must be per projected band; "
            f"got {len(active)} labels for n_band={data.basis.n_band}"
        )
    labels = np.zeros((int(data.basis.nt),), dtype=int)
    state_index = _state_index(data)
    for ispin in range(int(data.basis.n_spin)):
        for ieta in range(int(data.basis.n_flavor)):
            for iband, band_index in enumerate(active):
                labels[int(state_index[ispin, ieta, iband])] = int(band_index)
    return tuple(int(value) for value in labels)


def _basis_energies(data: RLGhBNProjectedBasisData) -> np.ndarray:
    raw = np.asarray(data.band_energies, dtype=float)
    n_band = int(data.basis.n_band)
    n_eta = int(data.basis.n_flavor)
    nk = int(data.basis.nk)
    if raw.shape == (int(data.basis.nt), nk):
        return raw
    if raw.shape != (n_band, n_eta, nk):
        raise ValueError(
            "RnG/hBN band_energies must have shape (n_band, n_flavor, n_k) "
            f"or (n_state, n_k), got {raw.shape}"
        )
    out = np.zeros((int(data.basis.nt), nk), dtype=float)
    state_index = _state_index(data)
    for ispin in range(int(data.basis.n_spin)):
        for ieta in range(n_eta):
            for iband in range(n_band):
                out[int(state_index[ispin, ieta, iband]), :] = raw[iband, ieta, :]
    return out


def _flavor_labels(data: RLGhBNProjectedBasisData) -> tuple[str, ...]:
    labels = [""] * int(data.basis.nt)
    valleys = tuple(int(value) for value in data.valleys)
    state_index = _state_index(data)
    for ispin in range(int(data.basis.n_spin)):
        for ieta in range(int(data.basis.n_flavor)):
            valley_label = valleys[ieta] if ieta < len(valleys) else ieta
            for iband in range(int(data.basis.n_band)):
                labels[int(state_index[ispin, ieta, iband])] = f"spin{ispin}_valley{valley_label}_band{iband}"
    return tuple(labels)


def _band_labels(data: RLGhBNProjectedBasisData) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "active_window_index": int(index),
            "physical_band_index": int(band_index),
        }
        for index, band_index in enumerate(data.active_band_indices)
    )


def _projection_mode(data: RLGhBNProjectedBasisData) -> str:
    return "screened" if bool(data.interaction.use_screened_basis) else "bare"


def _projected_basis(data: RLGhBNProjectedBasisData) -> ContractProjectedBasis:
    physical_model = _single_particle_model(data, role="physical")
    basis_model = _single_particle_model(data, role="basis")
    projection_mode = _projection_mode(data)
    physical_displacement = float(data.model.params.displacement_field_mev)
    basis_displacement = float(data.basis_model.params.displacement_field_mev)
    return ContractProjectedBasis(
        physical_model=physical_model,
        basis_model=basis_model,
        kvec=np.asarray(data.kvec, dtype=np.complex128),
        k_grid_frac=_flatten_k_grid_frac(data),
        h0=np.asarray(data.h0, dtype=np.complex128),
        basis_energies=_basis_energies(data),
        active_band_indices=_active_band_indices(data),
        active_valence_bands=int(data.interaction.active_valence_bands),
        active_conduction_bands=int(data.interaction.active_conduction_bands),
        micro_wavefunctions=np.asarray(data.basis.wavefunctions, dtype=np.complex128),
        flavor_labels=_flavor_labels(data),
        band_labels=_band_labels(data),
        metadata={
            "projected_basis_source": "RLGhBNProjectedBasisData",
            "wavefunctions_axis_order": "basis,band,flavor,k",
            "density_axis_order": "abk",
            "active_band_semantics": "physical_active_band_indices_repeated_over_spin_valley",
            "active_band_indices_per_band": [int(index) for index in data.active_band_indices],
            "flat_band_indices": [int(index) for index in data.flat_band_indices],
            "valleys": [int(value) for value in data.valleys],
            "projection_mode": projection_mode,
            "h0_rule": "project_H_sp_V_into_H_sp_U_basis"
            if projection_mode == "screened"
            else "project_H_sp_V_into_own_basis",
            "h0_includes_fixed_remote_hamiltonian": data.fixed_remote_hamiltonian is not None,
            "physical_h0_available": data.physical_h0 is not None,
            "fixed_remote_hamiltonian_available": data.fixed_remote_hamiltonian is not None,
            "screening_available": data.screening is not None,
            "supports_crpa": False,
            "physical_model_displacement_mev": physical_displacement,
            "basis_model_displacement_mev": basis_displacement,
            "screened_u_mev": float(data.screened_u_mev),
            "interaction_scheme": str(data.interaction.scheme),
            "interaction_dimension": str(data.interaction.interaction_dimension),
            "reciprocal_grid_shape": [int(value) for value in data.reciprocal_grid_shape],
            "reciprocal_grid_origin": [int(value) for value in data.reciprocal_grid_origin],
            "moire_cell_area_nm2": float(data.moire_cell_area_nm2),
        },
    )


def _density_state(run: RLGhBNHartreeFockRun) -> ContractDensityState:
    state = run.state
    reference = np.asarray(state.reference_density, dtype=np.complex128)
    return density_state_from_delta(
        state.density,
        reference,
        reference_scheme=state.scheme,
        filling=float(state.nu),
        n_occupied_total=rlg_hbn_occupied_state_count(
            state.nu,
            state.nt,
            state.nk,
            active_valence_bands=state.active_valence_bands,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ),
        reference_metadata={
            "system": "RnG_hBN",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "state_scheme": str(state.scheme),
            "active_valence_bands": int(state.active_valence_bands),
            "reference_scheme_source": "RLGhBNHartreeFockState.reference_density",
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P-R",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.RnG_hBN.hf_contracts",
            "filling_from_density": float(
                rlg_hbn_filling_from_density(
                    state.density,
                    reference,
                    active_valence_bands=state.active_valence_bands,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
        },
    )


def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _hamiltonian_parts(run: RLGhBNHartreeFockRun) -> ContractHamiltonianParts:
    state = run.state
    data = run.basis_data
    h0 = np.asarray(state.h0, dtype=np.complex128)
    total = np.asarray(state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="rlg_hbn_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "supports_crpa": False,
            "h0_rule": "project_H_sp_V_into_H_sp_U_basis"
            if _projection_mode(data) == "screened"
            else "project_H_sp_V_into_own_basis",
            "h0_includes_fixed_remote_hamiltonian": data.fixed_remote_hamiltonian is not None,
            "physical_h0_available": data.physical_h0 is not None,
            "fixed_remote_hamiltonian_available": data.fixed_remote_hamiltonian is not None,
            "interaction_scheme": str(data.interaction.scheme),
        },
    )


def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _iteration_history(run: RLGhBNHartreeFockRun) -> list[dict[str, Any]]:
    count = max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))
    history: list[dict[str, Any]] = []
    for idx in range(count):
        history.append(
            {
                "iteration": int(idx + 1),
                "energy": float(run.iter_energy[idx]) if idx < len(run.iter_energy) else None,
                "error": float(run.iter_err[idx]) if idx < len(run.iter_err) else None,
                "oda_lambda": float(run.iter_oda[idx]) if idx < len(run.iter_oda) else None,
            }
        )
    return history


def rlg_hbn_hf_run_to_hf_run_result(
    run: RLGhBNHartreeFockRun,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap an RnG/hBN HF run in canonical core contracts.

    The raw RnG/hBN run remains the source of truth.  This adapter preserves the
    stored density delta ``P-R`` and creates a typed I/O view with collapsed
    Hamiltonian parts.  It does not recompute HF, split Hartree/Fock components,
    reconstruct final active eigenvectors, or touch cRPA.
    """

    state = run.state
    reference = np.asarray(state.reference_density, dtype=np.complex128)
    final_state = ContractHFState(
        basis=_projected_basis(run.basis_data),
        density=_density_state(run),
        hamiltonian=_hamiltonian_parts(run),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables={
            "eigenvectors_active_available": False,
            "primitive_nu": float(state.nu),
            "filling_from_density": float(
                rlg_hbn_filling_from_density(
                    state.density,
                    reference,
                    active_valence_bands=state.active_valence_bands,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
            "screening_available": run.basis_data.screening is not None,
            "fixed_remote_hamiltonian_available": run.basis_data.fixed_remote_hamiltonian is not None,
            "occupation_counts": None
            if state.occupation_counts is None
            else [int(value) for value in state.occupation_counts],
        },
        diagnostics=_float_diagnostics(state.diagnostics),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_iteration_history(run),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        best_seed=int(run.seed),
        init_mode=str(run.init_mode),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )


__all__ = ["rlg_hbn_hf_run_to_hf_run_result"]
