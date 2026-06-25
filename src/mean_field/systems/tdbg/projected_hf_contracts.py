from __future__ import annotations

"""Canonical mean-field contract adapters for TDBG projected HF results.

The functions here are post-run I/O adapters. They do not change the TDBG SCF
loop, density builders, interaction contractions, or cRPA behavior.
"""

from typing import Any

import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from mean_field.core.hf.contracts_bridge import basis_energies_from_h0, density_state_from_projector, float_diagnostics

from .projected_hf_state import TDBGProjectedHFData, TDBGProjectedHFResult


def _unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "TDBG projected-HF contract records an already-built projected basis; "
        "use mean_field.systems.tdbg model builders for fresh Hamiltonians."
    )


def _unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "TDBG projected-HF contract records post-run arrays; "
        "final diagonalization is not performed by the adapter."
    )


def _single_particle_model(data: TDBGProjectedHFData) -> ContractSingleParticleModel:
    return ContractSingleParticleModel(
        system="tdbg",
        lattice=getattr(data.model, "lattice", None),
        params=data.valley_params,
        hamiltonian_builder=_unavailable_hamiltonian_builder,
        diagonalizer=_unavailable_diagonalizer,
        metadata={
            "theta_deg": float(data.config.theta_deg),
            "cut": float(data.config.cut),
            "paper_ud_ev": float(data.config.paper_ud_ev),
            "paper_ud_convention": data.config.paper_ud_convention,
            "stacking": data.config.stacking,
            "source": "mean_field.systems.tdbg.projected_hf_data",
        },
    )



def _projected_basis(data: TDBGProjectedHFData) -> ContractProjectedBasis:
    model = _single_particle_model(data)
    k_grid_frac = np.asarray(data.k_grid_frac, dtype=float).reshape((data.nk, 2))
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(data.kvec, dtype=np.complex128),
        k_grid_frac=k_grid_frac,
        h0=np.asarray(data.h0, dtype=np.complex128),
        basis_energies=basis_energies_from_h0(data.h0),
        active_band_indices=tuple(int(label.band_index) for label in data.labels),
        active_valence_bands=int(data.lower_band_count),
        active_conduction_bands=int(data.n_band - data.lower_band_count),
        micro_wavefunctions=np.asarray(data.wavefunctions, dtype=np.complex128),
        flavor_labels=tuple(f"{label.valley_label}_{label.spin}" for label in data.labels),
        band_labels=tuple(label.to_dict() for label in data.labels),
        metadata={
            "projected_basis_source": "TDBGProjectedHFData",
            "wavefunctions_axis_order": "state,k,q_site,local",
            "canonical_dense_reconstruction_available": False,
            "reconstruction_adapter": "TDBGProjectedHFResult.reconstruct_micro_wavefunctions",
            "reconstruction_note": "Raw TDBG wavefunctions need spin/valley direct-sum expansion before common k,microscopic_basis,active_basis contraction.",
            "density_axis_order": "abk",
            "window_name": data.config.window.name,
            "window_band_indices": None
            if data.config.window.band_indices is None
            else [int(index) for index in data.config.window.band_indices],
            "moire_area_nm2": float(data.moire_area_nm2),
        },
    )


def _reference_scheme(data: TDBGProjectedHFData) -> str:
    return "CN" if int(data.lower_band_count) > 0 else "custom"


def _density_state(result: TDBGProjectedHFResult) -> ContractDensityState:
    data = result.data
    return density_state_from_projector(
        result.run.state.density,
        data.reference_density,
        reference_scheme=_reference_scheme(data),
        filling=float(data.config.filling),
        n_occupied_total=int(data.n_occupied_per_k * data.nk),
        reference_metadata={
            "system": "tdbg",
            "raw_density_convention": "projector",
            "density_axis_order": "abk",
            "hartree_reference": data.config.interaction.hartree_reference,
            "fock_density": data.config.interaction.fock_density,
            "lower_band_count": int(data.lower_band_count),
        },
        metadata={
            "raw_density_convention": "projector",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.tdbg.projected_hf_contracts",
        },
    )


def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _hamiltonian_parts(result: TDBGProjectedHFResult) -> ContractHamiltonianParts:
    data = result.data
    h0 = np.asarray(data.h0, dtype=np.complex128)
    total = np.asarray(result.run.state.hamiltonian, dtype=np.complex128)
    components = result.hamiltonian_components
    if components is None:
        return ContractHamiltonianParts(
            h0=h0,
            fixed=total - h0,
            hartree=_zero_field_like(h0),
            fock=_zero_field_like(h0),
            total=total,
            density_input_convention="tdbg_projector_policy_collapsed",
            metadata={
                "component_resolution": "collapsed_interaction_minus_h0",
                "supports_crpa": False,
            },
        )

    component_arrays: dict[str, np.ndarray] = {
        str(key): np.asarray(value, dtype=np.complex128) for key, value in components.items()
    }
    hartree = component_arrays.get("hartree", _zero_field_like(h0))
    fock = component_arrays.get("fock", _zero_field_like(h0))
    fixed = _zero_field_like(h0)
    fixed_component_names: list[str] = []
    for name, value in component_arrays.items():
        if name in {"hartree", "fock"}:
            continue
        fixed += value
        fixed_component_names.append(name)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=fixed,
        hartree=hartree,
        fock=fock,
        total=total,
        density_input_convention="tdbg_projector_with_policy_effective_hartree_fock",
        metadata={
            "component_resolution": "tdbg_projected_hf_components",
            "fixed_component_names": fixed_component_names,
            "hartree_reference": data.config.interaction.hartree_reference,
            "fock_density": data.config.interaction.fock_density,
            "supports_crpa": False,
        },
    )



def _iteration_history(result: TDBGProjectedHFResult) -> list[dict[str, Any]]:
    run = result.run
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


def tdbg_projected_hf_result_to_hf_run_result(
    result: TDBGProjectedHFResult,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a TDBG projected-HF run in canonical core contracts.

    The raw TDBG result remains the source of truth for physics arrays. This
    adapter creates a typed I/O view with canonical density, projected-basis, and
    Hamiltonian-part contracts. It does not claim cRPA compatibility and does
    not reconstruct final HF eigenvectors.
    """

    data = result.data
    final_state = ContractHFState(
        basis=_projected_basis(data),
        density=_density_state(result),
        hamiltonian=_hamiltonian_parts(result),
        energies=np.asarray(result.run.state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(result.run.state.mu),
        observables={
            "order_parameters": dict(result.order_parameters),
            "energy_components_ev": dict(result.energy_components),
            "eigenvectors_active_available": False,
        },
        diagnostics=float_diagnostics(result.run.state.diagnostics, finite_only=False),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_iteration_history(result),
        converged=bool(result.run.converged),
        exit_reason=str(result.run.exit_reason),
        best_seed=int(result.seed),
        init_mode=str(result.init_mode),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )


__all__ = ["tdbg_projected_hf_result_to_hf_run_result"]
