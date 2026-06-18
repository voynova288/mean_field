from __future__ import annotations

"""Canonical mean-field contract adapters for HTG supercell HF runs.

The functions here are post-run I/O adapters.  They wrap arrays produced by the
existing HTG supercell HF path and do not change SCF, interaction contractions,
topology, or cRPA behavior.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
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

from .supercell import (
    HTGSupercellHartreeFockRun,
    HTGSupercellProjectedBasisData,
    _supercell_reference_density_blocks,
    htg_supercell_filling_from_density,
    htg_supercell_occupied_count_per_k,
)

if TYPE_CHECKING:
    from mean_field.api import HFConfig, HFResult


def _unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "HTG supercell contract records an already-built projected basis; "
        "use mean_field.systems.htg builders for fresh Hamiltonians."
    )


def _unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "HTG supercell contract records post-run arrays; "
        "fresh diagonalization is not performed by the adapter."
    )


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _single_particle_model(data: HTGSupercellProjectedBasisData) -> ContractSingleParticleModel:
    model = data.model
    params = model.params
    interaction = data.interaction
    metadata: dict[str, object] = {
        "theta_deg": float(model.theta_deg),
        "n_shells": int(model.n_shells),
        "model_name": str(params.model_name),
        "mesh_size": int(data.mesh_size),
        "supercell_matrix": [
            [int(data.supercell.n11), int(data.supercell.n12)],
            [int(data.supercell.n21), int(data.supercell.n22)],
        ],
        "area_ratio": int(data.supercell.area_ratio),
        "primitive_projected_band_count": int(data.primitive_band_count),
        "interaction_epsilon_r": float(interaction.epsilon_r),
        "interaction_d_sc_nm": float(interaction.d_sc_nm),
        "interaction_U_ev": float(interaction.U_ev),
        "interaction_subtraction": str(interaction.subtraction),
        "interaction_g_shells": int(interaction.g_shells),
        "finite_zero_limit": bool(interaction.finite_zero_limit),
        "source": "mean_field.systems.htg.supercell",
    }
    return ContractSingleParticleModel(
        system="htg_supercell",
        lattice=model.lattice,
        params={
            "theta_deg": float(model.theta_deg),
            "n_shells": int(model.n_shells),
            "model_name": str(params.model_name),
            "kappa": float(params.kappa),
            "w_ev": float(params.w_ev),
            "vf_ev_nm": float(params.vf_ev_nm),
        },
        hamiltonian_builder=_unavailable_hamiltonian_builder,
        diagonalizer=_unavailable_diagonalizer,
        metadata=metadata,
    )


def _basis_energies_from_h0(h0: np.ndarray) -> np.ndarray:
    h0_array = np.asarray(h0, dtype=np.complex128)
    out = np.zeros((h0_array.shape[0], h0_array.shape[2]), dtype=float)
    for ik in range(h0_array.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(h0_array[:, :, ik])
    return out


def _flatten_k_grid_frac(data: HTGSupercellProjectedBasisData) -> np.ndarray:
    if data.k_grid_frac is None:
        raise ValueError("HTG supercell canonical ProjectedBasis requires basis_data.k_grid_frac")
    return np.asarray(data.k_grid_frac, dtype=float).reshape((data.nk, 2))


def _folded_band_labels(data: HTGSupercellProjectedBasisData) -> tuple[dict[str, object], ...]:
    primitive_indices = tuple(int(index) for index in data.primitive_projected_indices)
    fold_reps = tuple(tuple(int(v) for v in rep) for rep in data.fold_representatives)
    expected = len(primitive_indices) * len(fold_reps)
    if expected != int(data.basis.n_band):
        raise ValueError(
            "HTG supercell folded band count mismatch: "
            f"{len(primitive_indices)} primitive bands x {len(fold_reps)} folds != {data.basis.n_band}"
        )
    labels: list[dict[str, object]] = []
    for primitive_band_index in primitive_indices:
        for fold_index, fold_rep in enumerate(fold_reps):
            labels.append(
                {
                    "primitive_band_index": int(primitive_band_index),
                    "fold_index": int(fold_index),
                    "fold_representative": [int(fold_rep[0]), int(fold_rep[1])],
                }
            )
    return tuple(labels)


def _active_band_indices(data: HTGSupercellProjectedBasisData) -> tuple[int, ...]:
    band_labels = _folded_band_labels(data)
    n_spin = int(data.basis.n_spin)
    n_eta = int(data.basis.n_flavor)
    n_band = int(data.basis.n_band)
    state_labels = np.zeros((n_spin * n_eta * n_band,), dtype=int)
    state_index = np.arange(state_labels.size, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband, label in enumerate(band_labels):
                state_labels[int(state_index[ispin, ieta, iband])] = int(label["primitive_band_index"])
    return tuple(int(value) for value in state_labels)


def _flavor_labels(data: HTGSupercellProjectedBasisData) -> tuple[str, ...]:
    n_spin = int(data.basis.n_spin)
    n_eta = int(data.basis.n_flavor)
    n_band = int(data.basis.n_band)
    labels = [""] * (n_spin * n_eta * n_band)
    state_index = np.arange(len(labels), dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                labels[int(state_index[ispin, ieta, iband])] = f"spin{ispin}_eta{ieta}_band{iband}"
    return tuple(labels)


def _reference_scheme(data: HTGSupercellProjectedBasisData) -> str:
    reference = np.asarray(data.reference_diagonal, dtype=float).reshape(-1)
    return "average" if np.allclose(reference, 0.5, atol=1.0e-12, rtol=0.0) else "central_average"


def _projected_basis(data: HTGSupercellProjectedBasisData) -> ContractProjectedBasis:
    model = _single_particle_model(data)
    n_band = int(data.basis.n_band)
    active_valence = n_band // 2
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(data.kvec, dtype=np.complex128),
        k_grid_frac=_flatten_k_grid_frac(data),
        h0=np.asarray(data.h0, dtype=np.complex128),
        basis_energies=_basis_energies_from_h0(data.h0),
        active_band_indices=_active_band_indices(data),
        active_valence_bands=int(active_valence),
        active_conduction_bands=int(n_band - active_valence),
        micro_wavefunctions=np.asarray(data.basis.wavefunctions, dtype=np.complex128),
        flavor_labels=_flavor_labels(data),
        band_labels=_folded_band_labels(data),
        metadata={
            "projected_basis_source": "HTGSupercellProjectedBasisData",
            "wavefunctions_axis_order": "basis,band,flavor,k",
            "density_axis_order": "abk",
            "active_band_semantics": "primitive_projected_indices_repeated_over_supercell_folds_spin_valley",
            "primitive_projected_indices": [int(index) for index in data.primitive_projected_indices],
            "primitive_projected_band_count": int(data.primitive_band_count),
            "fold_representatives": [[int(a), int(b)] for a, b in data.fold_representatives],
            "supercell_matrix": [
                [int(data.supercell.n11), int(data.supercell.n12)],
                [int(data.supercell.n21), int(data.supercell.n22)],
            ],
            "area_ratio": int(data.supercell.area_ratio),
            "moire_supercell_area_nm2": float(data.moire_supercell_area_nm2),
            "reciprocal_grid_shape": [int(value) for value in data.reciprocal_grid_shape],
            "reciprocal_grid_origin": [int(value) for value in data.reciprocal_grid_origin],
        },
    )


def _reference_density(run: HTGSupercellHartreeFockRun) -> np.ndarray:
    state = run.state
    return _supercell_reference_density_blocks(
        state.nt,
        state.nk,
        reference_diagonal=state.reference_diagonal,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


def _density_state(run: HTGSupercellHartreeFockRun) -> ContractDensityState:
    data = run.basis_data
    state = run.state
    reference = _reference_density(run)
    area_ratio = int(data.supercell.area_ratio)
    return density_state_from_delta(
        state.density,
        reference,
        reference_scheme=_reference_scheme(data),
        filling=float(state.nu),
        n_occupied_total=int(
            htg_supercell_occupied_count_per_k(
                state.nu,
                reference_diagonal=state.reference_diagonal,
                area_ratio=area_ratio,
                n_sector=int(state.n_spin) * int(state.n_eta),
            )
            * int(state.nk)
        ),
        reference_metadata={
            "system": "htg_supercell",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_diagonal": [float(value) for value in np.asarray(state.reference_diagonal, dtype=float).reshape(-1)],
            "reference_scheme_source": "htg_supercell_reference_diagonal",
            "area_ratio": area_ratio,
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P-R",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.htg.supercell_contracts",
            "filling_from_density": float(
                htg_supercell_filling_from_density(
                    state.density,
                    reference_diagonal=state.reference_diagonal,
                    area_ratio=area_ratio,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
        },
    )


def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _hamiltonian_parts(run: HTGSupercellHartreeFockRun) -> ContractHamiltonianParts:
    h0 = np.asarray(run.state.h0, dtype=np.complex128)
    total = np.asarray(run.state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="htg_supercell_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "supports_crpa": False,
        },
    )


def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _iteration_history(run: HTGSupercellHartreeFockRun) -> list[dict[str, Any]]:
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

def _iteration_count(run: HTGSupercellHartreeFockRun) -> int:
    return max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))

def _supercell_matrix_metadata(data: HTGSupercellProjectedBasisData) -> list[list[int]]:
    return [
        [int(data.supercell.n11), int(data.supercell.n12)],
        [int(data.supercell.n21), int(data.supercell.n22)],
    ]

def _default_hf_config_from_run(run: HTGSupercellHartreeFockRun) -> "HFConfig":
    from mean_field.api.hf import HFConfig

    data = run.basis_data
    state = run.state
    interaction = data.interaction
    return HFConfig(
        filling=float(state.nu),
        mesh=(int(data.mesh_size), int(data.mesh_size)),
        interaction_scheme="average",
        density_convention="stored_delta",
        epsilon_r=float(interaction.epsilon_r),
        dsc_nm=float(interaction.d_sc_nm),
        coulomb_kernel="2d_gate",
        max_iter=max(_iteration_count(run), 1),
        precision=float(state.precision),
        seeds=(str(int(run.seed)),),
        metadata={
            "source": "derived_from_HTGSupercellHartreeFockRun",
            "max_iter_semantics": "observed_iteration_count_when_original_limit_is_unavailable",
            "init_mode": str(run.init_mode),
            "supercell_matrix": _supercell_matrix_metadata(data),
            "area_ratio": int(data.supercell.area_ratio),
            "primitive_projected_indices": [int(index) for index in data.primitive_projected_indices],
            "primitive_projected_band_count": int(data.primitive_band_count),
            "interaction_subtraction": str(interaction.subtraction),
            "interaction_g_shells": int(interaction.g_shells),
            "interaction_n_k": int(interaction.n_k),
        },
    )

def _validate_hf_config_matches_run(config: "HFConfig", run: HTGSupercellHartreeFockRun) -> None:
    mesh = (int(run.basis_data.mesh_size), int(run.basis_data.mesh_size))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"HTG supercell HFResult config.mesh must match raw mesh_size {mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(run.state.nu)):
        raise ValueError(
            f"HTG supercell HFResult config.filling={config.filling} does not match raw primitive_nu={run.state.nu}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "HTG supercell raw density is stored as P-R; use HFConfig.density_convention='stored_delta'"
        )

def _result_observables(run: HTGSupercellHartreeFockRun) -> dict[str, object]:
    state = run.state
    data = run.basis_data
    return {
        "primitive_nu": float(state.nu),
        "filling_from_density": float(
            htg_supercell_filling_from_density(
                state.density,
                reference_diagonal=state.reference_diagonal,
                area_ratio=data.supercell.area_ratio,
                n_spin=state.n_spin,
                n_eta=state.n_eta,
            )
        ),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "init_mode": str(run.init_mode),
        "seed": int(run.seed),
        "iterations": int(_iteration_count(run)),
        "supercell_area_ratio": int(data.supercell.area_ratio),
        "raw_density_convention": "stored_delta",
    }


def htg_supercell_hf_run_to_hf_run_result(
    run: HTGSupercellHartreeFockRun,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap an HTG supercell HF run in canonical core contracts.

    The raw HTG run remains the source of truth.  This adapter preserves the
    stored density delta ``P-R`` and creates a typed I/O view with collapsed
    Hamiltonian parts.  It does not recompute HF, split Hartree/Fock components,
    reconstruct final active eigenvectors, or claim cRPA support.
    """

    state = run.state
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
                htg_supercell_filling_from_density(
                    state.density,
                    reference_diagonal=state.reference_diagonal,
                    area_ratio=run.basis_data.supercell.area_ratio,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
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


def htg_supercell_hf_run_to_hf_result(
    run: HTGSupercellHartreeFockRun,
    *,
    config: "HFConfig | None" = None,
    archive_manifest: Mapping[str, Any] | None = None,
    observables: Mapping[str, object] | None = None,
) -> "HFResult":
    """Return a public :class:`HFResult` view of an existing HTG supercell run.

    The raw :class:`HTGSupercellHartreeFockRun` remains ``HFResult.state`` and
    the source of truth.  The attached ``canonical_run_result`` is the canonical
    I/O view produced by :func:`htg_supercell_hf_run_to_hf_run_result`; no SCF,
    interaction, topology, or cRPA calculation is rerun here.
    """

    from pathlib import Path

    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    resolved_config = _default_hf_config_from_run(run) if config is None else config
    _validate_hf_config_matches_run(resolved_config, run)
    canonical = htg_supercell_hf_run_to_hf_run_result(
        run,
        archive_manifest=None if archive_manifest is None else dict(archive_manifest),
    )
    result_observables = _result_observables(run)
    if observables is not None:
        result_observables.update(dict(observables))
    record = model_record(run.basis_data.model, system_name="htg_supercell")
    return HFResult(
        model=record,
        config=resolved_config,
        state=run,
        observables=result_observables,
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="eV",
                density_convention="stored_delta",
                density_axis_order="abk",
                hamiltonian_axis_order="abk",
                wavefunction_axis_order="basis,band,flavor,k",
                gauge="htg_supercell_projected_basis_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "htg.supercell_hf.raw_run_result",
                "system_name": "htg_supercell",
                "adapter": "mean_field.systems.htg.supercell_contracts.htg_supercell_hf_run_to_hf_result",
                "canonical_adapter": "mean_field.systems.htg.supercell_contracts.htg_supercell_hf_run_to_hf_run_result",
                "raw_state_type": type(run).__name__,
            },
        ),
        canonical_run_result=canonical,
    )

__all__ = ["htg_supercell_hf_run_to_hf_result", "htg_supercell_hf_run_to_hf_run_result"]
