from __future__ import annotations

"""Canonical mean-field contract adapters for RnG/hBN HF runs.

The functions here are post-run I/O adapters.  They wrap arrays already produced
by the existing RnG/hBN Hartree-Fock path and do not change SCF, screening,
interaction contractions, topology, or cRPA behavior.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
from mean_field.core.hf.contracts_bridge import density_state_from_delta, float_diagnostics

from .hf import (
    RLGhBNHartreeFockRun,
    RLGhBNModel,
    RLGhBNProjectedBasisData,
    build_rlg_hbn_projected_basis,
    rlg_hbn_filling_from_density,
    rlg_hbn_occupied_state_count,
    run_rlg_hbn_hartree_fock,
)
from .interaction import RLGhBNInteractionParams


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


@dataclass(frozen=True)
class RLGhBNRunHFConfig:
    """Explicit public config for RnG/hBN ``run_hf`` dispatch.

    This mirrors the existing system-owned RnG/hBN runner.  The public
    :class:`mean_field.api.hf.HFConfig` must still match filling, mesh,
    iteration controls, density convention, and interaction scalars; no generic
    ``HFConfig -> RnG/hBN`` inference is performed here.
    """

    interaction: RLGhBNInteractionParams = field(default_factory=RLGhBNInteractionParams)
    nu: float = 1.0
    mesh_size: int | None = None
    init_mode: str = "flavor"
    seed: int = 1
    beta: float = 1.0
    max_iter: int = 80
    precision: float = 1.0e-6
    oda_stall_threshold: float = 1.0e-3
    occupation_counts: tuple[int, ...] | None = None
    initial_density: np.ndarray | None = None
    frac_shift: tuple[float, float] = (0.0, 0.0)
    valleys: tuple[int, ...] = (1, -1)
    screening_mesh_size: int | None = None
    screening_max_iter: int = 50
    screening_tolerance_mev: float = 1.0e-6
    screening_mixing: float = 0.5
    screening_solver: str = "fixed_point"
    screening_u_min_mev: float = -100.0
    screening_u_max_mev: float = 200.0
    screening_u_grid_points: int = 121
    screening_root_tolerance_mev: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.mesh_size is not None and int(self.mesh_size) <= 0:
            raise ValueError(f"mesh_size must be positive when provided, got {self.mesh_size}")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if float(self.precision) <= 0.0:
            raise ValueError("precision must be positive")
        if float(self.oda_stall_threshold) <= 0.0:
            raise ValueError("oda_stall_threshold must be positive")
        if len(tuple(self.valleys)) == 0:
            raise ValueError("at least one valley is required")
        if len(tuple(self.frac_shift)) != 2:
            raise ValueError(f"frac_shift must have length 2, got {self.frac_shift}")
        if self.screening_mesh_size is not None and int(self.screening_mesh_size) <= 0:
            raise ValueError("screening_mesh_size must be positive when provided")
        if int(self.screening_max_iter) <= 0:
            raise ValueError("screening_max_iter must be positive")
        if float(self.screening_tolerance_mev) <= 0.0:
            raise ValueError("screening_tolerance_mev must be positive")
        if not (0.0 < float(self.screening_mixing) <= 1.0):
            raise ValueError("screening_mixing must lie in (0, 1]")
        if int(self.screening_u_grid_points) <= 0:
            raise ValueError("screening_u_grid_points must be positive")
        if float(self.screening_root_tolerance_mev) <= 0.0:
            raise ValueError("screening_root_tolerance_mev must be positive")

    @property
    def resolved_mesh_size(self) -> int:
        return int(self.interaction.k_mesh_size if self.mesh_size is None else self.mesh_size)


def _rlg_hbn_coulomb_kernel_name(interaction: RLGhBNInteractionParams) -> str:
    return "2d_gate" if interaction.interaction_dimension == "2d_diagnostic" else "3d_layered"


def _validate_rlg_hbn_public_hf_config(config: "HFConfig", rlg_config: RLGhBNRunHFConfig) -> None:
    if not isinstance(rlg_config.interaction, RLGhBNInteractionParams):
        raise TypeError(
            "rlg_hbn_config.interaction must be RLGhBNInteractionParams, "
            f"got {type(rlg_config.interaction).__name__}"
        )
    mesh = (int(rlg_config.resolved_mesh_size), int(rlg_config.resolved_mesh_size))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"RnG/hBN public run_hf requires HFConfig.mesh={mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(rlg_config.nu)):
        raise ValueError(f"RnG/hBN public run_hf requires HFConfig.filling={rlg_config.nu}, got {config.filling}")
    if int(config.max_iter) != int(rlg_config.max_iter):
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.max_iter={rlg_config.max_iter}, got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(rlg_config.precision)):
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.precision={rlg_config.precision}, got {config.precision}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "RnG/hBN HF stores density as P-R; set HFConfig.density_convention='stored_delta'"
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "RnG/hBN public run_hf takes the projected window from rlg_hbn_config.interaction; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )
    interaction = rlg_config.interaction
    if config.interaction_scheme != interaction.scheme:
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.interaction_scheme={interaction.scheme!r}, "
            f"got {config.interaction_scheme!r}"
        )
    expected_kernel = _rlg_hbn_coulomb_kernel_name(interaction)
    if config.coulomb_kernel != expected_kernel:
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.coulomb_kernel={expected_kernel!r} "
            f"for interaction_dimension={interaction.interaction_dimension!r}, got {config.coulomb_kernel!r}"
        )
    if not np.isclose(float(config.epsilon_r), float(interaction.epsilon_r)):
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.epsilon_r={interaction.epsilon_r}, got {config.epsilon_r}"
        )
    if not np.isclose(float(config.dsc_nm), float(interaction.gate_distance_nm)):
        raise ValueError(
            f"RnG/hBN public run_hf requires HFConfig.dsc_nm={interaction.gate_distance_nm}, got {config.dsc_nm}"
        )


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
        diagnostics=float_diagnostics(state.diagnostics),
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



def _default_hf_config_from_run(run: RLGhBNHartreeFockRun) -> "HFConfig":
    from mean_field.api.hf import HFConfig

    interaction = run.basis_data.interaction
    iteration_count = max(1, len(run.iter_err))
    return HFConfig(
        filling=float(run.state.nu),
        mesh=(int(run.basis_data.mesh_size), int(run.basis_data.mesh_size)),
        interaction_scheme=str(interaction.scheme),  # type: ignore[arg-type]
        density_convention="stored_delta",
        epsilon_r=float(interaction.epsilon_r),
        dsc_nm=float(interaction.gate_distance_nm),
        coulomb_kernel=_rlg_hbn_coulomb_kernel_name(interaction),  # type: ignore[arg-type]
        max_iter=iteration_count,
        precision=float(run.state.precision),
        seeds=(str(int(run.seed)),),
        metadata={
            "source": "derived_from_RLGhBNHartreeFockRun",
            "max_iter_semantics": "observed_iteration_count_when_original_limit_is_unavailable",
            "init_mode": str(run.init_mode),
            "interaction_dimension": str(interaction.interaction_dimension),
            "active_valence_bands": int(interaction.active_valence_bands),
            "active_conduction_bands": int(interaction.active_conduction_bands),
            "use_screened_basis": bool(interaction.use_screened_basis),
        },
    )


def _validate_hf_config_matches_run(config: "HFConfig", run: RLGhBNHartreeFockRun) -> None:
    mesh = (int(run.basis_data.mesh_size), int(run.basis_data.mesh_size))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"RnG/hBN HFResult config.mesh must match raw mesh_size {mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(run.state.nu)):
        raise ValueError(
            f"RnG/hBN HFResult config.filling={config.filling} does not match raw nu={run.state.nu}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError("RnG/hBN raw density is stored as P-R; use HFConfig.density_convention='stored_delta'")


def _result_observables(run: RLGhBNHartreeFockRun) -> dict[str, object]:
    state = run.state
    return {
        "primitive_nu": float(state.nu),
        "filling_from_density": float(
            rlg_hbn_filling_from_density(
                state.density,
                state.reference_density,
                active_valence_bands=state.active_valence_bands,
                n_spin=state.n_spin,
                n_eta=state.n_eta,
            )
        ),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "init_mode": str(run.init_mode),
        "seed": int(run.seed),
        "iterations": int(max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))),
        "raw_density_convention": "stored_delta",
        "screening_available": run.basis_data.screening is not None,
    }


def rlg_hbn_hf_run_to_hf_result(
    run: RLGhBNHartreeFockRun,
    *,
    config: "HFConfig | None" = None,
    archive_manifest: Mapping[str, Any] | None = None,
    observables: Mapping[str, object] | None = None,
) -> "HFResult":
    """Return a public :class:`HFResult` view of an existing RnG/hBN HF run.

    The raw :class:`RLGhBNHartreeFockRun` remains ``HFResult.state`` and the
    source of truth.  The attached ``canonical_run_result`` is the canonical I/O
    view produced by :func:`rlg_hbn_hf_run_to_hf_run_result`; no SCF,
    interaction, topology, or cRPA calculation is rerun here.
    """

    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    resolved_config = _default_hf_config_from_run(run) if config is None else config
    _validate_hf_config_matches_run(resolved_config, run)
    canonical = rlg_hbn_hf_run_to_hf_run_result(
        run,
        archive_manifest=None if archive_manifest is None else dict(archive_manifest),
    )
    result_observables = _result_observables(run)
    if observables is not None:
        result_observables.update(dict(observables))
    record = model_record(run.basis_data.model, system_name="rlg_hbn")
    return HFResult(
        model=record,
        config=resolved_config,
        state=run,
        observables=result_observables,
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="meV",
                density_convention="stored_delta",
                density_axis_order="abk",
                hamiltonian_axis_order="abk",
                wavefunction_axis_order="basis,band,flavor,k",
                gauge="rlg_hbn_projected_basis_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "rlg_hbn.hf.raw_run_result",
                "system_name": "rlg_hbn",
                "adapter": "mean_field.systems.RnG_hBN.hf_contracts.rlg_hbn_hf_run_to_hf_result",
                "canonical_adapter": "mean_field.systems.RnG_hBN.hf_contracts.rlg_hbn_hf_run_to_hf_run_result",
                "raw_state_type": type(run).__name__,
            },
        ),
        canonical_run_result=canonical,
    )


def run_rlg_hbn_hf_config_adapter(model: object, config: "HFConfig", **kwargs: Any) -> "HFResult | None":
    """Run RnG/hBN HF from an explicit system config.

    The adapter is intentionally narrow: callers must provide
    ``rlg_hbn_config=RLGhBNRunHFConfig(...)`` and a matching public
    ``HFConfig``.  The raw :class:`RLGhBNHartreeFockRun` remains the source of
    truth and is wrapped by the canonical RnG/hBN post-run adapter.
    """

    if not isinstance(model, RLGhBNModel):
        return None
    if "rlg_hbn_config" not in kwargs:
        raise NotImplementedError(
            "Unified run_hf has an RnG/hBN adapter only for explicit "
            "rlg_hbn_config=RLGhBNRunHFConfig(...); generic HFConfig -> RnG/hBN runner mapping is not implemented"
        )
    rlg_config = kwargs.pop("rlg_hbn_config")
    if not isinstance(rlg_config, RLGhBNRunHFConfig):
        raise TypeError(f"rlg_hbn_config must be RLGhBNRunHFConfig, got {type(rlg_config).__name__}")
    if kwargs:
        raise TypeError(f"Unsupported RnG/hBN run_hf kwargs: {sorted(kwargs)}")

    _validate_rlg_hbn_public_hf_config(config, rlg_config)
    basis_data = build_rlg_hbn_projected_basis(
        model,
        rlg_config.interaction,
        mesh_size=rlg_config.resolved_mesh_size,
        frac_shift=tuple(float(value) for value in rlg_config.frac_shift),
        valleys=tuple(int(value) for value in rlg_config.valleys),
        screening_mesh_size=rlg_config.screening_mesh_size,
        screening_max_iter=int(rlg_config.screening_max_iter),
        screening_tolerance_mev=float(rlg_config.screening_tolerance_mev),
        screening_mixing=float(rlg_config.screening_mixing),
        screening_solver=str(rlg_config.screening_solver),
        screening_u_min_mev=float(rlg_config.screening_u_min_mev),
        screening_u_max_mev=float(rlg_config.screening_u_max_mev),
        screening_u_grid_points=int(rlg_config.screening_u_grid_points),
        screening_root_tolerance_mev=float(rlg_config.screening_root_tolerance_mev),
    )
    raw = run_rlg_hbn_hartree_fock(
        basis_data,
        nu=float(rlg_config.nu),
        init_mode=str(rlg_config.init_mode),
        seed=int(rlg_config.seed),
        beta=float(rlg_config.beta),
        max_iter=int(rlg_config.max_iter),
        precision=float(rlg_config.precision),
        oda_stall_threshold=float(rlg_config.oda_stall_threshold),
        occupation_counts=rlg_config.occupation_counts,
        initial_density=rlg_config.initial_density,
    )
    return rlg_hbn_hf_run_to_hf_result(
        raw,
        config=config,
        observables={
            "public_run_hf_adapter": "mean_field.systems.RnG_hBN.hf_contracts.run_rlg_hbn_hf_config_adapter",
            "explicit_config_type": "RLGhBNRunHFConfig",
        },
    )


__all__ = [
    "RLGhBNRunHFConfig",
    "rlg_hbn_hf_run_to_hf_result",
    "rlg_hbn_hf_run_to_hf_run_result",
    "run_rlg_hbn_hf_config_adapter",
]
