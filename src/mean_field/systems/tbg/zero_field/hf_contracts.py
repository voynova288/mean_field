from __future__ import annotations

"""Canonical mean-field contract adapters for TBG zero-field HF runs.

The functions here are post-run I/O adapters.  They wrap arrays already produced
by the existing TBG zero-field B0/BM workflow and do not change the SCF loop,
interaction contractions, topology, path reconstruction, or cRPA behavior.

A bare :class:`RestrictedHartreeFockRun` is not self-describing enough for the
canonical projected-basis contract: it lacks the k-grid coordinates and BM
micro-wavefunctions.  The safe boundary therefore requires the matching
``grid_solution`` (or a higher-level ``B0HFBenchmarkRun`` that owns it) and
requires the carried canonical half-open torus mesh before creating the
canonical view. Endpoint-inclusive B0 meshes remain legacy diagnostics only.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
import math

import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
    assert_density_state_consistent,
)
from mean_field.core.hf.contracts_bridge import density_state_from_delta

from .hf import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    TBGZeroFieldHFRunProvenance,
    TBGZeroFieldHFSourceReceipt,
    TBGZeroFieldScreenedBlockBundle,
    build_tbg_zero_field_screened_block_bundle,
    normalize_full_init_mode,
    normalize_restricted_init_mode,
    restricted_filling,
    restricted_occupied_state_count,
    run_restricted_hartree_fock,
    tbg_zero_field_lattice_kvec_sha256,
)
from ._hf_basis_overlap import (
    validate_tbg_zero_field_primitive_cell_nu,
    validate_tbg_zero_field_seed,
    validate_tbg_zero_field_typed_bm_lg,
    validate_tbg_zero_field_typed_hf_source,
    validate_tbg_zero_field_typed_max_iter,
    validate_tbg_zero_field_typed_overlap_lg,
)
from .interaction import (
    TBGZeroFieldInteractionSpec,
    TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
)
from .model import (
    BMSolution,
    TBGZeroFieldBMModel,
    TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
    TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
    _b0_real_to_nm,
    _b0_reciprocal_to_nm_inv,
    tbg_zero_field_bm_generation_fingerprint,
)

_TBG_RUN_FLOAT_RTOL = 1.0e-13
_TBG_RUN_FLOAT_ATOL = 0.0
_TBG_RUN_FILLING_ATOL = 1.0e-12
_TBG_DENSITY_PROJECTOR_TOL = 1.0e-10

def _numeric_values_match(
    left: object,
    right: object,
    *,
    atol: float = _TBG_RUN_FLOAT_ATOL,
) -> bool:
    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=_TBG_RUN_FLOAT_RTOL,
            atol=atol,
        )
    )


def _unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "TBG zero-field contract records an already-built BM projected basis; "
        "use mean_field.systems.tbg.zero_field.solve_bm_model for fresh Hamiltonians."
    )


def _unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "TBG zero-field contract records post-run arrays; "
        "fresh BM diagonalization is not performed by the adapter."
    )


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _hf_source_receipt_metadata(
    state: RestrictedHartreeFockState,
) -> dict[str, object] | None:
    receipt = state.hf_source_receipt
    if receipt is None:
        return None
    if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
        raise TypeError("hf_source_receipt must be a TBGZeroFieldHFSourceReceipt")
    return receipt.to_metadata()


def _archive_manifest_with_hf_source_receipt(
    archive_manifest: Mapping[str, Any] | None,
    run: RestrictedHartreeFockRun,
) -> dict[str, Any]:
    resolved = dict(archive_manifest or {})
    state = run.state
    receipt_metadata = _hf_source_receipt_metadata(state)
    if receipt_metadata is None:
        return resolved
    existing = resolved.get("hf_source_receipt")
    if existing is not None and existing != receipt_metadata:
        raise ValueError(
            "archive_manifest hf_source_receipt conflicts with the typed HF state receipt"
        )
    resolved["hf_source_receipt"] = receipt_metadata
    interaction_spec = state.interaction_spec
    if interaction_spec is not None:
        if not isinstance(interaction_spec, TBGZeroFieldInteractionSpec):
            raise TypeError("state.interaction_spec must be TBGZeroFieldInteractionSpec")
        if (
            not isinstance(state.hf_source_receipt, TBGZeroFieldHFSourceReceipt)
            or state.hf_source_receipt.interaction_spec_fingerprint
            != interaction_spec.fingerprint
        ):
            raise ValueError(
                "typed interaction_spec does not match the HF source receipt fingerprint"
            )
        spec_metadata = interaction_spec.to_metadata()
        existing_spec = resolved.get("interaction_spec")
        if existing_spec is not None and existing_spec != spec_metadata:
            raise ValueError(
                "archive_manifest interaction_spec conflicts with the typed HF state spec"
            )
        resolved["interaction_spec"] = spec_metadata
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("Typed TBG zero-field archive export requires immutable run provenance")
    provenance_metadata = provenance.to_metadata()
    existing_provenance = resolved.get("hf_run_provenance")
    if existing_provenance is not None and existing_provenance != provenance_metadata:
        raise ValueError(
            "archive_manifest hf_run_provenance conflicts with the typed HF run"
        )
    resolved["hf_run_provenance"] = provenance_metadata
    return resolved


def _bm_source_attestation_metadata(solution: BMSolution) -> dict[str, object] | None:
    attestation = solution.source_attestation
    return None if attestation is None else attestation.to_metadata()


def _single_particle_model(solution: BMSolution) -> ContractSingleParticleModel:
    params = solution.params
    return ContractSingleParticleModel(
        system="tbg_zero_field",
        lattice={
            "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "g1_b0_code_pair": _complex_pair(params.g1),
            "g2_b0_code_pair": _complex_pair(params.g2),
            "a1_b0_code_pair": _complex_pair(params.a1),
            "a2_b0_code_pair": _complex_pair(params.a2),
            "kt_b0_code_pair": _complex_pair(params.kt),
            "kb_point_b0_code_pair": _complex_pair(params.kb_point),
            "g1_nm_inv_pair": _complex_pair(_b0_reciprocal_to_nm_inv(params.g1)),
            "g2_nm_inv_pair": _complex_pair(_b0_reciprocal_to_nm_inv(params.g2)),
            "a1_nm_pair": _complex_pair(_b0_real_to_nm(params.a1)),
            "a2_nm_pair": _complex_pair(_b0_real_to_nm(params.a2)),
            "theta12_rad": float(params.theta12),
            "kt_nm_inv_pair": _complex_pair(_b0_reciprocal_to_nm_inv(params.kt)),
            "kb_point_nm_inv_pair": _complex_pair(_b0_reciprocal_to_nm_inv(params.kb_point)),
        },
        params={
            "dtheta_rad": float(params.dtheta_rad),
            "convention": str(params.convention),
            "vf": float(params.vf),
            "chemical_potential": float(params.chemical_potential),
            "w0": float(params.w0),
            "w1": float(params.w1),
            "delta": float(params.delta),
            "strain": float(params.strain),
            "strain_angle_rad": float(params.strain_angle_rad),
            "poisson": float(params.poisson),
            "beta_g": float(params.beta_g),
            "alpha": float(params.alpha),
            "deformation_potential": float(params.deformation_potential),
        },
        hamiltonian_builder=_unavailable_hamiltonian_builder,
        diagonalizer=_unavailable_diagonalizer,
        metadata={
            "source": "mean_field.systems.tbg.zero_field.BMSolution",
            "model_name": "zero_field_bm",
            "lg": int(solution.lg),
            "nlocal": int(solution.nlocal),
            "n_eta": int(solution.n_eta),
            "n_spin": int(solution.n_spin),
            "nb": int(solution.nb),
            "sigma_rotation": bool(solution.sigma_rotation),
            "calculate_chern_operator": bool(solution.calculate_chern_operator),
            "periodic_g_grid": bool(solution.periodic_g_grid),
            "bm_generation_fingerprint": solution.generation_fingerprint,
            "bm_source_attestation": _bm_source_attestation_metadata(solution),
            "supports_crpa": False,
        },
    )


def _carried_torus_mesh(solution: BMSolution):
    mesh = solution.torus_mesh
    if mesh is None:
        raise ValueError(
            "TBG zero-field canonical adapter requires BMSolution.torus_mesh from "
            "solve_bm_model_on_torus; endpoint-inclusive B0 is an explicit legacy diagnostic only"
        )
    actual_kvec = np.asarray(solution.lattice_kvec, dtype=np.complex128).reshape(-1)
    if not np.array_equal(actual_kvec, mesh.kvec):
        raise ValueError("grid_solution.lattice_kvec does not match its carried torus mesh")
    return mesh

def _legacy_diagnostic_infer_b0_lk(solution: BMSolution) -> int:
    nk = int(solution.nk)
    side = int(round(math.sqrt(nk)))
    if side * side != nk:
        raise ValueError(
            "Legacy endpoint-inclusive B0 diagnostic requires a square uniform grid_solution; "
            f"got grid_solution.nk={nk}.  A bare RestrictedHartreeFockRun does not carry enough k-grid metadata."
        )
    lk = side - 1
    if lk <= 0:
        raise ValueError(
            "Legacy endpoint-inclusive B0 diagnostic requires lk >= 1; "
            f"got inferred lk={lk} from grid_solution.nk={nk}."
        )
    return lk


def _legacy_diagnostic_b0_uniform_k_grid_frac(solution: BMSolution) -> np.ndarray:
    """Endpoint-inclusive B0 coordinates retained only for legacy diagnostics."""

    lk = _legacy_diagnostic_infer_b0_lk(solution)
    frac = np.arange(lk + 1, dtype=float) / float(lk)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    k_grid_frac = np.stack([np.ravel(f1, order="F"), np.ravel(f2, order="F")], axis=1)
    expected_kvec = np.ravel(
        frac[:, None] * solution.params.g1 + frac[None, :] * solution.params.g2,
        order="F",
    ).astype(np.complex128)
    actual_kvec = np.asarray(solution.lattice_kvec, dtype=np.complex128).reshape(-1)
    if actual_kvec.shape != expected_kvec.shape or not np.allclose(actual_kvec, expected_kvec, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "Legacy endpoint-inclusive B0 diagnostic requires grid_solution.lattice_kvec to match "
            f"the B0 uniform mesh inferred from nk={solution.nk} (lk={lk}); received a non-uniform or reordered grid."
        )
    return k_grid_frac


def _central_bm_band_indices(solution: BMSolution) -> tuple[int, ...]:
    dim = int(solution.nlocal) * int(solution.lg) * int(solution.lg)
    start = dim // 2 - 1
    return tuple(range(start, start + int(solution.nb)))


def _active_band_indices(solution: BMSolution) -> tuple[int, ...]:
    central = _central_bm_band_indices(solution)
    labels: list[int] = []
    for band_index in central:
        for _ieta in range(int(solution.n_eta)):
            for _ispin in range(int(solution.n_spin)):
                labels.append(int(band_index))
    return tuple(labels)


def _flavor_labels(solution: BMSolution) -> tuple[str, ...]:
    labels: list[str] = []
    valley_labels = ("K", "Kprime")
    for iband in range(int(solution.nb)):
        for ieta in range(int(solution.n_eta)):
            valley = valley_labels[ieta] if ieta < len(valley_labels) else f"eta{ieta}"
            for ispin in range(int(solution.n_spin)):
                labels.append(f"spin{ispin}_{valley}_bm_band{iband}")
    return tuple(labels)


def _band_labels(solution: BMSolution) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "bm_window_index": int(index),
            "full_bm_matrix_band_index": int(band_index),
        }
        for index, band_index in enumerate(_central_bm_band_indices(solution))
    )


def _validate_solution_matches_state(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> None:
    state = run.state
    if int(state.nk) != int(grid_solution.nk):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.nk to match grid_solution.nk; "
            f"got {state.nk} and {grid_solution.nk}."
        )
    if int(state.nt) != int(grid_solution.nt):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.nt to match grid_solution.nt; "
            f"got {state.nt} and {grid_solution.nt}."
        )
    if int(state.n_spin) != int(grid_solution.n_spin) or int(state.n_eta) != int(grid_solution.n_eta):
        raise ValueError(
            "TBG zero-field canonical adapter requires matching spin/valley dimensions between hf_run and grid_solution; "
            f"got state (spin={state.n_spin}, eta={state.n_eta}) and "
            f"grid_solution (spin={grid_solution.n_spin}, eta={grid_solution.n_eta})."
        )
    if int(state.n_band) != int(grid_solution.nb):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.n_band to match grid_solution.nb; "
            f"got {state.n_band} and {grid_solution.nb}."
        )
    expected_uk_shape = (
        int(grid_solution.nlocal) * int(grid_solution.lg) * int(grid_solution.lg),
        int(grid_solution.nb),
        int(grid_solution.n_eta),
        int(grid_solution.nk),
    )
    if np.asarray(grid_solution.uk).shape != expected_uk_shape:
        raise ValueError(
            "TBG zero-field canonical adapter requires grid_solution.uk with raw BM shape "
            f"{expected_uk_shape}; got {np.asarray(grid_solution.uk).shape}."
        )

    flattened = np.asarray(grid_solution.flattened_energies(), dtype=float)
    h0 = np.asarray(state.h0, dtype=np.complex128)
    expected_h0 = np.zeros_like(h0)
    for ik in range(int(grid_solution.nk)):
        np.fill_diagonal(expected_h0[:, :, ik], flattened[:, ik])
    if h0.shape != expected_h0.shape or not np.allclose(h0, expected_h0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "TBG zero-field canonical adapter cannot safely combine hf_run.state.h0 with grid_solution: "
            "state.h0 is not the diagonal BM h0 built from grid_solution.flattened_energies()."
        )


def _validate_typed_run_source(
    run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
) -> tuple[TBGZeroFieldInteractionSpec, TBGZeroFieldScreenedBlockBundle, TBGZeroFieldHFSourceReceipt]:
    interaction_spec = run.state.interaction_spec
    bundle = run.screened_block_bundle
    receipt = run.state.hf_source_receipt
    if not isinstance(interaction_spec, TBGZeroFieldInteractionSpec):
        raise ValueError("Canonical TBG zero-field export requires a typed interaction spec")
    if not isinstance(bundle, TBGZeroFieldScreenedBlockBundle):
        raise ValueError("Canonical TBG zero-field export requires the screened-block bundle")
    if not isinstance(receipt, TBGZeroFieldHFSourceReceipt) or receipt.interaction_contract != "typed":
        raise ValueError("Canonical TBG zero-field export requires a typed HF source receipt")
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("Canonical TBG zero-field export requires immutable typed run provenance")
    provenance._validate_solver_issued_live_run(run)
    validate_tbg_zero_field_typed_hf_source(
        run.state,
        grid_solution,
        bundle,
        overlap_blocks=run.overlap_blocks,
        lattice_kvec=grid_solution.lattice_kvec,
        params=grid_solution.params,
    )
    if interaction_spec.fingerprint != bundle.interaction_spec.fingerprint:
        raise ValueError("Typed HF state interaction_spec does not match the screened bundle")
    if receipt.v0 != float(run.state.v0):
        raise ValueError("Typed HF receipt v0 does not match run.state.v0 exactly")
    if provenance.hf_mode != receipt.hf_mode:
        raise ValueError("Typed HF run provenance hf_mode does not match its source receipt")
    if provenance.beta != receipt.beta:
        raise ValueError("Typed HF run provenance beta does not match its source receipt exactly")
    diagnostic_beta = run.state.diagnostics.get("beta")
    if diagnostic_beta is None or float(diagnostic_beta) != provenance.beta:
        raise ValueError("Typed HF run provenance beta does not match run.state diagnostics exactly")
    state_checks = (
        (
            "state.nu",
            run.state.nu,
            provenance.nu,
            _TBG_RUN_FILLING_ATOL,
        ),
        (
            "state.precision",
            run.state.precision,
            provenance.precision,
            _TBG_RUN_FLOAT_ATOL,
        ),
    )
    for name, actual, expected_value, atol in state_checks:
        if not _numeric_values_match(actual, expected_value, atol=atol):
            raise ValueError(
                f"Typed HF run provenance {name} does not match within "
                f"rtol={_TBG_RUN_FLOAT_RTOL:g}, atol={atol:g}"
            )
    diagnostic_oda_stall_threshold = run.state.diagnostics.get("oda_stall_threshold")
    if diagnostic_oda_stall_threshold is None or not _numeric_values_match(
        diagnostic_oda_stall_threshold,
        provenance.oda_stall_threshold,
    ):
        raise ValueError(
            "Typed HF run provenance oda_stall_threshold does not match "
            "run.state diagnostics within the explicit numeric tolerance"
        )
    if provenance.seed != int(run.seed):
        raise ValueError("Typed HF run provenance seed does not match run.seed exactly")
    normalizer = normalize_full_init_mode if provenance.hf_mode == "full" else normalize_restricted_init_mode
    normalized_run_init_mode = normalizer(str(run.init_mode))
    if (
        str(run.init_mode) != normalized_run_init_mode
        or provenance.normalized_init_mode != normalized_run_init_mode
    ):
        raise ValueError(
            "Typed HF run provenance normalized init_mode does not match run.init_mode exactly"
        )
    diagnostic_max_iterations = run.state.diagnostics.get("requested_max_iterations")
    if (
        diagnostic_max_iterations is None
        or int(diagnostic_max_iterations) != provenance.requested_max_iterations
        or float(diagnostic_max_iterations) != float(int(diagnostic_max_iterations))
    ):
        raise ValueError(
            "Typed HF run provenance requested_max_iterations does not match "
            "run.state diagnostics exactly"
        )
    observed_iterations = max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))
    if observed_iterations > provenance.requested_max_iterations:
        raise ValueError(
            "Typed HF observed iteration count exceeds provenance requested_max_iterations"
        )
    if provenance.typed_receipt_fingerprint != receipt.fingerprint:
        raise ValueError("Typed HF run provenance receipt fingerprint does not match the live receipt")
    if provenance.interaction_spec_fingerprint != interaction_spec.fingerprint:
        raise ValueError("Typed HF run provenance interaction fingerprint does not match the live spec")
    if provenance.bm_generation_fingerprint != grid_solution.generation_fingerprint:
        raise ValueError(
            "Typed HF run provenance BM generation fingerprint does not match BMSolution"
        )
    mesh = _carried_torus_mesh(grid_solution)
    if provenance.mesh_fingerprint != mesh.fingerprint:
        raise ValueError("Typed HF run provenance mesh fingerprint does not match BMSolution")
    if run.overlap_blocks is not bundle.screened_blocks:
        raise ValueError("Returned HF run does not carry the exact screened bundle blocks")
    expected = {
        "active_shift_inventory": bundle.active_shifts,
        "active_shift_inventory_sha256": bundle.active_shift_inventory_sha256,
        "bm_generation_fingerprint": bundle.bm_generation_fingerprint,
        "bm_solution_sha256": bundle.bm_solution_sha256,
        "companion_circular_total_q_cutoff_parity": bundle.companion_circular_total_q_cutoff_parity,
        "interaction_spec_fingerprint": interaction_spec.fingerprint,
        "mesh_fingerprint": bundle.mesh_fingerprint,
        "n_band": bundle.n_band,
        "overlap_kernel_inventory_sha256": bundle.overlap_kernel_inventory_sha256,
        "overlap_lg": bundle.overlap_lg,
        "reference_projector_convention": bundle.reference_projector_convention,
        "reference_projector_dimensions": bundle.reference_projector_dimensions,
        "reference_projector_sha256": bundle.reference_projector_sha256,
        "screened_block_bundle_sha256": bundle.fingerprint,
    }
    mismatched = [name for name, value in expected.items() if getattr(receipt, name) != value]
    if mismatched:
        raise ValueError(
            "Typed HF receipt fields do not match the screened bundle: "
            f"{sorted(mismatched)}"
        )
    if receipt.lattice_kvec_sha256 != tbg_zero_field_lattice_kvec_sha256(
        grid_solution.lattice_kvec
    ):
        raise ValueError("Typed HF receipt lattice does not match BMSolution")
    return interaction_spec, bundle, receipt

def validate_tbg_zero_field_typed_hf_run_source(
    run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
) -> tuple[TBGZeroFieldInteractionSpec, TBGZeroFieldScreenedBlockBundle, TBGZeroFieldHFSourceReceipt]:
    """Validate the exact saved typed SCF source before any derived diagnostic."""

    return _validate_typed_run_source(run, grid_solution)

def _projected_basis(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> ContractProjectedBasis:
    _validate_solution_matches_state(run, grid_solution)
    mesh = _carried_torus_mesh(grid_solution)
    model = _single_particle_model(grid_solution)
    n_band = int(grid_solution.nb)
    active_valence = n_band // 2
    raw_kvec = np.asarray(mesh.kvec, dtype=np.complex128)
    physical_kvec = raw_kvec / TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=physical_kvec,
        k_grid_frac=np.asarray(mesh.k_grid_frac, dtype=float),
        h0=np.asarray(run.state.h0, dtype=np.complex128),
        basis_energies=np.asarray(grid_solution.flattened_energies(), dtype=float),
        active_band_indices=_active_band_indices(grid_solution),
        active_valence_bands=int(active_valence),
        active_conduction_bands=int(n_band - active_valence),
        micro_wavefunctions=np.asarray(grid_solution.uk, dtype=np.complex128),
        flavor_labels=_flavor_labels(grid_solution),
        band_labels=_band_labels(grid_solution),
        metadata={
            "projected_basis_source": "BMSolution grid_solution + RestrictedHartreeFockState.h0",
            "k_grid_frac_source": "BMSolution.torus_mesh (validated half-open Fortran order)",
            "kvec_coordinate_convention": "physical_cartesian_nm^-1",
            "kvec_unit": "nm^-1",
            "source_kvec_coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "source_kvec_unit": "dimensionless_b0_code",
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "kvec_conversion": "kvec_nm_inv = source_kvec_b0_code / graphene_a_nm",
            "source_kvec_b0_code_sha256": tbg_zero_field_lattice_kvec_sha256(raw_kvec),
            "kvec_nm_inv_sha256": tbg_zero_field_lattice_kvec_sha256(physical_kvec),
            "torus_mesh_fingerprint": mesh.fingerprint,
            "wavefunctions_axis_order": "bm_micro_basis,bm_band,valley,k",
            "spin_degeneracy_implicit_in_micro_wavefunctions": True,
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "active_state_order": "bm_band,valley,spin with spin fastest",
            "active_band_semantics": "central_full_BM_matrix_band_indices_repeated_over_valley_spin",
            "active_band_indices_per_bm_band": [int(index) for index in _central_bm_band_indices(grid_solution)],
            "lg": int(grid_solution.lg),
            "nlocal": int(grid_solution.nlocal),
            "sigma_rotation": bool(grid_solution.sigma_rotation),
            "calculate_chern_operator": bool(grid_solution.calculate_chern_operator),
            "periodic_g_grid": bool(grid_solution.periodic_g_grid),
            "bm_generation_fingerprint": grid_solution.generation_fingerprint,
            "bm_source_attestation_fingerprint": (
                None
                if grid_solution.source_attestation is None
                else grid_solution.source_attestation.fingerprint
            ),
            "supports_crpa": False,
        },
    )


def _reference_density(run: RestrictedHartreeFockRun) -> np.ndarray:
    state = run.state
    reference = np.zeros((state.nt, state.nt, state.nk), dtype=np.complex128)
    for ik in range(state.nk):
        np.fill_diagonal(reference[:, :, ik], 0.5)
    return reference


def _require_finite_density_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"TBG zero-field export requires finite {name}; found NaN/Inf")
    return array

def _validate_stored_projector_domain(
    density_delta: np.ndarray,
    reference: np.ndarray,
    *,
    tol: float,
) -> np.ndarray:
    density = _require_finite_density_array(density_delta, name="density_delta")
    stored_reference = _require_finite_density_array(reference, name="reference projector")
    projector = _require_finite_density_array(
        density + stored_reference,
        name="stored projector P=D+R",
    )
    hermiticity_residual = float(
        np.max(np.abs(projector - projector.conjugate().swapaxes(0, 1)))
    )
    if hermiticity_residual > tol:
        raise ValueError(
            "TBG zero-field stored projector P=D+R must be Hermitian at every k; "
            f"max residual {hermiticity_residual:.6e} exceeds {tol:.6e}"
        )
    for ik in range(projector.shape[2]):
        eigenvalues = np.linalg.eigvalsh(projector[:, :, ik])
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        if minimum < -tol or maximum > 1.0 + tol:
            raise ValueError(
                "TBG zero-field stored projector P=D+R must have every k-point "
                "eigenvalue in [0,1] within tolerance; "
                f"k={ik}, min={minimum:.16g}, max={maximum:.16g}, tol={tol:.3g}"
            )
    return projector

def _density_state(run: RestrictedHartreeFockRun) -> ContractDensityState:
    state = run.state
    reference = _reference_density(run)
    _validate_stored_projector_domain(
        state.density,
        reference,
        tol=_TBG_DENSITY_PROJECTOR_TOL,
    )
    filling_from_density = restricted_filling(state.density)
    if not np.isclose(
        filling_from_density,
        float(state.nu),
        rtol=0.0,
        atol=_TBG_DENSITY_PROJECTOR_TOL,
    ):
        raise ValueError(
            "TBG zero-field density-derived filling does not match state.nu: "
            f"restricted_filling={filling_from_density:.16g}, state.nu={float(state.nu):.16g}"
        )
    density_state = density_state_from_delta(
        state.density,
        reference,
        reference_scheme="average",
        filling=float(state.nu),
        n_occupied_total=restricted_occupied_state_count(state.nu, state.nt, state.nk),
        reference_metadata={
            "system": "tbg_zero_field",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_scheme_source": "0.5 * identity in the projected BM active basis",
            "reference_diagonal": 0.5,
            "stored_density_definition": "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab",
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.tbg.zero_field.hf_contracts",
            "filling_from_density": float(filling_from_density),
            "projector_domain": "Hermitian spectrum in [0,1]; ensemble occupations allowed",
            "projector_domain_tolerance": _TBG_DENSITY_PROJECTOR_TOL,
        },
    )
    _require_finite_density_array(density_state.density_delta, name="canonical density_delta")
    _require_finite_density_array(
        density_state.reference.reference,
        name="canonical reference projector",
    )
    _require_finite_density_array(density_state.projector, name="canonical stored projector")
    assert_density_state_consistent(
        density_state,
        require_projector=False,
        tol=_TBG_DENSITY_PROJECTOR_TOL,
    )
    return density_state


def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _hamiltonian_parts(run: RestrictedHartreeFockRun) -> ContractHamiltonianParts:
    state = run.state
    h0 = np.asarray(state.h0, dtype=np.complex128)
    total = np.asarray(state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="tbg_zero_field_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "raw_interaction_components_available": False,
            "supports_crpa": False,
            "v0_mev": float(state.v0),
            "beta": _finite_or_none(state.diagnostics.get("beta")),
            "overlap_lg": _finite_or_none(state.diagnostics.get("overlap_lg")),
        },
    )


def _iteration_history(run: RestrictedHartreeFockRun) -> list[dict[str, Any]]:
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
class TBGZeroFieldRunHFConfig:
    """Explicit public config for typed TBG zero-field HF dispatch.

    ``hf_mode`` is part of the source contract. The unified public runner still
    dispatches restricted HF only, while post-run export accepts either typed
    full or restricted provenance.

    The matching :class:`BMSolution` is required because the canonical TBG
    adapter needs the exact half-open torus and BM micro-wavefunctions used by the SCF
    grid.  The public :class:`mean_field.api.hf.HFConfig` is validated as a
    matching contract; it is not translated into hidden BM-grid construction.
    """

    grid_solution: BMSolution
    nu: float
    init_mode: str = "educated"
    seed: int = 1
    beta: float = 1.0
    max_iter: int = 300
    overlap_lg: int | None = None
    precision: float = 1.0e-5
    oda_stall_threshold: float = 1.0e-3
    interaction_spec: TBGZeroFieldInteractionSpec | None = None
    relative_permittivity: float | None = None
    screening_lm: float | None = None
    finite_zero_limit: bool | None = None
    zero_cutoff: float | None = None
    hf_mode: Literal["full", "restricted"] = "restricted"

    def __post_init__(self) -> None:
        if not isinstance(self.grid_solution, BMSolution):
            raise TypeError(f"grid_solution must be BMSolution, got {type(self.grid_solution).__name__}")
        _carried_torus_mesh(self.grid_solution)
        validate_tbg_zero_field_typed_bm_lg(self.grid_solution.lg)
        object.__setattr__(
            self,
            "nu",
            validate_tbg_zero_field_primitive_cell_nu(self.nu),
        )
        if self.hf_mode not in ("full", "restricted"):
            raise ValueError(f"Unsupported TBG zero-field hf_mode {self.hf_mode!r}")
        normalizer = normalize_full_init_mode if self.hf_mode == "full" else normalize_restricted_init_mode
        normalizer(str(self.init_mode))
        object.__setattr__(self, "seed", validate_tbg_zero_field_seed(self.seed))
        if not np.isfinite(float(self.beta)):
            raise ValueError("beta must be finite")
        resolved_max_iter = validate_tbg_zero_field_typed_max_iter(self.max_iter)
        object.__setattr__(self, "max_iter", resolved_max_iter)
        if not np.isfinite(float(self.precision)) or float(self.precision) <= 0.0:
            raise ValueError("precision must be finite and positive")
        if (
            not np.isfinite(float(self.oda_stall_threshold))
            or float(self.oda_stall_threshold) <= 0.0
        ):
            raise ValueError("oda_stall_threshold must be finite and positive")
        if not isinstance(self.interaction_spec, TBGZeroFieldInteractionSpec):
            raise ValueError(
                "TBGZeroFieldRunHFConfig requires a typed "
                "interaction_spec=TBGZeroFieldInteractionSpec; raw screening parameters "
                "are not a canonical physical-nm contract"
            )
        spec = self.interaction_spec
        resolved_overlap_lg = validate_tbg_zero_field_typed_overlap_lg(
            self.grid_solution.lg if self.overlap_lg is None else self.overlap_lg
        )
        if self.overlap_lg is not None:
            object.__setattr__(self, "overlap_lg", resolved_overlap_lg)
        if self.screening_lm is not None:
            raise ValueError(
                "Raw screening_lm is rejected in the typed TBG path; HFConfig.dsc_nm and "
                "interaction_spec.dsc_nm remain physical distances in nm"
            )
        raw_checks = (
            ("relative_permittivity", self.relative_permittivity, spec.epsr),
            ("zero_cutoff", self.zero_cutoff, spec.zero_cutoff),
        )
        for name, raw_value, expected in raw_checks:
            if raw_value is not None and float(raw_value) != float(expected):
                raise ValueError(
                    f"Raw {name}={raw_value!r} conflicts with typed interaction_spec value {expected!r}"
                )
        if self.finite_zero_limit is not None:
            if not isinstance(self.finite_zero_limit, (bool, np.bool_)) or bool(self.finite_zero_limit) != spec.finite_zero_limit:
                raise ValueError(
                    "Raw finite_zero_limit conflicts with typed interaction_spec value "
                    f"{spec.finite_zero_limit!r}"
                )


def _tbg_zero_field_grid_side(solution: BMSolution) -> int:
    return int(_carried_torus_mesh(solution).mesh_size)




def _resolved_screening_lm(config: TBGZeroFieldRunHFConfig) -> float:
    spec = config.interaction_spec
    if not isinstance(spec, TBGZeroFieldInteractionSpec):
        raise ValueError("TBG zero-field canonical adapter requires a typed interaction_spec")
    return float(spec.screening_lm)

def _validate_tbg_zero_field_public_hf_config(config: "HFConfig", tbg_config: TBGZeroFieldRunHFConfig) -> None:
    solution = tbg_config.grid_solution
    side = _tbg_zero_field_grid_side(solution)
    mesh = (side, side)
    if config.mesh != mesh:
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.mesh to match the carried "
            f"half-open torus mesh {mesh}, got {config.mesh}"
        )
    try:
        normalized_config_filling = validate_tbg_zero_field_primitive_cell_nu(
            config.filling
        )
    except ValueError as exc:
        raise ValueError(
            "TBG zero-field public run_hf rejected "
            f"HFConfig.filling={config.filling!r}: {exc}"
        ) from exc
    if not _numeric_values_match(
        normalized_config_filling,
        tbg_config.nu,
        atol=_TBG_RUN_FILLING_ATOL,
    ):
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.filling to match "
            f"nu={tbg_config.nu} within rtol={_TBG_RUN_FLOAT_RTOL:g}, "
            f"atol={_TBG_RUN_FILLING_ATOL:g}; got {config.filling}"
        )
    config_max_iter = validate_tbg_zero_field_typed_max_iter(config.max_iter)
    if config_max_iter != tbg_config.max_iter:
        raise ValueError(
            f"TBG zero-field public run_hf requires HFConfig.max_iter={tbg_config.max_iter}, got {config.max_iter}"
        )
    if not _numeric_values_match(config.precision, tbg_config.precision):
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.precision to match "
            f"{tbg_config.precision} within rtol={_TBG_RUN_FLOAT_RTOL:g}, "
            f"atol={_TBG_RUN_FLOAT_ATOL:g}; got {config.precision}"
        )
    chosen_seed = str(tbg_config.seed)
    config_seeds = tuple(str(value) for value in config.seeds)
    if chosen_seed not in config_seeds:
        raise ValueError(
            "TBG zero-field public run_hf requires the exact chosen seed to be a member "
            f"of HFConfig.seeds; chosen={chosen_seed!r}, seeds={config_seeds!r}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "TBG zero-field HF stores "
            "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab; "
            "set HFConfig.density_convention='stored_delta'"
        )
    if config.interaction_scheme != "average":
        raise ValueError("TBG zero-field public run_hf currently requires HFConfig.interaction_scheme='average'")
    if config.coulomb_kernel != "2d_gate":
        raise ValueError("TBG zero-field public run_hf currently requires HFConfig.coulomb_kernel='2d_gate'")
    spec = tbg_config.interaction_spec
    if not isinstance(spec, TBGZeroFieldInteractionSpec):
        raise ValueError("TBG zero-field public run_hf requires a typed interaction_spec")
    if float(config.epsilon_r) != float(spec.epsr):
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.epsilon_r to match "
            f"interaction_spec.epsr={spec.epsr}, got {config.epsilon_r}"
        )
    if float(config.dsc_nm) != float(spec.dsc_nm):
        raise ValueError(
            "TBG zero-field HFConfig.dsc_nm is a physical gate distance in nm and must match "
            f"interaction_spec.dsc_nm={spec.dsc_nm}; got {config.dsc_nm}.  It is not dimensionless screening_lm."
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "TBG zero-field public run_hf takes the active two-band BM window from grid_solution; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )
    normalizer = normalize_full_init_mode if tbg_config.hf_mode == "full" else normalize_restricted_init_mode
    normalized_init_mode = normalizer(str(tbg_config.init_mode))
    claimed_metadata = {
        "hf_mode": tbg_config.hf_mode,
        "mode": tbg_config.hf_mode,
        "beta": float(tbg_config.beta),
        "nu": float(tbg_config.nu),
        "filling": float(tbg_config.nu),
        "precision": float(tbg_config.precision),
        "oda_stall_threshold": float(tbg_config.oda_stall_threshold),
        "seed": int(tbg_config.seed),
        "init_mode": normalized_init_mode,
        "normalized_init_mode": normalized_init_mode,
        "max_iter": int(tbg_config.max_iter),
        "max_iterations": int(tbg_config.max_iter),
        "requested_max_iterations": int(tbg_config.max_iter),
    }
    tolerant_metadata = {
        "beta": _TBG_RUN_FLOAT_ATOL,
        "nu": _TBG_RUN_FILLING_ATOL,
        "filling": _TBG_RUN_FILLING_ATOL,
        "precision": _TBG_RUN_FLOAT_ATOL,
        "oda_stall_threshold": _TBG_RUN_FLOAT_ATOL,
    }
    for key, expected in claimed_metadata.items():
        if key not in config.metadata:
            continue
        actual = config.metadata[key]
        matches = (
            _numeric_values_match(actual, expected, atol=tolerant_metadata[key])
            if key in tolerant_metadata
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"HFConfig.metadata[{key!r}]={actual!r} conflicts with "
                f"the TBG run config value {expected!r}"
            )


def _validate_tbg_zero_field_model_matches_grid(model: TBGZeroFieldBMModel, solution: BMSolution) -> None:
    mesh = _carried_torus_mesh(solution)
    model_generation_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        model.params,
        lg=model.lg,
        periodic_g_grid=model.periodic_g_grid,
        sigma_rotation=model.sigma_rotation,
        calculate_chern_operator=solution.calculate_chern_operator,
        torus_mesh_fingerprint=mesh.fingerprint,
    )
    if model_generation_fingerprint != solution.generation_fingerprint:
        raise ValueError(
            "TBG zero-field model BM generation fingerprint does not exactly match "
            "grid_solution (all independent TBGParameters inputs, lg, periodic_g_grid, "
            "sigma_rotation, and torus mesh are bound)"
        )
    solution_theta_deg = float(solution.params.dtheta_rad) * 180.0 / math.pi
    if not _numeric_values_match(
        model.theta_deg,
        solution_theta_deg,
        atol=_TBG_RUN_FILLING_ATOL,
    ):
        raise ValueError(
            f"TBG zero-field model theta_deg={model.theta_deg} does not match grid_solution theta_deg={solution_theta_deg}"
        )


def tbg_zero_field_hf_run_to_hf_run_result(
    run: RestrictedHartreeFockRun,
    *,
    grid_solution: BMSolution | None = None,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a TBG zero-field HF run in canonical core contracts.

    ``RestrictedHartreeFockRun`` itself does not store the k-grid fractional
    coordinates or BM micro-wavefunctions required by ``ProjectedBasis``. Pass
    the matching torus-owning ``grid_solution`` (or use
    :func:`b0_hf_benchmark_run_to_hf_run_result`) so the adapter can validate the
    grid and avoid fabricating canonical basis data.
    """

    if grid_solution is None:
        raise ValueError(
            "TBG zero-field canonical HFRunResult adapter requires the matching BMSolution grid_solution; "
            "a bare RestrictedHartreeFockRun has no k-grid coordinates or BM micro-wavefunctions. "
            "Use b0_hf_benchmark_run_to_hf_run_result(result) for benchmark results, or pass "
            "grid_solution=<BMSolution> from the same SCF grid."
        )

    interaction_spec, screened_block_bundle, receipt = _validate_typed_run_source(
        run,
        grid_solution,
    )
    state = run.state
    mesh = _carried_torus_mesh(grid_solution)
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("Canonical TBG zero-field export requires immutable typed run provenance")
    canonical_observables: dict[str, object] = {
        "eigenvectors_active_available": False,
        "hf_mode": provenance.hf_mode,
        "beta": provenance.beta,
        "nu": provenance.nu,
        "precision": provenance.precision,
        "oda_stall_threshold": provenance.oda_stall_threshold,
        "requested_max_iterations": provenance.requested_max_iterations,
        "normalized_init_mode": provenance.normalized_init_mode,
        "seed": provenance.seed,
        "hf_run_provenance": provenance.to_metadata(),
        "grid_solution_available": True,
        "grid_mesh_size": int(mesh.mesh_size),
        "torus_mesh_fingerprint": mesh.fingerprint,
        "bm_generation_fingerprint": screened_block_bundle.bm_generation_fingerprint,
        "bm_solution_sha256": screened_block_bundle.bm_solution_sha256,
        "bm_source_attestation": _bm_source_attestation_metadata(grid_solution),
        "lattice_kvec_sha256": receipt.lattice_kvec_sha256,
        "overlap_kernel_inventory_sha256": receipt.overlap_kernel_inventory_sha256,
        "screened_block_bundle_sha256": screened_block_bundle.fingerprint,
        "interaction_spec_fingerprint": interaction_spec.fingerprint,
        "bm_lg": int(grid_solution.lg),
        "sigma_rotation": bool(grid_solution.sigma_rotation),
        "calculate_chern_operator": bool(grid_solution.calculate_chern_operator),
        "periodic_g_grid": bool(grid_solution.periodic_g_grid),
        "filling_from_density": float(restricted_filling(state.density)),
        "micro_wavefunctions_source": "BMSolution.uk",
        "micro_wavefunctions_spin_degeneracy_implicit": True,
        "screened_block_bundle": screened_block_bundle.to_metadata(),
    }
    canonical_observables["hf_source_receipt"] = receipt.to_metadata()
    canonical_observables["interaction_spec"] = interaction_spec.to_metadata()
    final_state = ContractHFState(
        basis=_projected_basis(run, grid_solution),
        density=_density_state(run),
        hamiltonian=_hamiltonian_parts(run),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables=canonical_observables,
        diagnostics=_float_diagnostics(state.diagnostics),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_iteration_history(run),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        best_seed=int(run.seed),
        init_mode=str(run.init_mode),
        archive_manifest=_archive_manifest_with_hf_source_receipt(
            archive_manifest,
            run,
        ),
    )


def b0_hf_benchmark_run_to_hf_run_result(
    result: object,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a ``B0HFBenchmarkRun``-like result in canonical core contracts.

    The higher-level result must carry a typed half-open torus solution and its
    matching screened bundle. Historical endpoint-inclusive B0 results are
    intentionally diagnostic-only and are refused here.
    """

    hf_run = getattr(result, "hf_run")
    grid_solution = getattr(result, "grid_solution")
    _carried_torus_mesh(grid_solution)
    return tbg_zero_field_hf_run_to_hf_run_result(
        hf_run,
        grid_solution=grid_solution,
        archive_manifest=archive_manifest,
    )



def _default_hf_config_from_run(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> "HFConfig":
    from mean_field.api.hf import HFConfig

    interaction_spec, _bundle, _receipt = _validate_typed_run_source(run, grid_solution)
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("Cannot derive HFConfig without immutable typed run provenance")
    return HFConfig(
        filling=provenance.nu,
        mesh=(_tbg_zero_field_grid_side(grid_solution), _tbg_zero_field_grid_side(grid_solution)),
        interaction_scheme="average",
        density_convention="stored_delta",
        epsilon_r=float(interaction_spec.epsr),
        dsc_nm=float(interaction_spec.dsc_nm),
        coulomb_kernel="2d_gate",
        max_iter=provenance.requested_max_iterations,
        precision=provenance.precision,
        seeds=(str(provenance.seed),),
        metadata={
            "source": "immutable_TBGZeroFieldHFRunProvenance_and_typed_sources",
            "max_iter_semantics": "requested_solver_limit",
            "hf_mode": provenance.hf_mode,
            "beta": provenance.beta,
            "nu": provenance.nu,
            "filling": provenance.nu,
            "precision": provenance.precision,
            "oda_stall_threshold": provenance.oda_stall_threshold,
            "seed": provenance.seed,
            "init_mode": provenance.normalized_init_mode,
            "normalized_init_mode": provenance.normalized_init_mode,
            "requested_max_iterations": provenance.requested_max_iterations,
            "typed_receipt_fingerprint": provenance.typed_receipt_fingerprint,
            "interaction_spec_fingerprint": provenance.interaction_spec_fingerprint,
            "bm_generation_fingerprint": provenance.bm_generation_fingerprint,
            "mesh_fingerprint": provenance.mesh_fingerprint,
            "grid_mesh_size": int(_carried_torus_mesh(grid_solution).mesh_size),
            "bm_lg": int(grid_solution.lg),
            "interaction_spec": interaction_spec.to_metadata(),
        },
    )


def _validate_hf_config_matches_run(
    config: "HFConfig",
    run: RestrictedHartreeFockRun,
    grid_solution: BMSolution,
    *,
    tbg_config: TBGZeroFieldRunHFConfig | None = None,
) -> None:
    interaction_spec, bundle, receipt = _validate_typed_run_source(run, grid_solution)
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("TBG zero-field HFResult requires immutable typed run provenance")
    strict_run_config = TBGZeroFieldRunHFConfig(
        grid_solution=grid_solution,
        nu=provenance.nu,
        hf_mode=provenance.hf_mode,
        init_mode=provenance.normalized_init_mode,
        seed=provenance.seed,
        beta=provenance.beta,
        max_iter=provenance.requested_max_iterations,
        overlap_lg=int(bundle.overlap_lg),
        precision=provenance.precision,
        oda_stall_threshold=provenance.oda_stall_threshold,
        interaction_spec=interaction_spec,
    )
    _validate_tbg_zero_field_public_hf_config(config, strict_run_config)

    expected_metadata = {
        "hf_mode": provenance.hf_mode,
        "mode": provenance.hf_mode,
        "beta": provenance.beta,
        "nu": provenance.nu,
        "filling": provenance.nu,
        "precision": provenance.precision,
        "oda_stall_threshold": provenance.oda_stall_threshold,
        "seed": provenance.seed,
        "init_mode": provenance.normalized_init_mode,
        "normalized_init_mode": provenance.normalized_init_mode,
        "max_iter": provenance.requested_max_iterations,
        "max_iterations": provenance.requested_max_iterations,
        "requested_max_iterations": provenance.requested_max_iterations,
        "receipt_fingerprint": provenance.typed_receipt_fingerprint,
        "hf_source_receipt_fingerprint": provenance.typed_receipt_fingerprint,
        "typed_receipt_fingerprint": provenance.typed_receipt_fingerprint,
        "interaction_spec_fingerprint": provenance.interaction_spec_fingerprint,
        "bm_generation_fingerprint": provenance.bm_generation_fingerprint,
        "mesh_fingerprint": provenance.mesh_fingerprint,
        "grid_mesh_size": int(_carried_torus_mesh(grid_solution).mesh_size),
        "bm_lg": int(grid_solution.lg),
        "interaction_spec": interaction_spec.to_metadata(),
    }
    tolerant_metadata = {
        "beta": _TBG_RUN_FLOAT_ATOL,
        "nu": _TBG_RUN_FILLING_ATOL,
        "filling": _TBG_RUN_FILLING_ATOL,
        "precision": _TBG_RUN_FLOAT_ATOL,
        "oda_stall_threshold": _TBG_RUN_FLOAT_ATOL,
    }
    for key, expected in expected_metadata.items():
        if key not in config.metadata:
            continue
        actual = config.metadata[key]
        matches = (
            _numeric_values_match(actual, expected, atol=tolerant_metadata[key])
            if key in tolerant_metadata
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"HFConfig.metadata[{key!r}]={actual!r} conflicts with "
                f"immutable run provenance {expected!r}"
            )

    if tbg_config is not None:
        tbg_spec = tbg_config.interaction_spec
        if not isinstance(tbg_spec, TBGZeroFieldInteractionSpec):
            raise ValueError("TBGZeroFieldRunHFConfig lacks its typed interaction spec")
        bundle.validate_for_solution(tbg_config.grid_solution)
        resolved_overlap_lg = (
            int(tbg_config.grid_solution.lg)
            if tbg_config.overlap_lg is None
            else int(tbg_config.overlap_lg)
        )
        tbg_normalizer = (
            normalize_full_init_mode
            if tbg_config.hf_mode == "full"
            else normalize_restricted_init_mode
        )
        tbg_values = {
            "hf_mode": tbg_config.hf_mode,
            "beta": float(tbg_config.beta),
            "requested_max_iterations": int(tbg_config.max_iter),
            "seed": int(tbg_config.seed),
            "normalized_init_mode": tbg_normalizer(str(tbg_config.init_mode)),
            "interaction_spec_fingerprint": tbg_spec.fingerprint,
            "bm_generation_fingerprint": tbg_config.grid_solution.generation_fingerprint,
            "mesh_fingerprint": _carried_torus_mesh(tbg_config.grid_solution).fingerprint,
            "overlap_lg": resolved_overlap_lg,
            "nu": float(tbg_config.nu),
            "precision": float(tbg_config.precision),
            "oda_stall_threshold": float(tbg_config.oda_stall_threshold),
        }
        expected_tbg_values = {
            "hf_mode": provenance.hf_mode,
            "beta": provenance.beta,
            "requested_max_iterations": provenance.requested_max_iterations,
            "seed": provenance.seed,
            "normalized_init_mode": provenance.normalized_init_mode,
            "interaction_spec_fingerprint": provenance.interaction_spec_fingerprint,
            "bm_generation_fingerprint": provenance.bm_generation_fingerprint,
            "mesh_fingerprint": provenance.mesh_fingerprint,
            "overlap_lg": int(bundle.overlap_lg),
            "nu": provenance.nu,
            "precision": provenance.precision,
            "oda_stall_threshold": provenance.oda_stall_threshold,
        }
        tolerant_tbg_fields = {
            "beta": _TBG_RUN_FLOAT_ATOL,
            "nu": _TBG_RUN_FILLING_ATOL,
            "precision": _TBG_RUN_FLOAT_ATOL,
            "oda_stall_threshold": _TBG_RUN_FLOAT_ATOL,
        }
        mismatched = []
        for name, value in tbg_values.items():
            expected_value = expected_tbg_values[name]
            matches = (
                _numeric_values_match(
                    value,
                    expected_value,
                    atol=tolerant_tbg_fields[name],
                )
                if name in tolerant_tbg_fields
                else value == expected_value
            )
            if not matches:
                mismatched.append(name)
        if mismatched:
            raise ValueError(
                "TBGZeroFieldRunHFConfig does not match immutable run provenance/state: "
                f"{sorted(mismatched)}"
            )

    if not isinstance(receipt, TBGZeroFieldHFSourceReceipt) or receipt.interaction_contract != "typed":
        raise ValueError("TBG zero-field HFResult requires a typed HF source receipt")
    if receipt.interaction_spec_fingerprint != interaction_spec.fingerprint:
        raise ValueError("HF source receipt interaction fingerprint does not match the live typed spec")
    if receipt.screened_block_bundle_sha256 != bundle.fingerprint:
        raise ValueError("HF source receipt does not match the returned screened-block bundle")


def _result_observables(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> dict[str, object]:
    state = run.state
    interaction_spec, bundle, receipt = _validate_typed_run_source(run, grid_solution)
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("TBG zero-field HFResult requires immutable typed run provenance")
    mesh = _carried_torus_mesh(grid_solution)
    return {
        "hf_mode": provenance.hf_mode,
        "beta": provenance.beta,
        "nu": provenance.nu,
        "precision": provenance.precision,
        "oda_stall_threshold": provenance.oda_stall_threshold,
        "hf_run_provenance": provenance.to_metadata(),
        "requested_max_iterations": provenance.requested_max_iterations,
        "filling_from_density": float(restricted_filling(state.density)),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "init_mode": str(run.init_mode),
        "seed": int(run.seed),
        "iterations": int(max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))),
        "raw_density_convention": "stored_delta",
        "grid_mesh_size": int(mesh.mesh_size),
        "torus_mesh_fingerprint": mesh.fingerprint,
        "bm_lg": int(grid_solution.lg),
        "sigma_rotation": bool(grid_solution.sigma_rotation),
        "calculate_chern_operator": bool(grid_solution.calculate_chern_operator),
        "periodic_g_grid": bool(grid_solution.periodic_g_grid),
        "bm_generation_fingerprint": bundle.bm_generation_fingerprint,
        "bm_solution_sha256": bundle.bm_solution_sha256,
        "bm_source_attestation": _bm_source_attestation_metadata(grid_solution),
        "lattice_kvec_sha256": receipt.lattice_kvec_sha256,
        "overlap_kernel_inventory_sha256": receipt.overlap_kernel_inventory_sha256,
        "screened_block_bundle_sha256": bundle.fingerprint,
        "interaction_spec_fingerprint": interaction_spec.fingerprint,
        "hf_source_receipt": receipt.to_metadata(),
        "interaction_spec": interaction_spec.to_metadata(),
        "screened_block_bundle": bundle.to_metadata(),
    }


def tbg_zero_field_hf_run_to_hf_result(
    run: RestrictedHartreeFockRun,
    *,
    grid_solution: BMSolution,
    config: "HFConfig | None" = None,
    archive_manifest: Mapping[str, Any] | None = None,
    observables: Mapping[str, object] | None = None,
) -> "HFResult":
    """Return a public :class:`HFResult` view of an existing TBG zero-field HF run.

    The matching ``grid_solution`` is required for the same reason as the
    canonical adapter: the raw HF run does not carry BM micro-wavefunctions or
    fractional k-grid coordinates.
    """

    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    resolved_config = _default_hf_config_from_run(run, grid_solution) if config is None else config
    _validate_hf_config_matches_run(resolved_config, run, grid_solution)
    canonical = tbg_zero_field_hf_run_to_hf_run_result(
        run,
        grid_solution=grid_solution,
        archive_manifest=None if archive_manifest is None else dict(archive_manifest),
    )
    result_observables = _result_observables(run, grid_solution)
    reserved_observable_keys = set(result_observables) | {
        "solver_provenance",
        "run_provenance",
        "max_iter",
        "max_iterations",
        "normalized_init_mode",
        "mesh",
        "mesh_fingerprint",
        "mesh_provenance",
        "source",
        "source_fingerprint",
        "source_provenance",
        "source_receipt",
        "hf_mode",
        "beta",
    }
    if observables is not None:
        caller_observables = dict(observables)
        conflicts = sorted(set(caller_observables) & reserved_observable_keys)
        if conflicts:
            raise ValueError(
                "Caller observables cannot overwrite reserved verified TBG HF keys: "
                f"{conflicts}"
            )
        result_observables.update(caller_observables)
    provenance = run.provenance
    if not isinstance(provenance, TBGZeroFieldHFRunProvenance):
        raise ValueError("TBG zero-field HFResult requires immutable typed run provenance")
    canonical_model = TBGZeroFieldBMModel.from_config(
        grid_solution.params.dtheta_rad * 180.0 / math.pi,
        lg=grid_solution.lg,
        params=grid_solution.params,
        sigma_rotation=grid_solution.sigma_rotation,
        periodic_g_grid=grid_solution.periodic_g_grid,
    )
    record = model_record(canonical_model, system_name="tbg_zero_field")
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
                wavefunction_axis_order="bm_micro_basis,bm_band,valley,k",
                gauge="tbg_zero_field_bm_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": f"tbg.zero_field.{provenance.hf_mode}_hf.raw_run_result",
                "system_name": "tbg_zero_field",
                "hf_mode": provenance.hf_mode,
                "beta": provenance.beta,
                "nu": provenance.nu,
                "precision": provenance.precision,
                "oda_stall_threshold": provenance.oda_stall_threshold,
                "bm_generation_fingerprint": provenance.bm_generation_fingerprint,
                "requested_max_iterations": provenance.requested_max_iterations,
                "normalized_init_mode": provenance.normalized_init_mode,
                "seed": provenance.seed,
                "hf_run_provenance": provenance.to_metadata(),
                "adapter": "mean_field.systems.tbg.zero_field.hf_contracts.tbg_zero_field_hf_run_to_hf_result",
                "canonical_adapter": "mean_field.systems.tbg.zero_field.hf_contracts.tbg_zero_field_hf_run_to_hf_run_result",
                "raw_state_type": type(run).__name__,
            },
        ),
        canonical_run_result=canonical,
    )


def run_tbg_zero_field_hf_config_adapter(model: object, config: "HFConfig", **kwargs: Any) -> "HFResult | None":
    """Run TBG zero-field restricted HF from an explicit grid-owning config."""

    if not isinstance(model, TBGZeroFieldBMModel):
        return None
    if "tbg_zero_field_config" not in kwargs:
        raise NotImplementedError(
            "Unified run_hf has a TBG zero-field adapter only for explicit "
            "tbg_zero_field_config=TBGZeroFieldRunHFConfig(grid_solution=...); "
            "generic HFConfig -> BMSolution/grid workflow mapping is not implemented"
        )
    tbg_config = kwargs.pop("tbg_zero_field_config")
    if not isinstance(tbg_config, TBGZeroFieldRunHFConfig):
        raise TypeError(f"tbg_zero_field_config must be TBGZeroFieldRunHFConfig, got {type(tbg_config).__name__}")
    if kwargs:
        raise TypeError(f"Unsupported TBG zero-field run_hf kwargs: {sorted(kwargs)}")

    normalized_filling = validate_tbg_zero_field_primitive_cell_nu(config.filling)
    if float(config.filling) != normalized_filling:
        config = replace(config, filling=normalized_filling)

    _validate_tbg_zero_field_model_matches_grid(model, tbg_config.grid_solution)
    if tbg_config.hf_mode != "restricted":
        raise NotImplementedError(
            "The public TBG zero-field config adapter currently dispatches restricted HF only; "
            "full typed runs can still be exported with their actual full-mode provenance"
        )
    _validate_tbg_zero_field_public_hf_config(config, tbg_config)
    overlap_lg = validate_tbg_zero_field_typed_overlap_lg(
        tbg_config.grid_solution.lg
        if tbg_config.overlap_lg is None
        else tbg_config.overlap_lg
    )
    state = RestrictedHartreeFockState.from_bm_solution(
        tbg_config.grid_solution,
        nu=float(tbg_config.nu),
        precision=float(tbg_config.precision),
    )
    spec = tbg_config.interaction_spec
    if not isinstance(spec, TBGZeroFieldInteractionSpec):
        raise ValueError("TBG zero-field canonical adapter requires a typed interaction_spec")
    screened_block_bundle = build_tbg_zero_field_screened_block_bundle(
        tbg_config.grid_solution,
        overlap_lg=overlap_lg,
        interaction_spec=spec,
    )
    overlap_blocks = screened_block_bundle.screened_blocks
    raw = run_restricted_hartree_fock(
        state,
        overlap_blocks,
        tbg_config.grid_solution.lattice_kvec,
        tbg_config.grid_solution.params,
        init_mode=str(tbg_config.init_mode),
        seed=tbg_config.seed,
        beta=float(tbg_config.beta),
        max_iter=tbg_config.max_iter,
        oda_stall_threshold=float(tbg_config.oda_stall_threshold),
        interaction_spec=spec,
        source_solution=tbg_config.grid_solution,
        screened_block_bundle=screened_block_bundle,
    )
    _validate_hf_config_matches_run(
        config,
        raw,
        tbg_config.grid_solution,
        tbg_config=tbg_config,
    )
    return tbg_zero_field_hf_run_to_hf_result(
        raw,
        grid_solution=tbg_config.grid_solution,
        config=config,
        observables={
            "public_run_hf_adapter": "mean_field.systems.tbg.zero_field.hf_contracts.run_tbg_zero_field_hf_config_adapter",
            "explicit_config_type": "TBGZeroFieldRunHFConfig",
        },
    )


__all__ = [
    "TBGZeroFieldRunHFConfig",
    "b0_hf_benchmark_run_to_hf_run_result",
    "run_tbg_zero_field_hf_config_adapter",
    "tbg_zero_field_hf_run_to_hf_result",
    "tbg_zero_field_hf_run_to_hf_run_result",
]
