"""Pickle-free archive loading for canonical InAs/GaSb Kane--Poisson results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from mean_field.core.io import (
    file_object_sha256,
    file_sha256,
    open_stable_binary_file,
    parse_json_artifact,
    publish_staged_file_noreplace,
    unique_staging_path,
    write_json_artifact,
    write_npz_artifact,
)

from .kane_poisson import (
    CANONICAL_FLATTENING_LABEL,
    KANE_POISSON_SCHEMA_VERSION,
    POTENTIAL_COUPLING_LABEL,
    CanonicalKanePoissonConfig,
    CanonicalKanePoissonResult,
    KaneParentSourceSpec,
    KanePoissonIterationRecord,
    KaneWindowAtPotential,
    canonical_kane_poisson_fingerprint,
    kane_poisson_array_sha256,
)
from .normal_reference import (
    ChemicalPotentialRootTolerances,
    KaneChargeNeutralReference,
    KaneNormalReferenceContract,
)
from .ordinary_electron_hf_bcs import OrdinaryElectronConventionLock

CANONICAL_KANE_POISSON_ARCHIVE_SCHEMA = "canonical_kane_poisson_archive_v1"
_CANONICAL_ARRAY_KEYS = frozenset(
    {
        "k_cart_nm_inv",
        "k_weights_nm2",
        "z_nm",
        "z_weights_nm",
        "epsilon_r",
        "potential_mev",
        "electron_density_nm3",
        "hole_density_nm3",
        "hamiltonian_mev",
        "micro_wavefunctions",
        "density_matrix",
        "normal_eigenvalues_mev",
    }
)


@dataclass(frozen=True)
class CanonicalKanePoissonSourceAuthority:
    claim_scope: str
    credibility_blockers: tuple[str, ...]
    ordinary_electron_convention: OrdinaryElectronConventionLock
    physical_fixed_gate_claim: bool = False
    hartree_fock_authorized: bool = False
    matrix_hf_scope: str = "frozen_poisson_exchange_only_diagnostic"

    def __post_init__(self) -> None:
        if not isinstance(self.claim_scope, str) or not self.claim_scope:
            raise ValueError("canonical source claim_scope must be nonempty text")
        if (
            not isinstance(self.credibility_blockers, tuple)
            or not self.credibility_blockers
            or any(
                not isinstance(item, str) or not item
                for item in self.credibility_blockers
            )
        ):
            raise ValueError(
                "canonical source credibility_blockers must be nonempty text"
            )
        if not isinstance(
            self.ordinary_electron_convention, OrdinaryElectronConventionLock
        ):
            raise TypeError(
                "ordinary_electron_convention must be OrdinaryElectronConventionLock"
            )
        if self.physical_fixed_gate_claim is not False:
            raise ValueError("canonical source cannot authorize a physical fixed-gate claim")
        if self.hartree_fock_authorized is not False:
            raise ValueError("canonical source cannot authorize a physical HF claim")
        if self.matrix_hf_scope != "frozen_poisson_exchange_only_diagnostic":
            raise ValueError("canonical source matrix-HF scope changed")

    def to_metadata(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "credibility_blockers": list(self.credibility_blockers),
            "ordinary_electron_convention": asdict(
                self.ordinary_electron_convention
            ),
            "physical_fixed_gate_claim": self.physical_fixed_gate_claim,
            "hartree_fock_authorized": self.hartree_fock_authorized,
            "matrix_hf_scope": self.matrix_hf_scope,
        }


@dataclass(frozen=True)
class LoadedCanonicalKanePoissonArchive:
    result: CanonicalKanePoissonResult
    source_authority: CanonicalKanePoissonSourceAuthority
    npz_path: Path
    metadata_path: Path
    npz_sha256: str
    metadata_sha256: str




@dataclass(frozen=True)
class CanonicalKanePoissonArchiveReceipt:
    npz_path: Path
    metadata_path: Path
    npz_sha256: str
    metadata_sha256: str
    result_fingerprint: str
    source_authority: CanonicalKanePoissonSourceAuthority


def canonical_kane_poisson_archive_arrays(
    result: CanonicalKanePoissonResult,
) -> dict[str, np.ndarray]:
    """Project a complete typed result to the arrays required for reconstruction."""

    return {
        "k_cart_nm_inv": result.parent_spec.k_cart_nm_inv,
        "k_weights_nm2": result.k_weights_nm2,
        "z_nm": result.z_nm,
        "z_weights_nm": result.z_weights_nm,
        "epsilon_r": result.epsilon_r,
        "potential_mev": result.potential_mev,
        "electron_density_nm3": result.electron_density_nm3,
        "hole_density_nm3": result.hole_density_nm3,
        "hamiltonian_mev": result.final_eigensystem.hamiltonian_mev,
        "micro_wavefunctions": result.final_eigensystem.micro_wavefunctions,
        "density_matrix": result.reference.density,
        "normal_eigenvalues_mev": result.reference.energies_mev,
    }


def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": str(np.asarray(values).dtype),
            "shape": list(np.asarray(values).shape),
            "sha256": kane_poisson_array_sha256(np.asarray(values)),
        }
        for name, values in arrays.items()
    }


def canonical_kane_poisson_archive_metadata(
    result: CanonicalKanePoissonResult,
    *,
    arrays: dict[str, np.ndarray],
    npz_sha256: str,
) -> dict[str, object]:
    """Build strict typed metadata for the canonical archive loader."""

    parent = result.parent_spec
    source = result.final_eigensystem
    reference = result.reference
    return {
        "schema": CANONICAL_KANE_POISSON_ARCHIVE_SCHEMA,
        "result_fingerprint": canonical_kane_poisson_fingerprint(result),
        "npz_sha256": str(npz_sha256),
        "array_manifest": _array_manifest(arrays),
        "source_authority": CanonicalKanePoissonSourceAuthority(
            claim_scope=result.claim_scope,
            credibility_blockers=result.credibility_blockers,
            ordinary_electron_convention=OrdinaryElectronConventionLock(),
        ).to_metadata(),
        "config": asdict(result.config),
        "parent_spec": {
            "fingerprint": parent.fingerprint,
            "parent_hilbert_dimension": parent.parent_hilbert_dimension,
            "selected_band_indices": list(parent.selected_band_indices),
            "candidate_eigenpair_count": parent.candidate_eigenpair_count,
            "target_energy_mev": parent.target_energy_mev,
            "window_selection_label": parent.window_selection_label,
            "static_parent_sha256": parent.static_parent_sha256,
            "material_profile_sha256": parent.material_profile_sha256,
            "kdotpy_source_sha256": parent.kdotpy_source_sha256,
            "hamiltonian_options": list(parent.hamiltonian_options),
            "basis_flattening_label": parent.basis_flattening_label,
            "potential_coupling_label": parent.potential_coupling_label,
            "schema_version": parent.schema_version,
        },
        "final_eigensystem": {
            "selected_band_indices": list(source.selected_band_indices),
            "parent_hilbert_dimension": source.parent_hilbert_dimension,
            "calculation_split_mev": source.calculation_split_mev,
            "energy_zero_label": source.energy_zero_label,
            "input_potential_sha256": source.input_potential_sha256,
            "potential_operator_fingerprint": source.potential_operator_fingerprint,
            "parent_spec_fingerprint": source.parent_spec_fingerprint,
            "replayed_static_parent_sha256": source.replayed_static_parent_sha256,
            "static_parent_replay_max_error_mev": source.static_parent_replay_max_error_mev,
            "potential_operator_max_error_mev": source.potential_operator_max_error_mev,
            "provenance_fingerprint": source.provenance_fingerprint,
        },
        "reference": {
            "contract": asdict(reference.contract),
            "tolerances": asdict(reference.tolerances),
            "temperature_K": reference.temperature_K,
            "mu_mev": reference.mu_mev,
            "electron_density_nm2": reference.electron_density_nm2,
            "hole_density_nm2": reference.hole_density_nm2,
            "target_metric_occupation_nm2": reference.target_metric_occupation_nm2,
            "achieved_metric_occupation_nm2": reference.achieved_metric_occupation_nm2,
            "number_residual_nm2": reference.number_residual_nm2,
            "neutrality_residual_nm2": reference.neutrality_residual_nm2,
            "chemical_potential_bracket_mev": list(reference.chemical_potential_bracket_mev),
            "bracket_residuals_nm2": list(reference.bracket_residuals_nm2),
            "root_iterations": reference.root_iterations,
            "root_function_calls": reference.root_function_calls,
            "root_converged": reference.root_converged,
            "frame_orthonormality_error": reference.frame_orthonormality_error,
            "orbital_projector_completeness_error": reference.orbital_projector_completeness_error,
            "hamiltonian_hermiticity_error_mev": reference.hamiltonian_hermiticity_error_mev,
        },
        "history": [asdict(record) for record in result.history],
        "result": {
            "fixed_point_residual_mev": result.fixed_point_residual_mev,
            "gauss_residual_mev_nm2": result.gauss_residual_mev_nm2,
            "converged": result.converged,
            "replay_verified": result.replay_verified,
        },
    }


def save_canonical_kane_poisson_archive(
    result: CanonicalKanePoissonResult,
    npz_path: str | Path,
    metadata_path: str | Path,
) -> CanonicalKanePoissonArchiveReceipt:
    """Publish and immediately reconstruct a typed canonical result."""

    npz_output = Path(npz_path)
    metadata_output = Path(metadata_path)
    if npz_output.resolve() == metadata_output.resolve():
        raise ValueError("canonical NPZ and metadata paths must be distinct")
    if npz_output.parent.resolve() != metadata_output.parent.resolve():
        raise ValueError("canonical NPZ and metadata must share one directory")
    if npz_output.exists() or metadata_output.exists():
        raise FileExistsError("canonical archive outputs must not already exist")
    arrays = canonical_kane_poisson_archive_arrays(result)
    staged_npz = unique_staging_path(npz_output)
    staged_metadata = unique_staging_path(metadata_output)
    try:
        write_npz_artifact(arrays, staged_npz, compressed=True)
        with np.load(staged_npz, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("canonical archive array keys changed during publication")
            for name, expected in arrays.items():
                if not np.array_equal(archive[name], expected, equal_nan=True):
                    raise ValueError(
                        f"canonical archive array {name!r} changed during publication"
                    )
        npz_digest = file_sha256(staged_npz)
        metadata = canonical_kane_poisson_archive_metadata(
            result,
            arrays=arrays,
            npz_sha256=npz_digest,
        )
        write_json_artifact(metadata, staged_metadata)
        reloaded = load_canonical_kane_poisson_archive(
            staged_npz,
            staged_metadata,
        )
        result_fingerprint = canonical_kane_poisson_fingerprint(result)
        if canonical_kane_poisson_fingerprint(reloaded.result) != result_fingerprint:
            raise ValueError("canonical archive typed reload changed the result")
        publish_staged_file_noreplace(staged_npz, npz_output)
        publish_staged_file_noreplace(staged_metadata, metadata_output)
    except Exception:
        staged_metadata.unlink(missing_ok=True)
        staged_npz.unlink(missing_ok=True)
        # Never unlink a public path after publication: an orphaned first member
        # is nonauthorizing without its typed metadata partner.
        raise
    return CanonicalKanePoissonArchiveReceipt(
        npz_path=npz_output,
        metadata_path=metadata_output,
        npz_sha256=npz_digest,
        metadata_sha256=reloaded.metadata_sha256,
        result_fingerprint=result_fingerprint,
        source_authority=reloaded.source_authority,
    )


def _decode_canonical_kane_poisson_archive(
    npz_path: str | Path,
    metadata_path: str | Path,
) -> tuple[
    CanonicalKanePoissonResult,
    CanonicalKanePoissonSourceAuthority,
    str,
    str,
]:
    with open_stable_binary_file(metadata_path) as metadata_handle:
        metadata_bytes = metadata_handle.read()
        metadata_sha256 = file_object_sha256(metadata_handle)
        metadata_value = parse_json_artifact(metadata_bytes)
    if not isinstance(metadata_value, dict):
        raise TypeError("canonical Kane--Poisson metadata must be an object")
    metadata = metadata_value
    schema = metadata.get("schema")
    if schema is None:
        raise ValueError("canonical Kane--Poisson archive schema is required")
    if schema != CANONICAL_KANE_POISSON_ARCHIVE_SCHEMA:
        raise ValueError(f"unsupported canonical Kane--Poisson archive schema {schema!r}")
    expected_top_level = {
        "schema",
        "result_fingerprint",
        "npz_sha256",
        "array_manifest",
        "source_authority",
        "config",
        "parent_spec",
        "final_eigensystem",
        "reference",
        "history",
        "result",
    }
    if set(metadata) != expected_top_level:
        raise ValueError("canonical Kane--Poisson metadata keys changed")
    with open_stable_binary_file(npz_path) as npz_handle:
        npz_sha256 = file_object_sha256(npz_handle)
        if metadata.get("npz_sha256") != npz_sha256:
            raise ValueError("canonical Kane--Poisson archive NPZ hash mismatch")
        with np.load(npz_handle, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if file_object_sha256(npz_handle) != npz_sha256:
            raise RuntimeError("canonical Kane--Poisson NPZ changed during load")
    if set(arrays) != _CANONICAL_ARRAY_KEYS:
        raise ValueError("canonical Kane--Poisson archive array keys changed")
    manifest = metadata.get("array_manifest")
    if not isinstance(manifest, dict) or set(manifest) != set(arrays):
        raise ValueError("canonical Kane--Poisson array manifest keys changed")
    for name, values in arrays.items():
        record = manifest[name]
        if not isinstance(record, dict) or set(record) != {
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(
                f"canonical Kane--Poisson array {name!r} manifest changed"
            )
        if (
            str(values.dtype) != record.get("dtype")
            or list(values.shape) != record.get("shape")
            or kane_poisson_array_sha256(values) != record.get("sha256")
        ):
            raise ValueError(
                f"canonical Kane--Poisson array {name!r} failed integrity"
            )
    parent_meta = metadata["parent_spec"]
    if not isinstance(parent_meta, dict):
        raise TypeError("canonical Kane--Poisson parent_spec must be an object")
    strict_parent_keys = {
        "fingerprint",
        "parent_hilbert_dimension",
        "selected_band_indices",
        "candidate_eigenpair_count",
        "target_energy_mev",
        "window_selection_label",
        "static_parent_sha256",
        "material_profile_sha256",
        "kdotpy_source_sha256",
        "hamiltonian_options",
        "basis_flattening_label",
        "potential_coupling_label",
        "schema_version",
    }
    if set(parent_meta) != strict_parent_keys:
        raise ValueError("canonical Kane--Poisson parent_spec keys changed")
    parent_spec = KaneParentSourceSpec(
        k_cart_nm_inv=arrays["k_cart_nm_inv"],
        k_weights_nm2=arrays["k_weights_nm2"],
        z_nm=arrays["z_nm"],
        z_weights_nm=arrays["z_weights_nm"],
        epsilon_r=arrays["epsilon_r"],
        parent_hilbert_dimension=int(parent_meta["parent_hilbert_dimension"]),
        selected_band_indices=tuple(parent_meta["selected_band_indices"]),
        candidate_eigenpair_count=int(parent_meta["candidate_eigenpair_count"]),
        target_energy_mev=float(parent_meta["target_energy_mev"]),
        window_selection_label=str(parent_meta["window_selection_label"]),
        static_parent_sha256=str(parent_meta["static_parent_sha256"]),
        material_profile_sha256=str(parent_meta["material_profile_sha256"]),
        kdotpy_source_sha256=str(parent_meta["kdotpy_source_sha256"]),
        hamiltonian_options=tuple(tuple(item) for item in parent_meta["hamiltonian_options"]),
        basis_flattening_label=str(
            parent_meta.get("basis_flattening_label", CANONICAL_FLATTENING_LABEL)
        ),
        potential_coupling_label=str(
            parent_meta.get("potential_coupling_label", POTENTIAL_COUPLING_LABEL)
        ),
        schema_version=int(
            parent_meta.get("schema_version", KANE_POISSON_SCHEMA_VERSION)
        ),
    )
    if parent_spec.fingerprint != parent_meta["fingerprint"]:
        raise ValueError("reloaded parent spec fingerprint differs from metadata")

    if not isinstance(metadata["config"], dict):
        raise TypeError("canonical Kane--Poisson config must be an object")
    config_meta = dict(metadata["config"])
    if set(config_meta) != {
        field.name for field in fields(CanonicalKanePoissonConfig)
    }:
        raise ValueError("canonical Kane--Poisson config keys changed")
    root_tolerances = ChemicalPotentialRootTolerances(
        **config_meta.pop("root_tolerances")
    )
    config = CanonicalKanePoissonConfig(
        **config_meta,
        root_tolerances=root_tolerances,
    )
    source_meta = metadata["final_eigensystem"]
    if not isinstance(source_meta, dict):
        raise TypeError("canonical final_eigensystem must be an object")
    if set(source_meta) != {
        "selected_band_indices",
        "parent_hilbert_dimension",
        "calculation_split_mev",
        "energy_zero_label",
        "input_potential_sha256",
        "potential_operator_fingerprint",
        "parent_spec_fingerprint",
        "replayed_static_parent_sha256",
        "static_parent_replay_max_error_mev",
        "potential_operator_max_error_mev",
        "provenance_fingerprint",
    }:
        raise ValueError("canonical final_eigensystem keys changed")
    final_source = KaneWindowAtPotential(
        hamiltonian_mev=arrays["hamiltonian_mev"],
        micro_wavefunctions=arrays["micro_wavefunctions"],
        selected_band_indices=tuple(source_meta["selected_band_indices"]),
        parent_hilbert_dimension=int(source_meta["parent_hilbert_dimension"]),
        calculation_split_mev=float(source_meta["calculation_split_mev"]),
        energy_zero_label=str(source_meta["energy_zero_label"]),
        input_potential_sha256=str(source_meta["input_potential_sha256"]),
        potential_operator_fingerprint=str(source_meta["potential_operator_fingerprint"]),
        parent_spec_fingerprint=str(source_meta["parent_spec_fingerprint"]),
        replayed_static_parent_sha256=str(source_meta["replayed_static_parent_sha256"]),
        static_parent_replay_max_error_mev=float(
            source_meta["static_parent_replay_max_error_mev"]
        ),
        potential_operator_max_error_mev=float(source_meta["potential_operator_max_error_mev"]),
        provenance_fingerprint=str(source_meta["provenance_fingerprint"]),
    )
    reference_meta = metadata["reference"]
    if not isinstance(reference_meta, dict):
        raise TypeError("canonical reference metadata must be an object")
    expected_reference_keys = {
        "contract",
        "tolerances",
        "temperature_K",
        "mu_mev",
        "electron_density_nm2",
        "hole_density_nm2",
        "target_metric_occupation_nm2",
        "achieved_metric_occupation_nm2",
        "number_residual_nm2",
        "neutrality_residual_nm2",
        "chemical_potential_bracket_mev",
        "bracket_residuals_nm2",
        "root_iterations",
        "root_function_calls",
        "root_converged",
        "frame_orthonormality_error",
        "orbital_projector_completeness_error",
        "hamiltonian_hermiticity_error_mev",
    }
    if set(reference_meta) != expected_reference_keys:
        raise ValueError("canonical reference metadata keys changed")
    if not isinstance(reference_meta["contract"], dict):
        raise TypeError("canonical reference contract must be an object")
    contract_meta = dict(reference_meta["contract"])
    if set(contract_meta) != {
        field.name for field in fields(KaneNormalReferenceContract)
    }:
        raise ValueError("canonical reference contract keys changed")
    contract_meta["selected_band_indices"] = tuple(contract_meta["selected_band_indices"])
    contract_meta["kane_orbital_labels"] = tuple(contract_meta["kane_orbital_labels"])
    contract = KaneNormalReferenceContract(**contract_meta)
    if not isinstance(reference_meta["tolerances"], dict):
        raise TypeError("canonical reference tolerances must be an object")
    if set(reference_meta["tolerances"]) != {
        field.name for field in fields(ChemicalPotentialRootTolerances)
    }:
        raise ValueError("canonical reference tolerance keys changed")
    reference_tolerances = ChemicalPotentialRootTolerances(
        **reference_meta["tolerances"]
    )
    loaded_density = np.array(
        arrays["density_matrix"], dtype=np.complex128, copy=True, order="C"
    )
    loaded_energies = np.array(
        arrays["normal_eigenvalues_mev"], dtype=np.float64, copy=True, order="C"
    )
    loaded_density.setflags(write=False)
    loaded_energies.setflags(write=False)
    if type(reference_meta.get("root_converged")) is not bool:
        raise TypeError("canonical reference root_converged must be an exact boolean")
    reference = KaneChargeNeutralReference(
        contract=contract,
        tolerances=reference_tolerances,
        temperature_K=float(reference_meta["temperature_K"]),
        density=loaded_density,
        energies_mev=loaded_energies,
        mu_mev=float(reference_meta["mu_mev"]),
        electron_density_nm2=float(reference_meta["electron_density_nm2"]),
        hole_density_nm2=float(reference_meta["hole_density_nm2"]),
        target_metric_occupation_nm2=float(reference_meta["target_metric_occupation_nm2"]),
        achieved_metric_occupation_nm2=float(reference_meta["achieved_metric_occupation_nm2"]),
        number_residual_nm2=float(reference_meta["number_residual_nm2"]),
        neutrality_residual_nm2=float(reference_meta["neutrality_residual_nm2"]),
        chemical_potential_bracket_mev=tuple(reference_meta["chemical_potential_bracket_mev"]),
        bracket_residuals_nm2=tuple(reference_meta["bracket_residuals_nm2"]),
        root_iterations=int(reference_meta["root_iterations"]),
        root_function_calls=int(reference_meta["root_function_calls"]),
        root_converged=bool(reference_meta["root_converged"]),
        frame_orthonormality_error=float(reference_meta["frame_orthonormality_error"]),
        orbital_projector_completeness_error=float(reference_meta["orbital_projector_completeness_error"]),
        hamiltonian_hermiticity_error_mev=float(reference_meta["hamiltonian_hermiticity_error_mev"]),
    )
    history_payload = metadata["history"]
    if not isinstance(history_payload, list):
        raise TypeError("canonical history must be a list")
    expected_history_keys = {
        field.name for field in fields(KanePoissonIterationRecord)
    }
    if any(
        not isinstance(record, dict) or set(record) != expected_history_keys
        for record in history_payload
    ):
        raise ValueError("canonical history record keys changed")
    history = tuple(
        KanePoissonIterationRecord(**record) for record in history_payload
    )
    result_meta = metadata["result"]
    if not isinstance(result_meta, dict):
        raise TypeError("canonical result metadata must be an object")
    if set(result_meta) != {
        "fixed_point_residual_mev",
        "gauss_residual_mev_nm2",
        "converged",
        "replay_verified",
    }:
        raise ValueError("canonical result metadata keys changed")
    if any(
        type(result_meta.get(name)) is not bool
        for name in ("converged", "replay_verified")
    ):
        raise TypeError("canonical result status fields must be exact booleans")
    result = CanonicalKanePoissonResult(
        config=config,
        parent_spec=parent_spec,
        z_nm=arrays["z_nm"],
        z_weights_nm=arrays["z_weights_nm"],
        epsilon_r=arrays["epsilon_r"],
        k_weights_nm2=arrays["k_weights_nm2"],
        potential_mev=arrays["potential_mev"],
        electron_density_nm3=arrays["electron_density_nm3"],
        hole_density_nm3=arrays["hole_density_nm3"],
        reference=reference,
        final_eigensystem=final_source,
        history=history,
        fixed_point_residual_mev=float(result_meta["fixed_point_residual_mev"]),
        gauss_residual_mev_nm2=float(result_meta["gauss_residual_mev_nm2"]),
        converged=bool(result_meta["converged"]),
        replay_verified=bool(result_meta["replay_verified"]),
    )
    if canonical_kane_poisson_fingerprint(result) != metadata["result_fingerprint"]:
        raise ValueError("reloaded canonical Kane--Poisson fingerprint mismatch")
    authority = metadata.get("source_authority")
    if not isinstance(authority, dict):
        raise TypeError("canonical Kane--Poisson source_authority must be an object")
    expected_authority_keys = {
        "claim_scope",
        "credibility_blockers",
        "ordinary_electron_convention",
        "physical_fixed_gate_claim",
        "hartree_fock_authorized",
        "matrix_hf_scope",
    }
    if set(authority) != expected_authority_keys:
        raise ValueError("canonical Kane--Poisson source_authority keys changed")
    convention = authority["ordinary_electron_convention"]
    if not isinstance(convention, dict):
        raise TypeError("ordinary_electron_convention must be an object")
    source_authority = CanonicalKanePoissonSourceAuthority(
        claim_scope=str(authority["claim_scope"]),
        credibility_blockers=tuple(authority["credibility_blockers"]),
        ordinary_electron_convention=OrdinaryElectronConventionLock(
            **convention
        ),
        physical_fixed_gate_claim=authority["physical_fixed_gate_claim"],
        hartree_fock_authorized=authority["hartree_fock_authorized"],
        matrix_hf_scope=str(authority["matrix_hf_scope"]),
    )
    if (
        source_authority.claim_scope != result.claim_scope
        or source_authority.credibility_blockers != result.credibility_blockers
    ):
        raise ValueError("canonical Kane--Poisson source authority changed")
    return result, source_authority, npz_sha256, metadata_sha256


def load_canonical_kane_poisson_archive(
    npz_path: str | Path,
    metadata_path: str | Path,
) -> LoadedCanonicalKanePoissonArchive:
    """Load a strict current archive and retain its typed negative authority."""

    result, source_authority, npz_sha256, metadata_sha256 = (
        _decode_canonical_kane_poisson_archive(
            npz_path,
            metadata_path,
        )
    )
    return LoadedCanonicalKanePoissonArchive(
        result=result,
        source_authority=source_authority,
        npz_path=Path(npz_path),
        metadata_path=Path(metadata_path),
        npz_sha256=npz_sha256,
        metadata_sha256=metadata_sha256,
    )




__all__ = [
    "CANONICAL_KANE_POISSON_ARCHIVE_SCHEMA",
    "CanonicalKanePoissonArchiveReceipt",
    "CanonicalKanePoissonSourceAuthority",
    "LoadedCanonicalKanePoissonArchive",
    "canonical_kane_poisson_archive_arrays",
    "canonical_kane_poisson_archive_metadata",
    "load_canonical_kane_poisson_archive",
    "save_canonical_kane_poisson_archive",
]
