"""HTQG full-microscopic adapter for the reusable HF problem/engine API.

This module supplies orientation, occupation, and density-vertex callbacks.
It also provides an explicit source-bound builder for the screened-Coulomb,
reference-sea, zero-mode, and filling conventions used by completed prior HTQG
reciprocal-space HF runs. Wall profiles remain caller-supplied physical inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import threading
import weakref

import numpy as np

from mean_field.core.hf.coulomb import (
    ScreenedCoulombParams,
    real_space_cell_area_nm2_from_reciprocal,
    screened_coulomb,
)
from mean_field.core.hf.density_vertex import (
    DensityVertexInteractionSpec,
    DensityVertexReferenceSpec,
    SparseDensityVertex,
    hartree_fock_action,
    hartree_fock_action_and_energy,
    hartree_fock_energy,
    prepare_density_vertex_family,
)
from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
    InteractionEnergyResult,
)
from mean_field.core.hf.occupations import (
    GlobalOccupationResult,
    fermionic_density_diagnostics,
    global_canonical_occupations,
    global_fermi_occupations,
    helmholtz_free_energy_ev,
)
from mean_field.core.hf.problem import HartreeFockKernel, HartreeFockProblem

from .lattice import HTQGLattice, hex_shell_indices
from .microscopic_supercell import (
    BatchAFoldMap,
    BatchAPhysicalLabelSupport,
    MicroscopicBasisLayout,
    MicroscopicH0Artifact,
    validate_batch_a_layout_support,
    validate_microscopic_h0_artifact,
    batch_a_physical_label_support_sha256,
    build_plane_wave_density_vertices,
    ket_blocks_to_repository_stored,
    ket_hamiltonian_to_repository_blocks,
    repository_hamiltonian_to_ket_blocks,
    repository_stored_to_ket_blocks,
)

Array = np.ndarray

# Completed reciprocal-space HTQG HF authority used for this convention port.
RECIPROCAL_HF_AUTHORITY = (
    "results/HTQG_Fujimoto2025_hf/active8_hf_eps5_nu_m3_reduced_v1/"
    "source_capsule_v14_controlplane_manifest_key_fix"
)
RECIPROCAL_HF_AUTHORITY_SHA256 = {
    "SOURCE_MANIFEST.json": "562d0d8140fb5277a5fe19c6167032204df767c2cfc1c2a3c2a7a4220827ee95",
    "SOURCE_MANIFEST.sha256": "361c2c47c3bbab69c13e86ffbf1066066dbcaf9ca47d13a6714269f95708ec41",
    "configs/production.json": "0258e92974eb352f6704f72a73b2e3d2ce4d9036cd1f55075b6b1772cf9ffd8f",
    "source_snapshot/src/mean_field/systems/htqg/hf.py": "d70b2158d13ebcb2f88d3d10b01ea31ff07f4f94eea3aa6dc920b885a3aece8b",
    "source_snapshot/src/mean_field/core/hf/coulomb.py": "27c26a4491fb5d713dcc96a281647787cf97310c7397452ed9cbda0dd84e80a7",
    "source_snapshot/src/mean_field/core/hf/interaction.py": "d54b0b388988471127583fe9757ba856dd05f4e062d14e1f52d11cb834ae8dbe",
}


def _validate_support_sha256(value: str) -> str:
    resolved = str(value)
    if len(resolved) != 64 or any(
        character not in "0123456789abcdef" for character in resolved
    ):
        raise ValueError("support_sha256 must be one lowercase SHA-256 digest")
    return resolved


def _validate_support_source_binding(
    vertices: Sequence[SparseDensityVertex],
    interaction_spec: DensityVertexInteractionSpec,
    references: DensityVertexReferenceSpec,
    support_sha256: str,
) -> str:
    """Require one support identity on every coupled HF input surface."""

    resolved = _validate_support_sha256(support_sha256)
    token = f"support_sha256={resolved}"
    for index, vertex in enumerate(vertices):
        if token not in str(vertex.provenance):
            raise ValueError(
                f"density vertex {index} is not bound to the supplied support_sha256"
            )
    if token not in str(interaction_spec.normalization_provenance):
        raise ValueError("interaction spec is not bound to the supplied support_sha256")
    if token not in str(references.provenance):
        raise ValueError("references are not bound to the supplied support_sha256")
    return resolved


def _sha256_array(values: Array, *, dtype: np.dtype) -> str:
    array = np.asarray(values, dtype=dtype, order="C")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_vertices(vertices: Sequence[SparseDensityVertex]) -> str:
    digest = hashlib.sha256()
    items = tuple(vertices)
    digest.update(np.asarray((len(items),), dtype="<i8").tobytes())
    for vertex in items:
        digest.update(np.asarray(vertex.label, dtype="<i8").tobytes())
        for name in ("target_k", "source_k", "rows", "columns"):
            digest.update(
                bytes.fromhex(
                    _sha256_array(getattr(vertex, name), dtype=np.dtype("<i8"))
                )
            )
        digest.update(
            bytes.fromhex(_sha256_array(vertex.values, dtype=np.dtype("<c16")))
        )
        digest.update(np.asarray((vertex.weight,), dtype="<f8").tobytes())
        digest.update(vertex.provenance.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _frozen_array_copy(values: Array, *, dtype: np.dtype) -> Array:
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )


def _frozen_vertex_copy(vertex: SparseDensityVertex) -> SparseDensityVertex:
    return SparseDensityVertex(
        label=vertex.label,
        target_k=_frozen_array_copy(vertex.target_k, dtype=np.dtype(np.int64)),
        source_k=_frozen_array_copy(vertex.source_k, dtype=np.dtype(np.int64)),
        rows=_frozen_array_copy(vertex.rows, dtype=np.dtype(np.int64)),
        columns=_frozen_array_copy(vertex.columns, dtype=np.dtype(np.int64)),
        values=_frozen_array_copy(vertex.values, dtype=np.dtype(np.complex128)),
        weight=vertex.weight,
        provenance=vertex.provenance,
    )


def _frozen_reference_copy(
    references: DensityVertexReferenceSpec,
) -> DensityVertexReferenceSpec:
    return DensityVertexReferenceSpec(
        hartree=(
            None
            if references.hartree is None
            else _frozen_array_copy(references.hartree, dtype=np.dtype(np.complex128))
        ),
        fock=(
            None
            if references.fock is None
            else _frozen_array_copy(references.fock, dtype=np.dtype(np.complex128))
        ),
        provenance=str(references.provenance),
    )


def _frozen_interaction_spec_copy(
    spec: DensityVertexInteractionSpec,
) -> DensityVertexInteractionSpec:
    """Detach scalar interaction policy from a caller-visible frozen dataclass."""

    if type(spec) is not DensityVertexInteractionSpec:
        raise TypeError("interaction_spec must be an exact DensityVertexInteractionSpec")
    return DensityVertexInteractionSpec(
        include_hartree=spec.include_hartree,
        include_fock=spec.include_fock,
        hartree_reference_policy=spec.hartree_reference_policy,
        fock_reference_policy=spec.fock_reference_policy,
        zero_mode_policy=spec.zero_mode_policy,
        closure_tolerance=spec.closure_tolerance,
        normalization_provenance=spec.normalization_provenance,
        unresolved_physical_choices=tuple(spec.unresolved_physical_choices),
    )

def _validate_reference_arrays(
    references: DensityVertexReferenceSpec,
    interaction_spec: DensityVertexInteractionSpec,
    *,
    ensemble_shape: tuple[int, int, int],
) -> None:
    for name in ("hartree", "fock"):
        values = getattr(references, name)
        enabled = bool(getattr(interaction_spec, f"include_{name}"))
        policy = str(getattr(interaction_spec, f"{name}_reference_policy"))
        if enabled and policy == "subtract_explicit" and values is None:
            raise ValueError(f"enabled {name} term requires an explicit reference")
        if enabled and policy == "absolute" and values is not None:
            raise ValueError(f"absolute {name} term forbids an explicit reference")
        if values is None:
            continue
        array = np.asarray(values, dtype=np.complex128)
        if array.shape != ensemble_shape or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} reference must be finite with shape {ensemble_shape}"
            )
        scale = max(1.0, float(np.max(np.abs(array), initial=0.0)))
        tolerance = 128.0 * np.finfo(float).eps * scale
        residual = float(
            np.max(
                np.abs(array - array.conj().transpose(0, 2, 1)),
                initial=0.0,
            )
        )
        if residual > tolerance:
            raise ValueError(
                f"{name} reference must be Hermitian; residual={residual:.3e}"
            )


def _sha256_interaction_spec(spec: DensityVertexInteractionSpec) -> str:
    payload = (
        f"{type(spec).__module__}.{type(spec).__qualname__}",
        ("include_hartree", spec.include_hartree),
        ("include_fock", spec.include_fock),
        ("hartree_reference_policy", spec.hartree_reference_policy),
        ("fock_reference_policy", spec.fock_reference_policy),
        ("zero_mode_policy", spec.zero_mode_policy),
        ("closure_tolerance", float(spec.closure_tolerance).hex()),
        ("normalization_provenance", spec.normalization_provenance),
        ("unresolved_physical_choices", tuple(spec.unresolved_physical_choices)),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _sha256_references(
    references: DensityVertexReferenceSpec,
    *,
    ensemble_shape: tuple[int, int, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{type(references).__module__}.{type(references).__qualname__}".encode(
            "utf-8"
        )
        + b"\0"
    )
    digest.update(np.asarray(ensemble_shape, dtype="<i8").tobytes(order="C"))
    for name in ("hartree", "fock"):
        values = getattr(references, name)
        digest.update(name.encode("ascii") + b"\0")
        if values is None:
            digest.update(b"NONE\0")
        else:
            digest.update(
                bytes.fromhex(_sha256_array(values, dtype=np.dtype("<c16")))
            )
    digest.update(references.provenance.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _resolve_occupation_ensemble(
    *,
    nk: int,
    kbt_ev: float | None,
    k_weights: Array | None,
) -> tuple[str, float | None, Array, str | None]:
    """Canonicalize the one-global-mu occupation ensemble for source binding."""

    if kbt_ev is None:
        if k_weights is not None:
            raise ValueError("finite-temperature options require explicit finite kbt_ev")
        return (
            "zero_temperature_global_canonical",
            None,
            np.ones(nk, dtype=float),
            None,
        )
    temperature = float(kbt_ev)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("finite-temperature kbt_ev must be finite and positive")
    if k_weights is None:
        raise ValueError("finite-temperature HTQG requires explicit k_weights")
    frozen_weights = _frozen_array_copy(
        np.asarray(k_weights), dtype=np.dtype(np.float64)
    )
    if frozen_weights.shape != (nk,):
        raise ValueError(
            f"k_weights must have shape {(nk,)}, got {frozen_weights.shape}"
        )
    if not np.all(np.isfinite(frozen_weights)) or np.any(frozen_weights <= 0.0):
        raise ValueError("finite-temperature HTQG k_weights must be finite and positive")
    if not np.array_equal(
        frozen_weights,
        np.full(frozen_weights.shape, frozen_weights[0], dtype=np.float64),
    ):
        raise ValueError(
            "finite-temperature HTQG currently requires exactly uniform k_weights"
        )
    return (
        "finite_temperature_global_fermi",
        temperature,
        frozen_weights,
        _sha256_array(frozen_weights, dtype=np.dtype("<f8")),
    )


def _reciprocal_authority_sha256() -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(RECIPROCAL_HF_AUTHORITY_SHA256.items()):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


_MICROSCOPIC_HF_SOURCE_BINDING_FACTORY_TOKEN = object()
_MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY_LOCK = threading.RLock()
_MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType["MicroscopicHFSourceBinding"],
        tuple[object, ...],
        Path,
    ],
] = {}


@dataclass(frozen=True, init=False)
class MicroscopicHFSourceBinding:
    """Factory-only content identity joining one exact microscopic-HF ensemble."""

    support_sha256: str
    h0_sha256: str
    vertices_sha256: str
    interaction_sha256: str
    references_sha256: str
    authority_sha256: str
    derivation_authority: str
    occupation_ensemble: str
    kbt_ev: float | None
    k_weights_sha256: str | None
    nk: int
    dimension: int
    total_occupied: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "MicroscopicHFSourceBinding must be created by "
            "build_microscopic_hf_source_binding"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("MicroscopicHFSourceBinding does not permit subclasses")

    def __copy__(self) -> "MicroscopicHFSourceBinding":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "MicroscopicHFSourceBinding":
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("MicroscopicHFSourceBinding does not permit pickling")


def _new_microscopic_hf_source_binding(
    *,
    support_sha256: str,
    h0_sha256: str,
    vertices_sha256: str,
    interaction_sha256: str,
    references_sha256: str,
    authority_sha256: str,
    derivation_authority: str,
    occupation_ensemble: str,
    kbt_ev: float | None,
    k_weights_sha256: str | None,
    nk: int,
    dimension: int,
    total_occupied: int,
    authority_capsule_root: str | Path,
) -> MicroscopicHFSourceBinding:
    if type(nk) is not int or nk <= 0:
        raise ValueError("source-binding nk must be a positive exact integer")
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("source-binding dimension must be a positive exact integer")
    if (
        type(total_occupied) is not int
        or total_occupied < 0
        or total_occupied > nk * dimension
    ):
        raise ValueError(
            "source-binding total_occupied must be an exact in-range integer"
        )
    out = object.__new__(MicroscopicHFSourceBinding)
    object.__setattr__(
        out,
        "_factory_token",
        _MICROSCOPIC_HF_SOURCE_BINDING_FACTORY_TOKEN,
    )
    for name, value in (
        ("support_sha256", support_sha256),
        ("h0_sha256", h0_sha256),
        ("vertices_sha256", vertices_sha256),
        ("interaction_sha256", interaction_sha256),
        ("references_sha256", references_sha256),
        ("authority_sha256", authority_sha256),
    ):
        object.__setattr__(out, name, _validate_support_sha256(value))
    if derivation_authority not in {
        "identity_only_unverified",
        "batch_a_direct_h0_artifact",
    }:
        raise ValueError("unsupported microscopic-HF derivation authority")
    object.__setattr__(out, "derivation_authority", derivation_authority)
    if occupation_ensemble not in {
        "zero_temperature_global_canonical",
        "finite_temperature_global_fermi",
    }:
        raise ValueError("unsupported microscopic-HF occupation ensemble")
    if occupation_ensemble == "zero_temperature_global_canonical":
        if kbt_ev is not None or k_weights_sha256 is not None:
            raise ValueError("zero-temperature binding must not carry thermal data")
        resolved_kbt = None
        resolved_weights_sha256 = None
    else:
        resolved_kbt = float(kbt_ev) if kbt_ev is not None else float("nan")
        if not math.isfinite(resolved_kbt) or resolved_kbt <= 0.0:
            raise ValueError("finite-temperature binding requires positive kbt_ev")
        if k_weights_sha256 is None:
            raise ValueError("finite-temperature binding requires k-weight identity")
        resolved_weights_sha256 = _validate_support_sha256(k_weights_sha256)
    object.__setattr__(out, "occupation_ensemble", occupation_ensemble)
    object.__setattr__(out, "kbt_ev", resolved_kbt)
    object.__setattr__(out, "k_weights_sha256", resolved_weights_sha256)
    object.__setattr__(out, "nk", nk)
    object.__setattr__(out, "dimension", dimension)
    object.__setattr__(out, "total_occupied", total_occupied)
    expected = tuple(
        getattr(out, name)
        for name in (
            "support_sha256",
            "h0_sha256",
            "vertices_sha256",
            "interaction_sha256",
            "references_sha256",
            "authority_sha256",
            "derivation_authority",
            "occupation_ensemble",
            "kbt_ev",
            "k_weights_sha256",
            "nk",
            "dimension",
            "total_occupied",
        )
    )
    registry_key = id(out)

    def remove_registry_entry(reference: object, *, key: int = registry_key) -> None:
        with _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY_LOCK:
            current = _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY.pop(key, None)

    authority_root = Path(authority_capsule_root).resolve(strict=True)
    reference = weakref.ref(out, remove_registry_entry)
    with _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY_LOCK:
        _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY[registry_key] = (
            reference,
            expected,
            authority_root,
        )
    return out


def _require_microscopic_hf_source_binding(
    value: object,
    *,
    verify_authority_files: bool = False,
) -> MicroscopicHFSourceBinding:
    if type(value) is not MicroscopicHFSourceBinding:
        raise TypeError(
            "source_binding must be an exact factory-built "
            "MicroscopicHFSourceBinding"
        )
    with _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY_LOCK:
        registered = _MICROSCOPIC_HF_SOURCE_BINDING_REGISTRY.get(id(value))
    current = tuple(
        getattr(value, name, None)
        for name in (
            "support_sha256",
            "h0_sha256",
            "vertices_sha256",
            "interaction_sha256",
            "references_sha256",
            "authority_sha256",
            "derivation_authority",
            "occupation_ensemble",
            "kbt_ev",
            "k_weights_sha256",
            "nk",
            "dimension",
            "total_occupied",
        )
    )
    if (
        registered is None
        or registered[0]() is not value
        or current != registered[1]
    ):
        raise TypeError(
            "source_binding must be an exact factory-built "
            "MicroscopicHFSourceBinding"
        )
    if verify_authority_files:
        verify_reciprocal_hf_authority(registered[2])
        if value.authority_sha256 != _reciprocal_authority_sha256():
            raise ValueError(
                "source binding authority digest does not match pinned authority"
            )
    return value


@dataclass(frozen=True)
class MicroscopicScreenedCoulombInputs:
    """Resolved reciprocal-space inputs matching prior HTQG HF conventions."""

    vertices: tuple[SparseDensityVertex, ...]
    interaction_spec: DensityVertexInteractionSpec
    references: DensityVertexReferenceSpec
    total_occupied: int
    neutral_occupied: int
    primitive_filling: int
    primitive_area_nm2: float
    supercell_area_nm2: float
    quadrature_normalization_nm_inv_sq: float
    screened_params: ScreenedCoulombParams
    support_sha256: str
    source_binding: MicroscopicHFSourceBinding
    authority_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        vertices = tuple(_frozen_vertex_copy(vertex) for vertex in self.vertices)
        if not vertices:
            raise ValueError("screened-Coulomb inputs require nonempty vertices")
        interaction_spec = _frozen_interaction_spec_copy(self.interaction_spec)
        references = _frozen_reference_copy(self.references)
        resolved = _validate_support_source_binding(
            vertices,
            interaction_spec,
            references,
            self.support_sha256,
        )
        binding = _require_microscopic_hf_source_binding(self.source_binding)
        _validate_reference_arrays(
            references,
            interaction_spec,
            ensemble_shape=(binding.nk, binding.dimension, binding.dimension),
        )
        if binding.support_sha256 != resolved:
            raise ValueError("source binding support digest does not match inputs")
        if binding.total_occupied != self.total_occupied:
            raise ValueError(
                "source binding total_occupied does not match screened inputs"
            )
        if binding.vertices_sha256 != _sha256_vertices(vertices):
            raise ValueError("source binding vertex digest does not match inputs")
        if binding.interaction_sha256 != _sha256_interaction_spec(
            interaction_spec
        ):
            raise ValueError("source binding interaction digest does not match inputs")
        if binding.references_sha256 != _sha256_references(
            references,
            ensemble_shape=(binding.nk, binding.dimension, binding.dimension),
        ):
            raise ValueError("source binding reference digest does not match inputs")
        if binding.authority_sha256 != _reciprocal_authority_sha256():
            raise ValueError("source binding authority digest does not match pinned authority")
        if not self.authority_paths or any(
            not str(path).strip() for path in self.authority_paths
        ):
            raise ValueError("screened-Coulomb inputs require authority paths")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "interaction_spec", interaction_spec)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "support_sha256", resolved)
        object.__setattr__(
            self, "authority_paths", tuple(str(path) for path in self.authority_paths)
        )


_MICROSCOPIC_HF_PROBLEM_FACTORY_TOKEN = object()
_MICROSCOPIC_HF_PROBLEM_REGISTRY_LOCK = threading.RLock()
_MICROSCOPIC_HF_PROBLEM_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType["MicroscopicHartreeFockProblem"],
        HartreeFockProblem,
        str | None,
        MicroscopicHFSourceBinding | None,
        int,
        int,
    ],
] = {}


def _detached_hf_execution_lease(execution: HartreeFockProblem) -> HartreeFockProblem:
    """Return a one-run shell detached from the canonical registry snapshot."""

    return HartreeFockProblem(
        initializer=execution.initializer,
        kernel=replace(execution.kernel),
    )


@dataclass(frozen=True, init=False)
class MicroscopicHartreeFockProblem(HartreeFockProblem):
    """HF problem carrying the support identity used by caches/checkpoints."""

    support_sha256: str | None = None
    source_binding: MicroscopicHFSourceBinding | None = None
    nk: int | None = None
    dimension: int | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "MicroscopicHartreeFockProblem must be created by "
            "build_microscopic_hf_problem"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("MicroscopicHartreeFockProblem does not permit subclasses")

    def __copy__(self) -> "MicroscopicHartreeFockProblem":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "MicroscopicHartreeFockProblem":
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("MicroscopicHartreeFockProblem does not permit pickling")

    def __post_init__(self) -> None:
        if getattr(self, "_factory_token", None) is not (
            _MICROSCOPIC_HF_PROBLEM_FACTORY_TOKEN
        ):
            raise TypeError(
                "MicroscopicHartreeFockProblem must be exact and factory-built"
            )
        if self.support_sha256 is not None:
            object.__setattr__(
                self, "support_sha256", _validate_support_sha256(self.support_sha256)
            )
        if self.source_binding is not None:
            _require_microscopic_hf_source_binding(self.source_binding)
        if self.source_binding is not None and (
            self.support_sha256 != self.source_binding.support_sha256
        ):
            raise ValueError("problem support digest does not match source binding")
        if self.source_binding is not None and (
            self.nk != self.source_binding.nk
            or self.dimension != self.source_binding.dimension
        ):
            raise ValueError("problem ensemble shape does not match source binding")
        if self.nk is not None and (type(self.nk) is not int or self.nk <= 0):
            raise ValueError("problem nk must be a positive exact integer")
        if self.dimension is not None and (
            type(self.dimension) is not int or self.dimension <= 0
        ):
            raise ValueError("problem dimension must be a positive exact integer")

    def validate_before_execution(
        self, state: HartreeFockStateProtocol
    ) -> HartreeFockProblem:
        """Return the private execution snapshot after source/state preflight."""

        if type(self) is not MicroscopicHartreeFockProblem:
            raise TypeError(
                "MicroscopicHartreeFockProblem must be exact and factory-built"
            )
        with _MICROSCOPIC_HF_PROBLEM_REGISTRY_LOCK:
            registered = _MICROSCOPIC_HF_PROBLEM_REGISTRY.get(id(self))
        if registered is None or registered[0]() is not self:
            raise TypeError(
                "MicroscopicHartreeFockProblem must be exact and factory-built"
            )
        execution, support_sha256, source_binding, nk, dimension = registered[1:]
        if (
            self.support_sha256 != support_sha256
            or self.source_binding is not source_binding
            or self.nk != nk
            or self.dimension != dimension
        ):
            raise TypeError(
                "MicroscopicHartreeFockProblem must be exact and factory-built"
            )
        if source_binding is None:
            if getattr(state, "source_binding", None) is not None:
                raise ValueError("unbound HF problem rejects a source-bound state")
            return _detached_hf_execution_lease(execution)
        binding = _require_microscopic_hf_source_binding(
            source_binding, verify_authority_files=True
        )
        if type(state) is not MicroscopicHFState:
            raise TypeError(
                "source-bound microscopic HF requires an exact MicroscopicHFState"
            )
        if state.source_binding is not source_binding:
            raise ValueError("HF state is not bound to the problem source lineage")
        _require_microscopic_hf_source_binding(state.source_binding)
        if state.nk != nk or state.h0.shape[:2] != (dimension, dimension):
            raise ValueError("HF state ensemble shape does not match problem binding")
        h0_ket = repository_hamiltonian_to_ket_blocks(state.h0)
        if _sha256_array(h0_ket, dtype=np.dtype("<c16")) != binding.h0_sha256:
            raise ValueError("HF state H0 digest does not match problem binding")
        if support_sha256 != binding.support_sha256:
            raise ValueError("problem support digest does not match source binding")
        return _detached_hf_execution_lease(execution)


def _new_microscopic_hf_problem(
    *,
    initializer: object,
    kernel: HartreeFockKernel,
    support_sha256: str | None,
    source_binding: MicroscopicHFSourceBinding | None,
    nk: int,
    dimension: int,
) -> MicroscopicHartreeFockProblem:
    out = object.__new__(MicroscopicHartreeFockProblem)
    object.__setattr__(out, "_factory_token", _MICROSCOPIC_HF_PROBLEM_FACTORY_TOKEN)
    object.__setattr__(out, "_factory_initializer", initializer)
    object.__setattr__(out, "_factory_kernel", kernel)
    object.__setattr__(out, "initializer", initializer)
    object.__setattr__(out, "kernel", kernel)
    object.__setattr__(out, "support_sha256", support_sha256)
    object.__setattr__(out, "source_binding", source_binding)
    object.__setattr__(out, "nk", nk)
    object.__setattr__(out, "dimension", dimension)
    out.__post_init__()
    # Keep the executable kernel structurally separate from the caller-visible
    # frozen dataclass.  ``object.__setattr__`` can replace fields even on a
    # frozen instance; a shared kernel would therefore let public shadow
    # mutation alter the private registry execution graph.
    execution = HartreeFockProblem(initializer=initializer, kernel=replace(kernel))
    registry_key = id(out)

    def remove_registry_entry(reference: object, *, key: int = registry_key) -> None:
        with _MICROSCOPIC_HF_PROBLEM_REGISTRY_LOCK:
            current = _MICROSCOPIC_HF_PROBLEM_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _MICROSCOPIC_HF_PROBLEM_REGISTRY.pop(key, None)

    reference = weakref.ref(out, remove_registry_entry)
    with _MICROSCOPIC_HF_PROBLEM_REGISTRY_LOCK:
        _MICROSCOPIC_HF_PROBLEM_REGISTRY[registry_key] = (
            reference,
            execution,
            support_sha256,
            source_binding,
            nk,
            dimension,
        )
    return out


@dataclass(frozen=True)
class MicroscopicHFFinalGateTolerances:
    """Explicit finite-temperature final-state acceptance thresholds."""

    particle_number_abs: float
    density_spectrum_abs: float
    commutator_rms_ev: float
    fermi_map_rms: float

    def __post_init__(self) -> None:
        for name in (
            "particle_number_abs",
            "density_spectrum_abs",
            "commutator_rms_ev",
            "fermi_map_rms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MicroscopicHFFiniteTConvergence:
    """Finite-temperature objective criteria supplementing density convergence."""

    helmholtz_change_abs_ev: float
    required_consecutive_iterations: int

    def __post_init__(self) -> None:
        tolerance = float(self.helmholtz_change_abs_ev)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "helmholtz_change_abs_ev must be finite and strictly positive"
            )
        if type(self.required_consecutive_iterations) is not int:
            raise TypeError("required_consecutive_iterations must be an exact integer")
        if self.required_consecutive_iterations < 1:
            raise ValueError("required_consecutive_iterations must be positive")
        object.__setattr__(self, "helmholtz_change_abs_ev", tolerance)


@dataclass
class MicroscopicHFState:
    """Mutable stored-orientation state accepted by the generic HF engine."""

    h0: Array
    density: Array
    hamiltonian: Array
    energies: Array
    mu: float
    precision: float
    source_binding: MicroscopicHFSourceBinding | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)
    _lineage_sealed: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name: str, value: object) -> None:
        if (
            name in {"h0", "source_binding"}
            and getattr(self, "_lineage_sealed", False)
        ):
            raise AttributeError(f"{name} is immutable after state construction")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.h0 = _frozen_array_copy(
            _stored_blocks(self.h0, name="h0"), dtype=np.dtype(np.complex128)
        )
        self.density = _stored_blocks(self.density, name="density").copy()
        self.hamiltonian = _stored_blocks(
            self.hamiltonian, name="hamiltonian"
        ).copy()
        if self.density.shape != self.h0.shape or self.hamiltonian.shape != self.h0.shape:
            raise ValueError("h0, density, and hamiltonian shapes must match")
        nb, _, nk = self.h0.shape
        energies = np.asarray(self.energies, dtype=float)
        if energies.shape != (nb, nk) or not np.all(np.isfinite(energies)):
            raise ValueError("energies must be finite with shape (nb,nk)")
        self.energies = energies.copy()
        self.mu = float(self.mu)
        self.precision = float(self.precision)
        if not math.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be finite and positive")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")
        if self.source_binding is not None:
            _require_microscopic_hf_source_binding(self.source_binding)
            if (
                nk != self.source_binding.nk
                or nb != self.source_binding.dimension
            ):
                raise ValueError("state ensemble shape does not match source binding")
            h0_ket = repository_hamiltonian_to_ket_blocks(self.h0)
            if _sha256_array(h0_ket, dtype=np.dtype("<c16")) != (
                self.source_binding.h0_sha256
            ):
                raise ValueError("state H0 digest does not match source binding")
        object.__setattr__(self, "_lineage_sealed", True)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_ket_h0(
        cls,
        h0_ket: Array,
        *,
        precision: float,
        initial_projector_ket: Array | None = None,
        source_binding: MicroscopicHFSourceBinding | None = None,
    ) -> "MicroscopicHFState":
        """Create an engine state without selecting the physical occupation."""

        h0 = np.asarray(h0_ket, dtype=np.complex128)
        if h0.ndim != 3 or h0.shape[1] != h0.shape[2] or not np.all(np.isfinite(h0)):
            raise ValueError("h0_ket must be finite with shape (nk,nb,nb)")
        _require_ket_hermitian(h0, name="h0_ket")
        if initial_projector_ket is None:
            projector = np.zeros_like(h0)
        else:
            projector = np.asarray(initial_projector_ket, dtype=np.complex128)
            if projector.shape != h0.shape or not np.all(np.isfinite(projector)):
                raise ValueError("initial_projector_ket must be finite and match h0_ket")
            _require_ket_hermitian(projector, name="initial_projector_ket")
        stored_h0 = ket_hamiltonian_to_repository_blocks(h0)
        return cls(
            h0=stored_h0,
            density=ket_blocks_to_repository_stored(projector),
            hamiltonian=stored_h0.copy(),
            energies=np.zeros((h0.shape[1], h0.shape[0]), dtype=float),
            mu=0.0,
            precision=precision,
            source_binding=source_binding,
        )


def _stored_blocks(values: Array, *, name: str) -> Array:
    out = np.asarray(values, dtype=np.complex128)
    if out.ndim != 3 or out.shape[0] != out.shape[1] or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite with shape (nb,nb,nk)")
    return out


def _require_ket_hermitian(values: Array, *, name: str) -> None:
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    tolerance = 128.0 * np.finfo(float).eps * scale
    residual = float(np.max(np.abs(values - values.conj().transpose(0, 2, 1)), initial=0.0))
    if residual > tolerance:
        raise ValueError(f"{name} must be Hermitian; residual={residual:.3e}")


def verify_reciprocal_hf_authority(capsule_root: str | Path) -> None:
    """Fail closed unless the convention authority matches pinned bytes."""

    root = Path(capsule_root)
    for relative, expected in RECIPROCAL_HF_AUTHORITY_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing reciprocal-HF authority file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"reciprocal-HF authority hash mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )


def _build_microscopic_hf_source_binding(
    *,
    h0_ket: Array,
    vertices: Sequence[SparseDensityVertex],
    interaction_spec: DensityVertexInteractionSpec,
    references: DensityVertexReferenceSpec,
    support: BatchAPhysicalLabelSupport,
    authority_capsule_root: str | Path,
    total_occupied: int,
    derivation_authority: str,
    h0_artifact: MicroscopicH0Artifact | None,
    kbt_ev: float | None,
    k_weights: Array | None,
) -> MicroscopicHFSourceBinding:
    """Internal common builder; only strict physical adapters grant authority."""

    verify_reciprocal_hf_authority(authority_capsule_root)
    if type(support) is not BatchAPhysicalLabelSupport:
        raise TypeError("support must be an exact BatchAPhysicalLabelSupport")
    if type(interaction_spec) is not DensityVertexInteractionSpec:
        raise TypeError("interaction_spec must be an exact DensityVertexInteractionSpec")
    if type(references) is not DensityVertexReferenceSpec:
        raise TypeError("references must be an exact DensityVertexReferenceSpec")

    # Snapshot every mutable numerical payload before validation and hashing.
    # The digest closure therefore describes one internally consistent read,
    # not caller-owned arrays that may change between validation and hashing.
    h0 = _frozen_array_copy(h0_ket, dtype=np.dtype(np.complex128))
    frozen_vertices = tuple(_frozen_vertex_copy(vertex) for vertex in vertices)
    frozen_interaction_spec = _frozen_interaction_spec_copy(interaction_spec)
    frozen_references = _frozen_reference_copy(references)
    resolved = batch_a_physical_label_support_sha256(support)
    if derivation_authority == "batch_a_direct_h0_artifact":
        if h0_artifact is None:
            raise ValueError("Batch-A derivation authority requires an H0 artifact")
        artifact = validate_microscopic_h0_artifact(h0_artifact)
        if artifact.support_sha256 != resolved:
            raise ValueError("H0 artifact support does not match source support")
        if not np.array_equal(h0, artifact.h0_ket):
            raise ValueError("h0_ket does not exactly replay the H0 artifact")
    elif derivation_authority != "identity_only_unverified":
        raise ValueError("unsupported microscopic-HF derivation authority")
    _validate_support_source_binding(
        frozen_vertices, frozen_interaction_spec, frozen_references, resolved
    )
    if h0.ndim != 3 or h0.shape[1] != h0.shape[2] or not np.all(np.isfinite(h0)):
        raise ValueError("h0_ket must be finite with shape (nk,nb,nb)")
    _require_ket_hermitian(h0, name="h0_ket")
    nk, dimension, _ = h0.shape
    occupation_ensemble, resolved_kbt, resolved_weights, k_weights_sha256 = (
        _resolve_occupation_ensemble(
            nk=nk,
            kbt_ev=kbt_ev,
            k_weights=k_weights,
        )
    )
    if occupation_ensemble == "finite_temperature_global_fermi" and not (
        np.array_equal(resolved_weights, np.ones(nk, dtype=np.float64))
    ):
        raise ValueError(
            "source-bound finite-temperature HTQG requires exact unit k_weights"
        )
    frozen_interaction_spec.validate(require_resolved=True)
    _validate_reference_arrays(
        frozen_references,
        frozen_interaction_spec,
        ensemble_shape=(nk, dimension, dimension),
    )
    return _new_microscopic_hf_source_binding(
        support_sha256=resolved,
        h0_sha256=_sha256_array(h0, dtype=np.dtype("<c16")),
        vertices_sha256=_sha256_vertices(frozen_vertices),
        interaction_sha256=_sha256_interaction_spec(frozen_interaction_spec),
        references_sha256=_sha256_references(
            frozen_references, ensemble_shape=(nk, dimension, dimension)
        ),
        authority_sha256=_reciprocal_authority_sha256(),
        derivation_authority=derivation_authority,
        occupation_ensemble=occupation_ensemble,
        kbt_ev=resolved_kbt,
        k_weights_sha256=k_weights_sha256,
        nk=int(nk),
        dimension=int(dimension),
        total_occupied=total_occupied,
        authority_capsule_root=authority_capsule_root,
    )


def build_microscopic_hf_source_binding(
    *,
    h0_ket: Array,
    vertices: Sequence[SparseDensityVertex],
    interaction_spec: DensityVertexInteractionSpec,
    references: DensityVertexReferenceSpec,
    support: BatchAPhysicalLabelSupport,
    authority_capsule_root: str | Path,
    total_occupied: int,
    kbt_ev: float | None = None,
    k_weights: Array | None = None,
) -> MicroscopicHFSourceBinding:
    """Build an identity-only closure; this is not production authorization."""

    return _build_microscopic_hf_source_binding(
        h0_ket=h0_ket,
        vertices=vertices,
        interaction_spec=interaction_spec,
        references=references,
        support=support,
        authority_capsule_root=authority_capsule_root,
        total_occupied=total_occupied,
        derivation_authority="identity_only_unverified",
        h0_artifact=None,
        kbt_ev=kbt_ev,
        k_weights=k_weights,
    )


def build_global_neutral_h0_reference(
    h0_ket: Array,
    *,
    fermi_degeneracy_tolerance: float,
    eigensolver_workers: int = 1,
) -> GlobalOccupationResult:
    """Build Scheme A: one-global-mu neutral projector of the supplied wall H0.

    Neutrality is half filling of the complete finite one-particle space.  The
    occupied rank may redistribute among reduced-K blocks.  The generic global
    occupation solver fails closed when the neutral Fermi boundary is degenerate
    within the caller's explicit tolerance.
    """

    h0 = np.asarray(h0_ket, dtype=np.complex128)
    if h0.ndim != 3 or h0.shape[1] != h0.shape[2] or not np.all(np.isfinite(h0)):
        raise ValueError("h0_ket must be finite with shape (nk,nb,nb)")
    _require_ket_hermitian(h0, name="h0_ket")
    capacity = int(h0.shape[0] * h0.shape[1])
    if capacity % 2:
        raise ValueError("the global charge-neutral H0 reference requires even capacity")
    return global_canonical_occupations(
        h0,
        total_occupied=capacity // 2,
        fermi_degeneracy_tolerance=fermi_degeneracy_tolerance,
        eigensolver_workers=eigensolver_workers,
    )


def _batch_a_authorized_shell(
    lattice: HTQGLattice,
    layout: MicroscopicBasisLayout,
) -> int | None:
    """Return the exact canonical shell authorized for Batch-A, if any."""

    if type(lattice) is not HTQGLattice or type(layout) is not MicroscopicBasisLayout:
        return None
    authorized = {
        1: (7, 896),
        2: (19, 2432),
        3: (37, 4736),
    }
    for shell, (n_g, dimension) in authorized.items():
        canonical_indices = hex_shell_indices(shell)
        if (
            lattice.n_shells == shell
            and lattice.n_g == n_g
            and lattice.matrix_dim == 8 * n_g
            and layout.dimension == dimension
            and np.array_equal(lattice.g_indices, canonical_indices)
            and np.array_equal(layout.g_indices, canonical_indices)
        ):
            return shell
    return None


def build_half_filled_h0_reference(h0_ket: Array) -> Array:
    """Build the historical neutral sea: lowest half of H0 at every K.

    This fixed-rank-per-K construction is retained for uniform historical replay.
    It is not Scheme A for an inhomogeneous wall; use
    :func:`build_global_neutral_h0_reference` for that choice.
    """

    h0 = np.asarray(h0_ket, dtype=np.complex128)
    if h0.ndim != 3 or h0.shape[1] != h0.shape[2] or not np.all(np.isfinite(h0)):
        raise ValueError("h0_ket must be finite with shape (nk,nb,nb)")
    _require_ket_hermitian(h0, name="h0_ket")
    nk, nb, _ = h0.shape
    if nb % 2:
        raise ValueError("the charge-neutral half-filled H0 reference requires even nb")
    reference = np.zeros_like(h0)
    neutral_rank = nb // 2
    for k in range(nk):
        _values, vectors = np.linalg.eigh(h0[k])
        occupied = vectors[:, :neutral_rank]
        reference[k] = occupied @ occupied.conj().T
    return reference


def build_screened_coulomb_hf_inputs(
    *,
    h0_ket: Array,
    lattice: HTQGLattice,
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
    authority_capsule_root: str | Path,
    epsilon_r: float,
    d_sc_nm: float,
    primitive_filling: int,
    max_vertex_entries: int,
    provenance: str,
    hartree_reference_ket: Array | None = None,
    reference_provenance: str | None = None,
    kbt_ev: float | None = None,
    k_weights: Array | None = None,
) -> MicroscopicScreenedCoulombInputs:
    """Apply the completed reciprocal-HF convention to a supplied 4x1 cutoff.

    The interaction is evaluated in reciprocal space,

    ``V(Q)=2*pi*(e^2/4*pi*eps0)/(epsilon_r*|Q|)*tanh(|Q|*d_sc)``,

    including its finite screened ``Q=0`` limit. Vertex weights are
    ``V(Q)/(A_supercell*N_K)``. Hartree uses ``P-P_ref`` for every Q and Fock
    uses absolute ``P``, including Fock Q=0. Outside the exact Batch-A
    geometry, omitting a reference retains the convenience convention of the
    lowest half of the supplied H0 at each K. Batch A requires an explicit
    representation-matched reference instead. Integer primitive filling
    changes the total electron count by ``filling*N_k_primitive``; occupation
    itself remains global under one chemical potential. An explicit reference
    must be a finite Hermitian block projector whose total trace is the neutral
    particle count. Its reduced-K ranks may vary, and explicit provenance is
    required. The returned source binding is content-identity only; production
    derivation authority is available exclusively through the strict Batch-A
    adapter.
    """

    verify_reciprocal_hf_authority(authority_capsule_root)
    if type(primitive_filling) is not int:
        raise TypeError("primitive_filling must be an exact integer")
    if type(max_vertex_entries) is not int or max_vertex_entries <= 0:
        raise ValueError("max_vertex_entries must be a positive exact integer")
    if not str(provenance).strip():
        raise ValueError("screened-Coulomb input provenance must be explicit")
    validate_batch_a_layout_support(layout, support)
    if not np.array_equal(support.seed_g_indices, layout.g_indices):
        raise ValueError("screened-Coulomb support seed must replay layout.g_indices")
    if support.labels.shape[:4] != (
        fold_map.reduced_nk,
        2,
        4,
        layout.g_indices.shape[0],
    ):
        raise ValueError("screened-Coulomb support does not match layout/mesh")
    support_sha256 = batch_a_physical_label_support_sha256(support)
    is_batch_a_geometry = (
        _batch_a_authorized_shell(lattice, layout) is not None
        and fold_map.primitive_mesh == (8, 4)
        and fold_map.reduced_mesh == (2, 4)
        and fold_map.supercell.area_ratio == 4
    )
    if hartree_reference_ket is None and is_batch_a_geometry:
        raise ValueError(
            "UNRESOLVED Batch-A supercell Hartree reference: supply an explicit "
            "globally neutral projector and provenance"
        )
    if hartree_reference_ket is not None and (
        reference_provenance is None or not str(reference_provenance).strip()
    ):
        raise ValueError("explicit Hartree reference provenance is required")

    h0 = np.asarray(h0_ket, dtype=np.complex128)
    if h0.shape != (fold_map.reduced_nk, layout.dimension, layout.dimension):
        raise ValueError("h0_ket does not match the supplied Batch-A basis and mesh")
    _require_ket_hermitian(h0, name="h0_ket")
    if hartree_reference_ket is None:
        reference = build_half_filled_h0_reference(h0)
        resolved_reference_provenance = "lowest nb/2 H0 states per supplied reduced K"
    else:
        reference = np.asarray(hartree_reference_ket, dtype=np.complex128)
        if reference.shape != h0.shape or not np.all(np.isfinite(reference)):
            raise ValueError("hartree_reference_ket must be finite and match h0_ket")
        _require_ket_hermitian(reference, name="hartree_reference_ket")
        total_trace = float(np.trace(reference, axis1=1, axis2=2).real.sum())
        neutral_trace = float(fold_map.reduced_nk * layout.dimension // 2)
        if not math.isclose(total_trace, neutral_trace, abs_tol=1.0e-8, rel_tol=0.0):
            raise ValueError(
                "Hartree reference must carry the neutral trace globally; "
                f"expected {neutral_trace:.17g}, got {total_trace:.17g}"
            )
        reference_eigenvalues = np.linalg.eigvalsh(reference)
        distance_to_projector_spectrum = np.minimum(
            np.abs(reference_eigenvalues), np.abs(reference_eigenvalues - 1.0)
        )
        if np.max(distance_to_projector_spectrum) > 1.0e-8:
            raise ValueError("Hartree reference must be an idempotent projector")
        reference = reference.copy()
        resolved_reference_provenance = str(reference_provenance)

    screened_params = ScreenedCoulombParams(
        epsilon_r=float(epsilon_r),
        d_sc_nm=float(d_sc_nm),
        finite_zero_limit=True,
    )
    primitive_area = real_space_cell_area_nm2_from_reciprocal(
        lattice.b_m1, lattice.b_m2
    )
    supercell_area = float(fold_map.supercell.area_ratio) * primitive_area
    normalization = 1.0 / (supercell_area * fold_map.reduced_nk)

    def weight_for_transfer(_label: tuple[int, int], qvec: complex) -> float:
        return normalization * float(screened_coulomb(qvec, screened_params))

    vertices = build_plane_wave_density_vertices(
        layout,
        fold_map,
        support,
        b_m1=lattice.b_m1,
        b_m2=lattice.b_m2,
        weight_for_transfer=weight_for_transfer,
        provenance=(
            f"{provenance}; screened Coulomb with finite Q=0; "
            f"weight=V(Q)/(A_supercell*N_K); support_sha256={support_sha256}; "
            f"support_provenance={support.provenance}; "
            f"authority={RECIPROCAL_HF_AUTHORITY}"
        ),
        max_entries=max_vertex_entries,
    )
    vertices = tuple(_frozen_vertex_copy(vertex) for vertex in vertices)
    neutral_occupied = int(fold_map.reduced_nk * layout.dimension // 2)
    total_occupied = neutral_occupied + int(
        primitive_filling * fold_map.primitive_nk
    )
    full_dimension = fold_map.reduced_nk * layout.dimension
    if total_occupied < 0 or total_occupied > full_dimension:
        raise ValueError("primitive filling lies outside the microscopic cutoff space")

    authority_paths = tuple(
        f"{RECIPROCAL_HF_AUTHORITY}/{relative}"
        for relative in RECIPROCAL_HF_AUTHORITY_SHA256
    )
    interaction_spec = DensityVertexInteractionSpec(
        include_hartree=True,
        include_fock=True,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="absolute",
        zero_mode_policy="include_supplied",
        closure_tolerance=1.0e-12,
        normalization_provenance=(
            "V(Q)/(A_supercell*N_K) = V(Q)/(A_primitive*N_k_primitive); "
            f"A_supercell={supercell_area:.17g} nm^2, N_K={fold_map.reduced_nk}; "
            f"support_sha256={support_sha256}; authority={RECIPROCAL_HF_AUTHORITY}"
        ),
        unresolved_physical_choices=(),
    )
    reference = _frozen_array_copy(reference, dtype=np.dtype(np.complex128))
    references = DensityVertexReferenceSpec(
        hartree=reference,
        fock=None,
        provenance=(
            f"Hartree P-P_ref with P_ref provenance={resolved_reference_provenance}; "
            "Fock uses absolute P; finite screened Q=0 retained in both terms; "
            f"support_sha256={support_sha256}; authority={RECIPROCAL_HF_AUTHORITY}"
        ),
    )
    source_binding = build_microscopic_hf_source_binding(
        h0_ket=h0,
        vertices=vertices,
        interaction_spec=interaction_spec,
        references=references,
        support=support,
        authority_capsule_root=authority_capsule_root,
        total_occupied=total_occupied,
        kbt_ev=kbt_ev,
        k_weights=k_weights,
    )
    return MicroscopicScreenedCoulombInputs(
        vertices=vertices,
        interaction_spec=interaction_spec,
        references=references,
        total_occupied=total_occupied,
        neutral_occupied=neutral_occupied,
        primitive_filling=primitive_filling,
        primitive_area_nm2=float(primitive_area),
        supercell_area_nm2=float(supercell_area),
        quadrature_normalization_nm_inv_sq=float(normalization),
        screened_params=screened_params,
        support_sha256=support_sha256,
        source_binding=source_binding,
        authority_paths=authority_paths,
    )


def build_batch_a_screened_coulomb_hf_inputs(
    *,
    h0_ket: Array | None,
    lattice: HTQGLattice,
    layout: MicroscopicBasisLayout,
    fold_map: BatchAFoldMap,
    support: BatchAPhysicalLabelSupport,
    authority_capsule_root: str | Path,
    max_vertex_entries: int,
    provenance: str,
    hartree_reference_ket: Array | None = None,
    reference_provenance: str | None = None,
    h0_artifact: MicroscopicH0Artifact | None = None,
    reference_fermi_degeneracy_tolerance: float | None = None,
    reference_replay_tolerance: float | None = None,
    kbt_ev: float | None = None,
    k_weights: Array | None = None,
    allow_unverified_test_h0: bool = False,
) -> MicroscopicScreenedCoulombInputs:
    """Strict shell-1/2/3, eps=5, d_sc=25 nm, nu=-3 Batch-A constructor.

    Shell 6 is intentionally outside this authority. Production construction
    requires an H0 artifact emitted by the direct-H0
    assembler. Raw arrays are accepted only through the explicit test-only
    escape hatch and do not establish physical derivation authority.
    """

    authorized_shell = _batch_a_authorized_shell(lattice, layout)
    if authorized_shell is None:
        raise ValueError(
            "Batch A requires canonical shell 1, 2, or 3 layout; shell 6 is not authorized"
        )
    if not np.array_equal(layout.g_indices, lattice.g_indices):
        raise ValueError("Batch-A layout must exactly replay lattice.g_indices")
    if (
        fold_map.primitive_mesh != (8, 4)
        or fold_map.reduced_mesh != (2, 4)
        or fold_map.supercell.area_ratio != 4
    ):
        raise ValueError("Batch-A strict authority requires canonical 8x4 -> 2x4 folding")
    if hartree_reference_ket is None:
        raise ValueError(
            "UNRESOLVED Batch-A supercell Hartree reference: supply an explicit "
            "globally neutral projector and provenance"
        )
    if type(allow_unverified_test_h0) is not bool:
        raise TypeError("allow_unverified_test_h0 must be an exact Boolean")
    if h0_artifact is not None:
        artifact = validate_microscopic_h0_artifact(h0_artifact)
        if artifact.support_sha256 != support.support_sha256:
            raise ValueError("H0 artifact support does not match Batch-A support")
        if h0_ket is not None and not np.array_equal(
            np.asarray(h0_ket, dtype=np.complex128), artifact.h0_ket
        ):
            raise ValueError("h0_ket does not exactly replay the H0 artifact")
        resolved_h0 = artifact.h0_ket
        if not allow_unverified_test_h0:
            if reference_fermi_degeneracy_tolerance is None:
                raise ValueError(
                    "Batch-A production requires explicit reference Fermi tolerance"
                )
            if reference_replay_tolerance is None:
                raise ValueError(
                    "Batch-A production requires explicit reference replay tolerance"
                )
            replay_tolerance = float(reference_replay_tolerance)
            if not math.isfinite(replay_tolerance) or replay_tolerance < 0.0:
                raise ValueError(
                    "reference_replay_tolerance must be finite and nonnegative"
                )
            replay = build_global_neutral_h0_reference(
                artifact.h0_ket,
                fermi_degeneracy_tolerance=float(
                    reference_fermi_degeneracy_tolerance
                ),
            ).projector_ket
            reference_array = np.asarray(
                hartree_reference_ket, dtype=np.complex128
            )
            if reference_array.shape != replay.shape:
                raise ValueError(
                    "Batch-A Hartree reference does not match H0 artifact ensemble"
                )
            replay_residual = float(np.max(np.abs(reference_array - replay)))
            if replay_residual > replay_tolerance:
                raise ValueError(
                    "Batch-A Hartree reference does not replay the global-neutral "
                    f"H0 projector: residual={replay_residual:.17g}"
                )
    else:
        if not allow_unverified_test_h0:
            raise ValueError(
                "Batch-A production requires a factory-built MicroscopicH0Artifact"
            )
        if h0_ket is None:
            raise ValueError("test-only raw Batch-A construction requires h0_ket")
        resolved_h0 = h0_ket
    result = build_screened_coulomb_hf_inputs(
        h0_ket=resolved_h0,
        lattice=lattice,
        layout=layout,
        fold_map=fold_map,
        support=support,
        authority_capsule_root=authority_capsule_root,
        epsilon_r=5.0,
        d_sc_nm=25.0,
        primitive_filling=-3,
        max_vertex_entries=max_vertex_entries,
        provenance=provenance,
        hartree_reference_ket=hartree_reference_ket,
        reference_provenance=reference_provenance,
        kbt_ev=kbt_ev,
        k_weights=k_weights,
    )
    if h0_artifact is not None and not allow_unverified_test_h0:
        authoritative_binding = _build_microscopic_hf_source_binding(
            h0_ket=resolved_h0,
            vertices=result.vertices,
            interaction_spec=result.interaction_spec,
            references=result.references,
            support=support,
            authority_capsule_root=authority_capsule_root,
            total_occupied=result.total_occupied,
            derivation_authority="batch_a_direct_h0_artifact",
            h0_artifact=h0_artifact,
            kbt_ev=kbt_ev,
            k_weights=k_weights,
        )
        result = replace(result, source_binding=authoritative_binding)
    expected_neutral = fold_map.reduced_nk * layout.dimension // 2
    expected_total = expected_neutral - 3 * fold_map.primitive_nk
    if (
        result.neutral_occupied != expected_neutral
        or result.total_occupied != expected_total
    ):
        raise RuntimeError(
            f"Batch-A shell-{authorized_shell} neutral/nu=-3 occupation count drift"
        )
    return result


def build_microscopic_hf_problem(
    *,
    vertices: Sequence[SparseDensityVertex],
    interaction_spec: DensityVertexInteractionSpec,
    references: DensityVertexReferenceSpec,
    nk: int,
    dimension: int,
    total_occupied: int,
    fermi_degeneracy_tolerance: float,
    mixing: float,
    support_sha256: str | None = None,
    source_binding: MicroscopicHFSourceBinding | None = None,
    kbt_ev: float | None = None,
    k_weights: Array | None = None,
    final_gate_tolerances: MicroscopicHFFinalGateTolerances | None = None,
    finite_t_convergence: MicroscopicHFFiniteTConvergence | None = None,
    interaction_workers: int = 1,
    eigensolver_workers: int = 1,
    fock_tile_rows: int = 256,
    fock_output_row_stripes: int = 1,
    allow_unbound_test_problem: bool = False,
    allow_identity_only_test_binding: bool = False,
) -> MicroscopicHartreeFockProblem:
    """Bind explicit HTQG microscopic inputs to the generic HF engine.

    ``init_mode='h0_global'`` builds the globally occupied H0 projector.
    ``init_mode='preserve'`` validates and retains the caller's state density.
    Constant ``mixing`` is implemented through the engine's step parameterizer;
    no system-specific SCF loop is introduced. ``kbt_ev=None`` preserves the
    legacy zero-temperature projector path. Finite temperature requires equal,
    positive explicit k weights because the current HTQG interaction vertices
    embed a uniform k quadrature.
    """

    if type(nk) is not int or nk <= 0:
        raise TypeError("nk must be an exact positive integer")
    if type(dimension) is not int or dimension <= 0:
        raise TypeError("dimension must be an exact positive integer")
    if (
        type(total_occupied) is not int
        or total_occupied < 0
        or total_occupied > nk * dimension
    ):
        raise TypeError("total_occupied must be an exact in-range integer")
    if type(allow_unbound_test_problem) is not bool:
        raise TypeError("allow_unbound_test_problem must be an exact boolean")
    if type(allow_identity_only_test_binding) is not bool:
        raise TypeError(
            "allow_identity_only_test_binding must be an exact boolean"
        )
    (
        occupation_ensemble,
        resolved_kbt,
        weights,
        k_weights_sha256,
    ) = _resolve_occupation_ensemble(
        nk=nk,
        kbt_ev=kbt_ev,
        k_weights=k_weights,
    )
    finite_temperature = occupation_ensemble == "finite_temperature_global_fermi"
    temperature = 0.0 if resolved_kbt is None else resolved_kbt
    interaction_spec = _frozen_interaction_spec_copy(interaction_spec)
    interaction_spec.validate(require_resolved=True)
    vertices = tuple(_frozen_vertex_copy(vertex) for vertex in vertices)
    references = _frozen_reference_copy(references)
    if not vertices:
        raise ValueError("vertices must be nonempty")
    _validate_reference_arrays(
        references,
        interaction_spec,
        ensemble_shape=(nk, dimension, dimension),
    )
    binding_texts = (
        *(str(vertex.provenance) for vertex in vertices),
        str(interaction_spec.normalization_provenance),
        str(references.provenance),
    )
    if source_binding is None:
        if not allow_unbound_test_problem:
            raise ValueError(
                "microscopic HF problems require an explicit typed source_binding; "
                "only algebraic tests may set allow_unbound_test_problem=True"
            )
        if support_sha256 is not None or any(
            "support_sha256=" in text for text in binding_texts
        ):
            raise ValueError(
                "support-derived HF inputs require an explicit typed source_binding"
            )
        resolved_support_sha256 = None
    else:
        source_binding = _require_microscopic_hf_source_binding(source_binding)
        resolved_support_sha256 = source_binding.support_sha256
        if support_sha256 is not None and (
            _validate_support_sha256(support_sha256) != resolved_support_sha256
        ):
            raise ValueError("support_sha256 does not match typed source_binding")
        if source_binding.vertices_sha256 != _sha256_vertices(vertices):
            raise ValueError("source binding vertex digest does not match HF inputs")
        if source_binding.interaction_sha256 != _sha256_interaction_spec(
            interaction_spec
        ):
            raise ValueError("source binding interaction digest does not match HF inputs")
        if (
            source_binding.nk != nk
            or source_binding.dimension != dimension
        ):
            raise ValueError("source binding ensemble shape does not match HF inputs")
        if source_binding.references_sha256 != _sha256_references(
            references, ensemble_shape=(nk, dimension, dimension)
        ):
            raise ValueError("source binding reference digest does not match HF inputs")
        if source_binding.total_occupied != total_occupied:
            raise ValueError(
                "source binding total_occupied does not match HF problem"
            )
        if source_binding.occupation_ensemble != occupation_ensemble:
            raise ValueError("source binding occupation ensemble does not match HF problem")
        if source_binding.kbt_ev != resolved_kbt:
            raise ValueError("source binding kbt_ev does not match HF problem")
        if source_binding.k_weights_sha256 != k_weights_sha256:
            raise ValueError("source binding k-weight digest does not match HF problem")
        if source_binding.authority_sha256 != _reciprocal_authority_sha256():
            raise ValueError("source binding authority digest does not match pinned authority")
        if (
            source_binding.derivation_authority == "identity_only_unverified"
            and not allow_identity_only_test_binding
        ):
            raise ValueError(
                "production microscopic HF requires source-bound H0 derivation; "
                "identity-only bindings are test-only"
            )
    if type(total_occupied) is not int:
        raise TypeError("total_occupied must be an exact integer")
    fermi_tolerance = float(fermi_degeneracy_tolerance)
    if not math.isfinite(fermi_tolerance) or fermi_tolerance < 0.0:
        raise ValueError("fermi_degeneracy_tolerance must be finite and nonnegative")
    mix = float(mixing)
    if not math.isfinite(mix) or not (0.0 < mix <= 1.0):
        raise ValueError("mixing must lie in (0,1]")
    if type(interaction_workers) is not int or interaction_workers <= 0:
        raise TypeError("interaction_workers must be an exact positive integer")
    if type(eigensolver_workers) is not int or eigensolver_workers <= 0:
        raise TypeError("eigensolver_workers must be an exact positive integer")
    if type(fock_tile_rows) is not int or fock_tile_rows <= 0:
        raise TypeError("fock_tile_rows must be an exact positive integer")
    if (
        type(fock_output_row_stripes) is not int
        or fock_output_row_stripes <= 0
        or fock_output_row_stripes > dimension
    ):
        raise TypeError(
            "fock_output_row_stripes must be an exact positive integer "
            "not exceeding dimension"
        )

    if not finite_temperature:
        if final_gate_tolerances is not None or finite_t_convergence is not None:
            raise ValueError(
                "finite-temperature options require explicit finite kbt_ev"
            )
        weight_scale = 1.0
        weighted_target = float(total_occupied)
    else:
        if final_gate_tolerances is None:
            raise ValueError(
                "finite-temperature HTQG requires explicit final_gate_tolerances"
            )
        if finite_t_convergence is None:
            raise ValueError(
                "finite-temperature HTQG requires explicit finite_t_convergence"
            )
        weight_scale = float(weights[0])
        weighted_target = weight_scale * float(total_occupied)
        weighted_capacity = float(dimension * np.sum(weights))
        occupation_particle_tolerance = min(
            1.0e-12,
            final_gate_tolerances.particle_number_abs / max(1.0, weighted_capacity),
        )

    problem_references = references
    prepared_vertices = prepare_density_vertex_family(
        vertices,
        nk=nk,
        dimension=dimension,
        closure_tolerance=interaction_spec.closure_tolerance,
        zero_mode_policy=interaction_spec.zero_mode_policy,
        fock_output_row_stripes=fock_output_row_stripes,
    )
    free_energy_history: list[float] = []
    convergence_streak = [0]
    cached_h0_source: list[Array | None] = [None]
    cached_h0_ket: list[Array | None] = [None]

    def bind_h0_cache(stored_h0: Array) -> Array:
        h0_ket = repository_hamiltonian_to_ket_blocks(stored_h0)
        _require_ket_hermitian(h0_ket, name="state.h0")
        h0_ket.setflags(write=False)
        cached_h0_source[0] = stored_h0
        cached_h0_ket[0] = h0_ket
        return h0_ket

    def resolve_h0_ket(stored_h0: Array) -> Array:
        if stored_h0 is cached_h0_source[0] and cached_h0_ket[0] is not None:
            return cached_h0_ket[0]
        h0_ket = repository_hamiltonian_to_ket_blocks(stored_h0)
        _require_ket_hermitian(h0_ket, name="stored_h0")
        return h0_ket

    def initialize(
        state: HartreeFockStateProtocol, *, init_mode: str, seed: int
    ) -> None:
        del seed
        free_energy_history.clear()
        convergence_streak[0] = 0
        for key in tuple(state.diagnostics):
            if key.startswith("finite_t_"):
                state.diagnostics.pop(key)
        h0_ket = bind_h0_cache(state.h0)
        if source_binding is not None:
            if type(state) is not MicroscopicHFState:
                raise TypeError(
                    "source-bound microscopic HF requires an exact MicroscopicHFState"
                )
            if getattr(state, "source_binding", None) != source_binding:
                raise ValueError("HF state is not bound to the problem source lineage")
            if _sha256_array(h0_ket, dtype=np.dtype("<c16")) != (
                source_binding.h0_sha256
            ):
                raise ValueError("HF state H0 digest does not match problem binding")
        if init_mode == "h0_global":
            if finite_temperature:
                occupied = global_fermi_occupations(
                    h0_ket,
                    target_particle_number=weighted_target,
                    kbt_ev=temperature,
                    k_weights=weights,
                    particle_tolerance=occupation_particle_tolerance,
                    eigensolver_workers=eigensolver_workers,
                )
                initial_density_ket = occupied.density_ket
            else:
                occupied = global_canonical_occupations(
                    h0_ket,
                    total_occupied=total_occupied,
                    fermi_degeneracy_tolerance=fermi_tolerance,
                    eigensolver_workers=eigensolver_workers,
                )
                initial_density_ket = occupied.projector_ket
            state.density[:, :, :] = ket_blocks_to_repository_stored(
                initial_density_ket
            )
            state.energies[:, :] = occupied.energies
            state.mu = occupied.chemical_potential
        elif init_mode == "preserve":
            density_ket = repository_stored_to_ket_blocks(state.density)
            _require_ket_hermitian(density_ket, name="state.density")
            if finite_temperature:
                density_diagnostics = fermionic_density_diagnostics(
                    density_ket,
                    k_weights=weights,
                    spectrum_tolerance=final_gate_tolerances.density_spectrum_abs,
                    eigensolver_workers=eigensolver_workers,
                )
                particle_number = density_diagnostics.particle_number
                particle_tolerance = final_gate_tolerances.particle_number_abs
                target_for_check = weighted_target
            else:
                particle_number = float(
                    np.trace(density_ket, axis1=1, axis2=2).real.sum()
                )
                particle_tolerance = 1.0e-10
                target_for_check = float(total_occupied)
            if abs(particle_number - target_for_check) > particle_tolerance:
                raise ValueError("preserved density does not match total_occupied")
        else:
            raise ValueError("init_mode must be 'h0_global' or 'preserve'")
        state.hamiltonian[:, :, :] = state.h0

    def interaction_builder(stored_density: Array) -> Array:
        projector = repository_stored_to_ket_blocks(stored_density)
        _require_ket_hermitian(projector, name="density")
        action = hartree_fock_action(
            projector,
            prepared_vertices,
            interaction_spec,
            problem_references,
            target_workers=interaction_workers,
            fock_tile_rows=fock_tile_rows,
        )
        _require_ket_hermitian(action.total, name="interaction action")
        return ket_hamiltonian_to_repository_blocks(action.total)

    def interaction_energy_builder(
        stored_density: Array, stored_h0: Array
    ) -> InteractionEnergyResult:
        projector = repository_stored_to_ket_blocks(stored_density)
        _require_ket_hermitian(projector, name="density")
        evaluation = hartree_fock_action_and_energy(
            projector,
            resolve_h0_ket(stored_h0),
            prepared_vertices,
            interaction_spec,
            problem_references,
            target_workers=interaction_workers,
            fock_tile_rows=fock_tile_rows,
        )
        _require_ket_hermitian(evaluation.action.total, name="interaction action")
        return InteractionEnergyResult(
            interaction_h=ket_hamiltonian_to_repository_blocks(
                evaluation.action.total
            ),
            energy=float(weight_scale * evaluation.energy),
        )

    def density_builder(stored_hamiltonian: Array) -> DensityUpdateResult:
        hamiltonian = repository_hamiltonian_to_ket_blocks(stored_hamiltonian)
        if finite_temperature:
            occupied = global_fermi_occupations(
                hamiltonian,
                target_particle_number=weighted_target,
                kbt_ev=temperature,
                k_weights=weights,
                particle_tolerance=occupation_particle_tolerance,
                eigensolver_workers=eigensolver_workers,
            )
            density_ket = occupied.density_ket
            observables = {
                "particle_number": occupied.particle_number,
                "particle_residual": occupied.particle_residual,
                "entropy_dimensionless": occupied.entropy_dimensionless,
                "kbt_ev": occupied.kbt_ev,
            }
        else:
            occupied = global_canonical_occupations(
                hamiltonian,
                total_occupied=total_occupied,
                fermi_degeneracy_tolerance=fermi_tolerance,
                eigensolver_workers=eigensolver_workers,
            )
            density_ket = occupied.projector_ket
            observables = {"fermi_gap": occupied.fermi_gap}
        return DensityUpdateResult(
            density=ket_blocks_to_repository_stored(density_ket),
            energies=occupied.energies,
            mu=occupied.chemical_potential,
            observables=observables,
        )

    def internal_energy(stored_h0: Array, stored_density: Array) -> float:
        unscaled = hartree_fock_energy(
            repository_stored_to_ket_blocks(stored_density),
            resolve_h0_ket(stored_h0),
            prepared_vertices,
            interaction_spec,
            problem_references,
            target_workers=interaction_workers,
            fock_tile_rows=fock_tile_rows,
        )
        return float(weight_scale * unscaled)

    def energy_functional(
        _stored_interaction: Array, stored_h0: Array, stored_density: Array
    ) -> float:
        return internal_energy(stored_h0, stored_density)

    def iteration_convergence_gate(
        state: HartreeFockStateProtocol,
        step: HartreeFockStepResult,
        density_converged: bool,
    ) -> bool:
        if not finite_temperature:
            return True
        diagnostics = fermionic_density_diagnostics(
            repository_stored_to_ket_blocks(step.previous_density),
            k_weights=weights,
            spectrum_tolerance=final_gate_tolerances.density_spectrum_abs,
            eigensolver_workers=eigensolver_workers,
        )
        free_energy = helmholtz_free_energy_ev(
            step.energy,
            kbt_ev=temperature,
            entropy_dimensionless=diagnostics.entropy_dimensionless,
        )
        if free_energy_history:
            delta_free_energy = float(free_energy - free_energy_history[-1])
            state.diagnostics["finite_t_free_energy_delta_ev"] = delta_free_energy
            qualified = (
                density_converged
                and abs(delta_free_energy)
                <= finite_t_convergence.helmholtz_change_abs_ev
            )
            convergence_streak[0] = (
                convergence_streak[0] + 1 if qualified else 0
            )
        else:
            convergence_streak[0] = 0
        free_energy_history.append(free_energy)
        state.diagnostics["finite_t_particle_number"] = diagnostics.particle_number
        state.diagnostics["finite_t_particle_residual_abs"] = abs(
            diagnostics.particle_number - weighted_target
        )
        state.diagnostics["finite_t_entropy_dimensionless"] = (
            diagnostics.entropy_dimensionless
        )
        state.diagnostics["finite_t_internal_energy_ev"] = step.energy
        state.diagnostics["finite_t_helmholtz_objective_ev"] = free_energy
        state.diagnostics["finite_t_free_energy_tolerance_ev"] = (
            finite_t_convergence.helmholtz_change_abs_ev
        )
        state.diagnostics["finite_t_convergence_streak"] = float(
            convergence_streak[0]
        )
        state.diagnostics["finite_t_required_consecutive_iterations"] = float(
            finite_t_convergence.required_consecutive_iterations
        )
        return (
            convergence_streak[0]
            >= finite_t_convergence.required_consecutive_iterations
        )

    def final_acceptance(
        state: HartreeFockStateProtocol,
        final_density_update: DensityUpdateResult,
        _final_norm: float,
    ) -> FinalAcceptanceResult:
        if not finite_temperature:
            return FinalAcceptanceResult(accepted=True)
        for key in (
            "finite_t_commutator_rms_ev",
            "finite_t_fermi_map_rms",
            "finite_t_free_energy_delta_ev",
            "finite_t_entropy_dimensionless",
            "finite_t_internal_energy_ev",
            "finite_t_helmholtz_objective_ev",
        ):
            state.diagnostics.pop(key, None)
        density = repository_stored_to_ket_blocks(state.density)
        hamiltonian = repository_hamiltonian_to_ket_blocks(state.hamiltonian)
        fermi_density = repository_stored_to_ket_blocks(
            final_density_update.density
        )
        density_scale = max(1.0, float(np.max(np.abs(density))))
        hermiticity_residual = float(
            np.max(np.abs(density - density.conj().transpose(0, 2, 1)))
        )
        hermitian_density = 0.5 * (density + density.conj().transpose(0, 2, 1))
        density_eigenvalues = np.linalg.eigvalsh(hermitian_density)
        eigenvalue_min = float(np.min(density_eigenvalues))
        eigenvalue_max = float(np.max(density_eigenvalues))
        particle_number = float(
            np.einsum(
                "k,k->",
                weights,
                np.trace(hermitian_density, axis1=1, axis2=2).real,
                optimize=True,
            )
        )
        particle_residual = abs(particle_number - weighted_target)
        common_diagnostics = {
            "finite_t_particle_number": particle_number,
            "finite_t_particle_residual_abs": particle_residual,
            "finite_t_density_eigenvalue_min": eigenvalue_min,
            "finite_t_density_eigenvalue_max": eigenvalue_max,
            "finite_t_density_hermiticity_residual": hermiticity_residual,
            "finite_t_kbt_ev": temperature,
            "finite_t_k_weight_sum": float(np.sum(weights)),
            "finite_t_energy_weight_scale": weight_scale,
            "finite_t_target_particle_number": weighted_target,
            "finite_t_free_energy_tolerance_ev": (
                finite_t_convergence.helmholtz_change_abs_ev
            ),
            "finite_t_convergence_streak": float(convergence_streak[0]),
            "finite_t_required_consecutive_iterations": float(
                finite_t_convergence.required_consecutive_iterations
            ),
        }
        hermiticity_limit = 64.0 * np.finfo(float).eps * density_scale
        spectrum_ok = (
            hermiticity_residual <= hermiticity_limit
            and eigenvalue_min >= -final_gate_tolerances.density_spectrum_abs
            and eigenvalue_max <= 1.0 + final_gate_tolerances.density_spectrum_abs
        )
        if not spectrum_ok:
            return FinalAcceptanceResult(
                accepted=False,
                reason="final_density_spectrum",
                diagnostics=common_diagnostics,
            )

        thermodynamics = fermionic_density_diagnostics(
            density,
            k_weights=weights,
            spectrum_tolerance=final_gate_tolerances.density_spectrum_abs,
            eigensolver_workers=eigensolver_workers,
        )
        if particle_residual > final_gate_tolerances.particle_number_abs:
            return FinalAcceptanceResult(
                accepted=False,
                reason="final_particle_number",
                diagnostics=common_diagnostics,
            )

        commutator = hamiltonian @ density - density @ hamiltonian
        weight_sum = float(np.sum(weights))
        commutator_rms = float(
            np.sqrt(
                np.einsum(
                    "k,kij,kij->",
                    weights,
                    commutator.conj(),
                    commutator,
                    optimize=True,
                ).real
                / (dimension * weight_sum)
            )
        )
        fermi_delta = density - fermi_density
        fermi_map_rms = float(
            np.sqrt(
                np.einsum(
                    "k,kij,kij->",
                    weights,
                    fermi_delta.conj(),
                    fermi_delta,
                    optimize=True,
                ).real
                / (dimension * weight_sum)
            )
        )
        final_internal_energy = (
            float(state.diagnostics["hf_energy"])
            if "hf_energy" in state.diagnostics
            else internal_energy(state.h0, state.density)
        )
        final_free_energy = helmholtz_free_energy_ev(
            final_internal_energy,
            kbt_ev=temperature,
            entropy_dimensionless=thermodynamics.entropy_dimensionless,
        )
        common_diagnostics.update(
            {
                "finite_t_commutator_rms_ev": commutator_rms,
                "finite_t_fermi_map_rms": fermi_map_rms,
                "finite_t_entropy_dimensionless": thermodynamics.entropy_dimensionless,
                "finite_t_internal_energy_ev": final_internal_energy,
                "finite_t_helmholtz_objective_ev": final_free_energy,
            }
        )
        final_helmholtz_change_ok = False
        if free_energy_history:
            final_delta_free_energy = float(
                final_free_energy - free_energy_history[-1]
            )
            common_diagnostics["finite_t_free_energy_delta_ev"] = (
                final_delta_free_energy
            )
            final_helmholtz_change_ok = (
                abs(final_delta_free_energy)
                <= finite_t_convergence.helmholtz_change_abs_ev
                and convergence_streak[0]
                >= finite_t_convergence.required_consecutive_iterations
            )
        if commutator_rms > final_gate_tolerances.commutator_rms_ev:
            reason = "final_commutator"
        elif fermi_map_rms > final_gate_tolerances.fermi_map_rms:
            reason = "final_fermi_map"
        elif not final_helmholtz_change_ok:
            reason = "final_helmholtz_change"
        else:
            reason = "accepted"
        return FinalAcceptanceResult(
            accepted=reason == "accepted",
            reason=reason,
            diagnostics=common_diagnostics,
        )

    return _new_microscopic_hf_problem(
        initializer=initialize,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            interaction_energy_builder=interaction_energy_builder,
            oda_parameterizer=lambda _state, _delta: mix,
            iteration_convergence_gate=(
                iteration_convergence_gate if finite_temperature else None
            ),
            final_acceptance=final_acceptance if finite_temperature else None,
            convergence_rule="raw",
        ),
        support_sha256=resolved_support_sha256,
        source_binding=source_binding,
        nk=nk,
        dimension=dimension,
    )


__all__ = [
    "MicroscopicHFFinalGateTolerances",
    "MicroscopicHFFiniteTConvergence",
    "MicroscopicHFSourceBinding",
    "MicroscopicHFState",
    "MicroscopicHartreeFockProblem",
    "MicroscopicScreenedCoulombInputs",
    "RECIPROCAL_HF_AUTHORITY",
    "RECIPROCAL_HF_AUTHORITY_SHA256",
    "build_batch_a_screened_coulomb_hf_inputs",
    "build_global_neutral_h0_reference",
    "build_half_filled_h0_reference",
    "build_microscopic_hf_problem",
    "build_microscopic_hf_source_binding",
    "build_screened_coulomb_hf_inputs",
    "verify_reciprocal_hf_authority",
]
