"""Factory-only frozen-source pocket-refinement replay for Vituri-2024 HF.

This system-local gate does not solve a refined SCF problem.  It binds the
factory array receipt plus an exact uninterrupted-SCF approval/receipt pair,
reconstructs their source/spec/state identities, loads a detached refinement
archive before calling a trusted live evaluator, registers one same-domain
nested rectangular mesh, and independently recomputes finite-domain digital
pocket topology and a discrete threshold-topology ("Lifshitz") margin.

The evidence model deliberately proves only parity under a trusted evaluator:
distinct authority objects, immutable copies, manifests, and call ordering do
not exclude a same-code provider that reads hidden archive state.  Real-artifact,
continuum, convergence, ground-state, paper, and TDHF-readiness claims remain
false even after a successful synthetic replay.
"""
from __future__ import annotations

import ast
from dataclasses import InitVar, asdict, dataclass, field, fields as dataclass_fields
import hashlib
import inspect
import json
import marshal
import math
from numbers import Integral, Real
from pathlib import Path
import stat
import subprocess
import textwrap
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .vituri2024_hf_preflight import (
    VITURI2024_BASE_PROVIDER_METADATA_FIELDS,
    Vituri2024HalfMetalHFProviderBinding,
    Vituri2024ValleyPocketEvidenceReceipt,
)
from .vituri2024_hf_replay import (
    INTERNAL_FLAVOR_ORDER,
    Vituri2024HalfMetalHFReplayReceipt,
    canonical_array_sha256,
)
from .vituri2024_hf_scf_replay import (
    Vituri2024SCFReplayApproval,
    Vituri2024SCFReplayReceipt,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE = (
    "vituri2024_frozen_selected_source_pocket_refinement_replay_v1"
)
VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT = (
    "ae6fadf3b7e4a70e5390d73f724b9484bcbc7abd"
)
POCKET_REFINEMENT_EVIDENCE_MODEL = (
    "trusted_live_selected_source_evaluator_distinct_refinement_archive_object"
)
POCKET_REFINEMENT_MESH_SCHEMA_LABEL = (
    "vituri2024_nested_same_domain_rectangular_no_wrap_mesh_v1"
)
POCKET_REFINEMENT_REQUEST_SCHEMA_LABEL = (
    "vituri2024_frozen_selected_source_refinement_request_v1"
)
POCKET_REFINEMENT_EVALUATION_SCHEMA_LABEL = (
    "vituri2024_frozen_selected_source_refinement_evaluation_v1"
)
POCKET_REFINEMENT_ARCHIVE_SCHEMA_LABEL = (
    "vituri2024_immutable_pocket_refinement_archive_v1"
)
POCKET_REFINEMENT_ARCHIVE_GENERATION_PHASE = (
    "detached_before_live_frozen_selected_source_evaluation"
)
POCKET_REFINEMENT_EVIDENCE_SCHEMA_LABEL = (
    "vituri2024_refinement_topology_lifshitz_evidence_v1"
)
POCKET_REFINEMENT_TOPOLOGY_CONVENTION = (
    "foreground_four_neighbor_complement_eight_neighbor_finite_domain_no_wrap"
)
POCKET_REFINEMENT_LIFSHITZ_CONVENTION = (
    "maximal_consecutive_open_energy_threshold_intervals_with_accepted_topology"
)

POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV = 1.0e-12
POCKET_REPLAY_V1_RELATIVE_TOLERANCE = 1.0e-12
POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM = 1.0e-14
POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE = 1.0e-12
POCKET_REPLAY_V1_HERMITICITY_TOLERANCE_EV = 1.0e-12
POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV = 1.0e-12
POCKET_REPLAY_V1_FOCK_DECOMPOSITION_TOLERANCE_EV = 1.0e-12

_POCKET_SUCCESS_TOKEN = object()
_GIT_PROVENANCE_MODE = "git_ancestor_head_index_worktree_verified"
_SOURCE_EXPORT_PROVENANCE_MODE = "pinned_hash_verified_source_export"

_PREREQUISITE_SOURCE_EXPECTATIONS: MappingProxyType[str, tuple[str, str]] = (
    MappingProxyType(
        {
            "src/mean_field/systems/abc_trilayer/vituri2024_hf_preflight.py": (
                "9d59c3cbc27ec102e8c0cd8703fefa27dac7f856c5380394657f2eb73dc3c9a4",
                "019039629a4f5b7b5a06e358d842f052dc964b82b9241badf434a77b55eaa30d",
            ),
            "src/mean_field/systems/abc_trilayer/vituri2024_hf_replay.py": (
                "1e963f2e1d439d1c99eb610473a8bcb01c70938e93189486134adb4b7ba9113b",
                "b73c29949fac8284d24db70ebd04d7c4dd72924cdea8c591e120f525494dd3e1",
            ),
            "src/mean_field/systems/abc_trilayer/vituri2024_hf_scf_replay.py": (
                "75138177ef0ceda0840f733052533badda3ec89f9feb76a1b129b89c83cf200a",
                "7241aa786a4c53be2a0ed9fa3acaf2b75453ac414d0db3e9786a15b217333cd2",
            ),
        }
    )
)


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase 40-character commit")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a strict integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _immutable_array(
    value: object,
    *,
    label: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    array = value
    if array.dtype != dtype:
        raise TypeError(f"{label} dtype must be exactly {dtype.name}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    result = np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)
    result.flags.writeable = False
    return result


def _array_manifest(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": canonical_array_sha256(array),
    }


def _max_abs(array: np.ndarray) -> float:
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _scale_bound(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    absolute: float = POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV,
    relative: float = POCKET_REPLAY_V1_RELATIVE_TOLERANCE,
) -> float:
    scale = max(_max_abs(np.asarray(actual)), _max_abs(np.asarray(expected)), 1.0)
    return absolute + relative * scale


def _require_close(
    actual: np.ndarray | float,
    expected: np.ndarray | float,
    label: str,
    *,
    absolute: float = POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV,
    relative: float = POCKET_REPLAY_V1_RELATIVE_TOLERANCE,
) -> float:
    residual = _max_abs(np.asarray(actual) - np.asarray(expected))
    if residual > _scale_bound(
        actual, expected, absolute=absolute, relative=relative
    ):
        raise ValueError(f"{label} residual exceeds locked v1 tolerance")
    return residual


def canonical_half_metal_hf_replay_receipt_fingerprint(
    receipt: Vituri2024HalfMetalHFReplayReceipt,
) -> str:
    """Canonical complete fingerprint of one factory array-replay receipt."""

    if type(receipt) is not Vituri2024HalfMetalHFReplayReceipt:
        raise TypeError("array prerequisite requires the exact replay receipt type")
    return _fingerprint(asdict(receipt))


def canonical_scf_replay_receipt_fingerprint(
    receipt: Vituri2024SCFReplayReceipt,
) -> str:
    """Canonical complete fingerprint of one factory SCF-replay receipt."""

    if type(receipt) is not Vituri2024SCFReplayReceipt:
        raise TypeError("SCF prerequisite requires the exact replay receipt type")
    return _fingerprint(asdict(receipt))


def pocket_refinement_replay_module_ast_manifest_sha256(
    source: str | None = None,
) -> str:
    """Hash the canonical full-module AST, excluding comments and locations."""

    module_source = Path(__file__).read_text(encoding="utf-8") if source is None else source
    if not isinstance(module_source, str):
        raise TypeError("pocket replay module source must be text")
    tree = ast.parse(module_source, filename="vituri2024_hf_pocket_replay.py")
    return hashlib.sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Vituri2024PocketCallableManifest:
    role: str
    module: str
    qualname: str
    signature: str
    source_sha256: str
    canonical_ast_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.role, "callable role"),
            (self.module, "callable module"),
            (self.qualname, "callable qualname"),
            (self.signature, "callable signature"),
        ):
            _text(value, label)
        for value, label in (
            (self.source_sha256, "callable source"),
            (self.canonical_ast_sha256, "callable AST"),
            (self.code_sha256, "callable code"),
        ):
            _sha256(value, label)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def vituri2024_pocket_callable_manifest(
    role: str, callable_object: object
) -> Vituri2024PocketCallableManifest:
    if not callable(callable_object):
        raise TypeError(f"{role} must be callable")
    function = getattr(callable_object, "__func__", callable_object)
    code = getattr(function, "__code__", None)
    if code is None:
        raise TypeError(f"{role} must expose Python code for detached approval")
    try:
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        signature = str(inspect.signature(callable_object))
    except (OSError, TypeError, SyntaxError, ValueError) as error:
        raise TypeError(f"{role} must expose inspectable source/signature") from error
    return Vituri2024PocketCallableManifest(
        role=role,
        module=_text(getattr(function, "__module__", None), f"{role} module"),
        qualname=_text(getattr(function, "__qualname__", None), f"{role} qualname"),
        signature=signature,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        canonical_ast_sha256=hashlib.sha256(
            ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
        ).hexdigest(),
        code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class Vituri2024PocketPrerequisiteSourceManifest:
    relative_path: str
    source_bytes_sha256: str
    canonical_ast_sha256: str

    def __post_init__(self) -> None:
        _text(self.relative_path, "prerequisite source path")
        _sha256(self.source_bytes_sha256, "prerequisite source bytes")
        _sha256(self.canonical_ast_sha256, "prerequisite source AST")


@dataclass(frozen=True, slots=True)
class Vituri2024PocketPrerequisiteProvenance:
    provenance_mode: Literal[
        "git_ancestor_head_index_worktree_verified",
        "pinned_hash_verified_source_export",
    ]
    baseline_commit: str
    repository_checks_available: bool
    repository_ancestry_verified: bool
    repository_head_sources_verified: bool
    repository_index_sources_verified: bool
    repository_worktree_sources_verified: bool
    source_manifests: tuple[Vituri2024PocketPrerequisiteSourceManifest, ...]

    def __post_init__(self) -> None:
        if self.provenance_mode not in (
            _GIT_PROVENANCE_MODE,
            _SOURCE_EXPORT_PROVENANCE_MODE,
        ):
            raise ValueError("unsupported pocket prerequisite provenance mode")
        if (
            _commit(self.baseline_commit, "pocket prerequisite baseline")
            != VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT
        ):
            raise ValueError("pocket prerequisite baseline commit changed")
        expected = self.provenance_mode == _GIT_PROVENANCE_MODE
        flags = (
            self.repository_checks_available,
            self.repository_ancestry_verified,
            self.repository_head_sources_verified,
            self.repository_index_sources_verified,
            self.repository_worktree_sources_verified,
        )
        if any(type(value) is not bool for value in flags) or any(
            value is not expected for value in flags
        ):
            raise ValueError("pocket prerequisite provenance flags contradict mode")
        manifests = tuple(self.source_manifests)
        if tuple(item.relative_path for item in manifests) != tuple(
            _PREREQUISITE_SOURCE_EXPECTATIONS
        ):
            raise ValueError("pocket prerequisite source inventory/order changed")
        object.__setattr__(self, "source_manifests", manifests)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(
            "pocket prerequisite git verification failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _source_manifest(relative_path: str, source: bytes) -> Vituri2024PocketPrerequisiteSourceManifest:
    try:
        tree = ast.parse(source.decode("utf-8"), filename=relative_path)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"invalid prerequisite Python source: {relative_path}") from error
    manifest = Vituri2024PocketPrerequisiteSourceManifest(
        relative_path=relative_path,
        source_bytes_sha256=hashlib.sha256(source).hexdigest(),
        canonical_ast_sha256=hashlib.sha256(
            ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
        ).hexdigest(),
    )
    if (
        manifest.source_bytes_sha256,
        manifest.canonical_ast_sha256,
    ) != _PREREQUISITE_SOURCE_EXPECTATIONS[relative_path]:
        raise ValueError(f"pocket prerequisite source drift: {relative_path}")
    return manifest


def verified_vituri2024_pocket_prerequisite_provenance() -> Vituri2024PocketPrerequisiteProvenance:
    """Verify the ae6 prerequisite sources without pinning the new module itself."""

    root = _repository_root()
    git_path = root / ".git"
    try:
        mode = git_path.lstat().st_mode
        repository_available = stat.S_ISDIR(mode) or stat.S_ISREG(mode)
    except FileNotFoundError:
        repository_available = False
    if repository_available:
        _git(
            root,
            "merge-base",
            "--is-ancestor",
            VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT,
            "HEAD",
        )
        manifests: list[Vituri2024PocketPrerequisiteSourceManifest] = []
        for path in _PREREQUISITE_SOURCE_EXPECTATIONS:
            baseline = _git(
                root,
                "show",
                f"{VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT}:{path}",
            )
            head = _git(root, "show", f"HEAD:{path}")
            index = _git(root, "show", f":{path}")
            worktree = (root / path).read_bytes()
            if not (baseline == head == index == worktree):
                raise ValueError(f"pocket prerequisite HEAD/index/worktree drift: {path}")
            manifests.append(_source_manifest(path, worktree))
        return Vituri2024PocketPrerequisiteProvenance(
            provenance_mode=_GIT_PROVENANCE_MODE,
            baseline_commit=VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT,
            repository_checks_available=True,
            repository_ancestry_verified=True,
            repository_head_sources_verified=True,
            repository_index_sources_verified=True,
            repository_worktree_sources_verified=True,
            source_manifests=tuple(manifests),
        )
    manifests = tuple(
        _source_manifest(path, (root / path).read_bytes())
        for path in _PREREQUISITE_SOURCE_EXPECTATIONS
    )
    return Vituri2024PocketPrerequisiteProvenance(
        provenance_mode=_SOURCE_EXPORT_PROVENANCE_MODE,
        baseline_commit=VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT,
        repository_checks_available=False,
        repository_ancestry_verified=False,
        repository_head_sources_verified=False,
        repository_index_sources_verified=False,
        repository_worktree_sources_verified=False,
        source_manifests=manifests,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024NestedNoWrapRefinementMesh:
    """Affine row-major nested refinement of one closed rectangular domain."""

    base_shape: tuple[int, int]
    subdivision_factors: tuple[int, int]
    base_mesh: FloatArray
    refined_mesh: FloatArray
    boundary_policy: Literal["finite_domain_no_wrap"] = "finite_domain_no_wrap"
    halo_policy: Literal["no_halo"] = "no_halo"
    reciprocal_carry_policy: Literal["no_reciprocal_carry"] = "no_reciprocal_carry"

    def __post_init__(self) -> None:
        if type(self.base_shape) is not tuple or len(self.base_shape) != 2:
            raise TypeError("base mesh shape must be a strict length-two tuple")
        n1, n2 = (
            _strict_int(self.base_shape[0], "base mesh axis-1 size"),
            _strict_int(self.base_shape[1], "base mesh axis-2 size"),
        )
        if n1 < 2 or n2 < 2:
            raise ValueError("base rectangular mesh axes must each contain at least two points")
        if type(self.subdivision_factors) is not tuple or len(self.subdivision_factors) != 2:
            raise TypeError("subdivision factors must be a strict length-two tuple")
        r1, r2 = (
            _strict_int(self.subdivision_factors[0], "axis-1 subdivision factor"),
            _strict_int(self.subdivision_factors[1], "axis-2 subdivision factor"),
        )
        if r1 < 1 or r2 < 1 or (r1, r2) == (1, 1):
            raise ValueError("subdivision factors must be positive and not both one")
        refined_shape = (r1 * (n1 - 1) + 1, r2 * (n2 - 1) + 1)
        base = _immutable_array(
            self.base_mesh,
            label="base mesh",
            dtype=np.dtype(np.float64),
            shape=(n1 * n2, 2),
        )
        refined = _immutable_array(
            self.refined_mesh,
            label="refined mesh",
            dtype=np.dtype(np.float64),
            shape=(refined_shape[0] * refined_shape[1], 2),
        )
        if (
            self.boundary_policy != "finite_domain_no_wrap"
            or self.halo_policy != "no_halo"
            or self.reciprocal_carry_policy != "no_reciprocal_carry"
        ):
            raise ValueError("refinement mesh must remain finite-domain/no-wrap/no-halo/no-carry")
        base_grid = base.reshape(n1, n2, 2)
        origin = base_grid[0, 0]
        axis1 = base_grid[1, 0] - origin
        axis2 = base_grid[0, 1] - origin
        determinant = float(axis1[0] * axis2[1] - axis1[1] * axis2[0])
        axis_scale = max(float(np.linalg.norm(axis1) * np.linalg.norm(axis2)), 1.0e-300)
        if abs(determinant) <= 256.0 * math.ulp(axis_scale):
            raise ValueError("rectangular mesh affine axes must be non-collinear")
        expected_base = np.asarray(
            [origin + i * axis1 + j * axis2 for i in range(n1) for j in range(n2)],
            dtype=np.float64,
        )
        expected_refined = np.asarray(
            [
                origin + i * axis1 / r1 + j * axis2 / r2
                for i in range(refined_shape[0])
                for j in range(refined_shape[1])
            ],
            dtype=np.float64,
        )
        for actual, expected, label in (
            (base, expected_base, "base row-major affine mesh"),
            (refined, expected_refined, "refined row-major affine mesh"),
        ):
            residual = _max_abs(actual - expected)
            bound = (
                POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM
                + POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE
                * max(_max_abs(actual), _max_abs(expected), 1.0)
            )
            if residual > bound:
                raise ValueError(f"{label} residual exceeds locked scale-aware tolerance")
        embedding = np.asarray(
            [
                (r1 * i) * refined_shape[1] + r2 * j
                for i in range(n1)
                for j in range(n2)
            ],
            dtype=np.int64,
        )
        if len(set(int(value) for value in embedding)) != n1 * n2 or np.any(
            (embedding < 0) | (embedding >= refined.shape[0])
        ):
            raise ValueError("base embedding must be unique and in range")
        embedded = refined[embedding]
        residual = _max_abs(embedded - base)
        bound = (
            POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM
            + POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE
            * max(_max_abs(embedded), _max_abs(base), 1.0)
        )
        if residual > bound:
            raise ValueError("embedded base coordinates do not close scale-aware")
        object.__setattr__(self, "base_shape", (n1, n2))
        object.__setattr__(self, "subdivision_factors", (r1, r2))
        object.__setattr__(self, "base_mesh", base)
        object.__setattr__(self, "refined_mesh", refined)

    @property
    def refined_shape(self) -> tuple[int, int]:
        n1, n2 = self.base_shape
        r1, r2 = self.subdivision_factors
        return (r1 * (n1 - 1) + 1, r2 * (n2 - 1) + 1)

    @property
    def base_embedding_indices(self) -> IntegerArray:
        n1, n2 = self.base_shape
        r1, r2 = self.subdivision_factors
        refined_columns = self.refined_shape[1]
        result = np.asarray(
            [
                (r1 * i) * refined_columns + r2 * j
                for i in range(n1)
                for j in range(n2)
            ],
            dtype=np.int64,
        )
        result.flags.writeable = False
        return result

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": POCKET_REFINEMENT_MESH_SCHEMA_LABEL,
                "base_shape": self.base_shape,
                "refined_shape": self.refined_shape,
                "subdivision_factors": self.subdivision_factors,
                "base_mesh": _array_manifest(self.base_mesh),
                "refined_mesh": _array_manifest(self.refined_mesh),
                "base_embedding_indices": _array_manifest(self.base_embedding_indices),
                "boundary_policy": self.boundary_policy,
                "halo_policy": self.halo_policy,
                "reciprocal_carry_policy": self.reciprocal_carry_policy,
            }
        )


POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema": POCKET_REFINEMENT_REQUEST_SCHEMA_LABEL,
        "source_identity": (
            "provider_fingerprint",
            "source_commit",
            "source_artifact_sha256",
            "spec_fingerprint",
            "source_state_sha256",
            "selected_branch_label",
            "selected_spin",
            "chemical_potential_ev",
        ),
        "base_hashes": (
            "mesh",
            "h0",
            "interaction_h",
            "fock",
            "energies",
            "occupations",
            "projector",
        ),
        "mesh_schema": POCKET_REFINEMENT_MESH_SCHEMA_LABEL,
        "archive_fields_forbidden": True,
    }
)
POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema": POCKET_REFINEMENT_EVALUATION_SCHEMA_LABEL,
        "arrays": ("h0", "interaction_h", "fock"),
        "shape": "4x4xNref",
        "dtype": "complex128",
        "derived_fields_forbidden": (
            "energies",
            "occupations",
            "topology",
            "lifshitz_margin",
        ),
    }
)
POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema": POCKET_REFINEMENT_ARCHIVE_SCHEMA_LABEL,
        "generation_phase": POCKET_REFINEMENT_ARCHIVE_GENERATION_PHASE,
        "mesh_schema": POCKET_REFINEMENT_MESH_SCHEMA_LABEL,
        "arrays": ("h0", "interaction_h", "fock"),
    }
)


@dataclass(frozen=True, slots=True)
class Vituri2024PocketBaseHashes:
    ordered_momentum_mesh_sha256: str
    h0_sha256: str
    interaction_h_sha256: str
    fock_sha256: str
    energies_sha256: str
    occupations_sha256: str
    projector_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _sha256(value, name.replace("_", " "))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementFieldHashes:
    h0_sha256: str
    interaction_h_sha256: str
    fock_sha256: str
    energies_sha256: str
    occupations_sha256: str
    projector_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _sha256(value, name.replace("_", " "))


@dataclass(frozen=True, slots=True)
class Vituri2024ArchivedPocketRefinementFields:
    h0: ComplexArray
    interaction_h: ComplexArray
    fock: ComplexArray

    def __post_init__(self) -> None:
        h0 = _immutable_array(
            self.h0, label="archived refined h0", dtype=np.dtype(np.complex128), ndim=3
        )
        if h0.shape[:2] != (4, 4) or h0.shape[2] < 1:
            raise ValueError("archived refined h0 shape must be (4,4,Nref)")
        shape = h0.shape
        interaction = _immutable_array(
            self.interaction_h,
            label="archived refined interaction_h",
            dtype=np.dtype(np.complex128),
            shape=shape,
        )
        fock = _immutable_array(
            self.fock,
            label="archived refined Fock",
            dtype=np.dtype(np.complex128),
            shape=shape,
        )
        object.__setattr__(self, "h0", h0)
        object.__setattr__(self, "interaction_h", interaction)
        object.__setattr__(self, "fock", fock)


@dataclass(frozen=True, slots=True)
class Vituri2024FrozenHFRefinementRequest:
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: Literal[-1, 1]
    chemical_potential_ev: float
    base_hashes: Vituri2024PocketBaseHashes
    mesh: Vituri2024NestedNoWrapRefinementMesh
    request_schema_fingerprint: str = POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT

    def __post_init__(self) -> None:
        for value, label in (
            (self.provider_fingerprint, "request provider"),
            (self.source_artifact_sha256, "request source artifact"),
            (self.spec_fingerprint, "request spec"),
            (self.source_state_sha256, "request source state"),
            (self.request_schema_fingerprint, "request schema"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "request source commit")
        _text(self.selected_branch_label, "request selected branch")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("request selected spin must be exactly -1 or +1")
        object.__setattr__(
            self, "chemical_potential_ev", _finite(self.chemical_potential_ev, "request mu")
        )
        if type(self.base_hashes) is not Vituri2024PocketBaseHashes:
            raise TypeError("request requires typed base hashes")
        if type(self.mesh) is not Vituri2024NestedNoWrapRefinementMesh:
            raise TypeError("request requires a typed nested refinement mesh")
        if self.request_schema_fingerprint != POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT:
            raise ValueError("request schema fingerprint changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": POCKET_REFINEMENT_REQUEST_SCHEMA_LABEL,
                "provider_fingerprint": self.provider_fingerprint,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
                "spec_fingerprint": self.spec_fingerprint,
                "source_state_sha256": self.source_state_sha256,
                "selected_branch_label": self.selected_branch_label,
                "selected_spin": self.selected_spin,
                "chemical_potential_ev": self.chemical_potential_ev,
                "base_hashes": asdict(self.base_hashes),
                "mesh_fingerprint": self.mesh.fingerprint,
                "request_schema_fingerprint": self.request_schema_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class Vituri2024FrozenHFRefinementEvaluation:
    pocket_refinement_provider_fingerprint: str
    evaluator_implementation_fingerprint: str
    evaluation_schema_fingerprint: str
    request_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    h0: ComplexArray
    interaction_h: ComplexArray
    fock: ComplexArray

    def __post_init__(self) -> None:
        for value, label in (
            (self.pocket_refinement_provider_fingerprint, "evaluation provider"),
            (self.evaluator_implementation_fingerprint, "evaluation implementation"),
            (self.evaluation_schema_fingerprint, "evaluation schema"),
            (self.request_fingerprint, "evaluation request"),
            (self.source_artifact_sha256, "evaluation source artifact"),
            (self.spec_fingerprint, "evaluation spec"),
            (self.source_state_sha256, "evaluation source state"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "evaluation source commit")
        _text(self.selected_branch_label, "evaluation selected branch")
        if self.evaluation_schema_fingerprint != POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT:
            raise ValueError("evaluation schema fingerprint changed")
        fields = Vituri2024ArchivedPocketRefinementFields(
            self.h0, self.interaction_h, self.fock
        )
        object.__setattr__(self, "h0", fields.h0)
        object.__setattr__(self, "interaction_h", fields.interaction_h)
        object.__setattr__(self, "fock", fields.fock)


@dataclass(frozen=True, slots=True)
class Vituri2024ImmutablePocketRefinementArchive:
    archive_authority_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: Literal[-1, 1]
    chemical_potential_ev: float
    archive_loader_implementation_fingerprint: str
    archive_schema_fingerprint: str
    generation_phase: Literal[
        "detached_before_live_frozen_selected_source_evaluation"
    ]
    mesh: Vituri2024NestedNoWrapRefinementMesh
    fields: Vituri2024ArchivedPocketRefinementFields

    def __post_init__(self) -> None:
        for value, label in (
            (self.archive_authority_fingerprint, "archive authority"),
            (self.source_artifact_sha256, "archive source artifact"),
            (self.spec_fingerprint, "archive spec"),
            (self.source_state_sha256, "archive source state"),
            (self.archive_loader_implementation_fingerprint, "archive loader"),
            (self.archive_schema_fingerprint, "archive schema"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "archive source commit")
        _text(self.selected_branch_label, "archive selected branch")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("archive selected spin must be exactly -1 or +1")
        object.__setattr__(
            self, "chemical_potential_ev", _finite(self.chemical_potential_ev, "archive mu")
        )
        if self.archive_schema_fingerprint != POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT:
            raise ValueError("archive schema fingerprint changed")
        if self.generation_phase != POCKET_REFINEMENT_ARCHIVE_GENERATION_PHASE:
            raise ValueError("archive must be generated in the detached pre-live phase")
        if type(self.mesh) is not Vituri2024NestedNoWrapRefinementMesh:
            raise TypeError("archive requires a typed nested refinement mesh")
        if type(self.fields) is not Vituri2024ArchivedPocketRefinementFields:
            raise TypeError("archive requires typed refinement fields")
        if self.fields.h0.shape[2] != self.mesh.refined_mesh.shape[0]:
            raise ValueError("archive field cardinality does not equal refined mesh cardinality")


@runtime_checkable
class Vituri2024PocketRefinementArchiveAuthorityProtocol(Protocol):
    archive_authority_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    archive_loader_implementation_fingerprint: str
    archive_schema_fingerprint: str

    def load_immutable_pocket_refinement_archive(
        self, source_artifact_sha256: str
    ) -> Vituri2024ImmutablePocketRefinementArchive: ...


POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS = VITURI2024_BASE_PROVIDER_METADATA_FIELDS + (
    "replay_loader_implementation_fingerprint",
    "replay_payload_schema_fingerprint",
    "pocket_refinement_provider_fingerprint",
    "refinement_evaluator_implementation_fingerprint",
    "refinement_request_schema_fingerprint",
    "refinement_evaluation_schema_fingerprint",
)
POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS: tuple[str, ...] = (
    "archive_authority_fingerprint",
    "source_commit",
    "source_artifact_sha256",
    "spec_fingerprint",
    "source_state_sha256",
    "archive_loader_implementation_fingerprint",
    "archive_schema_fingerprint",
)


@runtime_checkable
class Vituri2024PocketRefinementProviderProtocol(Protocol):
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    replay_loader_implementation_fingerprint: str
    replay_payload_schema_fingerprint: str
    pocket_refinement_provider_fingerprint: str
    refinement_evaluator_implementation_fingerprint: str
    refinement_request_schema_fingerprint: str
    refinement_evaluation_schema_fingerprint: str

    def evaluate_frozen_selected_hf_source(
        self, request: Vituri2024FrozenHFRefinementRequest
    ) -> Vituri2024FrozenHFRefinementEvaluation: ...


def pocket_refinement_provider_fingerprint(
    *,
    base_provider_fingerprint: str,
    evaluator_implementation_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "base_provider_fingerprint": _sha256(
                base_provider_fingerprint, "base provider"
            ),
            "evaluator_implementation_fingerprint": _sha256(
                evaluator_implementation_fingerprint, "refinement evaluator"
            ),
            "request_schema_fingerprint": POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT,
            "evaluation_schema_fingerprint": POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT,
        }
    )


def pocket_refinement_archive_authority_fingerprint(
    *,
    source_commit: str,
    source_artifact_sha256: str,
    spec_fingerprint: str,
    source_state_sha256: str,
    archive_loader_implementation_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "source_commit": _commit(source_commit, "archive authority source commit"),
            "source_artifact_sha256": _sha256(
                source_artifact_sha256, "archive authority source artifact"
            ),
            "spec_fingerprint": _sha256(spec_fingerprint, "archive authority spec"),
            "source_state_sha256": _sha256(
                source_state_sha256, "archive authority source state"
            ),
            "archive_loader_implementation_fingerprint": _sha256(
                archive_loader_implementation_fingerprint, "archive authority loader"
            ),
            "archive_schema_fingerprint": POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT,
        }
    )


def _metadata_snapshot(provider: object, fields: tuple[str, ...], label: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for name in fields:
        value = getattr(provider, name, None)
        if name == "source_commit":
            clean = _commit(value, f"{label} {name}")
        else:
            clean = _sha256(value, f"{label} {name}")
        result.append((name, clean))
    return tuple(result)


def pocket_refinement_archive_manifest_sha256(
    archive: Vituri2024ImmutablePocketRefinementArchive,
) -> str:
    if type(archive) is not Vituri2024ImmutablePocketRefinementArchive:
        raise TypeError("archive manifest requires the exact typed archive")
    return _fingerprint(
        {
            "schema": POCKET_REFINEMENT_ARCHIVE_SCHEMA_LABEL,
            "archive_authority_fingerprint": archive.archive_authority_fingerprint,
            "source_commit": archive.source_commit,
            "source_artifact_sha256": archive.source_artifact_sha256,
            "spec_fingerprint": archive.spec_fingerprint,
            "source_state_sha256": archive.source_state_sha256,
            "selected_branch_label": archive.selected_branch_label,
            "selected_spin": archive.selected_spin,
            "chemical_potential_ev": archive.chemical_potential_ev,
            "archive_loader_implementation_fingerprint": (
                archive.archive_loader_implementation_fingerprint
            ),
            "archive_schema_fingerprint": archive.archive_schema_fingerprint,
            "generation_phase": archive.generation_phase,
            "mesh_fingerprint": archive.mesh.fingerprint,
            "fields": {
                "h0": _array_manifest(archive.fields.h0),
                "interaction_h": _array_manifest(archive.fields.interaction_h),
                "fock": _array_manifest(archive.fields.fock),
            },
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementPrerequisites:
    binding: Vituri2024HalfMetalHFProviderBinding
    array_replay_receipt: Vituri2024HalfMetalHFReplayReceipt
    scf_replay_approval: Vituri2024SCFReplayApproval
    scf_replay_receipt: Vituri2024SCFReplayReceipt

    def __post_init__(self) -> None:
        if type(self.binding) is not Vituri2024HalfMetalHFProviderBinding:
            raise TypeError("pocket replay requires the exact typed provider binding")
        if type(self.array_replay_receipt) is not Vituri2024HalfMetalHFReplayReceipt:
            raise TypeError("pocket replay requires a factory array replay receipt")
        if type(self.scf_replay_approval) is not Vituri2024SCFReplayApproval:
            raise TypeError("pocket replay requires the exact typed SCF approval")
        if type(self.scf_replay_receipt) is not Vituri2024SCFReplayReceipt:
            raise TypeError("pocket replay requires a factory SCF replay receipt")
        spec = self.binding.spec
        spec.require_receipt_set_complete()
        assert spec.attested_source is not None
        assert spec.scf_policy is not None and spec.shared_functional is not None
        source = spec.attested_source
        provider = self.binding.provider
        array = self.array_replay_receipt
        scf_approval = self.scf_replay_approval
        scf = self.scf_replay_receipt
        if not (
            array.status.arrays_loaded
            and array.status.array_hashes_verified
            and array.status.source_structure_verified
            and array.status.pocket_refinement_replayed is False
        ):
            raise ValueError("array replay prerequisite status is not the factory array-only status")
        if not (
            scf.status.selected_final_source_reproduced
            and scf.status.branch_table_replayed
            and scf.status.scientific_execution_verified is False
        ):
            raise ValueError("SCF selected-source prerequisite status is not closed")
        if scf.approval_fingerprint != scf_approval.fingerprint:
            raise ValueError("SCF receipt/approval fingerprint mismatch")
        expected = (
            (array.provider_fingerprint, source.provider_fingerprint, "array provider"),
            (array.source_commit, source.source_commit, "array source commit"),
            (array.source_artifact_sha256, source.source_artifact_sha256, "array source artifact"),
            (array.spec_fingerprint, spec.fingerprint, "array spec"),
            (
                array.attested_source_receipt_fingerprint,
                source.fingerprint,
                "array attested source receipt",
            ),
            (
                array.hashes.reconstructed_source_state_sha256,
                source.source_state_sha256,
                "array source state",
            ),
            (
                array.replay_loader_implementation_fingerprint,
                source.replay_loader_implementation_fingerprint,
                "array replay loader",
            ),
            (
                array.replay_payload_schema_fingerprint,
                source.replay_payload_schema_fingerprint,
                "array replay schema",
            ),
            (scf_approval.provider_fingerprint, source.provider_fingerprint, "SCF approval provider"),
            (
                scf_approval.functional_provider_fingerprint,
                getattr(provider, "functional_provider_fingerprint", None),
                "SCF approval functional provider",
            ),
            (
                scf_approval.scf_provider_fingerprint,
                getattr(provider, "scf_provider_fingerprint", None),
                "SCF approval SCF provider",
            ),
            (scf_approval.source_commit, source.source_commit, "SCF approval source commit"),
            (
                scf_approval.source_artifact_sha256,
                source.source_artifact_sha256,
                "SCF approval source artifact",
            ),
            (scf_approval.spec_fingerprint, spec.fingerprint, "SCF approval spec"),
            (
                scf_approval.scf_policy_fingerprint,
                spec.scf_policy.fingerprint,
                "SCF approval policy",
            ),
            (
                scf_approval.shared_functional_fingerprint,
                spec.shared_functional.fingerprint,
                "SCF approval shared functional",
            ),
            (
                scf_approval.attested_source_fingerprint,
                source.fingerprint,
                "SCF approval attested source",
            ),
            (
                scf_approval.source_state_sha256,
                source.source_state_sha256,
                "SCF approval source state",
            ),
            (
                scf_approval.scalar_energy_function_fingerprint,
                spec.shared_functional.scalar_energy.fingerprint,
                "SCF approval scalar energy",
            ),
            (
                scf_approval.state_builder_implementation_fingerprint,
                getattr(provider, "state_builder_implementation_fingerprint", None),
                "SCF approval state builder",
            ),
            (
                scf_approval.problem_builder_implementation_fingerprint,
                getattr(provider, "problem_builder_implementation_fingerprint", None),
                "SCF approval problem builder",
            ),
            (
                scf_approval.scf_adapter_schema_fingerprint,
                getattr(provider, "scf_adapter_schema_fingerprint", None),
                "SCF approval adapter schema",
            ),
            (
                scf_approval.scf_adapter_abi_fingerprint,
                getattr(provider, "scf_adapter_abi_fingerprint", None),
                "SCF approval adapter ABI",
            ),
            (
                scf_approval.scf_dependency_archive_fingerprint,
                getattr(provider, "scf_dependency_archive_fingerprint", None),
                "SCF approval dependency archive",
            ),
            (
                scf.archive_manifest_sha256,
                scf_approval.expected_archive_manifest_sha256,
                "SCF approval/archive manifest",
            ),
            (
                scf.branch_table_sha256,
                scf_approval.expected_branch_table_sha256,
                "SCF approval/branch table",
            ),
            (
                scf.core_provenance_fingerprint,
                scf_approval.core_provenance_fingerprint,
                "SCF approval/core provenance",
            ),
            (
                scf.verifier_module_ast_manifest_sha256,
                scf_approval.verifier_module_ast_manifest_sha256,
                "SCF approval/verifier AST",
            ),
            (
                scf.archive_authority_fingerprint,
                scf_approval.archive_authority_fingerprint,
                "SCF approval/archive authority",
            ),
            (scf.selected_branch_label, source.selected_branch_label, "selected branch"),
        )
        for actual, required, label in expected:
            if actual != required:
                raise ValueError(f"pocket prerequisite {label} mismatch")
        if tuple(item.valley for item in source.pocket_evidence) != (-1, 1):
            raise ValueError("preflight pocket receipts must be ordered (-1,+1)")

    @property
    def array_receipt_fingerprint(self) -> str:
        return canonical_half_metal_hf_replay_receipt_fingerprint(
            self.array_replay_receipt
        )

    @property
    def scf_approval_fingerprint(self) -> str:
        return self.scf_replay_approval.fingerprint

    @property
    def scf_receipt_fingerprint(self) -> str:
        return canonical_scf_replay_receipt_fingerprint(self.scf_replay_receipt)

    @property
    def base_hashes(self) -> Vituri2024PocketBaseHashes:
        hashes = self.array_replay_receipt.hashes
        return Vituri2024PocketBaseHashes(
            ordered_momentum_mesh_sha256=hashes.ordered_momentum_mesh_sha256,
            h0_sha256=hashes.h0_sha256,
            interaction_h_sha256=hashes.interaction_h_sha256,
            fock_sha256=hashes.ordered_fock_sha256,
            energies_sha256=hashes.ordered_energies_sha256,
            occupations_sha256=hashes.ordered_occupations_sha256,
            projector_sha256=hashes.ordered_projector_sha256,
        )


@dataclass(frozen=True, slots=True)
class Vituri2024PocketTopologySignature:
    hole_state_count: int
    hole_component_count: int
    component_cardinalities: tuple[int, ...]
    boundary_hole_state_count: int
    enclosed_complement_component_count: int
    accepted: bool

    def __post_init__(self) -> None:
        counts = (
            _strict_int(self.hole_state_count, "topology hole count"),
            _strict_int(self.hole_component_count, "topology component count"),
            _strict_int(self.boundary_hole_state_count, "topology boundary count"),
            _strict_int(
                self.enclosed_complement_component_count,
                "topology enclosed-complement count",
            ),
        )
        if any(value < 0 for value in counts):
            raise ValueError("topology counts must be non-negative")
        cardinalities = tuple(self.component_cardinalities)
        if (
            len(cardinalities) != counts[1]
            or any(type(value) is not int or value < 1 for value in cardinalities)
            or cardinalities != tuple(sorted(cardinalities, reverse=True))
            or sum(cardinalities) != counts[0]
        ):
            raise ValueError("topology component cardinalities do not close")
        expected = counts[0] > 0 and counts[1] == 1 and counts[2] == 0 and counts[3] == 0
        if type(self.accepted) is not bool or self.accepted is not expected:
            raise ValueError("topology acceptance flag contradicts its signature")
        object.__setattr__(self, "component_cardinalities", cardinalities)


@dataclass(frozen=True, slots=True)
class Vituri2024RefinedValleyTopologyEvidence:
    valley: Literal[-1, 1]
    selected_spin: Literal[-1, 1]
    energy_sha256: str
    occupation_mask_sha256: str
    topology_convention: Literal[
        "foreground_four_neighbor_complement_eight_neighbor_finite_domain_no_wrap"
    ]
    signature: Vituri2024PocketTopologySignature

    def __post_init__(self) -> None:
        if type(self.valley) is not int or self.valley not in (-1, 1):
            raise ValueError("topology valley must be exactly -1 or +1")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("topology selected spin must be exactly -1 or +1")
        _sha256(self.energy_sha256, "topology energy")
        _sha256(self.occupation_mask_sha256, "topology occupation mask")
        if self.topology_convention != POCKET_REFINEMENT_TOPOLOGY_CONVENTION:
            raise ValueError("topology convention changed")
        if type(self.signature) is not Vituri2024PocketTopologySignature:
            raise TypeError("topology evidence requires a typed signature")


@dataclass(frozen=True, slots=True)
class Vituri2024DiscreteLifshitzLaneEvidence:
    energy_sha256: str
    minimum_absolute_energy_distance_to_mu_ev: float
    lower_critical_level_ev: float
    upper_critical_level_ev: float
    lower_critical_level_multiplicity: int
    upper_critical_level_multiplicity: int
    lower_rejected_signature: Vituri2024PocketTopologySignature
    upper_rejected_signature: Vituri2024PocketTopologySignature
    lower_margin_ev: float
    upper_margin_ev: float
    raw_margin_ev: float

    def __post_init__(self) -> None:
        _sha256(self.energy_sha256, "Lifshitz energy")
        minimum = _positive(
            self.minimum_absolute_energy_distance_to_mu_ev,
            "minimum energy distance to mu",
        )
        lower = _finite(self.lower_critical_level_ev, "lower critical level")
        upper = _finite(self.upper_critical_level_ev, "upper critical level")
        lower_mult = _strict_int(
            self.lower_critical_level_multiplicity, "lower critical multiplicity"
        )
        upper_mult = _strict_int(
            self.upper_critical_level_multiplicity, "upper critical multiplicity"
        )
        if lower >= upper or lower_mult < 1 or upper_mult < 1:
            raise ValueError("critical levels/multiplicities are invalid")
        if (
            type(self.lower_rejected_signature) is not Vituri2024PocketTopologySignature
            or type(self.upper_rejected_signature) is not Vituri2024PocketTopologySignature
            or self.lower_rejected_signature.accepted
            or self.upper_rejected_signature.accepted
        ):
            raise ValueError("critical outside signatures must both be rejected")
        lower_margin = _positive(self.lower_margin_ev, "lower Lifshitz margin")
        upper_margin = _positive(self.upper_margin_ev, "upper Lifshitz margin")
        raw = _positive(self.raw_margin_ev, "raw Lifshitz margin")
        if not math.isclose(
            raw,
            min(lower_margin, upper_margin),
            rel_tol=1.0e-13,
            abs_tol=64.0 * math.ulp(max(raw, lower_margin, upper_margin)),
        ):
            raise ValueError("raw Lifshitz margin is not min(lower,upper)")
        object.__setattr__(self, "minimum_absolute_energy_distance_to_mu_ev", minimum)
        object.__setattr__(self, "lower_critical_level_ev", lower)
        object.__setattr__(self, "upper_critical_level_ev", upper)
        object.__setattr__(self, "lower_critical_level_multiplicity", lower_mult)
        object.__setattr__(self, "upper_critical_level_multiplicity", upper_mult)
        object.__setattr__(self, "lower_margin_ev", lower_margin)
        object.__setattr__(self, "upper_margin_ev", upper_margin)
        object.__setattr__(self, "raw_margin_ev", raw)


@dataclass(frozen=True, slots=True)
class Vituri2024DiscreteLifshitzEvidence:
    valley: Literal[-1, 1]
    selected_spin: Literal[-1, 1]
    convention: Literal[
        "maximal_consecutive_open_energy_threshold_intervals_with_accepted_topology"
    ]
    archive: Vituri2024DiscreteLifshitzLaneEvidence
    live: Vituri2024DiscreteLifshitzLaneEvidence
    locked_threshold_uncertainty_ev: float
    maximum_archive_live_energy_residual_ev: float
    certified_margin_ev: float

    def __post_init__(self) -> None:
        if type(self.valley) is not int or self.valley not in (-1, 1):
            raise ValueError("Lifshitz valley must be exactly -1 or +1")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("Lifshitz selected spin must be exactly -1 or +1")
        if self.convention != POCKET_REFINEMENT_LIFSHITZ_CONVENTION:
            raise ValueError("Lifshitz convention changed")
        if (
            type(self.archive) is not Vituri2024DiscreteLifshitzLaneEvidence
            or type(self.live) is not Vituri2024DiscreteLifshitzLaneEvidence
        ):
            raise TypeError("Lifshitz evidence requires typed archive/live lanes")
        eta = _positive(
            self.locked_threshold_uncertainty_ev, "locked threshold uncertainty"
        )
        residual = _nonnegative(
            self.maximum_archive_live_energy_residual_ev,
            "archive/live energy residual",
        )
        certified = _finite(self.certified_margin_ev, "certified Lifshitz margin")
        expected = min(self.archive.raw_margin_ev, self.live.raw_margin_ev) - eta - residual
        if not math.isclose(
            certified,
            expected,
            rel_tol=1.0e-13,
            abs_tol=64.0 * math.ulp(max(abs(certified), abs(expected), 1.0)),
        ):
            raise ValueError("certified Lifshitz margin formula mismatch")
        if certified <= 0.0:
            raise ValueError("certified Lifshitz margin must be positive")
        object.__setattr__(self, "locked_threshold_uncertainty_ev", eta)
        object.__setattr__(self, "maximum_archive_live_energy_residual_ev", residual)
        object.__setattr__(self, "certified_margin_ev", certified)


def _neighbors(
    row: int, column: int, shape: tuple[int, int], connectivity: Literal[4, 8]
) -> tuple[tuple[int, int], ...]:
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result: list[tuple[int, int]] = []
    for dr, dc in offsets:
        nr, nc = row + dr, column + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            result.append((nr, nc))
    return tuple(result)


def _components(mask: BoolArray, connectivity: Literal[4, 8]) -> tuple[tuple[tuple[int, int], ...], ...]:
    if type(mask) is not np.ndarray or mask.dtype != np.dtype(np.bool_) or mask.ndim != 2:
        raise TypeError("digital topology mask must be an exact rank-two bool array")
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[tuple[tuple[int, int], ...]] = []
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if not mask[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            component: list[tuple[int, int]] = []
            while stack:
                vertex = stack.pop()
                component.append(vertex)
                for neighbor in _neighbors(*vertex, mask.shape, connectivity):
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(tuple(component))
    return tuple(components)


def _touches_boundary(component: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> bool:
    return any(
        row in (0, shape[0] - 1) or column in (0, shape[1] - 1)
        for row, column in component
    )


def vituri2024_pocket_topology_signature(mask: BoolArray) -> Vituri2024PocketTopologySignature:
    if type(mask) is not np.ndarray or mask.dtype != np.dtype(np.bool_) or mask.ndim != 2:
        raise TypeError("pocket topology requires an exact rank-two bool mask")
    holes = _components(mask, 4)
    complement = _components(np.asarray(~mask, dtype=np.bool_), 8)
    boundary_count = int(
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[1:-1, 0])
        + np.count_nonzero(mask[1:-1, -1])
    )
    enclosed = sum(not _touches_boundary(component, mask.shape) for component in complement)
    cardinalities = tuple(sorted((len(component) for component in holes), reverse=True))
    return Vituri2024PocketTopologySignature(
        hole_state_count=int(np.count_nonzero(mask)),
        hole_component_count=len(holes),
        component_cardinalities=cardinalities,
        boundary_hole_state_count=boundary_count,
        enclosed_complement_component_count=int(enclosed),
        accepted=(
            len(holes) == 1
            and boundary_count == 0
            and enclosed == 0
            and bool(np.any(mask))
        ),
    )


def _lifshitz_lane(
    energies: FloatArray,
    *,
    mu: float,
    uncertainty: float,
    shape: tuple[int, int],
) -> Vituri2024DiscreteLifshitzLaneEvidence:
    values = _immutable_array(
        energies,
        label="valley refined energies",
        dtype=np.dtype(np.float64),
        shape=(shape[0] * shape[1],),
    )
    clean_mu = _finite(mu, "Lifshitz chemical potential")
    eta = _positive(uncertainty, "Lifshitz threshold uncertainty")
    minimum_distance = float(np.min(np.abs(values - clean_mu)))
    if minimum_distance <= eta:
        raise ValueError("refined energy lies within locked threshold uncertainty of mu")
    levels, multiplicities = np.unique(values, return_counts=True)
    start = int(np.searchsorted(levels, clean_mu, side="right"))

    def interval_signature(interval: int) -> Vituri2024PocketTopologySignature:
        if interval == 0:
            mask = np.ones(values.shape, dtype=np.bool_)
        else:
            mask = np.asarray(values > levels[interval - 1], dtype=np.bool_)
        return vituri2024_pocket_topology_signature(mask.reshape(shape))

    signatures = tuple(interval_signature(index) for index in range(len(levels) + 1))
    if not signatures[start].accepted:
        raise ValueError("topology at chemical potential is not accepted")
    left = start
    while left > 0 and signatures[left - 1].accepted:
        left -= 1
    right = start
    while right < len(levels) and signatures[right + 1].accepted:
        right += 1
    if left == 0 or right == len(levels):
        raise ValueError("accepted topology lacks finite lower/upper critical energy levels")
    lower_level = float(levels[left - 1])
    upper_level = float(levels[right])
    lower_margin = clean_mu - lower_level
    upper_margin = upper_level - clean_mu
    raw_margin = min(lower_margin, upper_margin)
    if raw_margin <= eta:
        raise ValueError("raw Lifshitz margin does not exceed locked uncertainty")
    return Vituri2024DiscreteLifshitzLaneEvidence(
        energy_sha256=canonical_array_sha256(values),
        minimum_absolute_energy_distance_to_mu_ev=minimum_distance,
        lower_critical_level_ev=lower_level,
        upper_critical_level_ev=upper_level,
        lower_critical_level_multiplicity=int(multiplicities[left - 1]),
        upper_critical_level_multiplicity=int(multiplicities[right]),
        lower_rejected_signature=signatures[left - 1],
        upper_rejected_signature=signatures[right + 1],
        lower_margin_ev=lower_margin,
        upper_margin_ev=upper_margin,
        raw_margin_ev=raw_margin,
    )


def _derived_arrays(fields: Vituri2024ArchivedPocketRefinementFields, mu: float, uncertainty: float) -> tuple[FloatArray, IntegerArray, ComplexArray]:
    energies = np.asarray(np.diagonal(fields.fock, axis1=0, axis2=1).T.real, dtype=np.float64)
    distance = np.abs(energies - mu)
    if np.any(distance <= uncertainty):
        raise ValueError("refined Fock diagonal contains threshold-ambiguous states")
    occupations = np.asarray(energies < mu, dtype=np.int64)
    projector = np.zeros(fields.fock.shape, dtype=np.complex128)
    diagonal = np.arange(4)
    projector[diagonal, diagonal, :] = occupations
    return energies, occupations, projector


def _field_hashes(
    fields: Vituri2024ArchivedPocketRefinementFields,
    energies: FloatArray,
    occupations: IntegerArray,
    projector: ComplexArray,
) -> Vituri2024PocketRefinementFieldHashes:
    return Vituri2024PocketRefinementFieldHashes(
        h0_sha256=canonical_array_sha256(fields.h0),
        interaction_h_sha256=canonical_array_sha256(fields.interaction_h),
        fock_sha256=canonical_array_sha256(fields.fock),
        energies_sha256=canonical_array_sha256(energies),
        occupations_sha256=canonical_array_sha256(occupations),
        projector_sha256=canonical_array_sha256(projector),
    )


def _validate_refined_fields(
    fields: Vituri2024ArchivedPocketRefinementFields,
    point_count: int,
    label: str,
) -> None:
    if fields.h0.shape != (4, 4, point_count):
        raise ValueError(f"{label} refined fields must have shape (4,4,Nref)")
    decomposition = _max_abs(fields.fock - (fields.h0 + fields.interaction_h))
    if decomposition > POCKET_REPLAY_V1_FOCK_DECOMPOSITION_TOLERANCE_EV:
        raise ValueError(f"{label} refined fock=h0+interaction_h closure failed")
    for array, name in (
        (fields.h0, "h0"),
        (fields.interaction_h, "interaction_h"),
        (fields.fock, "fock"),
    ):
        hermiticity = _max_abs(array - np.swapaxes(array.conj(), 0, 1))
        if hermiticity > POCKET_REPLAY_V1_HERMITICITY_TOLERANCE_EV:
            raise ValueError(f"{label} refined {name} Hermiticity failed")
        diagonal = np.zeros_like(array)
        indices = np.arange(4)
        diagonal[indices, indices, :] = array[indices, indices, :]
        if _max_abs(array - diagonal) > POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV:
            raise ValueError(f"{label} refined {name} is not diagonal in the locked flavor basis")
        if _max_abs(np.imag(array[indices, indices, :])) > POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV:
            raise ValueError(f"{label} refined {name} diagonal is not real")


def _valley_topology(
    energies: FloatArray,
    occupations: IntegerArray,
    *,
    valley: int,
    selected_spin: int,
    shape: tuple[int, int],
) -> Vituri2024RefinedValleyTopologyEvidence:
    flavor = INTERNAL_FLAVOR_ORDER.index((valley, selected_spin))
    mask = np.asarray(occupations[flavor] == 0, dtype=np.bool_)
    signature = vituri2024_pocket_topology_signature(mask.reshape(shape))
    if not signature.accepted:
        raise ValueError(f"valley {valley:+d} refined pocket topology is not accepted")
    return Vituri2024RefinedValleyTopologyEvidence(
        valley=valley,  # type: ignore[arg-type]
        selected_spin=selected_spin,  # type: ignore[arg-type]
        energy_sha256=canonical_array_sha256(np.asarray(energies[flavor], dtype=np.float64)),
        occupation_mask_sha256=canonical_array_sha256(mask),
        topology_convention=POCKET_REFINEMENT_TOPOLOGY_CONVENTION,
        signature=signature,
    )


def vituri2024_refinement_evidence_sha256(
    *,
    valley: Literal[-1, 1],
    selected_spin: Literal[-1, 1],
    source_commit: str,
    source_artifact_sha256: str,
    source_state_sha256: str,
    selected_branch_label: str,
    base_hashes: Vituri2024PocketBaseHashes,
    mesh: Vituri2024NestedNoWrapRefinementMesh,
    archive_fields: Vituri2024ArchivedPocketRefinementFields,
    live_fields: Vituri2024ArchivedPocketRefinementFields,
    chemical_potential_ev: float,
    locked_threshold_uncertainty_ev: float,
) -> tuple[
    str,
    Vituri2024RefinedValleyTopologyEvidence,
    Vituri2024RefinedValleyTopologyEvidence,
    Vituri2024DiscreteLifshitzEvidence,
    Vituri2024PocketRefinementFieldHashes,
    Vituri2024PocketRefinementFieldHashes,
]:
    """Build the verifier-defined, acyclic per-valley refinement evidence."""

    _commit(source_commit, "evidence source commit")
    _sha256(source_artifact_sha256, "evidence source artifact")
    _sha256(source_state_sha256, "evidence source state")
    _text(selected_branch_label, "evidence selected branch")
    if type(base_hashes) is not Vituri2024PocketBaseHashes:
        raise TypeError("refinement evidence requires typed base hashes")
    if type(mesh) is not Vituri2024NestedNoWrapRefinementMesh:
        raise TypeError("refinement evidence requires typed mesh")
    if type(archive_fields) is not Vituri2024ArchivedPocketRefinementFields or type(
        live_fields
    ) is not Vituri2024ArchivedPocketRefinementFields:
        raise TypeError("refinement evidence requires typed archive/live fields")
    mu = _finite(chemical_potential_ev, "evidence chemical potential")
    eta = _positive(locked_threshold_uncertainty_ev, "evidence uncertainty")
    point_count = mesh.refined_mesh.shape[0]
    _validate_refined_fields(archive_fields, point_count, "archive")
    _validate_refined_fields(live_fields, point_count, "live")
    archive_energies, archive_occupations, archive_projector = _derived_arrays(
        archive_fields, mu, eta
    )
    live_energies, live_occupations, live_projector = _derived_arrays(
        live_fields, mu, eta
    )
    archive_hashes = _field_hashes(
        archive_fields, archive_energies, archive_occupations, archive_projector
    )
    live_hashes = _field_hashes(
        live_fields, live_energies, live_occupations, live_projector
    )
    archive_topology = _valley_topology(
        archive_energies,
        archive_occupations,
        valley=valley,
        selected_spin=selected_spin,
        shape=mesh.refined_shape,
    )
    live_topology = _valley_topology(
        live_energies,
        live_occupations,
        valley=valley,
        selected_spin=selected_spin,
        shape=mesh.refined_shape,
    )
    flavor = INTERNAL_FLAVOR_ORDER.index((valley, selected_spin))
    archive_lane = _lifshitz_lane(
        np.asarray(archive_energies[flavor], dtype=np.float64),
        mu=mu,
        uncertainty=eta,
        shape=mesh.refined_shape,
    )
    live_lane = _lifshitz_lane(
        np.asarray(live_energies[flavor], dtype=np.float64),
        mu=mu,
        uncertainty=eta,
        shape=mesh.refined_shape,
    )
    energy_residual = _max_abs(archive_energies[flavor] - live_energies[flavor])
    lifshitz = Vituri2024DiscreteLifshitzEvidence(
        valley=valley,
        selected_spin=selected_spin,
        convention=POCKET_REFINEMENT_LIFSHITZ_CONVENTION,
        archive=archive_lane,
        live=live_lane,
        locked_threshold_uncertainty_ev=eta,
        maximum_archive_live_energy_residual_ev=energy_residual,
        certified_margin_ev=(
            min(archive_lane.raw_margin_ev, live_lane.raw_margin_ev)
            - eta
            - energy_residual
        ),
    )
    digest = _fingerprint(
        {
            "schema": POCKET_REFINEMENT_EVIDENCE_SCHEMA_LABEL,
            "topology_convention": POCKET_REFINEMENT_TOPOLOGY_CONVENTION,
            "lifshitz_convention": POCKET_REFINEMENT_LIFSHITZ_CONVENTION,
            "interpretation": (
                "finite_grid_threshold_topology_margin_not_nearest_level_"
                "not_continuum_saddle_not_refinement_convergence"
            ),
            "valley": valley,
            "selected_spin": selected_spin,
            "source_commit": source_commit,
            "source_artifact_sha256": source_artifact_sha256,
            "source_state_sha256": source_state_sha256,
            "selected_branch_label": selected_branch_label,
            "base_hashes": asdict(base_hashes),
            "mesh_registration_fingerprint": mesh.fingerprint,
            "base_point_count": int(mesh.base_mesh.shape[0]),
            "refined_point_count": int(mesh.refined_mesh.shape[0]),
            "archive_field_hashes": asdict(archive_hashes),
            "live_field_hashes": asdict(live_hashes),
            "archive_topology": asdict(archive_topology),
            "live_topology": asdict(live_topology),
            "lifshitz": asdict(lifshitz),
            "locked_numeric_tolerances": {
                "absolute_ev": POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV,
                "relative": POCKET_REPLAY_V1_RELATIVE_TOLERANCE,
                "mesh_absolute_inverse_angstrom": (
                    POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM
                ),
                "mesh_relative": POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE,
                "hermiticity_ev": POCKET_REPLAY_V1_HERMITICITY_TOLERANCE_EV,
                "diagonal_ev": POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV,
                "fock_decomposition_ev": (
                    POCKET_REPLAY_V1_FOCK_DECOMPOSITION_TOLERANCE_EV
                ),
                "threshold_uncertainty_ev": eta,
                "maximum_archive_live_energy_residual_ev": energy_residual,
            },
        }
    )
    return (
        digest,
        archive_topology,
        live_topology,
        lifshitz,
        archive_hashes,
        live_hashes,
    )


def _validate_live_provider_semantics(
    binding: Vituri2024HalfMetalHFProviderBinding,
    snapshot: tuple[tuple[str, str], ...],
    evaluator_manifest: Vituri2024PocketCallableManifest,
) -> None:
    """Re-derive live evaluator metadata from the callable and current source."""

    spec = binding.spec
    assert spec.attested_source is not None
    source = spec.attested_source
    values = dict(snapshot)
    expected = {
        "provider_fingerprint": source.provider_fingerprint,
        "source_commit": source.source_commit,
        "source_artifact_sha256": source.source_artifact_sha256,
        "spec_fingerprint": spec.fingerprint,
        "source_state_sha256": source.source_state_sha256,
        "replay_loader_implementation_fingerprint": (
            source.replay_loader_implementation_fingerprint
        ),
        "replay_payload_schema_fingerprint": (
            source.replay_payload_schema_fingerprint
        ),
        "refinement_evaluator_implementation_fingerprint": (
            evaluator_manifest.fingerprint
        ),
        "refinement_request_schema_fingerprint": (
            POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT
        ),
        "refinement_evaluation_schema_fingerprint": (
            POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT
        ),
    }
    for name, required in expected.items():
        if values[name] != required:
            raise ValueError(f"live pocket provider semantic mismatch: {name}")
    required_provider = pocket_refinement_provider_fingerprint(
        base_provider_fingerprint=values["provider_fingerprint"],
        evaluator_implementation_fingerprint=values[
            "refinement_evaluator_implementation_fingerprint"
        ],
    )
    if values["pocket_refinement_provider_fingerprint"] != required_provider:
        raise ValueError("live pocket provider helper-derived fingerprint mismatch")


def _validate_archive_authority_semantics(
    binding: Vituri2024HalfMetalHFProviderBinding,
    snapshot: tuple[tuple[str, str], ...],
    loader_manifest: Vituri2024PocketCallableManifest,
) -> None:
    """Re-derive archive authority metadata from its callable and current source."""

    spec = binding.spec
    assert spec.attested_source is not None
    source = spec.attested_source
    values = dict(snapshot)
    expected = {
        "source_commit": source.source_commit,
        "source_artifact_sha256": source.source_artifact_sha256,
        "spec_fingerprint": spec.fingerprint,
        "source_state_sha256": source.source_state_sha256,
        "archive_loader_implementation_fingerprint": loader_manifest.fingerprint,
        "archive_schema_fingerprint": POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT,
    }
    for name, required in expected.items():
        if values[name] != required:
            raise ValueError(f"pocket archive authority semantic mismatch: {name}")
    required_authority = pocket_refinement_archive_authority_fingerprint(
        source_commit=source.source_commit,
        source_artifact_sha256=source.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=source.source_state_sha256,
        archive_loader_implementation_fingerprint=loader_manifest.fingerprint,
    )
    if values["archive_authority_fingerprint"] != required_authority:
        raise ValueError("pocket archive helper-derived authority fingerprint mismatch")


def _derived_pocket_approval_fields(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    *,
    prerequisite_provenance: Vituri2024PocketPrerequisiteProvenance,
    authority_snapshot: tuple[tuple[str, str], ...],
    loader_manifest: Vituri2024PocketCallableManifest,
    provider_snapshot: tuple[tuple[str, str], ...],
    evaluator_manifest: Vituri2024PocketCallableManifest,
) -> dict[str, object]:
    """Reconstruct every scientific approval field from live bound authorities."""

    spec = prerequisites.binding.spec
    assert spec.attested_source is not None
    source = spec.attested_source
    pockets = source.pocket_evidence
    if (
        pockets[0].refinement_mesh_sha256
        != pockets[1].refinement_mesh_sha256
        or pockets[0].refinement_point_count
        != pockets[1].refinement_point_count
    ):
        raise ValueError("bilateral preflight pockets must bind one refinement mesh/count")
    return {
        "scope": VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE,
        "prerequisite_provenance": prerequisite_provenance,
        "verifier_module_ast_manifest_sha256": (
            pocket_refinement_replay_module_ast_manifest_sha256()
        ),
        "array_replay_receipt_fingerprint": prerequisites.array_receipt_fingerprint,
        "scf_replay_approval_fingerprint": prerequisites.scf_approval_fingerprint,
        "scf_replay_receipt_fingerprint": prerequisites.scf_receipt_fingerprint,
        "source_commit": source.source_commit,
        "source_artifact_sha256": source.source_artifact_sha256,
        "spec_fingerprint": spec.fingerprint,
        "source_state_sha256": source.source_state_sha256,
        "selected_branch_label": source.selected_branch_label,
        "selected_spin": source.selected_spin,
        "base_hashes": prerequisites.base_hashes,
        "scf_contract_fingerprint": (
            prerequisites.scf_replay_receipt.contract_fingerprint
        ),
        "scf_archive_manifest_sha256": (
            prerequisites.scf_replay_receipt.archive_manifest_sha256
        ),
        "scf_core_provenance_fingerprint": (
            prerequisites.scf_replay_receipt.core_provenance_fingerprint
        ),
        "scf_selected_source_status": "selected_final_source_reproduced",
        "ordered_preflight_pocket_receipt_fingerprints": (
            pockets[0].fingerprint,
            pockets[1].fingerprint,
        ),
        "preflight_refinement_mesh_sha256": pockets[0].refinement_mesh_sha256,
        "preflight_refinement_point_count": pockets[0].refinement_point_count,
        "preflight_refinement_evidence_sha256": (
            pockets[0].refinement_evidence_sha256,
            pockets[1].refinement_evidence_sha256,
        ),
        "preflight_raw_lifshitz_margins_ev": (
            pockets[0].lifshitz_margin_ev,
            pockets[1].lifshitz_margin_ev,
        ),
        "preflight_lifshitz_uncertainties_ev": (
            pockets[0].lifshitz_tolerance_ev,
            pockets[1].lifshitz_tolerance_ev,
        ),
        "archive_authority_metadata_snapshot": authority_snapshot,
        "archive_loader_callable_manifest": loader_manifest,
        "live_provider_metadata_snapshot": provider_snapshot,
        "live_evaluator_callable_manifest": evaluator_manifest,
    }


_POCKET_APPROVAL_EXTERNAL_FIELDS = frozenset(
    ("expected_archive_manifest_sha256", "detached_approval_provenance")
)


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementReplayApproval:
    scope: Literal["vituri2024_frozen_selected_source_pocket_refinement_replay_v1"]
    prerequisite_provenance: Vituri2024PocketPrerequisiteProvenance
    verifier_module_ast_manifest_sha256: str
    array_replay_receipt_fingerprint: str
    scf_replay_approval_fingerprint: str
    scf_replay_receipt_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: Literal[-1, 1]
    base_hashes: Vituri2024PocketBaseHashes
    scf_contract_fingerprint: str
    scf_archive_manifest_sha256: str
    scf_core_provenance_fingerprint: str
    scf_selected_source_status: Literal["selected_final_source_reproduced"]
    ordered_preflight_pocket_receipt_fingerprints: tuple[str, str]
    preflight_refinement_mesh_sha256: str
    preflight_refinement_point_count: int
    preflight_refinement_evidence_sha256: tuple[str, str]
    preflight_raw_lifshitz_margins_ev: tuple[float, float]
    preflight_lifshitz_uncertainties_ev: tuple[float, float]
    expected_archive_manifest_sha256: str
    archive_authority_metadata_snapshot: tuple[tuple[str, str], ...]
    archive_loader_callable_manifest: Vituri2024PocketCallableManifest
    live_provider_metadata_snapshot: tuple[tuple[str, str], ...]
    live_evaluator_callable_manifest: Vituri2024PocketCallableManifest
    detached_approval_provenance: str

    def __post_init__(self) -> None:
        if self.scope != VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE:
            raise ValueError("pocket replay approval scope changed")
        if type(self.prerequisite_provenance) is not Vituri2024PocketPrerequisiteProvenance:
            raise TypeError("approval requires typed prerequisite provenance")
        for value, label in (
            (self.verifier_module_ast_manifest_sha256, "approval verifier AST"),
            (self.array_replay_receipt_fingerprint, "approval array receipt"),
            (self.scf_replay_approval_fingerprint, "approval SCF approval"),
            (self.scf_replay_receipt_fingerprint, "approval SCF receipt"),
            (self.source_artifact_sha256, "approval source artifact"),
            (self.spec_fingerprint, "approval spec"),
            (self.source_state_sha256, "approval source state"),
            (self.scf_contract_fingerprint, "approval SCF contract"),
            (self.scf_archive_manifest_sha256, "approval SCF archive"),
            (self.scf_core_provenance_fingerprint, "approval SCF core provenance"),
            (self.preflight_refinement_mesh_sha256, "approval refinement mesh"),
            (self.expected_archive_manifest_sha256, "approval archive manifest"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "approval source commit")
        _text(self.selected_branch_label, "approval selected branch")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("approval selected spin must be exactly -1 or +1")
        if type(self.base_hashes) is not Vituri2024PocketBaseHashes:
            raise TypeError("approval requires typed base hashes")
        if self.verifier_module_ast_manifest_sha256 != pocket_refinement_replay_module_ast_manifest_sha256():
            raise ValueError("pocket replay verifier full-module AST changed")
        if self.scf_selected_source_status != "selected_final_source_reproduced":
            raise ValueError("approval SCF selected-source status changed")
        pocket_fingerprints = tuple(self.ordered_preflight_pocket_receipt_fingerprints)
        if len(pocket_fingerprints) != 2:
            raise ValueError("approval requires two ordered preflight pocket fingerprints")
        for digest in pocket_fingerprints:
            _sha256(digest, "approval preflight pocket receipt")
        object.__setattr__(
            self, "ordered_preflight_pocket_receipt_fingerprints", pocket_fingerprints
        )
        count = _strict_int(
            self.preflight_refinement_point_count, "approval refinement point count"
        )
        if count < 1:
            raise ValueError("approval refinement point count must be positive")
        object.__setattr__(self, "preflight_refinement_point_count", count)
        evidences = tuple(self.preflight_refinement_evidence_sha256)
        margins = tuple(self.preflight_raw_lifshitz_margins_ev)
        uncertainties = tuple(self.preflight_lifshitz_uncertainties_ev)
        if len(evidences) != 2 or len(margins) != 2 or len(uncertainties) != 2:
            raise ValueError("approval bilateral evidence/margin inventories changed")
        for digest in evidences:
            _sha256(digest, "approval preflight refinement evidence")
        for value in margins:
            _positive(value, "approval raw Lifshitz margin")
        for value in uncertainties:
            _positive(value, "approval Lifshitz uncertainty")
        object.__setattr__(self, "preflight_refinement_evidence_sha256", evidences)
        object.__setattr__(self, "preflight_raw_lifshitz_margins_ev", margins)
        object.__setattr__(self, "preflight_lifshitz_uncertainties_ev", uncertainties)
        if tuple(name for name, _ in self.archive_authority_metadata_snapshot) != POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS:
            raise ValueError("approval archive-authority metadata inventory changed")
        if tuple(name for name, _ in self.live_provider_metadata_snapshot) != POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS:
            raise ValueError("approval live-provider metadata inventory changed")
        if type(self.archive_loader_callable_manifest) is not Vituri2024PocketCallableManifest or self.archive_loader_callable_manifest.role != "load_immutable_pocket_refinement_archive":
            raise TypeError("approval archive loader callable manifest changed")
        if type(self.live_evaluator_callable_manifest) is not Vituri2024PocketCallableManifest or self.live_evaluator_callable_manifest.role != "evaluate_frozen_selected_hf_source":
            raise TypeError("approval live evaluator callable manifest changed")
        _text(self.detached_approval_provenance, "detached pocket approval provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _validate_pocket_approval_ingress(
    approval: Vituri2024PocketRefinementReplayApproval,
    derived_fields: dict[str, object],
) -> None:
    """Fail closed unless every non-external approval field is reconstructed."""

    approval_fields = tuple(item.name for item in dataclass_fields(approval))
    expected_derived = set(approval_fields) - _POCKET_APPROVAL_EXTERNAL_FIELDS
    if set(derived_fields) != expected_derived:
        missing = tuple(sorted(expected_derived - set(derived_fields)))
        extra = tuple(sorted(set(derived_fields) - expected_derived))
        raise RuntimeError(
            f"pocket approval reconstruction inventory drift: missing={missing}, extra={extra}"
        )
    for name in approval_fields:
        if name in _POCKET_APPROVAL_EXTERNAL_FIELDS:
            continue
        if getattr(approval, name) != derived_fields[name]:
            raise ValueError(f"pocket approval field mismatch: {name}")


def make_vituri2024_pocket_refinement_replay_approval(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    archive_authority: Vituri2024PocketRefinementArchiveAuthorityProtocol,
    *,
    expected_archive_manifest_sha256: str,
    provenance: str,
) -> Vituri2024PocketRefinementReplayApproval:
    """Create detached approval without invoking archive or evaluator methods."""

    if type(prerequisites) is not Vituri2024PocketRefinementPrerequisites:
        raise TypeError("pocket approval requires exact typed prerequisites")
    if not isinstance(archive_authority, Vituri2024PocketRefinementArchiveAuthorityProtocol):
        raise TypeError("archive authority does not satisfy the pocket authority protocol")
    spec = prerequisites.binding.spec
    provider = prerequisites.binding.provider
    # Re-attest mutable provider metadata and all source/spec/state prerequisites.
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    Vituri2024PocketRefinementPrerequisites(
        prerequisites.binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    if not isinstance(provider, Vituri2024PocketRefinementProviderProtocol):
        raise TypeError("live provider does not satisfy the pocket evaluator protocol")
    loader = getattr(archive_authority, "load_immutable_pocket_refinement_archive", None)
    evaluator = getattr(provider, "evaluate_frozen_selected_hf_source", None)
    if not callable(loader) or not callable(evaluator):
        raise TypeError("pocket authority/provider methods must be callable")
    if archive_authority is provider:
        raise ValueError("archive authority and live evaluator provider must be distinct objects")
    authority_snapshot = _metadata_snapshot(
        archive_authority,
        POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS,
        "archive authority",
    )
    provider_snapshot = _metadata_snapshot(
        provider, POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS, "live provider"
    )
    loader_manifest = vituri2024_pocket_callable_manifest(
        "load_immutable_pocket_refinement_archive", loader
    )
    evaluator_manifest = vituri2024_pocket_callable_manifest(
        "evaluate_frozen_selected_hf_source", evaluator
    )
    _validate_archive_authority_semantics(
        prerequisites.binding, authority_snapshot, loader_manifest
    )
    _validate_live_provider_semantics(
        prerequisites.binding, provider_snapshot, evaluator_manifest
    )
    authority_values = dict(authority_snapshot)
    provider_values = dict(provider_snapshot)
    if authority_values["archive_authority_fingerprint"] in (
        provider_values["provider_fingerprint"],
        provider_values["pocket_refinement_provider_fingerprint"],
    ):
        raise ValueError("archive authority and live provider fingerprints must be distinct")
    prerequisite_provenance = verified_vituri2024_pocket_prerequisite_provenance()
    derived_fields = _derived_pocket_approval_fields(
        prerequisites,
        prerequisite_provenance=prerequisite_provenance,
        authority_snapshot=authority_snapshot,
        loader_manifest=loader_manifest,
        provider_snapshot=provider_snapshot,
        evaluator_manifest=evaluator_manifest,
    )
    return Vituri2024PocketRefinementReplayApproval(
        **derived_fields,  # type: ignore[arg-type]
        expected_archive_manifest_sha256=_sha256(
            expected_archive_manifest_sha256, "expected pocket archive manifest"
        ),
        detached_approval_provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementReplayContract:
    approval_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    expected_archive_manifest_sha256: str
    archive_authority_metadata_fingerprint: str
    archive_loader_callable_fingerprint: str
    live_provider_metadata_fingerprint: str
    live_evaluator_callable_fingerprint: str
    request_schema_fingerprint: str
    evaluation_schema_fingerprint: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _sha256(value, name.replace("_", " "))
        if self.verifier_module_ast_manifest_sha256 != pocket_refinement_replay_module_ast_manifest_sha256():
            raise ValueError("pocket replay contract verifier AST changed")
        if self.request_schema_fingerprint != POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT or self.evaluation_schema_fingerprint != POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT:
            raise ValueError("pocket replay contract request/evaluation schemas changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementReplayStatus:
    _factory_token: InitVar[object | None] = None
    evidence_model: Literal[
        "trusted_live_selected_source_evaluator_distinct_refinement_archive_object"
    ] = field(default=POCKET_REFINEMENT_EVIDENCE_MODEL, init=False)
    array_replay_prerequisite_bound: bool = field(default=True, init=False)
    scf_selected_source_prerequisite_bound: bool = field(default=True, init=False)
    detached_refinement_archive_loaded: bool = field(default=True, init=False)
    structured_refinement_mesh_registered: bool = field(default=True, init=False)
    live_frozen_selected_source_evaluated: bool = field(default=True, init=False)
    refined_occupations_recomputed: bool = field(default=True, init=False)
    bilateral_hole_topology_recomputed: bool = field(default=True, init=False)
    discrete_lifshitz_margin_recomputed: bool = field(default=True, init=False)
    pocket_refinement_replayed: bool = field(default=True, init=False)
    real_vituri_artifact_replayed: bool = field(default=False, init=False)
    archive_live_computational_independence_verified: bool = field(default=False, init=False)
    hostile_provider_resistance_verified: bool = field(default=False, init=False)
    hidden_live_dependency_state_excluded: bool = field(default=False, init=False)
    refined_scf_executed: bool = field(default=False, init=False)
    refined_fixed_density_resolved: bool = field(default=False, init=False)
    continuum_pocket_stability_verified: bool = field(default=False, init=False)
    refinement_convergence_verified: bool = field(default=False, init=False)
    global_ground_state_verified: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)
    tdhf_readiness_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _POCKET_SUCCESS_TOKEN:
            raise ValueError("successful pocket replay status is factory-only")
        if self.evidence_model != POCKET_REFINEMENT_EVIDENCE_MODEL:
            raise ValueError("pocket replay evidence model changed")
        positives = (
            self.array_replay_prerequisite_bound,
            self.scf_selected_source_prerequisite_bound,
            self.detached_refinement_archive_loaded,
            self.structured_refinement_mesh_registered,
            self.live_frozen_selected_source_evaluated,
            self.refined_occupations_recomputed,
            self.bilateral_hole_topology_recomputed,
            self.discrete_lifshitz_margin_recomputed,
            self.pocket_refinement_replayed,
        )
        negatives = (
            self.real_vituri_artifact_replayed,
            self.archive_live_computational_independence_verified,
            self.hostile_provider_resistance_verified,
            self.hidden_live_dependency_state_excluded,
            self.refined_scf_executed,
            self.refined_fixed_density_resolved,
            self.continuum_pocket_stability_verified,
            self.refinement_convergence_verified,
            self.global_ground_state_verified,
            self.scientific_execution_verified,
            self.paper_reproduction_verified,
            self.tdhf_readiness_verified,
        )
        if not all(positives) or any(negatives):
            raise ValueError("pocket replay status claims changed")


@dataclass(frozen=True, slots=True)
class Vituri2024PocketRefinementReplayReceipt:
    approval_fingerprint: str
    contract_fingerprint: str
    array_replay_receipt_fingerprint: str
    scf_replay_approval_fingerprint: str
    scf_replay_receipt_fingerprint: str
    prerequisite_provenance_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    source_state_sha256: str
    selected_branch_label: str
    selected_spin: Literal[-1, 1]
    base_hashes: Vituri2024PocketBaseHashes
    base_point_count: int
    refined_point_count: int
    refinement_mesh_sha256: str
    mesh_registration_fingerprint: str
    archive_manifest_sha256: str
    request_fingerprint: str
    archive_field_hashes: Vituri2024PocketRefinementFieldHashes
    live_field_hashes: Vituri2024PocketRefinementFieldHashes
    embedded_base_hashes: Vituri2024PocketRefinementFieldHashes
    archive_live_field_max_abs_residual_ev: float
    archive_topology: tuple[
        Vituri2024RefinedValleyTopologyEvidence,
        Vituri2024RefinedValleyTopologyEvidence,
    ]
    live_topology: tuple[
        Vituri2024RefinedValleyTopologyEvidence,
        Vituri2024RefinedValleyTopologyEvidence,
    ]
    lifshitz_evidence: tuple[
        Vituri2024DiscreteLifshitzEvidence,
        Vituri2024DiscreteLifshitzEvidence,
    ]
    refinement_evidence_sha256: tuple[str, str]
    archive_authority_outer_call_sequence: tuple[str, ...]
    live_provider_outer_call_sequence: tuple[str, ...]
    locked_tolerance_manifest_sha256: str
    status: Vituri2024PocketRefinementReplayStatus
    evidence_model: Literal[
        "trusted_live_selected_source_evaluator_distinct_refinement_archive_object"
    ] = field(default=POCKET_REFINEMENT_EVIDENCE_MODEL, init=False)
    archive_live_computational_independence_verified: bool = field(default=False, init=False)
    hostile_provider_resistance_verified: bool = field(default=False, init=False)
    hidden_live_dependency_state_excluded: bool = field(default=False, init=False)
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _POCKET_SUCCESS_TOKEN:
            raise ValueError("successful pocket replay receipt is factory-only")
        for value, label in (
            (self.approval_fingerprint, "receipt approval"),
            (self.contract_fingerprint, "receipt contract"),
            (self.array_replay_receipt_fingerprint, "receipt array prerequisite"),
            (self.scf_replay_approval_fingerprint, "receipt SCF approval prerequisite"),
            (self.scf_replay_receipt_fingerprint, "receipt SCF prerequisite"),
            (self.prerequisite_provenance_fingerprint, "receipt prerequisite provenance"),
            (self.verifier_module_ast_manifest_sha256, "receipt verifier AST"),
            (self.source_artifact_sha256, "receipt source artifact"),
            (self.spec_fingerprint, "receipt spec"),
            (self.source_state_sha256, "receipt source state"),
            (self.refinement_mesh_sha256, "receipt refinement mesh"),
            (self.mesh_registration_fingerprint, "receipt mesh registration"),
            (self.archive_manifest_sha256, "receipt archive manifest"),
            (self.request_fingerprint, "receipt request"),
            (self.locked_tolerance_manifest_sha256, "receipt tolerance manifest"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "receipt source commit")
        _text(self.selected_branch_label, "receipt selected branch")
        if type(self.selected_spin) is not int or self.selected_spin not in (-1, 1):
            raise ValueError("receipt selected spin must be exactly -1 or +1")
        if type(self.base_hashes) is not Vituri2024PocketBaseHashes:
            raise TypeError("receipt requires typed base hashes")
        base_count = _strict_int(self.base_point_count, "receipt base count")
        refined_count = _strict_int(self.refined_point_count, "receipt refined count")
        if base_count < 1 or refined_count <= base_count:
            raise ValueError("receipt refinement counts are invalid")
        object.__setattr__(self, "base_point_count", base_count)
        object.__setattr__(self, "refined_point_count", refined_count)
        for name in ("archive_field_hashes", "live_field_hashes", "embedded_base_hashes"):
            if type(getattr(self, name)) is not Vituri2024PocketRefinementFieldHashes:
                raise TypeError(f"receipt {name} must be typed field hashes")
        object.__setattr__(
            self,
            "archive_live_field_max_abs_residual_ev",
            _nonnegative(
                self.archive_live_field_max_abs_residual_ev,
                "receipt archive/live field residual",
            ),
        )
        for name, evidence_type in (
            ("archive_topology", Vituri2024RefinedValleyTopologyEvidence),
            ("live_topology", Vituri2024RefinedValleyTopologyEvidence),
            ("lifshitz_evidence", Vituri2024DiscreteLifshitzEvidence),
        ):
            values = tuple(getattr(self, name))
            if len(values) != 2 or tuple(item.valley for item in values) != (-1, 1) or any(type(item) is not evidence_type for item in values):
                raise ValueError(f"receipt {name} must be typed and ordered (-1,+1)")
            object.__setattr__(self, name, values)
        evidence_hashes = tuple(self.refinement_evidence_sha256)
        if len(evidence_hashes) != 2:
            raise ValueError("receipt requires two refinement evidence hashes")
        for digest in evidence_hashes:
            _sha256(digest, "receipt refinement evidence")
        object.__setattr__(self, "refinement_evidence_sha256", evidence_hashes)
        if self.archive_authority_outer_call_sequence != ("load_immutable_pocket_refinement_archive",) or self.live_provider_outer_call_sequence != ("evaluate_frozen_selected_hf_source",):
            raise ValueError("receipt archive/live call ordering changed")
        if type(self.status) is not Vituri2024PocketRefinementReplayStatus:
            raise TypeError("receipt requires factory-created pocket status")
        if self.evidence_model != POCKET_REFINEMENT_EVIDENCE_MODEL or any(
            (
                self.archive_live_computational_independence_verified,
                self.hostile_provider_resistance_verified,
                self.hidden_live_dependency_state_excluded,
            )
        ):
            raise ValueError("receipt trust limitations changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _locked_tolerance_manifest_sha256() -> str:
    return _fingerprint(
        {
            "absolute_ev": POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV,
            "relative": POCKET_REPLAY_V1_RELATIVE_TOLERANCE,
            "mesh_absolute_inverse_angstrom": (
                POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM
            ),
            "mesh_relative": POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE,
            "hermiticity_ev": POCKET_REPLAY_V1_HERMITICITY_TOLERANCE_EV,
            "diagonal_ev": POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV,
            "fock_decomposition_ev": POCKET_REPLAY_V1_FOCK_DECOMPOSITION_TOLERANCE_EV,
        }
    )


def _check_embedded_base(
    fields: Vituri2024ArchivedPocketRefinementFields,
    *,
    mesh: Vituri2024NestedNoWrapRefinementMesh,
    mu: float,
    uncertainty: float,
    base_hashes: Vituri2024PocketBaseHashes,
    label: str,
) -> Vituri2024PocketRefinementFieldHashes:
    indices = mesh.base_embedding_indices
    embedded = Vituri2024ArchivedPocketRefinementFields(
        np.asarray(fields.h0[:, :, indices], dtype=np.complex128),
        np.asarray(fields.interaction_h[:, :, indices], dtype=np.complex128),
        np.asarray(fields.fock[:, :, indices], dtype=np.complex128),
    )
    energies, occupations, projector = _derived_arrays(embedded, mu, uncertainty)
    hashes = _field_hashes(embedded, energies, occupations, projector)
    expected = Vituri2024PocketRefinementFieldHashes(
        h0_sha256=base_hashes.h0_sha256,
        interaction_h_sha256=base_hashes.interaction_h_sha256,
        fock_sha256=base_hashes.fock_sha256,
        energies_sha256=base_hashes.energies_sha256,
        occupations_sha256=base_hashes.occupations_sha256,
        projector_sha256=base_hashes.projector_sha256,
    )
    if hashes != expected:
        raise ValueError(f"{label} embedded base arrays/energies/occupations/hash parity failed")
    return hashes


def _check_base_pockets(
    fields: Vituri2024ArchivedPocketRefinementFields,
    *,
    mesh: Vituri2024NestedNoWrapRefinementMesh,
    mu: float,
    uncertainty: float,
    selected_spin: int,
    receipt: Vituri2024HalfMetalHFReplayReceipt,
    label: str,
) -> None:
    indices = mesh.base_embedding_indices
    base_fields = Vituri2024ArchivedPocketRefinementFields(
        np.asarray(fields.h0[:, :, indices], dtype=np.complex128),
        np.asarray(fields.interaction_h[:, :, indices], dtype=np.complex128),
        np.asarray(fields.fock[:, :, indices], dtype=np.complex128),
    )
    energies, occupations, _ = _derived_arrays(base_fields, mu, uncertainty)
    for valley, expected in zip((-1, 1), receipt.base_pocket_evidence):
        flavor = INTERNAL_FLAVOR_ORDER.index((valley, selected_spin))
        mask = np.asarray(occupations[flavor] == 0, dtype=np.bool_).reshape(mesh.base_shape)
        signature = vituri2024_pocket_topology_signature(mask)
        if (
            signature.hole_component_count != expected.hole_component_count
            or signature.component_cardinalities != expected.component_cardinalities
            or signature.hole_state_count != expected.hole_state_count
        ):
            raise ValueError(f"{label} embedded base pocket evidence mismatch for valley {valley:+d}")


def replay_vituri2024_half_metal_hf_pocket_refinement(
    prerequisites: Vituri2024PocketRefinementPrerequisites,
    archive_authority: Vituri2024PocketRefinementArchiveAuthorityProtocol,
    approval: Vituri2024PocketRefinementReplayApproval,
) -> Vituri2024PocketRefinementReplayReceipt:
    """Replay frozen-source refinement parity; never execute SCF or fix density."""

    if type(prerequisites) is not Vituri2024PocketRefinementPrerequisites:
        raise TypeError("pocket replay requires exact typed prerequisites")
    if type(approval) is not Vituri2024PocketRefinementReplayApproval:
        raise TypeError("pocket replay requires exact detached approval")
    if not isinstance(archive_authority, Vituri2024PocketRefinementArchiveAuthorityProtocol):
        raise TypeError("pocket replay archive authority protocol mismatch")
    # Reconstruct binding, source/spec/state prerequisites, metadata semantics,
    # and every derivable approval field before either delegated method call.
    spec = prerequisites.binding.spec
    provider = prerequisites.binding.provider
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    Vituri2024PocketRefinementPrerequisites(
        prerequisites.binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    if not isinstance(provider, Vituri2024PocketRefinementProviderProtocol):
        raise TypeError("pocket replay live provider protocol mismatch")
    if archive_authority is provider:
        raise ValueError("archive authority and live evaluator provider must be distinct objects")
    if callable(getattr(provider, "load_immutable_pocket_refinement_archive", None)):
        raise ValueError("live provider must not expose the refinement archive loader")
    authority_before = _metadata_snapshot(
        archive_authority,
        POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS,
        "archive authority",
    )
    provider_before = _metadata_snapshot(
        provider, POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS, "live provider"
    )
    loader_before = vituri2024_pocket_callable_manifest(
        "load_immutable_pocket_refinement_archive",
        archive_authority.load_immutable_pocket_refinement_archive,
    )
    evaluator_before = vituri2024_pocket_callable_manifest(
        "evaluate_frozen_selected_hf_source",
        provider.evaluate_frozen_selected_hf_source,
    )
    _validate_archive_authority_semantics(
        prerequisites.binding, authority_before, loader_before
    )
    _validate_live_provider_semantics(
        prerequisites.binding, provider_before, evaluator_before
    )
    provenance = verified_vituri2024_pocket_prerequisite_provenance()
    derived_approval_fields = _derived_pocket_approval_fields(
        prerequisites,
        prerequisite_provenance=provenance,
        authority_snapshot=authority_before,
        loader_manifest=loader_before,
        provider_snapshot=provider_before,
        evaluator_manifest=evaluator_before,
    )
    _validate_pocket_approval_ingress(approval, derived_approval_fields)
    current_ast = str(derived_approval_fields["verifier_module_ast_manifest_sha256"])
    authority_values = dict(authority_before)
    provider_values = dict(provider_before)
    if authority_values["archive_authority_fingerprint"] in (
        provider_values["provider_fingerprint"],
        provider_values["pocket_refinement_provider_fingerprint"],
    ):
        raise ValueError("archive authority and live provider fingerprints must be distinct")
    contract = Vituri2024PocketRefinementReplayContract(
        approval_fingerprint=approval.fingerprint,
        verifier_module_ast_manifest_sha256=current_ast,
        expected_archive_manifest_sha256=approval.expected_archive_manifest_sha256,
        archive_authority_metadata_fingerprint=_fingerprint(authority_before),
        archive_loader_callable_fingerprint=loader_before.fingerprint,
        live_provider_metadata_fingerprint=_fingerprint(provider_before),
        live_evaluator_callable_fingerprint=evaluator_before.fingerprint,
        request_schema_fingerprint=POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT,
        evaluation_schema_fingerprint=POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT,
    )

    # Authority is called first, once.  No live request exists before this return.
    archive_calls: list[str] = []
    archive = archive_authority.load_immutable_pocket_refinement_archive(
        approval.source_artifact_sha256
    )
    archive_calls.append("load_immutable_pocket_refinement_archive")
    if type(archive) is not Vituri2024ImmutablePocketRefinementArchive:
        raise TypeError("archive authority must return the exact immutable archive type")
    authority_after = _metadata_snapshot(
        archive_authority,
        POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS,
        "archive authority",
    )
    loader_after = vituri2024_pocket_callable_manifest(
        "load_immutable_pocket_refinement_archive",
        archive_authority.load_immutable_pocket_refinement_archive,
    )
    provider_after_archive = _metadata_snapshot(
        provider, POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS, "live provider"
    )
    evaluator_after_archive = vituri2024_pocket_callable_manifest(
        "evaluate_frozen_selected_hf_source",
        provider.evaluate_frozen_selected_hf_source,
    )
    if authority_after != authority_before or loader_after != loader_before:
        raise ValueError("archive authority metadata/callable mutated during load")
    if (
        provider_after_archive != provider_before
        or evaluator_after_archive != evaluator_before
    ):
        raise ValueError("live provider metadata/callable mutated during archive load")
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    Vituri2024PocketRefinementPrerequisites(
        prerequisites.binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    _validate_archive_authority_semantics(
        prerequisites.binding, authority_after, loader_after
    )
    _validate_live_provider_semantics(
        prerequisites.binding, provider_after_archive, evaluator_after_archive
    )
    _validate_pocket_approval_ingress(
        approval,
        _derived_pocket_approval_fields(
            prerequisites,
            prerequisite_provenance=provenance,
            authority_snapshot=authority_after,
            loader_manifest=loader_after,
            provider_snapshot=provider_after_archive,
            evaluator_manifest=evaluator_after_archive,
        ),
    )
    # The sole detached scientific value closes immediately on the loaded object.
    archive_manifest = pocket_refinement_archive_manifest_sha256(archive)
    if archive_manifest != approval.expected_archive_manifest_sha256:
        raise ValueError("detached pocket archive manifest mismatch")
    assert spec.geometry is not None and spec.attested_source is not None
    source = spec.attested_source
    archive_identity = (
        (archive.archive_authority_fingerprint, authority_values["archive_authority_fingerprint"], "authority"),
        (archive.source_commit, source.source_commit, "source commit"),
        (archive.source_artifact_sha256, source.source_artifact_sha256, "source artifact"),
        (archive.spec_fingerprint, spec.fingerprint, "spec"),
        (archive.source_state_sha256, source.source_state_sha256, "source state"),
        (archive.selected_branch_label, source.selected_branch_label, "selected branch"),
        (archive.selected_spin, source.selected_spin, "selected spin"),
        (archive.chemical_potential_ev, source.chemical_potential_ev, "chemical potential"),
        (archive.archive_loader_implementation_fingerprint, authority_values["archive_loader_implementation_fingerprint"], "loader implementation"),
        (archive.archive_schema_fingerprint, POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT, "schema"),
    )
    for actual, expected, label in archive_identity:
        if actual != expected:
            raise ValueError(f"pocket archive {label} mismatch")
    mesh = archive.mesh
    if mesh.base_shape != spec.geometry.mesh_shape:
        raise ValueError("refinement base shape does not equal preflight geometry")
    if canonical_array_sha256(mesh.base_mesh) != approval.base_hashes.ordered_momentum_mesh_sha256:
        raise ValueError("archive base mesh hash does not equal factory array replay")
    refined_mesh_hash = canonical_array_sha256(mesh.refined_mesh)
    if refined_mesh_hash != approval.preflight_refinement_mesh_sha256 or mesh.refined_mesh.shape[0] != approval.preflight_refinement_point_count:
        raise ValueError("registered refinement mesh hash/count does not equal both preflight pockets")
    eta_max = max(approval.preflight_lifshitz_uncertainties_ev)
    _validate_refined_fields(archive.fields, mesh.refined_mesh.shape[0], "archive")
    archive_embedded_hashes = _check_embedded_base(
        archive.fields,
        mesh=mesh,
        mu=source.chemical_potential_ev,
        uncertainty=eta_max,
        base_hashes=approval.base_hashes,
        label="archive",
    )
    _check_base_pockets(
        archive.fields,
        mesh=mesh,
        mu=source.chemical_potential_ev,
        uncertainty=eta_max,
        selected_spin=source.selected_spin,
        receipt=prerequisites.array_replay_receipt,
        label="archive",
    )

    # Reconstruct a separately copied request containing source/mesh identities only.
    request_mesh = Vituri2024NestedNoWrapRefinementMesh(
        base_shape=mesh.base_shape,
        subdivision_factors=mesh.subdivision_factors,
        base_mesh=np.asarray(mesh.base_mesh, dtype=np.float64),
        refined_mesh=np.asarray(mesh.refined_mesh, dtype=np.float64),
    )
    request = Vituri2024FrozenHFRefinementRequest(
        provider_fingerprint=source.provider_fingerprint,
        source_commit=source.source_commit,
        source_artifact_sha256=source.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=source.source_state_sha256,
        selected_branch_label=source.selected_branch_label,
        selected_spin=source.selected_spin,
        chemical_potential_ev=source.chemical_potential_ev,
        base_hashes=approval.base_hashes,
        mesh=request_mesh,
    )
    if any(
        np.shares_memory(request_array, archive_array)
        for request_array in (request.mesh.base_mesh, request.mesh.refined_mesh)
        for archive_array in (
            archive.mesh.base_mesh,
            archive.mesh.refined_mesh,
            archive.fields.h0,
            archive.fields.interaction_h,
            archive.fields.fock,
        )
    ):
        raise ValueError("live request shares memory with detached archive")
    live_calls: list[str] = []
    result = provider.evaluate_frozen_selected_hf_source(request)
    live_calls.append("evaluate_frozen_selected_hf_source")
    if type(result) is not Vituri2024FrozenHFRefinementEvaluation:
        raise TypeError("live evaluator must return only the exact typed H0/interaction/Fock result")
    provider_after = _metadata_snapshot(
        provider, POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS, "live provider"
    )
    evaluator_after = vituri2024_pocket_callable_manifest(
        "evaluate_frozen_selected_hf_source",
        provider.evaluate_frozen_selected_hf_source,
    )
    authority_after_evaluation = _metadata_snapshot(
        archive_authority,
        POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS,
        "archive authority",
    )
    loader_after_evaluation = vituri2024_pocket_callable_manifest(
        "load_immutable_pocket_refinement_archive",
        archive_authority.load_immutable_pocket_refinement_archive,
    )
    if provider_after != provider_before or evaluator_after != evaluator_before:
        raise ValueError("live provider metadata/callable mutated during evaluation")
    if (
        authority_after_evaluation != authority_before
        or loader_after_evaluation != loader_before
    ):
        raise ValueError("archive authority metadata/callable mutated during evaluation")
    Vituri2024HalfMetalHFProviderBinding(spec, provider)
    Vituri2024PocketRefinementPrerequisites(
        prerequisites.binding,
        prerequisites.array_replay_receipt,
        prerequisites.scf_replay_approval,
        prerequisites.scf_replay_receipt,
    )
    _validate_live_provider_semantics(
        prerequisites.binding, provider_after, evaluator_after
    )
    _validate_archive_authority_semantics(
        prerequisites.binding, authority_after_evaluation, loader_after_evaluation
    )
    _validate_pocket_approval_ingress(
        approval,
        _derived_pocket_approval_fields(
            prerequisites,
            prerequisite_provenance=provenance,
            authority_snapshot=authority_after_evaluation,
            loader_manifest=loader_after_evaluation,
            provider_snapshot=provider_after,
            evaluator_manifest=evaluator_after,
        ),
    )
    result_identity = (
        (result.pocket_refinement_provider_fingerprint, provider_values["pocket_refinement_provider_fingerprint"], "provider"),
        (result.evaluator_implementation_fingerprint, provider_values["refinement_evaluator_implementation_fingerprint"], "evaluator implementation"),
        (result.evaluation_schema_fingerprint, provider_values["refinement_evaluation_schema_fingerprint"], "evaluation schema"),
        (result.request_fingerprint, request.fingerprint, "request"),
        (result.source_commit, source.source_commit, "source commit"),
        (result.source_artifact_sha256, source.source_artifact_sha256, "source artifact"),
        (result.spec_fingerprint, spec.fingerprint, "spec"),
        (result.source_state_sha256, source.source_state_sha256, "source state"),
        (result.selected_branch_label, source.selected_branch_label, "selected branch"),
    )
    for actual, expected, label in result_identity:
        if actual != expected:
            raise ValueError(f"live refinement result {label} mismatch")
    live_fields = Vituri2024ArchivedPocketRefinementFields(
        result.h0, result.interaction_h, result.fock
    )
    if any(
        np.shares_memory(live_array, other)
        for live_array in (live_fields.h0, live_fields.interaction_h, live_fields.fock)
        for other in (
            archive.fields.h0,
            archive.fields.interaction_h,
            archive.fields.fock,
            request.mesh.base_mesh,
            request.mesh.refined_mesh,
        )
    ):
        raise ValueError("live result shares storage with archive/request arrays")
    _validate_refined_fields(live_fields, mesh.refined_mesh.shape[0], "live")
    live_embedded_hashes = _check_embedded_base(
        live_fields,
        mesh=mesh,
        mu=source.chemical_potential_ev,
        uncertainty=eta_max,
        base_hashes=approval.base_hashes,
        label="live",
    )
    if live_embedded_hashes != archive_embedded_hashes:
        raise ValueError("archive/live embedded base hashes differ")
    _check_base_pockets(
        live_fields,
        mesh=mesh,
        mu=source.chemical_potential_ev,
        uncertainty=eta_max,
        selected_spin=source.selected_spin,
        receipt=prerequisites.array_replay_receipt,
        label="live",
    )
    field_residual = max(
        _require_close(result_array, archive_array, f"archive/live refined {name}")
        for name, result_array, archive_array in (
            ("h0", live_fields.h0, archive.fields.h0),
            ("interaction_h", live_fields.interaction_h, archive.fields.interaction_h),
            ("fock", live_fields.fock, archive.fields.fock),
        )
    )
    archive_energies, archive_occupations, archive_projector = _derived_arrays(
        archive.fields, source.chemical_potential_ev, eta_max
    )
    live_energies, live_occupations, live_projector = _derived_arrays(
        live_fields, source.chemical_potential_ev, eta_max
    )
    opposite_indices = tuple(
        index
        for index, (_, spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if spin != source.selected_spin
    )
    if np.any(archive_occupations[np.asarray(opposite_indices)] == 0) or np.any(
        live_occupations[np.asarray(opposite_indices)] == 0
    ):
        raise ValueError("opposite-spin refined holes are forbidden")
    if not np.any(archive_occupations == 1) or not np.any(archive_occupations == 0):
        raise ValueError("refined selected source must contain occupied and unoccupied points")

    archive_topologies: list[Vituri2024RefinedValleyTopologyEvidence] = []
    live_topologies: list[Vituri2024RefinedValleyTopologyEvidence] = []
    lifshitz_evidence: list[Vituri2024DiscreteLifshitzEvidence] = []
    evidence_hashes: list[str] = []
    archive_hashes_final: Vituri2024PocketRefinementFieldHashes | None = None
    live_hashes_final: Vituri2024PocketRefinementFieldHashes | None = None
    for index, (valley, pocket) in enumerate(zip((-1, 1), source.pocket_evidence)):
        (
            evidence_hash,
            archive_topology,
            live_topology,
            lifshitz,
            archive_hashes,
            live_hashes,
        ) = vituri2024_refinement_evidence_sha256(
            valley=valley,  # type: ignore[arg-type]
            selected_spin=source.selected_spin,
            source_commit=source.source_commit,
            source_artifact_sha256=source.source_artifact_sha256,
            source_state_sha256=source.source_state_sha256,
            selected_branch_label=source.selected_branch_label,
            base_hashes=approval.base_hashes,
            mesh=mesh,
            archive_fields=archive.fields,
            live_fields=live_fields,
            chemical_potential_ev=source.chemical_potential_ev,
            locked_threshold_uncertainty_ev=pocket.lifshitz_tolerance_ev,
        )
        if evidence_hash != pocket.refinement_evidence_sha256 or evidence_hash != approval.preflight_refinement_evidence_sha256[index]:
            raise ValueError(f"valley {valley:+d} verifier-defined refinement evidence hash mismatch")
        _require_close(
            lifshitz.archive.raw_margin_ev,
            pocket.lifshitz_margin_ev,
            f"valley {valley:+d} preflight raw Lifshitz margin",
        )
        archive_topologies.append(archive_topology)
        live_topologies.append(live_topology)
        lifshitz_evidence.append(lifshitz)
        evidence_hashes.append(evidence_hash)
        archive_hashes_final = archive_hashes
        live_hashes_final = live_hashes
    assert archive_hashes_final is not None and live_hashes_final is not None
    status = Vituri2024PocketRefinementReplayStatus(_factory_token=_POCKET_SUCCESS_TOKEN)
    return Vituri2024PocketRefinementReplayReceipt(
        approval_fingerprint=approval.fingerprint,
        contract_fingerprint=contract.fingerprint,
        array_replay_receipt_fingerprint=prerequisites.array_receipt_fingerprint,
        scf_replay_approval_fingerprint=prerequisites.scf_approval_fingerprint,
        scf_replay_receipt_fingerprint=prerequisites.scf_receipt_fingerprint,
        prerequisite_provenance_fingerprint=provenance.fingerprint,
        verifier_module_ast_manifest_sha256=current_ast,
        source_commit=source.source_commit,
        source_artifact_sha256=source.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        source_state_sha256=source.source_state_sha256,
        selected_branch_label=source.selected_branch_label,
        selected_spin=source.selected_spin,
        base_hashes=approval.base_hashes,
        base_point_count=mesh.base_mesh.shape[0],
        refined_point_count=mesh.refined_mesh.shape[0],
        refinement_mesh_sha256=refined_mesh_hash,
        mesh_registration_fingerprint=mesh.fingerprint,
        archive_manifest_sha256=archive_manifest,
        request_fingerprint=request.fingerprint,
        archive_field_hashes=archive_hashes_final,
        live_field_hashes=live_hashes_final,
        embedded_base_hashes=archive_embedded_hashes,
        archive_live_field_max_abs_residual_ev=field_residual,
        archive_topology=(archive_topologies[0], archive_topologies[1]),
        live_topology=(live_topologies[0], live_topologies[1]),
        lifshitz_evidence=(lifshitz_evidence[0], lifshitz_evidence[1]),
        refinement_evidence_sha256=(evidence_hashes[0], evidence_hashes[1]),
        archive_authority_outer_call_sequence=tuple(archive_calls),
        live_provider_outer_call_sequence=tuple(live_calls),
        locked_tolerance_manifest_sha256=_locked_tolerance_manifest_sha256(),
        status=status,
        _factory_token=_POCKET_SUCCESS_TOKEN,
    )


__all__ = [
    "POCKET_REFINEMENT_ARCHIVE_AUTHORITY_METADATA_FIELDS",
    "POCKET_REFINEMENT_ARCHIVE_GENERATION_PHASE",
    "POCKET_REFINEMENT_ARCHIVE_SCHEMA_FINGERPRINT",
    "POCKET_REFINEMENT_EVALUATION_SCHEMA_FINGERPRINT",
    "POCKET_REFINEMENT_EVIDENCE_MODEL",
    "POCKET_REFINEMENT_LIFSHITZ_CONVENTION",
    "POCKET_REFINEMENT_PROVIDER_METADATA_FIELDS",
    "POCKET_REFINEMENT_REQUEST_SCHEMA_FINGERPRINT",
    "POCKET_REFINEMENT_TOPOLOGY_CONVENTION",
    "POCKET_REPLAY_V1_ABSOLUTE_TOLERANCE_EV",
    "POCKET_REPLAY_V1_DIAGONAL_TOLERANCE_EV",
    "POCKET_REPLAY_V1_FOCK_DECOMPOSITION_TOLERANCE_EV",
    "POCKET_REPLAY_V1_HERMITICITY_TOLERANCE_EV",
    "POCKET_REPLAY_V1_MESH_ABSOLUTE_TOLERANCE_INVERSE_ANGSTROM",
    "POCKET_REPLAY_V1_MESH_RELATIVE_TOLERANCE",
    "POCKET_REPLAY_V1_RELATIVE_TOLERANCE",
    "VITURI2024_POCKET_REFINEMENT_REPLAY_SCOPE",
    "VITURI2024_POCKET_REPLAY_PREREQUISITE_BASELINE_COMMIT",
    "Vituri2024ArchivedPocketRefinementFields",
    "Vituri2024DiscreteLifshitzEvidence",
    "Vituri2024DiscreteLifshitzLaneEvidence",
    "Vituri2024FrozenHFRefinementEvaluation",
    "Vituri2024FrozenHFRefinementRequest",
    "Vituri2024ImmutablePocketRefinementArchive",
    "Vituri2024NestedNoWrapRefinementMesh",
    "Vituri2024PocketBaseHashes",
    "Vituri2024PocketCallableManifest",
    "Vituri2024PocketPrerequisiteProvenance",
    "Vituri2024PocketPrerequisiteSourceManifest",
    "Vituri2024PocketRefinementArchiveAuthorityProtocol",
    "Vituri2024PocketRefinementFieldHashes",
    "Vituri2024PocketRefinementPrerequisites",
    "Vituri2024PocketRefinementProviderProtocol",
    "Vituri2024PocketRefinementReplayApproval",
    "Vituri2024PocketRefinementReplayContract",
    "Vituri2024PocketRefinementReplayReceipt",
    "Vituri2024PocketRefinementReplayStatus",
    "Vituri2024PocketTopologySignature",
    "Vituri2024RefinedValleyTopologyEvidence",
    "canonical_half_metal_hf_replay_receipt_fingerprint",
    "canonical_scf_replay_receipt_fingerprint",
    "make_vituri2024_pocket_refinement_replay_approval",
    "pocket_refinement_archive_authority_fingerprint",
    "pocket_refinement_archive_manifest_sha256",
    "pocket_refinement_provider_fingerprint",
    "pocket_refinement_replay_module_ast_manifest_sha256",
    "replay_vituri2024_half_metal_hf_pocket_refinement",
    "verified_vituri2024_pocket_prerequisite_provenance",
    "vituri2024_pocket_callable_manifest",
    "vituri2024_pocket_topology_signature",
    "vituri2024_refinement_evidence_sha256",
]
