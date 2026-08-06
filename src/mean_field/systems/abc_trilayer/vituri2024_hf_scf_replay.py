"""Uninterrupted, detached-archive SCF replay for the Vituri-2024 HF source.

The verifier executes every attested seed through the baseline generic
:class:`~mean_field.core.hf.HartreeFockProblem` entrypoint.  It does not add a
second SCF loop.  A successful receipt records deterministic trajectory and
selected-source parity under a trusted-live-provider model.  Distinct authority
objects and loading a detached immutable archive before live builders prevent
accidental same-object coupling; they do not prove archive/live data
independence.  A trusted same-class/same-code provider could still read or copy
archive data into unmanifested instance state.  Hostile-provider resistance and
independent pinning of live-builder dependency state therefore remain explicitly
false.  The current generic core also has no continuation API and keeps
``cached_interaction_h`` local to the loop, so restart/checkpoint equivalence is
unverified.
"""
from __future__ import annotations

import ast
from dataclasses import InitVar, asdict, dataclass, field, fields
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import marshal
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import stat
import subprocess
import sys
import textwrap
from types import CodeType, MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ...core.hf import engine as _hf_engine
from ...core.hf import occupations as _hf_occupations
from ...core.hf import problem as _hf_problem
from ...core.hf.engine import DensityUpdateResult, HartreeFockStepResult
from ...core.hf.problem import HartreeFockKernel, HartreeFockProblem
from .vituri2024_hf_functional_replay import (
    FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS,
    Vituri2024FunctionalReplayProviderProtocol,
)
from .vituri2024_hf_preflight import (
    Vituri2024BranchEnergyReceipt,
    Vituri2024HalfMetalHFProviderBinding,
    Vituri2024SCFSeedReceipt,
)
from .vituri2024_hf_replay import canonical_array_sha256

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

VITURI2024_SCF_REPLAY_SCOPE = (
    "vituri2024_uninterrupted_registered_seed_trajectory_replay_v1"
)
VITURI2024_SCF_BASELINE_COMMIT = "0f7d9b9190001d2bdb6f6ec8f6e36a16864667dc"
_CORE_PROVENANCE_GIT_MODE = "git_ancestor_head_index_worktree_verified"
_CORE_PROVENANCE_SOURCE_EXPORT_MODE = "pinned_hash_verified_source_export"
_CORE_BASELINE_COMMIT_AUTHORITY = (
    "hardcoded_vituri2024_scf_baseline_commit_and_core_hash_manifest"
)
SCF_REPLAY_ARCHIVE_SCHEMA_LABEL = "vituri2024_immutable_historical_scf_archive_v1"
SCF_REPLAY_ADAPTER_ABI_LABEL = "vituri2024_live_hf_problem_adapter_abi_v1"
SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_LABEL = "vituri2024_scf_archive_authority_abi_v1"
SCF_REPLAY_VERIFIER_SCHEMA_LABEL = "vituri2024_uninterrupted_scf_verifier_v1"
SCF_REPLAY_ARCHIVE_GENERATION_PHASE = "detached_before_state_and_problem_builders"
_SCF_REPLAY_EVIDENCE_MODEL = "trusted_live_provider_distinct_archive_object"
SCF_REPLAY_SELECTED_HASH_FIELDS: tuple[str, ...] = (
    "h0",
    "effective_interaction_h",
    "fock",
    "projector",
    "energies",
)

_CORE_SOURCE_EXPECTATIONS: MappingProxyType[str, tuple[str, str]] = MappingProxyType(
    {
        "src/mean_field/core/hf/problem.py": (
            "d118cdd8ff5a086043688813caa869ef5eb08df8a421edd297bdb3962051da9d",
            "ffd2680dab3b2907f5fafaeb0ee6f73d37e41d23c3e7351012bdc6e39296bd01",
        ),
        "src/mean_field/core/hf/engine.py": (
            "147a9bbec5d3269348b5d99c7428affffaa620792a59cbcece446eaafba1596e",
            "a2ed4453ebd841b81431ec1d4b505db37c6a4bb026c2b7d30502f9cf124d3e6b",
        ),
        "src/mean_field/core/hf/occupations.py": (
            "9a2036c700ba6b2992e4af507e175d34aca01c3669d82ed4f2b8040d1b15a2d6",
            "21143b44a268c4949314924e3fcce75cfce42535e6b92121abb6759485ee815c",
        ),
    }
)
_CORE_CALLABLE_EXPECTATIONS: MappingProxyType[
    str, tuple[str, str, str, str]
] = MappingProxyType(
    {
        "run_hartree_fock_problem": (
            "mean_field.core.hf.problem",
            "run_hartree_fock_problem",
            "(state: 'HartreeFockStateProtocol', problem: 'HartreeFockProblem', "
            "*, init_mode: 'str', seed: 'int', max_iter: 'int' = 300, "
            "oda_stall_threshold: 'float' = 0.001, max_oda_lambda: "
            "'float | None' = None) -> 'HartreeFockRun'",
            "e82974e4c3bc410c48009c0f39b2b9347ecd5e8670ca0377ee428e680862ea28",
        ),
        "run_hartree_fock_iterations": (
            "mean_field.core.hf.engine",
            "run_hartree_fock_iterations",
            "(state: 'HartreeFockStateProtocol', *, init_mode: 'str', seed: 'int', "
            "interaction_builder: 'Callable[[np.ndarray], np.ndarray]', "
            "density_builder: 'Callable[[np.ndarray], DensityUpdateResult]', "
            "energy_functional: 'Callable[[np.ndarray, np.ndarray, np.ndarray], "
            "float]', oda_parameterizer: 'Callable[[HartreeFockStateProtocol, "
            "np.ndarray], float] | None' = None, oda_delta_interaction_builder: "
            "'Callable[[np.ndarray], np.ndarray] | None' = None, "
            "hamiltonian_postprocessor: 'Callable[[np.ndarray], None] | None' = None, "
            "density_postprocessor: 'Callable[[np.ndarray], None] | None' = None, "
            "step_callback: 'Callable[[HartreeFockStateProtocol, "
            "HartreeFockStepResult], None] | None' = None, final_state_callback: "
            "'Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None' "
            "= None, convergence_rule: \"Literal['raw', 'mixed']\" = 'raw', "
            "max_iter: 'int' = 300, oda_stall_threshold: 'float' = 0.001, "
            "max_oda_lambda: 'float | None' = None) -> 'HartreeFockRun'",
            "1987404d940fc1adac7dfe5c412b37f9ba089fec68296076c543adef93d687a6",
        ),
        "compute_oda_parameter": (
            "mean_field.core.hf.engine",
            "compute_oda_parameter",
            "(state: 'HartreeFockStateProtocol', delta_density: 'np.ndarray', *, "
            "interaction_builder: 'Callable[[np.ndarray], np.ndarray] | None' = "
            "None, delta_h: 'np.ndarray | None' = None, interaction_h: "
            "'np.ndarray | None' = None) -> 'float'",
            "abd86012c1436d1e4c83b86b91ea4333f6a1046598fbf0e0dd57c774874a5b0a",
        ),
        "calculate_norm_convergence": (
            "mean_field.core.hf.occupations",
            "calculate_norm_convergence",
            "(updated_density: 'np.ndarray', previous_density: 'np.ndarray') -> 'float'",
            "e292c1e42662eac146ff8ce91a94121ea602789d62b3da3d1278db7909eb4307",
        ),
    }
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
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if value.dtype != dtype:
        raise TypeError(f"{label} dtype must be exactly {dtype.name}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{label} shape must be exactly {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain only finite values")
    result = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(value.shape)
    result.flags.writeable = False
    return result


def _immutable_complex(value: object, label: str, shape: tuple[int, ...] | None = None) -> ComplexArray:
    return _immutable_array(
        value, label=label, dtype=np.dtype(np.complex128), ndim=3, shape=shape
    )  # type: ignore[return-value]


def _immutable_float(value: object, label: str, shape: tuple[int, ...] | None = None) -> FloatArray:
    return _immutable_array(
        value, label=label, dtype=np.dtype(np.float64), ndim=2, shape=shape
    )  # type: ignore[return-value]


def _canonical_ast_sha256(source: str, filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    return hashlib.sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _assert_core_source_matches(
    raw: bytes,
    relative_path: str,
    source_label: str,
) -> tuple[str, str]:
    expected_bytes, expected_ast = _CORE_SOURCE_EXPECTATIONS[relative_path]
    actual_bytes = hashlib.sha256(raw).hexdigest()
    try:
        actual_ast = _canonical_ast_sha256(raw.decode("utf-8"), relative_path)
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise RuntimeError(
            f"SCF replay {source_label} core source is not valid Python: {relative_path}"
        ) from error
    if actual_bytes != expected_bytes or actual_ast != expected_ast:
        raise RuntimeError(
            f"SCF replay {source_label} core source manifest mismatch: {relative_path}"
        )
    return actual_bytes, actual_ast

def scf_replay_module_ast_manifest_sha256(source: str | None = None) -> str:
    module_source = Path(__file__).read_text(encoding="utf-8") if source is None else source
    if not isinstance(module_source, str):
        raise TypeError("SCF replay module source must be text")
    return _canonical_ast_sha256(module_source, "vituri2024_hf_scf_replay.py")


def _callable_node_sha256(callback: object) -> str:
    source = textwrap.dedent(inspect.getsource(callback))
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(nodes) != 1:
        raise RuntimeError("callable source did not contain exactly one function")
    return hashlib.sha256(
        ast.dump(nodes[0], annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _callable_code_sha256(callback: object) -> str:
    target = callback.__func__ if inspect.ismethod(callback) else callback
    code = getattr(target, "__code__", None)
    if not isinstance(code, CodeType):
        raise RuntimeError("callable has no inspectable Python code object")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


@dataclass(frozen=True, slots=True)
class Vituri2024CoreSourceManifest:
    relative_path: str
    source_bytes_sha256: str
    canonical_ast_sha256: str

    def __post_init__(self) -> None:
        _text(self.relative_path, "core source path")
        _sha256(self.source_bytes_sha256, "core source bytes")
        _sha256(self.canonical_ast_sha256, "core source AST")


@dataclass(frozen=True, slots=True)
class Vituri2024CoreCallableIdentity:
    symbol: str
    module: str
    qualname: str
    signature: str
    canonical_function_ast_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.symbol, "core callable symbol"),
            (self.module, "core callable module"),
            (self.qualname, "core callable qualname"),
            (self.signature, "core callable signature"),
        ):
            _text(value, label)
        _sha256(self.canonical_function_ast_sha256, "core callable AST")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFCallableManifest:
    role: str
    module: str
    qualname: str
    signature: str
    source_sha256: str
    canonical_function_ast_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.role, "SCF callable role"),
            (self.module, "SCF callable module"),
            (self.qualname, "SCF callable qualname"),
            (self.signature, "SCF callable signature"),
        ):
            _text(value, label)
        _sha256(self.source_sha256, "SCF callable source")
        _sha256(self.canonical_function_ast_sha256, "SCF callable AST")
        _sha256(self.code_sha256, "SCF callable code")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFProblemCallbackManifest:
    role: str
    implementation_kind: Literal["callable", "none"]
    callable_manifest: Vituri2024SCFCallableManifest | None

    def __post_init__(self) -> None:
        _text(self.role, "SCF problem callback role")
        if self.implementation_kind == "none":
            if self.callable_manifest is not None:
                raise ValueError("None callback manifest cannot carry a callable identity")
        elif self.implementation_kind == "callable":
            if type(self.callable_manifest) is not Vituri2024SCFCallableManifest:
                raise TypeError("callable callback manifest requires an exact callable identity")
            if self.callable_manifest.role != self.role:
                raise ValueError("callback manifest role mismatch")
        else:
            raise ValueError("unsupported callback implementation kind")


def vituri2024_scf_callable_manifest(
    role: str, callback: object
) -> Vituri2024SCFCallableManifest:
    """Derive a fail-closed source/AST/code identity for one Python callable."""

    if not callable(callback):
        raise TypeError(f"{role} must be callable")
    target = callback.__func__ if inspect.ismethod(callback) else callback
    try:
        source = textwrap.dedent(inspect.getsource(target))
        source.encode("utf-8")
        ast_digest = _callable_node_sha256(target)
        code_digest = _callable_code_sha256(target)
        signature = str(inspect.signature(callback))
    except (OSError, TypeError, UnicodeError, SyntaxError, IndentationError, RuntimeError) as error:
        raise RuntimeError(f"SCF callable source/code is not inspectable: {role}") from error
    return Vituri2024SCFCallableManifest(
        role=role,
        module=getattr(target, "__module__", ""),
        qualname=getattr(target, "__qualname__", ""),
        signature=signature,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        canonical_function_ast_sha256=ast_digest,
        code_sha256=code_digest,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024CoreProvenance:
    provenance_mode: Literal[
        "git_ancestor_head_index_worktree_verified",
        "pinned_hash_verified_source_export",
    ]
    baseline_commit: str
    baseline_commit_authority: Literal[
        "hardcoded_vituri2024_scf_baseline_commit_and_core_hash_manifest"
    ]
    repository_checks_available: bool
    repository_ancestry_verified: bool
    repository_head_core_verified: bool
    repository_index_core_verified: bool
    repository_worktree_core_verified: bool
    source_manifests: tuple[Vituri2024CoreSourceManifest, ...]
    callable_identities: tuple[Vituri2024CoreCallableIdentity, ...]
    package_version: str
    python_version: str
    python_implementation: str
    numpy_version: str

    def __post_init__(self) -> None:
        if self.provenance_mode not in (
            _CORE_PROVENANCE_GIT_MODE,
            _CORE_PROVENANCE_SOURCE_EXPORT_MODE,
        ):
            raise ValueError("unsupported SCF core provenance mode")
        if _commit(self.baseline_commit, "core baseline commit") != VITURI2024_SCF_BASELINE_COMMIT:
            raise ValueError("SCF replay baseline commit changed")
        if self.baseline_commit_authority != _CORE_BASELINE_COMMIT_AUTHORITY:
            raise ValueError("SCF replay baseline commit authority changed")
        repository_flags = (
            self.repository_checks_available,
            self.repository_ancestry_verified,
            self.repository_head_core_verified,
            self.repository_index_core_verified,
            self.repository_worktree_core_verified,
        )
        if any(type(value) is not bool for value in repository_flags):
            raise TypeError("SCF repository provenance flags must be exact booleans")
        expected_repository_flag = self.provenance_mode == _CORE_PROVENANCE_GIT_MODE
        if any(value is not expected_repository_flag for value in repository_flags):
            raise ValueError("SCF repository provenance flags do not match mode")
        if type(self.source_manifests) is not tuple or len(self.source_manifests) != 3:
            raise TypeError("core provenance requires three ordered source manifests")
        if type(self.callable_identities) is not tuple or len(self.callable_identities) != 4:
            raise TypeError("core provenance requires four ordered callable identities")
        for value, label in (
            (self.package_version, "package version"),
            (self.python_version, "Python version"),
            (self.python_implementation, "Python implementation"),
            (self.numpy_version, "NumPy version"),
        ):
            _text(value, label)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _package_version() -> str:
    try:
        return importlib_metadata.version("mean-field")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0+source-tree"


def _repository_metadata_available(root: Path) -> bool:
    metadata_path = root / ".git"
    try:
        mode = metadata_path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("SCF replay could not inspect repository metadata") from error
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise RuntimeError("SCF replay repository metadata has unsupported file type")
    return True

def _verify_git_core_baseline(root: Path) -> None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _commit(head, "SCF replay HEAD")
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                VITURI2024_SCF_BASELINE_COMMIT,
                head,
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError(
            "SCF replay requires an inspectable git HEAD and bound baseline commit"
        ) from error
    if ancestry.returncode == 1:
        raise RuntimeError(
            "SCF replay bound baseline commit is not an ancestor of HEAD"
        )
    if ancestry.returncode != 0:
        raise RuntimeError("SCF replay could not verify bound baseline ancestry")

    for relative_path in _CORE_SOURCE_EXPECTATIONS:
        git_sources = (
            (
                "baseline",
                f"{VITURI2024_SCF_BASELINE_COMMIT}:{relative_path}",
            ),
            ("HEAD", f"{head}:{relative_path}"),
            ("index", f":{relative_path}"),
        )
        for source_label, object_spec in git_sources:
            try:
                raw = subprocess.run(
                    ["git", "show", object_spec],
                    cwd=root,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as error:
                raise RuntimeError(
                    f"SCF replay could not inspect {source_label} core source: "
                    f"{relative_path}"
                ) from error
            _assert_core_source_matches(raw, relative_path, source_label)
        try:
            worktree_raw = (root / relative_path).read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"SCF replay could not inspect working-tree core source: {relative_path}"
            ) from error
        _assert_core_source_matches(worktree_raw, relative_path, "working-tree")


def verified_vituri2024_core_provenance() -> Vituri2024CoreProvenance:
    """Fail closed on baseline source, runtime identity, or monkeypatch drift."""

    root = _repository_root()
    repository_checks_available = _repository_metadata_available(root)
    if repository_checks_available:
        _verify_git_core_baseline(root)
        provenance_mode = _CORE_PROVENANCE_GIT_MODE
        local_source_label = "working-tree"
    else:
        provenance_mode = _CORE_PROVENANCE_SOURCE_EXPORT_MODE
        local_source_label = "source-export"
    source_manifests: list[Vituri2024CoreSourceManifest] = []
    for relative_path in _CORE_SOURCE_EXPECTATIONS:
        path = root / relative_path
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"SCF replay could not inspect {local_source_label} core source: "
                f"{relative_path}"
            ) from error
        actual_bytes, actual_ast = _assert_core_source_matches(
            raw, relative_path, local_source_label
        )
        source_manifests.append(
            Vituri2024CoreSourceManifest(relative_path, actual_bytes, actual_ast)
        )

    if (
        _hf_problem.run_hartree_fock_iterations
        is not _hf_engine.run_hartree_fock_iterations
    ):
        raise RuntimeError(
            "core problem-module SCF iteration alias runtime identity was monkeypatched"
        )
    if _hf_engine.calculate_norm_convergence is not _hf_occupations.calculate_norm_convergence:
        raise RuntimeError("core convergence metric runtime identity was monkeypatched")
    runtime = (
        ("run_hartree_fock_problem", _hf_problem.run_hartree_fock_problem),
        ("run_hartree_fock_iterations", _hf_problem.run_hartree_fock_iterations),
        ("compute_oda_parameter", _hf_engine.compute_oda_parameter),
        ("calculate_norm_convergence", _hf_occupations.calculate_norm_convergence),
    )
    callable_identities: list[Vituri2024CoreCallableIdentity] = []
    for symbol, callback in runtime:
        (
            expected_module,
            expected_qualname,
            expected_signature,
            expected_ast,
        ) = _CORE_CALLABLE_EXPECTATIONS[symbol]
        actual_module = getattr(callback, "__module__", "")
        actual_qualname = getattr(callback, "__qualname__", "")
        try:
            actual_signature = str(inspect.signature(callback))
            actual_ast = _callable_node_sha256(callback)
        except (OSError, TypeError, SyntaxError, IndentationError, RuntimeError) as error:
            raise RuntimeError(f"core runtime callable identity drift: {symbol}") from error
        if (
            actual_module != expected_module
            or actual_qualname != expected_qualname
            or actual_signature != expected_signature
            or actual_ast != expected_ast
        ):
            raise RuntimeError(f"core runtime callable identity drift: {symbol}")
        callable_identities.append(
            Vituri2024CoreCallableIdentity(
                symbol=symbol,
                module=actual_module,
                qualname=actual_qualname,
                signature=actual_signature,
                canonical_function_ast_sha256=actual_ast,
            )
        )
    return Vituri2024CoreProvenance(
        provenance_mode=provenance_mode,
        baseline_commit=VITURI2024_SCF_BASELINE_COMMIT,
        baseline_commit_authority=_CORE_BASELINE_COMMIT_AUTHORITY,
        repository_checks_available=repository_checks_available,
        repository_ancestry_verified=repository_checks_available,
        repository_head_core_verified=repository_checks_available,
        repository_index_core_verified=repository_checks_available,
        repository_worktree_core_verified=repository_checks_available,
        source_manifests=tuple(source_manifests),
        callable_identities=tuple(callable_identities),
        package_version=_package_version(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        numpy_version=np.__version__,
    )


SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "label": SCF_REPLAY_ARCHIVE_SCHEMA_LABEL,
        "seed_fields": (
            "seed",
            "transfer_source_receipts",
            "pre_init",
            "post_init",
            "steps",
            "final_recomputation",
            "callback_sequence",
            "exit_reason",
            "converged",
            "iterations",
        ),
        "step_fields": (
            "iteration",
            "previous_density",
            "interaction_h",
            "total_hamiltonian",
            "raw_density",
            "raw_energies",
            "raw_mu",
            "density_update_observables_sha256",
            "mixed_density",
            "state_density",
            "state_hamiltonian",
            "state_energies",
            "state_mu",
            "delta_interaction_h",
            "oda_lambda",
            "norm_raw",
            "norm_mixed",
            "norm_selected",
            "energy",
            "interaction_h_from_cache",
            "state_diagnostics_manifest_sha256",
        ),
        "state_snapshot_diagnostics_manifest_required": True,
        "final_state_diagnostics_manifest_required": True,
        "selected_canonical_hash_fields": SCF_REPLAY_SELECTED_HASH_FIELDS,
        "current_replay_receipt_or_transcript_allowed": False,
    }
)
SCF_REPLAY_ADAPTER_ABI_FINGERPRINT = _fingerprint(
    {
        "label": SCF_REPLAY_ADAPTER_ABI_LABEL,
        "methods": ("build_fresh_scf_state", "build_scf_problem"),
        "provider_exposes_archive_loader_method": False,
        "provider_exposes_scf_entrypoint_method": False,
        "archive_data_independence_verified": False,
        "problem_type": "mean_field.core.hf.HartreeFockProblem",
    }
)
SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_FINGERPRINT = _fingerprint(
    {
        "label": SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_LABEL,
        "metadata": (
            "archive_authority_fingerprint",
            "source_artifact_sha256",
            "archive_loader_implementation_fingerprint",
            "archive_schema_fingerprint",
        ),
        "methods": ("load_immutable_scf_archive",),
        "authority_exposes_live_builder_methods": False,
    }
)
SCF_REPLAY_VERIFIER_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "label": SCF_REPLAY_VERIFIER_SCHEMA_LABEL,
        "baseline_commit": VITURI2024_SCF_BASELINE_COMMIT,
        "archive_schema": SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT,
        "live_adapter_abi": SCF_REPLAY_ADAPTER_ABI_FINGERPRINT,
        "archive_authority_abi": SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_FINGERPRINT,
        "actual_entrypoint": "mean_field.core.hf.problem.run_hartree_fock_problem",
        "problem_iteration_alias_identity_required": True,
        "callback_identity": "canonical_source_ast_and_python_code_v1",
        "provider_step_and_final_callbacks": "required_none_verifier_observers_only",
        "evidence_model": _SCF_REPLAY_EVIDENCE_MODEL,
        "archive_data_independence_verified": False,
        "hostile_provider_resistance_verified": False,
        "live_builder_dependency_state_independently_pinned": False,
        "restart_capability": "unavailable",
    }
)


def scf_archive_authority_fingerprint(
    *,
    source_artifact_sha256: str,
    archive_loader_implementation_fingerprint: str,
    archive_schema_fingerprint: str = SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT,
) -> str:
    return _fingerprint(
        {
            "source_artifact_sha256": _sha256(
                source_artifact_sha256, "archive-authority source artifact"
            ),
            "archive_loader_implementation_fingerprint": _sha256(
                archive_loader_implementation_fingerprint,
                "archive-authority loader implementation",
            ),
            "archive_schema_fingerprint": _sha256(
                archive_schema_fingerprint, "archive-authority schema"
            ),
            "archive_authority_abi_fingerprint": (
                SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_FINGERPRINT
            ),
        }
    )


def scf_dependency_archive_fingerprint(
    *,
    source_commit: str,
    source_artifact_sha256: str,
    state_builder_implementation_fingerprint: str,
    problem_builder_implementation_fingerprint: str,
    scf_adapter_schema_fingerprint: str,
    scf_adapter_abi_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "source_commit": _commit(source_commit, "dependency source commit"),
            "source_artifact_sha256": _sha256(
                source_artifact_sha256, "dependency source artifact"
            ),
            "state_builder_implementation_fingerprint": _sha256(
                state_builder_implementation_fingerprint, "state builder implementation"
            ),
            "problem_builder_implementation_fingerprint": _sha256(
                problem_builder_implementation_fingerprint, "problem builder implementation"
            ),
            "scf_adapter_schema_fingerprint": _sha256(
                scf_adapter_schema_fingerprint, "SCF adapter schema"
            ),
            "scf_adapter_abi_fingerprint": _sha256(
                scf_adapter_abi_fingerprint, "SCF adapter ABI"
            ),
        }
    )


def scf_provider_fingerprint(
    *,
    functional_provider_fingerprint: str,
    state_builder_implementation_fingerprint: str,
    problem_builder_implementation_fingerprint: str,
    scf_adapter_schema_fingerprint: str,
    scf_adapter_abi_fingerprint: str,
    scf_dependency_archive_fingerprint: str,
) -> str:
    return _fingerprint(
        {
            "functional_provider_fingerprint": _sha256(
                functional_provider_fingerprint, "functional provider"
            ),
            "state_builder_implementation_fingerprint": _sha256(
                state_builder_implementation_fingerprint, "state builder"
            ),
            "problem_builder_implementation_fingerprint": _sha256(
                problem_builder_implementation_fingerprint, "problem builder"
            ),
            "scf_adapter_schema_fingerprint": _sha256(
                scf_adapter_schema_fingerprint, "adapter schema"
            ),
            "scf_adapter_abi_fingerprint": _sha256(
                scf_adapter_abi_fingerprint, "adapter ABI"
            ),
            "scf_dependency_archive_fingerprint": _sha256(
                scf_dependency_archive_fingerprint, "dependency archive"
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SCFTransferSourceReceipt:
    density_side: Literal["lower_density", "higher_density"]
    source_label: str
    source_commit: str
    source_artifact_sha256: str
    source_state_sha256: str

    def __post_init__(self) -> None:
        if self.density_side not in ("lower_density", "higher_density"):
            raise ValueError("transfer source must identify one density side")
        _text(self.source_label, "transfer source label")
        _commit(self.source_commit, "transfer source commit")
        _sha256(self.source_artifact_sha256, "transfer source artifact")
        _sha256(self.source_state_sha256, "transfer source state")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024SCFStateSnapshot:
    h0: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float
    precision: float
    diagnostics_manifest_sha256: str

    def __post_init__(self) -> None:
        h0 = _immutable_complex(self.h0, "state h0")
        shape = h0.shape
        if shape[0] != shape[1] or shape[2] < 1:
            raise ValueError("state matrix arrays must have shape (n,n,nk)")
        object.__setattr__(self, "h0", h0)
        object.__setattr__(self, "density", _immutable_complex(self.density, "state density", shape))
        object.__setattr__(
            self, "hamiltonian", _immutable_complex(self.hamiltonian, "state Hamiltonian", shape)
        )
        object.__setattr__(
            self,
            "energies",
            _immutable_float(self.energies, "state energies", (shape[0], shape[2])),
        )
        object.__setattr__(self, "mu", _finite(self.mu, "state chemical potential"))
        precision = _finite(self.precision, "state precision")
        if precision <= 0.0:
            raise ValueError("state precision must be positive")
        object.__setattr__(self, "precision", precision)
        _sha256(self.diagnostics_manifest_sha256, "state diagnostics manifest")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFStepArchive:
    iteration: int
    previous_density: ComplexArray
    interaction_h: ComplexArray
    total_hamiltonian: ComplexArray
    raw_density: ComplexArray
    raw_energies: FloatArray
    raw_mu: float
    density_update_observables_sha256: str
    mixed_density: ComplexArray
    state_density: ComplexArray
    state_hamiltonian: ComplexArray
    state_energies: FloatArray
    state_mu: float
    state_diagnostics_manifest_sha256: str
    delta_interaction_h: ComplexArray | None
    oda_lambda: float
    norm_raw: float
    norm_mixed: float
    norm_selected: float
    energy: float
    interaction_h_from_cache: bool

    def __post_init__(self) -> None:
        iteration = _strict_int(self.iteration, "SCF step iteration")
        if iteration < 1:
            raise ValueError("SCF step iteration must be positive")
        object.__setattr__(self, "iteration", iteration)
        previous = _immutable_complex(self.previous_density, "previous density")
        shape = previous.shape
        object.__setattr__(self, "previous_density", previous)
        for name, label in (
            ("interaction_h", "interaction Hamiltonian"),
            ("total_hamiltonian", "total Hamiltonian"),
            ("raw_density", "raw density"),
            ("mixed_density", "mixed density"),
            ("state_density", "state density"),
            ("state_hamiltonian", "state Hamiltonian"),
        ):
            object.__setattr__(self, name, _immutable_complex(getattr(self, name), label, shape))
        object.__setattr__(
            self, "raw_energies", _immutable_float(self.raw_energies, "raw energies", (shape[0], shape[2]))
        )
        object.__setattr__(
            self,
            "state_energies",
            _immutable_float(self.state_energies, "state energies", (shape[0], shape[2])),
        )
        if self.delta_interaction_h is not None:
            object.__setattr__(
                self,
                "delta_interaction_h",
                _immutable_complex(self.delta_interaction_h, "delta interaction Hamiltonian", shape),
            )
        _sha256(self.density_update_observables_sha256, "density-update observables")
        _sha256(self.state_diagnostics_manifest_sha256, "step state diagnostics manifest")
        for name in (
            "raw_mu",
            "state_mu",
            "oda_lambda",
            "norm_raw",
            "norm_mixed",
            "norm_selected",
            "energy",
        ):
            value = _finite(getattr(self, name), f"SCF step {name}")
            if name in ("oda_lambda", "norm_raw", "norm_mixed", "norm_selected") and value < 0.0:
                raise ValueError(f"SCF step {name} must be non-negative")
            object.__setattr__(self, name, value)
        if type(self.interaction_h_from_cache) is not bool:
            raise TypeError("SCF cache flag must be bool")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFFinalRecomputationArchive:
    h0: ComplexArray
    state_density: ComplexArray
    effective_interaction_h: ComplexArray
    total_hamiltonian: ComplexArray
    raw_density: ComplexArray
    energies: FloatArray
    mu: float
    energy: float
    raw_norm: float
    density_update_observables_sha256: str
    state_diagnostics_manifest_sha256: str

    def __post_init__(self) -> None:
        h0 = _immutable_complex(self.h0, "final h0")
        shape = h0.shape
        object.__setattr__(self, "h0", h0)
        for name, label in (
            ("state_density", "final state density"),
            ("effective_interaction_h", "final effective interaction"),
            ("total_hamiltonian", "final total Hamiltonian"),
            ("raw_density", "final raw density"),
        ):
            object.__setattr__(self, name, _immutable_complex(getattr(self, name), label, shape))
        object.__setattr__(
            self, "energies", _immutable_float(self.energies, "final energies", (shape[0], shape[2]))
        )
        object.__setattr__(self, "mu", _finite(self.mu, "final chemical potential"))
        object.__setattr__(self, "energy", _finite(self.energy, "final energy"))
        object.__setattr__(self, "raw_norm", _nonnegative(self.raw_norm, "final raw norm"))
        _sha256(self.density_update_observables_sha256, "final density-update observables")
        _sha256(self.state_diagnostics_manifest_sha256, "final state diagnostics manifest")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFSeedTrajectoryArchive:
    seed: Vituri2024SCFSeedReceipt
    transfer_source_receipts: tuple[
        Vituri2024SCFTransferSourceReceipt, Vituri2024SCFTransferSourceReceipt
    ]
    pre_init: Vituri2024SCFStateSnapshot
    post_init: Vituri2024SCFStateSnapshot
    steps: tuple[Vituri2024SCFStepArchive, ...]
    final_recomputation: Vituri2024SCFFinalRecomputationArchive
    callback_sequence: tuple[str, ...]
    exit_reason: Literal["converged", "oda_stall", "max_iter"]
    converged: bool
    iterations: int

    def __post_init__(self) -> None:
        if type(self.seed) is not Vituri2024SCFSeedReceipt:
            raise TypeError("trajectory requires a typed SCF seed")
        sources = tuple(self.transfer_source_receipts)
        if len(sources) != 2 or any(type(item) is not Vituri2024SCFTransferSourceReceipt for item in sources):
            raise TypeError("trajectory requires two typed transfer-source receipts")
        if tuple(item.density_side for item in sources) != ("lower_density", "higher_density"):
            raise ValueError("trajectory transfer sources must be ordered lower/higher density")
        if sources[0].source_state_sha256 == sources[1].source_state_sha256:
            raise ValueError("two-sided transfer sources must be distinct")
        object.__setattr__(self, "transfer_source_receipts", sources)
        if type(self.pre_init) is not Vituri2024SCFStateSnapshot or type(self.post_init) is not Vituri2024SCFStateSnapshot:
            raise TypeError("trajectory requires typed pre/post initializer snapshots")
        steps = tuple(self.steps)
        if not steps or any(type(item) is not Vituri2024SCFStepArchive for item in steps):
            raise TypeError("trajectory requires typed SCF steps")
        if tuple(item.iteration for item in steps) != tuple(range(1, len(steps) + 1)):
            raise ValueError("trajectory step iterations must be consecutive and one-based")
        object.__setattr__(self, "steps", steps)
        if type(self.final_recomputation) is not Vituri2024SCFFinalRecomputationArchive:
            raise TypeError("trajectory requires a typed final recomputation")
        if type(self.callback_sequence) is not tuple or any(
            not isinstance(item, str) or not item for item in self.callback_sequence
        ):
            raise TypeError("trajectory callback sequence must be a tuple of labels")
        if self.exit_reason not in ("converged", "oda_stall", "max_iter"):
            raise ValueError("unsupported trajectory exit reason")
        if type(self.converged) is not bool or self.converged != (self.exit_reason == "converged"):
            raise ValueError("trajectory converged flag contradicts exit reason")
        iterations = _strict_int(self.iterations, "trajectory iterations")
        if iterations != len(steps):
            raise ValueError("trajectory iteration count does not match step archive")
        object.__setattr__(self, "iterations", iterations)


@dataclass(frozen=True, slots=True)
class Vituri2024SCFSelectedSource:
    selected_branch_label: str
    source_state_sha256: str
    h0: ComplexArray
    effective_interaction_h: ComplexArray
    fock: ComplexArray
    projector: ComplexArray
    energies: FloatArray
    mu: float
    registered_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text(self.selected_branch_label, "selected branch label")
        _sha256(self.source_state_sha256, "selected source state")
        h0 = _immutable_complex(self.h0, "selected h0")
        shape = h0.shape
        object.__setattr__(self, "h0", h0)
        for name, label in (
            ("effective_interaction_h", "selected effective interaction"),
            ("fock", "selected Fock"),
            ("projector", "selected projector"),
        ):
            object.__setattr__(self, name, _immutable_complex(getattr(self, name), label, shape))
        object.__setattr__(
            self, "energies", _immutable_float(self.energies, "selected energies", (shape[0], shape[2]))
        )
        object.__setattr__(self, "mu", _finite(self.mu, "selected chemical potential"))
        hashes = tuple(self.registered_hashes)
        if tuple(name for name, _ in hashes) != SCF_REPLAY_SELECTED_HASH_FIELDS:
            raise ValueError("selected-source registered hash inventory changed")
        arrays = {
            "h0": self.h0,
            "effective_interaction_h": self.effective_interaction_h,
            "fock": self.fock,
            "projector": self.projector,
            "energies": self.energies,
        }
        for name, digest in hashes:
            _sha256(digest, f"selected {name} hash")
            if digest != canonical_array_sha256(arrays[name]):
                raise ValueError(f"selected-source {name} registered hash mismatch")
        object.__setattr__(self, "registered_hashes", hashes)


@dataclass(frozen=True, slots=True)
class Vituri2024ImmutableHistoricalSCFArchive:
    archive_authority_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    archive_loader_implementation_fingerprint: str
    archive_schema_fingerprint: str
    generation_phase: Literal["detached_before_state_and_problem_builders"]
    seed_trajectories: tuple[Vituri2024SCFSeedTrajectoryArchive, ...]
    branch_records: tuple[Vituri2024BranchEnergyReceipt, ...]
    original_branch_table_bytes: bytes
    original_branch_table_sha256: str
    selected_branch_label: str
    selected_source: Vituri2024SCFSelectedSource

    def __post_init__(self) -> None:
        for value, label in (
            (self.archive_authority_fingerprint, "archive authority"),
            (self.source_artifact_sha256, "archive source artifact"),
            (self.spec_fingerprint, "archive spec"),
            (self.archive_loader_implementation_fingerprint, "archive loader"),
            (self.archive_schema_fingerprint, "archive schema"),
            (self.original_branch_table_sha256, "original branch table"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "archive source commit")
        if self.archive_schema_fingerprint != SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT:
            raise ValueError("historical SCF archive schema changed")
        if self.generation_phase != SCF_REPLAY_ARCHIVE_GENERATION_PHASE:
            raise ValueError("historical archive was not generated before builders")
        trajectories = tuple(self.seed_trajectories)
        records = tuple(self.branch_records)
        if not trajectories or any(type(item) is not Vituri2024SCFSeedTrajectoryArchive for item in trajectories):
            raise TypeError("historical archive requires typed seed trajectories")
        if len(records) != len(trajectories) or any(type(item) is not Vituri2024BranchEnergyReceipt for item in records):
            raise TypeError("historical archive branch rows must match trajectory count")
        if tuple(item.seed for item in trajectories) != tuple(item.seed for item in records):
            raise ValueError("historical branch rows and trajectories have different seed order")
        object.__setattr__(self, "seed_trajectories", trajectories)
        object.__setattr__(self, "branch_records", records)
        if type(self.original_branch_table_bytes) is not bytes or not self.original_branch_table_bytes:
            raise TypeError("historical branch table must be non-empty immutable bytes")
        if hashlib.sha256(self.original_branch_table_bytes).hexdigest() != self.original_branch_table_sha256:
            raise ValueError("historical branch table bytes/hash mismatch")
        _text(self.selected_branch_label, "archive selected branch")
        if type(self.selected_source) is not Vituri2024SCFSelectedSource:
            raise TypeError("historical archive requires a typed selected source")
        if self.selected_source.selected_branch_label != self.selected_branch_label:
            raise ValueError("archive selected-source label mismatch")



def _array_manifest(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": canonical_array_sha256(array),
    }


def _snapshot_manifest(snapshot: Vituri2024SCFStateSnapshot) -> dict[str, object]:
    return {
        "h0": _array_manifest(snapshot.h0),
        "density": _array_manifest(snapshot.density),
        "hamiltonian": _array_manifest(snapshot.hamiltonian),
        "energies": _array_manifest(snapshot.energies),
        "mu": snapshot.mu,
        "precision": snapshot.precision,
        "diagnostics_manifest_sha256": snapshot.diagnostics_manifest_sha256,
    }


def _step_manifest(step: Vituri2024SCFStepArchive) -> dict[str, object]:
    result: dict[str, object] = {
        "iteration": step.iteration,
        "raw_mu": step.raw_mu,
        "state_mu": step.state_mu,
        "oda_lambda": step.oda_lambda,
        "norm_raw": step.norm_raw,
        "norm_mixed": step.norm_mixed,
        "norm_selected": step.norm_selected,
        "energy": step.energy,
        "density_update_observables_sha256": step.density_update_observables_sha256,
        "interaction_h_from_cache": step.interaction_h_from_cache,
        "state_diagnostics_manifest_sha256": step.state_diagnostics_manifest_sha256,
    }
    for name in (
        "previous_density",
        "interaction_h",
        "total_hamiltonian",
        "raw_density",
        "raw_energies",
        "mixed_density",
        "state_density",
        "state_hamiltonian",
        "state_energies",
    ):
        result[name] = _array_manifest(getattr(step, name))
    result["delta_interaction_h"] = (
        None if step.delta_interaction_h is None else _array_manifest(step.delta_interaction_h)
    )
    return result


def scf_archive_manifest_sha256(archive: Vituri2024ImmutableHistoricalSCFArchive) -> str:
    if type(archive) is not Vituri2024ImmutableHistoricalSCFArchive:
        raise TypeError("SCF archive manifest requires the exact typed archive")
    trajectories: list[dict[str, object]] = []
    for trajectory in archive.seed_trajectories:
        final = trajectory.final_recomputation
        trajectories.append(
            {
                "seed": asdict(trajectory.seed),
                "transfer_source_receipts": [
                    asdict(item) for item in trajectory.transfer_source_receipts
                ],
                "pre_init": _snapshot_manifest(trajectory.pre_init),
                "post_init": _snapshot_manifest(trajectory.post_init),
                "steps": [_step_manifest(step) for step in trajectory.steps],
                "final_recomputation": {
                    "h0": _array_manifest(final.h0),
                    "state_density": _array_manifest(final.state_density),
                    "effective_interaction_h": _array_manifest(final.effective_interaction_h),
                    "total_hamiltonian": _array_manifest(final.total_hamiltonian),
                    "raw_density": _array_manifest(final.raw_density),
                    "energies": _array_manifest(final.energies),
                    "mu": final.mu,
                    "energy": final.energy,
                    "raw_norm": final.raw_norm,
                    "density_update_observables_sha256": final.density_update_observables_sha256,
                    "state_diagnostics_manifest_sha256": final.state_diagnostics_manifest_sha256,
                },
                "callback_sequence": trajectory.callback_sequence,
                "exit_reason": trajectory.exit_reason,
                "converged": trajectory.converged,
                "iterations": trajectory.iterations,
            }
        )
    selected = archive.selected_source
    return _fingerprint(
        {
            "schema": SCF_REPLAY_ARCHIVE_SCHEMA_LABEL,
            "archive_authority_fingerprint": archive.archive_authority_fingerprint,
            "source_commit": archive.source_commit,
            "source_artifact_sha256": archive.source_artifact_sha256,
            "spec_fingerprint": archive.spec_fingerprint,
            "archive_loader_implementation_fingerprint": archive.archive_loader_implementation_fingerprint,
            "archive_schema_fingerprint": archive.archive_schema_fingerprint,
            "generation_phase": archive.generation_phase,
            "seed_trajectories": trajectories,
            "branch_records": [asdict(item) for item in archive.branch_records],
            "original_branch_table_sha256": archive.original_branch_table_sha256,
            "selected_branch_label": archive.selected_branch_label,
            "selected_source": {
                "selected_branch_label": selected.selected_branch_label,
                "source_state_sha256": selected.source_state_sha256,
                "h0": _array_manifest(selected.h0),
                "effective_interaction_h": _array_manifest(selected.effective_interaction_h),
                "fock": _array_manifest(selected.fock),
                "projector": _array_manifest(selected.projector),
                "energies": _array_manifest(selected.energies),
                "mu": selected.mu,
                "registered_hashes": selected.registered_hashes,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SCFReplayTolerance:
    field_name: str
    absolute: float
    relative: float
    require_canonical_hash: bool = False
    require_bitwise_parity: bool = False

    def __post_init__(self) -> None:
        _text(self.field_name, "SCF tolerance field")
        object.__setattr__(self, "absolute", _nonnegative(self.absolute, "absolute tolerance"))
        object.__setattr__(self, "relative", _nonnegative(self.relative, "relative tolerance"))
        if type(self.require_canonical_hash) is not bool or type(self.require_bitwise_parity) is not bool:
            raise TypeError("SCF tolerance parity flags must be bool")


_TOLERANCE_FIELDS: tuple[str, ...] = (
    "pre_init.h0", "pre_init.density", "pre_init.hamiltonian", "pre_init.energies",
    "pre_init.mu", "pre_init.precision", "post_init.h0", "post_init.density",
    "post_init.hamiltonian", "post_init.energies", "post_init.mu", "post_init.precision",
    "step.previous_density", "step.interaction_h", "step.total_hamiltonian",
    "step.raw_density", "step.raw_energies", "step.raw_mu", "step.mixed_density",
    "step.state_density", "step.state_hamiltonian", "step.state_energies", "step.state_mu",
    "step.delta_interaction_h", "step.oda_lambda", "step.norm_raw", "step.norm_mixed",
    "step.norm_selected", "step.energy", "final.h0", "final.state_density",
    "final.effective_interaction_h", "final.total_hamiltonian", "final.raw_density",
    "final.energies", "final.mu", "final.energy", "final.raw_norm", "run.iter_energy",
    "run.iter_err", "run.iter_oda", "branch.energy", "branch.terminal_norm_raw",
    "branch.terminal_norm_mixed", "branch.terminal_norm_selected",
    "branch.terminal_oda_lambda", "branch.final_replay_raw_metric", "selected.h0",
    "selected.effective_interaction_h", "selected.fock", "selected.projector",
    "selected.energies", "selected.mu",
)


_PROBLEM_CALLBACK_ROLES: tuple[str, ...] = (
    "initializer",
    "interaction_builder",
    "density_builder",
    "energy_functional",
    "oda_parameterizer",
    "oda_delta_interaction_builder",
    "hamiltonian_postprocessor",
    "density_postprocessor",
    "step_callback",
    "final_state_callback",
)
SCF_REPLAY_V1_ABSOLUTE_TOLERANCE_MAXIMUM = 1.0e-12
SCF_REPLAY_V1_RELATIVE_TOLERANCE_MAXIMUM = 1.0e-12


def _locked_v1_tolerances() -> tuple[Vituri2024SCFReplayTolerance, ...]:
    selected_hash_fields = {f"selected.{name}" for name in SCF_REPLAY_SELECTED_HASH_FIELDS}
    return tuple(
        Vituri2024SCFReplayTolerance(
            field_name=name,
            absolute=SCF_REPLAY_V1_ABSOLUTE_TOLERANCE_MAXIMUM,
            relative=SCF_REPLAY_V1_RELATIVE_TOLERANCE_MAXIMUM,
            require_canonical_hash=name in selected_hash_fields,
            require_bitwise_parity=False,
        )
        for name in _TOLERANCE_FIELDS
    )


_LOCKED_V1_TOLERANCES = _locked_v1_tolerances()


def default_vituri2024_scf_replay_tolerances() -> tuple[Vituri2024SCFReplayTolerance, ...]:
    """Return the reviewed v1 inventory; no caller override is accepted."""

    return _LOCKED_V1_TOLERANCES


@dataclass(frozen=True, slots=True)
class Vituri2024SCFReplayApproval:
    scope: Literal["vituri2024_uninterrupted_registered_seed_trajectory_replay_v1"]
    core_provenance_mode: Literal[
        "git_ancestor_head_index_worktree_verified",
        "pinned_hash_verified_source_export",
    ]
    baseline_commit: str
    core_baseline_commit_authority: Literal[
        "hardcoded_vituri2024_scf_baseline_commit_and_core_hash_manifest"
    ]
    core_provenance_fingerprint: str
    core_source_manifests: tuple[Vituri2024CoreSourceManifest, ...]
    core_callable_identities: tuple[Vituri2024CoreCallableIdentity, ...]
    package_version: str
    python_version: str
    python_implementation: str
    numpy_version: str
    verifier_schema_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    archive_authority_fingerprint: str
    archive_authority_source_artifact_sha256: str
    archive_authority_loader_implementation_fingerprint: str
    provider_fingerprint: str
    functional_provider_fingerprint: str
    scf_provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    scf_policy_fingerprint: str
    shared_functional_fingerprint: str
    attested_source_fingerprint: str
    source_state_sha256: str
    scalar_energy_function_fingerprint: str
    state_builder_implementation_fingerprint: str
    problem_builder_implementation_fingerprint: str
    scf_adapter_schema_fingerprint: str
    scf_adapter_abi_fingerprint: str
    scf_dependency_archive_fingerprint: str
    archive_schema_fingerprint: str
    expected_archive_manifest_sha256: str
    expected_branch_table_sha256: str
    exact_seed_inventory: tuple[Vituri2024SCFSeedReceipt, ...]
    live_provider_metadata_snapshot: tuple[tuple[str, object], ...]
    live_builder_manifests: tuple[
        Vituri2024SCFCallableManifest, Vituri2024SCFCallableManifest
    ]
    problem_callback_manifests: tuple[Vituri2024SCFProblemCallbackManifest, ...]
    tolerances: tuple[Vituri2024SCFReplayTolerance, ...]
    detached_approval_provenance: str

    def __post_init__(self) -> None:
        if self.scope != VITURI2024_SCF_REPLAY_SCOPE:
            raise ValueError("SCF replay approval scope changed")
        if self.core_provenance_mode not in (
            _CORE_PROVENANCE_GIT_MODE,
            _CORE_PROVENANCE_SOURCE_EXPORT_MODE,
        ):
            raise ValueError("unsupported approval SCF core provenance mode")
        if _commit(self.baseline_commit, "approval baseline") != VITURI2024_SCF_BASELINE_COMMIT:
            raise ValueError("SCF replay approval baseline changed")
        if self.core_baseline_commit_authority != _CORE_BASELINE_COMMIT_AUTHORITY:
            raise ValueError("SCF replay approval baseline authority changed")
        for value, label in (
            (self.core_provenance_fingerprint, "core provenance"),
            (self.verifier_schema_fingerprint, "verifier schema"),
            (self.verifier_module_ast_manifest_sha256, "verifier module AST"),
            (self.archive_authority_fingerprint, "approval archive authority"),
            (self.archive_authority_source_artifact_sha256, "approval archive artifact"),
            (
                self.archive_authority_loader_implementation_fingerprint,
                "approval archive loader",
            ),
            (self.provider_fingerprint, "approval provider"),
            (self.functional_provider_fingerprint, "approval functional provider"),
            (self.scf_provider_fingerprint, "approval SCF provider"),
            (self.source_artifact_sha256, "approval source artifact"),
            (self.spec_fingerprint, "approval spec"),
            (self.scf_policy_fingerprint, "approval SCF policy"),
            (self.shared_functional_fingerprint, "approval shared functional"),
            (self.attested_source_fingerprint, "approval attested source"),
            (self.source_state_sha256, "approval source state"),
            (self.scalar_energy_function_fingerprint, "approval scalar energy"),
            (self.state_builder_implementation_fingerprint, "approval state builder"),
            (self.problem_builder_implementation_fingerprint, "approval problem builder"),
            (self.scf_adapter_schema_fingerprint, "approval adapter schema"),
            (self.scf_adapter_abi_fingerprint, "approval adapter ABI"),
            (self.scf_dependency_archive_fingerprint, "approval dependency archive"),
            (self.archive_schema_fingerprint, "approval archive schema"),
            (self.expected_archive_manifest_sha256, "approval archive manifest"),
            (self.expected_branch_table_sha256, "approval branch table"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "approval source commit")
        if self.verifier_schema_fingerprint != SCF_REPLAY_VERIFIER_SCHEMA_FINGERPRINT:
            raise ValueError("SCF replay verifier schema changed")
        if self.verifier_module_ast_manifest_sha256 != scf_replay_module_ast_manifest_sha256():
            raise ValueError("SCF replay verifier AST/source manifest changed")
        if self.archive_schema_fingerprint != SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT:
            raise ValueError("SCF replay archive schema changed")
        if self.scf_adapter_abi_fingerprint != SCF_REPLAY_ADAPTER_ABI_FINGERPRINT:
            raise ValueError("SCF replay adapter ABI changed")
        seeds = tuple(self.exact_seed_inventory)
        if len(seeds) < 2 or any(type(item) is not Vituri2024SCFSeedReceipt for item in seeds):
            raise TypeError("approval requires multiple typed seeds")
        object.__setattr__(self, "exact_seed_inventory", seeds)
        metadata_snapshot = tuple(self.live_provider_metadata_snapshot)
        if tuple(name for name, _ in metadata_snapshot) != SCF_REPLAY_PROVIDER_METADATA_FIELDS:
            raise ValueError("SCF replay live-provider metadata inventory/order changed")
        object.__setattr__(self, "live_provider_metadata_snapshot", metadata_snapshot)
        builders = tuple(self.live_builder_manifests)
        if (
            len(builders) != 2
            or tuple(item.role for item in builders)
            != ("build_fresh_scf_state", "build_scf_problem")
        ):
            raise ValueError("SCF replay live-builder manifest inventory/order changed")
        object.__setattr__(self, "live_builder_manifests", builders)
        callback_manifests = tuple(self.problem_callback_manifests)
        if tuple(item.role for item in callback_manifests) != _PROBLEM_CALLBACK_ROLES:
            raise ValueError("SCF replay problem-callback manifest inventory/order changed")
        for role in ("step_callback", "final_state_callback"):
            item = callback_manifests[_PROBLEM_CALLBACK_ROLES.index(role)]
            if item.implementation_kind != "none" or item.callable_manifest is not None:
                raise ValueError(f"SCF replay v1 requires provider {role} is None")
        object.__setattr__(self, "problem_callback_manifests", callback_manifests)
        tolerances = tuple(self.tolerances)
        if tolerances != _LOCKED_V1_TOLERANCES:
            raise ValueError("SCF replay locked v1 tolerance/value/hash inventory changed")
        object.__setattr__(self, "tolerances", tolerances)
        _text(self.detached_approval_provenance, "detached approval provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def make_vituri2024_scf_replay_approval(
    binding: Vituri2024HalfMetalHFProviderBinding,
    archive_authority: Vituri2024SCFArchiveAuthorityProtocol,
    *,
    expected_archive_manifest_sha256: str,
    expected_branch_table_sha256: str,
    problem_callback_manifests: tuple[Vituri2024SCFProblemCallbackManifest, ...],
    provenance: str,
) -> Vituri2024SCFReplayApproval:
    """Create a detached approval without executing a provider method."""

    if type(binding) is not Vituri2024HalfMetalHFProviderBinding:
        raise TypeError("SCF replay approval requires the exact provider binding")
    core = verified_vituri2024_core_provenance()
    spec = binding.spec
    assert spec.scf_policy is not None and spec.shared_functional is not None
    assert spec.attested_source is not None
    provider = binding.provider
    if provider is archive_authority:
        raise TypeError("archive authority and live provider must be distinct objects")
    if not isinstance(archive_authority, Vituri2024SCFArchiveAuthorityProtocol):
        raise TypeError("SCF replay approval requires the archive-authority protocol")
    authority_snapshot = _archive_authority_snapshot(archive_authority)
    live_snapshot = _provider_snapshot(provider)
    _reject_same_authority_fingerprint(authority_snapshot, live_snapshot)
    if authority_snapshot["archive_schema_fingerprint"] != SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT:
        raise ValueError("SCF replay archive-authority schema changed")
    if authority_snapshot["source_artifact_sha256"] != provider.source_artifact_sha256:
        raise ValueError("archive authority and live provider bind different source artifacts")
    required_authority_fingerprint = scf_archive_authority_fingerprint(
        source_artifact_sha256=authority_snapshot["source_artifact_sha256"],
        archive_loader_implementation_fingerprint=authority_snapshot[
            "archive_loader_implementation_fingerprint"
        ],
        archive_schema_fingerprint=authority_snapshot["archive_schema_fingerprint"],
    )
    if authority_snapshot["archive_authority_fingerprint"] != required_authority_fingerprint:
        raise ValueError("derived SCF archive-authority fingerprint mismatch")
    live_builder_manifests = (
        vituri2024_scf_callable_manifest(
            "build_fresh_scf_state", getattr(provider, "build_fresh_scf_state")
        ),
        vituri2024_scf_callable_manifest(
            "build_scf_problem", getattr(provider, "build_scf_problem")
        ),
    )
    return Vituri2024SCFReplayApproval(
        scope=VITURI2024_SCF_REPLAY_SCOPE,
        core_provenance_mode=core.provenance_mode,
        baseline_commit=VITURI2024_SCF_BASELINE_COMMIT,
        core_baseline_commit_authority=core.baseline_commit_authority,
        core_provenance_fingerprint=core.fingerprint,
        core_source_manifests=core.source_manifests,
        core_callable_identities=core.callable_identities,
        package_version=core.package_version,
        python_version=core.python_version,
        python_implementation=core.python_implementation,
        numpy_version=core.numpy_version,
        verifier_schema_fingerprint=SCF_REPLAY_VERIFIER_SCHEMA_FINGERPRINT,
        verifier_module_ast_manifest_sha256=scf_replay_module_ast_manifest_sha256(),
        archive_authority_fingerprint=authority_snapshot["archive_authority_fingerprint"],
        archive_authority_source_artifact_sha256=authority_snapshot["source_artifact_sha256"],
        archive_authority_loader_implementation_fingerprint=authority_snapshot[
            "archive_loader_implementation_fingerprint"
        ],
        provider_fingerprint=provider.provider_fingerprint,
        functional_provider_fingerprint=getattr(provider, "functional_provider_fingerprint"),
        scf_provider_fingerprint=getattr(provider, "scf_provider_fingerprint"),
        source_commit=provider.source_commit,
        source_artifact_sha256=provider.source_artifact_sha256,
        spec_fingerprint=spec.fingerprint,
        scf_policy_fingerprint=spec.scf_policy.fingerprint,
        shared_functional_fingerprint=spec.shared_functional.fingerprint,
        attested_source_fingerprint=spec.attested_source.fingerprint,
        source_state_sha256=spec.attested_source.source_state_sha256,
        scalar_energy_function_fingerprint=spec.shared_functional.scalar_energy.fingerprint,
        state_builder_implementation_fingerprint=getattr(
            provider, "state_builder_implementation_fingerprint"
        ),
        problem_builder_implementation_fingerprint=getattr(
            provider, "problem_builder_implementation_fingerprint"
        ),
        scf_adapter_schema_fingerprint=getattr(provider, "scf_adapter_schema_fingerprint"),
        scf_adapter_abi_fingerprint=getattr(provider, "scf_adapter_abi_fingerprint"),
        scf_dependency_archive_fingerprint=getattr(
            provider, "scf_dependency_archive_fingerprint"
        ),
        archive_schema_fingerprint=SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT,
        expected_archive_manifest_sha256=_sha256(
            expected_archive_manifest_sha256, "expected archive manifest"
        ),
        expected_branch_table_sha256=_sha256(
            expected_branch_table_sha256, "expected branch table"
        ),
        exact_seed_inventory=spec.scf_policy.seed_records,
        live_provider_metadata_snapshot=tuple(live_snapshot.items()),
        live_builder_manifests=live_builder_manifests,
        problem_callback_manifests=problem_callback_manifests,
        tolerances=default_vituri2024_scf_replay_tolerances(),
        detached_approval_provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SCFReplayContract:
    approval_fingerprint: str
    core_provenance_mode: Literal[
        "git_ancestor_head_index_worktree_verified",
        "pinned_hash_verified_source_export",
    ]
    core_baseline_commit_authority: Literal[
        "hardcoded_vituri2024_scf_baseline_commit_and_core_hash_manifest"
    ]
    core_provenance_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    expected_archive_manifest_sha256: str
    expected_branch_table_sha256: str
    exact_seed_inventory: tuple[Vituri2024SCFSeedReceipt, ...]
    archive_authority_fingerprint: str
    archive_authority_metadata_fingerprint: str
    tolerance_inventory_fingerprint: str
    provider_metadata_fingerprint: str

    def __post_init__(self) -> None:
        if self.core_provenance_mode not in (
            _CORE_PROVENANCE_GIT_MODE,
            _CORE_PROVENANCE_SOURCE_EXPORT_MODE,
        ):
            raise ValueError("unsupported contract SCF core provenance mode")
        if self.core_baseline_commit_authority != _CORE_BASELINE_COMMIT_AUTHORITY:
            raise ValueError("SCF replay contract baseline authority changed")
        for value, label in (
            (self.approval_fingerprint, "contract approval"),
            (self.core_provenance_fingerprint, "contract core provenance"),
            (self.verifier_module_ast_manifest_sha256, "contract verifier AST"),
            (self.expected_archive_manifest_sha256, "contract archive manifest"),
            (self.expected_branch_table_sha256, "contract branch table"),
            (self.archive_authority_fingerprint, "contract archive authority"),
            (
                self.archive_authority_metadata_fingerprint,
                "contract archive-authority metadata",
            ),
            (self.tolerance_inventory_fingerprint, "contract tolerance inventory"),
            (self.provider_metadata_fingerprint, "contract provider metadata"),
        ):
            _sha256(value, label)
        if self.verifier_module_ast_manifest_sha256 != scf_replay_module_ast_manifest_sha256():
            raise ValueError("SCF replay contract verifier AST changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS: tuple[str, ...] = (
    "archive_authority_fingerprint",
    "source_artifact_sha256",
    "archive_loader_implementation_fingerprint",
    "archive_schema_fingerprint",
)
SCF_REPLAY_PROVIDER_METADATA_FIELDS = FUNCTIONAL_REPLAY_PROVIDER_METADATA_FIELDS + (
    "scf_provider_fingerprint",
    "state_builder_implementation_fingerprint",
    "problem_builder_implementation_fingerprint",
    "scf_adapter_schema_fingerprint",
    "scf_adapter_abi_fingerprint",
    "scf_dependency_archive_fingerprint",
)


@runtime_checkable
class Vituri2024SCFArchiveAuthorityProtocol(Protocol):
    archive_authority_fingerprint: str
    source_artifact_sha256: str
    archive_loader_implementation_fingerprint: str
    archive_schema_fingerprint: str

    def load_immutable_scf_archive(
        self, source_artifact_sha256: str
    ) -> Vituri2024ImmutableHistoricalSCFArchive: ...

@runtime_checkable
class Vituri2024SCFReplayProviderProtocol(Vituri2024FunctionalReplayProviderProtocol, Protocol):
    scf_provider_fingerprint: str
    state_builder_implementation_fingerprint: str
    problem_builder_implementation_fingerprint: str
    scf_adapter_schema_fingerprint: str
    scf_adapter_abi_fingerprint: str
    scf_dependency_archive_fingerprint: str

    def build_fresh_scf_state(self, seed: Vituri2024SCFSeedReceipt) -> object: ...

    def build_scf_problem(
        self, state: object, seed: Vituri2024SCFSeedReceipt
    ) -> HartreeFockProblem: ...


def _provider_snapshot(provider: object) -> dict[str, object]:
    return {name: getattr(provider, name) for name in SCF_REPLAY_PROVIDER_METADATA_FIELDS}

def _archive_authority_snapshot(archive_authority: object) -> dict[str, str]:
    return {
        name: getattr(archive_authority, name)
        for name in SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS
    }

def _reject_same_authority_fingerprint(
    authority_snapshot: dict[str, str], provider_snapshot: dict[str, object]
) -> None:
    archive_fingerprint = authority_snapshot["archive_authority_fingerprint"]
    live_fingerprints = {
        value
        for name, value in provider_snapshot.items()
        if name.endswith("provider_fingerprint") and isinstance(value, str)
    }
    if archive_fingerprint in live_fingerprints:
        raise ValueError("archive authority and live provider have the same authority fingerprint")


def _validate_provider_snapshot(
    binding: Vituri2024HalfMetalHFProviderBinding,
    approval: Vituri2024SCFReplayApproval,
    snapshot: dict[str, object],
) -> None:
    if tuple(snapshot) != SCF_REPLAY_PROVIDER_METADATA_FIELDS:
        raise RuntimeError("SCF provider metadata snapshot inventory changed")
    del binding  # Detached approval, not provider self-comparison, is authoritative.
    expected = dict(approval.live_provider_metadata_snapshot)
    if snapshot != expected:
        changed = tuple(
            name
            for name in SCF_REPLAY_PROVIDER_METADATA_FIELDS
            if snapshot[name] != expected[name]
        )
        raise ValueError("SCF provider metadata/approval mismatch: " + ", ".join(changed))
    approval_bindings = {
        "provider_fingerprint": approval.provider_fingerprint,
        "functional_provider_fingerprint": approval.functional_provider_fingerprint,
        "scf_provider_fingerprint": approval.scf_provider_fingerprint,
        "source_commit": approval.source_commit,
        "source_artifact_sha256": approval.source_artifact_sha256,
        "spec_fingerprint": approval.spec_fingerprint,
        "state_builder_implementation_fingerprint": approval.state_builder_implementation_fingerprint,
        "problem_builder_implementation_fingerprint": approval.problem_builder_implementation_fingerprint,
        "scf_adapter_schema_fingerprint": approval.scf_adapter_schema_fingerprint,
        "scf_adapter_abi_fingerprint": approval.scf_adapter_abi_fingerprint,
        "scf_dependency_archive_fingerprint": approval.scf_dependency_archive_fingerprint,
    }
    if any(snapshot[name] != value for name, value in approval_bindings.items()):
        raise ValueError("SCF approval fields do not close to its live-provider snapshot")
    required_dependency = scf_dependency_archive_fingerprint(
        source_commit=approval.source_commit,
        source_artifact_sha256=approval.source_artifact_sha256,
        state_builder_implementation_fingerprint=approval.state_builder_implementation_fingerprint,
        problem_builder_implementation_fingerprint=approval.problem_builder_implementation_fingerprint,
        scf_adapter_schema_fingerprint=approval.scf_adapter_schema_fingerprint,
        scf_adapter_abi_fingerprint=approval.scf_adapter_abi_fingerprint,
    )
    if approval.scf_dependency_archive_fingerprint != required_dependency:
        raise ValueError("SCF provider dependency archive fingerprint mismatch")
    required_provider = scf_provider_fingerprint(
        functional_provider_fingerprint=approval.functional_provider_fingerprint,
        state_builder_implementation_fingerprint=approval.state_builder_implementation_fingerprint,
        problem_builder_implementation_fingerprint=approval.problem_builder_implementation_fingerprint,
        scf_adapter_schema_fingerprint=approval.scf_adapter_schema_fingerprint,
        scf_adapter_abi_fingerprint=approval.scf_adapter_abi_fingerprint,
        scf_dependency_archive_fingerprint=approval.scf_dependency_archive_fingerprint,
    )
    if approval.scf_provider_fingerprint != required_provider:
        raise ValueError("derived SCF provider fingerprint mismatch")


def _assert_snapshot_unchanged(provider: object, baseline: dict[str, object], label: str) -> None:
    after = _provider_snapshot(provider)
    if after != baseline:
        changed = tuple(name for name in SCF_REPLAY_PROVIDER_METADATA_FIELDS if after[name] != baseline[name])
        raise ValueError(f"SCF provider metadata mutated during {label}: " + ", ".join(changed))


def _validate_archive_authority_snapshot(
    approval: Vituri2024SCFReplayApproval, snapshot: dict[str, str]
) -> None:
    if tuple(snapshot) != SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS:
        raise RuntimeError("SCF archive-authority metadata inventory changed")
    expected = {
        "archive_authority_fingerprint": approval.archive_authority_fingerprint,
        "source_artifact_sha256": approval.archive_authority_source_artifact_sha256,
        "archive_loader_implementation_fingerprint": (
            approval.archive_authority_loader_implementation_fingerprint
        ),
        "archive_schema_fingerprint": approval.archive_schema_fingerprint,
    }
    if snapshot != expected:
        changed = tuple(
            name
            for name in SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS
            if snapshot[name] != expected[name]
        )
        raise ValueError("SCF archive-authority metadata/approval mismatch: " + ", ".join(changed))
    required = scf_archive_authority_fingerprint(
        source_artifact_sha256=snapshot["source_artifact_sha256"],
        archive_loader_implementation_fingerprint=snapshot[
            "archive_loader_implementation_fingerprint"
        ],
        archive_schema_fingerprint=snapshot["archive_schema_fingerprint"],
    )
    if snapshot["archive_authority_fingerprint"] != required:
        raise ValueError("derived SCF archive-authority fingerprint mismatch")

def _assert_archive_authority_snapshot_unchanged(
    archive_authority: object, baseline: dict[str, str], label: str
) -> None:
    after = _archive_authority_snapshot(archive_authority)
    if after != baseline:
        changed = tuple(
            name
            for name in SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS
            if after[name] != baseline[name]
        )
        raise ValueError(
            f"SCF archive-authority metadata mutated during {label}: "
            + ", ".join(changed)
        )

def _validate_live_builder_manifests(
    provider: object, approval: Vituri2024SCFReplayApproval
) -> None:
    actual = (
        vituri2024_scf_callable_manifest(
            "build_fresh_scf_state", getattr(provider, "build_fresh_scf_state")
        ),
        vituri2024_scf_callable_manifest(
            "build_scf_problem", getattr(provider, "build_scf_problem")
        ),
    )
    if actual != approval.live_builder_manifests:
        raise ValueError("SCF live state/problem builder source/AST/code manifest drift")

def _make_contract(
    binding: Vituri2024HalfMetalHFProviderBinding,
    approval: Vituri2024SCFReplayApproval,
    core: Vituri2024CoreProvenance,
    baseline: dict[str, object],
    authority_baseline: dict[str, str],
) -> Vituri2024SCFReplayContract:
    spec = binding.spec
    assert spec.scf_policy is not None and spec.shared_functional is not None
    assert spec.attested_source is not None
    if approval.core_provenance_mode != core.provenance_mode:
        raise ValueError("detached approval core provenance mode no longer matches runtime")
    if approval.core_baseline_commit_authority != core.baseline_commit_authority:
        raise ValueError("detached approval core baseline authority no longer matches runtime")
    if approval.core_provenance_fingerprint != core.fingerprint:
        raise ValueError("detached approval core provenance no longer matches runtime")
    if approval.core_source_manifests != core.source_manifests or approval.core_callable_identities != core.callable_identities:
        raise ValueError("detached approval core source/callable manifest drift")
    versions = (
        approval.package_version,
        approval.python_version,
        approval.python_implementation,
        approval.numpy_version,
    )
    if versions != (
        core.package_version,
        core.python_version,
        core.python_implementation,
        core.numpy_version,
    ):
        raise ValueError("detached approval package/Python/NumPy runtime drift")
    if approval.spec_fingerprint != spec.fingerprint or approval.scf_policy_fingerprint != spec.scf_policy.fingerprint:
        raise ValueError("detached approval spec/SCF policy drift")
    if (
        approval.shared_functional_fingerprint != spec.shared_functional.fingerprint
        or approval.attested_source_fingerprint != spec.attested_source.fingerprint
        or approval.source_state_sha256 != spec.attested_source.source_state_sha256
    ):
        raise ValueError("detached approval source/shared-functional drift")
    if approval.scalar_energy_function_fingerprint != spec.shared_functional.scalar_energy.fingerprint:
        raise ValueError("detached approval scalar-energy function drift")
    if approval.exact_seed_inventory != spec.scf_policy.seed_records:
        raise ValueError("detached approval seed order/inventory drift")
    _validate_provider_snapshot(binding, approval, baseline)
    _validate_archive_authority_snapshot(approval, authority_baseline)
    if approval.archive_authority_source_artifact_sha256 != approval.source_artifact_sha256:
        raise ValueError("archive authority and live provider bind different source artifacts")
    return Vituri2024SCFReplayContract(
        approval_fingerprint=approval.fingerprint,
        core_provenance_mode=core.provenance_mode,
        core_baseline_commit_authority=core.baseline_commit_authority,
        core_provenance_fingerprint=core.fingerprint,
        verifier_module_ast_manifest_sha256=approval.verifier_module_ast_manifest_sha256,
        expected_archive_manifest_sha256=approval.expected_archive_manifest_sha256,
        expected_branch_table_sha256=approval.expected_branch_table_sha256,
        exact_seed_inventory=approval.exact_seed_inventory,
        archive_authority_fingerprint=approval.archive_authority_fingerprint,
        archive_authority_metadata_fingerprint=_fingerprint(authority_baseline),
        tolerance_inventory_fingerprint=_fingerprint([asdict(item) for item in approval.tolerances]),
        provider_metadata_fingerprint=_fingerprint(baseline),
    )


def _observable_manifest_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.all(np.isfinite(value)):
            raise ValueError("deterministic manifests require finite non-object arrays")
        return _array_manifest(value)
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (Integral, Real, np.integer, np.floating)):
        result = float(value) if isinstance(value, (Real, np.floating)) else int(value)
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("deterministic manifest scalar must be finite")
        return result
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("deterministic manifest complex scalar must be finite")
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): _observable_manifest_value(item) for key, item in sorted(value.items())}
    raise TypeError(f"unsupported density-update observable type: {type(value)!r}")


def _observables_sha256(observables: object) -> str:
    if not isinstance(observables, dict):
        raise TypeError("DensityUpdateResult.observables must remain a dict")
    return _fingerprint(_observable_manifest_value(observables))


def _diagnostics_sha256(state: object) -> str:
    diagnostics = getattr(state, "diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError("state.diagnostics must remain a dict")
    return _fingerprint(_observable_manifest_value(diagnostics))

def _state_snapshot(state: object) -> Vituri2024SCFStateSnapshot:
    return Vituri2024SCFStateSnapshot(
        h0=np.asarray(getattr(state, "h0")),
        density=np.asarray(getattr(state, "density")),
        hamiltonian=np.asarray(getattr(state, "hamiltonian")),
        energies=np.asarray(getattr(state, "energies")),
        mu=getattr(state, "mu"),
        precision=getattr(state, "precision"),
        diagnostics_manifest_sha256=_diagnostics_sha256(state),
    )


def _state_mutation_digest(state: object) -> str:
    snapshot = _state_snapshot(state)
    return _fingerprint(_snapshot_manifest(snapshot))


def _array_digest(value: object) -> str:
    if not isinstance(value, np.ndarray):
        raise TypeError("callback array input changed type")
    return canonical_array_sha256(value)


def _density_update_mutation_digest(update: object) -> str:
    if type(update) is not DensityUpdateResult:
        raise TypeError("callback density update input changed type")
    return _fingerprint(
        {
            "density": _array_manifest(update.density),
            "energies": _array_manifest(update.energies),
            "mu": _finite(update.mu, "callback density-update mu"),
            "observables": _observables_sha256(update.observables),
        }
    )


def _step_mutation_digest(step: object) -> str:
    if type(step) is not HartreeFockStepResult:
        raise TypeError("callback step input changed type")
    return _fingerprint(
        {
            "iteration": step.iteration,
            "previous_density": _array_manifest(step.previous_density),
            "interaction_h": _array_manifest(step.interaction_h),
            "total_hamiltonian": _array_manifest(step.total_hamiltonian),
            "density_update": _density_update_mutation_digest(step.density_update),
            "mixed_density": _array_manifest(step.mixed_density),
            "oda_lambda": step.oda_lambda,
            "norm_raw": step.norm_raw,
            "norm_mixed": step.norm_mixed,
            "norm_selected": step.norm_selected,
            "energy": step.energy,
            "delta_interaction_h": (
                None
                if step.delta_interaction_h is None
                else _array_manifest(step.delta_interaction_h)
            ),
            "interaction_h_from_cache": step.interaction_h_from_cache,
        }
    )



def _problem_callbacks(problem: HartreeFockProblem) -> dict[str, object | None]:
    return {
        "initializer": problem.initializer,
        "interaction_builder": problem.kernel.interaction_builder,
        "density_builder": problem.kernel.density_builder,
        "energy_functional": problem.kernel.energy_functional,
        "oda_parameterizer": problem.kernel.oda_parameterizer,
        "oda_delta_interaction_builder": problem.kernel.oda_delta_interaction_builder,
        "hamiltonian_postprocessor": problem.kernel.hamiltonian_postprocessor,
        "density_postprocessor": problem.kernel.density_postprocessor,
        "step_callback": problem.kernel.step_callback,
        "final_state_callback": problem.kernel.final_state_callback,
    }


def vituri2024_scf_problem_callback_manifests(
    problem: HartreeFockProblem,
) -> tuple[Vituri2024SCFProblemCallbackManifest, ...]:
    if type(problem) is not HartreeFockProblem:
        raise TypeError("callback manifests require the exact HartreeFockProblem type")
    result: list[Vituri2024SCFProblemCallbackManifest] = []
    for role, callback in _problem_callbacks(problem).items():
        result.append(
            Vituri2024SCFProblemCallbackManifest(
                role=role,
                implementation_kind="none" if callback is None else "callable",
                callable_manifest=(
                    None
                    if callback is None
                    else vituri2024_scf_callable_manifest(role, callback)
                ),
            )
        )
    return tuple(result)

def _validate_problem_callbacks(
    problem: HartreeFockProblem,
    policy: object,
    approval: Vituri2024SCFReplayApproval,
) -> None:
    callbacks = _problem_callbacks(problem)
    receipts = getattr(policy, "callback_receipts")
    if tuple(callbacks) != _PROBLEM_CALLBACK_ROLES:
        raise ValueError("problem callback role inventory changed")
    if tuple(callbacks) != tuple(item.role for item in receipts):
        raise ValueError("problem callback/policy role inventory changed")
    for receipt in receipts:
        callback = callbacks[receipt.role]
        if receipt.implementation_kind == "none" and callback is not None:
            raise ValueError(f"{receipt.role} callback changed from None")
        if receipt.implementation_kind == "callable" and not callable(callback):
            raise TypeError(f"{receipt.role} callback must be callable")
    if callbacks["step_callback"] is not None or callbacks["final_state_callback"] is not None:
        raise ValueError("SCF replay v1 requires provider step/final callbacks are None")
    actual = vituri2024_scf_problem_callback_manifests(problem)
    if actual != approval.problem_callback_manifests:
        raise ValueError("SCF problem callback source/AST/code manifest drift")


class _TrajectoryRecorder:
    def __init__(
        self,
        provider: object,
        baseline: dict[str, object],
        expected: Vituri2024SCFSeedTrajectoryArchive,
    ) -> None:
        self.provider = provider
        self.baseline = baseline
        self.expected = expected
        self.callback_sequence: list[str] = []
        self.pre_init: Vituri2024SCFStateSnapshot | None = None
        self.post_init: Vituri2024SCFStateSnapshot | None = None
        self.steps: list[Vituri2024SCFStepArchive] = []
        self.final: Vituri2024SCFFinalRecomputationArchive | None = None
        self.density_postprocessor_outputs: list[np.ndarray] = []

    def invoke(
        self,
        role: str,
        callback: object,
        *args: object,
        immutable_array_args: tuple[int, ...] = (),
        immutable_state_args: tuple[int, ...] = (),
        immutable_step_args: tuple[int, ...] = (),
        immutable_update_args: tuple[int, ...] = (),
        **kwargs: object,
    ) -> object:
        _assert_snapshot_unchanged(self.provider, self.baseline, f"before {role}")
        array_before = {index: _array_digest(args[index]) for index in immutable_array_args}
        state_before = {index: _state_mutation_digest(args[index]) for index in immutable_state_args}
        step_before = {index: _step_mutation_digest(args[index]) for index in immutable_step_args}
        update_before = {
            index: _density_update_mutation_digest(args[index])
            for index in immutable_update_args
        }
        self.callback_sequence.append(role)
        result = callback(*args, **kwargs)  # type: ignore[operator]
        _assert_snapshot_unchanged(self.provider, self.baseline, role)
        for index, digest in array_before.items():
            if _array_digest(args[index]) != digest:
                raise ValueError(f"provider mutated verifier input during {role}")
        for index, digest in state_before.items():
            if _state_mutation_digest(args[index]) != digest:
                raise ValueError(f"provider mutated verifier state during {role}")
        for index, digest in step_before.items():
            if _step_mutation_digest(args[index]) != digest:
                raise ValueError(f"provider mutated verifier step input during {role}")
        for index, digest in update_before.items():
            if _density_update_mutation_digest(args[index]) != digest:
                raise ValueError(f"provider mutated verifier density-update input during {role}")
        return result

    def capture_step(self, state: object, step: HartreeFockStepResult) -> None:
        update = step.density_update
        self.steps.append(
            Vituri2024SCFStepArchive(
                iteration=step.iteration,
                previous_density=np.asarray(step.previous_density),
                interaction_h=np.asarray(step.interaction_h),
                total_hamiltonian=np.asarray(step.total_hamiltonian),
                raw_density=np.asarray(update.density),
                raw_energies=np.asarray(update.energies),
                raw_mu=update.mu,
                density_update_observables_sha256=_observables_sha256(update.observables),
                mixed_density=np.asarray(step.mixed_density),
                state_density=np.asarray(getattr(state, "density")),
                state_hamiltonian=np.asarray(getattr(state, "hamiltonian")),
                state_energies=np.asarray(getattr(state, "energies")),
                state_mu=getattr(state, "mu"),
                state_diagnostics_manifest_sha256=_diagnostics_sha256(state),
                delta_interaction_h=(
                    None if step.delta_interaction_h is None else np.asarray(step.delta_interaction_h)
                ),
                oda_lambda=step.oda_lambda,
                norm_raw=step.norm_raw,
                norm_mixed=step.norm_mixed,
                norm_selected=step.norm_selected,
                energy=step.energy,
                interaction_h_from_cache=step.interaction_h_from_cache,
            )
        )

    def capture_final(self, state: object, update: DensityUpdateResult) -> None:
        raw_density = np.asarray(update.density)
        if self.density_postprocessor_outputs:
            raw_density = self.density_postprocessor_outputs[-1]
        diagnostics = getattr(state, "diagnostics")
        self.final = Vituri2024SCFFinalRecomputationArchive(
            h0=np.asarray(getattr(state, "h0")),
            state_density=np.asarray(getattr(state, "density")),
            effective_interaction_h=np.asarray(getattr(state, "hamiltonian")) - np.asarray(getattr(state, "h0")),
            total_hamiltonian=np.asarray(getattr(state, "hamiltonian")),
            raw_density=raw_density,
            energies=np.asarray(getattr(state, "energies")),
            mu=getattr(state, "mu"),
            energy=diagnostics["hf_energy"],
            raw_norm=diagnostics["final_raw_norm"],
            density_update_observables_sha256=_observables_sha256(update.observables),
            state_diagnostics_manifest_sha256=_diagnostics_sha256(state),
        )


def _wrapped_problem(
    problem: HartreeFockProblem,
    recorder: _TrajectoryRecorder,
) -> HartreeFockProblem:
    original = _problem_callbacks(problem)

    def initializer(state: object, *, init_mode: str, seed: int) -> None:
        recorder.pre_init = _state_snapshot(state)
        recorder.invoke("initializer", original["initializer"], state, init_mode=init_mode, seed=seed)
        recorder.post_init = _state_snapshot(state)

    def interaction_builder(density: np.ndarray) -> np.ndarray:
        result = recorder.invoke(
            "interaction_builder", original["interaction_builder"], density, immutable_array_args=(0,)
        )
        return np.asarray(result)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        result = recorder.invoke(
            "density_builder", original["density_builder"], hamiltonian, immutable_array_args=(0,)
        )
        if type(result) is not DensityUpdateResult:
            raise TypeError("density builder must return the exact DensityUpdateResult type")
        return result

    def energy_functional(interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray) -> float:
        return float(
            recorder.invoke(
                "energy_functional",
                original["energy_functional"],
                interaction_h,
                h0,
                density,
                immutable_array_args=(0, 1, 2),
            )
        )

    oda_parameterizer = None
    if original["oda_parameterizer"] is not None:
        def oda_parameterizer(state: object, delta_density: np.ndarray) -> float:
            return float(
                recorder.invoke(
                    "oda_parameterizer",
                    original["oda_parameterizer"],
                    state,
                    delta_density,
                    immutable_array_args=(1,),
                    immutable_state_args=(0,),
                )
            )

    oda_delta_interaction_builder = None
    if original["oda_delta_interaction_builder"] is not None:
        def oda_delta_interaction_builder(delta_density: np.ndarray) -> np.ndarray:
            return np.asarray(
                recorder.invoke(
                    "oda_delta_interaction_builder",
                    original["oda_delta_interaction_builder"],
                    delta_density,
                    immutable_array_args=(0,),
                )
            )

    hamiltonian_postprocessor = None
    if original["hamiltonian_postprocessor"] is not None:
        def hamiltonian_postprocessor(hamiltonian: np.ndarray) -> None:
            recorder.invoke("hamiltonian_postprocessor", original["hamiltonian_postprocessor"], hamiltonian)

    density_postprocessor = None
    if original["density_postprocessor"] is not None:
        def density_postprocessor(density: np.ndarray) -> None:
            recorder.invoke("density_postprocessor", original["density_postprocessor"], density)
            recorder.density_postprocessor_outputs.append(np.asarray(density).copy())

    def step_observer(state: object, step: HartreeFockStepResult) -> None:
        # Provider step callbacks are forbidden in v1; this observer is verifier-only.
        recorder.capture_step(state, step)

    def final_observer(state: object, update: DensityUpdateResult) -> None:
        # Provider final callbacks are forbidden in v1; this observer is verifier-only.
        recorder.capture_final(state, update)

    return HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            oda_parameterizer=oda_parameterizer,
            oda_delta_interaction_builder=oda_delta_interaction_builder,
            hamiltonian_postprocessor=hamiltonian_postprocessor,
            density_postprocessor=density_postprocessor,
            step_callback=step_observer,
            final_state_callback=final_observer,
            convergence_rule=problem.kernel.convergence_rule,
        ),
    )


def _archive_arrays_by_seed(archive: Vituri2024ImmutableHistoricalSCFArchive) -> list[list[np.ndarray]]:
    result: list[list[np.ndarray]] = []
    for trajectory in archive.seed_trajectories:
        arrays: list[np.ndarray] = []
        for snapshot in (trajectory.pre_init, trajectory.post_init):
            arrays.extend((snapshot.h0, snapshot.density, snapshot.hamiltonian, snapshot.energies))
        for step in trajectory.steps:
            for item in fields(step):
                value = getattr(step, item.name)
                if isinstance(value, np.ndarray):
                    arrays.append(value)
        for item in fields(trajectory.final_recomputation):
            value = getattr(trajectory.final_recomputation, item.name)
            if isinstance(value, np.ndarray):
                arrays.append(value)
        result.append(arrays)
    return result


def _reject_archive_shared_storage(archive: Vituri2024ImmutableHistoricalSCFArchive) -> None:
    by_seed = _archive_arrays_by_seed(archive)
    selected_arrays = [
        archive.selected_source.h0,
        archive.selected_source.effective_interaction_h,
        archive.selected_source.fock,
        archive.selected_source.projector,
        archive.selected_source.energies,
    ]
    all_arrays = [array for group in by_seed for array in group] + selected_arrays
    for index, left in enumerate(all_arrays):
        if any(np.shares_memory(left, right) for right in all_arrays[index + 1 :]):
            raise ValueError("historical archive contains shared array storage")


def _reject_live_shared_storage(
    state: object,
    previous_states: list[object],
    archive: Vituri2024ImmutableHistoricalSCFArchive,
) -> None:
    live = [
        np.asarray(getattr(state, name)) for name in ("h0", "density", "hamiltonian", "energies")
    ]
    archived = [array for group in _archive_arrays_by_seed(archive) for array in group]
    archived.extend(
        (
            archive.selected_source.h0,
            archive.selected_source.effective_interaction_h,
            archive.selected_source.fock,
            archive.selected_source.projector,
            archive.selected_source.energies,
        )
    )
    if any(np.shares_memory(left, right) for left in live for right in archived):
        raise ValueError("live SCF state shares storage with historical archive")
    for previous in previous_states:
        old = [np.asarray(getattr(previous, name)) for name in ("h0", "density", "hamiltonian", "energies")]
        if any(np.shares_memory(left, right) for left in live for right in old):
            raise ValueError("live SCF states share storage across seeds")


def _tolerance_map(approval: Vituri2024SCFReplayApproval) -> dict[str, Vituri2024SCFReplayTolerance]:
    return {item.field_name: item for item in approval.tolerances}


def _compare_numeric(
    field_name: str,
    actual: object,
    expected: object,
    tolerances: dict[str, Vituri2024SCFReplayTolerance],
) -> tuple[bool, bool]:
    tolerance = tolerances[field_name]
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"{field_name} shape/dtype mismatch")
        close = bool(np.allclose(actual, expected, atol=tolerance.absolute, rtol=tolerance.relative, equal_nan=False))
        hash_equal = canonical_array_sha256(actual) == canonical_array_sha256(expected)
        bitwise_equal = actual.tobytes(order="C") == expected.tobytes(order="C")
    else:
        actual_float = _finite(actual, field_name)
        expected_float = _finite(expected, field_name)
        scale = max(abs(actual_float), abs(expected_float))
        close = abs(actual_float - expected_float) <= tolerance.absolute + tolerance.relative * scale
        actual_bytes = np.float64(actual_float).tobytes()
        expected_bytes = np.float64(expected_float).tobytes()
        hash_equal = hashlib.sha256(actual_bytes).digest() == hashlib.sha256(expected_bytes).digest()
        bitwise_equal = actual_bytes == expected_bytes
    if not close:
        raise ValueError(f"{field_name} exceeds registered scale-aware tolerance")
    if tolerance.require_canonical_hash and not hash_equal:
        raise ValueError(f"{field_name} canonical hash mismatch")
    if tolerance.require_bitwise_parity and not bitwise_equal:
        raise ValueError(f"{field_name} bitwise parity mismatch")
    return hash_equal, bitwise_equal


def _compare_snapshot(
    prefix: str,
    actual: Vituri2024SCFStateSnapshot,
    expected: Vituri2024SCFStateSnapshot,
    tolerances: dict[str, Vituri2024SCFReplayTolerance],
) -> list[tuple[bool, bool]]:
    if actual.diagnostics_manifest_sha256 != expected.diagnostics_manifest_sha256:
        raise ValueError(f"{prefix} state.diagnostics manifest mismatch")
    return [
        _compare_numeric(f"{prefix}.{name}", getattr(actual, name), getattr(expected, name), tolerances)
        for name in ("h0", "density", "hamiltonian", "energies", "mu", "precision")
    ]


def _compare_trajectory(
    actual_pre: Vituri2024SCFStateSnapshot,
    actual_post: Vituri2024SCFStateSnapshot,
    actual_steps: tuple[Vituri2024SCFStepArchive, ...],
    actual_final: Vituri2024SCFFinalRecomputationArchive,
    callback_sequence: tuple[str, ...],
    run: object,
    expected: Vituri2024SCFSeedTrajectoryArchive,
    tolerances: dict[str, Vituri2024SCFReplayTolerance],
) -> list[tuple[bool, bool]]:
    parity = _compare_snapshot("pre_init", actual_pre, expected.pre_init, tolerances)
    parity.extend(_compare_snapshot("post_init", actual_post, expected.post_init, tolerances))
    if len(actual_steps) != len(expected.steps):
        raise ValueError("SCF trajectory step count mismatch")
    scalar_fields = (
        "raw_mu", "state_mu", "oda_lambda", "norm_raw", "norm_mixed",
        "norm_selected", "energy",
    )
    array_fields = (
        "previous_density", "interaction_h", "total_hamiltonian", "raw_density",
        "raw_energies", "mixed_density", "state_density", "state_hamiltonian", "state_energies",
    )
    for actual, reference in zip(actual_steps, expected.steps):
        if actual.iteration != reference.iteration:
            raise ValueError("SCF step iteration mismatch")
        if actual.interaction_h_from_cache != reference.interaction_h_from_cache:
            raise ValueError("SCF interaction cache flag mismatch")
        if actual.density_update_observables_sha256 != reference.density_update_observables_sha256:
            raise ValueError("SCF density-update observables mismatch")
        if actual.state_diagnostics_manifest_sha256 != reference.state_diagnostics_manifest_sha256:
            raise ValueError("SCF step state.diagnostics manifest mismatch")
        if (actual.delta_interaction_h is None) != (reference.delta_interaction_h is None):
            raise ValueError("SCF delta interaction None/present mismatch")
        for name in array_fields:
            parity.append(_compare_numeric(f"step.{name}", getattr(actual, name), getattr(reference, name), tolerances))
        if actual.delta_interaction_h is not None:
            parity.append(_compare_numeric("step.delta_interaction_h", actual.delta_interaction_h, reference.delta_interaction_h, tolerances))
        for name in scalar_fields:
            parity.append(_compare_numeric(f"step.{name}", getattr(actual, name), getattr(reference, name), tolerances))
    final_fields = (
        "h0", "state_density", "effective_interaction_h", "total_hamiltonian",
        "raw_density", "energies", "mu", "energy", "raw_norm",
    )
    if actual_final.density_update_observables_sha256 != expected.final_recomputation.density_update_observables_sha256:
        raise ValueError("final recomputation observables mismatch")
    if (
        actual_final.state_diagnostics_manifest_sha256
        != expected.final_recomputation.state_diagnostics_manifest_sha256
    ):
        raise ValueError("final recomputation state.diagnostics manifest mismatch")
    for name in final_fields:
        parity.append(_compare_numeric(f"final.{name}", getattr(actual_final, name), getattr(expected.final_recomputation, name), tolerances))
    if callback_sequence != expected.callback_sequence:
        raise ValueError("SCF callback sequence mismatch")
    if getattr(run, "iterations") != expected.iterations or getattr(run, "exit_reason") != expected.exit_reason or getattr(run, "converged") != expected.converged:
        raise ValueError("SCF iteration/exit/converged mismatch")
    if getattr(run, "init_mode") != expected.seed.init_mode or getattr(run, "seed") != expected.seed.seed_value:
        raise ValueError("SCF run seed identity mismatch")
    parity.append(_compare_numeric("run.iter_energy", np.asarray(getattr(run, "iter_energy")), np.asarray([item.energy for item in expected.steps], dtype=np.float64), tolerances))
    parity.append(_compare_numeric("run.iter_err", np.asarray(getattr(run, "iter_err")), np.asarray([item.norm_selected for item in expected.steps], dtype=np.float64), tolerances))
    parity.append(_compare_numeric("run.iter_oda", np.asarray(getattr(run, "iter_oda")), np.asarray([item.oda_lambda for item in expected.steps], dtype=np.float64), tolerances))
    return parity


@dataclass(frozen=True, slots=True)
class Vituri2024RestartCapabilityAudit:
    public_continuation_api_available: bool = field(default=False, init=False)
    cached_interaction_h_publicly_exposed: bool = field(default=False, init=False)
    rng_state_captured: bool = field(default=False, init=False)
    callback_state_captured: bool = field(default=False, init=False)
    checkpoint_snapshot_payloads_present: bool = field(default=False, init=False)
    exact_restart_verified: bool = field(default=False, init=False)
    blocker: str = (
        "HEAD HartreeFockProblem/run_hartree_fock_problem has no public continuation API; "
        "run_hartree_fock_iterations keeps cached_interaction_h local and exposes no RNG or callback continuation state."
    )

    def __post_init__(self) -> None:
        if any(
            (
                self.public_continuation_api_available,
                self.cached_interaction_h_publicly_exposed,
                self.rng_state_captured,
                self.callback_state_captured,
                self.checkpoint_snapshot_payloads_present,
                self.exact_restart_verified,
            )
        ):
            raise ValueError("restart capability audit cannot become positive on this HEAD")
        _text(self.blocker, "restart capability blocker")


_SUCCESS_TOKEN = object()


@dataclass(frozen=True, slots=True)
class Vituri2024SCFReplayStatus:
    """Trusted-provider deterministic-parity status, not independence evidence."""

    _factory_token: InitVar[object | None] = None
    evidence_model: Literal[
        "trusted_live_provider_distinct_archive_object"
    ] = field(default=_SCF_REPLAY_EVIDENCE_MODEL, init=False)
    archive_data_independence_verified: bool = field(default=False, init=False)
    hostile_provider_resistance_verified: bool = field(default=False, init=False)
    live_builder_dependency_state_independently_pinned: bool = field(
        default=False, init=False
    )
    uninterrupted_registered_seed_trajectories_replayed: bool = field(default=True, init=False)
    all_attested_seed_branches_replayed: bool = field(default=True, init=False)
    branch_table_replayed: bool = field(default=True, init=False)
    selected_final_source_reproduced: bool = field(default=True, init=False)
    global_ground_state_verified: bool = field(default=False, init=False)
    transfer_learning_physics_verified: bool = field(default=False, init=False)
    checkpoint_snapshot_hash_verified: bool = field(default=False, init=False)
    atomic_checkpoint_publication_verified: bool = field(default=False, init=False)
    exact_restart_verified: bool = field(default=False, init=False)
    interrupted_vs_uninterrupted_trajectory_equivalent: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _SUCCESS_TOKEN:
            raise ValueError("successful SCF replay status is factory-only")
        if self.evidence_model != _SCF_REPLAY_EVIDENCE_MODEL:
            raise ValueError("SCF replay evidence model changed")
        positives = (
            self.uninterrupted_registered_seed_trajectories_replayed,
            self.all_attested_seed_branches_replayed,
            self.branch_table_replayed,
            self.selected_final_source_reproduced,
        )
        negatives = (
            self.archive_data_independence_verified,
            self.hostile_provider_resistance_verified,
            self.live_builder_dependency_state_independently_pinned,
            self.global_ground_state_verified,
            self.transfer_learning_physics_verified,
            self.checkpoint_snapshot_hash_verified,
            self.atomic_checkpoint_publication_verified,
            self.exact_restart_verified,
            self.interrupted_vs_uninterrupted_trajectory_equivalent,
            self.scientific_execution_verified,
            self.paper_reproduction_verified,
        )
        if not all(positives) or any(negatives):
            raise ValueError("SCF replay status claims changed")


@dataclass(frozen=True, slots=True)
class Vituri2024SCFReplayReceipt:
    """Parity receipt scoped to the explicit trusted-live-provider model."""

    approval_fingerprint: str
    contract_fingerprint: str
    archive_manifest_sha256: str
    branch_table_sha256: str
    core_provenance_mode: Literal[
        "git_ancestor_head_index_worktree_verified",
        "pinned_hash_verified_source_export",
    ]
    core_baseline_commit_authority: Literal[
        "hardcoded_vituri2024_scf_baseline_commit_and_core_hash_manifest"
    ]
    core_provenance_fingerprint: str
    verifier_module_ast_manifest_sha256: str
    archive_authority_fingerprint: str
    live_provider_metadata_fingerprint: str
    seed_order: tuple[str, ...]
    archive_authority_outer_call_sequence: tuple[str, ...]
    provider_outer_call_sequence: tuple[str, ...]
    replayed_branch_energies_ev: tuple[float, ...]
    converged_branch_labels: tuple[str, ...]
    tolerance_degenerate_minimum_labels: tuple[str, ...]
    selected_branch_label: str
    selected_branch_residual_ev: float
    canonical_hash_comparison_count: int
    canonical_hash_equal_count: int
    bitwise_comparison_count: int
    bitwise_equal_count: int
    effective_tolerances: tuple[Vituri2024SCFReplayTolerance, ...]
    unique_ground_state_claimed: bool
    restart_capability_audit: Vituri2024RestartCapabilityAudit
    status: Vituri2024SCFReplayStatus
    evidence_model: Literal[
        "trusted_live_provider_distinct_archive_object"
    ] = field(default=_SCF_REPLAY_EVIDENCE_MODEL, init=False)
    archive_data_independence_verified: bool = field(default=False, init=False)
    hostile_provider_resistance_verified: bool = field(default=False, init=False)
    live_builder_dependency_state_independently_pinned: bool = field(
        default=False, init=False
    )
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _SUCCESS_TOKEN:
            raise ValueError("successful SCF replay receipt is factory-only")
        if self.core_provenance_mode not in (
            _CORE_PROVENANCE_GIT_MODE,
            _CORE_PROVENANCE_SOURCE_EXPORT_MODE,
        ):
            raise ValueError("unsupported receipt SCF core provenance mode")
        if self.core_baseline_commit_authority != _CORE_BASELINE_COMMIT_AUTHORITY:
            raise ValueError("SCF replay receipt baseline authority changed")
        for value, label in (
            (self.approval_fingerprint, "receipt approval"),
            (self.contract_fingerprint, "receipt contract"),
            (self.archive_manifest_sha256, "receipt archive manifest"),
            (self.branch_table_sha256, "receipt branch table"),
            (self.core_provenance_fingerprint, "receipt core provenance"),
            (self.verifier_module_ast_manifest_sha256, "receipt verifier AST"),
            (self.archive_authority_fingerprint, "receipt archive authority"),
            (self.live_provider_metadata_fingerprint, "receipt live-provider metadata"),
        ):
            _sha256(value, label)
        if tuple(self.effective_tolerances) != _LOCKED_V1_TOLERANCES:
            raise ValueError("receipt effective tolerance/hash flags differ from locked v1")
        if self.unique_ground_state_claimed:
            raise ValueError("SCF replay must not claim a unique/global ground state")
        if self.evidence_model != _SCF_REPLAY_EVIDENCE_MODEL or any(
            (
                self.archive_data_independence_verified,
                self.hostile_provider_resistance_verified,
                self.live_builder_dependency_state_independently_pinned,
            )
        ):
            raise ValueError("SCF replay receipt evidence limitations changed")
        if type(self.restart_capability_audit) is not Vituri2024RestartCapabilityAudit:
            raise TypeError("SCF replay receipt requires a restart capability audit")
        if type(self.status) is not Vituri2024SCFReplayStatus:
            raise TypeError("SCF replay receipt requires a factory status")
        if (
            self.status.evidence_model != self.evidence_model
            or self.status.archive_data_independence_verified
            != self.archive_data_independence_verified
            or self.status.hostile_provider_resistance_verified
            != self.hostile_provider_resistance_verified
            or self.status.live_builder_dependency_state_independently_pinned
            != self.live_builder_dependency_state_independently_pinned
        ):
            raise ValueError("SCF replay status/receipt evidence scope mismatch")


def _validate_archive_against_source(
    binding: Vituri2024HalfMetalHFProviderBinding,
    approval: Vituri2024SCFReplayApproval,
    archive: Vituri2024ImmutableHistoricalSCFArchive,
) -> None:
    spec = binding.spec
    source = spec.attested_source
    policy = spec.scf_policy
    assert source is not None and policy is not None
    if archive.generation_phase != SCF_REPLAY_ARCHIVE_GENERATION_PHASE:
        raise ValueError("historical archive was generated after state/problem builders")
    expected_identity = (
        (
            archive.archive_authority_fingerprint,
            approval.archive_authority_fingerprint,
            "archive authority",
        ),
        (archive.source_commit, approval.source_commit, "source commit"),
        (archive.source_artifact_sha256, approval.source_artifact_sha256, "source artifact"),
        (archive.spec_fingerprint, approval.spec_fingerprint, "spec"),
        (
            archive.archive_loader_implementation_fingerprint,
            approval.archive_authority_loader_implementation_fingerprint,
            "archive-authority loader",
        ),
        (archive.archive_schema_fingerprint, approval.archive_schema_fingerprint, "archive schema"),
        (archive.original_branch_table_sha256, approval.expected_branch_table_sha256, "branch table approval"),
        (archive.original_branch_table_sha256, source.branch_energy_table_sha256, "branch table source"),
        (archive.selected_branch_label, source.selected_branch_label, "selected branch"),
        (archive.selected_source.source_state_sha256, source.source_state_sha256, "selected source state"),
    )
    for actual, expected, label in expected_identity:
        if actual != expected:
            raise ValueError(f"historical archive {label} mismatch")
    if tuple(item.seed for item in archive.seed_trajectories) != policy.seed_records:
        raise ValueError("historical archive seed order/inventory mismatch")
    if archive.branch_records != source.branch_records:
        raise ValueError("historical archive branch rows differ from attested source")
    selected_hashes = dict(archive.selected_source.registered_hashes)
    expected_hashes = {
        "h0": source.h0_sha256,
        "effective_interaction_h": source.interaction_h_sha256,
        "fock": source.ordered_fock_sha256,
        "projector": source.ordered_projector_sha256,
        "energies": source.ordered_energies_sha256,
    }
    if selected_hashes != expected_hashes:
        raise ValueError("historical selected-source hashes do not close to canonical source")
    if archive.selected_source.mu != source.chemical_potential_ev:
        raise ValueError("historical selected-source chemical potential mismatch")
    manifest = scf_archive_manifest_sha256(archive)
    if manifest != approval.expected_archive_manifest_sha256:
        raise ValueError("historical archive manifest differs from detached approval")
    _reject_archive_shared_storage(archive)


def replay_vituri2024_half_metal_hf_scf(
    binding: Vituri2024HalfMetalHFProviderBinding,
    archive_authority: Vituri2024SCFArchiveAuthorityProtocol,
    approval: Vituri2024SCFReplayApproval,
) -> Vituri2024SCFReplayReceipt:
    """Replay all policy seeds uninterrupted through the actual baseline core."""

    if type(binding) is not Vituri2024HalfMetalHFProviderBinding:
        raise TypeError("SCF replay requires the exact provider binding")
    if type(approval) is not Vituri2024SCFReplayApproval:
        raise TypeError("SCF replay requires a detached typed approval")

    # Pin the problem-module facade alias before inspecting or calling either
    # authority.  run_hartree_fock_problem resolves this imported alias.
    core = verified_vituri2024_core_provenance()
    provider = binding.provider
    if provider is archive_authority:
        raise TypeError("archive authority and live provider must be distinct objects")
    if not isinstance(archive_authority, Vituri2024SCFArchiveAuthorityProtocol):
        raise TypeError("archive authority is missing the detached archive protocol")
    if not isinstance(provider, Vituri2024SCFReplayProviderProtocol):
        raise TypeError("provider is missing the functional-derived live SCF replay protocol")
    if callable(getattr(provider, "load_immutable_scf_archive", None)):
        raise TypeError("live SCF provider cannot own the immutable SCF archive loader")
    for forbidden in ("build_fresh_scf_state", "build_scf_problem"):
        if callable(getattr(archive_authority, forbidden, None)):
            raise TypeError("SCF archive authority cannot own live state/problem builders")
    for forbidden in ("run_scf", "run_hartree_fock_problem", "replay_scf"):
        if callable(getattr(provider, forbidden, None)):
            raise TypeError("SCF provider cannot expose or run the SCF entrypoint")

    # All remaining approval checks and the contract are complete before the
    # first archive/live-builder method call.
    baseline = _provider_snapshot(provider)
    authority_baseline = _archive_authority_snapshot(archive_authority)
    _reject_same_authority_fingerprint(authority_baseline, baseline)
    contract = _make_contract(
        binding, approval, core, baseline, authority_baseline
    )
    authority_outer_calls: list[str] = []
    outer_calls: list[str] = []

    # The detached archive is loaded and fully validated before either live
    # state/problem builder is invoked.
    authority_outer_calls.append("load_immutable_scf_archive")
    _assert_archive_authority_snapshot_unchanged(
        archive_authority, authority_baseline, "before historical archive loader"
    )
    archive = archive_authority.load_immutable_scf_archive(
        approval.archive_authority_source_artifact_sha256
    )
    _assert_archive_authority_snapshot_unchanged(
        archive_authority, authority_baseline, "historical archive loader"
    )
    _assert_snapshot_unchanged(provider, baseline, "historical archive load")
    if type(archive) is not Vituri2024ImmutableHistoricalSCFArchive:
        raise TypeError("archive loader must return the exact immutable historical archive type")
    _validate_archive_against_source(binding, approval, archive)
    _validate_live_builder_manifests(provider, approval)

    spec = binding.spec
    policy = spec.scf_policy
    source = spec.attested_source
    assert policy is not None and source is not None
    tolerances = _tolerance_map(approval)
    live_states: list[object] = []
    parity: list[tuple[bool, bool]] = []
    replayed_energies: list[float] = []
    actual_finals: dict[str, Vituri2024SCFFinalRecomputationArchive] = {}
    actual_branch_facts: dict[
        str,
        tuple[object, Vituri2024SCFStepArchive, Vituri2024SCFFinalRecomputationArchive],
    ] = {}

    for expected in archive.seed_trajectories:
        seed = expected.seed
        outer_calls.append(f"build_fresh_scf_state:{seed.seed_label}")
        _assert_snapshot_unchanged(provider, baseline, f"before state builder {seed.seed_label}")
        state = provider.build_fresh_scf_state(seed)
        _assert_snapshot_unchanged(provider, baseline, f"state builder {seed.seed_label}")
        _reject_live_shared_storage(state, live_states, archive)
        state_before_problem = _state_mutation_digest(state)

        outer_calls.append(f"build_scf_problem:{seed.seed_label}")
        _assert_snapshot_unchanged(provider, baseline, f"before problem builder {seed.seed_label}")
        problem = provider.build_scf_problem(state, seed)
        _assert_snapshot_unchanged(provider, baseline, f"problem builder {seed.seed_label}")
        if _state_mutation_digest(state) != state_before_problem:
            raise ValueError("problem builder mutated its live-state input")
        if type(problem) is not HartreeFockProblem:
            raise TypeError("problem builder must return the exact HartreeFockProblem type")
        _validate_problem_callbacks(problem, policy, approval)
        if problem.kernel.convergence_rule != policy.convergence_rule:
            raise ValueError("problem convergence rule differs from exact SCF policy")
        if getattr(state, "precision") != policy.precision:
            raise ValueError("fresh state precision differs from exact SCF policy")

        recorder = _TrajectoryRecorder(provider, baseline, expected)
        wrapped = _wrapped_problem(problem, recorder)
        run = _hf_problem.run_hartree_fock_problem(
            state,
            wrapped,
            init_mode=seed.init_mode,
            seed=seed.seed_value,
            max_iter=policy.max_iter,
            oda_stall_threshold=policy.oda_stall_threshold,
            max_oda_lambda=policy.max_oda_lambda,
        )
        _assert_snapshot_unchanged(provider, baseline, f"uninterrupted SCF {seed.seed_label}")
        if recorder.pre_init is None or recorder.post_init is None or recorder.final is None:
            raise RuntimeError("SCF verifier observers did not capture complete trajectory")
        parity.extend(
            _compare_trajectory(
                recorder.pre_init,
                recorder.post_init,
                tuple(recorder.steps),
                recorder.final,
                tuple(recorder.callback_sequence),
                run,
                expected,
                tolerances,
            )
        )
        replayed_energies.append(recorder.final.energy)
        actual_finals[seed.seed_label] = recorder.final
        actual_branch_facts[seed.seed_label] = (
            run,
            recorder.steps[-1],
            recorder.final,
        )
        live_states.append(state)

    # First close every branch-table field separately to the detached trajectory
    # and the actual run.  Only then may convergence be classified.
    branch_is_converged: list[bool] = []
    for trajectory, energy, row in zip(
        archive.seed_trajectories, replayed_energies, archive.branch_records
    ):
        run, actual_terminal, actual_final = actual_branch_facts[
            trajectory.seed.seed_label
        ]
        if (
            row.attested_exit_reason != trajectory.exit_reason
            or row.attested_exit_reason != getattr(run, "exit_reason")
        ):
            raise ValueError("branch exit reason does not close to archive and actual run")
        if row.iterations != trajectory.iterations or row.iterations != getattr(run, "iterations"):
            raise ValueError("branch iterations do not close to archive and actual run")
        archived_terminal = trajectory.steps[-1]
        numeric_relations = (
            ("branch.energy", row.canonical_energy_ev, trajectory.final_recomputation.energy, energy),
            ("branch.terminal_norm_raw", row.terminal_norm_raw, archived_terminal.norm_raw, actual_terminal.norm_raw),
            ("branch.terminal_norm_mixed", row.terminal_norm_mixed, archived_terminal.norm_mixed, actual_terminal.norm_mixed),
            ("branch.terminal_norm_selected", row.terminal_norm_selected, archived_terminal.norm_selected, actual_terminal.norm_selected),
            ("branch.terminal_oda_lambda", row.terminal_oda_lambda, archived_terminal.oda_lambda, actual_terminal.oda_lambda),
            ("branch.final_replay_raw_metric", row.final_replay_raw_metric, trajectory.final_recomputation.raw_norm, actual_final.raw_norm),
        )
        for field_name, row_value, archived_value, actual_value in numeric_relations:
            parity.append(_compare_numeric(field_name, archived_value, row_value, tolerances))
            parity.append(_compare_numeric(field_name, actual_value, row_value, tolerances))
        branch_is_converged.append(row.attested_exit_reason == "converged")

    converged_labels: list[str] = []
    converged_energies: list[float] = []
    for trajectory, energy, is_converged in zip(
        archive.seed_trajectories, replayed_energies, branch_is_converged
    ):
        if is_converged:
            converged_labels.append(trajectory.seed.seed_label)
            converged_energies.append(energy)
    if not converged_energies:
        raise ValueError("branch table contains no replayed converged branch")
    minimum = min(converged_energies)
    degenerate_labels = tuple(
        label
        for label, energy in zip(converged_labels, converged_energies)
        if abs(energy - minimum) <= policy.branch_energy_tolerance_ev
    )
    if archive.selected_branch_label not in degenerate_labels:
        raise ValueError("selected branch is not a tolerance-degenerate replayed minimum")
    selected_final = actual_finals[archive.selected_branch_label]
    selected = archive.selected_source
    selected_pairs = (
        ("h0", selected_final.h0, selected.h0),
        ("effective_interaction_h", selected_final.effective_interaction_h, selected.effective_interaction_h),
        ("fock", selected_final.total_hamiltonian, selected.fock),
        ("projector", selected_final.raw_density, selected.projector),
        ("energies", selected_final.energies, selected.energies),
        ("mu", selected_final.mu, selected.mu),
    )
    selected_residual = 0.0
    for name, actual, expected_value in selected_pairs:
        parity.append(_compare_numeric(f"selected.{name}", actual, expected_value, tolerances))
        if isinstance(expected_value, np.ndarray):
            selected_residual = max(selected_residual, float(np.max(np.abs(actual - expected_value))))
        else:
            selected_residual = max(selected_residual, abs(float(actual) - float(expected_value)))
    if abs(selected_final.energy - source.selected_branch_energy_ev) > policy.branch_energy_tolerance_ev:
        raise ValueError("selected final energy differs from canonical source")
    if selected_final.raw_norm > policy.precision:
        raise ValueError("selected final recomputation residual exceeds policy precision")

    hash_equal = sum(int(item[0]) for item in parity)
    bitwise_equal = sum(int(item[1]) for item in parity)
    status = Vituri2024SCFReplayStatus(_factory_token=_SUCCESS_TOKEN)
    return Vituri2024SCFReplayReceipt(
        approval_fingerprint=approval.fingerprint,
        contract_fingerprint=contract.fingerprint,
        archive_manifest_sha256=scf_archive_manifest_sha256(archive),
        branch_table_sha256=archive.original_branch_table_sha256,
        core_provenance_mode=core.provenance_mode,
        core_baseline_commit_authority=core.baseline_commit_authority,
        core_provenance_fingerprint=core.fingerprint,
        verifier_module_ast_manifest_sha256=scf_replay_module_ast_manifest_sha256(),
        archive_authority_fingerprint=approval.archive_authority_fingerprint,
        live_provider_metadata_fingerprint=_fingerprint(baseline),
        seed_order=tuple(item.seed.seed_label for item in archive.seed_trajectories),
        archive_authority_outer_call_sequence=tuple(authority_outer_calls),
        provider_outer_call_sequence=tuple(outer_calls),
        replayed_branch_energies_ev=tuple(replayed_energies),
        converged_branch_labels=tuple(converged_labels),
        tolerance_degenerate_minimum_labels=degenerate_labels,
        selected_branch_label=archive.selected_branch_label,
        selected_branch_residual_ev=selected_residual,
        canonical_hash_comparison_count=len(parity),
        canonical_hash_equal_count=hash_equal,
        bitwise_comparison_count=len(parity),
        bitwise_equal_count=bitwise_equal,
        effective_tolerances=approval.tolerances,
        unique_ground_state_claimed=False,
        restart_capability_audit=Vituri2024RestartCapabilityAudit(),
        status=status,
        _factory_token=_SUCCESS_TOKEN,
    )


__all__ = [
    "SCF_REPLAY_ADAPTER_ABI_FINGERPRINT",
    "SCF_REPLAY_ARCHIVE_AUTHORITY_ABI_FINGERPRINT",
    "SCF_REPLAY_ARCHIVE_AUTHORITY_METADATA_FIELDS",
    "SCF_REPLAY_ARCHIVE_GENERATION_PHASE",
    "SCF_REPLAY_ARCHIVE_SCHEMA_FINGERPRINT",
    "SCF_REPLAY_PROVIDER_METADATA_FIELDS",
    "SCF_REPLAY_SELECTED_HASH_FIELDS",
    "SCF_REPLAY_V1_ABSOLUTE_TOLERANCE_MAXIMUM",
    "SCF_REPLAY_V1_RELATIVE_TOLERANCE_MAXIMUM",
    "SCF_REPLAY_VERIFIER_SCHEMA_FINGERPRINT",
    "VITURI2024_SCF_BASELINE_COMMIT",
    "VITURI2024_SCF_REPLAY_SCOPE",
    "Vituri2024CoreCallableIdentity",
    "Vituri2024CoreProvenance",
    "Vituri2024CoreSourceManifest",
    "Vituri2024ImmutableHistoricalSCFArchive",
    "Vituri2024RestartCapabilityAudit",
    "Vituri2024SCFArchiveAuthorityProtocol",
    "Vituri2024SCFCallableManifest",
    "Vituri2024SCFFinalRecomputationArchive",
    "Vituri2024SCFProblemCallbackManifest",
    "Vituri2024SCFReplayApproval",
    "Vituri2024SCFReplayContract",
    "Vituri2024SCFReplayProviderProtocol",
    "Vituri2024SCFReplayReceipt",
    "Vituri2024SCFReplayStatus",
    "Vituri2024SCFReplayTolerance",
    "Vituri2024SCFSeedTrajectoryArchive",
    "Vituri2024SCFSelectedSource",
    "Vituri2024SCFStateSnapshot",
    "Vituri2024SCFStepArchive",
    "Vituri2024SCFTransferSourceReceipt",
    "default_vituri2024_scf_replay_tolerances",
    "make_vituri2024_scf_replay_approval",
    "replay_vituri2024_half_metal_hf_scf",
    "scf_archive_authority_fingerprint",
    "scf_archive_manifest_sha256",
    "scf_dependency_archive_fingerprint",
    "scf_provider_fingerprint",
    "scf_replay_module_ast_manifest_sha256",
    "verified_vituri2024_core_provenance",
    "vituri2024_scf_callable_manifest",
    "vituri2024_scf_problem_callback_manifests",
]
