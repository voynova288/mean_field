from __future__ import annotations

"""Strict, system-independent contracts for exact saved-grid curves."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Literal, Protocol, runtime_checkable

import numpy as np

BranchNodeStatus = Literal["expanded", "terminal", "rejected"]
BranchClosureAuthority = Literal["supplied_finite_tree_structurally_resolved"]
CurveDomainTopology = Literal["open_interval", "periodic"]

_SOURCE_AUTHORITY_TOKEN = object()


def _require_text(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _require_sha256(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def immutable_finite_array(
    value: object,
    *,
    dtype: np.dtype[object] | str,
    name: str,
    ndim: int = 1,
) -> np.ndarray:
    """Return a C-contiguous finite array backed by immutable bytes."""

    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"{name} must be nonempty, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(contiguous.shape)
    if not frozen.flags.c_contiguous or frozen.flags.writeable:
        raise RuntimeError(f"failed to freeze {name}")
    return frozen


def canonical_array_sha256(value: np.ndarray) -> str:
    """Hash canonical shape, dtype, and C-order bytes for a finite array."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("canonical array hashes do not support object dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("canonical array hashes require finite values")
    canonical = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": [int(size) for size in canonical.shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"mean_field.ndarray.v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: object, *, namespace: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _strict_finite_json(value: object, *, path: str = "payload") -> object:
    """Copy exact JSON-native values while rejecting non-finite or coerced data."""

    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if type(value) is list:
        return [
            _strict_finite_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} object keys must be exact strings")
            result[key] = _strict_finite_json(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"{path} must contain only exact JSON-native values")


@dataclass(frozen=True, init=False)
class SourceAuthorityReceipt:
    """Opaque adapter authority payload bound to immutable canonical JSON."""

    authority_id: str
    canonical_payload_json: str
    payload_sha256: str

    def __init__(
        self,
        *,
        _token: object,
        authority_id: str,
        canonical_payload_json: str,
        payload_sha256: str,
    ) -> None:
        if _token is not _SOURCE_AUTHORITY_TOKEN:
            raise TypeError(
                "SourceAuthorityReceipt is factory-only; use make_source_authority_receipt"
            )
        authority = _require_text(authority_id, name="authority_id")
        canonical = _require_text(canonical_payload_json, name="canonical_payload_json")
        try:
            parsed = json.loads(canonical)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_payload_json is not valid JSON") from exc
        normalized = _strict_finite_json(parsed)
        expected_canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical != expected_canonical:
            raise ValueError("canonical_payload_json is not in canonical form")
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if _require_sha256(payload_sha256, name="payload_sha256") != expected_hash:
            raise ValueError(
                "source-authority payload SHA-256 does not match canonical JSON"
            )
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(self, "canonical_payload_json", canonical)
        object.__setattr__(self, "payload_sha256", expected_hash)

    def validate_live_state(self) -> None:
        SourceAuthorityReceipt(
            _token=_SOURCE_AUTHORITY_TOKEN,
            authority_id=self.authority_id,
            canonical_payload_json=self.canonical_payload_json,
            payload_sha256=self.payload_sha256,
        )

    def payload(self) -> object:
        """Return a fresh JSON value without granting generic semantic authority."""

        self.validate_live_state()
        return json.loads(self.canonical_payload_json)


def make_source_authority_receipt(
    authority_id: str,
    payload: object,
) -> SourceAuthorityReceipt:
    """Canonicalize strict finite JSON without interpreting adapter-owned semantics."""

    normalized = _strict_finite_json(payload)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return SourceAuthorityReceipt(
        _token=_SOURCE_AUTHORITY_TOKEN,
        authority_id=authority_id,
        canonical_payload_json=canonical,
        payload_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class CurveDomainReceipt:
    """Topology of the one-dimensional curve domain.

    Only ``open_interval`` comparison/interpolation is implemented.  Periodic
    domains must still declare their period and seam so no caller can silently
    apply open-interval behavior to wrapped data.
    """

    topology: CurveDomainTopology
    period: float | None = None
    seam: float | None = None

    def __post_init__(self) -> None:
        if self.topology not in {"open_interval", "periodic"}:
            raise ValueError(f"unsupported curve-domain topology {self.topology!r}")
        if self.topology == "open_interval":
            if self.period is not None or self.seam is not None:
                raise ValueError("open_interval domains must not declare period or seam")
            return
        if self.period is None or self.seam is None:
            raise ValueError("periodic domains require explicit period and seam")
        period = float(self.period)
        seam = float(self.seam)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("period must be finite and positive")
        if not np.isfinite(seam):
            raise ValueError("seam must be finite")
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "seam", seam)

    def require_open_interval(self, *, operation: str) -> None:
        if self.topology != "open_interval":
            raise NotImplementedError(
                f"{operation} is implemented only for open_interval domains; periodic seam handling is not implemented"
            )


@dataclass(frozen=True)
class SavedGridReceipt:
    """Identity of the exact saved point indices, abscissa, and domain."""

    source_id: str
    point_indices: np.ndarray
    x: np.ndarray
    x_units: str
    domain: CurveDomainReceipt
    point_indices_sha256: str = field(init=False)
    x_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_text(self.source_id, name="source_id"))
        object.__setattr__(self, "x_units", _require_text(self.x_units, name="x_units"))
        if not isinstance(self.domain, CurveDomainReceipt):
            raise TypeError("domain must be a CurveDomainReceipt")
        raw_indices = np.asarray(self.point_indices)
        if raw_indices.dtype.kind not in {"i", "u"}:
            raise TypeError("point_indices must have an integer dtype")
        if raw_indices.dtype.kind == "u" and np.any(
            raw_indices > np.iinfo(np.int64).max
        ):
            raise OverflowError("unsigned point_indices exceed signed int64 range")
        indices = immutable_finite_array(raw_indices, dtype=np.dtype("<i8"), name="point_indices")
        x = immutable_finite_array(self.x, dtype=np.dtype("<f8"), name="x")
        if indices.shape != x.shape:
            raise ValueError(f"point_indices shape {indices.shape} does not match x shape {x.shape}")
        if np.any(indices < 0):
            raise ValueError("point_indices must be nonnegative")
        if np.unique(indices).size != indices.size:
            raise ValueError("point_indices must be unique")
        if x.size > 1 and np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be strictly increasing")
        object.__setattr__(self, "point_indices", indices)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "point_indices_sha256", canonical_array_sha256(indices))
        object.__setattr__(self, "x_sha256", canonical_array_sha256(x))


@dataclass(frozen=True)
class ObservableReceipt:
    """Raw observable identity, convention, units, and validity statement."""

    kind: str
    basis: str
    units: str
    validity: str

    def __post_init__(self) -> None:
        for name in ("kind", "basis", "units", "validity"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name=name))


@dataclass(frozen=True)
class ValueTransformReceipt:
    """Declared affine conversion from a raw value to its output convention."""

    input_units: str
    output_units: str
    scale: float
    offset: float
    semantics: str
    common_across_branches: bool

    def __post_init__(self) -> None:
        for name in ("input_units", "output_units", "semantics"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name=name))
        for name in ("scale", "offset"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if type(self.common_across_branches) is not bool:
            raise TypeError("common_across_branches must be bool")

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.offset == 0.0


@dataclass(frozen=True)
class ExactGridObservableEvaluation:
    """One terminal callback result before bundle-level certification."""

    branch_source_id: str
    terminal_payload_sha256: str
    saved_grid: SavedGridReceipt
    observable: ObservableReceipt
    value_transform: ValueTransformReceipt
    raw_y: np.ndarray
    output_y: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_source_id", _require_text(self.branch_source_id, name="branch_source_id"))
        object.__setattr__(
            self,
            "terminal_payload_sha256",
            _require_sha256(self.terminal_payload_sha256, name="terminal_payload_sha256"),
        )
        if not isinstance(self.saved_grid, SavedGridReceipt):
            raise TypeError("saved_grid must be a SavedGridReceipt")
        if not isinstance(self.observable, ObservableReceipt):
            raise TypeError("observable must be an ObservableReceipt")
        if not isinstance(self.value_transform, ValueTransformReceipt):
            raise TypeError("value_transform must be a ValueTransformReceipt")
        if self.observable.units != self.value_transform.input_units:
            raise ValueError("observable units must equal value-transform input units")
        raw = immutable_finite_array(self.raw_y, dtype=np.dtype("<f8"), name="raw_y")
        output = immutable_finite_array(self.output_y, dtype=np.dtype("<f8"), name="output_y")
        expected_shape = self.saved_grid.x.shape
        if raw.shape != expected_shape or output.shape != expected_shape:
            raise ValueError(
                "raw_y and output_y must match the saved grid shape; "
                f"got {raw.shape}, {output.shape}, and {expected_shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = self.value_transform.scale * raw
            expected = scaled + self.value_transform.offset
            residual = output - expected
        if (
            not np.all(np.isfinite(scaled))
            or not np.all(np.isfinite(expected))
            or not np.all(np.isfinite(residual))
        ):
            raise ValueError("value transform produced a non-finite affine intermediate")
        tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            (
                np.ones_like(expected),
                np.abs(scaled),
                np.full_like(expected, abs(self.value_transform.offset)),
                np.abs(expected),
            )
        )
        if np.any(np.abs(residual) > tolerance):
            raise ValueError("output_y must equal scale * raw_y + offset within floating-point operation-order tolerance")
        object.__setattr__(self, "raw_y", raw)
        object.__setattr__(self, "output_y", output)


@dataclass(frozen=True)
class DiscreteBranchNode:
    """One node in a finite, explicitly supplied branch tree."""

    node_id: str
    parent_id: str | None
    child_ids: tuple[str, ...] = ()
    status: BranchNodeStatus = "terminal"
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, name="node_id"))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", _require_text(self.parent_id, name="parent_id"))
        children = tuple(_require_text(value, name="child_id") for value in self.child_ids)
        if len(children) != len(set(children)):
            raise ValueError(f"node {self.node_id!r} has duplicate child IDs")
        object.__setattr__(self, "child_ids", children)
        if self.status not in {"expanded", "terminal", "rejected"}:
            raise ValueError(f"node {self.node_id!r} has unsupported status {self.status!r}")
        if self.status == "rejected":
            object.__setattr__(self, "rejection_reason", _require_text(self.rejection_reason or "", name="rejection_reason"))
        elif self.rejection_reason is not None:
            raise ValueError("rejection_reason is only valid for rejected nodes")


@dataclass(frozen=True)
class EnumerationReceipt:
    """System-owned identity and frontier receipt for a supplied enumeration."""

    algorithm_id: str
    algorithm_version: str
    source_input_sha256: str
    choice_inventory_sha256: str
    unconsumed_frontier_count: int
    terminal_payload_hashes: Mapping[str, str] | tuple[tuple[str, str], ...]
    system_claims_exhaustive_enumeration: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "algorithm_id", _require_text(self.algorithm_id, name="algorithm_id"))
        object.__setattr__(self, "algorithm_version", _require_text(self.algorithm_version, name="algorithm_version"))
        object.__setattr__(self, "source_input_sha256", _require_sha256(self.source_input_sha256, name="source_input_sha256"))
        object.__setattr__(
            self,
            "choice_inventory_sha256",
            _require_sha256(self.choice_inventory_sha256, name="choice_inventory_sha256"),
        )
        if type(self.unconsumed_frontier_count) is not int or self.unconsumed_frontier_count < 0:
            raise ValueError("unconsumed_frontier_count must be a nonnegative exact int")
        if isinstance(self.terminal_payload_hashes, Mapping):
            items = tuple(self.terminal_payload_hashes.items())
        else:
            items = tuple(self.terminal_payload_hashes)
        normalized = tuple(
            sorted(
                (
                    _require_text(terminal_id, name="terminal payload ID"),
                    _require_sha256(digest, name=f"terminal payload hash for {terminal_id!r}"),
                )
                for terminal_id, digest in items
            )
        )
        if len({terminal_id for terminal_id, _digest in normalized}) != len(normalized):
            raise ValueError("terminal_payload_hashes contains duplicate terminal IDs")
        if not normalized:
            raise ValueError("terminal_payload_hashes must be nonempty")
        object.__setattr__(self, "terminal_payload_hashes", normalized)
        if type(self.system_claims_exhaustive_enumeration) is not bool:
            raise TypeError("system_claims_exhaustive_enumeration must be bool")

    def payload_hash(self, terminal_id: str) -> str:
        for candidate, digest in self.terminal_payload_hashes:
            if candidate == terminal_id:
                return digest
        raise KeyError(terminal_id)


def _validated_branch_closure_fields(
    nodes: Sequence[DiscreteBranchNode],
    enumeration_receipt: EnumerationReceipt,
) -> tuple[
    tuple[DiscreteBranchNode, ...],
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    records = tuple(nodes)
    if not records:
        raise ValueError("branch tree must contain at least one node")
    if any(not isinstance(node, DiscreteBranchNode) for node in records):
        raise TypeError("branch tree records must be DiscreteBranchNode instances")
    if not isinstance(enumeration_receipt, EnumerationReceipt):
        raise TypeError("enumeration_receipt must be an EnumerationReceipt")
    by_id: dict[str, DiscreteBranchNode] = {}
    for node in records:
        if node.node_id in by_id:
            raise ValueError(f"duplicate branch node ID {node.node_id!r}")
        by_id[node.node_id] = node
    roots = tuple(sorted(node.node_id for node in records if node.parent_id is None))
    if len(roots) != 1:
        raise ValueError(f"branch tree must have exactly one root, found {len(roots)}")
    root_id = roots[0]
    for node in records:
        if node.status == "expanded" and not node.child_ids:
            raise ValueError(f"expanded node {node.node_id!r} must have nonempty children")
        if node.status in {"terminal", "rejected"} and node.child_ids:
            raise ValueError(f"terminal node {node.node_id!r} must not have children")
        if node.parent_id is not None:
            parent = by_id.get(node.parent_id)
            if parent is None:
                raise ValueError(f"node {node.node_id!r} references missing parent {node.parent_id!r}")
            if node.node_id not in parent.child_ids:
                raise ValueError(f"node {node.node_id!r} is absent from its parent child list")
        for child_id in node.child_ids:
            child = by_id.get(child_id)
            if child is None:
                raise ValueError(f"node {node.node_id!r} references missing child {child_id!r}")
            if child.parent_id != node.node_id:
                raise ValueError(f"child {child_id!r} has an inconsistent parent backlink")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("branch tree contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child_id in by_id[node_id].child_ids:
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(root_id)
    if visited != set(by_id):
        missing = sorted(set(by_id) - visited)
        raise ValueError(f"branch nodes are not reachable from the root: {missing}")
    leaves = tuple(sorted(node.node_id for node in records if not node.child_ids))
    terminals = tuple(sorted(node.node_id for node in records if node.status in {"terminal", "rejected"}))
    if leaves != terminals:
        raise ValueError("branch-tree leaves must equal resolved terminal records")
    receipt_ids = tuple(terminal_id for terminal_id, _digest in enumeration_receipt.terminal_payload_hashes)
    if receipt_ids != terminals:
        raise ValueError("enumeration terminal payload hashes must match every supplied tree leaf")
    if enumeration_receipt.unconsumed_frontier_count != 0:
        raise ValueError("supplied finite tree is not structurally resolved while the enumeration frontier is nonempty")
    computed = tuple(sorted(node.node_id for node in records if node.status == "terminal"))
    rejected = tuple(sorted(node.node_id for node in records if node.status == "rejected"))
    if not computed:
        raise ValueError("branch closure must contain at least one computed terminal")
    return tuple(sorted(records, key=lambda node: node.node_id)), root_id, terminals, computed, rejected


@dataclass(frozen=True)
class BranchClosureCertificate:
    """Generic structural certificate over only the supplied finite tree."""

    nodes: tuple[DiscreteBranchNode, ...]
    enumeration_receipt: EnumerationReceipt
    root_id: str
    terminal_ids: tuple[str, ...]
    computed_terminal_ids: tuple[str, ...]
    rejected_terminal_ids: tuple[str, ...]
    supplied_finite_tree_structurally_resolved: bool = True
    authority: BranchClosureAuthority = "supplied_finite_tree_structurally_resolved"

    def __post_init__(self) -> None:
        if self.authority != "supplied_finite_tree_structurally_resolved":
            raise ValueError("invalid branch-closure authority")
        if type(self.supplied_finite_tree_structurally_resolved) is not bool or not self.supplied_finite_tree_structurally_resolved:
            raise ValueError("a BranchClosureCertificate must represent a structurally resolved supplied tree")
        expected = _validated_branch_closure_fields(self.nodes, self.enumeration_receipt)
        actual = (
            tuple(self.nodes),
            self.root_id,
            tuple(self.terminal_ids),
            tuple(self.computed_terminal_ids),
            tuple(self.rejected_terminal_ids),
        )
        if actual != expected:
            raise ValueError("branch closure fields must be derived canonically from nodes and enumeration receipt")
        object.__setattr__(self, "nodes", expected[0])
        object.__setattr__(self, "root_id", expected[1])
        object.__setattr__(self, "terminal_ids", expected[2])
        object.__setattr__(self, "computed_terminal_ids", expected[3])
        object.__setattr__(self, "rejected_terminal_ids", expected[4])


@dataclass(frozen=True)
class ComputeCertificate:
    """Execution receipt without selection, reproduction, paper, or production authority."""

    closure_authority: BranchClosureAuthority
    computed_terminal_ids: tuple[str, ...]
    callback_count: int
    exact_saved_grid: bool = True

    def __post_init__(self) -> None:
        if self.closure_authority != "supplied_finite_tree_structurally_resolved":
            raise ValueError("invalid closure authority")
        if type(self.exact_saved_grid) is not bool or not self.exact_saved_grid:
            raise ValueError("ComputeCertificate must certify exact_saved_grid=True")
        computed = tuple(self.computed_terminal_ids)
        if computed != tuple(sorted(computed)) or len(computed) != len(set(computed)):
            raise ValueError("computed_terminal_ids must be unique and in canonical order")
        object.__setattr__(self, "computed_terminal_ids", computed)
        if type(self.callback_count) is not int or self.callback_count != len(computed):
            raise ValueError("callback_count must equal the number of computed terminal IDs")


@dataclass(frozen=True)
class ExactSavedGridCurve:
    """One immutable computed terminal curve, preserving raw and output values."""

    terminal_id: str
    terminal_payload_sha256: str
    saved_grid: SavedGridReceipt
    observable: ObservableReceipt
    value_transform: ValueTransformReceipt
    raw_y: np.ndarray
    output_y: np.ndarray
    raw_y_sha256: str = field(init=False)
    output_y_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        evaluation = ExactGridObservableEvaluation(
            branch_source_id=self.terminal_id,
            terminal_payload_sha256=self.terminal_payload_sha256,
            saved_grid=self.saved_grid,
            observable=self.observable,
            value_transform=self.value_transform,
            raw_y=self.raw_y,
            output_y=self.output_y,
        )
        object.__setattr__(self, "terminal_id", evaluation.branch_source_id)
        object.__setattr__(self, "terminal_payload_sha256", evaluation.terminal_payload_sha256)
        object.__setattr__(self, "raw_y", evaluation.raw_y)
        object.__setattr__(self, "output_y", evaluation.output_y)
        object.__setattr__(self, "raw_y_sha256", canonical_array_sha256(evaluation.raw_y))
        object.__setattr__(self, "output_y_sha256", canonical_array_sha256(evaluation.output_y))


def _bundle_fingerprint_payload(
    curves: tuple[ExactSavedGridCurve, ...],
    closure: BranchClosureCertificate,
    certificate: ComputeCertificate,
    source_authority: SourceAuthorityReceipt,
) -> dict[str, object]:
    return {
        "schema": "mean_field.exact_saved_grid_curve_bundle.v3",
        "source_authority": {
            "authority_id": source_authority.authority_id,
            "canonical_payload_json": source_authority.canonical_payload_json,
            "payload_sha256": source_authority.payload_sha256,
        },
        "closure": {
            "authority": closure.authority,
            "nodes": [asdict(node) for node in closure.nodes],
            "enumeration_receipt": asdict(closure.enumeration_receipt),
            "root_id": closure.root_id,
            "terminal_ids": list(closure.terminal_ids),
            "computed_terminal_ids": list(closure.computed_terminal_ids),
            "rejected_terminal_ids": list(closure.rejected_terminal_ids),
        },
        "compute_certificate": asdict(certificate),
        "curves": [
            {
                "terminal_id": curve.terminal_id,
                "terminal_payload_sha256": curve.terminal_payload_sha256,
                "saved_grid": {
                    "source_id": curve.saved_grid.source_id,
                    "x_units": curve.saved_grid.x_units,
                    "domain": asdict(curve.saved_grid.domain),
                    "point_indices_sha256": curve.saved_grid.point_indices_sha256,
                    "x_sha256": curve.saved_grid.x_sha256,
                },
                "observable": asdict(curve.observable),
                "value_transform": asdict(curve.value_transform),
                "raw_y_sha256": curve.raw_y_sha256,
                "output_y_sha256": curve.output_y_sha256,
            }
            for curve in curves
        ],
    }


@dataclass(frozen=True)
class ExactSavedGridCurveBundle:
    """Every computed terminal curve on one exact grid, with no selection lineage."""

    curves: tuple[ExactSavedGridCurve, ...]
    branch_closure: BranchClosureCertificate
    compute_certificate: ComputeCertificate
    source_authority: SourceAuthorityReceipt
    bundle_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        curves = tuple(self.curves)
        if not curves or any(not isinstance(curve, ExactSavedGridCurve) for curve in curves):
            raise ValueError("curves must contain ExactSavedGridCurve records")
        if not isinstance(self.branch_closure, BranchClosureCertificate):
            raise TypeError("branch_closure must be a BranchClosureCertificate")
        if not isinstance(self.compute_certificate, ComputeCertificate):
            raise TypeError("compute_certificate must be a ComputeCertificate")
        if not isinstance(self.source_authority, SourceAuthorityReceipt):
            raise TypeError("source_authority must be a SourceAuthorityReceipt")
        self.source_authority.validate_live_state()
        ids = tuple(curve.terminal_id for curve in curves)
        expected_ids = self.branch_closure.computed_terminal_ids
        if ids != expected_ids or ids != tuple(sorted(ids)):
            raise ValueError("curves must contain every computed terminal exactly once in canonical order")
        if self.branch_closure.rejected_terminal_ids:
            raise ValueError("comparison-ready curve bundles cannot contain rejected terminals")
        if self.compute_certificate.computed_terminal_ids != expected_ids:
            raise ValueError("compute certificate terminal IDs do not match branch closure")
        recertified = certify_enumerated_branch_closure(
            self.branch_closure.nodes,
            self.branch_closure.enumeration_receipt,
        )
        if recertified != self.branch_closure:
            raise ValueError("branch closure certificate does not match its supplied records")
        first = curves[0]
        for curve in curves:
            expected_payload = self.branch_closure.enumeration_receipt.payload_hash(curve.terminal_id)
            if curve.terminal_payload_sha256 != expected_payload:
                raise ValueError(f"terminal payload digest mismatch for {curve.terminal_id!r}")
            if (
                curve.observable.kind != first.observable.kind
                or curve.observable.basis != first.observable.basis
                or curve.observable.validity != first.observable.validity
            ):
                raise ValueError("all branch curves must use the same observable identity, basis, and validity")
            if curve.saved_grid.source_id != first.saved_grid.source_id:
                raise ValueError("all branch curves must identify the same saved-grid source")
            if curve.saved_grid.x_units != first.saved_grid.x_units:
                raise ValueError("all branch curves must use the same x units")
            if curve.saved_grid.domain != first.saved_grid.domain:
                raise ValueError("all branch curves must use the same curve domain")
            if not np.array_equal(curve.saved_grid.point_indices, first.saved_grid.point_indices):
                raise ValueError("all branch curves must use identical exact point indices")
            if not np.array_equal(curve.saved_grid.x, first.saved_grid.x):
                raise ValueError("all branch curves must use identical exact x values")
        identical_transforms = all(
            curve.value_transform == first.value_transform for curve in curves[1:]
        )
        common_flags = tuple(curve.value_transform.common_across_branches for curve in curves)
        if any(common_flags) and (not all(common_flags) or not identical_transforms):
            raise ValueError(
                "a transform declared common_across_branches must be identical and declared common on every curve"
            )
        object.__setattr__(self, "curves", curves)
        object.__setattr__(
            self,
            "bundle_fingerprint",
            canonical_json_sha256(
                _bundle_fingerprint_payload(
                    curves,
                    self.branch_closure,
                    self.compute_certificate,
                    self.source_authority,
                ),
                namespace="mean_field.curve_bundle.v3",
            ),
        )

    @property
    def x(self) -> np.ndarray:
        return self.curves[0].saved_grid.x

    @property
    def point_indices(self) -> np.ndarray:
        return self.curves[0].saved_grid.point_indices

    @property
    def domain(self) -> CurveDomainReceipt:
        return self.curves[0].saved_grid.domain

    @property
    def transforms_identical(self) -> bool:
        first = self.curves[0].value_transform
        return all(curve.value_transform == first for curve in self.curves[1:])


@runtime_checkable
class ExactGridCurveAdapter(Protocol):
    """System adapter consumed by the generic comparison-ready bundle builder."""

    branch_tree: Sequence[DiscreteBranchNode]
    enumeration_receipt: EnumerationReceipt
    source_authority: SourceAuthorityReceipt

    def evaluate_terminal(self, terminal_id: str) -> ExactGridObservableEvaluation:
        """Evaluate one computed terminal and return its exact payload digest."""


def certify_enumerated_branch_closure(
    nodes: Sequence[DiscreteBranchNode],
    enumeration_receipt: EnumerationReceipt,
) -> BranchClosureCertificate:
    """Certify only that the supplied finite tree and payload inventory resolve structurally."""

    records, root_id, terminals, computed, rejected = _validated_branch_closure_fields(
        nodes,
        enumeration_receipt,
    )
    return BranchClosureCertificate(
        nodes=records,
        enumeration_receipt=enumeration_receipt,
        root_id=root_id,
        terminal_ids=terminals,
        computed_terminal_ids=computed,
        rejected_terminal_ids=rejected,
    )


def build_exact_grid_curve_bundle(adapter: ExactGridCurveAdapter) -> ExactSavedGridCurveBundle:
    """Compute every supplied leaf exactly once; any rejected leaf blocks callbacks."""

    source_authority = getattr(adapter, "source_authority", None)
    if not isinstance(source_authority, SourceAuthorityReceipt):
        raise TypeError("adapter.source_authority must be a SourceAuthorityReceipt")
    source_authority.validate_live_state()
    closure = certify_enumerated_branch_closure(adapter.branch_tree, adapter.enumeration_receipt)
    if closure.rejected_terminal_ids:
        raise ValueError(
            "supplied finite tree contains rejected terminals; a comparison-ready bundle requires every supplied leaf computed"
        )
    curves: list[ExactSavedGridCurve] = []
    for terminal_id in closure.computed_terminal_ids:
        evaluation = adapter.evaluate_terminal(terminal_id)
        if not isinstance(evaluation, ExactGridObservableEvaluation):
            raise TypeError("evaluate_terminal must return ExactGridObservableEvaluation")
        if evaluation.branch_source_id != terminal_id:
            raise ValueError(
                f"callback branch source {evaluation.branch_source_id!r} does not match {terminal_id!r}"
            )
        expected_payload = closure.enumeration_receipt.payload_hash(terminal_id)
        if evaluation.terminal_payload_sha256 != expected_payload:
            raise ValueError(f"terminal payload digest mismatch for {terminal_id!r}")
        curves.append(
            ExactSavedGridCurve(
                terminal_id=terminal_id,
                terminal_payload_sha256=evaluation.terminal_payload_sha256,
                saved_grid=evaluation.saved_grid,
                observable=evaluation.observable,
                value_transform=evaluation.value_transform,
                raw_y=evaluation.raw_y,
                output_y=evaluation.output_y,
            )
        )
    certificate = ComputeCertificate(
        closure_authority=closure.authority,
        computed_terminal_ids=closure.computed_terminal_ids,
        callback_count=len(curves),
    )
    return ExactSavedGridCurveBundle(
        curves=tuple(curves),
        branch_closure=closure,
        compute_certificate=certificate,
        source_authority=source_authority,
    )


__all__ = [
    "BranchClosureAuthority",
    "BranchClosureCertificate",
    "BranchNodeStatus",
    "ComputeCertificate",
    "CurveDomainReceipt",
    "CurveDomainTopology",
    "DiscreteBranchNode",
    "EnumerationReceipt",
    "ExactGridCurveAdapter",
    "ExactGridObservableEvaluation",
    "ExactSavedGridCurve",
    "ExactSavedGridCurveBundle",
    "ObservableReceipt",
    "SavedGridReceipt",
    "SourceAuthorityReceipt",
    "ValueTransformReceipt",
    "build_exact_grid_curve_bundle",
    "canonical_array_sha256",
    "canonical_json_sha256",
    "certify_enumerated_branch_closure",
    "immutable_finite_array",
    "make_source_authority_receipt",
]
