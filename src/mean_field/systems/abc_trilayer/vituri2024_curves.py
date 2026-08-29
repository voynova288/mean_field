"""Exact saved-grid curve adapter for candidate-only Vituri fixed-sector BFS results.

The adapter exposes every accepted stationary branch on the exact saved
``k_y=0`` grid.  It does not select a branch or grant reproduction, UV,
unrestricted-state, TDHF, or production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Final

import numpy as np

from mean_field.core.curve_workflow import (
    CurveDomainReceipt,
    DiscreteBranchNode,
    EnumerationReceipt,
    ExactGridCurveAdapter,
    ExactGridObservableEvaluation,
    ExactSavedGridCurveBundle,
    ObservableReceipt,
    SavedGridReceipt,
    SourceAuthorityReceipt,
    ValueTransformReceipt,
    build_exact_grid_curve_bundle,
    canonical_array_sha256,
    canonical_json_sha256,
    make_source_authority_receipt,
)

from .vituri2024 import VITURI2024_PARAMETERS
from .vituri2024_hf_fixed_sector import (
    VITURI2024_FIXED_SECTOR_HF_API_VERSION,
    Vituri2024FixedSectorBFSNode,
    Vituri2024FixedSectorEndpoint,
    Vituri2024FixedSectorSearchResult,
)
from .vituri2024_hf_preflight import INTERNAL_FLAVOR_ORDER
from .vituri2024_hf_scf import Vituri2024PreparedHomogeneousHF

VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION: Final[str] = (
    "vituri2024_fixed_sector_exact_grid_curve_adapter.v1"
)

_ENUMERATION_ALGORITHM_ID: Final[str] = (
    "Vituri2024 fixed-sector discrete coordinate-shell branch universe BFS"
)
_SOURCE_AUTHORITY_ID: Final[str] = (
    "vituri2024_candidate_only_fixed_sector_source.v1"
)


def _strict_flavor(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("flavor must be an integer")
    flavor = int(value)
    if flavor not in range(4):
        raise ValueError("flavor must be in range(4)")
    return flavor


def _terminal_payload_sha256(endpoint: Vituri2024FixedSectorEndpoint) -> str:
    """Bind all endpoint state used to authorize and evaluate one terminal."""

    endpoint.validate_live_state()
    payload = {
        "path_fingerprint": endpoint.path.fingerprint,
        "consumed_choice_fingerprints": list(endpoint.consumed_choice_fingerprints),
        "array_hashes": {
            "final_density": endpoint.final_density_sha256,
            "fresh_raw_density": endpoint.fresh_raw_density_sha256,
            "engine_final_raw_density": endpoint.engine_final_raw_density_sha256,
            "fresh_hamiltonian": endpoint.final_hamiltonian_sha256,
            "fresh_energies": endpoint.final_energies_sha256,
        },
        "metrics": asdict(endpoint.metrics),
        "fresh_map": asdict(endpoint.fresh_map),
        "outcome": endpoint.outcome,
        "iterations": endpoint.iterations,
    }
    return canonical_json_sha256(
        payload,
        namespace="mean_field.vituri2024_fixed_sector_terminal_payload.v1",
    )


def _common_reference_interval(
    endpoints: tuple[Vituri2024FixedSectorEndpoint, ...],
) -> tuple[float, float, float]:
    """Return the closed common-mu intersection and its midpoint."""

    if type(endpoints) is not tuple or not endpoints:
        raise ValueError("common chemical-potential reference requires endpoints")
    for endpoint in endpoints:
        if type(endpoint) is not Vituri2024FixedSectorEndpoint:
            raise TypeError("common-reference endpoints must be exactly typed")
        endpoint.validate_live_state()
    lower = max(endpoint.fresh_map.common_mu_lower_ev for endpoint in endpoints)
    upper = min(endpoint.fresh_map.common_mu_upper_ev for endpoint in endpoints)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("endpoint common chemical-potential intersection is empty")
    midpoint = 0.5 * (lower + upper)
    if not math.isfinite(midpoint):
        raise ValueError("common chemical-potential midpoint is not finite")
    return float(lower), float(upper), float(midpoint)


def _exact_ky_zero_grid(prepared: Vituri2024PreparedHomogeneousHF) -> SavedGridReceipt:
    labels = prepared.integer_mesh_labels
    mesh_size = prepared.spec.mesh_size
    cut_indices = np.flatnonzero(labels[:, 1] == 0).astype(np.int64, copy=False)
    if cut_indices.size != mesh_size:
        raise ValueError("exact ky=0 cut must contain exactly N saved-grid points")
    order = np.argsort(labels[cut_indices, 0], kind="stable")
    cut_indices = np.asarray(cut_indices[order], dtype=np.int64)
    integer_kx = np.asarray(labels[cut_indices, 0], dtype=np.int64)
    expected_kx = np.arange(
        -(mesh_size // 2), mesh_size // 2 + 1, dtype=np.int64
    )
    if (
        not np.array_equal(integer_kx, expected_kx)
        or not np.array_equal(labels[cut_indices, 1], np.zeros(mesh_size, dtype=np.int64))
        or (cut_indices.size > 1 and np.any(np.diff(cut_indices) <= 0))
    ):
        raise ValueError("exact ky=0 saved-grid labels or point indices drifted")
    x = (
        integer_kx.astype(np.float64)
        * prepared.spec.delta_k_inverse_angstrom
        * VITURI2024_PARAMETERS.a0
    )
    if x.size > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError("exact ky=0 kx*a0 coordinates are not strictly increasing")
    cut_labels = np.asarray(labels[cut_indices], dtype=np.int64)
    source_digest = canonical_json_sha256(
        {
            "prepared_fingerprint": prepared.fingerprint,
            "integer_grid_sha256": canonical_array_sha256(labels),
            "ordered_mesh_sha256": canonical_array_sha256(prepared.ordered_mesh),
            "cut_point_indices_sha256": canonical_array_sha256(cut_indices),
            "cut_integer_labels_sha256": canonical_array_sha256(cut_labels),
            "cut_x_sha256": canonical_array_sha256(x),
        },
        namespace="mean_field.vituri2024_exact_ky0_saved_grid.v1",
    )
    return SavedGridReceipt(
        source_id=f"vituri2024-exact-ky0-{source_digest}",
        point_indices=cut_indices,
        x=x,
        x_units="k_x a0",
        domain=CurveDomainReceipt(topology="open_interval"),
    )



def _validate_expanded_node_child_inventory(
    node: Vituri2024FixedSectorBFSNode,
    node_by_id: dict[str, Vituri2024FixedSectorBFSNode],
) -> dict[str, object]:
    """Independently prove one expanded node has its full canonical sibling set."""

    if type(node) is not Vituri2024FixedSectorBFSNode:
        raise TypeError("expanded-node inventory requires an exact BFS node")
    node.validate_live_state()
    if node.outcome != "expanded_exact_frontier" or not node.child_path_ids:
        raise ValueError("canonical child inventory requires an expanded BFS node")
    if len(node.child_path_ids) != len(set(node.child_path_ids)):
        raise ValueError("expanded node contains duplicate sibling IDs")

    expected_generation = len(node.path.choices)
    trigger_fingerprint: str | None = None
    canonical_choice_count: int | None = None
    indices: list[int] = []
    child_fingerprints: list[str] = []
    for child_id in node.child_path_ids:
        child = node_by_id.get(child_id)
        if child is None:
            raise ValueError("expanded node references an omitted child record")
        child.path.validate_live_state()
        if (
            len(child.path.choices) != expected_generation + 1
            or child.path.choices[:-1] != node.path.choices
        ):
            raise ValueError("every child must extend exactly its parent by one choice")
        choice = child.path.choices[-1]
        choice.validate_live_state()
        if choice.trigger.generation != expected_generation:
            raise ValueError("expanded siblings do not share the parent generation")
        if trigger_fingerprint is None:
            trigger_fingerprint = choice.trigger.fingerprint
            canonical_choice_count = choice.canonical_choice_count
        elif (
            choice.trigger.fingerprint != trigger_fingerprint
            or choice.canonical_choice_count != canonical_choice_count
        ):
            raise ValueError("expanded siblings do not share one trigger and choice count")
        if choice.canonical_choice_count != choice.trigger.canonical_choice_count:
            raise ValueError("sibling canonical choice count does not match its trigger")
        indices.append(choice.canonical_choice_index)
        child_fingerprints.append(choice.fingerprint)

    assert trigger_fingerprint is not None and canonical_choice_count is not None
    if len(node.child_path_ids) != canonical_choice_count:
        raise ValueError("expanded child count does not equal canonical_choice_count")
    expected_indices = list(range(canonical_choice_count))
    if indices != expected_indices:
        raise ValueError(
            "expanded canonical choice indices must be exactly 0..count-1 in order"
        )
    return {
        "parent_path_fingerprint": node.path.fingerprint,
        "trigger_fingerprint": trigger_fingerprint,
        "generation": expected_generation,
        "canonical_choice_count": canonical_choice_count,
        "canonical_choice_indices": expected_indices,
        "child_path_ids": list(node.child_path_ids),
        "child_choice_fingerprints": child_fingerprints,
        "all_children_extend_parent_by_one_choice": True,
        "no_omitted_or_duplicate_sibling": True,
    }


def _branch_tree_and_inventory(
    result: Vituri2024FixedSectorSearchResult,
) -> tuple[tuple[DiscreteBranchNode, ...], list[dict[str, object]]]:
    endpoint_by_id = {endpoint.path.path_id: endpoint for endpoint in result.endpoints}
    path_id_by_choices = {
        node.path.choices: node.path.path_id
        for node in result.nodes
    }
    node_by_id = {node.path.path_id: node for node in result.nodes}
    if len(node_by_id) != len(result.nodes):
        raise ValueError("BFS node inventory contains duplicate path IDs")
    records: list[DiscreteBranchNode] = []
    inventory: list[dict[str, object]] = []
    matched_endpoints: set[str] = set()
    for node in result.nodes:
        node_id = node.path.path_id
        parent_id = None
        if node.path.choices:
            parent_id = path_id_by_choices.get(node.path.choices[:-1])
            if parent_id is None:
                raise ValueError(f"BFS node {node_id!r} has no exact path-prefix parent")
        children = tuple(node.child_path_ids)
        endpoint = endpoint_by_id.get(node_id)
        child_inventory: dict[str, object] | None = None
        if children:
            if node.outcome != "expanded_exact_frontier" or endpoint is not None:
                raise ValueError("only exact-frontier nodes may be expanded")
            child_inventory = _validate_expanded_node_child_inventory(
                node, node_by_id
            )
            status = "expanded"
        else:
            if (
                endpoint is None
                or node.outcome != "stationary"
                or endpoint.outcome != "stationary"
                or not endpoint.stationary
                or not endpoint.converged
            ):
                raise ValueError(
                    "every supplied leaf must match one accepted stationary endpoint"
                )
            matched_endpoints.add(node_id)
            status = "terminal"
        records.append(
            DiscreteBranchNode(
                node_id=node_id,
                parent_id=parent_id,
                child_ids=children,
                status=status,
            )
        )
        inventory.append(
            {
                "node_id": node_id,
                "path_fingerprint": node.path.fingerprint,
                "parent_id": parent_id,
                "child_ids": list(children),
                "outcome": node.outcome,
                "choices": [asdict(choice) for choice in node.path.choices],
                "canonical_child_inventory": child_inventory,
            }
        )
    if matched_endpoints != set(endpoint_by_id):
        raise ValueError("BFS endpoint inventory contains unmatched or unresolved records")
    return tuple(records), inventory


@dataclass(frozen=True, slots=True)
class Vituri2024FixedSectorCurveAdapter:
    """Frozen structural adapter to the generic exact-grid curve workflow."""

    prepared: Vituri2024PreparedHomogeneousHF
    result: Vituri2024FixedSectorSearchResult
    flavor: int = 3
    branch_tree: tuple[DiscreteBranchNode, ...] = field(init=False)
    enumeration_receipt: EnumerationReceipt = field(init=False)
    source_authority: SourceAuthorityReceipt = field(init=False)
    saved_grid: SavedGridReceipt = field(init=False)
    observable: ObservableReceipt = field(init=False)
    value_transform: ValueTransformReceipt = field(init=False)
    common_mu_lower_ev: float = field(init=False)
    common_mu_upper_ev: float = field(init=False)
    common_mu_ev: float = field(init=False)
    authority: str = field(init=False)
    in_process_candidate_only: bool = field(init=False)
    independent_finite_volume_fixed_sector_full_scf_discriminator: bool = field(init=False)
    local_hessian_stability_established: bool = field(init=False)
    author_cutoff_identified: bool = field(init=False)
    uv_plateau_established: bool = field(init=False)
    unrestricted_ground_state_established: bool = field(init=False)
    full_paper_reproduction_verified: bool = field(init=False)
    tdhf_authority: bool = field(init=False)
    production_authority: bool = field(init=False)
    visual_match_promotes_authority: bool = field(init=False)
    _terminal_records: tuple[tuple[str, Vituri2024FixedSectorEndpoint, str], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if type(self.prepared) is not Vituri2024PreparedHomogeneousHF:
            raise TypeError("prepared must be exactly Vituri2024PreparedHomogeneousHF")
        if type(self.result) is not Vituri2024FixedSectorSearchResult:
            raise TypeError("result must be exactly Vituri2024FixedSectorSearchResult")
        self.prepared.validate_live_state()
        self.result.validate_live_state()
        if self.prepared.fingerprint != self.result.prepared_fingerprint:
            raise ValueError("prepared/result fingerprint binding mismatch")
        if (
            self.result.branch_tree_exhausted is not True
            or self.result.unconsumed_frontier_count != 0
            or self.result.rejections
            or self.result.replayed_path_count != len(self.result.nodes)
            or self.result.endpoint_count != len(self.result.endpoints)
            or not self.result.endpoints
            or self.result.all_normal_endpoints_stationary is not True
            or any(
                endpoint.outcome != "stationary"
                or not endpoint.stationary
                or not endpoint.converged
                for endpoint in self.result.endpoints
            )
        ):
            raise ValueError(
                "curve adapter requires a closed rejection-free all-stationary BFS result"
            )

        flavor = _strict_flavor(self.flavor)
        if flavor not in self.result.policy.partial_flavors:
            raise ValueError(
                f"flavor {flavor} is not one of policy partial flavors "
                f"{self.result.policy.partial_flavors}"
            )
        object.__setattr__(self, "flavor", flavor)

        branch_tree, inventory = _branch_tree_and_inventory(self.result)
        terminal_records = tuple(
            sorted(
                (
                    endpoint.path.path_id,
                    endpoint,
                    _terminal_payload_sha256(endpoint),
                )
                for endpoint in self.result.endpoints
            )
        )
        source_authority = make_source_authority_receipt(
            _SOURCE_AUTHORITY_ID,
            {
                "source_scope": self.result.authority,
                "in_process_candidate_only": (
                    self.result.in_process_candidate_only
                ),
                "independent_finite_volume_fixed_sector_full_scf_discriminator": (
                    self.result.independent_finite_volume_fixed_sector_full_scf_discriminator
                ),
                "local_hessian_stability_established": (
                    self.result.local_hessian_stability_established
                ),
                "author_cutoff_identified": self.result.author_cutoff_identified,
                "uv_plateau_established": self.result.uv_plateau_established,
                "unrestricted_ground_state_established": (
                    self.result.unrestricted_ground_state_established
                ),
                "full_paper_reproduction_verified": (
                    self.result.full_paper_reproduction_verified
                ),
                "tdhf_authority": self.result.tdhf_authority,
                "production_authority": self.result.production_authority,
                "visual_match_promotes_authority": (
                    self.result.visual_match_promotes_authority
                ),
            },
        )
        source_input_sha256 = canonical_json_sha256(
            {
                "adapter_api_version": VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION,
                "prepared_fingerprint": self.prepared.fingerprint,
                "policy_fingerprint": self.result.policy.fingerprint,
                "initializer_fingerprint": self.result.initializer.fingerprint,
            },
            namespace="mean_field.vituri2024_curve_source_input.v1",
        )
        enumeration_receipt = EnumerationReceipt(
            algorithm_id=_ENUMERATION_ALGORITHM_ID,
            algorithm_version=VITURI2024_FIXED_SECTOR_HF_API_VERSION,
            source_input_sha256=source_input_sha256,
            choice_inventory_sha256=canonical_json_sha256(
                inventory,
                namespace="mean_field.vituri2024_curve_node_choice_inventory.v1",
            ),
            unconsumed_frontier_count=0,
            terminal_payload_hashes=tuple(
                (terminal_id, payload_sha256)
                for terminal_id, _endpoint, payload_sha256 in terminal_records
            ),
            system_claims_exhaustive_enumeration=True,
        )

        saved_grid = _exact_ky_zero_grid(self.prepared)
        mu_lower, mu_upper, mu_common = _common_reference_interval(
            self.result.endpoints
        )
        flavor_label = INTERNAL_FLAVOR_ORDER[flavor]
        observable = ObservableReceipt(
            kind="real_diagonal_matrix_element",
            basis=(
                f"fixed Vituri internal flavor basis index {flavor}={flavor_label}; "
                "no Hamiltonian diagonalization"
            ),
            units="eV",
            validity=(
                "Re fresh H_ff is valid only for fixed-sector stationary and "
                "converged endpoints whose off-diagonal Hamiltonian/coherence and "
                "fresh/raw stationarity gates passed; the discarded imaginary "
                "diagonal is bounded by upstream Hermiticity/stationarity gates; "
                "this is neither raw complex H_ff nor an eigenvalue"
            ),
        )
        value_transform = ValueTransformReceipt(
            input_units="eV",
            output_units="meV",
            scale=1000.0,
            offset=-1000.0 * mu_common,
            semantics=(
                "one common-intersection midpoint mu_common across every endpoint; "
                "output=1000*(Re fresh H_ff-mu_common), an additive gauge plotting "
                "convention"
            ),
            common_across_branches=True,
        )

        object.__setattr__(self, "branch_tree", branch_tree)
        object.__setattr__(self, "enumeration_receipt", enumeration_receipt)
        object.__setattr__(self, "source_authority", source_authority)
        object.__setattr__(self, "saved_grid", saved_grid)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "value_transform", value_transform)
        object.__setattr__(self, "common_mu_lower_ev", mu_lower)
        object.__setattr__(self, "common_mu_upper_ev", mu_upper)
        object.__setattr__(self, "common_mu_ev", mu_common)
        object.__setattr__(self, "_terminal_records", terminal_records)
        for name in (
            "authority",
            "in_process_candidate_only",
            "independent_finite_volume_fixed_sector_full_scf_discriminator",
            "local_hessian_stability_established",
            "author_cutoff_identified",
            "uv_plateau_established",
            "unrestricted_ground_state_established",
            "full_paper_reproduction_verified",
            "tdhf_authority",
            "production_authority",
            "visual_match_promotes_authority",
        ):
            object.__setattr__(self, name, getattr(self.result, name))

        if not isinstance(self, ExactGridCurveAdapter):
            raise TypeError("Vituri curve adapter does not satisfy ExactGridCurveAdapter")

    def evaluate_terminal(self, terminal_id: str) -> ExactGridObservableEvaluation:
        if type(terminal_id) is not str or not terminal_id:
            raise TypeError("terminal_id must be a nonempty exact string")
        for candidate_id, endpoint, payload_sha256 in self._terminal_records:
            if candidate_id == terminal_id:
                endpoint.validate_live_state()
                if _terminal_payload_sha256(endpoint) != payload_sha256:
                    raise ValueError("terminal endpoint payload drifted")
                point_indices = self.saved_grid.point_indices
                raw_y = np.real(
                    endpoint.fresh_hamiltonian[
                        self.flavor, self.flavor, point_indices
                    ]
                )
                output_y = 1000.0 * (raw_y - self.common_mu_ev)
                return ExactGridObservableEvaluation(
                    branch_source_id=terminal_id,
                    terminal_payload_sha256=payload_sha256,
                    saved_grid=self.saved_grid,
                    observable=self.observable,
                    value_transform=self.value_transform,
                    raw_y=raw_y,
                    output_y=output_y,
                )
        raise KeyError(terminal_id)


def make_vituri2024_fixed_sector_curve_adapter(
    prepared: Vituri2024PreparedHomogeneousHF,
    result: Vituri2024FixedSectorSearchResult,
    *,
    flavor: int = 3,
) -> Vituri2024FixedSectorCurveAdapter:
    """Create the fail-closed candidate-only exact-grid adapter."""

    return Vituri2024FixedSectorCurveAdapter(prepared, result, flavor)


def build_vituri2024_fixed_sector_curve_bundle(
    prepared: Vituri2024PreparedHomogeneousHF,
    result: Vituri2024FixedSectorSearchResult,
    *,
    flavor: int = 3,
) -> ExactSavedGridCurveBundle:
    """Build every accepted Vituri branch through the generic curve builder."""

    adapter = make_vituri2024_fixed_sector_curve_adapter(
        prepared, result, flavor=flavor
    )
    return build_exact_grid_curve_bundle(adapter)


__all__ = [
    "VITURI2024_FIXED_SECTOR_CURVE_ADAPTER_API_VERSION",
    "Vituri2024FixedSectorCurveAdapter",
    "build_vituri2024_fixed_sector_curve_bundle",
    "make_vituri2024_fixed_sector_curve_adapter",
]
