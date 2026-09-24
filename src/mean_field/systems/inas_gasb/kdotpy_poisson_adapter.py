"""Optional kdotpy adapter for the fresh split-zero Kane--Poisson core.

The adapter constructs, attests, diagonalizes, and residual-checks the same sparse
full-parent matrix object at each k point. The package façade may import its typed
contracts, but the optional external ``kdotpy`` runtime is imported only when a
window builder is instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


from .kane_poisson import (
    KaneParentSourceSpec,
    KaneWindowAtPotential,
    canonical_parent_matrix_sequence_sha256,
    kane_poisson_array_sha256,
    kane_potential_operator_fingerprint,
)

DEFAULT_MINIMUM_ASSIGNMENT_MARGIN = 1.0e-3
DECLARED_GAMMA_SEED_STATUS = (
    "declared-Gamma-local-candidate-position-seed-not-independent-identity-evidence"
)
LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL = (
    "local-position-in-per-k-energy-ordered-Rayleigh-Ritz-candidate-table:v1"
)
XML_RUNTIME_BINDING_STATUS = "external-not-attested-by-pair-selection"


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    contiguous = np.array(values, dtype=dtype, copy=True, order="C")
    # A bytes-backed view is not merely write-disabled: callers cannot restore
    # WRITEABLE with ``setflags(write=True)`` and mutate nested evidence later.
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return result.reshape(contiguous.shape)


def _validate_sha256(value: str, *, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _hash_length_delimited_text(
    digest: Any, *, field_label: str, value: str
) -> None:
    """Hash text without concatenation ambiguity between adjacent fields."""

    for text in (str(field_label), str(value)):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)


@dataclass(frozen=True, slots=True)
class KdotpyPreviousUSameKHomotopySpec:
    """Opt-in finite refinement contract for previous-U same-k pair tracking.

    The algorithm is intentionally fixed rather than configurable: linear
    potential interpolation, dyadic searches, N+1 and 2N endpoint
    confirmation, and reverse-2N recovery.  Only the finite work bound is an
    input.  All numerical gates are inherited unchanged from the enclosing
    pair-selection and builder contracts.
    """

    maximum_substeps: int = 64

    def __post_init__(self) -> None:
        value = self.maximum_substeps
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("maximum_substeps must be an exact integer")
        maximum = int(value)
        if maximum < 4 or maximum & (maximum - 1):
            raise ValueError(
                "maximum_substeps must be a power of two greater than or equal to four"
            )
        object.__setattr__(self, "maximum_substeps", maximum)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"KdotpyPreviousUSameKHomotopySpec:v1")
        _hash_length_delimited_text(
            digest, field_label="maximum_substeps", value=str(self.maximum_substeps)
        )
        _hash_length_delimited_text(
            digest,
            field_label="path_contract",
            value="exact-linear-potential-fixed-k",
        )
        _hash_length_delimited_text(
            digest,
            field_label="certification_contract",
            value="dyadic-first-match-confirm-N-plus-1-and-2N-reverse-2N",
        )
        _hash_length_delimited_text(
            digest,
            field_label="gate_contract",
            value="inherit-pair-overlap-margin-ritz-complement-potential-attestation",
        )
        return digest.hexdigest()








def _maximum_sparse_abs(matrix: sparse.spmatrix) -> float:
    values = sparse.csr_matrix(matrix).data
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _rayleigh_ritz_near_target(
    hamiltonian: sparse.spmatrix,
    *,
    candidate_count: int,
    target_energy_mev: float,
    tolerance: float,
    maximum_iterations: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Diagonalize one exact sparse parent matrix and certify the candidate frame."""

    dimension = hamiltonian.shape[0]
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError("parent Hamiltonian must be square")
    if not 1 <= candidate_count < dimension:
        raise ValueError("candidate_count must lie in [1,parent_dimension)")
    raw_energies, raw_vectors = eigsh(
        sparse.csc_matrix(hamiltonian),
        k=int(candidate_count),
        sigma=float(target_energy_mev),
        which="LM",
        tol=float(tolerance),
        maxiter=maximum_iterations,
        return_eigenvectors=True,
    )
    raw_vectors = np.asarray(raw_vectors, dtype=np.complex128)
    raw_orthonormality = float(
        np.max(
            np.abs(
                raw_vectors.conj().T @ raw_vectors
                - np.eye(candidate_count, dtype=np.complex128)
            )
        )
    )
    frame, _triangular = np.linalg.qr(raw_vectors)
    projected = frame.conj().T @ (hamiltonian @ frame)
    projected = 0.5 * (projected + projected.conj().T)
    energies, rotation = np.linalg.eigh(projected)
    vectors = frame @ rotation
    order = np.argsort(energies)
    energies = np.asarray(energies[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    residual = hamiltonian @ vectors - vectors * energies[None, :]
    residual_norm = float(np.max(np.linalg.norm(residual, axis=0)))
    orthonormality = float(
        np.max(
            np.abs(
                vectors.conj().T @ vectors
                - np.eye(candidate_count, dtype=np.complex128)
            )
        )
    )
    return energies, vectors, raw_orthonormality, max(residual_norm, orthonormality)


@dataclass(frozen=True, slots=True)
class KdotpyE1H1PairSelectionSpec:
    """Opt-in declaration and tracking contract for an ordered E1/H1 quartet.

    ``gamma_*_local_candidate_positions`` are positions in the energy-ordered
    Gamma candidate table, not global band IDs.  They are a declared identity
    seed and are explicitly *not* independent physical-identity evidence.
    Physical output labels are declared exactly and must match the caller's
    selected labels in E-then-H order.  Ordering inside either rank-two pair is
    still a U(2) gauge choice.
    """

    gamma_electron_local_candidate_positions: tuple[int, int]
    gamma_hole_local_candidate_positions: tuple[int, int]
    electron_selected_labels: tuple[int, int]
    hole_selected_labels: tuple[int, int]
    minimum_principal_overlap: float
    minimum_assignment_margin: float = DEFAULT_MINIMUM_ASSIGNMENT_MARGIN
    electron_pair_label: str = "E1"
    hole_pair_label: str = "H1"
    previous_u_same_k_homotopy_spec: KdotpyPreviousUSameKHomotopySpec | None = None

    def __post_init__(self) -> None:
        electron_positions = self._validate_local_position_pair(
            self.gamma_electron_local_candidate_positions,
            label="gamma_electron_local_candidate_positions",
        )
        hole_positions = self._validate_local_position_pair(
            self.gamma_hole_local_candidate_positions,
            label="gamma_hole_local_candidate_positions",
        )
        if set(electron_positions) & set(hole_positions):
            raise ValueError("Gamma electron and hole candidate pairs must be disjoint")
        electron_labels = self._validate_selected_label_pair(
            self.electron_selected_labels, label="electron_selected_labels"
        )
        hole_labels = self._validate_selected_label_pair(
            self.hole_selected_labels, label="hole_selected_labels"
        )
        if set(electron_labels) & set(hole_labels):
            raise ValueError("declared electron and hole selected labels must be disjoint")
        overlap_threshold = float(self.minimum_principal_overlap)
        if not np.isfinite(overlap_threshold) or not 0.0 < overlap_threshold <= 1.0:
            raise ValueError("minimum_principal_overlap must lie in (0,1]")
        assignment_threshold = float(self.minimum_assignment_margin)
        if (
            not np.isfinite(assignment_threshold)
            or not 0.0 < assignment_threshold <= 1.0
        ):
            raise ValueError("minimum_assignment_margin must lie in (0,1]")
        homotopy_spec = self.previous_u_same_k_homotopy_spec
        if homotopy_spec is not None and not isinstance(
            homotopy_spec, KdotpyPreviousUSameKHomotopySpec
        ):
            raise TypeError(
                "previous_u_same_k_homotopy_spec must be "
                "KdotpyPreviousUSameKHomotopySpec or None"
            )
        electron_label = str(self.electron_pair_label)
        hole_label = str(self.hole_pair_label)
        if (
            not electron_label.strip()
            or not hole_label.strip()
            or electron_label == hole_label
        ):
            raise ValueError("electron and hole pair labels must be nonempty and distinct")
        object.__setattr__(
            self, "gamma_electron_local_candidate_positions", electron_positions
        )
        object.__setattr__(self, "gamma_hole_local_candidate_positions", hole_positions)
        object.__setattr__(self, "electron_selected_labels", electron_labels)
        object.__setattr__(self, "hole_selected_labels", hole_labels)
        object.__setattr__(self, "minimum_principal_overlap", overlap_threshold)
        object.__setattr__(self, "minimum_assignment_margin", assignment_threshold)
        object.__setattr__(self, "electron_pair_label", electron_label)
        object.__setattr__(self, "hole_pair_label", hole_label)

    @staticmethod
    def _validate_local_position_pair(
        values: tuple[int, int], *, label: str
    ) -> tuple[int, int]:
        raw = tuple(values)
        if len(raw) != 2:
            raise ValueError(f"{label} must be a rank-two candidate pair")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw
        ):
            raise ValueError(
                f"{label} must contain exact integer local candidate positions"
            )
        pair = tuple(int(value) for value in raw)
        if pair[0] == pair[1] or min(pair) < 0:
            raise ValueError(
                f"{label} must contain two distinct nonnegative local positions"
            )
        return pair

    @staticmethod
    def _validate_selected_label_pair(
        values: tuple[int, int], *, label: str
    ) -> tuple[int, int]:
        raw = tuple(values)
        if len(raw) != 2:
            raise ValueError(f"{label} must contain exactly two physical labels")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) == 0
            for value in raw
        ):
            raise ValueError(f"{label} must contain exact nonzero integer labels")
        pair = tuple(int(value) for value in raw)
        if pair[0] == pair[1]:
            raise ValueError(f"{label} must contain two distinct physical labels")
        return pair

    @property
    def ordered_selected_labels(self) -> tuple[int, int, int, int]:
        return self.electron_selected_labels + self.hole_selected_labels

    def validate_ordered_selected_labels(self, values: tuple[int, ...]) -> None:
        raw = tuple(values)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw
        ):
            raise ValueError("pair-resolved selected labels must be exact integers")
        labels = tuple(int(value) for value in raw)
        if labels != self.ordered_selected_labels:
            if labels == self.hole_selected_labels + self.electron_selected_labels:
                raise ValueError(
                    "legacy H-first selected labels are rejected; pair mode requires "
                    "the exactly declared E-then-H order"
                )
            raise ValueError(
                "pair-resolved selected labels must exactly match the declared "
                "E-then-H labels"
            )

    def validate_candidate_count(self, candidate_count: int) -> None:
        if (
            isinstance(candidate_count, (bool, np.bool_))
            or not isinstance(candidate_count, (int, np.integer))
            or int(candidate_count) < 4
        ):
            raise ValueError("candidate_count must be an integer at least four")
        if max(
            *self.gamma_electron_local_candidate_positions,
            *self.gamma_hole_local_candidate_positions,
        ) >= int(candidate_count):
            raise ValueError("Gamma E/H local candidate positions must be in range")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"KdotpyE1H1PairSelectionSpec:v2")
        digest.update(repr(self.gamma_electron_local_candidate_positions).encode())
        digest.update(repr(self.gamma_hole_local_candidate_positions).encode())
        digest.update(repr(self.electron_selected_labels).encode())
        digest.update(repr(self.hole_selected_labels).encode())
        digest.update(repr(self.minimum_principal_overlap).encode())
        digest.update(repr(self.minimum_assignment_margin).encode())
        _hash_length_delimited_text(
            digest, field_label="electron_pair_label", value=self.electron_pair_label
        )
        _hash_length_delimited_text(
            digest, field_label="hole_pair_label", value=self.hole_pair_label
        )
        _hash_length_delimited_text(
            digest, field_label="gamma_seed_status", value=DECLARED_GAMMA_SEED_STATUS
        )
        legacy_fingerprint = digest.hexdigest()
        if self.previous_u_same_k_homotopy_spec is None:
            # This branch is deliberately byte-for-byte identical to the
            # established v2 fingerprint contract.
            return legacy_fingerprint
        homotopy_digest = hashlib.sha256()
        homotopy_digest.update(b"KdotpyE1H1PairSelectionSpec:v3-homotopy")
        _hash_length_delimited_text(
            homotopy_digest,
            field_label="legacy_pair_selection_spec_fingerprint",
            value=legacy_fingerprint,
        )
        _hash_length_delimited_text(
            homotopy_digest,
            field_label="previous_u_same_k_homotopy_spec_fingerprint",
            value=self.previous_u_same_k_homotopy_spec.fingerprint,
        )
        return homotopy_digest.hexdigest()

@dataclass(frozen=True, slots=True)
class _KdotpyE1H1PairAssignment:
    electron_local_candidate_positions: tuple[int, int]
    hole_local_candidate_positions: tuple[int, int]
    electron_principal_singular_values: np.ndarray
    hole_principal_singular_values: np.ndarray
    electron_selected_projection_weights: np.ndarray
    hole_selected_projection_weights: np.ndarray
    electron_assignment_margin: float
    hole_assignment_margin: float

    def __post_init__(self) -> None:
        for name in (
            "electron_principal_singular_values",
            "hole_principal_singular_values",
            "electron_selected_projection_weights",
            "hole_selected_projection_weights",
        ):
            values = _readonly_copy(getattr(self, name), dtype=np.dtype(np.float64))
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite rank-two array")
            object.__setattr__(self, name, values)


def _validate_pair_selection_frame(
    frame: np.ndarray,
    *,
    label: str,
    expected_columns: int | None,
) -> np.ndarray:
    result = np.asarray(frame, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] < 1:
        raise ValueError(f"{label} must be a nonempty frame")
    if expected_columns is not None and result.shape[1] != expected_columns:
        raise ValueError(f"{label} must contain exactly {expected_columns} columns")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    gram_error = float(
        np.max(
            np.abs(
                result.conj().T @ result
                - np.eye(result.shape[1], dtype=np.complex128)
            )
        )
    )
    if gram_error > 1.0e-8:
        raise ValueError(f"{label} must be an orthonormal frame")
    return result


def _best_rank_two_projection_assignment(
    reference_frame: np.ndarray,
    candidate_frame: np.ndarray,
    *,
    minimum_assignment_margin: float,
    pair_label: str,
    context: str,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, float]:
    """Return a separated best pair in normalized projection-score units.

    The score is the mean squared principal singular value, equivalently half
    the reference-projector weight captured by the rank-two candidate pair.
    A configured margin is required in addition to a floating-point tie gate;
    therefore a finite near-tie is rejected rather than silently tie-broken.
    """

    overlap = reference_frame.conj().T @ candidate_frame
    assignments: list[tuple[float, float, tuple[int, int], np.ndarray]] = []
    for pair in combinations(range(candidate_frame.shape[1]), 2):
        singular_values = np.linalg.svd(
            overlap[:, pair], compute_uv=False
        ).astype(np.float64, copy=False)
        projection_score = float(np.sum(singular_values**2) / 2.0)
        assignments.append(
            (projection_score, float(np.min(singular_values)), pair, singular_values)
        )
    assignments.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = assignments[0]
    second_score = assignments[1][0]
    margin = float(best[0] - second_score)
    roundoff_gate = (
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(best[0]), abs(second_score))
    )
    required_margin = max(float(minimum_assignment_margin), roundoff_gate)
    if margin < required_margin:
        raise ValueError(
            f"{context}: {pair_label} rank-two assignment is a near-tie: "
            f"margin={margin:.3e}, minimum_assignment_margin="
            f"{minimum_assignment_margin:.3e}, roundoff_gate={roundoff_gate:.3e}"
        )
    selected = np.asarray(best[2], dtype=np.int64)
    selected_projection_weights = np.sum(
        np.abs(overlap[:, selected]) ** 2, axis=0
    )
    return (
        best[2],
        np.asarray(best[3], dtype=np.float64),
        np.asarray(selected_projection_weights, dtype=np.float64),
        margin,
    )


def _select_e1h1_candidate_pairs_by_overlap(
    electron_reference_frame: np.ndarray,
    hole_reference_frame: np.ndarray,
    candidate_frame: np.ndarray,
    *,
    minimum_principal_overlap: float,
    minimum_assignment_margin: float = DEFAULT_MINIMUM_ASSIGNMENT_MARGIN,
    electron_pair_label: str = "E1",
    hole_pair_label: str = "H1",
    context: str = "E1/H1 candidate selection",
) -> _KdotpyE1H1PairAssignment:
    """Pure fail-closed rank-two selector used by Gamma, radial, and U tracking."""

    electron_reference = _validate_pair_selection_frame(
        electron_reference_frame,
        label="electron_reference_frame",
        expected_columns=2,
    )
    hole_reference = _validate_pair_selection_frame(
        hole_reference_frame,
        label="hole_reference_frame",
        expected_columns=2,
    )
    candidates = _validate_pair_selection_frame(
        candidate_frame,
        label="candidate_frame",
        expected_columns=None,
    )
    if (
        electron_reference.shape[0] != candidates.shape[0]
        or hole_reference.shape[0] != candidates.shape[0]
        or candidates.shape[1] < 4
    ):
        raise ValueError("E/H references and at least four candidates must share a basis")
    cross_overlap = float(
        np.max(np.abs(electron_reference.conj().T @ hole_reference))
    )
    if cross_overlap > 1.0e-8:
        raise ValueError("electron and hole reference subspaces must be disjoint")
    overlap_threshold = float(minimum_principal_overlap)
    if not np.isfinite(overlap_threshold) or not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("minimum_principal_overlap must lie in (0,1]")
    assignment_threshold = float(minimum_assignment_margin)
    if not np.isfinite(assignment_threshold) or not 0.0 < assignment_threshold <= 1.0:
        raise ValueError("minimum_assignment_margin must lie in (0,1]")

    electron = _best_rank_two_projection_assignment(
        electron_reference,
        candidates,
        minimum_assignment_margin=assignment_threshold,
        pair_label=str(electron_pair_label),
        context=context,
    )
    hole = _best_rank_two_projection_assignment(
        hole_reference,
        candidates,
        minimum_assignment_margin=assignment_threshold,
        pair_label=str(hole_pair_label),
        context=context,
    )
    collision = set(electron[0]) & set(hole[0])
    if collision:
        raise ValueError(
            f"{context}: E/H rank-two candidate assignments collide at local "
            f"positions {sorted(collision)}"
        )
    for label, singular_values in (
        (str(electron_pair_label), electron[1]),
        (str(hole_pair_label), hole[1]),
    ):
        if float(np.min(singular_values)) < overlap_threshold:
            raise ValueError(
                f"{context}: {label} principal overlap is below threshold: "
                f"min_s={np.min(singular_values):.3e}, "
                f"required={overlap_threshold:.3e}"
            )
    return _KdotpyE1H1PairAssignment(
        electron_local_candidate_positions=electron[0],
        hole_local_candidate_positions=hole[0],
        electron_principal_singular_values=electron[1],
        hole_principal_singular_values=hole[1],
        electron_selected_projection_weights=electron[2],
        hole_selected_projection_weights=hole[2],
        electron_assignment_margin=electron[3],
        hole_assignment_margin=hole[3],
    )

def _validated_local_position_pair(
    values: tuple[int, int] | np.ndarray, *, label: str
) -> tuple[int, int]:
    raw = tuple(values)
    if len(raw) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError(f"{label} must contain exactly two integer local positions")
    result = tuple(int(value) for value in raw)
    if result[0] == result[1] or min(result) < 0:
        raise ValueError(f"{label} must contain distinct nonnegative local positions")
    return result


def _finite_float(value: float, *, label: str, nonnegative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class _KdotpyE1H1EndpointComparison:
    electron_positions_match: bool
    hole_positions_match: bool
    electron_principal_values: np.ndarray
    hole_principal_values: np.ndarray
    quartet_principal_values: np.ndarray
    projector_roundoff_bound: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("electron_principal_values", (2,)),
            ("hole_principal_values", (2,)),
            ("quartet_principal_values", (4,)),
        ):
            values = _readonly_copy(getattr(self, name), dtype=np.dtype(np.float64))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            object.__setattr__(self, name, values)
        bound = _finite_float(
            self.projector_roundoff_bound,
            label="projector_roundoff_bound",
            nonnegative=True,
        )
        object.__setattr__(self, "projector_roundoff_bound", bound)

    @property
    def matches(self) -> bool:
        bound = self.projector_roundoff_bound
        return bool(
            self.electron_positions_match
            and self.hole_positions_match
            and np.max(np.abs(self.electron_principal_values - 1.0)) <= bound
            and np.max(np.abs(self.hole_principal_values - 1.0)) <= bound
        )


def _compare_e1h1_endpoint_projectors(
    reference_frame: np.ndarray,
    transported_frame: np.ndarray,
    *,
    reference_electron_local_candidate_positions: tuple[int, int] | np.ndarray,
    reference_hole_local_candidate_positions: tuple[int, int] | np.ndarray,
    transported_electron_local_candidate_positions: tuple[int, int] | np.ndarray,
    transported_hole_local_candidate_positions: tuple[int, int] | np.ndarray,
) -> _KdotpyE1H1EndpointComparison:
    """Compare E and H endpoint projectors separately with a roundoff-only gate."""

    reference = _validate_pair_selection_frame(
        reference_frame, label="endpoint_reference_frame", expected_columns=4
    )
    transported = _validate_pair_selection_frame(
        transported_frame, label="endpoint_transported_frame", expected_columns=4
    )
    if reference.shape != transported.shape:
        raise ValueError("endpoint frames must share one parent Hilbert space")
    electron_values = np.linalg.svd(
        reference[:, :2].conj().T @ transported[:, :2], compute_uv=False
    )
    hole_values = np.linalg.svd(
        reference[:, 2:4].conj().T @ transported[:, 2:4], compute_uv=False
    )
    quartet_values = np.linalg.svd(
        reference.conj().T @ transported, compute_uv=False
    )
    reference_gram_error = float(
        np.max(np.abs(reference.conj().T @ reference - np.eye(4)))
    )
    transported_gram_error = float(
        np.max(np.abs(transported.conj().T @ transported - np.eye(4)))
    )
    roundoff_bound = max(
        2048.0 * np.finfo(np.float64).eps * max(reference.shape),
        16.0 * reference_gram_error,
        16.0 * transported_gram_error,
    )
    return _KdotpyE1H1EndpointComparison(
        electron_positions_match=(
            _validated_local_position_pair(
                reference_electron_local_candidate_positions,
                label="reference_electron_local_candidate_positions",
            )
            == _validated_local_position_pair(
                transported_electron_local_candidate_positions,
                label="transported_electron_local_candidate_positions",
            )
        ),
        hole_positions_match=(
            _validated_local_position_pair(
                reference_hole_local_candidate_positions,
                label="reference_hole_local_candidate_positions",
            )
            == _validated_local_position_pair(
                transported_hole_local_candidate_positions,
                label="transported_hole_local_candidate_positions",
            )
        ),
        electron_principal_values=electron_values,
        hole_principal_values=hole_values,
        quartet_principal_values=quartet_values,
        projector_roundoff_bound=roundoff_bound,
    )


@dataclass(frozen=True, slots=True)
class KdotpyPreviousUSameKHomotopyConflictReceipt:
    """Compact certification evidence for one direct E/H assignment conflict."""

    k_index: int
    direct_electron_local_candidate_positions: tuple[int, int]
    direct_hole_local_candidate_positions: tuple[int, int]
    direct_electron_principal_singular_values: np.ndarray
    direct_hole_principal_singular_values: np.ndarray
    direct_electron_assignment_margin: float
    direct_hole_assignment_margin: float
    radial_electron_local_candidate_positions: tuple[int, int]
    radial_hole_local_candidate_positions: tuple[int, int]
    attempted_dyadic_substeps: tuple[int, ...]
    first_matching_substeps: int
    confirmation_substeps: tuple[int, int]
    forward_endpoint_electron_local_candidate_positions: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]
    ]
    forward_endpoint_hole_local_candidate_positions: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int]
    ]
    forward_endpoint_electron_principal_values: np.ndarray
    forward_endpoint_hole_principal_values: np.ndarray
    forward_endpoint_quartet_principal_values: np.ndarray
    previous_electron_local_candidate_positions: tuple[int, int]
    previous_hole_local_candidate_positions: tuple[int, int]
    reverse_endpoint_electron_local_candidate_positions: tuple[int, int]
    reverse_endpoint_hole_local_candidate_positions: tuple[int, int]
    reverse_endpoint_electron_principal_values: np.ndarray
    reverse_endpoint_hole_principal_values: np.ndarray
    reverse_endpoint_quartet_principal_values: np.ndarray
    endpoint_projector_roundoff_bound: float
    minimum_electron_principal_overlap: float
    minimum_hole_principal_overlap: float
    minimum_electron_assignment_margin: float
    minimum_hole_assignment_margin: float
    minimum_selected_complement_gap_mev: float
    maximum_eigensystem_residual_mev: float
    maximum_static_parent_replay_error_mev: float
    maximum_potential_operator_error_mev: float
    transport_trace_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.k_index, (bool, np.bool_))
            or not isinstance(self.k_index, (int, np.integer))
            or int(self.k_index) < 0
        ):
            raise ValueError("k_index must be a nonnegative exact integer")
        object.__setattr__(self, "k_index", int(self.k_index))
        pair_names = (
            "direct_electron_local_candidate_positions",
            "direct_hole_local_candidate_positions",
            "radial_electron_local_candidate_positions",
            "radial_hole_local_candidate_positions",
            "previous_electron_local_candidate_positions",
            "previous_hole_local_candidate_positions",
            "reverse_endpoint_electron_local_candidate_positions",
            "reverse_endpoint_hole_local_candidate_positions",
        )
        for name in pair_names:
            object.__setattr__(
                self,
                name,
                _validated_local_position_pair(getattr(self, name), label=name),
            )
        for name in (
            "forward_endpoint_electron_local_candidate_positions",
            "forward_endpoint_hole_local_candidate_positions",
        ):
            rows = tuple(
                _validated_local_position_pair(row, label=f"{name}[{index}]")
                for index, row in enumerate(tuple(getattr(self, name)))
            )
            if len(rows) != 3:
                raise ValueError(f"{name} must contain N, N+1, and 2N endpoints")
            object.__setattr__(self, name, rows)
        for name, shape in (
            ("direct_electron_principal_singular_values", (2,)),
            ("direct_hole_principal_singular_values", (2,)),
            ("forward_endpoint_electron_principal_values", (3, 2)),
            ("forward_endpoint_hole_principal_values", (3, 2)),
            ("forward_endpoint_quartet_principal_values", (3, 4)),
            ("reverse_endpoint_electron_principal_values", (2,)),
            ("reverse_endpoint_hole_principal_values", (2,)),
            ("reverse_endpoint_quartet_principal_values", (4,)),
        ):
            values = _readonly_copy(getattr(self, name), dtype=np.dtype(np.float64))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, values)
        attempted = tuple(int(value) for value in self.attempted_dyadic_substeps)
        if (
            not attempted
            or attempted[0] != 2
            or any(value < 2 or value & (value - 1) for value in attempted)
            or any(right != 2 * left for left, right in zip(attempted, attempted[1:]))
        ):
            raise ValueError("attempted_dyadic_substeps must be consecutive powers from two")
        first = int(self.first_matching_substeps)
        if first != attempted[-1]:
            raise ValueError("first_matching_substeps must be the last attempted dyadic N")
        confirmation = tuple(int(value) for value in self.confirmation_substeps)
        if confirmation != (first + 1, 2 * first):
            raise ValueError("confirmation_substeps must be exactly (N+1,2N)")
        object.__setattr__(self, "attempted_dyadic_substeps", attempted)
        object.__setattr__(self, "first_matching_substeps", first)
        object.__setattr__(self, "confirmation_substeps", confirmation)
        if (
            self.direct_electron_local_candidate_positions
            == self.radial_electron_local_candidate_positions
            and self.direct_hole_local_candidate_positions
            == self.radial_hole_local_candidate_positions
        ):
            raise ValueError("a homotopy conflict receipt requires direct E/H disagreement")
        if any(
            row != self.radial_electron_local_candidate_positions
            for row in self.forward_endpoint_electron_local_candidate_positions
        ) or any(
            row != self.radial_hole_local_candidate_positions
            for row in self.forward_endpoint_hole_local_candidate_positions
        ):
            raise ValueError("all confirmed forward endpoints must match radial E/H positions")
        bound = _finite_float(
            self.endpoint_projector_roundoff_bound,
            label="endpoint_projector_roundoff_bound",
            nonnegative=True,
        )
        object.__setattr__(self, "endpoint_projector_roundoff_bound", bound)
        for name in (
            "forward_endpoint_electron_principal_values",
            "forward_endpoint_hole_principal_values",
            "reverse_endpoint_electron_principal_values",
            "reverse_endpoint_hole_principal_values",
        ):
            if np.max(np.abs(getattr(self, name) - 1.0)) > bound:
                raise ValueError(f"{name} fails separate E/H projector recovery")
        for name in (
            "direct_electron_assignment_margin",
            "direct_hole_assignment_margin",
            "minimum_electron_principal_overlap",
            "minimum_hole_principal_overlap",
            "minimum_electron_assignment_margin",
            "minimum_hole_assignment_margin",
            "minimum_selected_complement_gap_mev",
            "maximum_eigensystem_residual_mev",
            "maximum_static_parent_replay_error_mev",
            "maximum_potential_operator_error_mev",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), label=name, nonnegative=True),
            )
        if self.minimum_selected_complement_gap_mev <= 1.0e-6:
            raise ValueError("homotopy selected-complement gap fails the existing gate")
        if self.maximum_eigensystem_residual_mev > 1.0e-7:
            raise ValueError("homotopy eigensystem residual fails the existing gate")
        if self.maximum_static_parent_replay_error_mev > 1.0e-10:
            raise ValueError("homotopy static-parent replay fails the existing gate")
        if self.maximum_potential_operator_error_mev > 1.0e-12:
            raise ValueError("homotopy potential operator fails the existing gate")
        object.__setattr__(
            self,
            "transport_trace_sha256",
            _validate_sha256(
                self.transport_trace_sha256, label="transport_trace_sha256"
            ),
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"KdotpyPreviousUSameKHomotopyConflictReceipt:v1")
        for name in (
            "k_index",
            "direct_electron_local_candidate_positions",
            "direct_hole_local_candidate_positions",
            "direct_electron_assignment_margin",
            "direct_hole_assignment_margin",
            "radial_electron_local_candidate_positions",
            "radial_hole_local_candidate_positions",
            "attempted_dyadic_substeps",
            "first_matching_substeps",
            "confirmation_substeps",
            "forward_endpoint_electron_local_candidate_positions",
            "forward_endpoint_hole_local_candidate_positions",
            "previous_electron_local_candidate_positions",
            "previous_hole_local_candidate_positions",
            "reverse_endpoint_electron_local_candidate_positions",
            "reverse_endpoint_hole_local_candidate_positions",
            "endpoint_projector_roundoff_bound",
            "minimum_electron_principal_overlap",
            "minimum_hole_principal_overlap",
            "minimum_electron_assignment_margin",
            "minimum_hole_assignment_margin",
            "minimum_selected_complement_gap_mev",
            "maximum_eigensystem_residual_mev",
            "maximum_static_parent_replay_error_mev",
            "maximum_potential_operator_error_mev",
            "transport_trace_sha256",
        ):
            _hash_length_delimited_text(
                digest, field_label=name, value=repr(getattr(self, name))
            )
        for name in (
            "direct_electron_principal_singular_values",
            "direct_hole_principal_singular_values",
            "forward_endpoint_electron_principal_values",
            "forward_endpoint_hole_principal_values",
            "forward_endpoint_quartet_principal_values",
            "reverse_endpoint_electron_principal_values",
            "reverse_endpoint_hole_principal_values",
            "reverse_endpoint_quartet_principal_values",
        ):
            _hash_length_delimited_text(
                digest,
                field_label=name,
                value=kane_poisson_array_sha256(getattr(self, name)),
            )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class KdotpyPreviousUSameKHomotopyCallReceipt:
    """One-call immutable summary; intermediate homotopy frames are excluded."""

    pair_selection_spec_fingerprint: str
    homotopy_spec_fingerprint: str
    parent_spec_fingerprint: str
    current_potential_sha256: str
    previous_potential_sha256: str | None
    potential_step_max_abs_mev: float | None
    potential_step_rms_mev: float | None
    k_modes: tuple[str, ...]
    conflicts: tuple[KdotpyPreviousUSameKHomotopyConflictReceipt, ...]

    def __post_init__(self) -> None:
        for name in (
            "pair_selection_spec_fingerprint",
            "homotopy_spec_fingerprint",
            "parent_spec_fingerprint",
            "current_potential_sha256",
        ):
            object.__setattr__(
                self, name, _validate_sha256(getattr(self, name), label=name)
            )
        modes = tuple(str(value) for value in self.k_modes)
        if not modes or any(value not in {"initial", "direct", "homotopy"} for value in modes):
            raise ValueError("k_modes must contain initial/direct/homotopy labels")
        conflicts = tuple(self.conflicts)
        if any(
            not isinstance(value, KdotpyPreviousUSameKHomotopyConflictReceipt)
            for value in conflicts
        ):
            raise TypeError("conflicts must contain typed homotopy conflict receipts")
        conflict_indices = tuple(value.k_index for value in conflicts)
        if len(set(conflict_indices)) != len(conflict_indices) or any(
            index >= len(modes) for index in conflict_indices
        ):
            raise ValueError("homotopy conflict k indices must be unique and in range")
        if tuple(index for index, mode in enumerate(modes) if mode == "homotopy") != conflict_indices:
            raise ValueError("k_modes and homotopy conflict receipts disagree")
        if self.previous_potential_sha256 is None:
            if any(value != "initial" for value in modes) or conflicts:
                raise ValueError("an initial call cannot carry previous-U transport evidence")
            if self.potential_step_max_abs_mev is not None or self.potential_step_rms_mev is not None:
                raise ValueError("an initial call cannot carry potential-step metrics")
        else:
            object.__setattr__(
                self,
                "previous_potential_sha256",
                _validate_sha256(
                    self.previous_potential_sha256,
                    label="previous_potential_sha256",
                ),
            )
            if any(value == "initial" for value in modes):
                raise ValueError("a previous-U call cannot carry initial k modes")
            for name in ("potential_step_max_abs_mev", "potential_step_rms_mev"):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required when a previous potential exists")
                object.__setattr__(
                    self,
                    name,
                    _finite_float(
                        getattr(self, name), label=name, nonnegative=True
                    ),
                )
        object.__setattr__(self, "k_modes", modes)
        object.__setattr__(self, "conflicts", conflicts)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"KdotpyPreviousUSameKHomotopyCallReceipt:v1")
        for name in (
            "pair_selection_spec_fingerprint",
            "homotopy_spec_fingerprint",
            "parent_spec_fingerprint",
            "current_potential_sha256",
            "previous_potential_sha256",
            "potential_step_max_abs_mev",
            "potential_step_rms_mev",
            "k_modes",
        ):
            _hash_length_delimited_text(
                digest, field_label=name, value=repr(getattr(self, name))
            )
        for index, conflict in enumerate(self.conflicts):
            _hash_length_delimited_text(
                digest,
                field_label=f"conflict_{index}_fingerprint",
                value=conflict.fingerprint,
            )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class KdotpyPreviousUSameKHomotopyFailureReceipt:
    """Immutable evidence attached to a fail-closed homotopy exception."""

    reason_code: str
    detail: str
    k_index: int
    pair_selection_spec_fingerprint: str
    homotopy_spec_fingerprint: str
    parent_spec_fingerprint: str
    previous_potential_sha256: str
    current_potential_sha256: str
    potential_step_max_abs_mev: float
    potential_step_rms_mev: float
    attempted_dyadic_substeps: tuple[int, ...]
    first_matching_substeps: int | None
    direct_electron_local_candidate_positions: tuple[int, int]
    direct_hole_local_candidate_positions: tuple[int, int]
    direct_electron_principal_singular_values: np.ndarray
    direct_hole_principal_singular_values: np.ndarray
    direct_electron_assignment_margin: float
    direct_hole_assignment_margin: float
    radial_electron_local_candidate_positions: tuple[int, int]
    radial_hole_local_candidate_positions: tuple[int, int]
    transport_trace_sha256: str

    def __post_init__(self) -> None:
        reason = str(self.reason_code)
        if reason not in {
            "maximum_substeps_exhausted",
            "transport_gate_failed",
            "confirmation_failed",
            "reverse_recovery_failed",
        }:
            raise ValueError("unsupported homotopy failure reason_code")
        detail = str(self.detail)
        if not detail:
            raise ValueError("homotopy failure detail must be nonempty")
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "detail", detail)
        if int(self.k_index) != self.k_index or int(self.k_index) < 0:
            raise ValueError("k_index must be a nonnegative exact integer")
        object.__setattr__(self, "k_index", int(self.k_index))
        for name in (
            "pair_selection_spec_fingerprint",
            "homotopy_spec_fingerprint",
            "parent_spec_fingerprint",
            "previous_potential_sha256",
            "current_potential_sha256",
            "transport_trace_sha256",
        ):
            object.__setattr__(
                self, name, _validate_sha256(getattr(self, name), label=name)
            )
        for name in (
            "direct_electron_local_candidate_positions",
            "direct_hole_local_candidate_positions",
            "radial_electron_local_candidate_positions",
            "radial_hole_local_candidate_positions",
        ):
            object.__setattr__(
                self,
                name,
                _validated_local_position_pair(getattr(self, name), label=name),
            )
        for name in (
            "direct_electron_principal_singular_values",
            "direct_hole_principal_singular_values",
        ):
            values = _readonly_copy(getattr(self, name), dtype=np.dtype(np.float64))
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite rank-two array")
            object.__setattr__(self, name, values)
        attempted = tuple(int(value) for value in self.attempted_dyadic_substeps)
        if attempted and (
            attempted[0] != 2
            or any(value < 2 or value & (value - 1) for value in attempted)
            or any(right != 2 * left for left, right in zip(attempted, attempted[1:]))
        ):
            raise ValueError("attempted dyadic substeps are malformed")
        object.__setattr__(self, "attempted_dyadic_substeps", attempted)
        if self.first_matching_substeps is not None:
            first = int(self.first_matching_substeps)
            if first not in attempted:
                raise ValueError("first_matching_substeps must be one attempted dyadic N")
            object.__setattr__(self, "first_matching_substeps", first)
        for name in (
            "potential_step_max_abs_mev",
            "potential_step_rms_mev",
            "direct_electron_assignment_margin",
            "direct_hole_assignment_margin",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), label=name, nonnegative=True),
            )


class KdotpyPreviousUSameKHomotopyError(RuntimeError):
    """Typed fail-closed error carrying compact immutable transport evidence."""

    def __init__(self, receipt: KdotpyPreviousUSameKHomotopyFailureReceipt) -> None:
        if not isinstance(receipt, KdotpyPreviousUSameKHomotopyFailureReceipt):
            raise TypeError("receipt must be KdotpyPreviousUSameKHomotopyFailureReceipt")
        self.receipt = receipt
        super().__init__(
            "previous-U same-k potential homotopy failed closed at "
            f"ik={receipt.k_index}: {receipt.reason_code}: {receipt.detail}"
        )


@dataclass(frozen=True, slots=True)
class _KdotpyFixedKTransportResult:
    assignment: _KdotpyE1H1PairAssignment
    endpoint_frame: np.ndarray
    minimum_electron_principal_overlap: float
    minimum_hole_principal_overlap: float
    minimum_electron_assignment_margin: float
    minimum_hole_assignment_margin: float
    minimum_selected_complement_gap_mev: float
    maximum_eigensystem_residual_mev: float
    maximum_static_parent_replay_error_mev: float
    maximum_potential_operator_error_mev: float
    trace_sha256: str

    def __post_init__(self) -> None:
        frame = _readonly_copy(self.endpoint_frame, dtype=np.dtype(np.complex128))
        if frame.ndim != 2 or frame.shape[1] != 4 or not np.all(np.isfinite(frame)):
            raise ValueError("transport endpoint_frame must be a finite quartet")
        object.__setattr__(self, "endpoint_frame", frame)
        object.__setattr__(
            self,
            "trace_sha256",
            _validate_sha256(self.trace_sha256, label="trace_sha256"),
        )


class _KdotpyHomotopyPathGateError(RuntimeError):
    def __init__(self, reason: str, detail: str, trace_sha256: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        self.trace_sha256 = _validate_sha256(
            trace_sha256, label="failed_path_trace_sha256"
        )
        super().__init__(f"{self.reason}: {self.detail}")


@dataclass(frozen=True)
class KdotpyParentPotentialAttestation:
    input_potential_sha256: str
    potential_operator_fingerprint: str
    actual_full_parent_sequence_sha256: str
    raw_replayed_static_parent_sha256: str
    replayed_static_parent_sha256: str
    static_parent_replay_max_error_mev: float
    potential_operator_max_error_mev: float


@dataclass(frozen=True)
class KdotpyFullParentEigensystem:
    energies_mev: np.ndarray
    eigenvectors: np.ndarray
    attestation: KdotpyParentPotentialAttestation
    hermiticity_error_mev: float
    eigen_residual_mev: float
    orthonormality_error: float
    completeness_error: float

    def __post_init__(self) -> None:
        energies = np.array(self.energies_mev, dtype=np.float64, copy=True, order="C")
        vectors = np.array(self.eigenvectors, dtype=np.complex128, copy=True, order="C")
        energies.setflags(write=False)
        vectors.setflags(write=False)
        object.__setattr__(self, "energies_mev", energies)
        object.__setattr__(self, "eigenvectors", vectors)


@dataclass(frozen=True, slots=True)
class KdotpyWindowDiagnostics:
    """Immutable candidate-table and continuity evidence for one builder call."""

    anchor_candidate_energies_mev: np.ndarray
    anchor_selected_local_candidate_positions: np.ndarray
    anchor_selection_projection_weights: np.ndarray
    anchor_subspace_singular_values: np.ndarray
    anchor_candidate_edge_gap_mev: float
    anchor_eigen_residual_mev: float
    anchor_full_parent_sha256: str
    candidate_energies_mev: np.ndarray
    selected_local_candidate_positions: np.ndarray
    selected_projection_weights: np.ndarray
    selected_subspace_singular_values: np.ndarray
    candidate_edge_gap_mev: np.ndarray
    candidate_raw_orthonormality_error: np.ndarray
    selected_eigen_residual_mev: np.ndarray
    actual_full_parent_sequence_sha256: str
    raw_replayed_static_parent_sha256: str
    replayed_static_parent_sha256: str
    static_parent_replay_max_error_mev: float
    potential_operator_max_error_mev: float
    pair_selection_spec_fingerprint: str
    pair_labels: tuple[str, str]
    ordered_physical_selected_labels: tuple[int, int, int, int]
    gamma_seed_status: str
    minimum_principal_overlap: float
    minimum_assignment_margin: float
    anchor_electron_selected_local_candidate_positions: np.ndarray
    anchor_hole_selected_local_candidate_positions: np.ndarray
    anchor_electron_principal_singular_values: np.ndarray
    anchor_hole_principal_singular_values: np.ndarray
    anchor_electron_assignment_margin: float
    anchor_hole_assignment_margin: float
    electron_selected_local_candidate_positions: np.ndarray
    hole_selected_local_candidate_positions: np.ndarray
    current_u_adjacent_k_electron_principal_singular_values: np.ndarray
    current_u_adjacent_k_hole_principal_singular_values: np.ndarray
    current_u_adjacent_k_electron_assignment_margin: np.ndarray
    current_u_adjacent_k_hole_assignment_margin: np.ndarray
    previous_u_same_k_electron_selected_local_candidate_positions: np.ndarray | None = None
    previous_u_same_k_hole_selected_local_candidate_positions: np.ndarray | None = None
    previous_u_same_k_electron_principal_singular_values: np.ndarray | None = None
    previous_u_same_k_hole_principal_singular_values: np.ndarray | None = None
    previous_u_same_k_electron_assignment_margin: np.ndarray | None = None
    previous_u_same_k_hole_assignment_margin: np.ndarray | None = None
    previous_u_same_k_homotopy_receipt: (
        KdotpyPreviousUSameKHomotopyCallReceipt | None
    ) = None
    pair_selection_diagnostics_sha256: str | None = None

    def __post_init__(self) -> None:
        float_array_names = (
            "anchor_candidate_energies_mev",
            "anchor_selection_projection_weights",
            "anchor_subspace_singular_values",
            "candidate_energies_mev",
            "selected_projection_weights",
            "selected_subspace_singular_values",
            "candidate_edge_gap_mev",
            "candidate_raw_orthonormality_error",
            "selected_eigen_residual_mev",
            "anchor_electron_principal_singular_values",
            "anchor_hole_principal_singular_values",
            "current_u_adjacent_k_electron_principal_singular_values",
            "current_u_adjacent_k_hole_principal_singular_values",
            "current_u_adjacent_k_electron_assignment_margin",
            "current_u_adjacent_k_hole_assignment_margin",
            "previous_u_same_k_electron_principal_singular_values",
            "previous_u_same_k_hole_principal_singular_values",
            "previous_u_same_k_electron_assignment_margin",
            "previous_u_same_k_hole_assignment_margin",
        )
        integer_array_names = (
            "anchor_selected_local_candidate_positions",
            "selected_local_candidate_positions",
            "anchor_electron_selected_local_candidate_positions",
            "anchor_hole_selected_local_candidate_positions",
            "electron_selected_local_candidate_positions",
            "hole_selected_local_candidate_positions",
            "previous_u_same_k_electron_selected_local_candidate_positions",
            "previous_u_same_k_hole_selected_local_candidate_positions",
        )
        for name in float_array_names:
            values = getattr(self, name)
            if values is not None:
                array = _readonly_copy(values, dtype=np.dtype(np.float64))
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be finite")
                object.__setattr__(self, name, array)
        for name in integer_array_names:
            values = getattr(self, name)
            if values is not None:
                raw = np.asarray(values)
                if not np.issubdtype(raw.dtype, np.integer):
                    raise ValueError(f"{name} must contain integer local positions")
                array = _readonly_copy(raw, dtype=np.dtype(np.int64))
                if np.any(array < 0):
                    raise ValueError(f"{name} must contain nonnegative local positions")
                object.__setattr__(self, name, array)

        candidate_energies = self.candidate_energies_mev
        selected_positions = self.selected_local_candidate_positions
        if candidate_energies.ndim != 2 or candidate_energies.shape[0] < 1:
            raise ValueError("candidate_energies_mev must have shape (nk,ncandidate)")
        nk, candidate_count = candidate_energies.shape
        if self.anchor_candidate_energies_mev.shape != (candidate_count,):
            raise ValueError("anchor and radial candidate tables must have equal width")
        if selected_positions.ndim != 2 or selected_positions.shape[0] != nk:
            raise ValueError(
                "selected_local_candidate_positions must have shape (nk,nwindow)"
            )
        nwindow = selected_positions.shape[1]
        expected_window_shape = (nk, nwindow)
        for name in ("selected_projection_weights", "selected_subspace_singular_values"):
            if getattr(self, name).shape != expected_window_shape:
                raise ValueError(f"{name} must have shape (nk,nwindow)")
        for name in (
            "candidate_edge_gap_mev",
            "candidate_raw_orthonormality_error",
            "selected_eigen_residual_mev",
        ):
            if getattr(self, name).shape != (nk,):
                raise ValueError(f"{name} must have shape (nk,)")
        if self.anchor_selected_local_candidate_positions.shape != (nwindow,):
            raise ValueError(
                "anchor_selected_local_candidate_positions must have shape (nwindow,)"
            )
        for name in (
            "anchor_selection_projection_weights",
            "anchor_subspace_singular_values",
        ):
            if getattr(self, name).shape != (nwindow,):
                raise ValueError(f"{name} must have shape (nwindow,)")
        for row in selected_positions:
            if np.unique(row).size != nwindow or np.max(row) >= candidate_count:
                raise ValueError(
                    "selected local candidate positions must be distinct and in range"
                )
        if (
            np.unique(self.anchor_selected_local_candidate_positions).size != nwindow
            or np.max(self.anchor_selected_local_candidate_positions) >= candidate_count
        ):
            raise ValueError(
                "anchor local candidate positions must be distinct and in range"
            )
        for name in (
            "anchor_candidate_edge_gap_mev",
            "anchor_eigen_residual_mev",
            "static_parent_replay_max_error_mev",
            "potential_operator_max_error_mev",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in (
            "anchor_full_parent_sha256",
            "actual_full_parent_sequence_sha256",
            "raw_replayed_static_parent_sha256",
            "replayed_static_parent_sha256",
        ):
            object.__setattr__(
                self, name, _validate_sha256(getattr(self, name), label=name)
            )

        previous_u_fields = (
            "previous_u_same_k_electron_selected_local_candidate_positions",
            "previous_u_same_k_hole_selected_local_candidate_positions",
            "previous_u_same_k_electron_principal_singular_values",
            "previous_u_same_k_hole_principal_singular_values",
            "previous_u_same_k_electron_assignment_margin",
            "previous_u_same_k_hole_assignment_margin",
        )
        object.__setattr__(
            self,
            "pair_selection_spec_fingerprint",
            _validate_sha256(
                self.pair_selection_spec_fingerprint,
                label="pair_selection_spec_fingerprint",
            ),
        )
        previous_presence = tuple(
            getattr(self, name) is not None for name in previous_u_fields
        )
        if any(previous_presence) and not all(previous_presence):
            raise ValueError(
                "previous-U same-k diagnostics must be present as one complete set"
            )
        labels = tuple(str(value) for value in self.pair_labels)
        if (
            len(labels) != 2
            or not all(value.strip() for value in labels)
            or labels[0] == labels[1]
        ):
            raise ValueError("pair_labels must contain distinct nonempty E/H labels")
        object.__setattr__(self, "pair_labels", labels)
        physical_labels = tuple(self.ordered_physical_selected_labels)
        if (
            len(physical_labels) != 4
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) == 0
                for value in physical_labels
            )
            or len(set(int(value) for value in physical_labels)) != 4
        ):
            raise ValueError(
                "ordered_physical_selected_labels must contain four exact unique labels"
            )
        object.__setattr__(
            self,
            "ordered_physical_selected_labels",
            tuple(int(value) for value in physical_labels),
        )
        if self.gamma_seed_status != DECLARED_GAMMA_SEED_STATUS:
            raise ValueError("Gamma seed must be labeled as a declaration, not evidence")
        for name in ("minimum_principal_overlap", "minimum_assignment_margin"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0,1]")
            object.__setattr__(self, name, value)
        for name in ("anchor_electron_assignment_margin", "anchor_hole_assignment_margin"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < self.minimum_assignment_margin:
                raise ValueError(f"{name} fails minimum_assignment_margin")
            object.__setattr__(self, name, value)

        pair_shape = (nk, 2)
        for name in (
            "electron_selected_local_candidate_positions",
            "hole_selected_local_candidate_positions",
            "current_u_adjacent_k_electron_principal_singular_values",
            "current_u_adjacent_k_hole_principal_singular_values",
        ):
            if getattr(self, name).shape != pair_shape:
                raise ValueError(f"{name} must have shape (nk,2)")
        for name in (
            "current_u_adjacent_k_electron_assignment_margin",
            "current_u_adjacent_k_hole_assignment_margin",
        ):
            values = getattr(self, name)
            if values.shape != (nk,) or np.any(values < self.minimum_assignment_margin):
                raise ValueError(f"{name} fails minimum_assignment_margin")
        for name in (
            "anchor_electron_principal_singular_values",
            "anchor_hole_principal_singular_values",
            "current_u_adjacent_k_electron_principal_singular_values",
            "current_u_adjacent_k_hole_principal_singular_values",
        ):
            if np.any(getattr(self, name) < self.minimum_principal_overlap):
                raise ValueError(f"{name} fails minimum_principal_overlap")
        for name in (
            "anchor_electron_selected_local_candidate_positions",
            "anchor_hole_selected_local_candidate_positions",
            "anchor_electron_principal_singular_values",
            "anchor_hole_principal_singular_values",
        ):
            if getattr(self, name).shape != (2,):
                raise ValueError(f"{name} must have shape (2,)")
        combined_anchor = np.concatenate(
            (
                self.anchor_electron_selected_local_candidate_positions,
                self.anchor_hole_selected_local_candidate_positions,
            )
        )
        combined_rows = np.concatenate(
            (
                self.electron_selected_local_candidate_positions,
                self.hole_selected_local_candidate_positions,
            ),
            axis=1,
        )
        if not np.array_equal(combined_anchor, self.anchor_selected_local_candidate_positions):
            raise ValueError("anchor E/H local positions do not match ordered columns")
        if not np.array_equal(combined_rows, self.selected_local_candidate_positions):
            raise ValueError("radial E/H local positions do not match ordered columns")
        if all(previous_presence):
            for name in (
                "previous_u_same_k_electron_selected_local_candidate_positions",
                "previous_u_same_k_hole_selected_local_candidate_positions",
                "previous_u_same_k_electron_principal_singular_values",
                "previous_u_same_k_hole_principal_singular_values",
            ):
                if getattr(self, name).shape != pair_shape:
                    raise ValueError(f"{name} must have shape (nk,2)")
            for name in (
                "previous_u_same_k_electron_assignment_margin",
                "previous_u_same_k_hole_assignment_margin",
            ):
                values = getattr(self, name)
                if values.shape != (nk,) or np.any(values < self.minimum_assignment_margin):
                    raise ValueError(f"{name} fails minimum_assignment_margin")
            for name in (
                "previous_u_same_k_electron_principal_singular_values",
                "previous_u_same_k_hole_principal_singular_values",
            ):
                if np.any(getattr(self, name) < self.minimum_principal_overlap):
                    raise ValueError(f"{name} fails minimum_principal_overlap")

        homotopy_receipt = self.previous_u_same_k_homotopy_receipt
        if homotopy_receipt is None:
            # Preserve the established direct-only invariant and fingerprint
            # exactly when homotopy is not enabled.
            if all(previous_presence) and (
                not np.array_equal(
                    self.previous_u_same_k_electron_selected_local_candidate_positions,
                    self.electron_selected_local_candidate_positions,
                )
                or not np.array_equal(
                    self.previous_u_same_k_hole_selected_local_candidate_positions,
                    self.hole_selected_local_candidate_positions,
                )
            ):
                raise ValueError(
                    "current-U adjacent-k and previous-U same-k assignments disagree"
                )
        else:
            if not isinstance(
                homotopy_receipt, KdotpyPreviousUSameKHomotopyCallReceipt
            ):
                raise TypeError(
                    "previous_u_same_k_homotopy_receipt must be a typed call receipt"
                )
            if (
                homotopy_receipt.pair_selection_spec_fingerprint
                != self.pair_selection_spec_fingerprint
            ):
                raise ValueError("homotopy receipt carries a different pair-selection spec")
            if len(homotopy_receipt.k_modes) != nk:
                raise ValueError("homotopy receipt k_modes length differs from nk")
            if all(previous_presence):
                if homotopy_receipt.previous_potential_sha256 is None:
                    raise ValueError("previous-U diagnostics require a previous potential hash")
                conflicts_by_k = {
                    conflict.k_index: conflict
                    for conflict in homotopy_receipt.conflicts
                }
                for ik, mode in enumerate(homotopy_receipt.k_modes):
                    direct_electron = tuple(
                        int(value)
                        for value in self.previous_u_same_k_electron_selected_local_candidate_positions[
                            ik
                        ]
                    )
                    direct_hole = tuple(
                        int(value)
                        for value in self.previous_u_same_k_hole_selected_local_candidate_positions[
                            ik
                        ]
                    )
                    radial_electron = tuple(
                        int(value)
                        for value in self.electron_selected_local_candidate_positions[ik]
                    )
                    radial_hole = tuple(
                        int(value)
                        for value in self.hole_selected_local_candidate_positions[ik]
                    )
                    if mode == "direct":
                        if direct_electron != radial_electron or direct_hole != radial_hole:
                            raise ValueError(
                                "a direct-mode k point carries unresolved E/H disagreement"
                            )
                        continue
                    if mode != "homotopy" or ik not in conflicts_by_k:
                        raise ValueError("previous-U k mode lacks matching transport evidence")
                    conflict = conflicts_by_k[ik]
                    if (
                        conflict.direct_electron_local_candidate_positions
                        != direct_electron
                        or conflict.direct_hole_local_candidate_positions != direct_hole
                        or conflict.radial_electron_local_candidate_positions
                        != radial_electron
                        or conflict.radial_hole_local_candidate_positions != radial_hole
                    ):
                        raise ValueError("homotopy conflict positions differ from diagnostics")
                    if not np.array_equal(
                        conflict.direct_electron_principal_singular_values,
                        self.previous_u_same_k_electron_principal_singular_values[ik],
                    ) or not np.array_equal(
                        conflict.direct_hole_principal_singular_values,
                        self.previous_u_same_k_hole_principal_singular_values[ik],
                    ):
                        raise ValueError("homotopy direct singular values differ from diagnostics")
                    if (
                        conflict.direct_electron_assignment_margin
                        != self.previous_u_same_k_electron_assignment_margin[ik]
                        or conflict.direct_hole_assignment_margin
                        != self.previous_u_same_k_hole_assignment_margin[ik]
                    ):
                        raise ValueError("homotopy direct margins differ from diagnostics")
                    if (
                        conflict.minimum_electron_principal_overlap
                        < self.minimum_principal_overlap
                        or conflict.minimum_hole_principal_overlap
                        < self.minimum_principal_overlap
                        or conflict.minimum_electron_assignment_margin
                        < self.minimum_assignment_margin
                        or conflict.minimum_hole_assignment_margin
                        < self.minimum_assignment_margin
                    ):
                        raise ValueError("homotopy receipt relaxed configured E/H gates")
            else:
                if homotopy_receipt.previous_potential_sha256 is not None or any(
                    mode != "initial" for mode in homotopy_receipt.k_modes
                ):
                    raise ValueError("initial pair diagnostics carry warmed homotopy state")

        expected_pair_fingerprint = self._computed_pair_selection_fingerprint()
        if self.pair_selection_diagnostics_sha256 is None:
            object.__setattr__(
                self,
                "pair_selection_diagnostics_sha256",
                expected_pair_fingerprint,
            )
        elif _validate_sha256(
            self.pair_selection_diagnostics_sha256,
            label="pair_selection_diagnostics_sha256",
        ) != expected_pair_fingerprint:
            raise ValueError("pair-selection diagnostics fingerprint is stale or tampered")

    def _computed_pair_selection_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"KdotpyE1H1PairSelectionDiagnostics:v2")
        _hash_length_delimited_text(
            digest,
            field_label="pair_selection_spec_fingerprint",
            value=str(self.pair_selection_spec_fingerprint),
        )
        for index, label in enumerate(self.pair_labels):
            _hash_length_delimited_text(
                digest, field_label=f"pair_label_{index}", value=label
            )
        _hash_length_delimited_text(
            digest,
            field_label="gamma_seed_status",
            value=str(self.gamma_seed_status),
        )
        digest.update(repr(self.ordered_physical_selected_labels).encode())
        digest.update(repr(self.minimum_principal_overlap).encode())
        digest.update(repr(self.minimum_assignment_margin).encode())
        for name in (
            "anchor_electron_selected_local_candidate_positions",
            "anchor_hole_selected_local_candidate_positions",
            "anchor_electron_principal_singular_values",
            "anchor_hole_principal_singular_values",
            "electron_selected_local_candidate_positions",
            "hole_selected_local_candidate_positions",
            "current_u_adjacent_k_electron_principal_singular_values",
            "current_u_adjacent_k_hole_principal_singular_values",
            "current_u_adjacent_k_electron_assignment_margin",
            "current_u_adjacent_k_hole_assignment_margin",
            "previous_u_same_k_electron_selected_local_candidate_positions",
            "previous_u_same_k_hole_selected_local_candidate_positions",
            "previous_u_same_k_electron_principal_singular_values",
            "previous_u_same_k_hole_principal_singular_values",
            "previous_u_same_k_electron_assignment_margin",
            "previous_u_same_k_hole_assignment_margin",
        ):
            values = getattr(self, name)
            _hash_length_delimited_text(
                digest, field_label=f"{name}_presence", value=str(values is not None)
            )
            if values is not None:
                digest.update(kane_poisson_array_sha256(values).encode())
        digest.update(
            kane_poisson_array_sha256(
                np.asarray(
                    [
                        self.anchor_electron_assignment_margin,
                        self.anchor_hole_assignment_margin,
                    ],
                    dtype=np.float64,
                )
            ).encode()
        )
        if self.previous_u_same_k_homotopy_receipt is not None:
            _hash_length_delimited_text(
                digest,
                field_label="previous_u_same_k_homotopy_receipt_fingerprint",
                value=self.previous_u_same_k_homotopy_receipt.fingerprint,
            )
        return digest.hexdigest()

    def verify_integrity(self) -> None:
        """Recheck pair evidence in case frozen fields were forcibly replaced."""

        expected = self._computed_pair_selection_fingerprint()
        if self.pair_selection_diagnostics_sha256 != expected:
            raise ValueError("pair-selection diagnostics are stale or tampered")

    @property
    def fingerprint(self) -> str:
        self.verify_integrity()
        digest = hashlib.sha256()
        digest.update(b"KdotpyWindowDiagnostics:v2")
        for name in (
            "anchor_candidate_energies_mev",
            "anchor_selected_local_candidate_positions",
            "anchor_selection_projection_weights",
            "anchor_subspace_singular_values",
            "candidate_energies_mev",
            "selected_local_candidate_positions",
            "selected_projection_weights",
            "selected_subspace_singular_values",
            "candidate_edge_gap_mev",
            "candidate_raw_orthonormality_error",
            "selected_eigen_residual_mev",
        ):
            digest.update(kane_poisson_array_sha256(getattr(self, name)).encode())
        for name in (
            "anchor_candidate_edge_gap_mev",
            "anchor_eigen_residual_mev",
            "static_parent_replay_max_error_mev",
            "potential_operator_max_error_mev",
        ):
            _hash_length_delimited_text(
                digest, field_label=name, value=repr(getattr(self, name))
            )
        for name in (
            "anchor_full_parent_sha256",
            "actual_full_parent_sequence_sha256",
            "raw_replayed_static_parent_sha256",
            "replayed_static_parent_sha256",
            "pair_selection_diagnostics_sha256",
        ):
            _hash_length_delimited_text(
                digest, field_label=name, value=str(getattr(self, name))
            )
        return digest.hexdigest()


def _candidate_energy_table_sha256(diagnostics: KdotpyWindowDiagnostics) -> str:
    digest = hashlib.sha256()
    digest.update(b"KdotpyLocalCandidateEnergyTables:v1")
    digest.update(
        kane_poisson_array_sha256(diagnostics.anchor_candidate_energies_mev).encode()
    )
    digest.update(kane_poisson_array_sha256(diagnostics.candidate_energies_mev).encode())
    return digest.hexdigest()


def _local_candidate_positions_namespace_fingerprint(
    *,
    namespace_label: str,
    candidate_energy_table_sha256: str,
    parent_spec_fingerprint: str,
    material_profile_sha256: str,
    diagnostics: KdotpyWindowDiagnostics,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"KdotpyLocalCandidatePositionsNamespace:v1")
    _hash_length_delimited_text(
        digest, field_label="namespace_label", value=namespace_label
    )
    for label, value in (
        ("candidate_energy_table_sha256", candidate_energy_table_sha256),
        ("parent_spec_fingerprint", parent_spec_fingerprint),
        ("material_profile_sha256", material_profile_sha256),
        (
            "actual_full_parent_sequence_sha256",
            diagnostics.actual_full_parent_sequence_sha256,
        ),
        ("anchor_full_parent_sha256", diagnostics.anchor_full_parent_sha256),
    ):
        _hash_length_delimited_text(digest, field_label=label, value=value)
    return digest.hexdigest()


def _pair_resolved_same_call_fingerprint(
    *,
    window: KaneWindowAtPotential,
    diagnostics: KdotpyWindowDiagnostics,
    parent_spec_fingerprint: str,
    material_profile_sha256: str,
    candidate_energy_table_sha256: str,
    local_candidate_positions_namespace_label: str,
    local_candidate_positions_namespace_fingerprint: str,
    input_potential_mev: np.ndarray,
    xml_runtime_binding_status: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"KdotpyPairResolvedWindow:v1")
    for label, value in (
        ("window_exact_archive_fingerprint", window.exact_archive_fingerprint),
        ("diagnostics_fingerprint", diagnostics.fingerprint),
        ("parent_spec_fingerprint", parent_spec_fingerprint),
        ("material_profile_sha256", material_profile_sha256),
        ("candidate_energy_table_sha256", candidate_energy_table_sha256),
        (
            "local_candidate_positions_namespace_label",
            local_candidate_positions_namespace_label,
        ),
        (
            "local_candidate_positions_namespace_fingerprint",
            local_candidate_positions_namespace_fingerprint,
        ),
        ("xml_runtime_binding_status", xml_runtime_binding_status),
    ):
        _hash_length_delimited_text(digest, field_label=label, value=value)
    digest.update(
        kane_poisson_array_sha256(
            np.asarray(input_potential_mev, dtype=np.float64)
        ).encode()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class KdotpyPairResolvedWindow:
    """Same-call immutable pair window and local-position provenance receipt.

    The receipt carries the parent and material-profile fingerprints but does
    not attest that an external XML source was the one loaded at runtime.  The
    Gamma positions remain a declared seed, not independent identity evidence.
    """

    window: KaneWindowAtPotential
    diagnostics: KdotpyWindowDiagnostics
    parent_spec: KaneParentSourceSpec
    input_potential_mev: np.ndarray
    parent_spec_fingerprint: str
    material_profile_sha256: str
    candidate_energy_table_sha256: str
    local_candidate_positions_namespace_label: str
    local_candidate_positions_namespace_fingerprint: str
    same_call_fingerprint: str
    xml_runtime_binding_status: str = XML_RUNTIME_BINDING_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_potential_mev",
            _readonly_copy(self.input_potential_mev, dtype=np.dtype(np.float64)),
        )
        self._validate_integrity()

    def _expected_same_call_fingerprint(self) -> str:
        return _pair_resolved_same_call_fingerprint(
            window=self.window,
            diagnostics=self.diagnostics,
            parent_spec_fingerprint=self.parent_spec_fingerprint,
            material_profile_sha256=self.material_profile_sha256,
            candidate_energy_table_sha256=self.candidate_energy_table_sha256,
            local_candidate_positions_namespace_label=(
                self.local_candidate_positions_namespace_label
            ),
            local_candidate_positions_namespace_fingerprint=(
                self.local_candidate_positions_namespace_fingerprint
            ),
            input_potential_mev=self.input_potential_mev,
            xml_runtime_binding_status=self.xml_runtime_binding_status,
        )

    def _validate_integrity(self) -> None:
        if not isinstance(self.window, KaneWindowAtPotential):
            raise TypeError("window must be KaneWindowAtPotential")
        if not isinstance(self.diagnostics, KdotpyWindowDiagnostics):
            raise TypeError("diagnostics must be KdotpyWindowDiagnostics")
        if not isinstance(self.parent_spec, KaneParentSourceSpec):
            raise TypeError("parent_spec must be KaneParentSourceSpec")
        self.diagnostics.verify_integrity()
        homotopy_receipt = self.diagnostics.previous_u_same_k_homotopy_receipt
        if homotopy_receipt is not None:
            if (
                homotopy_receipt.current_potential_sha256
                != kane_poisson_array_sha256(self.input_potential_mev)
            ):
                raise ValueError("homotopy receipt is bound to a different current potential")
            if (
                homotopy_receipt.parent_spec_fingerprint
                != self.parent_spec.fingerprint
            ):
                raise ValueError("homotopy receipt is bound to a different parent spec")
            parent_options = dict(self.parent_spec.hamiltonian_options)
            if parent_options.get(
                "pair_selection_previous_u_same_k_homotopy_spec_sha256"
            ) != homotopy_receipt.homotopy_spec_fingerprint:
                raise ValueError("homotopy receipt differs from the parent option contract")
            if parent_options.get("pair_selection_spec_sha256") != (
                homotopy_receipt.pair_selection_spec_fingerprint
            ):
                raise ValueError("homotopy receipt differs from the parent pair spec")
        self.window.validate(
            input_potential_mev=self.input_potential_mev,
            parent_spec=self.parent_spec,
        )
        expected_parent = self.parent_spec.fingerprint
        if _validate_sha256(
            self.parent_spec_fingerprint, label="parent_spec_fingerprint"
        ) != expected_parent:
            raise ValueError("pair return carries a stale parent-spec fingerprint")
        if self.window.parent_spec_fingerprint != expected_parent:
            raise ValueError("window and pair return have different parent fingerprints")
        expected_material = self.parent_spec.material_profile_sha256
        if _validate_sha256(
            self.material_profile_sha256, label="material_profile_sha256"
        ) != expected_material:
            raise ValueError("pair return carries a stale material-profile fingerprint")
        if (
            tuple(self.window.selected_band_indices)
            != self.diagnostics.ordered_physical_selected_labels
        ):
            raise ValueError("returned labels do not match the ordered E-then-H columns")
        expected_table = _candidate_energy_table_sha256(self.diagnostics)
        if _validate_sha256(
            self.candidate_energy_table_sha256,
            label="candidate_energy_table_sha256",
        ) != expected_table:
            raise ValueError("local candidate-position energy table is stale or tampered")
        if self.local_candidate_positions_namespace_label != (
            LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL
        ):
            raise ValueError("unsupported local candidate-position namespace")
        expected_namespace = _local_candidate_positions_namespace_fingerprint(
            namespace_label=self.local_candidate_positions_namespace_label,
            candidate_energy_table_sha256=self.candidate_energy_table_sha256,
            parent_spec_fingerprint=self.parent_spec_fingerprint,
            material_profile_sha256=self.material_profile_sha256,
            diagnostics=self.diagnostics,
        )
        if _validate_sha256(
            self.local_candidate_positions_namespace_fingerprint,
            label="local_candidate_positions_namespace_fingerprint",
        ) != expected_namespace:
            raise ValueError("local candidate-position namespace is stale or tampered")
        if self.xml_runtime_binding_status != XML_RUNTIME_BINDING_STATUS:
            raise ValueError("XML runtime binding must remain explicitly external")
        if _validate_sha256(
            self.same_call_fingerprint, label="same_call_fingerprint"
        ) != self._expected_same_call_fingerprint():
            raise ValueError("pair window and diagnostics are not from the same call")
