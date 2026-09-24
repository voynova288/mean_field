"""Stateful kdotpy parent/window builders for the Kane--Poisson adapter.

Pair-selection, homotopy, and immutable receipt types remain owned by
:mod:`.kdotpy_poisson_adapter`; this module owns only the stateful parent
construction and canonical-window orchestration.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from .kane_poisson import (
    KaneParentSourceSpec,
    KaneWindowAtPotential,
    canonical_parent_matrix_sequence_sha256,
    kane_poisson_array_sha256,
    kane_potential_operator_fingerprint,
)
from .kdotpy_poisson_adapter import (
    DECLARED_GAMMA_SEED_STATUS,
    LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL,
    XML_RUNTIME_BINDING_STATUS,
    KdotpyE1H1PairSelectionSpec,
    KdotpyFullParentEigensystem,
    KdotpyPairResolvedWindow,
    KdotpyParentPotentialAttestation,
    KdotpyPreviousUSameKHomotopyCallReceipt,
    KdotpyPreviousUSameKHomotopyConflictReceipt,
    KdotpyPreviousUSameKHomotopyError,
    KdotpyPreviousUSameKHomotopyFailureReceipt,
    KdotpyPreviousUSameKHomotopySpec,
    KdotpyWindowDiagnostics,
    _KdotpyE1H1EndpointComparison,
    _KdotpyE1H1PairAssignment,
    _KdotpyFixedKTransportResult,
    _KdotpyHomotopyPathGateError,
    _candidate_energy_table_sha256,
    _compare_e1h1_endpoint_projectors,
    _hash_length_delimited_text,
    _local_candidate_positions_namespace_fingerprint,
    _maximum_sparse_abs,
    _pair_resolved_same_call_fingerprint,
 _rayleigh_ritz_near_target,
 _select_e1h1_candidate_pairs_by_overlap,
    _readonly_copy,
    _validate_pair_selection_frame,
)





class KdotpyCanonicalWindowBuilder:
    """Stateful radial/window tracker for repeated kdotpy parent diagonalization."""

    def __init__(
        self,
        *,
        params: Any,
        k_cart_nm_inv: np.ndarray,
        k_weights_nm2: np.ndarray,
        z_nm: np.ndarray,
        z_weights_nm: np.ndarray,
        epsilon_r: np.ndarray,
        selected_band_indices: tuple[int, ...],
        candidate_eigenpair_count: int,
        target_energy_mev: float,
        material_profile_sha256: str,
        kdotpy_source_sha256: str,
        hamiltonian_options: tuple[tuple[str, str], ...],
        energy_zero_label: str,
        pair_selection_spec: KdotpyE1H1PairSelectionSpec,
        minimum_subspace_singular_value: float = 0.7,
        eigensolver_tolerance: float = 1e-11,
        eigensolver_maximum_iterations: int | None = None,
    ) -> None:
        from kdotpy import hamiltonian as hm
        from kdotpy.vector import Vector

        self._hm = hm
        self._params = params
        self._bfield = Vector(0.0, astype="z")
        self._k_cart = np.asarray(k_cart_nm_inv, dtype=np.float64)
        self._z_nm = np.asarray(z_nm, dtype=np.float64)
        self._z_weights = np.asarray(z_weights_nm, dtype=np.float64)
        raw_selected_labels = tuple(selected_band_indices)
        self._selected_labels = tuple(int(label) for label in raw_selected_labels)
        self._candidate_count = int(candidate_eigenpair_count)
        self._target_energy = float(target_energy_mev)
        self._energy_zero_label = str(energy_zero_label)
        if not isinstance(pair_selection_spec, KdotpyE1H1PairSelectionSpec):
            raise TypeError("pair_selection_spec must be KdotpyE1H1PairSelectionSpec")
        self._pair_selection_spec = pair_selection_spec
        self._minimum_subspace_singular_value = float(minimum_subspace_singular_value)
        self._eigensolver_tolerance = float(eigensolver_tolerance)
        self._eigensolver_maximum_iterations = eigensolver_maximum_iterations
        if self._k_cart.ndim != 2 or self._k_cart.shape[1] != 2:
            raise ValueError("k_cart_nm_inv must have shape (nk,2)")
        if len(self._selected_labels) % 2 != 0:
            raise ValueError(
                "the split-zero selected window must contain an even number of states"
            )
        if self._candidate_count <= len(self._selected_labels):
            raise ValueError(
                "candidate eigensystem must be strictly larger than the selected window"
            )
        if len(self._selected_labels) != 4:
            raise ValueError("pair-resolved selection requires an E/H quartet")
        self._pair_selection_spec.validate_candidate_count(self._candidate_count)
        self._pair_selection_spec.validate_ordered_selected_labels(raw_selected_labels)
        if (
            not np.isfinite(self._minimum_subspace_singular_value)
            or not 0.0 < self._minimum_subspace_singular_value <= 1.0
        ):
            raise ValueError("minimum_subspace_singular_value must lie in (0,1]")
        if int(params.nz) != self._z_nm.size or int(params.norbitals) != 8:
            raise ValueError("kdotpy parent must use nz points and eight canonical Kane orbitals")

        self._static_parent = tuple(
            sparse.csr_matrix(
                hm.hz_sparse(
                    [float(kx), float(ky)],
                    self._bfield,
                    params,
                    lattice_reg=False,
                    ignorestrain=True,
                    axial=True,
                    bia=False,
                    ignore_magnxy=False,
                ),
                dtype=np.complex128,
                copy=True,
            )
            for kx, ky in self._k_cart
        )
        self._static_parent_sha256 = canonical_parent_matrix_sequence_sha256(
            list(self._static_parent)
        )
        self._anchor_static_parent = sparse.csr_matrix(
            hm.hz_sparse(
                [0.0, 0.0],
                self._bfield,
                params,
                lattice_reg=False,
                ignorestrain=True,
                axial=True,
                bia=False,
                ignore_magnxy=False,
            ),
            dtype=np.complex128,
            copy=True,
        )
        self._anchor_static_parent_sha256 = canonical_parent_matrix_sequence_sha256(
            [self._anchor_static_parent]
        )
        pair_spec = self._pair_selection_spec
        ordered_anchor_positions = (
            pair_spec.gamma_electron_local_candidate_positions
            + pair_spec.gamma_hole_local_candidate_positions
        )
        selection_anchor_positions = ",".join(
            str(position) for position in ordered_anchor_positions
        )
        pair_hamiltonian_options = (
                ("pair_selection_mode", "explicit-Gamma-E-H-rank2-principal-overlap"),
                ("pair_selection_spec_sha256", pair_spec.fingerprint),
                (
                    "pair_selection_gamma_electron_local_candidate_positions",
                    ",".join(
                        str(position)
                        for position
                        in pair_spec.gamma_electron_local_candidate_positions
                    ),
                ),
                (
                    "pair_selection_gamma_hole_local_candidate_positions",
                    ",".join(
                        str(position)
                        for position in pair_spec.gamma_hole_local_candidate_positions
                    ),
                ),
                (
                    "pair_selection_ordered_physical_selected_labels",
                    ",".join(str(label) for label in pair_spec.ordered_selected_labels),
                ),
                ("pair_selection_electron_label", pair_spec.electron_pair_label),
                ("pair_selection_hole_label", pair_spec.hole_pair_label),
                (
                    "pair_selection_minimum_principal_overlap",
                    repr(pair_spec.minimum_principal_overlap),
                ),
                (
                    "pair_selection_minimum_assignment_margin",
                    repr(pair_spec.minimum_assignment_margin),
                ),
                ("pair_selection_gamma_seed_status", DECLARED_GAMMA_SEED_STATUS),
                (
                    "pair_selection_local_candidate_positions_namespace",
                    LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL,
                ),
        )
        window_selection_label = (
            "declared-Gamma-E-H-seed-not-evidence_then-current-U-adjacent-k-and-"
            "previous-U-same-k-rank2-principal-overlap"
        )
        homotopy_spec = pair_spec.previous_u_same_k_homotopy_spec
        if homotopy_spec is not None:
            pair_hamiltonian_options += (
                    (
                        "pair_selection_previous_u_same_k_homotopy_spec_sha256",
                        homotopy_spec.fingerprint,
                    ),
                    (
                        "pair_selection_previous_u_same_k_homotopy_maximum_substeps",
                        str(homotopy_spec.maximum_substeps),
                    ),
                    (
                        "pair_selection_previous_u_same_k_homotopy_contract",
                        "linear-fixed-k_first-match-confirm-N-plus-1-2N_reverse-2N",
                    ),
            )
            window_selection_label += (
                "_conditional-fail-closed-linear-potential-homotopy-certification"
            )
        bound_hamiltonian_options = tuple(hamiltonian_options) + (
            ("selection_anchor_k_nm_inv", "0,0"),
            ("selection_anchor_static_parent_sha256", self._anchor_static_parent_sha256),
            (
                "selection_anchor_local_candidate_positions",
                selection_anchor_positions,
            ),
            (
                "minimum_subspace_singular_value",
                repr(self._minimum_subspace_singular_value),
            ),
        ) + pair_hamiltonian_options
        self.parent_spec = KaneParentSourceSpec(
            k_cart_nm_inv=self._k_cart,
            k_weights_nm2=np.asarray(k_weights_nm2, dtype=np.float64),
            z_nm=self._z_nm,
            z_weights_nm=self._z_weights,
            epsilon_r=np.asarray(epsilon_r, dtype=np.float64),
            parent_hilbert_dimension=8 * self._z_nm.size,
            selected_band_indices=self._selected_labels,
            candidate_eigenpair_count=self._candidate_count,
            target_energy_mev=self._target_energy,
            window_selection_label=window_selection_label,
            static_parent_sha256=self._static_parent_sha256,
            material_profile_sha256=material_profile_sha256,
            kdotpy_source_sha256=kdotpy_source_sha256,
            hamiltonian_options=bound_hamiltonian_options,
        )
        self._previous_anchor_frame: np.ndarray | None = None
        self._previous_iteration_frames: tuple[np.ndarray, ...] | None = None
        self._previous_successful_potential_mev: np.ndarray | None = None
        self.last_diagnostics: KdotpyWindowDiagnostics | None = None
        self.diagnostics_history: list[KdotpyWindowDiagnostics] = []

    @property
    def static_parent_matrices(self) -> tuple[sparse.csr_matrix, ...]:
        return tuple(matrix.copy() for matrix in self._static_parent)

    def verify_cold_start(self) -> bool:
        """Fail closed unless every builder tracking field is still pristine."""

        populated = tuple(
            name
            for name in (
                "_previous_anchor_frame",
                "_previous_iteration_frames",
                "_previous_successful_potential_mev",
                "last_diagnostics",
            )
            if getattr(self, name) is not None
        )
        if populated or self.diagnostics_history:
            detail = ", ".join(populated) if populated else "diagnostics_history"
            raise ValueError(f"kdotpy builder is not cold: {detail} is populated")
        return True

    def _parent_matrices_at_potential(
        self, potential_mev: np.ndarray
    ) -> tuple[
        np.ndarray,
        tuple[sparse.csr_matrix, ...],
        KdotpyParentPotentialAttestation,
    ]:
        potential = np.asarray(potential_mev, dtype=np.float64)
        if potential.shape != (self._z_nm.size,) or not np.all(np.isfinite(potential)):
            raise ValueError("potential_mev must be finite with shape (nz,)")
        hpot = sparse.csr_matrix(
            self._hm.hz_sparse_pot(self._params, potential),
            dtype=np.complex128,
            copy=True,
        )
        canonical_potential = sparse.diags(
            np.repeat(potential, 8),
            offsets=0,
            shape=hpot.shape,
            dtype=np.complex128,
            format="csr",
        )
        potential_operator_error = _maximum_sparse_abs(hpot - canonical_potential)
        full_parent = tuple(
            sparse.csr_matrix(static + hpot, dtype=np.complex128, copy=True)
            for static in self._static_parent
        )
        replayed_static = tuple(
            sparse.csr_matrix(full - canonical_potential, dtype=np.complex128, copy=True)
            for full in full_parent
        )
        raw_replayed_static_sha256 = canonical_parent_matrix_sequence_sha256(
            list(replayed_static)
        )
        static_replay_error = max(
            _maximum_sparse_abs(replayed - static)
            for replayed, static in zip(replayed_static, self._static_parent)
        )
        replayed_static_sha256 = (
            self._static_parent_sha256
            if static_replay_error <= 1e-10
            else raw_replayed_static_sha256
        )
        actual_full_hash = canonical_parent_matrix_sequence_sha256(list(full_parent))
        attestation = KdotpyParentPotentialAttestation(
            input_potential_sha256=kane_poisson_array_sha256(potential),
            potential_operator_fingerprint=kane_potential_operator_fingerprint(potential),
            actual_full_parent_sequence_sha256=actual_full_hash,
            raw_replayed_static_parent_sha256=raw_replayed_static_sha256,
            replayed_static_parent_sha256=replayed_static_sha256,
            static_parent_replay_max_error_mev=static_replay_error,
            potential_operator_max_error_mev=potential_operator_error,
        )
        return potential, full_parent, attestation

    def _anchor_matrix_at_potential(
        self, potential_mev: np.ndarray
    ) -> tuple[sparse.csr_matrix, str, float, float]:
        potential = np.asarray(potential_mev, dtype=np.float64)
        hpot = sparse.csr_matrix(
            self._hm.hz_sparse_pot(self._params, potential),
            dtype=np.complex128,
            copy=True,
        )
        canonical_potential = sparse.diags(
            np.repeat(potential, 8),
            offsets=0,
            shape=hpot.shape,
            dtype=np.complex128,
            format="csr",
        )
        operator_error = _maximum_sparse_abs(hpot - canonical_potential)
        full = sparse.csr_matrix(
            self._anchor_static_parent + hpot,
            dtype=np.complex128,
            copy=True,
        )
        static_replay_error = _maximum_sparse_abs(
            full - canonical_potential - self._anchor_static_parent
        )
        full_hash = canonical_parent_matrix_sequence_sha256([full])
        return full, full_hash, static_replay_error, operator_error

    def _fixed_k_candidate_eigensystem_at_potential(
        self, *, ik: int, potential_mev: np.ndarray
    ) -> tuple[
        sparse.csr_matrix,
        np.ndarray,
        np.ndarray,
        float,
        str,
        float,
        float,
        str,
    ]:
        """Build, attest, and diagonalize one exact fixed-k parent in scratch state."""

        potential = np.asarray(potential_mev, dtype=np.float64)
        if potential.shape != (self._z_nm.size,) or not np.all(np.isfinite(potential)):
            raise ValueError("homotopy potential must be finite with shape (nz,)")
        if not 0 <= int(ik) < len(self._static_parent):
            raise ValueError("homotopy k index is out of range")
        hpot = sparse.csr_matrix(
            self._hm.hz_sparse_pot(self._params, potential),
            dtype=np.complex128,
            copy=True,
        )
        canonical_potential = sparse.diags(
            np.repeat(potential, 8),
            offsets=0,
            shape=hpot.shape,
            dtype=np.complex128,
            format="csr",
        )
        potential_operator_error = _maximum_sparse_abs(hpot - canonical_potential)
        matrix = sparse.csr_matrix(
            self._static_parent[int(ik)] + hpot,
            dtype=np.complex128,
            copy=True,
        )
        static_replay_error = _maximum_sparse_abs(
            matrix - canonical_potential - self._static_parent[int(ik)]
        )
        energies, candidates, _raw_orthonormality, candidate_error = (
            _rayleigh_ritz_near_target(
                matrix,
                candidate_count=self._candidate_count,
                target_energy_mev=self._target_energy,
                tolerance=self._eigensolver_tolerance,
                maximum_iterations=self._eigensolver_maximum_iterations,
            )
        )
        return (
            matrix,
            energies,
            candidates,
            candidate_error,
            canonical_parent_matrix_sequence_sha256([matrix]),
            static_replay_error,
            potential_operator_error,
            kane_poisson_array_sha256(potential),
        )

    def _transport_previous_u_same_k_path(
        self,
        *,
        ik: int,
        previous_potential_mev: np.ndarray,
        current_potential_mev: np.ndarray,
        start_frame: np.ndarray,
        substeps: int,
        direction: str,
        pair_spec: KdotpyE1H1PairSelectionSpec,
        target_matrix: sparse.csr_matrix | None = None,
        target_energies: np.ndarray | None = None,
        target_candidates: np.ndarray | None = None,
        target_candidate_error: float | None = None,
        target_static_replay_error: float | None = None,
        target_potential_operator_error: float | None = None,
    ) -> _KdotpyFixedKTransportResult:
        """Transport one E/H frame without mutating any committed builder state."""

        if direction not in {"forward", "reverse"}:
            raise ValueError("homotopy direction must be forward or reverse")
        if isinstance(substeps, (bool, np.bool_)) or int(substeps) != substeps:
            raise ValueError("homotopy substeps must be an exact integer")
        count = int(substeps)
        if count < 1:
            raise ValueError("homotopy substeps must be positive")
        previous = np.asarray(previous_potential_mev, dtype=np.float64)
        current = np.asarray(current_potential_mev, dtype=np.float64)
        if previous.shape != current.shape or previous.shape != (self._z_nm.size,):
            raise ValueError("homotopy endpoint potentials must share shape (nz,)")
        reference_frame = np.asarray(
            _validate_pair_selection_frame(
                start_frame, label="homotopy_start_frame", expected_columns=4
            ),
            dtype=np.complex128,
        )
        trace = hashlib.sha256()
        trace.update(b"KdotpyPreviousUSameKFixedKTransportTrace:v1")
        for label, value in (
            ("parent_spec_fingerprint", self.parent_spec.fingerprint),
            ("pair_selection_spec_fingerprint", pair_spec.fingerprint),
            ("direction", direction),
            ("k_index", str(int(ik))),
            ("substeps", str(count)),
            ("previous_potential_sha256", kane_poisson_array_sha256(previous)),
            ("current_potential_sha256", kane_poisson_array_sha256(current)),
            ("start_frame_sha256", kane_poisson_array_sha256(reference_frame)),
        ):
            _hash_length_delimited_text(trace, field_label=label, value=value)

        def fail(reason: str, detail: str) -> None:
            failed_trace = trace.copy()
            _hash_length_delimited_text(
                failed_trace, field_label="failure_reason", value=reason
            )
            _hash_length_delimited_text(
                failed_trace, field_label="failure_detail", value=detail
            )
            raise _KdotpyHomotopyPathGateError(
                reason, detail, failed_trace.hexdigest()
            )

        minima = {
            "electron_overlap": np.inf,
            "hole_overlap": np.inf,
            "electron_margin": np.inf,
            "hole_margin": np.inf,
            "gap": np.inf,
        }
        maxima = {
            "residual": 0.0,
            "static_replay": 0.0,
            "potential_operator": 0.0,
        }
        endpoint_assignment: _KdotpyE1H1PairAssignment | None = None
        endpoint_frame: np.ndarray | None = None
        for step in range(1, count + 1):
            lambda_numerator = step if direction == "forward" else count - step
            if lambda_numerator == 0:
                potential = previous.copy()
            elif lambda_numerator == count:
                potential = current.copy()
            else:
                potential = (
                    (count - lambda_numerator) * previous
                    + lambda_numerator * current
                ) / float(count)
            use_cached_target = direction == "forward" and step == count
            if use_cached_target:
                if any(
                    value is None
                    for value in (
                        target_matrix,
                        target_energies,
                        target_candidates,
                        target_candidate_error,
                        target_static_replay_error,
                        target_potential_operator_error,
                    )
                ):
                    fail("target_cache_incomplete", "forward target cache is incomplete")
                matrix = sparse.csr_matrix(target_matrix, copy=False)
                energies = np.asarray(target_energies, dtype=np.float64)
                candidates = np.asarray(target_candidates, dtype=np.complex128)
                candidate_error = float(target_candidate_error)
                parent_sha256 = canonical_parent_matrix_sequence_sha256([matrix])
                static_replay_error = float(target_static_replay_error)
                potential_operator_error = float(target_potential_operator_error)
                potential_sha256 = kane_poisson_array_sha256(current)
            else:
                try:
                    (
                        matrix,
                        energies,
                        candidates,
                        candidate_error,
                        parent_sha256,
                        static_replay_error,
                        potential_operator_error,
                        potential_sha256,
                    ) = self._fixed_k_candidate_eigensystem_at_potential(
                        ik=ik, potential_mev=potential
                    )
                except Exception as exc:
                    fail(
                        "candidate_eigensystem_failed",
                        f"{type(exc).__name__}: {exc}",
                    )
            for label, value in (
                ("step", str(step)),
                ("lambda_numerator", str(lambda_numerator)),
                ("lambda_denominator", str(count)),
                ("potential_sha256", potential_sha256),
                ("fixed_k_parent_sha256", parent_sha256),
                (
                    "candidate_energy_table_sha256",
                    kane_poisson_array_sha256(energies),
                ),
                ("candidate_error", repr(float(candidate_error))),
                ("static_replay_error", repr(float(static_replay_error))),
                ("potential_operator_error", repr(float(potential_operator_error))),
            ):
                _hash_length_delimited_text(trace, field_label=label, value=value)
            if not np.isfinite(static_replay_error) or static_replay_error > 1.0e-10:
                fail(
                    "static_parent_replay_gate",
                    f"error={static_replay_error:.3e} exceeds 1e-10 meV",
                )
            if (
                not np.isfinite(potential_operator_error)
                or potential_operator_error > 1.0e-12
            ):
                fail(
                    "potential_operator_gate",
                    f"error={potential_operator_error:.3e} exceeds 1e-12 meV",
                )
            try:
                assignment = _select_e1h1_candidate_pairs_by_overlap(
                    reference_frame[:, :2],
                    reference_frame[:, 2:4],
                    candidates,
                    minimum_principal_overlap=pair_spec.minimum_principal_overlap,
                    minimum_assignment_margin=pair_spec.minimum_assignment_margin,
                    electron_pair_label=pair_spec.electron_pair_label,
                    hole_pair_label=pair_spec.hole_pair_label,
                    context=(
                        f"previous-U same-k homotopy {direction} at ik={ik}, "
                        f"lambda={lambda_numerator}/{count}"
                    ),
                )
            except Exception as exc:
                fail("pair_selection_gate", f"{type(exc).__name__}: {exc}")
            electron_positions = np.asarray(
                assignment.electron_local_candidate_positions, dtype=np.int64
            )
            hole_positions = np.asarray(
                assignment.hole_local_candidate_positions, dtype=np.int64
            )
            chosen = np.concatenate((electron_positions, hole_positions))
            frame = np.asarray(candidates[:, chosen], dtype=np.complex128)
            values = np.asarray(energies[chosen], dtype=np.float64)
            residual = matrix @ frame - frame * values[None, :]
            residual_norm = float(np.max(np.linalg.norm(residual, axis=0)))
            remaining = np.delete(energies, chosen)
            complement_gap = (
                np.inf
                if remaining.size == 0
                else float(np.min(np.abs(values[:, None] - remaining[None, :])))
            )
            quartet_principal_values = np.linalg.svd(
                reference_frame.conj().T @ frame, compute_uv=False
            )
            for label, values_to_hash in (
                ("electron_positions", electron_positions),
                ("hole_positions", hole_positions),
                (
                    "electron_principal_singular_values",
                    assignment.electron_principal_singular_values,
                ),
                (
                    "hole_principal_singular_values",
                    assignment.hole_principal_singular_values,
                ),
                ("quartet_principal_values", quartet_principal_values),
            ):
                _hash_length_delimited_text(
                    trace,
                    field_label=label,
                    value=kane_poisson_array_sha256(values_to_hash),
                )
            for label, value in (
                ("electron_margin", assignment.electron_assignment_margin),
                ("hole_margin", assignment.hole_assignment_margin),
                ("selected_residual", residual_norm),
                ("selected_complement_gap", complement_gap),
            ):
                _hash_length_delimited_text(
                    trace, field_label=label, value=repr(float(value))
                )
            eigensystem_error = max(float(candidate_error), residual_norm)
            if not np.isfinite(eigensystem_error) or eigensystem_error > 1.0e-7:
                fail(
                    "eigensystem_residual_gate",
                    f"error={eigensystem_error:.3e} exceeds 1e-7",
                )
            if not np.isfinite(complement_gap) or complement_gap <= 1.0e-6:
                fail(
                    "selected_complement_gap_gate",
                    f"gap={complement_gap:.3e} is not above 1e-6 meV",
                )
            if (
                not np.all(np.isfinite(quartet_principal_values))
                or float(np.min(quartet_principal_values))
                < self._minimum_subspace_singular_value
            ):
                fail(
                    "quartet_continuity_gate",
                    "quartet principal overlap is below the unchanged builder gate",
                )
            minima["electron_overlap"] = min(
                minima["electron_overlap"],
                float(np.min(assignment.electron_principal_singular_values)),
            )
            minima["hole_overlap"] = min(
                minima["hole_overlap"],
                float(np.min(assignment.hole_principal_singular_values)),
            )
            minima["electron_margin"] = min(
                minima["electron_margin"], assignment.electron_assignment_margin
            )
            minima["hole_margin"] = min(
                minima["hole_margin"], assignment.hole_assignment_margin
            )
            minima["gap"] = min(minima["gap"], complement_gap)
            maxima["residual"] = max(maxima["residual"], eigensystem_error)
            maxima["static_replay"] = max(
                maxima["static_replay"], static_replay_error
            )
            maxima["potential_operator"] = max(
                maxima["potential_operator"], potential_operator_error
            )
            reference_frame = frame
            endpoint_assignment = assignment
            endpoint_frame = frame
        if endpoint_assignment is None or endpoint_frame is None:
            fail("empty_transport", "homotopy path produced no endpoint")
        return _KdotpyFixedKTransportResult(
            assignment=endpoint_assignment,
            endpoint_frame=endpoint_frame,
            minimum_electron_principal_overlap=minima["electron_overlap"],
            minimum_hole_principal_overlap=minima["hole_overlap"],
            minimum_electron_assignment_margin=minima["electron_margin"],
            minimum_hole_assignment_margin=minima["hole_margin"],
            minimum_selected_complement_gap_mev=minima["gap"],
            maximum_eigensystem_residual_mev=maxima["residual"],
            maximum_static_parent_replay_error_mev=maxima["static_replay"],
            maximum_potential_operator_error_mev=maxima["potential_operator"],
            trace_sha256=trace.hexdigest(),
        )

    def _certify_previous_u_same_k_homotopy(
        self,
        *,
        ik: int,
        previous_potential_mev: np.ndarray,
        current_potential_mev: np.ndarray,
        previous_frame: np.ndarray,
        previous_electron_local_candidate_positions: tuple[int, int],
        previous_hole_local_candidate_positions: tuple[int, int],
        radial_assignment: _KdotpyE1H1PairAssignment,
        radial_frame: np.ndarray,
        direct_assignment: _KdotpyE1H1PairAssignment,
        target_matrix: sparse.csr_matrix,
        target_energies: np.ndarray,
        target_candidates: np.ndarray,
        target_candidate_error: float,
        target_static_replay_error: float,
        target_potential_operator_error: float,
        pair_spec: KdotpyE1H1PairSelectionSpec,
        homotopy_spec: KdotpyPreviousUSameKHomotopySpec,
    ) -> KdotpyPreviousUSameKHomotopyConflictReceipt:
        """Certify one direct disagreement or raise a typed transactional error."""

        previous = np.asarray(previous_potential_mev, dtype=np.float64)
        current = np.asarray(current_potential_mev, dtype=np.float64)
        potential_step = current - previous
        attempted: list[int] = []
        trace = hashlib.sha256()
        trace.update(b"KdotpyPreviousUSameKHomotopyConflictTrace:v1")
        for label, value in (
            ("parent_spec_fingerprint", self.parent_spec.fingerprint),
            ("pair_selection_spec_fingerprint", pair_spec.fingerprint),
            ("homotopy_spec_fingerprint", homotopy_spec.fingerprint),
            ("k_index", str(int(ik))),
            ("previous_potential_sha256", kane_poisson_array_sha256(previous)),
            ("current_potential_sha256", kane_poisson_array_sha256(current)),
            ("previous_frame_sha256", kane_poisson_array_sha256(previous_frame)),
            ("radial_frame_sha256", kane_poisson_array_sha256(radial_frame)),
            (
                "direct_electron_positions",
                repr(direct_assignment.electron_local_candidate_positions),
            ),
            (
                "direct_hole_positions",
                repr(direct_assignment.hole_local_candidate_positions),
            ),
            (
                "radial_electron_positions",
                repr(radial_assignment.electron_local_candidate_positions),
            ),
            (
                "radial_hole_positions",
                repr(radial_assignment.hole_local_candidate_positions),
            ),
        ):
            _hash_length_delimited_text(trace, field_label=label, value=value)

        def raise_failure(
            reason_code: str,
            detail: str,
            *,
            first_matching_substeps: int | None = None,
            failed_trace_sha256: str | None = None,
        ) -> None:
            failure_trace = trace.copy()
            if failed_trace_sha256 is not None:
                _hash_length_delimited_text(
                    failure_trace,
                    field_label="failed_path_trace_sha256",
                    value=failed_trace_sha256,
                )
            _hash_length_delimited_text(
                failure_trace, field_label="reason_code", value=reason_code
            )
            _hash_length_delimited_text(
                failure_trace, field_label="detail", value=detail
            )
            raise KdotpyPreviousUSameKHomotopyError(
                KdotpyPreviousUSameKHomotopyFailureReceipt(
                    reason_code=reason_code,
                    detail=detail,
                    k_index=ik,
                    pair_selection_spec_fingerprint=pair_spec.fingerprint,
                    homotopy_spec_fingerprint=homotopy_spec.fingerprint,
                    parent_spec_fingerprint=self.parent_spec.fingerprint,
                    previous_potential_sha256=kane_poisson_array_sha256(previous),
                    current_potential_sha256=kane_poisson_array_sha256(current),
                    potential_step_max_abs_mev=float(np.max(np.abs(potential_step))),
                    potential_step_rms_mev=float(
                        np.sqrt(np.mean(potential_step**2))
                    ),
                    attempted_dyadic_substeps=tuple(attempted),
                    first_matching_substeps=first_matching_substeps,
                    direct_electron_local_candidate_positions=(
                        direct_assignment.electron_local_candidate_positions
                    ),
                    direct_hole_local_candidate_positions=(
                        direct_assignment.hole_local_candidate_positions
                    ),
                    direct_electron_principal_singular_values=(
                        direct_assignment.electron_principal_singular_values
                    ),
                    direct_hole_principal_singular_values=(
                        direct_assignment.hole_principal_singular_values
                    ),
                    direct_electron_assignment_margin=(
                        direct_assignment.electron_assignment_margin
                    ),
                    direct_hole_assignment_margin=(
                        direct_assignment.hole_assignment_margin
                    ),
                    radial_electron_local_candidate_positions=(
                        radial_assignment.electron_local_candidate_positions
                    ),
                    radial_hole_local_candidate_positions=(
                        radial_assignment.hole_local_candidate_positions
                    ),
                    transport_trace_sha256=failure_trace.hexdigest(),
                )
            )

        direct_positions = np.asarray(
            direct_assignment.electron_local_candidate_positions
            + direct_assignment.hole_local_candidate_positions,
            dtype=np.int64,
        )
        direct_frame = np.asarray(
            target_candidates[:, direct_positions], dtype=np.complex128
        )
        direct_values = np.asarray(
            target_energies[direct_positions], dtype=np.float64
        )
        direct_residual = target_matrix @ direct_frame - direct_frame * direct_values[
            None, :
        ]
        direct_residual_norm = float(
            np.max(np.linalg.norm(direct_residual, axis=0))
        )
        direct_remaining = np.delete(target_energies, direct_positions)
        direct_complement_gap = float(
            np.min(
                np.abs(
                    direct_values[:, None] - direct_remaining[None, :]
                )
            )
        )
        direct_previous_quartet_values = np.linalg.svd(
            previous_frame.conj().T @ direct_frame, compute_uv=False
        )
        direct_radial_quartet_values = np.linalg.svd(
            radial_frame.conj().T @ direct_frame, compute_uv=False
        )
        direct_gram_error = max(
            float(
                np.max(
                    np.abs(direct_frame.conj().T @ direct_frame - np.eye(4))
                )
            ),
            float(
                np.max(
                    np.abs(radial_frame.conj().T @ radial_frame - np.eye(4))
                )
            ),
        )
        direct_radial_roundoff_bound = max(
            2048.0 * np.finfo(np.float64).eps * max(direct_frame.shape),
            16.0 * direct_gram_error,
        )
        for label, value in (
            ("direct_selected_residual", direct_residual_norm),
            ("direct_selected_complement_gap", direct_complement_gap),
            ("direct_target_candidate_error", float(target_candidate_error)),
            ("direct_target_static_replay_error", target_static_replay_error),
            (
                "direct_target_potential_operator_error",
                target_potential_operator_error,
            ),
        ):
            _hash_length_delimited_text(
                trace, field_label=label, value=repr(float(value))
            )
        for label, values in (
            ("direct_previous_quartet_values", direct_previous_quartet_values),
            ("direct_radial_quartet_values", direct_radial_quartet_values),
        ):
            _hash_length_delimited_text(
                trace,
                field_label=label,
                value=kane_poisson_array_sha256(values),
            )
        direct_eigensystem_error = max(
            float(target_candidate_error), direct_residual_norm
        )
        if (
            not np.isfinite(target_static_replay_error)
            or target_static_replay_error > 1.0e-10
            or not np.isfinite(target_potential_operator_error)
            or target_potential_operator_error > 1.0e-12
            or not np.isfinite(direct_eigensystem_error)
            or direct_eigensystem_error > 1.0e-7
            or not np.isfinite(direct_complement_gap)
            or direct_complement_gap <= 1.0e-6
            or float(np.min(direct_previous_quartet_values))
            < self._minimum_subspace_singular_value
        ):
            raise_failure(
                "transport_gate_failed",
                "the direct previous-U selector did not pass all unchanged parent, "
                "eigensystem, complement, and quartet gates",
            )
        if np.max(np.abs(direct_radial_quartet_values - 1.0)) > (
            direct_radial_roundoff_bound
        ):
            raise_failure(
                "transport_gate_failed",
                "direct and radial assignments do not span the same quartet; "
                "the approved homotopy check is partition-only",
            )

        def run_forward(substeps: int) -> _KdotpyFixedKTransportResult:
            try:
                result = self._transport_previous_u_same_k_path(
                    ik=ik,
                    previous_potential_mev=previous,
                    current_potential_mev=current,
                    start_frame=previous_frame,
                    substeps=substeps,
                    direction="forward",
                    pair_spec=pair_spec,
                    target_matrix=target_matrix,
                    target_energies=target_energies,
                    target_candidates=target_candidates,
                    target_candidate_error=target_candidate_error,
                    target_static_replay_error=target_static_replay_error,
                    target_potential_operator_error=target_potential_operator_error,
                )
            except _KdotpyHomotopyPathGateError as exc:
                raise_failure(
                    "transport_gate_failed",
                    f"forward N={substeps}: {exc}",
                    failed_trace_sha256=exc.trace_sha256,
                )
            _hash_length_delimited_text(
                trace,
                field_label=f"forward_N_{substeps}_trace_sha256",
                value=result.trace_sha256,
            )
            return result

        def compare_forward(
            result: _KdotpyFixedKTransportResult, *, label: str
        ) -> _KdotpyE1H1EndpointComparison:
            comparison = _compare_e1h1_endpoint_projectors(
                radial_frame,
                result.endpoint_frame,
                reference_electron_local_candidate_positions=(
                    radial_assignment.electron_local_candidate_positions
                ),
                reference_hole_local_candidate_positions=(
                    radial_assignment.hole_local_candidate_positions
                ),
                transported_electron_local_candidate_positions=(
                    result.assignment.electron_local_candidate_positions
                ),
                transported_hole_local_candidate_positions=(
                    result.assignment.hole_local_candidate_positions
                ),
            )
            for suffix, values in (
                ("electron", comparison.electron_principal_values),
                ("hole", comparison.hole_principal_values),
                ("quartet", comparison.quartet_principal_values),
            ):
                _hash_length_delimited_text(
                    trace,
                    field_label=f"{label}_{suffix}_endpoint_principal_values",
                    value=kane_poisson_array_sha256(values),
                )
            _hash_length_delimited_text(
                trace,
                field_label=f"{label}_endpoint_matches",
                value=str(comparison.matches),
            )
            return comparison

        first_result: _KdotpyFixedKTransportResult | None = None
        first_comparison: _KdotpyE1H1EndpointComparison | None = None
        candidate_substeps = 2
        while 2 * candidate_substeps <= homotopy_spec.maximum_substeps:
            attempted.append(candidate_substeps)
            result = run_forward(candidate_substeps)
            comparison = compare_forward(
                result, label=f"forward_N_{candidate_substeps}"
            )
            if comparison.matches:
                first_result = result
                first_comparison = comparison
                break
            candidate_substeps *= 2
        if first_result is None or first_comparison is None:
            raise_failure(
                "maximum_substeps_exhausted",
                "no separately matching E/H endpoint had room for 2N confirmation",
            )
        first_matching = candidate_substeps
        noncommensurate_result = run_forward(first_matching + 1)
        noncommensurate_comparison = compare_forward(
            noncommensurate_result, label=f"forward_N_{first_matching + 1}"
        )
        dyadic_result = run_forward(2 * first_matching)
        dyadic_comparison = compare_forward(
            dyadic_result, label=f"forward_N_{2 * first_matching}"
        )
        if not noncommensurate_comparison.matches or not dyadic_comparison.matches:
            raise_failure(
                "confirmation_failed",
                "the first matching endpoint was not confirmed by both N+1 and 2N",
                first_matching_substeps=first_matching,
            )
        try:
            reverse_result = self._transport_previous_u_same_k_path(
                ik=ik,
                previous_potential_mev=previous,
                current_potential_mev=current,
                start_frame=dyadic_result.endpoint_frame,
                substeps=2 * first_matching,
                direction="reverse",
                pair_spec=pair_spec,
            )
        except _KdotpyHomotopyPathGateError as exc:
            raise_failure(
                "transport_gate_failed",
                f"reverse 2N={2 * first_matching}: {exc}",
                first_matching_substeps=first_matching,
                failed_trace_sha256=exc.trace_sha256,
            )
        _hash_length_delimited_text(
            trace,
            field_label="reverse_2N_trace_sha256",
            value=reverse_result.trace_sha256,
        )
        reverse_comparison = _compare_e1h1_endpoint_projectors(
            previous_frame,
            reverse_result.endpoint_frame,
            reference_electron_local_candidate_positions=(
                previous_electron_local_candidate_positions
            ),
            reference_hole_local_candidate_positions=(
                previous_hole_local_candidate_positions
            ),
            transported_electron_local_candidate_positions=(
                reverse_result.assignment.electron_local_candidate_positions
            ),
            transported_hole_local_candidate_positions=(
                reverse_result.assignment.hole_local_candidate_positions
            ),
        )
        for suffix, values in (
            ("electron", reverse_comparison.electron_principal_values),
            ("hole", reverse_comparison.hole_principal_values),
            ("quartet", reverse_comparison.quartet_principal_values),
        ):
            _hash_length_delimited_text(
                trace,
                field_label=f"reverse_{suffix}_endpoint_principal_values",
                value=kane_poisson_array_sha256(values),
            )
        reverse_projectors_match = bool(
            np.max(
                np.abs(reverse_comparison.electron_principal_values - 1.0)
            )
            <= reverse_comparison.projector_roundoff_bound
            and np.max(
                np.abs(reverse_comparison.hole_principal_values - 1.0)
            )
            <= reverse_comparison.projector_roundoff_bound
        )
        if not reverse_projectors_match:
            raise_failure(
                "reverse_recovery_failed",
                "reverse 2N transport did not recover previous E/H projectors",
                first_matching_substeps=first_matching,
            )
        accepted_results = (
            first_result,
            noncommensurate_result,
            dyadic_result,
            reverse_result,
        )
        forward_results = (
            first_result,
            noncommensurate_result,
            dyadic_result,
        )
        forward_comparisons = (
            first_comparison,
            noncommensurate_comparison,
            dyadic_comparison,
        )
        return KdotpyPreviousUSameKHomotopyConflictReceipt(
            k_index=ik,
            direct_electron_local_candidate_positions=(
                direct_assignment.electron_local_candidate_positions
            ),
            direct_hole_local_candidate_positions=(
                direct_assignment.hole_local_candidate_positions
            ),
            direct_electron_principal_singular_values=(
                direct_assignment.electron_principal_singular_values
            ),
            direct_hole_principal_singular_values=(
                direct_assignment.hole_principal_singular_values
            ),
            direct_electron_assignment_margin=(
                direct_assignment.electron_assignment_margin
            ),
            direct_hole_assignment_margin=direct_assignment.hole_assignment_margin,
            radial_electron_local_candidate_positions=(
                radial_assignment.electron_local_candidate_positions
            ),
            radial_hole_local_candidate_positions=(
                radial_assignment.hole_local_candidate_positions
            ),
            attempted_dyadic_substeps=tuple(attempted),
            first_matching_substeps=first_matching,
            confirmation_substeps=(first_matching + 1, 2 * first_matching),
            forward_endpoint_electron_local_candidate_positions=tuple(
                result.assignment.electron_local_candidate_positions
                for result in forward_results
            ),
            forward_endpoint_hole_local_candidate_positions=tuple(
                result.assignment.hole_local_candidate_positions
                for result in forward_results
            ),
            forward_endpoint_electron_principal_values=np.asarray(
                [value.electron_principal_values for value in forward_comparisons]
            ),
            forward_endpoint_hole_principal_values=np.asarray(
                [value.hole_principal_values for value in forward_comparisons]
            ),
            forward_endpoint_quartet_principal_values=np.asarray(
                [value.quartet_principal_values for value in forward_comparisons]
            ),
            previous_electron_local_candidate_positions=(
                previous_electron_local_candidate_positions
            ),
            previous_hole_local_candidate_positions=(
                previous_hole_local_candidate_positions
            ),
            reverse_endpoint_electron_local_candidate_positions=(
                reverse_result.assignment.electron_local_candidate_positions
            ),
            reverse_endpoint_hole_local_candidate_positions=(
                reverse_result.assignment.hole_local_candidate_positions
            ),
            reverse_endpoint_electron_principal_values=(
                reverse_comparison.electron_principal_values
            ),
            reverse_endpoint_hole_principal_values=(
                reverse_comparison.hole_principal_values
            ),
            reverse_endpoint_quartet_principal_values=(
                reverse_comparison.quartet_principal_values
            ),
            endpoint_projector_roundoff_bound=max(
                value.projector_roundoff_bound
                for value in (*forward_comparisons, reverse_comparison)
            ),
            minimum_electron_principal_overlap=min(
                value.minimum_electron_principal_overlap
                for value in accepted_results
            ),
            minimum_hole_principal_overlap=min(
                value.minimum_hole_principal_overlap for value in accepted_results
            ),
            minimum_electron_assignment_margin=min(
                value.minimum_electron_assignment_margin
                for value in accepted_results
            ),
            minimum_hole_assignment_margin=min(
                value.minimum_hole_assignment_margin for value in accepted_results
            ),
            minimum_selected_complement_gap_mev=min(
                value.minimum_selected_complement_gap_mev
                for value in accepted_results
            ),
            maximum_eigensystem_residual_mev=max(
                value.maximum_eigensystem_residual_mev
                for value in accepted_results
            ),
            maximum_static_parent_replay_error_mev=max(
                value.maximum_static_parent_replay_error_mev
                for value in accepted_results
            ),
            maximum_potential_operator_error_mev=max(
                value.maximum_potential_operator_error_mev
                for value in accepted_results
            ),
            transport_trace_sha256=trace.hexdigest(),
        )

    def attest_potential_operator(
        self, potential_mev: np.ndarray
    ) -> KdotpyParentPotentialAttestation:
        """Attest the actual kdotpy matrices without diagonalizing them."""

        _potential, _full_parent, attestation = self._parent_matrices_at_potential(
            potential_mev
        )
        return attestation

    def diagonalize_full_parent_at_potential(
        self, potential_mev: np.ndarray
    ) -> KdotpyFullParentEigensystem:
        """Dense-diagonalize the same complete matrices used by the attestation."""

        _potential, full_parent, attestation = self._parent_matrices_at_potential(
            potential_mev
        )
        if attestation.static_parent_replay_max_error_mev > 1e-10:
            raise ValueError("full-parent static replay residual exceeds tolerance")
        if attestation.potential_operator_max_error_mev > 1e-12:
            raise ValueError("full-parent potential operator residual exceeds tolerance")
        dimension = self.parent_spec.parent_hilbert_dimension
        identity = np.eye(dimension, dtype=np.complex128)
        energies = np.empty((len(full_parent), dimension), dtype=np.float64)
        vectors = np.empty(
            (len(full_parent), dimension, dimension), dtype=np.complex128
        )
        hermiticity_error = 0.0
        residual_error = 0.0
        orthonormality_error = 0.0
        completeness_error = 0.0
        for ik, matrix in enumerate(full_parent):
            dense = np.asarray(matrix.toarray(), dtype=np.complex128)
            hermiticity_error = max(
                hermiticity_error,
                float(np.max(np.abs(dense - dense.conj().T))),
            )
            values, frame = eigh(
                dense,
                lower=True,
                overwrite_a=True,
                check_finite=False,
                driver="evd",
            )
            energies[ik] = values
            vectors[ik] = frame
            residual = matrix @ frame - frame * values[None, :]
            residual_error = max(
                residual_error,
                float(np.max(np.linalg.norm(residual, axis=0))),
            )
            orthonormality_error = max(
                orthonormality_error,
                float(np.max(np.abs(frame.conj().T @ frame - identity))),
            )
            completeness_error = max(
                completeness_error,
                float(np.max(np.abs(frame @ frame.conj().T - identity))),
            )
        if hermiticity_error > 1e-10:
            raise ValueError("complete kdotpy parent is not Hermitian")
        if residual_error > 1e-7:
            raise ValueError("complete kdotpy eigensystem residual exceeds tolerance")
        if max(orthonormality_error, completeness_error) > 1e-8:
            raise ValueError("complete kdotpy eigensystem fails unitary closure")
        return KdotpyFullParentEigensystem(
            energies_mev=energies,
            eigenvectors=vectors,
            attestation=attestation,
            hermiticity_error_mev=hermiticity_error,
            eigen_residual_mev=residual_error,
            orthonormality_error=orthonormality_error,
            completeness_error=completeness_error,
        )

    def __call__(self, potential_mev: np.ndarray) -> KdotpyPairResolvedWindow:
        potential, full_parent, attestation = self._parent_matrices_at_potential(
            potential_mev
        )
        potential_operator_error = attestation.potential_operator_max_error_mev
        replayed_static_sha256 = attestation.replayed_static_parent_sha256

        nwindow = len(self._selected_labels)
        (
            anchor_matrix,
            anchor_full_hash,
            anchor_static_replay_error,
            anchor_operator_error,
        ) = self._anchor_matrix_at_potential(potential)
        if anchor_static_replay_error > 1e-10 or anchor_operator_error > 1e-12:
            raise ValueError("Gamma selection anchor failed full-parent potential attestation")
        (
            anchor_energies,
            anchor_candidates,
            _anchor_raw_orthonormality,
            anchor_candidate_error,
        ) = _rayleigh_ritz_near_target(
            anchor_matrix,
            candidate_count=self._candidate_count,
            target_energy_mev=self._target_energy,
            tolerance=self._eigensolver_tolerance,
            maximum_iterations=self._eigensolver_maximum_iterations,
        )
        pair_spec = self._pair_selection_spec
        homotopy_spec = pair_spec.previous_u_same_k_homotopy_spec
        if homotopy_spec is not None:
            previous_frames_present = self._previous_iteration_frames is not None
            previous_potential_present = (
                self._previous_successful_potential_mev is not None
            )
            if previous_frames_present != previous_potential_present:
                raise RuntimeError(
                    "homotopy tracking state is inconsistent: previous frames and "
                    "previous potential must be committed together"
                )
            if previous_frames_present and self.last_diagnostics is None:
                raise RuntimeError(
                    "homotopy tracking state has previous frames without diagnostics"
                )
            if self.last_diagnostics is not None:
                self.last_diagnostics.verify_integrity()
        anchor_electron_positions: np.ndarray | None = None
        anchor_hole_positions: np.ndarray | None = None
        anchor_electron_singular_values: np.ndarray | None = None
        anchor_hole_singular_values: np.ndarray | None = None
        anchor_electron_margin: float | None = None
        anchor_hole_margin: float | None = None
        if self._previous_anchor_frame is None:
            anchor_electron_positions = np.asarray(
                pair_spec.gamma_electron_local_candidate_positions,
                dtype=np.int64,
            )
            anchor_hole_positions = np.asarray(
                pair_spec.gamma_hole_local_candidate_positions,
                dtype=np.int64,
            )
            pinned_assignment = _select_e1h1_candidate_pairs_by_overlap(
                anchor_candidates[:, anchor_electron_positions],
                anchor_candidates[:, anchor_hole_positions],
                anchor_candidates,
                minimum_principal_overlap=pair_spec.minimum_principal_overlap,
                minimum_assignment_margin=pair_spec.minimum_assignment_margin,
                electron_pair_label=pair_spec.electron_pair_label,
                hole_pair_label=pair_spec.hole_pair_label,
                context=(
                    "declared initial Gamma E/H seed (not independent identity "
                    "evidence)"
                ),
            )
            if (
                set(pinned_assignment.electron_local_candidate_positions)
                != set(anchor_electron_positions.tolist())
                or set(pinned_assignment.hole_local_candidate_positions)
                != set(anchor_hole_positions.tolist())
            ):
                raise ValueError(
                    "declared Gamma E/H local positions are not self-identifying"
                )
            anchor_electron_singular_values = (
                pinned_assignment.electron_principal_singular_values
            )
            anchor_hole_singular_values = (
                pinned_assignment.hole_principal_singular_values
            )
            anchor_electron_margin = pinned_assignment.electron_assignment_margin
            anchor_hole_margin = pinned_assignment.hole_assignment_margin
            anchor_projection_weights = np.zeros_like(anchor_energies)
            anchor_projection_weights[anchor_electron_positions] = (
                pinned_assignment.electron_selected_projection_weights
            )
            anchor_projection_weights[anchor_hole_positions] = (
                pinned_assignment.hole_selected_projection_weights
            )
            anchor_chosen = np.concatenate(
                (anchor_electron_positions, anchor_hole_positions)
            )
            anchor_singular_values = np.ones(nwindow)
        else:
            anchor_assignment = _select_e1h1_candidate_pairs_by_overlap(
                self._previous_anchor_frame[:, :2],
                self._previous_anchor_frame[:, 2:4],
                anchor_candidates,
                minimum_principal_overlap=pair_spec.minimum_principal_overlap,
                minimum_assignment_margin=pair_spec.minimum_assignment_margin,
                electron_pair_label=pair_spec.electron_pair_label,
                hole_pair_label=pair_spec.hole_pair_label,
                context="previous-U same-Gamma E/H selection",
            )
            anchor_electron_positions = np.asarray(
                anchor_assignment.electron_local_candidate_positions,
                dtype=np.int64,
            )
            anchor_hole_positions = np.asarray(
                anchor_assignment.hole_local_candidate_positions,
                dtype=np.int64,
            )
            anchor_chosen = np.concatenate(
                (anchor_electron_positions, anchor_hole_positions)
            )
            anchor_projection_weights = np.zeros_like(anchor_energies)
            anchor_projection_weights[anchor_electron_positions] = (
                anchor_assignment.electron_selected_projection_weights
            )
            anchor_projection_weights[anchor_hole_positions] = (
                anchor_assignment.hole_selected_projection_weights
            )
            anchor_singular_values = np.linalg.svd(
                self._previous_anchor_frame.conj().T
                @ anchor_candidates[:, anchor_chosen],
                compute_uv=False,
            )
            anchor_electron_singular_values = (
                anchor_assignment.electron_principal_singular_values
            )
            anchor_hole_singular_values = (
                anchor_assignment.hole_principal_singular_values
            )
            anchor_electron_margin = anchor_assignment.electron_assignment_margin
            anchor_hole_margin = anchor_assignment.hole_assignment_margin
        anchor_frame = np.asarray(
            anchor_candidates[:, anchor_chosen], dtype=np.complex128
        )
        anchor_values = np.asarray(anchor_energies[anchor_chosen], dtype=np.float64)
        anchor_residual = anchor_matrix @ anchor_frame - anchor_frame * anchor_values[None, :]
        anchor_residual_norm = float(np.max(np.linalg.norm(anchor_residual, axis=0)))
        anchor_remaining = np.delete(anchor_energies, anchor_chosen)
        anchor_edge_gap = float(
            np.min(np.abs(anchor_values[:, None] - anchor_remaining[None, :]))
        )
        if max(anchor_candidate_error, anchor_residual_norm) > 1e-7:
            raise ValueError("Gamma selection-anchor eigensystem failed residual tolerance")
        if anchor_edge_gap <= 1e-6:
            raise ValueError("Gamma selection-anchor candidate complement is not isolated")
        if float(np.min(anchor_singular_values)) < self._minimum_subspace_singular_value:
            raise ValueError("Gamma selection-anchor subspace continuity was lost")

        candidate_energies: list[np.ndarray] = []
        selected_local_positions: list[np.ndarray] = []
        selection_weights: list[np.ndarray] = []
        continuation_singular_values: list[np.ndarray] = []
        edge_gaps: list[float] = []
        raw_orthonormality: list[float] = []
        selected_residuals: list[float] = []
        selected_energies: list[np.ndarray] = []
        selected_frames: list[np.ndarray] = []
        electron_selected_local_positions: list[np.ndarray] = []
        hole_selected_local_positions: list[np.ndarray] = []
        current_adjacent_electron_singular_values: list[np.ndarray] = []
        current_adjacent_hole_singular_values: list[np.ndarray] = []
        current_adjacent_electron_margins: list[float] = []
        current_adjacent_hole_margins: list[float] = []
        previous_u_electron_local_positions: list[np.ndarray] = []
        previous_u_hole_local_positions: list[np.ndarray] = []
        previous_u_electron_singular_values: list[np.ndarray] = []
        previous_u_hole_singular_values: list[np.ndarray] = []
        previous_u_electron_margins: list[float] = []
        previous_u_hole_margins: list[float] = []
        homotopy_k_modes: list[str] = []
        homotopy_conflicts: list[
            KdotpyPreviousUSameKHomotopyConflictReceipt
        ] = []
        previous_radial_frame: np.ndarray | None = anchor_frame

        for ik, matrix in enumerate(full_parent):
            energies, candidates, raw_orth, candidate_error = _rayleigh_ritz_near_target(
                matrix,
                candidate_count=self._candidate_count,
                target_energy_mev=self._target_energy,
                tolerance=self._eigensolver_tolerance,
                maximum_iterations=self._eigensolver_maximum_iterations,
            )
            pair_assignment: _KdotpyE1H1PairAssignment | None = None
            previous_u_assignment: _KdotpyE1H1PairAssignment | None = None
            previous_u_reference_frame: np.ndarray | None = None
            pair_assignments_disagree = False
            if previous_radial_frame is None:
                raise ValueError(
                    "pair-resolved current-U adjacent-k selection has no reference"
                )
            # Always preserve current-U radial continuity, including after a
            # previous potential iteration exists.
            reference_frame = previous_radial_frame
            pair_assignment = _select_e1h1_candidate_pairs_by_overlap(
                reference_frame[:, :2],
                reference_frame[:, 2:4],
                candidates,
                minimum_principal_overlap=pair_spec.minimum_principal_overlap,
                minimum_assignment_margin=pair_spec.minimum_assignment_margin,
                electron_pair_label=pair_spec.electron_pair_label,
                hole_pair_label=pair_spec.hole_pair_label,
                context=f"current-U adjacent-k E/H selection at ik={ik}",
            )
            if self._previous_iteration_frames is not None:
                # Independently require previous-U same-k continuity.  Do
                # not replace the adjacent-k check with this second axis.
                previous_u_reference_frame = self._previous_iteration_frames[ik]
                previous_u_assignment = _select_e1h1_candidate_pairs_by_overlap(
                    previous_u_reference_frame[:, :2],
                    previous_u_reference_frame[:, 2:4],
                    candidates,
                    minimum_principal_overlap=pair_spec.minimum_principal_overlap,
                    minimum_assignment_margin=pair_spec.minimum_assignment_margin,
                    electron_pair_label=pair_spec.electron_pair_label,
                    hole_pair_label=pair_spec.hole_pair_label,
                    context=f"previous-U same-k E/H selection at ik={ik}",
                )
                pair_assignments_disagree = bool(
                    pair_assignment.electron_local_candidate_positions
                    != previous_u_assignment.electron_local_candidate_positions
                    or pair_assignment.hole_local_candidate_positions
                    != previous_u_assignment.hole_local_candidate_positions
                )
                if pair_assignments_disagree and homotopy_spec is None:
                    # Preserve the direct-only fail-closed behavior exactly
                    # when the opt-in homotopy spec is absent.
                    raise ValueError(
                        "current-U adjacent-k and previous-U same-k pair "
                        f"assignments disagree at ik={ik}"
                    )
            electron_positions = np.asarray(
                pair_assignment.electron_local_candidate_positions,
                dtype=np.int64,
            )
            hole_positions = np.asarray(
                pair_assignment.hole_local_candidate_positions,
                dtype=np.int64,
            )
            chosen = np.concatenate((electron_positions, hole_positions))
            weights = np.zeros_like(energies)
            weights[electron_positions] = (
                pair_assignment.electron_selected_projection_weights
            )
            weights[hole_positions] = (
                pair_assignment.hole_selected_projection_weights
            )
            frame = np.asarray(candidates[:, chosen], dtype=np.complex128)
            values = np.asarray(energies[chosen], dtype=float)
            singular_values = (
                np.ones(nwindow)
                if reference_frame is None
                else np.linalg.svd(reference_frame.conj().T @ frame, compute_uv=False)
            )
            previous_u_quartet_singular_values = (
                None
                if previous_u_reference_frame is None
                else np.linalg.svd(
                    previous_u_reference_frame.conj().T @ frame,
                    compute_uv=False,
                )
            )
            residual = matrix @ frame - frame * values[None, :]
            residual_norm = float(np.max(np.linalg.norm(residual, axis=0)))
            remaining = np.delete(energies, chosen)
            edge_gap = (
                np.inf
                if remaining.size == 0
                else float(np.min(np.abs(values[:, None] - remaining[None, :])))
            )
            if max(candidate_error, residual_norm) > 1e-7:
                raise ValueError(
                    f"kdotpy parent eigensystem failed at ik={ik}: "
                    f"candidate={candidate_error:.3e}, selected={residual_norm:.3e}"
                )
            if edge_gap <= 1e-6:
                raise ValueError(
                    f"selected quartet candidate complement is not isolated at ik={ik}"
                )
            if float(np.min(singular_values)) < self._minimum_subspace_singular_value:
                raise ValueError(
                    f"selected quartet subspace continuity was lost at ik={ik}: "
                    f"min_s={np.min(singular_values):.3e}"
                )
            if (
                previous_u_quartet_singular_values is not None
                and float(np.min(previous_u_quartet_singular_values))
                < self._minimum_subspace_singular_value
            ):
                raise ValueError(
                    "selected quartet previous-U same-k continuity was lost at "
                    f"ik={ik}: min_s={np.min(previous_u_quartet_singular_values):.3e}"
                )
            if homotopy_spec is not None:
                if self._previous_iteration_frames is None:
                    homotopy_k_modes.append("initial")
                elif pair_assignments_disagree:
                    if (
                        previous_u_reference_frame is None
                        or previous_u_assignment is None
                        or pair_assignment is None
                        or self._previous_successful_potential_mev is None
                        or self.last_diagnostics is None
                        or self.last_diagnostics.electron_selected_local_candidate_positions
                        is None
                        or self.last_diagnostics.hole_selected_local_candidate_positions
                        is None
                    ):
                        raise RuntimeError(
                            "homotopy conflict lacks committed previous-U E/H evidence"
                        )
                    conflict_receipt = self._certify_previous_u_same_k_homotopy(
                        ik=ik,
                        previous_potential_mev=(
                            self._previous_successful_potential_mev
                        ),
                        current_potential_mev=potential,
                        previous_frame=previous_u_reference_frame,
                        previous_electron_local_candidate_positions=tuple(
                            int(value)
                            for value in self.last_diagnostics.electron_selected_local_candidate_positions[
                                ik
                            ]
                        ),
                        previous_hole_local_candidate_positions=tuple(
                            int(value)
                            for value in self.last_diagnostics.hole_selected_local_candidate_positions[
                                ik
                            ]
                        ),
                        radial_assignment=pair_assignment,
                        radial_frame=frame,
                        direct_assignment=previous_u_assignment,
                        target_matrix=matrix,
                        target_energies=energies,
                        target_candidates=candidates,
                        target_candidate_error=candidate_error,
                        target_static_replay_error=(
                            attestation.static_parent_replay_max_error_mev
                        ),
                        target_potential_operator_error=potential_operator_error,
                        pair_spec=pair_spec,
                        homotopy_spec=homotopy_spec,
                    )
                    homotopy_conflicts.append(conflict_receipt)
                    homotopy_k_modes.append("homotopy")
                else:
                    homotopy_k_modes.append("direct")
            candidate_energies.append(energies)
            selected_local_positions.append(chosen)
            selection_weights.append(weights[chosen])
            continuation_singular_values.append(singular_values)
            edge_gaps.append(edge_gap)
            raw_orthonormality.append(raw_orth)
            selected_residuals.append(residual_norm)
            selected_energies.append(values)
            selected_frames.append(frame)
            electron_selected_local_positions.append(electron_positions)
            hole_selected_local_positions.append(hole_positions)
            current_adjacent_electron_singular_values.append(
                pair_assignment.electron_principal_singular_values
            )
            current_adjacent_hole_singular_values.append(
                pair_assignment.hole_principal_singular_values
            )
            current_adjacent_electron_margins.append(
                pair_assignment.electron_assignment_margin
            )
            current_adjacent_hole_margins.append(
                pair_assignment.hole_assignment_margin
            )
            if previous_u_assignment is not None:
                previous_u_electron_local_positions.append(
                    np.asarray(
                        previous_u_assignment.electron_local_candidate_positions,
                        dtype=np.int64,
                    )
                )
                previous_u_hole_local_positions.append(
                    np.asarray(
                        previous_u_assignment.hole_local_candidate_positions,
                        dtype=np.int64,
                    )
                )
                previous_u_electron_singular_values.append(
                    previous_u_assignment.electron_principal_singular_values
                )
                previous_u_hole_singular_values.append(
                    previous_u_assignment.hole_principal_singular_values
                )
                previous_u_electron_margins.append(
                    previous_u_assignment.electron_assignment_margin
                )
                previous_u_hole_margins.append(
                    previous_u_assignment.hole_assignment_margin
                )
            previous_radial_frame = frame

        hamiltonian = np.zeros(
            (nwindow, nwindow, self._k_cart.shape[0]), dtype=np.complex128
        )
        for ik, values in enumerate(selected_energies):
            hamiltonian[:, :, ik] = np.diag(values)
        dz = float(self._z_weights[0])
        micro_wavefunctions = np.asarray(
            [
                frame.reshape(self._z_nm.size, 8, nwindow) / np.sqrt(dz)
                for frame in selected_frames
            ],
            dtype=np.complex128,
        )
        actual_full_hash = attestation.actual_full_parent_sequence_sha256
        provenance = hashlib.sha256()
        provenance.update(self.parent_spec.fingerprint.encode())
        provenance.update(kane_poisson_array_sha256(potential).encode())
        provenance.update(actual_full_hash.encode())
        provenance.update(anchor_full_hash.encode())
        provenance.update(replayed_static_sha256.encode())
        provenance.update(kane_poisson_array_sha256(hamiltonian).encode())
        provenance.update(kane_poisson_array_sha256(micro_wavefunctions).encode())
        provenance.update(
            kane_poisson_array_sha256(
                np.asarray(selected_local_positions, dtype=np.int64)
            ).encode()
        )
        if (
            anchor_electron_positions is None
            or anchor_hole_positions is None
            or anchor_electron_singular_values is None
            or anchor_hole_singular_values is None
            or anchor_electron_margin is None
            or anchor_hole_margin is None
        ):
            raise RuntimeError("pair-resolved anchor diagnostics are incomplete")
        previous_u_diagnostic_kwargs: dict[str, Any] = {}
        if self._previous_iteration_frames is not None:
            previous_u_diagnostic_kwargs = {
                "previous_u_same_k_electron_selected_local_candidate_positions": np.asarray(
                    previous_u_electron_local_positions, dtype=np.int64
                ),
                "previous_u_same_k_hole_selected_local_candidate_positions": np.asarray(
                    previous_u_hole_local_positions, dtype=np.int64
                ),
                "previous_u_same_k_electron_principal_singular_values": np.asarray(
                    previous_u_electron_singular_values, dtype=np.float64
                ),
                "previous_u_same_k_hole_principal_singular_values": np.asarray(
                    previous_u_hole_singular_values, dtype=np.float64
                ),
                "previous_u_same_k_electron_assignment_margin": np.asarray(
                    previous_u_electron_margins, dtype=np.float64
                ),
                "previous_u_same_k_hole_assignment_margin": np.asarray(
                    previous_u_hole_margins, dtype=np.float64
                ),
            }
        homotopy_call_receipt = None
        if homotopy_spec is not None:
            previous_potential_sha256 = None
            potential_step_max_abs_mev = None
            potential_step_rms_mev = None
            if self._previous_successful_potential_mev is not None:
                previous_potential_sha256 = kane_poisson_array_sha256(
                    self._previous_successful_potential_mev
                )
                potential_step = (
                    potential - self._previous_successful_potential_mev
                )
                potential_step_max_abs_mev = float(
                    np.max(np.abs(potential_step))
                )
                potential_step_rms_mev = float(
                    np.sqrt(np.mean(potential_step**2))
                )
            homotopy_call_receipt = KdotpyPreviousUSameKHomotopyCallReceipt(
                pair_selection_spec_fingerprint=pair_spec.fingerprint,
                homotopy_spec_fingerprint=homotopy_spec.fingerprint,
                parent_spec_fingerprint=self.parent_spec.fingerprint,
                current_potential_sha256=kane_poisson_array_sha256(potential),
                previous_potential_sha256=previous_potential_sha256,
                potential_step_max_abs_mev=potential_step_max_abs_mev,
                potential_step_rms_mev=potential_step_rms_mev,
                k_modes=tuple(homotopy_k_modes),
                conflicts=tuple(homotopy_conflicts),
            )
        pair_diagnostic_kwargs = {
            "pair_selection_spec_fingerprint": pair_spec.fingerprint,
            "pair_labels": (
                pair_spec.electron_pair_label,
                pair_spec.hole_pair_label,
            ),
            "ordered_physical_selected_labels": pair_spec.ordered_selected_labels,
            "gamma_seed_status": DECLARED_GAMMA_SEED_STATUS,
            "minimum_principal_overlap": pair_spec.minimum_principal_overlap,
            "minimum_assignment_margin": pair_spec.minimum_assignment_margin,
            "anchor_electron_selected_local_candidate_positions": np.asarray(
                anchor_electron_positions, dtype=np.int64
            ),
            "anchor_hole_selected_local_candidate_positions": np.asarray(
                anchor_hole_positions, dtype=np.int64
            ),
            "anchor_electron_principal_singular_values": np.asarray(
                anchor_electron_singular_values, dtype=np.float64
            ),
            "anchor_hole_principal_singular_values": np.asarray(
                anchor_hole_singular_values, dtype=np.float64
            ),
            "anchor_electron_assignment_margin": anchor_electron_margin,
            "anchor_hole_assignment_margin": anchor_hole_margin,
            "electron_selected_local_candidate_positions": np.asarray(
                electron_selected_local_positions, dtype=np.int64
            ),
            "hole_selected_local_candidate_positions": np.asarray(
                hole_selected_local_positions, dtype=np.int64
            ),
            "current_u_adjacent_k_electron_principal_singular_values": np.asarray(
                current_adjacent_electron_singular_values, dtype=np.float64
            ),
            "current_u_adjacent_k_hole_principal_singular_values": np.asarray(
                current_adjacent_hole_singular_values, dtype=np.float64
            ),
            "current_u_adjacent_k_electron_assignment_margin": np.asarray(
                current_adjacent_electron_margins, dtype=np.float64
            ),
            "current_u_adjacent_k_hole_assignment_margin": np.asarray(
                current_adjacent_hole_margins, dtype=np.float64
            ),
            "previous_u_same_k_homotopy_receipt": homotopy_call_receipt,
            **previous_u_diagnostic_kwargs,
        }

        diagnostics = KdotpyWindowDiagnostics(
            anchor_candidate_energies_mev=np.asarray(anchor_energies),
            anchor_selected_local_candidate_positions=np.asarray(
                anchor_chosen, dtype=np.int64
            ),
            anchor_selection_projection_weights=np.asarray(
                anchor_projection_weights[anchor_chosen]
            ),
            anchor_subspace_singular_values=np.asarray(anchor_singular_values),
            anchor_candidate_edge_gap_mev=anchor_edge_gap,
            anchor_eigen_residual_mev=anchor_residual_norm,
            anchor_full_parent_sha256=anchor_full_hash,
            candidate_energies_mev=np.asarray(candidate_energies),
            selected_local_candidate_positions=np.asarray(
                selected_local_positions, dtype=np.int64
            ),
            selected_projection_weights=np.asarray(selection_weights),
            selected_subspace_singular_values=np.asarray(continuation_singular_values),
            candidate_edge_gap_mev=np.asarray(edge_gaps),
            candidate_raw_orthonormality_error=np.asarray(raw_orthonormality),
            selected_eigen_residual_mev=np.asarray(selected_residuals),
            actual_full_parent_sequence_sha256=actual_full_hash,
            raw_replayed_static_parent_sha256=(
                attestation.raw_replayed_static_parent_sha256
            ),
            replayed_static_parent_sha256=replayed_static_sha256,
            static_parent_replay_max_error_mev=(
                attestation.static_parent_replay_max_error_mev
            ),
            potential_operator_max_error_mev=potential_operator_error,
            **pair_diagnostic_kwargs,
        )
        provenance.update(diagnostics.pair_selection_diagnostics_sha256.encode())
        window = KaneWindowAtPotential(
            hamiltonian_mev=hamiltonian,
            micro_wavefunctions=micro_wavefunctions,
            selected_band_indices=self._selected_labels,
            parent_hilbert_dimension=self.parent_spec.parent_hilbert_dimension,
            calculation_split_mev=0.0,
            energy_zero_label=self._energy_zero_label,
            input_potential_sha256=kane_poisson_array_sha256(potential),
            potential_operator_fingerprint=kane_potential_operator_fingerprint(potential),
            parent_spec_fingerprint=self.parent_spec.fingerprint,
            replayed_static_parent_sha256=replayed_static_sha256,
            static_parent_replay_max_error_mev=(
                attestation.static_parent_replay_max_error_mev
            ),
            potential_operator_max_error_mev=potential_operator_error,
            provenance_fingerprint=provenance.hexdigest(),
        )
        candidate_table_sha256 = _candidate_energy_table_sha256(diagnostics)
        namespace_fingerprint = (
            _local_candidate_positions_namespace_fingerprint(
                namespace_label=LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL,
                candidate_energy_table_sha256=candidate_table_sha256,
                parent_spec_fingerprint=self.parent_spec.fingerprint,
                material_profile_sha256=self.parent_spec.material_profile_sha256,
                diagnostics=diagnostics,
            )
        )
        same_call_fingerprint = _pair_resolved_same_call_fingerprint(
            window=window,
            diagnostics=diagnostics,
            parent_spec_fingerprint=self.parent_spec.fingerprint,
            material_profile_sha256=self.parent_spec.material_profile_sha256,
            candidate_energy_table_sha256=candidate_table_sha256,
            local_candidate_positions_namespace_label=(
                LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL
            ),
            local_candidate_positions_namespace_fingerprint=(
                namespace_fingerprint
            ),
            input_potential_mev=potential,
            xml_runtime_binding_status=XML_RUNTIME_BINDING_STATUS,
        )
        result = KdotpyPairResolvedWindow(
            window=window,
            diagnostics=diagnostics,
            parent_spec=self.parent_spec,
            input_potential_mev=potential,
            parent_spec_fingerprint=self.parent_spec.fingerprint,
            material_profile_sha256=self.parent_spec.material_profile_sha256,
            candidate_energy_table_sha256=candidate_table_sha256,
            local_candidate_positions_namespace_label=(
                LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL
            ),
            local_candidate_positions_namespace_fingerprint=(
                namespace_fingerprint
            ),
            same_call_fingerprint=same_call_fingerprint,
        )

        # Prepare tracking copies before the commit boundary as well, so even a
        # late allocation failure leaves every prior tracking field untouched.
        next_anchor_frame = anchor_frame.copy()
        next_iteration_frames = tuple(frame.copy() for frame in selected_frames)
        next_successful_potential = (
            None
            if homotopy_spec is None
            else _readonly_copy(potential, dtype=np.dtype(np.float64))
        )

        # Commit state only after diagnostics, provenance, window, typed pair
        # receipt, immutable potential, and tracking copies have all been
        # constructed and validated.  Every earlier exception leaves all
        # tracking fields at the last successful call.
        self.diagnostics_history.append(diagnostics)
        self.last_diagnostics = diagnostics
        self._previous_anchor_frame = next_anchor_frame
        self._previous_iteration_frames = next_iteration_frames
        if homotopy_spec is not None:
            self._previous_successful_potential_mev = next_successful_potential
        return result


class KdotpyCanonicalWindowOnlyBuilder:
    """Receipt-preserving adapter from kdotpy pair returns to canonical windows.

    The canonical neutrality solver consumes only :class:`KaneWindowAtPotential`,
    while pair-resolved kdotpy selection returns a richer immutable receipt.
    This adapter retains every successful pair receipt and unwraps only its
    validated window. It never changes the canonical orbital-transfer density
    rule and never supplies fixed-pair occupations.
    """

    def __init__(self, builder: KdotpyCanonicalWindowBuilder) -> None:
        if not isinstance(builder, KdotpyCanonicalWindowBuilder):
            raise TypeError("builder must be KdotpyCanonicalWindowBuilder")
        self._builder = builder
        self._pair_resolved_history: list[KdotpyPairResolvedWindow] = []
        self.parent_spec = builder.parent_spec

    @property
    def raw_builder(self) -> KdotpyCanonicalWindowBuilder:
        return self._builder

    @property
    def pair_resolved_history(self) -> tuple[KdotpyPairResolvedWindow, ...]:
        return tuple(self._pair_resolved_history)

    @property
    def diagnostics_history(self) -> tuple[KdotpyWindowDiagnostics, ...]:
        return tuple(self._builder.diagnostics_history)

    @property
    def last_diagnostics(self) -> KdotpyWindowDiagnostics | None:
        return self._builder.last_diagnostics

    @property
    def static_parent_matrices(self) -> tuple[sparse.csr_matrix, ...]:
        return self._builder.static_parent_matrices

    def verify_cold_start(self) -> bool:
        if self._pair_resolved_history:
            raise ValueError("canonical window-only adapter has retained pair receipts")
        return self._builder.verify_cold_start()

    def __call__(self, potential_mev: np.ndarray) -> KaneWindowAtPotential:
        result = self._builder(potential_mev)
        if result.parent_spec.fingerprint != self.parent_spec.fingerprint:
            raise ValueError("pair-resolved return changed the canonical parent spec")
        result.window.validate(
            input_potential_mev=np.asarray(potential_mev, dtype=np.float64),
            parent_spec=self.parent_spec,
        )
        self._pair_resolved_history.append(result)
        return result.window

__all__ = [
    "KdotpyCanonicalWindowBuilder",
    "KdotpyCanonicalWindowOnlyBuilder",
]
