from __future__ import annotations

"""Dense q=0 TDHF cross-validation for zero-field full-HF MA-TBG runs.

The matrix convention follows Kwan et al., arXiv:2511.21683, Eqs. (81)-(84),
and the interaction tangent follows Khalaf et al., arXiv:2009.14827,
Appendix A, especially Eq. (A7).  This module intentionally implements only
q=0.  It does not solve finite-q sectors, run HF, plot spectra, or infer an
interaction model from BM parameters.

The zero-field HF implementation stores the centered transpose of the
conventional one-body projector,

    D_stored = (P_conventional - 1/2 I)^T.

Consequently an interaction tangent must be supplied in the same stored
convention.  All screening kernels are taken directly from the
``HFOverlapBlockSet`` attached to the source run; dielectric, gate, and
reference-density choices are never reconstructed here.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Literal, Sequence

import numpy as np

from ....core.hf import (
    HFOverlapBlockSet,
    ParticleHolePair,
    TDHFMatrices,
    assemble_tdhf_liouvillian,
    build_projected_interaction_hamiltonian,
    validate_tdhf_structures,
)
from ._hf_basis_overlap import (
    RestrictedHartreeFockRun,
    TBGZeroFieldHFSourceReceipt,
    TBG_ZERO_FIELD_CENTERED_REFERENCE_CONVENTION,
    TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA,
    TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA_VERSION,
    build_h0_from_bm,
    restricted_occupied_state_count,
    tbg_zero_field_lattice_kvec_sha256,
    tbg_zero_field_overlap_kernel_inventory_fingerprint,
)
from .model import BMSolution


@dataclass(frozen=True)
class TBGZeroFieldTDHFProvenance:
    """Explicit receipts for the HF functional linearized by TDHF.

    Source labels identify external evidence, while
    ``expected_hf_source_receipt_sha256`` binds that evidence to every typed
    receipt field.  Numerical kernels still come only from the overlap blocks
    attached to the run.
    """

    hf_run_source: str
    overlap_blocks_source: str
    interaction_parameters_source: str
    reference_density_source: str
    expected_hf_source_receipt_sha256: str
    hf_mode: Literal["full"]

    def __post_init__(self) -> None:
        for field_name in (
            "hf_run_source",
            "overlap_blocks_source",
            "interaction_parameters_source",
            "reference_density_source",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"TDHF provenance field {field_name!r} must be non-empty")
            object.__setattr__(self, field_name, value)
        expected_fingerprint = _validate_sha256(
            self.expected_hf_source_receipt_sha256,
            name="expected_hf_source_receipt_sha256",
        )
        object.__setattr__(
            self,
            "expected_hf_source_receipt_sha256",
            expected_fingerprint,
        )
        if self.hf_mode != "full":
            raise ValueError("TBG zero-field TDHF accepts only explicitly identified full-HF runs")


@dataclass(frozen=True)
class TBGZeroFieldTDHFContext:
    """A source-bound q=0 TDHF linearization context.

    ``grid_solution`` identifies the exact BM source grid but is not used to
    regenerate screening.  ``run.overlap_blocks`` is the sole interaction-block
    inventory, so this adapter cannot silently choose epsilon, gate distance,
    a q=0 prescription, or a reference density.
    """

    grid_solution: BMSolution
    run: RestrictedHartreeFockRun
    beta: float
    provenance: TBGZeroFieldTDHFProvenance
    closure_tolerance: float = 1.0e-10
    _bound_live_source_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        beta = float(self.beta)
        tolerance = float(self.closure_tolerance)
        if not np.isfinite(beta):
            raise ValueError(f"TDHF beta must be finite, got {self.beta}")
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError(
                f"TDHF closure_tolerance must be finite and non-negative, got {self.closure_tolerance}"
            )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "closure_tolerance", tolerance)
        if not isinstance(self.provenance, TBGZeroFieldTDHFProvenance):
            raise TypeError("provenance must be a TBGZeroFieldTDHFProvenance")
        if not bool(self.run.converged):
            raise ValueError("q=0 TDHF requires a converged full-HF source run")

        state = self.run.state
        source = self.grid_solution
        self._validated_source_receipt()
        expected_dimensions = (int(source.n_spin), int(source.n_eta), int(source.nb), int(source.nk))
        state_dimensions = (int(state.n_spin), int(state.n_eta), int(state.n_band), int(state.nk))
        if state_dimensions != expected_dimensions:
            raise ValueError(
                "HF state and BM source-grid dimensions differ: "
                f"state={state_dimensions}, source={expected_dimensions}"
            )
        if int(state.nt) != int(source.nt):
            raise ValueError(f"HF state nt={state.nt} does not match BM source nt={source.nt}")

        source_h0 = build_h0_from_bm(source)
        saved_h0 = _as_square_matrix_field(state.h0, name="run.state.h0")
        if source_h0.shape != saved_h0.shape:
            raise ValueError(
                f"BM source h0 shape {source_h0.shape} does not match saved HF h0 shape {saved_h0.shape}"
            )
        h0_residual = _max_abs(source_h0 - saved_h0)
        if not np.isfinite(h0_residual) or h0_residual > tolerance:
            raise ValueError(
                "BM source h0 does not match run.state.h0: "
                f"max residual={h0_residual:.6e}, tolerance={tolerance:.6e}"
            )

        _validate_overlap_inventory(self.run.overlap_blocks, nt=state.nt, nk=state.nk)
        saved_hamiltonian = _as_square_matrix_field(
            state.hamiltonian,
            name="run.state.hamiltonian",
        )
        density = _as_square_matrix_field(state.density, name="run.state.density")
        expected_hamiltonian = saved_h0 + _apply_source_interaction_hamiltonian(
            self,
            density,
        )
        hamiltonian_residual = _max_abs(saved_hamiltonian - expected_hamiltonian)
        if not np.isfinite(hamiltonian_residual) or hamiltonian_residual > tolerance:
            raise ValueError(
                "Saved full-HF Hamiltonian is not closed by h0 plus the saved interaction functional: "
                f"max residual={hamiltonian_residual:.6e}, tolerance={tolerance:.6e}"
            )
        object.__setattr__(
            self,
            "_bound_live_source_sha256",
            self._current_live_source_sha256(),
        )

    def _validated_source_receipt(self) -> TBGZeroFieldHFSourceReceipt:
        receipt = self.run.state.hf_source_receipt
        if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
            raise ValueError(
                "TBG zero-field TDHF requires a typed TBGZeroFieldHFSourceReceipt"
            )
        if receipt.schema != TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA:
            raise ValueError("TBG zero-field TDHF source receipt schema mismatch")
        if receipt.schema_version != TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("TBG zero-field TDHF source receipt schema-version mismatch")
        if receipt.hf_mode != "full" or self.provenance.hf_mode != receipt.hf_mode:
            raise ValueError("TBG zero-field TDHF requires an hf_mode='full' source receipt")
        if receipt.beta != self.beta:
            raise ValueError(
                "Explicit TDHF beta does not match the source receipt: "
                f"tdhf={self.beta:.16g}, receipt={receipt.beta:.16g}"
            )
        if receipt.v0 != self.run.state.v0:
            raise ValueError(
                "HF source receipt v0 does not match run.state.v0 exactly: "
                f"receipt.v0={receipt.v0!r}, run.state.v0={self.run.state.v0!r}"
            )
        lattice_fingerprint = tbg_zero_field_lattice_kvec_sha256(
            self.grid_solution.lattice_kvec
        )
        if receipt.lattice_kvec_sha256 != lattice_fingerprint:
            raise ValueError(
                "BM source lattice_kvec does not match the exact source receipt"
            )
        inventory_fingerprint = tbg_zero_field_overlap_kernel_inventory_fingerprint(
            self.run.overlap_blocks
        )
        if receipt.overlap_kernel_inventory_sha256 != inventory_fingerprint:
            raise ValueError(
                "Live overlap/kernel inventory does not match the exact source receipt"
            )
        if (
            receipt.centered_reference_convention
            != TBG_ZERO_FIELD_CENTERED_REFERENCE_CONVENTION
        ):
            raise ValueError("TBG zero-field TDHF centered-reference receipt mismatch")
        if (
            self.provenance.expected_hf_source_receipt_sha256
            != receipt.fingerprint
        ):
            raise ValueError(
                "TDHF provenance expected fingerprint does not match the HF source receipt"
            )
        return receipt

    def _current_live_source_sha256(self) -> str:
        receipt = self._validated_source_receipt()
        return _tbg_zero_field_tdhf_live_source_sha256(
            self.run,
            beta=self.beta,
            receipt=receipt,
        )

    def _revalidate_live_source(self) -> None:
        receipt = self.run.state.hf_source_receipt
        if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
            raise ValueError(
                "TBG zero-field TDHF live HF source changed after context construction"
            )
        current = _tbg_zero_field_tdhf_live_source_sha256(
            self.run,
            beta=self.beta,
            receipt=receipt,
        )
        if current != self._bound_live_source_sha256:
            raise ValueError(
                "TBG zero-field TDHF live HF source changed after context construction"
            )
        self._validated_source_receipt()

    @property
    def overlap_blocks(self) -> HFOverlapBlockSet:
        """Return the exact overlap-block object carried by the HF run."""

        return self.run.overlap_blocks

    def build_interaction_hamiltonian(self, delta_density: np.ndarray) -> np.ndarray:
        """Apply the source HF interaction tangent to a stored-convention density.

        ``build_projected_interaction_hamiltonian`` is linear in its density
        argument.  Calling it directly with the run's already-screened blocks
        is the scalar oracle for the vectorized A/B contractions below.  In
        particular, this method does not call the TBG screening wrapper and
        therefore cannot fill missing kernels from default dielectric or gate
        parameters.
        """

        self._revalidate_live_source()
        tangent = np.asarray(delta_density, dtype=np.complex128)
        if tangent.shape != self.run.state.density.shape:
            raise ValueError(
                f"Expected interaction tangent shape {self.run.state.density.shape}, got {tangent.shape}"
            )
        result = _apply_source_interaction_hamiltonian(self, tangent)
        self._revalidate_live_source()
        return result


@dataclass(frozen=True)
class TBGZeroFieldTDHFOccupationResiduals:
    projector_hermitian: float
    projector_hf_offdiagonal: float
    projector_hf_diagonal_imaginary: float
    projector_hf_zero_one: float
    projector_trace: float
    tolerance: float
    trace_tolerance: float

    @property
    def ok(self) -> bool:
        return (
            self.projector_hermitian <= self.tolerance
            and self.projector_hf_offdiagonal <= self.tolerance
            and self.projector_hf_diagonal_imaginary <= self.tolerance
            and self.projector_hf_zero_one <= self.tolerance
            and self.projector_trace <= self.trace_tolerance
        )


@dataclass(frozen=True)
class TBGZeroFieldTDHFOrbitals:
    """Per-k HF orbitals with stable Fortran-order global indices."""

    energies: np.ndarray
    eigenvectors: np.ndarray
    occupied_mask: np.ndarray
    conventional_projector: np.ndarray
    mu: float
    occupation_residuals: TBGZeroFieldTDHFOccupationResiduals
    source_hamiltonian_sha256: str
    source_density_sha256: str

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=float)
        if energies.ndim != 2 or min(energies.shape, default=0) <= 0:
            raise ValueError(f"Expected non-empty orbital energies shape (nt, nk), got {energies.shape}")
        nt, nk = int(energies.shape[0]), int(energies.shape[1])
        eigenvectors = np.asarray(self.eigenvectors, dtype=np.complex128)
        occupied_mask = np.asarray(self.occupied_mask, dtype=bool)
        projector = np.asarray(self.conventional_projector, dtype=np.complex128)
        if eigenvectors.shape != (nt, nt, nk):
            raise ValueError(f"Expected eigenvectors shape {(nt, nt, nk)}, got {eigenvectors.shape}")
        if occupied_mask.shape != (nt, nk):
            raise ValueError(f"Expected occupied_mask shape {(nt, nk)}, got {occupied_mask.shape}")
        if projector.shape != (nt, nt, nk):
            raise ValueError(f"Expected conventional_projector shape {(nt, nt, nk)}, got {projector.shape}")
        for field_name, values in (
            ("energies", energies),
            ("eigenvectors", eigenvectors),
            ("conventional_projector", projector),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{field_name} must contain only finite values")
        if not isinstance(self.occupation_residuals, TBGZeroFieldTDHFOccupationResiduals):
            raise TypeError("occupation_residuals must be a TBGZeroFieldTDHFOccupationResiduals")
        for field_name in ("source_hamiltonian_sha256", "source_density_sha256"):
            value = str(getattr(self, field_name)).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a SHA-256 hexadecimal digest")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "occupied_mask", occupied_mask)
        object.__setattr__(self, "conventional_projector", projector)
        object.__setattr__(self, "mu", float(self.mu))

    @property
    def nt(self) -> int:
        return int(self.energies.shape[0])

    @property
    def nk(self) -> int:
        return int(self.energies.shape[1])

    @property
    def global_energies(self) -> np.ndarray:
        return self.energies.reshape(-1, order="F")

    def global_index(self, local_index: int, k_index: int) -> int:
        local = int(local_index)
        ik = int(k_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        if ik < 0 or ik >= self.nk:
            raise IndexError(f"k_index={ik} outside [0, {self.nk})")
        return local + self.nt * ik

    def decode_global_index(self, global_index: int) -> tuple[int, int]:
        index = int(global_index)
        if index < 0 or index >= self.nt * self.nk:
            raise IndexError(f"global_index={index} outside [0, {self.nt * self.nk})")
        return index % self.nt, index // self.nt


@dataclass(frozen=True)
class TBGZeroFieldTDHFTangentParity:
    """Selected-column parity against the scalar HF interaction tangent."""

    columns: tuple[int, ...]
    a_column_residuals: tuple[float, ...]
    b_column_residuals: tuple[float, ...]
    tolerance: float

    @property
    def max_a_residual(self) -> float:
        return max(self.a_column_residuals, default=0.0)

    @property
    def max_b_residual(self) -> float:
        return max(self.b_column_residuals, default=0.0)

    @property
    def ok(self) -> bool:
        return max(self.max_a_residual, self.max_b_residual) <= self.tolerance


def stored_density_to_conventional_projector(stored_density: np.ndarray) -> np.ndarray:
    """Convert ``D_stored`` to ``P = D_stored.T + 1/2 I`` at every k."""

    stored = _as_square_matrix_field(stored_density, name="stored_density")
    nt = int(stored.shape[0])
    identity = np.eye(nt, dtype=np.complex128)[:, :, None]
    return np.swapaxes(stored, 0, 1) + 0.5 * identity


def conventional_projector_to_stored_density(projector: np.ndarray) -> np.ndarray:
    """Convert a conventional projector to the centered stored HF convention."""

    conventional = _as_square_matrix_field(projector, name="projector")
    nt = int(conventional.shape[0])
    identity = np.eye(nt, dtype=np.complex128)[:, :, None]
    return np.swapaxes(conventional - 0.5 * identity, 0, 1)


def stored_tangent_to_conventional_tangent(stored_tangent: np.ndarray) -> np.ndarray:
    """Transpose a centered-density tangent without adding an identity term."""

    tangent = _as_square_matrix_field(stored_tangent, name="stored_tangent")
    return np.swapaxes(tangent, 0, 1)


def conventional_tangent_to_stored_tangent(conventional_tangent: np.ndarray) -> np.ndarray:
    """Transpose a conventional projector tangent into the HF stored convention."""

    tangent = _as_square_matrix_field(conventional_tangent, name="conventional_tangent")
    return np.swapaxes(tangent, 0, 1)


def build_tbg_zero_field_tdhf_orbitals(
    run: RestrictedHartreeFockRun,
    *,
    projector_tolerance: float = 1.0e-7,
) -> TBGZeroFieldTDHFOrbitals:
    """Reconstruct HF orbitals and 0/1 occupations from a full-HF run.

    Occupations are not inferred from a chemical potential or an energy sort.
    The conventional projector ``run.state.density.T + 1/2 I`` must be diagonal
    in the freshly reconstructed HF eigenbasis and have occupations near 0/1.
    """

    tolerance = float(projector_tolerance)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError(f"projector_tolerance must be finite and non-negative, got {projector_tolerance}")

    state = run.state
    hamiltonian = _as_square_matrix_field(state.hamiltonian, name="run.state.hamiltonian")
    nt, _nt_rhs, nk = hamiltonian.shape
    if nt != int(state.n_spin) * int(state.n_eta) * int(state.n_band):
        raise ValueError(
            f"HF dimension {nt} is incompatible with n_spin={state.n_spin}, "
            f"n_eta={state.n_eta}, n_band={state.n_band}"
        )
    hamiltonian_residual = _max_abs(hamiltonian - np.swapaxes(np.conjugate(hamiltonian), 0, 1))
    if hamiltonian_residual > max(tolerance, 1.0e-12):
        raise ValueError(
            "run.state.hamiltonian is not Hermitian; "
            f"max residual {hamiltonian_residual:.6e}"
        )

    energies = np.empty((nt, nk), dtype=float)
    eigenvectors = np.empty((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], eigenvectors[:, :, ik] = np.linalg.eigh(hamiltonian[:, :, ik])

    projector = stored_density_to_conventional_projector(state.density)
    projector_hermitian = _max_abs(projector - np.swapaxes(np.conjugate(projector), 0, 1))
    projector_hf = np.einsum(
        "aik,abk,bjk->ijk",
        np.conjugate(eigenvectors),
        projector,
        eigenvectors,
        optimize=True,
    )
    diagonal = np.einsum("iik->ik", projector_hf)
    offdiagonal = projector_hf.copy()
    indices = np.arange(nt)
    offdiagonal[indices, indices, :] = 0.0

    diagonal_real = diagonal.real
    offdiagonal_residual = _max_abs(offdiagonal)
    diagonal_imaginary_residual = _max_abs(diagonal.imag)
    zero_one_residual = _max_abs(np.minimum(np.abs(diagonal_real), np.abs(diagonal_real - 1.0)))
    expected_occupied = restricted_occupied_state_count(float(state.nu), nt, nk)
    trace_residual = abs(float(np.sum(diagonal_real)) - float(expected_occupied))
    trace_tolerance = tolerance * max(1, nt * nk)

    failures: list[str] = []
    if projector_hermitian > tolerance:
        failures.append(f"Hermitian={projector_hermitian:.6e}")
    if offdiagonal_residual > tolerance:
        failures.append(f"HF-offdiagonal={offdiagonal_residual:.6e}")
    if diagonal_imaginary_residual > tolerance:
        failures.append(f"HF-diagonal-imaginary={diagonal_imaginary_residual:.6e}")
    if zero_one_residual > tolerance:
        failures.append(f"HF-0/1={zero_one_residual:.6e}")
    if trace_residual > trace_tolerance:
        failures.append(f"trace={trace_residual:.6e} (bound {trace_tolerance:.6e})")
    if failures:
        raise ValueError(
            "Stored full-HF density does not define a 0/1 projector in the reconstructed HF eigenbasis: "
            + ", ".join(failures)
        )

    occupied_mask = diagonal_real > 0.5
    occupied_count = int(np.count_nonzero(occupied_mask))
    if occupied_count != expected_occupied:
        raise ValueError(
            f"HF projector occupation count {occupied_count} does not match filling count {expected_occupied}"
        )
    if np.any(occupied_mask) and np.any(~occupied_mask):
        mu = 0.5 * (float(np.max(energies[occupied_mask])) + float(np.min(energies[~occupied_mask])))
    else:
        mu = float(np.mean(energies))

    residuals = TBGZeroFieldTDHFOccupationResiduals(
        projector_hermitian=projector_hermitian,
        projector_hf_offdiagonal=offdiagonal_residual,
        projector_hf_diagonal_imaginary=diagonal_imaginary_residual,
        projector_hf_zero_one=zero_one_residual,
        projector_trace=trace_residual,
        tolerance=tolerance,
        trace_tolerance=trace_tolerance,
    )
    return TBGZeroFieldTDHFOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occupied_mask,
        conventional_projector=projector,
        mu=mu,
        occupation_residuals=residuals,
        source_hamiltonian_sha256=_complex_array_sha256(state.hamiltonian),
        source_density_sha256=_complex_array_sha256(state.density),
    )


def build_tbg_zero_field_tdhf_q0_pairs(
    orbitals: TBGZeroFieldTDHFOrbitals,
) -> tuple[ParticleHolePair, ...]:
    """Build all q=0 pairs in ``k -> hole -> particle`` order."""

    pairs: list[ParticleHolePair] = []
    for ik in range(orbitals.nk):
        occupied = np.flatnonzero(orbitals.occupied_mask[:, ik])
        unoccupied = np.flatnonzero(~orbitals.occupied_mask[:, ik])
        for hole in occupied:
            for particle in unoccupied:
                pairs.append(
                    ParticleHolePair(
                        particle=orbitals.global_index(int(particle), ik),
                        hole=orbitals.global_index(int(hole), ik),
                        particle_momentum=ik,
                        hole_momentum=ik,
                    )
                )
    return tuple(pairs)


def build_tbg_zero_field_tdhf_q0_matrices(
    context: TBGZeroFieldTDHFContext,
    orbitals: TBGZeroFieldTDHFOrbitals | None = None,
    *,
    structure_tolerance: float = 1.0e-8,
) -> TDHFMatrices:
    """Build dense q=0 A/B/L matrices from the source run's scalar blocks.

    For pair ``i=(p,h)`` and ``j=(p',h')``, the HF-energy gap supplies the
    diagonal term in Kwan Eq. (82).  Hartree and Fock terms are vectorized from
    exactly the diagonal-overlap, Hartree-kernel, full-overlap, and Fock-kernel
    arrays consumed by ``build_projected_interaction_hamiltonian``.  ``L`` is
    assembled by the common core helper according to Kwan Eq. (84).
    """

    context._revalidate_live_source()
    resolved_orbitals = (
        build_tbg_zero_field_tdhf_orbitals(context.run)
        if orbitals is None
        else orbitals
    )
    pairs = build_tbg_zero_field_tdhf_q0_pairs(resolved_orbitals)
    result = build_tbg_zero_field_tdhf_q0_matrices_from_pairs(
        context,
        resolved_orbitals,
        pairs,
        structure_tolerance=structure_tolerance,
    )
    context._revalidate_live_source()
    return result


def build_tbg_zero_field_tdhf_q0_matrices_from_pairs(
    context: TBGZeroFieldTDHFContext,
    orbitals: TBGZeroFieldTDHFOrbitals,
    pairs: Sequence[ParticleHolePair],
    *,
    structure_tolerance: float = 1.0e-8,
) -> TDHFMatrices:
    """Vectorized dense q=0 A/B assembly for an explicitly ordered pair list."""

    context._revalidate_live_source()
    tolerance = float(structure_tolerance)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError(f"structure_tolerance must be finite and non-negative, got {structure_tolerance}")
    if orbitals.nt != context.run.state.nt or orbitals.nk != context.run.state.nk:
        raise ValueError("TDHF orbitals do not match the source HF run dimensions")
    _validate_orbitals_source(context, orbitals)

    ph_pairs = tuple(pairs)
    p_local, h_local, pair_k = _validated_q0_pair_arrays(orbitals, ph_pairs)
    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    if n_pairs:
        global_energies = orbitals.global_energies
        gaps = np.asarray(
            [global_energies[pair.particle] - global_energies[pair.hole] for pair in ph_pairs],
            dtype=float,
        )
        A[np.diag_indices(n_pairs)] = gaps

    indices_by_k = tuple(np.flatnonzero(pair_k == ik) for ik in range(orbitals.nk))
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    blocks = context.overlap_blocks
    _validate_overlap_inventory(blocks, nt=orbitals.nt, nk=orbitals.nk)
    scale = float(context.beta) * float(context.run.state.v0) / float(orbitals.nk)

    for shift in blocks.shifts:
        overlap = np.asarray(blocks.overlaps[shift], dtype=np.complex128)

        hartree_kernel = blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            diagonal_overlap = blocks.diagonal_overlaps.get(shift)
            if diagonal_overlap is None:
                raise ValueError(f"Missing diagonal overlap for active Hartree shift {shift}")
            diagonal = np.asarray(diagonal_overlap, dtype=np.complex128)
            if diagonal.shape != (orbitals.nt, orbitals.nt, orbitals.nk):
                raise ValueError(
                    f"Expected diagonal overlap shape {(orbitals.nt, orbitals.nt, orbitals.nk)}, "
                    f"got {diagonal.shape} for shift {shift}"
                )
            if n_pairs and float(hartree_kernel) != 0.0:
                form_ph = np.empty(n_pairs, dtype=np.complex128)
                form_hp = np.empty(n_pairs, dtype=np.complex128)
                for ik, pair_indices in enumerate(indices_by_k):
                    if pair_indices.size == 0:
                        continue
                    form = U[:, :, ik].conjugate().T @ diagonal[:, :, ik] @ U[:, :, ik]
                    form_ph[pair_indices] = form[p_local[pair_indices], h_local[pair_indices]]
                    form_hp[pair_indices] = form[h_local[pair_indices], p_local[pair_indices]]
                prefactor = scale * float(hartree_kernel)
                # Direct terms: A=F_ph F_ph^*, B=F_ph F_hp^*.
                A += prefactor * form_ph[:, None] * np.conjugate(form_ph[None, :])
                B += prefactor * form_ph[:, None] * np.conjugate(form_hp[None, :])

        fock_kernel = blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            kernel = np.asarray(fock_kernel, dtype=np.complex128)
            if kernel.shape != (orbitals.nk, orbitals.nk):
                raise ValueError(
                    f"Expected Fock kernel shape {(orbitals.nk, orbitals.nk)}, got {kernel.shape} for shift {shift}"
                )
            for target_k, target_indices in enumerate(indices_by_k):
                if target_indices.size == 0:
                    continue
                target_u = U[:, :, target_k]
                target_p = p_local[target_indices]
                target_h = h_local[target_indices]
                for source_k, source_indices in enumerate(indices_by_k):
                    if source_indices.size == 0 or kernel[target_k, source_k] == 0.0:
                        continue
                    source_u = U[:, :, source_k]
                    source_p = p_local[source_indices]
                    source_h = h_local[source_indices]
                    form = (
                        target_u.conjugate().T
                        @ overlap[:, target_k, :, source_k]
                        @ source_u
                    )
                    prefactor = scale * kernel[target_k, source_k]
                    # Exchange terms obtained by differentiating the scalar
                    # Fock contraction in the stored-projector convention.
                    A[np.ix_(target_indices, source_indices)] -= prefactor * (
                        form[np.ix_(target_p, source_p)]
                        * np.conjugate(form[np.ix_(target_h, source_h)])
                    )
                    B[np.ix_(target_indices, source_indices)] -= prefactor * (
                        form[np.ix_(target_p, source_h)]
                        * np.conjugate(form[np.ix_(target_h, source_p)])
                    )

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(
        A,
        B,
        L,
        tolerance=tolerance,
        raise_on_fail=True,
    )
    result = TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)
    context._revalidate_live_source()
    return result


def validate_tbg_zero_field_tdhf_tangent_columns(
    context: TBGZeroFieldTDHFContext,
    orbitals: TBGZeroFieldTDHFOrbitals,
    matrices: TDHFMatrices,
    columns: Sequence[int],
    *,
    tolerance: float = 1.0e-10,
    raise_on_fail: bool = True,
) -> TBGZeroFieldTDHFTangentParity:
    """Compare selected vectorized A/B columns to the scalar HF tangent.

    An A column uses the stored-HF-basis tangent ``D[h',p']=1`` (conventional
    ``delta P=|p'><h'|``); a B column uses ``D[p',h']=1`` (conventional
    ``delta P=|h'><p'|``).  Projecting
    ``context.build_interaction_hamiltonian(delta_density)`` back onto every
    row pair must reproduce ``A-gap`` and ``B`` respectively.
    """

    context._revalidate_live_source()
    resolved_tolerance = float(tolerance)
    if resolved_tolerance < 0.0 or not np.isfinite(resolved_tolerance):
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}")
    resolved_columns = tuple(int(value) for value in columns)
    if not resolved_columns:
        raise ValueError("At least one TDHF column is required for tangent parity")
    if len(set(resolved_columns)) != len(resolved_columns):
        raise ValueError("Tangent-parity column indices must be unique")
    n_pairs = len(matrices.pairs)
    if matrices.A.shape != (n_pairs, n_pairs) or matrices.B.shape != (n_pairs, n_pairs):
        raise ValueError("TDHF matrix shapes do not match their pair inventory")
    if any(column < 0 or column >= n_pairs for column in resolved_columns):
        raise IndexError(f"Tangent-parity columns must lie in [0, {n_pairs})")
    _validate_orbitals_source(context, orbitals)

    p_local, h_local, pair_k = _validated_q0_pair_arrays(orbitals, matrices.pairs)
    global_energies = orbitals.global_energies
    kinetic = np.zeros_like(matrices.A)
    for index, pair in enumerate(matrices.pairs):
        kinetic[index, index] = global_energies[pair.particle] - global_energies[pair.hole]
    expected_a_interaction = matrices.A - kinetic

    a_residuals: list[float] = []
    b_residuals: list[float] = []
    for column in resolved_columns:
        a_stored_hf = np.zeros((orbitals.nt, orbitals.nt, orbitals.nk), dtype=np.complex128)
        b_stored_hf = np.zeros_like(a_stored_hf)
        source_k = int(pair_k[column])
        a_stored_hf[h_local[column], p_local[column], source_k] = 1.0
        b_stored_hf[p_local[column], h_local[column], source_k] = 1.0

        a_response = context.build_interaction_hamiltonian(
            _stored_hf_tangent_to_basis(a_stored_hf, orbitals.eigenvectors)
        )
        b_response = context.build_interaction_hamiltonian(
            _stored_hf_tangent_to_basis(b_stored_hf, orbitals.eigenvectors)
        )
        a_column = _project_response_onto_pairs(a_response, orbitals, p_local, h_local, pair_k)
        b_column = _project_response_onto_pairs(b_response, orbitals, p_local, h_local, pair_k)
        a_residuals.append(_max_abs(a_column - expected_a_interaction[:, column]))
        b_residuals.append(_max_abs(b_column - matrices.B[:, column]))

    result = TBGZeroFieldTDHFTangentParity(
        columns=resolved_columns,
        a_column_residuals=tuple(a_residuals),
        b_column_residuals=tuple(b_residuals),
        tolerance=resolved_tolerance,
    )
    if raise_on_fail and not result.ok:
        raise ValueError(
            "TBG zero-field TDHF tangent-column parity failed: "
            f"max A residual={result.max_a_residual:.6e}, "
            f"max B residual={result.max_b_residual:.6e}, "
            f"tolerance={result.tolerance:.6e}"
        )
    context._revalidate_live_source()
    return result


def _stored_hf_tangent_to_basis(stored_hf: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    tangent = np.asarray(stored_hf, dtype=np.complex128)
    U = np.asarray(eigenvectors, dtype=np.complex128)
    if tangent.shape != U.shape:
        raise ValueError(f"Stored HF tangent shape {tangent.shape} does not match eigenvectors {U.shape}")
    return np.einsum("aik,ijk,bjk->abk", np.conjugate(U), tangent, U, optimize=True)


def _project_response_onto_pairs(
    response: np.ndarray,
    orbitals: TBGZeroFieldTDHFOrbitals,
    p_local: np.ndarray,
    h_local: np.ndarray,
    pair_k: np.ndarray,
) -> np.ndarray:
    field = _as_square_matrix_field(response, name="interaction_response")
    if field.shape != (orbitals.nt, orbitals.nt, orbitals.nk):
        raise ValueError(
            f"Expected interaction response shape {(orbitals.nt, orbitals.nt, orbitals.nk)}, got {field.shape}"
        )
    projected = np.empty(p_local.size, dtype=np.complex128)
    for ik in range(orbitals.nk):
        indices = np.flatnonzero(pair_k == ik)
        if indices.size == 0:
            continue
        U = orbitals.eigenvectors[:, :, ik]
        response_hf = U.conjugate().T @ field[:, :, ik] @ U
        projected[indices] = response_hf[p_local[indices], h_local[indices]]
    return projected


def _validated_q0_pair_arrays(
    orbitals: TBGZeroFieldTDHFOrbitals,
    pairs: Sequence[ParticleHolePair],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ph_pairs = tuple(pairs)
    if len({(int(pair.particle), int(pair.hole)) for pair in ph_pairs}) != len(ph_pairs):
        raise ValueError("q=0 TDHF pair inventory contains duplicates")
    p_local = np.empty(len(ph_pairs), dtype=int)
    h_local = np.empty(len(ph_pairs), dtype=int)
    pair_k = np.empty(len(ph_pairs), dtype=int)
    for index, pair in enumerate(ph_pairs):
        particle, particle_k = orbitals.decode_global_index(pair.particle)
        hole, hole_k = orbitals.decode_global_index(pair.hole)
        if particle_k != hole_k:
            raise ValueError("q=0 TDHF pair has particle and hole at different k points")
        if orbitals.occupied_mask[particle, particle_k]:
            raise ValueError("q=0 TDHF pair particle is occupied")
        if not orbitals.occupied_mask[hole, hole_k]:
            raise ValueError("q=0 TDHF pair hole is unoccupied")
        if pair.particle_momentum is not None and int(pair.particle_momentum) != particle_k:
            raise ValueError("q=0 TDHF pair particle_momentum metadata is inconsistent")
        if pair.hole_momentum is not None and int(pair.hole_momentum) != hole_k:
            raise ValueError("q=0 TDHF pair hole_momentum metadata is inconsistent")
        p_local[index] = particle
        h_local[index] = hole
        pair_k[index] = particle_k
    return p_local, h_local, pair_k


def _validate_orbitals_source(
    context: TBGZeroFieldTDHFContext,
    orbitals: TBGZeroFieldTDHFOrbitals,
) -> None:
    state = context.run.state
    current_hamiltonian_hash = _complex_array_sha256(state.hamiltonian)
    current_density_hash = _complex_array_sha256(state.density)
    if orbitals.source_hamiltonian_sha256 != current_hamiltonian_hash:
        raise ValueError("TDHF orbitals are stale: source Hamiltonian hash changed")
    if orbitals.source_density_sha256 != current_density_hash:
        raise ValueError("TDHF orbitals are stale: source density hash changed")

    hamiltonian = _as_square_matrix_field(state.hamiltonian, name="run.state.hamiltonian")
    projector = stored_density_to_conventional_projector(state.density)
    tolerance = max(float(orbitals.occupation_residuals.tolerance), 1.0e-10)
    identity = np.eye(orbitals.nt, dtype=np.complex128)
    reconstructed_projector = np.empty_like(projector)
    max_unitarity_residual = 0.0
    max_diagonalization_residual = 0.0
    for ik in range(orbitals.nk):
        unitary = orbitals.eigenvectors[:, :, ik]
        max_unitarity_residual = max(
            max_unitarity_residual,
            _max_abs(unitary.conjugate().T @ unitary - identity),
        )
        transformed_hamiltonian = unitary.conjugate().T @ hamiltonian[:, :, ik] @ unitary
        max_diagonalization_residual = max(
            max_diagonalization_residual,
            _max_abs(
                transformed_hamiltonian
                - np.diag(np.asarray(orbitals.energies[:, ik], dtype=np.complex128))
            ),
        )
        occupied_vectors = unitary[:, orbitals.occupied_mask[:, ik]]
        reconstructed_projector[:, :, ik] = occupied_vectors @ occupied_vectors.conjugate().T

    projector_source_residual = _max_abs(orbitals.conventional_projector - projector)
    projector_orbital_residual = _max_abs(reconstructed_projector - projector)
    failures: list[str] = []
    if not orbitals.occupation_residuals.ok:
        failures.append("stored occupation residual receipt is not valid")
    if max_unitarity_residual > tolerance:
        failures.append(f"unitarity={max_unitarity_residual:.6e}")
    if max_diagonalization_residual > tolerance:
        failures.append(f"diagonalization={max_diagonalization_residual:.6e}")
    if projector_source_residual > tolerance:
        failures.append(f"source-projector={projector_source_residual:.6e}")
    if projector_orbital_residual > tolerance:
        failures.append(f"orbital-projector={projector_orbital_residual:.6e}")
    if failures:
        raise ValueError(
            "TDHF orbitals fail source-bound eigensystem validation: " + ", ".join(failures)
        )

def _validate_overlap_inventory(blocks: HFOverlapBlockSet, *, nt: int, nk: int) -> None:
    shifts = tuple(blocks.shifts)
    shift_set = set(shifts)
    gvecs = np.asarray(blocks.gvecs)
    if gvecs.shape != (len(shifts),):
        raise ValueError("Overlap shift and reciprocal-vector inventories differ")
    if len(shift_set) != len(shifts):
        raise ValueError("Overlap shifts must be unique")
    if set(blocks.overlaps) != shift_set:
        raise ValueError("Overlap block keys must exactly match the ordered shift inventory")

    diagonal_keys = set(blocks.diagonal_overlaps)
    hartree_keys = set(blocks.hartree_screening)
    fock_keys = set(blocks.fock_screening)
    active_shifts = hartree_keys | fock_keys
    if not active_shifts:
        raise ValueError("TDHF source overlap blocks are unscreened: no active kernel inventory")
    if not (diagonal_keys | active_shifts) <= shift_set:
        raise ValueError("Active overlap/kernel inventory contains shifts absent from overlaps")
    if not hartree_keys <= diagonal_keys:
        missing = sorted(hartree_keys - diagonal_keys)
        raise ValueError(
            f"Active Hartree shifts require diagonal-overlap entries; missing {missing}"
        )
    for shift in shifts:
        overlap = np.asarray(blocks.overlaps[shift])
        if overlap.shape != (nt, nk, nt, nk):
            raise ValueError(
                f"Expected overlap shape {(nt, nk, nt, nk)}, got {overlap.shape} for shift {shift}"
            )
        if not np.all(np.isfinite(overlap)):
            raise ValueError(f"Overlap block contains non-finite values for shift {shift}")
    for shift in diagonal_keys:
        diagonal = np.asarray(blocks.diagonal_overlaps[shift])
        if diagonal.shape != (nt, nt, nk):
            raise ValueError(
                f"Expected diagonal overlap shape {(nt, nt, nk)}, got {diagonal.shape} for shift {shift}"
            )
        if not np.all(np.isfinite(diagonal)):
            raise ValueError(f"Diagonal overlap contains non-finite values for shift {shift}")
    for shift in hartree_keys:
        hartree_kernel = float(blocks.hartree_screening[shift])
        if not np.isfinite(hartree_kernel):
            raise ValueError(f"Hartree kernel is non-finite for shift {shift}")
    for shift in fock_keys:
        fock_kernel = np.asarray(blocks.fock_screening[shift])
        if fock_kernel.shape != (nk, nk):
            raise ValueError(
                f"Expected Fock kernel shape {(nk, nk)}, got {fock_kernel.shape} for shift {shift}"
            )
        if not np.all(np.isfinite(fock_kernel)):
            raise ValueError(f"Fock kernel contains non-finite values for shift {shift}")


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return digest


def _update_live_source_bytes(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, byteorder="little", signed=False))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _update_live_source_array(digest: Any, label: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype("<c16")))
    _update_live_source_bytes(
        digest,
        f"{label}.shape",
        np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"),
    )
    _update_live_source_bytes(
        digest,
        f"{label}.values",
        array.tobytes(order="C"),
    )


def _apply_source_interaction_hamiltonian(
    context: TBGZeroFieldTDHFContext,
    tangent: np.ndarray,
) -> np.ndarray:
    return build_projected_interaction_hamiltonian(
        tangent,
        context.run.overlap_blocks,
        v0=float(context.run.state.v0),
        beta=context.beta,
    )


def _tbg_zero_field_tdhf_live_source_sha256(
    run: RestrictedHartreeFockRun,
    *,
    beta: float,
    receipt: TBGZeroFieldHFSourceReceipt,
) -> str:
    state = run.state
    digest = hashlib.sha256()
    _update_live_source_bytes(
        digest,
        "domain",
        b"TBGZeroFieldTDHFLiveSource/v1",
    )
    _update_live_source_array(digest, "h0", state.h0)
    _update_live_source_array(digest, "hamiltonian", state.hamiltonian)
    _update_live_source_array(digest, "density", state.density)
    _update_live_source_bytes(
        digest,
        "v0",
        np.asarray(float(state.v0), dtype=np.dtype("<f8")).tobytes(),
    )
    _update_live_source_bytes(
        digest,
        "beta",
        np.asarray(float(beta), dtype=np.dtype("<f8")).tobytes(),
    )
    _update_live_source_bytes(
        digest,
        "receipt",
        receipt.fingerprint.encode("ascii"),
    )
    _update_live_source_bytes(
        digest,
        "overlap_kernel_inventory",
        tbg_zero_field_overlap_kernel_inventory_fingerprint(
            run.overlap_blocks
        ).encode("ascii"),
    )
    return digest.hexdigest()


def _complex_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.complex128))
    return hashlib.sha256(array.tobytes()).hexdigest()

def _as_square_matrix_field(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must have shape (nt, nt, nk), got {array.shape}")
    return array


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


__all__ = [
    "TBGZeroFieldTDHFContext",
    "TBGZeroFieldTDHFOccupationResiduals",
    "TBGZeroFieldTDHFOrbitals",
    "TBGZeroFieldTDHFProvenance",
    "TBGZeroFieldTDHFTangentParity",
    "build_tbg_zero_field_tdhf_orbitals",
    "build_tbg_zero_field_tdhf_q0_matrices",
    "build_tbg_zero_field_tdhf_q0_matrices_from_pairs",
    "build_tbg_zero_field_tdhf_q0_pairs",
    "conventional_projector_to_stored_density",
    "conventional_tangent_to_stored_tangent",
    "stored_density_to_conventional_projector",
    "stored_tangent_to_conventional_tangent",
    "validate_tbg_zero_field_tdhf_tangent_columns",
]
