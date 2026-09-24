"""Exact fixed-rank zero-temperature Hartree--Fock orbital Hessians.

For an idempotent physical ket projector ``P`` and occupied--virtual coordinate
``X``, the tangent is

``delta P = V X O^dagger + O X^dagger V^dagger``.

At a stationary projector the orbital-gradient Jacobian is

``A[X] = H_vv X - X H_oo + V^dagger delta_H[delta P] O``.

The real energy Hessian is ``2 A`` in the weighted real coordinate whose norm
is ``sum_k w_k ||X_k||_F^2``.  The response callback must include every
interaction normalization already present in the physical HF Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh


@dataclass(frozen=True)
class OrbitalTangentFrame:
    basis: np.ndarray
    occupied_per_k: int
    k_weights: np.ndarray

    @property
    def n(self) -> int:
        return int(self.basis.shape[0])

    @property
    def nk(self) -> int:
        return int(self.basis.shape[2])

    @property
    def nocc(self) -> int:
        return int(self.occupied_per_k)

    @property
    def nvir(self) -> int:
        return int(self.n - self.nocc)

    @property
    def complex_size(self) -> int:
        return int(self.nvir * self.nocc * self.nk)

    @property
    def real_size(self) -> int:
        return int(2 * self.complex_size)

    def unpack_weighted_real(self, vector: np.ndarray) -> np.ndarray:
        """Convert a weighted real vector to physical complex ``X_k``."""

        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.real_size,):
            raise ValueError("orbital tangent vector has the wrong shape")
        weighted = (
            values[: self.complex_size]
            + 1.0j * values[self.complex_size :]
        ).reshape(self.nvir, self.nocc, self.nk)
        return weighted / np.sqrt(self.k_weights)[None, None, :]

    def pack_weighted_complex(
        self, coordinates: np.ndarray, *, factor: float = 1.0
    ) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.complex128)
        expected = (self.nvir, self.nocc, self.nk)
        if values.shape != expected:
            raise ValueError("orbital coordinates have the wrong shape")
        weighted = (
            float(factor)
            * values
            * np.sqrt(self.k_weights)[None, None, :]
        )
        return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

    def tangent_projector(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.complex128)
        expected = (self.nvir, self.nocc, self.nk)
        if values.shape != expected:
            raise ValueError("orbital coordinates have the wrong shape")
        tangent = np.empty_like(self.basis)
        for k_index in range(self.nk):
            occupied = self.basis[:, : self.nocc, k_index]
            virtual = self.basis[:, self.nocc :, k_index]
            xblock = values[:, :, k_index]
            tangent[:, :, k_index] = (
                virtual @ xblock @ occupied.conj().T
                + occupied @ xblock.conj().T @ virtual.conj().T
            )
        return tangent

    def unitary_basis(
        self, weighted_vector: np.ndarray, *, amplitude: float = 1.0
    ) -> np.ndarray:
        """Return the exact block-unitarily rotated occupied/virtual basis."""

        coordinates = self.unpack_weighted_real(weighted_vector)
        rotated_basis = np.empty_like(self.basis)
        for k_index in range(self.nk):
            generator = np.zeros((self.n, self.n), dtype=np.complex128)
            xblock = float(amplitude) * coordinates[:, :, k_index]
            generator[self.nocc :, : self.nocc] = xblock
            generator[: self.nocc, self.nocc :] = -xblock.conj().T
            rotated_basis[:, :, k_index] = (
                self.basis[:, :, k_index] @ expm(generator)
            )
        return rotated_basis

    def unitary_projector(
        self, weighted_vector: np.ndarray, *, amplitude: float = 1.0
    ) -> np.ndarray:
        """Retract a weighted tangent vector by an exact block-unitary map."""

        rotated_basis = self.unitary_basis(
            weighted_vector, amplitude=amplitude
        )
        projector = np.empty_like(self.basis)
        for k_index in range(self.nk):
            occupied = rotated_basis[:, : self.nocc, k_index]
            projector[:, :, k_index] = occupied @ occupied.conj().T
        return projector


@dataclass(frozen=True)
class HessianSelfAdjointnessReport:
    pair_count: int
    maximum_absolute_defect: float
    maximum_relative_defect: float


@dataclass(frozen=True)
class ProjectedHessianProbeReport:
    probe_count: int
    projector_idempotency_max: float
    projector_self_adjoint_max: float
    projector_self_adjoint_relative_max: float
    hessian_self_adjoint_max_ev: float
    hessian_self_adjoint_relative_max: float
    sector_commutator_max_ev: float
    sector_commutator_relative_max: float
    cross_sector_coupling_max_ev: float
    cross_sector_coupling_relative_max: float


@dataclass
class PenalizedProjectedHessianOperator:
    size: int
    hessian_action: Callable[[np.ndarray], np.ndarray]
    projector_action: Callable[[np.ndarray], np.ndarray]
    penalty_ev: float
    matvec_count: int = 0
    operator: LinearOperator = field(init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("projected Hessian size must be positive")
        if not np.isfinite(self.penalty_ev) or self.penalty_ev <= 0.0:
            raise ValueError("penalty_ev must be finite and positive")
        self.operator = LinearOperator(
            (self.size, self.size),
            matvec=self._matvec,
            dtype=np.dtype(np.float64),
        )

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError("projected Hessian vector has the wrong shape")
        projected = np.asarray(self.projector_action(values), dtype=np.float64)
        h_projected = np.asarray(
            self.hessian_action(projected), dtype=np.float64
        )
        projected_h_projected = np.asarray(
            self.projector_action(h_projected), dtype=np.float64
        )
        if any(
            result.shape != values.shape or not np.all(np.isfinite(result))
            for result in (projected, h_projected, projected_h_projected)
        ):
            raise ValueError("projected Hessian callback returned invalid values")
        output = projected_h_projected + self.penalty_ev * (values - projected)
        if not np.all(np.isfinite(output)):
            raise ValueError("penalized projected Hessian action is not finite")
        self.matvec_count += 1
        return output


@dataclass(frozen=True)
class LowestProjectedHessianResult:
    eigenvalues_ev: np.ndarray
    eigenvectors: np.ndarray
    projected_ritz_residuals_ev: np.ndarray
    penalized_ritz_residuals_ev: np.ndarray
    full_ritz_residuals_ev: np.ndarray
    sector_residuals_ev: np.ndarray
    sector_leakage_norms: np.ndarray
    penalty_distances_ev: np.ndarray
    matvec_count: int
    initial_vector_sha256: str


@dataclass(frozen=True)
class ZeroTemperatureOrbitalHessian:
    frame: OrbitalTangentFrame
    hamiltonian: np.ndarray
    hamiltonian_response: Callable[[np.ndarray], np.ndarray]
    stationarity_residual: np.ndarray
    stationarity_rms_ev: float
    stationarity_max_abs_ev: float

    @property
    def shape(self) -> tuple[int, int]:
        size = self.frame.real_size
        return (size, size)

    def gradient_jacobian_action(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the complex real-linear orbital-gradient Jacobian ``A``."""

        values = np.asarray(coordinates, dtype=np.complex128)
        expected = (self.frame.nvir, self.frame.nocc, self.frame.nk)
        if values.shape != expected:
            raise ValueError("orbital coordinates have the wrong shape")
        delta_projector = self.frame.tangent_projector(values)
        delta_hamiltonian = np.asarray(
            self.hamiltonian_response(delta_projector), dtype=np.complex128
        )
        if delta_hamiltonian.shape != self.hamiltonian.shape:
            raise ValueError("Hamiltonian response has the wrong shape")
        if not np.all(np.isfinite(delta_hamiltonian)):
            raise ValueError("Hamiltonian response is not finite")
        hermiticity = float(
            np.max(
                np.abs(
                    delta_hamiltonian
                    - np.swapaxes(delta_hamiltonian.conj(), 0, 1)
                ),
                initial=0.0,
            )
        )
        if hermiticity > 1.0e-9:
            raise ValueError("Hamiltonian response to a Hermitian tangent is not Hermitian")
        output = np.empty_like(values)
        for k_index in range(self.frame.nk):
            basis = self.frame.basis[:, :, k_index]
            rotated_h = basis.conj().T @ self.hamiltonian[:, :, k_index] @ basis
            rotated_delta_h = (
                basis.conj().T @ delta_hamiltonian[:, :, k_index] @ basis
            )
            h_oo = rotated_h[: self.frame.nocc, : self.frame.nocc]
            h_vv = rotated_h[self.frame.nocc :, self.frame.nocc :]
            output[:, :, k_index] = (
                h_vv @ values[:, :, k_index]
                - values[:, :, k_index] @ h_oo
                + rotated_delta_h[self.frame.nocc :, : self.frame.nocc]
            )
        return output

    def energy_hessian_action(self, weighted_vector: np.ndarray) -> np.ndarray:
        coordinates = self.frame.unpack_weighted_real(weighted_vector)
        action = self.gradient_jacobian_action(coordinates)
        return self.frame.pack_weighted_complex(action, factor=2.0)

    def verify_self_adjointness(
        self, probe_vectors: np.ndarray
    ) -> HessianSelfAdjointnessReport:
        probes = np.asarray(probe_vectors, dtype=np.float64)
        if probes.ndim != 2 or probes.shape[1] != self.shape[0]:
            raise ValueError("self-adjointness probes must have shape (nprobe, size)")
        if probes.shape[0] < 2 or not np.all(np.isfinite(probes)):
            raise ValueError("at least two finite self-adjointness probes are required")
        actions = np.asarray(
            [self.energy_hessian_action(vector) for vector in probes],
            dtype=np.float64,
        )
        maximum_absolute = 0.0
        maximum_relative = 0.0
        pair_count = 0
        for left_index in range(probes.shape[0]):
            for right_index in range(left_index + 1, probes.shape[0]):
                lhs = float(np.dot(probes[left_index], actions[right_index]))
                rhs = float(np.dot(actions[left_index], probes[right_index]))
                defect = abs(lhs - rhs)
                scale = max(abs(lhs), abs(rhs), 1.0e-300)
                maximum_absolute = max(maximum_absolute, defect)
                maximum_relative = max(maximum_relative, defect / scale)
                pair_count += 1
        return HessianSelfAdjointnessReport(
            pair_count=pair_count,
            maximum_absolute_defect=maximum_absolute,
            maximum_relative_defect=maximum_relative,
        )

    def as_linear_operator(
        self,
        *,
        verification_vectors: np.ndarray,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> LinearOperator:
        report = self.verify_self_adjointness(verification_vectors)
        if (
            report.maximum_absolute_defect > float(absolute_tolerance)
            and report.maximum_relative_defect > float(relative_tolerance)
        ):
            raise ValueError("orbital Hessian failed real self-adjointness verification")
        return LinearOperator(
            self.shape,
            matvec=self.energy_hessian_action,
            dtype=np.dtype(np.float64),
        )

    def bilinear_symmetry_defect(
        self, left: np.ndarray, right: np.ndarray
    ) -> float:
        lvalue = np.asarray(left, dtype=np.float64)
        rvalue = np.asarray(right, dtype=np.float64)
        lhs = float(np.dot(lvalue, self.energy_hessian_action(rvalue)))
        rhs = float(np.dot(self.energy_hessian_action(lvalue), rvalue))
        return abs(lhs - rhs)


def build_penalized_projected_hessian_operator(
    hessian: ZeroTemperatureOrbitalHessian,
    projector_action: Callable[[np.ndarray], np.ndarray],
    *,
    penalty_ev: float,
) -> PenalizedProjectedHessianOperator:
    """Embed a projected Hessian with a positive complementary penalty."""

    return PenalizedProjectedHessianOperator(
        size=hessian.shape[0],
        hessian_action=hessian.energy_hessian_action,
        projector_action=projector_action,
        penalty_ev=float(penalty_ev),
    )


def probe_projected_hessian_operator(
    hessian: ZeroTemperatureOrbitalHessian,
    projector_action: Callable[[np.ndarray], np.ndarray],
    probe_vectors: np.ndarray,
) -> ProjectedHessianProbeReport:
    """Probe projector geometry, Hessian symmetry, and sector invariance."""

    probes = np.asarray(probe_vectors, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != hessian.shape[0]:
        raise ValueError("projected-Hessian probes have the wrong shape")
    if probes.shape[0] < 2 or not np.all(np.isfinite(probes)):
        raise ValueError("at least two finite projected-Hessian probes are required")
    projected = np.asarray(
        [projector_action(vector) for vector in probes], dtype=np.float64
    )
    projected_twice = np.asarray(
        [projector_action(vector) for vector in projected], dtype=np.float64
    )
    hessian_actions = np.asarray(
        [hessian.energy_hessian_action(vector) for vector in probes],
        dtype=np.float64,
    )
    hessian_projected = np.asarray(
        [hessian.energy_hessian_action(vector) for vector in projected],
        dtype=np.float64,
    )
    projected_hessian = np.asarray(
        [projector_action(vector) for vector in hessian_actions],
        dtype=np.float64,
    )
    if not all(
        values.shape == probes.shape and np.all(np.isfinite(values))
        for values in (
            projected,
            projected_twice,
            hessian_actions,
            hessian_projected,
            projected_hessian,
        )
    ):
        raise ValueError("projected-Hessian callback returned invalid probe values")

    projector_idempotency = max(
        float(np.linalg.norm(projected_twice[index] - projected[index]))
        for index in range(probes.shape[0])
    )
    commutator_absolute = 0.0
    commutator_relative = 0.0
    cross_absolute = 0.0
    cross_relative = 0.0
    for index in range(probes.shape[0]):
        defect = hessian_projected[index] - projected_hessian[index]
        absolute = float(np.linalg.norm(defect))
        scale = max(
            float(np.linalg.norm(hessian_projected[index])),
            float(np.linalg.norm(projected_hessian[index])),
            1.0e-300,
        )
        commutator_absolute = max(commutator_absolute, absolute)
        commutator_relative = max(commutator_relative, absolute / scale)
        complement = probes[index] - projected[index]
        h_complement = hessian.energy_hessian_action(complement)
        projected_h_complement = np.asarray(
            projector_action(h_complement), dtype=np.float64
        )
        coupling = float(np.linalg.norm(projected_h_complement))
        coupling_scale = max(float(np.linalg.norm(h_complement)), 1.0e-300)
        cross_absolute = max(cross_absolute, coupling)
        cross_relative = max(cross_relative, coupling / coupling_scale)

    projector_self_absolute = 0.0
    projector_self_relative = 0.0
    hessian_self_absolute = 0.0
    hessian_self_relative = 0.0
    for left in range(probes.shape[0]):
        for right in range(left + 1, probes.shape[0]):
            q_lhs = float(np.dot(probes[left], projected[right]))
            q_rhs = float(np.dot(projected[left], probes[right]))
            q_defect = abs(q_lhs - q_rhs)
            q_scale = max(abs(q_lhs), abs(q_rhs), 1.0e-300)
            projector_self_absolute = max(projector_self_absolute, q_defect)
            projector_self_relative = max(
                projector_self_relative, q_defect / q_scale
            )
            h_lhs = float(np.dot(probes[left], hessian_actions[right]))
            h_rhs = float(np.dot(hessian_actions[left], probes[right]))
            h_defect = abs(h_lhs - h_rhs)
            h_scale = max(abs(h_lhs), abs(h_rhs), 1.0e-300)
            hessian_self_absolute = max(hessian_self_absolute, h_defect)
            hessian_self_relative = max(hessian_self_relative, h_defect / h_scale)

    return ProjectedHessianProbeReport(
        probe_count=int(probes.shape[0]),
        projector_idempotency_max=projector_idempotency,
        projector_self_adjoint_max=projector_self_absolute,
        projector_self_adjoint_relative_max=projector_self_relative,
        hessian_self_adjoint_max_ev=hessian_self_absolute,
        hessian_self_adjoint_relative_max=hessian_self_relative,
        sector_commutator_max_ev=commutator_absolute,
        sector_commutator_relative_max=commutator_relative,
        cross_sector_coupling_max_ev=cross_absolute,
        cross_sector_coupling_relative_max=cross_relative,
    )


def solve_lowest_projected_hessian_eigenpairs(
    penalized: PenalizedProjectedHessianOperator,
    *,
    count: int,
    seed: int,
    ncv: int,
    tolerance: float,
    max_iterations: int,
) -> LowestProjectedHessianResult:
    """Solve the lowest projected Hessian modes with deterministic ARPACK."""

    size = int(penalized.size)
    mode_count = int(count)
    if not 0 < mode_count < size:
        raise ValueError("count must lie strictly between zero and operator size")
    ncv_value = int(ncv)
    if not mode_count < ncv_value <= size:
        raise ValueError("ncv must be greater than count and no larger than size")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    rng = np.random.default_rng(int(seed))
    initial = np.asarray(
        penalized.projector_action(rng.normal(size=size)), dtype=np.float64
    )
    initial_norm = float(np.linalg.norm(initial))
    if not np.isfinite(initial_norm) or initial_norm <= 1.0e-14:
        raise ValueError("projected eigensolver initial vector is numerically zero")
    initial /= initial_norm
    initial_sha256 = hashlib.sha256(initial.tobytes(order="C")).hexdigest()
    starting_matvec_count = penalized.matvec_count
    try:
        eigenvalues, eigenvectors = eigsh(
            penalized.operator,
            k=mode_count,
            which="SA",
            v0=initial,
            ncv=ncv_value,
            tol=float(tolerance),
            maxiter=int(max_iterations),
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as error:
        raise RuntimeError("projected Hessian ARPACK solve did not converge") from error
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    for index in range(mode_count):
        pivot = int(np.argmax(np.abs(eigenvectors[:, index])))
        if eigenvectors[pivot, index] < 0.0:
            eigenvectors[:, index] *= -1.0

    projected_ritz = np.empty(mode_count, dtype=np.float64)
    penalized_ritz = np.empty(mode_count, dtype=np.float64)
    full_ritz = np.empty(mode_count, dtype=np.float64)
    sector_residuals = np.empty(mode_count, dtype=np.float64)
    sector_leakage = np.empty(mode_count, dtype=np.float64)
    for index in range(mode_count):
        vector = eigenvectors[:, index]
        projected_vector = np.asarray(
            penalized.projector_action(vector), dtype=np.float64
        )
        h_projected = np.asarray(
            penalized.hessian_action(projected_vector), dtype=np.float64
        )
        projected_h_projected = np.asarray(
            penalized.projector_action(h_projected), dtype=np.float64
        )
        h_full = np.asarray(penalized.hessian_action(vector), dtype=np.float64)
        projected_ritz[index] = np.linalg.norm(
            projected_h_projected - eigenvalues[index] * projected_vector
        )
        penalized_action = (
            projected_h_projected
            + penalized.penalty_ev * (vector - projected_vector)
        )
        penalized_ritz[index] = np.linalg.norm(
            penalized_action - eigenvalues[index] * vector
        )
        full_ritz[index] = np.linalg.norm(
            h_full - eigenvalues[index] * vector
        )
        sector_residuals[index] = np.linalg.norm(
            h_full - penalized.projector_action(h_full)
        )
        sector_leakage[index] = np.linalg.norm(vector - projected_vector)

    return LowestProjectedHessianResult(
        eigenvalues_ev=eigenvalues,
        eigenvectors=eigenvectors,
        projected_ritz_residuals_ev=projected_ritz,
        penalized_ritz_residuals_ev=penalized_ritz,
        full_ritz_residuals_ev=full_ritz,
        sector_residuals_ev=sector_residuals,
        sector_leakage_norms=sector_leakage,
        penalty_distances_ev=np.abs(penalized.penalty_ev - eigenvalues),
        matvec_count=int(penalized.matvec_count - starting_matvec_count),
        initial_vector_sha256=initial_sha256,
    )


def build_zero_temperature_orbital_hessian(
    physical_projector: np.ndarray,
    hamiltonian: np.ndarray,
    hamiltonian_response: Callable[[np.ndarray], np.ndarray],
    *,
    occupied_per_k: int,
    k_weights: np.ndarray | None = None,
    projector_tolerance: float = 1.0e-8,
    stationarity_tolerance_ev: float | None = None,
    tangent_basis: np.ndarray | None = None,
) -> ZeroTemperatureOrbitalHessian:
    """Build an exact matrix-free fixed-rank zero-temperature HF Hessian."""

    projector = np.asarray(physical_projector, dtype=np.complex128)
    hvalue = np.asarray(hamiltonian, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[0] != projector.shape[1]:
        raise ValueError("physical_projector must have shape (n,n,nk)")
    if hvalue.shape != projector.shape:
        raise ValueError("Hamiltonian and projector shapes differ")
    if not np.all(np.isfinite(projector)) or not np.all(np.isfinite(hvalue)):
        raise ValueError("projector and Hamiltonian must be finite")
    n, _, nk = projector.shape
    nocc = int(occupied_per_k)
    if not 0 < nocc < n:
        raise ValueError("occupied_per_k must lie strictly between zero and n")
    weights = (
        np.full(nk, 1.0 / nk, dtype=np.float64)
        if k_weights is None
        else np.asarray(k_weights, dtype=np.float64)
    )
    if weights.shape != (nk,) or np.any(weights <= 0.0):
        raise ValueError("k_weights must be positive with shape (nk,)")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_weights must sum to one")

    supplied_basis = (
        None
        if tangent_basis is None
        else np.asarray(tangent_basis, dtype=np.complex128)
    )
    if supplied_basis is not None and (
        supplied_basis.shape != projector.shape
        or not np.all(np.isfinite(supplied_basis))
    ):
        raise ValueError("tangent_basis must be finite with the projector shape")
    basis = np.empty_like(projector)
    residual = np.empty((n - nocc, nocc, nk), dtype=np.complex128)
    for k_index in range(nk):
        pblock = projector[:, :, k_index]
        hblock = hvalue[:, :, k_index]
        if np.linalg.norm(pblock - pblock.conj().T) > projector_tolerance:
            raise ValueError("physical projector is not Hermitian")
        if np.linalg.norm(pblock @ pblock - pblock) > projector_tolerance:
            raise ValueError("physical projector is not idempotent")
        if np.linalg.norm(hblock - hblock.conj().T) > 1.0e-9:
            raise ValueError("Hamiltonian is not Hermitian")
        if supplied_basis is None:
            eigenvalues, eigenvectors = np.linalg.eigh(pblock)
            occupied_indices = np.arange(n - nocc, n, dtype=np.int64)
            virtual_indices = np.arange(0, n - nocc, dtype=np.int64)
            if eigenvalues[virtual_indices[-1]] > projector_tolerance:
                raise ValueError("projector virtual eigenvalue is too large")
            if eigenvalues[occupied_indices[0]] < 1.0 - projector_tolerance:
                raise ValueError("projector occupied eigenvalue is too small")
            order = np.concatenate([occupied_indices, virtual_indices])
            block_basis = eigenvectors[:, order]
        else:
            block_basis = supplied_basis[:, :, k_index]
            if np.linalg.norm(block_basis.conj().T @ block_basis - np.eye(n)) > 1.0e-8:
                raise ValueError("supplied tangent basis is not unitary")
            occupied = block_basis[:, :nocc]
            if np.linalg.norm(occupied @ occupied.conj().T - pblock) > projector_tolerance:
                raise ValueError("supplied tangent basis does not represent the projector")
        basis[:, :, k_index] = block_basis
        rotated_h = block_basis.conj().T @ hblock @ block_basis
        residual[:, :, k_index] = rotated_h[nocc:, :nocc]

    stationarity_max = float(np.max(np.abs(residual), initial=0.0))
    if stationarity_tolerance_ev is not None:
        tolerance = float(stationarity_tolerance_ev)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("stationarity_tolerance_ev must be finite and nonnegative")
        if stationarity_max > tolerance:
            raise ValueError(
                "projector is not stationary enough for an orbital Hessian: "
                f"{stationarity_max:.6e} > {tolerance:.6e} eV"
            )
    frame = OrbitalTangentFrame(
        basis=basis,
        occupied_per_k=nocc,
        k_weights=weights,
    )
    return ZeroTemperatureOrbitalHessian(
        frame=frame,
        hamiltonian=hvalue,
        hamiltonian_response=hamiltonian_response,
        stationarity_residual=residual,
        stationarity_rms_ev=float(np.linalg.norm(residual) / np.sqrt(residual.size)),
        stationarity_max_abs_ev=stationarity_max,
    )


__all__ = [
    "HessianSelfAdjointnessReport",
    "LowestProjectedHessianResult",
    "OrbitalTangentFrame",
    "PenalizedProjectedHessianOperator",
    "ProjectedHessianProbeReport",
    "ZeroTemperatureOrbitalHessian",
    "build_penalized_projected_hessian_operator",
    "build_zero_temperature_orbital_hessian",
    "probe_projected_hessian_operator",
    "solve_lowest_projected_hessian_eigenpairs",
]
