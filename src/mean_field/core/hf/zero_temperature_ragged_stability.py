"""Fixed-per-block-rank, block-preserving orbital-Hessian candidates.

This API retains only orbital rotations at fixed occupied rank in every
block.  It cannot test inter-block occupation transfers or zero-temperature
Aufbau ordering.  A separate occupation-gap gate covering all admissible
occupied/virtual transfers is therefore required before *any* stability claim.
In particular, an all-full/all-empty input has zero retained directions; its
zero-dimensional candidate says nothing about stability.

Inputs use the conventional physical layout ``(n, n, nblock)``.  In block
``b``, ``U_b = [O_b, V_b]`` and only virtual-by-occupied coordinates ``X_b``
are retained.  Their Hermitian projector tangent and candidate action are

``D_b[X] = V_b X_b O_b^dagger + O_b X_b^dagger V_b^dagger``,
``J_b[X] = F_vv,b X_b - X_b F_oo,b + V_b^dagger dF[D[X]]_b O_b``.

Crucially, ``hamiltonians`` supplies the full source Fock matrices
``F[P0]``, including the source mean-field field; it is not the bare ``h0``.
For raw positive weights ``w_b``, the packed real candidate operator returns
``2 w_b J_b``.  Euclidean-self-adjoint status, and therefore eligibility for
a Hermitian eigensolver, requires separate same-functional authority:

``dE[P](D) = sum_b w_b Tr(F_b[P] D_b)``

and the exact weighted differential reciprocity identity

``sum_b w_b Tr(D_b dF[P0][Dtilde]_b)``
``    = sum_b w_b Tr(Dtilde_b dF[P0][D]_b)``

for every pair of Hermitian tangents ``D, Dtilde``.  Shape, finiteness, and
Hermiticity checks do not establish this identity, so the exposed
``LinearOperator`` deliberately has no ``rmatvec``.  Deterministic bilinear-
symmetry samples and a single-step five-point stencil are diagnostics, not
authority.  No finite sample can prove self-adjointness.

Weights are never normalized here.  The repository's raw-unweighted ABI is
the special case ``w_b = 1`` for every block; all other total/per-cell/per-k or
quadrature normalization remains caller-owned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

Array = np.ndarray
FockDerivative = Callable[[Array], Array]
EnergyCallback = Callable[[Array], complex | float]


def _max_abs(value: Array) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _frobenius_norm(value: Array) -> float:
    return float(np.linalg.norm(value, ord="fro")) if value.size else 0.0


def _dimensionless_tolerance(atol: float, rtol: float, *values: Array) -> float:
    scale = max(1.0, *(_max_abs(value) for value in values))
    return atol + rtol * scale


def _dimensional_tolerance(atol: float, rtol: float, *values: Array) -> float:
    return atol + rtol * max((_max_abs(value) for value in values), default=0.0)


def _remove_real_scalar_shift(matrix: Array) -> Array:
    """Remove the real identity component without hiding anti-Hermiticity."""

    mean_real_diagonal = float(np.trace(matrix).real) / matrix.shape[0]
    return matrix - mean_real_diagonal * np.eye(matrix.shape[0], dtype=matrix.dtype)


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class RaggedBlockLayout:
    """Deterministic complex-coordinate slice for one orbital block."""

    block: int
    occupied_count: int
    virtual_count: int
    start: int
    stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.virtual_count, self.occupied_count)

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class BilinearSymmetryProbe:
    """One sampled pair in a deterministic bilinear-symmetry diagnostic."""

    index: int
    left_right: float
    right_left: float
    residual: float
    tolerance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class BilinearSymmetryDiagnostic:
    """Finite sampled-pair diagnostic, never self-adjointness authority.

    ``inconclusive`` is true when no pair was evaluated, including both a
    positive-dimensional zero-probe request and a zero-dimensional
    full/empty layout.  There is deliberately no vacuously true ``passed``
    property.
    """

    seed: int
    requested_probe_count: int
    retained_real_dimension: int
    probes: tuple[BilinearSymmetryProbe, ...]

    @property
    def inconclusive(self) -> bool:
        return not self.probes

    @property
    def all_evaluated_pairs_symmetric(self) -> bool:
        return bool(self.probes) and all(probe.passed for probe in self.probes)

    @property
    def asymmetry_detected(self) -> bool:
        return any(not probe.passed for probe in self.probes)

    @property
    def outcome(self) -> str:
        if self.inconclusive:
            return "inconclusive"
        if self.asymmetry_detected:
            return "sampled_bilinear_asymmetry_detected"
        return "sampled_pairs_symmetric_not_proof"

    @property
    def maximum_residual(self) -> float:
        return max((probe.residual for probe in self.probes), default=0.0)


@dataclass(frozen=True, slots=True)
class FivePointCurvatureCheck:
    """Diagnostic-only exact-unitary single-step five-point comparison."""

    step: float
    input_direction_norm: float
    evaluated_direction_norm: float
    diagnostic_only: bool
    energies_minus_2h_to_plus_2h: tuple[float, float, float, float, float]
    stationarity_derivative: float
    predicted_curvature: float
    finite_difference_curvature: float
    curvature_residual: float
    curvature_tolerance: float
    stationarity_tolerance: float
    passed: bool


class ZeroTemperatureRaggedOrbitalHessian:
    """Fixed-per-block-rank, block-preserving matrix-free candidate action.

    This object never tests inter-block occupation transfers or Aufbau
    ordering.  A separate occupation-gap gate is mandatory before any
    stability claim.  Validation here also does not prove that the action is a
    scalar-functional Hessian.  Hermitian eigensolvers require external
    authority for the exact weighted same-functional reciprocity identity
    stated in the module docs.

    Parameters
    ----------
    projectors, hamiltonians, orbital_basis
        Complex arrays with conventional physical shape ``(n, n, nblock)``.
        ``hamiltonians`` must be the full source ``F[P0]``, not bare ``h0``.
        ``orbital_basis[:, :, b]`` must be unitary and have occupied columns
        first.  The projector must equal ``O_b O_b^dagger``.
    occupied_counts
        Per-block occupied ranks, each in the closed interval ``[0, n]``.
    fock_derivative
        Real-linear callback mapping an exact Hermitian tangent ``D`` in the
        same physical layout to ``dF[D]``.  Its output is checked for shape,
        finiteness, and Hermiticity on every action.
    block_weights
        Positive raw metric weights.  They multiply the real candidate rows
        and are deliberately not normalized or required to sum to one.  The
        default all-one weights are the raw-unweighted repository ABI.
    validation_atol, validation_rtol
        Dimensionless tolerances used only for projector/basis geometry.
    hamiltonian_atol, hamiltonian_rtol
        Hamiltonian-unit tolerances for Hermiticity of ``F[P0]`` and ``dF``.
    stationarity_atol, stationarity_rtol
        Hamiltonian-unit absolute and dimensionless relative tolerances for
        ``V^dagger F[P0] O``.  Relative scaling uses the traceless-block scale,
        so adding a real ``c I`` changes neither the residual nor tolerance.
    """

    scope = "fixed-per-block-rank, block-preserving candidate"
    tests_inter_block_occupation_transfers = False
    tests_aufbau_ordering = False
    requires_separate_occupation_gap_gate = True

    def __init__(
        self,
        projectors: Array,
        hamiltonians: Array,
        orbital_basis: Array,
        occupied_counts: Sequence[int],
        fock_derivative: FockDerivative,
        *,
        block_weights: Sequence[float] | None = None,
        validation_atol: float = 2.0e-10,
        validation_rtol: float = 2.0e-12,
        hamiltonian_atol: float = 2.0e-10,
        hamiltonian_rtol: float = 2.0e-12,
        stationarity_atol: float = 2.0e-10,
        stationarity_rtol: float = 2.0e-12,
    ) -> None:
        atol = _finite_nonnegative("validation_atol", validation_atol)
        rtol = _finite_nonnegative("validation_rtol", validation_rtol)
        hatol = _finite_nonnegative("hamiltonian_atol", hamiltonian_atol)
        hrtol = _finite_nonnegative("hamiltonian_rtol", hamiltonian_rtol)
        satol = _finite_nonnegative("stationarity_atol", stationarity_atol)
        srtol = _finite_nonnegative("stationarity_rtol", stationarity_rtol)
        if not callable(fock_derivative):
            raise TypeError("fock_derivative must be callable")

        projector_array = self._source_array(projectors, "projectors")
        hamiltonian_array = self._source_array(hamiltonians, "hamiltonians")
        basis_array = self._source_array(orbital_basis, "orbital_basis")
        if projector_array.ndim != 3 or projector_array.shape[0] != projector_array.shape[1]:
            raise ValueError("projectors must have shape (n, n, nblock)")
        if hamiltonian_array.shape != projector_array.shape:
            raise ValueError("hamiltonians must have the same shape as projectors")
        if basis_array.shape != projector_array.shape:
            raise ValueError("orbital_basis must have the same shape as projectors")
        n, _, nblock = projector_array.shape
        if n == 0 or nblock == 0:
            raise ValueError("n and nblock must both be positive")

        counts = self._occupied_counts(occupied_counts, n=n, nblock=nblock)
        if block_weights is None:
            weights = np.ones(nblock, dtype=np.float64)
        else:
            weights = np.asarray(block_weights, dtype=np.float64)
            if weights.shape != (nblock,):
                raise ValueError(f"block_weights must have shape ({nblock},)")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("block_weights must be finite and strictly positive")
            weights = np.array(weights, copy=True)
        weights.setflags(write=False)

        layouts: list[RaggedBlockLayout] = []
        offset = 0
        for block, occupied in enumerate(counts):
            virtual = n - occupied
            stop = offset + virtual * occupied
            layouts.append(
                RaggedBlockLayout(block, occupied, virtual, offset, stop)
            )
            offset = stop

        self.projectors = projector_array
        self.hamiltonians = hamiltonian_array
        self.orbital_basis = basis_array
        self.occupied_counts = counts
        self.block_weights = weights
        self.layouts = tuple(layouts)
        self.complex_dimension = offset
        self.real_dimension = 2 * offset
        self.n = n
        self.nblock = nblock
        self.fock_derivative = fock_derivative
        self.validation_atol = atol
        self.validation_rtol = rtol
        self.hamiltonian_atol = hatol
        self.hamiltonian_rtol = hrtol
        self.stationarity_atol = satol
        self.stationarity_rtol = srtol

        offsets = np.asarray([layout.start for layout in layouts] + [offset], dtype=np.int64)
        offsets.setflags(write=False)
        self.complex_offsets = offsets
        (
            self.stationarity_residuals,
            self.stationarity_tolerances,
        ) = self._validate_source()
        self._candidate_linear_operator = LinearOperator(
            shape=(self.real_dimension, self.real_dimension),
            matvec=self.matvec,
            dtype=np.dtype(np.float64),
        )

    @staticmethod
    def _source_array(value: Any, name: str) -> Array:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be numeric")
        result = np.array(array, dtype=np.complex128, copy=True, order="C")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        result.setflags(write=False)
        return result

    @staticmethod
    def _occupied_counts(
        values: Sequence[int], *, n: int, nblock: int
    ) -> tuple[int, ...]:
        if len(values) != nblock:
            raise ValueError(f"occupied_counts must have length {nblock}")
        counts: list[int] = []
        for block, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"occupied_counts[{block}] must be an integer")
            count = int(value)
            if count < 0 or count > n:
                raise ValueError(f"occupied_counts[{block}] must lie in [0, {n}]")
            counts.append(count)
        return tuple(counts)

    def _validate_source(self) -> tuple[Array, Array]:
        identity = np.eye(self.n, dtype=np.complex128)
        residuals = np.zeros(self.nblock, dtype=np.float64)
        stationarity_tolerances = np.zeros(self.nblock, dtype=np.float64)
        for layout in self.layouts:
            block = layout.block
            projector = self.projectors[:, :, block]
            hamiltonian = self.hamiltonians[:, :, block]
            basis = self.orbital_basis[:, :, block]
            occupied = basis[:, : layout.occupied_count]
            expected_projector = occupied @ occupied.conj().T

            geometry_tolerance = _dimensionless_tolerance(
                self.validation_atol, self.validation_rtol, projector, basis
            )
            if _max_abs(projector - projector.conj().T) > geometry_tolerance:
                raise ValueError(f"projectors[:, :, {block}] is not Hermitian")
            if _max_abs(projector @ projector - projector) > geometry_tolerance:
                raise ValueError(f"projectors[:, :, {block}] is not idempotent")
            if (
                abs(float(np.trace(projector).real) - layout.occupied_count)
                > geometry_tolerance
            ):
                raise ValueError(f"projectors[:, :, {block}] has the wrong rank/trace")
            if abs(float(np.trace(projector).imag)) > geometry_tolerance:
                raise ValueError(f"projectors[:, :, {block}] has a non-real trace")
            if _max_abs(basis.conj().T @ basis - identity) > geometry_tolerance:
                raise ValueError(f"orbital_basis[:, :, {block}] is not unitary")
            if _max_abs(projector - expected_projector) > geometry_tolerance:
                raise ValueError(
                    f"projector/basis occupied-column mismatch in block {block}"
                )

            shift_free_hamiltonian = _remove_real_scalar_shift(hamiltonian)
            hamiltonian_tolerance = _dimensional_tolerance(
                self.hamiltonian_atol,
                self.hamiltonian_rtol,
                shift_free_hamiltonian,
            )
            if (
                _max_abs(hamiltonian - hamiltonian.conj().T)
                > hamiltonian_tolerance
            ):
                raise ValueError(f"hamiltonians[:, :, {block}] is not Hermitian")

            virtual = basis[:, layout.occupied_count :]
            gradient = virtual.conj().T @ shift_free_hamiltonian @ occupied
            residual = _frobenius_norm(gradient)
            residuals[block] = residual
            stationarity_tolerance = (
                self.stationarity_atol
                + self.stationarity_rtol
                * _frobenius_norm(shift_free_hamiltonian)
            )
            stationarity_tolerances[block] = stationarity_tolerance
            if residual > stationarity_tolerance:
                raise ValueError(
                    f"source is not orbital-stationary in block {block}: "
                    f"residual {residual:.6e} exceeds {stationarity_tolerance:.6e}"
                )
        residuals.setflags(write=False)
        stationarity_tolerances.setflags(write=False)
        return residuals, stationarity_tolerances

    @property
    def candidate_linear_operator(self) -> LinearOperator:
        """Real candidate action with no declared adjoint implementation.

        Passing this operator to a Hermitian eigensolver requires separate
        same-functional weighted-reciprocity authority.  Probe success alone
        is not such authority.
        """

        return self._candidate_linear_operator

    @property
    def linear_operator(self) -> LinearOperator:
        """Compatibility alias for :attr:`candidate_linear_operator`."""

        return self._candidate_linear_operator

    @property
    def operator(self) -> LinearOperator:
        """Compatibility alias for :attr:`candidate_linear_operator`."""

        return self._candidate_linear_operator

    def pack_complex(self, blocks: Sequence[Array]) -> Array:
        """Pack ragged ``(virtual, occupied)`` blocks in block/C order."""

        if len(blocks) != self.nblock:
            raise ValueError(f"expected {self.nblock} coordinate blocks")
        packed = np.empty(self.complex_dimension, dtype=np.complex128)
        for layout, value in zip(self.layouts, blocks):
            array = np.asarray(value, dtype=np.complex128)
            if array.shape != layout.shape:
                raise ValueError(
                    f"coordinate block {layout.block} must have shape {layout.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"coordinate block {layout.block} must be finite")
            packed[layout.start : layout.stop] = array.reshape(-1, order="C")
        return packed

    def unpack_complex(self, coordinates: Array) -> tuple[Array, ...]:
        """Unpack one complex coordinate vector into ragged block matrices."""

        packed = np.asarray(coordinates, dtype=np.complex128)
        if packed.shape != (self.complex_dimension,):
            raise ValueError(
                f"complex coordinates must have shape ({self.complex_dimension},)"
            )
        if not np.all(np.isfinite(packed)):
            raise ValueError("complex coordinates must be finite")
        return tuple(
            np.array(
                packed[layout.start : layout.stop].reshape(layout.shape, order="C"),
                copy=True,
            )
            for layout in self.layouts
        )

    def pack_real(self, blocks: Sequence[Array]) -> Array:
        """Pack as ``(all Re X, all Im X)`` without metric rescaling."""

        packed = self.pack_complex(blocks)
        return np.concatenate((packed.real, packed.imag))

    def unpack_real(self, coordinates: Array) -> tuple[Array, ...]:
        """Inverse of :meth:`pack_real`."""

        array = np.asarray(coordinates)
        if np.iscomplexobj(array):
            raise TypeError("real candidate coordinates must have a real dtype")
        array = np.asarray(array, dtype=np.float64)
        if array.shape != (self.real_dimension,):
            raise ValueError(f"real coordinates must have shape ({self.real_dimension},)")
        if not np.all(np.isfinite(array)):
            raise ValueError("real coordinates must be finite")
        complex_coordinates = (
            array[: self.complex_dimension]
            + 1j * array[self.complex_dimension :]
        )
        return self.unpack_complex(complex_coordinates)

    def tangent(self, coordinates: Array) -> Array:
        """Return the exact Hermitian first-order physical tangent ``D=X+X†``."""

        blocks = self.unpack_real(coordinates)
        tangent = np.zeros_like(self.projectors)
        for layout, x_block in zip(self.layouts, blocks):
            block = layout.block
            basis = self.orbital_basis[:, :, block]
            occupied = basis[:, : layout.occupied_count]
            virtual = basis[:, layout.occupied_count :]
            lowering = virtual @ x_block @ occupied.conj().T
            tangent[:, :, block] = lowering + lowering.conj().T
        return tangent

    def complex_action(self, coordinates: Array) -> tuple[Array, ...]:
        """Apply the unweighted complex action ``J[X]`` block by block."""

        blocks = self.unpack_real(coordinates)
        tangent = self.tangent(coordinates)
        response = np.asarray(self.fock_derivative(tangent), dtype=np.complex128)
        if response.shape != self.projectors.shape:
            raise ValueError(
                "fock_derivative output must have shape "
                f"{self.projectors.shape}, got {response.shape}"
            )
        if not np.all(np.isfinite(response)):
            raise ValueError("fock_derivative output must be finite")
        for block in range(self.nblock):
            response_block = response[:, :, block]
            tolerance = _dimensional_tolerance(
                self.hamiltonian_atol,
                self.hamiltonian_rtol,
                _remove_real_scalar_shift(response_block),
            )
            if _max_abs(response_block - response_block.conj().T) > tolerance:
                raise ValueError(
                    f"fock_derivative output block {block} is not Hermitian"
                )

        actions: list[Array] = []
        for layout, x_block in zip(self.layouts, blocks):
            block = layout.block
            basis = self.orbital_basis[:, :, block]
            occupied = basis[:, : layout.occupied_count]
            virtual = basis[:, layout.occupied_count :]
            hamiltonian = _remove_real_scalar_shift(
                self.hamiltonians[:, :, block]
            )
            h_oo = occupied.conj().T @ hamiltonian @ occupied
            h_vv = virtual.conj().T @ hamiltonian @ virtual
            response_vo = virtual.conj().T @ response[:, :, block] @ occupied
            actions.append(h_vv @ x_block - x_block @ h_oo + response_vo)
        return tuple(actions)

    def matvec(self, coordinates: Array) -> Array:
        """Apply the factor-two weighted fixed-rank candidate action."""

        actions = self.complex_action(coordinates)
        weighted = tuple(
            2.0 * self.block_weights[layout.block] * action
            for layout, action in zip(self.layouts, actions)
        )
        result = self.pack_real(weighted)
        # LinearOperator matvecs should always return a fresh contiguous vector.
        return np.asarray(result, dtype=np.float64, order="C")

    def retract(self, coordinates: Array, step: float) -> Array:
        """Return projectors on the exact block-unitary path at ``step``.

        In the supplied orbital basis the anti-Hermitian generator is
        ``K = [[0, -X†], [X, 0]]``.  Therefore ``P(step)`` is Hermitian,
        idempotent, and fixed-rank up to matrix-exponential roundoff, without
        a linearized projector approximation.
        """

        if isinstance(step, (bool, np.bool_)):
            raise TypeError("step must be a finite real scalar")
        step_value = float(step)
        if not np.isfinite(step_value):
            raise ValueError("step must be finite")
        blocks = self.unpack_real(coordinates)
        result = np.empty_like(self.projectors)
        for layout, x_block in zip(self.layouts, blocks):
            basis = self.orbital_basis[:, :, layout.block]
            generator = np.zeros((self.n, self.n), dtype=np.complex128)
            occupied = layout.occupied_count
            generator[occupied:, :occupied] = x_block
            generator[:occupied, occupied:] = -x_block.conj().T
            rotated_basis = basis @ expm(step_value * generator)
            rotated_occupied = rotated_basis[:, :occupied]
            result[:, :, layout.block] = rotated_occupied @ rotated_occupied.conj().T
        return result

    def diagnose_bilinear_symmetry(
        self,
        *,
        seed: int = 0,
        probe_count: int = 4,
        atol: float = 2.0e-10,
        rtol: float = 2.0e-10,
    ) -> BilinearSymmetryDiagnostic:
        """Sample ``x.T H y == y.T H x`` on normalized random pairs.

        This is only a bilinear-symmetry diagnostic for the fixed-per-block-
        rank, block-preserving candidate.  It tests neither exact reciprocity
        nor inter-block occupation/Aufbau stability.  Zero evaluated pairs are
        explicitly inconclusive; finite success is sampled evidence, not
        proof and not Hermitian-eigensolver authority.  ``atol`` has the same
        energy unit as the candidate action; ``rtol`` is dimensionless.
        """

        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if (
            isinstance(probe_count, (bool, np.bool_))
            or not isinstance(probe_count, (int, np.integer))
            or probe_count < 0
        ):
            raise ValueError("probe_count must be a nonnegative integer")
        absolute = _finite_nonnegative("atol", atol)
        relative = _finite_nonnegative("rtol", rtol)
        if self.real_dimension == 0 or probe_count == 0:
            return BilinearSymmetryDiagnostic(
                int(seed), int(probe_count), self.real_dimension, tuple()
            )

        rng = np.random.default_rng(int(seed))
        evidence: list[BilinearSymmetryProbe] = []
        for index in range(int(probe_count)):
            left = rng.standard_normal(self.real_dimension)
            right = rng.standard_normal(self.real_dimension)
            left /= np.linalg.norm(left)
            right /= np.linalg.norm(right)
            left_right = float(left @ self.matvec(right))
            right_left = float(right @ self.matvec(left))
            residual = abs(left_right - right_left)
            tolerance = absolute + relative * max(
                abs(left_right), abs(right_left)
            )
            evidence.append(
                BilinearSymmetryProbe(
                    index=index,
                    left_right=left_right,
                    right_left=right_left,
                    residual=residual,
                    tolerance=tolerance,
                    passed=residual <= tolerance,
                )
            )
        return BilinearSymmetryDiagnostic(
            int(seed), int(probe_count), self.real_dimension, tuple(evidence)
        )

    def check_five_point_curvature(
        self,
        coordinates: Array,
        energy: EnergyCallback,
        *,
        step: float,
        curvature_atol: float = 2.0e-8,
        curvature_rtol: float = 2.0e-7,
        stationarity_atol: float = 2.0e-8,
        raise_on_failure: bool = True,
    ) -> FivePointCurvatureCheck:
        """Run one diagnostic-only exact-unitary five-point comparison.

        The input direction is normalized internally, so rescaling a nonzero
        vector cannot make the check vacuous.  ``step`` is measured along that
        unit-norm direction, and both input and evaluated norms are reported.
        A single step does not certify finite-difference convergence or
        same-functional reciprocity.  ``curvature_atol`` and
        ``stationarity_atol`` have the energy unit returned by ``energy`` and
        used by the candidate action; both must be rescaled when that unit is
        rescaled.  Relative tolerances are dimensionless.
        """

        if not callable(energy):
            raise TypeError("energy must be callable")
        direction = np.asarray(coordinates)
        if np.iscomplexobj(direction):
            raise TypeError("curvature coordinates must have a real dtype")
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (self.real_dimension,):
            raise ValueError(
                f"curvature coordinates must have shape ({self.real_dimension},)"
            )
        if not np.all(np.isfinite(direction)):
            raise ValueError("curvature coordinates must be finite")
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction_norm) or direction_norm == 0.0:
            raise ValueError("curvature direction must have finite nonzero norm")
        direction = direction / direction_norm
        evaluated_direction_norm = float(np.linalg.norm(direction))
        if isinstance(step, (bool, np.bool_)):
            raise TypeError("step must be a finite positive real scalar")
        h = float(step)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("step must be finite and positive")
        catol = _finite_nonnegative("curvature_atol", curvature_atol)
        crtol = _finite_nonnegative("curvature_rtol", curvature_rtol)
        satol = _finite_nonnegative("stationarity_atol", stationarity_atol)

        energy_values: list[float] = []
        for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
            value = energy(self.retract(direction, multiple * h))
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("energy callback must return a real scalar")
            scalar = np.asarray(value)
            if scalar.shape != () or not np.issubdtype(scalar.dtype, np.number):
                raise TypeError("energy callback must return a real scalar")
            complex_value = complex(scalar.item())
            if not (
                np.isfinite(complex_value.real)
                and np.isfinite(complex_value.imag)
            ):
                raise ValueError(
                    "energy callback returned a non-finite real or imaginary component"
                )
            imaginary_tolerance = (
                64.0
                * np.finfo(float).eps
                * max(abs(complex_value.real), abs(complex_value.imag))
            )
            if abs(complex_value.imag) > imaginary_tolerance:
                raise ValueError("energy callback returned a non-real scalar")
            energy_values.append(float(complex_value.real))

        em2, em1, e0, ep1, ep2 = energy_values
        stationarity = (em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * h)
        finite_difference = (
            -ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2
        ) / (12.0 * h * h)
        predicted = float(direction @ self.matvec(direction))
        residual = abs(finite_difference - predicted)
        curvature_tolerance = catol + crtol * max(
            abs(finite_difference), abs(predicted)
        )
        passed = abs(stationarity) <= satol and residual <= curvature_tolerance
        report = FivePointCurvatureCheck(
            step=h,
            input_direction_norm=direction_norm,
            evaluated_direction_norm=evaluated_direction_norm,
            diagnostic_only=True,
            energies_minus_2h_to_plus_2h=(em2, em1, e0, ep1, ep2),
            stationarity_derivative=stationarity,
            predicted_curvature=predicted,
            finite_difference_curvature=finite_difference,
            curvature_residual=residual,
            curvature_tolerance=curvature_tolerance,
            stationarity_tolerance=satol,
            passed=passed,
        )
        if raise_on_failure and not passed:
            raise ValueError(
                "diagnostic-only exact-unitary five-point curvature check failed: "
                f"stationarity={stationarity:.6e} (tol={satol:.6e}), "
                f"curvature residual={residual:.6e} "
                f"(tol={curvature_tolerance:.6e})"
            )
        return report


# Concise compatibility spelling: both names denote the same reusable object.
RaggedZeroTemperatureOrbitalHessian = ZeroTemperatureRaggedOrbitalHessian


def build_zero_temperature_ragged_orbital_hessian(
    projectors: Array,
    hamiltonians: Array,
    orbital_basis: Array,
    occupied_counts: Sequence[int],
    fock_derivative: FockDerivative,
    *,
    block_weights: Sequence[float] | None = None,
    validation_atol: float = 2.0e-10,
    validation_rtol: float = 2.0e-12,
    hamiltonian_atol: float = 2.0e-10,
    hamiltonian_rtol: float = 2.0e-12,
    stationarity_atol: float = 2.0e-10,
    stationarity_rtol: float = 2.0e-12,
) -> ZeroTemperatureRaggedOrbitalHessian:
    """Build a fixed-per-block-rank, block-preserving candidate action.

    This does not test inter-block occupation transfers or Aufbau ordering;
    require a separate occupation-gap gate before any stability claim.
    """

    return ZeroTemperatureRaggedOrbitalHessian(
        projectors,
        hamiltonians,
        orbital_basis,
        occupied_counts,
        fock_derivative,
        block_weights=block_weights,
        validation_atol=validation_atol,
        validation_rtol=validation_rtol,
        hamiltonian_atol=hamiltonian_atol,
        hamiltonian_rtol=hamiltonian_rtol,
        stationarity_atol=stationarity_atol,
        stationarity_rtol=stationarity_rtol,
    )


def build_zero_temperature_ragged_hessian(
    projectors: Array,
    hamiltonians: Array,
    orbital_basis: Array,
    occupied_counts: Sequence[int],
    fock_derivative: FockDerivative,
    **kwargs: Any,
) -> ZeroTemperatureRaggedOrbitalHessian:
    """Short alias for the fixed-per-block-rank candidate builder.

    The alias does not broaden the block-preserving scope and does not supply
    the separately required occupation-gap gate.
    """

    return build_zero_temperature_ragged_orbital_hessian(
        projectors,
        hamiltonians,
        orbital_basis,
        occupied_counts,
        fock_derivative,
        **kwargs,
    )


__all__ = [
    "BilinearSymmetryDiagnostic",
    "BilinearSymmetryProbe",
    "FivePointCurvatureCheck",
    "RaggedBlockLayout",
    "RaggedZeroTemperatureOrbitalHessian",
    "ZeroTemperatureRaggedOrbitalHessian",
    "build_zero_temperature_ragged_hessian",
    "build_zero_temperature_ragged_orbital_hessian",
]
