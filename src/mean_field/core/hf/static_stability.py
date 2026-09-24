"""Finite-temperature static Hartree--Fock stability operators.

The module is deliberately system agnostic.  A caller supplies a validated
finite-temperature eigensystem and a *linear* interaction action ``F``.  No
eigensystem is recomputed or reordered here.  With ``chi`` the Frechet
derivative of the Fermi matrix function, the two operators are

    J = chi F,
    C = (-chi)^(1/2) (-F) (-chi)^(1/2).

If ``F`` is self-adjoint in the supplied weighted matrix inner product, ``C``
is self-adjoint and is similar to ``J`` at strictly positive temperature.
The interaction callback must already contain every physical sign and source
quadrature weight; this layer adds neither a factor of one half nor another
integration weight.

Matrices use shape ``(n, n, nk)``.  The standard Euclidean representation is
obtained by the exact similarity

    T(X) = ravel(X * sqrt(k_weights)[None, None, :], order="C").

``thermal_energy`` is in the same energy units as the Hamiltonian, energies,
and chemical potential.  This core module has no Kelvin convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

ComplexArray = np.ndarray
FloatArray = np.ndarray
NumberConstraint = Literal["none", "global"]

EIGENSYSTEM_TOLERANCE = 1.0e-10
_LOG_TWO = float(np.log(2.0))


def _maximum_absolute(value: np.ndarray) -> float:
    array = np.asarray(value)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


def _finite_real_scalar(value: float, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if np.iscomplexobj(raw) and float(abs(complex(raw).imag)) > 0.0:
        raise ValueError(f"{name} must be real")
    result = float(np.real(raw))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_log_cosh(value: np.ndarray) -> np.ndarray:
    """Return ``log(cosh(value))`` without evaluating a large cosh."""

    absolute = np.abs(np.asarray(value, dtype=float))
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - _LOG_TWO


def _stable_log_sinhc(value: np.ndarray) -> np.ndarray:
    """Return ``log(sinh(value) / value)`` with the value at zero included."""

    absolute = np.abs(np.asarray(value, dtype=float))
    output = np.empty_like(absolute)
    small = absolute < 1.0e-4
    x_small = absolute[small]
    # sinh(x)/x = 1 + x^2/6 + x^4/120 + x^6/5040 + O(x^8).
    correction = (
        x_small * x_small / 6.0
        + x_small**4 / 120.0
        + x_small**6 / 5040.0
    )
    output[small] = np.log1p(correction)

    large = ~small
    x_large = absolute[large]
    # log(sinh(x)) = x - log(2) + log(1-exp(-2x)).
    output[large] = (
        x_large
        - _LOG_TWO
        + np.log(-np.expm1(-2.0 * x_large))
        - np.log(x_large)
    )
    return output


def _validated_k_weights(k_weights: FloatArray, *, nk: int) -> np.ndarray:
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != (nk,):
        raise ValueError(f"k_weights must have shape ({nk},)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("k_weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("k_weights must be strictly positive")
    return weights


@dataclass(frozen=True, init=False)
class FermiFrechetResponse:
    """Frechet response of a supplied finite-temperature Fermi eigensystem.

    ``divided_differences[p,q,k]`` is the nonpositive coefficient ``L`` in
    ``chi[X] = V (L * (V^dagger X V)) V^dagger``.  Its negative and its
    operator square root are retained separately.  In particular, the square
    root is computed directly from ``exp(log(-L)/2)`` rather than from
    ``sqrt(-L)`` after ``L`` may already have underflowed.
    """

    hamiltonian: ComplexArray
    eigenvectors: ComplexArray
    energies: FloatArray
    mu: float
    thermal_energy: float
    divided_differences: FloatArray
    negative_divided_differences: FloatArray
    sqrt_negative_divided_differences: FloatArray
    log_negative_divided_differences: FloatArray
    hamiltonian_hermiticity_residual: float
    eigenvector_unitarity_residual: float
    eigenpair_residual: float

    @classmethod
    def from_eigensystem(
        cls,
        hamiltonian: ComplexArray,
        eigenvectors: ComplexArray,
        energies: FloatArray,
        *,
        mu: float,
        thermal_energy: float,
    ) -> "FermiFrechetResponse":
        """Validate and bind a supplied eigensystem without rediagonalizing it."""

        h = np.asarray(hamiltonian, dtype=np.complex128)
        vectors = np.asarray(eigenvectors, dtype=np.complex128)
        raw_energies = np.asarray(energies)
        if h.ndim != 3 or h.shape[0] != h.shape[1] or h.shape[2] == 0:
            raise ValueError("hamiltonian must have shape (n, n, nk) with nk>0")
        n, _, nk = h.shape
        if vectors.shape != h.shape:
            raise ValueError("eigenvectors must have the same (n, n, nk) shape as H")
        if raw_energies.shape != (n, nk):
            raise ValueError("energies must have shape (n, nk)")
        if not np.all(np.isfinite(raw_energies)):
            raise ValueError("H, V, and energies must all be finite")
        if np.iscomplexobj(raw_energies):
            imaginary_scale = max(1.0, _maximum_absolute(np.real(raw_energies)))
            if _maximum_absolute(np.imag(raw_energies)) > (
                EIGENSYSTEM_TOLERANCE * imaginary_scale
            ):
                raise ValueError("energies must be real")
        energy_values = np.asarray(np.real(raw_energies), dtype=float)
        if not (
            np.all(np.isfinite(h))
            and np.all(np.isfinite(vectors))
            and np.all(np.isfinite(energy_values))
        ):
            raise ValueError("H, V, and energies must all be finite")

        mu_value = _finite_real_scalar(mu, name="mu")
        thermal_value = _finite_real_scalar(
            thermal_energy, name="thermal_energy"
        )
        if thermal_value <= 0.0:
            raise ValueError("thermal_energy must be strictly positive")

        h_scale = max(1.0, _maximum_absolute(h))
        h_hermiticity = _maximum_absolute(
            h - np.swapaxes(h.conj(), 0, 1)
        ) / h_scale
        if h_hermiticity > EIGENSYSTEM_TOLERANCE:
            raise ValueError(
                "hamiltonian is not Hermitian within tolerance: "
                f"residual={h_hermiticity:.3e}"
            )

        gram = np.einsum(
            "api,aqi->pqi", vectors.conj(), vectors, optimize=True
        )
        identity = np.eye(n, dtype=np.complex128)[:, :, None]
        vector_unitarity = _maximum_absolute(gram - identity)
        if vector_unitarity > EIGENSYSTEM_TOLERANCE:
            raise ValueError(
                "eigenvectors are not unitary within tolerance: "
                f"residual={vector_unitarity:.3e}"
            )

        h_times_v = np.einsum("abi,bpi->api", h, vectors, optimize=True)
        v_times_e = vectors * energy_values[None, :, :]
        eigen_scale = max(
            1.0, _maximum_absolute(h), _maximum_absolute(energy_values)
        )
        eigen_residual = _maximum_absolute(h_times_v - v_times_e) / eigen_scale
        if eigen_residual > EIGENSYSTEM_TOLERANCE:
            raise ValueError(
                "supplied H, V, and energies fail the eigenpair residual: "
                f"residual={eigen_residual:.3e}"
            )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dimensionless = (energy_values - mu_value) / thermal_value
        if not np.all(np.isfinite(dimensionless)):
            raise ValueError(
                "thermal_energy is too small to form finite dimensionless energies"
            )

        z_p = dimensionless[:, None, :]
        z_q = dimensionless[None, :, :]
        z_average = 0.5 * z_p + 0.5 * z_q
        half_difference = 0.5 * z_p - 0.5 * z_q
        log_sinhc = _stable_log_sinhc(half_difference)
        log_cosh_sum = np.logaddexp(
            _stable_log_cosh(z_average),
            _stable_log_cosh(half_difference),
        )
        log_negative = (
            log_sinhc
            - float(np.log(thermal_value))
            - _LOG_TWO
            - log_cosh_sum
        )
        if not np.all(np.isfinite(log_negative)):
            raise ValueError("Fermi Frechet log-coefficients are non-finite")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            negative = np.exp(log_negative)
            sqrt_negative = np.exp(0.5 * log_negative)
        if not (
            np.all(np.isfinite(negative))
            and np.all(np.isfinite(sqrt_negative))
        ):
            raise ValueError(
                "thermal scale makes the Fermi Frechet coefficients non-finite"
            )
        if np.any(negative < 0.0) or np.any(sqrt_negative < 0.0):
            raise ValueError("Fermi Frechet negative-response coefficients are invalid")

        result = object.__new__(cls)
        object.__setattr__(
            result, "hamiltonian", _readonly_copy(h, dtype=np.complex128)
        )
        object.__setattr__(
            result,
            "eigenvectors",
            _readonly_copy(vectors, dtype=np.complex128),
        )
        object.__setattr__(
            result, "energies", _readonly_copy(energy_values, dtype=np.float64)
        )
        object.__setattr__(result, "mu", mu_value)
        object.__setattr__(result, "thermal_energy", thermal_value)
        object.__setattr__(
            result,
            "negative_divided_differences",
            _readonly_copy(negative, dtype=np.float64),
        )
        object.__setattr__(
            result,
            "divided_differences",
            _readonly_copy(-negative, dtype=np.float64),
        )
        object.__setattr__(
            result,
            "sqrt_negative_divided_differences",
            _readonly_copy(sqrt_negative, dtype=np.float64),
        )
        object.__setattr__(
            result,
            "log_negative_divided_differences",
            _readonly_copy(log_negative, dtype=np.float64),
        )
        object.__setattr__(
            result,
            "hamiltonian_hermiticity_residual",
            float(h_hermiticity),
        )
        object.__setattr__(
            result,
            "eigenvector_unitarity_residual",
            float(vector_unitarity),
        )
        object.__setattr__(result, "eigenpair_residual", float(eigen_residual))
        return result

    @property
    def matrix_dimension(self) -> int:
        return int(self.hamiltonian.shape[0])

    @property
    def nk(self) -> int:
        return int(self.hamiltonian.shape[2])

    @property
    def matrix_shape(self) -> tuple[int, int, int]:
        return self.hamiltonian.shape

    @property
    def divided_difference_underflow_count(self) -> int:
        return int(np.count_nonzero(self.negative_divided_differences == 0.0))

    @property
    def negative_sqrt_underflow_count(self) -> int:
        return int(
            np.count_nonzero(self.sqrt_negative_divided_differences == 0.0)
        )

    @property
    def negative_sqrt_coefficients(self) -> FloatArray:
        """Alias for the independently constructed ``(-chi)^(1/2)`` factors."""

        return self.sqrt_negative_divided_differences

    def _validated_input(self, value: ComplexArray, *, name: str) -> np.ndarray:
        matrices = np.asarray(value, dtype=np.complex128)
        if matrices.shape != self.matrix_shape:
            raise ValueError(f"{name} must have shape {self.matrix_shape}")
        if not np.all(np.isfinite(matrices)):
            raise ValueError(f"{name} must be finite")
        return matrices

    def _to_eigenbasis(self, value: ComplexArray, *, name: str) -> np.ndarray:
        matrices = self._validated_input(value, name=name)
        return np.einsum(
            "api,adi,dqi->pqi",
            self.eigenvectors.conj(),
            matrices,
            self.eigenvectors,
            optimize=True,
        )

    def _from_eigenbasis(self, value: ComplexArray) -> np.ndarray:
        output = np.einsum(
            "api,pqi,dqi->adi",
            self.eigenvectors,
            value,
            self.eigenvectors.conj(),
            optimize=True,
        )
        if not np.all(np.isfinite(output)):
            raise ValueError("Fermi Frechet action produced non-finite values")
        return output

    def apply(self, delta_h: ComplexArray) -> ComplexArray:
        """Apply ``chi`` to an arbitrary finite complex matrix perturbation."""

        eigenbasis = self._to_eigenbasis(delta_h, name="delta_h")
        return self._from_eigenbasis(
            self.divided_differences * eigenbasis
        )

    def apply_negative_sqrt(self, value: ComplexArray) -> ComplexArray:
        """Apply the positive operator square root ``(-chi)^(1/2)``."""

        eigenbasis = self._to_eigenbasis(value, name="value")
        return self._from_eigenbasis(
            self.sqrt_negative_divided_differences * eigenbasis
        )

    def canonical_delta_mu(
        self,
        delta_h: ComplexArray,
        k_weights: FloatArray,
    ) -> complex:
        """Return the global fixed-number first-order chemical-potential shift.

        The ratio is evaluated as a scaled weighted average of diagonal
        eigenbasis matrix elements.  This remains defined when the absolute
        number susceptibility underflows but its relative weights remain
        representable.
        """

        weights = _validated_k_weights(k_weights, nk=self.nk)
        eigenbasis = self._to_eigenbasis(delta_h, name="delta_h")
        indices = np.arange(self.matrix_dimension)
        diagonal = eigenbasis[indices, indices, :]
        diagonal_log_response = self.log_negative_divided_differences[
            indices, indices, :
        ]
        log_measure = diagonal_log_response + np.log(weights)[None, :]
        log_scale = float(np.max(log_measure))
        if not np.isfinite(log_scale):
            raise ValueError("global number susceptibility has no finite scale")
        scaled_measure = np.exp(log_measure - log_scale)
        denominator = float(np.sum(scaled_measure))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("global number susceptibility is numerically zero")
        delta_mu = complex(np.sum(scaled_measure * diagonal) / denominator)
        if not (np.isfinite(delta_mu.real) and np.isfinite(delta_mu.imag)):
            raise ValueError("canonical delta_mu is non-finite")
        return delta_mu

    def apply_canonical(
        self,
        delta_h: ComplexArray,
        k_weights: FloatArray,
    ) -> ComplexArray:
        """Apply the globally fixed-number response ``chi[dH-dmu I]``."""

        matrices = self._validated_input(delta_h, name="delta_h")
        delta_mu = self.canonical_delta_mu(matrices, k_weights)
        identity = np.eye(self.matrix_dimension, dtype=np.complex128)[:, :, None]
        return self.apply(matrices - delta_mu * identity)


@dataclass(frozen=True)
class SqrtUnderflowSandwichBound:
    """Immutable diagnostic for omitted ``(-chi)^(1/2)`` coefficients.

    ``omitted_mask`` is defined by coefficients that are zero in the stored
    float64 square-root response even though their exact logarithms remain
    finite.  ``omitted_indices`` lists ``(p, q, k)`` in C order and
    ``omitted_logs`` contains the corresponding ``log(-chi[p,q,k])`` values.

    For ``C=S B S`` and the stored, underflow-truncated ``S_tilde``, the
    reported bound is

    ``||C-S_tilde B S_tilde||_2 <= delta``

    with ``log_delta = log(2) + (ell_max + ell_Z_max)/2 + log_Bfro``.
    ``delta`` may be zero when its logarithm lies below float64 range; the
    finite logarithm remains the authoritative value.  With no omissions,
    ``ell_Z_max`` and ``log_delta`` are ``-inf`` and ``delta`` is exactly zero.
    """

    omitted_mask: np.ndarray = field(repr=False, compare=False)
    omitted_indices: np.ndarray = field(repr=False, compare=False)
    omitted_logs: FloatArray = field(repr=False, compare=False)
    ell_max: float
    ell_Z_max: float
    log_Bfro: float
    log_delta: float
    delta: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.omitted_mask, dtype=bool)
        indices = np.asarray(self.omitted_indices, dtype=np.intp)
        logs = np.asarray(self.omitted_logs, dtype=float)
        if mask.ndim != 3:
            raise ValueError("omitted_mask must have shape (n, n, nk)")
        if indices.shape != (int(np.count_nonzero(mask)), 3):
            raise ValueError("omitted_indices must list every omitted (p,q,k)")
        if not np.array_equal(indices, np.argwhere(mask)):
            raise ValueError("omitted_indices do not match omitted_mask")
        if logs.shape != (indices.shape[0],):
            raise ValueError("omitted_logs must match omitted_indices")
        if not np.all(np.isfinite(logs)):
            raise ValueError("omitted_logs must be finite")

        object.__setattr__(
            self, "omitted_mask", _readonly_copy(mask, dtype=np.bool_)
        )
        object.__setattr__(
            self, "omitted_indices", _readonly_copy(indices, dtype=np.intp)
        )
        object.__setattr__(
            self, "omitted_logs", _readonly_copy(logs, dtype=np.float64)
        )

    @property
    def omitted_count(self) -> int:
        return int(self.omitted_indices.shape[0])

    @property
    def ell_z_max(self) -> float:
        """Lower-case alias for ``ell_Z_max``."""

        return self.ell_Z_max

    @property
    def log_delta_uf(self) -> float:
        """Alias emphasizing that this is the underflow truncation bound."""

        return self.log_delta

    @property
    def delta_uf(self) -> float:
        """Alias emphasizing that this is the underflow truncation bound."""

        return self.delta

    def to_json_dict(self) -> dict[str, float | str | None]:
        """Return the scalar bound as a standards-compliant JSON record.

        The exact no-omission convention remains ``-inf`` on this object.  JSON
        has no infinity literal, so the two intentional sentinels are encoded
        as tagged nulls instead.  Positive-omission logarithms must be finite.
        """

        finite_values = {
            "ell_max": float(self.ell_max),
            "log_Bfro": float(self.log_Bfro),
            "delta_uf": float(self.delta),
        }
        if not all(np.isfinite(value) for value in finite_values.values()):
            raise ValueError(
                "ell_max, log_Bfro, and delta_uf must be finite for JSON"
            )
        if finite_values["delta_uf"] < 0.0:
            raise ValueError("delta_uf must be nonnegative")

        ell_z_max = float(self.ell_Z_max)
        log_delta_uf = float(self.log_delta)
        if self.omitted_count == 0:
            if ell_z_max != -np.inf or log_delta_uf != -np.inf:
                raise ValueError(
                    "no omissions require ell_Z_max=log_delta_uf=-inf"
                )
            ell_z_max_json: float | None = None
            log_delta_uf_json: float | None = None
            ell_z_max_state = "negative_infinity_no_omission"
            log_delta_uf_state = "negative_infinity_no_omission"
        else:
            if not (np.isfinite(ell_z_max) and np.isfinite(log_delta_uf)):
                raise ValueError(
                    "positive omissions require finite ell_Z_max and log_delta_uf"
                )
            ell_z_max_json = ell_z_max
            log_delta_uf_json = log_delta_uf
            ell_z_max_state = "finite"
            log_delta_uf_state = "finite"

        return {
            "ell_max": finite_values["ell_max"],
            "ell_Z_max": ell_z_max_json,
            "ell_Z_max_state": ell_z_max_state,
            "log_Bfro": finite_values["log_Bfro"],
            "delta_uf": finite_values["delta_uf"],
            "log_delta_uf": log_delta_uf_json,
            "log_delta_uf_state": log_delta_uf_state,
        }

def compute_sqrt_underflow_sandwich_bound(
    response: FermiFrechetResponse,
    *,
    interaction_frobenius_norm: float | None = None,
    log_interaction_frobenius_norm: float | None = None,
) -> SqrtUnderflowSandwichBound:
    """Bound the omitted-square-root error in ``S B S`` in log domain.

    Exactly one interaction norm representation must be supplied.  The
    preferred ``log_interaction_frobenius_norm`` is ``log(||B_E||_F)`` and may
    have either sign; it represents a strictly positive norm.  The direct
    ``interaction_frobenius_norm`` input must be finite and strictly positive.

    Eigenbasis conjugation is Frobenius-unitary at each k point and commutes
    with the scalar k weight.  Hence ``||S||_2=exp(ell_max/2)`` and the omitted
    part has norm ``exp(ell_Z_max/2)``.  Using ``||B_E||_2<=||B_E||_F`` gives

    ``||S B S - S_tilde B S_tilde||_2``
    ``<= 2 exp((ell_max+ell_Z_max)/2) ||B_E||_F``.
    """

    if not isinstance(response, FermiFrechetResponse):
        raise TypeError("response must be a FermiFrechetResponse")
    supplied_direct = interaction_frobenius_norm is not None
    supplied_log = log_interaction_frobenius_norm is not None
    if supplied_direct == supplied_log:
        raise ValueError(
            "supply exactly one of interaction_frobenius_norm or "
            "log_interaction_frobenius_norm"
        )
    if supplied_log:
        log_b_fro = _finite_real_scalar(
            log_interaction_frobenius_norm,
            name="log_interaction_frobenius_norm",
        )
    else:
        b_fro = _finite_real_scalar(
            interaction_frobenius_norm,
            name="interaction_frobenius_norm",
        )
        if b_fro <= 0.0:
            raise ValueError(
                "interaction_frobenius_norm must be strictly positive"
            )
        log_b_fro = float(np.log(b_fro))

    logs = np.asarray(response.log_negative_divided_differences, dtype=float)
    omitted_mask = np.asarray(
        response.sqrt_negative_divided_differences == 0.0, dtype=bool
    )
    omitted_indices = np.argwhere(omitted_mask)
    omitted_logs = logs[omitted_mask]
    ell_max = float(np.max(logs))

    if omitted_logs.size == 0:
        ell_z_max = -np.inf
        log_delta = -np.inf
        delta = 0.0
    else:
        # The least-suppressed omitted coefficient controls ||S-S_tilde||:
        # this is max over the omitted logs, never the global/minimum log.
        ell_z_max = float(np.max(omitted_logs))
        log_delta = float(
            _LOG_TWO + 0.5 * (ell_max + ell_z_max) + log_b_fro
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            delta = float(np.exp(log_delta))
        if np.isnan(delta):
            raise ValueError("sqrt-underflow sandwich bound is non-finite")

    return SqrtUnderflowSandwichBound(
        omitted_mask=omitted_mask,
        omitted_indices=omitted_indices,
        omitted_logs=omitted_logs,
        ell_max=ell_max,
        ell_Z_max=ell_z_max,
        log_Bfro=log_b_fro,
        log_delta=log_delta,
        delta=delta,
    )

@dataclass(frozen=True)
class WeightedKMatrixSpace:
    """Exact isometry from a weighted k-matrix space to Euclidean vectors."""

    matrix_dimension: int
    k_weights: FloatArray
    _sqrt_k_weights: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.matrix_dimension, (bool, np.bool_)):
            raise ValueError("matrix_dimension must be a positive integer")
        dimension = int(self.matrix_dimension)
        if dimension != self.matrix_dimension or dimension < 1:
            raise ValueError("matrix_dimension must be a positive integer")
        raw_weights = np.asarray(self.k_weights, dtype=float)
        weights = _validated_k_weights(raw_weights, nk=raw_weights.size)
        if weights.size == 0:
            raise ValueError("k_weights must not be empty")
        weights_copy = _readonly_copy(weights, dtype=np.float64)
        square_roots = _readonly_copy(np.sqrt(weights), dtype=np.float64)
        object.__setattr__(self, "matrix_dimension", dimension)
        object.__setattr__(self, "k_weights", weights_copy)
        object.__setattr__(self, "_sqrt_k_weights", square_roots)

    @property
    def nk(self) -> int:
        return int(self.k_weights.size)

    @property
    def matrix_shape(self) -> tuple[int, int, int]:
        return (self.matrix_dimension, self.matrix_dimension, self.nk)

    @property
    def size(self) -> int:
        return self.matrix_dimension * self.matrix_dimension * self.nk

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def _validated_matrices(self, matrices: ComplexArray) -> np.ndarray:
        values = np.asarray(matrices, dtype=np.complex128)
        if values.shape != self.matrix_shape:
            raise ValueError(f"matrices must have shape {self.matrix_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("matrices must be finite")
        return values

    def to_euclidean(self, matrices: ComplexArray) -> ComplexArray:
        """Return ``ravel(matrices*sqrt(w), order='C')``."""

        values = self._validated_matrices(matrices)
        return np.ravel(
            values * self._sqrt_k_weights[None, None, :],
            order="C",
        ).copy()

    def from_euclidean(self, vector: ComplexArray) -> ComplexArray:
        """Apply the exact inverse weighted flattening map."""

        values = np.asarray(vector, dtype=np.complex128)
        if values.shape != (self.size,):
            raise ValueError(f"Euclidean vector must have shape ({self.size},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("Euclidean vector must be finite")
        return (
            values.reshape(self.matrix_shape, order="C")
            / self._sqrt_k_weights[None, None, :]
        )

    def inner(self, left: ComplexArray, right: ComplexArray) -> complex:
        """Return ``sum_k w_k Tr(left_k^dagger right_k)``."""

        left_vector = self.to_euclidean(left)
        right_vector = self.to_euclidean(right)
        return complex(np.vdot(left_vector, right_vector))


@dataclass(frozen=True)
class StaticStabilityOperators:
    """Matrix actions and Euclidean ``LinearOperator`` views of ``J`` and ``C``."""

    response: FermiFrechetResponse
    space: WeightedKMatrixSpace
    number_constraint: NumberConstraint
    jacobian: LinearOperator
    self_adjoint: LinearOperator
    constraint_vector_euclidean: ComplexArray | None
    _apply_jacobian: Callable[[ComplexArray], ComplexArray] = field(
        repr=False, compare=False
    )
    _apply_self_adjoint: Callable[[ComplexArray], ComplexArray] = field(
        repr=False, compare=False
    )

    @property
    def J(self) -> LinearOperator:
        return self.jacobian

    @property
    def C(self) -> LinearOperator:
        return self.self_adjoint

    @property
    def jacobian_operator(self) -> LinearOperator:
        return self.jacobian

    @property
    def stability_operator(self) -> LinearOperator:
        return self.self_adjoint

    def apply_jacobian(self, matrices: ComplexArray) -> ComplexArray:
        """Apply ``J=chi F`` (with global canonical response when requested)."""

        return self._apply_jacobian(matrices)

    def apply_self_adjoint(self, matrices: ComplexArray) -> ComplexArray:
        """Apply ``C=(-chi)^(1/2)(-F)(-chi)^(1/2)`` and any global Q."""

        return self._apply_self_adjoint(matrices)

    def apply_stability(self, matrices: ComplexArray) -> ComplexArray:
        return self.apply_self_adjoint(matrices)


def build_static_stability_operators(
    response: FermiFrechetResponse,
    interaction_action: Callable[[ComplexArray], ComplexArray],
    *,
    k_weights: FloatArray,
    number_constraint: NumberConstraint = "none",
) -> StaticStabilityOperators:
    """Build weighted-Euclidean actions for the HF Jacobian and self-adjoint stability kernel.

    ``interaction_action`` is exactly ``F``.  It must accept and return
    ``(n,n,nk)`` arrays and, for ``C`` to be self-adjoint, must be self-adjoint
    under ``sum_k w_k Tr(X_k^dagger Y_k)``.  The callback is not rescaled:
    existing Fock signs, source weights, and normalization are preserved.

    ``number_constraint='global'`` applies the canonical response to ``J`` and
    uses ``Q C Q`` with ``u=(-chi)^(1/2) I``.  ``'none'`` performs no number
    projection, as required for nonzero angular harmonics.
    """

    if not isinstance(response, FermiFrechetResponse):
        raise TypeError("response must be a FermiFrechetResponse")
    if not callable(interaction_action):
        raise TypeError("interaction_action must be callable")
    if number_constraint not in {"none", "global"}:
        raise ValueError("number_constraint must be 'none' or 'global'")

    weights = _validated_k_weights(k_weights, nk=response.nk)
    space = WeightedKMatrixSpace(
        matrix_dimension=response.matrix_dimension,
        k_weights=weights,
    )

    def apply_interaction(matrices: ComplexArray) -> np.ndarray:
        values = response._validated_input(matrices, name="interaction input")
        output = np.asarray(interaction_action(values), dtype=np.complex128)
        if output.shape != response.matrix_shape:
            raise ValueError(
                "interaction_action must return shape "
                f"{response.matrix_shape}, got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("interaction_action returned non-finite values")
        return output

    constraint_vector: np.ndarray | None = None
    if number_constraint == "global":
        identity = np.repeat(
            np.eye(response.matrix_dimension, dtype=np.complex128)[:, :, None],
            response.nk,
            axis=2,
        )
        raw_constraint = space.to_euclidean(
            response.apply_negative_sqrt(identity)
        )
        scale = _maximum_absolute(raw_constraint)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(
                "global number constraint is unresolved because "
                "(-chi)^(1/2) I underflowed to zero"
            )
        scaled_constraint = raw_constraint / scale
        norm = float(np.linalg.norm(scaled_constraint))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("global number constraint vector has zero norm")
        constraint_vector = scaled_constraint / norm
        constraint_vector.setflags(write=False)

    def project_vector(vector: np.ndarray) -> np.ndarray:
        if constraint_vector is None:
            return vector
        return vector - constraint_vector * np.vdot(constraint_vector, vector)

    def apply_jacobian_matrices(matrices: ComplexArray) -> np.ndarray:
        interaction = apply_interaction(matrices)
        if number_constraint == "global":
            return response.apply_canonical(interaction, weights)
        return response.apply(interaction)

    def apply_jacobian_vector(vector: ComplexArray) -> np.ndarray:
        matrices = space.from_euclidean(vector)
        return space.to_euclidean(apply_jacobian_matrices(matrices))

    def apply_self_adjoint_vector(vector: ComplexArray) -> np.ndarray:
        projected_input = project_vector(np.asarray(vector, dtype=np.complex128))
        matrices = space.from_euclidean(projected_input)
        response_sqrt = response.apply_negative_sqrt(matrices)
        minus_interaction = -apply_interaction(response_sqrt)
        output = response.apply_negative_sqrt(minus_interaction)
        return project_vector(space.to_euclidean(output))

    def apply_self_adjoint_matrices(matrices: ComplexArray) -> np.ndarray:
        vector = space.to_euclidean(matrices)
        return space.from_euclidean(apply_self_adjoint_vector(vector))

    jacobian = LinearOperator(
        shape=space.shape,
        matvec=apply_jacobian_vector,
        dtype=np.dtype(np.complex128),
    )
    self_adjoint = LinearOperator(
        shape=space.shape,
        matvec=apply_self_adjoint_vector,
        rmatvec=apply_self_adjoint_vector,
        dtype=np.dtype(np.complex128),
    )
    return StaticStabilityOperators(
        response=response,
        space=space,
        number_constraint=number_constraint,
        jacobian=jacobian,
        self_adjoint=self_adjoint,
        constraint_vector_euclidean=constraint_vector,
        _apply_jacobian=apply_jacobian_matrices,
        _apply_self_adjoint=apply_self_adjoint_matrices,
    )


__all__ = [
    "EIGENSYSTEM_TOLERANCE",
    "FermiFrechetResponse",
    "NumberConstraint",
    "SqrtUnderflowSandwichBound",
    "StaticStabilityOperators",
    "WeightedKMatrixSpace",
    "build_static_stability_operators",
    "compute_sqrt_underflow_sandwich_bound",
]
