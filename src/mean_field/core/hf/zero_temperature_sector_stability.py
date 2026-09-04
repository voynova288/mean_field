"""System-agnostic paired-sector zero-temperature orbital Hessians.

A physical Hermitian tangent at non-self-conjugate quantum number ``q`` may
contain independent particle-hole coordinates in both ``q`` and ``-q`` lanes.
This module supplies only the generic real-coordinate plumbing:

``J_s[x] = (epsilon_p-epsilon_h) x_s + response_s[x_q,x_-q]``

and returns the real energy-Hessian action ``2 J``.  The caller owns sector
labels, transition enumeration, tangent embedding, interaction contractions,
and all physical authority.  No ``LinearOperator`` is exposed; a Hermitian
eigensolver requires a separate exact reciprocity and scalar-functional
qualification layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np
from .zero_temperature_ragged_stability import (
    BilinearSymmetryDiagnostic,
    BilinearSymmetryProbe,
)

Array = np.ndarray
PairedInteractionResponse = Callable[[Array, Array], tuple[Array, Array]]


def _readonly_array(value: object, dtype: np.dtype, shape: tuple[int, ...], label: str) -> Array:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite exact {dtype} {shape}")
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class OrbitalTransitionLane:
    """One deterministic list of occupied-to-virtual orbital transitions."""

    label: str
    particle_orbital_ids: Array
    hole_orbital_ids: Array
    particle_energies_ev: Array
    hole_energies_ev: Array
    one_body_gaps_ev: Array = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("transition lane label must be a nonempty exact string")
        arrays = tuple(np.asarray(value) for value in (
            self.particle_orbital_ids,
            self.hole_orbital_ids,
            self.particle_energies_ev,
            self.hole_energies_ev,
        ))
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("transition lane arrays must be one-dimensional")
        size = int(arrays[0].size)
        if any(array.size != size for array in arrays):
            raise ValueError("transition lane arrays must have one common size")
        particle_ids = _readonly_array(
            arrays[0], np.dtype(np.int64), (size,), "particle orbital ids"
        )
        hole_ids = _readonly_array(
            arrays[1], np.dtype(np.int64), (size,), "hole orbital ids"
        )
        particle_energies = _readonly_array(
            arrays[2], np.dtype(np.float64), (size,), "particle energies"
        )
        hole_energies = _readonly_array(
            arrays[3], np.dtype(np.float64), (size,), "hole energies"
        )
        if np.any(particle_ids < 0) or np.any(hole_ids < 0):
            raise ValueError("orbital ids must be nonnegative")
        if len(set(zip(particle_ids.tolist(), hole_ids.tolist(), strict=True))) != size:
            raise ValueError("transition lane contains duplicate particle-hole pairs")
        object.__setattr__(self, "particle_orbital_ids", particle_ids)
        object.__setattr__(self, "hole_orbital_ids", hole_ids)
        object.__setattr__(self, "particle_energies_ev", particle_energies)
        object.__setattr__(self, "hole_energies_ev", hole_energies)
        object.__setattr__(
            self,
            "one_body_gaps_ev",
            _readonly_array(
                np.asarray(particle_energies - hole_energies, dtype=np.float64),
                np.dtype(np.float64),
                (size,),
                "one-body gaps",
            ),
        )

    @property
    def complex_dimension(self) -> int:
        return int(self.particle_orbital_ids.size)


@dataclass(frozen=True, slots=True)
class PairedOrbitalTransitionFrame:
    """Real packing for two independently retained conjugate-sector lanes."""

    first: OrbitalTransitionLane
    second: OrbitalTransitionLane

    def __post_init__(self) -> None:
        if type(self.first) is not OrbitalTransitionLane or type(self.second) is not OrbitalTransitionLane:
            raise TypeError("paired transition frame requires exact lane objects")

    @property
    def first_complex_dimension(self) -> int:
        return self.first.complex_dimension

    @property
    def second_complex_dimension(self) -> int:
        return self.second.complex_dimension

    @property
    def complex_dimension(self) -> int:
        return self.first_complex_dimension + self.second_complex_dimension

    @property
    def real_dimension(self) -> int:
        return 2 * self.complex_dimension

    def unpack_real(self, vector: Array) -> tuple[Array, Array]:
        values = np.asarray(vector)
        if np.iscomplexobj(values):
            raise TypeError("paired-sector coordinates must have a real dtype")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.real_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"paired-sector coordinates must be finite ({self.real_dimension},)"
            )
        complex_values = values[: self.complex_dimension] + 1.0j * values[
            self.complex_dimension :
        ]
        split = self.first_complex_dimension
        return (
            np.asarray(complex_values[:split], dtype=np.complex128),
            np.asarray(complex_values[split:], dtype=np.complex128),
        )

    def pack_complex(
        self, first: Array, second: Array, *, factor: float = 1.0
    ) -> Array:
        first_values = np.asarray(first, dtype=np.complex128)
        second_values = np.asarray(second, dtype=np.complex128)
        if first_values.shape != (self.first_complex_dimension,) or second_values.shape != (
            self.second_complex_dimension,
        ):
            raise ValueError("paired-sector complex coordinates have the wrong shape")
        if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
            raise ValueError("paired-sector complex coordinates must be finite")
        scale = float(factor)
        if not np.isfinite(scale):
            raise ValueError("paired-sector packing factor must be finite")
        joined = scale * np.concatenate((first_values, second_values))
        return np.concatenate((joined.real, joined.imag))


@dataclass(frozen=True, slots=True, init=False)
class PairedSectorOrbitalHessian:
    """Matrix-free real Hessian for one caller-defined conjugate-sector orbit."""

    frame: PairedOrbitalTransitionFrame
    _interaction_response: PairedInteractionResponse

    scope = "paired conjugate-sector fixed-global-rank candidate"
    requires_separate_reciprocity_authority = True
    hermitian_eigensolver_authorized = False

    def __init__(
        self,
        frame: PairedOrbitalTransitionFrame,
        interaction_response: PairedInteractionResponse,
    ) -> None:
        if type(frame) is not PairedOrbitalTransitionFrame:
            raise TypeError("paired-sector Hessian requires an exact frame")
        if not callable(interaction_response):
            raise TypeError("paired-sector interaction response must be callable")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_interaction_response", interaction_response)

    @property
    def complex_dimension(self) -> int:
        return self.frame.complex_dimension

    @property
    def real_dimension(self) -> int:
        return self.frame.real_dimension

    def complex_gradient_jacobian_action(
        self, first: Array, second: Array
    ) -> tuple[Array, Array]:
        first_values = np.asarray(first, dtype=np.complex128)
        second_values = np.asarray(second, dtype=np.complex128)
        if first_values.shape != (self.frame.first_complex_dimension,) or second_values.shape != (
            self.frame.second_complex_dimension,
        ):
            raise ValueError("paired-sector complex coordinates have the wrong shape")
        if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
            raise ValueError("paired-sector complex coordinates must be finite")
        response = self._interaction_response(first_values, second_values)
        if type(response) is not tuple or len(response) != 2:
            raise TypeError("paired interaction response must return an exact pair")
        first_response = np.asarray(response[0], dtype=np.complex128)
        second_response = np.asarray(response[1], dtype=np.complex128)
        if first_response.shape != first_values.shape or second_response.shape != second_values.shape:
            raise ValueError("paired interaction response returned the wrong shape")
        if not np.all(np.isfinite(first_response)) or not np.all(np.isfinite(second_response)):
            raise ValueError("paired interaction response returned non-finite values")
        return (
            self.frame.first.one_body_gaps_ev * first_values + first_response,
            self.frame.second.one_body_gaps_ev * second_values + second_response,
        )

    def matvec(self, vector: Array) -> Array:
        first, second = self.frame.unpack_real(vector)
        first_output, second_output = self.complex_gradient_jacobian_action(
            first, second
        )
        return self.frame.pack_complex(first_output, second_output, factor=2.0)

    def check_bilinear_symmetry(
        self,
        *,
        seed: int,
        probe_count: int,
        atol: float = 2.0e-10,
        rtol: float = 2.0e-10,
    ) -> BilinearSymmetryDiagnostic:
        if type(seed) is not int or type(probe_count) is not int:
            raise TypeError("seed and probe_count must be exact integers")
        if probe_count < 0:
            raise ValueError("probe_count must be nonnegative")
        absolute = _finite_nonnegative("atol", atol)
        relative = _finite_nonnegative("rtol", rtol)
        if self.real_dimension == 0 or probe_count == 0:
            return BilinearSymmetryDiagnostic(seed, probe_count, self.real_dimension, ())
        rng = np.random.default_rng(seed)
        probes: list[BilinearSymmetryProbe] = []
        for index in range(probe_count):
            left = rng.standard_normal(self.real_dimension)
            right = rng.standard_normal(self.real_dimension)
            left /= np.linalg.norm(left)
            right /= np.linalg.norm(right)
            left_right = float(left @ self.matvec(right))
            right_left = float(right @ self.matvec(left))
            residual = abs(left_right - right_left)
            tolerance = absolute + relative * max(abs(left_right), abs(right_left))
            probes.append(
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
            seed, probe_count, self.real_dimension, tuple(probes)
        )


__all__ = [
    "OrbitalTransitionLane",
    "PairedInteractionResponse",
    "PairedOrbitalTransitionFrame",
    "PairedSectorOrbitalHessian",
]
