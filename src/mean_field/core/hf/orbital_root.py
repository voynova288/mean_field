"""Orbital-rotation root solver for zero-temperature Hartree--Fock stationarity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from scipy.linalg import expm
from scipy.optimize import root

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class OrbitalRootEvaluation(Generic[PayloadT]):
    hamiltonian: np.ndarray
    payload: PayloadT


@dataclass(frozen=True)
class OrbitalRotationRootRun(Generic[PayloadT]):
    projector: np.ndarray
    final_evaluation: OrbitalRootEvaluation[PayloadT]
    residual: np.ndarray
    residual_rms: float
    residual_max_abs: float
    iterations: int
    function_evaluations: int
    converged: bool
    exit_reason: str
    residual_history: np.ndarray


def run_orbital_rotation_root(
    initial_projector: np.ndarray,
    evaluate_projector: Callable[[np.ndarray], OrbitalRootEvaluation[PayloadT]],
    *,
    occupied_per_k: int,
    residual_tolerance: float = 1.0e-10,
    max_iter: int = 100,
) -> OrbitalRotationRootRun[PayloadT]:
    """Solve the occupied--virtual HF orbital gradient in a unitary chart.

    The chart is anchored at an idempotent physical ket projector. For each k,
    a complex virtual-by-occupied coordinate ``X`` generates an anti-Hermitian
    orbital rotation. The root function is the virtual--occupied block of the
    freshly rebuilt physical Hamiltonian in the rotated basis.
    """

    projector0 = np.asarray(initial_projector, dtype=np.complex128)
    if projector0.ndim != 3 or projector0.shape[0] != projector0.shape[1]:
        raise ValueError("initial_projector must have shape (n,n,nk)")
    if not np.all(np.isfinite(projector0)):
        raise ValueError("initial_projector must be finite")
    n, _, nk = projector0.shape
    nocc = int(occupied_per_k)
    if not 0 < nocc < n:
        raise ValueError("occupied_per_k must lie strictly between zero and n")
    nvir = n - nocc
    tolerance = float(residual_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")

    reference_basis = np.empty_like(projector0)
    for k_index in range(nk):
        pblock = projector0[:, :, k_index]
        if np.linalg.norm(pblock - pblock.conj().T) > 1.0e-9:
            raise ValueError("initial projector is not Hermitian")
        eigenvalues, eigenvectors = np.linalg.eigh(pblock)
        if np.max(np.minimum(np.abs(eigenvalues), np.abs(eigenvalues - 1.0))) > 1.0e-7:
            raise ValueError("initial projector is not idempotent")
        order = np.concatenate(
            [np.arange(n - nocc, n, dtype=np.int64), np.arange(0, n - nocc, dtype=np.int64)]
        )
        reference_basis[:, :, k_index] = eigenvectors[:, order]

    complex_size = nvir * nocc * nk
    real_size = 2 * complex_size
    residual_history: list[float] = []
    evaluation_count = 0

    def unpack(vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (real_size,):
            raise ValueError("orbital-root vector has the wrong shape")
        return (
            values[:complex_size] + 1.0j * values[complex_size:]
        ).reshape(nvir, nocc, nk)

    def rotated_projector_and_basis(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = unpack(vector)
        projector = np.empty_like(projector0)
        rotated_basis = np.empty_like(reference_basis)
        for k_index in range(nk):
            generator = np.zeros((n, n), dtype=np.complex128)
            xblock = coordinates[:, :, k_index]
            generator[nocc:, :nocc] = xblock
            generator[:nocc, nocc:] = -xblock.conj().T
            unitary = expm(generator)
            basis = reference_basis[:, :, k_index] @ unitary
            occupied = basis[:, :nocc]
            projector[:, :, k_index] = occupied @ occupied.conj().T
            rotated_basis[:, :, k_index] = basis
        return projector, rotated_basis

    def evaluate_vector(vector: np.ndarray) -> tuple[np.ndarray, OrbitalRootEvaluation[PayloadT], np.ndarray]:
        nonlocal evaluation_count
        projector, basis = rotated_projector_and_basis(vector)
        evaluation = evaluate_projector(projector)
        hamiltonian = np.asarray(evaluation.hamiltonian, dtype=np.complex128)
        if hamiltonian.shape != projector.shape or not np.all(np.isfinite(hamiltonian)):
            raise ValueError("orbital-root Hamiltonian has invalid shape or values")
        residual = np.empty((nvir, nocc, nk), dtype=np.complex128)
        for k_index in range(nk):
            rotated_h = (
                basis[:, :, k_index].conj().T
                @ hamiltonian[:, :, k_index]
                @ basis[:, :, k_index]
            )
            residual[:, :, k_index] = rotated_h[nocc:, :nocc]
        packed = np.concatenate([residual.real.ravel(), residual.imag.ravel()])
        evaluation_count += 1
        residual_history.append(float(np.max(np.abs(packed), initial=0.0)))
        return packed, evaluation, projector

    def function(vector: np.ndarray) -> np.ndarray:
        packed, _, _ = evaluate_vector(vector)
        return packed

    solution = root(
        function,
        np.zeros(real_size, dtype=np.float64),
        method="krylov",
        options={"fatol": tolerance, "maxiter": int(max_iter), "disp": False},
    )
    packed, final_evaluation, final_projector = evaluate_vector(solution.x)
    complex_residual = (
        packed[:complex_size] + 1.0j * packed[complex_size:]
    ).reshape(nvir, nocc, nk)
    maximum = float(np.max(np.abs(complex_residual), initial=0.0))
    rms = float(np.linalg.norm(complex_residual) / np.sqrt(complex_residual.size))
    converged = maximum <= tolerance
    exit_reason = "converged" if converged else str(solution.message)
    return OrbitalRotationRootRun(
        projector=final_projector,
        final_evaluation=final_evaluation,
        residual=complex_residual,
        residual_rms=rms,
        residual_max_abs=maximum,
        iterations=int(getattr(solution, "nit", 0)),
        function_evaluations=evaluation_count,
        converged=converged,
        exit_reason=exit_reason,
        residual_history=np.asarray(residual_history, dtype=np.float64),
    )


__all__ = [
    "OrbitalRootEvaluation",
    "OrbitalRotationRootRun",
    "run_orbital_rotation_root",
]
