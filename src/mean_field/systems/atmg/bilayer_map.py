from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import block_diag, eigh, svd

from .lattice import ATMGLattice
from .params import ATMGParameters
from .tbg import build_monolayer_hamiltonian, build_tbg_hamiltonian


def _normalize_couplings(
    n_layers: int,
    coupling: float | Sequence[float],
) -> tuple[float, ...]:
    if isinstance(coupling, (float, int, np.floating, np.integer)):
        if n_layers <= 1:
            return tuple()
        return tuple(float(coupling) for _ in range(n_layers - 1))
    normalized = tuple(float(value) for value in coupling)
    if len(normalized) != max(0, int(n_layers) - 1):
        raise ValueError(
            f"Expected {max(0, int(n_layers) - 1)} couplings for n_layers={n_layers}, got {len(normalized)}"
        )
    return normalized


@dataclass(frozen=True)
class ATMGSVDResult:
    w_matrix: np.ndarray
    left_unitary: np.ndarray
    singular_values: np.ndarray
    right_unitary: np.ndarray

    @property
    def reconstruction(self) -> np.ndarray:
        sigma = np.zeros_like(self.w_matrix, dtype=np.complex128)
        diag_size = min(sigma.shape[0], sigma.shape[1], self.singular_values.size)
        for idx in range(diag_size):
            sigma[idx, idx] = self.singular_values[idx]
        return self.left_unitary @ sigma @ self.right_unitary.conjugate().T


@dataclass(frozen=True)
class MappedSpectrumResult:
    singular_values: np.ndarray
    labels: tuple[str, ...]
    subspace_energies: tuple[np.ndarray, ...]
    combined_energies: np.ndarray


def build_W_matrix(
    n_layers: int,
    coupling: float | Sequence[float],
) -> np.ndarray:
    n_layers = int(n_layers)
    if n_layers <= 0:
        raise ValueError(f"Expected a positive layer count, got {n_layers}")
    n_odd = (n_layers + 1) // 2
    n_even = n_layers // 2
    couplings = _normalize_couplings(n_layers, coupling)

    w_matrix = np.zeros((n_odd, n_even), dtype=np.complex128)
    for interface_index, value in enumerate(couplings):
        if interface_index % 2 == 0:
            odd_index = interface_index // 2
            even_index = interface_index // 2
        else:
            odd_index = interface_index // 2 + 1
            even_index = interface_index // 2
        w_matrix[odd_index, even_index] = float(value)
    return w_matrix


def svd_decompose(w_matrix: np.ndarray) -> ATMGSVDResult:
    w_matrix = np.asarray(w_matrix, dtype=np.complex128)
    left_unitary, singular_values, right_unitary_h = svd(w_matrix, full_matrices=True)
    right_unitary = right_unitary_h.conjugate().T
    return ATMGSVDResult(
        w_matrix=w_matrix,
        left_unitary=np.asarray(left_unitary, dtype=np.complex128),
        singular_values=np.asarray(singular_values, dtype=float),
        right_unitary=np.asarray(right_unitary, dtype=np.complex128),
    )


def analytic_singular_values(n_layers: int, alpha: float) -> np.ndarray:
    n_layers = int(n_layers)
    n_even = n_layers // 2
    return np.asarray(
        [2.0 * math_cos_pi(k / (n_layers + 1)) * float(alpha) for k in range(1, n_even + 1)],
        dtype=float,
    )


def math_cos_pi(value: float) -> float:
    return float(np.cos(np.pi * float(value)))


def build_block_diagonal_mapped_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int = 1,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    svd_result = svd_decompose(build_W_matrix(params.n_layers, params.resolved_alpha_couplings))
    blocks: list[np.ndarray] = []
    labels: list[str] = []

    for subspace_index, lambda_coupling in enumerate(svd_result.singular_values, start=1):
        blocks.append(
            build_tbg_hamiltonian(
                k_tilde,
                lattice,
                lambda_coupling=float(lambda_coupling),
                kappa=params.kappa,
                vf=params.vf,
                valley=valley,
            )
        )
        labels.append(f"TBG-{subspace_index}")

    if params.n_layers % 2 == 1:
        blocks.append(build_monolayer_hamiltonian(k_tilde, lattice, vf=params.vf, valley=valley, sector="odd"))
        labels.append("MLG")

    if not blocks:
        return np.zeros((0, 0), dtype=np.complex128), tuple(), np.asarray([], dtype=float)

    block_matrix = block_diag(*blocks)
    combined_energies = np.sort(np.linalg.eigvalsh(block_matrix))
    return np.asarray(block_matrix, dtype=np.complex128), tuple(labels), np.asarray(combined_energies, dtype=float)


def build_atmg_via_tbg_sum(
    k_tilde: complex,
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int = 1,
) -> MappedSpectrumResult:
    svd_result = svd_decompose(build_W_matrix(params.n_layers, params.resolved_alpha_couplings))
    labels: list[str] = []
    subspace_energies: list[np.ndarray] = []

    for subspace_index, lambda_coupling in enumerate(svd_result.singular_values, start=1):
        evals, _ = eigh(
            build_tbg_hamiltonian(
                k_tilde,
                lattice,
                lambda_coupling=float(lambda_coupling),
                kappa=params.kappa,
                vf=params.vf,
                valley=valley,
            )
        )
        labels.append(f"TBG-{subspace_index}")
        subspace_energies.append(np.asarray(evals, dtype=float))

    if params.n_layers % 2 == 1:
        evals, _ = eigh(build_monolayer_hamiltonian(k_tilde, lattice, vf=params.vf, valley=valley, sector="odd"))
        labels.append("MLG")
        subspace_energies.append(np.asarray(evals, dtype=float))

    if subspace_energies:
        combined_energies = np.sort(np.concatenate(subspace_energies))
    else:
        combined_energies = np.asarray([], dtype=float)

    return MappedSpectrumResult(
        singular_values=np.asarray(svd_result.singular_values, dtype=float),
        labels=tuple(labels),
        subspace_energies=tuple(subspace_energies),
        combined_energies=np.asarray(combined_energies, dtype=float),
    )


__all__ = [
    "ATMGSVDResult",
    "MappedSpectrumResult",
    "analytic_singular_values",
    "build_W_matrix",
    "build_atmg_via_tbg_sum",
    "build_block_diagonal_mapped_hamiltonian",
    "svd_decompose",
]
