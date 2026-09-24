from __future__ import annotations

import numpy as np

from .lattice import AhnTMDLattice
from .params import AhnTMDParameters


def _add_coeff(coeffs: dict[tuple[int, int], complex], key: tuple[int, int], value: complex) -> None:
    coeffs[key] = coeffs.get(key, 0.0j) + complex(value)


def _neg_key(key: tuple[int, int]) -> tuple[int, int]:
    return (-int(key[0]), -int(key[1]))


def _sub_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(a[0]) - int(b[0]), int(a[1]) - int(b[1]))


def diagonal_potential_coefficients(
    params: AhnTMDParameters,
    *,
    layer_sign: int,
) -> dict[tuple[int, int], complex]:
    """Fourier coefficients of Ahn ``V_+`` or ``V_-`` in q0/q1 units."""

    if int(layer_sign) not in (-1, 1):
        raise ValueError(f"layer_sign must be +/-1, got {layer_sign}")
    coeffs: dict[tuple[int, int], complex] = {}
    first_shell = ((1, 0), (0, 1), (-1, -1))
    phase1 = int(layer_sign) * params.psi_rad
    for q in first_shell:
        _add_coeff(coeffs, q, params.v1_eV * np.exp(1j * phase1))
        _add_coeff(coeffs, _neg_key(q), params.v1_eV * np.exp(-1j * phase1))
    second_shell = ((1, 2), (-2, -1), (1, -1))
    phase2 = params.psi_prime_rad
    for q in second_shell:
        _add_coeff(coeffs, q, params.v2_eV * np.exp(1j * phase2))
        _add_coeff(coeffs, _neg_key(q), params.v2_eV * np.exp(-1j * phase2))
    return coeffs


def gamma_coefficients(params: AhnTMDParameters) -> dict[tuple[int, int], complex]:
    """Fourier coefficients of ``Gamma(r)``, not ``Gamma*(r)``."""

    q0 = (1, 0)
    q1 = (0, 1)
    q2 = (-1, -1)
    q0_second = (1, 2)
    coeffs: dict[tuple[int, int], complex] = {}
    _add_coeff(coeffs, (0, 0), params.gamma1_eV)
    _add_coeff(coeffs, q2, params.gamma1_eV)
    _add_coeff(coeffs, _neg_key(q1), params.gamma1_eV)
    _add_coeff(coeffs, _neg_key(q0_second), params.gamma2_eV)
    _add_coeff(coeffs, q0, params.gamma2_eV)
    _add_coeff(coeffs, _neg_key(q0), params.gamma2_eV)
    return coeffs


def _all_gamma_star_deltas(gamma: dict[tuple[int, int], complex]) -> set[tuple[int, int]]:
    return {_neg_key(key) for key in gamma}


def _complex_to_xy(z: complex) -> np.ndarray:
    return np.asarray([complex(z).real, complex(z).imag], dtype=float)


def build_k_valley_hamiltonian(
    k_tilde: complex,
    lattice: AhnTMDLattice,
    params: AhnTMDParameters,
) -> np.ndarray:
    """Return the K-valley electron continuum Hamiltonian in eV."""

    index_by_g = lattice.g_index_lookup()
    hamiltonian = np.zeros((lattice.matrix_dim, lattice.matrix_dim), dtype=np.complex128)
    k_xy = _complex_to_xy(k_tilde)
    k_plus = _complex_to_xy(lattice.k_plus)
    k_minus = _complex_to_xy(lattice.k_minus)
    prefactor = float(params.hbar2_over_2m_eV_nm2)
    for ig, gvec in enumerate(lattice.g_vectors):
        momentum = k_xy + _complex_to_xy(complex(gvec))
        hamiltonian[2 * ig, 2 * ig] = -prefactor * float(np.dot(momentum - k_plus, momentum - k_plus))
        hamiltonian[2 * ig + 1, 2 * ig + 1] = -prefactor * float(np.dot(momentum - k_minus, momentum - k_minus))

    v_plus = diagonal_potential_coefficients(params, layer_sign=+1)
    v_minus = diagonal_potential_coefficients(params, layer_sign=-1)
    gamma = gamma_coefficients(params)
    for row_key, row in index_by_g.items():
        for delta, coeff in v_plus.items():
            col = index_by_g.get(_sub_key(row_key, delta))
            if col is not None:
                hamiltonian[2 * row, 2 * col] += coeff
        for delta, coeff in v_minus.items():
            col = index_by_g.get(_sub_key(row_key, delta))
            if col is not None:
                hamiltonian[2 * row + 1, 2 * col + 1] += coeff
        for delta, coeff in gamma.items():
            col = index_by_g.get(_sub_key(row_key, delta))
            if col is not None:
                hamiltonian[2 * row + 1, 2 * col] += coeff
        for delta in _all_gamma_star_deltas(gamma):
            col = index_by_g.get(_sub_key(row_key, delta))
            if col is not None:
                hamiltonian[2 * row, 2 * col + 1] += np.conjugate(gamma.get(_neg_key(delta), 0.0j))
    return hamiltonian


def build_hamiltonian(
    k_tilde: complex,
    lattice: AhnTMDLattice,
    params: AhnTMDParameters,
    *,
    valley: int = 1,
) -> np.ndarray:
    """Return the valley-resolved electron continuum Hamiltonian in eV.

    ``valley=-1`` is represented by time reversal of the K valley:
    ``H_K' (k) = conj(H_K(-k))``.  This keeps the valley convention explicit in
    the system layer while leaving HF iteration to the generic core.
    """

    if int(valley) == 1:
        return build_k_valley_hamiltonian(complex(k_tilde), lattice, params)
    if int(valley) == -1:
        return np.conjugate(build_k_valley_hamiltonian(-complex(k_tilde), lattice, params))
    raise ValueError(f"valley must be +1 or -1, got {valley}")


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: AhnTMDLattice,
    params: AhnTMDParameters,
    *,
    valley: int = 1,
    descending: bool = True,
    n_bands: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(build_hamiltonian(k_tilde, lattice, params, valley=valley))
    if descending:
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
    if n_bands is not None:
        return np.asarray(vals[: int(n_bands)], dtype=float), np.asarray(vecs[:, : int(n_bands)], dtype=np.complex128)
    return np.asarray(vals, dtype=float), np.asarray(vecs, dtype=np.complex128)


__all__ = [
    "build_hamiltonian",
    "build_k_valley_hamiltonian",
    "diagonal_potential_coefficients",
    "diagonalize_hamiltonian",
    "gamma_coefficients",
]
