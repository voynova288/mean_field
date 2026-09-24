from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np

from ...core.lattice import KPath, build_kpath_from_nodes, build_moire_k_grid_from_reciprocal
from .params import AhnTMDParameters, DEFAULT_AHN_TMD_PARAMETERS


def c3_rotate(z: complex) -> complex:
    return complex(np.exp(2.0j * np.pi / 3.0) * complex(z))


def hex_shell_indices(n_shell: int) -> tuple[tuple[int, int], ...]:
    n = int(n_shell)
    if n < 0:
        raise ValueError(f"n_shell must be non-negative, got {n_shell}")
    out: list[tuple[int, int]] = []
    for m in range(-n, n + 1):
        for nn in range(-n, n + 1):
            if max(abs(m), abs(nn), abs(m + nn)) <= n:
                out.append((int(m), int(nn)))
    out.sort(key=lambda item: (max(abs(item[0]), abs(item[1]), abs(item[0] + item[1])), item[0], item[1]))
    return tuple(out)


@dataclass(frozen=True)
class AhnTMDLattice:
    theta_deg: float
    n_shell: int
    q0: complex
    q1: complex
    q2: complex
    g_indices: tuple[tuple[int, int], ...]
    g_vectors: np.ndarray
    k_plus: complex
    k_minus: complex

    @property
    def n_g(self) -> int:
        return len(self.g_indices)

    @property
    def matrix_dim(self) -> int:
        return 2 * self.n_g

    @property
    def reciprocal_matrix(self) -> np.ndarray:
        return np.asarray([[self.q0.real, self.q1.real], [self.q0.imag, self.q1.imag]], dtype=float)

    @property
    def moire_area_nm2(self) -> float:
        area_bz = abs(float(np.linalg.det(self.reciprocal_matrix)))
        return float((2.0 * pi) ** 2 / area_bz)

    def g_index_lookup(self) -> dict[tuple[int, int], int]:
        return {tuple(pair): idx for idx, pair in enumerate(self.g_indices)}

    def g_vector(self, pair: tuple[int, int]) -> complex:
        return int(pair[0]) * self.q0 + int(pair[1]) * self.q1


def build_ahn_tmd_lattice(
    theta_deg: float,
    *,
    n_shell: int = 5,
    params: AhnTMDParameters = DEFAULT_AHN_TMD_PARAMETERS,
) -> AhnTMDLattice:
    theta = float(theta_deg) * pi / 180.0
    a_m = float(params.a0_nm) / (2.0 * sin(theta / 2.0))
    q0 = complex(4.0 * pi / (a_m * sqrt(3.0)), 0.0)
    q1 = c3_rotate(q0)
    q2 = c3_rotate(q1)
    g_indices = hex_shell_indices(n_shell)
    g_vectors = np.asarray([int(m) * q0 + int(n) * q1 for m, n in g_indices], dtype=np.complex128)
    k_plus = (q0 - q2) / 3.0
    k_minus = (q0 - q1) / 3.0
    return AhnTMDLattice(
        theta_deg=float(theta_deg),
        n_shell=int(n_shell),
        q0=q0,
        q1=q1,
        q2=q2,
        g_indices=g_indices,
        g_vectors=g_vectors,
        k_plus=k_plus,
        k_minus=k_minus,
    )


def build_moire_k_grid(
    lattice: AhnTMDLattice,
    mesh_size: int,
    *,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    return build_moire_k_grid_from_reciprocal(
        lattice.q0,
        lattice.q1,
        int(mesh_size),
        endpoint=endpoint,
        frac_shift=frac_shift,
    )


def build_rectangular_moire_k_grid(
    lattice: AhnTMDLattice,
    nx: int,
    ny: int,
    *,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    if int(nx) <= 0 or int(ny) <= 0:
        raise ValueError(f"nx and ny must be positive, got {(nx, ny)}")
    fx = np.mod(np.arange(int(nx), dtype=float) / float(nx) + float(frac_shift[0]), 1.0)
    fy = np.mod(np.arange(int(ny), dtype=float) / float(ny) + float(frac_shift[1]), 1.0)
    frac_i, frac_j = np.meshgrid(fx, fy, indexing="ij")
    frac = np.stack([frac_i, frac_j], axis=-1)
    kvec = frac_i * lattice.q0 + frac_j * lattice.q1
    return frac, np.asarray(kvec, dtype=np.complex128)


def standard_kpath(lattice: AhnTMDLattice, *, points_per_segment: int = 120) -> KPath:
    # Keep the same Gamma-K-M-Gamma convention as the tTMD Ahn adapter.
    gamma = 0.0 + 0.0j
    k_point = (2.0 * lattice.q0 + lattice.q1) / 3.0
    m_point = 0.5 * lattice.q0
    return build_kpath_from_nodes(
        (gamma, k_point, m_point, gamma),
        ("Γ", "K", "M", "Γ"),
        int(points_per_segment),
    )


__all__ = [
    "AhnTMDLattice",
    "build_ahn_tmd_lattice",
    "build_moire_k_grid",
    "build_rectangular_moire_k_grid",
    "c3_rotate",
    "hex_shell_indices",
    "standard_kpath",
]
