from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.lattice import KPath, build_kpath_from_nodes as _build_core_kpath_from_nodes, build_moire_k_grid_from_reciprocal
from .params import DEFAULT_HBN_MISMATCH, GRAPHENE_LATTICE_CONSTANT_NM


def rotation_matrix(angle_rad: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )


def rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value * complex(math.cos(angle_rad), math.sin(angle_rad)))


def _complex_from_xy(xy: np.ndarray) -> complex:
    return complex(float(xy[0]), float(xy[1]))


def _xy_from_complex(value: complex) -> np.ndarray:
    return np.asarray([float(complex(value).real), float(complex(value).imag)], dtype=float)


def _cross_2d(a: complex, b: complex) -> float:
    return float(a.real * b.imag - a.imag * b.real)


def _build_g_shell(
    g_m1: complex,
    g_m2: complex,
    *,
    cutoff_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    shortest = max(min(abs(g_m1), abs(g_m2)), 1.0e-15)
    coefficient_bound = int(math.ceil(float(cutoff_nm_inv) / shortest)) + 4
    entries: list[tuple[int, int, complex]] = []
    for n1 in range(-coefficient_bound, coefficient_bound + 1):
        for n2 in range(-coefficient_bound, coefficient_bound + 1):
            vector = complex(n1 * g_m1 + n2 * g_m2)
            if abs(vector) <= float(cutoff_nm_inv) + 1.0e-12:
                entries.append((int(n1), int(n2), vector))
    entries.sort(key=lambda item: (round(abs(item[2]), 12), item[0] * item[0] + item[1] * item[1], item[0], item[1]))
    indices = np.asarray([[entry[0], entry[1]] for entry in entries], dtype=int)
    vectors = np.asarray([entry[2] for entry in entries], dtype=np.complex128)
    return indices, vectors


def _real_space_vectors_from_reciprocal(g_m1: complex, g_m2: complex) -> tuple[complex, complex]:
    reciprocal = np.asarray(
        [[float(g_m1.real), float(g_m2.real)], [float(g_m1.imag), float(g_m2.imag)]],
        dtype=float,
    )
    direct = 2.0 * math.pi * np.linalg.inv(reciprocal).T
    return _complex_from_xy(direct[:, 0]), _complex_from_xy(direct[:, 1])


@dataclass(frozen=True)
class RLGhBNLattice:
    theta_deg: float
    theta_rad: float
    hbn_lattice_mismatch: float
    graphene_lattice_constant_nm: float
    shell_count: int
    g_cutoff_nm_inv: float

    k_dirac_nm_inv: float
    q_vectors: np.ndarray
    q_complex: np.ndarray
    g_vectors_basis: np.ndarray
    g_complex_basis: np.ndarray
    g_indices: np.ndarray
    g_vectors: np.ndarray

    gamma_m: complex
    k_m: complex
    kprime_m: complex
    m_m: complex
    real_space_a1: complex
    real_space_a2: complex
    moire_period_nm: float
    mbz_area: float
    matrix_dim: int

    @property
    def n_g(self) -> int:
        return int(self.g_vectors.shape[0])

    @property
    def g_m1(self) -> complex:
        return complex(self.g_complex_basis[0])

    @property
    def g_m2(self) -> complex:
        return complex(self.g_complex_basis[1])

    @property
    def g_m3(self) -> complex:
        return complex(self.g_complex_basis[2])

    def basis_index(self, g_index: int, layer: int, sublattice: int, *, layer_count: int) -> int:
        return int((int(g_index) * int(layer_count) + int(layer)) * 2 + int(sublattice))

    def layer_slice(self, g_index: int, layer: int, *, layer_count: int) -> slice:
        start = self.basis_index(g_index, layer, 0, layer_count=layer_count)
        return slice(start, start + 2)

    def g_index_lookup(self) -> dict[tuple[int, int], int]:
        return {(int(value[0]), int(value[1])): int(idx) for idx, value in enumerate(self.g_indices)}

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "epsilon": float(self.hbn_lattice_mismatch),
            "a_graphene_nm": float(self.graphene_lattice_constant_nm),
            "shell_count": int(self.shell_count),
            "g_cutoff_nm_inv": float(self.g_cutoff_nm_inv),
            "k_dirac_nm_inv": float(self.k_dirac_nm_inv),
            "q_vectors_nm_inv": [[float(value[0]), float(value[1])] for value in self.q_vectors],
            "g_m1_nm_inv": [float(self.g_m1.real), float(self.g_m1.imag)],
            "g_m2_nm_inv": [float(self.g_m2.real), float(self.g_m2.imag)],
            "g_m3_nm_inv": [float(self.g_m3.real), float(self.g_m3.imag)],
            "moire_period_nm": float(self.moire_period_nm),
            "mbz_area_nm_inv_sq": float(self.mbz_area),
            "N_G": int(self.n_g),
            "matrix_dim": int(self.matrix_dim),
        }


def build_rlg_hbn_lattice(
    theta_deg: float = 0.77,
    *,
    shell_count: int = 4,
    hbn_lattice_mismatch: float = DEFAULT_HBN_MISMATCH,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
    layer_count: int = 5,
    g_cutoff_nm_inv: float | None = None,
) -> RLGhBNLattice:
    theta_rad = float(theta_deg) * math.pi / 180.0
    if shell_count <= 0:
        raise ValueError(f"Expected a positive shell_count, got {shell_count}")

    a_nm = float(graphene_lattice_constant_nm)
    epsilon = float(hbn_lattice_mismatch)
    k_dirac = float(4.0 * math.pi / (3.0 * a_nm))
    xhat = np.asarray([1.0, 0.0], dtype=float)
    q1 = k_dirac * (np.eye(2, dtype=float) - rotation_matrix(-theta_rad) / (1.0 + epsilon)) @ xhat
    q_vectors = np.asarray([rotation_matrix(2.0 * math.pi * idx / 3.0) @ q1 for idx in range(3)], dtype=float)
    q_complex = np.asarray([_complex_from_xy(value) for value in q_vectors], dtype=np.complex128)

    g1 = q_vectors[1] - q_vectors[2]
    g_vectors_basis = np.asarray([rotation_matrix(2.0 * math.pi * idx / 3.0) @ g1 for idx in range(3)], dtype=float)
    g_complex_basis = np.asarray([_complex_from_xy(value) for value in g_vectors_basis], dtype=np.complex128)
    resolved_cutoff = float(shell_count) * float(np.linalg.norm(q1)) if g_cutoff_nm_inv is None else float(g_cutoff_nm_inv)
    g_indices, g_vectors = _build_g_shell(g_complex_basis[0], g_complex_basis[1], cutoff_nm_inv=resolved_cutoff)

    gamma_m = 0.0 + 0.0j
    # g_m1 and g_m2 are chosen with a 120 degree angle.  In this basis every
    # edge midpoint has two adjacent corners.  Fig. 2 of 2312.11617v1 follows
    # the edge centered at M=(g_m1+g_m2)/2; the adjacent corners are
    # (2*g_m1+g_m2)/3 and (g_m1+2*g_m2)/3.  This convention matches the
    # paper's remote-band cusp near M in the xi=0, V=48 meV panel.
    #
    # The opposite M orbit is not equivalent in a single valley because time
    # reversal relates K valley at k to K' valley at -k, not K valley to itself.
    k_m = complex((2.0 * g_complex_basis[0] + g_complex_basis[1]) / 3.0)
    kprime_m = complex((g_complex_basis[0] + 2.0 * g_complex_basis[1]) / 3.0)
    m_m = complex((g_complex_basis[0] + g_complex_basis[1]) / 2.0)
    real_a1, real_a2 = _real_space_vectors_from_reciprocal(g_complex_basis[0], g_complex_basis[1])
    g_norm = float(abs(g_complex_basis[0]))
    moire_period_nm = float(4.0 * math.pi / (math.sqrt(3.0) * g_norm))
    mbz_area = abs(_cross_2d(g_complex_basis[0], g_complex_basis[1]))

    return RLGhBNLattice(
        theta_deg=float(theta_deg),
        theta_rad=float(theta_rad),
        hbn_lattice_mismatch=epsilon,
        graphene_lattice_constant_nm=a_nm,
        shell_count=int(shell_count),
        g_cutoff_nm_inv=resolved_cutoff,
        k_dirac_nm_inv=k_dirac,
        q_vectors=q_vectors,
        q_complex=q_complex,
        g_vectors_basis=g_vectors_basis,
        g_complex_basis=g_complex_basis,
        g_indices=g_indices,
        g_vectors=g_vectors,
        gamma_m=gamma_m,
        k_m=k_m,
        kprime_m=kprime_m,
        m_m=m_m,
        real_space_a1=real_a1,
        real_space_a2=real_a2,
        moire_period_nm=moire_period_nm,
        mbz_area=float(mbz_area),
        matrix_dim=int(2 * int(layer_count) * int(g_vectors.shape[0])),
    )


def build_kpath_from_nodes(
    nodes: tuple[complex, ...],
    labels: tuple[str, ...],
    segment_point_counts: tuple[int, ...],
    *,
    duplicate_nodes: bool = False,
) -> KPath:
    return _build_core_kpath_from_nodes(
        nodes,
        labels,
        segment_point_counts,
        duplicate_nodes=duplicate_nodes,
    )

def build_standard_kpath(lattice: RLGhBNLattice, points_per_segment: int = 80) -> KPath:
    if points_per_segment <= 0:
        raise ValueError(f"Expected a positive points_per_segment, got {points_per_segment}")
    return build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m, lattice.m_m, lattice.kprime_m),
        ("K", "Gamma", "M", "Kprime"),
        (int(points_per_segment), int(points_per_segment), int(points_per_segment)),
    )


def build_moire_k_grid(
    lattice: RLGhBNLattice,
    mesh_size: int,
    *,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    return build_moire_k_grid_from_reciprocal(
        lattice.g_m1,
        lattice.g_m2,
        mesh_size,
        endpoint=endpoint,
        frac_shift=frac_shift,
    )


__all__ = [
    "RLGhBNLattice",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_rlg_hbn_lattice",
    "build_standard_kpath",
    "rotate_complex",
    "rotation_matrix",
]
