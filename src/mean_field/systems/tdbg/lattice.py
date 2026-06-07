from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.lattice import KPath, build_kpath_from_nodes as _build_core_kpath_from_nodes
from .params import DEFAULT_BETA, DEFAULT_POISSON_RATIO, GRAPHENE_LATTICE_CONSTANT_NM


def rotation_matrix(angle_rad: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=float,
    )


def strain_twist_matrix(theta_rad: float, phi_rad: float, epsilon: float, poisson_ratio: float) -> np.ndarray:
    strain_tensor = np.asarray([[epsilon, 0.0], [0.0, -poisson_ratio * epsilon]], dtype=float)
    return rotation_matrix(-phi_rad) @ strain_tensor @ rotation_matrix(phi_rad) + np.asarray(
        [[0.0, -theta_rad], [theta_rad, 0.0]],
        dtype=float,
    )


def _cross_2d(a: complex, b: complex) -> float:
    return float(a.real * b.imag - a.imag * b.real)


def _complex_from_xy(xy: np.ndarray) -> complex:
    return complex(float(xy[0]), float(xy[1]))


def _build_neighbor_table(q_sites: np.ndarray, q_vectors: np.ndarray, *, digits: int = 3) -> tuple[tuple[tuple[int, int], ...], ...]:
    rounded_lookup = {
        (round(float(site[0]), digits), round(float(site[1]), digits)): idx
        for idx, site in enumerate(q_sites)
    }
    neighbors: list[tuple[tuple[int, int], ...]] = []
    for site in q_sites:
        local_entries: list[tuple[int, int]] = []
        for channel in range(q_vectors.shape[0]):
            target_xy = site[:2] + q_vectors[channel]
            key = (round(float(target_xy[0]), digits), round(float(target_xy[1]), digits))
            if key in rounded_lookup:
                local_entries.append((int(rounded_lookup[key]), int(channel)))
        neighbors.append(tuple(local_entries))
    return tuple(neighbors)


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

@dataclass(frozen=True)
class TDBGLattice:
    theta_deg: float
    theta_rad: float
    phi_deg: float
    phi_rad: float
    epsilon: float
    graphene_lattice_constant_nm: float
    beta: float
    poisson_ratio: float
    gauge_connection_nm_inv: float
    cut: float
    q_cutoff_nm_inv: float

    k_dirac_nm_inv: float
    k_m: complex
    gamma_m: complex
    m_m: complex
    kprime_m: complex
    g_m1: complex
    g_m2: complex
    mbz_area: float

    q_vectors: np.ndarray
    q_complex: np.ndarray
    b_vectors: np.ndarray
    b_complex: np.ndarray
    q_sites: np.ndarray
    q_neighbors: tuple[tuple[tuple[int, int], ...], ...]
    physical_g_vectors: np.ndarray
    matrix_dim: int

    @property
    def n_q(self) -> int:
        return int(self.q_sites.shape[0])

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "phi_deg": float(self.phi_deg),
            "epsilon": float(self.epsilon),
            "a_graphene_nm": float(self.graphene_lattice_constant_nm),
            "cut": float(self.cut),
            "q_cutoff_nm_inv": float(self.q_cutoff_nm_inv),
            "k_dirac_nm_inv": float(self.k_dirac_nm_inv),
            "q_vectors_nm_inv": [[float(value[0]), float(value[1])] for value in self.q_vectors],
            "g_m1_nm_inv": [float(self.g_m1.real), float(self.g_m1.imag)],
            "g_m2_nm_inv": [float(self.g_m2.real), float(self.g_m2.imag)],
            "mbz_area_nm_inv_sq": float(self.mbz_area),
            "N_q": int(self.n_q),
            "matrix_dim": int(self.matrix_dim),
        }


def build_tdbg_lattice(
    theta_deg: float,
    *,
    phi_deg: float = 0.0,
    epsilon: float = 0.0,
    cut: float = 4.0,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
    beta: float = DEFAULT_BETA,
    poisson_ratio: float = DEFAULT_POISSON_RATIO,
) -> TDBGLattice:
    theta_rad = float(theta_deg) * math.pi / 180.0
    phi_rad = float(phi_deg) * math.pi / 180.0
    if theta_rad <= 0.0:
        raise ValueError(f"Expected a positive twist angle, got {theta_deg}")
    if cut <= 0.0:
        raise ValueError(f"Expected a positive cutoff factor, got {cut}")

    a_nm = float(graphene_lattice_constant_nm)
    gauge_connection = float(math.sqrt(3.0) * float(beta) / (2.0 * a_nm))
    k_dirac = float(4.0 * math.pi / (3.0 * a_nm))

    k1 = np.asarray([k_dirac, 0.0], dtype=float)
    k2 = k_dirac * np.asarray([math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0)], dtype=float)
    k3 = -k_dirac * np.asarray([math.cos(math.pi / 3.0), math.sin(math.pi / 3.0)], dtype=float)

    q0 = strain_twist_matrix(theta_rad, phi_rad, epsilon, poisson_ratio) @ k1
    q1 = strain_twist_matrix(theta_rad, phi_rad, epsilon, poisson_ratio) @ k2
    q2 = strain_twist_matrix(theta_rad, phi_rad, epsilon, poisson_ratio) @ k3
    q_vectors = np.asarray([q0, q1, q2], dtype=float)
    q_complex = np.asarray([_complex_from_xy(value) for value in q_vectors], dtype=np.complex128)

    k_theta = float(max(np.linalg.norm(q0), np.linalg.norm(q1), np.linalg.norm(q2)))
    q_cutoff = float(math.sqrt(3.0) * k_theta * float(cut))

    b1 = q1 - q2
    b2 = q0 - q2
    b3 = q1 - q0
    b_vectors = np.asarray([b1, b2, b3], dtype=float)
    b_complex = np.asarray([_complex_from_xy(value) for value in b_vectors], dtype=np.complex128)

    q_sites = np.asarray(
        [
            np.asarray(list(np.asarray([i, j, 0], dtype=float) @ b_vectors - l * q0) + [float(l)], dtype=float)
            for i in range(-100, 100)
            for j in range(-100, 100)
            for l in (0, 1)
            if np.linalg.norm(np.asarray([i, j, 0], dtype=float) @ b_vectors - l * q0) <= q_cutoff + 1.0e-12
        ],
        dtype=float,
    )
    q_neighbors = _build_neighbor_table(q_sites, q_vectors)

    q_to_g = np.asarray([[sector, sector] for sector in q_sites[:, 2]], dtype=float) * q0 + q_sites[:, :2]
    physical_g_vectors = np.asarray(
        [_complex_from_xy(value) for value in q_to_g[q_sites[:, 2] == 1.0]],
        dtype=np.complex128,
    )

    g_m1 = _complex_from_xy(b2)
    g_m2 = _complex_from_xy(b1)
    gamma_m = q_complex[0] + q_complex[1]
    k_m = 0.0 + 0.0j
    m_m = q_complex[0] / 2.0
    kprime_m = q_complex[0]
    mbz_area = abs(_cross_2d(g_m1, g_m2))

    return TDBGLattice(
        theta_deg=float(theta_deg),
        theta_rad=float(theta_rad),
        phi_deg=float(phi_deg),
        phi_rad=float(phi_rad),
        epsilon=float(epsilon),
        graphene_lattice_constant_nm=a_nm,
        beta=float(beta),
        poisson_ratio=float(poisson_ratio),
        gauge_connection_nm_inv=gauge_connection,
        cut=float(cut),
        q_cutoff_nm_inv=q_cutoff,
        k_dirac_nm_inv=k_dirac,
        k_m=complex(k_m),
        gamma_m=complex(gamma_m),
        m_m=complex(m_m),
        kprime_m=complex(kprime_m),
        g_m1=complex(g_m1),
        g_m2=complex(g_m2),
        mbz_area=float(mbz_area),
        q_vectors=q_vectors,
        q_complex=q_complex,
        b_vectors=b_vectors,
        b_complex=b_complex,
        q_sites=q_sites,
        q_neighbors=q_neighbors,
        physical_g_vectors=physical_g_vectors,
        matrix_dim=4 * int(q_sites.shape[0]),
    )


def build_standard_kpath(lattice: TDBGLattice, resolution: int = 16) -> KPath:
    if resolution <= 0:
        raise ValueError(f"Expected a positive resolution, got {resolution}")
    l1 = int(resolution)
    l2 = int(math.sqrt(3.0) * resolution / 2.0)
    l3 = int(resolution / 2.0)
    return build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m, lattice.m_m, lattice.kprime_m),
        ("K", "Gamma", "M", "Kprime"),
        (l1, l2, l3),
        duplicate_nodes=True,
    )


def build_moire_k_grid(
    lattice: TDBGLattice,
    mesh_size: int,
    *,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    if mesh_size <= 0:
        raise ValueError(f"Expected a positive mesh_size, got {mesh_size}")

    shift_1 = float(frac_shift[0])
    shift_2 = float(frac_shift[1])
    if endpoint:
        frac_1 = np.linspace(0.0, 1.0, mesh_size, dtype=float) + shift_1
        frac_2 = np.linspace(0.0, 1.0, mesh_size, dtype=float) + shift_2
    else:
        frac_1 = np.mod(np.arange(mesh_size, dtype=float) / float(mesh_size) + shift_1, 1.0)
        frac_2 = np.mod(np.arange(mesh_size, dtype=float) / float(mesh_size) + shift_2, 1.0)
    frac_i, frac_j = np.meshgrid(frac_1, frac_2, indexing="ij")
    kvec = frac_i * lattice.g_m1 + frac_j * lattice.g_m2
    frac_grid = np.stack([frac_i, frac_j], axis=-1)
    return frac_grid, np.asarray(kvec, dtype=np.complex128)
