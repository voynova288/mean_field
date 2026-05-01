from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .params import GRAPHENE_LATTICE_CONSTANT_NM



@dataclass(frozen=True)
class KPathNode:
    label: str
    index: int
    k_value: complex
    k_dist: float


@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    kdist: np.ndarray
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

    @property
    def nodes(self) -> tuple[KPathNode, ...]:
        return tuple(
            KPathNode(
                label=str(label),
                index=int(index),
                k_value=complex(self.kvec[int(index) - 1]),
                k_dist=float(self.kdist[int(index) - 1]),
            )
            for label, index in zip(self.labels, self.node_indices, strict=True)
        )


def cumulative_distance(kvec: np.ndarray) -> np.ndarray:
    values = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    distances = np.zeros(values.size, dtype=float)
    if values.size > 1:
        distances[1:] = np.cumsum(np.abs(np.diff(values)))
    return distances

def _cross_2d(a: complex, b: complex) -> float:
    return float(a.real * b.imag - a.imag * b.real)


def dot_2d(a: complex, b: complex) -> float:
    return float(a.real * b.real + a.imag * b.imag)


def rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(math.cos(angle_rad), math.sin(angle_rad))


def _complex_key(value: complex, *, digits: int = 12) -> tuple[float, float]:
    return (round(float(value.real), digits), round(float(value.imag), digits))


@dataclass(frozen=True)
class HTGLattice:
    theta_deg: float
    theta_rad: float
    graphene_a1: complex
    graphene_a2: complex
    graphene_b1: complex
    graphene_b2: complex
    graphene_k_mag: float
    k_theta: float
    q_vectors: np.ndarray
    b_m1: complex
    b_m2: complex
    a_m1: complex
    a_m2: complex
    delta: complex
    g_indices: np.ndarray
    g_vectors: np.ndarray
    g_cutoff: float
    n_shells: int
    l_m: float
    gamma_m: complex
    kappa_m: complex
    kappa_prime_m: complex
    m_m: complex
    mbz_area: float
    matrix_dim: int

    @property
    def q0(self) -> complex:
        return complex(self.q_vectors[0])

    @property
    def q1(self) -> complex:
        return complex(self.q_vectors[1])

    @property
    def q2(self) -> complex:
        return complex(self.q_vectors[2])

    @property
    def n_g(self) -> int:
        return int(self.g_vectors.size)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "a_graphene_nm": float(abs(self.graphene_a1)),
            "L_M_nm": float(self.l_m),
            "K_graphene_nm_inv": float(self.graphene_k_mag),
            "k_theta_nm_inv": float(self.k_theta),
            "q_vectors_nm_inv": [[float(q.real), float(q.imag)] for q in self.q_vectors],
            "b_m1_nm_inv": [float(self.b_m1.real), float(self.b_m1.imag)],
            "b_m2_nm_inv": [float(self.b_m2.real), float(self.b_m2.imag)],
            "a_m1_nm": [float(self.a_m1.real), float(self.a_m1.imag)],
            "a_m2_nm": [float(self.a_m2.real), float(self.a_m2.imag)],
            "delta_nm": [float(self.delta.real), float(self.delta.imag)],
            "n_shells": int(self.n_shells),
            "N_G": int(self.n_g),
            "matrix_dim": int(self.matrix_dim),
            "mBZ_area_nm_inv_sq": float(self.mbz_area),
            "g_cutoff_nm_inv": float(self.g_cutoff),
        }


def build_htg_lattice(
    theta_deg: float,
    *,
    n_shells: int = 5,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> HTGLattice:
    if n_shells < 0:
        raise ValueError(f"Expected non-negative n_shells, got {n_shells}")
    theta_rad = float(theta_deg) * math.pi / 180.0
    if theta_rad <= 0.0:
        raise ValueError(f"Expected a positive twist angle, got {theta_deg}")

    a_nm = float(graphene_lattice_constant_nm)
    graphene_a1 = complex(a_nm, 0.0)
    graphene_a2 = complex(a_nm / 2.0, a_nm * math.sqrt(3.0) / 2.0)
    graphene_b1 = complex(2.0 * math.pi / a_nm, -2.0 * math.pi / (a_nm * math.sqrt(3.0)))
    graphene_b2 = complex(0.0, 4.0 * math.pi / (a_nm * math.sqrt(3.0)))
    graphene_k_mag = 4.0 * math.pi / (3.0 * a_nm)

    k_theta = 2.0 * graphene_k_mag * math.sin(theta_rad / 2.0)
    q_vectors = np.asarray(
        [
            complex(0.0, -k_theta),
            k_theta * complex(math.sqrt(3.0) / 2.0, 0.5),
            k_theta * complex(-math.sqrt(3.0) / 2.0, 0.5),
        ],
        dtype=np.complex128,
    )

    b_m1 = complex(q_vectors[1] - q_vectors[0])
    b_m2 = complex(q_vectors[2] - q_vectors[0])
    a_scale = 4.0 * math.pi / (3.0 * k_theta)
    a_m1 = a_scale * complex(math.sqrt(3.0) / 2.0, 0.5)
    a_m2 = a_scale * complex(-math.sqrt(3.0) / 2.0, 0.5)
    delta = (a_m2 - a_m1) / 3.0
    l_m = a_nm / (2.0 * math.sin(theta_rad / 2.0))
    g_cutoff = float((n_shells + 0.5) * abs(b_m1))

    index_limit = n_shells + 2
    entries: list[tuple[float, int, int, complex]] = []
    for n1 in range(-index_limit, index_limit + 1):
        for n2 in range(-index_limit, index_limit + 1):
            gvec = n1 * b_m1 + n2 * b_m2
            if abs(gvec) <= g_cutoff + 1.0e-12:
                entries.append((abs(gvec), n1, n2, complex(gvec)))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    g_indices = np.asarray([(n1, n2) for _, n1, n2, _ in entries], dtype=int)
    g_vectors = np.asarray([gvec for _, _, _, gvec in entries], dtype=np.complex128)

    gamma_m = 0.0 + 0.0j
    kappa_m = (b_m1 + b_m2) / 3.0
    kappa_prime_m = -(b_m1 + b_m2) / 3.0
    m_m = b_m1 / 2.0
    mbz_area = abs(_cross_2d(b_m1, b_m2))
    matrix_dim = 6 * int(g_vectors.size)

    return HTGLattice(
        theta_deg=float(theta_deg),
        theta_rad=float(theta_rad),
        graphene_a1=complex(graphene_a1),
        graphene_a2=complex(graphene_a2),
        graphene_b1=complex(graphene_b1),
        graphene_b2=complex(graphene_b2),
        graphene_k_mag=float(graphene_k_mag),
        k_theta=float(k_theta),
        q_vectors=q_vectors,
        b_m1=complex(b_m1),
        b_m2=complex(b_m2),
        a_m1=complex(a_m1),
        a_m2=complex(a_m2),
        delta=complex(delta),
        g_indices=g_indices,
        g_vectors=g_vectors,
        g_cutoff=float(g_cutoff),
        n_shells=int(n_shells),
        l_m=float(l_m),
        gamma_m=complex(gamma_m),
        kappa_m=complex(kappa_m),
        kappa_prime_m=complex(kappa_prime_m),
        m_m=complex(m_m),
        mbz_area=float(mbz_area),
        matrix_dim=int(matrix_dim),
    )


def build_kpath_from_nodes(
    nodes: tuple[complex, ...],
    labels: tuple[str, ...],
    points_per_segment: int,
) -> KPath:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be positive")
    if len(nodes) < 2:
        raise ValueError("At least two path nodes are required.")
    if len(nodes) != len(labels):
        raise ValueError(f"Expected {len(nodes)} labels, got {len(labels)}")

    kvec: list[complex] = [complex(nodes[0])]
    node_indices = [1]
    for start_k, end_k in zip(nodes[:-1], nodes[1:], strict=True):
        step = (end_k - start_k) / float(points_per_segment)
        for idx in range(1, points_per_segment + 1):
            kvec.append(complex(start_k + idx * step))
        node_indices.append(len(kvec))

    kvec_array = np.asarray(kvec, dtype=np.complex128)
    return KPath(
        kvec=kvec_array,
        kdist=cumulative_distance(kvec_array),
        labels=labels,
        node_indices=tuple(node_indices),
    )


def _paper_edge_kappa_prime(lattice: HTGLattice) -> complex:
    return complex(lattice.kappa_prime_m + lattice.b_m1)


def build_standard_kpath(lattice: HTGLattice, points_per_segment: int = 120) -> KPath:
    """High-symmetry path used in Devakul et al. Fig. 2B and Fig. 3B.

    The two inequivalent corners labelled kappa and kappa_prime in the paper
    are adjacent corners of the extended-zone moire Brillouin zone.  The
    central-zone coordinates ``kappa_m`` and ``kappa_prime_m`` are opposite
    corners, so the naive segment kappa_m -> kappa_prime_m passes through
    Gamma.  For the plotted path we therefore use the reciprocal-lattice
    equivalent point ``kappa_prime_m + b_m1``.  This makes the kappa ->
    kappa_prime segment an mBZ edge; its midpoint is M = b_m1/2.
    """
    kappa_prime_edge = _paper_edge_kappa_prime(lattice)
    return build_kpath_from_nodes(
        (lattice.gamma_m, lattice.kappa_m, kappa_prime_edge, lattice.gamma_m, lattice.m_m),
        ("Gamma", "kappa", "kappa_prime", "Gamma", "M"),
        points_per_segment,
    )


def build_paper_hf_kpath(lattice: HTGLattice, points_per_segment: int = 120) -> KPath:
    """High-symmetry path used for Kwan et al. HF band/potential figures.

    The HF figures in Kwan et al. 2023 extend the usual
    ``Gamma-kappa-kappa_prime-Gamma-M`` path by returning from ``M`` to
    ``Gamma``.
    """

    kappa_prime_edge = _paper_edge_kappa_prime(lattice)
    return build_kpath_from_nodes(
        (lattice.gamma_m, lattice.kappa_m, kappa_prime_edge, lattice.gamma_m, lattice.m_m, lattice.gamma_m),
        ("Gamma", "kappa", "kappa_prime", "Gamma", "M", "Gamma"),
        points_per_segment,
    )


def build_moire_k_grid(
    lattice: HTGLattice,
    mesh_size: int,
    *,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    if mesh_size <= 0:
        raise ValueError(f"Expected positive mesh_size, got {mesh_size}")
    if endpoint:
        frac_1d = np.linspace(0.0, 1.0, mesh_size, dtype=float)
        f1, f2 = np.meshgrid(frac_1d, frac_1d, indexing="ij")
    else:
        frac_1d = np.arange(mesh_size, dtype=float) / float(mesh_size)
        f1, f2 = np.meshgrid(
            frac_1d + float(frac_shift[0]) / float(mesh_size),
            frac_1d + float(frac_shift[1]) / float(mesh_size),
            indexing="ij",
        )
    frac = np.stack([f1, f2], axis=-1)
    kvec = f1 * lattice.b_m1 + f2 * lattice.b_m2
    return frac, np.asarray(kvec, dtype=np.complex128)
