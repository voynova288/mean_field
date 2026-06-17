from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.lattice import KPath, build_kpath_from_nodes as _build_core_kpath_from_nodes
from .params import GRAPHENE_LATTICE_CONSTANT_NM


def _cross_2d(a: complex, b: complex) -> float:
    return float(a.real * b.imag - a.imag * b.real)


def dot_2d(a: complex, b: complex) -> float:
    return float(a.real * b.real + a.imag * b.imag)


def rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(math.cos(angle_rad), math.sin(angle_rad))


@dataclass(frozen=True)
class HTQGLattice:
    """Moiré geometry for a single periodic HTQG domain.

    The high-symmetry points deliberately distinguish folded equivalence
    classes from paper path representatives.  ``kappap_class=-q0`` is used for
    layer-folding checks, while ``kappap_path=-q1`` is the adjacent mBZ corner
    selected by the Fig. 1(d,e) path-convention audit.
    """

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
    d_ba: complex
    g_indices: np.ndarray
    g_vectors: np.ndarray
    n_shells: int
    l_m: float
    gamma: complex
    kappa_class: complex
    kappap_class: complex
    kappa_path: complex
    kappap_path: complex
    m_path: complex
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

    @property
    def gamma_m(self) -> complex:
        """Backward-compatible alias matching other system modules."""

        return complex(self.gamma)

    @property
    def kappa_m(self) -> complex:
        return complex(self.kappa_path)

    @property
    def kappa_prime_m(self) -> complex:
        return complex(self.kappap_path)

    @property
    def m_m(self) -> complex:
        return complex(self.m_path)

    def g_index_lookup(self) -> dict[tuple[int, int], int]:
        return {tuple(int(x) for x in pair): idx for idx, pair in enumerate(self.g_indices)}

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
            "d_BA_nm": [float(self.d_ba.real), float(self.d_ba.imag)],
            "n_shells": int(self.n_shells),
            "N_G": int(self.n_g),
            "matrix_dim": int(self.matrix_dim),
            "mBZ_area_nm_inv_sq": float(self.mbz_area),
            "gamma_nm_inv": [float(self.gamma.real), float(self.gamma.imag)],
            "kappa_class_nm_inv": [float(self.kappa_class.real), float(self.kappa_class.imag)],
            "kappap_class_nm_inv": [float(self.kappap_class.real), float(self.kappap_class.imag)],
            "kappa_path_nm_inv": [float(self.kappa_path.real), float(self.kappa_path.imag)],
            "kappap_path_nm_inv": [float(self.kappap_path.real), float(self.kappap_path.imag)],
            "m_path_nm_inv": [float(self.m_path.real), float(self.m_path.imag)],
        }


def hex_shell_indices(n_shells: int) -> np.ndarray:
    """Return C3/C6-closed triangular-lattice indices.

    The shell condition is max(|n1|, |n2|, |n1+n2|) <= n_shells, giving
    N_G = 1 + 3 N (N + 1).  This is the cutoff required by the HTQG work
    document; do not replace it by a circular cutoff for paper checkpoints.
    """

    n_shells = int(n_shells)
    if n_shells < 0:
        raise ValueError(f"Expected non-negative n_shells, got {n_shells}")
    entries: list[tuple[int, int, int]] = []
    for n1 in range(-n_shells, n_shells + 1):
        for n2 in range(-n_shells, n_shells + 1):
            shell = max(abs(n1), abs(n2), abs(n1 + n2))
            if shell <= n_shells:
                entries.append((shell, n1, n2))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.asarray([(n1, n2) for _, n1, n2 in entries], dtype=int)


def build_htqg_lattice(
    theta_deg: float,
    *,
    n_shells: int = 4,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> HTQGLattice:
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
    # Fujimoto et al. write d_BA=(a_1^M+a_2^M)/3 in their moire basis.
    # With the reciprocal/vector convention used here (q0=(0,-kθ),
    # b_M1=q1-q0, b_M2=q2-q0 and the corresponding real-space pair below),
    # the paper's alpha-beta-gamma representative is obtained by the BA corner
    # (a_M2-a_M1)/3.  This sign choice matches the paper's K-valley Chern label
    # for alpha-beta-gamma: valence C=-2, conduction C=0.
    d_ba = (a_m2 - a_m1) / 3.0
    l_m = a_nm / (2.0 * math.sin(theta_rad / 2.0))

    g_indices = hex_shell_indices(int(n_shells))
    g_vectors = np.asarray([n1 * b_m1 + n2 * b_m2 for n1, n2 in g_indices], dtype=np.complex128)

    gamma = 0.0 + 0.0j
    kappa_class = complex(q_vectors[0])
    kappap_class = -complex(q_vectors[0])
    kappa_path = complex(q_vectors[0])
    # Adjacent K' representative for the paper plotting path.  The mirror edge
    # q0 -> -q1 gives the Fig. 1(d) flat-band bandwidth quoted in the text
    # while preserving the high-symmetry node gaps and topology.
    kappap_path = -complex(q_vectors[1])
    m_path = 0.5 * (kappa_path + kappap_path)
    mbz_area = abs(_cross_2d(b_m1, b_m2))
    matrix_dim = 8 * int(g_vectors.size)

    return HTQGLattice(
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
        d_ba=complex(d_ba),
        g_indices=g_indices,
        g_vectors=g_vectors,
        n_shells=int(n_shells),
        l_m=float(l_m),
        gamma=complex(gamma),
        kappa_class=complex(kappa_class),
        kappap_class=complex(kappap_class),
        kappa_path=complex(kappa_path),
        kappap_path=complex(kappap_path),
        m_path=complex(m_path),
        mbz_area=float(mbz_area),
        matrix_dim=int(matrix_dim),
    )


def build_kpath_from_nodes(
    nodes: tuple[complex, ...],
    labels: tuple[str, ...],
    points_per_segment: int,
) -> KPath:
    return _build_core_kpath_from_nodes(nodes, labels, int(points_per_segment))


def build_standard_kpath(lattice: HTQGLattice, points_per_segment: int = 120) -> KPath:
    """Return the paper path Gamma -> kappa -> kappa' -> Gamma -> M."""

    return build_kpath_from_nodes(
        (lattice.gamma, lattice.kappa_path, lattice.kappap_path, lattice.gamma, lattice.m_path),
        ("Gamma", "kappa", "kappa_prime", "Gamma", "M"),
        points_per_segment,
    )


def build_moire_k_grid(
    lattice: HTQGLattice,
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
            np.mod(frac_1d + float(frac_shift[0]) / float(mesh_size), 1.0),
            np.mod(frac_1d + float(frac_shift[1]) / float(mesh_size), 1.0),
            indexing="ij",
        )
    frac = np.stack([f1, f2], axis=-1)
    kvec = f1 * lattice.b_m1 + f2 * lattice.b_m2
    return frac, np.asarray(kvec, dtype=np.complex128)


__all__ = [
    "HTQGLattice",
    "KPath",
    "build_htqg_lattice",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_standard_kpath",
    "dot_2d",
    "hex_shell_indices",
    "rotate_complex",
]
