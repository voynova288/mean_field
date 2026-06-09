from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.lattice import build_kpath_from_nodes as _build_core_kpath_from_nodes, build_moire_k_grid_from_reciprocal
from .core_lattice import KPath
from .params import GRAPHENE_LATTICE_CONSTANT_NM


def _cross_2d(a: complex, b: complex) -> float:
    return float(a.real * b.imag - a.imag * b.real)



@dataclass(frozen=True)
class TMBGLattice:
    theta_deg: float
    theta_rad: float
    graphene_a1: complex
    graphene_a2: complex
    graphene_b1: complex
    graphene_b2: complex
    graphene_k_mag: float
    q0: complex
    q_plus: complex
    q_minus: complex
    g_m1: complex
    g_m2: complex
    g_indices: np.ndarray
    g_vectors: np.ndarray
    g_cutoff: float
    n_shells: int
    l_m: float
    gamma_m: complex
    k_m: complex
    kprime_m: complex
    m_m: complex
    mbz_area: float
    matrix_dim: int

    @property
    def q_vectors(self) -> dict[str, complex]:
        return {"0": self.q0, "+": self.q_plus, "-": self.q_minus}

    @property
    def n_g(self) -> int:
        return int(self.g_vectors.size)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "a_graphene_nm": float(abs(self.graphene_a1)),
            "L_M_nm": float(self.l_m),
            "G_M1_nm_inv": [float(self.g_m1.real), float(self.g_m1.imag)],
            "G_M2_nm_inv": [float(self.g_m2.real), float(self.g_m2.imag)],
            "Q0_nm_inv": [float(self.q0.real), float(self.q0.imag)],
            "Q_plus_nm_inv": [float(self.q_plus.real), float(self.q_plus.imag)],
            "Q_minus_nm_inv": [float(self.q_minus.real), float(self.q_minus.imag)],
            "n_shells": int(self.n_shells),
            "N_G": int(self.n_g),
            "matrix_dim": int(self.matrix_dim),
            "mBZ_area_nm_inv_sq": float(self.mbz_area),
        }


def build_tmbg_lattice(
    theta_deg: float,
    *,
    n_shells: int = 5,
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM,
) -> TMBGLattice:
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

    q_scale = 2.0 * graphene_k_mag * math.sin(theta_rad / 2.0)
    q0 = complex(0.0, -q_scale)
    q_plus = q_scale * complex(math.sqrt(3.0) / 2.0, 0.5)
    q_minus = q_scale * complex(-math.sqrt(3.0) / 2.0, 0.5)

    g_m1 = q_plus - q_minus
    g_m2 = q0 - q_minus
    g_cutoff = float((n_shells + 0.5) * abs(g_m1))
    l_m = float(a_nm / (2.0 * math.sin(theta_rad / 2.0)))

    index_limit = n_shells + 2
    entries: list[tuple[float, int, int, complex]] = []
    for n1 in range(-index_limit, index_limit + 1):
        for n2 in range(-index_limit, index_limit + 1):
            gvec = n1 * g_m1 + n2 * g_m2
            if abs(gvec) <= g_cutoff + 1.0e-12:
                entries.append((abs(gvec), n1, n2, complex(gvec)))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))

    g_indices = np.asarray([(n1, n2) for _, n1, n2, _ in entries], dtype=int)
    g_vectors = np.asarray([gvec for _, _, _, gvec in entries], dtype=np.complex128)

    gamma_m = 0.0 + 0.0j
    k_m = (g_m1 + g_m2) / 3.0
    kprime_m = (2.0 * g_m1 - g_m2) / 3.0
    m_m = g_m1 / 2.0
    mbz_area = abs(_cross_2d(g_m1, g_m2))
    matrix_dim = 6 * int(g_vectors.size)

    return TMBGLattice(
        theta_deg=float(theta_deg),
        theta_rad=float(theta_rad),
        graphene_a1=complex(graphene_a1),
        graphene_a2=complex(graphene_a2),
        graphene_b1=complex(graphene_b1),
        graphene_b2=complex(graphene_b2),
        graphene_k_mag=float(graphene_k_mag),
        q0=complex(q0),
        q_plus=complex(q_plus),
        q_minus=complex(q_minus),
        g_m1=complex(g_m1),
        g_m2=complex(g_m2),
        g_indices=g_indices,
        g_vectors=g_vectors,
        g_cutoff=g_cutoff,
        n_shells=int(n_shells),
        l_m=l_m,
        gamma_m=complex(gamma_m),
        k_m=complex(k_m),
        kprime_m=complex(kprime_m),
        m_m=complex(m_m),
        mbz_area=float(mbz_area),
        matrix_dim=int(matrix_dim),
    )


def build_kpath_from_nodes(nodes: tuple[complex, ...], labels: tuple[str, ...], points_per_segment: int) -> KPath:
    return _build_core_kpath_from_nodes(nodes, labels, int(points_per_segment))


def build_standard_kpath(lattice: TMBGLattice, points_per_segment: int = 120) -> KPath:
    return build_kpath_from_nodes(
        (lattice.k_m, lattice.gamma_m, lattice.m_m, lattice.kprime_m),
        ("K", "Gamma", "M", "Kprime"),
        points_per_segment,
    )




def park_fig2_nodes(lattice: TMBGLattice, *, gamma_prime_choice: str = "minus_g1") -> tuple[tuple[complex, ...], tuple[str, ...]]:
    """Return the extended-zone high-symmetry path used for Park et al. Fig. 2.

    The path is K~ -> K~' -> Gamma~ -> Gamma~' -> K~ in a neighboring
    moire Brillouin zone.  In the current code gauge the visually correct
    Fig. 2 comparison uses Gamma~' = -G_M1.  The choice is explicit and can
    be overridden from the plotting script for diagnostics.
    """
    k_bg = lattice.k_m
    k_mono = lattice.kprime_m
    gamma = 0.0 + 0.0j
    choices = {
        "g1": lattice.g_m1,
        "minus_g1": -lattice.g_m1,
        "g2": lattice.g_m2,
        "minus_g2": -lattice.g_m2,
        "g1_minus_g2": lattice.g_m1 - lattice.g_m2,
        "minus_g1_plus_g2": -lattice.g_m1 + lattice.g_m2,
    }
    try:
        gamma_prime = choices[gamma_prime_choice]
    except KeyError as exc:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported gamma_prime_choice={gamma_prime_choice!r}; valid choices are: {valid}") from exc
    k_bg_prime = gamma_prime + k_bg
    return (k_bg, k_mono, gamma, gamma_prime, k_bg_prime), ("K", "Kprime", "Gamma", "GammaPrime", "K")


def build_park_fig2_kpath(
    lattice: TMBGLattice,
    points_per_segment: int = 120,
    *,
    gamma_prime_choice: str = "minus_g1",
) -> KPath:
    nodes, labels = park_fig2_nodes(lattice, gamma_prime_choice=gamma_prime_choice)
    return build_kpath_from_nodes(nodes, labels, points_per_segment)

def build_moire_k_grid(
    lattice: TMBGLattice,
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
