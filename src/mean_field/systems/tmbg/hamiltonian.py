from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from .lattice import TMBGLattice, _complex_key
from .params import TMBGParameters


VALID_VALLEYS = (-1, 1)
MOIRE_CHANNELS = ("0", "+", "-")


@dataclass(frozen=True)
class MoireCouplingEntry:
    channel: str
    middle_index: int
    top_index: int


def _validate_valley(valley: int) -> int:
    valley = int(valley)
    if valley not in VALID_VALLEYS:
        raise ValueError(f"Expected valley in {VALID_VALLEYS}, got {valley}")
    return valley


def _rotated_complex_momentum(kvec: complex, phi: float) -> complex:
    return complex(kvec) * complex(math.cos(phi), -math.sin(phi))


def _valley_pi(kvec: complex, phi: float, valley: int) -> tuple[complex, complex]:
    q = _rotated_complex_momentum(kvec, phi)
    if _validate_valley(valley) == 1:
        return complex(q), complex(q.conjugate())
    return complex(-q.conjugate()), complex(-q)


def dirac_block(kvec: complex, phi: float, vf: float, valley: int) -> np.ndarray:
    pi, pi_dag = _valley_pi(kvec, phi, valley)
    return np.asarray([[0.0, vf * pi_dag], [vf * pi, 0.0]], dtype=np.complex128)


def blg_interlayer(
    kvec: complex, phi: float, params: TMBGParameters, valley: int
) -> np.ndarray:
    """Bernal bottom-middle interlayer block in the code gauge.

    The basis of each six-orbital moire block is
    (A_b, B_b, A_m, B_m, A_t, B_t), and t^- is inserted as
    H[bottom, middle].  Relative to the compact notation in Park et al.
    Eq. (3), the intralayer Dirac block used here has the opposite B-
    sublattice phase.  In this gauge the Park/PyTwist-compatible AB block is

        [[-v4*pi^dagger, -v3*pi],
         [ t1,            -v4*pi^dagger]].

    The v3 term connects the non-dimer pair A_b <-> B_m and is the term that
    fixes the trigonal-warping orientation of the complete-model bands.
    """
    pi, pi_dag = _valley_pi(kvec, phi, valley)
    return np.asarray(
        [
            [-params.v4 * pi_dag, -params.v3 * pi],
            [params.t1, -params.v4 * pi_dag],
        ],
        dtype=np.complex128,
    )


def build_diagonal_block(
    k_tilde: complex,
    gvec: complex,
    lattice: TMBGLattice,
    params: TMBGParameters,
    valley: int,
) -> np.ndarray:
    valley = _validate_valley(valley)
    # Use an extended-zone gauge in which the uncoupled bilayer low-energy point
    # sits at K~ and the monolayer Dirac point sits at K~'. This amounts to
    # measuring the intralayer Dirac momenta relative to valley-dependent mBZ
    # corners rather than relative to Γ~.
    k_bottom = complex(k_tilde + gvec - valley * lattice.k_m)
    k_top = complex(k_tilde + gvec - valley * lattice.kprime_m)
    h_bottom = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, valley)
    h_middle = dirac_block(k_bottom, -lattice.theta_rad / 2.0, params.vf, valley)
    h_top = dirac_block(k_top, lattice.theta_rad / 2.0, params.vf, valley)
    t_blg = blg_interlayer(k_bottom, -lattice.theta_rad / 2.0, params, valley)

    block = np.zeros((6, 6), dtype=np.complex128)
    block[0:2, 0:2] = h_bottom
    block[2:4, 2:4] = h_middle
    block[4:6, 4:6] = h_top

    block[0:2, 2:4] = t_blg
    block[2:4, 0:2] = t_blg.conjugate().T

    # Park Eq. (2) is written in the same sublattice gauge as the t^- block
    # above.  In the explicit (A_b, B_b, A_m, B_m, A_t, B_t) basis used here it
    # gives the dimer-site correction B_b += delta and A_m -= delta.
    # Placing delta on A_b/B_m, or putting +delta on both dimer orbitals, opens
    # an artificial visible gap at the Delta=0 K point of Fig. 2.
    block[1, 1] += params.delta
    block[2, 2] -= params.delta

    block[0:2, 0:2] += -params.interlayer_potential * np.eye(2, dtype=np.complex128)
    block[4:6, 4:6] += params.interlayer_potential * np.eye(2, dtype=np.complex128)
    block[4:6, 4:6] += params.staggered_potential * np.asarray(
        [[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128
    )
    return block


def moire_coupling_matrix(
    channel: str, params: TMBGParameters, valley: int
) -> np.ndarray:
    valley = _validate_valley(valley)
    phase = complex(
        math.cos(2.0 * math.pi * valley / 3.0), math.sin(2.0 * math.pi * valley / 3.0)
    )
    if channel == "0":
        return np.asarray(
            [[params.omega_prime, params.omega], [params.omega, params.omega_prime]],
            dtype=np.complex128,
        )
    if channel == "+":
        return np.asarray(
            [
                [params.omega_prime, params.omega * phase.conjugate()],
                [params.omega * phase, params.omega_prime],
            ],
            dtype=np.complex128,
        )
    if channel == "-":
        return np.asarray(
            [
                [params.omega_prime, params.omega * phase],
                [params.omega * phase.conjugate(), params.omega_prime],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported moire coupling channel: {channel}")


def build_coupling_table(
    g_vectors: np.ndarray,
    q_vectors: dict[str, complex],
    *,
    valley: int = 1,
) -> tuple[MoireCouplingEntry, ...]:
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    mapping = {_complex_key(complex(gvec)): idx for idx, gvec in enumerate(g_vectors)}
    valley = _validate_valley(valley)
    q0 = complex(valley * q_vectors["0"])

    entries: list[MoireCouplingEntry] = []
    for middle_index, g_middle in enumerate(g_vectors):
        for channel in MOIRE_CHANNELS:
            shift = complex(valley * q_vectors[channel] - q0)
            top_index = mapping.get(_complex_key(complex(g_middle + shift)))
            if top_index is None:
                continue
            entries.append(
                MoireCouplingEntry(
                    channel=channel,
                    middle_index=int(middle_index),
                    top_index=int(top_index),
                )
            )
    return tuple(entries)


def build_hamiltonian(
    k_tilde: complex, lattice: TMBGLattice, params: TMBGParameters, valley: int = 1
) -> np.ndarray:
    valley = _validate_valley(valley)
    n_g = lattice.n_g
    dim = 6 * n_g
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    for ig, gvec in enumerate(lattice.g_vectors):
        sl = slice(6 * ig, 6 * (ig + 1))
        hamiltonian[sl, sl] = build_diagonal_block(
            k_tilde, complex(gvec), lattice, params, valley
        )

    for entry in build_coupling_table(
        lattice.g_vectors, lattice.q_vectors, valley=valley
    ):
        middle_slice = slice(6 * entry.middle_index + 2, 6 * entry.middle_index + 4)
        top_slice = slice(6 * entry.top_index + 4, 6 * entry.top_index + 6)
        coupling = moire_coupling_matrix(entry.channel, params, valley)
        hamiltonian[middle_slice, top_slice] += coupling
        hamiltonian[top_slice, middle_slice] += coupling.conjugate().T

    return hamiltonian


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: TMBGLattice,
    params: TMBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    hmat = build_hamiltonian(k_tilde, lattice, params, valley)
    if not return_eigenvectors:
        evals = eigh(hmat, eigvals_only=True, driver="evr")
        if n_bands is None:
            return np.asarray(evals, dtype=float), None
        return np.asarray(evals[:n_bands], dtype=float), None

    evals, evecs = eigh(hmat, driver="evr")
    if n_bands is None:
        return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
    return np.asarray(evals[:n_bands], dtype=float), np.asarray(
        evecs[:, :n_bands], dtype=np.complex128
    )
