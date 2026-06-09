from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from mean_field.core.validation import validate_valley as _validate_valley

from .lattice import ATMGLattice


VALID_VALLEYS = (-1, 1)
MOIRE_CHANNELS = ("0", "+", "-")



def _complex_key(value: complex, *, digits: int = 12) -> tuple[float, float]:
    return (round(float(value.real), digits), round(float(value.imag), digits))


def _valley_pi(kvec: complex, valley: int) -> tuple[complex, complex]:
    kvec = complex(kvec)
    valley = _validate_valley(valley)
    if valley == 1:
        return kvec, kvec.conjugate()
    return -kvec.conjugate(), -kvec


def dirac_block(kvec: complex, vf: float, valley: int) -> np.ndarray:
    pi, pi_dag = _valley_pi(kvec, valley)
    return np.asarray([[0.0, vf * pi_dag], [vf * pi, 0.0]], dtype=np.complex128)


def moire_coupling_matrix(
    channel: str,
    *,
    w_ab: float,
    w_aa: float,
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    phase = complex(math.cos(2.0 * math.pi * valley / 3.0), math.sin(2.0 * math.pi * valley / 3.0))
    if channel == "0":
        return np.asarray([[w_aa, w_ab], [w_ab, w_aa]], dtype=np.complex128)
    if channel == "+":
        return np.asarray(
            [
                [w_aa, w_ab * phase.conjugate()],
                [w_ab * phase, w_aa],
            ],
            dtype=np.complex128,
        )
    if channel == "-":
        return np.asarray(
            [
                [w_aa, w_ab * phase],
                [w_ab * phase.conjugate(), w_aa],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported moire coupling channel: {channel}")


@dataclass(frozen=True)
class TBGCouplingEntry:
    channel: str
    odd_index: int
    even_index: int


def build_coupling_table(
    g_vectors: np.ndarray,
    q_vectors: dict[str, complex],
    *,
    valley: int = 1,
) -> tuple[TBGCouplingEntry, ...]:
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    valley = _validate_valley(valley)
    mapping = {_complex_key(complex(gvec)): idx for idx, gvec in enumerate(g_vectors)}

    entries: list[TBGCouplingEntry] = []
    for odd_index, g_odd in enumerate(g_vectors):
        for channel in MOIRE_CHANNELS:
            shift = complex(valley * (q_vectors[channel] - q_vectors["0"]))
            even_index = mapping.get(_complex_key(complex(g_odd + shift)))
            if even_index is None:
                continue
            entries.append(
                TBGCouplingEntry(
                    channel=str(channel),
                    odd_index=int(odd_index),
                    even_index=int(even_index),
                )
            )
    return tuple(entries)


def build_monolayer_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    *,
    vf: float,
    valley: int = 1,
    sector: str = "odd",
) -> np.ndarray:
    valley = _validate_valley(valley)
    shift = 0.0 + 0.0j if sector == "odd" else complex(valley * lattice.q0)
    dim = 2 * lattice.n_g
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for ig, gvec in enumerate(lattice.g_vectors):
        sl = slice(2 * ig, 2 * (ig + 1))
        hamiltonian[sl, sl] = dirac_block(complex(k_tilde + gvec + shift), vf, valley)
    return hamiltonian


def build_tbg_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    *,
    lambda_coupling: float,
    kappa: float,
    vf: float,
    valley: int = 1,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> np.ndarray:
    valley = _validate_valley(valley)
    n_g = lattice.n_g
    dim = 4 * n_g
    w_ab = float(lambda_coupling) * float(vf) * float(lattice.graphene_k_mag) * float(lattice.theta_rad)
    w_aa = float(kappa) * w_ab
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    for ig, gvec in enumerate(lattice.g_vectors):
        bottom_slice = slice(4 * ig, 4 * ig + 2)
        top_slice = slice(4 * ig + 2, 4 * ig + 4)
        hamiltonian[bottom_slice, bottom_slice] = dirac_block(complex(k_tilde + gvec), vf, valley)
        hamiltonian[top_slice, top_slice] = dirac_block(complex(k_tilde + gvec + valley * lattice.q0), vf, valley)

    resolved_table = coupling_table
    if resolved_table is None:
        resolved_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    for entry in resolved_table:
        bottom_slice = slice(4 * entry.odd_index, 4 * entry.odd_index + 2)
        top_slice = slice(4 * entry.even_index + 2, 4 * entry.even_index + 4)
        coupling = moire_coupling_matrix(
            entry.channel,
            w_ab=w_ab,
            w_aa=w_aa,
            valley=valley,
        )
        hamiltonian[bottom_slice, top_slice] += coupling
        hamiltonian[top_slice, bottom_slice] += coupling.conjugate().T

    return hamiltonian


def diagonalize_tbg_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    *,
    lambda_coupling: float,
    kappa: float,
    vf: float,
    valley: int = 1,
    n_bands: int | None = None,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(
        build_tbg_hamiltonian(
            k_tilde,
            lattice,
            lambda_coupling=lambda_coupling,
            kappa=kappa,
            vf=vf,
            valley=valley,
            coupling_table=coupling_table,
        )
    )
    if n_bands is None:
        return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
    return np.asarray(evals[:n_bands], dtype=float), np.asarray(evecs[:, :n_bands], dtype=np.complex128)


__all__ = [
    "MOIRE_CHANNELS",
    "TBGCouplingEntry",
    "VALID_VALLEYS",
    "build_coupling_table",
    "build_monolayer_hamiltonian",
    "build_tbg_hamiltonian",
    "diagonalize_tbg_hamiltonian",
    "dirac_block",
    "moire_coupling_matrix",
]
