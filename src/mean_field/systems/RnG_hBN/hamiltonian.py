from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.linalg import eigh

from mean_field.core.validation import validate_valley as _validate_valley

from .lattice import RLGhBNLattice
from .params import RLGhBNParams, VALID_VALLEYS


MOIRE_DELTAS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, -1))


@dataclass(frozen=True)
class MoireCouplingEntry:
    source_g_index: int
    target_g_index: int
    channel: int



def _momentum_complex(momentum: complex | Iterable[float] | np.ndarray) -> complex:
    if isinstance(momentum, complex):
        return complex(momentum)
    array = np.asarray(momentum, dtype=float)
    if array.shape != (2,):
        return complex(momentum)  # type: ignore[arg-type]
    return complex(float(array[0]), float(array[1]))


def basis_index(g_index: int, layer: int, sublattice: int, params: RLGhBNParams) -> int:
    return int((int(g_index) * params.layer_count + int(layer)) * 2 + int(sublattice))


def layer_slice(g_index: int, layer: int, params: RLGhBNParams) -> slice:
    start = basis_index(g_index, layer, 0, params)
    return slice(start, start + 2)


def dirac_block(momentum: complex | Iterable[float] | np.ndarray, layer: int, params: RLGhBNParams) -> np.ndarray:
    p_plus = _momentum_complex(momentum)
    p_minus = p_plus.conjugate()
    centered_layer = float(layer) - 0.5 * float(params.layer_count - 1)
    onsite = float(params.isp_mev) * abs(centered_layer) + float(params.displacement_field_mev) * centered_layer
    return np.asarray(
        [
            [onsite, params.fermi_velocity_mev_nm * p_minus],
            [params.fermi_velocity_mev_nm * p_plus, onsite],
        ],
        dtype=np.complex128,
    )


def interlayer_coupling(momentum: complex | Iterable[float] | np.ndarray, params: RLGhBNParams) -> tuple[np.ndarray, np.ndarray]:
    p_plus = _momentum_complex(momentum)
    p_minus = p_plus.conjugate()
    nearest = -np.asarray(
        [
            [params.v4_mev_nm * p_plus, -params.t1_mev],
            [params.v3_mev_nm * p_minus, params.v4_mev_nm * p_plus],
        ],
        dtype=np.complex128,
    )
    next_nearest = np.asarray([[0.0, 0.0], [params.t2_mev, 0.0]], dtype=np.complex128)
    return nearest, next_nearest


def build_rlg_block(momentum: complex | Iterable[float] | np.ndarray, params: RLGhBNParams) -> np.ndarray:
    dim = params.internal_dim
    block = np.zeros((dim, dim), dtype=np.complex128)
    for layer in range(params.layer_count):
        sl = slice(2 * layer, 2 * layer + 2)
        block[sl, sl] = dirac_block(momentum, layer, params)

    nearest, next_nearest = interlayer_coupling(momentum, params)
    for layer in range(params.layer_count - 1):
        lower = slice(2 * layer, 2 * layer + 2)
        upper = slice(2 * (layer + 1), 2 * (layer + 1) + 2)
        block[upper, lower] = nearest
        block[lower, upper] = nearest.conjugate().T
    for layer in range(params.layer_count - 2):
        lower = slice(2 * layer, 2 * layer + 2)
        upper2 = slice(2 * (layer + 2), 2 * (layer + 2) + 2)
        block[upper2, lower] = next_nearest
        block[lower, upper2] = next_nearest.conjugate().T
    return block


def moire_coupling_matrix(channel: int, params: RLGhBNParams) -> np.ndarray:
    channel = int(channel)
    if channel not in (1, 2, 3):
        raise ValueError(f"Expected moire channel 1, 2, or 3, got {channel}")
    omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))
    return np.asarray(
        [
            [1.0, omega ** (-channel)],
            [omega ** (channel + 1), omega],
        ],
        dtype=np.complex128,
    )


def moire_potential(g_i: tuple[int, int] | np.ndarray, g_j: tuple[int, int] | np.ndarray, params: RLGhBNParams) -> np.ndarray:
    gi = np.asarray(g_i, dtype=int)
    gj = np.asarray(g_j, dtype=int)
    if gi.shape != (2,) or gj.shape != (2,):
        raise ValueError(f"Expected integer G coordinates with shape (2,), got {gi.shape} and {gj.shape}")

    delta = (int(gi[0] - gj[0]), int(gi[1] - gj[1]))
    if delta == (0, 0):
        return float(params.moire_v0_mev) * np.eye(2, dtype=np.complex128)
    if delta in MOIRE_DELTAS:
        channel = MOIRE_DELTAS.index(delta) + 1
        return float(params.moire_v1_mev) * np.exp(1.0j * params.moire_phase_rad) * moire_coupling_matrix(channel, params)
    neg_delta = (-delta[0], -delta[1])
    if neg_delta in MOIRE_DELTAS:
        channel = MOIRE_DELTAS.index(neg_delta) + 1
        return (
            float(params.moire_v1_mev)
            * np.exp(-1.0j * params.moire_phase_rad)
            * moire_coupling_matrix(channel, params).conjugate().T
        )
    return np.zeros((2, 2), dtype=np.complex128)


def build_coupling_table(lattice: RLGhBNLattice) -> tuple[MoireCouplingEntry, ...]:
    lookup = lattice.g_index_lookup()
    entries: list[MoireCouplingEntry] = []
    for source_index, source_coords in enumerate(lattice.g_indices):
        source = (int(source_coords[0]), int(source_coords[1]))
        for channel, delta in enumerate(MOIRE_DELTAS, start=1):
            target = (source[0] + delta[0], source[1] + delta[1])
            if target in lookup:
                entries.append(
                    MoireCouplingEntry(
                        source_g_index=int(source_index),
                        target_g_index=int(lookup[target]),
                        channel=int(channel),
                    )
                )
    return tuple(entries)


def build_hamiltonian(
    k_tilde: complex,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    if valley == -1:
        return build_hamiltonian(-complex(k_tilde), lattice, params, valley=1).conjugate()

    dim = int(2 * params.layer_count * lattice.n_g)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for g_index, g_vector in enumerate(lattice.g_vectors):
        sl = slice(2 * params.layer_count * g_index, 2 * params.layer_count * (g_index + 1))
        hamiltonian[sl, sl] = build_rlg_block(complex(k_tilde + g_vector), params)

    for row_g_index, row_coords in enumerate(lattice.g_indices):
        row_slice = layer_slice(row_g_index, 0, params)
        for col_g_index, col_coords in enumerate(lattice.g_indices):
            potential = moire_potential(row_coords, col_coords, params)
            if np.any(np.abs(potential) > 0.0):
                col_slice = layer_slice(col_g_index, 0, params)
                hamiltonian[row_slice, col_slice] += potential
    return hamiltonian


def hamiltonian_dimension(lattice: RLGhBNLattice, params: RLGhBNParams) -> int:
    return int(2 * params.layer_count * lattice.n_g)


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
    n_bands: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if n_bands is None or int(n_bands) >= hamiltonian.shape[0]:
        evals, evecs = eigh(hamiltonian)
    else:
        evals, evecs = eigh(hamiltonian, subset_by_index=[0, int(n_bands) - 1])
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)


def valence_band_count(lattice: RLGhBNLattice, params: RLGhBNParams) -> int:
    return int(params.layer_count * lattice.n_g)


def flat_band_indices(lattice: RLGhBNLattice, params: RLGhBNParams) -> tuple[int, int]:
    valence_count = valence_band_count(lattice, params)
    return valence_count - 1, valence_count


__all__ = [
    "MOIRE_DELTAS",
    "MoireCouplingEntry",
    "basis_index",
    "build_coupling_table",
    "build_hamiltonian",
    "build_rlg_block",
    "diagonalize_hamiltonian",
    "dirac_block",
    "flat_band_indices",
    "hamiltonian_dimension",
    "interlayer_coupling",
    "layer_slice",
    "moire_coupling_matrix",
    "moire_potential",
    "valence_band_count",
]
