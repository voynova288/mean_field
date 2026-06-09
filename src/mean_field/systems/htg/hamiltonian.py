from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from mean_field.core.lattice import build_shift_coupling_edges, complex_lattice_key
from mean_field.core.validation import validate_valley as _validate_valley

from .lattice import HTGLattice, dot_2d
from .params import HTGParams, VALID_VALLEYS


@dataclass(frozen=True)
class MoireCouplingEntry:
    channel: int
    middle_index: int
    outer_index: int



def centered_band_indices(matrix_dim: int, band_count: int) -> tuple[int, ...]:
    if band_count <= 0:
        raise ValueError(f"Expected positive band_count, got {band_count}")
    if band_count > matrix_dim:
        raise ValueError(f"band_count={band_count} exceeds matrix_dim={matrix_dim}")
    center = matrix_dim // 2
    lower = max(0, center - band_count // 2)
    upper = min(matrix_dim, lower + band_count)
    lower = max(0, upper - band_count)
    return tuple(range(lower, upper))


def _rotated_complex_momentum(kvec: complex, angle_rad: float) -> complex:
    return complex(kvec) * complex(math.cos(angle_rad), -math.sin(angle_rad))


def _valley_pi(kvec: complex, angle_rad: float, valley: int) -> tuple[complex, complex]:
    q = _rotated_complex_momentum(kvec, angle_rad)
    if _validate_valley(valley) == 1:
        return complex(q), complex(q.conjugate())
    return complex(-q.conjugate()), complex(-q)


def dirac_block(kvec: complex, angle_rad: float, vf_ev_nm: float, valley: int) -> np.ndarray:
    pi, pi_dag = _valley_pi(kvec, angle_rad, valley)
    return np.asarray(
        [[0.0, vf_ev_nm * pi_dag], [vf_ev_nm * pi, 0.0]],
        dtype=np.complex128,
    )


def layer_rotation_angle(lattice: HTGLattice, params: HTGParams, layer: int) -> float:
    zeta = lattice.theta_rad if params.zeta_rad is None else float(params.zeta_rad)
    if layer == 1:
        return float(zeta)
    if layer == 2:
        return 0.0
    if layer == 3:
        return float(-zeta)
    raise ValueError(f"Expected layer in (1, 2, 3), got {layer}")


def layer_k_offset(lattice: HTGLattice, valley: int, layer: int) -> complex:
    valley = _validate_valley(valley)
    if layer == 1:
        return complex(valley * lattice.q0)
    if layer == 2:
        return 0.0 + 0.0j
    if layer == 3:
        return complex(-valley * lattice.q0)
    raise ValueError(f"Expected layer in (1, 2, 3), got {layer}")


def build_diagonal_block(
    k_tilde: complex,
    gvec: complex,
    lattice: HTGLattice,
    params: HTGParams,
    valley: int,
    layer: int,
) -> np.ndarray:
    momentum = complex(k_tilde + gvec + layer_k_offset(lattice, valley, layer))
    return dirac_block(
        momentum,
        layer_rotation_angle(lattice, params, layer),
        params.vf_ev_nm,
        valley,
    )


def moire_coupling_matrix(channel: int, params: HTGParams, valley: int = 1) -> np.ndarray:
    valley = _validate_valley(valley)
    if channel not in (0, 1, 2):
        raise ValueError(f"Expected channel in (0, 1, 2), got {channel}")
    phase = complex(
        math.cos(2.0 * math.pi * valley * channel / 3.0),
        math.sin(2.0 * math.pi * valley * channel / 3.0),
    )
    return params.w_ev * np.asarray(
        [
            [params.kappa, phase.conjugate()],
            [phase, params.kappa],
        ],
        dtype=np.complex128,
    )


def build_coupling_table(
    g_vectors: np.ndarray,
    q_vectors: np.ndarray,
    *,
    valley: int = 1,
    shift_sign: int = 1,
) -> tuple[MoireCouplingEntry, ...]:
    if int(shift_sign) not in (-1, 1):
        raise ValueError(f"shift_sign must be +/-1, got {shift_sign}")
    valley = _validate_valley(valley)
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    q_vectors = np.asarray(q_vectors, dtype=np.complex128)
    q0 = complex(q_vectors[0])
    signed_shift = int(shift_sign) * int(valley)
    channel_shifts = ((channel, complex(signed_shift * (q_vectors[channel] - q0))) for channel in (0, 1, 2))
    return tuple(
        MoireCouplingEntry(
            channel=int(edge.channel),
            middle_index=int(edge.source_index),
            outer_index=int(edge.target_index),
        )
        for edge in build_shift_coupling_edges(
            g_vectors,
            channel_shifts,
            key=complex_lattice_key,
        )
    )


def default_displacements(lattice: HTGLattice, *, domain: str = "h") -> tuple[complex, complex]:
    if domain == "h":
        return complex(-lattice.delta), complex(lattice.delta)
    if domain in {"hbar", "anti-h"}:
        return complex(lattice.delta), complex(-lattice.delta)
    raise ValueError(f"Unsupported HTG domain {domain!r}; expected 'h' or 'hbar'.")


def _phase_for_displacement(qvec: complex, displacement: complex, valley: int) -> complex:
    phase = float(valley) * dot_2d(qvec, displacement)
    return complex(math.cos(phase), math.sin(phase))


def _orbital_slice(g_index: int, layer: int) -> slice:
    start = 6 * int(g_index) + 2 * (int(layer) - 1)
    return slice(start, start + 2)


def build_hamiltonian(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    top_coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
    bottom_coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
) -> np.ndarray:
    valley = _validate_valley(valley)
    if d_top is None or d_bot is None:
        default_top, default_bot = default_displacements(lattice, domain="h")
        if d_top is None:
            d_top = default_top
        if d_bot is None:
            d_bot = default_bot

    dim = lattice.matrix_dim
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for ig, gvec in enumerate(lattice.g_vectors):
        for layer in (1, 2, 3):
            sl = _orbital_slice(ig, layer)
            hamiltonian[sl, sl] = build_diagonal_block(
                k_tilde,
                complex(gvec),
                lattice,
                params,
                valley,
                layer,
            )

    # Coupling tables depend only on lattice geometry, valley, and interface
    # orientation.  Path/grid helpers can build them once and reuse them for
    # every k-point in the sweep.
    top_entries = (
        build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
        if top_coupling_table is None
        else tuple(top_coupling_table)
    )
    bottom_entries = (
        build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)
        if bottom_coupling_table is None
        else tuple(bottom_coupling_table)
    )
    top_channel_couplings = tuple(
        _phase_for_displacement(complex(lattice.q_vectors[channel]), complex(d_top), valley)
        * moire_coupling_matrix(channel, params, valley=valley)
        for channel in (0, 1, 2)
    )
    bottom_channel_couplings = tuple(
        _phase_for_displacement(complex(lattice.q_vectors[channel]), complex(d_bot), valley)
        * moire_coupling_matrix(channel, params, valley=valley)
        for channel in (0, 1, 2)
    )

    for entry in top_entries:
        top_slice = _orbital_slice(entry.outer_index, 1)
        middle_slice = _orbital_slice(entry.middle_index, 2)

        top_coupling = top_channel_couplings[entry.channel]

        hamiltonian[top_slice, middle_slice] += top_coupling
        hamiltonian[middle_slice, top_slice] += top_coupling.conjugate().T

    for entry in bottom_entries:
        middle_slice = _orbital_slice(entry.middle_index, 2)
        bottom_slice = _orbital_slice(entry.outer_index, 3)

        bottom_coupling = bottom_channel_couplings[entry.channel]

        hamiltonian[middle_slice, bottom_slice] += bottom_coupling
        hamiltonian[bottom_slice, middle_slice] += bottom_coupling.conjugate().T

    return hamiltonian


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    top_coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
    bottom_coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
    band_indices: tuple[int, ...] | None = None,
    return_eigenvectors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    hmat = build_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        top_coupling_table=top_coupling_table,
        bottom_coupling_table=bottom_coupling_table,
    )
    subset_by_index = None
    if band_indices is not None:
        normalized = tuple(int(index) for index in band_indices)
        if not normalized:
            raise ValueError("band_indices must not be empty")
        if min(normalized) < 0 or max(normalized) >= lattice.matrix_dim:
            raise ValueError(f"band_indices out of range for matrix_dim={lattice.matrix_dim}: {normalized}")
        expected = tuple(range(min(normalized), max(normalized) + 1))
        if normalized != expected:
            raise ValueError("band_indices must be a contiguous sorted range for scipy.linalg.eigh")
        subset_by_index = (min(normalized), max(normalized))

    if not return_eigenvectors:
        evals = eigh(hmat, eigvals_only=True, subset_by_index=subset_by_index, driver="evr")
        return np.asarray(evals, dtype=float), None

    evals, evecs = eigh(hmat, subset_by_index=subset_by_index, driver="evr")
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
