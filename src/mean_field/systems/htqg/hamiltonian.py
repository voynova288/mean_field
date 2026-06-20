from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from mean_field.core.bands import centered_band_indices
from mean_field.core.lattice import build_shift_coupling_edges, complex_lattice_key
from mean_field.core.validation import validate_valley as _validate_valley

from .domains import HTQGDomain, domain_displacements
from .lattice import HTQGLattice, dot_2d
from .params import HTQGParams


@dataclass(frozen=True)
class MoireCouplingEntry:
    """Sparse plane-wave edge for an adjacent-layer moiré tunneling channel.

    ``source_index`` labels the G site in the higher-numbered layer (right
    block of Eq. (1)); ``target_index`` labels the G site in the lower-numbered
    layer (left block).  For all three HTQG interfaces the work-document
    convention is represented as ``G_left = G_right + q_j - q_0``.
    """

    channel: int
    source_index: int
    target_index: int



def dirac_block(kvec: complex, angle_rad: float, vf_ev_nm: float) -> np.ndarray:
    """Single-layer graphene K-valley Dirac block.

    The phase convention implements
        h_l(k) = hbar v [[0, (kx-i ky)e^{-i phi_l}],
                         [(kx+i ky)e^{+i phi_l}, 0]].
    """

    phase = complex(math.cos(float(angle_rad)), math.sin(float(angle_rad)))
    pi = complex(kvec) * phase
    return np.asarray(
        [[0.0, float(vf_ev_nm) * pi.conjugate()], [float(vf_ev_nm) * pi, 0.0]],
        dtype=np.complex128,
    )


def layer_rotation_angle(lattice: HTQGLattice, params: HTQGParams, layer: int) -> float:
    layer = int(layer)
    if layer not in (1, 2, 3, 4):
        raise ValueError(f"Expected layer in (1, 2, 3, 4), got {layer}")
    if not params.include_dirac_rotation:
        return 0.0
    return float((layer - 2.5) * lattice.theta_rad)


def layer_k_offset(lattice: HTQGLattice, layer: int) -> complex:
    """Return Delta K_l = (l-1) q0 for layer ``l``."""

    layer = int(layer)
    if layer not in (1, 2, 3, 4):
        raise ValueError(f"Expected layer in (1, 2, 3, 4), got {layer}")
    return complex((layer - 1) * lattice.q0)


def layer_momentum(k_tilde: complex, gvec: complex, lattice: HTQGLattice, layer: int) -> complex:
    return complex(k_tilde + gvec - layer_k_offset(lattice, layer))


def build_diagonal_block(
    k_tilde: complex,
    gvec: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
    layer: int,
) -> np.ndarray:
    momentum = layer_momentum(k_tilde, gvec, lattice, layer)
    return dirac_block(momentum, layer_rotation_angle(lattice, params, layer), params.vf_ev_nm)


def moire_coupling_matrix(channel: int, params: HTQGParams) -> np.ndarray:
    """Return the K-valley 2x2 BM tunneling matrix T_j."""

    channel = int(channel)
    if channel not in (0, 1, 2):
        raise ValueError(f"Expected channel in (0, 1, 2), got {channel}")
    phase = complex(math.cos(2.0 * math.pi * channel / 3.0), math.sin(2.0 * math.pi * channel / 3.0))
    return params.w_ev * np.asarray(
        [[params.kappa, phase.conjugate()], [phase, params.kappa]],
        dtype=np.complex128,
    )


def qhat_perp(channel: int) -> complex:
    channel = int(channel)
    return complex(math.cos(2.0 * math.pi * channel / 3.0), math.sin(2.0 * math.pi * channel / 3.0))


def build_coupling_table(g_vectors: np.ndarray, q_vectors: np.ndarray) -> tuple[MoireCouplingEntry, ...]:
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    q_vectors = np.asarray(q_vectors, dtype=np.complex128)
    q0 = complex(q_vectors[0])
    shifts = ((channel, complex(q_vectors[channel] - q0)) for channel in (0, 1, 2))
    return tuple(
        MoireCouplingEntry(
            channel=int(edge.channel),
            source_index=int(edge.source_index),
            target_index=int(edge.target_index),
        )
        for edge in build_shift_coupling_edges(g_vectors, shifts, key=complex_lattice_key)
    )


def _phase_for_displacement(qvec: complex, displacement: complex) -> complex:
    phase = dot_2d(qvec, displacement)
    return complex(math.cos(phase), math.sin(phase))


def _orbital_slice(g_index: int, layer: int) -> slice:
    start = 8 * int(g_index) + 2 * (int(layer) - 1)
    return slice(start, start + 2)


def _resolve_domain(
    lattice: HTQGLattice,
    domain: str | HTQGDomain,
    d12: complex | None,
    d34: complex | None,
) -> tuple[complex, complex, HTQGDomain]:
    resolved_domain = domain_displacements(lattice, domain)
    return (
        complex(resolved_domain.d12 if d12 is None else d12),
        complex(resolved_domain.d34 if d34 is None else d34),
        resolved_domain,
    )


def _mdt_factor(
    *,
    channel: int,
    k_tilde: complex,
    source_g: complex,
    target_g: complex,
    source_layer: int,
    target_layer: int,
    lattice: HTQGLattice,
    params: HTQGParams,
) -> complex:
    if params.lambda_mdt_nm == 0.0:
        return 1.0 + 0.0j
    p_source = layer_momentum(k_tilde, source_g, lattice, source_layer)
    p_target = layer_momentum(k_tilde, target_g, lattice, target_layer)
    if params.mdt_momentum == "source":
        p = p_source
    elif params.mdt_momentum == "target":
        p = p_target
    elif params.mdt_momentum == "midpoint":
        p = 0.5 * (p_source + p_target)
    else:  # pragma: no cover; HTQGParams validates the option.
        raise AssertionError(params.mdt_momentum)
    # Kwan-Tan-Devakul's derivation expands the Fourier tunneling amplitude as
    #   W(p) ~= w0 * (1 + lambda_MDT * qhat_perp · p_source).
    # Therefore the momentum-space factor is real in the plane-wave basis.  The
    # real-space shorthand ``(1 + lambda qhat·∇)`` in the HTQG text should not be
    # interpreted as adding an extra ``i`` when the operator is assembled in
    # momentum space.
    return 1.0 + float(params.lambda_mdt_nm) * dot_2d(qhat_perp(channel), p)


def build_hamiltonian(
    k_tilde: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
    d12: complex | None = None,
    d34: complex | None = None,
    coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
) -> np.ndarray:
    """Build the 8*N_G by 8*N_G HTQG continuum Hamiltonian.

    ``valley=-1`` is defined by time reversal, ``H_K'(k)=H_K(-k)^*``.  This
    keeps the K-valley convention explicit and guarantees the C5 spectral check.
    """

    valley = _validate_valley(valley)
    if valley == -1:
        return build_hamiltonian(
            -complex(k_tilde),
            lattice,
            params,
            domain=domain,
            valley=1,
            d12=d12,
            d34=d34,
            coupling_table=coupling_table,
        ).conjugate()

    resolved_d12, resolved_d34, _ = _resolve_domain(lattice, domain, d12, d34)
    entries = build_coupling_table(lattice.g_vectors, lattice.q_vectors) if coupling_table is None else tuple(coupling_table)

    dim = lattice.matrix_dim
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for ig, gvec in enumerate(lattice.g_vectors):
        for layer in (1, 2, 3, 4):
            sl = _orbital_slice(ig, layer)
            hamiltonian[sl, sl] = build_diagonal_block(complex(k_tilde), complex(gvec), lattice, params, layer)

    # Eq. (1): H_{l,l+1} blocks are T(r-d12), T(r), T(r+d34).
    interface_phase_displacements: tuple[tuple[int, int, complex | None, int], ...] = (
        (1, 2, resolved_d12, +1),
        (2, 3, None, 0),
        (3, 4, resolved_d34, -1),
    )
    for left_layer, right_layer, displacement, sign in interface_phase_displacements:
        for entry in entries:
            source_g = complex(lattice.g_vectors[entry.source_index])
            target_g = complex(lattice.g_vectors[entry.target_index])
            block = moire_coupling_matrix(entry.channel, params)
            if displacement is not None:
                phase = _phase_for_displacement(complex(lattice.q_vectors[entry.channel]), complex(displacement))
                if sign < 0:
                    phase = phase.conjugate()
                block = phase * block
            block = block * _mdt_factor(
                channel=entry.channel,
                k_tilde=complex(k_tilde),
                source_g=source_g,
                target_g=target_g,
                source_layer=right_layer,
                target_layer=left_layer,
                lattice=lattice,
                params=params,
            )

            left_slice = _orbital_slice(entry.target_index, left_layer)
            right_slice = _orbital_slice(entry.source_index, right_layer)
            hamiltonian[left_slice, right_slice] += block
            hamiltonian[right_slice, left_slice] += block.conjugate().T

    return hamiltonian


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
    d12: complex | None = None,
    d34: complex | None = None,
    coupling_table: tuple[MoireCouplingEntry, ...] | None = None,
    band_indices: tuple[int, ...] | None = None,
    return_eigenvectors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    hmat = build_hamiltonian(
        k_tilde,
        lattice,
        params,
        domain=domain,
        valley=valley,
        d12=d12,
        d34=d34,
        coupling_table=coupling_table,
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


__all__ = [
    "MoireCouplingEntry",
    "build_coupling_table",
    "build_diagonal_block",
    "build_hamiltonian",
    "centered_band_indices",
    "diagonalize_hamiltonian",
    "dirac_block",
    "layer_k_offset",
    "layer_momentum",
    "layer_rotation_angle",
    "moire_coupling_matrix",
    "qhat_perp",
]
