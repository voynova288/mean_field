from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .lattice import ATMGLattice
from .params import ATMGParameters
from .tbg import TBGCouplingEntry, build_coupling_table, dirac_block, moire_coupling_matrix


def _layer_slice(g_index: int, layer_index: int, n_layers: int) -> slice:
    start = 2 * (int(g_index) * int(n_layers) + int(layer_index))
    return slice(start, start + 2)


def _layer_shift(layer_index: int, lattice: ATMGLattice, valley: int) -> complex:
    return 0.0 + 0.0j if int(layer_index) % 2 == 0 else complex(int(valley) * lattice.q0)


def build_diagonal_block(
    k_tilde: complex,
    gvec: complex,
    lattice: ATMGLattice,
    params: ATMGParameters,
    valley: int,
) -> np.ndarray:
    n_layers = int(params.n_layers)
    block = np.zeros((2 * n_layers, 2 * n_layers), dtype=np.complex128)
    for layer_index in range(n_layers):
        sl = slice(2 * layer_index, 2 * layer_index + 2)
        k_layer = complex(k_tilde + gvec + _layer_shift(layer_index, lattice, valley))
        block[sl, sl] = dirac_block(k_layer, params.vf, valley)
    return block


def build_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    params: ATMGParameters,
    valley: int = 1,
    *,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> np.ndarray:
    n_layers = int(params.n_layers)
    dim = 2 * n_layers * lattice.n_g
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    for ig, gvec in enumerate(lattice.g_vectors):
        block = build_diagonal_block(k_tilde, complex(gvec), lattice, params, valley)
        for layer_index in range(n_layers):
            sl = _layer_slice(ig, layer_index, n_layers)
            local_slice = slice(2 * layer_index, 2 * layer_index + 2)
            hamiltonian[sl, sl] = block[local_slice, local_slice]

    resolved_table = coupling_table
    if resolved_table is None:
        resolved_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)

    for interface_index, w_ab in enumerate(params.resolved_w_ab_couplings):
        lower_layer = interface_index
        upper_layer = interface_index + 1
        w_aa = params.kappa * w_ab
        lower_is_odd = lower_layer % 2 == 0

        for entry in resolved_table:
            odd_slice = _layer_slice(entry.odd_index, lower_layer if lower_is_odd else upper_layer, n_layers)
            even_slice = _layer_slice(entry.even_index, upper_layer if lower_is_odd else lower_layer, n_layers)
            coupling = moire_coupling_matrix(
                entry.channel,
                w_ab=w_ab,
                w_aa=w_aa,
                valley=valley,
            )
            if lower_is_odd:
                lower_slice = odd_slice
                upper_slice = even_slice
                block = coupling
            else:
                lower_slice = even_slice
                upper_slice = odd_slice
                block = coupling.conjugate().T

            hamiltonian[lower_slice, upper_slice] += block
            hamiltonian[upper_slice, lower_slice] += block.conjugate().T

    return hamiltonian


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    params: ATMGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(
        build_hamiltonian(
            k_tilde,
            lattice,
            params,
            valley=valley,
            coupling_table=coupling_table,
        )
    )
    if n_bands is None:
        return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
    return np.asarray(evals[:n_bands], dtype=float), np.asarray(evecs[:, :n_bands], dtype=np.complex128)


__all__ = [
    "build_diagonal_block",
    "build_hamiltonian",
    "diagonalize_hamiltonian",
]
