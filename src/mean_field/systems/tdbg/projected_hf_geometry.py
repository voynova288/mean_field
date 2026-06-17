from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.hf import (
    ComponentGroup,
    ProjectedWavefunctionBasis,
    calculate_projected_overlap_between,
    real_space_cell_area_nm2_from_reciprocal,
)
from .lattice import TDBGLattice
from .projected_hf_config import SPIN_LABELS, TDBGProjectedWindow, VALLEY_SEQUENCE
from .projected_hf_state import TDBGProjectedHFData
from .topology import translation_srcmap

def tdbg_band_window_indices(matrix_dim: int, window: TDBGProjectedWindow | str = "two_flat") -> tuple[int, ...]:
    if isinstance(window, str):
        window = TDBGProjectedWindow(name=window)
    if window.band_indices is not None:
        indices = tuple(int(v) for v in window.band_indices)
        if not indices:
            raise ValueError("Explicit TDBG projected-HF band window cannot be empty")
        return indices

    name = window.name.strip().lower().replace("-", "_")
    aliases = {"isolated_cb": 1, "cb": 1, "two_flat": 2, "central2": 2, "central4": 4, "central6": 6}
    if name not in aliases:
        raise ValueError(f"Unsupported TDBG projected-HF window {window.name!r}")
    count = int(aliases[name])
    center = int(matrix_dim) // 2
    if count == 1:
        indices = (center,)
    else:
        start = center - count // 2
        indices = tuple(range(start, start + count))
    if min(indices) < 0 or max(indices) >= int(matrix_dim):
        raise ValueError(f"Band window {indices} is outside matrix_dim={matrix_dim}")
    return indices


def tdbg_moire_area_nm2(lattice: TDBGLattice) -> float:
    return real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)


def _shift_table(lattice: TDBGLattice, g_shells: int | None) -> tuple[tuple[tuple[int, int], ...], np.ndarray, tuple[np.ndarray, ...]]:
    shells = int(math.ceil(2.0 * lattice.cut) + 1) if g_shells is None else int(g_shells)
    shifts: list[tuple[int, int]] = []
    gvecs: list[complex] = []
    srcmaps: list[np.ndarray] = []
    for m in range(-shells, shells + 1):
        for n in range(-shells, shells + 1):
            gvec = m * lattice.g_m1 + n * lattice.g_m2
            src = translation_srcmap(lattice, gvec)
            if np.any(src >= 0):
                shifts.append((int(m), int(n)))
                gvecs.append(complex(gvec))
                srcmaps.append(np.asarray(src, dtype=int))
    return tuple(shifts), np.asarray(gvecs, dtype=np.complex128), tuple(srcmaps)



@dataclass(frozen=True)
class _TDBGQSiteEmbedding:
    grid_shape: tuple[int, int]
    local_basis_size: int
    basis_indices: np.ndarray  # (n_q, 4) indices into ProjectedWavefunctionBasis basis axis.

def tdbg_embedded_component_groups() -> tuple[ComponentGroup, ...]:
    """Return groups for the 8-component embedded projected-HF local basis.

    The embedded core/HF basis uses local index ``4 * sector + alpha`` with
    ``alpha=(A1, B1, A2, B2)``.  This is distinct from the full Hamiltonian
    basis index ``4 * q_site + alpha`` exposed by :class:`TDBGModel`.
    """

    return (
        ComponentGroup("sector_0", np.asarray([0, 1, 2, 3], dtype=int)),
        ComponentGroup("sector_1", np.asarray([4, 5, 6, 7], dtype=int)),
        ComponentGroup("layer_0", np.asarray([0, 1], dtype=int)),
        ComponentGroup("layer_1", np.asarray([2, 3], dtype=int)),
        ComponentGroup("layer_2", np.asarray([4, 5], dtype=int)),
        ComponentGroup("layer_3", np.asarray([6, 7], dtype=int)),
        ComponentGroup("sublattice_A", np.asarray([0, 2, 4, 6], dtype=int)),
        ComponentGroup("sublattice_B", np.asarray([1, 3, 5, 7], dtype=int)),
    )


def _tdbg_q_site_embedding(lattice: TDBGLattice) -> _TDBGQSiteEmbedding:
    """Embed TDBG's finite q-site disk into a rectangular core/hf basis grid.

    The generic core overlap code shifts rectangular reciprocal grids with
    zero-fill boundary conditions. TDBG's q-sites are a finite disk labelled by
    moire reciprocal coordinates plus a sector index. We embed sector `l=0,1`
    and local component `alpha=0..3` into an eight-component local basis on a
    rectangular `(g_m1, g_m2)` coordinate grid, so the trusted core overlap
    helpers can be reused without changing TDBG's finite-cutoff physics.
    """

    q_sites = np.asarray(lattice.q_sites, dtype=float)
    if q_sites.ndim != 2 or q_sites.shape[1] < 3:
        raise ValueError(f"Expected q_sites with columns (qx, qy, sector), got {q_sites.shape}")
    q0 = complex(np.asarray(lattice.q_complex, dtype=np.complex128)[0])
    g1 = complex(lattice.g_m1)
    g2 = complex(lattice.g_m2)
    matrix = np.asarray([[g1.real, g2.real], [g1.imag, g2.imag]], dtype=float)
    coords: list[tuple[int, int, int]] = []
    for site in q_sites:
        sector = int(round(float(site[2])))
        if sector not in {0, 1}:
            raise ValueError(f"TDBG q-site sector must be 0 or 1, got {sector}")
        vector = complex(float(site[0]), float(site[1])) + sector * q0
        coeff = np.linalg.solve(matrix, np.asarray([vector.real, vector.imag], dtype=float))
        axis0 = int(round(float(coeff[0])))
        axis1 = int(round(float(coeff[1])))
        if not np.allclose(coeff, (axis0, axis1), atol=1.0e-8):
            raise ValueError(f"Could not map q-site {site.tolist()} to integer moire coordinates: {coeff}")
        coords.append((axis0, axis1, sector))
    axis0_values = [item[0] for item in coords]
    axis1_values = [item[1] for item in coords]
    min0, max0 = min(axis0_values), max(axis0_values)
    min1, max1 = min(axis1_values), max(axis1_values)
    nx = max0 - min0 + 1
    ny = max1 - min1 + 1
    local_basis_size = 8
    basis_indices = np.zeros((q_sites.shape[0], 4), dtype=int)
    for iq, (axis0, axis1, sector) in enumerate(coords):
        x = axis0 - min0
        y = axis1 - min1
        for alpha in range(4):
            local = 4 * sector + alpha
            basis_indices[iq, alpha] = local + local_basis_size * (x + nx * y)
    return _TDBGQSiteEmbedding(grid_shape=(int(nx), int(ny)), local_basis_size=local_basis_size, basis_indices=basis_indices)

def _tdbg_core_order_permutation(data: TDBGProjectedHFData) -> np.ndarray:
    permutation = np.zeros(data.nt, dtype=int)
    n_spin = len(SPIN_LABELS)
    n_valley = len(VALLEY_SEQUENCE)
    for label in data.labels:
        spin_index = SPIN_LABELS.index(label.spin)
        valley_index = VALLEY_SEQUENCE.index(int(label.valley))
        core_index = spin_index + n_spin * (valley_index + n_valley * int(label.band_position))
        permutation[int(label.index)] = int(core_index)
    return permutation

def _tdbg_projected_wavefunction_basis(data: TDBGProjectedHFData, wavefunctions: np.ndarray, *, name: str) -> ProjectedWavefunctionBasis:
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    if wavefunctions.ndim != 4 or wavefunctions.shape[0] != data.nt or wavefunctions.shape[2:] != (data.model.lattice.n_q, 4):
        raise ValueError(
            f"Expected TDBG wavefunctions shape (nt, nk, n_q, 4) with nt={data.nt}, n_q={data.model.lattice.n_q}; "
            f"got {wavefunctions.shape}"
        )
    nk = int(wavefunctions.shape[1])
    embedding = _tdbg_q_site_embedding(data.model.lattice)
    basis_dim = embedding.local_basis_size * embedding.grid_shape[0] * embedding.grid_shape[1]
    core_wavefunctions = np.zeros((basis_dim, data.n_band, len(VALLEY_SEQUENCE), nk), dtype=np.complex128)
    assigned: set[tuple[int, int]] = set()
    for label in data.labels:
        valley_index = VALLEY_SEQUENCE.index(int(label.valley))
        key = (int(label.band_position), valley_index)
        if key in assigned:
            continue
        assigned.add(key)
        values = wavefunctions[int(label.index)]
        for alpha in range(4):
            core_wavefunctions[embedding.basis_indices[:, alpha], int(label.band_position), valley_index, :] = values[:, :, alpha].T
    return ProjectedWavefunctionBasis(
        core_wavefunctions,
        embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=embedding.local_basis_size,
        name=name,
        boundary_mode="zero_fill",
        component_groups=tdbg_embedded_component_groups(),
    )

def _tdbg_total_overlap_from_bases(
    data: TDBGProjectedHFData,
    target_basis: ProjectedWavefunctionBasis,
    source_basis: ProjectedWavefunctionBasis,
    shift: tuple[int, int],
) -> np.ndarray:
    overlap_core = calculate_projected_overlap_between(target_basis, source_basis, int(shift[0]), int(shift[1]))
    permutation = _tdbg_core_order_permutation(data)
    return overlap_core[permutation, :, :, :][:, :, permutation, :]

def _tdbg_total_overlap_between(
    data: TDBGProjectedHFData,
    target_wavefunctions: np.ndarray,
    source_wavefunctions: np.ndarray,
    shift: tuple[int, int],
    *,
    target_name: str = "tdbg-target",
    source_name: str = "tdbg-source",
) -> np.ndarray:
    target_basis = _tdbg_projected_wavefunction_basis(data, target_wavefunctions, name=target_name)
    source_basis = _tdbg_projected_wavefunction_basis(data, source_wavefunctions, name=source_name)
    return _tdbg_total_overlap_from_bases(data, target_basis, source_basis, shift)

__all__ = [
    "_TDBGQSiteEmbedding",
    "_shift_table",
    "_tdbg_core_order_permutation",
    "_tdbg_projected_wavefunction_basis",
    "_tdbg_q_site_embedding",
    "_tdbg_total_overlap_between",
    "_tdbg_total_overlap_from_bases",
    "tdbg_band_window_indices",
    "tdbg_embedded_component_groups",
    "tdbg_moire_area_nm2",
]
