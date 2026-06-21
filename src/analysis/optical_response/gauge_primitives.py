from __future__ import annotations

import numpy as np

def energy_difference_inverse(energies: np.ndarray, *, cutoff: float = 1.0e-10) -> np.ndarray:
    """Return ``1/(E_n-E_m)`` with diagonal/small denominators set to zero.

    This follows WannierBerri ``data_K.py::dEig_inv`` in spirit: near-degenerate
    denominators are excluded rather than assigned arbitrary phases.
    """

    e = np.asarray(energies, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    diff = e[:, None] - e[None, :]
    out = np.zeros_like(diff, dtype=float)
    mask = np.abs(diff) > float(cutoff)
    out[mask] = 1.0 / diff[mask]
    return out

def degenerate_band_groups(energies: np.ndarray, *, threshold: float = 1.0e-4) -> list[tuple[int, int]]:
    """Return contiguous WannierBerri-style degenerate band groups.

    Each group is a half-open interval ``(start, stop)``.  Adjacent bands are
    placed in the same group when their energy separation is at most
    ``threshold``.  This mirrors the band-group logic used before
    WannierBerri dynamic calculators call ``trace_ln`` over subspaces.
    """

    e = np.asarray(energies, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    if e.size == 0:
        return []
    borders = [0]
    gaps = np.diff(e)
    borders.extend((np.where(gaps > float(threshold))[0] + 1).tolist())
    borders.append(int(e.size))
    return [(int(a), int(b)) for a, b in zip(borders, borders[1:])]

def group_indices(group: tuple[int, int]) -> np.ndarray:
    """Convert a half-open band group to an integer index array."""

    start, stop = int(group[0]), int(group[1])
    if stop < start:
        raise ValueError(f"Invalid group {group}")
    return np.arange(start, stop, dtype=int)

def random_block_unitary(groups: list[tuple[int, int]], nb: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Return a block-diagonal random unitary for gauge-covariance tests.

    WannierBerri's ``random_gauge`` applies random unitary rotations inside
    degenerate subspaces.  This helper provides the same validation pattern for
    local tests.  Singleton groups receive a random U(1) phase.
    """

    generator = np.random.default_rng(rng)
    unitary = np.eye(int(nb), dtype=np.complex128)
    for start, stop in groups:
        start = int(start)
        stop = int(stop)
        size = stop - start
        if size <= 0:
            raise ValueError(f"Invalid group {(start, stop)}")
        z = generator.normal(size=(size, size)) + 1.0j * generator.normal(size=(size, size))
        q, r = np.linalg.qr(z)
        phases = np.diag(r)
        phases = phases / np.where(np.abs(phases) > 0.0, np.abs(phases), 1.0)
        unitary[start:stop, start:stop] = q * phases.conjugate()[None, :]
    return unitary

def apply_band_gauge_to_matrix(matrix: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply ``X -> G† X G`` to a band-space covariant matrix.

    ``matrix`` has shape ``(nb,nb,*extra)`` and ``gauge`` has shape
    ``(nb,nb)``.  This is the finite-dimensional version of the Hamiltonian
    gauge covariance used by WannierBerri.
    """

    X = np.asarray(matrix, dtype=np.complex128)
    G = np.asarray(gauge, dtype=np.complex128)
    if X.ndim < 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {X.shape}")
    nb = X.shape[0]
    if G.shape != (nb, nb):
        raise ValueError(f"gauge has shape {G.shape}, expected {(nb, nb)}")
    return np.einsum("am,ab...,bn->mn...", G.conjugate(), X, G, optimize=True)

def apply_band_gauge_to_axis_matrix(matrix: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply ``X^a -> G† X^a G`` to ``(ndim,nb,nb,*extra)`` arrays."""

    X = np.asarray(matrix, dtype=np.complex128)
    G = np.asarray(gauge, dtype=np.complex128)
    if X.ndim < 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"matrix must have shape (ndim,nb,nb,*extra), got {X.shape}")
    nb = X.shape[1]
    if G.shape != (nb, nb):
        raise ValueError(f"gauge has shape {G.shape}, expected {(nb, nb)}")
    return np.einsum("am,xab...,bn->xmn...", G.conjugate(), X, G, optimize=True)

def trace_subspace(matrix: np.ndarray, group: tuple[int, int]) -> np.ndarray:
    """Trace a covariant matrix over a band subspace.

    This is the scalar/subspace invariant that survives arbitrary unitary
    rotations inside the group, analogous to WannierBerri's ``trace_ln`` use in
    dynamic calculators.
    """

    X = np.asarray(matrix, dtype=np.complex128)
    idx = group_indices(group)
    if X.ndim < 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {X.shape}")
    return np.trace(X[np.ix_(idx, idx)], axis1=0, axis2=1)

def matrix_in_eigenbasis(eigenvectors: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Transform one or more basis-space operators into Hamiltonian gauge.

    Parameters
    ----------
    eigenvectors:
        Matrix with eigenvectors as columns, shape ``(basis, nb)``.
    operators:
        Either ``(basis,basis)`` or ``(..., basis, basis)``.

    Returns
    -------
    np.ndarray
        Shape ``(..., nb, nb)`` with entries ``<u_n|O|u_m>``.
    """

    u = np.asarray(eigenvectors, dtype=np.complex128)
    if u.ndim != 2:
        raise ValueError(f"eigenvectors must be 2D with eigenvectors as columns, got {u.shape}")
    ops = np.asarray(operators, dtype=np.complex128)
    if ops.shape[-2:] != (u.shape[0], u.shape[0]):
        raise ValueError(f"operator trailing shape {ops.shape[-2:]} incompatible with eigenvectors {u.shape}")
    return np.einsum("bn,...bc,cm->...nm", u.conjugate(), ops, u, optimize=True)

__all__ = [
    "energy_difference_inverse",
    "degenerate_band_groups",
    "group_indices",
    "random_block_unitary",
    "apply_band_gauge_to_matrix",
    "apply_band_gauge_to_axis_matrix",
    "trace_subspace",
    "matrix_in_eigenbasis",
]
