"""Local and integrated density vertices for projected Kane4 states."""

from __future__ import annotations

import numpy as np

ComplexArray = np.ndarray
FloatArray = np.ndarray

def local_density_vertices(micro_wavefunctions: ComplexArray) -> ComplexArray:
    """Return ``Lambda[k,p,z,a,b] = Phi[k,z]^dagger Phi[p,z]``."""

    phi = np.asarray(micro_wavefunctions, dtype=np.complex128)
    if phi.ndim != 4 or phi.shape[-1] == 0:
        raise ValueError("micro_wavefunctions must have shape (nk, nz, nmicro, nactive)")
    return np.einsum("kzma,pzmb->kpzab", phi.conj(), phi, optimize=True)


def integrated_density_vertices(local_vertices: ComplexArray, z_weights_nm: FloatArray) -> ComplexArray:
    """Integrate local density vertices over z."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    wz = np.asarray(z_weights_nm, dtype=float)
    if local.ndim != 5 or wz.shape != (local.shape[2],):
        raise ValueError("local vertices and z weights have incompatible shapes")
    return np.einsum("kpzab,z->kpab", local, wz, optimize=True)


def density_vertex_reciprocity_error(local_vertices: ComplexArray) -> float:
    """Return max error in ``Lambda[k,p]^dagger = Lambda[p,k]``."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    if local.ndim != 5 or local.shape[0] != local.shape[1] or local.shape[3] != local.shape[4]:
        raise ValueError("local_vertices must have shape (nk, nk, nz, n, n)")
    reverse = np.swapaxes(np.swapaxes(local.conj(), 0, 1), 3, 4)
    return float(np.max(np.abs(local - reverse)))

__all__ = [
    "density_vertex_reciprocity_error",
    "integrated_density_vertices",
    "local_density_vertices",
]
