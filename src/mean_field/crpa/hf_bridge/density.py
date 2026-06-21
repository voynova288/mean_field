from __future__ import annotations

from ._shared import *  # noqa: F401,F403

def half_reference_delta_like(density: np.ndarray) -> np.ndarray:
    """Return the stored-density reference term D_ref = -0.5 I."""

    template = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = template.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {template.shape}")
    out = np.zeros_like(template, dtype=np.complex128)
    diagonal = np.arange(nt)
    out[diagonal, diagonal, :] = -0.5
    if out.shape[2] != nk:
        raise RuntimeError("Internal density-reference construction changed k dimension unexpectedly.")
    return out


def physical_projector_from_delta(density_delta: np.ndarray) -> np.ndarray:
    """Convert the stored full-HF density D = P - 0.5 I to the physical projector P."""

    projector = np.asarray(density_delta, dtype=np.complex128).copy()
    nt, nt_rhs, _nk = projector.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {projector.shape}")
    diagonal = np.arange(nt)
    projector[diagonal, diagonal, :] += 0.5
    return projector


def active_lower_flat_projector_like(
    density: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> np.ndarray:
    """Projector for the CNP lower-flat active reference in current state ordering."""

    template = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = template.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {template.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(
            f"Density dimension {nt} does not match n_spin*n_eta*n_band="
            f"{int(n_spin) * int(n_eta) * int(n_band)}"
        )
    if int(n_band) < 2:
        raise ValueError(f"Expected at least two active bands, got n_band={n_band}")
    out = np.zeros_like(template, dtype=np.complex128)
    indices = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            state = int(indices[ispin, ieta, 0])
            out[state, state, :] = 1.0
    if out.shape[2] != nk:
        raise RuntimeError("Internal CNP reference construction changed k dimension unexpectedly.")
    return out


__all__ = [name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None) == __name__ and not name.startswith('_')]
