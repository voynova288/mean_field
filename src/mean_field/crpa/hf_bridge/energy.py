from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from .density import physical_projector_from_delta
from .split_scheme import crpa_active_density_from_delta

def crpa_split_energy_functional(interaction_hamiltonian: np.ndarray, h0: np.ndarray, density_delta: np.ndarray) -> float:
    """Energy functional for H = h_BM + DeltaH_I^bare + Sigma_cRPA[P]."""

    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    total = np.einsum("abk,abk->", h0, projector, optimize=True)
    total += 0.5 * np.einsum("abk,abk->", interaction_hamiltonian, projector, optimize=True)
    return float(total.real / float(nk))


def split_oda_parameter(
    state_obj,
    delta_density: np.ndarray,
    *,
    delta_h: np.ndarray,
    interaction_h: np.ndarray | None = None,
) -> float:
    """ODA parameter for split Hamiltonians using D = P - 0.5 I storage.

    The split Zhang-style functional is quadratic in the physical projector
    P, while the solver stores the shifted density D.  The last bilinear term
    must therefore contract ``delta_h`` with P, not with D.  Using the generic
    Wang ODA formula with a split ``h0`` would miss the +0.5 I reference term
    and can make the no-cRPA Zhang/Wang trajectories diverge.
    """

    delta = np.asarray(delta_density, dtype=np.complex128)
    delta_interaction = np.asarray(delta_h, dtype=np.complex128)
    active_interaction = (
        np.asarray(state_obj.hamiltonian - state_obj.h0, dtype=np.complex128)
        if interaction_h is None
        else np.asarray(interaction_h, dtype=np.complex128)
    )
    active_projector = physical_projector_from_delta(state_obj.density)

    a = np.einsum("abk,abk->", delta, delta_interaction, optimize=True)
    b = np.einsum("abk,abk->", delta, state_obj.h0, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", delta, active_interaction, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", active_projector, delta_interaction, optimize=True)
    a = float(a.real / state_obj.nk)
    b = float(b.real / state_obj.nk)

    if abs(a) < 1e-15:
        return 1.0 if b < 0.0 else 0.0
    lambda0 = -b / a
    if a > 0.0:
        if lambda0 <= 0.0:
            return 0.0
        if lambda0 < 1.0:
            return float(lambda0)
        return 1.0
    if lambda0 <= 0.5:
        return 1.0
    return 0.0


def crpa_hf_energy_components(
    h0: np.ndarray,
    density_delta: np.ndarray,
    hartree_hamiltonian: np.ndarray,
    fock_hamiltonian: np.ndarray,
) -> dict[str, float]:
    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    e_band = np.einsum("abk,abk->", h0, projector, optimize=True).real / float(nk)
    e_hartree = 0.5 * np.einsum("abk,abk->", hartree_hamiltonian, projector, optimize=True).real / float(nk)
    e_fock = 0.5 * np.einsum("abk,abk->", fock_hamiltonian, projector, optimize=True).real / float(nk)
    return {
        "E_band": float(e_band),
        "E_Hartree": float(e_hartree),
        "E_Fock": float(e_fock),
        "E_total": float(e_band + e_hartree + e_fock),
    }


def crpa_hartree_delta_fock_projector_energy_components(
    h0: np.ndarray,
    density_delta: np.ndarray,
    hartree_hamiltonian: np.ndarray,
    fock_hamiltonian: np.ndarray,
) -> dict[str, float]:
    projector = physical_projector_from_delta(density_delta)
    nk = int(projector.shape[2])
    e_band = np.einsum("abk,abk->", h0, projector, optimize=True).real / float(nk)
    e_hartree = 0.5 * np.einsum("abk,abk->", hartree_hamiltonian, density_delta, optimize=True).real / float(nk)
    e_fock = 0.5 * np.einsum("abk,abk->", fock_hamiltonian, projector, optimize=True).real / float(nk)
    return {
        "E_band": float(e_band),
        "E_Hartree": float(e_hartree),
        "E_Fock": float(e_fock),
        "E_total": float(e_band + e_hartree + e_fock),
    }


def hartree_delta_fock_projector_oda_parameter(
    state_obj,
    delta_density: np.ndarray,
    *,
    delta_hartree_h: np.ndarray,
    delta_fock_h: np.ndarray,
    interaction_h: np.ndarray | None = None,
) -> float:
    """ODA parameter for the diagnostic Hartree[D] + Fock[P] active split."""

    delta = np.asarray(delta_density, dtype=np.complex128)
    delta_hartree = np.asarray(delta_hartree_h, dtype=np.complex128)
    delta_fock = np.asarray(delta_fock_h, dtype=np.complex128)
    delta_interaction = delta_hartree + delta_fock
    active_interaction = (
        np.asarray(state_obj.hamiltonian - state_obj.h0, dtype=np.complex128)
        if interaction_h is None
        else np.asarray(interaction_h, dtype=np.complex128)
    )
    active_projector = physical_projector_from_delta(state_obj.density)

    a = np.einsum("abk,abk->", delta, delta_interaction, optimize=True)
    b = np.einsum("abk,abk->", delta, state_obj.h0, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", delta, active_interaction, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", state_obj.density, delta_hartree, optimize=True)
    b += 0.5 * np.einsum("abk,abk->", active_projector, delta_fock, optimize=True)
    a = float(a.real / state_obj.nk)
    b = float(b.real / state_obj.nk)

    if abs(a) < 1e-15:
        return 1.0 if b < 0.0 else 0.0
    lambda0 = -b / a
    if a > 0.0:
        if lambda0 <= 0.0:
            return 0.0
        if lambda0 < 1.0:
            return float(lambda0)
        return 1.0
    if lambda0 <= 0.5:
        return 1.0
    return 0.0


__all__ = [name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None) == __name__ and not name.startswith('_')]
