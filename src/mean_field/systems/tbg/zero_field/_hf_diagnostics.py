from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_basis_overlap import restricted_occupied_state_count

def restricted_gap_estimate(energies: np.ndarray, nu: float) -> float:
    nu_norm = restricted_occupied_state_count(nu, energies.shape[0], energies.shape[1])
    sorted_energies = np.sort(energies, axis=None)
    if nu_norm <= 0 or nu_norm >= sorted_energies.size:
        return float("nan")
    return float(sorted_energies[nu_norm] - sorted_energies[nu_norm - 1])


def occupied_sigma_mean(energies: np.ndarray, sigma_ztauz: np.ndarray, nu: float) -> float:
    nu_norm = restricted_occupied_state_count(nu, energies.shape[0], energies.shape[1])
    order = _occupied_state_linear_indices(energies, nu_norm)
    if order.size == 0:
        return float("nan")
    return float(np.mean(np.ravel(sigma_ztauz, order="F")[order]))


def offdiag_flavor_norm(density: np.ndarray, sectors: tuple[tuple[int, ...], ...] | None = None) -> float:
    nt = density.shape[0]
    mask = np.zeros((nt, nt), dtype=bool)
    if sectors is None:
        sectors = flavor_block_indices()
    for inds in sectors:
        idx = np.asarray(inds, dtype=int)
        mask[np.ix_(idx, idx)] = True

    total = 0.0
    for ik in range(density.shape[2]):
        block = density[:, :, ik].copy()
        block[mask] = 0.0
        total += float(np.linalg.norm(block) ** 2)
    return float(np.sqrt(total))

__all__ = [name for name in globals() if not name.startswith('__')]
