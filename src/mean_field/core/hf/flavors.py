from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class FlavorBandData:
    """Flavor-resolved band labeling shared by multiple HF systems."""

    band_labels: tuple[str, ...]
    energies: np.ndarray
    mean_weights: np.ndarray


def identity_block(size: int) -> np.ndarray:
    return np.eye(size, dtype=np.complex128)


def flavor_sector_metadata(
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    idx = np.arange(n_spin * n_eta * n_band, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    labels: list[str] = []
    sectors: list[tuple[int, ...]] = []
    spin_labels = ["up", "down"] + [f"spin_{ispin + 1}" for ispin in range(2, n_spin)]
    valley_labels = ["K", "Kprime"] + [f"eta_{ieta + 1}" for ieta in range(2, n_eta)]
    for ispin in range(n_spin):
        spin_label = spin_labels[ispin]
        for ieta in range(n_eta):
            valley_label = valley_labels[ieta]
            labels.append(f"{valley_label}_{spin_label}")
            sectors.append(tuple(int(value) for value in idx[ispin, ieta, :]))
    return tuple(labels), tuple(sectors)


def flavor_block_indices(
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[tuple[int, ...], ...]:
    return flavor_sector_metadata(n_spin=n_spin, n_eta=n_eta, n_band=n_band)[1]


def block_mask(
    *,
    sectors: tuple[tuple[int, ...], ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    if sectors is None:
        sectors = flavor_block_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    nt = max(max(inds) for inds in sectors) + 1
    mask = np.zeros((nt, nt), dtype=bool)
    for inds in sectors:
        idx = np.asarray(inds, dtype=int)
        mask[np.ix_(idx, idx)] = True
    return mask


def project_to_flavor_diagonal_inplace(
    array: np.ndarray,
    *,
    sectors: tuple[tuple[int, ...], ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> None:
    mask = block_mask(sectors=sectors, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    if array.ndim == 2:
        array *= mask
        return
    if array.ndim != 3:
        raise ValueError(f"Expected a rank-2 or rank-3 array, got shape {array.shape}")
    for ik in range(array.shape[2]):
        array[:, :, ik] *= mask


def project_to_flavor_diagonal(
    array: np.ndarray,
    *,
    sectors: tuple[tuple[int, ...], ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    projected = np.array(array, copy=True)
    project_to_flavor_diagonal_inplace(
        projected,
        sectors=sectors,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )
    return projected


def build_flavor_band_data(
    hamiltonian: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> FlavorBandData:
    nt, _, nk = hamiltonian.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    band_labels = [""] * nt
    energies = np.zeros((nt, nk), dtype=float)
    flavor_labels, sectors = flavor_sector_metadata(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    mean_weights = np.zeros((nt, len(sectors)), dtype=float)

    if nk == 0:
        generic_labels = tuple(f"b{ib + 1}" for ib in range(nt))
        return FlavorBandData(
            band_labels=generic_labels,
            energies=energies,
            mean_weights=mean_weights,
        )

    for ik in range(nk):
        evals, evecs = eigh(hamiltonian[:, :, ik])
        energies[:, ik] = evals
        for ib in range(nt):
            for ifl, inds in enumerate(sectors):
                mean_weights[ib, ifl] += float(np.sum(np.abs(evecs[np.asarray(inds), ib]) ** 2))

    mean_weights /= nk
    for ib in range(nt):
        dominant = int(np.argmax(mean_weights[ib, :]))
        band_labels[ib] = f"{flavor_labels[dominant]}_b{ib + 1}"

    return FlavorBandData(
        band_labels=tuple(band_labels),
        energies=energies,
        mean_weights=mean_weights,
    )
