from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandClassification:
    flat_indices: np.ndarray
    flat_mask: np.ndarray
    remote_below_mask: np.ndarray
    remote_above_mask: np.ndarray
    energies: np.ndarray
    cnp_energy_mev: float

    @property
    def n_eta(self) -> int:
        return int(self.energies.shape[1])

    @property
    def nk(self) -> int:
        return int(self.energies.shape[2])

    @property
    def n_band(self) -> int:
        return int(self.energies.shape[0])


def classify_flat_bands(
    energies: np.ndarray,
    *,
    n_flat: int = 2,
    cnp_energy_mev: float = 0.0,
    method: str = "center",
    max_flat_abs_mev: float | None = None,
) -> BandClassification:
    """Classify flat and remote bands on a sorted BM spectrum.

    The BM diagonalization returns ascending energies. For the Zhang TBG
    continuum model the two CNP flat bands are the two centered bands in the
    plane-wave spectrum. ``method='closest'`` is kept for diagnostics on small
    synthetic spectra.
    """

    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim != 3:
        raise ValueError(f"Expected energies shape (band, valley, k), got {energy_array.shape}")
    n_band, n_eta, nk = energy_array.shape
    n_flat = int(n_flat)
    if n_flat <= 0 or n_flat > n_band:
        raise ValueError(f"n_flat must be in [1, {n_band}], got {n_flat}")

    flat_indices = np.zeros((n_eta, nk, n_flat), dtype=int)
    method = str(method).lower()
    if method == "center":
        start = n_band // 2 - n_flat // 2
        indices = np.arange(start, start + n_flat, dtype=int)
        flat_indices[:, :, :] = indices[None, None, :]
    elif method == "closest":
        for ieta in range(n_eta):
            for ik in range(nk):
                order = np.argsort(np.abs(energy_array[:, ieta, ik] - cnp_energy_mev), kind="stable")
                flat_indices[ieta, ik, :] = np.sort(order[:n_flat])
    else:
        raise ValueError(f"Unsupported flat-band classification method: {method}")

    flat_mask = np.zeros((n_eta, nk, n_band), dtype=bool)
    remote_below_mask = np.zeros_like(flat_mask)
    remote_above_mask = np.zeros_like(flat_mask)
    for ieta in range(n_eta):
        for ik in range(nk):
            flat = flat_indices[ieta, ik]
            flat_mask[ieta, ik, flat] = True
            lo = int(np.min(flat))
            hi = int(np.max(flat))
            remote_below_mask[ieta, ik, :lo] = True
            remote_above_mask[ieta, ik, hi + 1 :] = True

    if max_flat_abs_mev is not None:
        flat_energies = np.take_along_axis(
            np.moveaxis(energy_array, 0, -1),
            flat_indices,
            axis=2,
        )
        max_abs = float(np.max(np.abs(flat_energies - cnp_energy_mev)))
        if max_abs > float(max_flat_abs_mev):
            raise ValueError(
                f"Flat-band classification failed max |E-CNP| <= {max_flat_abs_mev} meV: got {max_abs}"
            )

    return BandClassification(
        flat_indices=flat_indices,
        flat_mask=flat_mask,
        remote_below_mask=remote_below_mask,
        remote_above_mask=remote_above_mask,
        energies=energy_array,
        cnp_energy_mev=float(cnp_energy_mev),
    )
