"""Source-signed radial Kane carrier counting.

This module mirrors kdotpy's band-index radial linear-simplex IDOS convention:
positive band indices count occupied electron states and negative indices count
vacancies (holes).  It intentionally does not infer carrier labels from the
energy-sorted column number at each momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class SignedRadialBandSpectrum:
    """Radial dispersion with source-defined unique nonzero band indices."""

    k_nm_inv: np.ndarray
    energies_mev: np.ndarray
    band_indices: tuple[int, ...]
    band_labels: tuple[str, ...]
    source: str

    @classmethod
    def from_kdotpy_xml(
        cls,
        path: str | Path,
        *,
        selected_band_indices: tuple[int, ...] | None = None,
    ) -> "SignedRadialBandSpectrum":
        xml_path = Path(path)
        points = ET.parse(xml_path).getroot().findall("./dispersion/momentum")
        if len(points) < 2:
            raise ValueError("kdotpy XML must contain at least two radial momentum points")
        rows: list[tuple[float, np.ndarray]] = []
        reference_indices: np.ndarray | None = None
        reference_labels: tuple[str, ...] | None = None
        for point in points:
            energies_node = point.find("energies")
            indices_node = point.find("bandindex")
            if (
                "k" not in point.attrib
                or energies_node is None
                or not energies_node.text
                or indices_node is None
                or not indices_node.text
            ):
                raise ValueError("kdotpy momentum point lacks k, energies, or signed band indices")
            energies = np.fromstring(energies_node.text, sep=" ", dtype=float)
            indices = np.fromstring(indices_node.text, sep=" ", dtype=int)
            if energies.shape != indices.shape:
                raise ValueError("kdotpy energies and band indices have different lengths")
            if reference_indices is None:
                reference_indices = indices
                labels_node = point.find("characters")
                labels = () if labels_node is None or not labels_node.text else tuple(labels_node.text.split())
                reference_labels = labels if len(labels) == len(indices) else tuple("" for _ in indices)
            elif not np.array_equal(indices, reference_indices):
                raise ValueError("signed band-index order changes across the kdotpy radial export")
            rows.append((float(point.attrib["k"]), energies))
        rows.sort(key=lambda item: item[0])
        assert reference_indices is not None and reference_labels is not None
        if selected_band_indices is None:
            selected = np.arange(reference_indices.size)
        else:
            selected_rows = []
            for band_index in selected_band_indices:
                matches = np.flatnonzero(reference_indices == int(band_index))
                if matches.size != 1:
                    raise ValueError(f"signed band index {band_index} is not unique in the kdotpy export")
                selected_rows.append(int(matches[0]))
            selected = np.asarray(selected_rows, dtype=int)
        result = cls(
            k_nm_inv=np.asarray([item[0] for item in rows], dtype=float),
            energies_mev=np.asarray([item[1][selected] for item in rows], dtype=float),
            band_indices=tuple(int(reference_indices[i]) for i in selected),
            band_labels=tuple(reference_labels[i] for i in selected),
            source=str(xml_path),
        )
        result.validate()
        return result

    def validate(self) -> None:
        k = np.asarray(self.k_nm_inv, dtype=float)
        energies = np.asarray(self.energies_mev, dtype=float)
        if k.ndim != 1 or k.size < 2 or np.any(np.diff(k) <= 0.0) or k[0] < 0.0:
            raise ValueError("radial momenta must be strictly increasing and nonnegative")
        if energies.shape != (k.size, len(self.band_indices)):
            raise ValueError("signed radial energy array has an incompatible shape")
        if len(self.band_labels) != len(self.band_indices):
            raise ValueError("band labels and signed band indices have different lengths")
        if 0 in self.band_indices or len(set(self.band_indices)) != len(self.band_indices):
            raise ValueError("signed band indices must be unique and nonzero")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(energies)):
            raise ValueError("signed radial spectrum must be finite")


@dataclass(frozen=True)
class SignedBandCarrierDensity:
    fermi_energy_mev: float
    electron_density_nm2: float
    hole_density_nm2: float
    electron_density_by_band_nm2: dict[int, float]
    hole_density_by_band_nm2: dict[int, float]
    outer_interval_electron_by_band_nm2: dict[int, float]
    outer_interval_hole_by_band_nm2: dict[int, float]

    @property
    def charge_imbalance_nm2(self) -> float:
        return self.electron_density_nm2 - self.hole_density_nm2


def linear_interval_filled_fraction(endpoint_energies_mev: np.ndarray, mu_mev: float) -> np.ndarray:
    """Filled fraction used by kdotpy's radial linear-simplex IDOS."""

    endpoints = np.asarray(endpoint_energies_mev, dtype=float)
    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ValueError("endpoint energies must have shape (ninterval, 2, nband)")
    lower = np.min(endpoints, axis=1)
    upper = np.max(endpoints, axis=1)
    span = upper - lower
    fraction = np.zeros_like(lower)
    varying = span != 0.0
    fraction[varying] = np.clip((float(mu_mev) - lower[varying]) / span[varying], 0.0, 1.0)
    fraction[~varying] = (float(mu_mev) >= lower[~varying]).astype(float)
    return fraction


def signed_band_carrier_density(
    spectrum: SignedRadialBandSpectrum,
    *,
    fermi_energy_mev: float | None = None,
) -> SignedBandCarrierDensity:
    """Evaluate source-signed radial electron/hole IDOS at zero temperature.

    Radial interval weights are ``(k[i+1]^2-k[i]^2)/(4*pi)`` for
    ``int d^2k/(2*pi)^2``. No extra spin factor is applied because Kramers
    partners are explicit signed bands.
    """

    spectrum.validate()
    k = np.asarray(spectrum.k_nm_inv, dtype=float)
    energies = np.asarray(spectrum.energies_mev, dtype=float)
    band_indices = np.asarray(spectrum.band_indices, dtype=int)
    endpoints = np.stack([energies[:-1], energies[1:]], axis=1)
    weights = np.diff(k**2) / (4.0 * np.pi)
    positive = band_indices > 0
    negative = band_indices < 0
    if not np.any(positive) or not np.any(negative):
        raise ValueError("both positive electron and negative hole band indices are required")

    def components(mu: float) -> tuple[np.ndarray, np.ndarray]:
        filled = linear_interval_filled_fraction(endpoints, mu)
        electron = np.sum(weights[:, None] * filled[:, positive], axis=0)
        hole = np.sum(weights[:, None] * (1.0 - filled[:, negative]), axis=0)
        return electron, hole

    if fermi_energy_mev is None:
        margin = max(10.0, float(np.ptp(energies)))

        def neutrality(mu: float) -> float:
            electron, hole = components(mu)
            return float(np.sum(electron) - np.sum(hole))

        lower = float(np.min(energies) - margin)
        upper = float(np.max(energies) + margin)
        residual_lower = neutrality(lower)
        residual_upper = neutrality(upper)
        if not residual_lower < 0.0 < residual_upper:
            raise ValueError(
                "signed radial spectrum does not bracket charge neutrality: "
                f"({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
            )
        # Match kdotpy's insulating-plateau convention: if the signed IDOS is
        # exactly zero over an energy interval, use the center of that interval
        # instead of an arbitrary root selected by bracketing.
        breakpoints = np.unique(energies)
        zero_intervals: list[tuple[float, float]] = []
        density_scale = max(1.0, float(np.sum(weights)))
        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            if right <= left:
                continue
            if abs(neutrality(0.5 * (left + right))) <= 1e-13 * density_scale:
                zero_intervals.append((float(left), float(right)))
        if zero_intervals:
            mu = 0.5 * (zero_intervals[0][0] + zero_intervals[-1][1])
        else:
            mu = float(brentq(neutrality, lower, upper, xtol=1e-13, rtol=1e-14))
    else:
        mu = float(fermi_energy_mev)

    electron, hole = components(mu)
    filled = linear_interval_filled_fraction(endpoints, mu)
    electron_by_band = {
        int(b): float(value)
        for b, value in zip(band_indices[positive], electron)
    }
    hole_by_band = {
        int(b): float(value)
        for b, value in zip(band_indices[negative], hole)
    }
    outer_electron = {
        int(b): float(weights[-1] * filled[-1, i])
        for i, b in enumerate(band_indices)
        if b > 0
    }
    outer_hole = {
        int(b): float(weights[-1] * (1.0 - filled[-1, i]))
        for i, b in enumerate(band_indices)
        if b < 0
    }
    return SignedBandCarrierDensity(
        fermi_energy_mev=mu,
        electron_density_nm2=float(np.sum(electron)),
        hole_density_nm2=float(np.sum(hole)),
        electron_density_by_band_nm2=electron_by_band,
        hole_density_by_band_nm2=hole_by_band,
        outer_interval_electron_by_band_nm2=outer_electron,
        outer_interval_hole_by_band_nm2=outer_hole,
    )
