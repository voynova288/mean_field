"""Convert a kdotpy 8-band dispersion into Stage-B radial band input.

This module follows the E1/H1 branches by orbital character rather than by
energy ordering.  It is intended for the InAs/GaSb double-well Kane output used
by the NC EI reproduction.  The exported hole energy is the negative of the
valence-electron energy, as required by :class:`ei_mf.bands.RadialBandData`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from .wavefunction_form_factor import read_wavefunction_components

Array = NDArray[np.float64]


@dataclass(frozen=True)
class KaneE1H1Data:
    k_nm_inv: Array
    electron_energy_mev: Array
    valence_energy_mev: Array
    electron_gamma6_weight: Array
    hole_gamma8h_weight: Array
    source: str

    def aligned_stageB_bands(self, k_cross_nm_inv: float) -> tuple[Array, Array]:
        """Return electron and hole energies crossing at ``k_cross_nm_inv``.

        The branch-dependent constant shifts represent the gate/electrostatic
        band alignment needed to impose the measured CNP pocket density.  No
        digitized Fig. 2 order-parameter or pair-breaking curve enters here.
        """

        if not self.k_nm_inv[0] <= k_cross_nm_inv <= self.k_nm_inv[-1]:
            raise ValueError("requested crossing lies outside the Kane k range")
        ee_cross = float(np.interp(k_cross_nm_inv, self.k_nm_inv, self.electron_energy_mev))
        ev_cross = float(np.interp(k_cross_nm_inv, self.k_nm_inv, self.valence_energy_mev))
        epsilon_a = self.electron_energy_mev - ee_cross
        epsilon_b = ev_cross - self.valence_energy_mev
        return epsilon_a, epsilon_b


def _float_vector(element: ET.Element | None, *, label: str) -> Array:
    if element is None or not element.text:
        raise ValueError(f"missing {label} in kdotpy XML")
    return np.fromstring(element.text, sep=" ", dtype=np.float64)


def read_kdotpy_e1_h1(
    path: str | Path,
    *,
    active_band_indices: tuple[int, ...] = (-2, -1, 1, 2),
    tracking_mode: str = "orbital_projected",
) -> KaneE1H1Data:
    """Extract diabatic E1/H1 branches from a kdotpy XML dispersion.

    ``orbital_projected`` reconstructs diagonal E1/H1 energies by projecting
    the four hybridized eigenvalues onto their Gamma6 and Gamma8-heavy-hole
    weights.  This removes the avoided-crossing level repulsion from the band
    input.  ``max_character`` retains the older diagnostic behavior that
    follows the two most Gamma6-like eigenstates and must not be mistaken for a
    diabatic band near the E1/H1 anticrossing.
    """

    if tracking_mode not in {"orbital_projected", "max_character"}:
        raise ValueError("tracking_mode must be 'orbital_projected' or 'max_character'")

    path = Path(path)
    root = ET.parse(path).getroot()
    points = root.findall("./dispersion/momentum")
    if len(points) < 4:
        raise ValueError("kdotpy XML requires at least four dispersion points")

    k_values: list[float] = []
    electron_energy: list[float] = []
    valence_energy: list[float] = []
    electron_weight: list[float] = []
    hole_weight: list[float] = []
    active = set(active_band_indices)

    for point in points:
        k_values.append(float(point.attrib["k"]))
        energies = _float_vector(point.find("energies"), label="energies")
        indices = _float_vector(point.find("bandindex"), label="bandindex").astype(int)
        gamma6 = _float_vector(point.find("observable[@q='gamma6']"), label="gamma6 observable")
        gamma8h = _float_vector(point.find("observable[@q='gamma8h']"), label="gamma8h observable")
        if not (energies.size == indices.size == gamma6.size == gamma8h.size):
            raise ValueError("inconsistent kdotpy energy/index/orbital vector lengths")
        candidate = np.flatnonzero(np.isin(indices, tuple(active)))
        if candidate.size != 4:
            raise ValueError(
                f"expected four active E1/H1 states at k={k_values[-1]}, got {candidate.size}"
            )
        e_sel = candidate[np.argsort(gamma6[candidate])[-2:]]
        remaining = np.setdiff1d(candidate, e_sel, assume_unique=False)
        if remaining.size != 2:
            raise ValueError("E1/H1 orbital assignment did not split into two Kramers pairs")
        h_sel = remaining
        # Guard against a wrong energy-ordered quartet or missing orbital data.
        if float(np.mean(gamma6[e_sel])) <= float(np.mean(gamma6[h_sel])):
            raise ValueError("selected E1 branch is not more Gamma6-like than H1")
        if float(np.mean(gamma8h[h_sel])) <= float(np.mean(gamma8h[e_sel])):
            raise ValueError("selected H1 branch is not more heavy-hole-like than E1")
        if tracking_mode == "orbital_projected":
            e_orbital = gamma6[candidate]
            h_orbital = gamma8h[candidate]
            e_norm = float(np.sum(e_orbital))
            h_norm = float(np.sum(h_orbital))
            if e_norm <= 1e-8 or h_norm <= 1e-8:
                raise ValueError("active quartet has insufficient E1/H1 orbital weight")
            electron_energy.append(float(np.dot(e_orbital, energies[candidate]) / e_norm))
            valence_energy.append(float(np.dot(h_orbital, energies[candidate]) / h_norm))
            electron_weight.append(0.5 * e_norm)
            hole_weight.append(0.5 * h_norm)
        else:
            electron_energy.append(float(np.mean(energies[e_sel])))
            valence_energy.append(float(np.mean(energies[h_sel])))
            electron_weight.append(float(np.mean(gamma6[e_sel])))
            hole_weight.append(float(np.mean(gamma8h[h_sel])))

    order = np.argsort(k_values)
    return KaneE1H1Data(
        np.asarray(k_values, dtype=np.float64)[order],
        np.asarray(electron_energy, dtype=np.float64)[order],
        np.asarray(valence_energy, dtype=np.float64)[order],
        np.asarray(electron_weight, dtype=np.float64)[order],
        np.asarray(hole_weight, dtype=np.float64)[order],
        source=str(path),
    )


def read_kdotpy_e1_h1_subband_overlap(
    path: str | Path,
    wavefunction_directory: str | Path,
    *,
    output_id: str,
    active_band_indices: tuple[int, ...] = (-2, -1, 1, 2),
    electron_reference_indices: tuple[int, int] = (-2, -1),
    hole_reference_indices: tuple[int, int] = (1, 2),
) -> KaneE1H1Data:
    """Reconstruct diabatic E1/H1 energies in the frozen k=0 subband basis.

    For each k, spectral projector weights onto the two k=0 E1 and H1
    reference states are formed from complex envelope-function overlaps.  The
    weighted eigenvalue sums are the diagonal matrix elements of H(k) in that
    fixed subband basis, up to the explicitly reported active-space
    completeness.  This is distinct from projection onto atomic Gamma6/Gamma8
    orbitals, which need not identify a confined E1/H1 subband.
    """

    path = Path(path)
    wf_dir = Path(wavefunction_directory)
    root = ET.parse(path).getroot()
    points = root.findall("./dispersion/momentum")
    if len(points) < 4:
        raise ValueError("kdotpy XML requires at least four dispersion points")
    point_data = sorted(points, key=lambda point: float(point.attrib["k"]))
    if abs(float(point_data[0].attrib["k"])) > 1e-12:
        raise ValueError("subband-overlap tracking requires a k=0 reference point")

    active = tuple(active_band_indices)
    if not set(electron_reference_indices).issubset(active) or not set(
        hole_reference_indices
    ).issubset(active):
        raise ValueError("reference indices must belong to the active band set")

    def load_components(k: float, band_index: int) -> tuple[Array, dict[str, NDArray[np.complex128]]]:
        filename = wf_dir / f"wfs{output_id}_{k:.3f}_0.{band_index}.csv"
        z, components = read_wavefunction_components(filename)
        dz = float(np.mean(np.diff(z)))
        norm = float(sum(np.sum(np.abs(value) ** 2) for value in components.values()) * dz)
        if norm <= 1e-12 or not np.isfinite(norm):
            raise ValueError(f"invalid wavefunction norm for {filename}")
        scale = np.sqrt(norm)
        return z, {label: value / scale for label, value in components.items()}

    k0 = float(point_data[0].attrib["k"])
    references: dict[int, dict[str, NDArray[np.complex128]]] = {}
    z_ref: Array | None = None
    for band_index in (*electron_reference_indices, *hole_reference_indices):
        z, components = load_components(k0, band_index)
        if z_ref is None:
            z_ref = z
        elif not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
            raise ValueError("inconsistent reference-state z grids")
        references[band_index] = components
    assert z_ref is not None
    dz = float(np.mean(np.diff(z_ref)))

    def overlap(reference: dict[str, NDArray[np.complex128]], state: dict[str, NDArray[np.complex128]]) -> complex:
        return dz * sum(
            np.vdot(component, state.get(label, np.zeros_like(component)))
            for label, component in reference.items()
        )

    k_values: list[float] = []
    electron_energy: list[float] = []
    valence_energy: list[float] = []
    electron_completeness: list[float] = []
    hole_completeness: list[float] = []
    for point in point_data:
        k = float(point.attrib["k"])
        energies = _float_vector(point.find("energies"), label="energies")
        indices = _float_vector(point.find("bandindex"), label="bandindex").astype(int)
        state_components: dict[int, dict[str, NDArray[np.complex128]]] = {}
        state_energies: dict[int, float] = {}
        for band_index in active:
            matches = np.flatnonzero(indices == band_index)
            if matches.size != 1:
                raise ValueError(f"active band {band_index} is not unique at k={k}")
            z, components = load_components(k, band_index)
            if not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
                raise ValueError("inconsistent wavefunction z grids")
            state_components[band_index] = components
            state_energies[band_index] = float(energies[matches[0]])

        weights_e = {
            band_index: sum(
                abs(overlap(references[ref], state_components[band_index])) ** 2
                for ref in electron_reference_indices
            )
            for band_index in active
        }
        weights_h = {
            band_index: sum(
                abs(overlap(references[ref], state_components[band_index])) ** 2
                for ref in hole_reference_indices
            )
            for band_index in active
        }
        norm_e = float(sum(weights_e.values()))
        norm_h = float(sum(weights_h.values()))
        if norm_e <= 1e-8 or norm_h <= 1e-8:
            raise ValueError(f"insufficient active-subspace overlap completeness at k={k}")
        k_values.append(k)
        electron_energy.append(
            float(sum(weights_e[index] * state_energies[index] for index in active) / norm_e)
        )
        valence_energy.append(
            float(sum(weights_h[index] * state_energies[index] for index in active) / norm_h)
        )
        electron_completeness.append(0.5 * norm_e)
        hole_completeness.append(0.5 * norm_h)

    return KaneE1H1Data(
        np.asarray(k_values),
        np.asarray(electron_energy),
        np.asarray(valence_energy),
        np.asarray(electron_completeness),
        np.asarray(hole_completeness),
        source=str(path),
    )


def write_stageB_band_csv(
    output: str | Path,
    data: KaneE1H1Data,
    *,
    k_cross_nm_inv: float,
) -> None:
    """Write a Stage-B band CSV with explicit orbital-character diagnostics."""

    epsilon_a, epsilon_b = data.aligned_stageB_bands(k_cross_nm_inv)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.column_stack(
            [
                data.k_nm_inv,
                epsilon_a,
                epsilon_b,
                data.electron_energy_mev,
                data.valence_energy_mev,
                data.electron_gamma6_weight,
                data.hole_gamma8h_weight,
            ]
        ),
        delimiter=",",
        header=(
            "k_nm_inv,epsilon_a_meV,epsilon_b_meV,"
            "raw_E1_electron_meV,raw_H1_valence_electron_meV,"
            "E1_gamma6_weight,H1_gamma8h_weight"
        ),
        comments="",
    )
