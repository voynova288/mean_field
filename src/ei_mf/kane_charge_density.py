"""Paper-local charge density and periodic Poisson tools for Kane E1/H1 states.

This module implements the definitions in Du et al. Supplementary Note 6,
Eq. (5): occupied Gamma6 components contribute electron density and unoccupied
valence-orbital components contribute hole density.  It intentionally does not
use kdotpy's band-index CNP classifier or uniform full-spectrum reference
charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .fermi import fermi_mev
from .units import COULOMB_MEV_NM
from .wavefunction_form_factor import read_orbital_densities

Array = NDArray[np.float64]


def _float_vector(element: ET.Element | None, *, label: str) -> Array:
    if element is None or not element.text:
        raise ValueError(f"missing {label} in kdotpy XML")
    return np.fromstring(element.text, sep=" ", dtype=np.float64)


def radial_quadrature_weights(k_nm_inv: Array) -> Array:
    """Weights for ``int d^2k/(2*pi)^2`` on a radial endpoint grid."""

    k = np.asarray(k_nm_inv, dtype=np.float64)
    if k.ndim != 1 or k.size < 3 or np.any(np.diff(k) <= 0.0) or k[0] < 0.0:
        raise ValueError("k must be a strictly increasing nonnegative radial grid")
    edges = np.empty(k.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (k[:-1] + k[1:])
    edges[-1] = k[-1]
    if np.any(np.diff(edges) < 0.0):
        raise ValueError("invalid radial cell edges")
    return (edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * np.pi)


@dataclass(frozen=True)
class KaneRadialStates:
    k_nm_inv: Array
    energies_mev: Array  # shape (nk, nstate)
    z_nm: Array
    electron_orbital_density_nm_inv: Array  # shape (nk, nstate, nz)
    hole_orbital_density_nm_inv: Array  # shape (nk, nstate, nz)
    band_indices: tuple[int, ...]
    source_xml: str
    wavefunction_directory: str

    @classmethod
    def from_kdotpy_export(
        cls,
        xml_path: str | Path,
        wavefunction_directory: str | Path,
        *,
        output_id: str,
        active_band_indices: tuple[int, ...] = (13, 14, 15, 16),
    ) -> "KaneRadialStates":
        xml_path = Path(xml_path)
        wf_dir = Path(wavefunction_directory)
        root = ET.parse(xml_path).getroot()
        points = root.findall("./dispersion/momentum")
        if len(points) < 3:
            raise ValueError("kdotpy XML needs at least three momentum points")

        rows: list[tuple[float, Array, Array, Array]] = []
        z_ref: Array | None = None
        for point in points:
            k = float(point.attrib["k"])
            energies = _float_vector(point.find("energies"), label="energies")
            indices = _float_vector(point.find("bandindex"), label="bandindex").astype(int)
            state_energies: list[float] = []
            state_e_density: list[Array] = []
            state_h_density: list[Array] = []
            for band_index in active_band_indices:
                matches = np.flatnonzero(indices == band_index)
                if matches.size != 1:
                    raise ValueError(f"band index {band_index} is not unique at k={k}")
                filename = f"wfs{output_id}_{k:.3f}_0.{band_index}.csv"
                z, orbitals = read_orbital_densities(wf_dir / filename)
                if z_ref is None:
                    z_ref = z
                elif not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
                    raise ValueError("inconsistent z grids in wavefunction export")
                rho_e = np.zeros_like(z)
                rho_h = np.zeros_like(z)
                for label, density in orbitals.items():
                    if label.startswith("Γ6,"):
                        rho_e += density
                    else:
                        rho_h += density
                dz = float(np.mean(np.diff(z)))
                norm = float(np.sum(rho_e + rho_h) * dz)
                if not np.isfinite(norm) or norm <= 1e-12:
                    raise ValueError(f"invalid wavefunction norm {norm}: {filename}")
                # kdotpy CSVs contain lattice-normalized eigenvectors.  Their
                # continuum integral equals dz (0.5 nm here), so normalize each
                # exported state before applying Supplementary Eq. (5).
                state_energies.append(float(energies[matches[0]]))
                state_e_density.append(rho_e / norm)
                state_h_density.append(rho_h / norm)
            rows.append(
                (
                    k,
                    np.asarray(state_energies),
                    np.asarray(state_e_density),
                    np.asarray(state_h_density),
                )
            )

        rows.sort(key=lambda row: row[0])
        assert z_ref is not None
        result = cls(
            np.asarray([row[0] for row in rows]),
            np.asarray([row[1] for row in rows]),
            z_ref,
            np.asarray([row[2] for row in rows]),
            np.asarray([row[3] for row in rows]),
            tuple(active_band_indices),
            str(xml_path),
            str(wf_dir),
        )
        result.validate()
        return result

    def validate(self) -> None:
        nk = self.k_nm_inv.size
        ns = len(self.band_indices)
        nz = self.z_nm.size
        if self.energies_mev.shape != (nk, ns):
            raise ValueError("energy array shape mismatch")
        expected = (nk, ns, nz)
        if self.electron_orbital_density_nm_inv.shape != expected:
            raise ValueError("electron orbital-density shape mismatch")
        if self.hole_orbital_density_nm_inv.shape != expected:
            raise ValueError("hole orbital-density shape mismatch")
        if np.any(self.electron_orbital_density_nm_inv < 0.0) or np.any(
            self.hole_orbital_density_nm_inv < 0.0
        ):
            raise ValueError("negative orbital probability density")


@dataclass(frozen=True)
class ChargeNeutralDensity:
    fermi_energy_mev: float
    electron_density_cm2: float
    hole_density_cm2: float
    z_nm: Array
    electron_density_nm3: Array
    hole_density_nm3: Array

    @property
    def charge_imbalance_cm2(self) -> float:
        return self.electron_density_cm2 - self.hole_density_cm2


def charge_neutral_density(
    states: KaneRadialStates,
    *,
    temperature_K: float = 0.0,
) -> ChargeNeutralDensity:
    """Evaluate Supplementary Eq. (5) and solve its neutrality condition."""

    states.validate()
    wk = radial_quadrature_weights(states.k_nm_inv)
    z = states.z_nm
    dz = float(np.mean(np.diff(z)))
    if not np.allclose(np.diff(z), dz, rtol=0.0, atol=1e-10):
        raise ValueError("kdotpy wavefunction z grid must be uniform")
    # Plane-wave/FDM grid points carry equal periodic quadrature weight dz.
    electron_weight = np.sum(states.electron_orbital_density_nm_inv, axis=2) * dz
    hole_weight = np.sum(states.hole_orbital_density_nm_inv, axis=2) * dz

    def areal_densities(ef_mev: float) -> tuple[float, float, Array]:
        occupation = fermi_mev(states.energies_mev - ef_mev, temperature_K)
        ne_nm2 = float(np.sum(wk[:, None] * electron_weight * occupation))
        nh_nm2 = float(np.sum(wk[:, None] * hole_weight * (1.0 - occupation)))
        return ne_nm2, nh_nm2, occupation

    energy_margin = max(10.0, 50.0 * 0.08617333262 * max(temperature_K, 0.0))
    lower = float(np.min(states.energies_mev) - energy_margin)
    upper = float(np.max(states.energies_mev) + energy_margin)

    def neutrality(ef_mev: float) -> float:
        ne, nh, _ = areal_densities(ef_mev)
        return ne - nh

    f_lower = neutrality(lower)
    f_upper = neutrality(upper)
    if not (f_lower < 0.0 < f_upper):
        raise ValueError(
            "active-state energy window does not bracket charge neutrality: "
            f"residuals=({f_lower:.3e}, {f_upper:.3e}) nm^-2"
        )
    ef = float(brentq(neutrality, lower, upper, xtol=1e-12, rtol=1e-13))
    ne_nm2, nh_nm2, occupation = areal_densities(ef)
    weighted_occ = wk[:, None] * occupation
    weighted_empty = wk[:, None] * (1.0 - occupation)
    ne_z = np.einsum("ks,ksz->z", weighted_occ, states.electron_orbital_density_nm_inv)
    nh_z = np.einsum("ks,ksz->z", weighted_empty, states.hole_orbital_density_nm_inv)
    return ChargeNeutralDensity(
        ef,
        ne_nm2 * 1e14,
        nh_nm2 * 1e14,
        z,
        ne_z,
        nh_z,
    )


def periodic_poisson_electron_energy_mev(
    z_nm: Array,
    electron_density_nm3: Array,
    hole_density_nm3: Array,
    epsilon_r: Array,
) -> Array:
    """Solve neutral periodic 1D Poisson equation with zero-mean electron energy.

    The equation is ``d_z[epsilon_r d_z U] = 4*pi*C*(n_h-n_e)``,
    where ``C=e^2/(4*pi*epsilon_0)`` and ``U=-e*phi`` is the electron
    electrostatic potential energy.  The zero Fourier component is fixed by
    ``mean(U)=0``.
    """

    z = np.asarray(z_nm, dtype=np.float64)
    ne = np.asarray(electron_density_nm3, dtype=np.float64)
    nh = np.asarray(hole_density_nm3, dtype=np.float64)
    eps = np.asarray(epsilon_r, dtype=np.float64)
    if not (z.shape == ne.shape == nh.shape == eps.shape):
        raise ValueError("z, densities, and epsilon arrays must have matching shapes")
    dz_values = np.diff(z)
    if z.ndim != 1 or z.size < 4 or not np.allclose(dz_values, dz_values[0]):
        raise ValueError("periodic Poisson solver requires a uniform z grid")
    if np.any(eps <= 0.0):
        raise ValueError("dielectric constant must be positive")
    dz = float(dz_values[0])
    net_number = nh - ne
    integrated_net = float(np.sum(net_number) * dz)
    if abs(integrated_net) > 1e-8:
        raise ValueError(f"periodic Poisson source is not neutral: {integrated_net:.3e} nm^-2")
    # Remove residual quadrature roundoff from the zero Fourier component.
    net_number = net_number - np.mean(net_number)

    n = z.size
    matrix = np.zeros((n, n), dtype=np.float64)
    eps_plus = 2.0 * eps * np.roll(eps, -1) / (eps + np.roll(eps, -1))
    eps_minus = np.roll(eps_plus, 1)
    for i in range(n):
        matrix[i, i] = -(eps_plus[i] + eps_minus[i]) / dz**2
        matrix[i, (i + 1) % n] = eps_plus[i] / dz**2
        matrix[i, (i - 1) % n] = eps_minus[i] / dz**2
    source = 4.0 * np.pi * COULOMB_MEV_NM * net_number
    # Replace one singular Poisson row by the zero-mean gauge condition.
    matrix[-1, :] = 1.0 / n
    source[-1] = 0.0
    potential = np.linalg.solve(matrix, source)
    potential -= np.mean(potential)
    return potential


def wafer_b_dielectric_profile(z_nm: Array) -> Array:
    """Piecewise Supplementary-Table-1 dielectric profile for Wafer B."""

    z = np.asarray(z_nm, dtype=np.float64)
    eps = np.full_like(z, 14.4)  # AlSb barriers
    eps[(z >= -9.75) & (z < 1.75)] = 14.55  # InAs, 11.5 nm
    eps[(z >= 1.75) & (z < 9.75)] = 15.69  # GaSb, 8 nm
    return eps
