"""Finite-magnetic-field BM/Hofstadter spectrum for TBG.

This module ports the non-interacting magnetic spectrum part of the author
code in ``TBG_HartreeFock（作者原始代码）/libs/bmLL*.jl``.  It constructs the
Landau-level basis Hamiltonian at rational flux ``p/q``, diagonalizes the
central ``2q`` Hofstadter subbands on the magnetic Brillouin-zone mesh, and can
build projected density-overlap matrices for the finite-B HF module.

The implementation intentionally keeps file I/O out of the core.  Production
workflows may save the returned arrays in whatever format is convenient; the HF
adapter consumes the same arrays through :class:`MagneticOverlapData`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln

from ....core.magnetic_field import MagneticFlux, choose_magnetic_nq, magnetic_reciprocal_vector
from .hf import MagneticOverlapData

Array = np.ndarray
Valley = Literal["K", "Kprime"]


def _complex_matrix(values: list[list[complex]]) -> Array:
    return np.asarray(values, dtype=np.complex128)


@dataclass(frozen=True)
class FiniteFieldBMParameters:
    """Author-code finite-B BM parameters.

    Defaults mirror ``libs/params.jl`` and ``initParamsWithStrain`` in the
    original repository rather than the zero-field B0 adapter.  In particular,
    the original finite-B code does not apply the extra inversion-symmetric
    ``K``-point shift used by some B0 workflows.
    """

    dtheta_rad: float
    vf: float = 2482.0
    w0: float = 77.0
    w1: float = 110.0
    delta: float = 0.0
    strain: float = 0.0
    strain_angle_rad: float = 0.0
    poisson: float = 0.16
    beta_g: float = 3.14
    deformation_potential: float = -4100.0

    g1: complex = field(init=False)
    g2: complex = field(init=False)
    a1: complex = field(init=False)
    a2: complex = field(init=False)
    area: float = field(init=False)
    theta12: float = field(init=False)
    kt: complex = field(init=False)
    kb_point: complex = field(init=False)
    omega: complex = field(init=False)
    t0: Array = field(init=False, repr=False)
    t1: Array = field(init=False, repr=False)
    t2: Array = field(init=False, repr=False)
    strain_matrix: Array = field(init=False, repr=False)
    gauge_shift: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtheta = float(self.dtheta_rad)
        strain = float(self.strain)
        phi = float(self.strain_angle_rad)
        poisson = float(self.poisson)

        kb = 4.0 * np.pi / 3.0 * dtheta
        kt = kb / 2.0 * np.exp(1j * np.pi / 2.0)
        kb_point = -kb / 2.0 * np.exp(1j * np.pi / 2.0)

        omega = np.exp(1j * 2.0 * np.pi / 3.0)
        t0 = _complex_matrix([[self.w0, self.w1], [self.w1, self.w0]])
        # Match original Params.T1/T2, not the B0 neighbor-swapped wrapper.
        t1 = _complex_matrix([[self.w0, self.w1 * omega], [self.w1 * np.conj(omega), self.w0]])
        t2 = _complex_matrix([[self.w0, self.w1 * np.conj(omega)], [self.w1 * omega, self.w0]])

        exx = -strain * np.cos(phi) ** 2 + poisson * strain * np.sin(phi) ** 2
        eyy = poisson * strain * np.cos(phi) ** 2 - strain * np.sin(phi) ** 2
        exy = (1.0 + poisson) * strain * np.cos(phi) * np.sin(phi)
        gauge_shift = (np.sqrt(3.0) * self.beta_g / 2.0) * np.asarray([exx - eyy, -2.0 * exy], dtype=float)
        rotation_phi = np.asarray([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], dtype=float)
        strain_matrix = rotation_phi.T @ np.asarray([[-strain, 0.0], [0.0, poisson * strain]], dtype=float) @ rotation_phi

        twist_generator = dtheta / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
        g1_cart = 4.0 * np.pi / np.sqrt(3.0) * np.asarray([0.0, -1.0], dtype=float)
        g2_cart = 4.0 * np.pi / np.sqrt(3.0) * np.asarray([np.sqrt(3.0) / 2.0, 0.5], dtype=float)
        tmp1 = (2.0 * twist_generator - strain_matrix) @ g1_cart
        tmp2 = (2.0 * twist_generator - strain_matrix) @ g2_cart
        g1 = complex(tmp1[0], tmp1[1])
        g2 = complex(tmp2[0], tmp2[1])

        reciprocal_area = abs(g1.real * g2.imag - g1.imag * g2.real)
        a1 = 2.0 * np.pi / reciprocal_area * complex(g2.imag, -g2.real)
        a2 = 2.0 * np.pi / reciprocal_area * complex(-g1.imag, g1.real)
        area = abs((np.conj(a1) * a2).imag)
        theta12 = np.angle(a2) - np.angle(a1)

        kt = kt + complex(gauge_shift[0], gauge_shift[1]) / 2.0 - complex(strain_matrix[0, 0], strain_matrix[1, 0]) * 2.0 * np.pi / 3.0
        kb_point = kb_point - complex(gauge_shift[0], gauge_shift[1]) / 2.0 + complex(strain_matrix[0, 0], strain_matrix[1, 0]) * 2.0 * np.pi / 3.0

        object.__setattr__(self, "g1", complex(g1))
        object.__setattr__(self, "g2", complex(g2))
        object.__setattr__(self, "a1", complex(a1))
        object.__setattr__(self, "a2", complex(a2))
        object.__setattr__(self, "area", float(area))
        object.__setattr__(self, "theta12", float(theta12))
        object.__setattr__(self, "kt", complex(kt))
        object.__setattr__(self, "kb_point", complex(kb_point))
        object.__setattr__(self, "omega", complex(omega))
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "t2", t2)
        object.__setattr__(self, "strain_matrix", strain_matrix)
        object.__setattr__(self, "gauge_shift", gauge_shift)

    @classmethod
    def from_degrees(
        cls,
        theta_deg: float,
        *,
        w0: float = 77.0,
        w1: float = 110.0,
        strain: float = 0.0,
        strain_angle_deg: float = 0.0,
        deformation_potential: float = -4100.0,
        vf: float = 2482.0,
    ) -> "FiniteFieldBMParameters":
        return cls(
            dtheta_rad=float(theta_deg) * np.pi / 180.0,
            vf=vf,
            w0=w0,
            w1=w1,
            strain=strain,
            strain_angle_rad=float(strain_angle_deg) * np.pi / 180.0,
            deformation_potential=deformation_potential,
        )


@dataclass(frozen=True)
class MagneticSpectrumResult:
    """Central Hofstadter subbands and projected data for one valley."""

    params: FiniteFieldBMParameters
    flux: MagneticFlux
    valley: Valley
    n_landau: int
    n_h: int
    nq: int
    sigma_rotation: bool
    l_b: float
    q_phi: complex
    lattice_k1: Array
    lattice_k2: Array
    hamiltonian_ll: Array
    sigma_z_ll: Array
    spectrum: Array
    vec: Array
    p_sigma_z: Array
    sigma_z_eigenvalues: Array
    sigma_z_energy_diag: Array

    @property
    def q(self) -> int:
        return self.flux.q

    @property
    def p(self) -> int:
        return self.flux.p

    @property
    def n_subbands(self) -> int:
        return 2 * self.q

    @property
    def inner_dim(self) -> int:
        return 2 * self.n_h * self.p

    def overlap_data_for_shifts(self, shifts: tuple[tuple[int, int], ...], *, fast: bool = True) -> MagneticOverlapData:
        """Return 4D overlap data for the requested finite-B shifts."""

        overlaps: dict[tuple[int, int], Array] = {}
        gvecs: list[complex] = []
        nk_full = self.q * self.nq * self.nq
        overlap_fn = compute_coulomb_overlap_fast if fast else compute_coulomb_overlap
        for m, n in shifts:
            shift = (int(m), int(n))
            flat = overlap_fn(self, shift[0], shift[1])
            overlaps[shift] = flat.reshape((self.n_subbands, nk_full, self.n_subbands, nk_full), order="F")
            gvecs.append(magnetic_reciprocal_vector(shift[0], shift[1], g1=self.params.g1, g2=self.params.g2, q=self.q))
        return MagneticOverlapData(shifts=tuple((int(m), int(n)) for m, n in shifts), gvecs=np.asarray(gvecs, dtype=np.complex128), overlaps=overlaps)



@dataclass(frozen=True)
class MagneticSpectrumSweepCase:
    """One flux point in a paper-style Hofstadter spectrum sweep."""

    flux: MagneticFlux
    nq: int
    n_landau: int
    mesh_shift: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "nq", int(self.nq))
        object.__setattr__(self, "n_landau", int(self.n_landau))
        object.__setattr__(self, "mesh_shift", float(self.mesh_shift))
        if self.nq <= 0:
            raise ValueError(f"nq must be positive, got {self.nq}")
        if self.n_landau <= 0:
            raise ValueError(f"n_landau must be positive, got {self.n_landau}")

@dataclass(frozen=True)
class MagneticSpectrumSweepResult:
    """No-I/O result for a paper-style magnetic-spectrum sweep."""

    cases: tuple[MagneticSpectrumSweepCase, ...]
    spectra: tuple[MagneticSpectrumResult, ...]
    red_group_masks: tuple[Array, ...]

    def __post_init__(self) -> None:
        if not (len(self.cases) == len(self.spectra) == len(self.red_group_masks)):
            raise ValueError("cases, spectra, and red_group_masks must have the same length")
        masks = tuple(np.asarray(mask, dtype=bool) for mask in self.red_group_masks)
        for case, result, mask in zip(self.cases, self.spectra, masks, strict=True):
            if int(result.flux.p) != case.flux.p or int(result.flux.q) != case.flux.q:
                raise ValueError("Sweep case/result flux mismatch")
            if mask.shape != (result.n_subbands,):
                raise ValueError(f"Expected red-group mask shape {(result.n_subbands,)}, got {mask.shape}")
        object.__setattr__(self, "red_group_masks", masks)

    def as_point_table(self) -> dict[str, Array]:
        """Return flattened arrays suitable for paper-style scatter plots."""

        p_vals: list[int] = []
        q_vals: list[int] = []
        nq_vals: list[int] = []
        n_landau_vals: list[int] = []
        band_vals: list[int] = []
        i1_vals: list[int] = []
        i2_vals: list[int] = []
        energy_vals: list[float] = []
        red_vals: list[bool] = []
        for case, result, red_mask in zip(self.cases, self.spectra, self.red_group_masks, strict=True):
            for i2 in range(result.nq):
                for i1 in range(result.nq):
                    for band in range(result.n_subbands):
                        p_vals.append(case.flux.p)
                        q_vals.append(case.flux.q)
                        nq_vals.append(case.nq)
                        n_landau_vals.append(case.n_landau)
                        band_vals.append(band + 1)
                        i1_vals.append(i1 + 1)
                        i2_vals.append(i2 + 1)
                        energy_vals.append(float(result.spectrum[band, i1, i2]))
                        red_vals.append(bool(red_mask[band]))
        return {
            "p": np.asarray(p_vals, dtype=int),
            "q": np.asarray(q_vals, dtype=int),
            "phi": np.asarray(p_vals, dtype=float) / np.asarray(q_vals, dtype=float),
            "nq": np.asarray(nq_vals, dtype=int),
            "n_landau": np.asarray(n_landau_vals, dtype=int),
            "band": np.asarray(band_vals, dtype=int),
            "i1": np.asarray(i1_vals, dtype=int),
            "i2": np.asarray(i2_vals, dtype=int),
            "energy_mev": np.asarray(energy_vals, dtype=float),
            "red_group": np.asarray(red_vals, dtype=bool),
        }

def paper_hofstadter_fluxes(*, max_denominator: int = 12, phi_max: float = 0.5) -> tuple[MagneticFlux, ...]:
    """Return the rational fluxes used for paper-style Hofstadter panels."""

    max_denominator = int(max_denominator)
    if max_denominator <= 0:
        raise ValueError(f"max_denominator must be positive, got {max_denominator}")
    fracs = {
        Fraction(p, q)
        for q in range(1, max_denominator + 1)
        for p in range(1, q + 1)
        if float(Fraction(p, q)) <= float(phi_max) + 1e-15
    }
    return tuple(MagneticFlux(frac.numerator, frac.denominator) for frac in sorted(fracs))

def author_landau_cutoff(flux: MagneticFlux, *, cutoff: int = 25) -> int:
    """Return author production cutoff ``nLL = cutoff*q/p``."""

    if flux.p <= 0:
        raise ValueError(f"Author paper spectra expect positive p, got {flux.p}")
    return int(int(cutoff) * flux.q // flux.p)

def red_chern_minus_one_group_mask(flux: MagneticFlux) -> Array:
    """Mask the red C=-1 subband group below CNP in Fig. 3(a)."""

    p_flux, q_flux = int(flux.p), int(flux.q)
    if p_flux <= 0 or p_flux > q_flux:
        raise ValueError(f"Expected 0 < p <= q for red-group mask, got p/q={p_flux}/{q_flux}")
    mask = np.zeros(2 * q_flux, dtype=bool)
    mask[q_flux - p_flux : q_flux] = True
    return mask

def _value_for_flux(value, flux: MagneticFlux):
    if callable(value):
        return value(flux)
    if isinstance(value, Mapping):
        if (flux.p, flux.q) in value:
            return value[(flux.p, flux.q)]
        label = f"{flux.p}/{flux.q}"
        if label in value:
            return value[label]
        return 0.0
    return value

def compute_magnetic_spectrum_sweep(
    params: FiniteFieldBMParameters,
    *,
    fluxes: Sequence[MagneticFlux | tuple[int, int] | Fraction | str] | None = None,
    max_denominator: int = 12,
    phi_max: float = 0.5,
    valley: Valley = "K",
    landau_cutoff: int = 25,
    n_landau_by_flux: Callable[[MagneticFlux], int] | Mapping[tuple[int, int] | str, int] | None = None,
    nq_by_flux: Callable[[MagneticFlux], int] | Mapping[tuple[int, int] | str, int] | None = None,
    mesh_shift_by_flux: Callable[[MagneticFlux], float] | Mapping[tuple[int, int] | str, float] | float = 0.0,
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
) -> MagneticSpectrumSweepResult:
    """Compute a no-I/O paper-style finite-B Hofstadter spectrum sweep.

    By default this matches the Fig. 3(a)-style flux set ``q<=12`` and
    ``phi<=1/2`` with the author cutoff ``nLL=25*q/p`` and mesh rule
    ``choose_magnetic_nq(q)``. Tests and small diagnostics can override
    ``fluxes``, ``n_landau_by_flux``, and ``nq_by_flux``.
    """

    normalized_fluxes = (
        paper_hofstadter_fluxes(max_denominator=max_denominator, phi_max=phi_max)
        if fluxes is None
        else tuple(MagneticFlux.from_value(flux) if not isinstance(flux, MagneticFlux) else flux for flux in fluxes)
    )
    cases: list[MagneticSpectrumSweepCase] = []
    spectra: list[MagneticSpectrumResult] = []
    red_masks: list[Array] = []
    for flux in normalized_fluxes:
        n_landau = int(_value_for_flux(n_landau_by_flux, flux)) if n_landau_by_flux is not None else author_landau_cutoff(flux, cutoff=landau_cutoff)
        nq = int(_value_for_flux(nq_by_flux, flux)) if nq_by_flux is not None else choose_magnetic_nq(flux.q)
        mesh_shift = float(_value_for_flux(mesh_shift_by_flux, flux))
        case = MagneticSpectrumSweepCase(flux=flux, nq=nq, n_landau=n_landau, mesh_shift=mesh_shift)
        result = compute_magnetic_spectrum(
            params,
            flux=flux,
            n_landau=n_landau,
            nq=nq,
            valley=valley,
            sigma_rotation=sigma_rotation,
            hbn=hbn,
            include_strain=include_strain,
            q0=q0,
            mesh_shift=mesh_shift,
        )
        cases.append(case)
        spectra.append(result)
        red_masks.append(red_chern_minus_one_group_mask(flux))
    return MagneticSpectrumSweepResult(cases=tuple(cases), spectra=tuple(spectra), red_group_masks=tuple(red_masks))

def in_gamma(index: int) -> tuple[int, int]:
    """Return ``(n, gamma)`` for the original 1D LL/sublattice index."""

    i = int(index)
    if i < 0:
        raise ValueError(f"index must be nonnegative, got {index}")
    if i == 0:
        return 0, -3  # matches the original helper; gamma is unused for n=0.
    n = (i - 1) // 2 + 1
    i_gamma = (i - 1) % 2 + 1
    return int(n), int(2 * i_gamma - 3)


def projector_para(vec1: complex, vec2: complex) -> float:
    return float((np.conj(vec1) * vec2).real / abs(vec2))


def projector_norm(vec1: complex, vec2: complex) -> float:
    return float((np.conj(vec1) * vec2).imag / abs(vec2))


def associated_laguerre_element(n: int, m: int, cplus: complex, cminus: complex) -> complex:
    """Matrix element used by the LL translation operator."""

    n = int(n)
    m = int(m)
    x = -float((cplus * cminus).real)
    if n >= m:
        prefactor = np.exp(-x / 2.0 + 0.5 * (gammaln(m + 1) - gammaln(n + 1)))
        val = prefactor * cplus ** (n - m) * eval_genlaguerre(m, n - m, x)
    else:
        prefactor = np.exp(-x / 2.0 + 0.5 * (gammaln(n + 1) - gammaln(m + 1)))
        val = prefactor * cminus ** (m - n) * eval_genlaguerre(n, m - n, x)
    val = complex(val)
    return 0.0 + 0.0j if abs(val) < 1e-16 else val


def associated_laguerre_matrix(n_landau: int, qvec: complex, l_b: float) -> Array:
    cplus = -1j * float(l_b) / np.sqrt(2.0) * (qvec.real - 1j * qvec.imag)
    cminus = -1j * float(l_b) / np.sqrt(2.0) * (qvec.real + 1j * qvec.imag)
    n_ll = int(n_landau)
    out = np.zeros((n_ll, n_ll), dtype=np.complex128)
    x = -float((cplus * cminus).real)
    for delta in range(n_ll):
        idx = np.arange(n_ll - delta)
        prefactor = np.exp(
            -x / 2.0 + 0.5 * (gammaln(idx + 1.0) - gammaln(idx + delta + 1.0))
        )
        laguerre = eval_genlaguerre(idx, delta, x)
        lower = prefactor * cplus**delta * laguerre
        lower = np.asarray(lower, dtype=np.complex128)
        lower[np.abs(lower) < 1e-16] = 0.0
        out[idx + delta, idx] = lower
        if delta == 0:
            continue
        upper = prefactor * cminus**delta * laguerre
        upper = np.asarray(upper, dtype=np.complex128)
        upper[np.abs(upper) < 1e-16] = 0.0
        out[idx, idx + delta] = upper
    return out


def tll_matrix(
    tunnel: Array,
    qvec: complex,
    *,
    n_landau: int,
    n_h: int,
    l_b: float,
    theta0: float,
    theta1: float,
    theta2: float,
    sigma_rotation: bool = False,
    valley: Valley = "K",
) -> Array:
    """Return the LL matrix element of ``T exp(-i q.r)``.

    This is the Python form of ``_tLL_v1`` and ``_tLL_v1_valleyKprime``.
    """

    t = np.asarray(tunnel, dtype=np.complex128)
    al = associated_laguerre_matrix(n_landau, qvec, l_b)
    out = np.zeros((n_h, n_h), dtype=np.complex128)
    is_kprime = valley == "Kprime"

    for i1 in range(n_h):
        n1, gamma1 = in_gamma(i1)
        for i2 in range(n_h):
            n2, gamma2 = in_gamma(i2)
            if not is_kprime:
                if sigma_rotation:
                    if n1 != 0 and n2 != 0:
                        out[i1, i2] = (
                            t[0, 0] * gamma1 * gamma2 * np.exp(1j * (theta2 - theta1)) * al[n1 - 1, n2 - 1]
                            + t[1, 1] * al[n1, n2]
                            + t[0, 1] * (1j * gamma1 * np.exp(-1j * (theta1 - theta0))) * al[n1 - 1, n2]
                            + t[1, 0] * (-1j * gamma2 * np.exp(1j * (theta2 - theta0))) * al[n1, n2 - 1]
                        ) / 2.0
                    elif n1 == 0 and n2 == 0:
                        out[i1, i2] = t[1, 1] * al[0, 0]
                    elif n1 == 0:
                        out[i1, i2] = (t[1, 1] * al[0, n2] + t[1, 0] * (-1j * gamma2 * np.exp(1j * (theta2 - theta0))) * al[0, n2 - 1]) / np.sqrt(2.0)
                    else:
                        out[i1, i2] = (t[1, 1] * al[n1, 0] + t[0, 1] * (1j * gamma1 * np.exp(-1j * (theta1 - theta0))) * al[n1 - 1, 0]) / np.sqrt(2.0)
                else:
                    if n1 != 0 and n2 != 0:
                        out[i1, i2] = (
                            t[0, 0] * gamma1 * gamma2 * al[n1 - 1, n2 - 1]
                            + t[1, 1] * al[n1, n2]
                            + t[0, 1] * (1j * gamma1 * np.exp(1j * theta0)) * al[n1 - 1, n2]
                            + t[1, 0] * (-1j * gamma2 * np.exp(-1j * theta0)) * al[n1, n2 - 1]
                        ) / 2.0
                    elif n1 == 0 and n2 == 0:
                        out[i1, i2] = t[1, 1] * al[0, 0]
                    elif n1 == 0:
                        out[i1, i2] = (t[1, 1] * al[0, n2] + t[1, 0] * (-1j * gamma2 * np.exp(-1j * theta0)) * al[0, n2 - 1]) / np.sqrt(2.0)
                    else:
                        out[i1, i2] = (t[1, 1] * al[n1, 0] + t[0, 1] * (1j * gamma1 * np.exp(1j * theta0)) * al[n1 - 1, 0]) / np.sqrt(2.0)
            else:
                if sigma_rotation:
                    if n1 != 0 and n2 != 0:
                        out[i1, i2] = (
                            t[0, 0] * al[n1, n2]
                            + t[1, 1] * gamma1 * gamma2 * np.exp(1j * (theta2 - theta1)) * al[n1 - 1, n2 - 1]
                            + t[0, 1] * (1j * gamma2 * np.exp(1j * (theta2 - theta0))) * al[n1, n2 - 1]
                            + t[1, 0] * (-1j * gamma1 * np.exp(-1j * (theta1 - theta0))) * al[n1 - 1, n2]
                        ) / 2.0
                    elif n1 == 0 and n2 == 0:
                        out[i1, i2] = t[0, 0] * al[0, 0]
                    elif n2 == 0:
                        out[i1, i2] = (t[0, 0] * al[n1, 0] + t[1, 0] * (-1j * gamma1 * np.exp(-1j * (theta1 - theta0))) * al[n1 - 1, 0]) / np.sqrt(2.0)
                    else:
                        out[i1, i2] = (t[0, 0] * al[0, n2] + t[0, 1] * (1j * gamma2 * np.exp(1j * (theta2 - theta0))) * al[0, n2 - 1]) / np.sqrt(2.0)
                else:
                    if n1 != 0 and n2 != 0:
                        out[i1, i2] = (
                            t[0, 0] * al[n1, n2]
                            + t[1, 1] * gamma1 * gamma2 * al[n1 - 1, n2 - 1]
                            + t[0, 1] * (1j * gamma2 * np.exp(-1j * theta0)) * al[n1, n2 - 1]
                            + t[1, 0] * (-1j * gamma1 * np.exp(1j * theta0)) * al[n1 - 1, n2]
                        ) / 2.0
                    elif n1 == 0 and n2 == 0:
                        out[i1, i2] = t[0, 0] * al[0, 0]
                    elif n2 == 0:
                        out[i1, i2] = (t[0, 0] * al[n1, 0] + t[1, 0] * (-1j * gamma1 * np.exp(1j * theta0)) * al[n1 - 1, 0]) / np.sqrt(2.0)
                    else:
                        out[i1, i2] = (t[0, 0] * al[0, n2] + t[0, 1] * (1j * gamma2 * np.exp(-1j * theta0)) * al[0, n2 - 1]) / np.sqrt(2.0)
    return out


def magnetic_lattice_coordinates(nq: int, q: int, *, valley: Valley = "K", q0: complex = 0.0 + 0.0j, mesh_shift: float = 0.0) -> tuple[Array, Array]:
    """Return author-code lattice coordinates for ``lk=nq*q``."""

    lk = int(nq) * int(q)
    k1 = np.arange(lk, dtype=float) / float(lk) + float(mesh_shift)
    k2 = np.arange(lk, dtype=float) / float(lk) + float(mesh_shift)
    if valley == "Kprime":
        k1 = k1 + float(np.real(q0)) / float(lk)
        k2 = k2 + float(np.imag(q0)) / float(lk)
    return k1, k2


def qjs_for_valley(valley: Valley) -> tuple[complex, complex, complex]:
    return (0.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j) if valley == "K" else (0.0 + 0.0j, 0.0 - 1.0j, -1.0 - 1.0j)


def construct_ll_hamiltonian(
    params: FiniteFieldBMParameters,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nq: int,
    valley: Valley = "K",
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
    mesh_shift: float = 0.0,
) -> tuple[Array, Array, Array, Array, float, complex]:
    """Construct the LL Hamiltonian and ``Σz`` operator before projection."""

    p, q = int(flux.p), int(flux.q)
    n_h = int(n_landau) * 2 - 1
    k1, k2 = magnetic_lattice_coordinates(nq, q, valley=valley, q0=q0, mesh_shift=mesh_shift)
    q_phi = 2.0 * np.pi / abs(params.a2) ** 2 * params.a2 * flux.ratio
    l_b = float(np.sqrt(q / (2.0 * np.pi * abs(p)) * params.area))
    h = np.zeros((n_h, p, 2, n_h, p, 2, int(nq), int(nq)), dtype=np.complex128)

    eps_b = params.vf / l_b
    for ih in range(n_h):
        n, gamma = in_gamma(ih)
        for ip in range(p):
            h[ih, ip, 0, ih, ip, 0, :, :] += gamma * np.sqrt(2.0 * n) * eps_b
            h[ih, ip, 1, ih, ip, 1, :, :] += gamma * np.sqrt(2.0 * n) * eps_b

    if hbn:
        sign_hbn = 1.0 if valley == "K" else -1.0
        for ip in range(p):
            h[0, ip, 0, 0, ip, 0, :, :] += -params.delta * sign_hbn
            for n in range(1, int(n_landau)):
                for ig1 in range(2):
                    gamma1 = 2 * (ig1 + 1) - 3
                    ih1 = (n - 1) * 2 + ig1 + 1
                    for ig2 in range(2):
                        gamma2 = 2 * (ig2 + 1) - 3
                        ih2 = (n - 1) * 2 + ig2 + 1
                        h[ih1, ip, 0, ih2, ip, 0, :, :] += -(1.0 - gamma1 * gamma2) / 2.0 * params.delta * sign_hbn

    if include_strain:
        def_pot = params.deformation_potential * (params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0
        for ih in range(n_h):
            for ip in range(p):
                h[ih, ip, 0, ih, ip, 0, :, :] -= def_pot
                h[ih, ip, 1, ih, ip, 1, :, :] += def_pot

    theta_strain = np.angle(params.a2) - np.pi / 2.0
    tunnels = (params.t0, params.t1, params.t2)
    for qj, tunnel in zip(qjs_for_valley(valley), tunnels, strict=True):
        q_lab = qj.real * params.g1 + qj.imag * params.g2
        q_lab = q_lab + (params.kb_point - params.kt if valley == "K" else params.kt - params.kb_point)
        q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
        tmat = tll_matrix(
            tunnel if valley == "K" else np.conj(tunnel),
            q_rot,
            n_landau=int(n_landau),
            n_h=n_h,
            l_b=l_b,
            theta0=theta_strain,
            theta1=-params.dtheta_rad / 2.0,
            theta2=params.dtheta_rad / 2.0,
            sigma_rotation=sigma_rotation,
            valley=valley,
        )
        kl = params.kb_point if valley == "K" else -params.kb_point
        kr = params.kt if valley == "K" else -params.kt
        for ik2 in range(int(nq)):
            for r1 in range(p):
                k2l = projector_para(params.g2, params.a2) * k2[r1 * int(nq) + ik2] - projector_para(kl, params.a2)
                raw_r2 = r1 + q * int(qj.imag)
                r2 = raw_r2 % p
                s = -((raw_r2 - r2) // p)
                p2 = k2[r2 * int(nq) + ik2]
                for ik1 in range(int(nq)):
                    expfactor = (
                        np.exp(1j * 2.0 * np.pi * s * (k1[ik1] - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                        * np.exp(1j * s * s / 2.0 * projector_para(q_phi, params.a1) * abs(params.a1))
                        * np.exp(-1j * s * projector_norm(kr, params.a2) * projector_norm(params.a1, params.a2))
                        * np.exp(1j * q_rot.real * k2l * l_b**2)
                        * np.exp(1j * q_rot.real * q_rot.imag * l_b**2 / 2.0)
                    )
                    h[:, r1, 0, :, r2, 1, ik1, ik2] += expfactor * tmat

    sigma_z = construct_sigma_z_ll(n_landau=int(n_landau), p=p, valley=valley)
    return h, sigma_z, k1, k2, l_b, q_phi


def construct_sigma_z_ll(*, n_landau: int, p: int, valley: Valley = "K") -> Array:
    n_h = int(n_landau) * 2 - 1
    sigma = np.zeros((n_h, int(p), 2, n_h, int(p), 2), dtype=np.complex128)
    sign = 1.0 if valley == "K" else -1.0
    for ip in range(int(p)):
        sigma[0, ip, 0, 0, ip, 0] = -sign
        sigma[0, ip, 1, 0, ip, 1] = -sign
        for n in range(1, int(n_landau)):
            for ig1 in range(2):
                gamma1 = 2 * (ig1 + 1) - 3
                ih1 = (n - 1) * 2 + ig1 + 1
                for ig2 in range(2):
                    gamma2 = 2 * (ig2 + 1) - 3
                    ih2 = (n - 1) * 2 + ig2 + 1
                    value = -(1.0 - gamma1 * gamma2) / 2.0 * sign
                    sigma[ih1, ip, 0, ih2, ip, 0] = value
                    sigma[ih1, ip, 1, ih2, ip, 1] = value
    return sigma


def _matrix_from_ll_tensor(tensor: Array, i1: int, i2: int) -> Array:
    block = np.asarray(tensor[..., int(i1), int(i2)], dtype=np.complex128)
    dim = int(np.prod(block.shape[:3]))
    return block.reshape((dim, dim), order="F")


def _hermitian_from_upper(mat: Array) -> Array:
    """Return the Hermitian matrix represented by the upper triangle of ``mat``.

    Julia's ``Hermitian(H, :U)`` keeps the diagonal once and mirrors only the
    strict upper triangle.  Do the same here; adding the diagonal twice would
    double the LL kinetic energies.
    """

    return np.triu(mat) + np.triu(mat, 1).conj().T


def generate_magnetic_translation_orbit(vec: Array, *, q: int, p: int, nq: int) -> None:
    """Fill the ``r=1..q-1`` magnetic-translation orbit in-place."""

    base = vec[:, :, 0, :, :].reshape((-1, int(p), 2, 2 * int(q), int(nq), int(nq)), order="F")
    rk2 = np.arange(int(p), dtype=float)
    for r1 in range(1, int(q)):
        multiplier = 0
        while multiplier < int(q):
            if (multiplier * int(p)) % int(q) == r1:
                break
            multiplier += 1
        phase = np.exp(-1j * 2.0 * np.pi * multiplier * (rk2 / float(q)))
        shifted = base * phase.reshape((1, int(p), 1, 1, 1, 1))
        vec[:, :, r1, :, :] = shifted.reshape((vec.shape[0], 2 * int(q), int(nq), int(nq)), order="F")


def compute_magnetic_spectrum(
    params: FiniteFieldBMParameters,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nq: int,
    valley: Valley = "K",
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
    mesh_shift: float = 0.0,
) -> MagneticSpectrumResult:
    """Construct and diagonalize the central finite-B Hofstadter subbands."""

    h, sigma_z_ll, k1, k2, l_b, q_phi = construct_ll_hamiltonian(
        params,
        flux=flux,
        n_landau=n_landau,
        nq=nq,
        valley=valley,
        sigma_rotation=sigma_rotation,
        hbn=hbn,
        include_strain=include_strain,
        q0=q0,
        mesh_shift=mesh_shift,
    )
    p, q = flux.p, flux.q
    n_h = int(n_landau) * 2 - 1
    inner = 2 * n_h * p
    n_sub = 2 * q
    spectrum = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    vec = np.zeros((inner, n_sub, q, int(nq), int(nq)), dtype=np.complex128)
    p_sigma_z = np.zeros((n_sub, n_sub, int(nq), int(nq)), dtype=np.complex128)
    sigma_z_eigs = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    sigma_z_energy = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    sigma_mat = sigma_z_ll.reshape((inner, inner), order="F")
    start = n_h * p - q
    stop = n_h * p + q - 1
    if start < 0 or stop >= inner:
        raise ValueError(f"Central subband window [{start},{stop}] outside inner dimension {inner}; increase n_landau")

    for i2 in range(int(nq)):
        for i1 in range(int(nq)):
            hmat_upper = _matrix_from_ll_tensor(h, i1, i2)
            hmat = _hermitian_from_upper(hmat_upper)
            vals, evecs = eigh(hmat, subset_by_index=(start, stop), check_finite=False)
            order = np.argsort(vals.real, kind="stable")
            vals = vals[order].real
            evecs = evecs[:, order]
            spectrum[:, i1, i2] = vals
            vec[:, :, 0, i1, i2] = evecs
            psigma = evecs.conj().T @ sigma_mat @ evecs
            p_sigma_z[:, :, i1, i2] = psigma
            sigma_z_energy[:, i1, i2] = np.diag(psigma).real
            sigma_z_eigs[:, i1, i2] = np.linalg.eigvalsh((psigma + psigma.conj().T) / 2.0).real
    generate_magnetic_translation_orbit(vec, q=q, p=p, nq=int(nq))
    return MagneticSpectrumResult(
        params=params,
        flux=flux,
        valley=valley,
        n_landau=int(n_landau),
        n_h=n_h,
        nq=int(nq),
        sigma_rotation=bool(sigma_rotation),
        l_b=float(l_b),
        q_phi=complex(q_phi),
        lattice_k1=np.asarray(k1, dtype=float),
        lattice_k2=np.asarray(k2, dtype=float),
        hamiltonian_ll=h,
        sigma_z_ll=sigma_z_ll,
        spectrum=spectrum,
        vec=vec,
        p_sigma_z=p_sigma_z,
        sigma_z_eigenvalues=sigma_z_eigs,
        sigma_z_energy_diag=sigma_z_energy,
    )


def compute_coulomb_overlap(result: MagneticSpectrumResult, m: int, n: int) -> Array:
    """Compute one projected density-overlap matrix ``Λ_(m,n)``.

    This is a direct, readable implementation with the same layer/K-point
    convention as the author production routine ``computeCoulombOverlap_v2``.
    Use :func:`compute_coulomb_overlap_fast` for the symmetry-reduced version.
    """

    params = result.params
    p, q, nq = result.p, result.q, result.nq
    n_h = result.n_h
    inner = result.inner_dim
    n_sub = 2 * q
    theta_strain = np.angle(params.a2) - np.pi / 2.0
    tmp = np.zeros((n_sub, q, nq, nq, n_sub, q, nq, nq), dtype=np.complex128)
    lambda_psi = np.zeros((inner, inner), dtype=np.complex128)
    lambda_psi_view = lambda_psi.reshape((n_h, p, 2, n_h, p, 2), order="F")
    identity_sublattice = np.eye(2, dtype=np.complex128)

    for ip2 in range(nq):
        for ik2 in range(nq):
            for ip1 in range(nq):
                for rp1 in range(q):
                    for ik1 in range(nq):
                        for rk1 in range(q):
                            lambda_psi[:, :] = 0.0
                            k1 = result.lattice_k1[ik1 + rk1 * nq]
                            p1 = result.lattice_k1[ip1 + rp1 * nq]
                            for rk2 in range(p):
                                rp2 = (rk2 + int(n)) % p
                                s = -((rk2 + int(n) - rp2) // p)
                                k2 = result.lattice_k2[ik2 + rk2 * nq]
                                p2 = result.lattice_k2[ip2 + rp2 * nq]
                                k20 = result.lattice_k2[ik2]
                                p20 = result.lattice_k2[ip2]
                                q_lab = (p1 - k1 + int(m)) * params.g1 + (p20 - k20 + int(n) / q) * params.g2
                                q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
                                for layer in range(2):
                                    if result.valley == "K":
                                        kl = params.kb_point if layer == 0 else params.kt
                                    else:
                                        kl = params.kt if layer == 0 else params.kb_point
                                    theta_l = (2 * (layer + 1) - 3) * params.dtheta_rad / 2.0
                                    k2l = projector_para(params.g2, params.a2) * k2 - projector_para(kl, params.a2)
                                    expfactor = (
                                        np.exp(1j * 2.0 * np.pi * s * (p1 - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                                        * np.exp(1j * s * s / 2.0 * projector_para(result.q_phi, params.a1) * abs(params.a1))
                                        * np.exp(-1j * s * projector_norm(kl, params.a2) * projector_norm(params.a1, params.a2))
                                        * np.exp(1j * q_rot.real * k2l * result.l_b**2)
                                        * np.exp(1j * q_rot.real * q_rot.imag * result.l_b**2 / 2.0)
                                    )
                                    lambda_psi_view[:, rk2, layer, :, rp2, layer] = tll_matrix(
                                        identity_sublattice,
                                        q_rot,
                                        n_landau=result.n_landau,
                                        n_h=n_h,
                                        l_b=result.l_b,
                                        theta0=theta_strain,
                                        theta1=theta_l,
                                        theta2=theta_l,
                                        sigma_rotation=result.sigma_rotation,
                                        valley=result.valley,
                                    ) * expfactor
                            left = result.vec[:, :, rk1, ik1, ik2]
                            right = result.vec[:, :, rp1, ip1, ip2]
                            tmp[:, rk1, ik1, ik2, :, rp1, ip1, ip2] = left.conj().T @ lambda_psi @ right
    return tmp.reshape((n_sub * q * nq * nq, n_sub * q * nq * nq), order="F")


def _overlap_slice(
    result: MagneticSpectrumResult,
    m: int,
    n: int,
    *,
    ik2: int,
    ip2: int,
    ik1: int,
    ip1: int,
    rk1: int,
    rp1: int,
    tll_cache: dict[tuple[int, int, int], Array] | None = None,
) -> Array:
    params = result.params
    p, q, nq = result.p, result.q, result.nq
    n_h = result.n_h
    inner = result.inner_dim
    ik2_i = int(ik2)
    ip2_i = int(ip2)
    ik1_i = int(ik1)
    ip1_i = int(ip1)
    rk1_i = int(rk1)
    rp1_i = int(rp1)
    m_i = int(m)
    n_i = int(n)
    theta_strain = np.angle(params.a2) - np.pi / 2.0
    lambda_psi = np.zeros((inner, inner), dtype=np.complex128)
    lambda_psi_view = lambda_psi.reshape((n_h, p, 2, n_h, p, 2), order="F")
    identity_sublattice = np.eye(2, dtype=np.complex128)
    k1 = result.lattice_k1[ik1_i + rk1_i * nq]
    p1 = result.lattice_k1[ip1_i + rp1_i * nq]
    k20 = result.lattice_k2[ik2_i]
    p20 = result.lattice_k2[ip2_i]
    q_lab = (p1 - k1 + m_i) * params.g1 + (p20 - k20 + n_i / q) * params.g2
    q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
    # The lattice mesh is an arithmetic progression, so q0/mesh_shift cancel
    # from the momentum transfer.  Cache the expensive LL translation matrix by
    # this exact integer transfer and by layer-specific theta arguments.
    delta1_num = (ip1_i + rp1_i * nq) - (ik1_i + rk1_i * nq) + m_i * nq * q
    delta2_num = (ip2_i - ik2_i) + n_i * nq
    for rk2 in range(p):
        rp2 = (rk2 + n_i) % p
        s = -((rk2 + n_i - rp2) // p)
        k2 = result.lattice_k2[ik2_i + rk2 * nq]
        p2 = result.lattice_k2[ip2_i + rp2 * nq]
        for layer in range(2):
            if result.valley == "K":
                kl = params.kb_point if layer == 0 else params.kt
            else:
                kl = params.kt if layer == 0 else params.kb_point
            theta_l = (2 * (layer + 1) - 3) * params.dtheta_rad / 2.0
            k2l = projector_para(params.g2, params.a2) * k2 - projector_para(kl, params.a2)
            expfactor = (
                np.exp(1j * 2.0 * np.pi * s * (p1 - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                * np.exp(1j * s * s / 2.0 * projector_para(result.q_phi, params.a1) * abs(params.a1))
                * np.exp(-1j * s * projector_norm(kl, params.a2) * projector_norm(params.a1, params.a2))
                * np.exp(1j * q_rot.real * k2l * result.l_b**2)
                * np.exp(1j * q_rot.real * q_rot.imag * result.l_b**2 / 2.0)
            )
            cache_key = (delta1_num, delta2_num, layer)
            if tll_cache is None:
                tll = tll_matrix(
                    identity_sublattice,
                    q_rot,
                    n_landau=result.n_landau,
                    n_h=n_h,
                    l_b=result.l_b,
                    theta0=theta_strain,
                    theta1=theta_l,
                    theta2=theta_l,
                    sigma_rotation=result.sigma_rotation,
                    valley=result.valley,
                )
            else:
                tll = tll_cache.get(cache_key)
                if tll is None:
                    tll = tll_matrix(
                        identity_sublattice,
                        q_rot,
                        n_landau=result.n_landau,
                        n_h=n_h,
                        l_b=result.l_b,
                        theta0=theta_strain,
                        theta1=theta_l,
                        theta2=theta_l,
                        sigma_rotation=result.sigma_rotation,
                        valley=result.valley,
                    )
                    tll_cache[cache_key] = tll
            lambda_psi_view[:, rk2, layer, :, rp2, layer] = tll * expfactor
    left = result.vec[:, :, rk1_i, ik1_i, ik2_i]
    right = result.vec[:, :, rp1_i, ip1_i, ip2_i]
    return left.conj().T @ lambda_psi @ right


def compute_coulomb_overlap_fast(result: MagneticSpectrumResult, m: int, n: int) -> Array:
    """Symmetry-reduced port of author ``computeCoulombOverlap_v2``.

    The result has the same flattened shape as :func:`compute_coulomb_overlap`,
    but only explicitly computes the first row/column of magnetic-translation
    strip blocks and reconstructs the rest with the author phase factors.
    """

    p, q, nq = result.p, result.q, result.nq
    n_sub = 2 * q
    tmp = np.zeros((n_sub, q, nq, nq, n_sub, q, nq, nq), dtype=np.complex128)
    ips = np.asarray([(rp * p) % q for rp in range(q)], dtype=int)
    tll_cache: dict[tuple[int, int, int], Array] = {}
    for ip2 in range(nq):
        for ik2 in range(nq):
            for ip1 in range(nq):
                for rp1 in range(q):
                    for ik1 in range(nq):
                        tmp[:, 0, ik1, ik2, :, rp1, ip1, ip2] = _overlap_slice(
                            result, m, n, ik2=ik2, ip2=ip2, ik1=ik1, ip1=ip1, rk1=0, rp1=rp1, tll_cache=tll_cache
                        )
            for ip1 in range(nq):
                for ik1 in range(nq):
                    for rk1 in range(1, q):
                        tmp[:, rk1, ik1, ik2, :, 0, ip1, ip2] = _overlap_slice(
                            result, m, n, ik2=ik2, ip2=ip2, ik1=ik1, ip1=ip1, rk1=rk1, rp1=0, tll_cache=tll_cache
                        )

            for rp1 in range(1, q):
                for rk1 in range(1, q):
                    lhs_rk = int(ips[rk1])
                    lhs_rp = int(ips[rp1])
                    if ips[rp1] > ips[rk1]:
                        source_rp = int(ips[rp1] - ips[rk1])
                        tmp[:, lhs_rk, :, ik2, :, lhs_rp, :, ip2] = tmp[:, 0, :, ik2, :, source_rp, :, ip2] * np.exp(
                            -1j * 2.0 * np.pi * rk1 * int(n) / q
                        )
                    else:
                        source_rk = int(ips[rk1] - ips[rp1])
                        tmp[:, lhs_rk, :, ik2, :, lhs_rp, :, ip2] = tmp[:, source_rk, :, ik2, :, 0, :, ip2] * np.exp(
                            -1j * 2.0 * np.pi * rp1 * int(n) / q
                        )
    return tmp.reshape((n_sub * q * nq * nq, n_sub * q * nq * nq), order="F")


__all__ = [
    "FiniteFieldBMParameters",
    "MagneticSpectrumResult",
    "MagneticSpectrumSweepCase",
    "MagneticSpectrumSweepResult",
    "Valley",
    "associated_laguerre_element",
    "associated_laguerre_matrix",
    "author_landau_cutoff",
    "compute_coulomb_overlap",
    "compute_coulomb_overlap_fast",
    "compute_magnetic_spectrum",
    "compute_magnetic_spectrum_sweep",
    "construct_ll_hamiltonian",
    "construct_sigma_z_ll",
    "generate_magnetic_translation_orbit",
    "in_gamma",
    "magnetic_lattice_coordinates",
    "paper_hofstadter_fluxes",
    "projector_norm",
    "projector_para",
    "qjs_for_valley",
    "red_chern_minus_one_group_mask",
    "tll_matrix",
]
