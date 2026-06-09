"""Finite-magnetic-field Hartree-Fock adapter for TBG.

This module is a Python port of the Hartree-Fock part of the author code in
``TBG_HartreeFock（作者原始代码）/libs/MagneticFieldHF*.jl``.  It intentionally
keeps the TBG/finite-B physics in the system layer while reusing the generic
SCF/ODA contracts from :mod:`mean_field.core.hf`.

Reference convention
--------------------
The formulas follow Zhang et al. arXiv:2310.15982v3, SI Sec. III:

* density matrix ``P(k) = <d^† d> - I/2`` (Eq. S54), stored in the same
  transposed-projector convention as the Julia code;
* projected interaction ``1/2A sum_q V_q delta rho_q delta rho_-q`` (Eqs.
  S45-S61);
* mean-field Hamiltonian ``H_MF = T + Hartree - Fock`` (Eq. S64);
* self-consistency by diagonalizing ``H_MF`` and updating the projector so
  ``[H_MF, P] = 0`` (Eq. S67).

The non-interacting Hofstadter/LL basis generation is not duplicated here: a
system adapter should provide the projected Hofstadter energies, ``PΣz`` maps,
magnetic k-mesh, and precomputed density-overlap matrices ``Λ``.  This file
ports the HF assembly, finite-B normalization, filling/initialization, and the
optional magnetic-translation-symmetric IKS reduction used by the author code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np

from ....core.magnetic_field import (
    MagneticFlux,
    choose_magnetic_nq,
    diophantine_filling,
    in_hex_shell,
    magnetic_k_vectors,
    magnetic_normalization_count,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_reciprocal_vector,
    magnetic_shell_shifts,
)
from ....core.hf import (
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    calculate_norm_convergence,
    compute_density_overlap_trace_from_diagonal,
    compute_oda_parameter,
    contract_fock_term_from_overlap,
    find_chemical_potential,
    occupied_state_linear_indices,
    run_hartree_fock_problem,
)

Array = np.ndarray
InitMode = Literal["bm", "random", "flavor", "bm_cascade", "sublattice"]



@dataclass(frozen=True)
class MagneticOverlapData:
    """Projected density-overlap matrices for finite-B HF.

    ``overlaps[(m, n)]`` stores ``Λ_G`` with ``G = m*g1 + (n/q)*g2`` and shape
    ``(nt, nk_target, nt, nk_source)``.  For the full magnetic-BZ calculation
    ``nk_target == nk_source == q*nq**2``.  For the tL-symmetric reduced IKS
    calculation, overlaps are still stored on the full strip orbit with
    ``nk_full = q*nq**2`` while the density lives on ``nq**2`` reduced momenta.
    """

    shifts: tuple[tuple[int, int], ...]
    gvecs: Array
    overlaps: Mapping[tuple[int, int], Array]

    def __post_init__(self) -> None:
        shifts = tuple((int(m), int(n)) for m, n in self.shifts)
        gvecs = np.asarray(self.gvecs, dtype=np.complex128)
        if gvecs.shape != (len(shifts),):
            raise ValueError(f"Expected gvecs shape {(len(shifts),)}, got {gvecs.shape}")
        for shift in shifts:
            if shift not in self.overlaps:
                raise ValueError(f"Missing overlap block for shift {shift}")
            block = np.asarray(self.overlaps[shift], dtype=np.complex128)
            if block.ndim != 4 or block.shape[0] != block.shape[2]:
                raise ValueError(f"Overlap {shift} must have shape (nt,nk_t,nt,nk_s), got {block.shape}")
        object.__setattr__(self, "shifts", shifts)
        object.__setattr__(self, "gvecs", gvecs)

    @classmethod
    def from_overlap_mapping(
        cls,
        overlaps: Mapping[tuple[int, int], Array],
        *,
        g1: complex,
        g2: complex,
        q: int,
        shell_ng: int | None = None,
    ) -> "MagneticOverlapData":
        shifts = tuple(sorted((int(m), int(n)) for m, n in overlaps))
        if shell_ng is not None:
            shifts = tuple(shift for shift in shifts if in_hex_shell(shift[0], shift[1], g1=g1, g2=g2, q=q, shell_ng=shell_ng))
        gvecs = np.asarray([magnetic_reciprocal_vector(m, n, g1=g1, g2=g2, q=q) for m, n in shifts], dtype=np.complex128)
        return cls(shifts=shifts, gvecs=gvecs, overlaps={shift: np.asarray(overlaps[shift], dtype=np.complex128) for shift in shifts})


@dataclass(frozen=True)
class FiniteFieldHartreeFockInputs:
    """Fully assembled finite-B HF inputs for one flux point.

    This is a small no-I/O bundle: workflow code may decide how to cache or
    persist these arrays, while the system layer owns the physics conventions
    and array ordering needed by the HF kernel.
    """

    state: "FiniteFieldHartreeFockState"
    overlap_data: MagneticOverlapData
    k_vectors: Array
    normalization_count: int

    def __post_init__(self) -> None:
        kvec = np.asarray(self.k_vectors, dtype=np.complex128)
        if kvec.shape != (self.state.nk,):
            raise ValueError(f"Expected k_vectors shape {(self.state.nk,)}, got {kvec.shape}")
        if int(self.normalization_count) <= 0:
            raise ValueError("normalization_count must be positive")
        object.__setattr__(self, "k_vectors", kvec)
        object.__setattr__(self, "normalization_count", int(self.normalization_count))

@dataclass(frozen=True)
class FiniteFieldTLSymmetricHartreeFockInputs:
    """Assembled reduced tL-symmetric / IKS finite-B HF inputs."""

    state: "FiniteFieldHartreeFockState"
    overlap_data: MagneticOverlapData
    full_k_vectors: Array
    normalization_count: int

    def __post_init__(self) -> None:
        full_k = np.asarray(self.full_k_vectors, dtype=np.complex128)
        expected = self.state.q * self.state.nk
        if not self.state.reduced_translation:
            raise ValueError("tL-symmetric inputs require a reduced_translation=True state")
        if full_k.shape != (expected,):
            raise ValueError(f"Expected full_k_vectors shape {(expected,)}, got {full_k.shape}")
        if int(self.normalization_count) <= 0:
            raise ValueError("normalization_count must be positive")
        object.__setattr__(self, "full_k_vectors", full_k)
        object.__setattr__(self, "normalization_count", int(self.normalization_count))

FiniteFieldHartreeFockInputBundle = FiniteFieldHartreeFockInputs | FiniteFieldTLSymmetricHartreeFockInputs
"""Either full magnetic-BZ or reduced tL-symmetric finite-B HF input bundle."""

@dataclass(frozen=True)
class FiniteFieldHartreeFockSummary:
    """Small no-I/O summary of a finite-B HF state/run.

    ``single_particle_gap`` is the occupied/unoccupied HF eigenvalue gap in the
    stored finite system; it is not a many-body charge gap.
    """

    filling: float
    energy_per_muc: float
    mu: float
    single_particle_gap: float
    final_raw_norm: float
    iterations: int
    converged: bool
    exit_reason: str

@dataclass
class FiniteFieldHartreeFockState:
    """Finite-B HF state in the stored-projector convention of the Julia code."""

    h0: Array
    sigma_z: Array
    density: Array
    hamiltonian: Array
    energies: Array
    sigma_ztauz: Array
    nu: float
    flux: MagneticFlux
    nq: int
    v0: float
    mu: float = float("nan")
    precision: float = 1e-5
    n_eta: int = 2
    n_spin: int = 2
    n_band: int = 2
    reduced_translation: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.h0 = np.asarray(self.h0, dtype=np.complex128)
        self.sigma_z = np.asarray(self.sigma_z, dtype=np.complex128)
        self.density = np.asarray(self.density, dtype=np.complex128)
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        self.energies = np.asarray(self.energies, dtype=float)
        self.sigma_ztauz = np.asarray(self.sigma_ztauz, dtype=float)
        if self.h0.ndim != 3 or self.h0.shape[0] != self.h0.shape[1]:
            raise ValueError(f"Expected h0 shape (nt,nt,nk), got {self.h0.shape}")
        if self.density.shape != self.h0.shape or self.hamiltonian.shape != self.h0.shape:
            raise ValueError("density and hamiltonian must match h0 shape")
        if self.sigma_z.shape != self.h0.shape:
            raise ValueError("sigma_z must match h0 shape")
        if self.energies.shape != (self.nt, self.nk):
            raise ValueError(f"Expected energies shape {(self.nt, self.nk)}, got {self.energies.shape}")
        if self.sigma_ztauz.shape != (self.nt, self.nk):
            raise ValueError(f"Expected sigma_ztauz shape {(self.nt, self.nk)}, got {self.sigma_ztauz.shape}")

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def q(self) -> int:
        return int(self.flux.q)

    @property
    def p(self) -> int:
        return int(self.flux.p)

    @property
    def subbands_per_flavor(self) -> int:
        return int(self.n_band * self.q)

    @classmethod
    def from_h0(
        cls,
        h0: Array,
        *,
        sigma_z: Array | None = None,
        nu: float,
        flux: MagneticFlux,
        nq: int,
        v0: float,
        precision: float = 1e-5,
        reduced_translation: bool = False,
    ) -> "FiniteFieldHartreeFockState":
        h0_arr = np.asarray(h0, dtype=np.complex128)
        nt, _, nk = h0_arr.shape
        sigma = np.zeros_like(h0_arr) if sigma_z is None else np.asarray(sigma_z, dtype=np.complex128)
        return cls(
            h0=h0_arr,
            sigma_z=sigma,
            density=np.zeros_like(h0_arr),
            hamiltonian=h0_arr.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_ztauz=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            flux=flux,
            nq=int(nq),
            v0=float(v0),
            precision=float(precision),
            reduced_translation=bool(reduced_translation),
        )


def finite_field_diophantine_filling(
    s: int,
    t: int,
    flux: MagneticFlux | Fraction | tuple[int, int] | str,
) -> float:
    """Return the Fig. 6 / Streda-line filling ``nu = s + t*phi/phi0``.

    The paper labels finite-B CHF branches by integer pairs ``(s,t)``. For a
    rational flux ``phi/phi0=p/q``, the density plotted in Fig. 6 is
    ``nu = s + t*p/q`` in units of ``n_s``. Keeping this as a small helper
    avoids hard-coded decimal fillings in replay wrappers.
    """

    return diophantine_filling(s, t, flux)


def paper_fig6_finite_b_fluxes() -> tuple[MagneticFlux, ...]:
    """Return the selected finite-B flux grid used by the Fig. 6 replay scripts.

    The order mirrors ``proj/run_fig6_bfield_selected_analysis.jl`` in the
    author-code workspace: high-field points first, then lower fields down to
    ``1/12``.
    """

    return (
        MagneticFlux(1, 2),
        MagneticFlux(2, 5),
        MagneticFlux(1, 3),
        MagneticFlux(2, 7),
        MagneticFlux(1, 4),
        MagneticFlux(2, 9),
        MagneticFlux(1, 5),
        MagneticFlux(2, 11),
        MagneticFlux(1, 6),
        MagneticFlux(1, 8),
        MagneticFlux(1, 12),
    )


def paper_fig6_branch_cases(
    s: int,
    t: int,
    *,
    fluxes: Sequence[MagneticFlux | Fraction | tuple[int, int] | str] | None = None,
) -> tuple[tuple[MagneticFlux, float], ...]:
    """Return ``(flux, nu)`` cases for one Fig. 6 finite-B ``(s,t)`` branch."""

    selected_fluxes = paper_fig6_finite_b_fluxes() if fluxes is None else tuple(
        flux if isinstance(flux, MagneticFlux) else MagneticFlux.from_value(flux) for flux in fluxes
    )
    return tuple((flux, finite_field_diophantine_filling(s, t, flux)) for flux in selected_fluxes)


def screened_coulomb_finite_b(
    qvec: complex,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
) -> float:
    """Dual-gate screened Coulomb kernel used in 2310.15982v3.

    ``V(q)=2π/(epsilon_r |q|) tanh(|q|*4*Lm/2)`` and ``V(0)=0`` in the
    author code because the density operator is background subtracted.
    """

    q_abs = abs(qvec)
    if q_abs < zero_cutoff:
        return 0.0
    return float(2.0 * np.pi / (relative_permittivity * q_abs) * np.tanh(q_abs * 4.0 * float(lm) / 2.0))


def coulomb_unit_from_lattice(a1: complex, a2: complex) -> float:
    electron_charge = 1.6e-19
    vacuum_permittivity = 8.8541878128e-12
    graphene_lattice_constant = 2.46e-10
    area_moire = abs((complex(a1).conjugate() * complex(a2)).imag)
    return float(electron_charge / (4.0 * np.pi * vacuum_permittivity * area_moire * graphene_lattice_constant) * 1e3)


def zeeman_unit_from_area(area: float) -> float:
    """Return the meV Zeeman prefactor multiplying ``p/q`` in the Julia code."""

    hbar = 1.054571817e-34
    electron_mass = 9.1093837e-31
    electron_charge = 1.6e-19
    graphene_lattice_constant = 2.46e-10
    return float(2.0 * np.pi * hbar**2 / (2.0 * electron_mass * float(area) * graphene_lattice_constant**2) / electron_charge * 1000.0)


def state_index(subband: int, eta: int, spin: int, *, subbands_per_flavor: int, n_eta: int = 2) -> int:
    """Flatten ``(subband, eta, spin)`` with Julia/Fortran order."""

    return int(subband + subbands_per_flavor * (eta + n_eta * spin))


def build_h0_from_hofstadter_metadata(
    valley_energies: Sequence[Array],
    valley_sigma_z: Sequence[Array] | None,
    *,
    flux: MagneticFlux,
    nq: int,
    zeeman_unit: float = 0.0,
    reduced_translation: bool = False,
) -> tuple[Array, Array]:
    """Build ``H0`` and projected ``Σz`` from Hofstadter metadata arrays.

    Parameters
    ----------
    valley_energies:
        Two arrays, one for K and one for K', each with shape ``(2q,nq,nq)``.
    valley_sigma_z:
        Optional two arrays with shape ``(2q,2q,nq,nq)``.  The valley sign is
        applied as in ``BM_info``: K gets ``+PΣz`` and K' gets ``-PΣz``.
    reduced_translation:
        If false, repeat the Hofstadter data over the ``q`` magnetic strips and
        return ``nk=q*nq**2``.  If true, return the reduced tL-symmetric mesh
        with ``nk=nq**2``.
    """

    q = int(flux.q)
    nq = int(nq)
    n_sub = 2 * q
    n_eta = 2
    n_spin = 2
    nt = n_sub * n_eta * n_spin
    nk = nq * nq if reduced_translation else q * nq * nq
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    sigma = np.zeros_like(h0)
    if len(valley_energies) != 2:
        raise ValueError("Expected two valley energy arrays: K and K'")
    if valley_sigma_z is not None and len(valley_sigma_z) != 2:
        raise ValueError("Expected two valley sigma_z arrays when provided")

    for eta in range(2):
        energies = np.asarray(valley_energies[eta], dtype=float)
        if energies.shape != (n_sub, nq, nq):
            raise ValueError(f"Expected valley energy shape {(n_sub, nq, nq)}, got {energies.shape}")
        sigma_eta = None if valley_sigma_z is None else np.asarray(valley_sigma_z[eta], dtype=np.complex128)
        if sigma_eta is not None and sigma_eta.shape != (n_sub, n_sub, nq, nq):
            raise ValueError(f"Expected valley sigma_z shape {(n_sub, n_sub, nq, nq)}, got {sigma_eta.shape}")
        valley_sign = 1.0 if eta == 0 else -1.0
        for spin in range(2):
            spin_sign = 1.0 if spin == 0 else -1.0
            for i1 in range(nq):
                for i2 in range(nq):
                    reduced_k = i1 + nq * i2
                    strip_iter = (0,) if reduced_translation else range(q)
                    for r in strip_iter:
                        ik = reduced_k if reduced_translation else r + q * reduced_k
                        for a in range(n_sub):
                            ia = state_index(a, eta, spin, subbands_per_flavor=n_sub)
                            h0[ia, ia, ik] = energies[a, i1, i2] + spin_sign * float(zeeman_unit) * flux.ratio
                        if sigma_eta is not None:
                            for a in range(n_sub):
                                ia = state_index(a, eta, spin, subbands_per_flavor=n_sub)
                                for b in range(n_sub):
                                    ib = state_index(b, eta, spin, subbands_per_flavor=n_sub)
                                    sigma[ia, ib, ik] = valley_sign * sigma_eta[a, b, i1, i2]
    return h0, sigma


def _validate_spectrum_pair(valley_k_spectrum, valley_kprime_spectrum) -> tuple[MagneticFlux, int]:
    flux = valley_k_spectrum.flux
    if not isinstance(flux, MagneticFlux):
        flux = MagneticFlux(int(flux.p), int(flux.q))
    other_flux = valley_kprime_spectrum.flux
    if int(other_flux.p) != flux.p or int(other_flux.q) != flux.q:
        raise ValueError(f"K/Kprime flux mismatch: {flux} vs {other_flux}")
    nq = int(valley_k_spectrum.nq)
    if int(valley_kprime_spectrum.nq) != nq:
        raise ValueError(f"K/Kprime nq mismatch: {nq} vs {valley_kprime_spectrum.nq}")
    if getattr(valley_k_spectrum, "valley", "K") != "K":
        raise ValueError("First spectrum must be valley='K'")
    if getattr(valley_kprime_spectrum, "valley", "Kprime") != "Kprime":
        raise ValueError("Second spectrum must be valley='Kprime'")
    return flux, nq

def build_finite_field_hf_state_from_spectra(
    valley_k_spectrum,
    valley_kprime_spectrum,
    *,
    nu: float,
    v0: float,
    zeeman_unit: float = 0.0,
    precision: float = 1e-5,
    reduced_translation: bool = False,
) -> FiniteFieldHartreeFockState:
    """Build a finite-B HF state directly from K/K' spectrum results.

    The input objects are normally :class:`MagneticSpectrumResult` instances
    from ``spectrum.py``.  Keeping this helper in the HF adapter avoids copying
    spectrum arrays by hand in reproduction scripts.
    """

    flux, nq = _validate_spectrum_pair(valley_k_spectrum, valley_kprime_spectrum)
    h0, sigma = build_h0_from_hofstadter_metadata(
        [valley_k_spectrum.spectrum, valley_kprime_spectrum.spectrum],
        [valley_k_spectrum.p_sigma_z, valley_kprime_spectrum.p_sigma_z],
        flux=flux,
        nq=nq,
        zeeman_unit=zeeman_unit,
        reduced_translation=reduced_translation,
    )
    return FiniteFieldHartreeFockState.from_h0(
        h0,
        sigma_z=sigma,
        nu=nu,
        flux=flux,
        nq=nq,
        v0=v0,
        precision=precision,
        reduced_translation=reduced_translation,
    )

def build_full_flavor_overlap_data_from_spectra(
    valley_k_spectrum,
    valley_kprime_spectrum,
    *,
    shifts: Sequence[tuple[int, int]],
    fast: bool = True,
) -> MagneticOverlapData:
    """Compute K/K' overlap blocks and expand them to spin/valley flavors.

    ``shifts`` should usually come from :func:`magnetic_shell_shifts`.  The
    function does not perform any file I/O; workflow layers can cache the
    returned arrays if desired.
    """

    flux, _nq = _validate_spectrum_pair(valley_k_spectrum, valley_kprime_spectrum)
    normalized_shifts = tuple((int(m), int(n)) for m, n in shifts)
    if not normalized_shifts:
        raise ValueError("At least one overlap shift is required")
    valley_k_data = valley_k_spectrum.overlap_data_for_shifts(normalized_shifts, fast=fast)
    valley_kprime_data = valley_kprime_spectrum.overlap_data_for_shifts(normalized_shifts, fast=fast)
    return expand_valley_overlap_data_to_flavors(valley_k_data, valley_kprime_data, q=flux.q)

def _resolve_overlap_shifts(
    valley_k_spectrum,
    *,
    flux: MagneticFlux,
    shifts: Sequence[tuple[int, int]] | None,
    shell_ng: int | None,
) -> tuple[tuple[int, int], ...]:
    """Resolve explicit or shell-generated finite-B overlap shifts."""

    if shifts is None:
        if shell_ng is None:
            raise ValueError("Provide explicit shifts or shell_ng for overlap generation")
        return magnetic_shell_shifts(g1=valley_k_spectrum.params.g1, g2=valley_k_spectrum.params.g2, q=flux.q, shell_ng=shell_ng)
    normalized = tuple((int(m), int(n)) for m, n in shifts)
    if not normalized:
        raise ValueError("At least one overlap shift is required")
    return normalized

def build_finite_field_hf_inputs_from_spectra(
    valley_k_spectrum,
    valley_kprime_spectrum,
    *,
    nu: float,
    v0: float,
    shifts: Sequence[tuple[int, int]] | None = None,
    shell_ng: int | None = None,
    zeeman_unit: float = 0.0,
    precision: float = 1e-5,
    fast_overlap: bool = True,
    reduced_translation: bool = False,
) -> FiniteFieldHartreeFockInputBundle:
    """Assemble finite-B HF inputs from K/K' spectrum results.

    This is the unified no-I/O assembly path for both the full magnetic-BZ
    calculation and the reduced tL-symmetric/IKS calculation.  Set
    ``reduced_translation=True`` to return a
    :class:`FiniteFieldTLSymmetricHartreeFockInputs` bundle; otherwise a
    :class:`FiniteFieldHartreeFockInputs` bundle is returned.  Callers must
    provide either explicit ``shifts`` or an explicit ``shell_ng`` so expensive
    overlap generation is never accidental.
    """

    flux, nq = _validate_spectrum_pair(valley_k_spectrum, valley_kprime_spectrum)
    resolved_shifts = _resolve_overlap_shifts(valley_k_spectrum, flux=flux, shifts=shifts, shell_ng=shell_ng)
    state = build_finite_field_hf_state_from_spectra(
        valley_k_spectrum,
        valley_kprime_spectrum,
        nu=nu,
        v0=v0,
        zeeman_unit=zeeman_unit,
        precision=precision,
        reduced_translation=bool(reduced_translation),
    )
    overlap_data = build_full_flavor_overlap_data_from_spectra(
        valley_k_spectrum,
        valley_kprime_spectrum,
        shifts=resolved_shifts,
        fast=fast_overlap,
    )
    full_k_vectors = magnetic_k_vectors(g1=valley_k_spectrum.params.g1, g2=valley_k_spectrum.params.g2, flux=flux, nq=nq)
    normalization_count = magnetic_normalization_count(flux, nq)
    if reduced_translation:
        return FiniteFieldTLSymmetricHartreeFockInputs(
            state=state,
            overlap_data=overlap_data,
            full_k_vectors=full_k_vectors,
            normalization_count=normalization_count,
        )
    return FiniteFieldHartreeFockInputs(
        state=state,
        overlap_data=overlap_data,
        k_vectors=full_k_vectors,
        normalization_count=normalization_count,
    )

def build_finite_field_hf_inputs_from_parameters(
    params,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nu: float,
    v0: float,
    nq: int | None = None,
    shifts: Sequence[tuple[int, int]] | None = None,
    shell_ng: int | None = None,
    zeeman_unit: float = 0.0,
    precision: float = 1e-5,
    fast_overlap: bool = True,
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    mesh_shift: float = 0.0,
    kprime_q0: complex = 0.0 + 0.0j,
    reduced_translation: bool = False,
) -> FiniteFieldHartreeFockInputBundle:
    """Compute K/K' spectra and assemble finite-B HF inputs.

    This convenience helper is still no-I/O, but it can be numerically
    expensive because it diagonalizes both valleys and computes requested
    overlap blocks.  Callers must pass an explicit ``n_landau`` and either
    explicit ``shifts`` or ``shell_ng``.  Set ``reduced_translation=True`` for
    the tL-symmetric/IKS-reduced input bundle.
    """

    if int(n_landau) <= 0:
        raise ValueError(f"n_landau must be positive, got {n_landau}")
    if shifts is None and shell_ng is None:
        raise ValueError("Provide explicit shifts or shell_ng for overlap generation")
    nq_value = choose_magnetic_nq(flux.q) if nq is None else int(nq)
    if nq_value <= 0:
        raise ValueError(f"nq must be positive, got {nq_value}")
    from .spectrum import compute_magnetic_spectrum

    common = dict(
        params=params,
        flux=flux,
        n_landau=int(n_landau),
        nq=nq_value,
        sigma_rotation=sigma_rotation,
        hbn=hbn,
        include_strain=include_strain,
        mesh_shift=mesh_shift,
    )
    valley_k = compute_magnetic_spectrum(**common, valley="K")
    valley_kprime = compute_magnetic_spectrum(**common, valley="Kprime", q0=kprime_q0)
    return build_finite_field_hf_inputs_from_spectra(
        valley_k,
        valley_kprime,
        nu=nu,
        v0=v0,
        shifts=shifts,
        shell_ng=shell_ng,
        zeeman_unit=zeeman_unit,
        precision=precision,
        fast_overlap=fast_overlap,
        reduced_translation=reduced_translation,
    )

def build_tl_symmetric_finite_field_hf_inputs_from_spectra(
    valley_k_spectrum,
    valley_kprime_spectrum,
    *,
    nu: float,
    v0: float,
    shifts: Sequence[tuple[int, int]] | None = None,
    shell_ng: int | None = None,
    zeeman_unit: float = 0.0,
    precision: float = 1e-5,
    fast_overlap: bool = True,
) -> FiniteFieldTLSymmetricHartreeFockInputs:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF inputs.

    Prefer :func:`build_finite_field_hf_inputs_from_spectra` with
    ``reduced_translation=True`` for new code.
    """

    inputs = build_finite_field_hf_inputs_from_spectra(
        valley_k_spectrum,
        valley_kprime_spectrum,
        nu=nu,
        v0=v0,
        shifts=shifts,
        shell_ng=shell_ng,
        zeeman_unit=zeeman_unit,
        precision=precision,
        fast_overlap=fast_overlap,
        reduced_translation=True,
    )
    if not isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs):  # pragma: no cover - construction guards this.
        raise TypeError("Expected reduced tL-symmetric finite-field inputs")
    return inputs

def build_tl_symmetric_finite_field_hf_inputs_from_parameters(
    params,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nu: float,
    v0: float,
    nq: int | None = None,
    shifts: Sequence[tuple[int, int]] | None = None,
    shell_ng: int | None = None,
    zeeman_unit: float = 0.0,
    precision: float = 1e-5,
    fast_overlap: bool = True,
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    mesh_shift: float = 0.0,
    kprime_q0: complex = 0.0 + 0.0j,
) -> FiniteFieldTLSymmetricHartreeFockInputs:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF inputs.

    Prefer :func:`build_finite_field_hf_inputs_from_parameters` with
    ``reduced_translation=True`` for new code.
    """

    inputs = build_finite_field_hf_inputs_from_parameters(
        params,
        flux=flux,
        n_landau=n_landau,
        nq=nq,
        nu=nu,
        v0=v0,
        shifts=shifts,
        shell_ng=shell_ng,
        zeeman_unit=zeeman_unit,
        precision=precision,
        fast_overlap=fast_overlap,
        sigma_rotation=sigma_rotation,
        hbn=hbn,
        include_strain=include_strain,
        mesh_shift=mesh_shift,
        kprime_q0=kprime_q0,
        reduced_translation=True,
    )
    if not isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs):  # pragma: no cover - construction guards this.
        raise TypeError("Expected reduced tL-symmetric finite-field inputs")
    return inputs

def finite_field_occupied_state_count(nu: float, nt: int, nk: int) -> int:
    raw = (float(nu) + 4.0) / 8.0 * int(nt) * int(nk)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer occupied count {raw} for nt={nt}, nk={nk}")
    if rounded < 0 or rounded > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied count {rounded} outside [0,{int(nt) * int(nk)}]")
    return rounded


def finite_field_filling(density: Array) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    occupied = np.trace(density, axis1=0, axis2=1).real.sum() + 0.5 * nt * nk
    return float(8.0 * occupied / (nt * nk) - 4.0)


def density_update_from_hamiltonian(hamiltonian: Array, *, nu: float, sigma_z: Array | None = None) -> DensityUpdateResult:
    """Diagonalize ``H(k)`` and build the stored projector ``conj(U_occ) U_occ^T - I/2``."""

    h = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {h.shape}")
    vals = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros((nt, nt, nk), dtype=np.complex128)
    sigma_obs = np.zeros((nt, nk), dtype=float)
    for ik in range(nk):
        vals[:, ik], vecs[:, :, ik] = np.linalg.eigh(h[:, :, ik])
        if sigma_z is not None:
            sigma_block = np.asarray(sigma_z, dtype=np.complex128)[:, :, ik]
            for ib in range(nt):
                v = vecs[:, ib, ik]
                sigma_obs[ib, ik] = float(np.vdot(v, sigma_block @ v).real)

    n_occ = finite_field_occupied_state_count(nu, nt, nk)
    occupied = occupied_state_linear_indices(vals, n_occ)
    density = np.zeros_like(h)
    eye = np.eye(nt, dtype=np.complex128)
    bands = occupied % nt
    k_indices = occupied // nt
    for ik in range(nk):
        occ_bands = bands[k_indices == ik]
        if occ_bands.size:
            occ_vecs = vecs[:, occ_bands, ik]
            density[:, :, ik] = occ_vecs.conj() @ occ_vecs.T
        density[:, :, ik] -= 0.5 * eye
    mu = find_chemical_potential(vals, n_occ / float(nt * nk))
    return DensityUpdateResult(density=density, energies=vals, mu=mu, observables={"sigma_ztauz": sigma_obs})


def initialize_density_from_h0(state: FiniteFieldHartreeFockState, *, init_mode: str, seed: int = 0) -> None:
    """Initialize ``state.density`` using a subset of the original Julia modes."""

    mode = normalize_finite_field_init_mode(init_mode)
    rng = np.random.default_rng(seed)
    nt, nk = state.nt, state.nk
    n_occ = finite_field_occupied_state_count(state.nu, nt, nk)
    density = np.zeros_like(state.density)
    if mode == "bm":
        diag = np.diagonal(state.h0, axis1=0, axis2=1).T.real
        occupied = occupied_state_linear_indices(diag, n_occ)
    elif mode == "random":
        occupied = rng.permutation(nt * nk)[:n_occ]
    elif mode == "flavor":
        occupied = _flavor_polarized_indices(state, n_occ)
    elif mode == "bm_cascade":
        occupied = _bm_cascade_indices(state)
    elif mode == "sublattice":
        occupied = _sublattice_indices(state, n_occ)
    else:  # pragma: no cover - normalize_finite_field_init_mode guards this.
        raise ValueError(f"Unsupported init mode {init_mode!r}")
    eye = np.eye(nt, dtype=np.complex128)
    for linear in occupied:
        ib = int(linear % nt)
        ik = int(linear // nt)
        density[ib, ib, ik] = 1.0
    for ik in range(nk):
        density[:, :, ik] -= 0.5 * eye
    state.density[:, :, :] = density
    state.diagnostics["initial_filling"] = finite_field_filling(state.density)


def normalize_finite_field_init_mode(init_mode: str) -> InitMode:
    normalized = str(init_mode).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "bm": "bm",
        "random": "random",
        "flavor": "flavor",
        "flavor_u(4)": "flavor",
        "flavor_u4": "flavor",
        "bm_cascade": "bm_cascade",
        "cascade": "bm_cascade",
        "sublattice": "sublattice",
        "chern": "sublattice",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported finite-field init mode. Supported: bm, random, flavor, bm_cascade, sublattice")
    return aliases[normalized]  # type: ignore[return-value]


def _flavor_polarized_indices(state: FiniteFieldHartreeFockState, n_occ: int) -> Array:
    # Fill states flavor by flavor, lowest H0 energies within each chosen flavor.  This is a deterministic
    # Python analogue of the Julia educated flavor-polarized initializer.
    nt, nk = state.nt, state.nk
    diag = np.diagonal(state.h0, axis1=0, axis2=1).T.real
    sub = state.subbands_per_flavor
    flavor_order = [(1, 0), (0, 0), (1, 1), (0, 1)]  # (eta, spin), used by older TBG B0/VH runs too.
    candidates: list[int] = []
    for eta, spin in flavor_order:
        start = state_index(0, eta, spin, subbands_per_flavor=sub)
        flavor_states = [state_index(a, eta, spin, subbands_per_flavor=sub) for a in range(sub)]
        linear = np.asarray([ib + nt * ik for ik in range(nk) for ib in flavor_states], dtype=int)
        order = np.argsort(diag.reshape(-1, order="F")[linear], kind="stable")
        candidates.extend(linear[order].tolist())
        if len(candidates) >= n_occ:
            break
    _ = start  # keep the state_index call above explicit for convention readability.
    return np.asarray(candidates[:n_occ], dtype=int)


def _bm_cascade_indices(state: FiniteFieldHartreeFockState) -> Array:
    # Port of init_P_bm_cascade for the common (s,t) integer-fan seeds.
    pairs = [(0, 4), (0, -4), (1, 3), (-1, -3), (2, 2), (-2, -2), (3, 1), (-3, -1)]
    selected: tuple[int, int] | None = None
    for s, t in pairs:
        if abs(state.nu - (s + t * state.flux.ratio)) < 1e-3:
            selected = (s, t)
            break
    if selected is None:
        raise ValueError(f"nu={state.nu} is not on a supported (s,t) bm_cascade line")
    s, _t = selected
    sub = state.subbands_per_flavor
    nt, nk = state.nt, state.nk
    flavor_order = [2, 3, 0, 1]
    occupied: list[int] = []
    if s <= 1e-3:
        for flavor in flavor_order[: 4 - abs(s)]:
            eta = flavor % 2
            spin = flavor // 2
            for band in range(max(state.q - state.p, 0)):
                ib = state_index(band, eta, spin, subbands_per_flavor=sub)
                occupied.extend(ib + nt * ik for ik in range(nk))
    else:
        occupied.extend(range(nt * nk))
        remove: set[int] = set()
        for flavor in range(4 - abs(s)):
            eta = flavor % 2
            spin = flavor // 2
            for band in range(state.q + state.p, sub):
                ib = state_index(band, eta, spin, subbands_per_flavor=sub)
                remove.update(ib + nt * ik for ik in range(nk))
        occupied = [idx for idx in occupied if idx not in remove]
    return np.asarray(occupied, dtype=int)


def _sublattice_indices(state: FiniteFieldHartreeFockState, n_occ: int) -> Array:
    # A conservative deterministic Chern/sublattice seed: occupy the first n_occ states ordered by subband,
    # flavor, spin, and k.  The original Julia code has several variants; paper validation should choose
    # a physically motivated seed explicitly.
    return np.arange(state.nt * state.nk, dtype=int)[:n_occ]


def expand_valley_overlap_data_to_flavors(
    valley_k: MagneticOverlapData,
    valley_kprime: MagneticOverlapData,
    *,
    q: int,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> MagneticOverlapData:
    """Expand valley-resolved ``bmLL`` overlaps into the full HF flavor basis.

    Author ``MagneticFieldHF*.jl`` reads one metadata file per valley and copies
    each ``2q`` overlap block into both spin sectors, with no off-diagonal
    valley/spin density-overlap matrix elements.  This helper performs the same
    expansion for Python arrays.
    """

    if tuple(valley_k.shifts) != tuple(valley_kprime.shifts):
        raise ValueError("K and Kprime overlap shift tables must match")
    if np.asarray(valley_k.gvecs).shape != np.asarray(valley_kprime.gvecs).shape:
        raise ValueError("K and Kprime g-vector tables must have matching shape")
    n_sub = int(n_band) * int(q)
    nt = n_sub * int(n_eta) * int(n_spin)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    for shift in valley_k.shifts:
        block_k = np.asarray(valley_k.overlaps[shift], dtype=np.complex128)
        block_kp = np.asarray(valley_kprime.overlaps[shift], dtype=np.complex128)
        if block_k.shape != block_kp.shape:
            raise ValueError(f"K/Kprime overlap shape mismatch for {shift}: {block_k.shape} != {block_kp.shape}")
        if block_k.shape[0] != n_sub or block_k.shape[2] != n_sub:
            raise ValueError(f"Expected valley overlap flavor dimension {n_sub}, got {block_k.shape} for {shift}")
        nk_target = block_k.shape[1]
        nk_source = block_k.shape[3]
        full = np.zeros((nt, nk_target, nt, nk_source), dtype=np.complex128)
        for eta, valley_block in enumerate((block_k, block_kp)):
            for spin in range(int(n_spin)):
                indices = [state_index(band, eta, spin, subbands_per_flavor=n_sub, n_eta=n_eta) for band in range(n_sub)]
                full[np.ix_(indices, np.arange(nk_target), indices, np.arange(nk_source))] = valley_block
        overlaps[shift] = full
    return MagneticOverlapData(shifts=tuple(valley_k.shifts), gvecs=np.asarray(valley_k.gvecs, dtype=np.complex128).copy(), overlaps=overlaps)


def build_magnetic_interaction_hamiltonian(
    density: Array,
    overlap_data: MagneticOverlapData,
    *,
    k_vectors: Array,
    v0: float,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    relative_permittivity: float = 15.0,
    use_hartree: bool = True,
    use_fock: bool = True,
    use_numba: bool | None = None,
) -> Array:
    """Build full finite-B ``Hartree - Fock`` interaction Hamiltonian.

    This is the direct Python counterpart of ``add_HartreeFock`` in
    ``MagneticFieldHF.jl`` / ``MagneticFieldHF_IKS.jl``.  The finite-B
    normalization is ``1/hf.latt.nk = 1/(q*nq)^2``, not ``1/number_of_MBZ_k``.
    """

    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected density shape (nt,nt,nk), got {density.shape}")
    kvec = np.asarray(k_vectors, dtype=np.complex128)
    if kvec.shape != (nk,):
        raise ValueError(f"Expected k_vectors shape {(nk,)}, got {kvec.shape}")
    out = np.zeros_like(density)
    prefactor = float(beta) * float(v0) / float(normalization_count)
    for shift, gvec in zip(overlap_data.shifts, overlap_data.gvecs, strict=True):
        overlap = np.asarray(overlap_data.overlaps[shift], dtype=np.complex128)
        if overlap.shape != (nt, nk, nt, nk):
            raise ValueError(f"Expected overlap block shape {(nt, nk, nt, nk)}, got {overlap.shape} for shift {shift}")
        if use_hartree:
            diagonal = np.stack([overlap[:, ik, :, ik] for ik in range(nk)], axis=2).astype(np.complex128, copy=False)
            hartree_kernel = screened_coulomb_finite_b(gvec, screening_lm, relative_permittivity=relative_permittivity)
            if hartree_kernel != 0.0:
                tr_pg = compute_density_overlap_trace_from_diagonal(density, diagonal, use_numba=use_numba)
                out += prefactor * hartree_kernel * tr_pg * diagonal
        if use_fock:
            qvals = kvec.reshape(1, nk) - kvec.reshape(nk, 1) + gvec  # target ik rows, source ip cols.
            fock_kernel = np.vectorize(
                lambda z: screened_coulomb_finite_b(z, screening_lm, relative_permittivity=relative_permittivity),
                otypes=[float],
            )(qvals)
            out -= contract_fock_term_from_overlap(overlap, density, prefactor * fock_kernel, use_numba=use_numba)
    return out


def apply_iks_phase_to_transposed_density(
    transposed_density: Array,
    *,
    q: int,
    rp_position: int,
    phi: float,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> Array:
    """Apply Eq. S81/S78 IKS phase to ``transpose(P)`` for one tL2 orbit point."""

    mat = np.asarray(transposed_density, dtype=np.complex128).copy()
    if abs(float(phi)) < 1e-15:
        return mat
    n_sub = int(n_band) * int(q)
    expected = n_sub * int(n_eta) * int(n_spin)
    if mat.shape != (expected, expected):
        raise ValueError(f"Expected matrix shape {(expected, expected)}, got {mat.shape}")
    view = mat.reshape((n_sub, n_eta, n_spin, n_sub, n_eta, n_spin), order="F")
    phase = np.exp(1j * float(phi) * int(rp_position))
    view[:, 0, :, :, 1, :] *= np.conj(phase)
    view[:, 1, :, :, 0, :] *= phase
    return mat


def build_tl_symmetric_magnetic_interaction_hamiltonian(
    density: Array,
    overlap_data: MagneticOverlapData,
    *,
    full_k_vectors: Array,
    flux: MagneticFlux,
    nq: int,
    v0: float,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> Array:
    """Build the magnetic-translation-symmetric / IKS-reduced HF Hamiltonian.

    This ports ``MagneticFieldHF_tLSymmetric*_IKS*.jl``: the density is defined
    on ``nq**2`` reduced momenta, while Fock sums over the ``q``-point orbit
    generated by ``t_L2``.  Hartree terms survive only for ``n % q == 0`` and
    carry the author-code factor ``q``.
    """

    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk_reduced = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected density shape (nt,nt,nk), got {density.shape}")
    q = int(flux.q)
    nq = int(nq)
    if nk_reduced != nq * nq:
        raise ValueError(f"Expected reduced nk=nq**2={nq*nq}, got {nk_reduced}")
    indices = magnetic_orbit_indices(q, nq)
    full_k = np.asarray(full_k_vectors, dtype=np.complex128)
    if full_k.shape != (q * nk_reduced,):
        raise ValueError(f"Expected full_k_vectors shape {(q * nk_reduced,)}, got {full_k.shape}")
    rps = magnetic_r_orbit_positions(flux.p, flux.q)
    out = np.zeros_like(density)
    prefactor = float(beta) * float(v0) / float(normalization_count)

    for shift, gvec in zip(overlap_data.shifts, overlap_data.gvecs, strict=True):
        m, n = shift
        _ = m
        overlap = np.asarray(overlap_data.overlaps[shift], dtype=np.complex128)
        expected_shape = (nt, q * nk_reduced, nt, q * nk_reduced)
        if overlap.shape != expected_shape:
            raise ValueError(f"Expected tL overlap shape {expected_shape}, got {overlap.shape} for shift {shift}")
        kernel_g = screened_coulomb_finite_b(gvec, screening_lm, relative_permittivity=relative_permittivity)
        if n % q == 0 and kernel_g != 0.0:
            diag_full = indices[0, :]
            diagonal = np.stack([overlap[:, int(full_ik), :, int(full_ik)] for full_ik in diag_full], axis=2).astype(np.complex128, copy=False)
            tr_pg = compute_density_overlap_trace_from_diagonal(density, diagonal)
            out += prefactor * kernel_g * tr_pg * q * diagonal

        for ik in range(nk_reduced):
            full_target = indices[0, ik]
            tmp_fock = np.zeros((nt, nt), dtype=np.complex128)
            for ip in range(nk_reduced):
                for rp in range(q):
                    full_source = indices[rp, ip]
                    coeff = prefactor * screened_coulomb_finite_b(
                        full_k[full_source] - full_k[full_target] + gvec,
                        screening_lm,
                        relative_permittivity=relative_permittivity,
                    )
                    if coeff == 0.0:
                        continue
                    phased_density_t = apply_iks_phase_to_transposed_density(
                        density[:, :, ip].T,
                        q=q,
                        rp_position=int(rps[rp]),
                        phi=phi,
                        n_eta=n_eta,
                        n_spin=n_spin,
                        n_band=n_band,
                    )
                    lam = overlap[:, full_target, :, full_source]
                    tmp_fock += coeff * (lam @ phased_density_t @ lam.conj().T)
            out[:, :, ik] -= tmp_fock
    return out


def compute_finite_field_hf_energy(interaction_hamiltonian: Array, h0: Array, density: Array) -> float:
    """Author-code energy per moire unit cell, ``8/(nt*nk)`` times the trace."""

    interaction = np.asarray(interaction_hamiltonian, dtype=np.complex128)
    h0_arr = np.asarray(h0, dtype=np.complex128)
    p = np.asarray(density, dtype=np.complex128)
    total = np.einsum("abk,abk->", interaction, p, optimize=True) / 2.0
    total += np.einsum("abk,abk->", h0_arr, p, optimize=True)
    return float((8.0 * total.real) / (h0_arr.shape[0] * h0_arr.shape[2]))


def summarize_finite_field_hartree_fock(
    state: FiniteFieldHartreeFockState,
    run: HartreeFockRun | None = None,
) -> FiniteFieldHartreeFockSummary:
    """Return a compact no-I/O summary for checkpoint comparisons."""

    n_occ = finite_field_occupied_state_count(state.nu, state.nt, state.nk)
    flat_energies = np.sort(np.asarray(state.energies, dtype=float).reshape(-1))
    if 0 < n_occ < flat_energies.size:
        gap = float(flat_energies[n_occ] - flat_energies[n_occ - 1])
    else:
        gap = float("nan")
    energy = float(state.diagnostics.get("hf_energy", np.nan))
    final_raw_norm = float(state.diagnostics.get("final_raw_norm", np.nan))
    if run is None:
        iterations = int(round(float(state.diagnostics.get("iterations", 0.0))))
        converged = bool(final_raw_norm <= state.precision) if np.isfinite(final_raw_norm) else False
        exit_reason = "unknown"
    else:
        iterations = int(run.iterations)
        converged = bool(run.converged)
        exit_reason = str(run.exit_reason)
    return FiniteFieldHartreeFockSummary(
        filling=finite_field_filling(state.density),
        energy_per_muc=energy,
        mu=float(state.mu),
        single_particle_gap=gap,
        final_raw_norm=final_raw_norm,
        iterations=iterations,
        converged=converged,
        exit_reason=exit_reason,
    )

def calculate_valley_spin_order_parameters(
    hamiltonian: Array,
    energies: Array,
    mu: float,
    *,
    q: int,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> dict[str, float]:
    """Return ``s_i eta_j`` order parameters in the convention of the Julia code."""

    pauli = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    sub = int(n_band) * int(q)
    identity_band = np.eye(sub, dtype=np.complex128)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    eps = np.asarray(energies, dtype=float)
    out: dict[str, float] = {}
    nk = h.shape[2]
    for ispin, spin_mat in enumerate(pauli):
        for ieta, eta_mat in enumerate(pauli):
            op = np.kron(spin_mat, np.kron(eta_mat, identity_band))
            values = np.zeros_like(eps)
            for ik in range(nk):
                _vals, vecs = np.linalg.eigh(h[:, :, ik])
                values[:, ik] = np.diag(vecs.conj().T @ op @ vecs).real
            out[f"s{ispin}_eta{ieta}"] = float(values[eps <= float(mu)].sum() / eps.size * 8.0)
    return out


def build_finite_field_hf_kernel(
    state: FiniteFieldHartreeFockState,
    overlap_data: MagneticOverlapData,
    *,
    k_vectors: Array,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    interaction_builder = lambda density: build_magnetic_interaction_hamiltonian(
        density,
        overlap_data,
        k_vectors=k_vectors,
        v0=state.v0,
        normalization_count=normalization_count,
        screening_lm=screening_lm,
        beta=beta,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )

    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        result = density_update_from_hamiltonian(hamiltonian, nu=state.nu, sigma_z=state.sigma_z)
        sigma_obs = result.observables.get("sigma_ztauz")
        if isinstance(sigma_obs, np.ndarray) and sigma_obs.shape == state.sigma_ztauz.shape:
            state.sigma_ztauz[:, :] = sigma_obs
        return result

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        energy_functional=compute_finite_field_hf_energy,
        oda_delta_interaction_builder=interaction_builder,
        convergence_rule="mixed",
    )


def build_tl_symmetric_finite_field_hf_kernel(
    state: FiniteFieldHartreeFockState,
    overlap_data: MagneticOverlapData,
    *,
    full_k_vectors: Array,
    normalization_count: int,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
) -> HartreeFockKernel:
    interaction_builder = lambda density: build_tl_symmetric_magnetic_interaction_hamiltonian(
        density,
        overlap_data,
        full_k_vectors=full_k_vectors,
        flux=state.flux,
        nq=state.nq,
        v0=state.v0,
        normalization_count=normalization_count,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
    )

    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        result = density_update_from_hamiltonian(hamiltonian, nu=state.nu, sigma_z=state.sigma_z)
        sigma_obs = result.observables.get("sigma_ztauz")
        if isinstance(sigma_obs, np.ndarray) and sigma_obs.shape == state.sigma_ztauz.shape:
            state.sigma_ztauz[:, :] = sigma_obs
        return result

    return HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=density_builder,
        energy_functional=compute_finite_field_hf_energy,
        oda_delta_interaction_builder=interaction_builder,
        convergence_rule="mixed",
    )


def build_finite_field_hf_problem(
    kernel: HartreeFockKernel,
    *,
    initializer: Callable[[FiniteFieldHartreeFockState, str, int], None] | None = None,
) -> HartreeFockProblem:
    def default_initializer(state, *, init_mode: str, seed: int) -> None:
        if initializer is None:
            initialize_density_from_h0(state, init_mode=init_mode, seed=seed)
        else:
            initializer(state, init_mode, seed)

    return HartreeFockProblem(initializer=default_initializer, kernel=kernel)


def run_finite_field_hartree_fock(
    state: FiniteFieldHartreeFockState,
    kernel: HartreeFockKernel,
    *,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
) -> HartreeFockRun:
    problem = build_finite_field_hf_problem(kernel)
    return run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )

def build_finite_field_hf_kernel_from_inputs(
    inputs: FiniteFieldHartreeFockInputBundle,
    *,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockKernel:
    """Build an HF kernel from any assembled finite-B input bundle.

    Full magnetic-BZ bundles dispatch to :func:`build_finite_field_hf_kernel`.
    Reduced tL-symmetric/IKS bundles dispatch to
    :func:`build_tl_symmetric_finite_field_hf_kernel` and use ``phi`` for the
    IKS phase.  This keeps workflow code on one API while preserving the two
    physics contractions internally.
    """

    if isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs):
        return build_tl_symmetric_finite_field_hf_kernel(
            inputs.state,
            inputs.overlap_data,
            full_k_vectors=inputs.full_k_vectors,
            normalization_count=inputs.normalization_count,
            screening_lm=screening_lm,
            beta=beta,
            phi=phi,
            relative_permittivity=relative_permittivity,
        )
    if isinstance(inputs, FiniteFieldHartreeFockInputs):
        return build_finite_field_hf_kernel(
            inputs.state,
            inputs.overlap_data,
            k_vectors=inputs.k_vectors,
            normalization_count=inputs.normalization_count,
            screening_lm=screening_lm,
            beta=beta,
            relative_permittivity=relative_permittivity,
            use_numba=use_numba,
        )
    raise TypeError(f"Unsupported finite-field HF input bundle type: {type(inputs).__name__}")

def run_finite_field_hartree_fock_from_inputs(
    inputs: FiniteFieldHartreeFockInputBundle,
    *,
    screening_lm: float,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
    use_numba: bool | None = None,
) -> HartreeFockRun:
    """Run finite-B HF from a full or reduced no-I/O input bundle.

    The generic SCF/ODA loop still lives in :mod:`mean_field.core.hf`; this
    adapter only builds the correct finite-B TBG kernel for the provided bundle.
    """

    kernel = build_finite_field_hf_kernel_from_inputs(
        inputs,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
        use_numba=use_numba,
    )
    return run_finite_field_hartree_fock(
        inputs.state,
        kernel,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )

def build_tl_symmetric_finite_field_hf_kernel_from_inputs(
    inputs: FiniteFieldTLSymmetricHartreeFockInputs,
    *,
    screening_lm: float,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
) -> HartreeFockKernel:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF kernels."""

    return build_finite_field_hf_kernel_from_inputs(
        inputs,
        screening_lm=screening_lm,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
    )

def run_tl_symmetric_finite_field_hartree_fock_from_inputs(
    inputs: FiniteFieldTLSymmetricHartreeFockInputs,
    *,
    screening_lm: float,
    init_mode: str,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    beta: float = 1.0,
    phi: float = 0.0,
    relative_permittivity: float = 15.0,
) -> HartreeFockRun:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF runs."""

    return run_finite_field_hartree_fock_from_inputs(
        inputs,
        screening_lm=screening_lm,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        beta=beta,
        phi=phi,
        relative_permittivity=relative_permittivity,
    )

__all__ = [
    "FiniteFieldHartreeFockInputBundle",
    "FiniteFieldHartreeFockInputs",
    "FiniteFieldHartreeFockState",
    "FiniteFieldHartreeFockSummary",
    "FiniteFieldTLSymmetricHartreeFockInputs",
    "InitMode",
    "MagneticFlux",
    "MagneticOverlapData",
    "apply_iks_phase_to_transposed_density",
    "build_finite_field_hf_inputs_from_parameters",
    "build_finite_field_hf_inputs_from_spectra",
    "build_finite_field_hf_kernel",
    "build_finite_field_hf_kernel_from_inputs",
    "build_finite_field_hf_problem",
    "build_finite_field_hf_state_from_spectra",
    "build_full_flavor_overlap_data_from_spectra",
    "build_tl_symmetric_finite_field_hf_inputs_from_parameters",
    "build_tl_symmetric_finite_field_hf_inputs_from_spectra",
    "build_h0_from_hofstadter_metadata",
    "build_magnetic_interaction_hamiltonian",
    "build_tl_symmetric_finite_field_hf_kernel",
    "build_tl_symmetric_finite_field_hf_kernel_from_inputs",
    "build_tl_symmetric_magnetic_interaction_hamiltonian",
    "calculate_valley_spin_order_parameters",
    "choose_magnetic_nq",
    "compute_finite_field_hf_energy",
    "coulomb_unit_from_lattice",
    "density_update_from_hamiltonian",
    "expand_valley_overlap_data_to_flavors",
    "finite_field_diophantine_filling",
    "finite_field_filling",
    "finite_field_occupied_state_count",
    "in_hex_shell",
    "initialize_density_from_h0",
    "magnetic_k_vectors",
    "magnetic_normalization_count",
    "magnetic_orbit_indices",
    "magnetic_r_orbit_positions",
    "magnetic_reciprocal_vector",
    "magnetic_shell_shifts",
    "normalize_finite_field_init_mode",
    "paper_fig6_branch_cases",
    "paper_fig6_finite_b_fluxes",
    "run_finite_field_hartree_fock",
    "run_finite_field_hartree_fock_from_inputs",
    "run_tl_symmetric_finite_field_hartree_fock_from_inputs",
    "screened_coulomb_finite_b",
    "state_index",
    "summarize_finite_field_hartree_fock",
    "zeeman_unit_from_area",
]
