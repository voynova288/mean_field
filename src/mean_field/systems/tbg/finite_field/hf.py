"""TBG adapter for the generic finite-magnetic-field Hartree-Fock core.

The actual finite-B HF state, initialization, interaction contractions, kernels,
SCF problem assembly, and summaries live in :mod:`mean_field.core.hf.finite_field`.
This module is intentionally thin: it converts TBG ``MagneticSpectrumResult``
objects or TBG finite-field parameters into the generic finite-B HF input
bundles and re-exports the stable historical API.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from ....core.magnetic_field import (
    MagneticFlux,
    choose_magnetic_nq,
    diophantine_branch_cases,
    in_hex_shell,
    magnetic_k_vectors,
    magnetic_normalization_count,
    magnetic_orbit_indices,
    magnetic_r_orbit_positions,
    magnetic_reciprocal_vector,
    magnetic_shell_shifts,
)
from ....core.hf.finite_field import (
    FiniteFieldHartreeFockInputBundle,
    FiniteFieldHartreeFockInputs,
    FiniteFieldHartreeFockState,
    FiniteFieldHartreeFockSummary,
    FiniteFieldTLSymmetricHartreeFockInputs,
    InitMode,
    MagneticOverlapData,
    apply_iks_phase_to_transposed_density,
    build_finite_field_hf_kernel,
    build_finite_field_hf_kernel_from_inputs,
    build_finite_field_hf_problem,
    build_h0_from_hofstadter_metadata,
    build_magnetic_interaction_hamiltonian,
    build_tl_symmetric_finite_field_hf_kernel,
    build_tl_symmetric_finite_field_hf_kernel_from_inputs,
    build_tl_symmetric_magnetic_interaction_hamiltonian,
    calculate_valley_spin_order_parameters,
    compute_finite_field_hf_energy,
    coulomb_unit_from_lattice,
    density_update_from_hamiltonian,
    expand_valley_overlap_data_to_flavors,
    finite_field_diophantine_filling,
    finite_field_filling,
    finite_field_occupied_state_count,
    initialize_density_from_h0,
    normalize_finite_field_init_mode,
    run_finite_field_hartree_fock,
    run_finite_field_hartree_fock_from_inputs,
    run_tl_symmetric_finite_field_hartree_fock_from_inputs,
    screened_coulomb_finite_b,
    state_index,
    summarize_finite_field_hartree_fock,
    zeeman_unit_from_area,
)


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

    return diophantine_branch_cases(s, t, fluxes=paper_fig6_finite_b_fluxes() if fluxes is None else fluxes)


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
    """Build a generic finite-B HF state from TBG K/K' spectrum results."""

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
    """Compute TBG K/K' overlaps and expand them into the generic HF flavor basis."""

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
    """Assemble generic finite-B HF inputs from TBG K/K' spectrum results."""

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
    kprime_mesh_shift: float | None = None,
    kprime_q0: complex = 0.0 + 0.0j,
    reduced_translation: bool = False,
) -> FiniteFieldHartreeFockInputBundle:
    """Compute TBG K/K' spectra, then assemble generic finite-B HF inputs."""

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
    kprime_common = dict(common)
    kprime_common["mesh_shift"] = mesh_shift if kprime_mesh_shift is None else float(kprime_mesh_shift)
    valley_kprime = compute_magnetic_spectrum(**kprime_common, valley="Kprime", q0=kprime_q0)
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
    """Assemble reduced tL-symmetric generic finite-B HF inputs from TBG spectra."""

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
    kprime_mesh_shift: float | None = None,
    kprime_q0: complex = 0.0 + 0.0j,
) -> FiniteFieldTLSymmetricHartreeFockInputs:
    """Compatibility wrapper for reduced tL-symmetric/IKS HF inputs."""

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
        kprime_mesh_shift=kprime_mesh_shift,
        kprime_q0=kprime_q0,
        reduced_translation=True,
    )
    if not isinstance(inputs, FiniteFieldTLSymmetricHartreeFockInputs):  # pragma: no cover - construction guards this.
        raise TypeError("Expected reduced tL-symmetric finite-field inputs")
    return inputs


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
