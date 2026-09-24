"""Narrow typed API for finite-field TBG spectra and Hartree-Fock."""

from ....core.hf.finite_field import (
    FiniteFieldHartreeFockInputBundle,
    FiniteFieldHartreeFockInputs,
    FiniteFieldHartreeFockState,
    FiniteFieldHartreeFockSummary,
    FiniteFieldTLSymmetricHartreeFockInputs,
    MagneticOverlapData,
    run_finite_field_hartree_fock_from_inputs,
    summarize_finite_field_hartree_fock,
)
from ....core.magnetic_field import MagneticFlux
from .hf import (
    build_finite_field_hf_inputs_from_parameters,
    build_finite_field_hf_inputs_from_spectra,
    build_finite_field_hf_state_from_spectra,
    build_full_flavor_overlap_data_from_spectra,
)
from .spectrum import (
    FiniteFieldBMParameters,
    MagneticSpectrumResult,
    MagneticSpectrumSweepCase,
    MagneticSpectrumSweepResult,
    compute_magnetic_spectrum,
    compute_magnetic_spectrum_sweep,
)

__all__ = [
    "FiniteFieldBMParameters",
    "MagneticFlux",
    "MagneticSpectrumResult",
    "MagneticSpectrumSweepCase",
    "MagneticSpectrumSweepResult",
    "FiniteFieldHartreeFockInputBundle",
    "FiniteFieldHartreeFockInputs",
    "FiniteFieldTLSymmetricHartreeFockInputs",
    "FiniteFieldHartreeFockState",
    "FiniteFieldHartreeFockSummary",
    "MagneticOverlapData",
    "compute_magnetic_spectrum",
    "compute_magnetic_spectrum_sweep",
    "build_finite_field_hf_state_from_spectra",
    "build_full_flavor_overlap_data_from_spectra",
    "build_finite_field_hf_inputs_from_spectra",
    "build_finite_field_hf_inputs_from_parameters",
    "run_finite_field_hartree_fock_from_inputs",
    "summarize_finite_field_hartree_fock",
]
