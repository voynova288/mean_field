"""Common injection-current / CPGE helpers.

This package reuses the Hamiltonian-gauge ingredients from
:mod:`analysis.shift_current` and adds only the injection-current kernel
``Delta v * r * r``.  Paper-specific scans and plotting should live in
workflow scripts or system adapters, not here.
"""

from .core import (
    InjectionCurrentTensors,
    accumulate_injection_spectrum,
    component_label,
    cpge_part,
    injection_component_kernel,
    injection_component_kernel_from_gauge_pair,
    injection_spectra_from_transition_table,
    injection_transition_weight,
    linear_injection_part,
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)

__all__ = [
    "InjectionCurrentTensors",
    "accumulate_injection_spectrum",
    "component_label",
    "cpge_part",
    "injection_component_kernel",
    "injection_component_kernel_from_gauge_pair",
    "injection_spectra_from_transition_table",
    "injection_transition_weight",
    "linear_injection_part",
    "positive_injection_transition_terms",
    "precompute_injection_current_tensors",
]
