"""One-dimensional Hubbard-model benchmark adapters."""

from .tdhf_alavirad_sau import (
    AlaviradSauHubbard1DModel,
    AlaviradSauHubbardTDHFProvider,
    AlaviradSauSpinStiffness,
    apply_saturated_ferromagnet_interspin_tdhf_action,
    build_exact_one_magnon_hamiltonian,
    build_exact_one_magnon_hamiltonian_bitstring,
    fit_exact_one_magnon_spin_stiffness,
    saturated_ferromagnet_fock_matrix,
    saturated_ferromagnet_projector,
    saturated_ferromagnet_stationarity_residual,
    spin_lowering_nambu_generator,
)

__all__ = [
    "AlaviradSauHubbard1DModel",
    "AlaviradSauHubbardTDHFProvider",
    "AlaviradSauSpinStiffness",
    "apply_saturated_ferromagnet_interspin_tdhf_action",
    "build_exact_one_magnon_hamiltonian",
    "build_exact_one_magnon_hamiltonian_bitstring",
    "fit_exact_one_magnon_spin_stiffness",
    "saturated_ferromagnet_fock_matrix",
    "saturated_ferromagnet_projector",
    "saturated_ferromagnet_stationarity_residual",
    "spin_lowering_nambu_generator",
]
