"""Compatibility facade for finite-magnetic-field BM/Hofstadter spectrum helpers."""

from __future__ import annotations

from ._spectrum_shared import MagneticFlux, Valley
from ._spectrum_params import FiniteFieldBMParameters, MagneticSpectrumResult
from ._spectrum_sweep import (
    MagneticSpectrumSweepCase,
    MagneticSpectrumSweepResult,
    author_landau_cutoff,
    compute_magnetic_spectrum_sweep,
    paper_hofstadter_fluxes,
    red_chern_minus_one_group_mask,
)
from ._spectrum_ll import (
    associated_laguerre_element,
    associated_laguerre_matrix,
    in_gamma,
    projector_norm,
    projector_para,
    tll_matrix,
)
from ._spectrum_hamiltonian import (
    _hermitian_from_upper,
    compute_magnetic_spectrum,
    construct_ll_hamiltonian,
    construct_sigma_z_ll,
    generate_magnetic_translation_orbit,
    magnetic_lattice_coordinates,
    qjs_for_valley,
)
from ._spectrum_overlap import compute_coulomb_overlap, compute_coulomb_overlap_fast

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

for _name in __all__:
    _obj = globals()[_name]
    if getattr(_obj, "__module__", "").startswith("mean_field.systems.tbg.finite_field._spectrum_"):
        _obj.__module__ = __name__
del _name, _obj
