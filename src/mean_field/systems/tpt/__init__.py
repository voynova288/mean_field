"""Ta--Pd--Te (TPT) material adapters."""

from .longwave_form_factor_hf import (
    apply_periodic_wannier_fock_fft,
    build_fixed_rank_density_builder,
    build_periodic_scalar_2d_coulomb_kernel,
    direct_periodic_wannier_fock_elements,
    reconstruct_complete_wannier_h0_and_reference,
    rectangular_coulomb_self_cell_average,
    reference_relative_energy,
)
from .xue_style_hf import (
    TPTGammaPatch,
    TPTXueLikeHFResult,
    TPTXueLikeHFState,
    TPTXueLikeParameters,
    build_tpt_xue_like_hf_state,
    load_tpt_full_bz_energy_rank_diagnostic,
    load_tpt_gamma_patch,
    run_tpt_xue_like_hf,
)

__all__ = [
    "apply_periodic_wannier_fock_fft",
    "build_fixed_rank_density_builder",
    "build_periodic_scalar_2d_coulomb_kernel",
    "direct_periodic_wannier_fock_elements",
    "reconstruct_complete_wannier_h0_and_reference",
    "rectangular_coulomb_self_cell_average",
    "reference_relative_energy",
    "TPTGammaPatch",
    "TPTXueLikeHFResult",
    "TPTXueLikeHFState",
    "TPTXueLikeParameters",
    "build_tpt_xue_like_hf_state",
    "load_tpt_full_bz_energy_rank_diagnostic",
    "load_tpt_gamma_patch",
    "run_tpt_xue_like_hf",
]
