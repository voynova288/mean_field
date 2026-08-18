"""InAs/GaSb adapters for the generic excitonic matrix-HF API.

The common reference-subtracted solver lives in :mod:`mean_field.core.hf`;
this package owns Kane/E1/H1 basis conventions and microscopic interactions.
"""

from .angular_fock import (
    CoRotatingHarmonicFockOperator,
    PolarHarmonicProjectedFockOperator,
    direct_full_2d_co_rotating_fock_mode_contributions,
    direct_full_2d_co_rotating_fock_modes,
)
from .axial import (
    AxialRotationSpec,
    axial_eh_seed_hamiltonian,
    axial_radial_time_reversal_error,
    axial_radial_time_reversal_residuals,
    axial_radial_time_reversal_unitary,
    axial_time_reversal_residuals,
    expand_axial_radial_bundle,
    expand_axial_radial_matrices,
    matrix_time_reversal_error,
    project_axial_covariant_matrices,
    project_axial_radial_time_reversal_matrices,
    project_axial_time_reversal_matrices,
    project_time_reversal_matrices,
)
from .axial_fock import (
    AxialAveragedProjectedFockOperator,
    AxialE1H1CoherenceFockSuperoperator,
    AxialProjectedFockOperator,
    co_rotating_mode_alias,
    precompute_axial_harmonic_exchange_tensor_mev_nm2,
    precompute_axial_harmonic_exchange_tensors_mev_nm2,
    precompute_axial_reduced_exchange_tensor_mev_nm2,
)
from .carriers import (
    DensityCalibratedDetuning,
    KaneCarrierProjectors,
    NeutralDensityResult,
    apply_relative_detuning,
    calibrate_relative_detuning_to_density,
    calibrate_relative_detuning_to_pocket_density,
    charge_neutral_fermi_density,
    energy_sorted_pocket_density,
)
from .conventions import (
    E1H1BasisSpec,
    KANE8_JZ,
    active_electron_hole_areal_densities,
    block_unitary,
    kane8_time_reversal_unitary,
    ordinary_electron_pair_channels_mev,
    spinful_time_reversal_errors,
)
from .hartree import (
    ProjectedHartreeFockOperator,
    ProjectedHartreeOperator,
    ReferenceSubtractedChargeDensity,
    diagonal_local_density_vertices,
    periodic_poisson_electron_energy_mev,
    reference_subtracted_charge_density,
    wafer_b_dielectric_profile,
)
from .matrix_ei import (
    MatrixEIConfig,
    MatrixEIResult,
    MatrixEIState,
    fermionic_entropy_density_mev_per_K_nm2,
    fixed_mu_fermi_density,
    relative_hf_energy_density_mev_nm2,
    relative_internal_energy_components_mev_nm2,
    solve_reference_subtracted_matrix_ei,
    weighted_fermi_density,
)
from .projected_model import (
    Kane4Bundle,
    LiftedKaneFrame,
    ProjectedFockOperator,
    apply_e1h1_coherence_exchange_superoperator,
    density_vertex_reciprocity_error,
    e1h1_coherence_exchange_tensors_mev_nm2,
    excitonic_fock_singular_values,
    fock_energy_density_mev_nm2,
    gauge_transform_k_matrices,
    gauge_transform_local_vertices,
    hermitian_density_from_e1h1_coherence,
    integrated_density_vertices,
    lift_active_frame_to_reference,
    local_density_vertices,
    load_kane4_bundle,
    projected_fock_self_energy,
    remove_normal_e1_h1_hybridization,
    save_kane4_bundle,
    uniform_dielectric_green_mev_nm2,
    uniform_dielectric_green_on_mesh_mev_nm2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
