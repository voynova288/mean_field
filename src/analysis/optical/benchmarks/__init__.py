"""Lightweight optical-response benchmark helpers.

Benchmark helpers capture formula-level checks and symmetry constraints from
reference papers.  They must not launch paper-scale grid integrations or hide
system-specific conventions.
"""

from .liu_dai_2020 import (
    C3TensorDecomposition,
    c3_inplane_second_order_tensor,
    decompose_c3_inplane_second_order_tensor,
    qah_faraday_kerr_benchmark,
    qah_faraday_kerr_small_benchmark,
    qah_sheet_conductivity_tensor,
)
from .mikhailov_2016 import (
    IntegralMethod,
    Mikhailov2016THGResult,
    mikhailov2016_thg_xxxx_scattering,
)
from .okada_2016 import (
    OKADA_2016_INP_REFRACTIVE_INDEX,
    OKADA_2016_DC_ESTIMATED_FARADAY_RAD,
    OKADA_2016_DC_ESTIMATED_KERR_RAD,
    OKADA_2016_MEASURED_FARADAY_RAD,
    OKADA_2016_MEASURED_KERR_RAD,
    Okada2016QAHBenchmark,
    Okada2016ScalingUncertainty,
    okada2016_qah_benchmark,
    okada2016_qah_sheet_conductivity,
    okada2016_scaling_from_dc_conductivity,
    okada2016_scaling_function,
    okada2016_scaling_uncertainty,
)
from .passos_2018 import (
    Passos2018GrapheneParameters,
    RiceMeleParameters,
    graphene_direct_lattice_vectors,
    graphene_reciprocal_vectors,
    iter_passos2018_graphene_kpoints,
    iter_rice_mele_kpoints,
    passos2018_graphene_grid,
    passos2018_graphene_refined_grid,
    passos2018_graphene_hamiltonian_and_derivatives,
    passos2018_sigma0_si,
    rice_mele_grid,
    rice_mele_hamiltonian_and_derivatives,
)
from .qwz import (
    qwz_hamiltonian_and_derivatives,
    qwz_lower_band_chern_dvector,
    qwz_lower_band_chern_number,
    qwz_optical_kpoint_data,
)

__all__ = [
    "C3TensorDecomposition",
    "c3_inplane_second_order_tensor",
    "decompose_c3_inplane_second_order_tensor",
    "qah_faraday_kerr_benchmark",
    "qah_faraday_kerr_small_benchmark",
    "qah_sheet_conductivity_tensor",
    "IntegralMethod",
    "Mikhailov2016THGResult",
    "mikhailov2016_thg_xxxx_scattering",
    "OKADA_2016_INP_REFRACTIVE_INDEX",
    "OKADA_2016_DC_ESTIMATED_FARADAY_RAD",
    "OKADA_2016_DC_ESTIMATED_KERR_RAD",
    "OKADA_2016_MEASURED_FARADAY_RAD",
    "OKADA_2016_MEASURED_KERR_RAD",
    "Okada2016QAHBenchmark",
    "Okada2016ScalingUncertainty",
    "okada2016_qah_benchmark",
    "okada2016_qah_sheet_conductivity",
    "okada2016_scaling_from_dc_conductivity",
    "okada2016_scaling_function",
    "okada2016_scaling_uncertainty",
    "Passos2018GrapheneParameters",
    "RiceMeleParameters",
    "graphene_direct_lattice_vectors",
    "graphene_reciprocal_vectors",
    "iter_passos2018_graphene_kpoints",
    "iter_rice_mele_kpoints",
    "passos2018_graphene_grid",
    "passos2018_graphene_refined_grid",
    "passos2018_graphene_hamiltonian_and_derivatives",
    "passos2018_sigma0_si",
    "rice_mele_grid",
    "rice_mele_hamiltonian_and_derivatives",
    "qwz_hamiltonian_and_derivatives",
    "qwz_lower_band_chern_dvector",
    "qwz_lower_band_chern_number",
    "qwz_optical_kpoint_data",
]
