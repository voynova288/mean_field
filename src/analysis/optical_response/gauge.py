from __future__ import annotations

"""Gauge-safe optical-response derivative public facade.

Implementations are split across ``analysis.optical_response.gauge_*`` modules.
This module preserves the historical import surface used by
``analysis.response_derivative_gauge`` and by system adapters.
"""

from .gauge_data import *  # noqa: F401,F403
from .gauge_primitives import *  # noqa: F401,F403
from .gauge_hamiltonian import *  # noqa: F401,F403
from .gauge_derivatives import *  # noqa: F401,F403
from .gauge_shift import *  # noqa: F401,F403

__all__ = [
    "HamiltonianGaugeData",
    "GeneralizedDerivativeData",
    "PairGeneralizedDerivativeData",
    "energy_difference_inverse",
    "degenerate_band_groups",
    "group_indices",
    "random_block_unitary",
    "apply_band_gauge_to_matrix",
    "apply_band_gauge_to_axis_matrix",
    "trace_subspace",
    "matrix_in_eigenbasis",
    "hamiltonian_gauge_data",
    "covariant_derivative_matrix",
    "wannierberri_matrix_gen_derivative_ln",
    "wannierberri_matrix_gen_derivative_nn",
    "berry_connection_generalized_derivative",
    "berry_connection_pair",
    "berry_connection_generalized_derivative_pair",
    "wannierberri_shift_current_internal_imn",
    "wannierberri_shift_current_group_trace",
    "shift_integrand_from_pair_generalized_derivative",
    "shift_vector_from_pair_generalized_derivative",
    "shift_integrand_from_generalized_derivative",
    "shift_vector_from_generalized_derivative",
    "normalized_u1_link",
    "link_shift_vector",
]
