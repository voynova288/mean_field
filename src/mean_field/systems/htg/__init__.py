"""Helical trilayer graphene public facade."""

from .model import HTGModel
from .params import (
    HTGParams,
    InteractionParams,
    KWAN_2023_FERMI_VELOCITY_M_PER_S,
    KWAN_2023_TUNNELING_EV,
    theta_deg_from_alpha,
    velocity_m_per_s_to_ev_nm,
)
from .mean_field_adapter import (
    HTGRunHFConfig,
    run_htg_hf_config_adapter,
)
from .supercell_contracts import (
    HTGSupercellRunHFConfig,
    htg_supercell_hf_run_to_hf_result,
    htg_supercell_hf_run_to_hf_run_result,
    run_htg_supercell_hf_config_adapter,
)

__all__ = [
    "HTGModel",
    "HTGParams",
    "HTGRunHFConfig",
    "HTGSupercellRunHFConfig",
    "InteractionParams",
    "KWAN_2023_FERMI_VELOCITY_M_PER_S",
    "KWAN_2023_TUNNELING_EV",
    "htg_supercell_hf_run_to_hf_result",
    "htg_supercell_hf_run_to_hf_run_result",
    "run_htg_hf_config_adapter",
    "run_htg_supercell_hf_config_adapter",
    "theta_deg_from_alpha",
    "velocity_m_per_s_to_ev_nm",
]
