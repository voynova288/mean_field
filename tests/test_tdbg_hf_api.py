from __future__ import annotations

from mean_field.core.hf.problem import HartreeFockProblem
from mean_field.systems.tdbg import (
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
)
from mean_field.systems.tdbg.projected_hf_data import build_tdbg_projected_hf_data
from mean_field.systems.tdbg.projected_hf_run import build_tdbg_projected_hf_problem


def test_tdbg_projected_hf_owners_build_common_hf_problem() -> None:
    data = build_tdbg_projected_hf_data(
        TDBGProjectedHFConfig(
            theta_deg=1.38,
            cut=1.0,
            mesh_size=1,
            paper_ud_ev=0.09,
            paper_ud_convention="minus_xi_ud_over3",
            window=TDBGProjectedWindow("two_flat"),
            filling=2,
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        )
    )

    problem = build_tdbg_projected_hf_problem(data)

    assert isinstance(problem, HartreeFockProblem)
