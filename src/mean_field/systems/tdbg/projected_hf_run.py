from __future__ import annotations

import numpy as np

from ...core.hf import HartreeFockKernel, HartreeFockProblem, run_hartree_fock_problem
from .projected_hf_config import TDBGProjectedHFConfig
from .projected_hf_interactions import (
    TDBGProjectedHFInteractionBuilder,
    build_tdbg_interaction_builder,
    tdbg_energy_components,
)
from .projected_hf_state import (
    TDBGProjectedHFData,
    TDBGProjectedHFDensityBuilder,
    TDBGProjectedHFInitializer,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    tdbg_order_parameters,
)

def build_tdbg_projected_hf_state(data: TDBGProjectedHFData) -> TDBGProjectedHFState:
    return TDBGProjectedHFState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.zeros_like(data.h0),
        hamiltonian=np.asarray(data.h0, dtype=np.complex128).copy(),
        energies=np.zeros((data.nt, data.nk), dtype=float),
        precision=float(data.config.precision),
    )


def build_tdbg_projected_hf_kernel(
    data: TDBGProjectedHFData,
    *,
    interaction_builder: TDBGProjectedHFInteractionBuilder | None = None,
) -> HartreeFockKernel:
    resolved_interaction_builder = build_tdbg_interaction_builder(data) if interaction_builder is None else interaction_builder
    density_builder = TDBGProjectedHFDensityBuilder(data)

    def energy_functional(_interaction_h: np.ndarray, _h0: np.ndarray, density: np.ndarray) -> float:
        return tdbg_energy_components(
            data,
            density,
            interaction_components=resolved_interaction_builder.components(density),
        )["total_ev"]

    if data.config.mix_fallback is None:
        oda_parameterizer = None
    else:
        fixed_mix = float(data.config.mix_fallback)
        oda_parameterizer = lambda _state, _delta_density: fixed_mix

    return HartreeFockKernel(
        interaction_builder=resolved_interaction_builder,
        density_builder=density_builder,
        energy_functional=energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=None,
        convergence_rule="raw",
    )


def build_tdbg_projected_hf_problem(
    data: TDBGProjectedHFData,
    *,
    interaction_builder: TDBGProjectedHFInteractionBuilder | None = None,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=TDBGProjectedHFInitializer(data),
        kernel=build_tdbg_projected_hf_kernel(data, interaction_builder=interaction_builder),
    )


def run_tdbg_projected_hf(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> TDBGProjectedHFResult:
    state = build_tdbg_projected_hf_state(data)
    interaction_builder = build_tdbg_interaction_builder(data)
    problem = build_tdbg_projected_hf_problem(data, interaction_builder=interaction_builder)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=int(data.config.max_iter),
        oda_stall_threshold=0.0 if data.config.mix_fallback is not None else 1.0e-3,
    )
    hamiltonian_components = interaction_builder.components(run.state.density)
    components = tdbg_energy_components(
        data,
        run.state.density,
        interaction_components=hamiltonian_components,
    )
    order = tdbg_order_parameters(data, run.state.density)
    run.state.diagnostics.update({k: float(v) for k, v in components.items()})
    return TDBGProjectedHFResult(
        run=run,
        data=data,
        init_mode=init_mode,
        seed=int(seed),
        order_parameters=order,
        energy_components=components,
        hamiltonian_components={key: np.asarray(value, dtype=np.complex128).copy() for key, value in hamiltonian_components.items()},
    )

__all__ = [
    "build_tdbg_projected_hf_kernel",
    "build_tdbg_projected_hf_problem",
    "build_tdbg_projected_hf_state",
    "run_tdbg_projected_hf",
]
