from __future__ import annotations

import numpy as np

from mean_field.core.hf import HartreeFockProblem
from mean_field.systems.RnG_hBN import (
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    build_rlg_hbn_hf_problem,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    rlg_hbn_filling_from_density,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
)


def test_rlg_hbn_hf_problem_builder_uses_common_problem_api() -> None:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=1)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=((0, 0),))
    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=1.0,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=1.0,
        occupation_counts=counts,
    )

    problem = build_rlg_hbn_hf_problem(state, blocks)
    problem.initializer(state, init_mode="flavor", seed=1)

    assert isinstance(problem, HartreeFockProblem)
    assert state.density.shape == basis_data.h0.shape
    assert np.isclose(
        rlg_hbn_filling_from_density(
            state.density,
            state.reference_density,
            active_valence_bands=state.active_valence_bands,
        ),
        1.0,
    )
