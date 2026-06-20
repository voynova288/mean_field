from __future__ import annotations

import numpy as np

from mean_field.systems.RnG_hBN import RLGhBNInteractionParams, RLGhBNModel, build_rlg_hbn_projected_basis
from mean_field.systems.RnG_hBN.hf import (
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_layer_overlap_blocks_between,
)


def _small_model() -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )


def _small_interaction() -> RLGhBNInteractionParams:
    return RLGhBNInteractionParams(
        active_valence_bands=1,
        active_conduction_bands=1,
        k_mesh_size=1,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )


def test_rlg_hbn_self_overlap_blocks_match_between_builder() -> None:
    basis_data = build_rlg_hbn_projected_basis(_small_model(), _small_interaction(), mesh_size=1)
    shifts = ((0, 0), (1, 0), (-1, 0))

    self_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=shifts)
    between_blocks = build_rlg_hbn_layer_overlap_blocks_between(basis_data, basis_data, shifts=shifts)

    assert self_blocks.shifts == between_blocks.shifts == shifts
    np.testing.assert_allclose(self_blocks.gvecs, between_blocks.gvecs, rtol=0.0, atol=0.0)

    for shift in shifts:
        np.testing.assert_allclose(
            self_blocks.layer_overlaps[shift],
            between_blocks.layer_overlaps[shift],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            self_blocks.layer_diagonal_overlaps[shift],
            between_blocks.layer_diagonal_overlaps[shift],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            self_blocks.hartree_layer_coulomb[shift],
            between_blocks.hartree_layer_coulomb[shift],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            self_blocks.fock_layer_coulomb[shift],
            between_blocks.fock_layer_coulomb[shift],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
