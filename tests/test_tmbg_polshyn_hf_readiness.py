from __future__ import annotations

import numpy as np

from mean_field.systems.tmbg.polshyn_supercell import (
    cdw_density_blocks,
    flatten_sector_blocks,
    polshyn_nu_7over2_filling_summary,
    translation_order_parameters,
    unflatten_sector_blocks,
)


def test_polshyn_hf_filling_summary_matches_nu_7over2_convention() -> None:
    projected = (25, 26, 27, 28)
    summary = polshyn_nu_7over2_filling_summary(projected, target_band_index=27)

    assert summary.projected_indices == projected
    assert summary.target_primitive_position == 2
    assert summary.target_fold_indices == (4, 5)
    assert summary.nb == 8
    assert summary.area_ratio == 2
    assert np.allclose(summary.reference_diagonal, [1, 1, 1, 1, 0, 0, 0, 0])
    assert summary.occupation_counts.tolist() == [[5, 6], [6, 6]]
    assert np.isclose(summary.primitive_nu, 3.5, atol=1.0e-12)
    assert summary.matches_expected_filling

    as_dict = summary.to_dict()
    assert as_dict["target_fold_indices"] == [4, 5]
    assert as_dict["matches_expected_filling"] is True


def test_polshyn_cdw_initializer_has_maximal_target_fold_order() -> None:
    projected = (25, 26, 27, 28)
    summary = polshyn_nu_7over2_filling_summary(projected, target_band_index=27)

    density = cdw_density_blocks(
        projected_indices=projected,
        target_band_index=27,
        n_spin=2,
        n_eta=2,
        nb=summary.nb,
        nk=3,
        reference_diagonal=summary.reference_diagonal,
    )
    order = translation_order_parameters(
        density,
        projected_indices=projected,
        target_band_index=27,
        spin_index=0,
        valley_index=0,
    )

    assert np.allclose(order["target_x2"], np.ones(3))
    assert np.isclose(order["target_x2_mean"], 1.0, atol=1.0e-12)
    assert np.allclose(density[0, 0, 4, 5, :], 0.5)
    assert np.allclose(density[0, 0, 5, 4, :], 0.5)
    assert np.allclose(density[1, 0, 4, 5, :], 0.0)
    assert np.allclose(density[0, 1, 4, 5, :], 0.0)


def test_polshyn_sector_flatten_round_trip_preserves_block_layout() -> None:
    blocks = np.zeros((2, 2, 3, 3, 2), dtype=np.complex128)
    for ispin in range(2):
        for ieta in range(2):
            for ib in range(3):
                for jb in range(3):
                    for ik in range(2):
                        blocks[ispin, ieta, ib, jb, ik] = (
                            1000 * ispin
                            + 100 * ieta
                            + 10 * ib
                            + jb
                            + 0.1 * ik
                            + 1j * (ib - jb)
                        )

    flat = flatten_sector_blocks(blocks)
    restored = unflatten_sector_blocks(flat, n_spin=2, n_eta=2, nb=3)

    assert flat.shape == (12, 12, 2)
    assert np.allclose(restored, blocks)
