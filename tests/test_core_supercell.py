from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.supercell import (
    IntegerSupercell,
    fixed_sector_occupation_counts,
    folded_indices_for_primitive_band,
    folded_reference_diagonal_by_primitive_index,
    occupied_count_from_primitive_filling,
    primitive_filling_from_occupation_counts,
)
from mean_field.systems.tbg.zero_field.supercell import (
    filling_from_occupation_counts,
    occupation_counts_svp_8over3,
    zhang_sqrt3_tripled_supercell,
)
from mean_field.systems.tmbg.polshyn_supercell import polshyn_nu_7over2_filling_summary


def test_integer_supercell_geometry_matches_zhang_tripled_cell() -> None:
    cell = IntegerSupercell(n11=1, n12=1, n21=-1, n22=2)
    b1 = 1.0 + 0.0j
    b2 = 0.0 + 1.0j
    g1, g2 = cell.reciprocal_vectors(b1, b2)
    assert cell.area_ratio == 3
    assert np.isclose(g1, (2.0 * b1 + b2) / 3.0)
    assert np.isclose(g2, (-b1 + b2) / 3.0)
    assert cell.primitive_shift_to_supercell(1, 0) == (1, -1)
    assert cell.primitive_shift_to_supercell(0, 1) == (1, 2)


def test_tbg_zhang_svp_occupation_counts_are_case_guarded() -> None:
    cell = zhang_sqrt3_tripled_supercell()
    occ = occupation_counts_svp_8over3(6)
    assert occ.tolist() == [[6, 2], [6, 6]]
    assert np.isclose(filling_from_occupation_counts(occ, nb=6, area_ratio=cell.area_ratio), 8.0 / 3.0)
    with pytest.raises(ValueError, match="nb=6"):
        occupation_counts_svp_8over3(4)


def test_generic_fixed_sector_filling_formula_matches_zhang() -> None:
    occ = fixed_sector_occupation_counts(
        n_spin=2,
        n_eta=2,
        default_count=6,
        overrides={(0, 1): 2},
        n_band=6,
    )
    assert np.isclose(
        primitive_filling_from_occupation_counts(occ, reference_diagonal=0.5, n_band=6, area_ratio=3),
        8.0 / 3.0,
    )
    assert occupied_count_from_primitive_filling(
        8.0 / 3.0,
        reference_diagonal=0.5,
        n_band=6,
        area_ratio=3,
        n_sector=4,
    ) == 20


def test_generic_folded_reference_matches_polshyn_conduction_convention() -> None:
    projected = (25, 26, 27, 28)
    reference = folded_reference_diagonal_by_primitive_index(
        projected,
        target_band_index=27,
        folds_per_primitive=2,
        lower_reference=1.0,
        target_reference=0.0,
        upper_reference=0.0,
    )
    assert np.allclose(reference, [1, 1, 1, 1, 0, 0, 0, 0])
    assert folded_indices_for_primitive_band(projected, target_band_index=27, folds_per_primitive=2) == (4, 5)
    summary = polshyn_nu_7over2_filling_summary(projected, target_band_index=27)
    assert summary.occupation_counts.tolist() == [[5, 6], [6, 6]]
    assert np.isclose(summary.primitive_nu, 3.5)
    assert occupied_count_from_primitive_filling(
        3.5,
        reference_diagonal=reference,
        area_ratio=2,
        n_sector=4,
    ) == 23
