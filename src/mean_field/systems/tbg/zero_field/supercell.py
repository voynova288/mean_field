from __future__ import annotations

"""Lightweight zero-field TBG supercell convention helpers.

The old module also carried an unexported standalone supercell BM/SCF workflow.
That workflow duplicated the maintained zero-field BM/HF runners and generic
core-HF utilities, so the tracked surface now keeps only the Zhang sqrt(3)
filling-convention helpers used by public tests.
"""

import numpy as np

from ....core.supercell import (
    IntegerSupercell,
    fixed_sector_occupation_counts,
    primitive_filling_from_occupation_counts,
)

class MoireSupercell(IntegerSupercell):
    """TBG-facing alias for the generic integer-supercell convention."""


def zhang_sqrt3_tripled_supercell() -> MoireSupercell:
    """The ``sqrt(3) x sqrt(3)`` tripled cell used for nu=8/3 in Zhang Fig. 10."""

    return MoireSupercell(n11=1, n12=1, n21=-1, n22=2)


def occupation_counts_svp_8over3(nb: int) -> np.ndarray:
    """Fixed-sector occupation counts for Zhang's nu=8/3 SVP convention."""

    if int(nb) != 6:
        raise ValueError(
            "occupation_counts_svp_8over3 is specific to Zhang's sqrt(3) x sqrt(3) tripled cell, "
            f"where nb=6 folded bands are kept per spin/valley sector; got nb={nb}."
        )
    return fixed_sector_occupation_counts(
        n_spin=2,
        n_eta=2,
        default_count=6,
        overrides={(0, 1): 2},
        n_band=int(nb),
    )


def filling_from_occupation_counts(occupation_counts: np.ndarray, *, nb: int, area_ratio: int) -> float:
    return primitive_filling_from_occupation_counts(
        occupation_counts,
        reference_diagonal=0.5,
        n_band=int(nb),
        area_ratio=int(area_ratio),
    )


__all__ = [
    "MoireSupercell",
    "filling_from_occupation_counts",
    "occupation_counts_svp_8over3",
    "zhang_sqrt3_tripled_supercell",
]
