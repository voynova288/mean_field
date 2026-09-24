"""Xue--MacDonald 2018 Q=0 BHZ reproduction conventions.

Source: F. Xue and A. H. MacDonald, Phys. Rev. Lett. 120, 186802
(2018), arXiv:1710.00410v3. The model reuses the one-slab Q=0 limit of
the Zeng--Xue--MacDonald folded-BHZ definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .zeng2022 import Zeng2022Parameters

FloatArray = NDArray[np.float64]


XUE2018_ELECTRON_MASS_M0 = 0.023
XUE2018_HOLE_MASS_M0 = 0.4
XUE2018_D_OVER_AB = 0.3


@dataclass(frozen=True)
class Xue2018Fig2Path:
    """Preregistered reconstruction of the gray Fig. 1/Fig. 2 path.

    The paper attests the path and every-tenth-point labels, but does not print
    a coordinate table. These coordinates are therefore tagged as inferred
    from the vector figure geometry rather than author-released raw data.
    """

    point_index: NDArray[np.int64]
    eg_ry: FloatArray
    hybridization_ab_ry: FloatArray
    authority: str = "figure_geometry_inferred"

    def anchor_indices(self) -> NDArray[np.int64]:
        return np.asarray([1, 11, 21, 31, 41, 61], dtype=np.int64)

    def select(self, paper_indices: NDArray[np.int64]) -> tuple[FloatArray, FloatArray]:
        requested = np.asarray(paper_indices, dtype=np.int64)
        if np.any(requested < 1) or np.any(requested > 62):
            raise ValueError("Fig. 2 paper indices must lie in [1,62]")
        positions = requested - 1
        return self.eg_ry[positions].copy(), self.hybridization_ab_ry[positions].copy()


def xue2018_standard_parameters(*, eg_ry: float, hybridization_ab_ry: float) -> Zeng2022Parameters:
    """Return the source production Q=0 parameter set in paper units.

    Xue--MacDonald Eq. (6) separates the particle-hole-asymmetric identity
    term ``zeta_k`` from the reduced-mass ``xi_k`` term and explicitly states
    that ``zeta_k`` is dropped below.  Therefore the numerical model has the
    equivalent symmetric masses ``m_e=m_h=2m``.  The quoted physical masses
    set ``m``, ``a_B*``, and ``Ry*``; they are not retained as ``zeta_k`` in
    the production Hamiltonian.
    """

    return Zeng2022Parameters(
        eg_ry=float(eg_ry),
        hybridization_ab_ry=float(hybridization_ab_ry),
        q_ab_inv=0.0,
        d_over_ab=XUE2018_D_OVER_AB,
        mass_e_over_reduced=2.0,
        mass_h_over_reduced=2.0,
    )


def xue2018_physical_asymmetry_diagnostic_parameters(
    *, eg_ry: float, hybridization_ab_ry: float
) -> Zeng2022Parameters:
    """Return the explicitly nonproduction lane that retains ``zeta_k``."""

    reduced_mass_m0 = (
        XUE2018_ELECTRON_MASS_M0
        * XUE2018_HOLE_MASS_M0
        / (XUE2018_ELECTRON_MASS_M0 + XUE2018_HOLE_MASS_M0)
    )
    return Zeng2022Parameters(
        eg_ry=float(eg_ry),
        hybridization_ab_ry=float(hybridization_ab_ry),
        q_ab_inv=0.0,
        d_over_ab=XUE2018_D_OVER_AB,
        mass_e_over_reduced=XUE2018_ELECTRON_MASS_M0 / reduced_mass_m0,
        mass_h_over_reduced=XUE2018_HOLE_MASS_M0 / reduced_mass_m0,
    )


def xue2018_fig2_inferred_path() -> Xue2018Fig2Path:
    """Return the 62-point L-shaped path inferred from Figs. 1 and 2.

    A 600-dpi marker count resolves 62 blue gap points in Fig. 2. Combined
    with the source Fig. 1 labels, this implies points 1--22 hold ``A=0.2``
    while ``-E_g`` increases from ``-1.6`` to ``+0.5`` in exact steps of
    0.1 (labels 1, 11, and 21 lie at -1.6, -0.6, and +0.4); points 22--62
    hold ``E_g=-0.5`` while ``A`` increases from 0.2 to 0.6 in steps of
    0.01. Point 22 is shared. This remains figure-geometry inference rather
    than an author coordinate table.
    """

    vertical_neg_eg = np.linspace(-1.6, 0.5, 22, dtype=np.float64)
    vertical_a = np.full(22, 0.2, dtype=np.float64)
    horizontal_a = np.linspace(0.21, 0.6, 40, dtype=np.float64)
    horizontal_eg = np.full(40, -0.5, dtype=np.float64)
    neg_eg = np.concatenate([vertical_neg_eg, -horizontal_eg])
    a_values = np.concatenate([vertical_a, horizontal_a])
    return Xue2018Fig2Path(
        point_index=np.arange(1, 63, dtype=np.int64),
        eg_ry=-neg_eg,
        hybridization_ab_ry=a_values,
    )


__all__ = [
    "XUE2018_D_OVER_AB",
    "XUE2018_ELECTRON_MASS_M0",
    "XUE2018_HOLE_MASS_M0",
    "Xue2018Fig2Path",
    "xue2018_fig2_inferred_path",
    "xue2018_physical_asymmetry_diagnostic_parameters",
    "xue2018_standard_parameters",
]
