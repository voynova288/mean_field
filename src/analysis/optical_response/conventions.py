from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

OpticalSymmetrization = Literal["none", "sum", "average"]

@dataclass(frozen=True)
class ShiftCurrentConvention:
    """Response-layer convention bundle.

    The derivative route is *not* configurable here: reusable code uses
    ``analysis.response_derivative_gauge`` and its WannierBerri Hamiltonian-gauge
    convention.  This dataclass records only response/plotting choices that
    differ between references: ordered vs symmetrized optical indices, the sign
    of the local geometric kernel, and whether the optical Lorentzian includes
    the normalizing ``1/pi``.
    """

    name: str
    optical_symmetrization: OpticalSymmetrization
    geometric_sign: float = 1.0
    normalized_lorentzian: bool = True
    description: str = ""


# Joya 2025 point audits showed that this ordered same-polarization kernel
# matches the explicit paper Eq.(7) c-sum directly, before omitted global
# conductivity prefactors, spin factor, SI conversion, and final colorbar sign.
JOYA_EQ7_GEOMETRIC_CONVENTION = ShiftCurrentConvention(
    name="joya2025_eq7_geometric",
    optical_symmetrization="none",
    geometric_sign=1.0,
    normalized_lorentzian=False,
    description="Ordered local pair kernel equals Joya 2025 Eq.(7) c-sum; optical Lorentzian has no 1/pi.",
)

# WannierBerri dynamic.py::ShiftCurrentFormula symmetrizes optical indices and
# its internal Imn has the opposite sign from the local pair product.  For b==c
# this is the audited relation Imn = -2 * ordered_pair_kernel.
WANNIERBERRI_INTERNAL_IMN_CONVENTION = ShiftCurrentConvention(
    name="wannierberri_internal_imn",
    optical_symmetrization="sum",
    geometric_sign=-1.0,
    normalized_lorentzian=True,
    description="Line-by-line convention of WannierBerri ShiftCurrentFormula internal Imn.",
)

HTG_LEGACY_CONVENTION = ShiftCurrentConvention(
    name="htg_legacy_sum",
    optical_symmetrization="sum",
    geometric_sign=1.0,
    normalized_lorentzian=True,
    description="Legacy hTG workspace convention: symmetrized optical product before the -i prefactor.",
)

E_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
HBAR_EV_S = 6.582119569e-16
KB_EV_PER_K = 8.617333262145e-5
SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 = math.pi * E_CHARGE_C**2 / HBAR_J_S * 1.0e6


__all__ = [
    "E_CHARGE_C",
    "HBAR_EV_S",
    "HBAR_J_S",
    "HTG_LEGACY_CONVENTION",
    "JOYA_EQ7_GEOMETRIC_CONVENTION",
    "KB_EV_PER_K",
    "OpticalSymmetrization",
    "SHIFT_CURRENT_PREFAC_UA_NM_PER_V2",
    "ShiftCurrentConvention",
    "WANNIERBERRI_INTERNAL_IMN_CONVENTION",
]
