from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class AhnTMDParameters:
    """Ahn et al. 2024 Table-I-like tMoTe2 continuum parameters.

    Mean_Field uses eV and nm internally in system adapters.  The published
    parameters are conventionally quoted in meV and Angstrom, so both are kept
    explicit at the boundary.
    """

    a0_nm: float = 0.352
    mstar_me: float = 0.62
    v1_eV: float = 20.51e-3
    psi_deg: float = -61.49
    gamma1_eV: float = -7.01e-3
    v2_eV: float = -9.08e-3
    gamma2_eV: float = 11.08e-3
    psi_prime_deg: float = 0.0
    hbar2_over_2me_eV_nm2: float = 0.03810

    @property
    def psi_rad(self) -> float:
        return self.psi_deg * pi / 180.0

    @property
    def psi_prime_rad(self) -> float:
        return self.psi_prime_deg * pi / 180.0

    @property
    def hbar2_over_2m_eV_nm2(self) -> float:
        return self.hbar2_over_2me_eV_nm2 / self.mstar_me


DEFAULT_AHN_TMD_PARAMETERS = AhnTMDParameters()


__all__ = ["AhnTMDParameters", "DEFAULT_AHN_TMD_PARAMETERS"]
