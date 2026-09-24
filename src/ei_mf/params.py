"""Parameter definitions for the Nature Communications InAs/GaSb EI reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Literal


@dataclass(frozen=True)
class EIParams:
    """Stage-A/default parameters in paper convention.

    Momentum is in nm^-1, length in nm, energy in meV, density in cm^-2.
    The solver reports ``E_pair = sqrt(xi**2 + Delta_order**2)``; it does not
    use the doubled superconducting-BCS quasiparticle convention.
    """

    # effective masses for Stage A
    m_e_over_m0: float = 0.032
    m_h_over_m0: float = 0.136

    # dielectric and electron-hole layer separation
    eps_r: float = 15.0
    d_eh_nm: float = 10.0

    # density and temperature
    n0_cm2: float = 5.5e10
    temperature_K: float = 0.1

    # radial momentum grid; k in nm^-1
    k_max_nm_inv: float = 0.12
    nk: int = 800
    nphi: int = 720

    # numerical stability
    q_floor_nm_inv: float | None = None
    energy_floor_meV: float = 1e-12
    mix: float = 0.05
    max_iter: int = 30000
    tol_meV: float = 1e-8

    # JDOS broadening
    eta_jdos_meV: float = 0.10

    # Explicit sensitivity controls. Non-source-backed overrides are forensic-only
    # and must never be selected against paper curves or presented as predictions.
    kF_nm_inv: float | None = None
    pairing_scale: float = 1.0
    screening_model: Literal["none", "tf"] = "none"
    q_tf_nm_inv: float = 0.0

    def validate(self) -> None:
        if self.m_e_over_m0 <= 0 or self.m_h_over_m0 <= 0:
            raise ValueError("effective masses must be positive")
        if not isfinite(self.eps_r) or self.eps_r <= 0:
            raise ValueError("eps_r must be finite and positive")
        if not isfinite(self.d_eh_nm) or self.d_eh_nm < 0:
            raise ValueError("d_eh_nm must be finite and non-negative")
        if self.n0_cm2 < 0:
            raise ValueError("n0_cm2 must be non-negative")
        if self.k_max_nm_inv <= 0 or self.nk < 4 or self.nphi < 4:
            raise ValueError("invalid radial/angular grid")
        if self.energy_floor_meV <= 0:
            raise ValueError("energy_floor_meV must be positive")
        if not (0.0 < self.mix <= 1.0):
            raise ValueError("mix must lie in (0, 1]")
        if self.max_iter < 1 or self.tol_meV <= 0:
            raise ValueError("invalid iteration controls")
        if self.eta_jdos_meV <= 0:
            raise ValueError("eta_jdos_meV must be positive")
        if self.kF_nm_inv is not None and self.kF_nm_inv <= 0:
            raise ValueError("kF_nm_inv override must be positive")
        if self.pairing_scale < 0:
            raise ValueError("pairing_scale must be non-negative")
        if self.screening_model not in {"none", "tf"}:
            raise ValueError(f"unsupported screening_model={self.screening_model!r}")
        if self.q_tf_nm_inv < 0:
            raise ValueError("q_tf_nm_inv must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def updated(self, **kwargs: object) -> "EIParams":
        params = replace(self, **kwargs)
        params.validate()
        return params
