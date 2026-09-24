"""Typed grounded-double-gate screening for projected PtSe2 Hartree-Fock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COULOMB_EV_NM = 1.439964548  # e^2/(4*pi*epsilon_0)


@dataclass(frozen=True)
class PtSe2DoubleGateScreening:
    epsilon_top: float
    epsilon_bottom: float
    d_top_nm: float
    d_bottom_nm: float

    def __post_init__(self) -> None:
        for name, value in (
            ("epsilon_top", self.epsilon_top),
            ("epsilon_bottom", self.epsilon_bottom),
            ("d_top_nm", self.d_top_nm),
            ("d_bottom_nm", self.d_bottom_nm),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def interaction_ev_nm2(
        self, q_nm_inv: float | np.ndarray
    ) -> float | np.ndarray:
        """Return W(q) in eV nm^2, including the analytic q=0 limit."""

        q = np.asarray(q_nm_inv, dtype=np.float64)
        if np.any(~np.isfinite(q)) or np.any(q < 0.0):
            raise ValueError("q must be finite and nonnegative")
        result = np.empty_like(q)
        zero = q == 0.0
        result[zero] = 4.0 * np.pi * COULOMB_EV_NM / (
            self.epsilon_top / self.d_top_nm
            + self.epsilon_bottom / self.d_bottom_nm
        )
        nonzero = ~zero
        if np.any(nonzero):
            values = q[nonzero]
            denominator = values * (
                self.epsilon_top / np.tanh(values * self.d_top_nm)
                + self.epsilon_bottom / np.tanh(values * self.d_bottom_nm)
            )
            result[nonzero] = 4.0 * np.pi * COULOMB_EV_NM / denominator
        if np.ndim(q_nm_inv) == 0:
            return float(result)
        return result
