from __future__ import annotations

import numpy as np

from .conventions import KB_EV_PER_K

def fermi_occupation(energies_ev: np.ndarray, *, mu_ev: float = 0.0, temperature_k: float = 0.0) -> np.ndarray:
    """Fermi occupation for energies in eV."""

    energies = np.asarray(energies_ev, dtype=float)
    if float(temperature_k) <= 0.0:
        return (energies < float(mu_ev)).astype(float)
    beta_arg = (energies - float(mu_ev)) / (KB_EV_PER_K * float(temperature_k))
    out = np.empty_like(beta_arg, dtype=float)
    out[beta_arg > 40.0] = 0.0
    out[beta_arg < -40.0] = 1.0
    mask = (beta_arg >= -40.0) & (beta_arg <= 40.0)
    out[mask] = 1.0 / (np.exp(beta_arg[mask]) + 1.0)
    return out


__all__ = ["fermi_occupation"]
