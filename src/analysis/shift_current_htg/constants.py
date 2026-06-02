from __future__ import annotations

import math

# SI constants.
E_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
HBAR_EV_S = 6.582119569e-16
KB_EV_PER_K = 8.617333262145e-5

# Graphene / Mao et al. PRB 111, 195408 (2025) defaults.
GRAPHENE_LATTICE_CONSTANT_NM = 0.246
CARBON_BOND_NM = GRAPHENE_LATTICE_CONSTANT_NM / math.sqrt(3.0)
MAO_HBAR_VF_K_EV = 9.905
MAO_W1_EV = 0.110
MAO_SUBLATTICE_MASS_EV = 0.030
MAO_REALISTIC_CORRUGATION = 0.8

# The gauge-free code uses k in nm^{-1}, energies in eV, r in nm, and
# delta(E_gamma - E_mn) in eV^{-1}.  For an already integrated quantity
# I with units nm/eV, the physical result in microampere*nm/V^2 is
# Re[-i * SHIFT_CURRENT_PREFAC * I].
SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 = math.pi * E_CHARGE_C**2 / HBAR_J_S * 1.0e6


def graphene_k_mag_nm_inv(a_nm: float = GRAPHENE_LATTICE_CONSTANT_NM) -> float:
    """Return |K| = 4*pi/(3a) in nm^{-1} for the graphene convention used here."""

    return 4.0 * math.pi / (3.0 * float(a_nm))


def mao_vf_ev_nm(a_nm: float = GRAPHENE_LATTICE_CONSTANT_NM) -> float:
    """Return hbar*v_F in eV*nm from Mao's hbar*v_F*|K| = 9.905 eV."""

    return MAO_HBAR_VF_K_EV / graphene_k_mag_nm_inv(a_nm)


def vf_ev_nm_to_m_per_s(vf_ev_nm: float) -> float:
    """Convert hbar*v_F from eV*nm to v_F in m/s."""

    return float(vf_ev_nm) / (HBAR_EV_S * 1.0e9)


def eta_mev_to_ev(eta_mev: float) -> float:
    return float(eta_mev) * 1.0e-3
