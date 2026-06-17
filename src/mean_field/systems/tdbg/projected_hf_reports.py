from __future__ import annotations

import numpy as np

from .projected_hf_config import (
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
    VALLEY_LABELS,
    VALLEY_SEQUENCE,
    tdbg_delta_from_paper_ud_for_valley,
)
from .projected_hf_state import TDBGProjectedHFResult

def tdbg_hf_grid_band_summary(result: TDBGProjectedHFResult) -> dict[str, object]:
    energies = np.asarray(result.run.state.energies, dtype=float)
    nocc = result.data.n_occupied_per_k
    if nocc <= 0 or nocc >= energies.shape[0]:
        gap = float("nan")
    else:
        gap = float(np.min(energies[nocc:, :]) - np.max(energies[:nocc, :]))
    return {
        "classification": result.order_parameters.get("classification"),
        "init_mode": result.init_mode,
        "seed": int(result.seed),
        "occupied_per_k": int(nocc),
        "hf_grid_gap_ev": gap,
        "hf_energy_ev": float(result.energy_components["total_ev"]),
        "energy_min_ev": float(np.min(energies)),
        "energy_max_ev": float(np.max(energies)),
    }


def liu2022_default_projected_hf_config(
    *,
    mesh_size: int = 9,
    cut: float = 5.0,
    window: str = "two_flat",
    include_intersite: bool = True,
    include_onsite: bool = True,
    filling: int = 2,
    max_iter: int = 300,
    precision: float = 1.0e-7,
) -> TDBGProjectedHFConfig:
    return TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=float(cut),
        mesh_size=int(mesh_size),
        paper_ud_ev=0.09,
        stacking="AB-BA",
        window=TDBGProjectedWindow(name=window),
        filling=int(filling),
        interaction=TDBGInteractionSettings(include_intersite=include_intersite, include_onsite=include_onsite),
        precision=float(precision),
        max_iter=int(max_iter),
    )

def liu2022_projected_hf_metadata(config: TDBGProjectedHFConfig) -> dict[str, object]:
    return {
        "paper": "Liu Nat Commun 2022 TDBG projected-HF pilot (not a reproduction claim)",
        "theta_deg": float(config.theta_deg),
        "paper_ud_ev": float(config.paper_ud_ev),
        "paper_ud_convention": config.paper_ud_convention,
        "code_delta_ev": float(tdbg_delta_from_paper_ud_for_valley(config.paper_ud_ev, 1, convention=config.paper_ud_convention)),
        "code_delta_by_valley_ev": {
            VALLEY_LABELS[valley]: float(tdbg_delta_from_paper_ud_for_valley(config.paper_ud_ev, valley, convention=config.paper_ud_convention))
            for valley in VALLEY_SEQUENCE
        },
        "stacking": config.stacking,
        "cut": float(config.cut),
        "mesh_size": int(config.mesh_size),
        "window": config.window.name,
        "explicit_band_indices": None if config.window.band_indices is None else list(config.window.band_indices),
        "filling": int(config.filling),
        "orbital_zeeman_b_t": float(config.orbital_zeeman_b_t),
        "orbital_zeeman_delta_k_nm_inv": float(config.orbital_zeeman_delta_k_nm_inv),
        "include_intersite": bool(config.interaction.include_intersite),
        "include_onsite": bool(config.interaction.include_onsite),
        "hubbard_u_ev": float(config.interaction.hubbard_u_ev),
        "epsilon_r": float(config.interaction.epsilon_r),
        "kappa_nm_inv": float(config.interaction.kappa_nm_inv),
        "hartree_reference": config.interaction.hartree_reference,
        "fock_density": config.interaction.fock_density,
        "onsite_valley_policy": config.interaction.onsite_valley_policy,
        "density_convention": "core stored projector P[a,b,k]=rho_conventional[b,a,k]",
        "reference_density_convention": "state density is absolute occupied projector; Hartree/Fock policies choose whether to subtract the explicit reference projector",
        "energy_convention": "component-resolved stored-projector functional: Hartree contracts with its policy density, Fock with its policy density, onsite with absolute density",
        "workflow": "self-consistent projected HF from multiple trial states, order-parameter classification, component-resolved diagnostic HF energies, and optional target-path reconstruction",
        "known_limitations": [
            "onsite intervalley/local-channel convention must be checked against Liu SI before any paper-comparison claim",
            "central4/central6 windows require separate topology/window diagnostics before production claims",
            "target-path HF bands are reconstructed from source-grid density; paper overlay and cutoff/mesh convergence remain external validation gates",
        ],
    }

__all__ = [
    "liu2022_default_projected_hf_config",
    "liu2022_projected_hf_metadata",
    "tdbg_hf_grid_band_summary",
]
