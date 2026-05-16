from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .params import DEFAULT_LAYER_SPACING_NM, VALID_LAYER_COUNTS


E2_OVER_2_EPS0_MEV_NM = 2.0 * math.pi * 1439.96448
VALID_INTERACTION_SCHEMES = ("average", "cn")
VALID_INTERACTION_DIMENSIONS = ("3d_layer_dependent", "2d_diagnostic")


@dataclass(frozen=True)
class RLGhBNInteractionParams:
    """Interaction settings for projected RLG/hBN Hartree-Fock calculations.

    Energies are in meV, lengths are in nm, and momenta are in nm^-1.
    The Coulomb kernels returned by this module therefore have units
    meV nm^2, matching a two-dimensional Fourier transform convention.
    """

    epsilon_r: float = 5.0
    gate_distance_nm: float = 10.0
    scheme: str = "average"
    interaction_dimension: str = "3d_layer_dependent"
    active_valence_bands: int = 4
    active_conduction_bands: int = 4
    k_mesh_size: int = 12
    hilbert_cutoff_q1: float = 4.0
    interaction_cutoff_q1: float = 3.0
    use_screened_basis: bool = True

    def __post_init__(self) -> None:
        epsilon_r = float(self.epsilon_r)
        gate_distance_nm = float(self.gate_distance_nm)
        if epsilon_r <= 0.0:
            raise ValueError(f"epsilon_r must be positive, got {self.epsilon_r}")
        if gate_distance_nm <= 0.0:
            raise ValueError(f"gate_distance_nm must be positive, got {self.gate_distance_nm}")
        if self.scheme not in VALID_INTERACTION_SCHEMES:
            raise ValueError(f"scheme must be one of {VALID_INTERACTION_SCHEMES}, got {self.scheme!r}")
        if self.interaction_dimension not in VALID_INTERACTION_DIMENSIONS:
            raise ValueError(
                "interaction_dimension must be one of "
                f"{VALID_INTERACTION_DIMENSIONS}, got {self.interaction_dimension!r}"
            )
        if int(self.active_valence_bands) < 0:
            raise ValueError(f"active_valence_bands must be nonnegative, got {self.active_valence_bands}")
        if int(self.active_conduction_bands) <= 0:
            raise ValueError(f"active_conduction_bands must be positive, got {self.active_conduction_bands}")
        if int(self.k_mesh_size) <= 0:
            raise ValueError(f"k_mesh_size must be positive, got {self.k_mesh_size}")
        if float(self.hilbert_cutoff_q1) <= 0.0:
            raise ValueError(f"hilbert_cutoff_q1 must be positive, got {self.hilbert_cutoff_q1}")
        if float(self.interaction_cutoff_q1) <= 0.0:
            raise ValueError(f"interaction_cutoff_q1 must be positive, got {self.interaction_cutoff_q1}")

        object.__setattr__(self, "epsilon_r", epsilon_r)
        object.__setattr__(self, "gate_distance_nm", gate_distance_nm)
        object.__setattr__(self, "active_valence_bands", int(self.active_valence_bands))
        object.__setattr__(self, "active_conduction_bands", int(self.active_conduction_bands))
        object.__setattr__(self, "k_mesh_size", int(self.k_mesh_size))
        object.__setattr__(self, "hilbert_cutoff_q1", float(self.hilbert_cutoff_q1))
        object.__setattr__(self, "interaction_cutoff_q1", float(self.interaction_cutoff_q1))
        object.__setattr__(self, "use_screened_basis", bool(self.use_screened_basis))

    @property
    def active_band_count(self) -> int:
        return int(self.active_valence_bands + self.active_conduction_bands)

    def to_summary_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "epsilon_r": float(self.epsilon_r),
            "gate_distance_nm": float(self.gate_distance_nm),
            "scheme": str(self.scheme),
            "interaction_dimension": str(self.interaction_dimension),
            "active_valence_bands": int(self.active_valence_bands),
            "active_conduction_bands": int(self.active_conduction_bands),
            "active_band_count": int(self.active_band_count),
            "k_mesh_size": int(self.k_mesh_size),
            "hilbert_cutoff_q1": float(self.hilbert_cutoff_q1),
            "interaction_cutoff_q1": float(self.interaction_cutoff_q1),
            "use_screened_basis": bool(self.use_screened_basis),
        }


def layer_z_coordinates_nm(layer_count: int, *, layer_spacing_nm: float = DEFAULT_LAYER_SPACING_NM) -> np.ndarray:
    layer_count = int(layer_count)
    if layer_count not in VALID_LAYER_COUNTS:
        raise ValueError(f"Expected layer_count in {VALID_LAYER_COUNTS}, got {layer_count}")
    layer_spacing_nm = float(layer_spacing_nm)
    if layer_spacing_nm <= 0.0:
        raise ValueError(f"layer_spacing_nm must be positive, got {layer_spacing_nm}")
    layers = np.arange(layer_count, dtype=float)
    return (layers - 0.5 * float(layer_count - 1)) * layer_spacing_nm


def screened_coulomb_2d_mev_nm2(
    q_nm_inv: float,
    *,
    epsilon_r: float = 5.0,
    gate_distance_nm: float = 10.0,
) -> float:
    q_abs = abs(float(q_nm_inv))
    epsilon_r = float(epsilon_r)
    gate_distance_nm = float(gate_distance_nm)
    if epsilon_r <= 0.0:
        raise ValueError(f"epsilon_r must be positive, got {epsilon_r}")
    if gate_distance_nm <= 0.0:
        raise ValueError(f"gate_distance_nm must be positive, got {gate_distance_nm}")
    if q_abs == 0.0:
        return float(E2_OVER_2_EPS0_MEV_NM * gate_distance_nm / epsilon_r)
    return float(E2_OVER_2_EPS0_MEV_NM * math.tanh(q_abs * gate_distance_nm) / (epsilon_r * q_abs))


def screened_coulomb_layer_mev_nm2(
    q_nm_inv: float,
    z_l_nm: float,
    z_lp_nm: float,
    *,
    epsilon_r: float = 5.0,
    gate_distance_nm: float = 10.0,
) -> float:
    """Dual-gate Coulomb kernel between two layers at finite in-plane q.

    The expression is the Dirichlet Green's function between metallic gates at
    z = +/- gate_distance_nm.  At z_l = z_lp = 0 it reduces to the familiar
    2D dual-gate kernel e^2 tanh(q d_sc) / (2 eps0 eps_r q).
    """

    q_abs = abs(float(q_nm_inv))
    z_l = float(z_l_nm)
    z_lp = float(z_lp_nm)
    epsilon_r = float(epsilon_r)
    gate_distance_nm = float(gate_distance_nm)
    if epsilon_r <= 0.0:
        raise ValueError(f"epsilon_r must be positive, got {epsilon_r}")
    if gate_distance_nm <= 0.0:
        raise ValueError(f"gate_distance_nm must be positive, got {gate_distance_nm}")
    if abs(z_l) > gate_distance_nm or abs(z_lp) > gate_distance_nm:
        raise ValueError("Layer coordinates must lie between the two metallic gates")
    if q_abs == 0.0:
        return q0_interlayer_hartree_mev_nm2(z_l, z_lp, epsilon_r=epsilon_r)

    z_min = min(z_l, z_lp)
    z_max = max(z_l, z_lp)
    x = q_abs * gate_distance_nm
    numerator = 2.0 * math.sinh(q_abs * (z_min + gate_distance_nm))
    numerator *= math.sinh(q_abs * (gate_distance_nm - z_max))
    denominator = q_abs * math.sinh(2.0 * x)
    return float(E2_OVER_2_EPS0_MEV_NM * numerator / (epsilon_r * denominator))


def q0_interlayer_hartree_mev_nm2(
    z_l_nm: float,
    z_lp_nm: float,
    *,
    epsilon_r: float = 5.0,
) -> float:
    epsilon_r = float(epsilon_r)
    if epsilon_r <= 0.0:
        raise ValueError(f"epsilon_r must be positive, got {epsilon_r}")
    return float(-E2_OVER_2_EPS0_MEV_NM * abs(float(z_l_nm) - float(z_lp_nm)) / epsilon_r)


def layer_coulomb_matrix_mev_nm2(
    q_nm_inv: float,
    layer_count: int,
    interaction: RLGhBNInteractionParams | None = None,
    *,
    layer_spacing_nm: float = DEFAULT_LAYER_SPACING_NM,
) -> np.ndarray:
    interaction = RLGhBNInteractionParams() if interaction is None else interaction
    z = layer_z_coordinates_nm(layer_count, layer_spacing_nm=layer_spacing_nm)
    matrix = np.zeros((int(layer_count), int(layer_count)), dtype=float)
    for il, z_l in enumerate(z):
        for jl, z_j in enumerate(z):
            if interaction.interaction_dimension == "2d_diagnostic":
                matrix[il, jl] = screened_coulomb_2d_mev_nm2(
                    q_nm_inv,
                    epsilon_r=interaction.epsilon_r,
                    gate_distance_nm=interaction.gate_distance_nm,
                )
            else:
                matrix[il, jl] = screened_coulomb_layer_mev_nm2(
                    q_nm_inv,
                    float(z_l),
                    float(z_j),
                    epsilon_r=interaction.epsilon_r,
                    gate_distance_nm=interaction.gate_distance_nm,
                )
    return matrix


__all__ = [
    "E2_OVER_2_EPS0_MEV_NM",
    "RLGhBNInteractionParams",
    "VALID_INTERACTION_DIMENSIONS",
    "VALID_INTERACTION_SCHEMES",
    "layer_coulomb_matrix_mev_nm2",
    "layer_z_coordinates_nm",
    "q0_interlayer_hartree_mev_nm2",
    "screened_coulomb_2d_mev_nm2",
    "screened_coulomb_layer_mev_nm2",
]
