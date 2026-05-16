from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .hamiltonian import valence_band_count
from .interaction import RLGhBNInteractionParams, layer_z_coordinates_nm, q0_interlayer_hartree_mev_nm2
from .model import RLGhBNModel
from .params import DEFAULT_LAYER_SPACING_NM, RLGhBNParams


@dataclass(frozen=True)
class LayerChargeResult:
    layer_charge: np.ndarray
    reference_layer_charge: np.ndarray
    delta_layer_charge: np.ndarray
    mesh_size: int
    n_spin: int
    valleys: tuple[int, ...]
    n_valence_bands: int


@dataclass(frozen=True)
class InterlayerHartreeResult:
    layer_potential_mev: np.ndarray
    interlayer_slope_mev: float
    delta_layer_charge: np.ndarray
    moire_cell_area_nm2: float


@dataclass(frozen=True)
class ScreeningIteration:
    iteration: int
    screened_u_mev: float
    interlayer_hartree_mev: float
    candidate_u_mev: float
    residual_mev: float


@dataclass(frozen=True)
class ScreenedInterlayerPotentialResult:
    external_v_mev: float
    screened_u_mev: float
    converged: bool
    iterations: tuple[ScreeningIteration, ...]
    layer_charge: LayerChargeResult
    hartree: InterlayerHartreeResult


def moire_cell_area_nm2(model: RLGhBNModel) -> float:
    a1 = complex(model.lattice.real_space_a1)
    a2 = complex(model.lattice.real_space_a2)
    return float(abs(a1.real * a2.imag - a1.imag * a2.real))


def _basis_layer_indices(lattice_n_g: int, params: RLGhBNParams, layer: int) -> np.ndarray:
    indices: list[int] = []
    for g_index in range(int(lattice_n_g)):
        for sublattice in range(2):
            indices.append((g_index * params.layer_count + int(layer)) * 2 + sublattice)
    return np.asarray(indices, dtype=int)


def compute_valence_layer_charge(
    model: RLGhBNModel,
    *,
    mesh_size: int,
    valleys: tuple[int, ...] = (1, -1),
    n_spin: int = 2,
    n_valence_bands: int | None = None,
) -> LayerChargeResult:
    if int(mesh_size) <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    if int(n_spin) <= 0:
        raise ValueError(f"n_spin must be positive, got {n_spin}")
    resolved_n_valence = (
        valence_band_count(model.lattice, model.params) if n_valence_bands is None else int(n_valence_bands)
    )
    layer_charge_per_spin = np.zeros(model.params.layer_count, dtype=float)
    layer_indices = tuple(
        _basis_layer_indices(model.lattice.n_g, model.params, layer) for layer in range(model.params.layer_count)
    )

    for valley in valleys:
        grid = model.bands_on_grid(
            int(mesh_size),
            valley=int(valley),
            return_eigenvectors=True,
            n_bands=None,
            endpoint=False,
        )
        if grid.eigenvectors is None:
            raise RuntimeError("Grid eigenvectors were not returned")
        norm_k = float(grid.eigenvectors.shape[0] * grid.eigenvectors.shape[1])
        for ix in range(grid.eigenvectors.shape[0]):
            for iy in range(grid.eigenvectors.shape[1]):
                occupied = grid.eigenvectors[ix, iy, :, :resolved_n_valence]
                weights = np.abs(occupied) ** 2
                for layer, indices in enumerate(layer_indices):
                    layer_charge_per_spin[layer] += float(np.sum(weights[indices, :])) / norm_k

    layer_charge = float(n_spin) * layer_charge_per_spin
    reference_layer_charge = np.full(
        model.params.layer_count,
        float(n_spin * len(valleys) * model.lattice.n_g),
        dtype=float,
    )
    return LayerChargeResult(
        layer_charge=layer_charge,
        reference_layer_charge=reference_layer_charge,
        delta_layer_charge=layer_charge - reference_layer_charge,
        mesh_size=int(mesh_size),
        n_spin=int(n_spin),
        valleys=tuple(int(valley) for valley in valleys),
        n_valence_bands=int(resolved_n_valence),
    )


def interlayer_hartree_potential_from_charge(
    delta_layer_charge: np.ndarray,
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    layer_spacing_nm: float = DEFAULT_LAYER_SPACING_NM,
) -> InterlayerHartreeResult:
    delta = np.asarray(delta_layer_charge, dtype=float)
    if delta.shape != (model.params.layer_count,):
        raise ValueError(f"Expected delta_layer_charge shape {(model.params.layer_count,)}, got {delta.shape}")
    z = layer_z_coordinates_nm(model.params.layer_count, layer_spacing_nm=layer_spacing_nm)
    q0_matrix = np.zeros((model.params.layer_count, model.params.layer_count), dtype=float)
    for il, z_l in enumerate(z):
        for jl, z_j in enumerate(z):
            q0_matrix[il, jl] = q0_interlayer_hartree_mev_nm2(
                float(z_l),
                float(z_j),
                epsilon_r=interaction.epsilon_r,
            )
    area = moire_cell_area_nm2(model)
    layer_potential = q0_matrix @ delta / area
    if model.params.layer_count <= 1:
        slope = 0.0
    else:
        slope = float((layer_potential[-1] - layer_potential[0]) / float(model.params.layer_count - 1))
    return InterlayerHartreeResult(
        layer_potential_mev=layer_potential,
        interlayer_slope_mev=slope,
        delta_layer_charge=delta.copy(),
        moire_cell_area_nm2=area,
    )


def solve_screened_interlayer_potential(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    mesh_size: int | None = None,
    max_iter: int = 50,
    tolerance_mev: float = 1.0e-6,
    mixing: float = 0.5,
) -> ScreenedInterlayerPotentialResult:
    if int(max_iter) <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if not 0.0 < float(mixing) <= 1.0:
        raise ValueError(f"mixing must be in (0, 1], got {mixing}")
    if float(tolerance_mev) < 0.0:
        raise ValueError(f"tolerance_mev must be nonnegative, got {tolerance_mev}")

    external_v = float(model.params.displacement_field_mev)
    screened_u = external_v
    iterations: list[ScreeningIteration] = []
    layer_charge: LayerChargeResult | None = None
    hartree: InterlayerHartreeResult | None = None
    converged = False
    resolved_mesh = interaction.k_mesh_size if mesh_size is None else int(mesh_size)

    for iteration in range(int(max_iter)):
        screened_params = replace(model.params, displacement_field_mev=screened_u)
        screened_model = RLGhBNModel(lattice=model.lattice, params=screened_params)
        layer_charge = compute_valence_layer_charge(screened_model, mesh_size=resolved_mesh)
        hartree = interlayer_hartree_potential_from_charge(layer_charge.delta_layer_charge, screened_model, interaction)
        candidate = external_v + hartree.interlayer_slope_mev
        residual = candidate - screened_u
        iterations.append(
            ScreeningIteration(
                iteration=int(iteration),
                screened_u_mev=float(screened_u),
                interlayer_hartree_mev=float(hartree.interlayer_slope_mev),
                candidate_u_mev=float(candidate),
                residual_mev=float(residual),
            )
        )
        if abs(residual) <= float(tolerance_mev):
            screened_u = candidate
            converged = True
            break
        screened_u = float((1.0 - mixing) * screened_u + mixing * candidate)

    if layer_charge is None or hartree is None:
        raise RuntimeError("Screening loop did not execute")
    return ScreenedInterlayerPotentialResult(
        external_v_mev=external_v,
        screened_u_mev=float(screened_u),
        converged=bool(converged),
        iterations=tuple(iterations),
        layer_charge=layer_charge,
        hartree=hartree,
    )


__all__ = [
    "InterlayerHartreeResult",
    "LayerChargeResult",
    "ScreenedInterlayerPotentialResult",
    "ScreeningIteration",
    "compute_valence_layer_charge",
    "interlayer_hartree_potential_from_charge",
    "moire_cell_area_nm2",
    "solve_screened_interlayer_potential",
]
