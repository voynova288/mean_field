from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import brentq

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
    residual_mev: float = float("nan")
    method: str = "fixed_point"
    mesh_size: int | None = None
    u_grid_min_mev: float | None = None
    u_grid_max_mev: float | None = None
    u_grid_points: int | None = None
    warning: str | None = None


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
    raw_layer_potential = q0_matrix @ delta / area
    # Appendix B5 reduces the layer-dependent q=0 Hartree potential to the
    # single adjacent-layer field U used by H_D.  With the centered layer
    # convention in dirac_block(), this scalar projection carries a 2/L factor.
    # The R5G V=48 meV screened-U checkpoints in the paper fix this
    # normalization; without it the q=0 Hartree field is too strong by about
    # L/2 and the prereq gate fails before any HF work starts.
    scalar_projection_scale = 2.0 / float(model.params.layer_count)
    layer_potential = scalar_projection_scale * raw_layer_potential
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
        residual_mev=float(iterations[-1].residual_mev),
        method="fixed_point",
        mesh_size=int(resolved_mesh),
    )


def solve_screened_interlayer_potential_grid(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    mesh_size: int | None = None,
    u_min_mev: float = -100.0,
    u_max_mev: float = 200.0,
    n_grid: int = 121,
    root_tolerance_mev: float = 1.0e-5,
) -> ScreenedInterlayerPotentialResult:
    """Solve the q=0 screened interlayer potential by grid bracketing.

    The root equation is ``external_V + interlayer_hartree_slope(U) - U = 0``.
    A sign-changing grid interval is refined with Brent's method.  If the grid
    does not bracket a root, the smallest-residual grid point is returned with
    ``converged=False`` and a warning for the caller to record before launching
    expensive HF work.
    """

    resolved_mesh = interaction.k_mesh_size if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    u_min = float(u_min_mev)
    u_max = float(u_max_mev)
    if not u_min < u_max:
        raise ValueError(f"u_min_mev must be smaller than u_max_mev, got {u_min_mev} >= {u_max_mev}")
    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError(f"n_grid must be at least 2, got {n_grid}")
    tolerance = float(root_tolerance_mev)
    if tolerance <= 0.0:
        raise ValueError(f"root_tolerance_mev must be positive, got {root_tolerance_mev}")

    external_v = float(model.params.displacement_field_mev)
    evaluation_cache: dict[float, tuple[float, LayerChargeResult, InterlayerHartreeResult]] = {}

    def evaluate(candidate_u: float) -> tuple[float, LayerChargeResult, InterlayerHartreeResult]:
        key = round(float(candidate_u), 12)
        cached = evaluation_cache.get(key)
        if cached is not None:
            return cached
        screened_params = replace(model.params, displacement_field_mev=float(candidate_u))
        screened_model = RLGhBNModel(lattice=model.lattice, params=screened_params)
        layer_charge = compute_valence_layer_charge(screened_model, mesh_size=resolved_mesh)
        hartree = interlayer_hartree_potential_from_charge(layer_charge.delta_layer_charge, screened_model, interaction)
        residual = float(external_v + hartree.interlayer_slope_mev - float(candidate_u))
        evaluation_cache[key] = (residual, layer_charge, hartree)
        return residual, layer_charge, hartree

    grid = np.linspace(u_min, u_max, n_grid, dtype=float)
    residuals = np.zeros(n_grid, dtype=float)
    iterations: list[ScreeningIteration] = []
    for index, u_value in enumerate(grid):
        residual, _charge, hartree = evaluate(float(u_value))
        residuals[index] = float(residual)
        iterations.append(
            ScreeningIteration(
                iteration=int(index),
                screened_u_mev=float(u_value),
                interlayer_hartree_mev=float(hartree.interlayer_slope_mev),
                candidate_u_mev=float(external_v + hartree.interlayer_slope_mev),
                residual_mev=float(residual),
            )
        )

    finite = np.isfinite(residuals)
    if not np.any(finite):
        raise RuntimeError("Screening grid produced no finite residuals")

    exact_indices = np.flatnonzero(np.abs(residuals) <= tolerance)
    bracket: tuple[float, float] | None = None
    if exact_indices.size:
        root_u = float(grid[int(exact_indices[0])])
        method = "grid_exact"
        converged = True
        warning = None
    else:
        for left in range(n_grid - 1):
            f_left = float(residuals[left])
            f_right = float(residuals[left + 1])
            if not (math.isfinite(f_left) and math.isfinite(f_right)):
                continue
            if f_left == 0.0:
                bracket = (float(grid[left]), float(grid[left]))
                break
            if f_left * f_right < 0.0:
                bracket = (float(grid[left]), float(grid[left + 1]))
                break
        if bracket is not None and bracket[0] != bracket[1]:
            root_u = float(
                brentq(
                    lambda u: evaluate(float(u))[0],
                    bracket[0],
                    bracket[1],
                    xtol=tolerance,
                    rtol=1.0e-12,
                    maxiter=100,
                )
            )
            method = "grid_brentq"
            converged = True
            warning = None
        elif bracket is not None:
            root_u = bracket[0]
            method = "grid_exact"
            converged = True
            warning = None
        else:
            best_index = int(np.nanargmin(np.abs(residuals)))
            root_u = float(grid[best_index])
            method = "grid_residual_min"
            converged = bool(abs(float(residuals[best_index])) <= tolerance)
            warning = (
                "screening grid did not bracket a root; returning the minimum-residual grid point"
            )

    final_residual, final_charge, final_hartree = evaluate(root_u)
    iterations.append(
        ScreeningIteration(
            iteration=len(iterations),
            screened_u_mev=float(root_u),
            interlayer_hartree_mev=float(final_hartree.interlayer_slope_mev),
            candidate_u_mev=float(external_v + final_hartree.interlayer_slope_mev),
            residual_mev=float(final_residual),
        )
    )
    if abs(float(final_residual)) > tolerance:
        converged = False
        if warning is None:
            warning = (
                "screening bracket was found but refined residual exceeds root_tolerance_mev"
            )

    return ScreenedInterlayerPotentialResult(
        external_v_mev=external_v,
        screened_u_mev=float(root_u),
        converged=bool(converged),
        iterations=tuple(iterations),
        layer_charge=final_charge,
        hartree=final_hartree,
        residual_mev=float(final_residual),
        method=method,
        mesh_size=int(resolved_mesh),
        u_grid_min_mev=float(u_min),
        u_grid_max_mev=float(u_max),
        u_grid_points=int(n_grid),
        warning=warning,
    )


def screening_result_to_dict(result: ScreenedInterlayerPotentialResult) -> dict[str, object]:
    return {
        "external_v_mev": float(result.external_v_mev),
        "screened_u_mev": float(result.screened_u_mev),
        "residual_mev": float(result.residual_mev),
        "converged": bool(result.converged),
        "method": str(result.method),
        "mesh_size": None if result.mesh_size is None else int(result.mesh_size),
        "u_grid_min_mev": None if result.u_grid_min_mev is None else float(result.u_grid_min_mev),
        "u_grid_max_mev": None if result.u_grid_max_mev is None else float(result.u_grid_max_mev),
        "u_grid_points": None if result.u_grid_points is None else int(result.u_grid_points),
        "warning": result.warning,
        "layer_charge": [float(value) for value in np.asarray(result.layer_charge.layer_charge, dtype=float)],
        "layer_charge_mesh_size": int(result.layer_charge.mesh_size),
        "layer_charge_n_spin": int(result.layer_charge.n_spin),
        "layer_charge_valleys": [int(value) for value in result.layer_charge.valleys],
        "layer_charge_n_valence_bands": int(result.layer_charge.n_valence_bands),
        "reference_layer_charge": [
            float(value) for value in np.asarray(result.layer_charge.reference_layer_charge, dtype=float)
        ],
        "delta_layer_charge": [
            float(value) for value in np.asarray(result.layer_charge.delta_layer_charge, dtype=float)
        ],
        "interlayer_hartree_mev": float(result.hartree.interlayer_slope_mev),
        "hartree_layer_potential_mev": [
            float(value) for value in np.asarray(result.hartree.layer_potential_mev, dtype=float)
        ],
        "moire_cell_area_nm2": float(result.hartree.moire_cell_area_nm2),
        "screening_iterations": [
            {
                "iteration": int(item.iteration),
                "screened_u_mev": float(item.screened_u_mev),
                "interlayer_hartree_mev": float(item.interlayer_hartree_mev),
                "candidate_u_mev": float(item.candidate_u_mev),
                "residual_mev": float(item.residual_mev),
            }
            for item in result.iterations
        ],
    }


def screening_result_from_dict(payload: dict[str, object]) -> ScreenedInterlayerPotentialResult:
    layer_charge = LayerChargeResult(
        layer_charge=np.asarray(payload["layer_charge"], dtype=float),
        reference_layer_charge=np.asarray(payload["reference_layer_charge"], dtype=float),
        delta_layer_charge=np.asarray(payload["delta_layer_charge"], dtype=float),
        mesh_size=int(payload.get("layer_charge_mesh_size") or payload.get("mesh_size") or 0),
        n_spin=int(payload.get("layer_charge_n_spin") or 2),
        valleys=tuple(int(value) for value in payload.get("layer_charge_valleys", (1, -1))),  # type: ignore[arg-type]
        n_valence_bands=int(payload.get("layer_charge_n_valence_bands") or 0),
    )
    hartree = InterlayerHartreeResult(
        layer_potential_mev=np.asarray(payload["hartree_layer_potential_mev"], dtype=float),
        interlayer_slope_mev=float(payload["interlayer_hartree_mev"]),
        delta_layer_charge=np.asarray(payload["delta_layer_charge"], dtype=float),
        moire_cell_area_nm2=float(payload["moire_cell_area_nm2"]),
    )
    raw_iterations = payload.get("screening_iterations", [])
    iterations: list[ScreeningIteration] = []
    if isinstance(raw_iterations, list):
        for raw in raw_iterations:
            if not isinstance(raw, dict):
                continue
            iterations.append(
                ScreeningIteration(
                    iteration=int(raw.get("iteration", len(iterations))),
                    screened_u_mev=float(raw.get("screened_u_mev", payload["screened_u_mev"])),
                    interlayer_hartree_mev=float(raw.get("interlayer_hartree_mev", payload["interlayer_hartree_mev"])),
                    candidate_u_mev=float(raw.get("candidate_u_mev", payload["screened_u_mev"])),
                    residual_mev=float(raw.get("residual_mev", payload.get("residual_mev", float("nan")))),
                )
            )
    if not iterations:
        iterations.append(
            ScreeningIteration(
                iteration=0,
                screened_u_mev=float(payload["screened_u_mev"]),
                interlayer_hartree_mev=float(payload["interlayer_hartree_mev"]),
                candidate_u_mev=float(payload["screened_u_mev"]) + float(payload.get("residual_mev", 0.0)),
                residual_mev=float(payload.get("residual_mev", float("nan"))),
            )
        )
    return ScreenedInterlayerPotentialResult(
        external_v_mev=float(payload["external_v_mev"]),
        screened_u_mev=float(payload["screened_u_mev"]),
        converged=bool(payload["converged"]),
        iterations=tuple(iterations),
        layer_charge=layer_charge,
        hartree=hartree,
        residual_mev=float(payload.get("residual_mev", iterations[-1].residual_mev)),
        method=str(payload.get("method", "unknown")),
        mesh_size=None if payload.get("mesh_size") is None else int(payload["mesh_size"]),
        u_grid_min_mev=None if payload.get("u_grid_min_mev") is None else float(payload["u_grid_min_mev"]),
        u_grid_max_mev=None if payload.get("u_grid_max_mev") is None else float(payload["u_grid_max_mev"]),
        u_grid_points=None if payload.get("u_grid_points") is None else int(payload["u_grid_points"]),
        warning=None if payload.get("warning") is None else str(payload["warning"]),
    )


__all__ = [
    "InterlayerHartreeResult",
    "LayerChargeResult",
    "ScreenedInterlayerPotentialResult",
    "ScreeningIteration",
    "compute_valence_layer_charge",
    "interlayer_hartree_potential_from_charge",
    "moire_cell_area_nm2",
    "screening_result_from_dict",
    "screening_result_to_dict",
    "solve_screened_interlayer_potential",
    "solve_screened_interlayer_potential_grid",
]
