from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from ...core.hf import real_space_cell_area_nm2_from_reciprocal
from .bands import GridBandsResult
from .lattice import TDBGLattice
from .projected_hf import (
    TDBGPaperUdConvention,
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedHFData,
    TDBGProjectedHFInitializer,
    TDBGProjectedHFInteractionBuilder,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    TDBGProjectedHFTargetData,
    TDBGProjectedWindow,
    TDBGStateLabel,
    build_tdbg_hf_target_hamiltonian,
    build_tdbg_interaction_builder,
    build_tdbg_interaction_components,
    build_tdbg_projected_hf_data,
    build_tdbg_projected_hf_kernel,
    build_tdbg_projected_hf_problem,
    build_tdbg_projected_hf_state,
    build_tdbg_projected_hf_target_data,
    build_tdbg_total_overlap_blocks,
    diagonalize_tdbg_hf_target_hamiltonian,
    initialize_tdbg_density,
    initialize_tdbg_nu2_density,
    liu2022_default_projected_hf_config,
    liu2022_projected_hf_metadata,
    run_tdbg_projected_hf,
    scan_tdbg_projected_hf_states,
    tdbg_density_from_hamiltonian,
    tdbg_delta_from_paper_ud_for_valley,
    tdbg_embedded_component_groups,
    tdbg_energy_components,
    tdbg_moire_area_nm2,
    tdbg_order_parameters,
    tdbg_projected_hf_result_to_hf_run_result,
    tdbg_parameters_from_paper_ud_for_valley,
    validate_tdbg_interaction_settings,
    validate_tdbg_projected_hf_config,
)
from .topology import TopologyResult, boundary_sewing_transforms, compute_topology_from_grid_result, translation_srcmap


LAYER_SUBLATTICE_LABELS: tuple[str, str, str, str] = ("A1", "B1", "A2", "B2")
VALLEY_LABELS: dict[int, str] = {1: "K", -1: "Kprime"}


@dataclass(frozen=True)
class TDBGActiveBandFlavorData:
    """Single-valley active-band data used by lightweight flavor-HF scouts."""

    valley: int
    band_index: int
    mean_energy_ev: float
    layer_sublattice_weights: np.ndarray
    topology: TopologyResult | None = None

    @property
    def valley_label(self) -> str:
        return VALLEY_LABELS.get(int(self.valley), f"valley{self.valley}")

    @property
    def rounded_chern_number(self) -> int | None:
        if self.topology is None:
            return None
        return int(self.topology.rounded_chern_number)


@dataclass(frozen=True)
class TDBGOnsiteHubbardFormFactorSummary:
    """Reciprocal-density form-factor norm for a single isolated active band."""

    g_shells: int
    included_shift_count: int
    q0_norm: float
    form_factor_norm: float
    max_valid_fraction: float
    min_nonzero_valid_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "g_shells": int(self.g_shells),
            "included_shift_count": int(self.included_shift_count),
            "q0_norm": float(self.q0_norm),
            "form_factor_norm": float(self.form_factor_norm),
            "ratio_to_q0_norm": float(self.form_factor_norm / self.q0_norm) if self.q0_norm > 0.0 else float("nan"),
            "max_valid_fraction": float(self.max_valid_fraction),
            "min_nonzero_valid_fraction": float(self.min_nonzero_valid_fraction),
        }


@dataclass(frozen=True)
class TDBGFlavorCandidateEnergy:
    """Energy bookkeeping for a fixed flavor projector, not a self-consistent HF state."""

    label: str
    occupied_flavors: tuple[tuple[int, str], ...]
    noninteracting_energy_ev: float
    onsite_hubbard_raw_ev: float
    onsite_hubbard_atomic_area_ev: float
    onsite_hubbard_form_factor_ev: float
    total_raw_ev: float
    total_atomic_area_ev: float
    total_form_factor_ev: float
    total_chern: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "occupied_flavors": [
                {"valley": VALLEY_LABELS.get(int(valley), f"valley{valley}"), "spin": str(spin)}
                for valley, spin in self.occupied_flavors
            ],
            "noninteracting_energy_ev_per_moire_cell_proxy": float(self.noninteracting_energy_ev),
            "onsite_hubbard_proxy_ev_per_moire_cell_raw": float(self.onsite_hubbard_raw_ev),
            "onsite_hubbard_proxy_ev_per_moire_cell_atomic_area_scaled": float(self.onsite_hubbard_atomic_area_ev),
            "onsite_hubbard_proxy_ev_per_moire_cell_form_factor_scaled": float(self.onsite_hubbard_form_factor_ev),
            "total_energy_proxy_ev_per_moire_cell_raw": float(self.total_raw_ev),
            "total_energy_proxy_ev_per_moire_cell_atomic_area_scaled": float(self.total_atomic_area_ev),
            "total_energy_proxy_ev_per_moire_cell_form_factor_scaled": float(self.total_form_factor_ev),
            "total_chern_from_active_band_cherns": None if self.total_chern is None else float(self.total_chern),
        }


def graphene_unit_cell_area_nm2(graphene_lattice_constant_nm: float) -> float:
    """Return the two-atom graphene unit-cell area for lattice constant ``a``."""

    a_nm = float(graphene_lattice_constant_nm)
    if a_nm <= 0.0:
        raise ValueError(f"graphene_lattice_constant_nm must be positive, got {graphene_lattice_constant_nm}")
    return float(math.sqrt(3.0) * a_nm * a_nm / 2.0)


def moire_cell_area_nm2(lattice: TDBGLattice) -> float:
    """Return the moire real-space unit-cell area from the TDBG reciprocal vectors."""

    return real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)


def onsite_hubbard_atomic_area_scale(lattice: TDBGLattice) -> float:
    """Estimate the continuum onsite-Hubbard scale ``A_graphene / A_moire``.

    This is a dimensional normalization diagnostic for projecting an atomic onsite
    term into moire-normalized continuum wavefunctions.  It is not a substitute
    for a full paper-specific Hubbard projection.
    """

    return float(graphene_unit_cell_area_nm2(lattice.graphene_lattice_constant_nm) / moire_cell_area_nm2(lattice))


def layer_sublattice_weights(active_eigenvectors: np.ndarray) -> np.ndarray:
    """Average ``|u|^2`` over k and q, retaining TDBG's four local components.

    The TDBG basis order is ``q``-major with local entries ``(A1, B1, A2, B2)``.
    The returned four-component vector is normalized to sum to one for one
    filled active flavor.
    """

    array = np.asarray(active_eigenvectors, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError(f"Expected active eigenvectors (mesh, mesh, basis), got {array.shape}")
    if array.shape[-1] % 4 != 0:
        raise ValueError(f"TDBG basis dimension must be divisible by four, got {array.shape[-1]}")

    weights = np.zeros(4, dtype=float)
    for alpha in range(4):
        weights[alpha] = float(np.mean(np.sum(np.abs(array[:, :, alpha::4]) ** 2, axis=-1)))
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Active eigenvector weights have zero norm.")
    return weights / total


def weights_to_label_dict(weights: Iterable[float]) -> dict[str, float]:
    values = np.asarray(tuple(weights), dtype=float)
    if values.shape != (4,):
        raise ValueError(f"Expected four layer/sublattice weights, got shape {values.shape}")
    return {label: float(value) for label, value in zip(LAYER_SUBLATTICE_LABELS, values, strict=True)}


def single_band_onsite_hubbard_proxy_ev(
    up_weights: np.ndarray,
    down_weights: np.ndarray,
    *,
    hubbard_u_ev: float,
    scale: float = 1.0,
) -> float:
    """Return ``U * scale * sum_alpha n_up(alpha) n_down(alpha)``.

    ``scale=1`` is the raw layer/sublattice-density proxy.  ``scale`` may be set
    to ``onsite_hubbard_atomic_area_scale(lattice)`` for a dimensional atomic-area
    estimate, but a final Liu-2022 reproduction still needs the full projected
    Hubbard normalization documented in the SI.
    """

    up = np.asarray(up_weights, dtype=float)
    down = np.asarray(down_weights, dtype=float)
    if up.shape != (4,) or down.shape != (4,):
        raise ValueError(f"Expected four-component weights, got {up.shape} and {down.shape}")
    return float(float(hubbard_u_ev) * float(scale) * np.dot(up, down))


def single_band_density_form_factor_summary(
    active_eigenvectors: np.ndarray,
    lattice: TDBGLattice,
    *,
    g_shells: int | None = None,
) -> TDBGOnsiteHubbardFormFactorSummary:
    """Compute ``sum_G,alpha |rho_alpha(G)|^2`` for one filled active band.

    The q=0 part is the simple layer/sublattice weight norm used by the first
    scout.  Nonzero moire reciprocal shifts add local-density Fourier components
    available within the finite q-site basis.  This is still a single-band
    diagnostic, not a full multi-band self-consistent HF projection.
    """

    array = np.asarray(active_eigenvectors, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError(f"Expected active eigenvectors (mesh, mesh, basis), got {array.shape}")
    if array.shape[-1] != 4 * lattice.n_q:
        raise ValueError(f"Expected basis dimension {4 * lattice.n_q}, got {array.shape[-1]}")
    resolved_shells = int(math.ceil(2.0 * lattice.cut) + 1) if g_shells is None else int(g_shells)
    if resolved_shells < 0:
        raise ValueError("g_shells must be non-negative")

    local = array.reshape(array.shape[0], array.shape[1], lattice.n_q, 4)
    total = 0.0
    q0_norm = float(np.dot(layer_sublattice_weights(array), layer_sublattice_weights(array)))
    valid_fractions: list[float] = []
    included = 0
    for m in range(-resolved_shells, resolved_shells + 1):
        for n in range(-resolved_shells, resolved_shells + 1):
            shift = m * lattice.g_m1 + n * lattice.g_m2
            src = translation_srcmap(lattice, shift)
            valid = src >= 0
            valid_fraction = float(np.mean(valid))
            if not np.any(valid):
                continue
            rho = np.mean(
                np.sum(np.conj(local[:, :, valid, :]) * local[:, :, src[valid], :], axis=2),
                axis=(0, 1),
            )
            total += float(np.sum(np.abs(rho) ** 2))
            valid_fractions.append(valid_fraction)
            included += 1

    return TDBGOnsiteHubbardFormFactorSummary(
        g_shells=resolved_shells,
        included_shift_count=included,
        q0_norm=q0_norm,
        form_factor_norm=float(total),
        max_valid_fraction=max(valid_fractions) if valid_fractions else 0.0,
        min_nonzero_valid_fraction=min(valid_fractions) if valid_fractions else 0.0,
    )


def active_band_flavor_data_from_grid(
    grid_result: GridBandsResult,
    *,
    lattice: TDBGLattice,
    valley: int,
    band_index: int,
    compute_topology: bool = True,
    boundary_sewing: bool = True,
) -> TDBGActiveBandFlavorData:
    """Build reusable data for one valley's active band from a grid result."""

    energies = np.asarray(grid_result.energies, dtype=float)
    if energies.ndim != 3:
        raise ValueError(f"Expected grid energies (mesh, mesh, band), got {energies.shape}")
    band = int(band_index)
    if band < 0 or band >= energies.shape[-1]:
        raise ValueError(f"band_index={band} outside available range [0, {energies.shape[-1]})")
    if grid_result.eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for active-band flavor data.")

    topology = None
    if compute_topology:
        topology = compute_topology_from_grid_result(
            grid_result,
            band,
            valley=int(valley),
            sewing_transforms=boundary_sewing_transforms(lattice) if boundary_sewing else None,
            metadata={"boundary_sewing": bool(boundary_sewing)},
        )

    return TDBGActiveBandFlavorData(
        valley=int(valley),
        band_index=band,
        mean_energy_ev=float(np.mean(energies[:, :, band])),
        layer_sublattice_weights=layer_sublattice_weights(grid_result.eigenvectors[:, :, :, band]),
        topology=topology,
    )


def _chern_sum(*items: TDBGActiveBandFlavorData) -> float | None:
    total = 0.0
    for item in items:
        rounded = item.rounded_chern_number
        if rounded is None:
            return None
        total += float(rounded)
    return total


def evaluate_nu2_sp_vp_single_band_candidates(
    flavor_data_by_valley: Mapping[int, TDBGActiveBandFlavorData],
    *,
    lattice: TDBGLattice,
    hubbard_u_ev: float,
    atomic_area_scale: float | None = None,
    form_factor_norm_by_valley: Mapping[int, float] | None = None,
) -> tuple[TDBGFlavorCandidateEnergy, ...]:
    """Evaluate fixed SP/VP flavor projectors for the Liu-2022 ``nu=2`` scout.

    This function deliberately does not run self-consistent HF.  It compares
    explicit projectors in the isolated active conduction band and reports both
    raw and atomic-area-scaled onsite-Hubbard proxies.
    """

    if 1 not in flavor_data_by_valley or -1 not in flavor_data_by_valley:
        raise ValueError("flavor_data_by_valley must contain valleys +1 and -1")
    k = flavor_data_by_valley[1]
    kp = flavor_data_by_valley[-1]
    scale = onsite_hubbard_atomic_area_scale(lattice) if atomic_area_scale is None else float(atomic_area_scale)

    def hub(raw_weights_a: np.ndarray, raw_weights_b: np.ndarray, *, local_scale: float) -> float:
        return single_band_onsite_hubbard_proxy_ev(
            raw_weights_a,
            raw_weights_b,
            hubbard_u_ev=hubbard_u_ev,
            scale=local_scale,
        )

    e0_sp = float(k.mean_energy_ev + kp.mean_energy_ev)
    e0_vp_k = float(2.0 * k.mean_energy_ev)
    e0_vp_kp = float(2.0 * kp.mean_energy_ev)
    zero = 0.0
    hub_vp_k_raw = hub(k.layer_sublattice_weights, k.layer_sublattice_weights, local_scale=1.0)
    hub_vp_kp_raw = hub(kp.layer_sublattice_weights, kp.layer_sublattice_weights, local_scale=1.0)
    hub_vp_k_scaled = hub(k.layer_sublattice_weights, k.layer_sublattice_weights, local_scale=scale)
    hub_vp_kp_scaled = hub(kp.layer_sublattice_weights, kp.layer_sublattice_weights, local_scale=scale)
    ff_norms = form_factor_norm_by_valley or {}
    hub_vp_k_ff = float(hubbard_u_ev) * scale * float(ff_norms.get(1, np.dot(k.layer_sublattice_weights, k.layer_sublattice_weights)))
    hub_vp_kp_ff = float(hubbard_u_ev) * scale * float(ff_norms.get(-1, np.dot(kp.layer_sublattice_weights, kp.layer_sublattice_weights)))

    return (
        TDBGFlavorCandidateEnergy(
            label="SP_KKprime_up",
            occupied_flavors=((1, "up"), (-1, "up")),
            noninteracting_energy_ev=e0_sp,
            onsite_hubbard_raw_ev=zero,
            onsite_hubbard_atomic_area_ev=zero,
            onsite_hubbard_form_factor_ev=zero,
            total_raw_ev=e0_sp,
            total_atomic_area_ev=e0_sp,
            total_form_factor_ev=e0_sp,
            total_chern=_chern_sum(k, kp),
        ),
        TDBGFlavorCandidateEnergy(
            label="VP_K_both_spins",
            occupied_flavors=((1, "up"), (1, "down")),
            noninteracting_energy_ev=e0_vp_k,
            onsite_hubbard_raw_ev=hub_vp_k_raw,
            onsite_hubbard_atomic_area_ev=hub_vp_k_scaled,
            onsite_hubbard_form_factor_ev=hub_vp_k_ff,
            total_raw_ev=e0_vp_k + hub_vp_k_raw,
            total_atomic_area_ev=e0_vp_k + hub_vp_k_scaled,
            total_form_factor_ev=e0_vp_k + hub_vp_k_ff,
            total_chern=None if k.rounded_chern_number is None else float(2 * k.rounded_chern_number),
        ),
        TDBGFlavorCandidateEnergy(
            label="VP_Kprime_both_spins",
            occupied_flavors=((-1, "up"), (-1, "down")),
            noninteracting_energy_ev=e0_vp_kp,
            onsite_hubbard_raw_ev=hub_vp_kp_raw,
            onsite_hubbard_atomic_area_ev=hub_vp_kp_scaled,
            onsite_hubbard_form_factor_ev=hub_vp_kp_ff,
            total_raw_ev=e0_vp_kp + hub_vp_kp_raw,
            total_atomic_area_ev=e0_vp_kp + hub_vp_kp_scaled,
            total_form_factor_ev=e0_vp_kp + hub_vp_kp_ff,
            total_chern=None if kp.rounded_chern_number is None else float(2 * kp.rounded_chern_number),
        ),
    )

__all__ = [
    "TDBGActiveBandFlavorData",
    "TDBGFlavorCandidateEnergy",
    "TDBGInteractionSettings",
    "TDBGOnsiteHubbardFormFactorSummary",
    "TDBGPaperUdConvention",
    "TDBGProjectedHFConfig",
    "TDBGProjectedHFData",
    "TDBGProjectedHFInitializer",
    "TDBGProjectedHFInteractionBuilder",
    "TDBGProjectedHFResult",
    "TDBGProjectedHFState",
    "TDBGProjectedHFTargetData",
    "TDBGProjectedWindow",
    "TDBGStateLabel",
    "active_band_flavor_data_from_grid",
    "build_tdbg_hf_target_hamiltonian",
    "build_tdbg_interaction_builder",
    "build_tdbg_interaction_components",
    "build_tdbg_projected_hf_data",
    "build_tdbg_projected_hf_kernel",
    "build_tdbg_projected_hf_problem",
    "build_tdbg_projected_hf_state",
    "build_tdbg_projected_hf_target_data",
    "build_tdbg_total_overlap_blocks",
    "diagonalize_tdbg_hf_target_hamiltonian",
    "evaluate_nu2_sp_vp_single_band_candidates",
    "graphene_unit_cell_area_nm2",
    "initialize_tdbg_density",
    "initialize_tdbg_nu2_density",
    "layer_sublattice_weights",
    "liu2022_default_projected_hf_config",
    "liu2022_projected_hf_metadata",
    "moire_cell_area_nm2",
    "onsite_hubbard_atomic_area_scale",
    "run_tdbg_projected_hf",
    "scan_tdbg_projected_hf_states",
    "single_band_density_form_factor_summary",
    "single_band_onsite_hubbard_proxy_ev",
    "tdbg_density_from_hamiltonian",
    "tdbg_delta_from_paper_ud_for_valley",
    "tdbg_embedded_component_groups",
    "tdbg_energy_components",
    "tdbg_moire_area_nm2",
    "tdbg_order_parameters",
    "tdbg_parameters_from_paper_ud_for_valley",
    "tdbg_projected_hf_result_to_hf_run_result",
    "validate_tdbg_interaction_settings",
    "validate_tdbg_projected_hf_config",
    "weights_to_label_dict",
]
