from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
import os

import numpy as np

from ...core.lattice import KPath
from ...core.hf import (
    average_reference_density as core_average_reference_density,
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    HartreeFockStepResult,
    apply_random_projector_rotation,
    random_unitary_from_hermitian,
    ComponentGroup,
    component_group_indices,
    compute_hf_energy,
    compute_oda_parameter,
    ProjectedWavefunctionBasis,
    calculate_projected_overlap_between,
    compute_density_overlap_trace_from_diagonal,
    find_chemical_potential,
    occupied_state_mask,
    run_hartree_fock_problem,
    shift_wavefunction_grid,
)
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, valence_band_count
from .interaction import RLGhBNInteractionParams, VALID_INTERACTION_SCHEMES, layer_coulomb_matrix_mev_nm2
from .lattice import RLGhBNLattice, build_moire_k_grid
from .model import RLGhBNModel
from .screening import (
    ScreenedInterlayerPotentialResult,
    moire_cell_area_nm2,
    solve_screened_interlayer_potential,
    solve_screened_interlayer_potential_grid,
)

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - exercised on systems without numba.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False
else:
    _NUMBA_AVAILABLE = True


VALLEY_SEQUENCE = (1, -1)
RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION = "centered_cell_reciprocal_relabel_pad1_v2"
RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING = 1
RLG_HBN_FORM_FACTOR_CONVENTION_VERSION = "physical_q_plus_g_valley_signed_raw_shift_v2"


def rlg_hbn_layer_component_groups(layer_count: int) -> tuple[ComponentGroup, ...]:
    """Return RnG/hBN layer groups in the local sublattice-resolved basis.

    The core HF layer only knows named local-component subsets.  RnG/hBN owns
    the physical convention that each layer contributes the two local sublattice
    components ``[2*layer, 2*layer+1]`` inside every reciprocal-grid cell.
    """

    resolved_layer_count = int(layer_count)
    if resolved_layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    return tuple(
        ComponentGroup(f"layer_{layer}", np.asarray([2 * layer, 2 * layer + 1], dtype=int))
        for layer in range(resolved_layer_count)
    )


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _rlg_hbn_require_numba() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA", default=False)


def _rlg_hbn_use_numba() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_USE_NUMBA", default=True)


def _rlg_hbn_zero_literal_q0_fock() -> bool:
    return _env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False)


def _maybe_zero_literal_q0_fock_kernel(shift: tuple[int, int], fock_kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(fock_kernel)
    if not _rlg_hbn_zero_literal_q0_fock() or (int(shift[0]), int(shift[1])) != (0, 0):
        return kernel
    if kernel.ndim != 4:
        raise ValueError(f"Expected fock kernel shape (nk_target, nk_source, layer, layer), got {kernel.shape}")
    adjusted = np.array(kernel, copy=True)
    n_diag = min(adjusted.shape[0], adjusted.shape[1])
    for ik in range(n_diag):
        adjusted[ik, ik, :, :] = 0.0
    return adjusted


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _contract_layer_fock_term_numba_kernel(
        left_overlap: np.ndarray,
        density_delta: np.ndarray,
        coeff_matrix: np.ndarray,
        right_overlap: np.ndarray,
    ) -> np.ndarray:
        nt_target = left_overlap.shape[0]
        nk_target = left_overlap.shape[1]
        nt_source = left_overlap.shape[2]
        nk_source = left_overlap.shape[3]
        out = np.empty((nt_target, nt_target, nk_target), dtype=np.complex128)
        output_pairs = nt_target * nt_target
        total_terms = nk_target * output_pairs
        for linear_index in prange(total_terms):
            ik_target = linear_index // output_pairs
            pair_index = linear_index - ik_target * output_pairs
            a = pair_index // nt_target
            b = pair_index - a * nt_target
            total = 0.0 + 0.0j
            for ik_source in range(nk_source):
                coeff = coeff_matrix[ik_target, ik_source]
                if coeff == 0.0:
                    continue
                source_total = 0.0 + 0.0j
                for d in range(nt_source):
                    left_density = 0.0 + 0.0j
                    for c in range(nt_source):
                        left_density += (
                            left_overlap[a, ik_target, c, ik_source]
                            * density_delta[d, c, ik_source]
                        )
                    source_total += left_density * np.conj(right_overlap[b, ik_target, d, ik_source])
                total += coeff * source_total
            out[a, b, ik_target] = total
        return out

else:

    def _contract_layer_fock_term_numba_kernel(
        left_overlap: np.ndarray,
        density_delta: np.ndarray,
        coeff_matrix: np.ndarray,
        right_overlap: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available for RnG/hBN Fock contraction")


@dataclass(frozen=True)
class RLGhBNProjectedBasisData:
    model: RLGhBNModel
    basis_model: RLGhBNModel
    interaction: RLGhBNInteractionParams
    screening: ScreenedInterlayerPotentialResult | None
    mesh_size: int
    kvec: np.ndarray
    k_grid_frac: np.ndarray
    basis: ProjectedWavefunctionBasis
    h0: np.ndarray
    band_energies: np.ndarray
    active_band_indices: tuple[int, ...]
    flat_band_indices: tuple[int, int]
    valleys: tuple[int, ...]
    reciprocal_grid_shape: tuple[int, int]
    reciprocal_grid_origin: tuple[int, int]
    moire_cell_area_nm2: float
    physical_h0: np.ndarray | None = None
    fixed_remote_hamiltonian: np.ndarray | None = None

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def n_band(self) -> int:
        return int(self.basis.n_band)

    @property
    def screened_u_mev(self) -> float:
        return float(self.basis_model.params.displacement_field_mev)

    @property
    def v0(self) -> float:
        return 1.0 / float(self.moire_cell_area_nm2)


@dataclass(frozen=True)
class RLGhBNLayerOverlapBlockSet:
    shifts: tuple[tuple[int, int], ...]
    gvecs: np.ndarray
    layer_overlaps: dict[tuple[int, int], np.ndarray]
    layer_diagonal_overlaps: dict[tuple[int, int], np.ndarray]
    hartree_layer_coulomb: dict[tuple[int, int], np.ndarray]
    fock_layer_coulomb: dict[tuple[int, int], np.ndarray]


@dataclass(frozen=True)
class RLGhBNInteractionComponents:
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray


@dataclass
class RLGhBNHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    reference_density: np.ndarray
    nu: float
    v0: float
    active_valence_bands: int
    scheme: str
    mu: float = float("nan")
    precision: float = 1.0e-6
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2
    occupation_counts: tuple[int, ...] | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_projected_basis(
        cls,
        basis_data: RLGhBNProjectedBasisData,
        *,
        nu: float,
        precision: float = 1.0e-6,
        occupation_counts: tuple[int, ...] | None = None,
    ) -> "RLGhBNHartreeFockState":
        h0 = np.asarray(basis_data.h0, dtype=np.complex128).copy()
        nt, _, nk = h0.shape
        reference = rlg_hbn_reference_density(
            nt,
            nk,
            scheme=basis_data.interaction.scheme,
            active_valence_bands=basis_data.interaction.active_valence_bands,
            n_spin=basis_data.basis.n_spin,
            n_eta=basis_data.basis.n_flavor,
        )
        return cls(
            h0=h0,
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            reference_density=reference,
            nu=float(nu),
            v0=float(basis_data.v0),
            active_valence_bands=int(basis_data.interaction.active_valence_bands),
            scheme=str(basis_data.interaction.scheme),
            precision=float(precision),
            n_spin=int(basis_data.basis.n_spin),
            n_eta=int(basis_data.basis.n_flavor),
            n_band=int(basis_data.basis.n_band),
            occupation_counts=occupation_counts,
        )


@dataclass(frozen=True)
class RLGhBNHartreeFockRun(HartreeFockRun):
    state: RLGhBNHartreeFockState
    overlap_blocks: RLGhBNLayerOverlapBlockSet
    basis_data: RLGhBNProjectedBasisData


@dataclass(frozen=True)
class RLGhBNHFPathResult:
    path: KPath
    basis_data: RLGhBNProjectedBasisData
    hamiltonian: np.ndarray
    energies: np.ndarray


@dataclass(frozen=True)
class _RLGhBNRemoteAverageSource:
    basis_data: RLGhBNProjectedBasisData
    weights: np.ndarray


@dataclass(frozen=True)
class RLGhBNGroundStateScan:
    runs: tuple[RLGhBNHartreeFockRun, ...]

    @property
    def best_run(self) -> RLGhBNHartreeFockRun:
        if not self.runs:
            raise ValueError("No RLG/hBN HF runs are available")
        return min(self.runs, key=lambda run: float(run.state.diagnostics.get("hf_energy", np.inf)))


def active_band_indices_for_interaction(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
) -> tuple[int, ...]:
    n_valence = int(interaction.active_valence_bands)
    n_conduction = int(interaction.active_conduction_bands)
    center = valence_band_count(model.lattice, model.params)
    start = center - n_valence
    stop = center + n_conduction
    if start < 0 or stop > model.matrix_dim:
        raise ValueError(
            "Active RLG/hBN band window is outside the single-particle spectrum: "
            f"requested {n_valence} valence and {n_conduction} conduction bands around "
            f"center={center}, but matrix_dim={model.matrix_dim}."
        )
    return tuple(range(start, stop))


def rlg_hbn_average_reference_density(nt: int, nk: int) -> np.ndarray:
    return core_average_reference_density(int(nt), int(nk), value=0.5)


def _infer_rlg_hbn_band_count(nt: int, *, n_spin: int = 2, n_eta: int = 2) -> int:
    n_flavor = int(n_spin) * int(n_eta)
    if int(nt) % n_flavor != 0:
        raise ValueError(f"Projected dimension nt={nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}")
    n_band = int(nt) // n_flavor
    if n_band <= 0:
        raise ValueError(f"Projected band count must be positive, got {n_band}")
    return n_band


def _rlg_hbn_reference_density_diagonal(
    nt: int,
    nk: int,
    *,
    scheme: str,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    if scheme not in VALID_INTERACTION_SCHEMES:
        raise ValueError(f"scheme must be one of {VALID_INTERACTION_SCHEMES}, got {scheme!r}")
    n_band = _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    n_valence = int(active_valence_bands)
    if n_valence < 0 or n_valence > n_band:
        raise ValueError(f"active_valence_bands must lie in [0, {n_band}], got {active_valence_bands}")

    diagonal = np.zeros((int(nt), int(nk)), dtype=float)
    idx = np.arange(int(nt), dtype=int).reshape((int(n_spin), int(n_eta), n_band), order="F")
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            for iband in range(n_band):
                if scheme == "average":
                    value = 0.5
                else:
                    value = 1.0 if iband < n_valence else 0.0
                diagonal[int(idx[ispin, ieta, iband]), :] = value
    return diagonal


def rlg_hbn_reference_density(
    nt: int,
    nk: int,
    *,
    scheme: str = "average",
    active_valence_bands: int = 0,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    if scheme == "average":
        return rlg_hbn_average_reference_density(nt, nk)
    diagonal = _rlg_hbn_reference_density_diagonal(
        nt,
        nk,
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    reference = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    rows = np.arange(int(nt), dtype=int)
    for ik in range(int(nk)):
        reference[rows, rows, ik] = diagonal[:, ik]
    return reference


def average_scheme_density_delta(occupation_density: np.ndarray) -> np.ndarray:
    density = np.asarray(occupation_density, dtype=np.complex128)
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected occupation_density shape (nt, nt, nk), got {density.shape}")
    return density - rlg_hbn_average_reference_density(density.shape[0], density.shape[2])


def rlg_hbn_density_delta(
    occupation_density: np.ndarray,
    *,
    scheme: str,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    density = np.asarray(occupation_density, dtype=np.complex128)
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected occupation_density shape (nt, nt, nk), got {density.shape}")
    return density - rlg_hbn_reference_density(
        density.shape[0],
        density.shape[2],
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )


def rlg_hbn_projector_from_density(density_delta: np.ndarray, reference_density: np.ndarray) -> np.ndarray:
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    if density_delta.shape != reference_density.shape:
        raise ValueError(f"Expected reference density shape {density_delta.shape}, got {reference_density.shape}")
    return density_delta + reference_density


def rlg_hbn_occupied_state_count(
    nu: float,
    nt: int,
    nk: int,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    raw = (float(int(n_spin) * int(n_eta) * int(active_valence_bands)) + float(nu)) * int(nk)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer occupied-state count {raw}")
    if rounded < 0 or rounded > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied-state count {rounded} outside [0, {int(nt) * int(nk)}]")
    return rounded


def rlg_hbn_occupied_bands_per_k(
    nu: float,
    nt: int,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    raw = float(int(n_spin) * int(n_eta) * int(active_valence_bands)) + float(nu)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer per-k occupation {raw}")
    if rounded < 0 or rounded > int(nt):
        raise ValueError(f"Filling nu={nu} gives per-k occupation {rounded} outside [0, {int(nt)}]")
    return rounded


def rlg_hbn_filling_from_density(
    density_delta: np.ndarray,
    reference_density: np.ndarray,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    projector = rlg_hbn_projector_from_density(density_delta, reference_density)
    total_particles = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    particles_per_k = total_particles / float(projector.shape[2])
    return float(particles_per_k - float(int(n_spin) * int(n_eta) * int(active_valence_bands)))


def _rectangular_g_embedding(
    lattice: RLGhBNLattice,
    *,
    padding: int = 0,
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    pad = int(padding)
    if pad < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    mins = np.min(lattice.g_indices, axis=0) - pad
    maxs = np.max(lattice.g_indices, axis=0) + pad
    grid_shape = (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1))
    origin = (int(mins[0]), int(mins[1]))
    positions = {
        (int(n1), int(n2)): (int(n1 - mins[0]), int(n2 - mins[1]))
        for n1 in range(int(mins[0]), int(maxs[0]) + 1)
        for n2 in range(int(mins[1]), int(maxs[1]) + 1)
    }
    return grid_shape, origin, positions


def _reciprocal_fractional_coordinates(k_tilde: complex, lattice: RLGhBNLattice) -> np.ndarray:
    reciprocal = np.asarray(
        [
            [float(lattice.g_m1.real), float(lattice.g_m2.real)],
            [float(lattice.g_m1.imag), float(lattice.g_m2.imag)],
        ],
        dtype=float,
    )
    vector = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    return np.linalg.solve(reciprocal, vector)


def _fold_k_to_centered_cell(k_tilde: complex, lattice: RLGhBNLattice) -> tuple[complex, tuple[int, int]]:
    fractional = _reciprocal_fractional_coordinates(k_tilde, lattice)
    shift = np.floor(fractional + 0.5).astype(int)
    k_can = complex(k_tilde - int(shift[0]) * lattice.g_m1 - int(shift[1]) * lattice.g_m2)
    return k_can, (int(shift[0]), int(shift[1]))


def _raw_pair_from_canonical_pair(
    canonical_pair: tuple[int, int] | np.ndarray,
    shift: tuple[int, int],
    *,
    valley: int,
) -> tuple[int, int]:
    pair = np.asarray(canonical_pair, dtype=int)
    sign = int(valley)
    if sign not in VALLEY_SEQUENCE:
        raise ValueError(f"Expected valley in {VALLEY_SEQUENCE}, got {valley}")
    return (
        int(pair[0] - sign * int(shift[0])),
        int(pair[1] - sign * int(shift[1])),
    )


def _raw_overlap_shift_for_physical_g(
    shift: tuple[int, int] | np.ndarray,
    *,
    valley: int,
) -> tuple[int, int]:
    """Return the raw reciprocal-grid shift implementing paper Eq. (18).

    In the embedded RLG/hBN basis the K valley uses raw labels equal to the
    physical reciprocal labels, while the K' valley is stored in the
    time-reversal relabelled convention ``G_raw = -G_phys``. For a physical
    Umklapp vector ``G = m g1 + n g2``, Eq. (18) requires
    ``target_raw = source_raw + valley * G``. The low-level grid shifter used
    by :func:`calculate_layer_projected_overlap_between` implements
    ``target_raw = source_raw - raw_shift``. Hence ``raw_shift = -valley * G``.
    """

    pair = np.asarray(shift, dtype=int).reshape(2)
    sign = int(valley)
    if sign not in VALLEY_SEQUENCE:
        raise ValueError(f"Expected valley in {VALLEY_SEQUENCE}, got {valley}")
    return (-sign * int(pair[0]), -sign * int(pair[1]))


def _screened_basis_model(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    screening_mesh_size: int | None,
    screening_max_iter: int,
    screening_tolerance_mev: float,
    screening_mixing: float,
    screening_solver: str = "fixed_point",
    screening_result: ScreenedInterlayerPotentialResult | None = None,
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    screening_root_tolerance_mev: float = 1.0e-5,
) -> tuple[RLGhBNModel, ScreenedInterlayerPotentialResult | None]:
    if not interaction.use_screened_basis:
        return model, None
    if screening_result is not None:
        screening = screening_result
    elif screening_solver == "grid":
        screening = solve_screened_interlayer_potential_grid(
            model,
            interaction,
            mesh_size=screening_mesh_size,
            u_min_mev=screening_u_min_mev,
            u_max_mev=screening_u_max_mev,
            n_grid=screening_u_grid_points,
            root_tolerance_mev=screening_root_tolerance_mev,
        )
    elif screening_solver == "fixed_point":
        screening = solve_screened_interlayer_potential(
            model,
            interaction,
            mesh_size=screening_mesh_size,
            max_iter=screening_max_iter,
            tolerance_mev=screening_tolerance_mev,
            mixing=screening_mixing,
        )
    else:
        raise ValueError(f"screening_solver must be 'grid' or 'fixed_point', got {screening_solver!r}")
    screened_params = replace(model.params, displacement_field_mev=screening.screened_u_mev)
    return RLGhBNModel(lattice=model.lattice, params=screened_params), screening


def _assert_average_remote_hamiltonian_contract(basis_data: RLGhBNProjectedBasisData) -> None:
    if basis_data.interaction.scheme != "average":
        return
    if basis_data.physical_h0 is None:
        raise AssertionError("average scheme requires physical_h0")
    if basis_data.fixed_remote_hamiltonian is None:
        raise AssertionError("average scheme requires fixed_remote_hamiltonian")
    expected = np.asarray(basis_data.physical_h0, dtype=np.complex128) + np.asarray(
        basis_data.fixed_remote_hamiltonian,
        dtype=np.complex128,
    )
    if not np.allclose(np.asarray(basis_data.h0, dtype=np.complex128), expected, atol=1.0e-9, rtol=1.0e-9):
        raise AssertionError("average scheme h0 must equal physical_h0 + fixed_remote_hamiltonian")


def _project_physical_hamiltonian(
    selected_basis: np.ndarray,
    *,
    k_tilde: complex,
    physical_model: RLGhBNModel,
    valley: int,
) -> np.ndarray:
    selected = np.asarray(selected_basis, dtype=np.complex128)
    hamiltonian = build_hamiltonian(
        complex(k_tilde),
        physical_model.lattice,
        physical_model.params,
        valley=int(valley),
    )
    projected = selected.conjugate().T @ hamiltonian @ selected
    return 0.5 * (projected + projected.conjugate().T)


def _build_projected_basis_for_indices(
    *,
    physical_model: RLGhBNModel,
    basis_model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    kvec: np.ndarray,
    band_indices: tuple[int, ...],
    valleys: tuple[int, ...],
    mesh_size: int,
    k_grid_frac: np.ndarray,
    screening: ScreenedInterlayerPotentialResult | None,
    name: str,
    build_h0: bool = True,
) -> RLGhBNProjectedBasisData:
    resolved_kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    resolved_indices = tuple(int(value) for value in band_indices)
    resolved_valleys = tuple(int(value) for value in valleys)
    if resolved_kvec.size == 0:
        raise ValueError("At least one k point is required")
    if not resolved_indices:
        raise ValueError("At least one band index is required")
    if not resolved_valleys:
        raise ValueError("At least one valley is required")
    if min(resolved_indices) < 0 or max(resolved_indices) >= basis_model.matrix_dim:
        raise ValueError(
            f"Band indices must lie in [0, {basis_model.matrix_dim}), got {resolved_indices}"
        )

    n_projected = len(resolved_indices)
    grid_shape, origin, positions = _rectangular_g_embedding(
        basis_model.lattice,
        padding=RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    )
    nx, ny = grid_shape
    local_basis_size = int(2 * basis_model.params.layer_count)
    embedded = np.zeros(
        (local_basis_size, nx, ny, n_projected, len(resolved_valleys), resolved_kvec.size),
        dtype=np.complex128,
    )
    band_energies = np.zeros((n_projected, len(resolved_valleys), resolved_kvec.size), dtype=float)
    physical_blocks = (
        np.zeros(
            (n_projected, n_projected, len(resolved_valleys), resolved_kvec.size),
            dtype=np.complex128,
        )
        if build_h0
        else None
    )

    index_array = np.asarray(resolved_indices, dtype=int)
    folded_k = tuple(_fold_k_to_centered_cell(complex(kval), basis_model.lattice) for kval in resolved_kvec)
    canonical_kvec = np.asarray([entry[0] for entry in folded_k], dtype=np.complex128)
    reciprocal_shifts = tuple(entry[1] for entry in folded_k)
    for iflavor, valley in enumerate(resolved_valleys):
        for ik, (k_can, reciprocal_shift) in enumerate(zip(canonical_kvec, reciprocal_shifts, strict=True)):
            evals, evecs = diagonalize_hamiltonian(
                complex(k_can),
                basis_model.lattice,
                basis_model.params,
                valley=int(valley),
            )
            selected_can = np.asarray(evecs[:, index_array], dtype=np.complex128)
            for source_g_index, pair in enumerate(basis_model.lattice.g_indices):
                raw_pair = _raw_pair_from_canonical_pair(
                    pair,
                    reciprocal_shift,
                    valley=int(valley),
                )
                if raw_pair not in positions:
                    raise ValueError(
                        "Periodic-gauge relabel moved a G component outside the embedded reciprocal grid: "
                        f"raw_pair={raw_pair}, shift={reciprocal_shift}, valley={valley}, "
                        f"origin={origin}, grid_shape={grid_shape}. Increase "
                        "RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING."
                    )
                ix, iy = positions[raw_pair]
                start = local_basis_size * source_g_index
                embedded[:, ix, iy, :, iflavor, ik] = selected_can[start : start + local_basis_size, :]
            band_energies[:, iflavor, ik] = np.asarray(evals[index_array], dtype=float)
            if physical_blocks is not None:
                physical_blocks[:, :, iflavor, ik] = _project_physical_hamiltonian(
                    selected_can,
                    k_tilde=complex(k_can),
                    physical_model=physical_model,
                    valley=int(valley),
                )

    wavefunction_array = embedded.reshape(
        (local_basis_size * nx * ny, n_projected, len(resolved_valleys), resolved_kvec.size),
        order="F",
    )
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=local_basis_size,
        name=name,
        component_groups=rlg_hbn_layer_component_groups(basis_model.params.layer_count),
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    idx = np.arange(basis.nt, dtype=int).reshape((basis.n_spin, basis.n_flavor, n_projected), order="F")
    if physical_blocks is not None:
        for ik in range(basis.nk):
            for ispin in range(basis.n_spin):
                for iflavor in range(basis.n_flavor):
                    block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                    h0[:, :, ik][np.ix_(block_indices, block_indices)] = physical_blocks[:, :, iflavor, ik]

    return RLGhBNProjectedBasisData(
        model=physical_model,
        basis_model=basis_model,
        interaction=interaction,
        screening=screening,
        mesh_size=int(mesh_size),
        kvec=resolved_kvec,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        band_energies=band_energies,
        active_band_indices=resolved_indices,
        flat_band_indices=basis_model.flat_band_indices,
        valleys=resolved_valleys,
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_cell_area_nm2=moire_cell_area_nm2(basis_model),
        physical_h0=h0.copy(),
        fixed_remote_hamiltonian=np.zeros_like(h0),
    )


def _remote_band_indices_and_average_weights(
    basis_model: RLGhBNModel,
    active_band_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], np.ndarray]:
    active = {int(value) for value in active_band_indices}
    valence_count = valence_band_count(basis_model.lattice, basis_model.params)
    remote_indices: list[int] = []
    weights: list[float] = []
    for band_index in range(basis_model.matrix_dim):
        if band_index in active:
            continue
        remote_indices.append(int(band_index))
        weights.append(0.5 if band_index < valence_count else -0.5)
    return tuple(remote_indices), np.asarray(weights, dtype=float)


def _remote_average_density_delta(remote_basis_data: RLGhBNProjectedBasisData, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != remote_basis_data.n_band:
        raise ValueError(
            f"Expected {remote_basis_data.n_band} remote weights, got {weights.size}"
        )
    density = np.zeros((remote_basis_data.nt, remote_basis_data.nt, remote_basis_data.nk), dtype=np.complex128)
    idx = np.arange(remote_basis_data.nt, dtype=int).reshape(
        (remote_basis_data.basis.n_spin, remote_basis_data.basis.n_flavor, remote_basis_data.n_band),
        order="F",
    )
    for ik in range(remote_basis_data.nk):
        for ispin in range(remote_basis_data.basis.n_spin):
            for iflavor in range(remote_basis_data.basis.n_flavor):
                density[idx[ispin, iflavor, :], idx[ispin, iflavor, :], ik] = weights
    return density


def _prepare_remote_average_source(
    source_basis_data: RLGhBNProjectedBasisData,
) -> _RLGhBNRemoteAverageSource | None:
    if source_basis_data.interaction.scheme != "average":
        return None
    remote_indices, remote_weights = _remote_band_indices_and_average_weights(
        source_basis_data.basis_model,
        source_basis_data.active_band_indices,
    )
    if not remote_indices:
        return None

    remote_basis_data = _build_projected_basis_for_indices(
        physical_model=source_basis_data.model,
        basis_model=source_basis_data.basis_model,
        interaction=source_basis_data.interaction,
        kvec=source_basis_data.kvec,
        band_indices=remote_indices,
        valleys=source_basis_data.valleys,
        mesh_size=source_basis_data.mesh_size,
        k_grid_frac=source_basis_data.k_grid_frac,
        screening=None,
        name="rlg_hbn_screened_remote",
        build_h0=False,
    )
    return _RLGhBNRemoteAverageSource(
        basis_data=remote_basis_data,
        weights=np.asarray(remote_weights, dtype=float),
    )


def _remote_average_chunk_size(n_band: int) -> int:
    raw = os.environ.get("MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS must be an integer, got {raw!r}") from exc
    else:
        value = 4
    return max(1, min(int(value), int(n_band)))


def _slice_projected_basis_data_bands(
    basis_data: RLGhBNProjectedBasisData,
    start: int,
    stop: int,
) -> RLGhBNProjectedBasisData:
    start = int(start)
    stop = int(stop)
    if start < 0 or stop <= start or stop > basis_data.n_band:
        raise ValueError(f"Invalid band slice [{start}, {stop}) for n_band={basis_data.n_band}")
    wavefunctions = np.asarray(basis_data.basis.wavefunctions[:, start:stop, :, :], dtype=np.complex128)
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunctions,
        grid_shape=basis_data.basis.grid_shape,
        n_spin=basis_data.basis.n_spin,
        local_basis_size=basis_data.basis.local_basis_size,
        name=f"{basis_data.basis.name}_bands_{start}_{stop}",
        component_groups=basis_data.basis.component_groups,
    )
    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    return replace(
        basis_data,
        basis=basis,
        h0=h0,
        band_energies=np.asarray(basis_data.band_energies[start:stop, :, :], dtype=float),
        active_band_indices=tuple(int(value) for value in basis_data.active_band_indices[start:stop]),
        physical_h0=None,
        fixed_remote_hamiltonian=None,
    )


def _layer_traces_for_diagonal_band_weights(
    basis: ProjectedWavefunctionBasis,
    weights: np.ndarray,
    m: int,
    n: int,
    *,
    layer_count: int,
    valleys: tuple[int, ...] | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != basis.n_band:
        raise ValueError(f"Expected {basis.n_band} band weights, got {weights.size}")
    layer_count = int(layer_count)
    if basis.local_basis_size != 2 * layer_count:
        raise ValueError(
            f"Expected local_basis_size={2 * layer_count} for {layer_count} layers, got {basis.local_basis_size}"
        )
    resolved_valleys = _resolve_basis_valleys(basis.n_flavor, valleys)

    nx, ny = basis.grid_shape
    band_k = basis.n_band * basis.nk
    band_k_weights = np.broadcast_to(weights[:, None], (basis.n_band, basis.nk)).reshape(-1, order="F")
    traces = np.zeros(layer_count, dtype=np.complex128)
    for iflavor, valley in enumerate(resolved_valleys):
        source_grid = basis.wavefunctions[:, :, iflavor, :].reshape(
            basis.local_basis_size,
            nx,
            ny,
            band_k,
            order="F",
        )
        raw_m, raw_n = _raw_overlap_shift_for_physical_g((m, n), valley=int(valley))
        shifted = shift_wavefunction_grid(source_grid, -raw_m, -raw_n, boundary_mode="zero_fill", grid_axes=(1, 2))
        for layer in range(layer_count):
            layer_indices = _rlg_hbn_layer_local_indices(basis, layer, layer_count=layer_count)
            diagonal = np.sum(
                np.conj(source_grid[layer_indices, :, :, :]) * shifted[layer_indices, :, :, :],
                axis=(0, 1, 2),
            )
            traces[layer] += basis.n_spin * np.sum(band_k_weights * np.conj(diagonal))
    return traces


def _remote_average_hamiltonian_from_source(
    target_basis_data: RLGhBNProjectedBasisData,
    source_basis_data: RLGhBNProjectedBasisData,
    remote_source: _RLGhBNRemoteAverageSource | None,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    if remote_source is None:
        return np.zeros_like(target_basis_data.h0)
    if source_basis_data.interaction.scheme != target_basis_data.interaction.scheme:
        raise ValueError(
            "Target/source interaction schemes differ: "
            f"{target_basis_data.interaction.scheme!r} != {source_basis_data.interaction.scheme!r}"
        )
    resolved_shifts = (
        shifts
        if shifts is not None
        else interaction_shifts_for_cutoff(source_basis_data.basis_model.lattice, source_basis_data.interaction)
    )
    resolved_shifts = tuple((int(m), int(n)) for m, n in resolved_shifts)
    gvecs = np.asarray(
        [
            m * source_basis_data.basis_model.lattice.g_m1 + n * source_basis_data.basis_model.lattice.g_m2
            for m, n in resolved_shifts
        ],
        dtype=np.complex128,
    )
    target_blocks = build_rlg_hbn_layer_overlap_blocks(target_basis_data, shifts=resolved_shifts)
    hamiltonian = np.zeros_like(target_basis_data.h0)
    remote_basis_data = remote_source.basis_data
    remote_weights = np.asarray(remote_source.weights, dtype=float).reshape(-1)
    if remote_weights.size != remote_basis_data.n_band:
        raise ValueError(f"Expected {remote_basis_data.n_band} remote weights, got {remote_weights.size}")

    nk_source = int(remote_basis_data.nk)
    scale = float(beta) * float(source_basis_data.v0) / float(nk_source)
    layer_count = int(source_basis_data.basis_model.params.layer_count)
    layer_spacing = float(source_basis_data.basis_model.params.layer_spacing_nm)
    chunk_size = _remote_average_chunk_size(remote_basis_data.n_band)

    for shift, gvec in zip(resolved_shifts, gvecs, strict=True):
        target_layer_diagonal = target_blocks.layer_diagonal_overlaps[shift]
        hartree_kernel = layer_coulomb_matrix_mev_nm2(
            abs(complex(gvec)),
            layer_count,
            source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )
        layer_traces = _layer_traces_for_diagonal_band_weights(
            remote_basis_data.basis,
            remote_weights,
            shift[0],
            shift[1],
            layer_count=layer_count,
            valleys=remote_basis_data.valleys,
        )
        for target_layer in range(layer_count):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hamiltonian += prefactor * target_layer_diagonal[target_layer]

        for start in range(0, remote_basis_data.n_band, chunk_size):
            stop = min(start + chunk_size, remote_basis_data.n_band)
            chunk_basis_data = _slice_projected_basis_data_bands(remote_basis_data, start, stop)
            chunk_density = _remote_average_density_delta(chunk_basis_data, remote_weights[start:stop])
            target_source_blocks = build_rlg_hbn_layer_overlap_blocks_between(
                target_basis_data,
                chunk_basis_data,
                shifts=(shift,),
            )
            target_source_layer_overlap = target_source_blocks.layer_overlaps[shift]
            fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, target_source_blocks.fock_layer_coulomb[shift])
            for target_layer in range(layer_count):
                for source_layer in range(layer_count):
                    coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                    if np.any(coeff != 0.0):
                        hamiltonian -= _contract_layer_fock_term(
                            target_source_layer_overlap[target_layer],
                            chunk_density,
                            coeff,
                            target_source_layer_overlap[source_layer],
                        )

    _hermitize_blocks_inplace(hamiltonian)
    return hamiltonian


def build_rlg_hbn_remote_average_hamiltonian(
    target_basis_data: RLGhBNProjectedBasisData,
    *,
    source_basis_data: RLGhBNProjectedBasisData | None = None,
    shifts: tuple[tuple[int, int], ...] | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    source_basis = target_basis_data if source_basis_data is None else source_basis_data
    remote_source = _prepare_remote_average_source(source_basis)
    return _remote_average_hamiltonian_from_source(
        target_basis_data,
        source_basis,
        remote_source,
        shifts=shifts,
        beta=beta,
    )


def build_rlg_hbn_projected_basis(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams | None = None,
    *,
    mesh_size: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    valleys: tuple[int, ...] = VALLEY_SEQUENCE,
    screening_mesh_size: int | None = None,
    screening_max_iter: int = 50,
    screening_tolerance_mev: float = 1.0e-6,
    screening_mixing: float = 0.5,
    screening_solver: str = "fixed_point",
    screening_result: ScreenedInterlayerPotentialResult | None = None,
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    screening_root_tolerance_mev: float = 1.0e-5,
) -> RLGhBNProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else RLGhBNInteractionParams()
    resolved_mesh = resolved_interaction.k_mesh_size if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    resolved_valleys = tuple(int(valley) for valley in valleys)
    if not resolved_valleys:
        raise ValueError("At least one valley is required")

    basis_model, screening = _screened_basis_model(
        model,
        resolved_interaction,
        screening_mesh_size=resolved_mesh if screening_mesh_size is None else int(screening_mesh_size),
        screening_max_iter=screening_max_iter,
        screening_tolerance_mev=screening_tolerance_mev,
        screening_mixing=screening_mixing,
        screening_solver=screening_solver,
        screening_result=screening_result,
        screening_u_min_mev=screening_u_min_mev,
        screening_u_max_mev=screening_u_max_mev,
        screening_u_grid_points=screening_u_grid_points,
        screening_root_tolerance_mev=screening_root_tolerance_mev,
    )
    k_grid_frac, kvec_grid = build_moire_k_grid(basis_model.lattice, resolved_mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid.reshape(-1), dtype=np.complex128)
    active_indices = active_band_indices_for_interaction(basis_model, resolved_interaction)
    basis_data = _build_projected_basis_for_indices(
        physical_model=model,
        basis_model=basis_model,
        interaction=resolved_interaction,
        kvec=kvec,
        band_indices=active_indices,
        valleys=resolved_valleys,
        mesh_size=int(resolved_mesh),
        k_grid_frac=np.asarray(k_grid_frac, dtype=float).reshape(-1, 2),
        screening=screening,
        name="rlg_hbn_screened_active",
    )
    fixed_remote = build_rlg_hbn_remote_average_hamiltonian(basis_data)
    completed = replace(
        basis_data,
        h0=np.asarray(basis_data.physical_h0, dtype=np.complex128) + fixed_remote,
        fixed_remote_hamiltonian=fixed_remote,
    )
    _assert_average_remote_hamiltonian_contract(completed)
    return completed


def build_rlg_hbn_projected_basis_for_kvec(
    basis_model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    kvec: np.ndarray,
    *,
    physical_model: RLGhBNModel | None = None,
    active_band_indices: tuple[int, ...] | np.ndarray | None = None,
    valleys: tuple[int, ...] = VALLEY_SEQUENCE,
) -> RLGhBNProjectedBasisData:
    resolved_kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    if resolved_kvec.size == 0:
        raise ValueError("At least one target k point is required")
    resolved_valleys = tuple(int(valley) for valley in valleys)
    if not resolved_valleys:
        raise ValueError("At least one valley is required")

    if active_band_indices is None:
        resolved_active_indices = active_band_indices_for_interaction(basis_model, interaction)
    else:
        resolved_active_indices = tuple(int(value) for value in np.asarray(active_band_indices, dtype=int).reshape(-1))
    if not resolved_active_indices:
        raise ValueError("At least one active band index is required")
    if min(resolved_active_indices) < 0 or max(resolved_active_indices) >= basis_model.matrix_dim:
        raise ValueError(
            f"Active band indices must lie in [0, {basis_model.matrix_dim}), got {resolved_active_indices}"
        )

    resolved_physical_model = basis_model if physical_model is None else physical_model
    return _build_projected_basis_for_indices(
        physical_model=resolved_physical_model,
        basis_model=basis_model,
        interaction=interaction,
        kvec=resolved_kvec,
        band_indices=resolved_active_indices,
        valleys=resolved_valleys,
        mesh_size=0,
        k_grid_frac=np.zeros((resolved_kvec.size, 2), dtype=float),
        screening=None,
        name="rlg_hbn_screened_active_path",
    )


def _resolve_basis_valleys(n_flavor: int, valleys: tuple[int, ...] | None) -> tuple[int, ...]:
    n = int(n_flavor)
    if n <= 0:
        raise ValueError(f"n_flavor must be positive, got {n_flavor}")
    if valleys is None:
        if n > len(VALLEY_SEQUENCE):
            raise ValueError(f"Need explicit valleys for n_flavor={n}")
        resolved = tuple(int(value) for value in VALLEY_SEQUENCE[:n])
    else:
        resolved = tuple(int(value) for value in valleys)
    if len(resolved) != n:
        raise ValueError(f"Expected {n} valley labels, got {resolved}")
    for valley in resolved:
        if valley not in VALLEY_SEQUENCE:
            raise ValueError(f"Expected valley labels in {VALLEY_SEQUENCE}, got {resolved}")
    return resolved


def _rlg_hbn_layer_local_indices(
    basis: ProjectedWavefunctionBasis,
    layer: int,
    *,
    layer_count: int,
) -> np.ndarray:
    layer_index = int(layer)
    if layer_index < 0 or layer_index >= int(layer_count):
        raise ValueError(f"layer index {layer_index} outside [0, {int(layer_count)})")
    group_name = f"layer_{layer_index}"
    if any(group.name == group_name for group in basis.component_groups):
        return component_group_indices(basis, group_name)
    return component_group_indices(basis, ComponentGroup(group_name, np.asarray([2 * layer_index, 2 * layer_index + 1])))



def calculate_layer_projected_overlap_between(
    target: ProjectedWavefunctionBasis,
    source: ProjectedWavefunctionBasis,
    m: int,
    n: int,
    *,
    layer_count: int,
    valleys: tuple[int, ...] | None = None,
) -> np.ndarray:
    if target.local_basis_size != source.local_basis_size:
        raise ValueError(f"local_basis_size mismatch: {target.local_basis_size} != {source.local_basis_size}")
    if target.n_flavor != source.n_flavor:
        raise ValueError(f"n_flavor mismatch: {target.n_flavor} != {source.n_flavor}")
    if target.n_spin != source.n_spin:
        raise ValueError(f"n_spin mismatch: {target.n_spin} != {source.n_spin}")
    if target.grid_shape != source.grid_shape:
        raise ValueError(f"grid_shape mismatch: {target.grid_shape} != {source.grid_shape}")
    layer_count = int(layer_count)
    if layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    if target.local_basis_size != 2 * layer_count:
        raise ValueError(
            f"Expected local_basis_size={2 * layer_count} for {layer_count} layers, "
            f"got {target.local_basis_size}"
        )
    resolved_valleys = _resolve_basis_valleys(target.n_flavor, valleys)

    nx, ny = target.grid_shape
    target_band_k = target.n_band * target.nk
    source_band_k = source.n_band * source.nk
    layer_blocks = np.zeros(
        (
            layer_count,
            target.n_spin,
            target.n_flavor,
            target_band_k,
            target.n_spin,
            target.n_flavor,
            source_band_k,
        ),
        dtype=np.complex128,
        order="F",
    )

    for iflavor, valley in enumerate(resolved_valleys):
        target_grid = target.wavefunctions[:, :, iflavor, :].reshape(
            target.local_basis_size,
            nx,
            ny,
            target_band_k,
            order="F",
        )
        source_grid = source.wavefunctions[:, :, iflavor, :].reshape(
            source.local_basis_size,
            nx,
            ny,
            source_band_k,
            order="F",
        )
        raw_m, raw_n = _raw_overlap_shift_for_physical_g((m, n), valley=int(valley))
        shifted = shift_wavefunction_grid(source_grid, -raw_m, -raw_n, boundary_mode="zero_fill", grid_axes=(1, 2))
        for layer in range(layer_count):
            target_layer_indices = _rlg_hbn_layer_local_indices(target, layer, layer_count=layer_count)
            source_layer_indices = _rlg_hbn_layer_local_indices(source, layer, layer_count=layer_count)
            if target_layer_indices.size != source_layer_indices.size:
                raise ValueError(
                    f"Target/source layer {layer} component sizes differ: "
                    f"{target_layer_indices.size} != {source_layer_indices.size}"
                )
            layer_local_size = int(target_layer_indices.size)
            target_layer = target_grid[target_layer_indices, :, :, :].reshape(
                layer_local_size * nx * ny, target_band_k, order="F"
            )
            shifted_layer = shifted[source_layer_indices, :, :, :].reshape(
                layer_local_size * nx * ny, source_band_k, order="F"
            )
            layer_overlap = target_layer.conj().T @ shifted_layer
            for ispin in range(target.n_spin):
                layer_blocks[layer, ispin, iflavor, :, ispin, iflavor, :] = layer_overlap

    return layer_blocks.reshape((layer_count, target.nt, target.nk, source.nt, source.nk), order="F")


def diagonal_layer_overlap_blocks(layer_overlap: np.ndarray) -> np.ndarray:
    layer_overlap = np.asarray(layer_overlap, dtype=np.complex128)
    if layer_overlap.ndim != 5:
        raise ValueError(f"Expected layer overlap shape (layer, nt, nk, nt, nk), got {layer_overlap.shape}")
    if layer_overlap.shape[1] != layer_overlap.shape[3]:
        raise ValueError(f"Expected square flavor dimensions in layer overlap, got {layer_overlap.shape}")
    if layer_overlap.shape[2] != layer_overlap.shape[4]:
        raise ValueError(f"Expected equal source/target k counts for diagonal overlap, got {layer_overlap.shape}")
    return np.diagonal(layer_overlap, axis1=2, axis2=4)


def interaction_shifts_for_cutoff(
    lattice: RLGhBNLattice,
    interaction: RLGhBNInteractionParams,
) -> tuple[tuple[int, int], ...]:
    q1_norm = float(abs(lattice.q_complex[0]))
    cutoff = float(interaction.interaction_cutoff_q1) * q1_norm
    shortest = max(min(abs(lattice.g_m1), abs(lattice.g_m2)), 1.0e-15)
    coefficient_bound = int(math.ceil(cutoff / shortest)) + 2
    entries: list[tuple[float, int, int]] = []
    for m in range(-coefficient_bound, coefficient_bound + 1):
        for n in range(-coefficient_bound, coefficient_bound + 1):
            gvec = complex(m * lattice.g_m1 + n * lattice.g_m2)
            if abs(gvec) <= cutoff + 1.0e-12:
                entries.append((round(abs(gvec), 12), int(m), int(n)))
    entries.sort(key=lambda item: (item[0], item[1] * item[1] + item[2] * item[2], item[1], item[2]))
    return tuple((m, n) for _, m, n in entries)


def _layer_coulomb_tensor_for_qvals(
    qvals: np.ndarray,
    *,
    layer_count: int,
    interaction: RLGhBNInteractionParams,
    layer_spacing_nm: float,
) -> np.ndarray:
    q_array = np.asarray(qvals, dtype=np.complex128)
    tensor = np.zeros(q_array.shape + (int(layer_count), int(layer_count)), dtype=float)
    for index in np.ndindex(q_array.shape):
        tensor[index] = layer_coulomb_matrix_mev_nm2(
            abs(complex(q_array[index])),
            int(layer_count),
            interaction,
            layer_spacing_nm=layer_spacing_nm,
        )
    return tensor


def build_rlg_hbn_layer_overlap_blocks(
    basis_data: RLGhBNProjectedBasisData,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNLayerOverlapBlockSet:
    return build_rlg_hbn_layer_overlap_blocks_between(basis_data, basis_data, shifts=shifts)


def build_rlg_hbn_layer_overlap_blocks_between(
    target_basis_data: RLGhBNProjectedBasisData,
    source_basis_data: RLGhBNProjectedBasisData,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNLayerOverlapBlockSet:
    if target_basis_data.reciprocal_grid_shape != source_basis_data.reciprocal_grid_shape:
        raise ValueError(
            "Target/source reciprocal grid shapes differ: "
            f"{target_basis_data.reciprocal_grid_shape} != {source_basis_data.reciprocal_grid_shape}"
        )
    if target_basis_data.reciprocal_grid_origin != source_basis_data.reciprocal_grid_origin:
        raise ValueError(
            "Target/source reciprocal grid origins differ: "
            f"{target_basis_data.reciprocal_grid_origin} != {source_basis_data.reciprocal_grid_origin}"
        )
    if target_basis_data.valleys != source_basis_data.valleys:
        raise ValueError(
            "Target/source valley order differs: "
            f"{target_basis_data.valleys} != {source_basis_data.valleys}"
        )

    resolved_shifts = (
        shifts
        if shifts is not None
        else interaction_shifts_for_cutoff(source_basis_data.basis_model.lattice, source_basis_data.interaction)
    )
    resolved_shifts = tuple((int(m), int(n)) for m, n in resolved_shifts)
    gvecs = np.asarray(
        [
            m * source_basis_data.basis_model.lattice.g_m1 + n * source_basis_data.basis_model.lattice.g_m2
            for m, n in resolved_shifts
        ],
        dtype=np.complex128,
    )

    layer_overlaps: dict[tuple[int, int], np.ndarray] = {}
    layer_diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    fock_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    layer_count = source_basis_data.basis_model.params.layer_count
    layer_spacing = source_basis_data.basis_model.params.layer_spacing_nm

    for shift, gvec in zip(resolved_shifts, gvecs, strict=True):
        overlap = calculate_layer_projected_overlap_between(
            target_basis_data.basis,
            source_basis_data.basis,
            shift[0],
            shift[1],
            layer_count=layer_count,
            valleys=target_basis_data.valleys,
        )
        layer_overlaps[shift] = overlap
        if target_basis_data.nk == source_basis_data.nk and target_basis_data.nt == source_basis_data.nt:
            layer_diagonal_overlaps[shift] = diagonal_layer_overlap_blocks(overlap)
        hartree_layer_coulomb[shift] = layer_coulomb_matrix_mev_nm2(
            abs(complex(gvec)),
            layer_count,
            source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )
        qvals = target_basis_data.kvec[:, None] - source_basis_data.kvec[None, :] + complex(gvec)
        fock_layer_coulomb[shift] = _layer_coulomb_tensor_for_qvals(
            qvals,
            layer_count=layer_count,
            interaction=source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )

    return RLGhBNLayerOverlapBlockSet(
        shifts=resolved_shifts,
        gvecs=gvecs,
        layer_overlaps=layer_overlaps,
        layer_diagonal_overlaps=layer_diagonal_overlaps,
        hartree_layer_coulomb=hartree_layer_coulomb,
        fock_layer_coulomb=fock_layer_coulomb,
    )


def _contract_layer_fock_term(
    left_overlap: np.ndarray,
    density_delta: np.ndarray,
    coeff_matrix: np.ndarray,
    right_overlap: np.ndarray,
) -> np.ndarray:
    left_overlap = np.asarray(left_overlap, dtype=np.complex128)
    right_overlap = np.asarray(right_overlap, dtype=np.complex128)
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    coeff_matrix = np.asarray(coeff_matrix)
    nt_target, nk_target, nt_source, nk_source = left_overlap.shape
    if right_overlap.shape != left_overlap.shape:
        raise ValueError(f"Expected right_overlap shape {left_overlap.shape}, got {right_overlap.shape}")
    if density_delta.shape != (nt_source, nt_source, nk_source):
        raise ValueError(f"Expected density_delta shape {(nt_source, nt_source, nk_source)}, got {density_delta.shape}")
    if coeff_matrix.shape != (nk_target, nk_source):
        raise ValueError(f"Expected coeff_matrix shape {(nk_target, nk_source)}, got {coeff_matrix.shape}")

    if _rlg_hbn_use_numba():
        if not _NUMBA_AVAILABLE:
            if _rlg_hbn_require_numba():
                raise RuntimeError("MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1 but numba is not available")
        else:
            return _contract_layer_fock_term_numba_kernel(
                np.ascontiguousarray(left_overlap),
                np.ascontiguousarray(density_delta),
                np.ascontiguousarray(coeff_matrix),
                np.ascontiguousarray(right_overlap),
            )

    left_blocks = np.transpose(left_overlap, (1, 3, 0, 2))
    right_blocks = np.transpose(right_overlap, (1, 3, 0, 2))
    density_t = np.transpose(density_delta, (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", left_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(right_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def build_rlg_hbn_interaction_components(
    density_delta: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
) -> RLGhBNInteractionComponents:
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    if density_delta.ndim != 3 or density_delta.shape[0] != density_delta.shape[1]:
        raise ValueError(f"Expected density_delta shape (nt, nt, nk), got {density_delta.shape}")
    nt, _, nk = density_delta.shape
    scale = float(beta) * float(v0) / float(nk)
    hartree = np.zeros_like(density_delta)
    fock = np.zeros_like(density_delta)

    for shift in overlap_blocks.shifts:
        layer_diagonal = overlap_blocks.layer_diagonal_overlaps[shift]
        layer_overlap = overlap_blocks.layer_overlaps[shift]
        hartree_kernel = overlap_blocks.hartree_layer_coulomb[shift]
        fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, overlap_blocks.fock_layer_coulomb[shift])
        if layer_diagonal.shape[1:] != (nt, nt, nk):
            raise ValueError(f"Layer diagonal overlap for {shift} is incompatible with density shape {density_delta.shape}")
        if layer_overlap.shape[1:] != (nt, nk, nt, nk):
            raise ValueError(f"Layer overlap for {shift} is incompatible with density shape {density_delta.shape}")

        layer_traces = np.asarray(
            [
                compute_density_overlap_trace_from_diagonal(density_delta, layer_diagonal[layer])
                for layer in range(layer_diagonal.shape[0])
            ],
            dtype=np.complex128,
        )
        for target_layer in range(layer_diagonal.shape[0]):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hartree += prefactor * layer_diagonal[target_layer]

        for target_layer in range(layer_overlap.shape[0]):
            for source_layer in range(layer_overlap.shape[0]):
                coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                if np.any(coeff != 0.0):
                    fock -= _contract_layer_fock_term(
                        layer_overlap[target_layer],
                        density_delta,
                        coeff,
                        layer_overlap[source_layer],
                    )

    return RLGhBNInteractionComponents(hartree=hartree, fock=fock, total=hartree + fock)


def build_rlg_hbn_hf_interaction_hamiltonian(
    density_delta: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
) -> np.ndarray:
    return build_rlg_hbn_interaction_components(
        density_delta,
        overlap_blocks,
        v0=float(v0),
        beta=float(beta),
    ).total


def build_rlg_hbn_target_hamiltonian(
    base_hamiltonian: np.ndarray,
    density_delta: np.ndarray,
    *,
    source_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    target_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    target_source_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    v0: float,
    beta: float = 1.0,
) -> np.ndarray:
    base = np.asarray(base_hamiltonian, dtype=np.complex128)
    density = np.asarray(density_delta, dtype=np.complex128)
    if base.ndim != 3 or base.shape[0] != base.shape[1]:
        raise ValueError(f"Expected base_hamiltonian shape (nt, nt, nk_target), got {base.shape}")
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected density_delta shape (nt, nt, nk_source), got {density.shape}")
    nt_target, _, nk_target = base.shape
    nt_source = int(density.shape[0])

    nk_source = int(density.shape[2])
    scale = float(beta) * float(v0) / float(nk_source)
    hamiltonian = base.copy()

    for shift in source_overlap_blocks.shifts:
        if shift not in target_overlap_blocks.layer_diagonal_overlaps:
            raise ValueError(f"Missing target diagonal overlaps for shift {shift}")
        if shift not in target_source_overlap_blocks.layer_overlaps:
            raise ValueError(f"Missing target-source overlaps for shift {shift}")

        source_layer_diagonal = source_overlap_blocks.layer_diagonal_overlaps[shift]
        target_layer_diagonal = target_overlap_blocks.layer_diagonal_overlaps[shift]
        target_source_layer_overlap = target_source_overlap_blocks.layer_overlaps[shift]
        hartree_kernel = source_overlap_blocks.hartree_layer_coulomb[shift]
        fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, target_source_overlap_blocks.fock_layer_coulomb[shift])

        if source_layer_diagonal.shape[1:] != (nt_source, nt_source, nk_source):
            raise ValueError(
                f"Source layer diagonal overlap for {shift} is incompatible with density shape {density.shape}"
            )
        if target_layer_diagonal.shape[1:] != (nt_target, nt_target, nk_target):
            raise ValueError(
                f"Target layer diagonal overlap for {shift} is incompatible with base shape {base.shape}"
            )
        if target_source_layer_overlap.shape[1:] != (nt_target, nk_target, nt_source, nk_source):
            raise ValueError(
                f"Target-source layer overlap for {shift} is incompatible with target/source shapes "
                f"{base.shape} and {density.shape}"
            )

        layer_traces = np.asarray(
            [
                compute_density_overlap_trace_from_diagonal(density, source_layer_diagonal[layer])
                for layer in range(source_layer_diagonal.shape[0])
            ],
            dtype=np.complex128,
        )
        for target_layer in range(target_layer_diagonal.shape[0]):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hamiltonian += prefactor * target_layer_diagonal[target_layer]

        for target_layer in range(target_source_layer_overlap.shape[0]):
            for source_layer in range(target_source_layer_overlap.shape[0]):
                coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                if np.any(coeff != 0.0):
                    hamiltonian -= _contract_layer_fock_term(
                        target_source_layer_overlap[target_layer],
                        density,
                        coeff,
                        target_source_layer_overlap[source_layer],
                    )

    _hermitize_blocks_inplace(hamiltonian)
    return hamiltonian


def _diagonalize_hf_path_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Expected Hamiltonian shape (nt, nt, nk), got {hamiltonian.shape}")
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def evaluate_rlg_hbn_hf_path(
    run: RLGhBNHartreeFockRun,
    path: KPath,
    *,
    beta: float = 1.0,
    chunk_size: int = 4,
) -> RLGhBNHFPathResult:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    source_basis_data = run.basis_data
    source_overlap_blocks = run.overlap_blocks
    density = np.asarray(run.state.density, dtype=np.complex128)
    kvec = np.asarray(path.kvec, dtype=np.complex128)
    remote_source = _prepare_remote_average_source(source_basis_data)

    hamiltonian_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []
    basis_chunks: list[RLGhBNProjectedBasisData] = []
    for start in range(0, kvec.size, int(chunk_size)):
        stop = min(start + int(chunk_size), kvec.size)
        target_basis_data = build_rlg_hbn_projected_basis_for_kvec(
            source_basis_data.basis_model,
            source_basis_data.interaction,
            kvec[start:stop],
            physical_model=source_basis_data.model,
            active_band_indices=source_basis_data.active_band_indices,
            valleys=source_basis_data.valleys,
        )
        fixed_remote = _remote_average_hamiltonian_from_source(
            target_basis_data,
            source_basis_data,
            remote_source,
            shifts=source_overlap_blocks.shifts,
            beta=beta,
        )
        target_basis_data = replace(
            target_basis_data,
            h0=np.asarray(target_basis_data.physical_h0, dtype=np.complex128) + fixed_remote,
            fixed_remote_hamiltonian=fixed_remote,
        )
        _assert_average_remote_hamiltonian_contract(target_basis_data)
        target_overlap_blocks = build_rlg_hbn_layer_overlap_blocks(
            target_basis_data,
            shifts=source_overlap_blocks.shifts,
        )
        target_source_overlap_blocks = build_rlg_hbn_layer_overlap_blocks_between(
            target_basis_data,
            source_basis_data,
            shifts=source_overlap_blocks.shifts,
        )
        chunk_hamiltonian = build_rlg_hbn_target_hamiltonian(
            target_basis_data.h0,
            density,
            source_overlap_blocks=source_overlap_blocks,
            target_overlap_blocks=target_overlap_blocks,
            target_source_overlap_blocks=target_source_overlap_blocks,
            v0=run.state.v0,
            beta=beta,
        )
        hamiltonian_chunks.append(chunk_hamiltonian)
        energy_chunks.append(_diagonalize_hf_path_hamiltonian(chunk_hamiltonian))
        basis_chunks.append(target_basis_data)

    hamiltonian = np.concatenate(hamiltonian_chunks, axis=2)
    energies = np.concatenate(energy_chunks, axis=1)
    first_basis = basis_chunks[0]
    basis_data = RLGhBNProjectedBasisData(
        model=first_basis.model,
        basis_model=first_basis.basis_model,
        interaction=first_basis.interaction,
        screening=None,
        mesh_size=0,
        kvec=kvec,
        k_grid_frac=np.zeros((kvec.size, 2), dtype=float),
        basis=ProjectedWavefunctionBasis(
            wavefunctions=np.concatenate([chunk.basis.wavefunctions for chunk in basis_chunks], axis=3),
            grid_shape=first_basis.basis.grid_shape,
            n_spin=first_basis.basis.n_spin,
            local_basis_size=first_basis.basis.local_basis_size,
            name=first_basis.basis.name,
            component_groups=first_basis.basis.component_groups,
        ),
        h0=np.concatenate([chunk.h0 for chunk in basis_chunks], axis=2),
        band_energies=np.concatenate([chunk.band_energies for chunk in basis_chunks], axis=2),
        active_band_indices=first_basis.active_band_indices,
        flat_band_indices=first_basis.flat_band_indices,
        valleys=first_basis.valleys,
        reciprocal_grid_shape=first_basis.reciprocal_grid_shape,
        reciprocal_grid_origin=first_basis.reciprocal_grid_origin,
        moire_cell_area_nm2=first_basis.moire_cell_area_nm2,
        physical_h0=np.concatenate([np.asarray(chunk.physical_h0, dtype=np.complex128) for chunk in basis_chunks], axis=2),
        fixed_remote_hamiltonian=np.concatenate(
            [np.asarray(chunk.fixed_remote_hamiltonian, dtype=np.complex128) for chunk in basis_chunks],
            axis=2,
        ),
    )
    return RLGhBNHFPathResult(
        path=path,
        basis_data=basis_data,
        hamiltonian=hamiltonian,
        energies=energies,
    )


def compute_rlg_hbn_oda_parameter(
    state: RLGhBNHartreeFockState,
    delta_density: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    beta: float = 1.0,
) -> float:
    return compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: build_rlg_hbn_hf_interaction_hamiltonian(
            density,
            overlap_blocks,
            v0=state.v0,
            beta=beta,
        ),
    )


def _hermitize_blocks_inplace(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)


def rlg_hbn_projector_idempotency_residual(density_delta: np.ndarray, reference_density: np.ndarray) -> float:
    projector = rlg_hbn_projector_from_density(density_delta, reference_density)
    residual = 0.0
    for ik in range(projector.shape[2]):
        block = projector[:, :, ik]
        residual = max(residual, float(np.max(np.abs(block @ block - block))))
    return float(residual)


def rlg_hbn_hermitian_residual(blocks: np.ndarray) -> float:
    blocks = np.asarray(blocks, dtype=np.complex128)
    residual = 0.0
    for ik in range(blocks.shape[2]):
        residual = max(residual, float(np.max(np.abs(blocks[:, :, ik] - blocks[:, :, ik].conjugate().T))))
    return float(residual)


def rlg_hbn_gap_estimate(
    energies: np.ndarray,
    nu: float,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    energies = np.asarray(energies, dtype=float)
    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        energies.shape[0],
        energies.shape[1],
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    sorted_energies = np.sort(energies, axis=None)
    if total_occupied <= 0 or total_occupied >= sorted_energies.size:
        return float("nan")
    return float(sorted_energies[total_occupied] - sorted_energies[total_occupied - 1])


def _update_rlg_hbn_diagnostics_from_density(state: RLGhBNHartreeFockState) -> None:
    state.diagnostics["filling"] = rlg_hbn_filling_from_density(
        state.density,
        state.reference_density,
        active_valence_bands=state.active_valence_bands,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    state.diagnostics["projector_idempotency_residual"] = rlg_hbn_projector_idempotency_residual(
        state.density,
        state.reference_density,
    )
    state.diagnostics["density_hermitian_residual"] = rlg_hbn_hermitian_residual(state.density)
    state.diagnostics["hamiltonian_hermitian_residual"] = rlg_hbn_hermitian_residual(state.hamiltonian)
    state.diagnostics["hf_gap"] = rlg_hbn_gap_estimate(
        state.energies,
        state.nu,
        active_valence_bands=state.active_valence_bands,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


def normalize_rlg_hbn_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "sp": "bm",
        "noninteracting": "bm",
        "random": "random",
        "diag_random": "random",
        "flavor": "flavor",
        "polarized": "flavor",
        "polarized_k_up": "flavor",
        "perturbed": "perturbed",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported RLG/hBN HF init mode: {init_mode}. "
            "Supported modes: bm, random, diag_random, flavor, polarized, polarized_k_up, perturbed"
        )
    return aliases[normalized]


def rlg_hbn_flavor_occupation_counts_for_init_mode(
    init_mode: str,
    *,
    nu: float,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
    seed: int | None = None,
) -> tuple[int, ...] | None:
    normalized = normalize_rlg_hbn_init_mode(init_mode)
    if normalized in {"bm", "random", "perturbed"}:
        return None

    integer_nu = int(round(float(nu)))
    if abs(float(nu) - float(integer_nu)) > 1.0e-9:
        return None
    n_spin = int(n_spin)
    n_eta = int(n_eta)
    n_band = int(n_band)
    n_valence = int(active_valence_bands)
    if n_valence < 0 or n_valence > n_band:
        raise ValueError(f"active_valence_bands must lie in [0, {n_band}], got {active_valence_bands}")

    counts = np.full((n_spin, n_eta), n_valence, dtype=int)
    flavor_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    flavor_order = [(s, e) for s, e in flavor_order if s < n_spin and e < n_eta]
    flavor_order.extend(
        (s, e)
        for s in range(n_spin)
        for e in range(n_eta)
        if (s, e) not in flavor_order
    )
    if seed is not None and flavor_order:
        start = (int(seed) - 1) % len(flavor_order)
        flavor_order = flavor_order[start:] + flavor_order[:start]
    if integer_nu > 0:
        if integer_nu > len(flavor_order):
            raise ValueError(f"Positive integer filling nu={nu} exceeds available flavors {len(flavor_order)}")
        for ispin, ieta in flavor_order[:integer_nu]:
            counts[ispin, ieta] += 1
    elif integer_nu < 0:
        if abs(integer_nu) > len(flavor_order):
            raise ValueError(f"Negative integer filling nu={nu} exceeds available flavors {len(flavor_order)}")
        for ispin, ieta in reversed(flavor_order[-abs(integer_nu) :]):
            counts[ispin, ieta] -= 1

    if np.any(counts < 0) or np.any(counts > n_band):
        raise ValueError(f"Invalid RLG/hBN flavor counts for nu={nu}: {counts}")
    return tuple(int(value) for value in counts.reshape(-1, order="C"))


def initialize_rlg_hbn_density(
    h0: np.ndarray,
    *,
    nu: float,
    reference_density: np.ndarray,
    active_valence_bands: int,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_rlg_hbn_init_mode(init_mode)
    h0 = np.asarray(h0, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    nt, _, nk = h0.shape
    if reference_density.shape != h0.shape:
        raise ValueError(f"Expected reference_density shape {h0.shape}, got {reference_density.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    if init_mode == "bm":
        return build_rlg_hbn_density_from_hamiltonian(
            h0,
            nu=nu,
            reference_density=reference_density,
            active_valence_bands=active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )[0]

    rng = np.random.default_rng(seed)
    density = np.zeros_like(h0)
    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        nt,
        nk,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    idx = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")

    if init_mode == "random":
        random_energies = rng.standard_normal((nt, nk))
        occ_mask = occupied_state_mask(random_energies, total_occupied)
        for ik in range(nk):
            unitary = random_unitary_from_hermitian(nt, rng)
            occupied = np.flatnonzero(occ_mask[:, ik])
            if occupied.size == 0:
                density[:, :, ik] = -reference_density[:, :, ik]
            else:
                occupied_vecs = unitary[:, occupied]
                density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density

    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=nu,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        seed=seed if init_mode == "flavor" else None,
    )
    if counts is None:
        raise ValueError(f"init_mode={init_mode!r} requires integer flavor occupation counts for nu={nu}")
    counts_2d = np.asarray(counts, dtype=int).reshape((int(n_spin), int(n_eta)), order="C")
    for ik in range(nk):
        density[:, :, ik] = -reference_density[:, :, ik]
        for ispin in range(int(n_spin)):
            for ieta in range(int(n_eta)):
                n_occ = int(counts_2d[ispin, ieta])
                if n_occ <= 0:
                    continue
                block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                occupied = block_indices[:n_occ]
                density[:, :, ik][np.ix_(occupied, occupied)] = (
                    np.eye(n_occ, dtype=np.complex128)
                    - reference_density[:, :, ik][np.ix_(occupied, occupied)]
                )

    if init_mode == "perturbed":
        apply_random_projector_rotation(
            density,
            reference_density=reference_density,
            alpha=0.05,
            seed=seed,
        )
    return density


def build_rlg_hbn_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    nu: float,
    reference_density: np.ndarray,
    active_valence_bands: int,
    occupation_counts: tuple[int, ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    if reference_density.shape != hamiltonian.shape:
        raise ValueError(f"Expected reference_density shape {hamiltonian.shape}, got {reference_density.shape}")
    if nt != int(n_spin) * int(n_eta) * int(n_band):
        raise ValueError(f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    energies = np.zeros((nt, nk), dtype=float)
    density = np.zeros_like(hamiltonian)

    if occupation_counts is not None:
        counts = np.asarray(occupation_counts, dtype=int).reshape(-1)
        if counts.size != int(n_spin) * int(n_eta):
            raise ValueError(f"Expected {int(n_spin) * int(n_eta)} flavor occupation counts, got {counts.size}")
        if np.any(counts < 0) or np.any(counts > int(n_band)):
            raise ValueError(f"Flavor occupation counts must lie in [0, {int(n_band)}], got {counts.tolist()}")
        if int(np.sum(counts)) != rlg_hbn_occupied_bands_per_k(
            nu,
            nt,
            active_valence_bands=active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
        ):
            raise ValueError("Flavor occupation counts do not match the requested filling")

        idx = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
        counts_2d = counts.reshape((int(n_spin), int(n_eta)), order="C")
        occ_mask = np.zeros((nt, nk), dtype=bool)
        for ik in range(nk):
            density[:, :, ik] = -reference_density[:, :, ik]
            for ispin in range(int(n_spin)):
                for ieta in range(int(n_eta)):
                    block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    reference_block = reference_density[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
                    n_occ = int(counts_2d[ispin, ieta])
                    if n_occ > 0:
                        occupied_vecs = eigvecs[:, :n_occ]
                        density[:, :, ik][np.ix_(block_indices, block_indices)] = (
                            occupied_vecs.conjugate() @ occupied_vecs.T - reference_block
                        )
                        occ_mask[block_indices[:n_occ], ik] = True
        if np.any(occ_mask) and not np.all(occ_mask):
            mu = 0.5 * (float(np.max(energies[occ_mask])) + float(np.min(energies[~occ_mask])))
        else:
            mu = float(np.mean(energies))
        return density, energies, float(mu), occ_mask

    vecs = np.zeros_like(hamiltonian)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs

    total_occupied = rlg_hbn_occupied_state_count(
        nu,
        nt,
        nk,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    occ_mask = occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, float(total_occupied) / float(energies.size))

    for ik in range(nk):
        occupied = np.flatnonzero(occ_mask[:, ik])
        if occupied.size == 0:
            density[:, :, ik] = -reference_density[:, :, ik]
            continue
        occupied_vecs = vecs[:, occupied, ik]
        density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]

    return density, energies, float(mu), occ_mask


def build_rlg_hbn_hf_problem(
    state: RLGhBNHartreeFockState,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
    step_callback: Callable[[RLGhBNHartreeFockState, HartreeFockStepResult], None] | None = None,
) -> HartreeFockProblem:
    """Build the reusable core-HF problem wrapper for an RLG/hBN state.

    The RLG/hBN system layer still owns the projected basis, layer-resolved
    Coulomb tables, filling convention, and ODA functional.  This adapter only
    packages those system-specific callables behind the shared
    :class:`mean_field.core.hf.HartreeFockProblem` interface.
    """

    def initialize_state(state_obj: RLGhBNHartreeFockState, *, init_mode: str, seed: int) -> None:
        if initial_density is not None:
            density = np.asarray(initial_density, dtype=np.complex128)
            if density.shape != state_obj.density.shape:
                raise ValueError(f"Expected initial_density shape {state_obj.density.shape}, got {density.shape}")
            state_obj.density[:, :, :] = density
        else:
            state_obj.density[:, :, :] = initialize_rlg_hbn_density(
                state_obj.h0,
                nu=state_obj.nu,
                reference_density=state_obj.reference_density,
                active_valence_bands=state_obj.active_valence_bands,
                init_mode=init_mode,
                seed=seed,
                n_spin=state_obj.n_spin,
                n_eta=state_obj.n_eta,
                n_band=state_obj.n_band,
            )
        _hermitize_blocks_inplace(state_obj.density)
        _update_rlg_hbn_diagnostics_from_density(state_obj)

    def build_density(hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occupation_mask = build_rlg_hbn_density_from_hamiltonian(
            hamiltonian,
            nu=state.nu,
            reference_density=state.reference_density,
            active_valence_bands=state.active_valence_bands,
            occupation_counts=state.occupation_counts,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        )
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables={"occupation_mask": occupation_mask},
        )

    kernel = HartreeFockKernel(
        interaction_builder=lambda density: build_rlg_hbn_hf_interaction_hamiltonian(
            density,
            overlap_blocks,
            v0=state.v0,
            beta=beta,
        ),
        density_builder=build_density,
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: compute_rlg_hbn_oda_parameter(
            state_obj,  # type: ignore[arg-type]
            delta_density,
            overlap_blocks,
            beta=beta,
        ),
        hamiltonian_postprocessor=_hermitize_blocks_inplace,
        density_postprocessor=_hermitize_blocks_inplace,
        step_callback=step_callback,  # type: ignore[arg-type]
        convergence_rule="raw",
    )
    return HartreeFockProblem(
        initializer=initialize_state,
        kernel=kernel,
    )


def run_rlg_hbn_hartree_fock(
    basis_data: RLGhBNProjectedBasisData,
    *,
    overlap_blocks: RLGhBNLayerOverlapBlockSet | None = None,
    nu: float = 1.0,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 80,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    occupation_counts: tuple[int, ...] | None = None,
    initial_density: np.ndarray | None = None,
    step_callback: Callable[[RLGhBNHartreeFockState, HartreeFockStepResult], None] | None = None,
) -> RLGhBNHartreeFockRun:
    resolved_counts = occupation_counts
    if resolved_counts is None:
        resolved_counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
            init_mode,
            nu=nu,
            active_valence_bands=basis_data.interaction.active_valence_bands,
            n_spin=basis_data.basis.n_spin,
            n_eta=basis_data.basis.n_flavor,
            n_band=basis_data.basis.n_band,
            seed=seed,
        )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=nu,
        precision=precision,
        occupation_counts=resolved_counts,
    )
    resolved_blocks = overlap_blocks if overlap_blocks is not None else build_rlg_hbn_layer_overlap_blocks(basis_data)
    problem = build_rlg_hbn_hf_problem(
        state,
        resolved_blocks,
        beta=beta,
        initial_density=initial_density,
        step_callback=step_callback,
    )
    core_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
    _update_rlg_hbn_diagnostics_from_density(state)
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=core_run.iter_energy,
        iter_err=core_run.iter_err,
        iter_oda=core_run.iter_oda,
        init_mode=core_run.init_mode,
        seed=core_run.seed,
        converged=core_run.converged,
        exit_reason=core_run.exit_reason,
        overlap_blocks=resolved_blocks,
        basis_data=basis_data,
    )


def scan_rlg_hbn_ground_state(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    nu: float = 1.0,
    init_modes: tuple[str, ...] = ("flavor", "bm", "perturbed"),
    seeds: tuple[int, ...] = (1,),
    beta: float = 1.0,
    max_iter: int = 80,
    precision: float = 1.0e-6,
    oda_stall_threshold: float = 1.0e-3,
    mesh_size: int | None = None,
    screening_mesh_size: int | None = None,
    run_callback: Callable[[RLGhBNHartreeFockRun], None] | None = None,
) -> RLGhBNGroundStateScan:
    basis_data = build_rlg_hbn_projected_basis(
        model,
        interaction,
        mesh_size=mesh_size,
        screening_mesh_size=screening_mesh_size,
    )
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data)
    runs: list[RLGhBNHartreeFockRun] = []
    for init_mode in init_modes:
        for seed in seeds:
            run = run_rlg_hbn_hartree_fock(
                basis_data,
                overlap_blocks=overlap_blocks,
                nu=nu,
                init_mode=init_mode,
                seed=int(seed),
                beta=beta,
                max_iter=max_iter,
                precision=precision,
                oda_stall_threshold=oda_stall_threshold,
            )
            runs.append(run)
            if run_callback is not None:
                run_callback(run)
    return RLGhBNGroundStateScan(runs=tuple(runs))


__all__ = [
    "RLGhBNGroundStateScan",
    "RLGhBNHartreeFockRun",
    "RLGhBNHartreeFockState",
    "RLGhBNInteractionComponents",
    "RLGhBNLayerOverlapBlockSet",
    "RLGhBNProjectedBasisData",
    "RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING",
    "RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION",
    "RLG_HBN_FORM_FACTOR_CONVENTION_VERSION",
    "VALLEY_SEQUENCE",
    "active_band_indices_for_interaction",
    "average_scheme_density_delta",
    "build_rlg_hbn_density_from_hamiltonian",
    "build_rlg_hbn_hf_interaction_hamiltonian",
    "build_rlg_hbn_hf_problem",
    "build_rlg_hbn_interaction_components",
    "build_rlg_hbn_layer_overlap_blocks",
    "build_rlg_hbn_projected_basis",
    "calculate_layer_projected_overlap_between",
    "compute_rlg_hbn_oda_parameter",
    "diagonal_layer_overlap_blocks",
    "initialize_rlg_hbn_density",
    "interaction_shifts_for_cutoff",
    "normalize_rlg_hbn_init_mode",
    "rlg_hbn_density_delta",
    "rlg_hbn_filling_from_density",
    "rlg_hbn_flavor_occupation_counts_for_init_mode",
    "rlg_hbn_gap_estimate",
    "rlg_hbn_layer_component_groups",
    "rlg_hbn_hermitian_residual",
    "rlg_hbn_occupied_bands_per_k",
    "rlg_hbn_occupied_state_count",
    "rlg_hbn_average_reference_density",
    "rlg_hbn_projector_from_density",
    "rlg_hbn_projector_idempotency_residual",
    "rlg_hbn_reference_density",
    "run_rlg_hbn_hartree_fock",
    "scan_rlg_hbn_ground_state",
]
