from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from ...core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from ...core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    ProjectedWavefunctionBasis,
    build_projected_interaction_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    density_from_fixed_sector_occupations as _core_density_from_fixed_sector_occupations,
    diagonal_overlap_blocks,
    flat_sector_indices as _core_flat_sector_indices,
    flatten_sector_blocks as _core_flatten_sector_blocks,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb,
    screened_coulomb_matrix,
    unflatten_sector_blocks as _core_unflatten_sector_blocks,
    unflatten_sector_energies as _core_unflatten_sector_energies,
)
from ...core.hf.contracts_bridge import density_state_from_delta
from ...core.supercell import (
    IntegerSupercell,
    fixed_sector_occupation_counts,
    folded_indices_for_primitive_band,
    folded_reference_diagonal_by_primitive_index,
    primitive_filling_from_occupation_counts,
)
from .lattice import TMBGLattice
from .model import TMBGModel
from .params import TMBGParameters


@dataclass(frozen=True)
class PolshynDoubledCell(IntegerSupercell):
    """Area-2 rectangular cell used for the Polshyn tMBG SBCI.

    In the primitive tMBG convention ``b1=G_M1`` and ``b2=G_M2``.  The doubled
    cell keeps the y-translation and doubles the other primitive direction:

    ``B1 = b1/2`` and ``B2 = b2 - b1/2``.

    The CDW wavevector is therefore ``Q = B1``.
    """

    n11: int = 2
    n12: int = 1
    n21: int = 0
    n22: int = 1

    def reciprocal_vectors(self, lattice: TMBGLattice) -> tuple[complex, complex]:
        return super().reciprocal_vectors(lattice.g_m1, lattice.g_m2)

    def primitive_to_supercell_coords(self, n1: int, n2: int, fold: int = 0) -> tuple[int, int]:
        sx, sy = self.primitive_shift_to_supercell(int(n1), int(n2))
        return (int(sx + int(fold)), int(sy))


def polshyn_doubled_cell() -> PolshynDoubledCell:
    return PolshynDoubledCell()


@dataclass(frozen=True)
class PolshynFillingSummary:
    projected_indices: tuple[int, ...]
    target_band_index: int
    target_primitive_position: int
    target_fold_indices: tuple[int, int]
    nb: int
    area_ratio: int
    reference_diagonal: np.ndarray
    occupation_counts: np.ndarray
    primitive_nu: float
    matches_expected_filling: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "projected_indices": [int(value) for value in self.projected_indices],
            "target_band_index": int(self.target_band_index),
            "target_primitive_position": int(self.target_primitive_position),
            "target_fold_indices": [int(value) for value in self.target_fold_indices],
            "nb": int(self.nb),
            "area_ratio": int(self.area_ratio),
            "reference_diagonal": [float(value) for value in self.reference_diagonal],
            "occupation_counts": self.occupation_counts.astype(int).tolist(),
            "primitive_nu": float(self.primitive_nu),
            "matches_expected_filling": bool(self.matches_expected_filling),
        }


@dataclass(frozen=True)
class PolshynProjectedBasis:
    model: TMBGModel
    supercell: PolshynDoubledCell
    kvec: np.ndarray
    k_grid_frac: np.ndarray | None
    projected_indices: tuple[int, ...]
    target_band_index: int
    wavefunctions: np.ndarray
    h0_blocks: np.ndarray
    reference_diagonal: np.ndarray
    super_b1: complex
    super_b2: complex
    embedding_shape: tuple[int, int]
    embedding_origin: tuple[int, int]
    embedding_positions: dict[tuple[int, int, int], tuple[int, int]]

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def n_eta(self) -> int:
        return int(self.wavefunctions.shape[2])

    @property
    def n_spin(self) -> int:
        return int(self.h0_blocks.shape[0])

    @property
    def nb(self) -> int:
        return int(self.wavefunctions.shape[1])

    @property
    def basis_dimension(self) -> int:
        return int(self.wavefunctions.shape[0])

    @property
    def local_basis_size(self) -> int:
        return 6


@dataclass
class PolshynWangHFState:
    """Minimal mutable state for the generic Wang/Xiaoyu HF iteration engine."""

    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float
    precision: float
    v0: float
    diagnostics: dict[str, float]

    @property
    def nk(self) -> int:
        return int(self.density.shape[2])


def _unavailable_polshyn_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "Polshyn-Wang canonical contract records an already-built projected basis; "
        "use mean_field.systems.tmbg.polshyn_supercell builders for fresh Hamiltonians."
    )

def _unavailable_polshyn_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "Polshyn-Wang canonical contract records post-run arrays; "
        "fresh diagonalization is not performed by the adapter."
    )

def _tmbg_params_summary(params: TMBGParameters) -> dict[str, object]:
    keys = (
        "graphene_lattice_constant_nm",
        "t0",
        "t1",
        "t3",
        "t4",
        "delta",
        "omega",
        "omega_prime",
        "interlayer_potential",
        "staggered_potential",
        "blg_stacking",
        "bernal_convention",
        "model_name",
        "vf",
        "v3",
        "v4",
    )
    out: dict[str, object] = {}
    for key in keys:
        value = getattr(params, key)
        if isinstance(value, str):
            out[key] = value
        else:
            out[key] = float(value)
    return out

def _polshyn_single_particle_model(basis: PolshynProjectedBasis) -> ContractSingleParticleModel:
    return ContractSingleParticleModel(
        system="tmbg_polshyn_doubled",
        lattice=basis.model.lattice_summary(),
        params=_tmbg_params_summary(basis.model.params),
        hamiltonian_builder=_unavailable_polshyn_hamiltonian_builder,
        diagonalizer=_unavailable_polshyn_diagonalizer,
        metadata={
            "source": "mean_field.systems.tmbg.polshyn_supercell",
            "theta_deg": float(basis.model.theta_deg),
            "n_shells": int(basis.model.n_shells),
            "supercell": basis.supercell.as_dict(),
            "projected_indices": [int(value) for value in basis.projected_indices],
            "target_band_index": int(basis.target_band_index),
        },
    )

def _basis_energies_from_flat_h0(h0: np.ndarray) -> np.ndarray:
    h0_array = np.asarray(h0, dtype=np.complex128)
    out = np.zeros((h0_array.shape[0], h0_array.shape[2]), dtype=float)
    for ik in range(h0_array.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(h0_array[:, :, ik])
    return out

def _polshyn_flat_state_index(basis: PolshynProjectedBasis) -> np.ndarray:
    return np.arange(int(basis.n_spin) * int(basis.n_eta) * int(basis.nb), dtype=int).reshape(
        (int(basis.n_spin), int(basis.n_eta), int(basis.nb)),
        order="F",
    )

def _polshyn_folded_band_labels(basis: PolshynProjectedBasis) -> tuple[dict[str, object], ...]:
    labels: list[dict[str, object]] = []
    for primitive_position, primitive_band_index in enumerate(basis.projected_indices):
        for fold_index in range(2):
            labels.append(
                {
                    "folded_band_index": int(2 * primitive_position + fold_index),
                    "primitive_position": int(primitive_position),
                    "primitive_band_index": int(primitive_band_index),
                    "fold_index": int(fold_index),
                    "is_target_band": bool(int(primitive_band_index) == int(basis.target_band_index)),
                }
            )
    if len(labels) != int(basis.nb):
        raise ValueError(f"Polshyn folded band labels length {len(labels)} does not match nb={basis.nb}")
    return tuple(labels)

def _polshyn_active_band_indices(basis: PolshynProjectedBasis) -> tuple[int, ...]:
    labels = np.zeros((int(basis.n_spin) * int(basis.n_eta) * int(basis.nb),), dtype=int)
    state_index = _polshyn_flat_state_index(basis)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            for iband in range(int(basis.nb)):
                primitive = int(basis.projected_indices[iband // 2])
                labels[int(state_index[ispin, ieta, iband])] = primitive
    return tuple(int(value) for value in labels)

def _polshyn_flavor_labels(basis: PolshynProjectedBasis) -> tuple[str, ...]:
    valley_labels = ("K", "Kprime")
    labels = [""] * (int(basis.n_spin) * int(basis.n_eta) * int(basis.nb))
    state_index = _polshyn_flat_state_index(basis)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            valley_label = valley_labels[ieta] if ieta < len(valley_labels) else f"eta{ieta}"
            for iband in range(int(basis.nb)):
                labels[int(state_index[ispin, ieta, iband])] = f"spin{ispin}_{valley_label}_folded_band{iband}"
    return tuple(labels)

def _polshyn_reference_density_flat(basis: PolshynProjectedBasis) -> np.ndarray:
    reference_diagonal = np.asarray(basis.reference_diagonal, dtype=float).reshape(-1)
    if reference_diagonal.shape != (int(basis.nb),):
        raise ValueError(
            f"Polshyn reference_diagonal shape {reference_diagonal.shape} does not match nb={basis.nb}"
        )
    reference_matrix = np.diag(reference_diagonal).astype(np.complex128)
    blocks = np.zeros((basis.n_spin, basis.n_eta, basis.nb, basis.nb, basis.nk), dtype=np.complex128)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            blocks[ispin, ieta, :, :, :] = reference_matrix[:, :, None]
    return _core_flatten_sector_blocks(blocks)

def _validate_polshyn_wang_bundle_shapes(basis: PolshynProjectedBasis, state: PolshynWangHFState) -> None:
    nt = int(basis.n_spin) * int(basis.n_eta) * int(basis.nb)
    nk = int(basis.nk)
    expected_matrix_shape = (nt, nt, nk)
    for name in ("h0", "density", "hamiltonian"):
        arr = np.asarray(getattr(state, name))
        if arr.shape != expected_matrix_shape:
            raise ValueError(f"Polshyn-Wang state.{name} shape {arr.shape} does not match {expected_matrix_shape}")
    energies = np.asarray(state.energies)
    if energies.shape != (nt, nk):
        raise ValueError(f"Polshyn-Wang state.energies shape {energies.shape} does not match {(nt, nk)}")
    expected_h0 = _core_flatten_sector_blocks(np.asarray(basis.h0_blocks, dtype=np.complex128))
    if not np.allclose(np.asarray(state.h0, dtype=np.complex128), expected_h0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("Polshyn-Wang state.h0 does not match flatten_sector_blocks(basis.h0_blocks)")

def _polshyn_projected_basis_contract(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState,
) -> ContractProjectedBasis:
    if basis.k_grid_frac is None:
        raise ValueError(
            "Polshyn-Wang canonical ProjectedBasis requires basis.k_grid_frac; "
            "the adapter does not reconstruct or guess a k-grid."
        )
    k_grid_frac = np.asarray(basis.k_grid_frac, dtype=float)
    if k_grid_frac.size != int(basis.nk) * 2:
        raise ValueError(f"basis.k_grid_frac shape {k_grid_frac.shape} incompatible with nk={basis.nk}")
    model = _polshyn_single_particle_model(basis)
    lower_folded_count = 2 * sum(1 for index in basis.projected_indices if int(index) < int(basis.target_band_index))
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(basis.kvec, dtype=np.complex128),
        k_grid_frac=k_grid_frac.reshape((int(basis.nk), 2)),
        h0=np.asarray(state.h0, dtype=np.complex128),
        basis_energies=_basis_energies_from_flat_h0(state.h0),
        active_band_indices=_polshyn_active_band_indices(basis),
        active_valence_bands=int(lower_folded_count),
        active_conduction_bands=int(basis.nb - lower_folded_count),
        micro_wavefunctions=np.asarray(basis.wavefunctions, dtype=np.complex128),
        flavor_labels=_polshyn_flavor_labels(basis),
        band_labels=_polshyn_folded_band_labels(basis),
        metadata={
            "projected_basis_source": "PolshynProjectedBasis",
            "wavefunctions_axis_order": "basis,folded_band,valley,k",
            "density_axis_order": "abk",
            "density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "active_band_semantics": "primitive_projected_indices_repeated_over_folds_spin_valley",
            "projected_indices": [int(value) for value in basis.projected_indices],
            "target_band_index": int(basis.target_band_index),
            "supercell": basis.supercell.as_dict(),
            "supercell_reciprocal_vectors_nm_inv": [
                [float(basis.super_b1.real), float(basis.super_b1.imag)],
                [float(basis.super_b2.real), float(basis.super_b2.imag)],
            ],
            "embedding_shape": [int(value) for value in basis.embedding_shape],
            "embedding_origin": [int(value) for value in basis.embedding_origin],
            "supports_crpa": False,
        },
    )

def _round_integer(value: float, *, name: str, atol: float = 1.0e-7) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - float(rounded)) > float(atol):
        raise ValueError(f"{name}={value:.12g} is not integer within atol={atol}")
    return rounded

def _polshyn_wang_density_state(basis: PolshynProjectedBasis, state: PolshynWangHFState) -> ContractDensityState:
    density_delta = np.asarray(state.density, dtype=np.complex128)
    reference = _polshyn_reference_density_flat(basis)
    projector = density_delta + reference
    trace_projector_total = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    n_occupied_total = _round_integer(trace_projector_total, name="Polshyn-Wang projector trace total")
    trace_delta_per_k = np.trace(density_delta, axis1=0, axis2=1).real
    primitive_nu_per_k = np.asarray(trace_delta_per_k, dtype=float) / float(basis.supercell.area_ratio)
    primitive_nu = float(np.mean(primitive_nu_per_k))
    max_nu_deviation = float(np.max(np.abs(primitive_nu_per_k - primitive_nu))) if primitive_nu_per_k.size else 0.0
    return density_state_from_delta(
        density_delta,
        reference,
        reference_scheme="custom",
        filling=primitive_nu,
        n_occupied_total=n_occupied_total,
        reference_metadata={
            "system": "tmbg_polshyn_doubled",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_scheme_source": "PolshynProjectedBasis.reference_diagonal",
            "reference_diagonal": [float(value) for value in np.asarray(basis.reference_diagonal, dtype=float).reshape(-1)],
            "area_ratio": int(basis.supercell.area_ratio),
            "convention": "Polshyn conduction-band filling: lower remote filled, target empty",
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P_store - R",
            "density_axis_order": "abk",
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "canonical_density_orientation": "stored_abk",
            "adapter": "mean_field.systems.tmbg.polshyn_supercell.polshyn_wang_hf_bundle_to_hf_run_result",
            "primitive_nu_from_density": primitive_nu,
            "primitive_nu_per_k_max_deviation": max_nu_deviation,
        },
    )

def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))

def _polshyn_hamiltonian_parts(state: PolshynWangHFState) -> ContractHamiltonianParts:
    h0 = np.asarray(state.h0, dtype=np.complex128)
    total = np.asarray(state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="polshyn_wang_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "supports_crpa": False,
        },
    )

def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool | np.bool_):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None

def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_float_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out

def _info_scalar_summary(info: Mapping[str, Any]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in info.items():
        if str(key) == "iteration_history":
            continue
        if value is None or isinstance(value, bool | np.bool_ | str):
            out[str(key)] = None if value is None else (bool(value) if isinstance(value, bool | np.bool_) else str(value))
            continue
        if isinstance(value, int | np.integer):
            out[str(key)] = int(value)
            continue
        finite = _finite_float_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out

def _coerce_iteration_history_value(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    finite = _finite_float_or_none(value)
    if finite is not None:
        return finite
    raise TypeError(f"Unsupported iteration_history value type {type(value).__name__}")

def _iteration_history_from_info(info: Mapping[str, Any]) -> list[dict[str, object]]:
    if "iteration_history" not in info:
        return []
    raw = info["iteration_history"]
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("info['iteration_history'] must be an explicit sequence of mapping rows")
    history: list[dict[str, object]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"info['iteration_history'][{row_index}] is not a mapping")
        history.append({str(key): _coerce_iteration_history_value(value) for key, value in row.items()})
    return history

def _require_info_key(info: Mapping[str, Any], key: str) -> Any:
    if key not in info:
        raise ValueError(f"Polshyn-Wang canonical adapter requires info[{key!r}]; refusing to fabricate it")
    return info[key]

def _require_info_bool(info: Mapping[str, Any], key: str) -> bool:
    value = _require_info_key(info, key)
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"Polshyn-Wang info[{key!r}] must be bool, got {type(value).__name__}")
    return bool(value)

def _resolve_polshyn_seed(info: Mapping[str, Any], explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    for key in ("best_seed", "seed"):
        if key in info:
            return int(info[key])
    raise ValueError(
        "Polshyn-Wang canonical adapter requires an explicit seed or info['seed']; "
        "refusing to invent best_seed"
    )

def polshyn_wang_hf_bundle_to_hf_run_result(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState,
    info: Mapping[str, Any],
    *,
    seed: int | None = None,
    archive_manifest: Mapping[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap an explicit ``(basis, state, info)`` Polshyn-Wang HF bundle.

    This is a boundary-only canonical I/O adapter.  It preserves the Wang/Xiaoyu
    stored-density orientation used by :class:`PolshynWangHFState`, records that
    orientation in metadata, and never reconstructs missing iteration history or
    full-state archives.  If ``info`` does not contain an explicit
    ``iteration_history`` sequence, the returned history is deliberately empty.
    """

    _validate_polshyn_wang_bundle_shapes(basis, state)
    info_map = dict(info)
    density = _polshyn_wang_density_state(basis, state)
    iteration_history = _iteration_history_from_info(info_map)
    history_source = "info.iteration_history" if "iteration_history" in info_map else "unavailable_in_polshyn_wang_info"
    diagnostics = _float_diagnostics(state.diagnostics)
    diagnostics.update(_float_diagnostics(info_map))
    final_state = ContractHFState(
        basis=_polshyn_projected_basis_contract(basis, state),
        density=density,
        hamiltonian=_polshyn_hamiltonian_parts(state),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables={
            "eigenvectors_active_available": False,
            "primitive_nu": float(density.filling),
            "filling_from_density": float(density.filling),
            "iteration_history_available": bool(iteration_history),
            "iteration_history_source": history_source,
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "info_summary": _info_scalar_summary(info_map),
        },
        diagnostics=diagnostics,
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=iteration_history,
        converged=_require_info_bool(info_map, "converged"),
        exit_reason=str(_require_info_key(info_map, "exit_reason")),
        best_seed=_resolve_polshyn_seed(info_map, seed),
        init_mode=str(_require_info_key(info_map, "init_mode")),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )









def reference_diagonal_for_projected_indices(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    """Reference density for Polshyn's conduction-band filling convention.

    The experimental/paper filling ``nu=7/2`` counts electrons added into the
    target conduction C=2 band.  Therefore the target band reference is empty,
    not half-filled as in charge-neutral two-flat-band TBG conventions.  Remote
    bands below the target are part of the subtraction-method sea and are filled
    in the reference; remote bands above the target are empty.
    """

    return folded_reference_diagonal_by_primitive_index(
        tuple(int(index) for index in projected_indices),
        target_band_index=int(target_band_index),
        folds_per_primitive=2,
        lower_reference=1.0,
        target_reference=0.0,
        upper_reference=0.0,
    )


def occupation_counts_nu_7over2(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    indices = tuple(int(index) for index in projected_indices)
    target_fold_indices = folded_indices_for_primitive_band(
        indices,
        target_band_index=int(target_band_index),
        folds_per_primitive=2,
    )
    if len(target_fold_indices) != 2:
        raise ValueError("Expected exactly one primitive target band, folded into two supercell bands")
    lower_count = sum(1 for index in indices if int(index) < int(target_band_index))
    full = 2 * int(lower_count) + len(target_fold_indices)
    partial = 2 * int(lower_count) + 1
    return fixed_sector_occupation_counts(
        n_spin=2,
        n_eta=2,
        default_count=full,
        overrides={(0, 0): partial},
        n_band=2 * len(indices),
    )


def primitive_nu_from_counts(occupation_counts: np.ndarray, reference_diagonal: np.ndarray, *, area_ratio: int) -> float:
    return primitive_filling_from_occupation_counts(
        occupation_counts,
        reference_diagonal=reference_diagonal,
        area_ratio=int(area_ratio),
        n_band=int(np.asarray(reference_diagonal, dtype=float).size),
    )



def polshyn_nu_7over2_filling_summary(
    projected_indices: tuple[int, ...],
    *,
    target_band_index: int,
    area_ratio: int = 2,
) -> PolshynFillingSummary:
    indices = tuple(int(index) for index in projected_indices)
    target = int(target_band_index)
    if target not in indices:
        raise ValueError(f"target_band_index={target} is not present in projected_indices={indices}")
    target_position = indices.index(target)
    target_fold_indices = folded_indices_for_primitive_band(
        indices,
        target_band_index=target,
        folds_per_primitive=2,
    )
    reference = reference_diagonal_for_projected_indices(indices, target)
    counts = occupation_counts_nu_7over2(indices, target)
    primitive_nu = primitive_nu_from_counts(counts, reference, area_ratio=int(area_ratio))
    return PolshynFillingSummary(
        projected_indices=indices,
        target_band_index=target,
        target_primitive_position=int(target_position),
        target_fold_indices=target_fold_indices,
        nb=2 * len(indices),
        area_ratio=int(area_ratio),
        reference_diagonal=reference,
        occupation_counts=counts,
        primitive_nu=float(primitive_nu),
        matches_expected_filling=bool(np.isclose(primitive_nu, 3.5, atol=1.0e-12)),
    )













def cdw_density_blocks(
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    n_spin: int,
    n_eta: int,
    nb: int,
    nk: int,
    reference_diagonal: np.ndarray,
) -> np.ndarray:
    """Maximal translation-breaking initializer for the K+ spin-up target band."""

    projected_indices = tuple(int(index) for index in projected_indices)
    target_primitive_pos = projected_indices.index(int(target_band_index))
    target_fold_indices = (2 * target_primitive_pos, 2 * target_primitive_pos + 1)
    reference = np.diag(np.asarray(reference_diagonal, dtype=float)).astype(np.complex128)
    density = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
            # Lower remote bands are filled in the reference and stay filled in the initializer.
            for iprim, band_index in enumerate(projected_indices):
                if int(band_index) < int(target_band_index):
                    projector[2 * iprim, 2 * iprim] = 1.0
                    projector[2 * iprim + 1, 2 * iprim + 1] = 1.0
            if ispin == 0 and ieta == 0:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 0.5
                projector[i1, i1] = 0.5
                projector[i0, i1] = 0.5
                projector[i1, i0] = 0.5
            else:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 1.0
                projector[i1, i1] = 1.0
            for ik in range(int(nk)):
                density[ispin, ieta, :, :, ik] = projector - reference
    return density














def wang_stored_density_from_sector_blocks(density_blocks: np.ndarray) -> np.ndarray:
    """Convert conventional sector density blocks to Wang/Xiaoyu stored layout."""

    return flatten_sector_blocks(np.conj(np.asarray(density_blocks, dtype=np.complex128)))


def scaled_overlap_blocks(
    overlap_blocks: HFOverlapBlockSet,
    *,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
) -> HFOverlapBlockSet:
    """Return an overlap table with Hartree/Fock kernels rescaled for diagnostics."""

    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={tuple(shift): float(hartree_scale) * float(value) for shift, value in overlap_blocks.hartree_screening.items()},
        fock_screening={tuple(shift): float(fock_scale) * np.asarray(value, dtype=float) for shift, value in overlap_blocks.fock_screening.items()},
    )


def overlap_blocks_with_hartree_q0_zeroed(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    """Return an overlap table with only the uniform Hartree shift removed."""

    hartree = {
        tuple(shift): (0.0 if tuple(shift) == (0, 0) else float(value))
        for shift, value in overlap_blocks.hartree_screening.items()
    }
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=hartree,
        fock_screening=overlap_blocks.fock_screening,
    )





def wang_projected_wavefunction_basis(basis: PolshynProjectedBasis) -> ProjectedWavefunctionBasis:
    """View a Polshyn projected basis in the generic Wang/Xiaoyu HF layout."""

    return ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(basis.wavefunctions, dtype=np.complex128),
        grid_shape=tuple(basis.embedding_shape),
        n_spin=int(basis.n_spin),
        local_basis_size=int(basis.local_basis_size),
        name="polshyn_doubled",
        boundary_mode="zero_fill",
    )


def _flat_sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    return _core_flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)

def flatten_sector_blocks(blocks: np.ndarray) -> np.ndarray:
    """Flatten (spin, valley, band, band, k) blocks to Wang nt x nt x nk layout."""

    return _core_flatten_sector_blocks(blocks)

def unflatten_sector_blocks(flat: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_blocks(flat, n_spin=n_spin, n_eta=n_eta, nb=nb)

def unflatten_sector_energies(flat_energies: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    return _core_unflatten_sector_energies(flat_energies, n_spin=n_spin, n_eta=n_eta, nb=nb)

def wang_density_from_fixed_sector_occupations(
    hamiltonian_flat: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> DensityUpdateResult:
    """Fixed-sector density update in Wang/Xiaoyu's stored-projector convention.

    The generic validated HF kernel stores the transpose/conjugate projector
    convention used by the original Wang/Xiaoyu code, namely ``P_store = P*``.
    This differs from the legacy Polshyn helper, which stored the conventional
    density matrix.  Keeping this builder separate prevents convention mixing.
    """

    h = np.asarray(hamiltonian_flat, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square flattened Hamiltonian, got {h.shape}")
    if nt != int(n_spin) * int(n_eta) * int(nb):
        raise ValueError(f"Flattened dimension {nt} incompatible with {(n_spin, n_eta, nb)}")
    occ = np.asarray(occupation_counts, dtype=int)
    if occ.shape != (int(n_spin), int(n_eta)):
        raise ValueError(f"occupation_counts shape {occ.shape} incompatible with {(n_spin, n_eta)}")
    reference = np.asarray(reference_diagonal, dtype=float)
    if reference.shape != (int(nb),):
        raise ValueError(f"reference_diagonal shape {reference.shape} incompatible with nb={nb}")
    ref_mat = np.diag(reference).astype(np.complex128)
    density = np.zeros_like(h)
    energies = np.zeros((nt, int(nk)), dtype=float)
    sector_energies = np.zeros((int(n_spin), int(n_eta), int(nb), int(nk)), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = _flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            n_occ = int(occ[ispin, ieta])
            for ik in range(int(nk)):
                block_h = h[idx[:, None], idx[None, :], ik]
                block_h = 0.5 * (block_h + block_h.conjugate().T)
                evals, evecs = np.linalg.eigh(block_h)
                energies[idx, ik] = evals
                sector_energies[ispin, ieta, :, ik] = evals
                if n_occ == 0:
                    projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
                elif n_occ == int(nb):
                    projector = np.eye(int(nb), dtype=np.complex128)
                else:
                    vecs = evecs[:, :n_occ]
                    projector = vecs.conj() @ vecs.T
                density[idx[:, None], idx[None, :], ik] = projector - ref_mat
    mu = estimate_fermi_level_from_sector_energies(sector_energies, occ)
    return DensityUpdateResult(density=density, energies=energies, mu=float(mu))


def build_wang_overlap_blocks(
    target: PolshynProjectedBasis,
    source: PolshynProjectedBasis,
    shifts: Iterable[tuple[int, int]],
    gvecs: np.ndarray,
    *,
    epsilon_r: float,
    d_sc_nm: float,
    include_hartree: bool = True,
    include_fock: bool = True,
    progress_prefix: str | None = None,
) -> HFOverlapBlockSet:
    """Build dense generic HF overlap blocks for the Wang/Xiaoyu engine."""

    target_core = wang_projected_wavefunction_basis(target)
    source_core = wang_projected_wavefunction_basis(source)
    shift_tuple = tuple(tuple(shift) for shift in shifts)
    gvec_array = np.asarray(gvecs, dtype=np.complex128)
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for ishift, (shift, gvec) in enumerate(zip(shift_tuple, gvec_array, strict=True), start=1):
        if progress_prefix and (ishift == 1 or ishift == len(shift_tuple) or ishift % 10 == 0):
            print(f"{progress_prefix} wang overlap {ishift}/{len(shift_tuple)} shift={shift}", flush=True)
        overlap = calculate_projected_overlap_between(target_core, source_core, int(shift[0]), int(shift[1]))
        overlaps[shift] = overlap
        if target.nk == source.nk:
            diagonal_overlaps[shift] = diagonal_overlap_blocks(overlap, nt=target_core.nt, nk=target.nk)
        if include_hartree:
            hartree_screening[shift] = screened_coulomb(complex(gvec), epsilon_r=float(epsilon_r), d_sc_nm=float(d_sc_nm))
        if include_fock:
            fock_screening[shift] = screened_coulomb_matrix(
                source.kvec[None, :] - target.kvec[:, None] + complex(gvec),
                epsilon_r=float(epsilon_r),
                d_sc_nm=float(d_sc_nm),
            )
    return HFOverlapBlockSet(
        shifts=shift_tuple,
        gvecs=gvec_array,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_wang_hf_problem(
    state: PolshynWangHFState,
    overlap_blocks: HFOverlapBlockSet,
    *,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray,
    n_spin: int,
    n_eta: int,
    nb: int,
) -> HartreeFockProblem:
    """Build the shared core-HF problem wrapper for Wang/Xiaoyu tMBG HF."""

    def interaction_builder(density_flat_in: np.ndarray) -> np.ndarray:
        return build_projected_interaction_hamiltonian(
            density_flat_in,
            overlap_blocks,
            v0=float(state.v0),
            beta=1.0,
        )

    def density_builder(hamiltonian_flat: np.ndarray) -> DensityUpdateResult:
        return wang_density_from_fixed_sector_occupations(
            hamiltonian_flat,
            occupation_counts,
            reference_diagonal,
            n_spin=n_spin,
            n_eta=n_eta,
            nb=nb,
        )

    return HartreeFockProblem(
        initializer=lambda _state, *, init_mode, seed: None,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=compute_hf_energy,
            oda_delta_interaction_builder=interaction_builder,
            convergence_rule="mixed",
        ),
    )


def run_projected_hf_scf_wang(
    basis: PolshynProjectedBasis,
    *,
    occupation_counts: np.ndarray,
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    v0: float,
    epsilon_r: float,
    d_sc_nm: float,
    max_iter: int = 80,
    precision: float = 1e-6,
    initial_density_blocks: np.ndarray | None = None,
    oda_stall_threshold: float = 1.0e-4,
    progress_prefix: str | None = None,
    overlap_blocks: HFOverlapBlockSet | None = None,
    seed: int = 0,
    hartree_scale: float = 1.0,
    fock_scale: float = 1.0,
    zero_hartree_q0: bool = False,
) -> tuple[PolshynWangHFState, HFOverlapBlockSet, dict[str, Any]]:
    """Run Polshyn projected HF through the generic Wang/Xiaoyu ODA engine."""

    if overlap_blocks is None:
        overlap_blocks = build_wang_overlap_blocks(
            basis,
            basis,
            shifts,
            gvecs,
            epsilon_r=epsilon_r,
            d_sc_nm=d_sc_nm,
            include_hartree=True,
            include_fock=True,
            progress_prefix=progress_prefix,
        )
    if float(hartree_scale) != 1.0 or float(fock_scale) != 1.0:
        overlap_blocks = scaled_overlap_blocks(
            overlap_blocks,
            hartree_scale=float(hartree_scale),
            fock_scale=float(fock_scale),
        )
    if bool(zero_hartree_q0):
        overlap_blocks = overlap_blocks_with_hartree_q0_zeroed(overlap_blocks)
    h0_flat = flatten_sector_blocks(basis.h0_blocks)
    if initial_density_blocks is None:
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        density_flat = init_update.density
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "bm_wang"
    else:
        # Existing initializers create conventional Hermitian density matrices;
        # Wang/Xiaoyu's kernel stores P* instead.
        density_flat = flatten_sector_blocks(np.conj(np.asarray(initial_density_blocks, dtype=np.complex128)))
        init_update = wang_density_from_fixed_sector_occupations(
            h0_flat,
            occupation_counts,
            basis.reference_diagonal,
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
        )
        energies = init_update.energies
        mu = init_update.mu
        init_mode = "provided_wang"

    state = PolshynWangHFState(
        h0=h0_flat.copy(),
        density=density_flat.copy(),
        hamiltonian=h0_flat.copy(),
        energies=np.asarray(energies, dtype=float).copy(),
        mu=float(mu),
        precision=float(precision),
        v0=float(v0),
        diagnostics={},
    )

    problem = build_wang_hf_problem(
        state,
        overlap_blocks,
        occupation_counts=occupation_counts,
        reference_diagonal=basis.reference_diagonal,
        n_spin=basis.n_spin,
        n_eta=basis.n_eta,
        nb=basis.nb,
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
    )
    iteration_history = [
        {
            "iteration": int(idx + 1),
            "energy": float(run.iter_energy[idx]) if idx < len(run.iter_energy) else None,
            "error": float(run.iter_err[idx]) if idx < len(run.iter_err) else None,
            "oda_lambda": float(run.iter_oda[idx]) if idx < len(run.iter_oda) else None,
        }
        for idx in range(max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda)))
    ]
    info: dict[str, Any] = {
        "mode": "polshyn_projected_hf_wang",
        "iterations": int(run.iterations),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "final_raw_norm": float(state.diagnostics.get("final_raw_norm", float("nan"))),
        "init_mode": init_mode,
        "seed": int(seed),
        "precision": float(precision),
        "oda_stall_threshold": float(oda_stall_threshold),
        "final_interaction_norm_ev": float(np.linalg.norm(state.hamiltonian - state.h0)),
        "hf_energy": float(state.diagnostics.get("hf_energy", float("nan"))),
        "hartree_scale": float(hartree_scale),
        "fock_scale": float(fock_scale),
        "zero_hartree_q0": bool(zero_hartree_q0),
        "iteration_history": iteration_history,
    }
    if run.iter_oda.size:
        info["last_oda_lambda"] = float(run.iter_oda[-1])
        info["min_oda_lambda"] = float(np.min(run.iter_oda))
    return state, overlap_blocks, info


def wang_sector_density_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    """Return conventional sector density blocks from a Wang/Xiaoyu HF state."""

    return np.conj(unflatten_sector_blocks(state.density, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb))


def wang_sector_hamiltonian_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_blocks(state.hamiltonian, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)


def wang_sector_energy_blocks(state: PolshynWangHFState, basis: PolshynProjectedBasis) -> np.ndarray:
    return unflatten_sector_energies(state.energies, n_spin=basis.n_spin, n_eta=basis.n_eta, nb=basis.nb)




def translation_order_parameters(
    density_blocks: np.ndarray,
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    spin_index: int = 0,
    valley_index: int = 0,
) -> dict[str, np.ndarray | float]:
    """Fold-off-diagonal CDW order diagnostic for the doubled supercell.

    In the area-2 folded basis, fold 0 and fold 1 differ by the paper's
    translation-breaking wavevector Q=B1.  The quantity analogous to Polshyn
    Eq. (1) is therefore the norm of density-matrix elements connecting even
    and odd folded copies at the same supercell k.  A single target-band state
    (|fold0>+|fold1>)/sqrt(2) has |rho_01|=1/2, so the reported ``*_x2``
    values are normalized to have maximal target-band order near one.
    """

    density = np.asarray(density_blocks, dtype=np.complex128)
    projected_indices = tuple(int(index) for index in projected_indices)
    target_pos = projected_indices.index(int(target_band_index))
    fold0 = np.asarray([2 * iprim for iprim in range(len(projected_indices))], dtype=int)
    fold1 = fold0 + 1
    sector = density[int(spin_index), int(valley_index)]
    target_raw = np.abs(sector[2 * target_pos, 2 * target_pos + 1, :])
    all_raw = np.sqrt(np.sum(np.abs(sector[np.ix_(fold0, fold1, np.arange(sector.shape[-1]))]) ** 2, axis=(0, 1)))
    return {
        "target_raw": np.asarray(target_raw, dtype=float),
        "all_raw": np.asarray(all_raw, dtype=float),
        "target_x2": np.asarray(2.0 * target_raw, dtype=float),
        "all_x2": np.asarray(2.0 * all_raw, dtype=float),
        "target_x2_min": float(np.min(2.0 * target_raw)),
        "target_x2_mean": float(np.mean(2.0 * target_raw)),
        "target_x2_max": float(np.max(2.0 * target_raw)),
        "all_x2_min": float(np.min(2.0 * all_raw)),
        "all_x2_mean": float(np.mean(2.0 * all_raw)),
        "all_x2_max": float(np.max(2.0 * all_raw)),
    }


def estimate_fermi_level_from_sector_energies(energies: np.ndarray, occupation_counts: np.ndarray) -> float:
    vals = np.asarray(energies, dtype=float)
    occ = np.asarray(occupation_counts, dtype=int)
    occupied_max: list[float] = []
    empty_min: list[float] = []
    for ispin in range(vals.shape[0]):
        for ieta in range(vals.shape[1]):
            n_occ = int(occ[ispin, ieta])
            if n_occ > 0:
                occupied_max.append(float(np.max(vals[ispin, ieta, n_occ - 1, :])))
            if n_occ < vals.shape[2]:
                empty_min.append(float(np.min(vals[ispin, ieta, n_occ, :])))
    if occupied_max and empty_min:
        return 0.5 * (max(occupied_max) + min(empty_min))
    if occupied_max:
        return max(occupied_max)
    if empty_min:
        return min(empty_min)
    return 0.0


def moire_cell_area_nm2(lattice: TMBGLattice, *, area_ratio: int = 1) -> float:
    primitive_area = real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)
    return float(area_ratio) * float(primitive_area)
