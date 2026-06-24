from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from analysis.order_parameters import StateLabel, analyze_tdbg_order_parameters

from ...core.contracts import MicroscopicWavefunctionBundle
from ...core.hf import (
    DensityConvention,
    DensityUpdateResult,
    conventional_projector_to_stored,
    density_to_stored_delta,
    stored_projector_to_conventional,
)
from .projected_hf_config import SPIN_LABELS, TDBGProjectedHFConfig, TDBG_LOCAL_LABELS, VALLEY_LABELS, VALLEY_SEQUENCE
from .projected_hf_geometry import tdbg_projected_hf_boundary_sewing_transforms

if TYPE_CHECKING:
    from ...core.hf import HartreeFockRun
    from .model import TDBGModel
    from .params import TDBGParameters

@dataclass(frozen=True)
class TDBGStateLabel:
    index: int
    spin: str
    valley: int
    band_position: int
    band_index: int

    @property
    def valley_label(self) -> str:
        return VALLEY_LABELS.get(int(self.valley), f"valley{self.valley}")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "spin": self.spin,
            "valley": int(self.valley),
            "valley_label": self.valley_label,
            "band_position": int(self.band_position),
            "band_index": int(self.band_index),
        }


@dataclass(frozen=True)
class TDBGProjectedHFData:
    model: TDBGModel
    config: TDBGProjectedHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    band_indices: tuple[int, ...]
    labels: tuple[TDBGStateLabel, ...]
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, nk, n_q, 4)
    reference_density: np.ndarray
    n_occupied_per_k: int
    lower_band_count: int
    moire_area_nm2: float
    shifts: tuple[tuple[int, int], ...]
    shift_gvecs: np.ndarray
    shift_srcmaps: tuple[np.ndarray, ...]
    valley_params: Mapping[int, TDBGParameters] | None = None

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def n_band(self) -> int:
        return int(len(self.band_indices))


@dataclass
class TDBGProjectedHFState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float = float("nan")
    precision: float = 1.0e-7
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class TDBGProjectedHFTargetData:
    kvec: np.ndarray
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, n_target, n_q, 4)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


_TDBG_RAW_WAVEFUNCTION_AXIS_ORDER = "state,k,q_site,local"
_TDBG_CANONICAL_MICRO_AXIS_ORDER = "k,microscopic_basis,active_basis"
_TDBG_MICRO_ROW_ORDER = "spin,valley,q_site,local"
_TDBG_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS = 5_000_000


def _tdbg_label_indices(label: TDBGStateLabel) -> tuple[int, int]:
    try:
        spin_index = SPIN_LABELS.index(str(label.spin))
    except ValueError as exc:
        raise ValueError(f"Unsupported TDBG spin label {label.spin!r}; expected one of {SPIN_LABELS}") from exc
    try:
        valley_index = VALLEY_SEQUENCE.index(int(label.valley))
    except ValueError as exc:
        raise ValueError(f"Unsupported TDBG valley label {label.valley!r}; expected one of {VALLEY_SEQUENCE}") from exc
    return spin_index, valley_index


def _tdbg_micro_basis_metadata(data: TDBGProjectedHFData) -> dict[str, Any]:
    n_q = int(data.model.lattice.n_q)
    n_local = len(TDBG_LOCAL_LABELS)
    mesh = int(getattr(data.config, "mesh_size", 0))
    topology_grid_shape = [mesh, mesh] if mesh > 0 and mesh * mesh == int(data.nk) else None
    return {
        "system": "tdbg",
        "projected_basis_source": "TDBGProjectedHFData.wavefunctions",
        "raw_wavefunctions_axis_order": _TDBG_RAW_WAVEFUNCTION_AXIS_ORDER,
        "wavefunctions_axis_order": _TDBG_CANONICAL_MICRO_AXIS_ORDER,
        "microscopic_basis_axis_order": _TDBG_MICRO_ROW_ORDER,
        "microscopic_basis_flattening": "(((spin_index * n_valley + valley_index) * n_q + q_site) * n_local + local)",
        "spin_order": list(SPIN_LABELS),
        "valley_order": [int(valley) for valley in VALLEY_SEQUENCE],
        "local_order": list(TDBG_LOCAL_LABELS),
        "q_site_count": n_q,
        "local_basis_size": n_local,
        "active_basis_axis_order": "TDBGStateLabel.index; band_position fastest inside valley inside spin",
        "active_basis_labels": [label.to_dict() for label in data.labels],
        "sewing_transforms_available": True,
        "sewing_policy": "available: block-diagonal q_site reciprocal-translation sewing for reconstructed row order spin,valley,q_site,local",
        "sewing_transforms_entrypoint": "mean_field.systems.tdbg.topology.projected_hf_boundary_sewing_transforms",
        "sewing_transform_axes": "acts on first axis of vectors/frames with row order spin,valley,q_site,local; spin and valley are spectator direct-sum blocks",
        "topology_eligible": True,
        "topology_eligibility_scope": "software/API eligibility for FHS on endpoint=False source-grid reconstructed projected-HF bundles; physical Chern validation remains separate",
        "topology_grid_shape": topology_grid_shape,
        "topology_grid_shape_source": "TDBGProjectedHFConfig.mesh_size when mesh_size**2 == nk",
        "topology_ineligible_reason": "",
        "topology_validation_status": "toy row-order sewing tests only; no TDBG projected-HF physical Chern or paper validation has been run here",
        "uncertainty": "Sewing is derived as direct-sum q_site translation; selected HF state/subspace isolation, min-link diagnostics, and physical projected-HF topology validation remain undone.",
        "evidence_paths": [
            "src/mean_field/systems/tdbg/projected_hf_data.py",
            "src/mean_field/systems/tdbg/projected_hf_geometry.py",
            "src/mean_field/systems/tdbg/projected_hf_state.py",
            "src/mean_field/systems/tdbg/projected_hf_contracts.py",
            "src/mean_field/systems/tdbg/topology.py",
        ],
    }


def _tdbg_micro_dim(n_q: int, n_local: int) -> int:
    return int(len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * int(n_q) * int(n_local))

def _tdbg_validated_raw_wavefunctions(data: TDBGProjectedHFData) -> tuple[np.ndarray, int, int, int, int]:
    raw = np.asarray(data.wavefunctions, dtype=np.complex128)
    if raw.ndim != 4:
        raise ValueError(
            f"Expected TDBG raw wavefunctions axis order {_TDBG_RAW_WAVEFUNCTION_AXIS_ORDER!r} with rank 4, got shape {raw.shape}"
        )
    nt, nk, n_q, n_local = (int(v) for v in raw.shape)
    if nt != int(data.nt) or nk != int(data.nk):
        raise ValueError(f"Raw TDBG wavefunctions shape {raw.shape} is incompatible with data.nt={data.nt}, data.nk={data.nk}")
    if n_q != int(data.model.lattice.n_q):
        raise ValueError(f"Raw TDBG q_site axis has length {n_q}, expected lattice.n_q={data.model.lattice.n_q}")
    if n_local != len(TDBG_LOCAL_LABELS):
        raise ValueError(f"Raw TDBG local axis has length {n_local}, expected {len(TDBG_LOCAL_LABELS)}")
    return raw, nt, nk, n_q, n_local

def _tdbg_validated_label_rows(data: TDBGProjectedHFData, *, nt: int, sector_stride: int) -> tuple[tuple[int, int], ...]:
    if len(data.labels) != nt:
        raise ValueError(f"TDBG label count {len(data.labels)} must match active dimension nt={nt}")
    entries: list[tuple[int, int]] = []
    seen: set[int] = set()
    for label in data.labels:
        col = int(label.index)
        if col < 0 or col >= nt:
            raise ValueError(f"TDBG label index {col} is outside active dimension nt={nt}")
        if col in seen:
            raise ValueError(f"Duplicate TDBG label index {col}")
        seen.add(col)
        spin_index, valley_index = _tdbg_label_indices(label)
        row0 = (spin_index * len(VALLEY_SEQUENCE) + valley_index) * int(sector_stride)
        entries.append((col, row0))
    if len(seen) != nt:
        missing = sorted(set(range(nt)) - seen)
        raise ValueError(f"TDBG labels did not cover active indices {missing}")
    return tuple(entries)

def tdbg_canonical_projected_micro_basis(data: TDBGProjectedHFData) -> np.ndarray:
    """Expand raw TDBG projected states to ``(k, microscopic_basis, active_basis)``.

    Raw TDBG projected-HF stores ``data.wavefunctions[state, k, q_site, local]``.
    The ``state`` axis is the active projected basis with labels carrying spin,
    valley, and band.  For microscopic reconstruction, spin and valley sectors
    must be orthogonal row blocks, so the canonical row index is
    ``(((spin * n_valley + valley) * n_q + q_site) * n_local + local)`` while
    the active column remains ``TDBGStateLabel.index``.
    """

    raw, nt, nk, n_q, n_local = _tdbg_validated_raw_wavefunctions(data)
    sector_stride = n_q * n_local
    micro_dim = _tdbg_micro_dim(n_q, n_local)
    canonical = np.zeros((nk, micro_dim, nt), dtype=np.complex128)
    for col, row0 in _tdbg_validated_label_rows(data, nt=nt, sector_stride=sector_stride):
        canonical[:, row0 : row0 + sector_stride, col] = raw[col].reshape(nk, sector_stride)
    return canonical


def tdbg_active_eigensystem_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    hermiticity_atol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Diagonalize final TDBG active Hamiltonians as ket coefficients.

    Returns eigenvalues ``(active_state, k)`` and eigenvectors
    ``(active_basis, hf_state, k)`` in the same active-basis order as
    ``TDBGStateLabel.index``.  These are ket coefficients from ``np.linalg.eigh``;
    no density-projector transpose/conjugation convention is used.  Non-Hermitian
    final Hamiltonians are rejected instead of being silently symmetrized.
    """

    hermiticity_atol_value = float(hermiticity_atol)
    if hermiticity_atol_value < 0.0:
        raise ValueError(f"hermiticity_atol must be non-negative, got {hermiticity_atol}")
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Expected TDBG Hamiltonian shape (nt, nt, nk), got {hamiltonian.shape}")
    nt, _nt_rhs, nk = (int(v) for v in hamiltonian.shape)
    max_hermitian_residual = (
        float(np.max(np.abs(hamiltonian - hamiltonian.conjugate().swapaxes(0, 1)))) if hamiltonian.size else 0.0
    )
    if max_hermitian_residual > hermiticity_atol_value:
        raise ValueError(
            "TDBG final Hamiltonian is not Hermitian enough for reconstruction; "
            f"max residual {max_hermitian_residual:.6e} exceeds hermiticity_atol={hermiticity_atol_value:.6e}. "
            "Refusing to symmetrize silently."
        )
    energies = np.zeros((nt, nk), dtype=float)
    eigenvectors = np.zeros((nt, nt, nk), dtype=np.complex128)
    max_eigen_residual = 0.0
    max_unitarity_residual = 0.0
    identity = np.eye(nt, dtype=np.complex128)
    for ik in range(nk):
        block = hamiltonian[:, :, ik]
        vals, vecs = np.linalg.eigh(block)
        energies[:, ik] = vals
        eigenvectors[:, :, ik] = vecs
        residual = block @ vecs - vecs * vals[None, :]
        max_eigen_residual = max(max_eigen_residual, float(np.max(np.abs(residual))) if residual.size else 0.0)
        unitary_residual = vecs.conjugate().T @ vecs - identity
        max_unitarity_residual = max(
            max_unitarity_residual,
            float(np.max(np.abs(unitary_residual))) if unitary_residual.size else 0.0,
        )
    return energies, eigenvectors, {
        "active_hamiltonian_hermitian_residual": max_hermitian_residual,
        "active_hamiltonian_hermiticity_atol": hermiticity_atol_value,
        "active_hamiltonian_hermiticity_policy": "reject_without_symmetrization",
        "active_eigensystem_residual": max_eigen_residual,
        "active_eigenvector_unitarity_residual": max_unitarity_residual,
    }


def _tdbg_normalize_reconstruction_state_indices(
    *,
    state_indices: int | Iterable[int] | None,
    band_indices: int | Iterable[int] | None,
    n_state: int,
) -> tuple[tuple[int, ...], str]:
    if state_indices is not None and band_indices is not None:
        raise ValueError("Pass only one of state_indices or band_indices for TDBG reconstruction")
    source = "all"
    raw_indices: int | Iterable[int] | None = None
    if state_indices is not None:
        source = "state_indices"
        raw_indices = state_indices
    elif band_indices is not None:
        source = "band_indices"
        raw_indices = band_indices
    if raw_indices is None:
        selected = tuple(range(int(n_state)))
    elif isinstance(raw_indices, (int, np.integer)):
        selected = (int(raw_indices),)
    else:
        selected = tuple(int(index) for index in raw_indices)
    if not selected:
        raise ValueError("TDBG reconstruction requires at least one selected HF state")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate TDBG reconstruction state indices {selected}")
    invalid = [int(index) for index in selected if int(index) < 0 or int(index) >= int(n_state)]
    if invalid:
        raise ValueError(f"TDBG reconstruction state indices {invalid} are outside [0, {int(n_state)})")
    return selected, source

def _tdbg_reconstruction_dense_element_count(data: TDBGProjectedHFData, *, n_selected: int) -> int:
    micro_dim = _tdbg_micro_dim(int(data.model.lattice.n_q), len(TDBG_LOCAL_LABELS))
    return int(data.nk) * int(micro_dim) * int(n_selected)

def _tdbg_validate_reconstruction_size(
    data: TDBGProjectedHFData,
    *,
    n_selected: int,
    max_dense_elements: int | None,
) -> int:
    dense_elements = _tdbg_reconstruction_dense_element_count(data, n_selected=n_selected)
    if max_dense_elements is None:
        return dense_elements
    max_elements = int(max_dense_elements)
    if max_elements < 0:
        raise ValueError("max_dense_elements must be non-negative or None")
    if dense_elements > max_elements:
        raise ValueError(
            "TDBG projected-HF dense reconstruction would exceed the explicit size guard: "
            f"estimated {dense_elements} complex elements for {int(n_selected)} selected HF states "
            f"> max_dense_elements={max_elements}. Pass selected state_indices/band_indices or increase "
            "max_dense_elements only for an intentional reconstruction call."
        )
    return dense_elements

def _tdbg_selected_eigenvector_unitarity_residual(eigenvectors: np.ndarray, selected: tuple[int, ...]) -> float:
    coeffs = eigenvectors[:, np.asarray(selected, dtype=int), :]
    gram = np.einsum("ahk,amk->hmk", coeffs.conjugate(), coeffs, optimize=True)
    identity = np.eye(len(selected), dtype=np.complex128)[:, :, None]
    return float(np.max(np.abs(gram - identity))) if gram.size else 0.0

def _tdbg_contract_selected_micro_wavefunctions(
    data: TDBGProjectedHFData,
    eigenvectors: np.ndarray,
    selected: tuple[int, ...],
) -> np.ndarray:
    raw, nt, nk, n_q, n_local = _tdbg_validated_raw_wavefunctions(data)
    if eigenvectors.shape != (nt, nt, nk):
        raise ValueError(f"TDBG active eigenvectors must have shape ({nt}, {nt}, {nk}), got {eigenvectors.shape}")
    sector_stride = n_q * n_local
    micro_dim = _tdbg_micro_dim(n_q, n_local)
    selected_array = np.asarray(selected, dtype=int)
    psi = np.zeros((nk, micro_dim, len(selected)), dtype=np.complex128)
    for col, row0 in _tdbg_validated_label_rows(data, nt=nt, sector_stride=sector_stride):
        raw_block = raw[col].reshape(nk, sector_stride)
        psi[:, row0 : row0 + sector_stride, :] += np.einsum(
            "kb,hk->kbh",
            raw_block,
            eigenvectors[col, selected_array, :],
            optimize=True,
        )
    return psi

def reconstruct_tdbg_projected_hf_micro_wavefunctions(
    result: "TDBGProjectedHFResult",
    *,
    state_indices: int | Iterable[int] | None = None,
    band_indices: int | Iterable[int] | None = None,
    max_dense_elements: int | None = _TDBG_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS,
    hermiticity_atol: float = 1.0e-8,
    unitarity_atol: float | None = 1.0e-8,
) -> MicroscopicWavefunctionBundle:
    """Return a core microscopic-wavefunction bundle for a TDBG projected-HF result.

    ``state_indices`` and ``band_indices`` select HF eigenstate columns after
    ``np.linalg.eigh`` sorting.  ``band_indices`` is an API-compatibility alias;
    it is not the noninteracting ``TDBGProjectedHFData.band_indices`` label.
    """

    data = result.data
    selected, selection_source = _tdbg_normalize_reconstruction_state_indices(
        state_indices=state_indices,
        band_indices=band_indices,
        n_state=data.nt,
    )
    dense_elements = _tdbg_validate_reconstruction_size(
        data,
        n_selected=len(selected),
        max_dense_elements=max_dense_elements,
    )
    energies, eigenvectors, eigensystem_metadata = tdbg_active_eigensystem_from_hamiltonian(
        result.run.state.hamiltonian,
        hermiticity_atol=hermiticity_atol,
    )
    selected_unitarity_residual = _tdbg_selected_eigenvector_unitarity_residual(eigenvectors, selected)
    if unitarity_atol is not None:
        unitarity_atol_value = float(unitarity_atol)
        if unitarity_atol_value < 0.0:
            raise ValueError(f"unitarity_atol must be non-negative or None, got {unitarity_atol}")
        if selected_unitarity_residual > unitarity_atol_value:
            raise ValueError(
                "Selected TDBG active eigenvectors are not unitary enough for reconstruction; "
                f"max column-Gram residual {selected_unitarity_residual:.6e} exceeds {unitarity_atol_value:.6e}"
            )
    kvec = np.asarray(data.kvec, dtype=np.complex128).reshape(-1)
    if kvec.shape != (data.nk,):
        raise ValueError(f"TDBG kvec must have shape ({data.nk},), got {kvec.shape}")
    try:
        k_grid_frac = np.asarray(data.k_grid_frac, dtype=float).reshape((data.nk, 2))
    except ValueError as exc:
        raise ValueError(f"TDBG k_grid_frac must reshape to ({data.nk}, 2), got {np.asarray(data.k_grid_frac).shape}") from exc
    psi = _tdbg_contract_selected_micro_wavefunctions(data, eigenvectors, selected)
    sewing_transforms = tdbg_projected_hf_boundary_sewing_transforms(data.model.lattice)
    energy_residual = None
    stored_energies = np.asarray(getattr(result.run.state, "energies", np.empty((0,))), dtype=float)
    if stored_energies.shape == energies.shape:
        energy_residual = float(np.max(np.abs(stored_energies - energies)))
    state_labels = tuple({"hf_state_index": int(index)} for index in selected)
    metadata = _tdbg_micro_basis_metadata(data)
    metadata.update(eigensystem_metadata)
    if energy_residual is not None:
        metadata["stored_energy_eigensystem_residual"] = energy_residual
    metadata.update(
        {
            "active_eigenvectors_source": "np.linalg.eigh(result.run.state.hamiltonian)",
            "active_eigenvectors_axis_order": "active_basis,hf_state,k",
            "selected_active_eigenvectors_unitarity_residual": selected_unitarity_residual,
            "hf_state_gauge": "np.linalg.eigh column phases; degenerate subspaces are not gauge-fixed",
            "reconstruction_adapter": "mean_field.systems.tdbg.projected_hf_state.reconstruct_tdbg_projected_hf_micro_wavefunctions",
            "projected_hf_reconstruction": "explicit_selected_dense_opt_in",
            "canonical_wrapping_dense_by_default": False,
            "canonical_micro_basis_materialized": False,
            "dense_reconstruction_estimated_elements": int(dense_elements),
            "dense_reconstruction_size_policy": "counts selected output psi_micro elements; all-state psi_micro is not materialized for selected calls",
            "max_dense_elements": None if max_dense_elements is None else int(max_dense_elements),
            "selection_argument": selection_source,
            "selected_hf_state_indices": [int(index) for index in selected],
            "selected_hf_band_indices": [int(index) for index in selected],
            "band_indices_argument_meaning": "HF eigenstate indices after np.linalg.eigh sorting, not TDBGProjectedHFData.band_indices",
            "all_hf_state_count": int(data.nt),
            "n_reconstructed_states": int(len(selected)),
            "state_labels": state_labels,
            "micro_basis_axis_order": _TDBG_CANONICAL_MICRO_AXIS_ORDER,
            "input_micro_basis_axes": {"k_axis": 0, "microscopic_basis_axis": 1, "active_axis": 2},
            "psi_micro_axis_order": "k,microscopic_basis,hf_state",
            "n_k": int(data.nk),
            "microscopic_basis_dim": int(psi.shape[1]),
            "n_active": int(data.nt),
            "kvec_provided": True,
            "k_grid_frac_shape": [int(k_grid_frac.shape[0]), int(k_grid_frac.shape[1])],
            "sewing_transforms_count": int(len(sewing_transforms)),
        }
    )
    return MicroscopicWavefunctionBundle(
        kvec=kvec,
        psi_micro=psi,
        sewing_transforms=sewing_transforms,
        basis_metadata=metadata,
        source="hf_reconstructed",
    )


@dataclass(frozen=True)
class TDBGProjectedHFResult:
    run: HartreeFockRun
    data: TDBGProjectedHFData
    init_mode: str
    seed: int
    order_parameters: dict[str, object]
    energy_components: dict[str, float]
    hamiltonian_components: Mapping[str, np.ndarray] | None = None

    def reconstruct_micro_wavefunctions(
        self,
        *,
        state_indices: int | Iterable[int] | None = None,
        band_indices: int | Iterable[int] | None = None,
        max_dense_elements: int | None = _TDBG_RECONSTRUCTION_DEFAULT_MAX_DENSE_ELEMENTS,
        hermiticity_atol: float = 1.0e-8,
        unitarity_atol: float | None = 1.0e-8,
    ):
        """Public HFResult state adapter for TDBG projected-HF wavefunctions."""

        core_bundle = reconstruct_tdbg_projected_hf_micro_wavefunctions(
            self,
            state_indices=state_indices,
            band_indices=band_indices,
            max_dense_elements=max_dense_elements,
            hermiticity_atol=hermiticity_atol,
            unitarity_atol=unitarity_atol,
        )
        metadata = dict(core_bundle.basis_metadata)
        metadata.update(
            {
                "source": core_bundle.source,
                "reconstruction_path": "TDBGProjectedHFResult.reconstruct_micro_wavefunctions",
                "core_reconstruction_sewing_transforms_available": bool(core_bundle.sewing_transforms),
                "core_reconstruction_sewing_transforms_count": int(len(core_bundle.sewing_transforms)),
                "sewing_transforms_available": False,
                "sewing_transforms_count": 0,
                "sewing_policy": "not attached to public flat WavefunctionBundle; use mean_field.systems.tdbg.topology.compute_projected_hf_topology",
                "topology_eligible": False,
                "topology_eligibility_scope": "public WavefunctionBundle is flat (nk,basis,state) and does not carry sewing_transforms; use mean_field.systems.tdbg.topology.compute_projected_hf_topology for FHS topology",
                "topology_ineligible_reason": "public API WavefunctionBundle drops core sewing transforms and is not reshaped to (mesh,mesh,basis,state); TDBG topology must use mean_field.systems.tdbg.topology.compute_projected_hf_topology, which reconstructs, reshapes, and passes sewing transforms explicitly",
            }
        )
        from mean_field.api import ConventionBundle, WavefunctionBundle

        return WavefunctionBundle(
            k=core_bundle.kvec,
            wavefunctions=core_bundle.psi_micro,
            metadata=metadata,
            convention=ConventionBundle(
                density_convention="projector",
                wavefunction_axis_order=str(metadata.get("psi_micro_axis_order", "k,microscopic_basis,hf_state")),
                gauge="tdbg_projected_hf_system_defined",
            ),
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "init_mode": self.init_mode,
            "seed": int(self.seed),
            "converged": bool(self.run.converged),
            "exit_reason": self.run.exit_reason,
            "iterations": int(self.run.iterations),
            "final_error": float(self.run.state.diagnostics.get("final_raw_norm", np.nan)),
            "hf_energy_ev": float(self.run.state.diagnostics.get("hf_energy", np.nan)),
            "order_parameters": self.order_parameters,
            "energy_components_ev": self.energy_components,
        }

def _conventional_projector_to_stored(projector: np.ndarray) -> np.ndarray:
    return conventional_projector_to_stored(projector)


def _stored_to_conventional(stored: np.ndarray) -> np.ndarray:
    return stored_projector_to_conventional(stored)


def _first_conduction_indices(data: TDBGProjectedHFData) -> list[int]:
    if data.n_band == 1:
        position = 0
    else:
        position = data.lower_band_count
    return [label.index for label in data.labels if label.band_position == position]

def _active_filling_indices(data: TDBGProjectedHFData) -> list[int]:
    filling = int(data.config.filling)
    if data.n_band == 1:
        position = 0
    elif filling >= 0:
        position = data.lower_band_count
    else:
        if data.lower_band_count <= 0:
            raise ValueError("Negative TDBG filling requires at least one valence band in the projected window")
        position = data.lower_band_count - 1
    return [label.index for label in data.labels if label.band_position == position]

def _reference_projector(data: TDBGProjectedHFData) -> np.ndarray:
    projector = np.zeros((data.nt, data.nt), dtype=np.complex128)
    for label in data.labels:
        if label.band_position < data.lower_band_count:
            projector[label.index, label.index] = 1.0
    return projector

def initialize_tdbg_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Return an absolute occupied projector in the core stored convention.

    The initializer supports positive and negative fillings relative to the
    charge-neutral reference. Positive fillings add projectors in the first
    conduction band; negative fillings remove hole projectors from the highest
    valence band. The unrestricted density builder then refills the lowest HF
    eigenstates at the configured occupation count.
    """

    mode = init_mode.strip().lower().replace("-", "_")
    nk = data.nk
    nt = data.nt
    filling = int(data.config.filling)
    count = abs(filling)
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    active_labels = [data.labels[idx] for idx in _active_filling_indices(data)]
    rng = np.random.default_rng(seed)

    def apply_active_projectors(projector: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        if len(projectors) != count:
            raise ValueError(f"init_mode={init_mode!r} produced {len(projectors)} projectors for filling {filling}")
        for active_projector in projectors:
            if filling >= 0:
                projector += active_projector
            else:
                projector -= active_projector
        return projector

    def basis_projectors(indices: list[int]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for idx in indices:
            vec = np.zeros(nt, dtype=np.complex128)
            vec[int(idx)] = 1.0
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def coherent_projectors(phase: complex, *, k_weight: float = 0.5) -> list[np.ndarray]:
        k_weight = float(k_weight)
        if not 0.0 < k_weight < 1.0:
            raise ValueError(f"IVC valley weight must be in (0, 1), got {k_weight}")
        kp_weight = 1.0 - k_weight
        out: list[np.ndarray] = []
        for spin in SPIN_LABELS:
            states = [label.index for label in active_labels if label.spin == spin]
            if len(states) != 2:
                raise ValueError("IVC initializer requires exactly two valley states per spin in the active filling band")
            vec = np.zeros(nt, dtype=np.complex128)
            vec[states[0]] = math.sqrt(k_weight)
            vec[states[1]] = complex(phase) * math.sqrt(kp_weight)
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def parse_ivc_weight_token(token: str) -> float:
        if not token:
            raise ValueError(f"Biased IVC initializer {init_mode!r} must include a valley weight, e.g. ivc_k85")
        value = float(token.replace("p", "."))
        if value > 1.0:
            value /= 100.0
        if not 0.0 < value < 1.0:
            raise ValueError(f"Biased IVC valley weight must be in (0, 1), got {value} from {init_mode!r}")
        return value

    def biased_coherent_projectors_from_mode() -> list[np.ndarray]:
        phase: complex = 1.0j if mode.endswith("_odd") else 1.0
        base = mode[:-4] if mode.endswith("_odd") else mode
        if base.startswith("ivc_kprime"):
            kp_weight = parse_ivc_weight_token(base[len("ivc_kprime") :])
            return coherent_projectors(phase, k_weight=1.0 - kp_weight)
        if base.startswith("ivc_k"):
            k_weight = parse_ivc_weight_token(base[len("ivc_k") :])
            return coherent_projectors(phase, k_weight=k_weight)
        raise ValueError(f"Unsupported biased IVC initializer {init_mode!r}")

    def random_projectors() -> list[np.ndarray]:
        states = [label.index for label in active_labels]
        if count > len(states):
            raise ValueError(f"Cannot choose {count} active projectors from {len(states)} states")
        z = rng.standard_normal((len(states), len(states))) + 1j * rng.standard_normal((len(states), len(states)))
        herm = z + z.conjugate().T
        _, vecs = np.linalg.eigh(herm)
        out: list[np.ndarray] = []
        for col in range(count):
            vec = np.zeros(nt, dtype=np.complex128)
            vec[np.asarray(states, dtype=int)] = vecs[:, col]
            out.append(np.outer(vec, vec.conjugate()))
        return out

    for ik in range(nk):
        projector = _reference_projector(data)
        if filling == 0:
            pass
        elif mode in {"sp", "sp_up"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "up"]))
        elif mode == "sp_down":
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "down"]))
        elif mode in {"vp", "vp_k"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == 1]))
        elif mode in {"vp_kprime", "vp_kp"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == -1]))
        elif mode in {"ivc", "ivc_even"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0))
        elif mode in {"ivc_odd", "kivc"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0j))
        elif mode.startswith("ivc_k"):
            projector = apply_active_projectors(projector, biased_coherent_projectors_from_mode())
        elif mode in {"random", "random_flavor"}:
            projector = apply_active_projectors(projector, random_projectors())
        elif mode in {"bm", "noninteracting"}:
            projector = np.zeros((nt, nt), dtype=np.complex128)
            evals = np.real(np.diag(data.h0[:, :, ik]))
            for idx in np.argsort(evals, kind="stable")[: data.n_occupied_per_k]:
                projector[int(idx), int(idx)] = 1.0
        else:
            raise ValueError(
                f"Unsupported TDBG projected-HF init_mode={init_mode!r}. "
                "Use sp, sp_down, vp_k, vp_kprime, ivc_even, ivc_odd, ivc_k85, ivc_kprime85, random, or bm."
            )
        density[:, :, ik] = _conventional_projector_to_stored(projector)
    return density

def initialize_tdbg_nu2_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Backward-compatible alias for the generic TDBG filling initializer."""

    return initialize_tdbg_density(data, init_mode=init_mode, seed=seed)

class TDBGProjectedHFInitializer:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, state: TDBGProjectedHFState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = initialize_tdbg_density(self.data, init_mode=init_mode, seed=seed)
        state.diagnostics.update(_numeric_order_parameters(self.data, state.density))


class TDBGProjectedHFDensityBuilder:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occ_mask = tdbg_density_from_hamiltonian(hamiltonian, self.data.n_occupied_per_k)
        observables = {"occupation_mask": occ_mask}
        observables.update(_numeric_order_parameters(self.data, density))
        return DensityUpdateResult(density=density, energies=energies, mu=mu, observables=observables)


def tdbg_density_from_hamiltonian(hamiltonian: np.ndarray, n_occupied_per_k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    nocc = int(n_occupied_per_k)
    if nocc < 0 or nocc > nt:
        raise ValueError(f"Invalid occupied count per k {nocc} for nt={nt}")
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    energies = np.zeros((nt, nk), dtype=float)
    occ_mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        vals, vecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = vals
        if nocc:
            occupied = vecs[:, :nocc]
            projector = occupied @ occupied.conjugate().T
            density[:, :, ik] = _conventional_projector_to_stored(projector)
            occ_mask[:nocc, ik] = True
    if nocc <= 0 or nocc >= nt:
        mu = float(np.mean(energies))
    else:
        mu = 0.5 * (float(np.max(energies[:nocc, :])) + float(np.min(energies[nocc:, :])))
    return density, energies, float(mu), occ_mask

def _reference_subtracted_tdbg_density(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    return density_to_stored_delta(
        density,
        DensityConvention.PROJECTOR,
        reference=data.reference_density,
        reference_policy="require",
    )


def _hartree_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.hartree_reference == "none":
        return density
    if settings.hartree_reference == "charge_neutral":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Hartree reference policy: {settings.hartree_reference!r}")


def _fock_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.fock_density == "absolute":
        return density
    if settings.fock_density == "reference_subtracted":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Fock density policy: {settings.fock_density!r}")

def _tdbg_order_labels(data: TDBGProjectedHFData) -> tuple[StateLabel, ...]:
    active_indices = set(_active_filling_indices(data))
    return tuple(
        StateLabel(
            index=int(label.index),
            spin=str(label.spin),
            valley=int(label.valley),
            band=int(label.band_index),
            active=bool(label.index in active_indices),
            metadata=label.to_dict(),
        )
        for label in data.labels
    )


def _conventional_projector_stack(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    stored = np.asarray(density, dtype=np.complex128)
    out = np.zeros_like(stored)
    for ik in range(data.nk):
        out[:, :, ik] = _stored_to_conventional(stored[:, :, ik])
    return out


def _numeric_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    result = analyze_tdbg_order_parameters(_conventional_projector_stack(data, density), _tdbg_order_labels(data))
    scalars = dict(result.scalars)
    # Backward-compatible names for the original nu=+2 conduction-band workflow.
    scalars["cb_spin_polarization"] = float(scalars["active_spin_polarization"])
    scalars["cb_valley_polarization"] = float(scalars["active_valley_polarization"])
    return {key: float(value) for key, value in scalars.items()}


def tdbg_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, object]:
    result = analyze_tdbg_order_parameters(_conventional_projector_stack(data, density), _tdbg_order_labels(data))
    numeric = dict(result.scalars)
    numeric["cb_spin_polarization"] = float(numeric["active_spin_polarization"])
    numeric["cb_valley_polarization"] = float(numeric["active_valley_polarization"])
    occupations: list[dict[str, object]] = []
    for item in result.tables.get("occupations", []):
        row = dict(item)
        row.pop("active", None)
        row.pop("band", None)
        row.pop("sector", None)
        row.pop("fold", None)
        row.pop("layer", None)
        row.pop("sublattice", None)
        occupations.append(row)
    return {**numeric, "classification": result.classification, "occupations": occupations}

__all__ = [
    "TDBGProjectedHFData",
    "TDBGProjectedHFInitializer",
    "TDBGProjectedHFDensityBuilder",
    "TDBGProjectedHFResult",
    "TDBGProjectedHFState",
    "TDBGProjectedHFTargetData",
    "TDBGStateLabel",
    "_active_filling_indices",
    "_conventional_projector_to_stored",
    "_first_conduction_indices",
    "_fock_density_for_policy",
    "_hartree_density_for_policy",
    "_numeric_order_parameters",
    "_reference_projector",
    "_reference_subtracted_tdbg_density",
    "_stored_to_conventional",
    "initialize_tdbg_density",
    "initialize_tdbg_nu2_density",
    "reconstruct_tdbg_projected_hf_micro_wavefunctions",
    "tdbg_active_eigensystem_from_hamiltonian",
    "tdbg_canonical_projected_micro_basis",
    "tdbg_density_from_hamiltonian",
    "tdbg_order_parameters",
]
