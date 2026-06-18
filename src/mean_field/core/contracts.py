from __future__ import annotations

"""Canonical mean-field I/O contracts.

This module is intentionally system-agnostic.  It defines the objects that
future HF/TDHF/cRPA/topology adapters should use at module boundaries, without
changing any existing physics implementation.  The central density convention is

    projector      P(k): physical occupied projector
    reference      R(k): interaction-scheme reference density
    density_delta  X(k): P(k) - R(k)

All matrix fields use the existing core-HF axis convention
``(n_state, n_state, n_k)``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

MatrixField = np.ndarray
EnergyField = np.ndarray
DensityField = np.ndarray

ProjectionMode = Literal["bare", "screened"]
ReferenceScheme = Literal["average", "CN", "central_average", "custom"]
DensityConvention = Literal["delta"]
InteractionKind = Literal["2d_coulomb", "layered_3d", "crpa", "onsite_intersite"]
SelfEnergyInputPolicy = Literal["delta", "projector", "custom"]
MicroscopicWavefunctionSource = Literal["single_particle", "hf_reconstructed"]


@dataclass(frozen=True)
class ProjectionConfig:
    mode: ProjectionMode
    active_valence_bands: int
    active_conduction_bands: int
    basis_displacement_mev: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionConfig:
    scheme: ReferenceScheme
    kind: InteractionKind = "2d_coulomb"
    epsilon_r: float | None = None
    d_sc_nm: float | None = None
    self_energy_input_policy: SelfEnergyInputPolicy = "delta"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverConfig:
    max_iter: int = 300
    precision: float = 1.0e-8
    mixing: float | None = None
    use_oda: bool = True
    seeds: tuple[int, ...] = (1,)
    init_modes: tuple[str, ...] = ("random",)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    root: str | None = None
    overwrite: bool = False
    save_wavefunctions: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MFConfig:
    system: str
    run_id: str
    layer_count: int | None
    xi: int | None
    theta_deg: float
    displacement_mev: float
    k_mesh: int
    g_shell: int | None
    projection: ProjectionConfig
    interaction: InteractionConfig
    solver: SolverConfig
    output: OutputConfig
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceDensity:
    scheme: ReferenceScheme
    reference: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference", np.asarray(self.reference, dtype=np.complex128))
        object.__setattr__(self, "metadata", dict(self.metadata))
        assert_matrix_field_shape(self.reference, name="reference")
        assert_hermitian_field(self.reference, name="reference")


@dataclass(frozen=True)
class DensityState:
    density_delta: np.ndarray
    reference: ReferenceDensity
    filling: float
    n_occupied_total: int
    convention: DensityConvention = "delta"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_delta", np.asarray(self.density_delta, dtype=np.complex128))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.convention != "delta":
            raise ValueError(f"DensityState convention must be 'delta', got {self.convention!r}")
        assert_matrix_field_shape(self.density_delta, name="density_delta")
        if self.density_delta.shape != self.reference.reference.shape:
            raise ValueError(
                "density_delta and reference must have the same shape, "
                f"got {self.density_delta.shape} and {self.reference.reference.shape}"
            )
        assert_hermitian_field(self.density_delta, name="density_delta")

    @property
    def projector(self) -> np.ndarray:
        return self.density_delta + self.reference.reference


@dataclass(frozen=True)
class SingleParticleModel:
    system: str
    lattice: Any
    params: Any
    hamiltonian_builder: Callable[[np.ndarray], np.ndarray]
    diagonalizer: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectedBasis:
    physical_model: SingleParticleModel
    basis_model: SingleParticleModel
    kvec: np.ndarray
    k_grid_frac: np.ndarray
    h0: np.ndarray
    basis_energies: np.ndarray
    active_band_indices: tuple[int, ...]
    active_valence_bands: int
    active_conduction_bands: int
    micro_wavefunctions: np.ndarray
    flavor_labels: tuple[Any, ...] = ()
    band_labels: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kvec", np.asarray(self.kvec, dtype=np.complex128).reshape(-1))
        object.__setattr__(self, "k_grid_frac", np.asarray(self.k_grid_frac, dtype=float))
        object.__setattr__(self, "h0", np.asarray(self.h0, dtype=np.complex128))
        object.__setattr__(self, "basis_energies", np.asarray(self.basis_energies, dtype=float))
        object.__setattr__(self, "micro_wavefunctions", np.asarray(self.micro_wavefunctions, dtype=np.complex128))
        object.__setattr__(self, "active_band_indices", tuple(int(index) for index in self.active_band_indices))
        object.__setattr__(self, "metadata", dict(self.metadata))
        assert_projected_basis_consistent(self)


@dataclass(frozen=True)
class InteractionKernel:
    kind: InteractionKind
    basis: ProjectedBasis
    overlap_blocks: Any
    coulomb_blocks: Any
    supports_layer_index: bool = False
    supports_crpa: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HamiltonianParts:
    h0: np.ndarray
    fixed: np.ndarray
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    density_input_convention: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("h0", "fixed", "hartree", "fock", "total"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.complex128))
        object.__setattr__(self, "metadata", dict(self.metadata))
        assert_hamiltonian_parts_consistent(self)


@dataclass(frozen=True)
class HFState:
    basis: ProjectedBasis
    density: DensityState
    hamiltonian: HamiltonianParts
    energies: np.ndarray
    eigenvectors_active: np.ndarray
    mu: float
    observables: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HFRunResult:
    final_state: HFState
    iteration_history: list[dict[str, Any]]
    converged: bool
    exit_reason: str
    best_seed: int
    init_mode: str
    archive_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MicroscopicWavefunctionBundle:
    kvec: np.ndarray
    psi_micro: np.ndarray
    sewing_transforms: tuple[Callable[..., Any], ...] = ()
    basis_metadata: dict[str, Any] = field(default_factory=dict)
    source: MicroscopicWavefunctionSource = "hf_reconstructed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kvec", np.asarray(self.kvec, dtype=np.complex128))
        object.__setattr__(self, "psi_micro", np.asarray(self.psi_micro, dtype=np.complex128))
        object.__setattr__(self, "basis_metadata", dict(self.basis_metadata))
        if self.psi_micro.ndim < 3:
            raise ValueError(f"psi_micro must include k, basis, and state axes, got {self.psi_micro.shape}")
        if self.source not in {"single_particle", "hf_reconstructed"}:
            raise ValueError(f"Invalid microscopic wavefunction source {self.source!r}")


@dataclass(frozen=True)
class TDHFResult:
    pairs: tuple[Any, ...]
    A: np.ndarray
    B: np.ndarray
    L: np.ndarray
    spectrum: np.ndarray
    mode_classification: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# Validators -----------------------------------------------------------------


def assert_matrix_field_shape(x: np.ndarray, *, name: str = "matrix_field") -> tuple[int, int, int]:
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (n_state, n_state, n_k), got {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square on the first two axes, got {arr.shape}")
    if arr.shape[2] <= 0:
        raise ValueError(f"{name} must contain at least one k point, got {arr.shape}")
    return (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))


def assert_hermitian_field(x: np.ndarray, *, name: str = "matrix_field", tol: float = 1.0e-10) -> None:
    arr = np.asarray(x, dtype=np.complex128)
    assert_matrix_field_shape(arr, name=name)
    residual = np.max(np.abs(arr - arr.conjugate().swapaxes(0, 1)))
    if float(residual) > float(tol):
        raise ValueError(f"{name} must be Hermitian at every k; max residual {residual:.6e} exceeds {tol:.6e}")


def assert_projector_field(P: np.ndarray, *, name: str = "projector", tol: float = 1.0e-8) -> None:
    projector = np.asarray(P, dtype=np.complex128)
    assert_hermitian_field(projector, name=name, tol=tol)
    residual = np.max(np.abs(np.einsum("abk,bck->ack", projector, projector, optimize=True) - projector))
    if float(residual) > float(tol):
        raise ValueError(f"{name} must be idempotent at every k; max residual {residual:.6e} exceeds {tol:.6e}")


def assert_density_state_consistent(state: DensityState, *, require_projector: bool = True, tol: float = 1.0e-8) -> None:
    if state.convention != "delta":
        raise ValueError(f"DensityState convention must be 'delta', got {state.convention!r}")
    assert_matrix_field_shape(state.density_delta, name="density_delta")
    assert_hermitian_field(state.density_delta, name="density_delta", tol=tol)
    projector = state.projector
    assert_hermitian_field(projector, name="projector", tol=tol)
    total_occupation = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    if not np.isclose(total_occupation, float(state.n_occupied_total), atol=tol, rtol=0.0):
        raise ValueError(
            f"projector trace total {total_occupation:.12g} does not match "
            f"n_occupied_total={state.n_occupied_total}"
        )
    if require_projector:
        assert_projector_field(projector, name="projector", tol=tol)


def assert_hamiltonian_parts_consistent(parts: HamiltonianParts, *, tol: float = 1.0e-10) -> None:
    shape = assert_matrix_field_shape(parts.h0, name="h0")
    for name in ("fixed", "hartree", "fock", "total"):
        arr_shape = assert_matrix_field_shape(getattr(parts, name), name=name)
        if arr_shape != shape:
            raise ValueError(f"{name} shape {arr_shape} does not match h0 shape {shape}")
    expected = parts.h0 + parts.fixed + parts.hartree + parts.fock
    residual = np.max(np.abs(parts.total - expected))
    if float(residual) > float(tol):
        raise ValueError(f"HamiltonianParts total sum residual {residual:.6e} exceeds {tol:.6e}")
    assert_hermitian_field(parts.total, name="total", tol=tol)


def assert_projected_basis_consistent(basis: ProjectedBasis) -> None:
    n_state, _n_state_rhs, n_k = assert_matrix_field_shape(basis.h0, name="h0")
    if basis.kvec.shape != (n_k,):
        raise ValueError(f"kvec must have shape ({n_k},), got {basis.kvec.shape}")
    if basis.k_grid_frac.shape != (n_k, 2):
        raise ValueError(f"k_grid_frac must have shape ({n_k}, 2), got {basis.k_grid_frac.shape}")
    if len(basis.active_band_indices) != n_state:
        raise ValueError(
            f"active_band_indices length {len(basis.active_band_indices)} must match projected dimension {n_state}"
        )
    expected_band_count = int(basis.active_valence_bands) + int(basis.active_conduction_bands)
    if expected_band_count > 0 and len(basis.band_labels) not in {0, expected_band_count, n_state}:
        raise ValueError(
            "band_labels must be empty, per-band, or per-active-state; "
            f"got {len(basis.band_labels)} labels for expected band count {expected_band_count} and n_state {n_state}"
        )
    if basis.basis_energies.ndim != 2 or basis.basis_energies.shape[-1] != n_k:
        raise ValueError(f"basis_energies must have shape (n_basis_band, {n_k}), got {basis.basis_energies.shape}")
    assert_hermitian_field(basis.h0, name="h0")


def assert_no_screened_diag_h0_for_RnG(basis: ProjectedBasis, *, tol: float = 1.0e-10) -> None:
    """Guard the RnG/hBN screened-basis projection rule.

    For screened-basis RnG/hBN, ``h0`` must be ``<u_U|H_sp(V)|u_U>`` and not
    simply ``diag(E[H_sp(U)])``.  The diagonal-eigenvalue rejection is applied
    only when metadata says the physical and basis displacements differ; at
    trivial ``U == V`` checkpoints the two can legitimately coincide.
    """

    system = str(basis.physical_model.system).lower().replace("-", "_")
    if system not in {"rng_hbn", "rlg_hbn", "rnghbn"}:
        return
    mode = str(basis.metadata.get("projection_mode", ""))
    if mode != "screened":
        return
    h0_rule = basis.metadata.get("h0_rule")
    if h0_rule != "project_H_sp_V_into_H_sp_U_basis":
        raise ValueError(
            "RnG/hBN screened ProjectedBasis must declare h0_rule='project_H_sp_V_into_H_sp_U_basis'"
        )
    physical_v = basis.metadata.get("physical_model_displacement_mev")
    basis_u = basis.metadata.get("basis_model_displacement_mev")
    if physical_v is None or basis_u is None or np.isclose(float(physical_v), float(basis_u), atol=tol, rtol=0.0):
        return
    if basis.basis_energies.shape[0] < basis.h0.shape[0]:
        return
    diag_eu = np.zeros_like(basis.h0)
    for ik in range(basis.h0.shape[2]):
        diag_eu[:, :, ik] = np.diag(basis.basis_energies[: basis.h0.shape[0], ik])
    if np.allclose(basis.h0, diag_eu, atol=tol, rtol=0.0):
        raise ValueError(
            "RnG/hBN screened ProjectedBasis h0 appears to be diag(E[H_sp(U)]); "
            "it must be <u_basis(U)|H_sp(V)|u_basis(U)>"
        )


__all__ = [
    "DensityConvention",
    "DensityField",
    "DensityState",
    "EnergyField",
    "HFRunResult",
    "HFState",
    "HamiltonianParts",
    "InteractionConfig",
    "InteractionKernel",
    "InteractionKind",
    "MFConfig",
    "MatrixField",
    "MicroscopicWavefunctionBundle",
    "MicroscopicWavefunctionSource",
    "OutputConfig",
    "ProjectedBasis",
    "ProjectionConfig",
    "ProjectionMode",
    "ReferenceDensity",
    "ReferenceScheme",
    "SelfEnergyInputPolicy",
    "SingleParticleModel",
    "SolverConfig",
    "TDHFResult",
    "assert_density_state_consistent",
    "assert_hamiltonian_parts_consistent",
    "assert_hermitian_field",
    "assert_matrix_field_shape",
    "assert_no_screened_diag_h0_for_RnG",
    "assert_projected_basis_consistent",
    "assert_projector_field",
]
