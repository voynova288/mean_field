from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
from scipy.linalg import eigh

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from mean_field.core.hf.contracts_bridge import density_state_from_delta

from ...core.hf import (
    DensityUpdateResult,
    FlavorBandData,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    HartreeFockStepResult,
    ProjectedWavefunctionBasis,
    apply_random_projector_rotation,
    random_unitary_from_hermitian,
    build_flavor_band_data,
    build_projected_hf_kernel,
    build_projected_hf_problem,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    compute_hf_energy,
    find_chemical_potential,
    occupied_state_mask,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb_matrix,
)
from .hamiltonian import build_hamiltonian, centered_band_indices
from .lattice import HTGLattice, KPath, build_moire_k_grid
from .model import HTGModel
from .params import HTGParams, InteractionParams
from .hamiltonian import sublattice_sigma_z

if TYPE_CHECKING:
    from mean_field.api import HFConfig, HFResult


VALLEY_SEQUENCE = (1, -1)


@dataclass(frozen=True)
class HTGSeedOccupationSummary:
    requested_init_mode: str
    normalized_init_mode: str
    nu: float
    n_spin: int
    n_eta: int
    n_band: int
    reference_band_occupations: tuple[float, ...]
    central_projected_band_indices: tuple[int, int]
    occupied_bands_per_k: int
    occupation_counts: tuple[int, ...] | None
    occupation_count_matrix: tuple[tuple[int, ...], ...] | None
    initial_state_labels: tuple[str, ...] | None
    constrained_flavor_counts: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_init_mode": self.requested_init_mode,
            "normalized_init_mode": self.normalized_init_mode,
            "nu": float(self.nu),
            "n_spin": int(self.n_spin),
            "n_eta": int(self.n_eta),
            "n_band": int(self.n_band),
            "reference_band_occupations": self.reference_band_occupations,
            "central_projected_band_indices": self.central_projected_band_indices,
            "occupied_bands_per_k": int(self.occupied_bands_per_k),
            "occupation_counts": self.occupation_counts,
            "occupation_count_matrix": self.occupation_count_matrix,
            "initial_state_labels": self.initial_state_labels,
            "constrained_flavor_counts": bool(self.constrained_flavor_counts),
        }


@dataclass(frozen=True)
class HTGProjectedBasisData:
    model: HTGModel
    interaction: InteractionParams
    mesh_size: int
    kvec: np.ndarray
    k_grid_frac: np.ndarray
    basis: ProjectedWavefunctionBasis
    h0: np.ndarray
    sigma_z: np.ndarray
    band_sigma_z: np.ndarray
    central_band_indices: tuple[int, int]
    projected_band_indices: tuple[int, ...]
    reciprocal_grid_shape: tuple[int, int]
    reciprocal_grid_origin: tuple[int, int]
    moire_cell_area_nm2: float

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])


@dataclass(frozen=True)
class HTGHartreeFockRun(HartreeFockRun):
    state: "HTGHartreeFockState"
    overlap_blocks: HFOverlapBlockSet
    basis_data: HTGProjectedBasisData


@dataclass(frozen=True)
class HTGGroundStateScan:
    runs: tuple[HTGHartreeFockRun, ...]

    @property
    def best_run(self) -> HTGHartreeFockRun:
        if not self.runs:
            raise ValueError("No HTG HF runs are available")
        return min(self.runs, key=lambda run: float(run.state.diagnostics.get("hf_energy", np.inf)))


@dataclass(frozen=True)
class HTGInteractionComponents:
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    hartree_eigenvalues: np.ndarray
    fock_eigenvalues: np.ndarray


@dataclass(frozen=True)
class HTGInteractionPathResult:
    path: KPath
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    hartree_diagonal_ev: np.ndarray
    fock_diagonal_ev: np.ndarray
    total_diagonal_ev: np.ndarray
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass(frozen=True)
class HTGHFPathResult:
    path: KPath
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_z_expectation: np.ndarray
    sigma_z_operator: np.ndarray
    band_data: FlavorBandData
    mu: float
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass
class HTGHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_z: np.ndarray
    nu: float
    v0: float
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
        basis_data: HTGProjectedBasisData,
        *,
        nu: float,
        precision: float = 1.0e-6,
        occupation_counts: tuple[int, ...] | None = None,
    ) -> "HTGHartreeFockState":
        h0 = np.asarray(basis_data.h0, dtype=np.complex128).copy()
        nt, _, nk = h0.shape
        return cls(
            h0=h0,
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_z=np.asarray(basis_data.sigma_z, dtype=np.complex128).copy(),
            nu=float(nu),
            v0=1.0 / float(basis_data.moire_cell_area_nm2),
            precision=float(precision),
            n_spin=int(basis_data.basis.n_spin),
            n_eta=int(basis_data.basis.n_flavor),
            n_band=int(basis_data.basis.n_band),
            occupation_counts=occupation_counts,
        )

__all__ = [name for name in globals() if not name.startswith('__')]
