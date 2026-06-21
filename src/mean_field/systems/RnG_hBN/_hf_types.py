from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_reference import rlg_hbn_reference_density

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

__all__ = [name for name in globals() if not name.startswith('__')]
