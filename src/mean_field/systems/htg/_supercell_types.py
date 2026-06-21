from __future__ import annotations

from ._supercell_shared import *  # noqa: F401,F403

class HTGSupercell(IntegerSupercell):
    """Integer supercell for folded-BZ HTG projected Hartree-Fock."""

    def reciprocal_vectors(self, lattice: HTGLattice) -> tuple[complex, complex]:
        return super().reciprocal_vectors(lattice.b_m1, lattice.b_m2)


@dataclass(frozen=True)
class HTGSupercellProjectedBasisData:
    model: HTGModel
    interaction: InteractionParams
    supercell: HTGSupercell
    mesh_size: int
    kvec: np.ndarray
    k_grid_frac: np.ndarray | None
    basis: ProjectedWavefunctionBasis
    h0: np.ndarray
    sigma_z: np.ndarray
    band_sigma_z: np.ndarray
    primitive_projected_indices: tuple[int, ...]
    primitive_band_count: int
    fold_representatives: tuple[tuple[int, int], ...]
    reference_diagonal: np.ndarray
    super_g1: complex
    super_g2: complex
    reciprocal_grid_shape: tuple[int, int]
    reciprocal_grid_origin: tuple[int, int]
    moire_supercell_area_nm2: float

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nb(self) -> int:
        return int(self.basis.n_band)


@dataclass
class HTGSupercellHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    nu: float
    reference_diagonal: np.ndarray
    v0: float
    mu: float = float("nan")
    precision: float = 1.0e-6
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 12
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
        basis_data: HTGSupercellProjectedBasisData,
        *,
        nu: float,
        precision: float = 1.0e-6,
    ) -> "HTGSupercellHartreeFockState":
        h0 = np.asarray(basis_data.h0, dtype=np.complex128).copy()
        nt, _, nk = h0.shape
        return cls(
            h0=h0,
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            reference_diagonal=np.asarray(basis_data.reference_diagonal, dtype=float).copy(),
            v0=1.0 / float(basis_data.moire_supercell_area_nm2),
            precision=float(precision),
            n_spin=int(basis_data.basis.n_spin),
            n_eta=int(basis_data.basis.n_flavor),
            n_band=int(basis_data.basis.n_band),
        )


@dataclass(frozen=True)
class HTGSupercellHartreeFockRun(HartreeFockRun):
    state: HTGSupercellHartreeFockState
    overlap_blocks: HFOverlapBlockSet
    basis_data: HTGSupercellProjectedBasisData


@dataclass(frozen=True)
class HTGSupercellGroundStateScan:
    runs: tuple[HTGSupercellHartreeFockRun, ...]

    @property
    def best_run(self) -> HTGSupercellHartreeFockRun:
        if not self.runs:
            raise ValueError("No HTG supercell HF runs are available")
        return min(self.runs, key=lambda run: float(run.state.diagnostics.get("hf_energy", np.inf)))


@dataclass(frozen=True)
class HTGSupercellPathResult:
    path: KPath
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float
    nu: float
    init_mode: str
    seed: int
    exit_reason: str
    points_per_segment: int


@dataclass(frozen=True)
class HTGSupercellSCFGridPathSamples:
    """Exact saved SCF-grid samples lying on a folded-BZ path."""

    kdist: np.ndarray
    grid_indices: np.ndarray
    frac_coords: np.ndarray
    segment_indices: np.ndarray
    node_kdist: np.ndarray
    labels: tuple[str, ...]
    exact_node_hit_mask: np.ndarray
    exact_tolerance: float

    @property
    def unique_grid_count(self) -> int:
        return int(np.unique(self.grid_indices).size)

    @property
    def exact_node_hit_count(self) -> int:
        return int(np.count_nonzero(self.exact_node_hit_mask))

    @property
    def segment_counts(self) -> tuple[int, ...]:
        n_segments = max(len(self.labels) - 1, 0)
        counts = np.bincount(self.segment_indices.astype(int), minlength=n_segments)
        return tuple(int(value) for value in counts[:n_segments])

@dataclass(frozen=True)
class HTGSupercellHFWavefunctionGrid:
    """Full physical wavefunction mesh reconstructed from a supercell HF Hamiltonian."""

    wavefunctions: np.ndarray
    energies: np.ndarray
    k_grid_frac: np.ndarray
    band_indices: tuple[int, ...]
    basis_data: HTGSupercellProjectedBasisData

__all__ = [name for name in globals() if not name.startswith('__')]
