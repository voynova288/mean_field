from __future__ import annotations

from ._polshyn_shared import *  # noqa: F401,F403

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

__all__ = [name for name in globals() if not name.startswith('__')]
