from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from analysis.topology import berry_curvature_from_links, chern_number_from_berry_curvature, compute_link_variables

from .hamiltonian import centered_band_indices, diagonalize_hamiltonian, build_hamiltonian
from .lattice import HTGLattice, build_moire_k_grid
from .params import HTGParams


@dataclass(frozen=True)
class ChernBasisResult:
    mesh_size: int
    valley: int
    band_indices: tuple[int, int]
    chern_a: float
    chern_b: float
    raw_chern_a: float
    raw_chern_b: float
    total_chern: float
    rounded_chern_a: int
    rounded_chern_b: int
    rounded_total_chern: int
    sigma_z_eigenvalue_min: float
    sigma_z_eigenvalue_max: float

    @property
    def integer_residual_a(self) -> float:
        return float(abs(self.chern_a - self.rounded_chern_a))

    @property
    def integer_residual_b(self) -> float:
        return float(abs(self.chern_b - self.rounded_chern_b))

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_size": int(self.mesh_size),
            "valley": int(self.valley),
            "band_indices": [int(index) for index in self.band_indices],
            "chern_a": float(self.chern_a),
            "chern_b": float(self.chern_b),
            "raw_chern_a": float(self.raw_chern_a),
            "raw_chern_b": float(self.raw_chern_b),
            "total_chern": float(self.total_chern),
            "rounded_chern_a": int(self.rounded_chern_a),
            "rounded_chern_b": int(self.rounded_chern_b),
            "rounded_total_chern": int(self.rounded_total_chern),
            "integer_residual_a": self.integer_residual_a,
            "integer_residual_b": self.integer_residual_b,
            "sigma_z_eigenvalue_min": float(self.sigma_z_eigenvalue_min),
            "sigma_z_eigenvalue_max": float(self.sigma_z_eigenvalue_max),
        }


def sublattice_sigma_z(lattice: HTGLattice) -> np.ndarray:
    """Layer-resolved graphene sublattice operator.

    The orbital ordering in the Hamiltonian is, for each moire reciprocal
    vector G, (top A, top B, middle A, middle B, bottom A, bottom B).
    The Chern basis is obtained by diagonalizing this operator after
    projection to the two central bands.
    """
    pattern = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    return np.diag(np.tile(pattern, lattice.n_g)).astype(np.complex128)



def _reciprocal_translation(lattice: HTGLattice, dn1: int, dn2: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return the transition function for crossing a moire BZ boundary.

    In the plane-wave representation a cell-periodic state is expanded as
        u_k(r) = sum_G c_G(k) exp(i G.r).
    At k + B, with B = dn1*b1 + dn2*b2, the same physical Bloch state is
    represented by c'_G = c_{G+B}.  This transition function is essential
    for Chern-number calculations on the torus; omitting it effectively glues
    the two BZ edges with the wrong gauge.
    """
    index_by_g = {tuple(int(x) for x in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector)
        out = np.zeros_like(array)
        for target_index, (n1, n2) in enumerate(lattice.g_indices):
            source_index = index_by_g.get((int(n1) + int(dn1), int(n2) + int(dn2)))
            if source_index is None:
                continue
            out[6 * target_index : 6 * target_index + 6, ...] = array[
                6 * source_index : 6 * source_index + 6, ...
            ]
        return out

    return apply


def _compute_link_arrays(vectors: np.ndarray, lattice: HTGLattice) -> tuple[np.ndarray, np.ndarray]:
    links = compute_link_variables(
        vectors,
        sewing_transforms=(
            _reciprocal_translation(lattice, 1, 0),
            _reciprocal_translation(lattice, 0, 1),
        ),
        link_method="determinant",
    )
    return links.link_1, links.link_2

def _chern_from_link_vectors(vectors: np.ndarray, lattice: HTGLattice) -> float:
    link_1, link_2 = _compute_link_arrays(vectors, lattice)
    return chern_number_from_berry_curvature(berry_curvature_from_links(link_1, link_2))

def _central_eigensystem(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int,
    d_top: complex | None,
    d_bot: complex | None,
    band_indices: tuple[int, int],
    use_full_eigensolver: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_full_eigensolver:
        # A full Hermitian solve is more robust at exact chiral degeneracies
        # than subset solvers, and the topology examples use moderate cutoffs.
        hamiltonian = build_hamiltonian(
            k_tilde,
            lattice,
            params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
        )
        evals, evecs = np.linalg.eigh(hamiltonian)
        selected = np.asarray(band_indices, dtype=int)
        return np.asarray(evals[selected], dtype=float), np.asarray(evecs[:, selected], dtype=np.complex128)

    evals, evecs = diagonalize_hamiltonian(
        k_tilde,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        band_indices=band_indices,
        return_eigenvectors=True,
    )
    if evecs is None:
        raise RuntimeError("Expected eigenvectors for Chern-basis construction.")
    return evals, evecs


def compute_chern_basis_on_grid(
    mesh_size: int,
    lattice: HTGLattice,
    params: HTGParams,
    *,
    valley: int = 1,
    d_top: complex | None = None,
    d_bot: complex | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    use_full_eigensolver: bool = True,
) -> ChernBasisResult:
    """Compute Chern numbers of the two central Chern-basis bands.

    The two central bands are first projected to the subspace spanned by the
    valence and conduction flat bands.  The projected sublattice operator
    sigma_z is then diagonalized; its positive and negative eigenvectors are
    labelled A and B respectively.

    The Berry flux is evaluated with the Fukui-Hatsugai-Suzuki lattice formula.
    The important implementation detail is the reciprocal-lattice transition
    function at the BZ boundary, because H(k + b_i) is only equal to H(k) after
    relabelling the plane-wave momenta G -> G + b_i.
    """
    if mesh_size <= 1:
        raise ValueError("mesh_size must exceed 1 for a Berry-curvature plaquette grid")

    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    sigma_z = sublattice_sigma_z(lattice)
    _, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=False, frac_shift=frac_shift)

    vectors_a = np.zeros((mesh_size, mesh_size, lattice.matrix_dim), dtype=np.complex128)
    vectors_b = np.zeros_like(vectors_a)
    subspace_vectors = np.zeros((mesh_size, mesh_size, lattice.matrix_dim, 2), dtype=np.complex128)
    sigma_eigs: list[float] = []

    for i in range(mesh_size):
        for j in range(mesh_size):
            _, evecs = _central_eigensystem(
                complex(kvec[i, j]),
                lattice,
                params,
                valley=valley,
                d_top=d_top,
                d_bot=d_bot,
                band_indices=central_pair,
                use_full_eigensolver=use_full_eigensolver,
            )
            subspace_vectors[i, j, :, :] = evecs

            projected = evecs.conjugate().T @ sigma_z @ evecs
            eigvals, eigvecs = np.linalg.eigh(projected)
            sigma_eigs.extend(float(value.real) for value in eigvals)

            # eigvals are sorted ascending: negative is B-like, positive is A-like.
            vectors_b[i, j, :] = evecs @ eigvecs[:, 0]
            vectors_a[i, j, :] = evecs @ eigvecs[:, -1]

    raw_chern_a = _chern_from_link_vectors(vectors_a, lattice)
    raw_chern_b = _chern_from_link_vectors(vectors_b, lattice)
    total_chern = _chern_from_link_vectors(subspace_vectors, lattice)

    return ChernBasisResult(
        mesh_size=int(mesh_size),
        valley=int(valley),
        band_indices=central_pair,
        chern_a=float(raw_chern_a),
        chern_b=float(raw_chern_b),
        raw_chern_a=float(raw_chern_a),
        raw_chern_b=float(raw_chern_b),
        total_chern=float(total_chern),
        rounded_chern_a=int(round(raw_chern_a)),
        rounded_chern_b=int(round(raw_chern_b)),
        rounded_total_chern=int(round(total_chern)),
        sigma_z_eigenvalue_min=float(min(sigma_eigs)),
        sigma_z_eigenvalue_max=float(max(sigma_eigs)),
    )
