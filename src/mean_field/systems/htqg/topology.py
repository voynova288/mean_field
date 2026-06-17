from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from analysis.topology import LatticeTopologyResult, WavefunctionIndex, compute_lattice_topology

from .bands import compute_bands_on_grid
from .domains import HTQGDomain, canonical_domain_key
from .hamiltonian import centered_band_indices, diagonalize_hamiltonian, build_hamiltonian
from .lattice import HTQGLattice, build_moire_k_grid
from .params import HTQGParams


@dataclass(frozen=True)
class ChernBasisResult:
    mesh_size: int
    valley: int
    domain: str
    band_indices: tuple[int, int]
    chern_a: float
    chern_b: float
    total_chern: float
    rounded_chern_a: int
    rounded_chern_b: int
    rounded_total_chern: int
    sigma_z_eigenvalue_min: float
    sigma_z_eigenvalue_max: float
    min_link_magnitude_a: float
    min_link_magnitude_b: float
    min_link_magnitude_subspace: float

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
            "domain": self.domain,
            "band_indices": [int(index) for index in self.band_indices],
            "chern_a": float(self.chern_a),
            "chern_b": float(self.chern_b),
            "total_chern": float(self.total_chern),
            "rounded_chern_a": int(self.rounded_chern_a),
            "rounded_chern_b": int(self.rounded_chern_b),
            "rounded_total_chern": int(self.rounded_total_chern),
            "integer_residual_a": self.integer_residual_a,
            "integer_residual_b": self.integer_residual_b,
            "sigma_z_eigenvalue_min": float(self.sigma_z_eigenvalue_min),
            "sigma_z_eigenvalue_max": float(self.sigma_z_eigenvalue_max),
            "min_link_magnitude_a": float(self.min_link_magnitude_a),
            "min_link_magnitude_b": float(self.min_link_magnitude_b),
            "min_link_magnitude_subspace": float(self.min_link_magnitude_subspace),
        }


def sublattice_sigma_z(lattice: HTQGLattice) -> np.ndarray:
    """Layer-resolved graphene sublattice operator in the HTQG basis."""

    pattern = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    return np.diag(np.tile(pattern, lattice.n_g)).astype(np.complex128)


def reciprocal_translation(lattice: HTQGLattice, dn1: int, dn2: int) -> Callable[[np.ndarray], np.ndarray]:
    """Boundary sewing for k -> k + dn1*b1 + dn2*b2 in the plane-wave basis."""

    index_by_g = {tuple(int(x) for x in pair): idx for idx, pair in enumerate(lattice.g_indices)}

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        out = np.zeros_like(array)
        for target_index, (n1, n2) in enumerate(lattice.g_indices):
            source_index = index_by_g.get((int(n1) + int(dn1), int(n2) + int(dn2)))
            if source_index is None:
                continue
            out[8 * target_index : 8 * target_index + 8, ...] = array[
                8 * source_index : 8 * source_index + 8,
                ...,
            ]
        return out

    return apply


def boundary_sewing_transforms(lattice: HTQGLattice) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    return (reciprocal_translation(lattice, 1, 0), reciprocal_translation(lattice, 0, 1))


def _central_eigensystem(
    k_tilde: complex,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain,
    valley: int,
    band_indices: tuple[int, int],
    use_full_eigensolver: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_full_eigensolver:
        hamiltonian = build_hamiltonian(k_tilde, lattice, params, domain=domain, valley=valley)
        evals, evecs = np.linalg.eigh(hamiltonian)
        selected = np.asarray(band_indices, dtype=int)
        return np.asarray(evals[selected], dtype=float), np.asarray(evecs[:, selected], dtype=np.complex128)

    evals, evecs = diagonalize_hamiltonian(
        k_tilde,
        lattice,
        params,
        domain=domain,
        valley=valley,
        band_indices=band_indices,
        return_eigenvectors=True,
    )
    if evecs is None:
        raise RuntimeError("Expected eigenvectors for Chern-basis construction.")
    return evals, evecs


def _compute_topology(
    vectors: np.ndarray,
    lattice: HTQGLattice,
    *,
    k_grid_frac: np.ndarray,
    index: WavefunctionIndex,
    link_method: str = "determinant",
) -> LatticeTopologyResult:
    return compute_lattice_topology(
        vectors,
        index=index,
        k_grid_frac=k_grid_frac,
        sewing_transforms=boundary_sewing_transforms(lattice),
        link_method=link_method,  # type: ignore[arg-type]
    )


def compute_chern_basis_on_grid(
    mesh_size: int,
    lattice: HTQGLattice,
    params: HTQGParams,
    *,
    domain: str | HTQGDomain = "alpha_beta_alpha",
    valley: int = 1,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    use_full_eigensolver: bool = True,
) -> ChernBasisResult:
    """Compute Chern numbers after sigma_z projection into the central pair.

    This is the correct route for the Type-I αβα central pair, whose two flat
    bands touch at Gamma and should not be treated as isolated energy bands.
    """

    if mesh_size <= 1:
        raise ValueError("mesh_size must exceed 1 for a Berry-curvature plaquette grid")

    domain_key = canonical_domain_key(domain)
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    sigma_z = sublattice_sigma_z(lattice)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=False, frac_shift=frac_shift)

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
                domain=domain,
                valley=valley,
                band_indices=central_pair,
                use_full_eigensolver=use_full_eigensolver,
            )
            subspace_vectors[i, j, :, :] = evecs
            projected = evecs.conjugate().T @ sigma_z @ evecs
            eigvals, eigvecs = np.linalg.eigh(projected)
            sigma_eigs.extend(float(value.real) for value in eigvals)
            vectors_b[i, j, :] = evecs @ eigvecs[:, 0]
            vectors_a[i, j, :] = evecs @ eigvecs[:, -1]

    topo_a = _compute_topology(
        vectors_a,
        lattice,
        k_grid_frac=k_grid_frac,
        index=WavefunctionIndex(
            indices=central_pair,
            role="chern_sublattice_basis",
            labels=("A_like_positive_sigma_z",),
            system="htqg",
            valley=valley,
            metadata={"domain": domain_key},
        ),
    )
    topo_b = _compute_topology(
        vectors_b,
        lattice,
        k_grid_frac=k_grid_frac,
        index=WavefunctionIndex(
            indices=central_pair,
            role="chern_sublattice_basis",
            labels=("B_like_negative_sigma_z",),
            system="htqg",
            valley=valley,
            metadata={"domain": domain_key},
        ),
    )
    topo_total = _compute_topology(
        subspace_vectors,
        lattice,
        k_grid_frac=k_grid_frac,
        index=WavefunctionIndex(
            indices=central_pair,
            role="central_two_band_subspace",
            labels=("valence", "conduction"),
            system="htqg",
            valley=valley,
            metadata={"domain": domain_key},
        ),
    )

    return ChernBasisResult(
        mesh_size=int(mesh_size),
        valley=int(valley),
        domain=domain_key,
        band_indices=central_pair,
        chern_a=float(topo_a.chern_number),
        chern_b=float(topo_b.chern_number),
        total_chern=float(topo_total.chern_number),
        rounded_chern_a=int(topo_a.rounded_chern_number),
        rounded_chern_b=int(topo_b.rounded_chern_number),
        rounded_total_chern=int(topo_total.rounded_chern_number),
        sigma_z_eigenvalue_min=float(min(sigma_eigs)),
        sigma_z_eigenvalue_max=float(max(sigma_eigs)),
        min_link_magnitude_a=float(topo_a.min_link_magnitude),
        min_link_magnitude_b=float(topo_b.min_link_magnitude),
        min_link_magnitude_subspace=float(topo_total.min_link_magnitude),
    )


def compute_band_topology_on_grid(
    mesh_size: int,
    lattice: HTQGLattice,
    params: HTQGParams,
    band_index: int,
    *,
    domain: str | HTQGDomain = "alpha_beta_gamma",
    valley: int = 1,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> LatticeTopologyResult:
    """Compute FHS topology for an isolated energy band.

    Do not use this for the αβα central touching pair; use
    :func:`compute_chern_basis_on_grid` instead.
    """

    band_index = int(band_index)
    grid = compute_bands_on_grid(
        mesh_size,
        lattice,
        params,
        domain=domain,
        valley=valley,
        band_indices=(band_index,),
        return_eigenvectors=True,
        frac_shift=frac_shift,
    )
    if grid.eigenvectors is None:
        raise RuntimeError("Expected eigenvectors for topology calculation")
    return compute_lattice_topology(
        grid.eigenvectors,
        state_indices=(0,),
        index=WavefunctionIndex(
            indices=(band_index,),
            role="energy_band",
            labels=(f"band_{band_index}",),
            system="htqg",
            valley=valley,
            metadata={"domain": canonical_domain_key(domain)},
        ),
        k_grid_frac=grid.k_grid_frac,
        sewing_transforms=boundary_sewing_transforms(lattice),
        link_method="determinant",
    )


__all__ = [
    "ChernBasisResult",
    "boundary_sewing_transforms",
    "compute_band_topology_on_grid",
    "compute_chern_basis_on_grid",
    "reciprocal_translation",
    "sublattice_sigma_z",
]
