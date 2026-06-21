from __future__ import annotations

from ._polshyn_shared import *  # noqa: F401,F403
from ._polshyn_types import *  # noqa: F401,F403
from ._polshyn_filling import reference_diagonal_for_projected_indices
from .model import TMBGModel


def _supercell_k_grid(
    *,
    super_b1: complex,
    super_b2: complex,
    mesh_size: int,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    mesh = int(mesh_size)
    if mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    shift = (float(frac_shift[0]), float(frac_shift[1]))
    frac: list[tuple[float, float]] = []
    kvec: list[complex] = []
    for iy in range(mesh):
        for ix in range(mesh):
            f1 = (float(ix) + shift[0]) / float(mesh)
            f2 = (float(iy) + shift[1]) / float(mesh)
            frac.append((f1, f2))
            kvec.append(f1 * complex(super_b1) + f2 * complex(super_b2))
    return np.asarray(frac, dtype=float), np.asarray(kvec, dtype=np.complex128)


def _polshyn_embedding_positions(
    model: TMBGModel,
    supercell: PolshynDoubledCell,
    *,
    folds_per_primitive: int = 2,
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int, int], tuple[int, int]]]:
    raw: dict[tuple[int, int, int], tuple[int, int]] = {}
    for n1, n2 in np.asarray(model.lattice.g_indices, dtype=int):
        for fold in range(int(folds_per_primitive)):
            raw[(int(n1), int(n2), int(fold))] = supercell.primitive_to_supercell_coords(int(n1), int(n2), fold)
    sx_values = [pos[0] for pos in raw.values()]
    sy_values = [pos[1] for pos in raw.values()]
    origin = (min(sx_values), min(sy_values))
    shape = (max(sx_values) - origin[0] + 1, max(sy_values) - origin[1] + 1)
    positions = {key: (pos[0] - origin[0], pos[1] - origin[1]) for key, pos in raw.items()}
    return (int(shape[0]), int(shape[1])), (int(origin[0]), int(origin[1])), positions


def build_polshyn_projected_basis(
    model: TMBGModel,
    *,
    mesh_size: int,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    valleys: tuple[int, int] = (1, -1),
    supercell: PolshynDoubledCell | None = None,
) -> PolshynProjectedBasis:
    """Build the explicit doubled-cell Polshyn projected basis.

    This is a system adapter: it diagonalizes the TMBG one-body Hamiltonian at
    the two folded primitive momenta ``k`` and ``k+B1`` for each supercell k,
    embeds the primitive plane-wave coefficients in the doubled-cell reciprocal
    grid, and leaves the HF iteration to the generic Wang/Xiaoyu core path.
    """

    if not isinstance(model, TMBGModel):
        raise TypeError(f"model must be TMBGModel, got {type(model).__name__}")
    indices = tuple(int(index) for index in projected_indices)
    if not indices:
        raise ValueError("projected_indices must not be empty")
    target = int(target_band_index)
    if target not in indices:
        raise ValueError(f"target_band_index={target} is not present in projected_indices={indices}")
    matrix_dim = int(model.lattice.matrix_dim)
    if min(indices) < 0 or max(indices) >= matrix_dim:
        raise ValueError(f"projected_indices={indices} outside TMBG matrix dimension {matrix_dim}")
    resolved_supercell = polshyn_doubled_cell() if supercell is None else supercell
    super_b1, super_b2 = resolved_supercell.reciprocal_vectors(model.lattice)
    k_grid_frac, kvec = _supercell_k_grid(
        super_b1=super_b1,
        super_b2=super_b2,
        mesh_size=int(mesh_size),
        frac_shift=frac_shift,
    )
    embedding_shape, embedding_origin, embedding_positions = _polshyn_embedding_positions(model, resolved_supercell)
    local_basis_size = 6
    basis_dimension = local_basis_size * int(embedding_shape[0]) * int(embedding_shape[1])
    n_eta = len(tuple(valleys))
    nb = 2 * len(indices)
    nk = int(kvec.size)
    wavefunctions_grid = np.zeros(
        (local_basis_size, int(embedding_shape[0]), int(embedding_shape[1]), nb, n_eta, nk),
        dtype=np.complex128,
    )
    h0_blocks = np.zeros((2, n_eta, nb, nb, nk), dtype=np.complex128)
    for ieta, valley in enumerate(tuple(int(value) for value in valleys)):
        if valley not in (-1, 1):
            raise ValueError(f"valleys must contain ±1 labels, got {valleys}")
        for ik, k_base in enumerate(kvec):
            for fold in range(2):
                evals, evecs = model.diagonalize(complex(k_base + fold * super_b1), valley=valley, n_bands=None)
                assert evecs is not None
                for iprim, band_index in enumerate(indices):
                    folded_index = 2 * iprim + fold
                    energy = float(evals[int(band_index)])
                    h0_blocks[:, ieta, folded_index, folded_index, ik] = energy
                    column = np.asarray(evecs[:, int(band_index)], dtype=np.complex128)
                    for ig, (n1, n2) in enumerate(np.asarray(model.lattice.g_indices, dtype=int)):
                        ix, iy = embedding_positions[(int(n1), int(n2), int(fold))]
                        wavefunctions_grid[:, ix, iy, folded_index, ieta, ik] = column[
                            local_basis_size * ig : local_basis_size * (ig + 1)
                        ]
    wavefunctions = wavefunctions_grid.reshape((basis_dimension, nb, n_eta, nk), order="F")
    reference = reference_diagonal_for_projected_indices(indices, target)
    return PolshynProjectedBasis(
        model=model,
        supercell=resolved_supercell,
        kvec=kvec,
        k_grid_frac=k_grid_frac,
        projected_indices=indices,
        target_band_index=target,
        wavefunctions=wavefunctions,
        h0_blocks=h0_blocks,
        reference_diagonal=reference,
        super_b1=complex(super_b1),
        super_b2=complex(super_b2),
        embedding_shape=embedding_shape,
        embedding_origin=embedding_origin,
        embedding_positions=embedding_positions,
    )


__all__ = [name for name in globals() if not name.startswith('__')]
