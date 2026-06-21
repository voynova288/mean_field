from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403

def _layer_potential_operator(lattice: HTGLattice, U_ev: float) -> np.ndarray:
    diagonal = np.zeros(lattice.matrix_dim, dtype=float)
    layer_values = (float(U_ev), 0.0, -float(U_ev))
    for ig in range(lattice.n_g):
        for layer_index, value in enumerate(layer_values):
            start = 6 * ig + 2 * layer_index
            diagonal[start : start + 2] = value
    return np.diag(diagonal).astype(np.complex128)


def _rectangular_g_embedding(lattice: HTGLattice) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    mins = np.min(lattice.g_indices, axis=0)
    maxs = np.max(lattice.g_indices, axis=0)
    grid_shape = (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1))
    origin = (int(mins[0]), int(mins[1]))
    positions = {
        (int(n1), int(n2)): (int(n1 - mins[0]), int(n2 - mins[1]))
        for n1, n2 in np.asarray(lattice.g_indices, dtype=int)
    }
    return grid_shape, origin, positions


def _central_chern_basis_at_k(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    interaction: InteractionParams,
    *,
    valley: int,
    central_pair: tuple[int, int],
    sigma_z_operator: np.ndarray,
    layer_potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hmat = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if interaction.U_ev != 0.0:
        hmat = hmat + layer_potential
    subset = (int(central_pair[0]), int(central_pair[1]))
    central_evals, central_evecs = eigh(hmat, subset_by_index=subset, driver="evr")
    central_evals = np.asarray(central_evals, dtype=float)
    central_evecs = np.asarray(central_evecs, dtype=np.complex128)

    projected_sigma = central_evecs.conjugate().T @ sigma_z_operator @ central_evecs
    sigma_eigs, sigma_rot = np.linalg.eigh(projected_sigma)
    # Return positive-sigma (A-like) then negative-sigma (B-like).
    order = np.asarray([int(np.argmax(sigma_eigs)), int(np.argmin(sigma_eigs))], dtype=int)
    rot = np.asarray(sigma_rot[:, order], dtype=np.complex128)
    wavefunctions = central_evecs @ rot
    h_projected = rot.conjugate().T @ np.diag(central_evals) @ rot
    sigma_projected = rot.conjugate().T @ projected_sigma @ rot
    return wavefunctions, h_projected, sigma_projected, sigma_eigs[order]


def _hybrid_projected_basis_at_k(
    k_tilde: complex,
    lattice: HTGLattice,
    params: HTGParams,
    interaction: InteractionParams,
    *,
    valley: int,
    projected_indices: tuple[int, ...],
    central_pair: tuple[int, int],
    sigma_z_operator: np.ndarray,
    layer_potential: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected_indices_array = np.asarray(projected_indices, dtype=int)
    central_pair_array = np.asarray(central_pair, dtype=int)
    if projected_indices_array.ndim != 1 or projected_indices_array.size < 2:
        raise ValueError(f"Expected at least two projected band indices, got {projected_indices}")
    if not set(int(index) for index in central_pair).issubset(set(int(index) for index in projected_indices)):
        raise ValueError(f"Projected indices {projected_indices} must contain central pair {central_pair}")

    hmat = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if interaction.U_ev != 0.0:
        hmat = hmat + layer_potential
    subset = (int(np.min(projected_indices_array)), int(np.max(projected_indices_array)))
    evals_subset, evecs_subset = eigh(hmat, subset_by_index=subset, driver="evr")
    evals_subset = np.asarray(evals_subset, dtype=float)
    evecs_subset = np.asarray(evecs_subset, dtype=np.complex128)
    local_position = {int(index): int(index - subset[0]) for index in projected_indices_array}

    central_evals = np.asarray([evals_subset[local_position[int(index)]] for index in central_pair_array], dtype=float)
    central_evecs = np.column_stack(
        [evecs_subset[:, local_position[int(index)]] for index in central_pair_array]
    ).astype(np.complex128, copy=False)
    projected_sigma = central_evecs.conjugate().T @ sigma_z_operator @ central_evecs
    sigma_eigs, sigma_rot = np.linalg.eigh(projected_sigma)
    central_order = np.asarray([int(np.argmax(sigma_eigs)), int(np.argmin(sigma_eigs))], dtype=int)
    central_rot = np.asarray(sigma_rot[:, central_order], dtype=np.complex128)
    central_wavefunctions = central_evecs @ central_rot
    central_h = central_rot.conjugate().T @ np.diag(central_evals) @ central_rot

    lower_indices = tuple(int(index) for index in projected_indices if int(index) < int(central_pair[0]))
    upper_indices = tuple(int(index) for index in projected_indices if int(index) > int(central_pair[1]))
    ordered_vectors: list[np.ndarray] = []
    for index in lower_indices:
        ordered_vectors.append(np.asarray(evecs_subset[:, local_position[index]], dtype=np.complex128))
    for col in range(2):
        ordered_vectors.append(np.asarray(central_wavefunctions[:, col], dtype=np.complex128))
    for index in upper_indices:
        ordered_vectors.append(np.asarray(evecs_subset[:, local_position[index]], dtype=np.complex128))

    wavefunctions = np.column_stack(ordered_vectors).astype(np.complex128, copy=False)
    n_projected = wavefunctions.shape[1]
    h_projected = np.zeros((n_projected, n_projected), dtype=np.complex128)
    for out_pos, index in enumerate(lower_indices):
        h_projected[out_pos, out_pos] = float(evals_subset[local_position[index]])
    central_start = len(lower_indices)
    h_projected[central_start : central_start + 2, central_start : central_start + 2] = central_h
    for offset, index in enumerate(upper_indices):
        pos = central_start + 2 + offset
        h_projected[pos, pos] = float(evals_subset[local_position[index]])

    sigma_projected = wavefunctions.conjugate().T @ sigma_z_operator @ wavefunctions
    sigma_diagonal = np.real(np.diag(sigma_projected))
    return wavefunctions, h_projected, sigma_projected, sigma_diagonal


def centered_projection_band_indices(matrix_dim: int, projected_band_count: int) -> tuple[int, ...]:
    projected_band_count = int(projected_band_count)
    if projected_band_count < 2 or projected_band_count % 2 != 0:
        raise ValueError(f"projected_band_count must be an even integer >= 2, got {projected_band_count}")
    return tuple(int(index) for index in centered_band_indices(int(matrix_dim), projected_band_count))


def _build_htg_projected_basis_from_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    mesh_size: int,
    k_grid_frac: np.ndarray,
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    lattice = model.lattice
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    projected_indices = centered_projection_band_indices(lattice.matrix_dim, projected_band_count)
    n_projected = len(projected_indices)
    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)

    grid_shape, origin, positions = _rectangular_g_embedding(lattice)
    nx, ny = grid_shape
    embedded = np.zeros((6, nx, ny, n_projected, 2, kvec.size), dtype=np.complex128)
    h_projected = np.zeros((n_projected, n_projected, 2, kvec.size), dtype=np.complex128)
    sigma_projected = np.zeros_like(h_projected)
    band_sigma_z = np.zeros((n_projected, 2, kvec.size), dtype=float)
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)

    for iflavor, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(kvec):
            wavefunctions, h_block, sigma_block, sigma_values = _hybrid_projected_basis_at_k(
                complex(kval),
                lattice,
                model.params,
                interaction,
                valley=valley,
                projected_indices=projected_indices,
                central_pair=central_pair,
                sigma_z_operator=sigma_z_operator,
                layer_potential=layer_potential,
            )
            for source_g_index, pair in enumerate(lattice.g_indices):
                ix, iy = positions[(int(pair[0]), int(pair[1]))]
                start = 6 * source_g_index
                embedded[:, ix, iy, :, iflavor, ik] = wavefunctions[start : start + 6, :]
            h_projected[:, :, iflavor, ik] = h_block
            sigma_projected[:, :, iflavor, ik] = sigma_block
            band_sigma_z[:, iflavor, ik] = np.real(sigma_values)

    wavefunction_array = embedded.reshape((6 * nx * ny, n_projected, 2, kvec.size), order="F")
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=6,
        name="htg_chern_sublattice",
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    sigma_z = np.zeros_like(h0)
    idx = np.arange(basis.nt, dtype=int).reshape((2, 2, n_projected), order="F")
    for ik in range(basis.nk):
        for ispin in range(2):
            for iflavor in range(2):
                block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                h0[:, :, ik][np.ix_(block_indices, block_indices)] = h_projected[:, :, iflavor, ik]
                sigma_z[:, :, ik][np.ix_(block_indices, block_indices)] = sigma_projected[:, :, iflavor, ik]

    return HTGProjectedBasisData(
        model=model,
        interaction=interaction,
        mesh_size=int(mesh_size),
        kvec=kvec,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        sigma_z=sigma_z,
        band_sigma_z=band_sigma_z,
        central_band_indices=central_pair,
        projected_band_indices=projected_indices,
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_cell_area_nm2=moire_cell_area_nm2(lattice),
    )


def build_htg_projected_basis(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    mesh_size: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_mesh = resolved_interaction.n_k if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError("mesh_size must be positive")

    k_grid_frac, kvec_grid = build_moire_k_grid(model.lattice, resolved_mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid.reshape(-1), dtype=np.complex128)
    return _build_htg_projected_basis_from_kvec(
        model,
        resolved_interaction,
        kvec,
        mesh_size=resolved_mesh,
        k_grid_frac=k_grid_frac,
        projected_band_count=projected_band_count,
    )


def build_htg_projected_basis_for_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    projected_band_count: int = 2,
) -> HTGProjectedBasisData:
    kvec_array = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    return _build_htg_projected_basis_from_kvec(
        model,
        interaction,
        kvec_array,
        mesh_size=0,
        k_grid_frac=np.zeros((kvec_array.size, 2), dtype=float),
        projected_band_count=projected_band_count,
    )


def reciprocal_shift_labels(g_shells: int) -> tuple[int, ...]:
    g_shells = int(g_shells)
    if g_shells < 0:
        raise ValueError("g_shells must be non-negative")
    return tuple(range(-g_shells, g_shells + 1))


def _infer_g_shells_from_overlap_blocks(overlap_blocks: HFOverlapBlockSet) -> int:
    if not overlap_blocks.shifts:
        return 0
    return int(max(max(abs(int(m)), abs(int(n))) for m, n in overlap_blocks.shifts))


def build_htg_overlap_blocks(
    basis_data: HTGProjectedBasisData,
    *,
    g_shells: int | None = None,
) -> HFOverlapBlockSet:
    interaction = basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray(
        [m * basis_data.model.lattice.b_m1 + n * basis_data.model.lattice.b_m2 for m, n in shifts],
        dtype=np.complex128,
    )
    overlaps = {
        shift: calculate_projected_overlap_between(basis_data.basis, basis_data.basis, shift[0], shift[1])
        for shift in shifts
    }
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        overlap = overlaps[shift]
        diagonal_overlaps[shift] = np.diagonal(overlap, axis1=1, axis2=3)
        hartree_screening[shift] = float(screened_coulomb_matrix(np.asarray(gvec), interaction))
        qvals = basis_data.kvec[None, :] - basis_data.kvec[:, None] + complex(gvec)
        fock_screening[shift] = screened_coulomb_matrix(qvals, interaction)

    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_htg_overlap_blocks_between(
    target_basis_data: HTGProjectedBasisData,
    source_basis_data: HTGProjectedBasisData,
    *,
    g_shells: int | None = None,
    include_hartree: bool = True,
) -> HFOverlapBlockSet:
    if target_basis_data.model.lattice is not source_basis_data.model.lattice:
        target_lattice = target_basis_data.model.lattice
        source_lattice = source_basis_data.model.lattice
        if not np.array_equal(target_lattice.g_indices, source_lattice.g_indices):
            raise ValueError("Target and source HTG bases must use the same plane-wave G-index set")
    if target_basis_data.basis.grid_shape != source_basis_data.basis.grid_shape:
        raise ValueError("Target and source HTG projected bases must use the same reciprocal embedding grid")

    interaction = target_basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray(
        [m * target_basis_data.model.lattice.b_m1 + n * target_basis_data.model.lattice.b_m2 for m, n in shifts],
        dtype=np.complex128,
    )
    overlaps = {
        shift: calculate_projected_overlap_between(
            target_basis_data.basis,
            source_basis_data.basis,
            shift[0],
            shift[1],
        )
        for shift in shifts
    }
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if target_basis_data.nk == source_basis_data.nk:
            diagonal_overlaps[shift] = np.diagonal(overlaps[shift], axis1=1, axis2=3)
        if include_hartree:
            hartree_screening[shift] = float(screened_coulomb_matrix(np.asarray(gvec), interaction))
        qvals = source_basis_data.kvec[None, :] - target_basis_data.kvec[:, None] + complex(gvec)
        fock_screening[shift] = screened_coulomb_matrix(qvals, interaction)

    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
