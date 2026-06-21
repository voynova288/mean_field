from __future__ import annotations

from ._supercell_shared import *  # noqa: F401,F403
from ._supercell_types import *  # noqa: F401,F403
from ._supercell_geometry import *  # noqa: F401,F403

def htg_supercell_reference_diagonal(primitive_projected_band_count: int, area_ratio: int) -> np.ndarray:
    primitive_projected_band_count = int(primitive_projected_band_count)
    area_ratio = int(area_ratio)
    if primitive_projected_band_count < 2 or primitive_projected_band_count % 2 != 0:
        raise ValueError(f"primitive_projected_band_count must be an even integer >= 2, got {primitive_projected_band_count}")
    lower_count = (primitive_projected_band_count - 2) // 2
    primitive_reference = np.zeros(primitive_projected_band_count, dtype=float)
    primitive_reference[:lower_count] = 1.0
    primitive_reference[lower_count : lower_count + 2] = 0.5
    return np.repeat(primitive_reference, area_ratio).astype(float)


def htg_supercell_occupied_count_per_k(
    primitive_nu: float,
    *,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_sector: int = 4,
    atol: float = 1.0e-9,
) -> int:
    return occupied_count_from_primitive_filling(
        float(primitive_nu),
        reference_diagonal=np.asarray(reference_diagonal, dtype=float),
        area_ratio=int(area_ratio),
        n_sector=int(n_sector),
        atol=float(atol),
    )


def htg_supercell_filling_from_density(
    density: np.ndarray,
    *,
    reference_diagonal: np.ndarray,
    area_ratio: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    projector = density + _supercell_reference_density_blocks(
        nt,
        nk,
        reference_diagonal=reference_diagonal,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    particles_per_k = float(np.trace(projector, axis1=0, axis2=1).real.sum()) / float(nk)
    reference_total = float(n_spin) * float(n_eta) * float(np.sum(np.asarray(reference_diagonal, dtype=float)))
    return float((particles_per_k - reference_total) / float(area_ratio))


def _supercell_reference_density_blocks(
    nt: int,
    nk: int,
    *,
    reference_diagonal: np.ndarray,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    reference = np.asarray(reference_diagonal, dtype=float).reshape(-1)
    n_band = int(reference.size)
    if int(nt) != int(n_spin) * int(n_eta) * n_band:
        raise ValueError(f"nt={nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    idx = np.arange(int(nt), dtype=int).reshape((int(n_spin), int(n_eta), n_band), order="F")
    blocks = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            for iband in range(n_band):
                row = int(idx[ispin, ieta, iband])
                blocks[row, row, :] = float(reference[iband])
    return blocks


def _build_htg_supercell_projected_basis_from_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    supercell: HTGSupercell,
    mesh_size: int,
    k_grid_frac: np.ndarray | None,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    lattice = model.lattice
    super_g1, super_g2 = supercell.reciprocal_vectors(lattice)
    fold_reps = supercell_fold_representatives(supercell)
    area_ratio = int(supercell.area_ratio)
    if len(fold_reps) != area_ratio:
        raise ValueError(f"Expected {area_ratio} fold representatives, got {len(fold_reps)}")
    central_pair_raw = centered_band_indices(lattice.matrix_dim, 2)
    central_pair = (int(central_pair_raw[0]), int(central_pair_raw[1]))
    projected_indices = centered_projection_band_indices(lattice.matrix_dim, projected_band_count)
    primitive_band_count = len(projected_indices)
    folded_band_count_expected = folded_band_count(primitive_band_count, area_ratio)
    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)

    grid_shape, origin, positions = _supercell_embedding_table(lattice, supercell, fold_reps)
    nx, ny = grid_shape
    embedded = np.zeros((6, nx, ny, folded_band_count_expected, 2, kvec.size), dtype=np.complex128)
    h_projected = np.zeros((folded_band_count_expected, folded_band_count_expected, 2, kvec.size), dtype=np.complex128)
    sigma_projected = np.zeros_like(h_projected)
    band_sigma_z = np.zeros((folded_band_count_expected, 2, kvec.size), dtype=float)
    sigma_z_operator = sublattice_sigma_z(lattice)
    layer_potential = _layer_potential_operator(lattice, interaction.U_ev)

    for iflavor, valley in enumerate((1, -1)):
        for ik, kval in enumerate(kvec):
            for ifold, (fold_x, fold_y) in enumerate(fold_reps):
                primitive_k_full = complex(kval + fold_x * super_g1 + fold_y * super_g2)
                primitive_k, primitive_shift = _centered_primitive_reduction(lattice, primitive_k_full)
                wavefunctions, h_block, sigma_block, sigma_values = _hybrid_projected_basis_at_k(
                    primitive_k,
                    lattice,
                    model.params,
                    interaction,
                    valley=valley,
                    projected_indices=projected_indices,
                    central_pair=central_pair,
                    sigma_z_operator=sigma_z_operator,
                    layer_potential=layer_potential,
                )
                shift_n1, shift_n2 = primitive_shift
                primitive_shift_sc = supercell.primitive_shift_to_supercell(-shift_n1, -shift_n2)
                for iprim, _band_index in enumerate(projected_indices):
                    out_band = iprim * area_ratio + ifold
                    h_projected[out_band, out_band, iflavor, ik] = h_block[iprim, iprim]
                    band_sigma_z[out_band, iflavor, ik] = float(np.real(sigma_values[iprim]))
                    for jprim in range(primitive_band_count):
                        out_band_rhs = jprim * area_ratio + ifold
                        h_projected[out_band, out_band_rhs, iflavor, ik] = h_block[iprim, jprim]
                        sigma_projected[out_band, out_band_rhs, iflavor, ik] = sigma_block[iprim, jprim]
                    for source_g_index, pair in enumerate(np.asarray(lattice.g_indices, dtype=int)):
                        pair_shift_sc = supercell.primitive_shift_to_supercell(int(pair[0]), int(pair[1]))
                        sx = int(fold_x + primitive_shift_sc[0] + pair_shift_sc[0])
                        sy = int(fold_y + primitive_shift_sc[1] + pair_shift_sc[1])
                        target_position = positions.get((sx, sy))
                        if target_position is None:
                            continue
                        ix, iy = target_position
                        start = 6 * source_g_index
                        embedded[:, ix, iy, out_band, iflavor, ik] = wavefunctions[start : start + 6, iprim]

    wavefunction_array = embedded.reshape((6 * nx * ny, folded_band_count_expected, 2, kvec.size), order="F")
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=6,
        name="htg_supercell_folded_chern_sublattice",
        boundary_mode="zero_fill",
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    sigma_z = np.zeros_like(h0)
    idx = np.arange(basis.nt, dtype=int).reshape((2, 2, folded_band_count_expected), order="F")
    for ik in range(basis.nk):
        for ispin in range(2):
            for iflavor in range(2):
                block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                h0[:, :, ik][np.ix_(block_indices, block_indices)] = h_projected[:, :, iflavor, ik]
                sigma_z[:, :, ik][np.ix_(block_indices, block_indices)] = sigma_projected[:, :, iflavor, ik]

    reference_diagonal = htg_supercell_reference_diagonal(primitive_band_count, area_ratio)
    primitive_area = real_space_cell_area_nm2_from_reciprocal(lattice.b_m1, lattice.b_m2)
    return HTGSupercellProjectedBasisData(
        model=model,
        interaction=interaction,
        supercell=supercell,
        mesh_size=int(mesh_size),
        kvec=kvec,
        k_grid_frac=None if k_grid_frac is None else np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        sigma_z=sigma_z,
        band_sigma_z=band_sigma_z,
        primitive_projected_indices=projected_indices,
        primitive_band_count=primitive_band_count,
        fold_representatives=fold_reps,
        reference_diagonal=reference_diagonal,
        super_g1=complex(super_g1),
        super_g2=complex(super_g2),
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_supercell_area_nm2=float(primitive_area) * float(area_ratio),
    )


def build_htg_supercell_projected_basis(
    model: HTGModel,
    interaction: InteractionParams | None = None,
    *,
    supercell: HTGSupercell | None = None,
    mesh_size: int | None = None,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else InteractionParams()
    resolved_supercell = htg_default_fractional_supercell() if supercell is None else supercell
    resolved_mesh = resolved_interaction.n_k if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError("mesh_size must be positive")
    k_grid_frac, kvec_grid = build_htg_supercell_uniform_grid(model.lattice, resolved_supercell, resolved_mesh, endpoint=False)
    return _build_htg_supercell_projected_basis_from_kvec(
        model,
        resolved_interaction,
        np.asarray(kvec_grid, dtype=np.complex128),
        supercell=resolved_supercell,
        mesh_size=resolved_mesh,
        k_grid_frac=k_grid_frac,
        projected_band_count=projected_band_count,
    )


def build_htg_supercell_projected_basis_for_kvec(
    model: HTGModel,
    interaction: InteractionParams,
    kvec: np.ndarray,
    *,
    supercell: HTGSupercell,
    projected_band_count: int = 2,
) -> HTGSupercellProjectedBasisData:
    kvec_array = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    return _build_htg_supercell_projected_basis_from_kvec(
        model,
        interaction,
        kvec_array,
        supercell=supercell,
        mesh_size=0,
        k_grid_frac=None,
        projected_band_count=projected_band_count,
    )


def build_htg_supercell_overlap_blocks(
    basis_data: HTGSupercellProjectedBasisData,
    *,
    g_shells: int | None = None,
) -> HFOverlapBlockSet:
    return build_htg_supercell_overlap_blocks_between(basis_data, basis_data, g_shells=g_shells, include_hartree=True)


def build_htg_supercell_overlap_blocks_between(
    target_basis_data: HTGSupercellProjectedBasisData,
    source_basis_data: HTGSupercellProjectedBasisData,
    *,
    g_shells: int | None = None,
    include_hartree: bool = True,
) -> HFOverlapBlockSet:
    if target_basis_data.basis.grid_shape != source_basis_data.basis.grid_shape:
        raise ValueError("Target and source HTG supercell bases must use the same reciprocal embedding grid")
    interaction = target_basis_data.interaction
    resolved_shells = interaction.g_shells if g_shells is None else int(g_shells)
    labels = reciprocal_shift_labels(resolved_shells)
    shifts = tuple((m, n) for m in labels for n in labels)
    gvecs = np.asarray(
        [m * target_basis_data.super_g1 + n * target_basis_data.super_g2 for m, n in shifts],
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
