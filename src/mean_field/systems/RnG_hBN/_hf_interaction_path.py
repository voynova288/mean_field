from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_types import *  # noqa: F401,F403
from ._hf_basis import *  # noqa: F401,F403

def _resolve_basis_valleys(n_flavor: int, valleys: tuple[int, ...] | None) -> tuple[int, ...]:
    n = int(n_flavor)
    if n <= 0:
        raise ValueError(f"n_flavor must be positive, got {n_flavor}")
    if valleys is None:
        if n > len(VALLEY_SEQUENCE):
            raise ValueError(f"Need explicit valleys for n_flavor={n}")
        resolved = tuple(int(value) for value in VALLEY_SEQUENCE[:n])
    else:
        resolved = tuple(int(value) for value in valleys)
    if len(resolved) != n:
        raise ValueError(f"Expected {n} valley labels, got {resolved}")
    for valley in resolved:
        if valley not in VALLEY_SEQUENCE:
            raise ValueError(f"Expected valley labels in {VALLEY_SEQUENCE}, got {resolved}")
    return resolved


def _rlg_hbn_layer_local_indices(
    basis: ProjectedWavefunctionBasis,
    layer: int,
    *,
    layer_count: int,
) -> np.ndarray:
    layer_index = int(layer)
    if layer_index < 0 or layer_index >= int(layer_count):
        raise ValueError(f"layer index {layer_index} outside [0, {int(layer_count)})")
    group_name = f"layer_{layer_index}"
    if any(group.name == group_name for group in basis.component_groups):
        return component_group_indices(basis, group_name)
    return component_group_indices(basis, ComponentGroup(group_name, np.asarray([2 * layer_index, 2 * layer_index + 1])))



def calculate_layer_projected_overlap_between(
    target: ProjectedWavefunctionBasis,
    source: ProjectedWavefunctionBasis,
    m: int,
    n: int,
    *,
    layer_count: int,
    valleys: tuple[int, ...] | None = None,
) -> np.ndarray:
    if target.local_basis_size != source.local_basis_size:
        raise ValueError(f"local_basis_size mismatch: {target.local_basis_size} != {source.local_basis_size}")
    if target.n_flavor != source.n_flavor:
        raise ValueError(f"n_flavor mismatch: {target.n_flavor} != {source.n_flavor}")
    if target.n_spin != source.n_spin:
        raise ValueError(f"n_spin mismatch: {target.n_spin} != {source.n_spin}")
    if target.grid_shape != source.grid_shape:
        raise ValueError(f"grid_shape mismatch: {target.grid_shape} != {source.grid_shape}")
    layer_count = int(layer_count)
    if layer_count <= 0:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    if target.local_basis_size != 2 * layer_count:
        raise ValueError(
            f"Expected local_basis_size={2 * layer_count} for {layer_count} layers, "
            f"got {target.local_basis_size}"
        )
    resolved_valleys = _resolve_basis_valleys(target.n_flavor, valleys)

    nx, ny = target.grid_shape
    target_band_k = target.n_band * target.nk
    source_band_k = source.n_band * source.nk
    layer_blocks = np.zeros(
        (
            layer_count,
            target.n_spin,
            target.n_flavor,
            target_band_k,
            target.n_spin,
            target.n_flavor,
            source_band_k,
        ),
        dtype=np.complex128,
        order="F",
    )

    for iflavor, valley in enumerate(resolved_valleys):
        target_grid = target.wavefunctions[:, :, iflavor, :].reshape(
            target.local_basis_size,
            nx,
            ny,
            target_band_k,
            order="F",
        )
        source_grid = source.wavefunctions[:, :, iflavor, :].reshape(
            source.local_basis_size,
            nx,
            ny,
            source_band_k,
            order="F",
        )
        raw_m, raw_n = _raw_overlap_shift_for_physical_g((m, n), valley=int(valley))
        shifted = shift_wavefunction_grid(source_grid, -raw_m, -raw_n, boundary_mode="zero_fill", grid_axes=(1, 2))
        for layer in range(layer_count):
            target_layer_indices = _rlg_hbn_layer_local_indices(target, layer, layer_count=layer_count)
            source_layer_indices = _rlg_hbn_layer_local_indices(source, layer, layer_count=layer_count)
            if target_layer_indices.size != source_layer_indices.size:
                raise ValueError(
                    f"Target/source layer {layer} component sizes differ: "
                    f"{target_layer_indices.size} != {source_layer_indices.size}"
                )
            layer_local_size = int(target_layer_indices.size)
            target_layer = target_grid[target_layer_indices, :, :, :].reshape(
                layer_local_size * nx * ny, target_band_k, order="F"
            )
            shifted_layer = shifted[source_layer_indices, :, :, :].reshape(
                layer_local_size * nx * ny, source_band_k, order="F"
            )
            layer_overlap = target_layer.conj().T @ shifted_layer
            for ispin in range(target.n_spin):
                layer_blocks[layer, ispin, iflavor, :, ispin, iflavor, :] = layer_overlap

    return layer_blocks.reshape((layer_count, target.nt, target.nk, source.nt, source.nk), order="F")


def diagonal_layer_overlap_blocks(layer_overlap: np.ndarray) -> np.ndarray:
    layer_overlap = np.asarray(layer_overlap, dtype=np.complex128)
    if layer_overlap.ndim != 5:
        raise ValueError(f"Expected layer overlap shape (layer, nt, nk, nt, nk), got {layer_overlap.shape}")
    if layer_overlap.shape[1] != layer_overlap.shape[3]:
        raise ValueError(f"Expected square flavor dimensions in layer overlap, got {layer_overlap.shape}")
    if layer_overlap.shape[2] != layer_overlap.shape[4]:
        raise ValueError(f"Expected equal source/target k counts for diagonal overlap, got {layer_overlap.shape}")
    return np.diagonal(layer_overlap, axis1=2, axis2=4)


def interaction_shifts_for_cutoff(
    lattice: RLGhBNLattice,
    interaction: RLGhBNInteractionParams,
) -> tuple[tuple[int, int], ...]:
    q1_norm = float(abs(lattice.q_complex[0]))
    cutoff = float(interaction.interaction_cutoff_q1) * q1_norm
    shortest = max(min(abs(lattice.g_m1), abs(lattice.g_m2)), 1.0e-15)
    coefficient_bound = int(math.ceil(cutoff / shortest)) + 2
    entries: list[tuple[float, int, int]] = []
    for m in range(-coefficient_bound, coefficient_bound + 1):
        for n in range(-coefficient_bound, coefficient_bound + 1):
            gvec = complex(m * lattice.g_m1 + n * lattice.g_m2)
            if abs(gvec) <= cutoff + 1.0e-12:
                entries.append((round(abs(gvec), 12), int(m), int(n)))
    entries.sort(key=lambda item: (item[0], item[1] * item[1] + item[2] * item[2], item[1], item[2]))
    return tuple((m, n) for _, m, n in entries)


def _layer_coulomb_tensor_for_qvals(
    qvals: np.ndarray,
    *,
    layer_count: int,
    interaction: RLGhBNInteractionParams,
    layer_spacing_nm: float,
) -> np.ndarray:
    q_array = np.asarray(qvals, dtype=np.complex128)
    tensor = np.zeros(q_array.shape + (int(layer_count), int(layer_count)), dtype=float)
    for index in np.ndindex(q_array.shape):
        tensor[index] = layer_coulomb_matrix_mev_nm2(
            abs(complex(q_array[index])),
            int(layer_count),
            interaction,
            layer_spacing_nm=layer_spacing_nm,
        )
    return tensor


def build_rlg_hbn_layer_overlap_blocks(
    basis_data: RLGhBNProjectedBasisData,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNLayerOverlapBlockSet:
    return build_rlg_hbn_layer_overlap_blocks_between(basis_data, basis_data, shifts=shifts)


def build_rlg_hbn_layer_overlap_blocks_between(
    target_basis_data: RLGhBNProjectedBasisData,
    source_basis_data: RLGhBNProjectedBasisData,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNLayerOverlapBlockSet:
    if target_basis_data.reciprocal_grid_shape != source_basis_data.reciprocal_grid_shape:
        raise ValueError(
            "Target/source reciprocal grid shapes differ: "
            f"{target_basis_data.reciprocal_grid_shape} != {source_basis_data.reciprocal_grid_shape}"
        )
    if target_basis_data.reciprocal_grid_origin != source_basis_data.reciprocal_grid_origin:
        raise ValueError(
            "Target/source reciprocal grid origins differ: "
            f"{target_basis_data.reciprocal_grid_origin} != {source_basis_data.reciprocal_grid_origin}"
        )
    if target_basis_data.valleys != source_basis_data.valleys:
        raise ValueError(
            "Target/source valley order differs: "
            f"{target_basis_data.valleys} != {source_basis_data.valleys}"
        )

    resolved_shifts = (
        shifts
        if shifts is not None
        else interaction_shifts_for_cutoff(source_basis_data.basis_model.lattice, source_basis_data.interaction)
    )
    resolved_shifts = tuple((int(m), int(n)) for m, n in resolved_shifts)
    gvecs = np.asarray(
        [
            m * source_basis_data.basis_model.lattice.g_m1 + n * source_basis_data.basis_model.lattice.g_m2
            for m, n in resolved_shifts
        ],
        dtype=np.complex128,
    )

    layer_overlaps: dict[tuple[int, int], np.ndarray] = {}
    layer_diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    fock_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    layer_count = source_basis_data.basis_model.params.layer_count
    layer_spacing = source_basis_data.basis_model.params.layer_spacing_nm

    for shift, gvec in zip(resolved_shifts, gvecs, strict=True):
        overlap = calculate_layer_projected_overlap_between(
            target_basis_data.basis,
            source_basis_data.basis,
            shift[0],
            shift[1],
            layer_count=layer_count,
            valleys=target_basis_data.valleys,
        )
        layer_overlaps[shift] = overlap
        if target_basis_data.nk == source_basis_data.nk and target_basis_data.nt == source_basis_data.nt:
            layer_diagonal_overlaps[shift] = diagonal_layer_overlap_blocks(overlap)
        hartree_layer_coulomb[shift] = layer_coulomb_matrix_mev_nm2(
            abs(complex(gvec)),
            layer_count,
            source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )
        qvals = target_basis_data.kvec[:, None] - source_basis_data.kvec[None, :] + complex(gvec)
        fock_layer_coulomb[shift] = _layer_coulomb_tensor_for_qvals(
            qvals,
            layer_count=layer_count,
            interaction=source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )

    return RLGhBNLayerOverlapBlockSet(
        shifts=resolved_shifts,
        gvecs=gvecs,
        layer_overlaps=layer_overlaps,
        layer_diagonal_overlaps=layer_diagonal_overlaps,
        hartree_layer_coulomb=hartree_layer_coulomb,
        fock_layer_coulomb=fock_layer_coulomb,
    )


def _contract_layer_fock_term(
    left_overlap: np.ndarray,
    density_delta: np.ndarray,
    coeff_matrix: np.ndarray,
    right_overlap: np.ndarray,
) -> np.ndarray:
    left_overlap = np.asarray(left_overlap, dtype=np.complex128)
    right_overlap = np.asarray(right_overlap, dtype=np.complex128)
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    coeff_matrix = np.asarray(coeff_matrix)
    nt_target, nk_target, nt_source, nk_source = left_overlap.shape
    if right_overlap.shape != left_overlap.shape:
        raise ValueError(f"Expected right_overlap shape {left_overlap.shape}, got {right_overlap.shape}")
    if density_delta.shape != (nt_source, nt_source, nk_source):
        raise ValueError(f"Expected density_delta shape {(nt_source, nt_source, nk_source)}, got {density_delta.shape}")
    if coeff_matrix.shape != (nk_target, nk_source):
        raise ValueError(f"Expected coeff_matrix shape {(nk_target, nk_source)}, got {coeff_matrix.shape}")

    if _rlg_hbn_use_numba():
        if not _NUMBA_AVAILABLE:
            if _rlg_hbn_require_numba():
                raise RuntimeError("MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1 but numba is not available")
        else:
            return _contract_layer_fock_term_numba_kernel(
                np.ascontiguousarray(left_overlap),
                np.ascontiguousarray(density_delta),
                np.ascontiguousarray(coeff_matrix),
                np.ascontiguousarray(right_overlap),
            )

    left_blocks = np.transpose(left_overlap, (1, 3, 0, 2))
    right_blocks = np.transpose(right_overlap, (1, 3, 0, 2))
    density_t = np.transpose(density_delta, (2, 1, 0))
    intermediate = np.einsum("tsac,scd->tsad", left_blocks, density_t, optimize=True)
    fock = np.einsum("ts,tsad,tsbd->tab", coeff_matrix, intermediate, np.conj(right_blocks), optimize=True)
    return np.transpose(fock, (1, 2, 0))


def build_rlg_hbn_interaction_components(
    density_delta: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
) -> RLGhBNInteractionComponents:
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    if density_delta.ndim != 3 or density_delta.shape[0] != density_delta.shape[1]:
        raise ValueError(f"Expected density_delta shape (nt, nt, nk), got {density_delta.shape}")
    nt, _, nk = density_delta.shape
    scale = float(beta) * float(v0) / float(nk)
    hartree = np.zeros_like(density_delta)
    fock = np.zeros_like(density_delta)

    for shift in overlap_blocks.shifts:
        layer_diagonal = overlap_blocks.layer_diagonal_overlaps[shift]
        layer_overlap = overlap_blocks.layer_overlaps[shift]
        hartree_kernel = overlap_blocks.hartree_layer_coulomb[shift]
        fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, overlap_blocks.fock_layer_coulomb[shift])
        if layer_diagonal.shape[1:] != (nt, nt, nk):
            raise ValueError(f"Layer diagonal overlap for {shift} is incompatible with density shape {density_delta.shape}")
        if layer_overlap.shape[1:] != (nt, nk, nt, nk):
            raise ValueError(f"Layer overlap for {shift} is incompatible with density shape {density_delta.shape}")

        layer_traces = np.asarray(
            [
                compute_density_overlap_trace_from_diagonal(density_delta, layer_diagonal[layer])
                for layer in range(layer_diagonal.shape[0])
            ],
            dtype=np.complex128,
        )
        for target_layer in range(layer_diagonal.shape[0]):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hartree += prefactor * layer_diagonal[target_layer]

        for target_layer in range(layer_overlap.shape[0]):
            for source_layer in range(layer_overlap.shape[0]):
                coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                if np.any(coeff != 0.0):
                    fock -= _contract_layer_fock_term(
                        layer_overlap[target_layer],
                        density_delta,
                        coeff,
                        layer_overlap[source_layer],
                    )

    return RLGhBNInteractionComponents(hartree=hartree, fock=fock, total=hartree + fock)


def build_rlg_hbn_hf_interaction_hamiltonian(
    density_delta: np.ndarray,
    overlap_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
) -> np.ndarray:
    return build_rlg_hbn_interaction_components(
        density_delta,
        overlap_blocks,
        v0=float(v0),
        beta=float(beta),
    ).total


def build_rlg_hbn_target_hamiltonian(
    base_hamiltonian: np.ndarray,
    density_delta: np.ndarray,
    *,
    source_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    target_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    target_source_overlap_blocks: RLGhBNLayerOverlapBlockSet,
    v0: float,
    beta: float = 1.0,
) -> np.ndarray:
    base = np.asarray(base_hamiltonian, dtype=np.complex128)
    density = np.asarray(density_delta, dtype=np.complex128)
    if base.ndim != 3 or base.shape[0] != base.shape[1]:
        raise ValueError(f"Expected base_hamiltonian shape (nt, nt, nk_target), got {base.shape}")
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected density_delta shape (nt, nt, nk_source), got {density.shape}")
    nt_target, _, nk_target = base.shape
    nt_source = int(density.shape[0])

    nk_source = int(density.shape[2])
    scale = float(beta) * float(v0) / float(nk_source)
    hamiltonian = base.copy()

    for shift in source_overlap_blocks.shifts:
        if shift not in target_overlap_blocks.layer_diagonal_overlaps:
            raise ValueError(f"Missing target diagonal overlaps for shift {shift}")
        if shift not in target_source_overlap_blocks.layer_overlaps:
            raise ValueError(f"Missing target-source overlaps for shift {shift}")

        source_layer_diagonal = source_overlap_blocks.layer_diagonal_overlaps[shift]
        target_layer_diagonal = target_overlap_blocks.layer_diagonal_overlaps[shift]
        target_source_layer_overlap = target_source_overlap_blocks.layer_overlaps[shift]
        hartree_kernel = source_overlap_blocks.hartree_layer_coulomb[shift]
        fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, target_source_overlap_blocks.fock_layer_coulomb[shift])

        if source_layer_diagonal.shape[1:] != (nt_source, nt_source, nk_source):
            raise ValueError(
                f"Source layer diagonal overlap for {shift} is incompatible with density shape {density.shape}"
            )
        if target_layer_diagonal.shape[1:] != (nt_target, nt_target, nk_target):
            raise ValueError(
                f"Target layer diagonal overlap for {shift} is incompatible with base shape {base.shape}"
            )
        if target_source_layer_overlap.shape[1:] != (nt_target, nk_target, nt_source, nk_source):
            raise ValueError(
                f"Target-source layer overlap for {shift} is incompatible with target/source shapes "
                f"{base.shape} and {density.shape}"
            )

        layer_traces = np.asarray(
            [
                compute_density_overlap_trace_from_diagonal(density, source_layer_diagonal[layer])
                for layer in range(source_layer_diagonal.shape[0])
            ],
            dtype=np.complex128,
        )
        for target_layer in range(target_layer_diagonal.shape[0]):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hamiltonian += prefactor * target_layer_diagonal[target_layer]

        for target_layer in range(target_source_layer_overlap.shape[0]):
            for source_layer in range(target_source_layer_overlap.shape[0]):
                coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                if np.any(coeff != 0.0):
                    hamiltonian -= _contract_layer_fock_term(
                        target_source_layer_overlap[target_layer],
                        density,
                        coeff,
                        target_source_layer_overlap[source_layer],
                    )

    _hermitize_blocks_inplace(hamiltonian)
    return hamiltonian


def _diagonalize_hf_path_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError(f"Expected Hamiltonian shape (nt, nt, nk), got {hamiltonian.shape}")
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def evaluate_rlg_hbn_hf_path(
    run: RLGhBNHartreeFockRun,
    path: KPath,
    *,
    beta: float = 1.0,
    chunk_size: int = 4,
) -> RLGhBNHFPathResult:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    source_basis_data = run.basis_data
    source_overlap_blocks = run.overlap_blocks
    density = np.asarray(run.state.density, dtype=np.complex128)
    kvec = np.asarray(path.kvec, dtype=np.complex128)
    remote_source = _prepare_remote_average_source(source_basis_data)

    hamiltonian_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []
    basis_chunks: list[RLGhBNProjectedBasisData] = []
    for start in range(0, kvec.size, int(chunk_size)):
        stop = min(start + int(chunk_size), kvec.size)
        target_basis_data = build_rlg_hbn_projected_basis_for_kvec(
            source_basis_data.basis_model,
            source_basis_data.interaction,
            kvec[start:stop],
            physical_model=source_basis_data.model,
            active_band_indices=source_basis_data.active_band_indices,
            valleys=source_basis_data.valleys,
        )
        fixed_remote = _remote_average_hamiltonian_from_source(
            target_basis_data,
            source_basis_data,
            remote_source,
            shifts=source_overlap_blocks.shifts,
            beta=beta,
        )
        target_basis_data = replace(
            target_basis_data,
            h0=np.asarray(target_basis_data.physical_h0, dtype=np.complex128) + fixed_remote,
            fixed_remote_hamiltonian=fixed_remote,
        )
        _assert_average_remote_hamiltonian_contract(target_basis_data)
        target_overlap_blocks = build_rlg_hbn_layer_overlap_blocks(
            target_basis_data,
            shifts=source_overlap_blocks.shifts,
        )
        target_source_overlap_blocks = build_rlg_hbn_layer_overlap_blocks_between(
            target_basis_data,
            source_basis_data,
            shifts=source_overlap_blocks.shifts,
        )
        chunk_hamiltonian = build_rlg_hbn_target_hamiltonian(
            target_basis_data.h0,
            density,
            source_overlap_blocks=source_overlap_blocks,
            target_overlap_blocks=target_overlap_blocks,
            target_source_overlap_blocks=target_source_overlap_blocks,
            v0=run.state.v0,
            beta=beta,
        )
        hamiltonian_chunks.append(chunk_hamiltonian)
        energy_chunks.append(_diagonalize_hf_path_hamiltonian(chunk_hamiltonian))
        basis_chunks.append(target_basis_data)

    hamiltonian = np.concatenate(hamiltonian_chunks, axis=2)
    energies = np.concatenate(energy_chunks, axis=1)
    first_basis = basis_chunks[0]
    basis_data = RLGhBNProjectedBasisData(
        model=first_basis.model,
        basis_model=first_basis.basis_model,
        interaction=first_basis.interaction,
        screening=None,
        mesh_size=0,
        kvec=kvec,
        k_grid_frac=np.zeros((kvec.size, 2), dtype=float),
        basis=ProjectedWavefunctionBasis(
            wavefunctions=np.concatenate([chunk.basis.wavefunctions for chunk in basis_chunks], axis=3),
            grid_shape=first_basis.basis.grid_shape,
            n_spin=first_basis.basis.n_spin,
            local_basis_size=first_basis.basis.local_basis_size,
            name=first_basis.basis.name,
            component_groups=first_basis.basis.component_groups,
        ),
        h0=np.concatenate([chunk.h0 for chunk in basis_chunks], axis=2),
        band_energies=np.concatenate([chunk.band_energies for chunk in basis_chunks], axis=2),
        active_band_indices=first_basis.active_band_indices,
        flat_band_indices=first_basis.flat_band_indices,
        valleys=first_basis.valleys,
        reciprocal_grid_shape=first_basis.reciprocal_grid_shape,
        reciprocal_grid_origin=first_basis.reciprocal_grid_origin,
        moire_cell_area_nm2=first_basis.moire_cell_area_nm2,
        physical_h0=np.concatenate([np.asarray(chunk.physical_h0, dtype=np.complex128) for chunk in basis_chunks], axis=2),
        fixed_remote_hamiltonian=np.concatenate(
            [np.asarray(chunk.fixed_remote_hamiltonian, dtype=np.complex128) for chunk in basis_chunks],
            axis=2,
        ),
    )
    return RLGhBNHFPathResult(
        path=path,
        basis_data=basis_data,
        hamiltonian=hamiltonian,
        energies=energies,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
