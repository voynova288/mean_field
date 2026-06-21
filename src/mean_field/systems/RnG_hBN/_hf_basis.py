from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_types import *  # noqa: F401,F403

def _rectangular_g_embedding(
    lattice: RLGhBNLattice,
    *,
    padding: int = 0,
) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
    pad = int(padding)
    if pad < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    mins = np.min(lattice.g_indices, axis=0) - pad
    maxs = np.max(lattice.g_indices, axis=0) + pad
    grid_shape = (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1))
    origin = (int(mins[0]), int(mins[1]))
    positions = {
        (int(n1), int(n2)): (int(n1 - mins[0]), int(n2 - mins[1]))
        for n1 in range(int(mins[0]), int(maxs[0]) + 1)
        for n2 in range(int(mins[1]), int(maxs[1]) + 1)
    }
    return grid_shape, origin, positions


def _reciprocal_fractional_coordinates(k_tilde: complex, lattice: RLGhBNLattice) -> np.ndarray:
    reciprocal = np.asarray(
        [
            [float(lattice.g_m1.real), float(lattice.g_m2.real)],
            [float(lattice.g_m1.imag), float(lattice.g_m2.imag)],
        ],
        dtype=float,
    )
    vector = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    return np.linalg.solve(reciprocal, vector)


def _fold_k_to_centered_cell(k_tilde: complex, lattice: RLGhBNLattice) -> tuple[complex, tuple[int, int]]:
    fractional = _reciprocal_fractional_coordinates(k_tilde, lattice)
    shift = np.floor(fractional + 0.5).astype(int)
    k_can = complex(k_tilde - int(shift[0]) * lattice.g_m1 - int(shift[1]) * lattice.g_m2)
    return k_can, (int(shift[0]), int(shift[1]))


def _raw_pair_from_canonical_pair(
    canonical_pair: tuple[int, int] | np.ndarray,
    shift: tuple[int, int],
    *,
    valley: int,
) -> tuple[int, int]:
    pair = np.asarray(canonical_pair, dtype=int)
    sign = int(valley)
    if sign not in VALLEY_SEQUENCE:
        raise ValueError(f"Expected valley in {VALLEY_SEQUENCE}, got {valley}")
    return (
        int(pair[0] - sign * int(shift[0])),
        int(pair[1] - sign * int(shift[1])),
    )


def _raw_overlap_shift_for_physical_g(
    shift: tuple[int, int] | np.ndarray,
    *,
    valley: int,
) -> tuple[int, int]:
    """Return the raw reciprocal-grid shift implementing paper Eq. (18).

    In the embedded RLG/hBN basis the K valley uses raw labels equal to the
    physical reciprocal labels, while the K' valley is stored in the
    time-reversal relabelled convention ``G_raw = -G_phys``. For a physical
    Umklapp vector ``G = m g1 + n g2``, Eq. (18) requires
    ``target_raw = source_raw + valley * G``. The low-level grid shifter used
    by :func:`calculate_layer_projected_overlap_between` implements
    ``target_raw = source_raw - raw_shift``. Hence ``raw_shift = -valley * G``.
    """

    pair = np.asarray(shift, dtype=int).reshape(2)
    sign = int(valley)
    if sign not in VALLEY_SEQUENCE:
        raise ValueError(f"Expected valley in {VALLEY_SEQUENCE}, got {valley}")
    return (-sign * int(pair[0]), -sign * int(pair[1]))


def _screened_basis_model(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    screening_mesh_size: int | None,
    screening_max_iter: int,
    screening_tolerance_mev: float,
    screening_mixing: float,
    screening_solver: str = "fixed_point",
    screening_result: ScreenedInterlayerPotentialResult | None = None,
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    screening_root_tolerance_mev: float = 1.0e-5,
) -> tuple[RLGhBNModel, ScreenedInterlayerPotentialResult | None]:
    if not interaction.use_screened_basis:
        return model, None
    if screening_result is not None:
        screening = screening_result
    elif screening_solver == "grid":
        screening = solve_screened_interlayer_potential_grid(
            model,
            interaction,
            mesh_size=screening_mesh_size,
            u_min_mev=screening_u_min_mev,
            u_max_mev=screening_u_max_mev,
            n_grid=screening_u_grid_points,
            root_tolerance_mev=screening_root_tolerance_mev,
        )
    elif screening_solver == "fixed_point":
        screening = solve_screened_interlayer_potential(
            model,
            interaction,
            mesh_size=screening_mesh_size,
            max_iter=screening_max_iter,
            tolerance_mev=screening_tolerance_mev,
            mixing=screening_mixing,
        )
    else:
        raise ValueError(f"screening_solver must be 'grid' or 'fixed_point', got {screening_solver!r}")
    screened_params = replace(model.params, displacement_field_mev=screening.screened_u_mev)
    return RLGhBNModel(lattice=model.lattice, params=screened_params), screening


def _assert_average_remote_hamiltonian_contract(basis_data: RLGhBNProjectedBasisData) -> None:
    if basis_data.interaction.scheme != "average":
        return
    if basis_data.physical_h0 is None:
        raise AssertionError("average scheme requires physical_h0")
    if basis_data.fixed_remote_hamiltonian is None:
        raise AssertionError("average scheme requires fixed_remote_hamiltonian")
    expected = np.asarray(basis_data.physical_h0, dtype=np.complex128) + np.asarray(
        basis_data.fixed_remote_hamiltonian,
        dtype=np.complex128,
    )
    if not np.allclose(np.asarray(basis_data.h0, dtype=np.complex128), expected, atol=1.0e-9, rtol=1.0e-9):
        raise AssertionError("average scheme h0 must equal physical_h0 + fixed_remote_hamiltonian")


def _project_physical_hamiltonian(
    selected_basis: np.ndarray,
    *,
    k_tilde: complex,
    physical_model: RLGhBNModel,
    valley: int,
) -> np.ndarray:
    selected = np.asarray(selected_basis, dtype=np.complex128)
    hamiltonian = build_hamiltonian(
        complex(k_tilde),
        physical_model.lattice,
        physical_model.params,
        valley=int(valley),
    )
    projected = selected.conjugate().T @ hamiltonian @ selected
    return 0.5 * (projected + projected.conjugate().T)


def _build_projected_basis_for_indices(
    *,
    physical_model: RLGhBNModel,
    basis_model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    kvec: np.ndarray,
    band_indices: tuple[int, ...],
    valleys: tuple[int, ...],
    mesh_size: int,
    k_grid_frac: np.ndarray,
    screening: ScreenedInterlayerPotentialResult | None,
    name: str,
    build_h0: bool = True,
) -> RLGhBNProjectedBasisData:
    resolved_kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    resolved_indices = tuple(int(value) for value in band_indices)
    resolved_valleys = tuple(int(value) for value in valleys)
    if resolved_kvec.size == 0:
        raise ValueError("At least one k point is required")
    if not resolved_indices:
        raise ValueError("At least one band index is required")
    if not resolved_valleys:
        raise ValueError("At least one valley is required")
    if min(resolved_indices) < 0 or max(resolved_indices) >= basis_model.matrix_dim:
        raise ValueError(
            f"Band indices must lie in [0, {basis_model.matrix_dim}), got {resolved_indices}"
        )

    n_projected = len(resolved_indices)
    grid_shape, origin, positions = _rectangular_g_embedding(
        basis_model.lattice,
        padding=RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    )
    nx, ny = grid_shape
    local_basis_size = int(2 * basis_model.params.layer_count)
    embedded = np.zeros(
        (local_basis_size, nx, ny, n_projected, len(resolved_valleys), resolved_kvec.size),
        dtype=np.complex128,
    )
    band_energies = np.zeros((n_projected, len(resolved_valleys), resolved_kvec.size), dtype=float)
    physical_blocks = (
        np.zeros(
            (n_projected, n_projected, len(resolved_valleys), resolved_kvec.size),
            dtype=np.complex128,
        )
        if build_h0
        else None
    )

    index_array = np.asarray(resolved_indices, dtype=int)
    folded_k = tuple(_fold_k_to_centered_cell(complex(kval), basis_model.lattice) for kval in resolved_kvec)
    canonical_kvec = np.asarray([entry[0] for entry in folded_k], dtype=np.complex128)
    reciprocal_shifts = tuple(entry[1] for entry in folded_k)
    for iflavor, valley in enumerate(resolved_valleys):
        for ik, (k_can, reciprocal_shift) in enumerate(zip(canonical_kvec, reciprocal_shifts, strict=True)):
            evals, evecs = diagonalize_hamiltonian(
                complex(k_can),
                basis_model.lattice,
                basis_model.params,
                valley=int(valley),
            )
            selected_can = np.asarray(evecs[:, index_array], dtype=np.complex128)
            for source_g_index, pair in enumerate(basis_model.lattice.g_indices):
                raw_pair = _raw_pair_from_canonical_pair(
                    pair,
                    reciprocal_shift,
                    valley=int(valley),
                )
                if raw_pair not in positions:
                    raise ValueError(
                        "Periodic-gauge relabel moved a G component outside the embedded reciprocal grid: "
                        f"raw_pair={raw_pair}, shift={reciprocal_shift}, valley={valley}, "
                        f"origin={origin}, grid_shape={grid_shape}. Increase "
                        "RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING."
                    )
                ix, iy = positions[raw_pair]
                start = local_basis_size * source_g_index
                embedded[:, ix, iy, :, iflavor, ik] = selected_can[start : start + local_basis_size, :]
            band_energies[:, iflavor, ik] = np.asarray(evals[index_array], dtype=float)
            if physical_blocks is not None:
                physical_blocks[:, :, iflavor, ik] = _project_physical_hamiltonian(
                    selected_can,
                    k_tilde=complex(k_can),
                    physical_model=physical_model,
                    valley=int(valley),
                )

    wavefunction_array = embedded.reshape(
        (local_basis_size * nx * ny, n_projected, len(resolved_valleys), resolved_kvec.size),
        order="F",
    )
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunction_array,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=local_basis_size,
        name=name,
        component_groups=rlg_hbn_layer_component_groups(basis_model.params.layer_count),
    )

    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    idx = np.arange(basis.nt, dtype=int).reshape((basis.n_spin, basis.n_flavor, n_projected), order="F")
    if physical_blocks is not None:
        for ik in range(basis.nk):
            for ispin in range(basis.n_spin):
                for iflavor in range(basis.n_flavor):
                    block_indices = np.asarray(idx[ispin, iflavor, :], dtype=int)
                    h0[:, :, ik][np.ix_(block_indices, block_indices)] = physical_blocks[:, :, iflavor, ik]

    return RLGhBNProjectedBasisData(
        model=physical_model,
        basis_model=basis_model,
        interaction=interaction,
        screening=screening,
        mesh_size=int(mesh_size),
        kvec=resolved_kvec,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        basis=basis,
        h0=h0,
        band_energies=band_energies,
        active_band_indices=resolved_indices,
        flat_band_indices=basis_model.flat_band_indices,
        valleys=resolved_valleys,
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=origin,
        moire_cell_area_nm2=moire_cell_area_nm2(basis_model),
        physical_h0=h0.copy(),
        fixed_remote_hamiltonian=np.zeros_like(h0),
    )


def _remote_band_indices_and_average_weights(
    basis_model: RLGhBNModel,
    active_band_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], np.ndarray]:
    active = {int(value) for value in active_band_indices}
    valence_count = valence_band_count(basis_model.lattice, basis_model.params)
    remote_indices: list[int] = []
    weights: list[float] = []
    for band_index in range(basis_model.matrix_dim):
        if band_index in active:
            continue
        remote_indices.append(int(band_index))
        weights.append(0.5 if band_index < valence_count else -0.5)
    return tuple(remote_indices), np.asarray(weights, dtype=float)


def _remote_average_density_delta(remote_basis_data: RLGhBNProjectedBasisData, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != remote_basis_data.n_band:
        raise ValueError(
            f"Expected {remote_basis_data.n_band} remote weights, got {weights.size}"
        )
    density = np.zeros((remote_basis_data.nt, remote_basis_data.nt, remote_basis_data.nk), dtype=np.complex128)
    idx = np.arange(remote_basis_data.nt, dtype=int).reshape(
        (remote_basis_data.basis.n_spin, remote_basis_data.basis.n_flavor, remote_basis_data.n_band),
        order="F",
    )
    for ik in range(remote_basis_data.nk):
        for ispin in range(remote_basis_data.basis.n_spin):
            for iflavor in range(remote_basis_data.basis.n_flavor):
                density[idx[ispin, iflavor, :], idx[ispin, iflavor, :], ik] = weights
    return density


def _prepare_remote_average_source(
    source_basis_data: RLGhBNProjectedBasisData,
) -> _RLGhBNRemoteAverageSource | None:
    if source_basis_data.interaction.scheme != "average":
        return None
    remote_indices, remote_weights = _remote_band_indices_and_average_weights(
        source_basis_data.basis_model,
        source_basis_data.active_band_indices,
    )
    if not remote_indices:
        return None

    remote_basis_data = _build_projected_basis_for_indices(
        physical_model=source_basis_data.model,
        basis_model=source_basis_data.basis_model,
        interaction=source_basis_data.interaction,
        kvec=source_basis_data.kvec,
        band_indices=remote_indices,
        valleys=source_basis_data.valleys,
        mesh_size=source_basis_data.mesh_size,
        k_grid_frac=source_basis_data.k_grid_frac,
        screening=None,
        name="rlg_hbn_screened_remote",
        build_h0=False,
    )
    return _RLGhBNRemoteAverageSource(
        basis_data=remote_basis_data,
        weights=np.asarray(remote_weights, dtype=float),
    )


def _remote_average_chunk_size(n_band: int) -> int:
    raw = os.environ.get("MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS must be an integer, got {raw!r}") from exc
    else:
        value = 4
    return max(1, min(int(value), int(n_band)))


def _slice_projected_basis_data_bands(
    basis_data: RLGhBNProjectedBasisData,
    start: int,
    stop: int,
) -> RLGhBNProjectedBasisData:
    start = int(start)
    stop = int(stop)
    if start < 0 or stop <= start or stop > basis_data.n_band:
        raise ValueError(f"Invalid band slice [{start}, {stop}) for n_band={basis_data.n_band}")
    wavefunctions = np.asarray(basis_data.basis.wavefunctions[:, start:stop, :, :], dtype=np.complex128)
    basis = ProjectedWavefunctionBasis(
        wavefunctions=wavefunctions,
        grid_shape=basis_data.basis.grid_shape,
        n_spin=basis_data.basis.n_spin,
        local_basis_size=basis_data.basis.local_basis_size,
        name=f"{basis_data.basis.name}_bands_{start}_{stop}",
        component_groups=basis_data.basis.component_groups,
    )
    h0 = np.zeros((basis.nt, basis.nt, basis.nk), dtype=np.complex128)
    return replace(
        basis_data,
        basis=basis,
        h0=h0,
        band_energies=np.asarray(basis_data.band_energies[start:stop, :, :], dtype=float),
        active_band_indices=tuple(int(value) for value in basis_data.active_band_indices[start:stop]),
        physical_h0=None,
        fixed_remote_hamiltonian=None,
    )


def _hermitize_blocks_inplace(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)


def _resolve_basis_valleys(n_flavor: int, valleys: tuple[int, ...] | None) -> tuple[int, ...]:
    if valleys is None:
        if int(n_flavor) == 1:
            return (1,)
        if int(n_flavor) == 2:
            return VALLEY_SEQUENCE
        return tuple(1 for _ in range(int(n_flavor)))
    resolved = tuple(int(valley) for valley in valleys)
    if len(resolved) != int(n_flavor):
        raise ValueError(f"Expected {n_flavor} valley labels, got {resolved}")
    if any(valley not in VALLEY_SEQUENCE for valley in resolved):
        raise ValueError(f"Expected valley labels in {VALLEY_SEQUENCE}, got {resolved}")
    return resolved


def _rlg_hbn_layer_local_indices(
    basis: ProjectedWavefunctionBasis,
    layer: int,
    *,
    layer_count: int,
) -> np.ndarray:
    layer = int(layer)
    layer_count = int(layer_count)
    if layer < 0 or layer >= layer_count:
        raise ValueError(f"layer must lie in [0, {layer_count}), got {layer}")
    local_basis_size = int(basis.local_basis_size)
    if local_basis_size % layer_count != 0:
        raise ValueError(f"local_basis_size={local_basis_size} is not divisible by layer_count={layer_count}")
    local_per_layer = local_basis_size // layer_count
    return np.arange(layer * local_per_layer, (layer + 1) * local_per_layer, dtype=int)


def _layer_traces_for_diagonal_band_weights(
    basis: ProjectedWavefunctionBasis,
    weights: np.ndarray,
    m: int,
    n: int,
    *,
    layer_count: int,
    valleys: tuple[int, ...] | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != basis.n_band:
        raise ValueError(f"Expected {basis.n_band} band weights, got {weights.size}")
    layer_count = int(layer_count)
    if basis.local_basis_size != 2 * layer_count:
        raise ValueError(
            f"Expected local_basis_size={2 * layer_count} for {layer_count} layers, got {basis.local_basis_size}"
        )
    resolved_valleys = _resolve_basis_valleys(basis.n_flavor, valleys)

    nx, ny = basis.grid_shape
    band_k = basis.n_band * basis.nk
    band_k_weights = np.broadcast_to(weights[:, None], (basis.n_band, basis.nk)).reshape(-1, order="F")
    traces = np.zeros(layer_count, dtype=np.complex128)
    for iflavor, valley in enumerate(resolved_valleys):
        source_grid = basis.wavefunctions[:, :, iflavor, :].reshape(
            basis.local_basis_size,
            nx,
            ny,
            band_k,
            order="F",
        )
        raw_m, raw_n = _raw_overlap_shift_for_physical_g((m, n), valley=int(valley))
        shifted = shift_wavefunction_grid(source_grid, -raw_m, -raw_n, boundary_mode="zero_fill", grid_axes=(1, 2))
        for layer in range(layer_count):
            layer_indices = _rlg_hbn_layer_local_indices(basis, layer, layer_count=layer_count)
            diagonal = np.sum(
                np.conj(source_grid[layer_indices, :, :, :]) * shifted[layer_indices, :, :, :],
                axis=(0, 1, 2),
            )
            traces[layer] += basis.n_spin * np.sum(band_k_weights * np.conj(diagonal))
    return traces


def _remote_average_hamiltonian_from_source(
    target_basis_data: RLGhBNProjectedBasisData,
    source_basis_data: RLGhBNProjectedBasisData,
    remote_source: _RLGhBNRemoteAverageSource | None,
    *,
    shifts: tuple[tuple[int, int], ...] | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    from ._hf_interaction_path import (
        _contract_layer_fock_term,
        build_rlg_hbn_layer_overlap_blocks,
        build_rlg_hbn_layer_overlap_blocks_between,
        interaction_shifts_for_cutoff,
    )

    if remote_source is None:
        return np.zeros_like(target_basis_data.h0)
    if source_basis_data.interaction.scheme != target_basis_data.interaction.scheme:
        raise ValueError(
            "Target/source interaction schemes differ: "
            f"{target_basis_data.interaction.scheme!r} != {source_basis_data.interaction.scheme!r}"
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
    target_blocks = build_rlg_hbn_layer_overlap_blocks(target_basis_data, shifts=resolved_shifts)
    hamiltonian = np.zeros_like(target_basis_data.h0)
    remote_basis_data = remote_source.basis_data
    remote_weights = np.asarray(remote_source.weights, dtype=float).reshape(-1)
    if remote_weights.size != remote_basis_data.n_band:
        raise ValueError(f"Expected {remote_basis_data.n_band} remote weights, got {remote_weights.size}")

    nk_source = int(remote_basis_data.nk)
    scale = float(beta) * float(source_basis_data.v0) / float(nk_source)
    layer_count = int(source_basis_data.basis_model.params.layer_count)
    layer_spacing = float(source_basis_data.basis_model.params.layer_spacing_nm)
    chunk_size = _remote_average_chunk_size(remote_basis_data.n_band)

    for shift, gvec in zip(resolved_shifts, gvecs, strict=True):
        target_layer_diagonal = target_blocks.layer_diagonal_overlaps[shift]
        hartree_kernel = layer_coulomb_matrix_mev_nm2(
            abs(complex(gvec)),
            layer_count,
            source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )
        layer_traces = _layer_traces_for_diagonal_band_weights(
            remote_basis_data.basis,
            remote_weights,
            shift[0],
            shift[1],
            layer_count=layer_count,
            valleys=remote_basis_data.valleys,
        )
        for target_layer in range(layer_count):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hamiltonian += prefactor * target_layer_diagonal[target_layer]

        for start in range(0, remote_basis_data.n_band, chunk_size):
            stop = min(start + chunk_size, remote_basis_data.n_band)
            chunk_basis_data = _slice_projected_basis_data_bands(remote_basis_data, start, stop)
            chunk_density = _remote_average_density_delta(chunk_basis_data, remote_weights[start:stop])
            target_source_blocks = build_rlg_hbn_layer_overlap_blocks_between(
                target_basis_data,
                chunk_basis_data,
                shifts=(shift,),
            )
            target_source_layer_overlap = target_source_blocks.layer_overlaps[shift]
            fock_kernel = _maybe_zero_literal_q0_fock_kernel(shift, target_source_blocks.fock_layer_coulomb[shift])
            for target_layer in range(layer_count):
                for source_layer in range(layer_count):
                    coeff = scale * fock_kernel[:, :, target_layer, source_layer]
                    if np.any(coeff != 0.0):
                        hamiltonian -= _contract_layer_fock_term(
                            target_source_layer_overlap[target_layer],
                            chunk_density,
                            coeff,
                            target_source_layer_overlap[source_layer],
                        )

    _hermitize_blocks_inplace(hamiltonian)
    return hamiltonian


def build_rlg_hbn_remote_average_hamiltonian(
    target_basis_data: RLGhBNProjectedBasisData,
    *,
    source_basis_data: RLGhBNProjectedBasisData | None = None,
    shifts: tuple[tuple[int, int], ...] | None = None,
    beta: float = 1.0,
) -> np.ndarray:
    source_basis = target_basis_data if source_basis_data is None else source_basis_data
    remote_source = _prepare_remote_average_source(source_basis)
    return _remote_average_hamiltonian_from_source(
        target_basis_data,
        source_basis,
        remote_source,
        shifts=shifts,
        beta=beta,
    )


def build_rlg_hbn_projected_basis(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams | None = None,
    *,
    mesh_size: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    valleys: tuple[int, ...] = VALLEY_SEQUENCE,
    screening_mesh_size: int | None = None,
    screening_max_iter: int = 50,
    screening_tolerance_mev: float = 1.0e-6,
    screening_mixing: float = 0.5,
    screening_solver: str = "fixed_point",
    screening_result: ScreenedInterlayerPotentialResult | None = None,
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    screening_root_tolerance_mev: float = 1.0e-5,
) -> RLGhBNProjectedBasisData:
    resolved_interaction = interaction if interaction is not None else RLGhBNInteractionParams()
    resolved_mesh = resolved_interaction.k_mesh_size if mesh_size is None else int(mesh_size)
    if resolved_mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    resolved_valleys = tuple(int(valley) for valley in valleys)
    if not resolved_valleys:
        raise ValueError("At least one valley is required")

    basis_model, screening = _screened_basis_model(
        model,
        resolved_interaction,
        screening_mesh_size=resolved_mesh if screening_mesh_size is None else int(screening_mesh_size),
        screening_max_iter=screening_max_iter,
        screening_tolerance_mev=screening_tolerance_mev,
        screening_mixing=screening_mixing,
        screening_solver=screening_solver,
        screening_result=screening_result,
        screening_u_min_mev=screening_u_min_mev,
        screening_u_max_mev=screening_u_max_mev,
        screening_u_grid_points=screening_u_grid_points,
        screening_root_tolerance_mev=screening_root_tolerance_mev,
    )
    k_grid_frac, kvec_grid = build_moire_k_grid(basis_model.lattice, resolved_mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid.reshape(-1), dtype=np.complex128)
    active_indices = active_band_indices_for_interaction(basis_model, resolved_interaction)
    basis_data = _build_projected_basis_for_indices(
        physical_model=model,
        basis_model=basis_model,
        interaction=resolved_interaction,
        kvec=kvec,
        band_indices=active_indices,
        valleys=resolved_valleys,
        mesh_size=int(resolved_mesh),
        k_grid_frac=np.asarray(k_grid_frac, dtype=float).reshape(-1, 2),
        screening=screening,
        name="rlg_hbn_screened_active",
    )
    fixed_remote = build_rlg_hbn_remote_average_hamiltonian(basis_data)
    completed = replace(
        basis_data,
        h0=np.asarray(basis_data.physical_h0, dtype=np.complex128) + fixed_remote,
        fixed_remote_hamiltonian=fixed_remote,
    )
    _assert_average_remote_hamiltonian_contract(completed)
    return completed


def build_rlg_hbn_projected_basis_for_kvec(
    basis_model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    kvec: np.ndarray,
    *,
    physical_model: RLGhBNModel | None = None,
    active_band_indices: tuple[int, ...] | np.ndarray | None = None,
    valleys: tuple[int, ...] = VALLEY_SEQUENCE,
) -> RLGhBNProjectedBasisData:
    resolved_kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    if resolved_kvec.size == 0:
        raise ValueError("At least one target k point is required")
    resolved_valleys = tuple(int(valley) for valley in valleys)
    if not resolved_valleys:
        raise ValueError("At least one valley is required")

    if active_band_indices is None:
        resolved_active_indices = active_band_indices_for_interaction(basis_model, interaction)
    else:
        resolved_active_indices = tuple(int(value) for value in np.asarray(active_band_indices, dtype=int).reshape(-1))
    if not resolved_active_indices:
        raise ValueError("At least one active band index is required")
    if min(resolved_active_indices) < 0 or max(resolved_active_indices) >= basis_model.matrix_dim:
        raise ValueError(
            f"Active band indices must lie in [0, {basis_model.matrix_dim}), got {resolved_active_indices}"
        )

    resolved_physical_model = basis_model if physical_model is None else physical_model
    return _build_projected_basis_for_indices(
        physical_model=resolved_physical_model,
        basis_model=basis_model,
        interaction=interaction,
        kvec=resolved_kvec,
        band_indices=resolved_active_indices,
        valleys=resolved_valleys,
        mesh_size=0,
        k_grid_frac=np.zeros((resolved_kvec.size, 2), dtype=float),
        screening=None,
        name="rlg_hbn_screened_active_path",
    )

__all__ = [name for name in globals() if not name.startswith('__')]
