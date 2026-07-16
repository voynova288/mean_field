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

def _reciprocal_vector_from_fractional_for_gauge(
    lattice: RLGhBNLattice,
    frac: tuple[float, float] | np.ndarray,
) -> complex:
    values = np.asarray(frac, dtype=float).reshape(2)
    return complex(float(values[0]) * lattice.g_m1 + float(values[1]) * lattice.g_m2)

def _wigner_seitz_reciprocal_shift_for_gauge(
    lattice: RLGhBNLattice,
    frac: tuple[float, float] | np.ndarray,
) -> tuple[int, int]:
    values = np.asarray(frac, dtype=float).reshape(2)
    nearest = np.rint(values).astype(int)
    best_key: tuple[float, int, int, int, int] | None = None
    best_shift: tuple[int, int] | None = None
    for dm in range(int(nearest[0]) - 2, int(nearest[0]) + 3):
        for dn in range(int(nearest[1]) - 2, int(nearest[1]) + 3):
            candidate = (float(values[0] - dm), float(values[1] - dn))
            norm = abs(_reciprocal_vector_from_fractional_for_gauge(lattice, candidate))
            key = (
                round(float(norm), 14),
                abs(int(dm)) + abs(int(dn)),
                int(dm) * int(dm) + int(dn) * int(dn),
                int(dm),
                int(dn),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_shift = (int(dm), int(dn))
    if best_shift is None:
        raise RuntimeError("failed to determine Wigner-Seitz reciprocal shift")
    return best_shift

def _c3_reciprocal_shift(shift: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    values = np.asarray(shift, dtype=int).reshape(2)
    return (-int(values[1]), int(values[0]) - int(values[1]))

def _c3_transform_raw_components(
    vector: np.ndarray,
    basis_data: RLGhBNProjectedBasisData,
    *,
    valley: int,
    source_total_shift: tuple[int, int],
    target_total_shift: tuple[int, int],
) -> np.ndarray:
    """Apply the microscopic C3 action between periodic-gauge representatives."""

    basis = basis_data.basis
    nx, ny = basis.grid_shape
    origin = basis_data.reciprocal_grid_origin
    local_size = int(basis.local_basis_size)
    omega = np.exp(2.0j * np.pi / 3.0)
    positions = {
        (int(origin[0]) + ix, int(origin[1]) + iy): (ix, iy)
        for ix in range(nx)
        for iy in range(ny)
    }
    c3_source = _c3_reciprocal_shift(source_total_shift)
    delta = (
        int(target_total_shift[0]) - int(c3_source[0]),
        int(target_total_shift[1]) - int(c3_source[1]),
    )
    valley_sign = int(valley)
    raw_offset = (-valley_sign * delta[0], -valley_sign * delta[1])
    source = np.asarray(vector, dtype=np.complex128).reshape(
        (local_size, nx, ny),
        order="F",
    )
    target = np.zeros_like(source)
    for ix in range(nx):
        for iy in range(ny):
            raw = (int(origin[0]) + ix, int(origin[1]) + iy)
            mapped = _c3_reciprocal_shift(raw)
            target_pair = (mapped[0] + raw_offset[0], mapped[1] + raw_offset[1])
            if target_pair not in positions:
                continue
            tx, ty = positions[target_pair]
            for local_index in range(local_size):
                layer = local_index // 2
                sublattice = local_index % 2
                phase = omega ** (layer + sublattice)
                if valley_sign == -1:
                    phase = np.conj(phase)
                target[local_index, tx, ty] = phase * source[local_index, ix, iy]
    return target.reshape(np.asarray(vector).shape, order="F")


def _c3_mesh_pair_and_representative_shift(
    pair: tuple[int, int] | np.ndarray,
    mesh_size: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    values = np.asarray(pair, dtype=int).reshape(2)
    mesh = int(mesh_size)
    if mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    raw = (-int(values[1]), int(values[0]) - int(values[1]))
    stored = (raw[0] % mesh, raw[1] % mesh)
    representative_shift = ((stored[0] - raw[0]) // mesh, (stored[1] - raw[1]) // mesh)
    return stored, (int(representative_shift[0]), int(representative_shift[1]))

def _pair_to_k_index(pair: tuple[int, int], mesh_size: int) -> int:
    return int(pair[0]) * int(mesh_size) + int(pair[1])

def _k_index_to_pair(index: int, mesh_size: int) -> tuple[int, int]:
    return (int(index) // int(mesh_size), int(index) % int(mesh_size))

def _c3_equivariant_reciprocal_shifts_for_mesh(
    mesh_size: int,
    lattice: RLGhBNLattice,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return a C3-equivariant reciprocal shift gauge for a regular square mesh.

    For stored C3 torus representatives ``k' = C3 k + R``, ordinary C3
    three-cycles can use integer shifts satisfying ``S(k') = C3 S(k) + R``.
    Nonzero C3-fixed torus sectors cannot satisfy that equation with one
    integer representative; they are returned separately so remote Fock can
    average over their three reciprocal representatives.
    """

    mesh = int(mesh_size)
    if mesh <= 0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")
    shifts: dict[tuple[int, int], tuple[int, int]] = {}
    impossible: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for i in range(mesh):
        for j in range(mesh):
            start = (int(i), int(j))
            if start in seen:
                continue
            orbit: list[tuple[int, int]] = []
            representative_shifts: list[tuple[int, int]] = []
            current = start
            while current not in orbit:
                orbit.append(current)
                seen.add(current)
                next_pair, representative_shift = _c3_mesh_pair_and_representative_shift(current, mesh)
                representative_shifts.append(representative_shift)
                current = next_pair
            values: list[tuple[int, int]] = [
                _wigner_seitz_reciprocal_shift_for_gauge(
                    lattice,
                    np.asarray(orbit[0], dtype=float) / float(mesh),
                )
            ]
            for idx in range(len(orbit) - 1):
                c3_value = _c3_reciprocal_shift(values[-1])
                rep = representative_shifts[idx]
                values.append((int(c3_value[0]) + int(rep[0]), int(c3_value[1]) + int(rep[1])))
            closure_c3 = _c3_reciprocal_shift(values[-1])
            closure_rep = representative_shifts[-1]
            closure = (int(closure_c3[0]) + int(closure_rep[0]), int(closure_c3[1]) + int(closure_rep[1]))
            if closure != values[0]:
                impossible.update(orbit)
            for pair_value, shift_value in zip(orbit, values, strict=True):
                shifts[pair_value] = shift_value
    ordered = tuple(shifts[_k_index_to_pair(index, mesh)] for index in range(mesh * mesh))
    return ordered, tuple(sorted(impossible))

def _regular_zero_shift_c3_reciprocal_shifts(
    *,
    mesh_size: int,
    k_grid_frac: np.ndarray,
    lattice: RLGhBNLattice,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    mesh = int(mesh_size)
    if mesh <= 0:
        return None
    frac = np.asarray(k_grid_frac, dtype=float)
    if frac.shape != (mesh * mesh, 2):
        return None
    expected = np.asarray(
        [[i / float(mesh), j / float(mesh)] for i in range(mesh) for j in range(mesh)],
        dtype=float,
    )
    if not np.allclose(frac, expected, atol=1.0e-12, rtol=0.0):
        return None
    return _c3_equivariant_reciprocal_shifts_for_mesh(mesh, lattice)

def _c3_fixed_representative_shift_orbit(
    representative_shift: tuple[int, int],
    *,
    seed: tuple[int, int] = (0, 0),
) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    current = (int(seed[0]), int(seed[1]))
    for _ in range(3):
        if current in values:
            break
        values.append(current)
        c3_current = _c3_reciprocal_shift(current)
        current = (
            int(c3_current[0]) + int(representative_shift[0]),
            int(c3_current[1]) + int(representative_shift[1]),
        )
    return tuple(values)



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
    reciprocal_shifts: tuple[tuple[int, int], ...] | None = None,
    c3_fixed_representative_pairs: tuple[tuple[int, int], ...] = (),
    periodic_gauge_padding: int | None = None,
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
    resolved_padding = (
        int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING)
        if periodic_gauge_padding is None
        else int(periodic_gauge_padding)
    )
    if resolved_padding < 0:
        raise ValueError(f"periodic_gauge_padding must be non-negative, got {periodic_gauge_padding}")
    grid_shape, origin, positions = _rectangular_g_embedding(
        basis_model.lattice,
        padding=resolved_padding,
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
    if reciprocal_shifts is None:
        folded_k = tuple(_fold_k_to_centered_cell(complex(kval), basis_model.lattice) for kval in resolved_kvec)
        canonical_kvec = np.asarray([entry[0] for entry in folded_k], dtype=np.complex128)
        resolved_reciprocal_shifts = tuple(entry[1] for entry in folded_k)
    else:
        resolved_reciprocal_shifts = tuple(
            (int(shift[0]), int(shift[1])) for shift in reciprocal_shifts
        )
        if len(resolved_reciprocal_shifts) != resolved_kvec.size:
            raise ValueError(
                "reciprocal_shifts length must match kvec size: "
                f"{len(resolved_reciprocal_shifts)} != {resolved_kvec.size}"
            )
        canonical_kvec = np.asarray(
            [
                complex(kval - int(shift[0]) * basis_model.lattice.g_m1 - int(shift[1]) * basis_model.lattice.g_m2)
                for kval, shift in zip(resolved_kvec, resolved_reciprocal_shifts, strict=True)
            ],
            dtype=np.complex128,
        )
    for iflavor, valley in enumerate(resolved_valleys):
        for ik, (k_can, reciprocal_shift) in enumerate(zip(canonical_kvec, resolved_reciprocal_shifts, strict=True)):
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
        periodic_reciprocal_shifts=resolved_reciprocal_shifts,
        c3_fixed_representative_pairs=tuple(
            (int(pair[0]), int(pair[1])) for pair in c3_fixed_representative_pairs
        ),
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


def _remote_average_nt_weights(remote_basis_data: RLGhBNProjectedBasisData, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != remote_basis_data.n_band:
        raise ValueError(
            f"Expected {remote_basis_data.n_band} remote weights, got {weights.size}"
        )
    nt_weights = np.zeros(remote_basis_data.nt, dtype=float)
    idx = np.arange(remote_basis_data.nt, dtype=int).reshape(
        (remote_basis_data.basis.n_spin, remote_basis_data.basis.n_flavor, remote_basis_data.n_band),
        order="F",
    )
    for ispin in range(remote_basis_data.basis.n_spin):
        for iflavor in range(remote_basis_data.basis.n_flavor):
            nt_weights[idx[ispin, iflavor, :]] = weights
    return nt_weights

def _remote_average_density_delta(remote_basis_data: RLGhBNProjectedBasisData, weights: np.ndarray) -> np.ndarray:
    nt_weights = _remote_average_nt_weights(remote_basis_data, weights)
    density = np.zeros((remote_basis_data.nt, remote_basis_data.nt, remote_basis_data.nk), dtype=np.complex128)
    diagonal = np.arange(remote_basis_data.nt, dtype=int)
    for ik in range(remote_basis_data.nk):
        density[diagonal, diagonal, ik] = nt_weights
    return density

def _contract_remote_diagonal_fock_term(
    left_overlap: np.ndarray,
    nt_weights: np.ndarray,
    coeff_matrix: np.ndarray,
    right_overlap: np.ndarray,
) -> np.ndarray:
    """Contract a Fock term for a k-independent diagonal remote density.

    Remote-average source density is diagonal in the projected remote-band
    basis with the same weights at every source k.  Using the generic
    ``_contract_layer_fock_term`` would sum over a full source density matrix
    and is unnecessarily expensive for Fig. S45-sized remote windows.
    """

    left = np.asarray(left_overlap, dtype=np.complex128)
    right = np.asarray(right_overlap, dtype=np.complex128)
    coeff = np.asarray(coeff_matrix, dtype=float)
    weights = np.asarray(nt_weights, dtype=float).reshape(-1)
    nt_target, nk_target, nt_source, nk_source = left.shape
    if right.shape != left.shape:
        raise ValueError(f"Expected right_overlap shape {left.shape}, got {right.shape}")
    if coeff.shape != (nk_target, nk_source):
        raise ValueError(f"Expected coeff_matrix shape {(nk_target, nk_source)}, got {coeff.shape}")
    if weights.shape != (nt_source,):
        raise ValueError(f"Expected nt_weights shape {(nt_source,)}, got {weights.shape}")
    return np.einsum(
        "ts,atcs,c,btcs->abt",
        coeff,
        left,
        weights,
        np.conj(right),
        optimize=True,
    )


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
        reciprocal_shifts=source_basis_data.periodic_reciprocal_shifts,
        c3_fixed_representative_pairs=source_basis_data.c3_fixed_representative_pairs,
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

def _slice_projected_basis_data_kpoints(
    basis_data: RLGhBNProjectedBasisData,
    indices: np.ndarray | list[int] | tuple[int, ...],
) -> RLGhBNProjectedBasisData:
    resolved = np.asarray(indices, dtype=int).reshape(-1)
    if resolved.size == 0:
        raise ValueError("At least one k point is required")
    if np.min(resolved) < 0 or np.max(resolved) >= basis_data.nk:
        raise ValueError(f"k-point indices out of range for nk={basis_data.nk}: {resolved.tolist()}")
    basis = ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(basis_data.basis.wavefunctions[:, :, :, resolved], dtype=np.complex128),
        grid_shape=basis_data.basis.grid_shape,
        n_spin=basis_data.basis.n_spin,
        local_basis_size=basis_data.basis.local_basis_size,
        name=f"{basis_data.basis.name}_k_slice",
        boundary_mode=basis_data.basis.boundary_mode,
        component_groups=basis_data.basis.component_groups,
    )
    h0 = np.asarray(basis_data.h0[:, :, resolved], dtype=np.complex128)
    periodic_shifts = (
        None
        if basis_data.periodic_reciprocal_shifts is None
        else tuple(basis_data.periodic_reciprocal_shifts[int(index)] for index in resolved)
    )
    return replace(
        basis_data,
        kvec=np.asarray(basis_data.kvec[resolved], dtype=np.complex128),
        k_grid_frac=np.asarray(basis_data.k_grid_frac[resolved], dtype=float),
        basis=basis,
        h0=h0,
        band_energies=np.asarray(basis_data.band_energies[:, :, resolved], dtype=float),
        physical_h0=None if basis_data.physical_h0 is None else np.asarray(basis_data.physical_h0[:, :, resolved], dtype=np.complex128),
        fixed_remote_hamiltonian=(
            None
            if basis_data.fixed_remote_hamiltonian is None
            else np.asarray(basis_data.fixed_remote_hamiltonian[:, :, resolved], dtype=np.complex128)
        ),
        periodic_reciprocal_shifts=periodic_shifts,
    )

def _build_c3_fixed_remote_representative_source(
    source_basis_data: RLGhBNProjectedBasisData,
    remote_basis_data: RLGhBNProjectedBasisData,
    fixed_pair: tuple[int, int],
) -> RLGhBNProjectedBasisData:
    mesh = int(source_basis_data.mesh_size)
    if mesh <= 0:
        raise ValueError("C3 fixed-sector representative source requires a regular mesh")
    source_index = _pair_to_k_index(fixed_pair, mesh)
    if source_index < 0 or source_index >= source_basis_data.nk:
        raise ValueError(f"fixed_pair={fixed_pair} is outside mesh_size={mesh}")
    c3_pair, representative_shift = _c3_mesh_pair_and_representative_shift(fixed_pair, mesh)
    if c3_pair != fixed_pair:
        raise ValueError(f"fixed_pair={fixed_pair} is not C3-fixed; maps to {c3_pair}")
    representative_orbit = _c3_fixed_representative_shift_orbit(representative_shift)
    return _build_projected_basis_for_indices(
        physical_model=remote_basis_data.model,
        basis_model=remote_basis_data.basis_model,
        interaction=remote_basis_data.interaction,
        kvec=np.repeat(np.asarray(source_basis_data.kvec[source_index], dtype=np.complex128), len(representative_orbit)),
        band_indices=remote_basis_data.active_band_indices,
        valleys=remote_basis_data.valleys,
        mesh_size=source_basis_data.mesh_size,
        k_grid_frac=np.repeat(
            np.asarray(source_basis_data.k_grid_frac[source_index], dtype=float).reshape(1, 2),
            len(representative_orbit),
            axis=0,
        ),
        screening=None,
        name=f"{remote_basis_data.basis.name}_fixed_c3_{fixed_pair[0]}_{fixed_pair[1]}",
        build_h0=False,
        reciprocal_shifts=representative_orbit,
        c3_fixed_representative_pairs=(fixed_pair,),
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
        _fock_overlap_shift_for_physical_transfer,
        build_rlg_hbn_layer_overlap_blocks,
        build_rlg_hbn_layer_overlap_blocks_between,
        fock_transfer_wrap_masks_between,
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

    fixed_pairs = tuple(remote_basis_data.c3_fixed_representative_pairs)
    use_fixed_sector_repair = (
        target_basis_data.periodic_reciprocal_shifts is not None
        and remote_basis_data.periodic_reciprocal_shifts is not None
        and source_basis_data.mesh_size > 0
        and bool(fixed_pairs)
    )
    source_groups: list[tuple[RLGhBNProjectedBasisData, np.ndarray]] = []
    if use_fixed_sector_repair:
        mesh = int(source_basis_data.mesh_size)
        fixed_indices = {_pair_to_k_index(pair, mesh) for pair in fixed_pairs}
        ordinary_indices = [index for index in range(remote_basis_data.nk) if index not in fixed_indices]
        if ordinary_indices:
            source_groups.append((_slice_projected_basis_data_kpoints(remote_basis_data, ordinary_indices), remote_weights))
        for fixed_pair in fixed_pairs:
            source_groups.append(
                (
                    _build_c3_fixed_remote_representative_source(
                        source_basis_data,
                        remote_basis_data,
                        fixed_pair,
                    ),
                    remote_weights / 3.0,
                )
            )
    else:
        source_groups.append((remote_basis_data, remote_weights))

    # Hartree uses the physical shell G directly and the diagonal remote
    # density trace.  The same fixed-sector representative average used for
    # Fock is needed here too once the active/remote bases use the C3 gauge.
    for shift, gvec in zip(resolved_shifts, gvecs, strict=True):
        target_layer_diagonal = target_blocks.layer_diagonal_overlaps[shift]
        hartree_kernel = layer_coulomb_matrix_mev_nm2(
            abs(complex(gvec)),
            layer_count,
            source_basis_data.interaction,
            layer_spacing_nm=layer_spacing,
        )
        layer_traces = np.zeros(layer_count, dtype=np.complex128)
        for source_group, group_weights in source_groups:
            layer_traces += _layer_traces_for_diagonal_band_weights(
                source_group.basis,
                group_weights,
                shift[0],
                shift[1],
                layer_count=layer_count,
                valleys=source_group.valleys,
            )
        for target_layer in range(layer_count):
            prefactor = scale * complex(np.dot(hartree_kernel[target_layer, :], layer_traces))
            if prefactor != 0.0:
                hamiltonian += prefactor * target_layer_diagonal[target_layer]

    # Fock uses the internal transfer k_target-k_source.  A finite G shell is
    # C3-covariant only when this internal transfer is represented in the first
    # mBZ.  For wrap W with delta_ws=delta-W, physical shell G is read from
    # cached overlap key H=G-W so that delta+H = delta_ws+G.
    def accumulate_fock_from_source_group(
        source_group: RLGhBNProjectedBasisData,
        group_weights: np.ndarray,
    ) -> None:
        group_weights = np.asarray(group_weights, dtype=float).reshape(-1)
        if group_weights.size != source_group.n_band:
            raise ValueError(f"Expected {source_group.n_band} remote weights, got {group_weights.size}")
        fock_wrap_masks = fock_transfer_wrap_masks_between(target_basis_data, source_group)
        all_fock_keys = tuple(
            sorted(
                {
                    _fock_overlap_shift_for_physical_transfer(shift, wrap)
                    for shift in resolved_shifts
                    for wrap in fock_wrap_masks
                }
            )
        )
        fock_key_by_shift_wrap = {
            (shift, wrap): _fock_overlap_shift_for_physical_transfer(shift, wrap)
            for shift in resolved_shifts
            for wrap in fock_wrap_masks
        }
        for start in range(0, source_group.n_band, chunk_size):
            stop = min(start + chunk_size, source_group.n_band)
            chunk_basis_data = _slice_projected_basis_data_bands(source_group, start, stop)
            chunk_weights = _remote_average_nt_weights(chunk_basis_data, group_weights[start:stop])
            target_source_blocks = build_rlg_hbn_layer_overlap_blocks_between(
                target_basis_data,
                chunk_basis_data,
                shifts=all_fock_keys,
            )
            for shift in resolved_shifts:
                for wrap, pair_mask in fock_wrap_masks.items():
                    fock_key = fock_key_by_shift_wrap[(shift, wrap)]
                    target_source_layer_overlap = target_source_blocks.layer_overlaps[fock_key]
                    fock_kernel = _maybe_zero_literal_q0_fock_kernel(
                        fock_key,
                        target_source_blocks.fock_layer_coulomb[fock_key],
                    )
                    masked = np.asarray(pair_mask, dtype=float)
                    for target_layer in range(layer_count):
                        for source_layer in range(layer_count):
                            coeff = scale * fock_kernel[:, :, target_layer, source_layer] * masked
                            if np.any(coeff != 0.0):
                                hamiltonian[:] -= _contract_remote_diagonal_fock_term(
                                    target_source_layer_overlap[target_layer],
                                    chunk_weights,
                                    coeff,
                                    target_source_layer_overlap[source_layer],
                                )

    for source_group, group_weights in source_groups:
        accumulate_fock_from_source_group(source_group, group_weights)

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
    flattened_k_grid_frac = np.asarray(k_grid_frac, dtype=float).reshape(-1, 2)
    c3_shift_data = _regular_zero_shift_c3_reciprocal_shifts(
        mesh_size=int(resolved_mesh),
        k_grid_frac=flattened_k_grid_frac,
        lattice=basis_model.lattice,
    )
    if c3_shift_data is None:
        periodic_reciprocal_shifts = None
        c3_fixed_representative_pairs: tuple[tuple[int, int], ...] = ()
    else:
        periodic_reciprocal_shifts, c3_fixed_representative_pairs = c3_shift_data
    basis_data = _build_projected_basis_for_indices(
        physical_model=model,
        basis_model=basis_model,
        interaction=resolved_interaction,
        kvec=kvec,
        band_indices=active_indices,
        valleys=resolved_valleys,
        mesh_size=int(resolved_mesh),
        k_grid_frac=flattened_k_grid_frac,
        screening=screening,
        name="rlg_hbn_screened_active",
        reciprocal_shifts=periodic_reciprocal_shifts,
        c3_fixed_representative_pairs=c3_fixed_representative_pairs,
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
