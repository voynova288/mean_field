from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403
from ._tdhf_orbitals import *  # noqa: F401,F403

def _mesh_shape_from_k_grid_frac(k_grid_frac: np.ndarray) -> tuple[int, int]:
    frac = np.asarray(k_grid_frac, dtype=float)
    if frac.ndim != 2 or frac.shape[1] != 2:
        raise ValueError(f"Expected k_grid_frac shape (nk, 2), got {frac.shape}")
    nx = int(np.unique(np.round(frac[:, 0], decimals=12)).size)
    ny = int(np.unique(np.round(frac[:, 1], decimals=12)).size)
    if nx <= 0 or ny <= 0 or nx * ny != frac.shape[0]:
        raise ValueError(f"Cannot infer rectangular mesh from k_grid_frac shape {frac.shape}")
    expected = np.asarray(
        [(ix / nx, iy / ny) for ix in range(nx) for iy in range(ny)],
        dtype=float,
    )
    if not np.allclose(frac, expected, atol=1.0e-10, rtol=0.0):
        raise ValueError("RLG/hBN finite-q TDHF currently requires row-major uniform fractional k_grid_frac")
    return nx, ny


def _shift_k_index_with_wrap(k_index: int, q_shift: tuple[int, int], mesh_shape: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    nx, ny = int(mesh_shape[0]), int(mesh_shape[1])
    index = int(k_index)
    ix = index // ny
    iy = index % ny
    raw_x = ix + int(q_shift[0])
    raw_y = iy + int(q_shift[1])
    target_x = raw_x % nx
    target_y = raw_y % ny
    wrap_x = (raw_x - target_x) // nx
    wrap_y = (raw_y - target_y) // ny
    return int(target_x * ny + target_y), (int(wrap_x), int(wrap_y))


def _add_shift(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (int(left[0]) + int(right[0]), int(left[1]) + int(right[1]))


def _sub_shift(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return (int(left[0]) - int(right[0]), int(left[1]) - int(right[1]))


def build_rlg_hbn_tdhf_q_pairs(
    orbitals: RLGhBNTDHFOrbitals,
    basis_data: RLGhBNProjectedBasisData,
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    require_y_partner: bool = True,
) -> tuple[ParticleHolePair, ...]:
    """Build finite-q X-sector ph pairs ``d†_{k+q,p} d_{k,h}``.

    The returned :class:`ParticleHolePair` stores the X particle momentum
    ``k+q`` and hole momentum ``k``.  For finite-q RPA the Y component in the
    paper convention uses the partner particle at ``k-q``; when
    ``require_y_partner`` is true we keep only pairs for which that partner is
    also an unoccupied HF orbital.  This is automatic for the insulating
    conduction-only Fig. 9/S45 checkpoints but catches accidental metallic or
    nonuniform occupations.
    """

    mesh_shape = _mesh_shape_from_k_grid_frac(basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))
    if basis_data.nk != orbitals.nk:
        raise ValueError(f"basis nk={basis_data.nk} does not match orbital nk={orbitals.nk}")

    pairs: list[ParticleHolePair] = []
    minus_shift = (-shift[0], -shift[1])
    for hole_k in range(orbitals.nk):
        particle_k, _wrap_plus = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        particle_k_minus, _wrap_minus = _shift_k_index_with_wrap(hole_k, minus_shift, mesh_shape)
        occupied = np.flatnonzero(orbitals.occupied_mask[:, hole_k])
        unoccupied_plus = np.flatnonzero(~orbitals.occupied_mask[:, particle_k])
        if require_y_partner:
            unoccupied_minus = set(int(value) for value in np.flatnonzero(~orbitals.occupied_mask[:, particle_k_minus]))
            unoccupied = [int(value) for value in unoccupied_plus if int(value) in unoccupied_minus]
        else:
            unoccupied = [int(value) for value in unoccupied_plus]
        for hole in occupied:
            for particle in unoccupied:
                pairs.append(
                    ParticleHolePair(
                        particle=orbitals.global_index(int(particle), particle_k),
                        hole=orbitals.global_index(int(hole), hole_k),
                        particle_momentum=particle_k,
                        hole_momentum=hole_k,
                        particle_flavor=orbitals.flavor_tag(int(particle)),
                        hole_flavor=orbitals.flavor_tag(int(hole)),
                    )
                )
    return tuple(pairs)


def required_rlg_hbn_tdhf_finite_q_overlap_shifts(
    orbitals: RLGhBNTDHFOrbitals,
    basis_data: RLGhBNProjectedBasisData,
    pairs: Sequence[ParticleHolePair],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    physical_shifts: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return all cached overlap-shift keys needed for finite-q exchange.

    ``physical_shifts`` are the paper Umklapp vectors G included in the
    Coulomb sum.  If a particle leg ``k+q`` wraps back into the stored mBZ,
    the stored form-factor key is not necessarily G but
    ``G + W_target - W_source``.  This helper computes the closure needed by
    the finite-q flavor-flip shortcut without changing the physical G cutoff.
    """

    mesh_shape = _mesh_shape_from_k_grid_frac(basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))

    ph_pairs = tuple(pairs)
    hole_k_values: list[int] = []
    wrap_by_hole_k: dict[int, tuple[int, int]] = {}
    for pair in ph_pairs:
        _p_local, particle_k = orbitals.decode_global_index(pair.particle)
        _h_local, hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, wrap = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if int(particle_k) != int(expected_particle_k):
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        if int(hole_k) not in wrap_by_hole_k:
            hole_k_values.append(int(hole_k))
            wrap_by_hole_k[int(hole_k)] = tuple(int(v) for v in wrap)

    required: set[tuple[int, int]] = set()
    resolved_physical_shifts = tuple((int(g[0]), int(g[1])) for g in physical_shifts)
    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        required.add(g0)
        for target_k in hole_k_values:
            wrap_t = wrap_by_hole_k[int(target_k)]
            for source_k in hole_k_values:
                wrap_s = wrap_by_hole_k[int(source_k)]
                required.add(_add_shift(g0, _sub_shift(wrap_t, wrap_s)))
    return tuple(sorted(required))


def required_rlg_hbn_tdhf_full_finite_q_overlap_shifts(
    orbitals: RLGhBNTDHFOrbitals,
    basis_data: RLGhBNProjectedBasisData,
    pairs: Sequence[ParticleHolePair],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    physical_shifts: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return cached overlap keys needed for full finite-q Eq. D19 TDHF.

    The X-sector pair is ``d†_{k+q,p} d_{k,h}``, while the Y-sector partner
    uses ``d†_{k,h} d_{k-q,p}``.  Therefore wrapped form-factor keys must cover
    both the particle leg at ``k+q`` and the particle leg at ``k-q``.  The
    physical Coulomb sum still runs only over ``physical_shifts``; extra keys
    are cache-closure labels, not extra Umklapp vectors.
    """

    mesh_shape = _mesh_shape_from_k_grid_frac(basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))

    wrap_plus_by_hole_k: dict[int, tuple[int, int]] = {}
    wrap_minus_by_hole_k: dict[int, tuple[int, int]] = {}
    hole_k_values: list[int] = []
    minus_shift = (-shift[0], -shift[1])
    for pair in tuple(pairs):
        _p_local, particle_k = orbitals.decode_global_index(pair.particle)
        _h_local, hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, wrap_plus = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if int(particle_k) != int(expected_particle_k):
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        _minus_k, wrap_minus = _shift_k_index_with_wrap(hole_k, minus_shift, mesh_shape)
        if int(hole_k) not in wrap_plus_by_hole_k:
            hole_k_values.append(int(hole_k))
            wrap_plus_by_hole_k[int(hole_k)] = tuple(int(v) for v in wrap_plus)
            wrap_minus_by_hole_k[int(hole_k)] = tuple(int(v) for v in wrap_minus)

    required: set[tuple[int, int]] = set()
    resolved_physical_shifts = tuple((int(g[0]), int(g[1])) for g in physical_shifts)
    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        required.add(g0)
        for target_k in hole_k_values:
            wrap_plus_t = wrap_plus_by_hole_k[int(target_k)]
            required.add(_add_shift(g0, wrap_plus_t))
            required.add(_sub_shift(g0, wrap_minus_by_hole_k[int(target_k)]))
            for source_k in hole_k_values:
                wrap_plus_s = wrap_plus_by_hole_k[int(source_k)]
                required.add(_add_shift(g0, _sub_shift(wrap_plus_t, wrap_plus_s)))
    return tuple(sorted(required))

def build_rlg_hbn_tdhf_q0_pairs(
    orbitals: RLGhBNTDHFOrbitals,
) -> tuple[ParticleHolePair, ...]:
    """Build q=0 ph pairs: particle and hole have the same mBZ k index."""

    pairs: list[ParticleHolePair] = []
    for ik in range(orbitals.nk):
        occupied = np.flatnonzero(orbitals.occupied_mask[:, ik])
        unoccupied = np.flatnonzero(~orbitals.occupied_mask[:, ik])
        for hole in occupied:
            for particle in unoccupied:
                pairs.append(
                    ParticleHolePair(
                        particle=orbitals.global_index(int(particle), ik),
                        hole=orbitals.global_index(int(hole), ik),
                        particle_momentum=ik,
                        hole_momentum=ik,
                        particle_flavor=orbitals.flavor_tag(int(particle)),
                        hole_flavor=orbitals.flavor_tag(int(hole)),
                    )
                )
    return tuple(pairs)


def build_rlg_hbn_tdhf_interaction(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals | None = None,
    *,
    beta: float = 1.0,
    momentum_policy: MomentumPolicy = "strict",
) -> RLGhBNTDHFInteraction:
    """Create the callable ``V_hf(a,b,c,d)`` for a converged RLG/hBN HF run."""

    resolved_orbitals = build_rlg_hbn_tdhf_orbitals(run.state) if orbitals is None else orbitals
    return RLGhBNTDHFInteraction(
        basis_data=run.basis_data,
        overlap_blocks=run.overlap_blocks,
        orbitals=resolved_orbitals,
        beta=beta,
        momentum_policy=momentum_policy,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
