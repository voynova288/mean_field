from __future__ import annotations

from ._finite_field_shared import *  # noqa: F401,F403
from ._finite_field_types import *  # noqa: F401,F403

def finite_field_diophantine_filling(
    s: int,
    t: int,
    flux: MagneticFlux | Fraction | tuple[int, int] | str,
) -> float:
    """Return the Fig. 6 / Streda-line filling ``nu = s + t*phi/phi0``.

    The paper labels finite-B CHF branches by integer pairs ``(s,t)``. For a
    rational flux ``phi/phi0=p/q``, the density plotted in Fig. 6 is
    ``nu = s + t*p/q`` in units of ``n_s``. Keeping this as a small helper
    avoids hard-coded decimal fillings in replay wrappers.
    """

    return diophantine_filling(s, t, flux)

def screened_coulomb_finite_b(
    qvec: complex,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
) -> float:
    """Dual-gate screened Coulomb kernel used in 2310.15982v3.

    ``V(q)=2π/(epsilon_r |q|) tanh(|q|*4*Lm/2)`` and ``V(0)=0`` in the
    author code because the density operator is background subtracted.
    """

    q_abs = abs(qvec)
    if q_abs < zero_cutoff:
        return 0.0
    return float(2.0 * np.pi / (relative_permittivity * q_abs) * np.tanh(q_abs * 4.0 * float(lm) / 2.0))

def _screened_coulomb_finite_b_array(
    qvec: Array,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
) -> Array:
    """Vectorized finite-B screened Coulomb kernel with the scalar helper's cutoff."""

    q_abs = np.abs(np.asarray(qvec, dtype=np.complex128))
    out = np.zeros(q_abs.shape, dtype=float)
    mask = q_abs >= float(zero_cutoff)
    if np.any(mask):
        q_selected = q_abs[mask]
        out[mask] = 2.0 * np.pi / (float(relative_permittivity) * q_selected) * np.tanh(q_selected * 4.0 * float(lm) / 2.0)
    return out

def coulomb_unit_from_lattice(a1: complex, a2: complex) -> float:
    electron_charge = 1.6e-19
    vacuum_permittivity = 8.8541878128e-12
    graphene_lattice_constant = 2.46e-10
    area_moire = abs((complex(a1).conjugate() * complex(a2)).imag)
    return float(electron_charge / (4.0 * np.pi * vacuum_permittivity * area_moire * graphene_lattice_constant) * 1e3)

def zeeman_unit_from_area(area: float) -> float:
    """Return the meV Zeeman prefactor multiplying ``p/q`` in the Julia code."""

    hbar = 1.054571817e-34
    electron_mass = 9.1093837e-31
    electron_charge = 1.6e-19
    graphene_lattice_constant = 2.46e-10
    return float(2.0 * np.pi * hbar**2 / (2.0 * electron_mass * float(area) * graphene_lattice_constant**2) / electron_charge * 1000.0)

def state_index(subband: int, eta: int, spin: int, *, subbands_per_flavor: int, n_eta: int = 2) -> int:
    """Flatten ``(subband, eta, spin)`` with Julia/Fortran order."""

    return int(subband + subbands_per_flavor * (eta + n_eta * spin))

def build_h0_from_hofstadter_metadata(
    valley_energies: Sequence[Array],
    valley_sigma_z: Sequence[Array] | None,
    *,
    flux: MagneticFlux,
    nq: int,
    zeeman_unit: float = 0.0,
    reduced_translation: bool = False,
) -> tuple[Array, Array]:
    """Build ``H0`` and projected ``Σz`` from Hofstadter metadata arrays.

    Parameters
    ----------
    valley_energies:
        Two arrays, one for K and one for K', each with shape ``(2q,nq,nq)``.
    valley_sigma_z:
        Optional two arrays with shape ``(2q,2q,nq,nq)``.  The valley sign is
        applied as in ``BM_info``: K gets ``+PΣz`` and K' gets ``-PΣz``.
    reduced_translation:
        If false, repeat the Hofstadter data over the ``q`` magnetic strips and
        return ``nk=q*nq**2``.  If true, return the reduced tL-symmetric mesh
        with ``nk=nq**2``.
    """

    q = int(flux.q)
    nq = int(nq)
    n_sub = 2 * q
    n_eta = 2
    n_spin = 2
    nt = n_sub * n_eta * n_spin
    nk = nq * nq if reduced_translation else q * nq * nq
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    sigma = np.zeros_like(h0)
    if len(valley_energies) != 2:
        raise ValueError("Expected two valley energy arrays: K and K'")
    if valley_sigma_z is not None and len(valley_sigma_z) != 2:
        raise ValueError("Expected two valley sigma_z arrays when provided")

    for eta in range(2):
        energies = np.asarray(valley_energies[eta], dtype=float)
        if energies.shape != (n_sub, nq, nq):
            raise ValueError(f"Expected valley energy shape {(n_sub, nq, nq)}, got {energies.shape}")
        sigma_eta = None if valley_sigma_z is None else np.asarray(valley_sigma_z[eta], dtype=np.complex128)
        if sigma_eta is not None and sigma_eta.shape != (n_sub, n_sub, nq, nq):
            raise ValueError(f"Expected valley sigma_z shape {(n_sub, n_sub, nq, nq)}, got {sigma_eta.shape}")
        valley_sign = 1.0 if eta == 0 else -1.0
        for spin in range(2):
            spin_sign = 1.0 if spin == 0 else -1.0
            for i1 in range(nq):
                for i2 in range(nq):
                    reduced_k = i1 + nq * i2
                    strip_iter = (0,) if reduced_translation else range(q)
                    for r in strip_iter:
                        ik = reduced_k if reduced_translation else r + q * reduced_k
                        for a in range(n_sub):
                            ia = state_index(a, eta, spin, subbands_per_flavor=n_sub)
                            h0[ia, ia, ik] = energies[a, i1, i2] + spin_sign * float(zeeman_unit) * flux.ratio
                        if sigma_eta is not None:
                            for a in range(n_sub):
                                ia = state_index(a, eta, spin, subbands_per_flavor=n_sub)
                                for b in range(n_sub):
                                    ib = state_index(b, eta, spin, subbands_per_flavor=n_sub)
                                    sigma[ia, ib, ik] = valley_sign * sigma_eta[a, b, i1, i2]
    return h0, sigma

def finite_field_occupied_state_count(nu: float, nt: int, nk: int) -> int:
    raw = (float(nu) + 4.0) / 8.0 * int(nt) * int(nk)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer occupied count {raw} for nt={nt}, nk={nk}")
    if rounded < 0 or rounded > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied count {rounded} outside [0,{int(nt) * int(nk)}]")
    return rounded

def finite_field_filling(density: Array) -> float:
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    occupied = np.trace(density, axis1=0, axis2=1).real.sum() + 0.5 * nt * nk
    return float(8.0 * occupied / (nt * nk) - 4.0)

def density_update_from_hamiltonian(hamiltonian: Array, *, nu: float, sigma_z: Array | None = None) -> DensityUpdateResult:
    """Diagonalize ``H(k)`` and build the stored projector ``conj(U_occ) U_occ^T - I/2``."""

    h = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {h.shape}")
    vals = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros((nt, nt, nk), dtype=np.complex128)
    sigma_obs = np.zeros((nt, nk), dtype=float)
    for ik in range(nk):
        vals[:, ik], vecs[:, :, ik] = np.linalg.eigh(h[:, :, ik])
        if sigma_z is not None:
            sigma_block = np.asarray(sigma_z, dtype=np.complex128)[:, :, ik]
            for ib in range(nt):
                v = vecs[:, ib, ik]
                sigma_obs[ib, ik] = float(np.vdot(v, sigma_block @ v).real)

    n_occ = finite_field_occupied_state_count(nu, nt, nk)
    occupied = occupied_state_linear_indices(vals, n_occ)
    density = np.zeros_like(h)
    eye = np.eye(nt, dtype=np.complex128)
    bands = occupied % nt
    k_indices = occupied // nt
    for ik in range(nk):
        occ_bands = bands[k_indices == ik]
        if occ_bands.size:
            occ_vecs = vecs[:, occ_bands, ik]
            density[:, :, ik] = occ_vecs.conj() @ occ_vecs.T
        density[:, :, ik] -= 0.5 * eye
    mu = find_chemical_potential(vals, n_occ / float(nt * nk))
    return DensityUpdateResult(density=density, energies=vals, mu=mu, observables={"sigma_ztauz": sigma_obs})

def initialize_density_from_h0(state: FiniteFieldHartreeFockState, *, init_mode: str, seed: int = 0) -> None:
    """Initialize ``state.density`` using a subset of the original Julia modes."""

    mode = normalize_finite_field_init_mode(init_mode)
    rng = np.random.default_rng(seed)
    nt, nk = state.nt, state.nk
    n_occ = finite_field_occupied_state_count(state.nu, nt, nk)
    density = np.zeros_like(state.density)
    if mode == "bm":
        diag = np.diagonal(state.h0, axis1=0, axis2=1).T.real
        occupied = occupied_state_linear_indices(diag, n_occ)
    elif mode == "random":
        occupied = rng.permutation(nt * nk)[:n_occ]
    elif mode == "flavor":
        occupied = _flavor_polarized_indices(state, n_occ)
    elif mode == "bm_cascade":
        occupied = _bm_cascade_indices(state)
    elif mode == "sublattice":
        occupied = _sublattice_indices(state, n_occ)
    else:  # pragma: no cover - normalize_finite_field_init_mode guards this.
        raise ValueError(f"Unsupported init mode {init_mode!r}")
    eye = np.eye(nt, dtype=np.complex128)
    for linear in occupied:
        ib = int(linear % nt)
        ik = int(linear // nt)
        density[ib, ib, ik] = 1.0
    for ik in range(nk):
        density[:, :, ik] -= 0.5 * eye
    if mode == "random":
        _apply_author_random_rotations(density, state, rng)
    state.density[:, :, :] = density
    state.diagnostics["initial_filling"] = finite_field_filling(state.density)

def _random_unitary_from_hermitian(dim: int, rng: np.random.Generator) -> Array:
    """Return a random unitary using the author-code Hermitian-eigenvector recipe."""

    z = rng.random((int(dim), int(dim))) + 1j * rng.random((int(dim), int(dim)))
    hermitian = np.triu(z) + np.triu(z, 1).conj().T
    hermitian[np.diag_indices(int(dim))] = hermitian.diagonal().real
    _vals, vecs = np.linalg.eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)

def _apply_author_random_rotations(density: Array, state: FiniteFieldHartreeFockState, rng: np.random.Generator) -> None:
    """Apply the two random rotations used after Julia ``init_P_random``.

    The original finite-B code does not leave the random seed as a diagonal BM
    occupation.  It first randomizes each valley-spin flavor within the magnetic
    subband space and then mixes valleys within each spin sector.  Keeping the
    seed diagonal traps the SCF loop in the wrong low-gap branch for the Fig. 3
    finite-B problem.
    """

    sub = state.subbands_per_flavor
    for ik in range(state.nk):
        for spin in range(state.n_spin):
            for eta in range(state.n_eta):
                indices = [state_index(band, eta, spin, subbands_per_flavor=sub, n_eta=state.n_eta) for band in range(sub)]
                unitary = _random_unitary_from_hermitian(len(indices), rng)
                block = density[np.ix_(indices, indices, [ik])][:, :, 0]
                density[np.ix_(indices, indices, [ik])] = (unitary.conj().T @ block @ unitary)[:, :, None]
        for spin in range(state.n_spin):
            indices = [
                state_index(band, eta, spin, subbands_per_flavor=sub, n_eta=state.n_eta)
                for eta in range(state.n_eta)
                for band in range(sub)
            ]
            unitary = _random_unitary_from_hermitian(len(indices), rng)
            block = density[np.ix_(indices, indices, [ik])][:, :, 0]
            density[np.ix_(indices, indices, [ik])] = (unitary.conj().T @ block @ unitary)[:, :, None]

def normalize_finite_field_init_mode(init_mode: str) -> InitMode:
    normalized = str(init_mode).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "bm": "bm",
        "random": "random",
        "flavor": "flavor",
        "flavor_u(4)": "flavor",
        "flavor_u4": "flavor",
        "bm_cascade": "bm_cascade",
        "cascade": "bm_cascade",
        "sublattice": "sublattice",
        "chern": "sublattice",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported finite-field init mode. Supported: bm, random, flavor, bm_cascade, sublattice")
    return aliases[normalized]  # type: ignore[return-value]

def _flavor_polarized_indices(state: FiniteFieldHartreeFockState, n_occ: int) -> Array:
    # Fill states flavor by flavor, lowest H0 energies within each chosen flavor.  This is a deterministic
    # Python analogue of the Julia educated flavor-polarized initializer.
    nt, nk = state.nt, state.nk
    diag = np.diagonal(state.h0, axis1=0, axis2=1).T.real
    sub = state.subbands_per_flavor
    flavor_order = [(1, 0), (0, 0), (1, 1), (0, 1)]  # (eta, spin), used by older TBG B0/VH runs too.
    candidates: list[int] = []
    for eta, spin in flavor_order:
        start = state_index(0, eta, spin, subbands_per_flavor=sub)
        flavor_states = [state_index(a, eta, spin, subbands_per_flavor=sub) for a in range(sub)]
        linear = np.asarray([ib + nt * ik for ik in range(nk) for ib in flavor_states], dtype=int)
        order = np.argsort(diag.reshape(-1, order="F")[linear], kind="stable")
        candidates.extend(linear[order].tolist())
        if len(candidates) >= n_occ:
            break
    _ = start  # keep the state_index call above explicit for convention readability.
    return np.asarray(candidates[:n_occ], dtype=int)

def _bm_cascade_indices(state: FiniteFieldHartreeFockState) -> Array:
    # Port of init_P_bm_cascade for the common (s,t) integer-fan seeds.
    pairs = [(0, 4), (0, -4), (1, 3), (-1, -3), (2, 2), (-2, -2), (3, 1), (-3, -1)]
    selected: tuple[int, int] | None = None
    for s, t in pairs:
        if abs(state.nu - (s + t * state.flux.ratio)) < 1e-3:
            selected = (s, t)
            break
    if selected is None:
        raise ValueError(f"nu={state.nu} is not on a supported (s,t) bm_cascade line")
    s, _t = selected
    sub = state.subbands_per_flavor
    nt, nk = state.nt, state.nk
    flavor_order = [2, 3, 0, 1]
    occupied: list[int] = []
    if s <= 1e-3:
        for flavor in flavor_order[: 4 - abs(s)]:
            eta = flavor % 2
            spin = flavor // 2
            for band in range(max(state.q - state.p, 0)):
                ib = state_index(band, eta, spin, subbands_per_flavor=sub)
                occupied.extend(ib + nt * ik for ik in range(nk))
    else:
        occupied.extend(range(nt * nk))
        remove: set[int] = set()
        for flavor in range(4 - abs(s)):
            eta = flavor % 2
            spin = flavor // 2
            for band in range(state.q + state.p, sub):
                ib = state_index(band, eta, spin, subbands_per_flavor=sub)
                remove.update(ib + nt * ik for ik in range(nk))
        occupied = [idx for idx in occupied if idx not in remove]
    return np.asarray(occupied, dtype=int)

def _sublattice_indices(state: FiniteFieldHartreeFockState, n_occ: int) -> Array:
    # A conservative deterministic Chern/sublattice seed: occupy the first n_occ states ordered by subband,
    # flavor, spin, and k.  The original Julia code has several variants; paper validation should choose
    # a physically motivated seed explicitly.
    return np.arange(state.nt * state.nk, dtype=int)[:n_occ]

__all__ = [name for name in globals() if not name.startswith('__')]
