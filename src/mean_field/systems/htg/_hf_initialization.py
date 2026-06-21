from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403


def _update_htg_diagnostics_from_density(state: HTGHartreeFockState) -> None:
    state.diagnostics["filling"] = htg_filling_from_density(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )
    state.diagnostics["projector_idempotency_residual"] = projector_idempotency_residual(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


@dataclass(frozen=True)
class HTGInitializer:
    initial_density: np.ndarray | None = None

    def __call__(self, state: HTGHartreeFockState, *, init_mode: str, seed: int) -> None:
        if self.initial_density is not None:
            density = np.asarray(self.initial_density, dtype=np.complex128)
            if density.shape != state.density.shape:
                raise ValueError(f"Expected initial_density shape {state.density.shape}, got {density.shape}")
            state.density[:, :, :] = density
        else:
            state.density[:, :, :] = initialize_htg_density(
                state.h0,
                nu=state.nu,
                init_mode=init_mode,
                seed=seed,
                n_spin=state.n_spin,
                n_eta=state.n_eta,
                n_band=state.n_band,
            )
        _update_htg_diagnostics_from_density(state)


@dataclass(frozen=True)
class HTGDensityBuilder:
    nu: float
    sigma_z: np.ndarray | None = None
    occupation_counts: tuple[int, ...] | None = None
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, sigma_z_expectation, mu, occupation_mask = build_htg_density_from_hamiltonian(
            hamiltonian,
            nu=self.nu,
            sigma_z=self.sigma_z,
            occupation_counts=self.occupation_counts,
            n_spin=self.n_spin,
            n_eta=self.n_eta,
            n_band=self.n_band,
        )
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables={
                "sigma_z": sigma_z_expectation,
                "occupation_mask": occupation_mask,
            },
        )

def normalize_htg_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "noninteracting": "bm",
        "random": "random",
        "diag_random": "diag_random",
        "flavor": "flavor",
        "fb": "fb",
        "d3a": "fb",
        "fb_d3a": "fb",
        "fb_d2a2": "fb",
        "d2a2": "fb",
        "d3b": "sublattice",
        "fb_d3b": "sublattice",
        "fb_d2b2": "sublattice",
        "d2b2": "sublattice",
        "fi": "fi",
        "fi_d3": "fi",
        "d3": "fi",
        "vp": "vp",
        "sp": "sp",
        "chern": "chern",
        "sublattice": "sublattice",
        "perturbed": "perturbed",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported HTG HF init mode: {init_mode}. "
            "Supported modes: bm, random, diag_random, flavor, fb/d3a, fi, vp, sp, chern, "
            "sublattice/d3b, perturbed"
        )
    return aliases[normalized]


def _flavor_priority(flag: str, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flavors = ((0, 0), (0, 1), (1, 0), (1, 1))
    n_band = int(idx.shape[2])
    lower_count = _remote_band_count_per_side(n_band)
    lower_bands = tuple(range(lower_count))
    central_a, central_b = _central_projected_band_indices(n_band)
    upper_bands = tuple(range(central_b + 1, n_band))

    def by_bands(bands: tuple[int, ...], flavor_order=flavors) -> list[int]:
        return [int(idx[ispin, ieta, iband]) for iband in bands for ispin, ieta in flavor_order]

    def by_flavors(bands: tuple[int, ...], flavor_order=flavors) -> list[int]:
        return [int(idx[ispin, ieta, iband]) for ispin, ieta in flavor_order for iband in bands]

    lower_states = by_bands(lower_bands)
    upper_states = by_bands(upper_bands)
    if flag in {"flavor", "fb", "chern"}:
        ordered = lower_states + by_bands((central_a, central_b)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "fi":
        ordered = lower_states + by_flavors((central_a, central_b)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "vp":
        flavor_order = tuple((ispin, ieta) for ieta in range(idx.shape[1]) for ispin in range(idx.shape[0]))
        ordered = by_bands(lower_bands, flavor_order) + by_flavors((central_a, central_b), flavor_order) + by_bands(upper_bands, flavor_order)
        return np.asarray(ordered, dtype=int)
    if flag == "sp":
        flavor_order = tuple((ispin, ieta) for ispin in range(idx.shape[0]) for ieta in range(idx.shape[1]))
        ordered = by_bands(lower_bands, flavor_order) + by_flavors((central_a, central_b), flavor_order) + by_bands(upper_bands, flavor_order)
        return np.asarray(ordered, dtype=int)
    if flag == "sublattice":
        ordered = lower_states + by_bands((central_b, central_a)) + upper_states
        return np.asarray(ordered, dtype=int)
    if flag == "random":
        return rng.permutation(idx.ravel(order="F"))
    raise ValueError(f"Unsupported HTG flavor priority flag: {flag}")


def htg_flavor_occupation_counts_for_init_mode(
    init_mode: str,
    *,
    nu: float,
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[int, ...] | None:
    """Return per-flavor occupation constraints implied by a strong-coupling seed.

    The tuple is flattened in ``(spin, valley)`` C-order and each entry gives
    how many Chern-sublattice bands are occupied in that flavor at every k.
    Stochastic/noninteracting seeds return ``None`` because they are intended
    to explore the unconstrained variational problem.
    """

    normalized = normalize_htg_init_mode(init_mode)
    if normalized in {"bm", "random", "diag_random", "perturbed"}:
        return None

    nt = int(n_spin) * int(n_eta) * int(n_band)
    occupied_per_k = htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta)
    rng = np.random.default_rng(seed)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    order = _flavor_priority(normalized, idx, rng)
    if occupied_per_k > order.size:
        raise ValueError(f"Filling nu={nu} requires {occupied_per_k} states per k, but only {order.size} are available")

    counts = np.zeros((n_spin, n_eta), dtype=int)
    reverse: dict[int, tuple[int, int, int]] = {}
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                reverse[int(idx[ispin, ieta, iband])] = (ispin, ieta, iband)
    for state_index in order[:occupied_per_k]:
        ispin, ieta, _ = reverse[int(state_index)]
        counts[ispin, ieta] += 1
    if np.any(counts < 0) or np.any(counts > n_band):
        raise ValueError(f"Invalid flavor occupation counts for init_mode={init_mode}, nu={nu}: {counts}")
    return tuple(int(value) for value in counts.reshape(-1, order="C"))


def _htg_seed_state_label(state_index: int, idx: np.ndarray) -> str:
    spin_labels = ["up", "down"] + [f"spin_{ispin + 1}" for ispin in range(2, idx.shape[0])]
    valley_labels = ["K", "Kprime"] + [f"eta_{ieta + 1}" for ieta in range(2, idx.shape[1])]
    lower_count = _remote_band_count_per_side(idx.shape[2])
    central_a, central_b = _central_projected_band_indices(idx.shape[2])
    for ispin in range(idx.shape[0]):
        for ieta in range(idx.shape[1]):
            for iband in range(idx.shape[2]):
                if int(idx[ispin, ieta, iband]) != int(state_index):
                    continue
                if iband < lower_count:
                    band_label = f"lower_remote_{iband + 1}"
                elif iband == central_a:
                    band_label = "central_A"
                elif iband == central_b:
                    band_label = "central_B"
                else:
                    band_label = f"upper_remote_{iband - central_b}"
                return f"{valley_labels[ieta]}_{spin_labels[ispin]}:{band_label}"
    raise ValueError(f"state_index={state_index} is not present in HTG seed layout")


def htg_seed_occupation_summary(
    init_mode: str,
    *,
    nu: float,
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> HTGSeedOccupationSummary:
    normalized = normalize_htg_init_mode(init_mode)
    nt = int(n_spin) * int(n_eta) * int(n_band)
    occupied_per_k = htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta)
    occupation_counts = htg_flavor_occupation_counts_for_init_mode(
        init_mode,
        nu=nu,
        seed=seed,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )
    occupation_count_matrix: tuple[tuple[int, ...], ...] | None = None
    initial_state_labels: tuple[str, ...] | None = None
    if occupation_counts is not None:
        counts = np.asarray(occupation_counts, dtype=int).reshape((int(n_spin), int(n_eta)), order="C")
        occupation_count_matrix = tuple(tuple(int(value) for value in row) for row in counts)
        rng = np.random.default_rng(seed)
        idx = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
        order = _flavor_priority(normalized, idx, rng)
        initial_state_labels = tuple(_htg_seed_state_label(int(state_index), idx) for state_index in order[:occupied_per_k])
    return HTGSeedOccupationSummary(
        requested_init_mode=str(init_mode),
        normalized_init_mode=normalized,
        nu=float(nu),
        n_spin=int(n_spin),
        n_eta=int(n_eta),
        n_band=int(n_band),
        reference_band_occupations=tuple(float(value) for value in htg_band_reference_occupations(n_band)),
        central_projected_band_indices=_central_projected_band_indices(n_band),
        occupied_bands_per_k=occupied_per_k,
        occupation_counts=occupation_counts,
        occupation_count_matrix=occupation_count_matrix,
        initial_state_labels=initial_state_labels,
        constrained_flavor_counts=occupation_counts is not None,
    )


def _apply_random_rotation(
    density: np.ndarray,
    *,
    reference_density: np.ndarray,
    alpha: float,
    seed: int,
) -> None:
    apply_random_projector_rotation(
        density,
        reference_density=reference_density,
        alpha=alpha,
        seed=seed,
    )

def initialize_htg_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_htg_init_mode(init_mode)
    h0 = np.asarray(h0, dtype=np.complex128)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    _validate_primitive_cell_integer_filling(nu)

    if init_mode == "bm":
        return build_htg_density_from_hamiltonian(
            h0,
            nu=nu,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )[0]
    if init_mode == "diag_random":
        init_mode = "random"

    rng = np.random.default_rng(seed)
    reference_density = _htg_reference_density_blocks(nt, nk, n_spin=n_spin, n_eta=n_eta)
    density = np.zeros_like(h0)
    total_occupied = htg_occupied_state_count(nu, nt, nk, n_spin=n_spin, n_eta=n_eta)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")

    if init_mode == "random":
        random_energies = rng.standard_normal((nt, nk))
        occ_mask = occupied_state_mask(random_energies, total_occupied)
        for ik in range(nk):
            unitary = random_unitary_from_hermitian(nt, rng)
            occupied = np.flatnonzero(occ_mask[:, ik])
            if occupied.size == 0:
                density[:, :, ik] = -reference_density[:, :, ik]
            else:
                occupied_vecs = unitary[:, occupied]
                density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]
        return density

    flag = "flavor" if init_mode == "perturbed" else init_mode
    order = _flavor_priority(flag, idx, rng)
    full_states = total_occupied // nk
    partial_count = total_occupied % nk
    if full_states > order.size:
        raise ValueError(f"Filling nu={nu} requires {full_states} full states, but only {order.size} are available")
    for state_index in order[:full_states]:
        density[int(state_index), int(state_index), :] = 1.0
    if partial_count:
        state_index = int(order[full_states])
        occupied_k = rng.permutation(nk)[:partial_count]
        density[state_index, state_index, occupied_k] = 1.0
    density -= reference_density

    if init_mode == "perturbed":
        _apply_random_rotation(density, reference_density=reference_density, alpha=0.05, seed=seed)
    return density


def build_htg_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    *,
    nu: float,
    sigma_z: np.ndarray | None = None,
    occupation_counts: tuple[int, ...] | None = None,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    if sigma_z is not None and np.asarray(sigma_z).shape != hamiltonian.shape:
        raise ValueError(f"Expected sigma_z shape {hamiltonian.shape}, got {np.asarray(sigma_z).shape}")
    _validate_primitive_cell_integer_filling(nu)

    energies = np.zeros((nt, nk), dtype=float)
    sigma_z_expectation = np.zeros((nt, nk), dtype=float)
    density = np.zeros_like(hamiltonian)
    reference_density = _htg_reference_density_blocks(nt, nk, n_spin=n_spin, n_eta=n_eta)

    if occupation_counts is not None:
        counts = np.asarray(occupation_counts, dtype=int).reshape(-1)
        if counts.size != int(n_spin) * int(n_eta):
            raise ValueError(
                f"Expected {int(n_spin) * int(n_eta)} flavor occupation counts, got {counts.size}"
            )
        if nt != int(n_spin) * int(n_eta) * int(n_band):
            raise ValueError(
                f"Hamiltonian dimension {nt} is incompatible with "
                f"n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}"
            )
        if np.any(counts < 0) or np.any(counts > int(n_band)):
            raise ValueError(f"Flavor occupation counts must lie in [0, {int(n_band)}], got {counts.tolist()}")
        if int(np.sum(counts)) != htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta):
            raise ValueError(
                f"Flavor occupation counts sum to {int(np.sum(counts))}, "
                f"but nu={nu} requires {htg_occupied_bands_per_k(nu, nt, n_spin=n_spin, n_eta=n_eta)} occupied bands per k"
            )

        idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
        counts_2d = counts.reshape((n_spin, n_eta), order="C")
        occ_mask = np.zeros((nt, nk), dtype=bool)
        for ik in range(nk):
            density[:, :, ik] = -reference_density[:, :, ik]
            for ispin in range(n_spin):
                for ieta in range(n_eta):
                    block_indices = np.asarray(idx[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    reference_block = reference_density[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
                    if sigma_z is not None:
                        sigma_block = sigma_z[:, :, ik][np.ix_(block_indices, block_indices)]
                        sigma_z_expectation[block_indices, ik] = np.real(
                            np.diag(eigvecs.conjugate().T @ sigma_block @ eigvecs)
                        )
                    n_occ = int(counts_2d[ispin, ieta])
                    if n_occ > 0:
                        occupied_vecs = eigvecs[:, :n_occ]
                        density[:, :, ik][np.ix_(block_indices, block_indices)] = (
                            occupied_vecs.conjugate() @ occupied_vecs.T - reference_block
                        )
                        occ_mask[block_indices[:n_occ], ik] = True

        if np.any(occ_mask) and not np.all(occ_mask):
            mu = 0.5 * (float(np.max(energies[occ_mask])) + float(np.min(energies[~occ_mask])))
        else:
            mu = float(np.mean(energies))
        return density, energies, sigma_z_expectation, float(mu), occ_mask

    vecs = np.zeros_like(hamiltonian)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs
        if sigma_z is not None:
            sigma_z_expectation[:, ik] = np.real(np.diag(eigvecs.conjugate().T @ sigma_z[:, :, ik] @ eigvecs))

    total_occupied = htg_occupied_state_count(nu, nt, nk, n_spin=n_spin, n_eta=n_eta)
    occ_mask = occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, float(total_occupied) / float(energies.size))

    for ik in range(nk):
        occupied = np.flatnonzero(occ_mask[:, ik])
        if occupied.size == 0:
            density[:, :, ik] = -reference_density[:, :, ik]
            continue
        occupied_vecs = vecs[:, occupied, ik]
        density[:, :, ik] = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density[:, :, ik]

    return density, energies, sigma_z_expectation, float(mu), occ_mask

__all__ = [name for name in globals() if not name.startswith('__')]
