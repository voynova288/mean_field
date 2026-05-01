from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from ....core.hf import (
    DensityUpdateResult,
    FlavorBandData,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    block_mask,
    build_projected_hf_kernel,
    build_projected_interaction_hamiltonian,
    build_flavor_band_data,
    calculate_norm_convergence,
    compute_density_overlap_trace_from_diagonal,
    compute_hf_energy,
    compute_oda_parameter,
    contract_fock_term_from_overlap,
    empty_overlap_block_set,
    find_chemical_potential,
    flavor_block_indices,
    flavor_sector_metadata,
    identity_block,
    occupied_state_linear_indices as _occupied_state_linear_indices,
    occupied_state_mask as _occupied_state_mask,
    project_to_flavor_diagonal,
    project_to_flavor_diagonal_inplace,
    run_hartree_fock_problem,
)
from .model import BMSolution
from ..params import TBGParameters


_compute_density_overlap_trace_from_diagonal = compute_density_overlap_trace_from_diagonal


@dataclass(frozen=True)
class RestrictedHartreeFockRun(HartreeFockRun):
    state: "RestrictedHartreeFockState"
    overlap_blocks: HFOverlapBlockSet


@dataclass
class RestrictedHartreeFockState:
    h0: np.ndarray
    sigma_z: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_ztauz: np.ndarray
    nu: float
    v0: float
    mu: float = float("nan")
    precision: float = 1e-5
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_bm_solution(
        cls,
        solution: BMSolution,
        *,
        nu: float,
        precision: float = 1e-5,
    ) -> "RestrictedHartreeFockState":
        h0 = build_h0_from_bm(solution)
        nt, nk = h0.shape[0], h0.shape[2]
        return cls(
            h0=h0,
            sigma_z=np.asarray(solution.sigma_z, dtype=np.complex128).copy(),
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_ztauz=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            v0=coulomb_unit(solution.params),
            precision=float(precision),
            n_spin=int(solution.n_spin),
            n_eta=int(solution.n_eta),
            n_band=int(solution.nb),
        )


def coulomb_unit(params: TBGParameters) -> float:
    electron_charge = 1.6e-19
    vacuum_permittivity = 8.8541878128e-12
    graphene_lattice_constant = 2.46e-10
    area_moire = abs((params.a1.conjugate() * params.a2).imag)
    return float(electron_charge / (4.0 * np.pi * vacuum_permittivity * area_moire * graphene_lattice_constant) * 1e3)


def screened_coulomb(
    q: complex,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
    finite_zero_limit: bool = False,
) -> float:
    q_abs = abs(q)
    if q_abs < zero_cutoff:
        return float(2.0 * np.pi * 2.0 * lm / relative_permittivity) if finite_zero_limit else 0.0
    return float(2.0 * np.pi / (relative_permittivity * q_abs) * np.tanh(q_abs * 4.0 * lm / 2.0))


def build_h0_from_bm(solution: BMSolution) -> np.ndarray:
    nt = solution.nt
    nk = solution.nk
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    flattened = solution.flattened_energies()
    for ik in range(nk):
        np.fill_diagonal(h0[:, :, ik], flattened[:, ik])
    return h0


def reciprocal_shift_labels(lg: int) -> tuple[int, ...]:
    if lg <= 0 or lg % 2 == 0:
        raise ValueError(f"Expected a positive odd lg, got {lg}")
    half_width = (lg - 1) // 2
    return tuple(range(-half_width, half_width + 1))


def build_overlap_block_set(
    target_solution: BMSolution,
    source_solution: BMSolution | None = None,
    *,
    lg: int | None = None,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> HFOverlapBlockSet:
    from .overlap import calculate_overlap_between

    source_solution = target_solution if source_solution is None else source_solution
    lG = target_solution.lg if lg is None else int(lg)
    labels = reciprocal_shift_labels(lG)
    shifts = tuple((m, n) for n in labels for m in labels)
    gvecs = np.asarray([m * target_solution.params.g1 + n * target_solution.params.g2 for m, n in shifts], dtype=np.complex128)
    overlaps = {shift: calculate_overlap_between(target_solution, source_solution, shift[0], shift[1]) for shift in shifts}
    diagonal_overlaps, hartree_screening, fock_screening = _precompute_overlap_screening(
        shifts,
        gvecs,
        overlaps,
        params=target_solution.params,
        target_kvec=np.asarray(target_solution.lattice_kvec, dtype=np.complex128),
        source_kvec=np.asarray(source_solution.lattice_kvec, dtype=np.complex128),
        relative_permittivity=relative_permittivity,
        screening_lm=screening_lm,
        finite_zero_limit=finite_zero_limit,
        zero_cutoff=zero_cutoff,
    )
    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def normalize_restricted_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "random": "random",
        "educated": "educated",
        "vp": "vp",
        "kspinpair": "kspinpair",
        "spindown": "spindown",
        "downpair": "downpair",
        # These two names appear in the packaged B0 benchmark manifest.
        "sp": "spindown",
        "chern": "vp",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported restricted init mode: {init_mode}. "
            "Supported modes: bm, random, educated, vp, kspinpair, spindown, downpair, sp, chern"
        )
    return aliases[normalized]


def canonical_fig6_flavor_sequence(init_mode: str) -> tuple[tuple[int, int], ...]:
    init_mode = normalize_restricted_init_mode(init_mode)
    if init_mode in ("educated", "vp", "kspinpair"):
        return ((1, 0), (0, 0), (1, 1), (0, 1))
    if init_mode in ("spindown", "downpair"):
        return ((1, 0), (1, 1), (0, 0), (0, 1))
    raise ValueError(f"Unsupported canonical restricted init mode: {init_mode}")


def is_canonical_restricted_init(init_mode: str) -> bool:
    try:
        normalized = normalize_restricted_init_mode(init_mode)
    except ValueError:
        return False
    return normalized in ("educated", "vp", "kspinpair", "spindown", "downpair")


def restricted_occupied_state_count(nu: float, nt: int, nk: int) -> int:
    raw = (nu + 4.0) / 8.0 * nt * nk
    rounded = int(round(float(raw)))
    if abs(float(raw) - rounded) > 1e-9:
        raise ValueError(
            f"Filling nu={nu} gives non-integer occupied-state count {raw} "
            f"for nt={nt}, nk={nk}."
        )
    if rounded < 0 or rounded > nt * nk:
        raise ValueError(f"Filling nu={nu} gives occupied-state count {rounded} outside [0, {nt * nk}].")
    return rounded


def restricted_occupied_bands_per_k(nu: float, nt: int) -> int:
    raw = (nu + 4.0) / 8.0 * nt
    rounded = int(round(float(raw)))
    if abs(float(raw) - rounded) > 1e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer per-k occupation {raw} for nt={nt}.")
    if rounded < 0 or rounded > nt:
        raise ValueError(f"Filling nu={nu} gives per-k occupation {rounded} outside [0, {nt}].")
    return rounded


def restricted_filling(density: np.ndarray) -> float:
    nt = density.shape[0]
    nk = density.shape[2]
    total = float(np.trace(density, axis1=0, axis2=1).real.sum() + 0.5 * nt * nk)
    return float(8.0 * total / (nk * nt) - 4.0)


def _screened_coulomb_matrix(
    qvals: np.ndarray,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
    finite_zero_limit: bool = False,
) -> np.ndarray:
    q_abs = np.abs(np.asarray(qvals, dtype=np.complex128))
    values = np.zeros_like(q_abs, dtype=float)
    if finite_zero_limit:
        values[q_abs < zero_cutoff] = 2.0 * np.pi * 2.0 * lm / relative_permittivity
    mask = q_abs >= zero_cutoff
    if np.any(mask):
        values[mask] = 2.0 * np.pi / (relative_permittivity * q_abs[mask]) * np.tanh(q_abs[mask] * 2.0 * lm)
    return values


def _hex_shell_contains(params: TBGParameters, gvec: complex) -> bool:
    g0 = abs(3.0 * params.g1 + 3.0 * params.g2) * 1.00001
    angle_mod = np.mod(np.angle(gvec), np.pi / 3.0) - np.pi / 6.0
    denominator = abs(np.cos(angle_mod))
    if denominator < 1e-15:
        return False
    shell_radius = g0 * np.cos(np.pi / 6.0) / denominator
    return abs(gvec) < shell_radius


def _precompute_overlap_screening(
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    overlaps: dict[tuple[int, int], np.ndarray],
    *,
    params: TBGParameters,
    target_kvec: np.ndarray,
    source_kvec: np.ndarray,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float], dict[tuple[int, int], np.ndarray]]:
    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)) if screening_lm is None else screening_lm)
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if not _hex_shell_contains(params, complex(gvec)):
            continue
        overlap = overlaps[shift]
        diagonal_overlaps[shift] = np.diagonal(overlap, axis1=1, axis2=3)
        hartree_screening[shift] = screened_coulomb(
            complex(gvec),
            lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
        fock_screening[shift] = _screened_coulomb_matrix(
            source_kvec[None, :] - target_kvec[:, None] + complex(gvec),
            lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
    return diagonal_overlaps, hartree_screening, fock_screening


def _with_tbg_overlap_screening(
    overlap_blocks: HFOverlapBlockSet,
    *,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> HFOverlapBlockSet:
    diagonal_overlaps = dict(overlap_blocks.diagonal_overlaps)
    hartree_screening = dict(overlap_blocks.hartree_screening)
    fock_screening = dict(overlap_blocks.fock_screening)
    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)) if screening_lm is None else screening_lm)
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        if not _hex_shell_contains(params, complex(gvec)):
            continue
        overlap = overlap_blocks.overlaps[shift]
        diagonal_overlaps.setdefault(shift, np.diagonal(overlap, axis1=1, axis2=3))
        hartree_screening.setdefault(
            shift,
            screened_coulomb(
                complex(gvec),
                lm,
                relative_permittivity=relative_permittivity,
                zero_cutoff=zero_cutoff,
                finite_zero_limit=finite_zero_limit,
            ),
        )
        fock_screening.setdefault(
            shift,
            _screened_coulomb_matrix(
                lattice_kvec[None, :] - lattice_kvec[:, None] + complex(gvec),
                lm,
                relative_permittivity=relative_permittivity,
                zero_cutoff=zero_cutoff,
                finite_zero_limit=finite_zero_limit,
            ),
        )
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_interaction_hamiltonian(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    v0: float,
    *,
    beta: float = 1.0,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
) -> np.ndarray:
    lattice_kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    if lattice_kvec.size != density.shape[2]:
        raise ValueError(f"Expected {density.shape[2]} k-points, got {lattice_kvec.size}")
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=lattice_kvec,
        params=params,
        relative_permittivity=relative_permittivity,
        screening_lm=screening_lm,
        finite_zero_limit=finite_zero_limit,
        zero_cutoff=zero_cutoff,
    )
    return build_projected_interaction_hamiltonian(
        density,
        screened_overlap_blocks,
        v0=v0,
        beta=beta,
    )


def oda_parametrization_restricted(
    state: RestrictedHartreeFockState,
    delta_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> float:
    return compute_oda_parameter(
        state,
        delta_density,
        interaction_builder=lambda density: build_interaction_hamiltonian(
            density,
            overlap_blocks,
            lattice_kvec,
            params,
            state.v0,
            beta=beta,
        ),
    )


def _restricted_density_update_result(state: RestrictedHartreeFockState, hamiltonian: np.ndarray) -> DensityUpdateResult:
    density, energies, sigma_ztauz, mu = build_restricted_density_from_hamiltonian(
        hamiltonian,
        state.sigma_z,
        nu=state.nu,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=mu,
        observables={"sigma_ztauz": sigma_ztauz},
    )


def _full_density_update_result(state: RestrictedHartreeFockState, hamiltonian: np.ndarray) -> DensityUpdateResult:
    density, energies, sigma_ztauz, mu = build_full_density_from_hamiltonian(
        hamiltonian,
        state.sigma_z,
        nu=state.nu,
    )
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=mu,
        observables={"sigma_ztauz": sigma_ztauz},
    )


def _update_tbg_hf_density_update_state(state: RestrictedHartreeFockState, density_update: DensityUpdateResult) -> None:
    sigma_ztauz = np.asarray(density_update.observables["sigma_ztauz"], dtype=float)
    state.sigma_ztauz[:, :] = sigma_ztauz
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)
    state.diagnostics["occupied_sigma_mean"] = occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu)


def _update_tbg_hf_step_state(state: RestrictedHartreeFockState, step) -> None:
    _update_tbg_hf_density_update_state(state, step.density_update)


def _flavor_diagonal_projector(state: RestrictedHartreeFockState):
    return lambda matrix: project_to_flavor_diagonal_inplace(
        matrix,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )


def build_restricted_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockKernel:
    flavor_projector = _flavor_diagonal_projector(state)
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
    )
    return build_projected_hf_kernel(
        state,
        screened_overlap_blocks,
        density_builder=lambda hamiltonian: _restricted_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: oda_parametrization_restricted(
            state_obj,
            delta_density,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
        hamiltonian_postprocessor=flavor_projector,
        density_postprocessor=flavor_projector,
        step_callback=_update_tbg_hf_step_state,
        final_state_callback=_update_tbg_hf_density_update_state,
        convergence_rule="raw",
        v0=state.v0,
        beta=beta,
    )


def build_restricted_hf_problem(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_restricted_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
        ),
        kernel=build_restricted_hf_kernel(
            state,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
    )


def initialize_restricted_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "educated",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_restricted_init_mode(init_mode)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    density = np.zeros_like(h0)
    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    full_id = identity_block(nt)
    sectors = flavor_block_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band)

    if init_mode == "bm":
        energies = np.zeros((nt, nk), dtype=float)
        for ik in range(nk):
            energies[:, ik] = np.diag(h0[:, :, ik]).real
        occupied = np.argsort(energies.ravel(order="F"))[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")
        for ik in range(nk):
            block = density[:, :, ik]
            block[np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
            block -= 0.5 * full_id
    elif is_canonical_restricted_init(init_mode):
        occupied_per_k = restricted_occupied_bands_per_k(nu, nt)
        if occupied_per_k < 0 or occupied_per_k > n_spin * n_eta:
            raise ValueError(f"Canonical restricted init only supports 0 <= occupied_per_k <= {n_spin * n_eta}, got {occupied_per_k}")
        for ispin, ieta in canonical_fig6_flavor_sequence(init_mode)[:occupied_per_k]:
            lower_band = int(idx[ispin, ieta, 0])
            density[lower_band, lower_band, :] = 1.0
        for ik in range(nk):
            density[:, :, ik] -= 0.5 * full_id
    elif init_mode == "random":
        rng = np.random.default_rng(seed)
        evals = np.zeros((nt, nk), dtype=float)
        vecs = np.zeros_like(h0)
        for ik in range(nk):
            vecs_k = vecs[:, :, ik]
            for inds in sectors:
                block_inds = np.asarray(inds, dtype=int)
                block_h = rng.standard_normal((block_inds.size, block_inds.size)) + 1j * rng.standard_normal((block_inds.size, block_inds.size))
                block_h = block_h + block_h.conj().T
                eigvals, eigvecs = eigh(block_h)
                evals[block_inds, ik] = eigvals
                vecs_k[np.ix_(block_inds, block_inds)] = eigvecs

        occupied = np.argsort(evals.ravel(order="F"))[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")

        for ik in range(nk):
            block_density = density[:, :, ik]
            vecs_k = vecs[:, :, ik]
            for inds in sectors:
                block_inds = np.asarray(inds, dtype=int)
                block_id = identity_block(block_inds.size)
                occ_local = np.flatnonzero(occ_mask[block_inds, ik])
                if occ_local.size == 0:
                    block_density[np.ix_(block_inds, block_inds)] = -0.5 * block_id
                    continue
                occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
                block_density[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id
    else:
        raise ValueError(f"Unsupported restricted init mode after normalization: {init_mode}")

    project_to_flavor_diagonal_inplace(density, sectors=sectors)
    return density


def initialize_restricted_state(
    state: RestrictedHartreeFockState,
    *,
    init_mode: str = "educated",
    seed: int = 1,
) -> float:
    state.density[:, :, :] = initialize_restricted_density(
        state.h0,
        nu=state.nu,
        init_mode=init_mode,
        seed=seed,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    filling = restricted_filling(state.density)
    state.diagnostics["filling"] = filling
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    return filling


def build_restricted_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
    *,
    nu: float,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nt, _, nk = hamiltonian.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(
            f"Hamiltonian dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}"
        )

    sectors = flavor_block_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    energies = np.zeros((nt, nk), dtype=float)
    sigma_ztauz = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros_like(hamiltonian)

    for ik in range(nk):
        vecs_k = vecs[:, :, ik]
        h_k = hamiltonian[:, :, ik]
        sigma_k = sigma_z[:, :, ik]
        for inds in sectors:
            block_inds = np.asarray(inds, dtype=int)
            block_h = h_k[np.ix_(block_inds, block_inds)]
            block_sigma = sigma_k[np.ix_(block_inds, block_inds)]
            eigvals, eigvecs = eigh(block_h)
            energies[block_inds, ik] = eigvals
            vecs_k[np.ix_(block_inds, block_inds)] = eigvecs
            sigma_ztauz[block_inds, ik] = np.real(np.diag(eigvecs.conj().T @ block_sigma @ eigvecs))

    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    occ_mask = _occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, (nu + 4.0) / 8.0)

    density = np.zeros_like(hamiltonian)
    for ik in range(nk):
        block_density = density[:, :, ik]
        vecs_k = vecs[:, :, ik]
        for inds in sectors:
            block_inds = np.asarray(inds, dtype=int)
            block_id = identity_block(block_inds.size)
            occ_local = np.flatnonzero(occ_mask[block_inds, ik])
            if occ_local.size == 0:
                block_density[np.ix_(block_inds, block_inds)] = -0.5 * block_id
                continue
            occupied_vecs = vecs_k[np.ix_(block_inds, block_inds)][:, occ_local]
            block_density[np.ix_(block_inds, block_inds)] = occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id

    project_to_flavor_diagonal_inplace(density, sectors=sectors)
    return density, energies, sigma_ztauz, mu


def update_restricted_density(
    state: RestrictedHartreeFockState,
    *,
    mixing_parameter: float = 1.0,
) -> tuple[float, float]:
    if mixing_parameter < 0.0 or mixing_parameter > 1.0:
        raise ValueError(f"mixing_parameter must lie in [0, 1], got {mixing_parameter}")

    old_density = state.density.copy()
    density_new, energies, sigma_ztauz, mu = build_restricted_density_from_hamiltonian(
        state.hamiltonian,
        state.sigma_z,
        nu=state.nu,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    mixed_density = mixing_parameter * density_new + (1.0 - mixing_parameter) * old_density
    norm_convergence = calculate_norm_convergence(mixed_density, old_density)

    state.density[:, :, :] = mixed_density
    project_to_flavor_diagonal_inplace(
        state.density,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
    )
    state.energies[:, :] = energies
    state.sigma_ztauz[:, :] = sigma_ztauz
    state.mu = float(mu)
    state.diagnostics["filling"] = restricted_filling(state.density)
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    state.diagnostics["restricted_gap"] = restricted_gap_estimate(state.energies, state.nu)
    state.diagnostics["occupied_sigma_mean"] = occupied_sigma_mean(state.energies, state.sigma_ztauz, state.nu)
    return norm_convergence, float(mixing_parameter)


def run_restricted_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    init_mode: str = "educated",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
) -> RestrictedHartreeFockRun:
    normalized_init_mode = normalize_restricted_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    problem = build_restricted_hf_problem(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        beta=beta,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=normalized_init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def run_restricted_hf_from_bm_solution(
    solution: BMSolution,
    *,
    nu: float,
    init_mode: str = "educated",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    overlap_lg: int | None = None,
    precision: float = 1e-5,
    oda_stall_threshold: float = 1e-3,
) -> RestrictedHartreeFockRun:
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=nu, precision=precision)
    resolved_overlap_lg = solution.lg if overlap_lg is None else int(overlap_lg)
    state.diagnostics["overlap_lg"] = float(resolved_overlap_lg)
    overlap_blocks = build_overlap_block_set(solution, lg=resolved_overlap_lg)
    return run_restricted_hartree_fock(
        state,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        seed=seed,
        beta=beta,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )


def normalize_full_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    supported = {
        "random",
        "diag_random",
        "educated",
        "tivc",
        "kivc",
        "bm",
        "vp",
        "sp",
        "chern",
        "flavor",
        "sublattice",
    }
    if normalized not in supported:
        raise ValueError(
            f"Unsupported full HF init mode: {init_mode}. "
            "Supported modes: random, diag_random, educated, tivc, kivc, bm, vp, sp, chern, flavor"
        )
    return normalized


def canonical_fig6_state_sequence(*, n_spin: int = 2, n_eta: int = 2, n_band: int = 2) -> tuple[tuple[int, int, int], ...]:
    if n_spin != 2 or n_eta != 2:
        raise ValueError("The canonical Fig.6 full-HF ordering is only defined for n_spin=2, n_eta=2.")
    flavor_order = ((1, 0), (0, 0), (1, 1), (0, 1))
    return tuple((ispin, ieta, iband) for iband in range(n_band) for ispin, ieta in flavor_order)


def _full_flavor_priority(flag: str, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if flag == "random":
        return rng.permutation(idx.ravel(order="F"))
    if flag == "vp":
        return np.transpose(idx, (0, 2, 1)).ravel(order="F")
    if flag == "sp":
        return np.transpose(idx, (2, 1, 0)).ravel(order="F")
    if flag == "chern":
        return idx.ravel(order="F")
    if flag == "sublattice":
        swapped = idx.copy()
        swapped[:, 0, 0], swapped[:, 0, 1] = idx[:, 0, 1].copy(), idx[:, 0, 0].copy()
        return swapped.ravel(order="F")
    raise ValueError(f"Unsupported full flavor-polarization flag: {flag}")


def _random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    # Match the Julia full-HF initializers more closely: they build the
    # rotation from `eigvecs(Hermitian(rand(ComplexF64, ...)))`, i.e. a
    # Hermitian view over a uniformly sampled complex matrix rather than a
    # symmetrized Gaussian draw.
    sampled = rng.random((dim, dim)) + 1j * rng.random((dim, dim))
    hermitian = np.triu(sampled).astype(np.complex128, copy=True)
    hermitian += np.triu(sampled, k=1).conj().T
    diag = np.real(np.diag(sampled))
    hermitian[np.diag_indices(dim)] = diag
    _, vecs = eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)


def _apply_full_valley_rotation(
    density: np.ndarray,
    *,
    alpha: float,
    seed: int,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> None:
    rng = np.random.default_rng(seed)
    idx = np.arange(density.shape[0], dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    for ik in range(density.shape[2]):
        for ispin in range(n_spin):
            block_inds = np.asarray(idx[ispin, :, :].ravel(order="F"), dtype=int)
            unitary = _random_unitary(block_inds.size, rng)
            block = density[np.ix_(block_inds, block_inds, [ik])][:, :, 0]
            density[np.ix_(block_inds, block_inds, [ik])] = (
                (1.0 - alpha) * block + alpha * (unitary.conj().T @ block @ unitary)
            )[:, :, None]


def _apply_full_ivc_rotation(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> None:
    if n_eta != 2:
        raise ValueError("IVC rotation currently expects exactly two valleys.")
    idx = np.arange(density.shape[0], dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    mat = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    for ik in range(density.shape[2]):
        for ispin in range(n_spin):
            for iband in range(n_band):
                block_inds = np.asarray(idx[ispin, :, iband], dtype=int)
                block = density[np.ix_(block_inds, block_inds, [ik])][:, :, 0]
                density[np.ix_(block_inds, block_inds, [ik])] = (mat @ block @ mat)[:, :, None]


def _apply_full_random_rotation(density: np.ndarray, *, alpha: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    nt = density.shape[0]
    for ik in range(density.shape[2]):
        unitary = _random_unitary(nt, rng)
        block = density[:, :, ik]
        density[:, :, ik] = (1.0 - alpha) * block + alpha * (unitary.conj().T @ block @ unitary)


def initialize_full_density(
    h0: np.ndarray,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    init_mode = normalize_full_init_mode(init_mode)
    nt, _, nk = h0.shape
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"H0 dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    density = np.zeros_like(h0)
    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    full_id = identity_block(nt)
    rng = np.random.default_rng(seed)
    valley_rotation_alpha = 0.0
    random_rotation_alpha = 0.0

    if init_mode == "random":
        occupied = rng.permutation(nt * nk)[:total_occupied]
        occ_mask = np.zeros(nt * nk, dtype=bool)
        occ_mask[occupied] = True
        occ_mask = occ_mask.reshape((nt, nk), order="F")
        for ik in range(nk):
            density[:, :, ik][np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
            density[:, :, ik] -= 0.5 * full_id
        valley_rotation_alpha = 1.0
        random_rotation_alpha = 1.0
    elif init_mode == "diag_random":
        return initialize_restricted_density(
            h0,
            nu=nu,
            init_mode="random",
            seed=seed,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
    elif init_mode == "educated":
        occupied_per_k = restricted_occupied_bands_per_k(nu, nt)
        ordered_states = canonical_fig6_state_sequence(n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        for ispin, ieta, iband in ordered_states[:occupied_per_k]:
            density[int(idx[ispin, ieta, iband]), int(idx[ispin, ieta, iband]), :] = 1.0
        for ik in range(nk):
            density[:, :, ik] -= 0.5 * full_id
    elif init_mode == "tivc":
        density = initialize_full_density(h0, nu=nu, init_mode="vp", seed=seed, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        _apply_full_ivc_rotation(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        return density
    elif init_mode == "kivc":
        density = initialize_full_density(h0, nu=nu, init_mode="sublattice", seed=seed, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        _apply_full_ivc_rotation(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
        return density
    else:
        flag = "random" if init_mode == "flavor" else init_mode
        if flag not in {"vp", "sp", "chern", "random", "sublattice", "bm"}:
            raise ValueError(f"Unsupported full flavor init flag: {flag}")

        if flag == "bm":
            energies = np.zeros((nt, nk), dtype=float)
            for ik in range(nk):
                energies[:, ik] = np.diag(h0[:, :, ik]).real
            occupied = np.argsort(energies.ravel(order="F"))[:total_occupied]
            occ_mask = np.zeros(nt * nk, dtype=bool)
            occ_mask[occupied] = True
            occ_mask = occ_mask.reshape((nt, nk), order="F")
            for ik in range(nk):
                density[:, :, ik][np.diag_indices(nt)] = occ_mask[:, ik].astype(np.float64)
                density[:, :, ik] -= 0.5 * full_id
            valley_rotation_alpha = 0.05
        else:
            n_per_flavor = nk
            num_full_flavors = total_occupied // n_per_flavor
            num_partial_flavors = 0 if total_occupied % n_per_flavor == 0 else 1
            flavor_order = _full_flavor_priority(flag, idx, rng)
            selected = np.asarray(flavor_order[: num_full_flavors + num_partial_flavors], dtype=int)
            for ifl in selected[: max(0, selected.size - num_partial_flavors)]:
                density[ifl, ifl, :] = 1.0
            if num_partial_flavors:
                ifl = int(selected[-1])
                remaining = total_occupied - (selected.size - 1) * n_per_flavor
                occupied_k = rng.permutation(n_per_flavor)[:remaining]
                density[ifl, ifl, occupied_k] = 1.0
            for ik in range(nk):
                density[:, :, ik] -= 0.5 * full_id
            valley_rotation_alpha = 0.05

    if valley_rotation_alpha > 0.0:
        _apply_full_valley_rotation(
            density,
            alpha=valley_rotation_alpha,
            seed=seed,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
    if random_rotation_alpha > 0.0:
        _apply_full_random_rotation(density, alpha=random_rotation_alpha, seed=seed)

    return density


def initialize_full_state(
    state: RestrictedHartreeFockState,
    *,
    init_mode: str = "flavor",
    seed: int = 1,
    initial_density: np.ndarray | None = None,
) -> float:
    if initial_density is None:
        state.density[:, :, :] = initialize_full_density(
            state.h0,
            nu=state.nu,
            init_mode=init_mode,
            seed=seed,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
            n_band=state.n_band,
        )
    else:
        initial_density = np.asarray(initial_density, dtype=np.complex128)
        if initial_density.shape != state.density.shape:
            raise ValueError(f"Expected initial_density shape {state.density.shape}, got {initial_density.shape}")
        state.density[:, :, :] = initial_density
    filling = restricted_filling(state.density)
    state.diagnostics["filling"] = filling
    state.diagnostics["offdiag_flavor_norm"] = offdiag_flavor_norm(
        state.density,
        flavor_block_indices(n_spin=state.n_spin, n_eta=state.n_eta, n_band=state.n_band),
    )
    return filling


def build_full_density_from_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
    *,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    nt, _, nk = hamiltonian.shape
    energies = np.zeros((nt, nk), dtype=float)
    sigma_ztauz = np.zeros((nt, nk), dtype=float)
    vecs = np.zeros_like(hamiltonian)

    for ik in range(nk):
        # Use the same dense Hermitian eigensolver family as the Julia
        # reference to reduce cross-language drift when frontier states become
        # nearly degenerate deep into the SCF iterations.
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik], UPLO="U")
        energies[:, ik] = eigvals
        vecs[:, :, ik] = eigvecs
        sigma_ztauz[:, ik] = np.real(np.diag(eigvecs.conj().T @ sigma_z[:, :, ik] @ eigvecs))

    total_occupied = restricted_occupied_state_count(nu, nt, nk)
    occ_mask = _occupied_state_mask(energies, total_occupied)
    mu = find_chemical_potential(energies, (nu + 4.0) / 8.0)

    density = np.zeros_like(hamiltonian)
    full_id = identity_block(nt)
    for ik in range(nk):
        occ_local = np.flatnonzero(occ_mask[:, ik])
        if occ_local.size == 0:
            density[:, :, ik] = -0.5 * full_id
            continue
        occupied_vecs = vecs[:, occ_local, ik]
        # Keep the current Julia full-HF projector convention for benchmark parity.
        density[:, :, ik] = occupied_vecs.conj() @ occupied_vecs.T - 0.5 * full_id

    return density, energies, sigma_ztauz, mu


def build_full_hf_kernel(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
) -> HartreeFockKernel:
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        params=params,
    )
    return build_projected_hf_kernel(
        state,
        screened_overlap_blocks,
        density_builder=lambda hamiltonian: _full_density_update_result(state, hamiltonian),
        energy_functional=compute_hf_energy,
        oda_parameterizer=lambda state_obj, delta_density: oda_parametrization_restricted(
            state_obj,
            delta_density,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
        step_callback=_update_tbg_hf_step_state,
        final_state_callback=_update_tbg_hf_density_update_state,
        convergence_rule="mixed",
        v0=state.v0,
        beta=beta,
    )


def build_full_hf_problem(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    beta: float = 1.0,
    initial_density: np.ndarray | None = None,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=lambda state_obj, *, init_mode, seed: initialize_full_state(
            state_obj,
            init_mode=init_mode,
            seed=seed,
            initial_density=initial_density,
        ),
        kernel=build_full_hf_kernel(
            state,
            overlap_blocks,
            lattice_kvec,
            params,
            beta=beta,
        ),
    )


def run_full_hartree_fock(
    state: RestrictedHartreeFockState,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    *,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1e-3,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    normalized_init_mode = normalize_full_init_mode(init_mode)
    state.diagnostics["beta"] = float(beta)
    state.diagnostics["oda_stall_threshold"] = float(oda_stall_threshold)
    problem = build_full_hf_problem(
        state,
        overlap_blocks,
        lattice_kvec,
        params,
        beta=beta,
        initial_density=initial_density,
    )
    base_run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=normalized_init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
    )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
    )


def run_full_hf_from_bm_solution(
    solution: BMSolution,
    *,
    nu: float,
    init_mode: str = "flavor",
    seed: int = 1,
    beta: float = 1.0,
    max_iter: int = 300,
    overlap_lg: int | None = None,
    precision: float = 1e-5,
    oda_stall_threshold: float = 1e-3,
    initial_density: np.ndarray | None = None,
) -> RestrictedHartreeFockRun:
    state = RestrictedHartreeFockState.from_bm_solution(solution, nu=nu, precision=precision)
    resolved_overlap_lg = solution.lg if overlap_lg is None else int(overlap_lg)
    state.diagnostics["overlap_lg"] = float(resolved_overlap_lg)
    overlap_blocks = build_overlap_block_set(solution, lg=resolved_overlap_lg)
    return run_full_hartree_fock(
        state,
        overlap_blocks,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        seed=seed,
        beta=beta,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        initial_density=initial_density,
    )


def restricted_gap_estimate(energies: np.ndarray, nu: float) -> float:
    nu_norm = restricted_occupied_state_count(nu, energies.shape[0], energies.shape[1])
    sorted_energies = np.sort(energies, axis=None)
    if nu_norm <= 0 or nu_norm >= sorted_energies.size:
        return float("nan")
    return float(sorted_energies[nu_norm] - sorted_energies[nu_norm - 1])


def occupied_sigma_mean(energies: np.ndarray, sigma_ztauz: np.ndarray, nu: float) -> float:
    nu_norm = restricted_occupied_state_count(nu, energies.shape[0], energies.shape[1])
    order = _occupied_state_linear_indices(energies, nu_norm)
    if order.size == 0:
        return float("nan")
    return float(np.mean(np.ravel(sigma_ztauz, order="F")[order]))


def offdiag_flavor_norm(density: np.ndarray, sectors: tuple[tuple[int, ...], ...] | None = None) -> float:
    nt = density.shape[0]
    mask = np.zeros((nt, nt), dtype=bool)
    if sectors is None:
        sectors = flavor_block_indices()
    for inds in sectors:
        idx = np.asarray(inds, dtype=int)
        mask[np.ix_(idx, idx)] = True

    total = 0.0
    for ik in range(density.shape[2]):
        block = density[:, :, ik].copy()
        block[mask] = 0.0
        total += float(np.linalg.norm(block) ** 2)
    return float(np.sqrt(total))
