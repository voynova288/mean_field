from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    ProjectedWavefunctionBasis,
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    conventional_projector_to_stored,
    diagonal_overlap_blocks,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    screened_coulomb,
)

from .domains import HTQGDomain, canonical_domain_key, domain_displacements
from .hamiltonian import centered_band_indices, diagonalize_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice, build_moire_k_grid
from .params import DEFAULT_THETA_DEG, HTQGParams

SPIN_LABELS: tuple[str, str] = ("up", "down")
VALLEY_SEQUENCE: tuple[int, int] = (1, -1)
VALLEY_LABELS: dict[int, str] = {1: "K", -1: "Kprime"}


@dataclass(frozen=True)
class HTQGInteractionSettings:
    """Projected long-range Coulomb settings for HTQG two-flat-band HF.

    ``epsilon_r`` is the dielectric constant.  ``d_sc_nm`` gives a double-gate
    screening length in the standard continuum convention
    ``V(q)=2π(e²/4πϵ0)/(ϵ|q|) tanh(|q| d_sc)``.  The current runner uses this
    as a screened-Coulomb pilot; changing the gate geometry is a physical input,
    not a postprocessing option.
    """

    epsilon_r: float = 10.0
    d_sc_nm: float = 25.0
    g_shells: int = 2
    include_hartree: bool = True
    include_fock: bool = True
    hartree_reference: Literal["charge_neutral", "none"] = "charge_neutral"
    fock_density: Literal["absolute", "reference_subtracted"] = "absolute"
    finite_zero_limit: bool = True


@dataclass(frozen=True)
class HTQGProjectedHFConfig:
    theta_deg: float = DEFAULT_THETA_DEG
    n_shells: int = 6
    mesh_size: int = 7
    active_band_count: int = 2
    domain: str = "alpha_beta_gamma"
    filling: int = 0
    params: HTQGParams = field(default_factory=lambda: HTQGParams.realistic(kappa=0.6))
    interaction: HTQGInteractionSettings = field(default_factory=HTQGInteractionSettings)
    precision: float = 1.0e-7
    max_iter: int = 80
    mixing: float = 0.5
    use_oda: bool = False
    active_basis: Literal["auto", "energy", "sublattice_chern"] = "energy"
    frac_shift: tuple[float, float] | None = None


@dataclass(frozen=True)
class HTQGStateLabel:
    index: int
    spin: str
    valley: int
    band_position: int
    band_index: int

    @property
    def valley_label(self) -> str:
        return VALLEY_LABELS.get(int(self.valley), f"valley{self.valley}")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "spin": self.spin,
            "valley": int(self.valley),
            "valley_label": self.valley_label,
            "band_position": int(self.band_position),
            "band_index": int(self.band_index),
        }


@dataclass(frozen=True)
class _HTQGEmbedding:
    grid_shape: tuple[int, int]
    local_basis_size: int
    basis_indices: np.ndarray  # (N_G, 8)
    origin: tuple[int, int]


@dataclass(frozen=True)
class HTQGProjectedHFData:
    lattice: HTQGLattice
    domain: HTQGDomain
    config: HTQGProjectedHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    band_indices: tuple[int, ...]
    labels: tuple[HTQGStateLabel, ...]
    h0: np.ndarray
    wavefunctions: np.ndarray  # ProjectedWavefunctionBasis layout: (basis, band, valley, k)
    reference_density: np.ndarray
    n_occupied_per_k: int
    moire_area_nm2: float
    shifts: tuple[tuple[int, int], ...]
    shift_gvecs: np.ndarray
    embedding: _HTQGEmbedding

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def n_band(self) -> int:
        return int(len(self.band_indices))

    @property
    def v0(self) -> float:
        return 1.0 / float(self.moire_area_nm2)


@dataclass
class HTQGHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float = float("nan")
    precision: float = 1.0e-7
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class HTQGProjectedHFTargetData:
    kvec: np.ndarray
    h0: np.ndarray
    wavefunctions: np.ndarray

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class HTQGHartreeFockResult:
    data: HTQGProjectedHFData
    run: HartreeFockRun
    init_mode: str
    seed: int
    energy_components: Mapping[str, float]

    @property
    def state(self) -> HTQGHartreeFockState:
        return self.run.state  # type: ignore[return-value]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "domain": self.data.domain.to_dict(),
            "filling": int(self.data.config.filling),
            "init_mode": self.init_mode,
            "seed": int(self.seed),
            "converged": bool(self.run.converged),
            "exit_reason": self.run.exit_reason,
            "iterations": int(self.run.iterations),
            "final_error": None if self.run.iter_err.size == 0 else float(self.run.iter_err[-1]),
            "mu_ev": float(self.state.mu),
            "energy_components_ev_per_cell": dict(self.energy_components),
            "occupation_by_label": occupation_by_label(self.data, self.state.density),
            "grid_gap_ev": gap_estimate(self.state.energies, self.data.n_occupied_per_k),
        }


def validate_htqg_projected_hf_config(config: HTQGProjectedHFConfig) -> None:
    if int(config.n_shells) < 0:
        raise ValueError("n_shells must be non-negative")
    if int(config.mesh_size) <= 0:
        raise ValueError("mesh_size must be positive")
    active_band_count = int(config.active_band_count)
    if active_band_count <= 0:
        raise ValueError("active_band_count must be positive")
    if active_band_count % 2 != 0:
        raise ValueError("active_band_count must be even so charge neutrality is unambiguous")
    if int(config.filling) != config.filling:
        raise ValueError(f"filling must be integer, got {config.filling}")
    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * (active_band_count // 2)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * active_band_count
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(
            f"HTQG filling gives invalid occupied states per k: n_occ={n_occupied_per_k}, nt={nt}, "
            f"active_band_count={active_band_count}, filling={config.filling}"
        )
    if config.interaction.epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive")
    if config.interaction.d_sc_nm < 0.0:
        raise ValueError("d_sc_nm must be non-negative")
    if int(config.interaction.g_shells) < 0:
        raise ValueError("g_shells must be non-negative")
    if config.interaction.hartree_reference not in {"charge_neutral", "none"}:
        raise ValueError(f"Unsupported hartree_reference={config.interaction.hartree_reference!r}")
    if config.interaction.fock_density not in {"absolute", "reference_subtracted"}:
        raise ValueError(f"Unsupported fock_density={config.interaction.fock_density!r}")
    if not (0.0 < float(config.mixing) <= 1.0):
        raise ValueError("mixing must lie in (0, 1]")
    if config.active_basis not in {"auto", "energy", "sublattice_chern"}:
        raise ValueError(f"Unsupported active_basis={config.active_basis!r}")


def htqg_moire_area_nm2(lattice: HTQGLattice) -> float:
    return real_space_cell_area_nm2_from_reciprocal(lattice.b_m1, lattice.b_m2)


def _embedding(lattice: HTQGLattice) -> _HTQGEmbedding:
    indices = np.asarray(lattice.g_indices, dtype=int)
    min1 = int(np.min(indices[:, 0]))
    min2 = int(np.min(indices[:, 1]))
    max1 = int(np.max(indices[:, 0]))
    max2 = int(np.max(indices[:, 1]))
    nx = max1 - min1 + 1
    ny = max2 - min2 + 1
    local = 8
    basis_indices = np.zeros((lattice.n_g, local), dtype=int)
    for ig, (n1, n2) in enumerate(indices):
        ix = int(n1) - min1
        iy = int(n2) - min2
        for alpha in range(local):
            basis_indices[ig, alpha] = alpha + local * (ix + nx * iy)
    return _HTQGEmbedding(grid_shape=(nx, ny), local_basis_size=local, basis_indices=basis_indices, origin=(min1, min2))


def _shift_table(lattice: HTQGLattice, g_shells: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    shells = int(g_shells)
    shifts: list[tuple[int, int]] = []
    gvecs: list[complex] = []
    for m in range(-shells, shells + 1):
        for n in range(-shells, shells + 1):
            shifts.append((int(m), int(n)))
            gvecs.append(complex(m * lattice.b_m1 + n * lattice.b_m2))
    return tuple(shifts), np.asarray(gvecs, dtype=np.complex128)


def _label_index(spin_index: int, valley_index: int, band_position: int) -> int:
    return int(spin_index + len(SPIN_LABELS) * (valley_index + len(VALLEY_SEQUENCE) * band_position))


def _sublattice_sigma_z(lattice: HTQGLattice) -> np.ndarray:
    pattern = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    return np.diag(np.tile(pattern, lattice.n_g)).astype(np.complex128)


def _resolve_active_basis(config: HTQGProjectedHFConfig, domain_key: str) -> str:
    # Keep auto conservative until the Type-I sublattice/Chern active-basis HF
    # convention has a separate validation gate.  The explicit
    # ``sublattice_chern`` option is available for diagnostics only.
    if config.active_basis == "auto":
        return "energy"
    return str(config.active_basis)


def _build_labels(band_indices: tuple[int, ...]) -> tuple[HTQGStateLabel, ...]:
    labels: list[HTQGStateLabel] = []
    for iband, band_index in enumerate(band_indices):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for ispin, spin in enumerate(SPIN_LABELS):
                labels.append(
                    HTQGStateLabel(
                        index=_label_index(ispin, ivalley, iband),
                        spin=spin,
                        valley=int(valley),
                        band_position=int(iband),
                        band_index=int(band_index),
                    )
                )
    return tuple(sorted(labels, key=lambda item: item.index))


def _projected_basis(data: HTQGProjectedHFData, wavefunctions: np.ndarray | None = None, *, name: str) -> ProjectedWavefunctionBasis:
    wf = data.wavefunctions if wavefunctions is None else np.asarray(wavefunctions, dtype=np.complex128)
    return ProjectedWavefunctionBasis(
        wf,
        data.embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=data.embedding.local_basis_size,
        name=name,
        boundary_mode="zero_fill",
    )


def build_htqg_projected_hf_data(config: HTQGProjectedHFConfig) -> HTQGProjectedHFData:
    validate_htqg_projected_hf_config(config)
    lattice = build_htqg_lattice(
        float(config.theta_deg),
        n_shells=int(config.n_shells),
        graphene_lattice_constant_nm=config.params.graphene_lattice_constant_nm,
    )
    domain = domain_displacements(lattice, config.domain)
    band_indices = tuple(int(index) for index in centered_band_indices(lattice.matrix_dim, int(config.active_band_count)))
    labels = _build_labels(band_indices)
    mesh = int(config.mesh_size)
    frac_shift = config.frac_shift if config.frac_shift is not None else (0.5 / mesh, 0.5 / mesh)
    k_grid_frac, kvec_grid = build_moire_k_grid(lattice, mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid, dtype=np.complex128).reshape(-1)
    nk = int(kvec.size)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * len(band_indices)
    emb = _embedding(lattice)
    basis_dim = emb.local_basis_size * emb.grid_shape[0] * emb.grid_shape[1]
    core_wavefunctions = np.zeros((basis_dim, len(band_indices), len(VALLEY_SEQUENCE), nk), dtype=np.complex128)
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    active_basis = _resolve_active_basis(config, domain.key)
    sigma_z = _sublattice_sigma_z(lattice) if active_basis == "sublattice_chern" else None
    for ivalley, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(kvec):
            evals, evecs = diagonalize_hamiltonian(
                complex(kval),
                lattice,
                config.params,
                domain=domain,
                valley=int(valley),
                band_indices=band_indices,
                return_eigenvectors=True,
            )
            if evecs is None:
                raise RuntimeError("Expected HTQG central-band eigenvectors")
            energy_diag = np.diag(np.asarray(evals, dtype=float)).astype(np.complex128)
            if sigma_z is None:
                active_vectors = np.asarray(evecs, dtype=np.complex128)
                local_h0 = energy_diag
            else:
                projected_sigma = np.asarray(evecs, dtype=np.complex128).conjugate().T @ sigma_z @ np.asarray(evecs, dtype=np.complex128)
                _sigma_evals, sigma_basis = np.linalg.eigh(projected_sigma)
                active_vectors = np.asarray(evecs, dtype=np.complex128) @ sigma_basis
                local_h0 = sigma_basis.conjugate().T @ energy_diag @ sigma_basis
            vec_grid = np.asarray(active_vectors, dtype=np.complex128).reshape(lattice.n_g, 8, len(band_indices))
            for iband in range(len(band_indices)):
                for jband in range(len(band_indices)):
                    for ispin in range(len(SPIN_LABELS)):
                        row = _label_index(ispin, ivalley, iband)
                        col = _label_index(ispin, ivalley, jband)
                        h0[row, col, ik] = local_h0[iband, jband]
                for alpha in range(8):
                    core_wavefunctions[emb.basis_indices[:, alpha], iband, ivalley, ik] = vec_grid[:, alpha, iband]
    reference_density = np.zeros((nt, nt, nk), dtype=np.complex128)
    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * (len(band_indices) // 2)
    for ik in range(nk):
        _vals, vecs = np.linalg.eigh(h0[:, :, ik])
        projector = vecs[:, :neutral_occupied_per_k] @ vecs[:, :neutral_occupied_per_k].conjugate().T
        reference_density[:, :, ik] = conventional_projector_to_stored(projector)
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(f"Invalid occupied states per k: {n_occupied_per_k} for nt={nt}")
    shifts, shift_gvecs = _shift_table(lattice, config.interaction.g_shells)
    return HTQGProjectedHFData(
        lattice=lattice,
        domain=domain,
        config=config,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec,
        band_indices=band_indices,
        labels=labels,
        h0=h0,
        wavefunctions=core_wavefunctions,
        reference_density=reference_density,
        n_occupied_per_k=int(n_occupied_per_k),
        moire_area_nm2=htqg_moire_area_nm2(lattice),
        shifts=shifts,
        shift_gvecs=shift_gvecs,
        embedding=emb,
    )


def _screening_values(data: HTQGProjectedHFData, gvec: complex) -> tuple[float, np.ndarray]:
    settings = data.config.interaction
    hartree = float(
        screened_coulomb(
            abs(complex(gvec)),
            epsilon_r=float(settings.epsilon_r),
            d_sc_nm=float(settings.d_sc_nm),
            finite_zero_limit=bool(settings.finite_zero_limit),
        )
    )
    qvals = data.kvec[None, :] - data.kvec[:, None] + complex(gvec)
    fock = np.asarray(
        screened_coulomb(
            qvals,
            epsilon_r=float(settings.epsilon_r),
            d_sc_nm=float(settings.d_sc_nm),
            finite_zero_limit=bool(settings.finite_zero_limit),
        ),
        dtype=float,
    )
    return hartree, fock


def build_htqg_overlap_blocks(data: HTQGProjectedHFData) -> HFOverlapBlockSet:
    basis = _projected_basis(data, name="htqg-source-grid")
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(data.shifts, data.shift_gvecs, strict=True):
        block = calculate_projected_overlap_between(basis, basis, int(shift[0]), int(shift[1]))
        overlaps[shift] = block
        diagonal[shift] = diagonal_overlap_blocks(block, nt=data.nt, nk=data.nk)
        hartree, fock = _screening_values(data, complex(gvec))
        hartree_screening[shift] = hartree
        fock_screening[shift] = fock
    return HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps=diagonal,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def _split_overlap_blocks(overlap_blocks: HFOverlapBlockSet) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet]:
    hartree = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
        fock_screening={},
    )
    fock = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={},
        fock_screening=overlap_blocks.fock_screening,
    )
    return hartree, fock


def _hartree_density(data: HTQGProjectedHFData, density: np.ndarray) -> np.ndarray:
    if data.config.interaction.hartree_reference == "charge_neutral":
        return np.asarray(density, dtype=np.complex128) - data.reference_density
    return np.asarray(density, dtype=np.complex128)


def _fock_density(data: HTQGProjectedHFData, density: np.ndarray) -> np.ndarray:
    if data.config.interaction.fock_density == "reference_subtracted":
        return np.asarray(density, dtype=np.complex128) - data.reference_density
    return np.asarray(density, dtype=np.complex128)


def _build_htqg_interaction_components_from_effective_densities(
    data: HTQGProjectedHFData,
    *,
    hartree_density: np.ndarray,
    fock_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    blocks = build_htqg_overlap_blocks(data) if overlap_blocks is None else overlap_blocks
    hartree_blocks, fock_blocks = _split_overlap_blocks(blocks)
    components: dict[str, np.ndarray] = {}
    if data.config.interaction.include_hartree:
        components["hartree"] = build_projected_interaction_hamiltonian(
            np.asarray(hartree_density, dtype=np.complex128),
            hartree_blocks,
            v0=data.v0,
            beta=1.0,
        )
    else:
        components["hartree"] = np.zeros_like(data.h0)
    if data.config.interaction.include_fock:
        components["fock"] = build_projected_interaction_hamiltonian(
            np.asarray(fock_density, dtype=np.complex128),
            fock_blocks,
            v0=data.v0,
            beta=1.0,
        )
    else:
        components["fock"] = np.zeros_like(data.h0)
    components["total"] = components["hartree"] + components["fock"]
    for value in components.values():
        for ik in range(value.shape[2]):
            value[:, :, ik] = 0.5 * (value[:, :, ik] + value[:, :, ik].conjugate().T)
    return components


def build_htqg_interaction_components(
    data: HTQGProjectedHFData,
    density: np.ndarray,
    *,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    return _build_htqg_interaction_components_from_effective_densities(
        data,
        hartree_density=_hartree_density(data, density),
        fock_density=_fock_density(data, density),
        overlap_blocks=overlap_blocks,
    )


def build_htqg_delta_interaction_components(
    data: HTQGProjectedHFData,
    delta_density: np.ndarray,
    *,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    """Linear HF response to a density update, with no reference offset.

    The SCF interaction is linear in the effective Hartree/Fock densities, but
    neutral-background and optional Fock reference subtractions are constant
    offsets. ODA therefore needs H[delta_P], not H[delta_P - P_ref].
    """
    delta = np.asarray(delta_density, dtype=np.complex128)
    return _build_htqg_interaction_components_from_effective_densities(
        data,
        hartree_density=delta,
        fock_density=delta,
        overlap_blocks=overlap_blocks,
    )


def _stored_inner(left: np.ndarray, right: np.ndarray, nk: int) -> float:
    return float((np.einsum("abk,abk->", left, right, optimize=True) / float(nk)).real)


def htqg_energy_components(
    data: HTQGProjectedHFData,
    density: np.ndarray,
    *,
    interaction_components: Mapping[str, np.ndarray] | None = None,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, float]:
    comps = build_htqg_interaction_components(data, density, overlap_blocks=overlap_blocks) if interaction_components is None else interaction_components
    one_body = _stored_inner(data.h0, density, data.nk)
    hartree = 0.5 * _stored_inner(comps["hartree"], _hartree_density(data, density), data.nk)
    fock = 0.5 * _stored_inner(comps["fock"], _fock_density(data, density), data.nk)
    return {
        "one_body": one_body,
        "hartree": hartree,
        "fock": fock,
        "total": one_body + hartree + fock,
    }


def _active_flavor_order(mode: str, *, seed: int) -> list[tuple[int, int]]:
    all_flavors = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (spin, valley): up-K, down-K, up-K', down-K'
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"flavor", "sp", "spin_valley"}:
        return all_flavors
    if mode in {"spin_up", "spin"}:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if mode in {"spin_down"}:
        return [(1, 0), (1, 1), (0, 0), (0, 1)]
    if mode in {"valley_k", "vp_k"}:
        return [(0, 0), (1, 0), (0, 1), (1, 1)]
    if mode in {"valley_kprime", "vp_kprime"}:
        return [(0, 1), (1, 1), (0, 0), (1, 0)]
    if mode in {"balanced", "bm", "noninteracting"}:
        return all_flavors
    if mode.startswith("random"):
        rng = np.random.default_rng(seed)
        order = all_flavors.copy()
        rng.shuffle(order)
        return order
    raise ValueError(
        f"Unsupported HTQG init_mode={mode!r}; use bm, flavor, spin_up, spin_down, valley_k, valley_kprime, balanced, or random."
    )


def initialize_htqg_density(data: HTQGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    mode = init_mode.strip().lower().replace("-", "_")
    nt, nk = data.nt, data.nk
    filling = int(data.config.filling)
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    if mode in {"bm", "noninteracting"}:
        for ik in range(nk):
            evals = np.real(np.diag(data.h0[:, :, ik]))
            projector = np.zeros((nt, nt), dtype=np.complex128)
            for idx in np.argsort(evals, kind="stable")[: data.n_occupied_per_k]:
                projector[int(idx), int(idx)] = 1.0
            density[:, :, ik] = conventional_projector_to_stored(projector)
        return density

    order = _active_flavor_order(mode, seed=seed)
    count = abs(filling)
    for ik in range(nk):
        projector = np.diag(np.real(np.diag(data.reference_density[:, :, ik]))).astype(np.complex128)
        neutral_band_count = data.n_band // 2
        band_position = neutral_band_count if filling >= 0 else neutral_band_count - 1
        sign = 1.0 if filling >= 0 else -1.0
        for spin_index, valley_index in order[:count]:
            idx = _label_index(spin_index, valley_index, band_position)
            projector[idx, idx] += sign
        density[:, :, ik] = conventional_projector_to_stored(projector)
    return density


class HTQGInitializer:
    def __init__(self, data: HTQGProjectedHFData):
        self.data = data

    def __call__(self, state: HTQGHartreeFockState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = initialize_htqg_density(self.data, init_mode=init_mode, seed=seed)
        state.diagnostics.update(_numeric_order_parameters(self.data, state.density))


class HTQGDensityBuilder:
    def __init__(self, data: HTQGProjectedHFData):
        self.data = data

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occ_mask = htqg_density_from_hamiltonian(hamiltonian, self.data.n_occupied_per_k)
        observables = {"occupation_mask": occ_mask}
        observables.update(_numeric_order_parameters(self.data, density))
        return DensityUpdateResult(density=density, energies=energies, mu=mu, observables=observables)


def htqg_density_from_hamiltonian(hamiltonian: np.ndarray, n_occupied_per_k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    nocc = int(n_occupied_per_k)
    if nocc < 0 or nocc > nt:
        raise ValueError(f"Invalid occupied count per k {nocc} for nt={nt}")
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    energies = np.zeros((nt, nk), dtype=float)
    occ_mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        vals, vecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = vals
        if nocc:
            occupied = vecs[:, :nocc]
            projector = occupied @ occupied.conjugate().T
            density[:, :, ik] = conventional_projector_to_stored(projector)
            occ_mask[:nocc, ik] = True
    if nocc <= 0 or nocc >= nt:
        mu = float(np.mean(energies))
    else:
        mu = 0.5 * (float(np.max(energies[:nocc, :])) + float(np.min(energies[nocc:, :])))
    return density, energies, float(mu), occ_mask


def _hermitize_blocks(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)


def build_htqg_hf_problem(
    data: HTQGProjectedHFData,
    overlap_blocks: HFOverlapBlockSet,
    *,
    step_callback=None,
) -> HartreeFockProblem:
    interaction_builder = lambda density: build_htqg_interaction_components(data, density, overlap_blocks=overlap_blocks)["total"]

    def energy_functional(_interaction_h: np.ndarray, _h0: np.ndarray, density: np.ndarray) -> float:
        return htqg_energy_components(data, density, overlap_blocks=overlap_blocks)["total"]

    if bool(data.config.use_oda):
        oda_parameterizer = None
        oda_delta_interaction_builder = lambda delta: build_htqg_delta_interaction_components(
            data,
            delta,
            overlap_blocks=overlap_blocks,
        )["total"]
        convergence_rule = "raw"
    else:
        oda_parameterizer = lambda _state, _delta: float(data.config.mixing)
        oda_delta_interaction_builder = None
        convergence_rule = "raw"

    kernel = HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=HTQGDensityBuilder(data),
        energy_functional=energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=oda_delta_interaction_builder,
        hamiltonian_postprocessor=_hermitize_blocks,
        step_callback=step_callback,
        convergence_rule=convergence_rule,
    )
    return HartreeFockProblem(initializer=HTQGInitializer(data), kernel=kernel)


def build_htqg_hf_state(data: HTQGProjectedHFData) -> HTQGHartreeFockState:
    return HTQGHartreeFockState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.zeros_like(data.h0, dtype=np.complex128),
        hamiltonian=np.asarray(data.h0, dtype=np.complex128).copy(),
        energies=np.zeros((data.nt, data.nk), dtype=float),
        precision=float(data.config.precision),
    )


def run_htqg_projected_hf(
    data: HTQGProjectedHFData,
    *,
    init_mode: str = "bm",
    seed: int = 1,
    overlap_blocks: HFOverlapBlockSet | None = None,
    step_callback=None,
) -> HTQGHartreeFockResult:
    blocks = build_htqg_overlap_blocks(data) if overlap_blocks is None else overlap_blocks
    state = build_htqg_hf_state(data)
    problem = build_htqg_hf_problem(data, blocks, step_callback=step_callback)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(data.config.max_iter),
        oda_stall_threshold=0.0,
    )
    comps = htqg_energy_components(data, state.density, overlap_blocks=blocks)
    return HTQGHartreeFockResult(data=data, run=run, init_mode=init_mode, seed=int(seed), energy_components=comps)


def build_htqg_target_data(data: HTQGProjectedHFData, kvec: np.ndarray) -> HTQGProjectedHFTargetData:
    target_k = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    nk = int(target_k.size)
    basis_dim = data.wavefunctions.shape[0]
    target_wavefunctions = np.zeros((basis_dim, data.n_band, len(VALLEY_SEQUENCE), nk), dtype=np.complex128)
    target_h0 = np.zeros((data.nt, data.nt, nk), dtype=np.complex128)
    for ivalley, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(target_k):
            evals, evecs = diagonalize_hamiltonian(
                complex(kval),
                data.lattice,
                data.config.params,
                domain=data.domain,
                valley=int(valley),
                band_indices=data.band_indices,
                return_eigenvectors=True,
            )
            if evecs is None:
                raise RuntimeError("Expected HTQG target eigenvectors")
            vec_grid = np.asarray(evecs, dtype=np.complex128).reshape(data.lattice.n_g, 8, data.n_band)
            for iband, energy in enumerate(evals):
                for ispin in range(len(SPIN_LABELS)):
                    idx = _label_index(ispin, ivalley, iband)
                    target_h0[idx, idx, ik] = float(energy)
                for alpha in range(8):
                    target_wavefunctions[data.embedding.basis_indices[:, alpha], iband, ivalley, ik] = vec_grid[:, alpha, iband]
    return HTQGProjectedHFTargetData(kvec=target_k, h0=target_h0, wavefunctions=target_wavefunctions)


def _target_overlap_blocks(data: HTQGProjectedHFData, target: HTQGProjectedHFTargetData) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet]:
    source_basis = _projected_basis(data, name="htqg-source-grid")
    target_basis = ProjectedWavefunctionBasis(
        target.wavefunctions,
        data.embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=data.embedding.local_basis_size,
        name="htqg-target-path",
        boundary_mode="zero_fill",
    )
    target_overlaps: dict[tuple[int, int], np.ndarray] = {}
    target_diagonal: dict[tuple[int, int], np.ndarray] = {}
    target_source_overlaps: dict[tuple[int, int], np.ndarray] = {}
    target_source_fock: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    for shift, gvec in zip(data.shifts, data.shift_gvecs, strict=True):
        target_block = calculate_projected_overlap_between(target_basis, target_basis, int(shift[0]), int(shift[1]))
        target_overlaps[shift] = target_block
        target_diagonal[shift] = diagonal_overlap_blocks(target_block, nt=data.nt, nk=target.nk)
        ts_block = calculate_projected_overlap_between(target_basis, source_basis, int(shift[0]), int(shift[1]))
        target_source_overlaps[shift] = ts_block
        settings = data.config.interaction
        hartree_screening[shift] = float(
            screened_coulomb(
                abs(complex(gvec)),
                epsilon_r=float(settings.epsilon_r),
                d_sc_nm=float(settings.d_sc_nm),
                finite_zero_limit=bool(settings.finite_zero_limit),
            )
        )
        qvals = data.kvec[None, :] - target.kvec[:, None] + complex(gvec)
        target_source_fock[shift] = np.asarray(
            screened_coulomb(
                qvals,
                epsilon_r=float(settings.epsilon_r),
                d_sc_nm=float(settings.d_sc_nm),
                finite_zero_limit=bool(settings.finite_zero_limit),
            ),
            dtype=float,
        )
    target_blocks = HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=target_overlaps,
        diagonal_overlaps=target_diagonal,
        hartree_screening=hartree_screening,
        fock_screening={},
    )
    target_source_blocks = HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=target_source_overlaps,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening=target_source_fock,
    )
    return target_blocks, target_source_blocks


def build_htqg_hf_target_hamiltonian(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
) -> np.ndarray:
    source_blocks = build_htqg_overlap_blocks(data) if source_overlap_blocks is None else source_overlap_blocks
    source_hartree, source_fock = _split_overlap_blocks(source_blocks)
    target_blocks, target_source_blocks = _target_overlap_blocks(data, target)
    hamiltonian = np.asarray(target.h0, dtype=np.complex128).copy()
    if data.config.interaction.include_hartree:
        hamiltonian = build_projected_target_hamiltonian(
            hamiltonian,
            _hartree_density(data, density),
            source_overlap_blocks=source_hartree,
            target_overlap_blocks=target_blocks,
            target_source_overlap_blocks=HFOverlapBlockSet(
                shifts=target_source_blocks.shifts,
                gvecs=target_source_blocks.gvecs,
                overlaps=target_source_blocks.overlaps,
                diagonal_overlaps={},
                hartree_screening={},
                fock_screening={},
            ),
            v0=data.v0,
            beta=1.0,
        )
    if data.config.interaction.include_fock:
        hamiltonian = build_projected_target_hamiltonian(
            hamiltonian,
            _fock_density(data, density),
            source_overlap_blocks=source_fock,
            target_overlap_blocks=target_blocks,
            target_source_overlap_blocks=target_source_blocks,
            v0=data.v0,
            beta=1.0,
        )
    _hermitize_blocks(hamiltonian)
    return hamiltonian


def diagonalize_htqg_hf_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def gap_estimate(energies: np.ndarray, n_occupied_per_k: int) -> float | None:
    arr = np.asarray(energies, dtype=float)
    nocc = int(n_occupied_per_k)
    if nocc <= 0 or nocc >= arr.shape[0]:
        return None
    return float(np.min(arr[nocc:, :]) - np.max(arr[:nocc, :]))


def occupation_by_label(data: HTQGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    diag = np.real(np.diagonal(np.asarray(density), axis1=0, axis2=1).T)
    out: dict[str, float] = {}
    for label in data.labels:
        key = f"{label.spin}_{label.valley_label}_band{label.band_position}"
        out[key] = float(np.mean(diag[label.index, :]))
    return out


def _numeric_order_parameters(data: HTQGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    occ = occupation_by_label(data, density)
    spin_up = sum(value for key, value in occ.items() if key.startswith("up_"))
    spin_down = sum(value for key, value in occ.items() if key.startswith("down_"))
    valley_k = sum(value for key, value in occ.items() if "_K_" in key)
    valley_kp = sum(value for key, value in occ.items() if "_Kprime_" in key)
    return {
        "spin_polarization": float(spin_up - spin_down),
        "valley_polarization": float(valley_k - valley_kp),
        "total_occupation": float(spin_up + spin_down),
    }


__all__ = [
    "HTQGInteractionSettings",
    "HTQGProjectedHFConfig",
    "HTQGProjectedHFData",
    "HTQGHartreeFockResult",
    "build_htqg_delta_interaction_components",
    "build_htqg_hf_target_hamiltonian",
    "build_htqg_overlap_blocks",
    "build_htqg_projected_hf_data",
    "build_htqg_target_data",
    "diagonalize_htqg_hf_hamiltonian",
    "gap_estimate",
    "occupation_by_label",
    "run_htqg_projected_hf",
]
