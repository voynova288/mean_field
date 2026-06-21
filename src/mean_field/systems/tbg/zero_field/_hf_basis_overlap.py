from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403

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

__all__ = [name for name in globals() if not name.startswith('__')]
