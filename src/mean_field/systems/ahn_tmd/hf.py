from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from collections.abc import Callable
from typing import Literal

import numpy as np

from mean_field.core.hf.engine import DensityUpdateResult
from mean_field.core.hf.overlap import (
    ProjectedWavefunctionBasis,
    build_projected_overlap_block_set,
)
from mean_field.core.hf.coulomb import (
    ScreenedCoulombParams,
    screened_coulomb,
    screened_coulomb_matrix,
)
from mean_field.core.hf.occupations import (
    flat_sector_indices,
    flatten_sector_blocks,
)
from mean_field.core.hf.interaction import run_projected_hartree_fock
from .lattice import build_rectangular_moire_k_grid, hex_shell_indices
from .model import AhnTMDModel
from .params import AhnTMDParameters, DEFAULT_AHN_TMD_PARAMETERS


@dataclass(frozen=True)
class AhnTMDHFConfig:
    """Paper-convention inputs for Ahn/tMoTe2 projected HF.

    This is only the Ahn system adapter.  SCF/ODA, projected-overlap
    contractions, Hartree/Fock assembly, and Coulomb evaluation are delegated to
    ``mean_field.core.hf``.
    """

    theta_deg: float = 2.1
    n_shell: int = 5
    mesh: tuple[int, int] = (24, 24)
    n_bands: int = 5
    g_shell: int = 5
    epsilon_r: float = 5.0
    xi_nm: float = inf
    finite_zero_limit: bool = False
    occupation_counts: tuple[int, int] = (1, 1)
    precision: float = 1.0e-7
    max_iter: int = 500
    seed: int = 1
    init_mode: str = "bare"
    convergence_rule: Literal["raw", "mixed"] = "raw"
    enforce_time_reversal: bool = True
    band_diagonal_hf: bool = False

    def __post_init__(self) -> None:
        if int(self.n_shell) < 0:
            raise ValueError("n_shell must be non-negative")
        if int(self.n_bands) <= 0:
            raise ValueError("n_bands must be positive")
        if tuple(int(v) for v in self.mesh)[0] <= 0 or tuple(int(v) for v in self.mesh)[1] <= 0:
            raise ValueError(f"mesh must be positive, got {self.mesh}")
        if len(self.occupation_counts) != 2:
            raise ValueError("occupation_counts must contain the two time-reversed valleys")
        if any(int(v) < 0 or int(v) > int(self.n_bands) for v in self.occupation_counts):
            raise ValueError("occupation_counts must lie in [0, n_bands]")

    @property
    def coulomb_params(self) -> ScreenedCoulombParams:
        if np.isinf(float(self.xi_nm)):
            return ScreenedCoulombParams(
                epsilon_r=float(self.epsilon_r),
                d_sc_nm=inf,
                finite_zero_limit=bool(self.finite_zero_limit),
            )
        return ScreenedCoulombParams.from_gate_separation_xi_nm(
            epsilon_r=float(self.epsilon_r),
            xi_nm=float(self.xi_nm),
            finite_zero_limit=bool(self.finite_zero_limit),
        )


@dataclass(frozen=True)
class AhnTMDProjectedHFData:
    model: AhnTMDModel
    config: AhnTMDHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    projected_basis: ProjectedWavefunctionBasis
    electron_energies: np.ndarray  # shape (n_valley, n_band, n_k)
    h0_hole: np.ndarray  # flattened core-HF layout, shape (nt, nt, n_k)
    initial_density: np.ndarray  # stored projector/delta convention, P_hole for nu_h=0 reference
    overlap_blocks: object


@dataclass
class AhnTMDHFState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    precision: float
    v0: float
    mu: float = float("nan")
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


def _hermitize_blocks(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)


def _project_to_valley_band_diagonal(blocks: np.ndarray, *, n_bands: int) -> None:
    arr = np.asarray(blocks, dtype=np.complex128)
    for iv in range(2):
        idx = flat_sector_indices(1, 2, int(n_bands), 0, iv)
        for ik in range(arr.shape[2]):
            block = arr[np.ix_(idx, idx, [ik])][:, :, 0]
            arr[np.ix_(idx, idx, [ik])] = np.diag(np.diag(block))[:, :, None]


def _enforce_time_reversal_blocks(blocks: np.ndarray, *, n_bands: int, mesh: tuple[int, int]) -> None:
    """Enforce K'(k) = conj(K(-k)) in the flattened two-valley layout."""

    arr = np.asarray(blocks, dtype=np.complex128)
    neg = _negative_k_indices(int(mesh[0]), int(mesh[1]))
    k_idx = flat_sector_indices(1, 2, int(n_bands), 0, 0)
    kp_idx = flat_sector_indices(1, 2, int(n_bands), 0, 1)
    k_blocks = arr[np.ix_(k_idx, k_idx, np.arange(arr.shape[2]))]
    kp_blocks = arr[np.ix_(kp_idx, kp_idx, np.arange(arr.shape[2]))]
    k_sym = np.empty_like(k_blocks)
    for ik in range(arr.shape[2]):
        k_sym[:, :, ik] = 0.5 * (k_blocks[:, :, ik] + np.conjugate(kp_blocks[:, :, int(neg[ik])]))
    arr[:, :, :] = 0.0
    for ik in range(arr.shape[2]):
        arr[np.ix_(k_idx, k_idx, [ik])] = k_sym[:, :, ik, None]
        arr[np.ix_(kp_idx, kp_idx, [ik])] = np.conjugate(k_sym[:, :, int(neg[ik])])[:, :, None]
    _hermitize_blocks(arr)


def _negative_k_indices(nx: int, ny: int) -> np.ndarray:
    out, _wrap = _negative_k_indices_and_wraps(nx, ny)
    return out


def _negative_k_indices_and_wraps(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros((int(nx) * int(ny),), dtype=int)
    wraps = np.zeros((int(nx) * int(ny), 2), dtype=int)
    for ix in range(int(nx)):
        for iy in range(int(ny)):
            rx = (-ix) % int(nx)
            ry = (-iy) % int(ny)
            ordinal = ix * int(ny) + iy
            indices[ordinal] = rx * int(ny) + ry
            wraps[ordinal, 0] = 0 if ix == 0 else 1
            wraps[ordinal, 1] = 0 if iy == 0 else 1
    return indices, wraps


def _time_reverse_embedded_wavefunctions(
    values: np.ndarray,
    *,
    width: int,
    local_basis_size: int = 2,
    reciprocal_shift: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply plane-wave time reversal with folded-k sewing.

    If the source K-valley wavefunction is stored at ``k_neg = -k + W`` on the
    periodic mesh, conjugation maps ``k_neg + G`` to ``k - (G + W)``.  Thus the
    K' reciprocal label is ``G' = -G - W``.
    """

    arr = np.asarray(values, dtype=np.complex128).reshape(int(local_basis_size), int(width), int(width), -1, order="F")
    out = np.zeros_like(arr)
    offset = int(width) // 2
    sx, sy = int(reciprocal_shift[0]), int(reciprocal_shift[1])
    for ix in range(int(width)):
        m = ix - offset
        tx = -m + sx + offset
        if tx < 0 or tx >= int(width):
            continue
        for iy in range(int(width)):
            n = iy - offset
            ty = -n + sy + offset
            if ty < 0 or ty >= int(width):
                continue
            out[:, tx, ty, :] = np.conjugate(arr[:, ix, iy, :])
    return out.reshape(values.shape, order="F")


def _embed_valley_wavefunctions(
    model: AhnTMDModel,
    kvec: np.ndarray,
    *,
    n_bands: int,
    mesh: tuple[int, int],
    valleys: tuple[int, int] = (1, -1),
    enforce_time_reversal: bool = True,
) -> tuple[ProjectedWavefunctionBasis, np.ndarray]:
    width = 2 * int(model.n_shell) + 1
    offset = int(model.n_shell)
    local_basis_size = 2
    basis_dim = local_basis_size * width * width
    nk = int(np.asarray(kvec).size)
    wavefunctions = np.zeros((basis_dim, int(n_bands), len(valleys), nk), dtype=np.complex128)
    electron_energies = np.zeros((len(valleys), int(n_bands), nk), dtype=float)

    if tuple(valleys) != (1, -1):
        raise ValueError("Ahn projected-HF adapter expects valleys=(+1,-1)")

    for ik, kval in enumerate(np.asarray(kvec, dtype=np.complex128)):
        vals, vecs = model.diagonalize(complex(kval), valley=1, descending=True, n_bands=int(n_bands))
        electron_energies[0, :, ik] = vals
        for ig, (m, n) in enumerate(model.lattice.g_indices):
            ix = int(m) + offset
            iy = int(n) + offset
            base = local_basis_size * (ix + width * iy)
            wavefunctions[base + 0, :, 0, ik] = vecs[2 * ig + 0, :]
            wavefunctions[base + 1, :, 0, ik] = vecs[2 * ig + 1, :]

    if enforce_time_reversal:
        neg, wraps = _negative_k_indices_and_wraps(int(mesh[0]), int(mesh[1]))
        for ik in range(nk):
            source = int(neg[ik])
            electron_energies[1, :, ik] = electron_energies[0, :, source]
            wrap = wraps[ik]
            wavefunctions[:, :, 1, ik] = _time_reverse_embedded_wavefunctions(
                wavefunctions[:, :, 0, source],
                width=width,
                local_basis_size=local_basis_size,
                reciprocal_shift=(-int(wrap[0]), -int(wrap[1])),
            )
    else:
        for ik, kval in enumerate(np.asarray(kvec, dtype=np.complex128)):
            vals, vecs = model.diagonalize(complex(kval), valley=-1, descending=True, n_bands=int(n_bands))
            electron_energies[1, :, ik] = vals
            for ig, (m, n) in enumerate(model.lattice.g_indices):
                ix = int(m) + offset
                iy = int(n) + offset
                base = local_basis_size * (ix + width * iy)
                wavefunctions[base + 0, :, 1, ik] = vecs[2 * ig + 0, :]
                wavefunctions[base + 1, :, 1, ik] = vecs[2 * ig + 1, :]

    return (
        ProjectedWavefunctionBasis(
            wavefunctions,
            grid_shape=(width, width),
            n_spin=1,
            local_basis_size=local_basis_size,
            name="ahn_tmd_five_valence_hole_basis",
            boundary_mode="zero_fill",
        ),
        electron_energies,
    )


def _hole_h0_from_electron_energies(electron_energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(electron_energies, dtype=float)
    n_valley, n_bands, nk = energies.shape
    blocks = np.zeros((1, n_valley, n_bands, n_bands, nk), dtype=np.complex128)
    for iv in range(n_valley):
        for ib in range(n_bands):
            blocks[0, iv, ib, ib, :] = -energies[iv, ib, :]
    return flatten_sector_blocks(blocks)


def _estimate_mu(sector_energies: np.ndarray, occupation_counts: np.ndarray) -> float:
    occupied: list[float] = []
    empty: list[float] = []
    n_spin, n_valley, n_bands, nk = sector_energies.shape
    for ispin in range(n_spin):
        for iv in range(n_valley):
            n_occ = int(occupation_counts[ispin, iv])
            if n_occ > 0:
                occupied.extend(np.ravel(sector_energies[ispin, iv, :n_occ, :]).tolist())
            if n_occ < n_bands:
                empty.extend(np.ravel(sector_energies[ispin, iv, n_occ:, :]).tolist())
    if occupied and empty:
        return 0.5 * (float(np.max(occupied)) + float(np.min(empty)))
    return float(np.mean(sector_energies))


def ahn_tmd_density_from_fixed_hole_occupations(
    hamiltonian: np.ndarray,
    occupation_counts: tuple[int, int] | np.ndarray,
    *,
    n_bands: int,
) -> DensityUpdateResult:
    """Return stored-projector hole density for fixed valley occupations.

    Ahn's ``nu_h=0`` normal-ordering reference is no-hole in a hole basis, so
    the reference-subtracted density is simply the stored hole projector.
    """

    h = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = h.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square hamiltonian blocks, got {h.shape}")
    n_valley = 2
    if nt != n_valley * int(n_bands):
        raise ValueError(f"Expected flattened size {n_valley * int(n_bands)}, got {nt}")
    occ = np.asarray(occupation_counts, dtype=int).reshape(1, n_valley)
    density = np.zeros_like(h)
    energies = np.zeros((nt, nk), dtype=float)
    sector_energies = np.zeros((1, n_valley, int(n_bands), nk), dtype=float)
    for iv in range(n_valley):
        idx = flat_sector_indices(1, n_valley, int(n_bands), 0, iv)
        n_occ = int(occ[0, iv])
        for ik in range(nk):
            block = h[np.ix_(idx, idx, [ik])][:, :, 0]
            block = 0.5 * (block + block.conjugate().T)
            evals, evecs = np.linalg.eigh(block)
            energies[idx, ik] = evals
            sector_energies[0, iv, :, ik] = evals
            if n_occ == 0:
                projector = np.zeros((int(n_bands), int(n_bands)), dtype=np.complex128)
            elif n_occ == int(n_bands):
                projector = np.eye(int(n_bands), dtype=np.complex128)
            else:
                vecs = evecs[:, :n_occ]
                projector = vecs.conjugate() @ vecs.T
            density[np.ix_(idx, idx, [ik])] = projector[:, :, None]
    return DensityUpdateResult(density=density, energies=energies, mu=_estimate_mu(sector_energies, occ))


def build_ahn_tmd_projected_hf_data(
    config: AhnTMDHFConfig,
    *,
    params: AhnTMDParameters = DEFAULT_AHN_TMD_PARAMETERS,
) -> AhnTMDProjectedHFData:
    model = AhnTMDModel.from_config(theta_deg=float(config.theta_deg), n_shell=int(config.n_shell), params=params)
    nx, ny = (int(config.mesh[0]), int(config.mesh[1]))
    k_grid_frac, k_grid = build_rectangular_moire_k_grid(model.lattice, nx, ny)
    kvec = np.asarray(k_grid.reshape(-1), dtype=np.complex128)
    projected_basis, electron_energies = _embed_valley_wavefunctions(
        model,
        kvec,
        n_bands=int(config.n_bands),
        mesh=(nx, ny),
        enforce_time_reversal=bool(config.enforce_time_reversal),
    )
    h0_hole = _hole_h0_from_electron_energies(electron_energies)
    initial = ahn_tmd_density_from_fixed_hole_occupations(
        h0_hole,
        config.occupation_counts,
        n_bands=int(config.n_bands),
    ).density

    shifts = tuple((-m, -n) for m, n in hex_shell_indices(int(config.g_shell)))
    gvecs = np.asarray([(-shift[0]) * model.lattice.q0 + (-shift[1]) * model.lattice.q1 for shift in shifts], dtype=np.complex128)
    coulomb_params = config.coulomb_params
    hartree_screening = {shift: float(screened_coulomb(gvec, coulomb_params)) for shift, gvec in zip(shifts, gvecs, strict=True)}
    fock_screening = {
        # For overlap Lambda_G(k_target, k_source), the physical transfer is
        # Q = k_target - k_source + G.  The generic Fock kernel indexes
        # coefficient matrices as (target, source), so the finite-k part is
        # kvec[:, None] - kvec[None, :].
        shift: screened_coulomb_matrix(kvec[:, None] - kvec[None, :] + complex(gvec), coulomb_params)
        for shift, gvec in zip(shifts, gvecs, strict=True)
    }
    overlap_blocks = build_projected_overlap_block_set(
        projected_basis,
        shifts=shifts,
        gvecs=gvecs,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )
    return AhnTMDProjectedHFData(
        model=model,
        config=config,
        k_grid_frac=k_grid_frac.reshape(-1, 2),
        kvec=kvec,
        projected_basis=projected_basis,
        electron_energies=electron_energies,
        h0_hole=h0_hole,
        initial_density=initial,
        overlap_blocks=overlap_blocks,
    )


def build_ahn_tmd_hf_state(data: AhnTMDProjectedHFData) -> AhnTMDHFState:
    return AhnTMDHFState(
        h0=np.asarray(data.h0_hole, dtype=np.complex128).copy(),
        density=np.asarray(data.initial_density, dtype=np.complex128).copy(),
        hamiltonian=np.asarray(data.h0_hole, dtype=np.complex128).copy(),
        energies=np.zeros((data.h0_hole.shape[0], data.h0_hole.shape[2]), dtype=float),
        precision=float(data.config.precision),
        v0=1.0 / float(data.model.lattice.moire_area_nm2),
    )


def run_ahn_tmd_projected_hf(
    data: AhnTMDProjectedHFData,
    *,
    step_callback: Callable[[AhnTMDHFState, object], None] | None = None,
    final_state_callback: Callable[[AhnTMDHFState, object], None] | None = None,
) -> object:
    """Run Ahn/tMoTe2 projected HF through the generic core engine."""

    state = build_ahn_tmd_hf_state(data)

    def _postprocess(blocks: np.ndarray) -> None:
        _hermitize_blocks(blocks)
        if bool(data.config.enforce_time_reversal):
            _enforce_time_reversal_blocks(blocks, n_bands=int(data.config.n_bands), mesh=data.config.mesh)
        if bool(data.config.band_diagonal_hf):
            _project_to_valley_band_diagonal(blocks, n_bands=int(data.config.n_bands))

    def initializer(target: AhnTMDHFState, *, init_mode: str, seed: int) -> None:
        if str(init_mode) != "bare":
            raise ValueError("Ahn adapter currently exposes only the documented bare fixed-sector initializer")
        target.density[:, :, :] = data.initial_density
        _postprocess(target.density)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return ahn_tmd_density_from_fixed_hole_occupations(
            hamiltonian,
            data.config.occupation_counts,
            n_bands=int(data.config.n_bands),
        )

    return run_projected_hartree_fock(
        state,
        initializer=initializer,
        density_builder=density_builder,
        overlap_blocks=data.overlap_blocks,
        init_mode=str(data.config.init_mode),
        seed=int(data.config.seed),
        v0=float(state.v0),
        hamiltonian_postprocessor=_postprocess,
        density_postprocessor=_postprocess,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_rule=data.config.convergence_rule,
        max_iter=int(data.config.max_iter),
        oda_stall_threshold=0.0,
    )


__all__ = [
    "AhnTMDHFConfig",
    "AhnTMDHFState",
    "AhnTMDProjectedHFData",
    "ahn_tmd_density_from_fixed_hole_occupations",
    "build_ahn_tmd_hf_state",
    "build_ahn_tmd_projected_hf_data",
    "run_ahn_tmd_projected_hf",
]
