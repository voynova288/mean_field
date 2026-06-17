from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import numpy as np

from ...core.hf import (
    DensityConvention,
    DensityUpdateResult,
    HFOverlapBlockSet,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    ProjectedWavefunctionBasis,
    build_projected_target_hamiltonian,
    calculate_projected_overlap_between,
    conventional_projector_to_stored,
    diagonal_overlap_blocks,
    real_space_cell_area_nm2_from_reciprocal,
    run_hartree_fock_problem,
    stored_projector_to_conventional,
    density_to_stored_delta,
)
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import TDBGLattice, build_moire_k_grid
from .model import TDBGModel
from .params import TDBGParameters
from .topology import translation_srcmap

from .projected_hf_config import (
    SPIN_LABELS,
    TDBGInteractionSettings,
    TDBG_LOCAL_LABELS,
    TDBGPaperUdConvention,
    TDBGProjectedHFConfig,
    TDBGProjectedWindow,
    VALID_PAPER_UD_CONVENTIONS,
    VALLEY_LABELS,
    VALLEY_SEQUENCE,
    tdbg_delta_from_paper_ud_for_valley,
    tdbg_parameters_from_paper_ud_for_valley,
    validate_tdbg_interaction_settings,
    validate_tdbg_projected_hf_config,
)

@dataclass(frozen=True)
class TDBGStateLabel:
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
class TDBGProjectedHFData:
    model: TDBGModel
    config: TDBGProjectedHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    band_indices: tuple[int, ...]
    labels: tuple[TDBGStateLabel, ...]
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, nk, n_q, 4)
    reference_density: np.ndarray
    n_occupied_per_k: int
    lower_band_count: int
    moire_area_nm2: float
    shifts: tuple[tuple[int, int], ...]
    shift_gvecs: np.ndarray
    shift_srcmaps: tuple[np.ndarray, ...]
    valley_params: Mapping[int, TDBGParameters] | None = None

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def n_band(self) -> int:
        return int(len(self.band_indices))


@dataclass
class TDBGProjectedHFState:
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
class TDBGProjectedHFTargetData:
    kvec: np.ndarray
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, n_target, n_q, 4)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class TDBGProjectedHFResult:
    run: HartreeFockRun
    data: TDBGProjectedHFData
    init_mode: str
    seed: int
    order_parameters: dict[str, object]
    energy_components: dict[str, float]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "init_mode": self.init_mode,
            "seed": int(self.seed),
            "converged": bool(self.run.converged),
            "exit_reason": self.run.exit_reason,
            "iterations": int(self.run.iterations),
            "final_error": float(self.run.state.diagnostics.get("final_raw_norm", np.nan)),
            "hf_energy_ev": float(self.run.state.diagnostics.get("hf_energy", np.nan)),
            "order_parameters": self.order_parameters,
            "energy_components_ev": self.energy_components,
        }


def tdbg_band_window_indices(matrix_dim: int, window: TDBGProjectedWindow | str = "two_flat") -> tuple[int, ...]:
    if isinstance(window, str):
        window = TDBGProjectedWindow(name=window)
    if window.band_indices is not None:
        indices = tuple(int(v) for v in window.band_indices)
        if not indices:
            raise ValueError("Explicit TDBG projected-HF band window cannot be empty")
        return indices

    name = window.name.strip().lower().replace("-", "_")
    aliases = {"isolated_cb": 1, "cb": 1, "two_flat": 2, "central2": 2, "central4": 4, "central6": 6}
    if name not in aliases:
        raise ValueError(f"Unsupported TDBG projected-HF window {window.name!r}")
    count = int(aliases[name])
    center = int(matrix_dim) // 2
    if count == 1:
        indices = (center,)
    else:
        start = center - count // 2
        indices = tuple(range(start, start + count))
    if min(indices) < 0 or max(indices) >= int(matrix_dim):
        raise ValueError(f"Band window {indices} is outside matrix_dim={matrix_dim}")
    return indices


def tdbg_moire_area_nm2(lattice: TDBGLattice) -> float:
    return real_space_cell_area_nm2_from_reciprocal(lattice.g_m1, lattice.g_m2)


def _shift_table(lattice: TDBGLattice, g_shells: int | None) -> tuple[tuple[tuple[int, int], ...], np.ndarray, tuple[np.ndarray, ...]]:
    shells = int(math.ceil(2.0 * lattice.cut) + 1) if g_shells is None else int(g_shells)
    shifts: list[tuple[int, int]] = []
    gvecs: list[complex] = []
    srcmaps: list[np.ndarray] = []
    for m in range(-shells, shells + 1):
        for n in range(-shells, shells + 1):
            gvec = m * lattice.g_m1 + n * lattice.g_m2
            src = translation_srcmap(lattice, gvec)
            if np.any(src >= 0):
                shifts.append((int(m), int(n)))
                gvecs.append(complex(gvec))
                srcmaps.append(np.asarray(src, dtype=int))
    return tuple(shifts), np.asarray(gvecs, dtype=np.complex128), tuple(srcmaps)



@dataclass(frozen=True)
class _TDBGQSiteEmbedding:
    grid_shape: tuple[int, int]
    local_basis_size: int
    basis_indices: np.ndarray  # (n_q, 4) indices into ProjectedWavefunctionBasis basis axis.

def _tdbg_q_site_embedding(lattice: TDBGLattice) -> _TDBGQSiteEmbedding:
    """Embed TDBG's finite q-site disk into a rectangular core/hf basis grid.

    The generic core overlap code shifts rectangular reciprocal grids with
    zero-fill boundary conditions. TDBG's q-sites are a finite disk labelled by
    moire reciprocal coordinates plus a sector index. We embed sector `l=0,1`
    and local component `alpha=0..3` into an eight-component local basis on a
    rectangular `(g_m1, g_m2)` coordinate grid, so the trusted core overlap
    helpers can be reused without changing TDBG's finite-cutoff physics.
    """

    q_sites = np.asarray(lattice.q_sites, dtype=float)
    if q_sites.ndim != 2 or q_sites.shape[1] < 3:
        raise ValueError(f"Expected q_sites with columns (qx, qy, sector), got {q_sites.shape}")
    q0 = complex(np.asarray(lattice.q_complex, dtype=np.complex128)[0])
    g1 = complex(lattice.g_m1)
    g2 = complex(lattice.g_m2)
    matrix = np.asarray([[g1.real, g2.real], [g1.imag, g2.imag]], dtype=float)
    coords: list[tuple[int, int, int]] = []
    for site in q_sites:
        sector = int(round(float(site[2])))
        if sector not in {0, 1}:
            raise ValueError(f"TDBG q-site sector must be 0 or 1, got {sector}")
        vector = complex(float(site[0]), float(site[1])) + sector * q0
        coeff = np.linalg.solve(matrix, np.asarray([vector.real, vector.imag], dtype=float))
        axis0 = int(round(float(coeff[0])))
        axis1 = int(round(float(coeff[1])))
        if not np.allclose(coeff, (axis0, axis1), atol=1.0e-8):
            raise ValueError(f"Could not map q-site {site.tolist()} to integer moire coordinates: {coeff}")
        coords.append((axis0, axis1, sector))
    axis0_values = [item[0] for item in coords]
    axis1_values = [item[1] for item in coords]
    min0, max0 = min(axis0_values), max(axis0_values)
    min1, max1 = min(axis1_values), max(axis1_values)
    nx = max0 - min0 + 1
    ny = max1 - min1 + 1
    local_basis_size = 8
    basis_indices = np.zeros((q_sites.shape[0], 4), dtype=int)
    for iq, (axis0, axis1, sector) in enumerate(coords):
        x = axis0 - min0
        y = axis1 - min1
        for alpha in range(4):
            local = 4 * sector + alpha
            basis_indices[iq, alpha] = local + local_basis_size * (x + nx * y)
    return _TDBGQSiteEmbedding(grid_shape=(int(nx), int(ny)), local_basis_size=local_basis_size, basis_indices=basis_indices)

def _tdbg_core_order_permutation(data: TDBGProjectedHFData) -> np.ndarray:
    permutation = np.zeros(data.nt, dtype=int)
    n_spin = len(SPIN_LABELS)
    n_valley = len(VALLEY_SEQUENCE)
    for label in data.labels:
        spin_index = SPIN_LABELS.index(label.spin)
        valley_index = VALLEY_SEQUENCE.index(int(label.valley))
        core_index = spin_index + n_spin * (valley_index + n_valley * int(label.band_position))
        permutation[int(label.index)] = int(core_index)
    return permutation

def _tdbg_projected_wavefunction_basis(data: TDBGProjectedHFData, wavefunctions: np.ndarray, *, name: str) -> ProjectedWavefunctionBasis:
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    if wavefunctions.ndim != 4 or wavefunctions.shape[0] != data.nt or wavefunctions.shape[2:] != (data.model.lattice.n_q, 4):
        raise ValueError(
            f"Expected TDBG wavefunctions shape (nt, nk, n_q, 4) with nt={data.nt}, n_q={data.model.lattice.n_q}; "
            f"got {wavefunctions.shape}"
        )
    nk = int(wavefunctions.shape[1])
    embedding = _tdbg_q_site_embedding(data.model.lattice)
    basis_dim = embedding.local_basis_size * embedding.grid_shape[0] * embedding.grid_shape[1]
    core_wavefunctions = np.zeros((basis_dim, data.n_band, len(VALLEY_SEQUENCE), nk), dtype=np.complex128)
    assigned: set[tuple[int, int]] = set()
    for label in data.labels:
        valley_index = VALLEY_SEQUENCE.index(int(label.valley))
        key = (int(label.band_position), valley_index)
        if key in assigned:
            continue
        assigned.add(key)
        values = wavefunctions[int(label.index)]
        for alpha in range(4):
            core_wavefunctions[embedding.basis_indices[:, alpha], int(label.band_position), valley_index, :] = values[:, :, alpha].T
    return ProjectedWavefunctionBasis(
        core_wavefunctions,
        embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=embedding.local_basis_size,
        name=name,
        boundary_mode="zero_fill",
    )

def _tdbg_total_overlap_from_bases(
    data: TDBGProjectedHFData,
    target_basis: ProjectedWavefunctionBasis,
    source_basis: ProjectedWavefunctionBasis,
    shift: tuple[int, int],
) -> np.ndarray:
    overlap_core = calculate_projected_overlap_between(target_basis, source_basis, int(shift[0]), int(shift[1]))
    permutation = _tdbg_core_order_permutation(data)
    return overlap_core[permutation, :, :, :][:, :, permutation, :]

def _tdbg_total_overlap_between(
    data: TDBGProjectedHFData,
    target_wavefunctions: np.ndarray,
    source_wavefunctions: np.ndarray,
    shift: tuple[int, int],
    *,
    target_name: str = "tdbg-target",
    source_name: str = "tdbg-source",
) -> np.ndarray:
    target_basis = _tdbg_projected_wavefunction_basis(data, target_wavefunctions, name=target_name)
    source_basis = _tdbg_projected_wavefunction_basis(data, source_wavefunctions, name=source_name)
    return _tdbg_total_overlap_from_bases(data, target_basis, source_basis, shift)


_EV_TO_J = 1.602176634e-19
_NM_TO_M = 1.0e-9
_ELECTRON_MASS_KG = 9.1093837015e-31
_HBAR_J_S = 1.054571817e-34
_MU_B_J_PER_T = 9.2740100783e-24

def _projected_orbital_g_matrix(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int,
    band_indices: tuple[int, ...],
    delta_k_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected-band energies, wavefunctions, and orbital-g matrix.

    This implements the Liu SI Eq. (8) structure in the zero-field continuum
    eigenbasis. The derivative matrices are finite differences of the full
    continuum Hamiltonian with respect to kx/ky in units of J*m, so
    `mu_B * g * B` is an energy. The selected subspace is not re-diagonalized
    here; callers add the resulting matrix to the projected one-body `h0`.
    """

    evals_ev, evecs = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=None)
    h_x_plus = build_hamiltonian(k_tilde + float(delta_k_nm_inv), lattice, params, valley=valley)
    h_x_minus = build_hamiltonian(k_tilde - float(delta_k_nm_inv), lattice, params, valley=valley)
    h_y_plus = build_hamiltonian(k_tilde + 1j * float(delta_k_nm_inv), lattice, params, valley=valley)
    h_y_minus = build_hamiltonian(k_tilde - 1j * float(delta_k_nm_inv), lattice, params, valley=valley)
    d_hx_j_m = ((h_x_plus - h_x_minus) / (2.0 * float(delta_k_nm_inv))) * _EV_TO_J * _NM_TO_M
    d_hy_j_m = ((h_y_plus - h_y_minus) / (2.0 * float(delta_k_nm_inv))) * _EV_TO_J * _NM_TO_M
    dx = evecs.conjugate().T @ d_hx_j_m @ evecs
    dy = evecs.conjugate().T @ d_hy_j_m @ evecs
    evals_j = np.asarray(evals_ev, dtype=float) * _EV_TO_J
    selected = np.asarray(band_indices, dtype=int)
    g_matrix = np.zeros((selected.size, selected.size), dtype=np.complex128)
    denom_cutoff = 1.0e-30
    prefactor = -1j * _ELECTRON_MASS_KG / (2.0 * _HBAR_J_S * _HBAR_J_S)
    all_indices = np.arange(evals_j.size, dtype=int)
    for ia, m in enumerate(selected):
        for ib, mp in enumerate(selected):
            terms = np.zeros(evals_j.size, dtype=np.complex128)
            denom_m = evals_j[int(m)] - evals_j
            denom_mp = evals_j[int(mp)] - evals_j
            valid = (np.abs(denom_m) > denom_cutoff) & (np.abs(denom_mp) > denom_cutoff)
            valid &= all_indices != int(m)
            valid &= all_indices != int(mp)
            if np.any(valid):
                berry_comm = dx[int(m), valid] * dy[valid, int(mp)] - dy[int(m), valid] * dx[valid, int(mp)]
                terms[valid] = (1.0 / denom_m[valid] + 1.0 / denom_mp[valid]) * berry_comm
                g_matrix[ia, ib] = prefactor * np.sum(terms[valid])
    g_matrix = 0.5 * (g_matrix + g_matrix.conjugate().T)
    return np.asarray(evals_ev[selected], dtype=float), evecs[:, selected], g_matrix

def _projected_onebody_and_wavefunctions(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int,
    band_indices: tuple[int, ...],
    orbital_zeeman_b_t: float,
    orbital_zeeman_delta_k_nm_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(orbital_zeeman_b_t)) <= 0.0:
        evals, vec = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=max(band_indices) + 1)
        selected = np.asarray(band_indices, dtype=int)
        return np.diag(np.asarray(evals, dtype=float)[selected]).astype(np.complex128), vec[:, selected]
    evals, vec, g_matrix = _projected_orbital_g_matrix(
        k_tilde,
        lattice,
        params,
        valley=valley,
        band_indices=band_indices,
        delta_k_nm_inv=float(orbital_zeeman_delta_k_nm_inv),
    )
    zeeman_ev = (_MU_B_J_PER_T * float(orbital_zeeman_b_t) / _EV_TO_J) * g_matrix
    h0 = np.diag(evals).astype(np.complex128) + zeeman_ev
    h0 = 0.5 * (h0 + h0.conjugate().T)
    return h0, vec

def build_tdbg_projected_hf_data(config: TDBGProjectedHFConfig) -> TDBGProjectedHFData:
    validate_tdbg_projected_hf_config(config)
    valley_params = {
        int(valley): tdbg_parameters_from_paper_ud_for_valley(
            config.paper_ud_ev,
            stacking=config.stacking,
            valley=int(valley),
            convention=config.paper_ud_convention,
        )
        for valley in VALLEY_SEQUENCE
    }
    params = valley_params[VALLEY_SEQUENCE[0]]
    model = TDBGModel.from_config(config.theta_deg, cut=config.cut, params=params)
    band_indices = tdbg_band_window_indices(model.matrix_dim, config.window)
    n_band = len(band_indices)
    if n_band < 1:
        raise ValueError("Projected TDBG window must include at least one band")
    lower_count = 0 if n_band == 1 else n_band // 2
    if n_band != 1 and n_band % 2 != 0:
        raise ValueError(f"Projected TDBG multi-band window must be even, got {n_band}")

    mesh = int(config.mesh_size)
    frac_shift = config.frac_shift if config.frac_shift is not None else (0.5 / mesh, 0.5 / mesh)
    k_grid_frac, kvec_grid = build_moire_k_grid(model.lattice, mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid, dtype=np.complex128).reshape(-1)
    nk = int(kvec.size)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * n_band

    labels: list[TDBGStateLabel] = []
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band_index in enumerate(band_indices):
                idx = iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)
                labels.append(
                    TDBGStateLabel(
                        index=int(idx),
                        spin=spin,
                        valley=int(valley),
                        band_position=int(iband),
                        band_index=int(band_index),
                    )
                )

    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    wavefunctions = np.zeros((nt, nk, model.lattice.n_q, 4), dtype=np.complex128)
    for valley in VALLEY_SEQUENCE:
        valley_labels = [label for label in labels if int(label.valley) == int(valley)]
        for ik, kval in enumerate(kvec):
            h_proj, vec = _projected_onebody_and_wavefunctions(
                kval,
                model.lattice,
                valley_params[int(valley)],
                valley=int(valley),
                band_indices=band_indices,
                orbital_zeeman_b_t=float(config.orbital_zeeman_b_t),
                orbital_zeeman_delta_k_nm_inv=float(config.orbital_zeeman_delta_k_nm_inv),
            )
            for spin in SPIN_LABELS:
                spin_indices = [label.index for label in valley_labels if label.spin == spin]
                h0[np.ix_(spin_indices, spin_indices, [ik])] = h_proj[:, :, None]
            for label in valley_labels:
                wavefunctions[label.index, ik, :, :] = vec[:, label.band_position].reshape(model.lattice.n_q, 4)

    reference_density = np.zeros((nt, nt, nk), dtype=np.complex128)
    if lower_count > 0:
        for label in labels:
            if label.band_position < lower_count:
                reference_density[label.index, label.index, :] = 1.0

    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * lower_count
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(
            f"Invalid TDBG occupied count per k: neutral={neutral_occupied_per_k}, "
            f"filling={config.filling}, occupied={n_occupied_per_k}, nt={nt}"
        )
    shifts, gvecs, srcmaps = _shift_table(model.lattice, config.interaction.g_shells)
    return TDBGProjectedHFData(
        model=model,
        config=config,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec,
        band_indices=band_indices,
        labels=tuple(labels),
        h0=h0,
        wavefunctions=wavefunctions,
        reference_density=reference_density,
        n_occupied_per_k=int(n_occupied_per_k),
        lower_band_count=int(lower_count),
        moire_area_nm2=tdbg_moire_area_nm2(model.lattice),
        shifts=shifts,
        shift_gvecs=gvecs,
        shift_srcmaps=srcmaps,
        valley_params=valley_params,
    )


def _conventional_projector_to_stored(projector: np.ndarray) -> np.ndarray:
    return conventional_projector_to_stored(projector)


def _stored_to_conventional(stored: np.ndarray) -> np.ndarray:
    return stored_projector_to_conventional(stored)


def _first_conduction_indices(data: TDBGProjectedHFData) -> list[int]:
    if data.n_band == 1:
        position = 0
    else:
        position = data.lower_band_count
    return [label.index for label in data.labels if label.band_position == position]

def _active_filling_indices(data: TDBGProjectedHFData) -> list[int]:
    filling = int(data.config.filling)
    if data.n_band == 1:
        position = 0
    elif filling >= 0:
        position = data.lower_band_count
    else:
        if data.lower_band_count <= 0:
            raise ValueError("Negative TDBG filling requires at least one valence band in the projected window")
        position = data.lower_band_count - 1
    return [label.index for label in data.labels if label.band_position == position]

def _reference_projector(data: TDBGProjectedHFData) -> np.ndarray:
    projector = np.zeros((data.nt, data.nt), dtype=np.complex128)
    for label in data.labels:
        if label.band_position < data.lower_band_count:
            projector[label.index, label.index] = 1.0
    return projector

def initialize_tdbg_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Return an absolute occupied projector in the core stored convention.

    The initializer supports positive and negative fillings relative to the
    charge-neutral reference. Positive fillings add projectors in the first
    conduction band; negative fillings remove hole projectors from the highest
    valence band. The unrestricted density builder then refills the lowest HF
    eigenstates at the configured occupation count.
    """

    mode = init_mode.strip().lower().replace("-", "_")
    nk = data.nk
    nt = data.nt
    filling = int(data.config.filling)
    count = abs(filling)
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    active_labels = [data.labels[idx] for idx in _active_filling_indices(data)]
    rng = np.random.default_rng(seed)

    def apply_active_projectors(projector: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        if len(projectors) != count:
            raise ValueError(f"init_mode={init_mode!r} produced {len(projectors)} projectors for filling {filling}")
        for active_projector in projectors:
            if filling >= 0:
                projector += active_projector
            else:
                projector -= active_projector
        return projector

    def basis_projectors(indices: list[int]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for idx in indices:
            vec = np.zeros(nt, dtype=np.complex128)
            vec[int(idx)] = 1.0
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def coherent_projectors(phase: complex, *, k_weight: float = 0.5) -> list[np.ndarray]:
        k_weight = float(k_weight)
        if not 0.0 < k_weight < 1.0:
            raise ValueError(f"IVC valley weight must be in (0, 1), got {k_weight}")
        kp_weight = 1.0 - k_weight
        out: list[np.ndarray] = []
        for spin in SPIN_LABELS:
            states = [label.index for label in active_labels if label.spin == spin]
            if len(states) != 2:
                raise ValueError("IVC initializer requires exactly two valley states per spin in the active filling band")
            vec = np.zeros(nt, dtype=np.complex128)
            vec[states[0]] = math.sqrt(k_weight)
            vec[states[1]] = complex(phase) * math.sqrt(kp_weight)
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def parse_ivc_weight_token(token: str) -> float:
        if not token:
            raise ValueError(f"Biased IVC initializer {init_mode!r} must include a valley weight, e.g. ivc_k85")
        value = float(token.replace("p", "."))
        if value > 1.0:
            value /= 100.0
        if not 0.0 < value < 1.0:
            raise ValueError(f"Biased IVC valley weight must be in (0, 1), got {value} from {init_mode!r}")
        return value

    def biased_coherent_projectors_from_mode() -> list[np.ndarray]:
        phase: complex = 1.0j if mode.endswith("_odd") else 1.0
        base = mode[:-4] if mode.endswith("_odd") else mode
        if base.startswith("ivc_kprime"):
            kp_weight = parse_ivc_weight_token(base[len("ivc_kprime") :])
            return coherent_projectors(phase, k_weight=1.0 - kp_weight)
        if base.startswith("ivc_k"):
            k_weight = parse_ivc_weight_token(base[len("ivc_k") :])
            return coherent_projectors(phase, k_weight=k_weight)
        raise ValueError(f"Unsupported biased IVC initializer {init_mode!r}")

    def random_projectors() -> list[np.ndarray]:
        states = [label.index for label in active_labels]
        if count > len(states):
            raise ValueError(f"Cannot choose {count} active projectors from {len(states)} states")
        z = rng.standard_normal((len(states), len(states))) + 1j * rng.standard_normal((len(states), len(states)))
        herm = z + z.conjugate().T
        _, vecs = np.linalg.eigh(herm)
        out: list[np.ndarray] = []
        for col in range(count):
            vec = np.zeros(nt, dtype=np.complex128)
            vec[np.asarray(states, dtype=int)] = vecs[:, col]
            out.append(np.outer(vec, vec.conjugate()))
        return out

    for ik in range(nk):
        projector = _reference_projector(data)
        if filling == 0:
            pass
        elif mode in {"sp", "sp_up"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "up"]))
        elif mode == "sp_down":
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "down"]))
        elif mode in {"vp", "vp_k"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == 1]))
        elif mode in {"vp_kprime", "vp_kp"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == -1]))
        elif mode in {"ivc", "ivc_even"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0))
        elif mode in {"ivc_odd", "kivc"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0j))
        elif mode.startswith("ivc_k"):
            projector = apply_active_projectors(projector, biased_coherent_projectors_from_mode())
        elif mode in {"random", "random_flavor"}:
            projector = apply_active_projectors(projector, random_projectors())
        elif mode in {"bm", "noninteracting"}:
            projector = np.zeros((nt, nt), dtype=np.complex128)
            evals = np.real(np.diag(data.h0[:, :, ik]))
            for idx in np.argsort(evals, kind="stable")[: data.n_occupied_per_k]:
                projector[int(idx), int(idx)] = 1.0
        else:
            raise ValueError(
                f"Unsupported TDBG projected-HF init_mode={init_mode!r}. "
                "Use sp, sp_down, vp_k, vp_kprime, ivc_even, ivc_odd, ivc_k85, ivc_kprime85, random, or bm."
            )
        density[:, :, ik] = _conventional_projector_to_stored(projector)
    return density

def initialize_tdbg_nu2_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Backward-compatible alias for the generic TDBG filling initializer."""

    return initialize_tdbg_density(data, init_mode=init_mode, seed=seed)

class TDBGProjectedHFInitializer:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, state: TDBGProjectedHFState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = initialize_tdbg_density(self.data, init_mode=init_mode, seed=seed)
        state.diagnostics.update(_numeric_order_parameters(self.data, state.density))


class TDBGProjectedHFDensityBuilder:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occ_mask = tdbg_density_from_hamiltonian(hamiltonian, self.data.n_occupied_per_k)
        observables = {"occupation_mask": occ_mask}
        observables.update(_numeric_order_parameters(self.data, density))
        return DensityUpdateResult(density=density, energies=energies, mu=mu, observables=observables)


def tdbg_density_from_hamiltonian(hamiltonian: np.ndarray, n_occupied_per_k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
            density[:, :, ik] = _conventional_projector_to_stored(projector)
            occ_mask[:nocc, ik] = True
    if nocc <= 0 or nocc >= nt:
        mu = float(np.mean(energies))
    else:
        mu = 0.5 * (float(np.max(energies[:nocc, :])) + float(np.min(energies[nocc:, :])))
    return density, energies, float(mu), occ_mask


def build_tdbg_total_overlap_blocks(data: TDBGProjectedHFData) -> HFOverlapBlockSet:
    """Build intersite total-density overlaps via the reusable core/hf overlap API."""

    settings = data.config.interaction
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    nt, nk = data.nt, data.nk
    source_basis = _tdbg_projected_wavefunction_basis(data, data.wavefunctions, name="tdbg-source-grid")
    for shift, gvec in zip(data.shifts, data.shift_gvecs, strict=True):
        block = _tdbg_total_overlap_from_bases(data, source_basis, source_basis, shift)
        if block.shape != (nt, nk, nt, nk):
            raise ValueError(f"Expected TDBG overlap block shape {(nt, nk, nt, nk)}, got {block.shape} for shift {shift}")
        overlaps[shift] = block
        diagonal[shift] = diagonal_overlap_blocks(block, nt=nt, nk=nk)
        qg = abs(complex(gvec))
        if not (settings.drop_g0_hartree and qg < 1.0e-14):
            hartree_screening[shift] = float(2.0 * math.pi * 1.439964547 / (settings.epsilon_r * math.sqrt(qg * qg + settings.kappa_nm_inv * settings.kappa_nm_inv)))
        qabs = np.abs(data.kvec[None, :] - data.kvec[:, None] + complex(gvec))
        fock_screening[shift] = 2.0 * math.pi * 1.439964547 / (settings.epsilon_r * np.sqrt(qabs * qabs + settings.kappa_nm_inv * settings.kappa_nm_inv))
    return HFOverlapBlockSet(
        shifts=tuple(overlaps.keys()),
        gvecs=np.asarray([complex(data.shift_gvecs[data.shifts.index(shift)]) for shift in overlaps.keys()], dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps=diagonal,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )

def _local_lambda(data: TDBGProjectedHFData, shift_index: int, *, valley_policy: str) -> np.ndarray:
    src = data.shift_srcmaps[shift_index]
    valid = src >= 0
    nt, nk = data.nt, data.nk
    lam = np.zeros((nt, nt, nk, 4), dtype=np.complex128)
    for a, la in enumerate(data.labels):
        wa = np.conj(data.wavefunctions[a][:, valid, :])
        for b, lb in enumerate(data.labels):
            if la.spin != lb.spin:
                continue
            if valley_policy == "valley_diagonal" and int(la.valley) != int(lb.valley):
                continue
            wb = data.wavefunctions[b][:, src[valid], :]
            lam[a, b, :, :] = np.einsum("tqa,tqa->ta", wa, wb, optimize=True)
    return lam


def build_tdbg_onsite_hamiltonian(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    density = np.asarray(density, dtype=np.complex128)
    nt, _, nk = density.shape
    if density.shape != (data.nt, data.nt, data.nk):
        raise ValueError(f"Expected density shape {(data.nt, data.nt, data.nk)}, got {density.shape}")
    scale = float(settings.hubbard_u_ev) * graphene_area_over_moire_area(data.model.lattice)
    out = np.zeros((nt, nt, nk), dtype=np.complex128)
    spin_indices = {spin: [label.index for label in data.labels if label.spin == spin] for spin in SPIN_LABELS}
    opposite = {"up": "down", "down": "up"}
    for ishift, _shift in enumerate(data.shifts):
        lam = _local_lambda(data, ishift, valley_policy=settings.onsite_valley_policy)
        for spin in SPIN_LABELS:
            opp = opposite[spin]
            opp_idx = np.asarray(spin_indices[opp], dtype=int)
            spin_idx = np.asarray(spin_indices[spin], dtype=int)
            rho_opp = np.zeros(4, dtype=np.complex128)
            # rho_alpha(G) = <n_alpha(G)> / nk, using stored P[a,b] = conventional projector[b,a].
            for ik in range(nk):
                pconv_opp = _stored_to_conventional(density[:, :, ik])[np.ix_(opp_idx, opp_idx)]
                lam_opp = lam[np.ix_(opp_idx, opp_idx, [ik], np.arange(4))][:, :, 0, :]
                rho_opp += np.einsum("ab,baq->q", pconv_opp, lam_opp, optimize=True)
            rho_opp /= float(nk)
            for ik in range(nk):
                lam_spin = lam[np.ix_(spin_idx, spin_idx, [ik], np.arange(4))][:, :, 0, :]
                hblock = scale * np.einsum("q,abq->ab", np.conj(rho_opp), lam_spin, optimize=True)
                out[np.ix_(spin_idx, spin_idx, [ik])] += hblock[:, :, None]
    for ik in range(nk):
        out[:, :, ik] = 0.5 * (out[:, :, ik] + out[:, :, ik].conjugate().T)
    return out


def graphene_area_over_moire_area(lattice: TDBGLattice) -> float:
    a = float(lattice.graphene_lattice_constant_nm)
    graphene_area = math.sqrt(3.0) * a * a / 2.0
    return float(graphene_area / tdbg_moire_area_nm2(lattice))


def _reference_subtracted_tdbg_density(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    return density_to_stored_delta(
        density,
        DensityConvention.PROJECTOR,
        reference=data.reference_density,
        reference_policy="require",
    )


def _hartree_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.hartree_reference == "none":
        return density
    if settings.hartree_reference == "charge_neutral":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Hartree reference policy: {settings.hartree_reference!r}")


def _fock_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.fock_density == "absolute":
        return density
    if settings.fock_density == "reference_subtracted":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Fock density policy: {settings.fock_density!r}")


def _split_intersite_overlap_blocks(overlap_blocks: HFOverlapBlockSet) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet]:
    hartree_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
        fock_screening={},
    )
    fock_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={},
        fock_screening=overlap_blocks.fock_screening,
    )
    return hartree_blocks, fock_blocks


def build_tdbg_interaction_components(
    data: TDBGProjectedHFData,
    density: np.ndarray,
    *,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    """Return separately contracted TDBG HF Hamiltonian components.

    Hartree and Fock can intentionally use different density/reference
    policies. Keeping the components separate is required for the energy
    functional: a charge-neutral Hartree potential must be contracted with
    ``P - P_ref``, not with the absolute occupied projector.
    """

    from ...core.hf import build_projected_interaction_hamiltonian

    validate_tdbg_interaction_settings(data.config.interaction)
    settings = data.config.interaction
    density = np.asarray(density, dtype=np.complex128)
    if density.shape != (data.nt, data.nt, data.nk):
        raise ValueError(f"Expected density shape {(data.nt, data.nt, data.nk)}, got {density.shape}")

    components: dict[str, np.ndarray] = {}
    if settings.include_intersite:
        resolved_overlap_blocks = build_tdbg_total_overlap_blocks(data) if overlap_blocks is None else overlap_blocks
        hartree_blocks, fock_blocks = _split_intersite_overlap_blocks(resolved_overlap_blocks)
        v0 = 1.0 / data.moire_area_nm2
        components["hartree"] = build_projected_interaction_hamiltonian(
            _hartree_density_for_policy(data, density),
            hartree_blocks,
            v0=v0,
            beta=1.0,
        )
        components["fock"] = build_projected_interaction_hamiltonian(
            _fock_density_for_policy(data, density),
            fock_blocks,
            v0=v0,
            beta=1.0,
        )
    if settings.include_onsite:
        components["onsite"] = build_tdbg_onsite_hamiltonian(data, density)
    return components


class TDBGProjectedHFInteractionBuilder:
    """Reusable interaction callback with component bookkeeping for core HF."""

    def __init__(self, data: TDBGProjectedHFData, *, overlap_blocks: HFOverlapBlockSet | None = None):
        validate_tdbg_interaction_settings(data.config.interaction)
        self.data = data
        self.overlap_blocks = (
            build_tdbg_total_overlap_blocks(data)
            if data.config.interaction.include_intersite and overlap_blocks is None
            else overlap_blocks
        )
        self._last_density: np.ndarray | None = None
        self._last_components: dict[str, np.ndarray] | None = None

    def components(self, density: np.ndarray) -> dict[str, np.ndarray]:
        density_array = np.asarray(density, dtype=np.complex128)
        if (
            self._last_density is not None
            and self._last_density.shape == density_array.shape
            and np.array_equal(self._last_density, density_array)
        ):
            assert self._last_components is not None
            return self._last_components
        components = build_tdbg_interaction_components(self.data, density_array, overlap_blocks=self.overlap_blocks)
        self._last_density = density_array.copy()
        self._last_components = components
        return components

    def __call__(self, density: np.ndarray) -> np.ndarray:
        density_array = np.asarray(density, dtype=np.complex128)
        total = np.zeros_like(density_array)
        for component in self.components(density_array).values():
            total += component
        return total


def build_tdbg_interaction_builder(data: TDBGProjectedHFData) -> TDBGProjectedHFInteractionBuilder:
    return TDBGProjectedHFInteractionBuilder(data)


def _stored_inner_ev(left: np.ndarray, right: np.ndarray, nk: int) -> float:
    return float(np.einsum("abk,abk->", left, right, optimize=True).real / float(nk))


def tdbg_energy_components(
    data: TDBGProjectedHFData,
    density: np.ndarray,
    *,
    interaction_components: Mapping[str, np.ndarray] | None = None,
    interaction_hamiltonian: np.ndarray | None = None,
) -> dict[str, float]:
    """Return TDBG HF energy components in the stored-projector convention.

    For reference-subtracted Hartree, the energy is
    ``1/2 <H_H[P-P_ref], P-P_ref>``. This avoids the old cross term
    ``1/2 <H_H[P-P_ref], P>`` and keeps state rankings tied to the same
    density policy used to build each potential. Onsite Hubbard is kept as an
    absolute opposite-spin density term; its local/intervalley convention is
    still reported as a physics limitation rather than a paper-comparison gate.
    """

    density = np.asarray(density, dtype=np.complex128)
    if density.shape != (data.nt, data.nt, data.nk):
        raise ValueError(f"Expected density shape {(data.nt, data.nt, data.nk)}, got {density.shape}")
    onebody = _stored_inner_ev(data.h0, density, data.nk)
    onebody_excess = _stored_inner_ev(data.h0, _reference_subtracted_tdbg_density(data, density), data.nk)

    zero = np.zeros_like(density)
    if interaction_components is None:
        if interaction_hamiltonian is None:
            interaction_components = build_tdbg_interaction_components(data, density)
        else:
            # Backward-compatible diagnostic path. Prefer separated components
            # when a reference-subtracted policy is active.
            interaction = 0.5 * _stored_inner_ev(np.asarray(interaction_hamiltonian, dtype=np.complex128), density, data.nk)
            return {
                "onebody_ev": onebody,
                "onebody_excess_ev": onebody_excess,
                "hartree_ev": float("nan"),
                "fock_ev": float("nan"),
                "onsite_ev": float("nan"),
                "interaction_ev": interaction,
                "total_ev": onebody + interaction,
            }

    hartree_ev = 0.5 * _stored_inner_ev(
        np.asarray(interaction_components.get("hartree", zero), dtype=np.complex128),
        _hartree_density_for_policy(data, density),
        data.nk,
    )
    fock_ev = 0.5 * _stored_inner_ev(
        np.asarray(interaction_components.get("fock", zero), dtype=np.complex128),
        _fock_density_for_policy(data, density),
        data.nk,
    )
    onsite_ev = 0.5 * _stored_inner_ev(
        np.asarray(interaction_components.get("onsite", zero), dtype=np.complex128),
        density,
        data.nk,
    )
    interaction = hartree_ev + fock_ev + onsite_ev
    return {
        "onebody_ev": onebody,
        "onebody_excess_ev": onebody_excess,
        "hartree_ev": hartree_ev,
        "fock_ev": fock_ev,
        "onsite_ev": onsite_ev,
        "interaction_ev": interaction,
        "total_ev": onebody + interaction,
    }


def _numeric_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    occ = np.zeros(data.nt, dtype=float)
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        occ += np.real(np.diag(projector))
    occ /= float(data.nk)
    spin_pol = 0.0
    valley_pol = 0.0
    active_spin_pol = 0.0
    active_valley_pol = 0.0
    active_indices = set(_active_filling_indices(data))
    for label in data.labels:
        value = float(occ[label.index])
        spin_sign = 1.0 if label.spin == "up" else -1.0
        valley_sign = 1.0 if int(label.valley) == 1 else -1.0
        spin_pol += spin_sign * value
        valley_pol += valley_sign * value
        if label.index in active_indices:
            active_spin_pol += spin_sign * value
            active_valley_pol += valley_sign * value
    ivc = 0.0
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        for spin in SPIN_LABELS:
            k_idx = [label.index for label in data.labels if label.spin == spin and int(label.valley) == 1 and label.index in active_indices]
            kp_idx = [label.index for label in data.labels if label.spin == spin and int(label.valley) == -1 and label.index in active_indices]
            if k_idx and kp_idx:
                ivc += float(np.linalg.norm(projector[np.ix_(k_idx, kp_idx)]))
    ivc /= float(data.nk)
    return {
        "spin_polarization": float(spin_pol),
        "valley_polarization": float(valley_pol),
        "active_spin_polarization": float(active_spin_pol),
        "active_valley_polarization": float(active_valley_pol),
        # Backward-compatible names for the original nu=+2 conduction-band workflow.
        "cb_spin_polarization": float(active_spin_pol),
        "cb_valley_polarization": float(active_valley_pol),
        "ivc_amplitude": float(ivc),
    }

def tdbg_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, object]:
    numeric = _numeric_order_parameters(data, density)
    occ_by_label: list[dict[str, object]] = []
    occ = np.zeros(data.nt, dtype=float)
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        occ += np.real(np.diag(projector))
    occ /= float(data.nk)
    for label in data.labels:
        item = label.to_dict()
        item["occupation"] = float(occ[label.index])
        occ_by_label.append(item)
    classification = "mixed"
    if abs(numeric["ivc_amplitude"]) > 0.15:
        classification = "IVC_or_valley_coherent"
    elif abs(numeric["cb_valley_polarization"]) > 1.2:
        classification = "VP_K" if numeric["cb_valley_polarization"] > 0 else "VP_Kprime"
    elif abs(numeric["cb_spin_polarization"]) > 1.2:
        classification = "SP_up" if numeric["cb_spin_polarization"] > 0 else "SP_down"
    return {**numeric, "classification": classification, "occupations": occ_by_label}


def build_tdbg_projected_hf_state(data: TDBGProjectedHFData) -> TDBGProjectedHFState:
    return TDBGProjectedHFState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.zeros_like(data.h0),
        hamiltonian=np.asarray(data.h0, dtype=np.complex128).copy(),
        energies=np.zeros((data.nt, data.nk), dtype=float),
        precision=float(data.config.precision),
    )


def build_tdbg_projected_hf_kernel(
    data: TDBGProjectedHFData,
    *,
    interaction_builder: TDBGProjectedHFInteractionBuilder | None = None,
) -> HartreeFockKernel:
    resolved_interaction_builder = build_tdbg_interaction_builder(data) if interaction_builder is None else interaction_builder
    density_builder = TDBGProjectedHFDensityBuilder(data)

    def energy_functional(_interaction_h: np.ndarray, _h0: np.ndarray, density: np.ndarray) -> float:
        return tdbg_energy_components(
            data,
            density,
            interaction_components=resolved_interaction_builder.components(density),
        )["total_ev"]

    if data.config.mix_fallback is None:
        oda_parameterizer = None
    else:
        fixed_mix = float(data.config.mix_fallback)
        oda_parameterizer = lambda _state, _delta_density: fixed_mix

    return HartreeFockKernel(
        interaction_builder=resolved_interaction_builder,
        density_builder=density_builder,
        energy_functional=energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=None,
        convergence_rule="raw",
    )


def build_tdbg_projected_hf_problem(
    data: TDBGProjectedHFData,
    *,
    interaction_builder: TDBGProjectedHFInteractionBuilder | None = None,
) -> HartreeFockProblem:
    return HartreeFockProblem(
        initializer=TDBGProjectedHFInitializer(data),
        kernel=build_tdbg_projected_hf_kernel(data, interaction_builder=interaction_builder),
    )


def run_tdbg_projected_hf(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> TDBGProjectedHFResult:
    state = build_tdbg_projected_hf_state(data)
    interaction_builder = build_tdbg_interaction_builder(data)
    problem = build_tdbg_projected_hf_problem(data, interaction_builder=interaction_builder)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=int(data.config.max_iter),
        oda_stall_threshold=0.0 if data.config.mix_fallback is not None else 1.0e-3,
    )
    components = tdbg_energy_components(
        data,
        run.state.density,
        interaction_components=interaction_builder.components(run.state.density),
    )
    order = tdbg_order_parameters(data, run.state.density)
    run.state.diagnostics.update({k: float(v) for k, v in components.items()})
    return TDBGProjectedHFResult(run=run, data=data, init_mode=init_mode, seed=int(seed), order_parameters=order, energy_components=components)


def scan_tdbg_projected_hf_states(
    config: TDBGProjectedHFConfig,
    *,
    init_modes: tuple[str, ...] = ("sp", "sp_down", "vp_k", "vp_kprime", "ivc_even", "ivc_odd", "random"),
    seeds: tuple[int, ...] = (1, 2, 3),
) -> tuple[TDBGProjectedHFResult, ...]:
    data = build_tdbg_projected_hf_data(config)
    results: list[TDBGProjectedHFResult] = []
    for init_mode in init_modes:
        mode_seeds = seeds if init_mode.startswith("random") else (seeds[0],)
        for seed in mode_seeds:
            results.append(run_tdbg_projected_hf(data, init_mode=init_mode, seed=int(seed)))
    return tuple(results)


def tdbg_hf_grid_band_summary(result: TDBGProjectedHFResult) -> dict[str, object]:
    energies = np.asarray(result.run.state.energies, dtype=float)
    nocc = result.data.n_occupied_per_k
    if nocc <= 0 or nocc >= energies.shape[0]:
        gap = float("nan")
    else:
        gap = float(np.min(energies[nocc:, :]) - np.max(energies[:nocc, :]))
    return {
        "classification": result.order_parameters.get("classification"),
        "init_mode": result.init_mode,
        "seed": int(result.seed),
        "occupied_per_k": int(nocc),
        "hf_grid_gap_ev": gap,
        "hf_energy_ev": float(result.energy_components["total_ev"]),
        "energy_min_ev": float(np.min(energies)),
        "energy_max_ev": float(np.max(energies)),
    }


def liu2022_default_projected_hf_config(
    *,
    mesh_size: int = 9,
    cut: float = 5.0,
    window: str = "two_flat",
    include_intersite: bool = True,
    include_onsite: bool = True,
    filling: int = 2,
    max_iter: int = 300,
    precision: float = 1.0e-7,
) -> TDBGProjectedHFConfig:
    return TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=float(cut),
        mesh_size=int(mesh_size),
        paper_ud_ev=0.09,
        stacking="AB-BA",
        window=TDBGProjectedWindow(name=window),
        filling=int(filling),
        interaction=TDBGInteractionSettings(include_intersite=include_intersite, include_onsite=include_onsite),
        precision=float(precision),
        max_iter=int(max_iter),
    )


def build_tdbg_projected_hf_target_data(data: TDBGProjectedHFData, kvec: np.ndarray) -> TDBGProjectedHFTargetData:
    """Project the same TDBG window on a target k-list for HF band plotting."""

    kvec = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    nt = data.nt
    h0 = np.zeros((nt, nt, kvec.size), dtype=np.complex128)
    wavefunctions = np.zeros((nt, kvec.size, data.model.lattice.n_q, 4), dtype=np.complex128)
    for valley in VALLEY_SEQUENCE:
        valley_labels = [label for label in data.labels if int(label.valley) == int(valley)]
        for ik, kval in enumerate(kvec):
            params = data.valley_params[int(valley)] if data.valley_params is not None else data.model.params
            h_proj, vec = _projected_onebody_and_wavefunctions(
                kval,
                data.model.lattice,
                params,
                valley=int(valley),
                band_indices=data.band_indices,
                orbital_zeeman_b_t=float(data.config.orbital_zeeman_b_t),
                orbital_zeeman_delta_k_nm_inv=float(data.config.orbital_zeeman_delta_k_nm_inv),
            )
            for spin in SPIN_LABELS:
                spin_indices = [label.index for label in valley_labels if label.spin == spin]
                h0[np.ix_(spin_indices, spin_indices, [ik])] = h_proj[:, :, None]
            for label in valley_labels:
                wavefunctions[label.index, ik, :, :] = vec[:, label.band_position].reshape(data.model.lattice.n_q, 4)
    return TDBGProjectedHFTargetData(kvec=kvec, h0=h0, wavefunctions=wavefunctions)


def _total_diagonal_overlap_from_wavefunctions(
    data: TDBGProjectedHFData,
    wavefunctions: np.ndarray,
    shift_index: int,
) -> np.ndarray:
    """Return ``Lambda_ab(k,k+G)`` diagonal blocks without forming full k-k' overlaps.

    Target-path Hartree reconstruction only needs the diagonal target overlap
    ``overlap[a, k_target, b, k_target]``.  Forming the full target-target
    matrix scales as ``(nt * n_target)^2`` and is prohibitively memory hungry
    for dense central-six path plots, so this routine contracts the TDBG q-site
    wavefunctions directly on the finite shifted q-grid.  The selection rules
    match :func:`_tdbg_total_overlap_from_bases`: same spin and same valley,
    summed over all local layer/sublattice components.
    """

    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    if wavefunctions.ndim != 4 or wavefunctions.shape[0] != data.nt or wavefunctions.shape[2:] != (data.model.lattice.n_q, 4):
        raise ValueError(
            f"Expected TDBG wavefunctions shape (nt, nk, n_q, 4) with nt={data.nt}, n_q={data.model.lattice.n_q}; "
            f"got {wavefunctions.shape}"
        )
    src = data.shift_srcmaps[int(shift_index)]
    valid = src >= 0
    nk = int(wavefunctions.shape[1])
    diagonal = np.zeros((data.nt, data.nt, nk), dtype=np.complex128)
    if not np.any(valid):
        return diagonal
    src_valid = src[valid]
    for a, la in enumerate(data.labels):
        wa = np.conj(wavefunctions[a][:, valid, :])
        for b, lb in enumerate(data.labels):
            if la.spin != lb.spin or int(la.valley) != int(lb.valley):
                continue
            wb = wavefunctions[b][:, src_valid, :]
            diagonal[a, b, :] = np.einsum("tqa,tqa->t", wa, wb, optimize=True)
    return diagonal

def _target_source_total_overlap(
    data: TDBGProjectedHFData,
    target: TDBGProjectedHFTargetData,
    shift_index: int,
) -> np.ndarray:
    shift = data.shifts[int(shift_index)]
    return _tdbg_total_overlap_between(
        data,
        target.wavefunctions,
        data.wavefunctions,
        shift,
        target_name="tdbg-target",
        source_name="tdbg-source",
    )

def _local_lambda_from_wavefunctions(
    data: TDBGProjectedHFData,
    wavefunctions: np.ndarray,
    shift_index: int,
    *,
    valley_policy: str,
) -> np.ndarray:
    src = data.shift_srcmaps[shift_index]
    valid = src >= 0
    nt = data.nt
    nk_target = int(wavefunctions.shape[1])
    lam = np.zeros((nt, nt, nk_target, 4), dtype=np.complex128)
    for a, la in enumerate(data.labels):
        wa = np.conj(wavefunctions[a][:, valid, :])
        for b, lb in enumerate(data.labels):
            if la.spin != lb.spin:
                continue
            if valley_policy == "valley_diagonal" and int(la.valley) != int(lb.valley):
                continue
            wb = wavefunctions[b][:, src[valid], :]
            lam[a, b, :, :] = np.einsum("tqa,tqa->ta", wa, wb, optimize=True)
    return lam


def build_tdbg_onsite_target_hamiltonian(
    data: TDBGProjectedHFData,
    target: TDBGProjectedHFTargetData,
    density: np.ndarray,
) -> np.ndarray:
    settings = data.config.interaction
    nt, nk_source = data.nt, data.nk
    out = np.zeros((nt, nt, target.nk), dtype=np.complex128)
    scale = float(settings.hubbard_u_ev) * graphene_area_over_moire_area(data.model.lattice)
    spin_indices = {spin: [label.index for label in data.labels if label.spin == spin] for spin in SPIN_LABELS}
    opposite = {"up": "down", "down": "up"}
    for ishift, _shift in enumerate(data.shifts):
        lam_source = _local_lambda_from_wavefunctions(data, data.wavefunctions, ishift, valley_policy=settings.onsite_valley_policy)
        lam_target = _local_lambda_from_wavefunctions(data, target.wavefunctions, ishift, valley_policy=settings.onsite_valley_policy)
        for spin in SPIN_LABELS:
            opp = opposite[spin]
            opp_idx = np.asarray(spin_indices[opp], dtype=int)
            spin_idx = np.asarray(spin_indices[spin], dtype=int)
            rho_opp = np.zeros(4, dtype=np.complex128)
            for ik in range(nk_source):
                pconv_opp = _stored_to_conventional(density[:, :, ik])[np.ix_(opp_idx, opp_idx)]
                lam_opp = lam_source[np.ix_(opp_idx, opp_idx, [ik], np.arange(4))][:, :, 0, :]
                rho_opp += np.einsum("ab,baq->q", pconv_opp, lam_opp, optimize=True)
            rho_opp /= float(nk_source)
            for it in range(target.nk):
                lam_spin = lam_target[np.ix_(spin_idx, spin_idx, [it], np.arange(4))][:, :, 0, :]
                hblock = scale * np.einsum("q,abq->ab", np.conj(rho_opp), lam_spin, optimize=True)
                out[np.ix_(spin_idx, spin_idx, [it])] += hblock[:, :, None]
    for ik in range(target.nk):
        out[:, :, ik] = 0.5 * (out[:, :, ik] + out[:, :, ik].conjugate().T)
    return out


def _build_tdbg_target_overlap_block_sets(
    data: TDBGProjectedHFData,
    target: TDBGProjectedHFTargetData,
) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet, HFOverlapBlockSet]:
    settings = data.config.interaction
    source_blocks = build_tdbg_total_overlap_blocks(data)
    target_diagonal: dict[tuple[int, int], np.ndarray] = {}
    target_source_overlaps: dict[tuple[int, int], np.ndarray] = {}
    target_source_fock: dict[tuple[int, int], np.ndarray] = {}
    source_basis = _tdbg_projected_wavefunction_basis(data, data.wavefunctions, name="tdbg-source-grid")
    target_basis = _tdbg_projected_wavefunction_basis(data, target.wavefunctions, name="tdbg-target-grid")
    for ishift, shift in enumerate(data.shifts):
        target_diagonal[shift] = _total_diagonal_overlap_from_wavefunctions(data, target.wavefunctions, ishift)
        target_source_overlaps[shift] = _tdbg_total_overlap_from_bases(data, target_basis, source_basis, shift)
        gvec = complex(data.shift_gvecs[ishift])
        qabs = np.abs(data.kvec[None, :] - target.kvec[:, None] + gvec)
        target_source_fock[shift] = 2.0 * math.pi * 1.439964547 / (
            settings.epsilon_r * np.sqrt(qabs * qabs + settings.kappa_nm_inv * settings.kappa_nm_inv)
        )
    target_blocks = HFOverlapBlockSet(
        shifts=source_blocks.shifts,
        gvecs=source_blocks.gvecs,
        overlaps={},
        diagonal_overlaps=target_diagonal,
        hartree_screening={},
        fock_screening={},
    )
    target_source_blocks = HFOverlapBlockSet(
        shifts=source_blocks.shifts,
        gvecs=source_blocks.gvecs,
        overlaps=target_source_overlaps,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening=target_source_fock,
    )
    return source_blocks, target_blocks, target_source_blocks


def _with_fock_screening(blocks: HFOverlapBlockSet, fock_screening: Mapping[tuple[int, int], np.ndarray]) -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=blocks.shifts,
        gvecs=blocks.gvecs,
        overlaps=blocks.overlaps,
        diagonal_overlaps=blocks.diagonal_overlaps,
        hartree_screening={},
        fock_screening=dict(fock_screening),
    )


def build_tdbg_hf_target_hamiltonian(
    data: TDBGProjectedHFData,
    target: TDBGProjectedHFTargetData,
    density: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``H_HF(k_target)`` for paper-path/grid band plots.

    TDBG owns the finite-q-site target/source overlap construction, while the
    reusable core projected-HF target contraction applies Hartree/Fock signs and
    stored-projector conventions. The source density is fixed and is not
    updated on the target path.
    """

    validate_tdbg_interaction_settings(data.config.interaction)
    settings = data.config.interaction
    density = np.asarray(density, dtype=np.complex128)
    if density.shape != (data.nt, data.nt, data.nk):
        raise ValueError(f"Expected density shape {(data.nt, data.nt, data.nk)}, got {density.shape}")

    hamiltonian = np.asarray(target.h0, dtype=np.complex128).copy()
    if settings.include_intersite:
        source_blocks, target_blocks, target_source_blocks = _build_tdbg_target_overlap_block_sets(data, target)
        source_hartree_blocks, source_fock_blocks = _split_intersite_overlap_blocks(source_blocks)
        target_source_hartree_blocks = _with_fock_screening(target_source_blocks, {})
        target_source_fock_blocks = _with_fock_screening(target_source_blocks, target_source_blocks.fock_screening)
        v0 = 1.0 / data.moire_area_nm2
        hamiltonian = build_projected_target_hamiltonian(
            hamiltonian,
            _hartree_density_for_policy(data, density),
            source_overlap_blocks=source_hartree_blocks,
            target_overlap_blocks=target_blocks,
            target_source_overlap_blocks=target_source_hartree_blocks,
            v0=v0,
            beta=1.0,
        )
        hamiltonian = build_projected_target_hamiltonian(
            hamiltonian,
            _fock_density_for_policy(data, density),
            source_overlap_blocks=source_fock_blocks,
            target_overlap_blocks=target_blocks,
            target_source_overlap_blocks=target_source_fock_blocks,
            v0=v0,
            beta=1.0,
        )
    if settings.include_onsite:
        hamiltonian += build_tdbg_onsite_target_hamiltonian(data, target, density)
    for ik in range(target.nk):
        hamiltonian[:, :, ik] = 0.5 * (hamiltonian[:, :, ik] + hamiltonian[:, :, ik].conjugate().T)
    return hamiltonian


def diagonalize_tdbg_hf_target_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def liu2022_projected_hf_metadata(config: TDBGProjectedHFConfig) -> dict[str, object]:
    return {
        "paper": "Liu Nat Commun 2022 TDBG projected-HF pilot (not a reproduction claim)",
        "theta_deg": float(config.theta_deg),
        "paper_ud_ev": float(config.paper_ud_ev),
        "paper_ud_convention": config.paper_ud_convention,
        "code_delta_ev": float(tdbg_delta_from_paper_ud_for_valley(config.paper_ud_ev, 1, convention=config.paper_ud_convention)),
        "code_delta_by_valley_ev": {
            VALLEY_LABELS[valley]: float(tdbg_delta_from_paper_ud_for_valley(config.paper_ud_ev, valley, convention=config.paper_ud_convention))
            for valley in VALLEY_SEQUENCE
        },
        "stacking": config.stacking,
        "cut": float(config.cut),
        "mesh_size": int(config.mesh_size),
        "window": config.window.name,
        "explicit_band_indices": None if config.window.band_indices is None else list(config.window.band_indices),
        "filling": int(config.filling),
        "orbital_zeeman_b_t": float(config.orbital_zeeman_b_t),
        "orbital_zeeman_delta_k_nm_inv": float(config.orbital_zeeman_delta_k_nm_inv),
        "include_intersite": bool(config.interaction.include_intersite),
        "include_onsite": bool(config.interaction.include_onsite),
        "hubbard_u_ev": float(config.interaction.hubbard_u_ev),
        "epsilon_r": float(config.interaction.epsilon_r),
        "kappa_nm_inv": float(config.interaction.kappa_nm_inv),
        "hartree_reference": config.interaction.hartree_reference,
        "fock_density": config.interaction.fock_density,
        "onsite_valley_policy": config.interaction.onsite_valley_policy,
        "density_convention": "core stored projector P[a,b,k]=rho_conventional[b,a,k]",
        "reference_density_convention": "state density is absolute occupied projector; Hartree/Fock policies choose whether to subtract the explicit reference projector",
        "energy_convention": "component-resolved stored-projector functional: Hartree contracts with its policy density, Fock with its policy density, onsite with absolute density",
        "workflow": "self-consistent projected HF from multiple trial states, order-parameter classification, component-resolved diagnostic HF energies, and optional target-path reconstruction",
        "known_limitations": [
            "onsite intervalley/local-channel convention must be checked against Liu SI before any paper-comparison claim",
            "central4/central6 windows require separate topology/window diagnostics before production claims",
            "target-path HF bands are reconstructed from source-grid density; paper overlay and cutoff/mesh convergence remain external validation gates",
        ],
    }
