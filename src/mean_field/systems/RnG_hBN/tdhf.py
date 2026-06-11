"""RLG/hBN adapter helpers for the generic TDHF/RPA core.

This module is intentionally a thin system layer: it extracts HF orbitals from a
converged RLG/hBN HF state, builds fixed-q particle-hole labels, and provides an
on-demand HF-basis two-body matrix element backed by the existing layer form
factors and full-Q Coulomb kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from ...core.hf import (
    ParticleHolePair,
    SpinValleyFlavor,
    TDHFMatrices,
    assemble_tdhf_liouvillian,
    build_tdhf_matrices,
    occupied_state_mask,
    validate_tdhf_structures,
)
from .cache import load_layer_overlap_blocks_cache, load_projected_basis_cache
from .hf import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
    rlg_hbn_occupied_state_count,
)

MomentumPolicy = Literal["strict", "mod_integer"]


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _reject_zero_literal_q0_fock_env() -> None:
    if _env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False):
        raise ValueError(
            "RLG/hBN TDHF does not support MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1; "
            "rerun/load HF with the physical q=0 Fock convention before TDHF."
        )


@dataclass(frozen=True)
class RLGhBNTDHFOrbitals:
    """HF orbitals and occupation mask with a stable global-index convention.

    ``eigenvectors[basis_index, hf_index, k]`` is the unitary returned by the
    per-k HF diagonalization.  The global TDHF index is
    ``local_hf_index + nt * k_index``, matching Fortran flattening of
    ``energies[local, k]``.
    """

    energies: np.ndarray
    eigenvectors: np.ndarray
    occupied_mask: np.ndarray
    mu: float
    n_spin: int
    n_eta: int
    n_band: int

    @property
    def nt(self) -> int:
        return int(self.energies.shape[0])

    @property
    def nk(self) -> int:
        return int(self.energies.shape[1])

    @property
    def global_energies(self) -> np.ndarray:
        return np.asarray(self.energies, dtype=float).reshape(-1, order="F")

    def global_index(self, local_index: int, k_index: int) -> int:
        local = int(local_index)
        ik = int(k_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        if ik < 0 or ik >= self.nk:
            raise IndexError(f"k_index={ik} outside [0, {self.nk})")
        return local + self.nt * ik

    def decode_global_index(self, global_index: int) -> tuple[int, int]:
        index = int(global_index)
        if index < 0 or index >= self.nt * self.nk:
            raise IndexError(f"global_index={index} outside [0, {self.nt * self.nk})")
        return index % self.nt, index // self.nt

    def flavor_tag(self, local_index: int) -> SpinValleyFlavor:
        local = int(local_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        ispin = local % self.n_spin
        ieta = (local // self.n_spin) % self.n_eta
        # RLG/hBN uses two valleys ordered as K, K'.  Keep integer valley labels
        # system-local to avoid imposing a plotting/string convention here.
        valley = 1 if ieta == 0 else -1 if self.n_eta == 2 else ieta
        return SpinValleyFlavor(spin=ispin, valley=valley)


@dataclass(frozen=True)
class RLGhBNTDHFInteraction:
    """On-demand RLG/hBN HF-basis two-body matrix element for TDHF.

    The returned value follows the generic core convention ``V[a,b,c,d]`` as the
    coefficient of ``c_b† c_a† c_c c_d``.  It is assembled from layer-resolved
    form factors as

    ``sum_Q,l,l' F_l(a,c; Q) conj(F_l'(d,b; Q)) V_ll'(Q) / (N_k Omega)``.

    This is deliberately callable-based; materializing the full four-index
    tensor is only suitable for very small smoke tests.
    """

    basis_data: RLGhBNProjectedBasisData
    overlap_blocks: RLGhBNLayerOverlapBlockSet
    orbitals: RLGhBNTDHFOrbitals
    beta: float = 1.0
    momentum_policy: MomentumPolicy = "strict"
    momentum_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.basis_data.nt != self.orbitals.nt or self.basis_data.nk != self.orbitals.nk:
            raise ValueError(
                "basis_data and orbitals dimensions differ: "
                f"basis nt/nk=({self.basis_data.nt}, {self.basis_data.nk}), "
                f"orbital nt/nk=({self.orbitals.nt}, {self.orbitals.nk})"
            )
        if self.momentum_policy not in {"strict", "mod_integer"}:
            raise ValueError(f"Unsupported momentum_policy={self.momentum_policy!r}")
        _reject_zero_literal_q0_fock_env()

    @property
    def scale(self) -> float:
        return float(self.beta) * float(self.basis_data.v0) / float(self.basis_data.nk)

    def __call__(self, a: int, b: int, c: int, d: int) -> complex:
        return self.matrix_element(a, b, c, d)

    def matrix_element(self, a: int, b: int, c: int, d: int) -> complex:
        a_local, a_k = self.orbitals.decode_global_index(a)
        b_local, b_k = self.orbitals.decode_global_index(b)
        c_local, c_k = self.orbitals.decode_global_index(c)
        d_local, d_k = self.orbitals.decode_global_index(d)
        if not self._momentum_conserved(a_k, b_k, c_k, d_k):
            return 0.0 + 0.0j

        total = 0.0 + 0.0j
        for shift in self.overlap_blocks.shifts:
            layer_overlap = self.overlap_blocks.layer_overlaps[shift]
            fock_kernel = self.overlap_blocks.fock_layer_coulomb[shift]
            if layer_overlap.shape[2] != self.basis_data.nk or layer_overlap.shape[4] != self.basis_data.nk:
                raise ValueError(f"Layer overlap for shift {shift} is incompatible with basis nk={self.basis_data.nk}")
            for target_layer in range(layer_overlap.shape[0]):
                left = self._hf_form_factor(
                    layer_overlap[target_layer],
                    a_local,
                    a_k,
                    c_local,
                    c_k,
                )
                if left == 0.0:
                    continue
                for source_layer in range(layer_overlap.shape[0]):
                    right = self._hf_form_factor(
                        layer_overlap[source_layer],
                        d_local,
                        d_k,
                        b_local,
                        b_k,
                    )
                    if right == 0.0:
                        continue
                    total += (
                        self.scale
                        * complex(fock_kernel[a_k, c_k, target_layer, source_layer])
                        * left
                        * np.conj(right)
                    )
        return complex(total)

    def _hf_form_factor(
        self,
        overlap: np.ndarray,
        target_hf: int,
        target_k: int,
        source_hf: int,
        source_k: int,
    ) -> complex:
        target_vec = self.orbitals.eigenvectors[:, int(target_hf), int(target_k)]
        source_vec = self.orbitals.eigenvectors[:, int(source_hf), int(source_k)]
        block = overlap[:, int(target_k), :, int(source_k)]
        return complex(np.vdot(target_vec, block @ source_vec))

    def _momentum_conserved(self, a_k: int, b_k: int, c_k: int, d_k: int) -> bool:
        frac = np.asarray(self.basis_data.k_grid_frac, dtype=float)
        if frac.shape[0] != self.basis_data.nk or frac.shape[1] != 2:
            raise ValueError(f"Expected k_grid_frac shape (nk, 2), got {frac.shape}")
        # The two form factors use transfers k_c-k_a and k_b-k_d with the same G.
        residual = (frac[int(c_k)] - frac[int(a_k)]) - (frac[int(b_k)] - frac[int(d_k)])
        if self.momentum_policy == "mod_integer":
            residual = residual - np.rint(residual)
        return bool(np.max(np.abs(residual)) <= float(self.momentum_tolerance))


def build_rlg_hbn_tdhf_orbitals(state: RLGhBNHartreeFockState) -> RLGhBNTDHFOrbitals:
    """Diagonalize the converged HF Hamiltonian in the same ordering as HF."""

    hamiltonian = np.asarray(state.hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square HF Hamiltonian blocks, got {hamiltonian.shape}")
    n_spin = int(state.n_spin)
    n_eta = int(state.n_eta)
    n_band = int(state.n_band)
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"HF dimension {nt} incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    energies = np.zeros((nt, nk), dtype=float)
    eigenvectors = np.zeros((nt, nt, nk), dtype=np.complex128)
    occ_mask = np.zeros((nt, nk), dtype=bool)

    if state.occupation_counts is not None:
        counts = np.asarray(state.occupation_counts, dtype=int).reshape((n_spin, n_eta), order="C")
        indices = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
        for ik in range(nk):
            for ispin in range(n_spin):
                for ieta in range(n_eta):
                    block_indices = np.asarray(indices[ispin, ieta, :], dtype=int)
                    block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
                    eigvals, eigvecs = np.linalg.eigh(block)
                    energies[block_indices, ik] = eigvals
                    eigenvectors[np.ix_(block_indices, block_indices, [ik])] = eigvecs[:, :, None]
                    n_occ = int(counts[ispin, ieta])
                    if n_occ > 0:
                        occ_mask[block_indices[:n_occ], ik] = True
    else:
        for ik in range(nk):
            eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
            energies[:, ik] = eigvals
            eigenvectors[:, :, ik] = eigvecs
        total_occupied = rlg_hbn_occupied_state_count(
            state.nu,
            nt,
            nk,
            active_valence_bands=state.active_valence_bands,
            n_spin=n_spin,
            n_eta=n_eta,
        )
        occ_mask[:, :] = occupied_state_mask(energies, total_occupied)

    if np.any(occ_mask) and not np.all(occ_mask):
        mu = 0.5 * (float(np.max(energies[occ_mask])) + float(np.min(energies[~occ_mask])))
    else:
        mu = float(np.mean(energies))
    return RLGhBNTDHFOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occ_mask,
        mu=mu,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )


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


def load_rlg_hbn_tdhf_run_from_archive(
    archive_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    precision: float = 1.0e-6,
) -> RLGhBNHartreeFockRun:
    """Load a saved RLG/hBN HF archive as a TDHF-ready run object.

    Archives written by ``run_rlg_hbn_paper_hf`` store final HF matrices plus
    cache keys for the projected basis and layer-overlap blocks.  This loader
    restores those cached objects and attaches the saved HF state without
    rerunning SCF.  It is intended for TDHF postprocessing jobs.
    """

    path = Path(archive_path).expanduser().resolve()
    with np.load(path) as data:
        archive = {key: data[key] for key in data.files}
    if _archive_bool(archive, "zero_literal_q0_fock", default=False):
        raise ValueError(
            "HF archive was generated with MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1; "
            "TDHF postprocessing requires the physical q=0 Fock convention."
        )
    resolved_summary_path = Path(summary_path) if summary_path is not None else path.with_name("hf_run_summary.json")
    summary: dict[str, object] = {}
    if resolved_summary_path.exists():
        summary = json.loads(resolved_summary_path.read_text(encoding="utf-8"))

    resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    if resolved_cache_dir is None:
        cache_dir_text = _archive_string(archive, "cache_dir") or str(summary.get("cache_dir", ""))
        if not cache_dir_text:
            raise ValueError("HF archive does not record cache_dir; pass cache_dir explicitly")
        resolved_cache_dir = Path(cache_dir_text).expanduser().resolve()

    basis_key = _archive_string(archive, "cache_key_basis") or str(summary.get("cache_key_basis", ""))
    overlap_key = _archive_string(archive, "cache_key_overlap") or str(summary.get("cache_key_overlap", ""))
    if not basis_key or not overlap_key:
        raise ValueError("HF archive must contain cache_key_basis and cache_key_overlap for TDHF postprocessing")

    basis_data = load_projected_basis_cache(resolved_cache_dir, basis_key)
    overlap_blocks = load_layer_overlap_blocks_cache(resolved_cache_dir, overlap_key)

    nu = _archive_scalar_float(archive, "nu", default=float(summary.get("filling", 1.0)))
    occupation_counts_array = np.asarray(archive.get("occupation_counts", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    occupation_counts = None if occupation_counts_array.size == 0 else tuple(int(v) for v in occupation_counts_array)
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=nu,
        precision=float(precision),
        occupation_counts=occupation_counts,
    )
    _assign_archive_array(state, "density", archive, "density")
    _assign_archive_array(state, "hamiltonian", archive, "hamiltonian")
    _assign_archive_array(state, "h0", archive, "h0")
    _assign_archive_array(state, "energies", archive, "energies_mev")
    if "reference_density" in archive:
        _assign_archive_array(state, "reference_density", archive, "reference_density")
    state.mu = _archive_scalar_float(archive, "mu_mev", default=float("nan"))
    state.diagnostics.update(
        {
            "hf_energy": float(summary.get("final_energy_mev", np.nan)),
            "hf_gap": float(summary.get("hf_gap_mev", np.nan)),
            "filling": float(summary.get("filling", nu)),
            "projector_idempotency_residual": float(summary.get("projector_idempotency_residual", np.nan)),
            "density_hermitian_residual": float(summary.get("density_hermitian_residual", np.nan)),
            "hamiltonian_hermitian_residual": float(summary.get("hamiltonian_hermitian_residual", np.nan)),
        }
    )

    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=np.asarray(archive.get("iter_energy_mev", np.asarray([], dtype=float)), dtype=float),
        iter_err=np.asarray(archive.get("iter_err", np.asarray([], dtype=float)), dtype=float),
        iter_oda=np.asarray(archive.get("iter_oda", np.asarray([], dtype=float)), dtype=float),
        init_mode=str(summary.get("init_mode", "archive")),
        seed=int(summary.get("seed", 0)),
        converged=bool(summary.get("converged", False)),
        exit_reason=str(summary.get("exit_reason", "loaded_archive")),
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


def _archive_string(archive: dict[str, np.ndarray], key: str) -> str:
    if key not in archive:
        return ""
    value = np.asarray(archive[key])
    if value.size == 0:
        return ""
    return str(value.reshape(-1)[0])


def _archive_scalar_float(archive: dict[str, np.ndarray], key: str, *, default: float) -> float:
    if key not in archive:
        return float(default)
    value = np.asarray(archive[key], dtype=float).reshape(-1)
    if value.size == 0:
        return float(default)
    return float(value[0])


def _archive_bool(archive: dict[str, np.ndarray], key: str, *, default: bool) -> bool:
    if key not in archive:
        return bool(default)
    value = np.asarray(archive[key]).reshape(-1)
    if value.size == 0:
        return bool(default)
    item = value[0]
    if isinstance(item, np.bool_ | bool):
        return bool(item)
    if isinstance(item, np.integer | int):
        return bool(int(item))
    return str(item).strip().lower() not in {"", "0", "false", "no", "off"}


def _assign_archive_array(
    state: RLGhBNHartreeFockState,
    attribute: str,
    archive: dict[str, np.ndarray],
    key: str,
) -> None:
    if key not in archive:
        raise ValueError(f"HF archive is missing required array {key!r}")
    value = np.asarray(archive[key])
    expected = np.asarray(getattr(state, attribute)).shape
    if value.shape != expected:
        raise ValueError(f"Archive array {key!r} has shape {value.shape}, expected {expected}")
    setattr(state, attribute, value.astype(np.asarray(getattr(state, attribute)).dtype, copy=True))


def build_rlg_hbn_tdhf_q0_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    *,
    beta: float = 1.0,
    include_direct_terms: bool = True,
    include_exchange_terms: bool = True,
    include_b_terms: bool = True,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
) -> TDHFMatrices:
    """Build dense q=0 TDHF matrices for a pre-filtered pair list.

    The ``vectorized`` path groups pairs by k and performs the layer/form-factor
    contractions with NumPy's compiled kernels.  ``generic`` is retained as a
    small-test reference because it calls the on-demand ``V_hf`` element-by-
    element in Python.
    """

    _reject_zero_literal_q0_fock_env()
    if assembly == "generic":
        interaction = build_rlg_hbn_tdhf_interaction(run, orbitals, beta=beta)
        return build_tdhf_matrices(
            orbitals.global_energies,
            pairs,
            interaction,
            include_direct_terms=include_direct_terms,
            include_exchange_terms=include_exchange_terms,
            include_b_terms=include_b_terms,
            structure_tolerance=structure_tolerance,
        )
    if assembly != "vectorized":
        raise ValueError(f"Unsupported RLG/hBN TDHF assembly mode: {assembly!r}")
    return _build_rlg_hbn_tdhf_q0_matrices_vectorized(
        run,
        orbitals,
        pairs,
        beta=beta,
        include_direct_terms=include_direct_terms,
        include_exchange_terms=include_exchange_terms,
        include_b_terms=include_b_terms,
        structure_tolerance=structure_tolerance,
    )


def _build_rlg_hbn_tdhf_q0_matrices_vectorized(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    *,
    beta: float,
    include_direct_terms: bool,
    include_exchange_terms: bool,
    include_b_terms: bool,
    structure_tolerance: float,
) -> TDHFMatrices:
    ph_pairs = tuple(pairs)
    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    if n_pairs == 0:
        L = assemble_tdhf_liouvillian(A, B)
        structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    p_local = np.empty(n_pairs, dtype=int)
    h_local = np.empty(n_pairs, dtype=int)
    pair_k = np.empty(n_pairs, dtype=int)
    for index, pair in enumerate(ph_pairs):
        p_local[index], p_k = orbitals.decode_global_index(pair.particle)
        h_local[index], h_k = orbitals.decode_global_index(pair.hole)
        if p_k != h_k:
            raise ValueError("RLG/hBN q=0 TDHF pair has particle and hole at different k")
        pair_k[index] = p_k
        A[index, index] = orbitals.global_energies[pair.particle] - orbitals.global_energies[pair.hole]

    indices_by_k = tuple(np.nonzero(pair_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)

    for shift in run.overlap_blocks.shifts:
        layer_overlap = np.asarray(run.overlap_blocks.layer_overlaps[shift], dtype=np.complex128)
        fock_kernel = np.asarray(run.overlap_blocks.fock_layer_coulomb[shift], dtype=float)
        n_layer = int(layer_overlap.shape[0])
        if include_direct_terms:
            F_ph = np.zeros((n_layer, n_pairs), dtype=np.complex128)
            F_hp = np.zeros((n_layer, n_pairs), dtype=np.complex128)
            for ik, indices in enumerate(indices_by_k):
                if indices.size == 0:
                    continue
                u_k = U[:, :, ik]
                p_idx = p_local[indices]
                h_idx = h_local[indices]
                for layer in range(n_layer):
                    full = u_k.conj().T @ layer_overlap[layer, :, ik, :, ik] @ u_k
                    F_ph[layer, indices] = full[p_idx, h_idx]
                    F_hp[layer, indices] = full[h_idx, p_idx]
            for ik, row_indices in enumerate(indices_by_k):
                if row_indices.size == 0:
                    continue
                kernel0 = fock_kernel[ik, ik]
                A[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                    "lm,li,mj->ij",
                    kernel0,
                    F_ph[:, row_indices],
                    np.conj(F_ph),
                    optimize=True,
                )
                if include_b_terms:
                    B[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                        "lm,li,mj->ij",
                        kernel0,
                        F_ph[:, row_indices],
                        np.conj(F_hp),
                        optimize=True,
                    )

        if include_exchange_terms:
            for kt, target_indices in enumerate(indices_by_k):
                if target_indices.size == 0:
                    continue
                u_target = U[:, :, kt]
                p_t = p_local[target_indices]
                h_t = h_local[target_indices]
                for ks, source_indices in enumerate(indices_by_k):
                    if source_indices.size == 0:
                        continue
                    u_source = U[:, :, ks]
                    p_s = p_local[source_indices]
                    h_s = h_local[source_indices]
                    pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                    hh = np.empty_like(pp)
                    ph = np.empty_like(pp) if include_b_terms else None
                    hp = np.empty_like(pp) if include_b_terms else None
                    for layer in range(n_layer):
                        full = u_target.conj().T @ layer_overlap[layer, :, kt, :, ks] @ u_source
                        pp[layer] = full[np.ix_(p_t, p_s)]
                        hh[layer] = full[np.ix_(h_t, h_s)]
                        if include_b_terms:
                            ph[layer] = full[np.ix_(p_t, h_s)]  # type: ignore[index]
                            hp[layer] = full[np.ix_(h_t, p_s)]  # type: ignore[index]
                    kernel = fock_kernel[kt, ks]
                    A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                        "lm,lij,mij->ij",
                        kernel,
                        pp,
                        np.conj(hh),
                        optimize=True,
                    )
                    if include_b_terms:
                        B[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                            "lm,lij,mij->ij",
                            kernel,
                            ph,
                            np.conj(hp),
                            optimize=True,
                        )

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)


def build_rlg_hbn_tdhf_q0_matrices(
    run: RLGhBNHartreeFockRun,
    *,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
) -> TDHFMatrices:
    """Dense q=0 TDHF matrices for smoke tests and small checkpoints.

    Large production runs should use channel filtering and eventually a matvec
    eigensolver.  The default dense assembly is vectorized over k blocks and
    layer form factors rather than calling ``V_hf`` for every matrix element.
    """

    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"q=0 TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering, a higher explicit max_pairs on a compute node, or a matvec workflow."
        )
    return build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        beta=beta,
        structure_tolerance=structure_tolerance,
        assembly=assembly,
    )


__all__ = [
    "RLGhBNTDHFInteraction",
    "RLGhBNTDHFOrbitals",
    "build_rlg_hbn_tdhf_interaction",
    "build_rlg_hbn_tdhf_orbitals",
    "build_rlg_hbn_tdhf_q0_matrices",
    "build_rlg_hbn_tdhf_q0_matrices_from_pairs",
    "build_rlg_hbn_tdhf_q0_pairs",
    "load_rlg_hbn_tdhf_run_from_archive",
]
