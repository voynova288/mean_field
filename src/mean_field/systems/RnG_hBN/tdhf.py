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
from typing import Literal, Sequence

import numpy as np

from ...core.contracts import HFRunResult as ContractHFRunResult, HFState as ContractHFState
from ...core.hf import (
    ParticleHolePair,
    SpinValleyFlavor,
    TDHFMatrices,
    TDHFOccupationPolicy,
    TDHFStructureResiduals,
    assemble_tdhf_liouvillian,
    build_tdhf_matrices,
    canonical_tdhf_orbitals_from_hf_run_result,
    canonical_tdhf_orbitals_from_hf_state,
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
FiniteQShortcutChannel = Literal["intervalley", "interspin", "inter_spin_valley"]
FiniteQChannel = Literal["intraflavor", "intervalley", "interspin", "inter_spin_valley"]
FINITE_Q_SHORTCUT_CHANNELS: tuple[str, ...] = ("intervalley", "interspin", "inter_spin_valley")
FINITE_Q_FULL_CHANNELS: tuple[str, ...] = ("intraflavor", *FINITE_Q_SHORTCUT_CHANNELS)
FINITE_Q_KNOWN_CHANNELS: tuple[str, ...] = ("all", *FINITE_Q_FULL_CHANNELS)

@dataclass(frozen=True)
class RLGhBNTDHFFiniteQSupport:
    """Introspection record for the currently implemented RLG/hBN finite-q TDHF modes.

    The canonical TDHF boundary only normalizes HF orbitals.  Whether finite-q
    direct terms, B terms, q/-q pair sectors, and the system-specific ``V_hf``
    are valid is decided in this RLG/hBN system layer.
    """

    supported: bool
    channel: str
    canonical_boundary: bool
    shortcut_exchange_only: bool
    supported_terms: tuple[str, ...]
    unsupported_terms: tuple[str, ...]
    runtime_guards: tuple[str, ...]
    blockers: tuple[str, ...]
    evidence: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": bool(self.supported),
            "channel": self.channel,
            "canonical_boundary": bool(self.canonical_boundary),
            "shortcut_exchange_only": bool(self.shortcut_exchange_only),
            "supported_terms": list(self.supported_terms),
            "unsupported_terms": list(self.unsupported_terms),
            "runtime_guards": list(self.runtime_guards),
            "blockers": list(self.blockers),
            "evidence": list(self.evidence),
            "reason": self.reason,
        }

def rlg_hbn_tdhf_finite_q_mode_support(
    channel: str,
    *,
    shortcut_exchange_only: bool = True,
    canonical_boundary: bool = False,
) -> RLGhBNTDHFFiniteQSupport:
    """Describe whether an RLG/hBN finite-q TDHF mode is implemented.

    This helper is intentionally conservative: it reports only the legacy
    system code paths that actually exist.  In particular, a canonical HF input
    does not supply finite-q direct/B-term formulas or construct ``V_hf``; it
    only supplies parity-checked HF orbitals before the system adapter builds
    the already-implemented flavor-flip exchange shortcut.
    """

    channel_key = str(channel)
    blockers: list[str] = []
    unsupported_terms = (
        ()
        if channel_key == "intraflavor"
        else (
            "finite_q_A_direct",
            "finite_q_B_direct",
            "finite_q_B_exchange",
            "finite_q_all_channel",
        )
    )
    runtime_guards = (
        "conduction_only_active_space",
        "saved_occupation_counts",
        "exactly_one_occupied_spin_valley_flavor",
        "flavor_flip_pairs_only",
        "complete_wrapped_umklapp_overlap_shifts",
        "canonical_orbital_legacy_parity" if canonical_boundary else "legacy_orbital_builder",
    )
    evidence = (
        "finite-q flavor-flip RLG/hBN assembly is build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs",
        "that shortcut assembly sets B=0 and includes only one-body plus A-exchange for flavor-flip sectors",
        "finite-q intraflavor assembly is build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs",
        "the intraflavor assembly implements Eq. D19 q/-q X/Y bookkeeping and reduces to q=0 direct/exchange/B assembly at q=0",
        "V_hf construction remains system-specific; the canonical boundary is only an orbital normalizer",
    )

    if channel_key == "all":
        blockers.append(
            "all-channel finite-q blocks mix flavor-flip and intraflavor sectors; use separated finite-q "
            "intraflavor/intervalley/interspin blocks."
        )
    elif channel_key == "intraflavor":
        pass
    elif channel_key in FINITE_Q_SHORTCUT_CHANNELS:
        if not bool(shortcut_exchange_only):
            blockers.append(
                "shortcut_exchange_only=False requests full finite-q direct/B terms for a flavor-flip channel; "
                "for the fully polarized conduction-only RLG/hBN sectors these terms vanish and the implemented "
                "path is the guarded exchange shortcut."
            )
    else:
        blockers.append(
            f"unknown finite-q channel {channel_key!r}; expected one of {FINITE_Q_KNOWN_CHANNELS}."
        )

    supported = not blockers
    boundary = "canonical" if canonical_boundary else "legacy"
    if supported and channel_key == "intraflavor":
        supported_terms = (
            "hf_energy_difference",
            "finite_q_A_direct",
            "finite_q_A_exchange",
            "finite_q_B_direct",
            "finite_q_B_exchange",
        )
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF supports channel='intraflavor' through the full Eq. D19 "
            "q/-q X/Y bookkeeping: X uses d†_{k+q,p} d_{k,h}, while Y uses d†_{k,h} d_{k-q,p}."
        )
    elif supported:
        supported_terms = ("hf_energy_difference", "finite_q_A_exchange")
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF supports channel={channel_key!r} only through the "
            "conduction-only, fully spin-valley-polarized, flavor-flip exchange shortcut; runtime guards still "
            "validate the active space, occupation_counts, pair flavors, and wrapped Umklapp cache coverage."
        )
    else:
        supported_terms = ()
        reason = (
            f"RLG/hBN {boundary} finite-q TDHF mode is not supported for channel={channel_key!r}, "
            f"shortcut_exchange_only={bool(shortcut_exchange_only)}. " + " ".join(blockers)
        )

    return RLGhBNTDHFFiniteQSupport(
        supported=bool(supported),
        channel=channel_key,
        canonical_boundary=bool(canonical_boundary),
        shortcut_exchange_only=bool(shortcut_exchange_only),
        supported_terms=supported_terms,
        unsupported_terms=unsupported_terms,
        runtime_guards=runtime_guards,
        blockers=tuple(blockers),
        evidence=evidence,
        reason=reason,
    )

def _require_rlg_hbn_tdhf_finite_q_mode_supported(
    channel: str,
    *,
    shortcut_exchange_only: bool,
    canonical_boundary: bool,
) -> RLGhBNTDHFFiniteQSupport:
    support = rlg_hbn_tdhf_finite_q_mode_support(
        channel,
        shortcut_exchange_only=shortcut_exchange_only,
        canonical_boundary=canonical_boundary,
    )
    if support.supported:
        return support
    if support.channel not in FINITE_Q_KNOWN_CHANNELS:
        raise ValueError(support.reason)
    raise NotImplementedError(support.reason)


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
class RLGhBNTDHFMomentumShift:
    """Discrete TDHF momentum transfer on the saved mBZ mesh."""

    shift: tuple[int, int]
    mesh_shape: tuple[int, int]

    @property
    def frac(self) -> tuple[float, float]:
        return (float(self.shift[0]) / float(self.mesh_shape[0]), float(self.shift[1]) / float(self.mesh_shape[1]))


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


def _flavor_block_offdiag_residual(
    hamiltonian: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> float:
    indices = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    max_residual = 0.0
    for ik in range(hamiltonian.shape[2]):
        for spin_a in range(int(n_spin)):
            for eta_a in range(int(n_eta)):
                rows = np.asarray(indices[spin_a, eta_a, :], dtype=int)
                for spin_b in range(int(n_spin)):
                    for eta_b in range(int(n_eta)):
                        if spin_a == spin_b and eta_a == eta_b:
                            continue
                        cols = np.asarray(indices[spin_b, eta_b, :], dtype=int)
                        block = hamiltonian[:, :, ik][np.ix_(rows, cols)]
                        if block.size:
                            max_residual = max(max_residual, float(np.max(np.abs(block))))
    return max_residual


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
        offdiag_residual = _flavor_block_offdiag_residual(
            hamiltonian,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
        if offdiag_residual > 1.0e-8:
            raise ValueError(
                "occupation_counts TDHF orbital shortcut requires a spin-valley block-diagonal HF Hamiltonian; "
                f"max off-block element is {offdiag_residual:.6e}. Use full diagonalization/occupation logic for "
                "flavor-mixed, IVC, or translation-breaking states."
            )
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


def _canonical_hf_state_from_input(canonical_hf: ContractHFState | ContractHFRunResult) -> ContractHFState:
    if isinstance(canonical_hf, ContractHFRunResult):
        return canonical_hf.final_state
    if isinstance(canonical_hf, ContractHFState):
        return canonical_hf
    raise TypeError("canonical_hf must be a mean_field.core.contracts.HFState or HFRunResult")


def _metadata_sequence_len(value: object) -> int:
    if value is None or isinstance(value, (str, bytes)):
        return 0
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _infer_rlg_hbn_dimensions_from_canonical_state(
    state: ContractHFState,
    *,
    n_spin: int | None,
    n_eta: int | None,
    n_band: int | None,
) -> tuple[int, int, int]:
    system = str(getattr(state.basis.physical_model, "system", ""))
    basis_system = str(getattr(state.basis.basis_model, "system", ""))
    if system != "RnG_hBN" or basis_system != "RnG_hBN":
        raise ValueError(
            "RLG/hBN TDHF canonical adapter only accepts canonical HFState/HFRunResult for system "
            f"'RnG_hBN'; got physical={system!r}, basis={basis_system!r}"
        )

    metadata = dict(state.basis.metadata)
    resolved_n_eta = int(n_eta) if n_eta is not None else _metadata_sequence_len(metadata.get("valleys"))
    if resolved_n_eta <= 0:
        resolved_n_eta = 2

    if n_band is not None:
        resolved_n_band = int(n_band)
    else:
        resolved_n_band = _metadata_sequence_len(metadata.get("active_band_indices_per_band"))
        if resolved_n_band <= 0:
            resolved_n_band = int(state.basis.active_valence_bands) + int(state.basis.active_conduction_bands)
    if resolved_n_band <= 0:
        raise ValueError("Cannot infer positive RLG/hBN n_band from canonical ProjectedBasis metadata")

    nt = int(state.hamiltonian.total.shape[0])
    if n_spin is not None:
        resolved_n_spin = int(n_spin)
    else:
        denom = resolved_n_eta * resolved_n_band
        if denom <= 0 or nt % denom != 0:
            raise ValueError(
                "Cannot infer RLG/hBN n_spin from canonical dimensions: "
                f"nt={nt}, n_eta={resolved_n_eta}, n_band={resolved_n_band}"
            )
        resolved_n_spin = nt // denom
    if resolved_n_spin <= 0:
        raise ValueError(f"RLG/hBN n_spin must be positive, got {resolved_n_spin}")
    if resolved_n_spin * resolved_n_eta * resolved_n_band != nt:
        raise ValueError(
            "Canonical HFState dimension is incompatible with requested RLG/hBN dimensions: "
            f"nt={nt}, n_spin={resolved_n_spin}, n_eta={resolved_n_eta}, n_band={resolved_n_band}"
        )
    return resolved_n_spin, resolved_n_eta, resolved_n_band


def _max_abs(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def _reorder_canonical_orbitals_to_rlg_hbn_flavor_order(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    occupied_mask: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    flavor_resolution_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nt, nk = int(energies.shape[0]), int(energies.shape[1])
    indices = np.arange(nt, dtype=int).reshape((int(n_spin), int(n_eta), int(n_band)), order="F")
    reordered_energies = np.empty_like(np.asarray(energies, dtype=float))
    reordered_vectors = np.empty_like(np.asarray(eigenvectors, dtype=np.complex128))
    reordered_occupied = np.empty_like(np.asarray(occupied_mask, dtype=bool))
    tolerance = float(flavor_resolution_tolerance)

    for ik in range(nk):
        assigned = np.zeros((nt,), dtype=bool)
        for ispin in range(int(n_spin)):
            for ieta in range(int(n_eta)):
                target_indices = np.asarray(indices[ispin, ieta, :], dtype=int)
                weights = np.sum(np.abs(eigenvectors[target_indices, :, ik]) ** 2, axis=0)
                selected = np.flatnonzero(weights >= 1.0 - tolerance)
                if selected.size != int(n_band):
                    best_weight = float(np.max(weights)) if weights.size else 0.0
                    raise ValueError(
                        "Canonical RLG/hBN TDHF orbitals are not spin-valley flavor-resolved. "
                        "The generic canonical diagonalization may mix degenerate or flavor-mixed sectors, "
                        "so RLG/hBN flavor-tagged TDHF pairs would be ambiguous; use the legacy system "
                        "orbital builder or add a system-specific flavor gauge. "
                        f"k={ik}, spin={ispin}, eta={ieta}, selected={selected.size}, "
                        f"expected={n_band}, best_block_weight={best_weight:.6e}"
                    )
                if np.any(assigned[selected]):
                    raise ValueError("Canonical RLG/hBN TDHF flavor assignment is not one-to-one")
                order = np.argsort(np.asarray(energies[selected, ik], dtype=float), kind="stable")
                selected = selected[order]
                reordered_energies[target_indices, ik] = energies[selected, ik]
                reordered_vectors[:, target_indices, ik] = eigenvectors[:, selected, ik]
                reordered_occupied[target_indices, ik] = occupied_mask[selected, ik]
                assigned[selected] = True
        if not np.all(assigned):
            missing = np.flatnonzero(~assigned)
            raise ValueError(f"Canonical RLG/hBN TDHF flavor assignment missed HF states at k={ik}: {missing.tolist()}")
    return reordered_energies, reordered_vectors, reordered_occupied


def build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    n_spin: int | None = None,
    n_eta: int | None = None,
    n_band: int | None = None,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> RLGhBNTDHFOrbitals:
    """Convert canonical HFState/HFRunResult orbitals into RLG/hBN TDHF ordering.

    The generic core boundary diagonalizes the full HF Hamiltonian.  RLG/hBN
    TDHF still assigns particle-hole flavor metadata from the local orbital
    index, so this adapter only accepts canonical orbitals that can be resolved
    unambiguously into the RLG/hBN spin-valley blocks.  Flavor-mixed or
    degenerate gauges are rejected instead of fabricating flavor-tagged TDHF
    sectors.
    """

    state = _canonical_hf_state_from_input(canonical_hf)
    resolved_n_spin, resolved_n_eta, resolved_n_band = _infer_rlg_hbn_dimensions_from_canonical_state(
        state,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
    )
    offdiag_residual = _flavor_block_offdiag_residual(
        np.asarray(state.hamiltonian.total, dtype=np.complex128),
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
    )
    if offdiag_residual > float(flavor_resolution_tolerance):
        raise ValueError(
            "Canonical RLG/hBN TDHF adapter requires a spin-valley block-diagonal HF Hamiltonian; "
            f"max off-block element is {offdiag_residual:.6e}. Full flavor-mixed RLG/hBN TDHF needs a "
            "separate system-specific flavor/gauge treatment."
        )

    if isinstance(canonical_hf, ContractHFRunResult):
        canonical = canonical_tdhf_orbitals_from_hf_run_result(
            canonical_hf,
            occupation_policy=occupation_policy,
            projector_tolerance=projector_tolerance,
            degeneracy_tolerance=degeneracy_tolerance,
        )
    else:
        canonical = canonical_tdhf_orbitals_from_hf_state(
            state,
            occupation_policy=occupation_policy,
            projector_tolerance=projector_tolerance,
            degeneracy_tolerance=degeneracy_tolerance,
        )
    energies, eigenvectors, occupied_mask = _reorder_canonical_orbitals_to_rlg_hbn_flavor_order(
        canonical.energies,
        canonical.eigenvectors,
        canonical.occupied_mask,
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    return RLGhBNTDHFOrbitals(
        energies=energies,
        eigenvectors=eigenvectors,
        occupied_mask=occupied_mask,
        mu=float(canonical.mu),
        n_spin=resolved_n_spin,
        n_eta=resolved_n_eta,
        n_band=resolved_n_band,
    )


def _occupied_projector_from_tdhf_orbitals(orbitals: RLGhBNTDHFOrbitals, k_index: int) -> np.ndarray:
    vectors = np.asarray(orbitals.eigenvectors[:, :, int(k_index)], dtype=np.complex128)
    occupations = np.diag(np.asarray(orbitals.occupied_mask[:, int(k_index)], dtype=float))
    return vectors @ occupations @ vectors.conjugate().T


def _rlg_hbn_tdhf_orbital_parity_metrics(
    legacy: RLGhBNTDHFOrbitals,
    canonical: RLGhBNTDHFOrbitals,
) -> dict[str, float]:
    if legacy.energies.shape != canonical.energies.shape:
        raise ValueError(f"TDHF orbital energy shapes differ: {legacy.energies.shape} vs {canonical.energies.shape}")
    if legacy.eigenvectors.shape != canonical.eigenvectors.shape:
        raise ValueError(
            f"TDHF orbital eigenvector shapes differ: {legacy.eigenvectors.shape} vs {canonical.eigenvectors.shape}"
        )
    if legacy.occupied_mask.shape != canonical.occupied_mask.shape:
        raise ValueError(
            f"TDHF occupied-mask shapes differ: {legacy.occupied_mask.shape} vs {canonical.occupied_mask.shape}"
        )

    energy_residual = _max_abs(np.asarray(legacy.energies) - np.asarray(canonical.energies))
    occupied_mask_mismatches = float(np.count_nonzero(np.asarray(legacy.occupied_mask) != np.asarray(canonical.occupied_mask)))
    vector_overlap_residual = 0.0
    occupied_projector_residual = 0.0
    for ik in range(legacy.nk):
        overlap = legacy.eigenvectors[:, :, ik].conjugate().T @ canonical.eigenvectors[:, :, ik]
        diagonal_abs = np.abs(np.diag(overlap))
        offdiag = overlap.copy()
        index = np.arange(overlap.shape[0])
        offdiag[index, index] = 0.0
        vector_overlap_residual = max(
            vector_overlap_residual,
            _max_abs(diagonal_abs - 1.0),
            _max_abs(offdiag),
        )
        occupied_projector_residual = max(
            occupied_projector_residual,
            _max_abs(
                _occupied_projector_from_tdhf_orbitals(legacy, ik)
                - _occupied_projector_from_tdhf_orbitals(canonical, ik)
            ),
        )
    return {
        "energy_residual": float(energy_residual),
        "occupied_mask_mismatches": occupied_mask_mismatches,
        "vector_overlap_residual": float(vector_overlap_residual),
        "occupied_projector_residual": float(occupied_projector_residual),
    }


def _validate_rlg_hbn_tdhf_orbital_parity(
    legacy: RLGhBNTDHFOrbitals,
    canonical: RLGhBNTDHFOrbitals,
    *,
    tolerance: float,
) -> dict[str, float]:
    metrics = _rlg_hbn_tdhf_orbital_parity_metrics(legacy, canonical)
    failures = {
        key: value
        for key, value in metrics.items()
        if (key == "occupied_mask_mismatches" and value != 0.0)
        or (key != "occupied_mask_mismatches" and value > float(tolerance))
    }
    if failures:
        raise ValueError(
            "Canonical RLG/hBN TDHF orbitals do not match the existing system orbital path within "
            f"tolerance {float(tolerance):.6e}: {failures}"
        )
    return metrics


def validate_rlg_hbn_tdhf_canonical_orbital_parity(
    state: RLGhBNHartreeFockState,
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> dict[str, float]:
    """Validate canonical HFState/HFRunResult TDHF orbitals against RLG/hBN legacy orbitals."""

    legacy = build_rlg_hbn_tdhf_orbitals(state)
    canonical = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    return _validate_rlg_hbn_tdhf_orbital_parity(legacy, canonical, tolerance=tolerance)


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


def _assert_finite_q_shortcut_is_safe(
    run: RLGhBNHartreeFockRun,
    pairs: tuple[ParticleHolePair, ...],
) -> None:
    if int(run.state.active_valence_bands) != 0:
        raise ValueError("finite-q exchange shortcut requires conduction-only active space")
    if run.state.occupation_counts is None:
        raise ValueError("finite-q exchange shortcut requires saved occupation_counts metadata")
    counts = np.asarray(run.state.occupation_counts, dtype=int).reshape((int(run.state.n_spin), int(run.state.n_eta)), order="C")
    occupied_flavors = [(int(s), int(e)) for s in range(counts.shape[0]) for e in range(counts.shape[1]) if int(counts[s, e]) > 0]
    if len(occupied_flavors) != 1:
        raise ValueError(f"finite-q exchange shortcut requires exactly one occupied flavor, got {occupied_flavors}")
    for pair in pairs:
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q exchange shortcut pairs must carry SpinValleyFlavor metadata")
        if particle.spin == hole.spin and particle.valley == hole.valley:
            raise ValueError("finite-q exchange shortcut is not valid for intraflavor pairs")


def build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    beta: float = 1.0,
    structure_tolerance: float = 1.0e-6,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
) -> TDHFMatrices:
    """Build finite-q TDHF matrices for flavor-flip shortcut channels.

    This is the first finite-q production path needed for Fig. S45 spin and
    valley dispersions.  It intentionally implements only the conduction-only,
    fully polarized shortcut case where direct and B terms vanish and the A
    block contains the one-body term plus exchange.  Intra-flavor finite-q RPA
    requires the full Eq. D19 X/Y q/-q bookkeeping and is deliberately not
    hidden behind this shortcut helper.

    Periodic wrapping is handled by treating the loop variable as the *physical*
    Umklapp ``G``.  For a form factor whose target/source momenta have integer
    reciprocal wraps ``W_target`` and ``W_source``, the cached overlap shift is
    ``G + W_target - W_source``.  If the overlap block set has been augmented
    with extra closure keys, pass the original Coulomb-cutoff keys through
    ``physical_shifts`` so they are used only as cached form factors, not as
    extra physical G terms in the sum.
    """

    _reject_zero_literal_q0_fock_env()
    _assert_finite_q_shortcut_is_safe(run, tuple(pairs))
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))
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
    h_k = np.empty(n_pairs, dtype=int)
    p_plus_k = np.empty(n_pairs, dtype=int)
    wrap_plus = np.empty((n_pairs, 2), dtype=int)
    for index, pair in enumerate(ph_pairs):
        p_local[index], particle_k = orbitals.decode_global_index(pair.particle)
        h_local[index], hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, wrap = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if particle_k != expected_particle_k:
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        h_k[index] = hole_k
        p_plus_k[index] = particle_k
        wrap_plus[index] = wrap
        A[index, index] = orbitals.energies[p_local[index], particle_k] - orbitals.energies[h_local[index], hole_k]

    indices_by_hole_k = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    overlap_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.layer_overlaps.items()}
    kernel_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.fock_layer_coulomb.items()}
    missing_shifts: set[tuple[int, int]] = set()
    resolved_physical_shifts = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )

    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        hh_overlap = overlap_by_shift.get(g0)
        if hh_overlap is None:
            missing_shifts.add(g0)
            continue
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            p_t_k = int(p_plus_k[target_indices[0]])
            wrap_t = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_k]
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                p_s_k = int(p_plus_k[source_indices[0]])
                wrap_s = tuple(int(v) for v in wrap_plus[source_indices[0]])
                pp_shift = _add_shift(g0, _sub_shift(wrap_t, wrap_s))
                pp_overlap = overlap_by_shift.get(pp_shift)
                pp_kernel = kernel_by_shift.get(pp_shift)
                if pp_overlap is None or pp_kernel is None:
                    missing_shifts.add(pp_shift)
                    continue
                u_p_source = U[:, :, p_s_k]
                u_h_source = U[:, :, ks]
                kernel = np.asarray(pp_kernel[p_t_k, p_s_k], dtype=float)
                n_layer = int(pp_overlap.shape[0])
                pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hh = np.empty_like(pp)
                for layer in range(n_layer):
                    pp_full = u_p_target.conj().T @ pp_overlap[layer, :, p_t_k, :, p_s_k] @ u_p_source
                    hh_full = u_h_target.conj().T @ hh_overlap[layer, :, kt, :, ks] @ u_h_source
                    pp[layer] = pp_full[np.ix_(p_t, p_s)]
                    hh[layer] = hh_full[np.ix_(h_t, h_s)]
                A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    pp,
                    np.conj(hh),
                    optimize=True,
                )
    if missing_shifts and require_complete_umklapp:
        preview = sorted(missing_shifts)[:10]
        raise ValueError(
            f"Finite-q exchange assembly requires cached overlap shifts not present in this HF run: {preview}"
        )
    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)


def _assert_finite_q_intraflavor_pairs(pairs: tuple[ParticleHolePair, ...]) -> None:
    for pair in pairs:
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q intraflavor pairs must carry SpinValleyFlavor metadata")
        if particle.spin != hole.spin or particle.valley != hole.valley:
            raise ValueError("finite-q intraflavor full TDHF requires particle and hole in the same flavor")

def build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
    run: RLGhBNHartreeFockRun,
    orbitals: RLGhBNTDHFOrbitals,
    pairs: tuple[ParticleHolePair, ...],
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    beta: float = 1.0,
    structure_tolerance: float = 1.0e-6,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
    _build_partner: bool = True,
) -> TDHFMatrices:
    """Build full finite-q intraflavor TDHF matrices using paper Eq. D19.

    Pair labels are the X-sector operators ``d†_{k+q,p} d_{k,h}``.  The B block
    columns use the corresponding Y-sector partner ``d†_{k,h} d_{k-q,p}`` with
    the same base hole momentum ``k`` and local HF band labels.  At ``q=0`` this
    reduces exactly to the existing q=0 direct/exchange/B assembly.  For nonzero
    ``q``, the returned Liouvillian is the Eq. D19 partner block
    ``[[A(q), B(q)], [-B(-q)*, -A(-q)*]]``; correspondingly the reported B
    residual checks ``B(q)=B(-q)^T`` rather than the q=0-only ``B(q)=B(q)^T``.
    """

    _reject_zero_literal_q0_fock_env()
    ph_pairs = tuple(pairs)
    _assert_finite_q_intraflavor_pairs(ph_pairs)
    mesh_shape = _mesh_shape_from_k_grid_frac(run.basis_data.k_grid_frac)
    if isinstance(q_shift, RLGhBNTDHFMomentumShift):
        if tuple(q_shift.mesh_shape) != tuple(mesh_shape):
            raise ValueError(f"q_shift mesh {q_shift.mesh_shape} does not match basis mesh {mesh_shape}")
        shift = tuple(int(v) for v in q_shift.shift)
    else:
        shift = (int(q_shift[0]), int(q_shift[1]))

    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    if n_pairs == 0:
        L = assemble_tdhf_liouvillian(A, B)
        structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    p_local = np.empty(n_pairs, dtype=int)
    h_local = np.empty(n_pairs, dtype=int)
    h_k = np.empty(n_pairs, dtype=int)
    p_plus_k = np.empty(n_pairs, dtype=int)
    p_minus_k = np.empty(n_pairs, dtype=int)
    wrap_plus = np.empty((n_pairs, 2), dtype=int)
    wrap_minus = np.empty((n_pairs, 2), dtype=int)
    minus_shift = (-shift[0], -shift[1])
    for index, pair in enumerate(ph_pairs):
        p_local[index], particle_k = orbitals.decode_global_index(pair.particle)
        h_local[index], hole_k = orbitals.decode_global_index(pair.hole)
        expected_particle_k, plus_wrap = _shift_k_index_with_wrap(hole_k, shift, mesh_shape)
        if int(particle_k) != int(expected_particle_k):
            raise ValueError(
                "finite-q pair does not have particle momentum k+q: "
                f"pair particle_k={particle_k}, expected {expected_particle_k}, q_shift={shift}"
            )
        minus_k, minus_wrap = _shift_k_index_with_wrap(hole_k, minus_shift, mesh_shape)
        if orbitals.occupied_mask[p_local[index], minus_k]:
            raise ValueError(
                "finite-q intraflavor Eq. D19 requires the Y-sector particle at k-q to be unoccupied; "
                f"local={p_local[index]} k_minus={minus_k} is occupied"
            )
        h_k[index] = int(hole_k)
        p_plus_k[index] = int(particle_k)
        p_minus_k[index] = int(minus_k)
        wrap_plus[index] = plus_wrap
        wrap_minus[index] = minus_wrap
        A[index, index] = orbitals.energies[p_local[index], particle_k] - orbitals.energies[h_local[index], hole_k]

    indices_by_hole_k = tuple(np.nonzero(h_k == ik)[0] for ik in range(orbitals.nk))
    scale = float(beta) * float(run.basis_data.v0) / float(run.basis_data.nk)
    U = np.asarray(orbitals.eigenvectors, dtype=np.complex128)
    overlap_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.layer_overlaps.items()}
    kernel_by_shift = {tuple(int(v) for v in shift_key): value for shift_key, value in run.overlap_blocks.fock_layer_coulomb.items()}
    resolved_physical_shifts = (
        tuple((int(g[0]), int(g[1])) for g in physical_shifts)
        if physical_shifts is not None
        else tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    )
    missing_shifts: set[tuple[int, int]] = set()

    for physical_shift in resolved_physical_shifts:
        g0 = (int(physical_shift[0]), int(physical_shift[1]))
        hh_overlap = overlap_by_shift.get(g0)
        if hh_overlap is None:
            missing_shifts.add(g0)
            continue
        n_layer = int(hh_overlap.shape[0])

        # Direct A/B terms: physical transfer q + G.  The X form factor uses
        # k+q -> k, while the Y partner uses k -> k-q.
        plus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        minus_direct = np.zeros((n_layer, n_pairs), dtype=np.complex128)
        direct_kernel_by_k: dict[int, np.ndarray] = {}
        for ik, indices in enumerate(indices_by_hole_k):
            if indices.size == 0:
                continue
            plus_key = _add_shift(g0, tuple(int(v) for v in wrap_plus[indices[0]]))
            minus_key = _sub_shift(g0, tuple(int(v) for v in wrap_minus[indices[0]]))
            plus_overlap = overlap_by_shift.get(plus_key)
            minus_overlap = overlap_by_shift.get(minus_key)
            plus_kernel = kernel_by_shift.get(plus_key)
            if plus_overlap is None or plus_kernel is None:
                missing_shifts.add(plus_key)
                continue
            if minus_overlap is None:
                missing_shifts.add(minus_key)
                continue
            p_plus = int(p_plus_k[indices[0]])
            p_minus = int(p_minus_k[indices[0]])
            u_h = U[:, :, ik]
            u_p_plus = U[:, :, p_plus]
            u_p_minus = U[:, :, p_minus]
            p_idx = p_local[indices]
            h_idx = h_local[indices]
            direct_kernel_by_k[int(ik)] = np.asarray(plus_kernel[p_plus, ik], dtype=float)
            for layer in range(n_layer):
                plus_full = u_p_plus.conj().T @ plus_overlap[layer, :, p_plus, :, ik] @ u_h
                minus_full = u_h.conj().T @ minus_overlap[layer, :, ik, :, p_minus] @ u_p_minus
                plus_direct[layer, indices] = plus_full[p_idx, h_idx]
                minus_direct[layer, indices] = minus_full[h_idx, p_idx]
        for ik, row_indices in enumerate(indices_by_hole_k):
            if row_indices.size == 0 or int(ik) not in direct_kernel_by_k:
                continue
            kernel = direct_kernel_by_k[int(ik)]
            A[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(plus_direct),
                optimize=True,
            )
            B[np.ix_(row_indices, np.arange(n_pairs))] += scale * np.einsum(
                "lm,li,mj->ij",
                kernel,
                plus_direct[:, row_indices],
                np.conj(minus_direct),
                optimize=True,
            )

        # A-exchange: V[p(k+q), h'(k'), p'(k'+q), h(k)].
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            p_t_plus = int(p_plus_k[target_indices[0]])
            wrap_t_plus = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_plus]
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s_plus = int(p_plus_k[source_indices[0]])
                wrap_s_plus = tuple(int(v) for v in wrap_plus[source_indices[0]])
                pp_shift = _add_shift(g0, _sub_shift(wrap_t_plus, wrap_s_plus))
                pp_overlap = overlap_by_shift.get(pp_shift)
                pp_kernel = kernel_by_shift.get(pp_shift)
                if pp_overlap is None or pp_kernel is None:
                    missing_shifts.add(pp_shift)
                    continue
                u_p_source = U[:, :, p_s_plus]
                u_h_source = U[:, :, ks]
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                kernel = np.asarray(pp_kernel[p_t_plus, p_s_plus], dtype=float)
                pp = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hh = np.empty_like(pp)
                for layer in range(n_layer):
                    pp_full = u_p_target.conj().T @ pp_overlap[layer, :, p_t_plus, :, p_s_plus] @ u_p_source
                    hh_full = u_h_target.conj().T @ hh_overlap[layer, :, kt, :, ks] @ u_h_source
                    pp[layer] = pp_full[np.ix_(p_t, p_s)]
                    hh[layer] = hh_full[np.ix_(h_t, h_s)]
                A[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    pp,
                    np.conj(hh),
                    optimize=True,
                )

        # B-exchange: V[p(k+q), p'(k'-q), h'(k'), h(k)].
        for kt, target_indices in enumerate(indices_by_hole_k):
            if target_indices.size == 0:
                continue
            p_t_plus = int(p_plus_k[target_indices[0]])
            wrap_t_plus = tuple(int(v) for v in wrap_plus[target_indices[0]])
            u_p_target = U[:, :, p_t_plus]
            u_h_target = U[:, :, kt]
            p_t = p_local[target_indices]
            h_t = h_local[target_indices]
            left_shift = _add_shift(g0, wrap_t_plus)
            left_overlap = overlap_by_shift.get(left_shift)
            left_kernel = kernel_by_shift.get(left_shift)
            if left_overlap is None or left_kernel is None:
                missing_shifts.add(left_shift)
                continue
            for ks, source_indices in enumerate(indices_by_hole_k):
                if source_indices.size == 0:
                    continue
                p_s_minus = int(p_minus_k[source_indices[0]])
                wrap_s_minus = tuple(int(v) for v in wrap_minus[source_indices[0]])
                right_shift = _sub_shift(g0, wrap_s_minus)
                right_overlap = overlap_by_shift.get(right_shift)
                if right_overlap is None:
                    missing_shifts.add(right_shift)
                    continue
                u_h_source = U[:, :, ks]
                u_p_minus_source = U[:, :, p_s_minus]
                p_s = p_local[source_indices]
                h_s = h_local[source_indices]
                kernel = np.asarray(left_kernel[p_t_plus, ks], dtype=float)
                ph = np.empty((n_layer, target_indices.size, source_indices.size), dtype=np.complex128)
                hp = np.empty_like(ph)
                for layer in range(n_layer):
                    ph_full = u_p_target.conj().T @ left_overlap[layer, :, p_t_plus, :, ks] @ u_h_source
                    hp_full = u_h_target.conj().T @ right_overlap[layer, :, kt, :, p_s_minus] @ u_p_minus_source
                    ph[layer] = ph_full[np.ix_(p_t, h_s)]
                    hp[layer] = hp_full[np.ix_(h_t, p_s)]
                B[np.ix_(target_indices, source_indices)] -= scale * np.einsum(
                    "lm,lij,mij->ij",
                    kernel,
                    ph,
                    np.conj(hp),
                    optimize=True,
                )

    if missing_shifts and require_complete_umklapp:
        preview = sorted(missing_shifts)[:10]
        raise ValueError(
            f"Finite-q intraflavor assembly requires cached overlap shifts not present in this HF run: {preview}"
        )

    if _build_partner and shift != (0, 0):
        minus_q_pairs_all = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, minus_shift)
        minus_q_pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(minus_q_pairs_all, "intraflavor")
        if len(minus_q_pairs) != n_pairs:
            raise ValueError(
                "finite-q intraflavor +q and -q pair spaces have different sizes: "
                f"{n_pairs} vs {len(minus_q_pairs)}"
            )
        for plus_pair, minus_pair in zip(ph_pairs, minus_q_pairs, strict=True):
            plus_p_local, _plus_p_k = orbitals.decode_global_index(plus_pair.particle)
            plus_h_local, plus_h_k = orbitals.decode_global_index(plus_pair.hole)
            minus_p_local, _minus_p_k = orbitals.decode_global_index(minus_pair.particle)
            minus_h_local, minus_h_k = orbitals.decode_global_index(minus_pair.hole)
            if (plus_p_local, plus_h_local, plus_h_k) != (minus_p_local, minus_h_local, minus_h_k):
                raise ValueError("finite-q intraflavor +q/-q pair order mismatch")
        minus_matrices = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            minus_q_pairs,
            minus_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
            require_complete_umklapp=require_complete_umklapp,
            physical_shifts=physical_shifts,
            _build_partner=False,
        )
        L = np.block(
            [
                [A, B],
                [-np.conj(minus_matrices.B), -np.conj(minus_matrices.A)],
            ]
        )
        a_residual = max(
            float(np.max(np.abs(A - np.conj(A.T)))) if A.size else 0.0,
            float(np.max(np.abs(minus_matrices.A - np.conj(minus_matrices.A.T)))) if minus_matrices.A.size else 0.0,
        )
        b_residual = float(np.max(np.abs(B - minus_matrices.B.T))) if B.size else 0.0
        structure = TDHFStructureResiduals(
            a_hermitian=a_residual,
            b_symmetric=b_residual,
            particle_hole_symmetry=0.0,
            tolerance=float(structure_tolerance),
        )
        return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=structure_tolerance)
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)

def _filter_rlg_hbn_tdhf_finite_q_pairs(
    all_pairs: Sequence[ParticleHolePair],
    channel: str,
) -> tuple[ParticleHolePair, ...]:
    ph_pairs = tuple(all_pairs)
    groups: dict[str, list[int]] = {"intraflavor": [], "intervalley": [], "interspin": [], "inter_spin_valley": []}
    for index, pair in enumerate(ph_pairs):
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q pairs must carry SpinValleyFlavor metadata")
        same_spin = particle.spin == hole.spin
        same_valley = particle.valley == hole.valley
        if same_spin and same_valley:
            groups["intraflavor"].append(index)
        elif same_spin and not same_valley:
            groups["intervalley"].append(index)
        elif not same_spin and same_valley:
            groups["interspin"].append(index)
        elif not same_spin and not same_valley:
            groups["inter_spin_valley"].append(index)
    if channel not in groups:
        raise ValueError(f"finite-q channel must be one of {tuple(groups)}, got {channel!r}")
    return tuple(ph_pairs[index] for index in groups[str(channel)])

def _filter_rlg_hbn_tdhf_finite_q_shortcut_pairs(
    all_pairs: Sequence[ParticleHolePair],
    channel: str,
) -> tuple[ParticleHolePair, ...]:
    ph_pairs = tuple(all_pairs)
    if channel not in {"intervalley", "interspin", "inter_spin_valley"}:
        raise ValueError(f"finite-q shortcut channel must be a flavor-flip channel, got {channel!r}")
    groups: dict[str, list[int]] = {"intervalley": [], "interspin": [], "inter_spin_valley": []}
    for index, pair in enumerate(ph_pairs):
        particle = pair.particle_flavor
        hole = pair.hole_flavor
        if not isinstance(particle, SpinValleyFlavor) or not isinstance(hole, SpinValleyFlavor):
            raise ValueError("finite-q pairs must carry SpinValleyFlavor metadata")
        same_spin = particle.spin == hole.spin
        same_valley = particle.valley == hole.valley
        if same_spin and not same_valley:
            groups["intervalley"].append(index)
        elif not same_spin and same_valley:
            groups["interspin"].append(index)
        elif not same_spin and not same_valley:
            groups["inter_spin_valley"].append(index)
    return tuple(ph_pairs[index] for index in groups[str(channel)])

def build_rlg_hbn_tdhf_q_matrices(
    run: RLGhBNHartreeFockRun,
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    channel: FiniteQChannel,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    shortcut_exchange_only: bool = True,
) -> TDHFMatrices:
    """Build a dense finite-q TDHF matrix for one supported RLG/hBN channel."""

    channel_text = str(channel)
    support = _require_rlg_hbn_tdhf_finite_q_mode_supported(
        channel_text,
        shortcut_exchange_only=(False if channel_text == "intraflavor" else shortcut_exchange_only),
        canonical_boundary=False,
    )
    channel_key = support.channel
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
    pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(all_pairs, channel_key)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"finite-q TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering or raise the explicit Slurm-side limit."
        )
    if channel_key == "intraflavor":
        return build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            pairs,
            q_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
        )
    return build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        beta=beta,
        structure_tolerance=structure_tolerance,
    )

def build_rlg_hbn_tdhf_q_matrices_from_canonical_hf(
    run: RLGhBNHartreeFockRun,
    canonical_hf: ContractHFState | ContractHFRunResult,
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    channel: FiniteQChannel,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    shortcut_exchange_only: bool = True,
    validate_legacy_parity: bool = True,
    parity_tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
) -> TDHFMatrices:
    """Finite-q TDHF matrices using canonical HFState/HFRunResult orbitals.

    This opt-in bridge reuses the system-specific RLG/hBN finite-q wrapping,
    pair filtering, and layer-overlap assembly.  In the intraflavor channel it
    builds the full Eq. D19 A/B block; in flavor-flip channels it builds the
    guarded conduction-only exchange shortcut.
    """

    channel_text = str(channel)
    support = _require_rlg_hbn_tdhf_finite_q_mode_supported(
        channel_text,
        shortcut_exchange_only=(False if channel_text == "intraflavor" else shortcut_exchange_only),
        canonical_boundary=True,
    )
    channel_key = support.channel
    orbitals = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    if validate_legacy_parity:
        legacy = build_rlg_hbn_tdhf_orbitals(run.state)
        _validate_rlg_hbn_tdhf_orbital_parity(legacy, orbitals, tolerance=parity_tolerance)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
    pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(all_pairs, channel_key)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"finite-q TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering or raise the explicit Slurm-side limit."
        )
    if channel_key == "intraflavor":
        return build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            pairs,
            q_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
            require_complete_umklapp=require_complete_umklapp,
            physical_shifts=physical_shifts,
        )
    return build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        beta=beta,
        structure_tolerance=structure_tolerance,
        require_complete_umklapp=require_complete_umklapp,
        physical_shifts=physical_shifts,
    )


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



def build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf(
    run: RLGhBNHartreeFockRun,
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
    validate_legacy_parity: bool = True,
    parity_tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> TDHFMatrices:
    """Dense q=0 TDHF matrices using canonical HFState/HFRunResult orbitals.

    This is an opt-in bridge from the canonical core TDHF boundary to the
    existing RLG/hBN q=0 matrix assembly.  It reuses the system-specific
    layer-form-factor ``V_hf`` path and, by default, validates that the
    canonical orbitals are parity-equivalent to the legacy RLG/hBN orbital
    builder before assembling matrices.
    """

    orbitals = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    if validate_legacy_parity:
        legacy = build_rlg_hbn_tdhf_orbitals(run.state)
        _validate_rlg_hbn_tdhf_orbital_parity(legacy, orbitals, tolerance=parity_tolerance)
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
    "RLGhBNTDHFFiniteQSupport",
    "RLGhBNTDHFMomentumShift",
    "RLGhBNTDHFOrbitals",
    "build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs",
    "build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs",
    "build_rlg_hbn_tdhf_interaction",
    "build_rlg_hbn_tdhf_orbitals",
    "build_rlg_hbn_tdhf_orbitals_from_canonical_hf",
    "build_rlg_hbn_tdhf_q_matrices",
    "build_rlg_hbn_tdhf_q_matrices_from_canonical_hf",
    "build_rlg_hbn_tdhf_q_pairs",
    "build_rlg_hbn_tdhf_q0_matrices",
    "build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf",
    "build_rlg_hbn_tdhf_q0_matrices_from_pairs",
    "build_rlg_hbn_tdhf_q0_pairs",
    "load_rlg_hbn_tdhf_run_from_archive",
    "required_rlg_hbn_tdhf_finite_q_overlap_shifts",
    "required_rlg_hbn_tdhf_full_finite_q_overlap_shifts",
    "rlg_hbn_tdhf_finite_q_mode_support",
    "validate_rlg_hbn_tdhf_canonical_orbital_parity",
]
