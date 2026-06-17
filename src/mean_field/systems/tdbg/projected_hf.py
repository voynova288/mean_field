from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from ...core.hf import (
    HFOverlapBlockSet,
    build_projected_target_hamiltonian,
    diagonal_overlap_blocks,
)
from .lattice import TDBGLattice, build_moire_k_grid
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
from .projected_hf_data import (
    _projected_onebody_and_wavefunctions,
    _projected_orbital_g_matrix,
    build_tdbg_projected_hf_data,
)
from .projected_hf_geometry import (
    _TDBGQSiteEmbedding,
    _shift_table,
    _tdbg_core_order_permutation,
    _tdbg_projected_wavefunction_basis,
    _tdbg_q_site_embedding,
    _tdbg_total_overlap_between,
    _tdbg_total_overlap_from_bases,
    tdbg_band_window_indices,
    tdbg_moire_area_nm2,
)
from .projected_hf_interactions import (
    TDBGProjectedHFInteractionBuilder,
    _local_lambda,
    _split_intersite_overlap_blocks,
    _stored_inner_ev,
    build_tdbg_interaction_builder,
    build_tdbg_interaction_components,
    build_tdbg_onsite_hamiltonian,
    build_tdbg_total_overlap_blocks,
    graphene_area_over_moire_area,
    tdbg_energy_components,
)
from .projected_hf_state import (
    TDBGProjectedHFData,
    TDBGProjectedHFDensityBuilder,
    TDBGProjectedHFInitializer,
    TDBGProjectedHFResult,
    TDBGProjectedHFState,
    TDBGProjectedHFTargetData,
    TDBGStateLabel,
    _active_filling_indices,
    _conventional_projector_to_stored,
    _first_conduction_indices,
    _fock_density_for_policy,
    _hartree_density_for_policy,
    _numeric_order_parameters,
    _reference_projector,
    _reference_subtracted_tdbg_density,
    _stored_to_conventional,
    initialize_tdbg_density,
    initialize_tdbg_nu2_density,
    tdbg_density_from_hamiltonian,
    tdbg_order_parameters,
)
from .projected_hf_reports import (
    liu2022_default_projected_hf_config,
    liu2022_projected_hf_metadata,
    tdbg_hf_grid_band_summary,
)
from .projected_hf_run import (
    build_tdbg_projected_hf_kernel,
    build_tdbg_projected_hf_problem,
    build_tdbg_projected_hf_state,
    run_tdbg_projected_hf,
)

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
