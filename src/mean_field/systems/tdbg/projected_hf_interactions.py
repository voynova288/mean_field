from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from ...core.hf import HFOverlapBlockSet
from .lattice import TDBGLattice
from .projected_hf_config import SPIN_LABELS, TDBG_LOCAL_LABELS, validate_tdbg_interaction_settings
from .projected_hf_geometry import _tdbg_projected_wavefunction_basis, _tdbg_total_overlap_from_bases, tdbg_moire_area_nm2
from .projected_hf_state import (
    TDBGProjectedHFData,
    _fock_density_for_policy,
    _hartree_density_for_policy,
    _reference_subtracted_tdbg_density,
    _stored_to_conventional,
)

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

__all__ = [
    "TDBGProjectedHFInteractionBuilder",
    "_local_lambda",
    "_split_intersite_overlap_blocks",
    "_stored_inner_ev",
    "build_tdbg_interaction_builder",
    "build_tdbg_interaction_components",
    "build_tdbg_onsite_hamiltonian",
    "build_tdbg_total_overlap_blocks",
    "graphene_area_over_moire_area",
    "tdbg_energy_components",
]
