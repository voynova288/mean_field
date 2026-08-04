"""Alavirad--Sau exact one-magnon oracle for the 1D Hubbard model.

The supplementary material of arXiv:1907.13633v1 uses the saturated
ferromagnet and the fixed-q one-magnon basis

    |psi_q> = sum_p phi[q,p] c^dagger[p+q,down] c[p,up] |FM>.

For the periodic one-band Hubbard Hamiltonian at one electron per site, the
exact Hamiltonian in that invariant Hilbert space is

    M_q[p,p'] = (epsilon[p+q] - epsilon[p] + U) delta[p,p'] - U/N.

This module deliberately constructs that exact matrix and the interspin TDHF
operator through separate code paths.  It is a narrow algebraic oracle for a
saturated SU(2) ferromagnet with B=0, not authority for a general K-IVC or
mixed particle-hole/hole-particle RPA sector.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from mean_field.core.hf import ParticleHolePair
from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    TDHFSignedQBlocks,
    TDHFTypedSector,
    build_standard_nambu_sewing,
    classify_tdhf_signed_q,
)

ALAVIRAD_SAU_ARXIV = "1907.13633v1"
ALAVIRAD_SAU_ONE_MAGNON_EQUATION = 13
ALAVIRAD_SAU_HUBBARD_HAMILTONIAN_EQUATION = 14
ALAVIRAD_SAU_EQ15_NOTE = (
    "Eq.15 prints two creation operators; this adapter uses the number-conserving "
    "spin-flip operator stated correctly in Eq.13"
)


@dataclass(frozen=True)
class AlaviradSauHubbard1DModel:
    site_count: int
    hopping: float
    interaction: float

    def __post_init__(self) -> None:
        if isinstance(self.site_count, bool) or int(self.site_count) != self.site_count:
            raise TypeError("site_count must be an integer")
        if self.site_count < 3:
            raise ValueError("site_count must be at least three")
        if not np.isfinite(self.hopping) or not np.isfinite(self.interaction):
            raise ValueError("Hubbard parameters must be finite")
        if self.interaction <= 0.0:
            raise ValueError("the benchmark requires repulsive U > 0")

    @property
    def momenta(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.site_count) / self.site_count

    @property
    def dispersion(self) -> np.ndarray:
        # Literal sign of Alavirad--Sau Eq. (14): +t(c_i^dagger c_i+1+h.c.).
        return 2.0 * self.hopping * np.cos(self.momenta)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "authority": ALAVIRAD_SAU_ARXIV,
                "dispersion": "epsilon_k=+2t*cos(k);literal_Eq14_sign",
                "interaction": float(self.interaction),
                "site_count": int(self.site_count),
                "hopping": float(self.hopping),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def interaction_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model_fingerprint": self.fingerprint,
                "onsite_vertex": "U/N in momentum space",
                "fourier_normalization": "c_k=N^-1/2 sum_j exp(-ikj)c_j",
                "interaction": float(self.interaction),
                "site_count": int(self.site_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def source_fingerprint(self) -> str:
        projector = saturated_ferromagnet_projector(self)
        payload = json.dumps(
            {
                "model_fingerprint": self.fingerprint,
                "boundary": "periodic",
                "electron_count": int(self.site_count),
                "filling": "one electron per site",
                "orbital_order": "up_k[0:N],down_k[0:N]",
                "projector_sha256": hashlib.sha256(projector.tobytes()).hexdigest(),
                "source": "all up-spin momentum orbitals occupied; down empty",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _raw_q_index(model: AlaviradSauHubbard1DModel, raw_q: int) -> int:
    if isinstance(raw_q, bool) or int(raw_q) != raw_q:
        raise TypeError("raw_q must be an integer")
    return int(raw_q)


def saturated_ferromagnet_projector(
    model: AlaviradSauHubbard1DModel,
) -> np.ndarray:
    """Reference projector in orbital order ``up_k`` then ``down_k``."""

    projector = np.zeros((2 * model.site_count, 2 * model.site_count))
    projector[np.arange(model.site_count), np.arange(model.site_count)] = 1.0
    projector.setflags(write=False)
    return projector


def saturated_ferromagnet_fock_matrix(
    model: AlaviradSauHubbard1DModel,
) -> np.ndarray:
    """HF Fock matrix for the half-filled fully up-polarized source."""

    epsilon = model.dispersion
    diagonal = np.concatenate([epsilon, epsilon + model.interaction])
    fock = np.diag(diagonal).astype(np.complex128)
    fock.setflags(write=False)
    return fock


def saturated_ferromagnet_stationarity_residual(
    model: AlaviradSauHubbard1DModel,
) -> float:
    """Compute, rather than assert, ``max|[F,P]|`` for the HF source."""

    fock = saturated_ferromagnet_fock_matrix(model)
    projector = saturated_ferromagnet_projector(model)
    return float(np.max(np.abs(fock @ projector - projector @ fock), initial=0.0))


def build_exact_one_magnon_hamiltonian(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
) -> np.ndarray:
    """Build the exact fixed-q Hamiltonian in the paper's ``|psi_q>`` basis."""

    q = _raw_q_index(model, raw_q)
    n = model.site_count
    epsilon = 2.0 * model.hopping * np.cos(
        2.0 * np.pi * np.arange(n) / n
    )
    kinetic = epsilon[(np.arange(n) + q) % n] - epsilon
    matrix = np.diag(kinetic + model.interaction).astype(np.complex128)
    matrix -= model.interaction / n
    matrix.setflags(write=False)
    return matrix


def _annihilate_bit(state: int, orbital: int) -> tuple[int, int] | None:
    if not (state >> orbital) & 1:
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state ^ (1 << orbital), sign


def _create_bit(state: int, orbital: int) -> tuple[int, int] | None:
    if (state >> orbital) & 1:
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state | (1 << orbital), sign


def build_exact_one_magnon_hamiltonian_bitstring(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
) -> np.ndarray:
    """Independent Eq.14 bitstring projection onto the Eq.13 momentum basis."""

    q = _raw_q_index(model, raw_q)
    n = model.site_count
    up_full = (1 << n) - 1
    basis = tuple(
        (up_full ^ (1 << hole_site)) | (1 << (n + down_site))
        for down_site in range(n)
        for hole_site in range(n)
    )
    index = {state: position for position, state in enumerate(basis)}
    hamiltonian = np.zeros((n * n, n * n), dtype=np.complex128)
    for column, state in enumerate(basis):
        double_occupancy = sum(
            ((state >> site) & 1) and ((state >> (n + site)) & 1)
            for site in range(n)
        )
        hamiltonian[column, column] += model.interaction * double_occupancy
        for spin_offset in (0, n):
            for site in range(n):
                neighbor = (site + 1) % n
                for target, source in ((site, neighbor), (neighbor, site)):
                    annihilated = _annihilate_bit(state, spin_offset + source)
                    if annihilated is None:
                        continue
                    intermediate, sign_a = annihilated
                    created = _create_bit(intermediate, spin_offset + target)
                    if created is None:
                        continue
                    final_state, sign_c = created
                    row = index.get(final_state)
                    if row is None:
                        raise RuntimeError("Hubbard hopping left the one-down-spin sector")
                    hamiltonian[row, column] += model.hopping * sign_a * sign_c

    # Build Eq.13 states by applying c^dagger_{p+q,down} c_{p,up}
    # to the real-space fully polarized determinant with explicit fermion signs.
    fm_state = up_full
    projected_basis = np.zeros((n * n, n), dtype=np.complex128)
    sites = np.arange(n)
    for momentum_index in range(n):
        momentum = 2.0 * np.pi * momentum_index / n
        target_momentum = 2.0 * np.pi * (momentum_index + q) / n
        for hole_site in sites:
            annihilated = _annihilate_bit(fm_state, int(hole_site))
            assert annihilated is not None
            intermediate, sign_a = annihilated
            hole_coefficient = np.exp(-1j * momentum * hole_site) / np.sqrt(n)
            for down_site in sites:
                created = _create_bit(intermediate, n + int(down_site))
                assert created is not None
                final_state, sign_c = created
                particle_coefficient = (
                    np.exp(1j * target_momentum * down_site) / np.sqrt(n)
                )
                projected_basis[index[final_state], momentum_index] += (
                    sign_a * sign_c * hole_coefficient * particle_coefficient
                )
    gram = np.conj(projected_basis.T) @ projected_basis
    if np.max(np.abs(gram - np.eye(n)), initial=0.0) > 1.0e-12:
        raise RuntimeError("Eq.13 bitstring basis is not orthonormal")
    projected = np.conj(projected_basis.T) @ hamiltonian @ projected_basis
    if np.max(np.abs(projected - np.conj(projected.T)), initial=0.0) > 1.0e-12:
        raise RuntimeError("projected exact one-magnon Hamiltonian is not Hermitian")
    projected.setflags(write=False)
    return projected


def apply_saturated_ferromagnet_interspin_tdhf_action(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Apply interspin TDHF without calling the exact one-magnon builder.

    The HF down-spin particle carries the opposite-spin Hartree shift ``+U``;
    the spin-flip ladder kernel contributes ``-(U/N) sum_p phi_p``.
    """

    q = _raw_q_index(model, raw_q)
    vector = np.asarray(amplitudes, dtype=np.complex128)
    if vector.shape != (model.site_count,):
        raise ValueError(
            f"amplitudes must have shape {(model.site_count,)}, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("amplitudes must be finite")
    epsilon = model.dispersion
    particle = epsilon[(np.arange(model.site_count) + q) % model.site_count]
    hole = epsilon
    hf_gap = particle + model.interaction - hole
    return hf_gap * vector - (model.interaction / model.site_count) * np.sum(vector)


def _tdhf_action_matrix(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
) -> np.ndarray:
    identity = np.eye(model.site_count, dtype=np.complex128)
    return np.column_stack(
        [
            apply_saturated_ferromagnet_interspin_tdhf_action(
                model, raw_q, identity[:, column]
            )
            for column in range(model.site_count)
        ]
    )


def _pairs(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
) -> tuple[ParticleHolePair, ...]:
    n = model.site_count
    return tuple(
        ParticleHolePair(
            particle=n + ((momentum + raw_q) % n),
            hole=momentum,
            particle_momentum=(int((momentum + raw_q) % n),),
            hole_momentum=(momentum,),
            particle_flavor="down",
            hole_flavor="up",
        )
        for momentum in range(n)
    )


def _typed_sector(
    model: AlaviradSauHubbard1DModel,
    raw_q: int,
) -> TDHFTypedSector:
    n = model.site_count
    minus_q = -raw_q
    plus_pairs = _pairs(model, raw_q)
    minus_pairs = _pairs(model, minus_q)
    a_plus = _tdhf_action_matrix(model, raw_q)
    a_minus = _tdhf_action_matrix(model, minus_q)
    zeros = np.zeros((n, n), dtype=np.complex128)
    blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=a_plus,
        B_plus_minus=zeros,
        A_minus=a_minus,
        B_minus_plus=zeros.copy(),
    )
    sewing = build_standard_nambu_sewing(
        plus_pairs,
        minus_pairs,
        source_fingerprint=model.source_fingerprint,
        construction="alavirad_sau_hubbard_spin_flip_block_swap_v1",
    )
    q_kind = classify_tdhf_signed_q(
        plus_raw=(raw_q, 0),
        minus_raw=(minus_q, 0),
        plus_canonical=(raw_q % n, 0),
        minus_canonical=(minus_q % n, 0),
        provenance=(
            f"periodic_1d_hubbard_N={n};one_magnon_eq13;"
            "independent_raw_signed_actions"
        ),
    )
    scope = "alavirad_sau_exact_one_magnon_interspin_tdhf_oracle_v1"
    if isinstance(q_kind, TDHFGenericSignedQ):
        return TDHFGenericSignedQSector(
            q=q_kind,
            blocks=blocks,
            sewing=sewing,
            source_fingerprint=model.source_fingerprint,
            interaction_fingerprint=model.interaction_fingerprint,
            response_scope=scope,
            static_hessian_authority="scalar_hessian",
        )
    if not isinstance(q_kind, TDHFSelfConjugateQ):
        raise TypeError("unexpected signed-q classification")
    if not np.array_equal(a_plus, a_minus):
        raise ValueError(
            "self-conjugate Hubbard aliases disagree; canonical branch is not unique"
        )
    raw_alias = q_kind.plus_raw != q_kind.minus_raw
    return TDHFSelfConjugateQSector(
        q=q_kind,
        canonical_pairs=plus_pairs,
        A=a_plus,
        B=zeros,
        source_fingerprint=model.source_fingerprint,
        interaction_fingerprint=model.interaction_fingerprint,
        response_scope=scope,
        static_hessian_authority="scalar_hessian",
        canonical_sewing_provenance=(
            "literal_q0_spin_flip_basis_v1"
            if not raw_alias
            else "independent_exact_boundary_actions_byte_identical_v1"
        ),
        raw_signed_diagnostic_blocks=blocks if raw_alias else None,
        raw_signed_diagnostic_sewing=sewing if raw_alias else None,
    )


@dataclass(frozen=True)
class AlaviradSauHubbardTDHFProvider:
    model: AlaviradSauHubbard1DModel

    def build_tdhf_sector(self, config: object, **kwargs: Any) -> TDHFTypedSector:
        if kwargs:
            raise TypeError(f"unexpected Hubbard TDHF kwargs: {sorted(kwargs)}")
        channel = getattr(config, "channel", None)
        if channel not in ("interspin", "spin_flip", "one_magnon"):
            raise ValueError("Hubbard oracle channel must be interspin/spin_flip")
        q_sector = getattr(config, "q_sector", None)
        if (
            not isinstance(q_sector, tuple)
            or len(q_sector) != 2
            or q_sector[1] != 0
        ):
            raise TypeError("Hubbard oracle q_sector must be the tuple (raw_q, 0)")
        raw_q = _raw_q_index(self.model, q_sector[0])
        return _typed_sector(self.model, raw_q)


@dataclass(frozen=True)
class AlaviradSauSpinStiffness:
    q_indices: np.ndarray
    physical_momenta: np.ndarray
    lowest_energies: np.ndarray
    stiffness: float
    fit_residual: float


def fit_exact_one_magnon_spin_stiffness(
    model: AlaviradSauHubbard1DModel,
    *,
    max_positive_q_index: int = 2,
) -> AlaviradSauSpinStiffness:
    """Fit the exact lowest branch to ``E(q)=rho_s q^2`` through the origin."""

    if max_positive_q_index < 1 or max_positive_q_index >= model.site_count / 2:
        raise ValueError("max_positive_q_index must lie below the Nyquist index")
    indices = np.arange(1, max_positive_q_index + 1, dtype=np.int64)
    momenta = 2.0 * np.pi * indices / model.site_count
    energies = np.asarray(
        [
            np.linalg.eigvalsh(build_exact_one_magnon_hamiltonian(model, int(q)))[0]
            for q in indices
        ]
    )
    q2 = momenta**2
    stiffness = float(np.dot(q2, energies) / np.dot(q2, q2))
    residual = float(np.max(np.abs(energies - stiffness * q2), initial=0.0))
    return AlaviradSauSpinStiffness(
        q_indices=indices,
        physical_momenta=momenta,
        lowest_energies=energies,
        stiffness=stiffness,
        fit_residual=residual,
    )


def spin_lowering_nambu_generator(
    model: AlaviradSauHubbard1DModel,
) -> np.ndarray:
    """Return the normalized q=0 total-spin-lowering tangent in ``[X,Y]``."""

    generator = np.zeros(2 * model.site_count, dtype=np.complex128)
    generator[: model.site_count] = 1.0 / np.sqrt(model.site_count)
    generator.setflags(write=False)
    return generator


__all__ = [
    "ALAVIRAD_SAU_ARXIV",
    "ALAVIRAD_SAU_EQ15_NOTE",
    "ALAVIRAD_SAU_HUBBARD_HAMILTONIAN_EQUATION",
    "ALAVIRAD_SAU_ONE_MAGNON_EQUATION",
    "AlaviradSauHubbard1DModel",
    "AlaviradSauHubbardTDHFProvider",
    "AlaviradSauSpinStiffness",
    "apply_saturated_ferromagnet_interspin_tdhf_action",
    "build_exact_one_magnon_hamiltonian",
    "build_exact_one_magnon_hamiltonian_bitstring",
    "fit_exact_one_magnon_spin_stiffness",
    "saturated_ferromagnet_fock_matrix",
    "saturated_ferromagnet_projector",
    "saturated_ferromagnet_stationarity_residual",
    "spin_lowering_nambu_generator",
]
