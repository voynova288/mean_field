"""Q=0 unrestricted HF adapter for the TPT Wannier--Keldysh model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    InteractionEnergyResult,
)
from mean_field.core.hf.problem import HartreeFockKernel, HartreeFockProblem
from mean_field.core.hf.real_space_density_density import (
    PeriodicDensityDensityKernel,
)

Array = np.ndarray


def _stored_hamiltonian_to_ket(stored: Array) -> Array:
    values = np.asarray(stored, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("stored Hamiltonian must have shape (nb,nb,nk)")
    return np.ascontiguousarray(values.transpose(2, 0, 1))


def _ket_projector_to_stored(projector: Array) -> Array:
    values = np.asarray(projector, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("ket projector must have shape (nk,nb,nb)")
    return np.ascontiguousarray(values.transpose(0, 2, 1).transpose(1, 2, 0))


def _stored_projector_to_ket(stored: Array) -> Array:
    values = np.asarray(stored, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("stored projector must have shape (nb,nb,nk)")
    return np.ascontiguousarray(values.swapaxes(0, 1).transpose(2, 0, 1))


@dataclass(frozen=True)
class _FixedRankOccupation:
    projector_ket: Array
    energies: Array
    chemical_potential: float
    minimum_direct_gap: float
    indirect_gap: float


def _fixed_rank_occupations(
    hamiltonian_ket: Array,
    *,
    occupied_per_k: int,
    direct_gap_tolerance: float,
    eigensolver_workers: int,
) -> _FixedRankOccupation:
    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[1] != hamiltonian.shape[2]:
        raise ValueError("hamiltonian_ket must have shape (nk,nb,nb)")
    nk, nb, _ = hamiltonian.shape
    nocc = int(occupied_per_k)
    if not (0 < nocc < nb):
        raise ValueError("occupied_per_k must lie inside the represented basis")
    tolerance = float(direct_gap_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("direct_gap_tolerance must be finite and positive")
    energies = np.empty((nb, nk), dtype=float)
    projector = np.empty_like(hamiltonian)

    def solve(index: int) -> tuple[int, Array, Array]:
        values, vectors = np.linalg.eigh(hamiltonian[index])
        occupied = vectors[:, :nocc]
        return index, values, occupied @ occupied.conj().T

    if eigensolver_workers == 1:
        iterator = map(solve, range(nk))
        for index, values, block in iterator:
            energies[:, index] = values
            projector[index] = block
    else:
        with ThreadPoolExecutor(
            max_workers=min(int(eigensolver_workers), nk)
        ) as executor:
            for index, values, block in executor.map(solve, range(nk)):
                energies[:, index] = values
                projector[index] = block
    direct_gaps = energies[nocc] - energies[nocc - 1]
    minimum_direct_gap = float(np.min(direct_gaps))
    if minimum_direct_gap <= tolerance:
        raise RuntimeError(
            "closed fixed-rank occupation boundary: "
            f"minimum direct gap={minimum_direct_gap:.16e} eV"
        )
    highest_occupied = float(np.max(energies[nocc - 1]))
    lowest_empty = float(np.min(energies[nocc]))
    return _FixedRankOccupation(
        projector_ket=projector,
        energies=energies,
        chemical_potential=0.5 * (highest_occupied + lowest_empty),
        minimum_direct_gap=minimum_direct_gap,
        indirect_gap=lowest_empty - highest_occupied,
    )


@dataclass(frozen=True)
class TPTWannierQ0HFInputs:
    """Immutable one-body/reference/interaction contract for one Q=0 run."""

    h0_abk: Array
    reference_projector_stored: Array
    parent_energies_ev: Array
    parent_chemical_potential_ev: float
    parent_fermi_gap_ev: float
    interaction_kernel: PeriodicDensityDensityKernel
    total_occupied_states: int
    mesh_shape: tuple[int, int]

    def __post_init__(self) -> None:
        h0 = np.asarray(self.h0_abk, dtype=np.complex128)
        reference = np.asarray(self.reference_projector_stored, dtype=np.complex128)
        energies = np.asarray(self.parent_energies_ev, dtype=float)
        n1, n2 = int(self.mesh_shape[0]), int(self.mesh_shape[1])
        if h0.ndim != 3 or h0.shape[0] != h0.shape[1]:
            raise ValueError("h0_abk must have shape (nb,nb,nk)")
        nb, _, nk = h0.shape
        if reference.shape != h0.shape or energies.shape != (nb, nk):
            raise ValueError("reference/energy shapes must match h0")
        if nk != n1 * n2 or self.interaction_kernel.mesh_shape != (n1, n2):
            raise ValueError("mesh shape is inconsistent across the TPT HF inputs")
        if self.interaction_kernel.n_basis != nb:
            raise ValueError("interaction basis dimension does not match h0")
        if not self.interaction_kernel.require_neutral_density:
            raise ValueError("TPT Q=0 inputs require fail-closed neutral-background policy")
        if type(self.total_occupied_states) is not int or not (
            0 < self.total_occupied_states < nb * nk
        ):
            raise ValueError("total_occupied_states is outside the represented space")
        if not np.all(np.isfinite(h0)) or not np.all(np.isfinite(reference)):
            raise ValueError("h0 and reference projector must be finite")
        h0_scale = max(1.0, float(np.max(np.abs(h0), initial=0.0)))
        h0_residual = float(
            np.max(np.abs(h0 - h0.conj().swapaxes(0, 1)), initial=0.0)
        )
        if h0_residual > 2.0e-11 * h0_scale:
            raise ValueError("h0_abk is not Hermitian")
        reference_ket = _stored_projector_to_ket(reference)
        reference_hermiticity = float(
            np.max(
                np.abs(reference_ket - reference_ket.conj().transpose(0, 2, 1)),
                initial=0.0,
            )
        )
        reference_idempotency = float(
            np.max(
                np.abs(reference_ket @ reference_ket - reference_ket), initial=0.0
            )
        )
        trace_by_k = np.trace(reference_ket, axis1=1, axis2=2).real
        expected_per_k = self.total_occupied_states / nk
        if (
            reference_hermiticity > 2.0e-11
            or reference_idempotency > 2.0e-10
            or np.max(np.abs(trace_by_k - expected_per_k), initial=0.0) > 2.0e-9
        ):
            raise ValueError("reference projector failed Hermiticity/rank/idempotency gates")
        gap = float(self.parent_fermi_gap_ev)
        mu = float(self.parent_chemical_potential_ev)
        if not np.isfinite(gap) or gap <= 0.0 or not np.isfinite(mu):
            raise ValueError("parent reference must have a finite positive global gap")
        h0 = np.ascontiguousarray(h0)
        reference = np.ascontiguousarray(reference)
        energies = np.ascontiguousarray(energies)
        h0.setflags(write=False)
        reference.setflags(write=False)
        energies.setflags(write=False)
        object.__setattr__(self, "h0_abk", h0)
        object.__setattr__(self, "reference_projector_stored", reference)
        object.__setattr__(self, "parent_energies_ev", energies)
        object.__setattr__(self, "mesh_shape", (n1, n2))

    @property
    def nb(self) -> int:
        return int(self.h0_abk.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0_abk.shape[2])

    @classmethod
    def from_h0(
        cls,
        h0_abk: Array,
        *,
        interaction_kernel: PeriodicDensityDensityKernel,
        occupied_per_k: int,
        mesh_shape: tuple[int, int],
        fermi_degeneracy_tolerance_ev: float = 1.0e-10,
        eigensolver_workers: int = 1,
    ) -> "TPTWannierQ0HFInputs":
        h0 = np.asarray(h0_abk, dtype=np.complex128)
        if h0.ndim != 3 or h0.shape[0] != h0.shape[1]:
            raise ValueError("h0_abk must have shape (nb,nb,nk)")
        nb, _, nk = h0.shape
        nocc = int(occupied_per_k)
        if not (0 < nocc < nb):
            raise ValueError("occupied_per_k must lie inside the represented basis")
        occupied = _fixed_rank_occupations(
            _stored_hamiltonian_to_ket(h0),
            occupied_per_k=nocc,
            direct_gap_tolerance=float(fermi_degeneracy_tolerance_ev),
            eigensolver_workers=eigensolver_workers,
        )
        trace_by_k = np.trace(occupied.projector_ket, axis1=1, axis2=2).real
        if np.max(np.abs(trace_by_k - nocc), initial=0.0) > 2.0e-9:
            raise ValueError(
                "global parent occupation is not the declared fixed-rank insulating reference"
            )
        return cls(
            h0_abk=h0,
            reference_projector_stored=_ket_projector_to_stored(
                occupied.projector_ket
            ),
            parent_energies_ev=occupied.energies,
            parent_chemical_potential_ev=occupied.chemical_potential,
            parent_fermi_gap_ev=occupied.indirect_gap,
            interaction_kernel=interaction_kernel,
            total_occupied_states=nocc * nk,
            mesh_shape=mesh_shape,
        )


@dataclass
class TPTWannierQ0HFState:
    h0: Array
    density: Array
    hamiltonian: Array
    energies: Array
    mu: float
    precision: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def create(
        cls,
        inputs: TPTWannierQ0HFInputs,
        *,
        precision: float,
    ) -> "TPTWannierQ0HFState":
        tolerance = float(precision)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("precision must be finite and positive")
        return cls(
            h0=np.asarray(inputs.h0_abk, dtype=np.complex128).copy(),
            density=np.zeros_like(inputs.h0_abk),
            hamiltonian=np.asarray(inputs.h0_abk, dtype=np.complex128).copy(),
            energies=np.asarray(inputs.parent_energies_ev, dtype=float).copy(),
            mu=float(inputs.parent_chemical_potential_ev),
            precision=tolerance,
        )


def build_projector_seed_from_source(
    inputs: TPTWannierQ0HFInputs,
    source_abk: Array,
    *,
    fermi_degeneracy_tolerance_ev: float = 1.0e-10,
    eigensolver_workers: int = 1,
) -> Array:
    """Build a fixed-number projector seed from ``H0+source`` and remove source."""

    source = np.asarray(source_abk, dtype=np.complex128)
    if source.shape != inputs.h0_abk.shape or not np.all(np.isfinite(source)):
        raise ValueError("source_abk must be finite and match h0")
    source_scale = max(1.0, float(np.max(np.abs(source), initial=0.0)))
    source_residual = float(
        np.max(np.abs(source - source.conj().swapaxes(0, 1)), initial=0.0)
    )
    if source_residual > 2.0e-11 * source_scale:
        raise ValueError("source_abk must be Hermitian")
    occupied = _fixed_rank_occupations(
        _stored_hamiltonian_to_ket(inputs.h0_abk + source),
        occupied_per_k=inputs.total_occupied_states // inputs.nk,
        direct_gap_tolerance=float(fermi_degeneracy_tolerance_ev),
        eigensolver_workers=eigensolver_workers,
    )
    seed = _ket_projector_to_stored(occupied.projector_ket)
    seed -= inputs.reference_projector_stored
    inputs.interaction_kernel.average_basis_charges(seed)
    return np.ascontiguousarray(seed)


def build_tpt_wannier_q0_hf_problem(
    inputs: TPTWannierQ0HFInputs,
    *,
    initial_densities: Mapping[str, Array] | None = None,
    eigensolver_workers: int = 1,
    fft_workers: int = 1,
    fermi_degeneracy_tolerance_ev: float = 1.0e-10,
    final_commutator_tolerance: float = 5.0e-8,
    final_projector_tolerance: float = 5.0e-8,
) -> HartreeFockProblem:
    """Connect the TPT physical model to the reusable ODA SCF engine."""

    initial: dict[str, Array] = {"reference": np.zeros_like(inputs.h0_abk)}
    if initial_densities is not None:
        for name, density in initial_densities.items():
            if type(name) is not str or not name or name == "preserve":
                raise ValueError("initial-density names must be nonempty and not 'preserve'")
            values = np.asarray(density, dtype=np.complex128)
            if values.shape != inputs.h0_abk.shape:
                raise ValueError(f"initial density {name!r} has the wrong shape")
            inputs.interaction_kernel.validate_density(values)
            inputs.interaction_kernel.average_basis_charges(values)
            initial[name] = np.ascontiguousarray(values.copy())
    initial = dict(MappingProxyType(initial))

    fermi_tolerance = float(fermi_degeneracy_tolerance_ev)
    commutator_tolerance = float(final_commutator_tolerance)
    projector_tolerance = float(final_projector_tolerance)
    if min(fermi_tolerance, commutator_tolerance, projector_tolerance) <= 0.0:
        raise ValueError("HF acceptance tolerances must be positive")

    def initializer(
        state: TPTWannierQ0HFState, *, init_mode: str, seed: int
    ) -> None:
        del seed
        if init_mode == "preserve":
            inputs.interaction_kernel.validate_density(state.density)
            inputs.interaction_kernel.average_basis_charges(state.density)
        elif init_mode in initial:
            state.density[:, :, :] = initial[init_mode]
        else:
            raise ValueError(f"unknown TPT Q=0 init_mode={init_mode!r}")
        state.hamiltonian[:, :, :] = state.h0
        state.energies[:, :] = inputs.parent_energies_ev
        state.mu = float(inputs.parent_chemical_potential_ev)

    def evaluate_interaction(stored_density: Array):
        return inputs.interaction_kernel.apply(
            stored_density, fft_workers=fft_workers
        )

    def interaction_builder(stored_density: Array) -> Array:
        return evaluate_interaction(stored_density).interaction_h

    def interaction_energy_builder(
        stored_density: Array, stored_h0: Array
    ) -> InteractionEnergyResult:
        action = evaluate_interaction(stored_density)
        one_body = float(
            np.einsum(
                "abk,abk->", stored_density, stored_h0, optimize=True
            ).real
            / inputs.nk
        )
        return InteractionEnergyResult(
            interaction_h=action.interaction_h,
            energy=float(one_body + action.interaction_energy),
        )

    def energy_functional(
        stored_interaction: Array,
        stored_h0: Array,
        stored_density: Array,
    ) -> float:
        one_body = np.einsum(
            "abk,abk->", stored_density, stored_h0, optimize=True
        )
        interaction = 0.5 * np.einsum(
            "abk,abk->", stored_density, stored_interaction, optimize=True
        )
        total = (one_body + interaction) / inputs.nk
        if abs(total.imag) > 2.0e-10 * max(1.0, abs(complex(total))):
            raise RuntimeError("HF energy functional is not real")
        return float(total.real)

    def density_builder(stored_hamiltonian: Array) -> DensityUpdateResult:
        occupied = _fixed_rank_occupations(
            _stored_hamiltonian_to_ket(stored_hamiltonian),
            occupied_per_k=inputs.total_occupied_states // inputs.nk,
            direct_gap_tolerance=fermi_tolerance,
            eigensolver_workers=eigensolver_workers,
        )
        stored_projector = _ket_projector_to_stored(occupied.projector_ket)
        stored_delta = stored_projector - inputs.reference_projector_stored
        neutrality = float(
            np.sum(inputs.interaction_kernel.average_basis_charges(stored_delta))
        )
        projector_residual = float(
            np.max(
                np.abs(
                    occupied.projector_ket @ occupied.projector_ket
                    - occupied.projector_ket
                ),
                initial=0.0,
            )
        )
        return DensityUpdateResult(
            density=stored_delta,
            energies=occupied.energies,
            mu=occupied.chemical_potential,
            observables={
                "minimum_direct_gap_ev": occupied.minimum_direct_gap,
                "indirect_gap_ev": occupied.indirect_gap,
                "neutrality_residual": neutrality,
                "projector_idempotency_residual": projector_residual,
            },
        )

    density_metric_scale = float(np.sqrt(inputs.total_occupied_states))

    def convergence_metric(updated: Array, previous: Array) -> float:
        return float(np.linalg.norm(np.asarray(updated) - np.asarray(previous))) / (
            density_metric_scale
        )

    def final_acceptance(
        state: TPTWannierQ0HFState,
        final_density_update: DensityUpdateResult,
        final_norm: float,
    ) -> FinalAcceptanceResult:
        del final_density_update
        stored_projector = state.density + inputs.reference_projector_stored
        projector_ket = _stored_projector_to_ket(stored_projector)
        hamiltonian_ket = _stored_hamiltonian_to_ket(state.hamiltonian)
        projector_residual = float(
            np.max(
                np.abs(projector_ket @ projector_ket - projector_ket), initial=0.0
            )
        )
        particle_by_k = np.trace(projector_ket, axis1=1, axis2=2).real
        expected_per_k = inputs.total_occupied_states // inputs.nk
        particle_residual = float(
            np.max(np.abs(particle_by_k - expected_per_k), initial=0.0)
        )
        commutator = hamiltonian_ket @ projector_ket - projector_ket @ hamiltonian_ket
        commutator_residual = float(np.max(np.abs(commutator), initial=0.0))
        neutrality_residual = abs(
            float(
                np.sum(
                    inputs.interaction_kernel.average_basis_charges(state.density)
                )
            )
        )
        diagnostics = {
            "final_commutator_residual": commutator_residual,
            "final_projector_idempotency_residual": projector_residual,
            "final_particle_number_residual": particle_residual,
            "final_neutrality_residual": neutrality_residual,
        }
        accepted = (
            final_norm <= state.precision
            and commutator_residual <= commutator_tolerance
            and projector_residual <= projector_tolerance
            and particle_residual <= 2.0e-8
            and neutrality_residual <= inputs.interaction_kernel.neutrality_tolerance
        )
        if accepted:
            return FinalAcceptanceResult(
                accepted=True, reason="converged", diagnostics=diagnostics
            )
        if commutator_residual > commutator_tolerance:
            reason = "final_commutator"
        elif projector_residual > projector_tolerance:
            reason = "final_projector"
        elif particle_residual > 2.0e-8:
            reason = "final_particle_number"
        elif neutrality_residual > inputs.interaction_kernel.neutrality_tolerance:
            reason = "final_neutrality"
        else:
            reason = "final_raw_residual"
        return FinalAcceptanceResult(
            accepted=False, reason=reason, diagnostics=diagnostics
        )

    return HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            interaction_energy_builder=interaction_energy_builder,
            oda_delta_interaction_builder=interaction_builder,
            final_acceptance=final_acceptance,
            convergence_metric=convergence_metric,
            convergence_rule="raw",
        ),
    )


__all__ = [
    "TPTWannierQ0HFInputs",
    "TPTWannierQ0HFState",
    "build_projector_seed_from_source",
    "build_tpt_wannier_q0_hf_problem",
]
