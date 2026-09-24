"""Nonlinear unrestricted TiSe2 SCF on the local WS-FFT regulator.

The stored density convention is ``P[a,b,k] = <c_a^dagger c_b>`` and the
iterated variable is the normal-state-relative density ``D = P - P_ref``.
This module is deliberately only a system adapter: all SCF iteration and
final raw-versus-mixed bookkeeping are delegated to
:func:`mean_field.core.hf.problem.run_hartree_fock_problem`.

The physical map is

``H[D] = H0 + Sigma_H[D] + Sigma_F[D]``

with the unrestricted 4x4 WS-FFT actions from :mod:`local_ws_fft`.  Finite-T
occupation, seed-field construction, and the Helmholtz functional relative to
``D=0`` are shared with :mod:`local_full_hf`.  Seed fields generate only the
initial density and are removed before the first physical SCF step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockRun,
)
from mean_field.core.hf.occupations import (
    fermionic_density_diagnostics,
    stored_projector_to_conventional,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)

from .folded_hf import FoldedTiSe2Geometry
from .local_full_hf import (
    ELECTRONS_PER_LOCAL_MOMENTUM,
    LOCAL_DIMENSION,
    LocalFullSpectrum,
    build_local_full_seed_field,
    diagonalize_local_full_finite_temperature,
    local_full_reference_relative_energy,
    local_full_reference_relative_free_energy,
)
from .local_ws_fft import (
    LocalWSNormalState,
    build_local_ws_normal_state,
    local_ws_fft_fock_action,
    local_ws_fft_hartree_action,
    local_ws_fft_interaction_action,
)
from .monney import MonneyTiSe2Parameters

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
LocalWSSCFInitMode = Literal[
    "normal", "symmetric_exciton", "random_full", "continuation"
]


@dataclass(frozen=True)
class LocalWSSCFConfig:
    """Physical regulator and fixed-mixing controls for WS-FFT SCF."""

    M: int
    L: int
    epsilon_r: float
    temperature_K: float = 65.0
    precision: float = 1.0e-9
    max_iter: int = 300
    fixed_mixing: float = 0.20
    seed_field_ev: float = 1.0e-3
    fft_workers: int = 1
    eigensolver_workers: int = 1
    max_fft_memory_gb: float = 1.0

    def __post_init__(self) -> None:
        if type(self.M) is not int or self.M < 0:
            raise TypeError("M must be an exact nonnegative integer")
        if type(self.L) is not int or self.L < 0:
            raise TypeError("L must be an exact nonnegative integer")
        finite = (
            self.epsilon_r,
            self.temperature_K,
            self.precision,
            self.fixed_mixing,
            self.seed_field_ev,
            self.max_fft_memory_gb,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("LocalWSSCFConfig scalar inputs must be finite")
        if self.epsilon_r <= 0.0 or self.temperature_K <= 0.0:
            raise ValueError("epsilon_r and temperature_K must be positive")
        if self.precision <= 0.0:
            raise ValueError("precision must be positive")
        if not 0.0 < self.fixed_mixing <= 1.0:
            raise ValueError("fixed_mixing must lie in (0,1]")
        if self.seed_field_ev <= 0.0 or self.max_fft_memory_gb <= 0.0:
            raise ValueError("seed_field_ev and max_fft_memory_gb must be positive")
        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise TypeError("max_iter must be an exact positive integer")
        if type(self.fft_workers) is not int or self.fft_workers <= 0:
            raise TypeError("fft_workers must be an exact positive integer")
        if type(self.eigensolver_workers) is not int or self.eigensolver_workers <= 0:
            raise TypeError("eigensolver_workers must be an exact positive integer")


@dataclass
class LocalWSSCFState:
    """Mutable state satisfying the generic ``core/hf`` engine protocol."""

    h0: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float
    precision: float
    diagnostics: dict[str, float]
    normal_state: LocalWSNormalState
    config: LocalWSSCFConfig
    initial_seed_field_ev: ComplexArray

    @property
    def nk(self) -> int:
        return self.normal_state.mesh.nk

    @property
    def reference_projector_stored(self) -> ComplexArray:
        return self.normal_state.reference_projector_stored


@dataclass(frozen=True)
class LocalWSBlockDiagnostics:
    """Unmasked 4x4 block diagnostics averaged over the normalized WS mesh.

    Complex ``*_average`` arrays preserve phase and stored orientation.  RMS
    arrays are nonnegative block magnitudes.  ``exciton_*`` selects the three
    valence--L coherence entries only as a reported order diagnostic; no such
    selection is applied to the density or self-energy during SCF.
    """

    density_average_stored: ComplexArray
    density_rms: FloatArray
    hartree_average_ev: ComplexArray
    hartree_rms_ev: FloatArray
    fock_average_ev: ComplexArray
    fock_rms_ev: FloatArray
    self_energy_average_ev: ComplexArray
    self_energy_rms_ev: FloatArray
    order_magnitude: FloatArray
    exciton_order_stored: ComplexArray
    exciton_self_energy_ev: ComplexArray


@dataclass(frozen=True)
class LocalWSSCFResult:
    """Accepted WS-FFT result with explicit mixed and raw density semantics.

    ``density_delta_stored`` is the final fixed-mixed iterate used to construct
    every reported H/F self-energy and thermodynamic scalar.  The complete
    eigensystem and occupations belong to ``raw_update_projector_stored``, the
    finite-T map of that mixed-density Hamiltonian.  The raw update is never
    silently substituted for the mixed iterate.
    """

    run: HartreeFockRun
    h0_ev: ComplexArray
    density_hartree_ev: ComplexArray
    density_fock_ev: ComplexArray
    interaction_h_ev: ComplexArray
    total_hamiltonian_ev: ComplexArray
    density_delta_stored: ComplexArray
    reference_projector_stored: ComplexArray
    mixed_projector_stored: ComplexArray
    raw_update_projector_stored: ComplexArray
    raw_update_density_delta_stored: ComplexArray
    energies_ev: FloatArray
    eigenvectors: ComplexArray
    occupations: FloatArray
    chemical_potential_ev: float
    raw_particle_residual: float
    mixed_particle_residual: float
    final_raw_norm: float
    reference_relative_energy_ev_A3: float
    reference_relative_free_energy_ev_A3: float
    hermiticity_residual_ev: float
    initial_seed_field_ev: ComplexArray
    block_diagnostics: LocalWSBlockDiagnostics

    @property
    def projector_stored(self) -> ComplexArray:
        """Return the projector associated with the preserved raw eigensystem."""

        return self.raw_update_projector_stored


def _validate_hermitian_field(
    value: np.ndarray,
    *,
    nk: int,
    name: str,
    tolerance: float = 2.0e-11,
) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    expected = (LOCAL_DIMENSION, LOCAL_DIMENSION, nk)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {expected}")
    scale = max(1.0, float(np.max(np.abs(array), initial=0.0)))
    residual = float(np.max(np.abs(array - array.conj().swapaxes(0, 1))))
    if residual > tolerance * scale:
        raise ValueError(f"{name} must be Hermitian in stored orientation")
    return array


def _spectrum(hamiltonian: np.ndarray, state: LocalWSSCFState) -> LocalFullSpectrum:
    return diagonalize_local_full_finite_temperature(
        hamiltonian,
        kbt_ev=state.normal_state.kbt_ev,
        k_weights=state.normal_state.mesh.normalized_weights,
        electrons_per_local_momentum=ELECTRONS_PER_LOCAL_MOMENTUM,
        eigensolver_workers=state.config.eigensolver_workers,
    )


def _physical_projector_diagnostics(
    projector_stored: np.ndarray, state: LocalWSSCFState
):
    projector = _validate_hermitian_field(
        projector_stored, nk=state.nk, name="projector_stored"
    )
    conventional = stored_projector_to_conventional(projector)
    return fermionic_density_diagnostics(
        np.moveaxis(conventional, 2, 0),
        k_weights=state.normal_state.mesh.normalized_weights,
        spectrum_tolerance=2.0e-10,
        eigensolver_workers=state.config.eigensolver_workers,
    )


def build_local_ws_scf_state(
    config: LocalWSSCFConfig,
    *,
    params: MonneyTiSe2Parameters | None = None,
    geometry: FoldedTiSe2Geometry | None = None,
) -> LocalWSSCFState:
    """Build one SCF state by reusing the geometry-bound normal-state adapter."""

    if not isinstance(config, LocalWSSCFConfig):
        raise TypeError("config must be a LocalWSSCFConfig")
    normal = build_local_ws_normal_state(
        M=config.M,
        L=config.L,
        epsilon_r=config.epsilon_r,
        temperature_K=config.temperature_K,
        params=params,
        geometry=geometry,
        fft_workers=config.fft_workers,
        max_fft_memory_gb=config.max_fft_memory_gb,
    )
    return LocalWSSCFState(
        h0=normal.h0_ev.copy(),
        density=np.zeros_like(normal.h0_ev),
        hamiltonian=normal.h0_ev.copy(),
        energies=normal.reference.energies_ev.copy(),
        mu=normal.reference.chemical_potential_ev,
        precision=float(config.precision),
        diagnostics={
            "reference_particle_residual": abs(normal.reference.particle_residual)
        },
        normal_state=normal,
        config=config,
        initial_seed_field_ev=np.zeros_like(normal.h0_ev),
    )


def initialize_local_ws_scf_state(
    state: LocalWSSCFState,
    *,
    init_mode: LocalWSSCFInitMode,
    seed: int = 0,
    continuation_density_delta_stored: np.ndarray | None = None,
) -> None:
    """Initialize ``D0`` and then remove every nonphysical seed field.

    A continuation is an explicit mixed ``D=P-P_ref`` density.  It is checked
    for Hermiticity, finite-T ensemble bounds, and the fixed global filling
    before use.  Continuation data and named seed modes are mutually exclusive.
    """

    if not isinstance(state, LocalWSSCFState):
        raise TypeError("state must be a LocalWSSCFState")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    supported = {"normal", "symmetric_exciton", "random_full", "continuation"}
    if init_mode not in supported:
        raise ValueError(f"init_mode must be one of {sorted(supported)}")
    if (init_mode == "continuation") != (
        continuation_density_delta_stored is not None
    ):
        raise ValueError(
            "continuation mode requires an explicit continuation density, and "
            "other modes forbid one"
        )

    state.initial_seed_field_ev.fill(0.0)
    if init_mode == "continuation":
        supplied = _validate_hermitian_field(
            continuation_density_delta_stored,  # type: ignore[arg-type]
            nk=state.nk,
            name="continuation_density_delta_stored",
        ).copy()
        projector = state.reference_projector_stored + supplied
        diagnostics = _physical_projector_diagnostics(projector, state)
        if (
            abs(
                diagnostics.particle_number
                - ELECTRONS_PER_LOCAL_MOMENTUM
            )
            > 2.0e-11
        ):
            raise ValueError("continuation density has the wrong global filling")
        state.density[:, :, :] = supplied
        state.diagnostics["continuation_initializer"] = 1.0
    else:
        field = build_local_full_seed_field(
            state.nk,
            init_mode=init_mode,  # type: ignore[arg-type]
            amplitude_ev=state.config.seed_field_ev,
            seed=seed,
        )
        state.initial_seed_field_ev[:, :, :] = field
        if init_mode == "normal":
            state.density.fill(0.0)
        else:
            seeded = _spectrum(state.h0 + field, state)
            state.density[:, :, :] = (
                seeded.projector_stored - state.reference_projector_stored
            )
        state.diagnostics["continuation_initializer"] = 0.0

    interaction = local_ws_fft_interaction_action(
        state.density,
        kernel=state.normal_state.kernel,
        fft_workers=state.config.fft_workers,
    )
    state.hamiltonian[:, :, :] = state.h0 + interaction
    physical = _spectrum(state.hamiltonian, state)
    state.energies[:, :] = physical.energies_ev
    state.mu = physical.chemical_potential_ev


def local_ws_block_diagnostics(
    density_delta_stored: np.ndarray,
    *,
    density_hartree_ev: np.ndarray,
    density_fock_ev: np.ndarray,
    state: LocalWSSCFState,
) -> LocalWSBlockDiagnostics:
    """Return unmasked density and H/F self-energy diagnostics for all blocks."""

    density = _validate_hermitian_field(
        density_delta_stored, nk=state.nk, name="density_delta_stored"
    )
    hartree = _validate_hermitian_field(
        density_hartree_ev, nk=state.nk, name="density_hartree_ev"
    )
    fock = _validate_hermitian_field(
        density_fock_ev, nk=state.nk, name="density_fock_ev"
    )
    weights = state.normal_state.mesh.normalized_weights

    def average(array: ComplexArray) -> ComplexArray:
        return np.asarray(
            np.einsum("abk,k->ab", array, weights, optimize=True),
            dtype=np.complex128,
        )

    def rms(array: ComplexArray) -> FloatArray:
        return np.asarray(
            np.sqrt(
                np.einsum(
                    "abk,abk,k->ab", array.conj(), array, weights, optimize=True
                ).real
            ),
            dtype=np.float64,
        )

    density_average = average(density)
    hartree_average = average(hartree)
    fock_average = average(fock)
    self_energy = hartree + fock
    self_energy_average = average(self_energy)
    return LocalWSBlockDiagnostics(
        density_average_stored=density_average,
        density_rms=rms(density),
        hartree_average_ev=hartree_average,
        hartree_rms_ev=rms(hartree),
        fock_average_ev=fock_average,
        fock_rms_ev=rms(fock),
        self_energy_average_ev=self_energy_average,
        self_energy_rms_ev=rms(self_energy),
        order_magnitude=np.asarray(np.abs(density_average), dtype=np.float64),
        exciton_order_stored=np.asarray(density_average[0, 1:], dtype=np.complex128),
        exciton_self_energy_ev=np.asarray(
            self_energy_average[0, 1:], dtype=np.complex128
        ),
    )


def build_local_ws_scf_problem(
    state: LocalWSSCFState,
    *,
    continuation_density_delta_stored: np.ndarray | None = None,
) -> HartreeFockProblem:
    """Bind TiSe2 WS callbacks to the generic ``HartreeFockProblem`` surface."""

    if not isinstance(state, LocalWSSCFState):
        raise TypeError("state must be a LocalWSSCFState")
    continuation = (
        None
        if continuation_density_delta_stored is None
        else np.asarray(
            continuation_density_delta_stored, dtype=np.complex128
        ).copy()
    )

    def initializer(target: LocalWSSCFState, *, init_mode: str, seed: int) -> None:
        initialize_local_ws_scf_state(
            target,
            init_mode=init_mode,  # type: ignore[arg-type]
            seed=seed,
            continuation_density_delta_stored=continuation,
        )

    def interaction_builder(density: np.ndarray) -> ComplexArray:
        return local_ws_fft_interaction_action(
            density,
            kernel=state.normal_state.kernel,
            fft_workers=state.config.fft_workers,
        )

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        spectrum = _spectrum(hamiltonian, state)
        return DensityUpdateResult(
            density=np.asarray(
                spectrum.projector_stored - state.reference_projector_stored,
                dtype=np.complex128,
            ),
            energies=spectrum.energies_ev,
            mu=spectrum.chemical_potential_ev,
            observables={
                "eigenvectors": spectrum.eigenvectors,
                "projector_stored": spectrum.projector_stored,
                "occupations": spectrum.occupations,
                "particle_residual": spectrum.particle_residual,
                "entropy_per_local_momentum": spectrum.entropy_per_local_momentum,
            },
        )

    def free_energy(
        interaction_h: np.ndarray, h0: np.ndarray, density: np.ndarray
    ) -> float:
        return local_full_reference_relative_free_energy(
            interaction_h,
            h0,
            density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.normal_state.mesh,
            kbt_ev=state.normal_state.kbt_ev,
            eigensolver_workers=state.config.eigensolver_workers,
        )

    def convergence_metric(updated: np.ndarray, previous: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(updated) - np.asarray(previous))))

    def final_acceptance(
        target: LocalWSSCFState,
        update: DensityUpdateResult,
        final_norm: float,
    ) -> FinalAcceptanceResult:
        hermiticity = float(
            np.max(
                np.abs(
                    target.hamiltonian
                    - target.hamiltonian.conj().swapaxes(0, 1)
                )
            )
        )
        raw_particle_residual = abs(float(update.observables["particle_residual"]))
        mixed_projector = target.reference_projector_stored + target.density
        try:
            mixed_diagnostics = _physical_projector_diagnostics(
                mixed_projector, target
            )
            mixed_particle_residual = abs(
                mixed_diagnostics.particle_number
                - ELECTRONS_PER_LOCAL_MOMENTUM
            )
            physical_mixed_density = True
        except ValueError:
            mixed_particle_residual = float("inf")
            physical_mixed_density = False
        accepted = (
            hermiticity <= 1.0e-10
            and raw_particle_residual <= 2.0e-11
            and mixed_particle_residual <= 2.0e-11
            and physical_mixed_density
            and final_norm <= target.precision
        )
        diagnostics = {
            "final_hermiticity_residual_ev": hermiticity,
            "final_particle_residual": raw_particle_residual,
            "final_mixed_particle_residual": mixed_particle_residual,
        }
        if not all(np.isfinite(value) for value in diagnostics.values()):
            diagnostics["final_mixed_particle_residual"] = np.finfo(float).max
        return FinalAcceptanceResult(
            accepted=bool(accepted),
            reason="accepted" if accepted else "local_ws_physical_final_gate",
            diagnostics=diagnostics,
        )

    return HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=free_energy,
            oda_parameterizer=lambda _target, _delta: state.config.fixed_mixing,
            final_acceptance=final_acceptance,
            convergence_metric=convergence_metric,
            convergence_rule="raw",
        ),
    )


def run_local_ws_scf(
    state: LocalWSSCFState,
    *,
    init_mode: LocalWSSCFInitMode = "normal",
    seed: int = 0,
    continuation_density_delta_stored: np.ndarray | None = None,
) -> LocalWSSCFResult:
    """Run unrestricted fixed-mixing WS-FFT HF through the generic engine."""

    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    problem = build_local_ws_scf_problem(
        state,
        continuation_density_delta_stored=continuation_density_delta_stored,
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=state.config.max_iter,
        oda_stall_threshold=1.0e-15,
    )
    if not run.converged:
        raise RuntimeError(
            "local WS-FFT TiSe2 HF did not pass the generic SCF/final gates: "
            f"exit_reason={run.exit_reason}, final_raw_norm="
            f"{state.diagnostics.get('final_raw_norm')}"
        )

    hartree = local_ws_fft_hartree_action(
        state.density, kernel=state.normal_state.kernel
    )
    fock = local_ws_fft_fock_action(
        state.density,
        kernel=state.normal_state.kernel,
        fft_workers=state.config.fft_workers,
    )
    interaction = hartree + fock
    total_hamiltonian = state.h0 + interaction
    final = _spectrum(total_hamiltonian, state)
    mixed_projector = state.reference_projector_stored + state.density
    raw_density = final.projector_stored - state.reference_projector_stored
    mixed_diagnostics = _physical_projector_diagnostics(mixed_projector, state)
    mixed_residual = (
        mixed_diagnostics.particle_number - ELECTRONS_PER_LOCAL_MOMENTUM
    )
    hermiticity = float(
        np.max(
            np.abs(
                total_hamiltonian
                - total_hamiltonian.conj().swapaxes(0, 1)
            )
        )
    )
    relative_energy = local_full_reference_relative_energy(
        interaction,
        state.h0,
        state.density,
        mesh=state.normal_state.mesh,
    )
    free_energy = local_full_reference_relative_free_energy(
        interaction,
        state.h0,
        state.density,
        reference_projector_stored=state.reference_projector_stored,
        mesh=state.normal_state.mesh,
        kbt_ev=state.normal_state.kbt_ev,
        eigensolver_workers=state.config.eigensolver_workers,
    )
    blocks = local_ws_block_diagnostics(
        state.density,
        density_hartree_ev=hartree,
        density_fock_ev=fock,
        state=state,
    )
    return LocalWSSCFResult(
        run=run,
        h0_ev=state.h0.copy(),
        density_hartree_ev=hartree,
        density_fock_ev=fock,
        interaction_h_ev=interaction,
        total_hamiltonian_ev=total_hamiltonian,
        density_delta_stored=state.density.copy(),
        reference_projector_stored=state.reference_projector_stored.copy(),
        mixed_projector_stored=np.asarray(mixed_projector, dtype=np.complex128),
        raw_update_projector_stored=final.projector_stored,
        raw_update_density_delta_stored=np.asarray(raw_density, dtype=np.complex128),
        energies_ev=final.energies_ev,
        eigenvectors=final.eigenvectors,
        occupations=final.occupations,
        chemical_potential_ev=final.chemical_potential_ev,
        raw_particle_residual=final.particle_residual,
        mixed_particle_residual=float(mixed_residual),
        final_raw_norm=float(state.diagnostics["final_raw_norm"]),
        reference_relative_energy_ev_A3=relative_energy,
        reference_relative_free_energy_ev_A3=free_energy,
        hermiticity_residual_ev=hermiticity,
        initial_seed_field_ev=state.initial_seed_field_ev.copy(),
        block_diagnostics=blocks,
    )


__all__ = [
    "LocalWSBlockDiagnostics",
    "LocalWSSCFConfig",
    "LocalWSSCFInitMode",
    "LocalWSSCFResult",
    "LocalWSSCFState",
    "build_local_ws_scf_problem",
    "build_local_ws_scf_state",
    "initialize_local_ws_scf_state",
    "local_ws_block_diagnostics",
    "run_local_ws_scf",
]
