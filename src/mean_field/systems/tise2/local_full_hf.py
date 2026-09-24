"""Local unrestricted full-4x4 Hartree--Fock adapter for 1T-TiSe2.

The basis at every local momentum ``p`` is
``(v_{Gamma+p}, c_{L1+p}, c_{L2+p}, c_{L3+p})``.  Stored densities use
``P[a,b,p] = <c_a^dagger c_b>``.  The one-body Hamiltonian is exactly
:func:`monney_bare_hamiltonian` on the existing C3-closed local mesh.

Because those phenomenological bands already describe the fitted normal
state, both interaction terms are reference subtracted::

    D = P - Pref
    H[D] = H0 + Sigma_H[D] + Sigma_F[D].

Unit form factors and the fixed-representative exact-``G=0`` route inventory
are reused from :mod:`folded_hf`.  The inventory contains 28 quartets.  Only
the ``q=0`` Hartree coefficient is removed; finite-transfer Hartree terms and
all 4x4 external blocks remain active.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockRun,
)
from mean_field.core.hf.occupations import (
    conventional_projector_to_stored,
    fermionic_density_diagnostics,
    global_fermi_occupations,
    stored_projector_to_conventional,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)

from .folded_hf import (
    FoldedCoulombKernel,
    FoldedTiSe2Geometry,
    physical_momentum_routes,
    physical_q_vectors,
    precompute_folded_coulomb_kernel,
)
from .hf import BOLTZMANN_EV_PER_K, MonneyLocalMesh, build_monney_local_mesh
from .monney import MonneyTiSe2Parameters, monney_bare_hamiltonian

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
SeedMode = Literal["normal", "symmetric_exciton", "random_full"]

LOCAL_DIMENSION = 4
ELECTRONS_PER_LOCAL_MOMENTUM = 1.0


@dataclass(frozen=True)
class LocalFullSpectrum:
    """Complete four-state eigensystem and one global finite-T density."""

    energies_ev: FloatArray
    eigenvectors: ComplexArray
    occupations: FloatArray
    projector_stored: ComplexArray
    chemical_potential_ev: float
    particle_number_per_local_momentum: float
    particle_residual: float
    entropy_per_local_momentum: float


@dataclass(frozen=True)
class LocalFullHFConfig:
    """Physical regulator and solver inputs for the local unrestricted model."""

    epsilon_r: float
    temperature_K: float = 65.0
    electrons_per_local_momentum: float = ELECTRONS_PER_LOCAL_MOMENTUM
    inplane_shells: int = 1
    inplane_spacing_Ainv: float = 0.040
    z_shells: int = 1
    z_spacing_Ainv: float = 0.050
    precision: float = 1.0e-9
    max_iter: int = 300
    max_dense_memory_gb: float = 1.0
    seed_field_ev: float = 1.0e-3
    interaction_workers: int = 1
    eigensolver_workers: int = 1
    oda_grid_points: int = 17
    mixing_policy: Literal["oda", "fixed"] = "oda"
    fixed_mixing: float = 0.20
    screening_provenance: str = "explicit constant-epsilon model input"
    regulator_provenance: str = "C3-closed local hexagonal-prism quadrature"
    route_policy: Literal["fixed_representative_g0"] = "fixed_representative_g0"

    def __post_init__(self) -> None:
        scalars = (
            self.epsilon_r,
            self.temperature_K,
            self.electrons_per_local_momentum,
            self.inplane_spacing_Ainv,
            self.z_spacing_Ainv,
            self.precision,
            self.max_dense_memory_gb,
            self.seed_field_ev,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("LocalFullHFConfig scalar inputs must be finite")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if self.electrons_per_local_momentum != ELECTRONS_PER_LOCAL_MOMENTUM:
            raise ValueError("the local TiSe2 contract fixes target filling to one")
        if type(self.inplane_shells) is not int or self.inplane_shells < 0:
            raise TypeError("inplane_shells must be an exact nonnegative integer")
        if type(self.z_shells) is not int or self.z_shells < 0:
            raise TypeError("z_shells must be an exact nonnegative integer")
        if self.inplane_spacing_Ainv <= 0.0 or self.z_spacing_Ainv <= 0.0:
            raise ValueError("mesh spacings must be positive")
        if self.precision <= 0.0:
            raise ValueError("precision must be positive")
        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise TypeError("max_iter must be an exact positive integer")
        if self.max_dense_memory_gb <= 0.0:
            raise ValueError("max_dense_memory_gb must be positive")
        if self.seed_field_ev <= 0.0:
            raise ValueError("seed_field_ev must be positive")
        if type(self.interaction_workers) is not int or self.interaction_workers <= 0:
            raise TypeError("interaction_workers must be an exact positive integer")
        if type(self.eigensolver_workers) is not int or self.eigensolver_workers <= 0:
            raise TypeError("eigensolver_workers must be an exact positive integer")
        if type(self.oda_grid_points) is not int or self.oda_grid_points < 5:
            raise TypeError("oda_grid_points must be an exact integer at least five")
        if self.mixing_policy not in {"oda", "fixed"}:
            raise ValueError("mixing_policy must be 'oda' or 'fixed'")
        if not np.isfinite(self.fixed_mixing) or not (
            0.0 < self.fixed_mixing <= 1.0
        ):
            raise ValueError("fixed_mixing must be finite and in (0, 1]")
        if not self.screening_provenance or not self.regulator_provenance:
            raise ValueError("assumption provenance strings must be nonempty")
        if self.route_policy != "fixed_representative_g0":
            raise ValueError("route_policy must be 'fixed_representative_g0'")

    @property
    def kbt_ev(self) -> float:
        return float(BOLTZMANN_EV_PER_K * self.temperature_K)


@dataclass
class LocalFullHFState:
    """Mutable state consumed by the generic :class:`HartreeFockProblem` engine."""

    h0: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float
    precision: float
    diagnostics: dict[str, float]
    mesh: MonneyLocalMesh
    coulomb: FoldedCoulombKernel
    reference_projector_stored: ComplexArray
    reference_occupations: FloatArray
    q_vectors_Ainv: FloatArray
    params: MonneyTiSe2Parameters
    geometry: FoldedTiSe2Geometry
    config: LocalFullHFConfig
    initial_seed_field_ev: ComplexArray

    @property
    def nk(self) -> int:
        return self.mesh.nk


@dataclass(frozen=True)
class LocalFullHFResult:
    """Closed SCF state with distinct mixed and raw finite-T densities.

    ``density_delta_stored`` and ``mixed_projector_stored`` are the final
    ODA-mixed iterate and are the density used for the reported interaction,
    Hamiltonian, energy, and free energy.  ``raw_update_projector_stored`` and
    ``raw_update_density_delta_stored`` are instead the finite-temperature
    Fermi-map output of that final Hamiltonian.  The reported energies,
    eigenvectors, occupations, and chemical potential belong to this raw
    eigensystem; the raw density is not silently substituted for the mixed
    iterate.
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

    @property
    def projector_stored(self) -> ComplexArray:
        """Projector associated with the preserved final raw eigensystem."""

        return self.raw_update_projector_stored


def _validate_local_matrix(array: np.ndarray, nk: int, *, name: str) -> ComplexArray:
    matrix = np.asarray(array, dtype=np.complex128)
    expected = (LOCAL_DIMENSION, LOCAL_DIMENSION, nk)
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape {expected}")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    residual = float(np.max(np.abs(matrix - matrix.conj().swapaxes(0, 1))))
    if residual > 128.0 * np.finfo(float).eps * scale:
        raise ValueError(f"{name} must be Hermitian; residual={residual:.3e}")
    return matrix


def _validate_hermitian_action(array: np.ndarray, *, name: str) -> None:
    scale = max(1.0, float(np.max(np.abs(array))))
    residual = float(np.max(np.abs(array - array.conj().swapaxes(0, 1))))
    if residual > 2.0e-11 * scale:
        raise ValueError(f"{name} is not Hermitian; residual={residual:.3e}")


def local_full_h0(
    points_Ainv: np.ndarray,
    *,
    params: MonneyTiSe2Parameters | None = None,
) -> ComplexArray:
    """Return exactly ``monney_bare_hamiltonian(points_Ainv, params)``."""

    return monney_bare_hamiltonian(points_Ainv, params=params)


def _validate_interaction_worker_environment(workers: int) -> None:
    """Fail closed on invalid workers or nested numerical threading."""

    if type(workers) is not int or workers <= 0:
        raise TypeError("workers must be an exact positive integer")
    if workers == 1:
        return
    invalid = {
        name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(
            "target-block interaction parallelism requires explicit one-thread "
            f"BLAS/OpenMP binding; invalid environment: {invalid}"
        )
    for name in ("MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        value = os.environ.get(name)
        if value not in (None, "1"):
            raise RuntimeError(
                "target-block interaction parallelism forbids nested numerical "
                f"threads; {name}={value!r}"
            )


def local_full_fock_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
    workers: int = 1,
) -> ComplexArray:
    r"""Apply the unrestricted local exchange map to ``D=P-Pref``.

    For every retained quartet ``(s,t,u,v)``,
    ``Sigma_F[s,t,k] -= sum_p w_p V(p_k-p_p+Q_s-Q_v) D[u,v,p]``.
    The 16 external ``(s,t)`` blocks are independent.  Parallel workers write
    disjoint blocks while preserving the serial route and source reductions
    within every block.
    """

    _validate_interaction_worker_environment(workers)
    density = _validate_local_matrix(
        density_delta_stored, coulomb.mesh.nk, name="density_delta_stored"
    )
    routes = physical_momentum_routes()
    sigma = np.zeros_like(density)

    def apply_external_block(block_index: int) -> None:
        sector, target_sector = divmod(block_index, LOCAL_DIMENSION)
        block = sigma[sector, target_sector]
        for density_row_sector, density_col_sector in routes[sector][target_sector]:
            kernel = coulomb.shifted_weighted_ev[sector, density_col_sector]
            block -= np.einsum(
                "kp,p->k",
                kernel,
                density[density_row_sector, density_col_sector],
                optimize=True,
            )

    block_indices = range(LOCAL_DIMENSION * LOCAL_DIMENSION)
    if workers == 1:
        for block_index in block_indices:
            apply_external_block(block_index)
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, LOCAL_DIMENSION * LOCAL_DIMENSION)
        ) as executor:
            for _ in executor.map(apply_external_block, block_indices):
                pass
    _validate_hermitian_action(sigma, name="local Fock action")
    return sigma


def local_full_hartree_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
) -> ComplexArray:
    r"""Apply finite-transfer Hartree while removing only its ``q=0`` term."""

    density = _validate_local_matrix(
        density_delta_stored, coulomb.mesh.nk, name="density_delta_stored"
    )
    routes = physical_momentum_routes()
    weights = coulomb.mesh.weights_Ainv3
    sigma = np.zeros_like(density)
    for sector in range(LOCAL_DIMENSION):
        for target_sector in range(LOCAL_DIMENSION):
            coefficient = coulomb.hartree_ev[sector, target_sector]
            if coefficient == 0.0:
                continue
            value = 0.0j
            for density_row_sector, density_col_sector in routes[sector][target_sector]:
                value += np.einsum(
                    "p,p->",
                    density[density_row_sector, density_col_sector],
                    weights,
                    optimize=True,
                )
            sigma[sector, target_sector, :] = coefficient * value
    _validate_hermitian_action(sigma, name="local Hartree action")
    return sigma


def local_full_interaction_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
    workers: int = 1,
) -> ComplexArray:
    """Return ``Sigma_H[D] + Sigma_F[D]`` with no block projection."""

    _validate_interaction_worker_environment(workers)
    return local_full_hartree_action(
        density_delta_stored, coulomb=coulomb
    ) + local_full_fock_action(
        density_delta_stored, coulomb=coulomb, workers=workers
    )


def local_full_interaction_energy(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
    workers: int = 1,
) -> float:
    """Return ``1/2 <D,L[D]>`` in eV/Angstrom^3."""

    density = _validate_local_matrix(
        density_delta_stored, coulomb.mesh.nk, name="density_delta_stored"
    )
    sigma = local_full_interaction_action(
        density, coulomb=coulomb, workers=workers
    )
    value = 0.5 * np.einsum(
        "abk,abk,k->", sigma, density, coulomb.mesh.weights_Ainv3, optimize=True
    )
    if abs(value.imag) > 2.0e-11 * max(1.0, abs(value.real)):
        raise ValueError(f"interaction energy is not real: {value!r}")
    return float(value.real)


def local_full_reference_relative_energy(
    interaction_h_ev: np.ndarray,
    h0_ev: np.ndarray,
    density_delta_stored: np.ndarray,
    *,
    mesh: MonneyLocalMesh,
) -> float:
    """Return ``<H0,D> + 1/2 <L[D],D>`` in eV/Angstrom^3."""

    density = _validate_local_matrix(
        density_delta_stored, mesh.nk, name="density_delta_stored"
    )
    interaction = _validate_local_matrix(interaction_h_ev, mesh.nk, name="interaction_h_ev")
    h0 = _validate_local_matrix(h0_ev, mesh.nk, name="h0_ev")
    value = np.einsum("abk,abk,k->", h0, density, mesh.weights_Ainv3, optimize=True)
    value += 0.5 * np.einsum(
        "abk,abk,k->", interaction, density, mesh.weights_Ainv3, optimize=True
    )
    if abs(value.imag) > 2.0e-11 * max(1.0, abs(value.real)):
        raise ValueError(f"reference-relative energy is not real: {value!r}")
    return float(value.real)


def _entropy_phi_density(
    projector_stored: np.ndarray,
    mesh: MonneyLocalMesh,
    *,
    eigensolver_workers: int = 1,
) -> float:
    projector = _validate_local_matrix(projector_stored, mesh.nk, name="projector_stored")
    projector_ket = stored_projector_to_conventional(projector)
    diagnostics = fermionic_density_diagnostics(
        np.moveaxis(projector_ket, 2, 0),
        k_weights=mesh.normalized_weights,
        spectrum_tolerance=2.0e-10,
        eigensolver_workers=eigensolver_workers,
    )
    return float(-np.sum(mesh.weights_Ainv3) * diagnostics.entropy_dimensionless)


def local_full_reference_relative_free_energy(
    interaction_h_ev: np.ndarray,
    h0_ev: np.ndarray,
    density_delta_stored: np.ndarray,
    *,
    reference_projector_stored: np.ndarray,
    mesh: MonneyLocalMesh,
    kbt_ev: float,
    eigensolver_workers: int = 1,
) -> float:
    """Return the reference-relative Helmholtz functional matching ``H[D]``."""

    density = _validate_local_matrix(
        density_delta_stored, mesh.nk, name="density_delta_stored"
    )
    reference = _validate_local_matrix(
        reference_projector_stored, mesh.nk, name="reference_projector_stored"
    )
    temperature = float(kbt_ev)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("kbt_ev must be finite and positive")
    internal = local_full_reference_relative_energy(
        interaction_h_ev, h0_ev, density, mesh=mesh
    )
    phi = _entropy_phi_density(
        reference + density, mesh, eigensolver_workers=eigensolver_workers
    )
    reference_phi = _entropy_phi_density(
        reference, mesh, eigensolver_workers=eigensolver_workers
    )
    return float(internal + temperature * (phi - reference_phi))


def diagonalize_local_full_finite_temperature(
    hamiltonian_abk: np.ndarray,
    *,
    kbt_ev: float,
    k_weights: np.ndarray | None = None,
    electrons_per_local_momentum: float = ELECTRONS_PER_LOCAL_MOMENTUM,
    particle_tolerance: float = 1.0e-12,
    eigensolver_workers: int = 1,
) -> LocalFullSpectrum:
    """Retain all four states and solve one global finite-T filling constraint."""

    hamiltonian = np.asarray(hamiltonian_abk, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[:2] != (LOCAL_DIMENSION, LOCAL_DIMENSION)
        or hamiltonian.shape[2] == 0
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("hamiltonian_abk must be finite with shape (4,4,nk)")
    _validate_hermitian_action(hamiltonian, name="local Hamiltonian")
    temperature = float(kbt_ev)
    tolerance = float(particle_tolerance)
    target = float(electrons_per_local_momentum)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("kbt_ev must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("particle_tolerance must be finite and positive")
    if target != ELECTRONS_PER_LOCAL_MOMENTUM:
        raise ValueError("electrons_per_local_momentum must equal one")

    nk = hamiltonian.shape[2]
    if k_weights is None:
        weights = np.full(nk, 1.0 / nk, dtype=np.float64)
    else:
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape != (nk,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("k_weights must be finite, positive, and have shape (nk,)")
        weights = weights / np.sum(weights)

    by_k = np.moveaxis(hamiltonian, 2, 0)
    occupation = global_fermi_occupations(
        by_k,
        target_particle_number=target,
        kbt_ev=temperature,
        k_weights=weights,
        particle_tolerance=tolerance,
        eigensolver_workers=eigensolver_workers,
    )
    if eigensolver_workers == 1:
        energies_by_k, eigenvectors = np.linalg.eigh(by_k)
    else:
        energies_by_k = np.empty((nk, LOCAL_DIMENSION), dtype=np.float64)
        eigenvectors = np.empty(
            (nk, LOCAL_DIMENSION, LOCAL_DIMENSION), dtype=np.complex128
        )

        def diagonalize(k_index: int) -> tuple[int, FloatArray, ComplexArray]:
            values, vectors = np.linalg.eigh(by_k[k_index])
            return k_index, values, vectors

        with ThreadPoolExecutor(max_workers=min(eigensolver_workers, nk)) as executor:
            for k_index, values, vectors in executor.map(diagonalize, range(nk)):
                energies_by_k[k_index] = values
                eigenvectors[k_index] = vectors

    energies = np.asarray(energies_by_k.T, dtype=np.float64)
    if not np.allclose(energies, occupation.energies, rtol=0.0, atol=2.0e-13):
        raise RuntimeError("preserved eigensystem energies disagree with core occupations")
    occupations = np.asarray(occupation.occupations, dtype=np.float64)
    density_ket = (
        eigenvectors * occupations.T[:, None, :]
    ) @ eigenvectors.conj().transpose(0, 2, 1)
    scale = max(1.0, float(np.max(np.abs(occupation.density_ket))))
    tolerance_density = 4096.0 * np.finfo(float).eps * scale
    residual = float(np.max(np.abs(density_ket - occupation.density_ket)))
    if residual > tolerance_density:
        raise RuntimeError(
            "projector reconstructed from preserved eigensystem disagrees with core "
            f"occupations: residual={residual:.3e}"
        )
    projector_stored = conventional_projector_to_stored(
        np.moveaxis(density_ket, 0, 2)
    )
    return LocalFullSpectrum(
        energies_ev=energies,
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        occupations=occupations,
        projector_stored=np.asarray(projector_stored, dtype=np.complex128),
        chemical_potential_ev=float(occupation.chemical_potential),
        particle_number_per_local_momentum=float(occupation.particle_number),
        particle_residual=float(occupation.particle_residual),
        entropy_per_local_momentum=float(occupation.entropy_dimensionless),
    )


def build_local_full_seed_field(
    nk: int,
    *,
    init_mode: SeedMode,
    amplitude_ev: float,
    seed: int,
) -> ComplexArray:
    """Build an unconstrained Hermitian seed field; it is never a block mask."""

    if type(nk) is not int or nk <= 0:
        raise TypeError("nk must be an exact positive integer")
    amplitude = float(amplitude_ev)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude_ev must be finite and positive")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if init_mode not in {"normal", "symmetric_exciton", "random_full"}:
        raise ValueError(
            "init_mode must be 'normal', 'symmetric_exciton', or 'random_full'"
        )

    field = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, nk), dtype=np.complex128)
    if init_mode == "symmetric_exciton":
        field[0, 1:, :] = -amplitude
        field[1:, 0, :] = -amplitude
    elif init_mode == "random_full":
        rng = np.random.default_rng(seed)
        while True:
            sampled = rng.normal(size=(LOCAL_DIMENSION, LOCAL_DIMENSION))
            sampled = sampled + 1j * rng.normal(size=sampled.shape)
            hermitian = 0.5 * (sampled + sampled.conj().T)
            if np.all(np.abs(hermitian) > 0.0):
                break
        local_seed = amplitude * hermitian / np.max(np.abs(hermitian))
        field[:, :, :] = local_seed[:, :, None]
    return field


def initialize_local_full_hf_state(
    state: LocalFullHFState,
    *,
    init_mode: SeedMode,
    seed: int = 0,
) -> None:
    """Initialize a coherent physical state while preserving the seeded ``D0``.

    For a non-normal initialization the external seed field is used only to
    generate the mixed starting density ``D0``.  The field is then removed:
    ``state.hamiltonian`` is set to ``H0 + L[D0]``, and ``state.energies`` and
    ``state.mu`` are computed from that physical Hamiltonian.  The corresponding
    raw finite-T update is deliberately *not* substituted for ``state.density``.
    For ``init_mode="normal"``, ``D0=0`` and the physical Hamiltonian is ``H0``.
    """

    field = build_local_full_seed_field(
        state.nk,
        init_mode=init_mode,
        amplitude_ev=state.config.seed_field_ev,
        seed=seed,
    )
    state.initial_seed_field_ev[:, :, :] = field
    if init_mode == "normal":
        state.density.fill(0.0)
        state.hamiltonian[:, :, :] = state.h0
    else:
        seed_spectrum = diagonalize_local_full_finite_temperature(
            state.h0 + field,
            kbt_ev=state.config.kbt_ev,
            k_weights=state.mesh.normalized_weights,
            eigensolver_workers=state.config.eigensolver_workers,
        )
        state.density[:, :, :] = (
            seed_spectrum.projector_stored - state.reference_projector_stored
        )
        state.hamiltonian[:, :, :] = state.h0 + local_full_interaction_action(
            state.density,
            coulomb=state.coulomb,
            workers=state.config.interaction_workers,
        )

    physical_spectrum = diagonalize_local_full_finite_temperature(
        state.hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
        eigensolver_workers=state.config.eigensolver_workers,
    )
    state.energies[:, :] = physical_spectrum.energies_ev
    state.mu = physical_spectrum.chemical_potential_ev


def build_local_full_hf_state(
    config: LocalFullHFConfig,
    *,
    params: MonneyTiSe2Parameters | None = None,
    geometry: FoldedTiSe2Geometry | None = None,
) -> LocalFullHFState:
    """Build the C3 local mesh, exact Monney ``H0``, reference, and kernels."""

    _validate_interaction_worker_environment(config.interaction_workers)
    model = MonneyTiSe2Parameters() if params is None else params
    lattice = (
        FoldedTiSe2Geometry(c_angstrom=model.c_angstrom)
        if geometry is None
        else geometry
    )
    if not np.isclose(lattice.c_angstrom, model.c_angstrom, rtol=0.0, atol=1.0e-12):
        raise ValueError("geometry.c_angstrom must equal params.c_angstrom")
    mesh = build_monney_local_mesh(
        inplane_shells=config.inplane_shells,
        inplane_spacing_Ainv=config.inplane_spacing_Ainv,
        z_shells=config.z_shells,
        z_spacing_Ainv=config.z_spacing_Ainv,
    )
    h0 = local_full_h0(mesh.points_Ainv, params=model)
    reference = diagonalize_local_full_finite_temperature(
        h0,
        kbt_ev=config.kbt_ev,
        k_weights=mesh.normalized_weights,
        electrons_per_local_momentum=config.electrons_per_local_momentum,
        eigensolver_workers=config.eigensolver_workers,
    )
    q_vectors = physical_q_vectors(lattice)
    coulomb = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=config.epsilon_r,
        max_dense_memory_gb=config.max_dense_memory_gb,
        route_policy=config.route_policy,
    )
    return LocalFullHFState(
        h0=h0.copy(),
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=reference.energies_ev.copy(),
        mu=reference.chemical_potential_ev,
        precision=float(config.precision),
        diagnostics={"reference_particle_residual": abs(reference.particle_residual)},
        mesh=mesh,
        coulomb=coulomb,
        reference_projector_stored=reference.projector_stored.copy(),
        reference_occupations=reference.occupations.copy(),
        q_vectors_Ainv=q_vectors,
        params=model,
        geometry=lattice,
        config=config,
        initial_seed_field_ev=np.zeros_like(h0),
    )


def _spectrum_for_state(
    hamiltonian: ComplexArray, state: LocalFullHFState
) -> LocalFullSpectrum:
    return diagonalize_local_full_finite_temperature(
        hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
        electrons_per_local_momentum=state.config.electrons_per_local_momentum,
        eigensolver_workers=state.config.eigensolver_workers,
    )


def _finite_temperature_oda_parameter(
    state: LocalFullHFState,
    delta_density: ComplexArray,
) -> float:
    """Minimize the matching free energy on the closed segment ``[0,1]``."""

    previous = state.density.copy()
    base_interaction = state.hamiltonian - state.h0
    delta_interaction = local_full_interaction_action(
        delta_density,
        coulomb=state.coulomb,
        workers=state.config.interaction_workers,
    )

    def objective(value: float) -> float:
        lam = float(value)
        density = previous + lam * delta_density
        interaction = base_interaction + lam * delta_interaction
        return local_full_reference_relative_free_energy(
            interaction,
            state.h0,
            density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.mesh,
            kbt_ev=state.config.kbt_ev,
            eigensolver_workers=state.config.eigensolver_workers,
        )

    grid = np.linspace(0.0, 1.0, state.config.oda_grid_points)
    values = np.asarray([objective(value) for value in grid], dtype=np.float64)
    candidates: list[tuple[float, float]] = [
        (float(values[0]), 0.0),
        (float(values[-1]), 1.0),
    ]
    for index in range(1, grid.size - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            result = minimize_scalar(
                objective,
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": 1.0e-10, "maxiter": 128},
            )
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.fun), float(result.x)))
            candidates.append((float(values[index]), float(grid[index])))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def run_local_full_hf(
    state: LocalFullHFState,
    *,
    init_mode: SeedMode = "normal",
    seed: int = 0,
) -> LocalFullHFResult:
    """Run unrestricted local HF through the generic ``core/hf`` engine."""

    if init_mode not in {"normal", "symmetric_exciton", "random_full"}:
        raise ValueError(
            "init_mode must be 'normal', 'symmetric_exciton', or 'random_full'"
        )
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")

    def density_update(hamiltonian: ComplexArray) -> DensityUpdateResult:
        spectrum = _spectrum_for_state(hamiltonian, state)
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

    def initializer(target: LocalFullHFState, *, init_mode: str, seed: int) -> None:
        initialize_local_full_hf_state(
            target, init_mode=init_mode, seed=seed  # type: ignore[arg-type]
        )

    def interaction_builder(density: ComplexArray) -> ComplexArray:
        return local_full_interaction_action(
            density,
            coulomb=state.coulomb,
            workers=state.config.interaction_workers,
        )

    def energy_functional(
        interaction_h: ComplexArray,
        h0: ComplexArray,
        density: ComplexArray,
    ) -> float:
        return local_full_reference_relative_free_energy(
            interaction_h,
            h0,
            density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.mesh,
            kbt_ev=state.config.kbt_ev,
            eigensolver_workers=state.config.eigensolver_workers,
        )

    def convergence_metric(updated: np.ndarray, previous: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(updated) - np.asarray(previous))))

    def final_acceptance(
        target: LocalFullHFState,
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
        mixed_number = float(
            np.einsum(
                "aak,k->",
                mixed_projector,
                target.mesh.normalized_weights,
                optimize=True,
            ).real
        )
        mixed_particle_residual = abs(
            mixed_number - target.config.electrons_per_local_momentum
        )
        accepted = (
            hermiticity <= 1.0e-10
            and raw_particle_residual <= 2.0e-11
            and mixed_particle_residual <= 2.0e-11
            and final_norm <= target.precision
        )
        return FinalAcceptanceResult(
            accepted=bool(accepted),
            reason="accepted" if accepted else "local_full_physical_final_gate",
            diagnostics={
                "final_hermiticity_residual_ev": hermiticity,
                "final_particle_residual": raw_particle_residual,
                "final_mixed_particle_residual": mixed_particle_residual,
            },
        )

    def mixing_parameter(
        target: LocalFullHFState, delta: ComplexArray
    ) -> float:
        if target.config.mixing_policy == "fixed":
            return target.config.fixed_mixing
        return _finite_temperature_oda_parameter(target, delta)

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_update,
            energy_functional=energy_functional,
            oda_parameterizer=lambda target, delta: mixing_parameter(
                target, delta  # type: ignore[arg-type]
            ),
            final_acceptance=final_acceptance,
            convergence_metric=convergence_metric,
            convergence_rule="raw",
        ),
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
            "local unrestricted TiSe2 HF did not pass the generic SCF/final gates: "
            f"exit_reason={run.exit_reason}, final_raw_norm="
            f"{state.diagnostics.get('final_raw_norm')}"
        )

    density_hartree = local_full_hartree_action(state.density, coulomb=state.coulomb)
    density_fock = local_full_fock_action(
        state.density,
        coulomb=state.coulomb,
        workers=state.config.interaction_workers,
    )
    interaction_h = density_hartree + density_fock
    total_h = state.h0 + interaction_h
    final = _spectrum_for_state(total_h, state)
    mixed_projector = state.reference_projector_stored + state.density
    raw_density = final.projector_stored - state.reference_projector_stored
    mixed_number = float(
        np.einsum(
            "aak,k->", mixed_projector, state.mesh.normalized_weights, optimize=True
        ).real
    )
    mixed_residual = mixed_number - state.config.electrons_per_local_momentum
    hermiticity = float(np.max(np.abs(total_h - total_h.conj().swapaxes(0, 1))))
    relative_energy = local_full_reference_relative_energy(
        interaction_h, state.h0, state.density, mesh=state.mesh
    )
    free_energy = energy_functional(interaction_h, state.h0, state.density)
    return LocalFullHFResult(
        run=run,
        h0_ev=state.h0.copy(),
        density_hartree_ev=density_hartree,
        density_fock_ev=density_fock,
        interaction_h_ev=interaction_h,
        total_hamiltonian_ev=total_h,
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
    )


__all__ = [
    "ELECTRONS_PER_LOCAL_MOMENTUM",
    "LOCAL_DIMENSION",
    "LocalFullHFConfig",
    "LocalFullHFResult",
    "LocalFullHFState",
    "LocalFullSpectrum",
    "build_local_full_hf_state",
    "build_local_full_seed_field",
    "diagonalize_local_full_finite_temperature",
    "initialize_local_full_hf_state",
    "local_full_fock_action",
    "local_full_h0",
    "local_full_hartree_action",
    "local_full_interaction_action",
    "local_full_interaction_energy",
    "local_full_reference_relative_energy",
    "local_full_reference_relative_free_energy",
    "run_local_full_hf",
]
