"""Unrestricted four-sector Hartree--Fock adapter for 1T-TiSe2.

The supplied primitive Hamiltonian ``H0(k)`` is a bare 4 by 4 noninteracting
Hamiltonian.  Four fixed sector representatives ``Q_s`` produce a complete
16 by 16 folded Hamiltonian at every reduced momentum.  No Hamiltonian or
density block is masked.

Basis and density conventions
-----------------------------
The basis is ``|s,a;p>`` with sector ``s=0,...,3`` and internal state
``a=(v,c1,c2,c3)``; ``I=4*s+a``.  Stored densities have orientation
``P[I,J,p]=<c_I^dagger c_J>``.  The noninteracting finite-temperature
reference is ``Pref=f_beta(H0-mu_ref)`` using one global chemical potential
and four electrons per reduced momentum *on average*.  All sixteen energies
and eigenvectors are retained.  The default temperature is 65 K.

The physical mean-field Hamiltonian is

``H_HF[P] = H0 + Sigma_H[P-Pref] + Sigma_F[P]``.

For ``D=P-Pref`` this is evaluated in the affine ODA form

``h0_eff = H0 + Sigma_F[Pref]`` and
``L[D] = Sigma_H[D] + Sigma_F[D]``.

The matching reference-relative free energy is

``<h0_eff,D> + 1/2 <L[D],D> + kBT (Phi[Pref+D]-Phi[Pref])``,

where ``Phi[P]=Tr[P log P + (1-P) log(1-P)]``.  The q=0 Hartree
coefficient is removed as the neutral background; global canonical filling
makes its integrated source vanish even though individual k-point traces need
not vanish.

Finite reciprocal-route contract
--------------------------------
The exact doubled fractional labels are ``2*Q_s``.  With only the four fixed
representatives supplied, the route policy is the finite ``G=0`` model: a
sector quartet ``(s,t,u,v)`` is retained only when
``Q_s+Q_u-Q_t-Q_v=0``, tested by requiring the doubled integer residual to be
exactly zero.  The fixed labels yield exactly 28 retained quartets: four for
each sector-diagonal target block and one for each off-diagonal target block.
Unit form factors are assigned to every retained quartet.  Fock uses the
single physical transfer ``V(p-p'+Q_s-Q_v)`` and Hartree uses
``V(Q_s-Q_t)``; reverse routes guarantee Hermiticity.  Adding Umklapp routes
requires a supplied G-resolved inventory and must not be emulated by modulo-G
routing or by averaging inequivalent Coulomb transfers.

This module remains separate from :mod:`mean_field.systems.tise2.hf`, whose
restricted Monney decoupling is diagnostic only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
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

from .hf import (
    BOLTZMANN_EV_PER_K,
    COULOMB_EV_ANGSTROM,
    MonneyLocalMesh,
    build_monney_local_mesh,
)
from .monney import HBAR2_OVER_2ME_EV_A2, MonneyTiSe2Parameters

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

N_SECTORS = 4
N_INTERNAL = 4
FOLDED_DIMENSION = N_SECTORS * N_INTERNAL
ELECTRONS_PER_REDUCED_MOMENTUM = 4.0
# Compatibility name: this is now an average global target, not a rank per k.
FILLING_PER_REDUCED_MOMENTUM = ELECTRONS_PER_REDUCED_MOMENTUM
Q_FRACTIONAL = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (-0.5, -0.5, 0.5),
    ),
    dtype=np.float64,
)
DOUBLED_Q_LABELS = np.asarray(
    ((0, 0, 0), (1, 0, 1), (0, 1, 1), (-1, -1, 1)),
    dtype=np.int64,
)


def _validate_outer_eigensolver_environment(eigensolver_workers: int) -> None:
    """Match the core outer-worker validation without importing a private API."""

    if type(eigensolver_workers) is not int or eigensolver_workers <= 0:
        raise TypeError("eigensolver_workers must be an exact positive integer")
    if eigensolver_workers == 1:
        return
    invalid = {
        name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(
            "outer eigensolver parallelism requires explicit one-thread "
            f"BLAS/OpenMP binding; invalid environment: {invalid}"
        )
    for name in ("MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        value = os.environ.get(name)
        if value not in (None, "1"):
            raise RuntimeError(
                "outer eigensolver parallelism forbids nested numerical "
                f"threads; {name}={value!r}"
            )

@dataclass(frozen=True)
class FoldedTiSe2Geometry:
    """Hexagonal lattice data used to define the fixed Q representatives."""

    a_angstrom: float = 3.54
    c_angstrom: float = 6.008
    provenance: str = (
        "external 1T-TiSe2 lattice constants; not Monney effective-mass fit parameters"
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.a_angstrom) or self.a_angstrom <= 0.0:
            raise ValueError("a_angstrom must be finite and positive")
        if not np.isfinite(self.c_angstrom) or self.c_angstrom <= 0.0:
            raise ValueError("c_angstrom must be finite and positive")
        if not self.provenance:
            raise ValueError("geometry provenance must be nonempty")

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray:
        scale = 2.0 * np.pi / self.a_angstrom
        return np.asarray(
            (
                (2.0 * scale / np.sqrt(3.0), 0.0, 0.0),
                (-scale / np.sqrt(3.0), scale, 0.0),
                (0.0, 0.0, 2.0 * np.pi / self.c_angstrom),
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class FoldedCoulombKernel:
    """Finite fixed-representative Coulomb data.

    ``shifted_weighted_ev[s,v,k,kp]`` is
    ``w[kp] V(p[k]-p[kp]+Q[s]-Q[v])`` and is the single Fock transfer for a
    retained exact-``G=0`` quartet.  ``hartree_ev[s,t]`` is
    ``V(Q[s]-Q[t])`` with its diagonal set to zero by the neutral-background
    policy.
    """

    shifted_weighted_ev: FloatArray
    hartree_ev: FloatArray
    q_vectors_Ainv: FloatArray
    mesh: MonneyLocalMesh
    epsilon_r: float
    self_cell_ev: float
    estimated_peak_bytes: int
    route_count: int = 28
    route_policy: Literal["fixed_representative_g0"] = "fixed_representative_g0"
    q0_hartree_policy: Literal["neutral_background_global_canonical"] = (
        "neutral_background_global_canonical"
    )

    def __post_init__(self) -> None:
        shifted = np.asarray(self.shifted_weighted_ev, dtype=np.float64)
        hartree = np.asarray(self.hartree_ev, dtype=np.float64)
        q_vectors = _validate_q_vectors(self.q_vectors_Ainv)
        expected = (N_SECTORS, N_SECTORS, self.mesh.nk, self.mesh.nk)
        if shifted.shape != expected:
            raise ValueError(
                f"shifted_weighted_ev must have shape {expected}, got {shifted.shape}"
            )
        if hartree.shape != (N_SECTORS, N_SECTORS):
            raise ValueError("hartree_ev must have shape (4,4)")
        if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0.0):
            raise ValueError("shifted Coulomb kernels must be finite and positive")
        if not np.all(np.isfinite(hartree)) or np.any(hartree < 0.0):
            raise ValueError("Hartree coefficients must be finite and nonnegative")
        if not np.array_equal(np.diag(hartree), np.zeros(N_SECTORS)):
            raise ValueError("q=0 Hartree coefficients must be exactly zero")
        if not np.isfinite(self.epsilon_r) or self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be finite and positive")
        if not np.isfinite(self.self_cell_ev) or self.self_cell_ev <= 0.0:
            raise ValueError("self_cell_ev must be finite and positive")
        if type(self.estimated_peak_bytes) is not int or self.estimated_peak_bytes <= 0:
            raise TypeError("estimated_peak_bytes must be an exact positive integer")
        if self.route_count != 28:
            raise ValueError("the finite fixed-representative G=0 inventory must contain 28 routes")
        if self.route_policy != "fixed_representative_g0":
            raise ValueError("route_policy must be 'fixed_representative_g0'")
        object.__setattr__(self, "shifted_weighted_ev", shifted)
        object.__setattr__(self, "hartree_ev", hartree)
        object.__setattr__(self, "q_vectors_Ainv", q_vectors)


@dataclass(frozen=True)
class FoldedSpectrum:
    """Complete 16-state eigensystem and global finite-T density."""

    energies_ev: FloatArray
    eigenvectors: ComplexArray
    occupations: FloatArray
    projector_stored: ComplexArray
    chemical_potential_ev: float
    particle_number_per_reduced_momentum: float
    particle_residual: float
    entropy_per_reduced_momentum: float


@dataclass(frozen=True)
class FoldedDerivativeOracleResult:
    direction_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    passed: bool
    finite_difference_step: float
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class FoldedHFConfig:
    """Explicit regulator, thermal, and solver inputs."""

    epsilon_r: float
    temperature_K: float = 65.0
    electrons_per_reduced_momentum: float = ELECTRONS_PER_REDUCED_MOMENTUM
    inplane_shells: int = 1
    inplane_spacing_Ainv: float = 0.040
    z_shells: int = 1
    z_spacing_Ainv: float = 0.050
    precision: float = 1.0e-9
    max_iter: int = 300
    max_dense_memory_gb: float = 1.0
    random_initial_rotation: float = 0.10
    eigensolver_workers: int = 1
    oda_grid_points: int = 17
    screening_provenance: str = "explicit constant-epsilon model input"
    regulator_provenance: str = "C3-closed local hexagonal-prism quadrature"
    route_policy: Literal["fixed_representative_g0"] = "fixed_representative_g0"

    def __post_init__(self) -> None:
        scalars = (
            self.epsilon_r,
            self.temperature_K,
            self.electrons_per_reduced_momentum,
            self.inplane_spacing_Ainv,
            self.z_spacing_Ainv,
            self.precision,
            self.max_dense_memory_gb,
            self.random_initial_rotation,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("FoldedHFConfig scalar inputs must be finite")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if self.electrons_per_reduced_momentum != 4.0:
            raise ValueError("the folded TiSe2 contract fixes four electrons on average")
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
        if not 0.0 <= self.random_initial_rotation <= 1.0:
            raise ValueError("random_initial_rotation must lie in [0,1]")
        if type(self.eigensolver_workers) is not int or self.eigensolver_workers <= 0:
            raise TypeError("eigensolver_workers must be an exact positive integer")
        if type(self.oda_grid_points) is not int or self.oda_grid_points < 5:
            raise TypeError("oda_grid_points must be an exact integer at least five")
        if not self.screening_provenance or not self.regulator_provenance:
            raise ValueError("assumption provenance strings must be nonempty")
        if self.route_policy != "fixed_representative_g0":
            raise ValueError("route_policy must be 'fixed_representative_g0'")

    @property
    def kbt_ev(self) -> float:
        return float(BOLTZMANN_EV_PER_K * self.temperature_K)


@dataclass
class FoldedHFState:
    # Generic engine sees the affine one-body term as h0.
    h0: ComplexArray
    bare_h0_ev: ComplexArray
    reference_fock_ev: ComplexArray
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
    config: FoldedHFConfig

    @property
    def nk(self) -> int:
        return self.mesh.nk

    @property
    def h0_effective_ev(self) -> ComplexArray:
        return self.h0


@dataclass(frozen=True)
class FoldedHFResult:
    run: HartreeFockRun
    bare_h0_ev: ComplexArray
    reference_fock_ev: ComplexArray
    h0_effective_ev: ComplexArray
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

    @property
    def projector_stored(self) -> ComplexArray:
        """Eigensystem-associated projector from the final raw density update."""

        return self.raw_update_projector_stored


def _validate_points(points_Ainv: np.ndarray) -> FloatArray:
    points = np.asarray(points_Ainv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points_Ainv must be nonempty with shape (nk,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_Ainv must be finite")
    return points


def _validate_q_vectors(q_vectors_Ainv: np.ndarray) -> FloatArray:
    vectors = np.asarray(q_vectors_Ainv, dtype=np.float64)
    if vectors.shape != (N_SECTORS, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError("q_vectors_Ainv must be finite with shape (4,3)")
    if not np.array_equal(vectors[0], np.zeros(3)):
        raise ValueError("Q0 must be exactly zero")

    scale = max(1.0, float(np.max(np.abs(vectors))))
    zero_tolerance = 512.0 * np.finfo(np.float64).eps * scale
    zero = np.zeros(3, dtype=np.int64)
    for sector in range(N_SECTORS):
        for target_sector in range(N_SECTORS):
            for density_row_sector in range(N_SECTORS):
                for density_col_sector in range(N_SECTORS):
                    physical_residual = (
                        vectors[sector]
                        + vectors[density_row_sector]
                        - vectors[target_sector]
                        - vectors[density_col_sector]
                    )
                    physical_is_zero = bool(
                        np.max(np.abs(physical_residual)) <= zero_tolerance
                    )
                    label_residual = (
                        DOUBLED_Q_LABELS[sector]
                        + DOUBLED_Q_LABELS[density_row_sector]
                        - DOUBLED_Q_LABELS[target_sector]
                        - DOUBLED_Q_LABELS[density_col_sector]
                    )
                    label_is_zero = bool(np.array_equal(label_residual, zero))
                    if physical_is_zero != label_is_zero:
                        quartet = (
                            sector,
                            target_sector,
                            density_row_sector,
                            density_col_sector,
                        )
                        raise ValueError(
                            "q_vectors_Ainv zero-residual routes do not match "
                            "DOUBLED_Q_LABELS for sector quartet "
                            f"(s,t,u,v)={quartet}"
                        )
    return vectors


def _validate_density(density_stored: np.ndarray, nk: int) -> ComplexArray:
    density = np.asarray(density_stored, dtype=np.complex128)
    expected = (FOLDED_DIMENSION, FOLDED_DIMENSION, nk)
    if density.shape != expected or not np.all(np.isfinite(density)):
        raise ValueError(f"density_stored must be finite with shape {expected}")
    scale = max(1.0, float(np.max(np.abs(density))))
    residual = float(np.max(np.abs(density - density.conj().swapaxes(0, 1))))
    if residual > 128.0 * np.finfo(float).eps * scale:
        raise ValueError(f"density_stored must be Hermitian; residual={residual:.3e}")
    return density


def _validate_hermitian_action(array: np.ndarray, *, name: str) -> None:
    scale = max(1.0, float(np.max(np.abs(array))))
    residual = float(np.max(np.abs(array - array.conj().swapaxes(0, 1))))
    if residual > 2.0e-11 * scale:
        raise ValueError(f"{name} is not Hermitian; residual={residual:.3e}")


def physical_q_vectors(
    geometry: FoldedTiSe2Geometry | None = None,
) -> FloatArray:
    lattice = FoldedTiSe2Geometry() if geometry is None else geometry
    vectors = Q_FRACTIONAL @ lattice.reciprocal_basis_Ainv
    vectors[0] = 0.0
    return _validate_q_vectors(vectors)


def global_primitive_h0(
    points_Ainv: np.ndarray,
    *,
    params: MonneyTiSe2Parameters | None = None,
    q_vectors_Ainv: np.ndarray | None = None,
    geometry: FoldedTiSe2Geometry | None = None,
) -> ComplexArray:
    """Return the bare global primitive ``H0(k)`` with shape ``(4,4,nk)``."""

    points = _validate_points(points_Ainv)
    model = MonneyTiSe2Parameters() if params is None else params
    lattice = FoldedTiSe2Geometry(c_angstrom=model.c_angstrom) if geometry is None else geometry
    if not np.isclose(lattice.c_angstrom, model.c_angstrom, rtol=0.0, atol=1.0e-12):
        raise ValueError("geometry.c_angstrom must equal params.c_angstrom")
    q_vectors = physical_q_vectors(lattice) if q_vectors_Ainv is None else _validate_q_vectors(q_vectors_Ainv)

    h0 = np.zeros((N_INTERNAL, N_INTERNAL, points.shape[0]), dtype=np.complex128)
    px, py, pz = points.T
    h0[0, 0] = (
        HBAR2_OVER_2ME_EV_A2 * (px**2 + py**2) / model.m_v_over_me
        + model.t_v_ev * np.cos(np.pi * pz / model.k_gamma_a_Ainv)
        + model.epsilon_v0_ev
    )
    sign = 1.0 if model.conduction_z_convention == "paper_literal" else -1.0
    for internal in range(1, N_INTERNAL):
        local = points - q_vectors[internal]
        qx, qy, qz = local.T
        angle = model.valley_angles_rad[internal - 1]
        c, s = np.cos(angle), np.sin(angle)
        q_long = c * qx + s * qy
        q_short = -s * qx + c * qy
        h0[internal, internal] = (
            HBAR2_OVER_2ME_EV_A2
            * (q_long**2 / model.m_long_over_me + q_short**2 / model.m_short_over_me)
            + sign * model.t_c_ev * np.cos(np.pi * qz / model.k_gamma_a_Ainv)
            + model.epsilon_c0_ev
        )
    return h0


def folded_h0(
    reduced_points_Ainv: np.ndarray,
    *,
    params: MonneyTiSe2Parameters | None = None,
    q_vectors_Ainv: np.ndarray | None = None,
    geometry: FoldedTiSe2Geometry | None = None,
) -> ComplexArray:
    """Return ``direct_sum_s H0(p+Q_s)`` with shape ``(16,16,nk)``."""

    points = _validate_points(reduced_points_Ainv)
    model = MonneyTiSe2Parameters() if params is None else params
    lattice = FoldedTiSe2Geometry(c_angstrom=model.c_angstrom) if geometry is None else geometry
    q_vectors = physical_q_vectors(lattice) if q_vectors_Ainv is None else _validate_q_vectors(q_vectors_Ainv)
    result = np.zeros((FOLDED_DIMENSION, FOLDED_DIMENSION, points.shape[0]), dtype=np.complex128)
    for sector in range(N_SECTORS):
        block = global_primitive_h0(
            points + q_vectors[sector],
            params=model,
            q_vectors_Ainv=q_vectors,
            geometry=lattice,
        )
        start = N_INTERNAL * sector
        result[start : start + N_INTERNAL, start : start + N_INTERNAL] = block
    return result


def physical_momentum_routes() -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Return the finite 28-quartet fixed-representative ``G=0`` inventory.

    For target block ``(s,t)``, each returned ``(u,v)`` satisfies the exact
    doubled-integer equality
    ``labels[s]+labels[u]-labels[t]-labels[v] == 0``.  A diagonal target block
    has four routes and an off-diagonal block has one.  Umklapp routes require
    a separately supplied G-resolved inventory; modulo-G routing is excluded.
    """

    all_routes: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for sector in range(N_SECTORS):
        target_row: list[tuple[tuple[int, int], ...]] = []
        for target_sector in range(N_SECTORS):
            routes: list[tuple[int, int]] = []
            for density_row_sector in range(N_SECTORS):
                for density_col_sector in range(N_SECTORS):
                    residual = (
                        DOUBLED_Q_LABELS[sector]
                        + DOUBLED_Q_LABELS[density_row_sector]
                        - DOUBLED_Q_LABELS[target_sector]
                        - DOUBLED_Q_LABELS[density_col_sector]
                    )
                    if np.array_equal(residual, np.zeros(3, dtype=np.int64)):
                        routes.append((density_row_sector, density_col_sector))
            expected_count = 4 if sector == target_sector else 1
            if len(routes) != expected_count:
                raise RuntimeError("unexpected fixed-representative G=0 route count")
            target_row.append(tuple(routes))
        all_routes.append(tuple(target_row))
    inventory = tuple(all_routes)
    if sum(len(routes) for row in inventory for routes in row) != 28:
        raise RuntimeError("unexpected finite fixed-representative G=0 route count")
    return inventory


def estimate_folded_coulomb_peak_bytes(mesh: MonneyLocalMesh) -> int:
    """Conservative peak estimate for the streamed shifted-kernel precompute."""

    n2 = mesh.nk * mesh.nk
    output = N_SECTORS**2 * n2 * np.dtype(np.float64).itemsize
    # One sector-pair at a time: 3-vector difference, q2, masks, ufunc/einsum
    # workspaces, plus fixed arrays and object overhead.  Eight float arrays is
    # deliberately above the observed live temporary inventory.
    streamed_work = 8 * n2 * np.dtype(np.float64).itemsize
    fixed_overhead = 4096 + (N_SECTORS**2 + 3 * N_SECTORS) * 8
    return int(output + streamed_work + fixed_overhead)


def precompute_folded_coulomb_kernel(
    mesh: MonneyLocalMesh,
    *,
    q_vectors_Ainv: np.ndarray,
    epsilon_r: float,
    max_dense_memory_gb: float = 1.0,
    route_policy: Literal["fixed_representative_g0"] = "fixed_representative_g0",
) -> FoldedCoulombKernel:
    """Precompute fixed-representative kernels with conservative accounting.

    Only one ``(s,v)`` temporary transfer array is live at a time.  The full
    ``(nk,nk,3)`` point-difference array is not retained alongside another
    transfer copy.  The cap is checked before the output allocation.
    """

    q_vectors = _validate_q_vectors(q_vectors_Ainv)
    eps = float(epsilon_r)
    cap = float(max_dense_memory_gb)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon_r must be finite and positive")
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("max_dense_memory_gb must be finite and positive")
    if route_policy != "fixed_representative_g0":
        raise ValueError("route_policy must be 'fixed_representative_g0'")
    estimated_peak = estimate_folded_coulomb_peak_bytes(mesh)
    if estimated_peak > int(cap * 1.0e9):
        raise MemoryError(
            "folded shifted-Coulomb precompute exceeds the conservative "
            f"peak-memory cap: estimate={estimated_peak} bytes"
        )

    cell_radius = float((3.0 * mesh.cell_volume_Ainv3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    self_cell_ev = float(2.0 * COULOMB_EV_ANGSTROM * cell_radius / (np.pi * eps))
    prefactor = 4.0 * np.pi * COULOMB_EV_ANGSTROM / eps
    shifted = np.empty((N_SECTORS, N_SECTORS, mesh.nk, mesh.nk), dtype=np.float64)
    weighted_source = np.broadcast_to(mesh.weights_Ainv3[None, :], (mesh.nk, mesh.nk))
    for sector in range(N_SECTORS):
        for density_col_sector in range(N_SECTORS):
            transfer = mesh.points_Ainv[:, None, :] - mesh.points_Ainv[None, :, :]
            transfer += q_vectors[sector] - q_vectors[density_col_sector]
            q2 = np.einsum("...d,...d->...", transfer, transfer, optimize=True)
            block = shifted[sector, density_col_sector]
            singular = q2 <= 64.0 * np.finfo(float).eps
            np.divide(weighted_source, q2, out=block, where=~singular)
            block *= prefactor
            block[singular] = self_cell_ev

    hartree = np.zeros((N_SECTORS, N_SECTORS), dtype=np.float64)
    for sector in range(N_SECTORS):
        for target_sector in range(N_SECTORS):
            if sector == target_sector:
                continue
            q = q_vectors[sector] - q_vectors[target_sector]
            hartree[sector, target_sector] = prefactor / float(np.dot(q, q))
    return FoldedCoulombKernel(
        shifted_weighted_ev=shifted,
        hartree_ev=hartree,
        q_vectors_Ainv=q_vectors,
        mesh=mesh,
        epsilon_r=eps,
        self_cell_ev=self_cell_ev,
        estimated_peak_bytes=estimated_peak,
        route_policy=route_policy,
    )


def folded_fock_action(
    density_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
) -> ComplexArray:
    r"""Return ``Sigma_F[P]`` (or the same linear map applied to ``D``).

    For each retained exact-``G=0`` quartet, the coefficient is the single
    fixed-representative transfer

    ``-V(p-p'+Qs-Qv) P[ub,va,p']``.

    Exact sector conservation makes ``Qs-Qv == Qt-Qu``; reverse routes provide
    the Hermitian-conjugate contribution without averaging transfers.
    """

    density = _validate_density(density_stored, coulomb.mesh.nk)
    routes = physical_momentum_routes()
    sigma = np.zeros_like(density)
    for sector in range(N_SECTORS):
        row = slice(N_INTERNAL * sector, N_INTERNAL * (sector + 1))
        for target_sector in range(N_SECTORS):
            col = slice(N_INTERNAL * target_sector, N_INTERNAL * (target_sector + 1))
            block = sigma[row, col]
            for density_row_sector, density_col_sector in routes[sector][target_sector]:
                density_row = slice(N_INTERNAL * density_row_sector, N_INTERNAL * (density_row_sector + 1))
                density_col = slice(N_INTERNAL * density_col_sector, N_INTERNAL * (density_col_sector + 1))
                source = density[density_row, density_col].swapaxes(0, 1)
                kernel = coulomb.shifted_weighted_ev[sector, density_col_sector]
                block -= np.einsum("kp,abp->abk", kernel, source, optimize=True)
    _validate_hermitian_action(sigma, name="Fock action")
    return sigma


def folded_hartree_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
) -> ComplexArray:
    r"""Return ``Sigma_H[D]`` with only the global q=0 background removed."""

    density = _validate_density(density_delta_stored, coulomb.mesh.nk)
    routes = physical_momentum_routes()
    sigma = np.zeros_like(density)
    weights = coulomb.mesh.weights_Ainv3
    eye = np.eye(N_INTERNAL, dtype=np.complex128)
    for sector in range(N_SECTORS):
        row = slice(N_INTERNAL * sector, N_INTERNAL * (sector + 1))
        for target_sector in range(N_SECTORS):
            col = slice(N_INTERNAL * target_sector, N_INTERNAL * (target_sector + 1))
            for density_row_sector, density_col_sector in routes[sector][target_sector]:
                coefficient = coulomb.hartree_ev[sector, target_sector]
                if coefficient == 0.0:
                    continue
                density_row = slice(N_INTERNAL * density_row_sector, N_INTERNAL * (density_row_sector + 1))
                density_col = slice(N_INTERNAL * density_col_sector, N_INTERNAL * (density_col_sector + 1))
                rho = np.einsum(
                    "aak,k->", density[density_row, density_col], weights, optimize=True
                )
                sigma[row, col, :] += coefficient * rho * eye[:, :, None]
    _validate_hermitian_action(sigma, name="Hartree action")
    return sigma


def folded_interaction_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
) -> ComplexArray:
    """Return the affine-density map ``L[D]=Sigma_H[D]+Sigma_F[D]``."""

    return folded_hartree_action(density_delta_stored, coulomb=coulomb) + folded_fock_action(
        density_delta_stored, coulomb=coulomb
    )


def folded_interaction_energy(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
) -> float:
    density = _validate_density(density_delta_stored, coulomb.mesh.nk)
    sigma = folded_interaction_action(density, coulomb=coulomb)
    value = 0.5 * np.einsum(
        "abk,abk,k->", sigma, density, coulomb.mesh.weights_Ainv3, optimize=True
    )
    if abs(value.imag) > 2.0e-11 * max(1.0, abs(value.real)):
        raise ValueError(f"interaction energy is not real: {value!r}")
    return float(value.real)


def folded_reference_relative_energy(
    interaction_h_ev: np.ndarray,
    h0_effective_ev: np.ndarray,
    density_delta_stored: np.ndarray,
    *,
    mesh: MonneyLocalMesh,
) -> float:
    """Return ``<h0_eff,D>+1/2<L[D],D>`` in eV/Angstrom^3."""

    density = _validate_density(density_delta_stored, mesh.nk)
    sigma = np.asarray(interaction_h_ev, dtype=np.complex128)
    h0_eff = np.asarray(h0_effective_ev, dtype=np.complex128)
    if sigma.shape != density.shape or h0_eff.shape != density.shape:
        raise ValueError("interaction_h_ev, h0_effective_ev, and density must match")
    value = np.einsum("abk,abk,k->", h0_eff, density, mesh.weights_Ainv3, optimize=True)
    value += 0.5 * np.einsum(
        "abk,abk,k->", sigma, density, mesh.weights_Ainv3, optimize=True
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
    """Return physical-weight ``Phi`` through the public core diagnostics path."""

    projector = _validate_density(projector_stored, mesh.nk)
    projector_ket = stored_projector_to_conventional(projector)
    diagnostics = fermionic_density_diagnostics(
        np.moveaxis(projector_ket, 2, 0),
        k_weights=mesh.normalized_weights,
        spectrum_tolerance=2.0e-10,
        eigensolver_workers=eigensolver_workers,
    )
    # Core reports positive S/k_B with the supplied normalized weights, whereas
    # this energy-density functional uses Phi=-S/k_B and physical quadrature
    # weights. Restore that physical normalization without changing the scalar.
    physical_weight = float(np.sum(mesh.weights_Ainv3))
    return float(-physical_weight * diagnostics.entropy_dimensionless)


def folded_reference_relative_free_energy(
    interaction_h_ev: np.ndarray,
    h0_effective_ev: np.ndarray,
    density_delta_stored: np.ndarray,
    *,
    reference_projector_stored: np.ndarray,
    mesh: MonneyLocalMesh,
    kbt_ev: float,
    eigensolver_workers: int = 1,
) -> float:
    """Return the matching finite-T reference-relative Helmholtz functional."""

    density = _validate_density(density_delta_stored, mesh.nk)
    reference = _validate_density(reference_projector_stored, mesh.nk)
    temperature = float(kbt_ev)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("kbt_ev must be finite and positive")
    internal = folded_reference_relative_energy(
        interaction_h_ev, h0_effective_ev, density, mesh=mesh
    )
    phi = _entropy_phi_density(
        reference + density, mesh, eigensolver_workers=eigensolver_workers
    )
    reference_phi = _entropy_phi_density(
        reference, mesh, eigensolver_workers=eigensolver_workers
    )
    return float(internal + temperature * (phi - reference_phi))


def diagonalize_folded_finite_temperature(
    hamiltonian_abk: np.ndarray,
    *,
    kbt_ev: float,
    k_weights: np.ndarray | None = None,
    electrons_per_reduced_momentum: float = ELECTRONS_PER_REDUCED_MOMENTUM,
    particle_tolerance: float = 1.0e-12,
    eigensolver_workers: int = 1,
) -> FoldedSpectrum:
    """Diagonalize all 16 states and apply one global finite-T chemical potential."""

    hamiltonian = np.asarray(hamiltonian_abk, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[:2] != (FOLDED_DIMENSION, FOLDED_DIMENSION)
        or hamiltonian.shape[2] == 0
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("hamiltonian_abk must be finite with shape (16,16,nk)")
    _validate_hermitian_action(hamiltonian, name="folded Hamiltonian")
    temperature = float(kbt_ev)
    tolerance = float(particle_tolerance)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("kbt_ev must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("particle_tolerance must be finite and positive")
    target = float(electrons_per_reduced_momentum)
    if not np.isfinite(target) or not 0.0 < target < FOLDED_DIMENSION:
        raise ValueError("electrons_per_reduced_momentum must lie strictly inside (0,16)")

    nk = hamiltonian.shape[2]
    if k_weights is None:
        weights = np.full(nk, 1.0 / nk, dtype=np.float64)
    else:
        weights = np.asarray(k_weights, dtype=np.float64)
        if weights.shape != (nk,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("k_weights must be finite, positive, and have shape (nk,)")
        weights = weights / np.sum(weights)

    _validate_outer_eigensolver_environment(eigensolver_workers)
    hamiltonian_by_k = np.moveaxis(hamiltonian, 2, 0)

    # Reuse the generic core/hf global finite-T occupation solver. Its public
    # result intentionally omits eigenvectors, so this system adapter performs
    # one additional all-16-state diagonalization solely to preserve the
    # complete eigensystem requested by the folded-state archive contract.
    occupation = global_fermi_occupations(
        hamiltonian_by_k,
        target_particle_number=target,
        kbt_ev=temperature,
        k_weights=weights,
        particle_tolerance=tolerance,
        eigensolver_workers=eigensolver_workers,
    )
    if eigensolver_workers == 1:
        energies_by_k, eigenvectors = np.linalg.eigh(hamiltonian_by_k)
    else:
        energies_by_k = np.empty((nk, FOLDED_DIMENSION), dtype=np.float64)
        eigenvectors = np.empty(
            (nk, FOLDED_DIMENSION, FOLDED_DIMENSION), dtype=np.complex128
        )

        def diagonalize(k_index: int) -> tuple[int, FloatArray, ComplexArray]:
            values, vectors = np.linalg.eigh(hamiltonian_by_k[k_index])
            return k_index, values, vectors

        with ThreadPoolExecutor(max_workers=min(eigensolver_workers, nk)) as executor:
            for k_index, values, vectors in executor.map(diagonalize, range(nk)):
                energies_by_k[k_index] = values
                eigenvectors[k_index] = vectors
    energies = np.asarray(energies_by_k.T, dtype=np.float64)
    if not np.allclose(energies, occupation.energies, rtol=0.0, atol=2.0e-13):
        raise RuntimeError("preserved eigensystem energies disagree with core occupations")

    occupations = np.asarray(occupation.occupations, dtype=np.float64)
    density_from_eigensystem = (
        eigenvectors * occupations.T[:, None, :]
    ) @ eigenvectors.conj().transpose(0, 2, 1)
    density_scale = max(1.0, float(np.max(np.abs(occupation.density_ket))))
    density_tolerance = 4096.0 * np.finfo(np.float64).eps * density_scale
    density_residual = float(
        np.max(np.abs(density_from_eigensystem - occupation.density_ket))
    )
    if density_residual > density_tolerance:
        raise RuntimeError(
            "projector reconstructed from the preserved eigensystem disagrees "
            "with the core occupation density: "
            f"residual={density_residual:.3e}, tolerance={density_tolerance:.3e}"
        )
    projector_stored = conventional_projector_to_stored(
        np.moveaxis(density_from_eigensystem, 0, 2)
    )
    return FoldedSpectrum(
        energies_ev=energies,
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        occupations=occupations,
        projector_stored=np.asarray(projector_stored, dtype=np.complex128),
        chemical_potential_ev=float(occupation.chemical_potential),
        particle_number_per_reduced_momentum=float(occupation.particle_number),
        particle_residual=float(occupation.particle_residual),
        entropy_per_reduced_momentum=float(occupation.entropy_dimensionless),
    )


def diagonalize_folded_fixed_filling(
    hamiltonian_abk: np.ndarray,
    *,
    kbt_ev: float = BOLTZMANN_EV_PER_K * 65.0,
    k_weights: np.ndarray | None = None,
    eigensolver_workers: int = 1,
) -> FoldedSpectrum:
    """Deprecated name for the global finite-T occupation path.

    It no longer imposes rank four at each k; the name is retained only to
    avoid a silent import break while callers migrate.
    """

    return diagonalize_folded_finite_temperature(
        hamiltonian_abk,
        kbt_ev=kbt_ev,
        k_weights=k_weights,
        eigensolver_workers=eigensolver_workers,
    )


def _entropy_gradient_stored(projector_stored: np.ndarray, mesh: MonneyLocalMesh) -> ComplexArray:
    projector = _validate_density(projector_stored, mesh.nk)
    result = np.empty_like(projector)
    for k_index in range(mesh.nk):
        values, vectors = np.linalg.eigh(projector[:, :, k_index])
        if np.min(values) <= 0.0 or np.max(values) >= 1.0:
            raise ValueError("entropy derivative requires a strictly interior density spectrum")
        logit = np.log(values) - np.log1p(-values)
        matrix_gradient = (vectors * logit[None, :]) @ vectors.conj().T
        # Stored densities use S_ab=<c_a^dagger c_b>=P_ket[b,a], while the
        # scalar derivative is paired elementwise as sum_ab G_ab dS_ab.
        # Therefore the matrix-calculus gradient f'(S) must be transposed.
        result[:, :, k_index] = matrix_gradient.T
    return result


def all_hermitian_direction_derivative_oracle(
    density_delta_stored: np.ndarray,
    *,
    coulomb: FoldedCoulombKernel,
    h0_effective_ev: np.ndarray | None = None,
    reference_projector_stored: np.ndarray | None = None,
    kbt_ev: float | None = None,
    finite_difference_step: float = 2.0e-6,
    absolute_tolerance: float = 2.0e-9,
    relative_tolerance: float = 2.0e-8,
) -> FoldedDerivativeOracleResult:
    """Check the scalar derivative on all ``nk*16^2`` Hermitian directions.

    With no optional arguments this checks the quadratic interaction.  With
    ``h0_effective_ev`` it checks the complete internal-energy derivative.
    Supplying both reference and ``kbt_ev`` additionally checks the relative
    entropy derivative.
    """

    density = _validate_density(density_delta_stored, coulomb.mesh.nk)
    step = float(finite_difference_step)
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite_difference_step must be finite and positive")
    if not np.isfinite(atol) or atol < 0.0 or not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("oracle tolerances must be finite and nonnegative")
    if (reference_projector_stored is None) != (kbt_ev is None):
        raise ValueError("reference_projector_stored and kbt_ev must be supplied together")

    sigma = folded_interaction_action(density, coulomb=coulomb)
    gradient = sigma.copy()
    h0_eff: ComplexArray | None = None
    if h0_effective_ev is not None:
        h0_eff = _validate_density(h0_effective_ev, coulomb.mesh.nk)
        gradient += h0_eff
    reference: ComplexArray | None = None
    temperature: float | None = None
    if reference_projector_stored is not None:
        reference = _validate_density(reference_projector_stored, coulomb.mesh.nk)
        temperature = float(kbt_ev)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("kbt_ev must be finite and positive")
        gradient += temperature * _entropy_gradient_stored(reference + density, coulomb.mesh)

    weights = coulomb.mesh.weights_Ainv3
    maximum_absolute = 0.0
    maximum_relative = 0.0
    all_passed = True
    count = 0
    direction = np.zeros_like(density)

    def scalar_at(trial: ComplexArray, trial_sigma: ComplexArray) -> float:
        if h0_eff is None:
            value = 0.5 * np.einsum(
                "abk,abk,k->", trial_sigma, trial, weights, optimize=True
            ).real
        else:
            value = folded_reference_relative_energy(
                trial_sigma, h0_eff, trial, mesh=coulomb.mesh
            )
        if reference is not None and temperature is not None:
            value += temperature * (
                _entropy_phi_density(reference + trial, coulomb.mesh)
                - _entropy_phi_density(reference, coulomb.mesh)
            )
        return float(value)

    def check_current_direction() -> None:
        nonlocal maximum_absolute, maximum_relative, all_passed, count
        sigma_direction = folded_interaction_action(direction, coulomb=coulomb)
        plus = scalar_at(density + step * direction, sigma + step * sigma_direction)
        minus = scalar_at(density - step * direction, sigma - step * sigma_direction)
        numerical = (plus - minus) / (2.0 * step)
        analytic_complex = np.einsum(
            "abk,abk,k->", gradient, direction, weights, optimize=True
        )
        analytic = float(analytic_complex.real)
        error = max(abs(numerical - analytic), abs(float(analytic_complex.imag)))
        scale = max(abs(numerical), abs(analytic), np.finfo(float).tiny)
        maximum_absolute = max(maximum_absolute, error)
        maximum_relative = max(maximum_relative, error / scale)
        all_passed = all_passed and error <= atol + rtol * scale
        count += 1

    normalization = 1.0 / np.sqrt(2.0)
    for k_index in range(coulomb.mesh.nk):
        for row in range(FOLDED_DIMENSION):
            direction.fill(0.0)
            direction[row, row, k_index] = 1.0
            check_current_direction()
        for row in range(FOLDED_DIMENSION):
            for col in range(row + 1, FOLDED_DIMENSION):
                direction.fill(0.0)
                direction[row, col, k_index] = normalization
                direction[col, row, k_index] = normalization
                check_current_direction()
                direction.fill(0.0)
                direction[row, col, k_index] = 1j * normalization
                direction[col, row, k_index] = -1j * normalization
                check_current_direction()

    return FoldedDerivativeOracleResult(
        direction_count=count,
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
        passed=bool(all_passed),
        finite_difference_step=step,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )


def _spectrum_for_state(hamiltonian: ComplexArray, state: FoldedHFState) -> FoldedSpectrum:
    return diagonalize_folded_finite_temperature(
        hamiltonian,
        kbt_ev=state.config.kbt_ev,
        k_weights=state.mesh.normalized_weights,
        electrons_per_reduced_momentum=state.config.electrons_per_reduced_momentum,
        eigensolver_workers=state.config.eigensolver_workers,
    )


def build_folded_hf_state(
    config: FoldedHFConfig,
    *,
    params: MonneyTiSe2Parameters | None = None,
    geometry: FoldedTiSe2Geometry | None = None,
) -> FoldedHFState:
    """Build ``Pref``, ``Sigma_F[Pref]``, and the affine SCF state."""

    model = MonneyTiSe2Parameters() if params is None else params
    lattice = FoldedTiSe2Geometry(c_angstrom=model.c_angstrom) if geometry is None else geometry
    mesh = build_monney_local_mesh(
        inplane_shells=config.inplane_shells,
        inplane_spacing_Ainv=config.inplane_spacing_Ainv,
        z_shells=config.z_shells,
        z_spacing_Ainv=config.z_spacing_Ainv,
    )
    q_vectors = physical_q_vectors(lattice)
    bare_h0 = folded_h0(
        mesh.points_Ainv, params=model, q_vectors_Ainv=q_vectors, geometry=lattice
    )
    reference = diagonalize_folded_finite_temperature(
        bare_h0,
        kbt_ev=config.kbt_ev,
        k_weights=mesh.normalized_weights,
        electrons_per_reduced_momentum=config.electrons_per_reduced_momentum,
        eigensolver_workers=config.eigensolver_workers,
    )
    coulomb = precompute_folded_coulomb_kernel(
        mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=config.epsilon_r,
        max_dense_memory_gb=config.max_dense_memory_gb,
        route_policy=config.route_policy,
    )
    reference_fock = folded_fock_action(reference.projector_stored, coulomb=coulomb)
    h0_effective = bare_h0 + reference_fock
    _validate_hermitian_action(h0_effective, name="h0_eff")
    return FoldedHFState(
        h0=h0_effective,
        bare_h0_ev=bare_h0,
        reference_fock_ev=reference_fock,
        density=np.zeros_like(bare_h0),
        hamiltonian=h0_effective.copy(),
        energies=reference.energies_ev.copy(),
        mu=reference.chemical_potential_ev,
        precision=float(config.precision),
        diagnostics={"reference_particle_residual": abs(reference.particle_residual)},
        mesh=mesh,
        coulomb=coulomb,
        reference_projector_stored=reference.projector_stored,
        reference_occupations=reference.occupations,
        q_vectors_Ainv=q_vectors,
        params=model,
        geometry=lattice,
        config=config,
    )


def _finite_temperature_oda_parameter(
    state: FoldedHFState,
    delta_density: ComplexArray,
) -> float:
    """Minimize the matching free energy on the closed ODA line segment."""

    previous = state.density.copy()
    base_interaction = state.hamiltonian - state.h0
    delta_interaction = folded_interaction_action(delta_density, coulomb=state.coulomb)

    def objective(value: float) -> float:
        lam = float(value)
        density = previous + lam * delta_density
        interaction = base_interaction + lam * delta_interaction
        return folded_reference_relative_free_energy(
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
    candidates: list[tuple[float, float]] = [(float(values[0]), 0.0), (float(values[-1]), 1.0)]
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


def run_folded_hf(
    state: FoldedHFState,
    *,
    init_mode: Literal["normal", "random"] = "normal",
    seed: int = 0,
) -> FoldedHFResult:
    """Run the generic engine with the affine full-Fock finite-T functional."""

    if init_mode not in {"normal", "random"}:
        raise ValueError("init_mode must be 'normal' or 'random'")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")

    def density_update(hamiltonian: ComplexArray) -> DensityUpdateResult:
        spectrum = _spectrum_for_state(hamiltonian, state)
        delta = spectrum.projector_stored - state.reference_projector_stored
        return DensityUpdateResult(
            density=np.asarray(delta, dtype=np.complex128),
            energies=spectrum.energies_ev,
            mu=spectrum.chemical_potential_ev,
            observables={
                "eigenvectors": spectrum.eigenvectors,
                "projector_stored": spectrum.projector_stored,
                "occupations": spectrum.occupations,
                "particle_residual": spectrum.particle_residual,
                "entropy_per_reduced_momentum": spectrum.entropy_per_reduced_momentum,
            },
        )

    def initializer(target: FoldedHFState, *, init_mode: str, seed: int) -> None:
        if init_mode == "normal":
            target.density.fill(0.0)
        else:
            rng = np.random.default_rng(seed)
            random_projector = np.empty_like(target.density)
            for k_index in range(target.nk):
                sampled = rng.normal(
                    size=(FOLDED_DIMENSION, FOLDED_DIMENSION)
                ) + 1j * rng.normal(size=(FOLDED_DIMENSION, FOLDED_DIMENSION))
                unitary, _ = np.linalg.qr(sampled)
                occupied = unitary[:, : int(ELECTRONS_PER_REDUCED_MOMENTUM)]
                random_projector[:, :, k_index] = (occupied @ occupied.conj().T).T
            target.density[:, :, :] = target.config.random_initial_rotation * (
                random_projector - target.reference_projector_stored
            )
        update = density_update(
            target.h0 + folded_interaction_action(target.density, coulomb=target.coulomb)
        )
        target.energies[:, :] = update.energies
        target.mu = float(update.mu)

    def interaction_builder(density: ComplexArray) -> ComplexArray:
        return folded_interaction_action(density, coulomb=state.coulomb)

    def energy_functional(
        interaction_h: ComplexArray,
        h0_effective: ComplexArray,
        density: ComplexArray,
    ) -> float:
        return folded_reference_relative_free_energy(
            interaction_h,
            h0_effective,
            density,
            reference_projector_stored=state.reference_projector_stored,
            mesh=state.mesh,
            kbt_ev=state.config.kbt_ev,
            eigensolver_workers=state.config.eigensolver_workers,
        )

    def convergence_metric(updated: np.ndarray, previous: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(updated) - np.asarray(previous))))

    def final_acceptance(
        target: FoldedHFState,
        update: DensityUpdateResult,
        final_norm: float,
    ) -> FinalAcceptanceResult:
        hermiticity = float(
            np.max(np.abs(target.hamiltonian - target.hamiltonian.conj().swapaxes(0, 1)))
        )
        particle_residual = abs(float(update.observables["particle_residual"]))
        accepted = (
            hermiticity <= 1.0e-10
            and particle_residual <= 2.0e-11
            and final_norm <= target.precision
        )
        return FinalAcceptanceResult(
            accepted=bool(accepted),
            reason="accepted" if accepted else "folded_physical_final_gate",
            diagnostics={
                "final_hermiticity_residual_ev": hermiticity,
                "final_particle_residual": particle_residual,
            },
        )

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_update,
            energy_functional=energy_functional,
            oda_parameterizer=lambda target, delta: _finite_temperature_oda_parameter(
                target, delta
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
            "folded TiSe2 HF did not pass the generic SCF/final-acceptance gates: "
            f"exit_reason={run.exit_reason}, final_raw_norm="
            f"{state.diagnostics.get('final_raw_norm')}"
        )

    density_hartree = folded_hartree_action(state.density, coulomb=state.coulomb)
    density_fock = folded_fock_action(state.density, coulomb=state.coulomb)
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
    mixed_residual = mixed_number - state.config.electrons_per_reduced_momentum
    hermiticity = float(np.max(np.abs(total_h - total_h.conj().swapaxes(0, 1))))
    relative_energy = folded_reference_relative_energy(
        interaction_h, state.h0, state.density, mesh=state.mesh
    )
    free_energy = energy_functional(interaction_h, state.h0, state.density)
    return FoldedHFResult(
        run=run,
        bare_h0_ev=state.bare_h0_ev.copy(),
        reference_fock_ev=state.reference_fock_ev.copy(),
        h0_effective_ev=state.h0.copy(),
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
    )


__all__ = [
    "DOUBLED_Q_LABELS",
    "ELECTRONS_PER_REDUCED_MOMENTUM",
    "FILLING_PER_REDUCED_MOMENTUM",
    "FOLDED_DIMENSION",
    "FoldedCoulombKernel",
    "FoldedDerivativeOracleResult",
    "FoldedHFConfig",
    "FoldedHFResult",
    "FoldedHFState",
    "FoldedSpectrum",
    "FoldedTiSe2Geometry",
    "Q_FRACTIONAL",
    "all_hermitian_direction_derivative_oracle",
    "build_folded_hf_state",
    "diagonalize_folded_finite_temperature",
    "diagonalize_folded_fixed_filling",
    "estimate_folded_coulomb_peak_bytes",
    "folded_fock_action",
    "folded_h0",
    "folded_hartree_action",
    "folded_interaction_action",
    "folded_interaction_energy",
    "folded_reference_relative_energy",
    "folded_reference_relative_free_energy",
    "global_primitive_h0",
    "physical_momentum_routes",
    "physical_q_vectors",
    "precompute_folded_coulomb_kernel",
    "run_folded_hf",
]
