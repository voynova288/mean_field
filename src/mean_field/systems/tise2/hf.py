"""Numerical excitonic mean field for the Monney 1T-TiSe2 model.

This adapter reuses ``mean_field.core.hf.problem.HartreeFockProblem`` for the
SCF loop.  System-owned code supplies the four-pocket Hamiltonian, the
paper's interband-only Coulomb decoupling, fixed-number occupations, quadrature
weights, and physical validation.

Authority boundary
------------------
arXiv:0809.1930 specifies ``V(q)=4*pi*e^2/[epsilon(q) q^2]`` and formally
defines the anomalous order parameter, but it does not specify ``epsilon(q)``,
a momentum regulator/mesh, the q=0 cell prescription, temperature/filling, or
a numerical self-consistency calculation.  Every such input is therefore
explicit below.  Results from this module are provisional model calculations,
not parameter-free reproductions of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, xlogy

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    FinalAcceptanceResult,
    HartreeFockRun,
)
from mean_field.core.hf.occupations import (
    conventional_projector_to_stored,
    global_fermi_occupations,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)

from .monney import MonneyTiSe2Parameters, monney_bare_hamiltonian


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SeedMode = Literal["normal", "symmetric", "single_q1", "random"]
SelfCellPolicy = Literal["equal_volume_sphere"]

COULOMB_EV_ANGSTROM = 14.3996454784255
BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class MonneyLocalMesh:
    """C3-closed local hexagonal-prism quadrature around the four pockets."""

    points_Ainv: FloatArray
    weights_Ainv3: FloatArray
    labels: IntArray
    inplane_shells: int
    inplane_spacing_Ainv: float
    z_shells: int
    z_spacing_Ainv: float
    cell_volume_Ainv3: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points_Ainv, dtype=np.float64)
        weights = np.asarray(self.weights_Ainv3, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points_Ainv must have shape (nk,3) and be nonempty")
        if weights.shape != (points.shape[0],):
            raise ValueError("weights_Ainv3 must have shape (nk,)")
        if labels.shape != (points.shape[0], 3):
            raise ValueError("labels must have shape (nk,3)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("mesh points and weights must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("mesh weights must be positive")
        if self.inplane_shells < 0 or self.z_shells < 0:
            raise ValueError("mesh shell counts must be nonnegative")
        if self.inplane_spacing_Ainv <= 0.0 or self.z_spacing_Ainv <= 0.0:
            raise ValueError("mesh spacings must be positive")
        if self.cell_volume_Ainv3 <= 0.0:
            raise ValueError("cell_volume_Ainv3 must be positive")
        object.__setattr__(self, "points_Ainv", points)
        object.__setattr__(self, "weights_Ainv3", weights)
        object.__setattr__(self, "labels", labels)

    @property
    def nk(self) -> int:
        return int(self.points_Ainv.shape[0])

    @property
    def normalized_weights(self) -> FloatArray:
        return np.asarray(self.weights_Ainv3 / np.sum(self.weights_Ainv3), dtype=np.float64)

    @property
    def gamma_index(self) -> int:
        hits = np.flatnonzero(np.all(self.labels == 0, axis=1))
        if hits.size != 1:
            raise ValueError("local mesh must contain exactly one p=0 point")
        return int(hits[0])


@dataclass(frozen=True)
class MonneyCoulombKernel:
    """Weighted direct Coulomb convolution matrix in eV."""

    weighted_matrix_ev: FloatArray
    epsilon_r: float
    self_cell_policy: SelfCellPolicy
    self_cell_ev: float
    source: str = "Monney V(q)=4*pi*e^2/[epsilon*q^2]; constant epsilon approximation"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.weighted_matrix_ev, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("weighted_matrix_ev must be square")
        if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
            raise ValueError("weighted Coulomb matrix must be finite and positive")
        if self.epsilon_r <= 0.0 or not np.isfinite(self.epsilon_r):
            raise ValueError("epsilon_r must be finite and positive")
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if np.max(np.abs(matrix - matrix.T)) > 1.0e-12 * scale:
            raise ValueError("uniform-cell Coulomb matrix must be symmetric")
        object.__setattr__(self, "weighted_matrix_ev", matrix)


@dataclass(frozen=True)
class MonneyHFConfig:
    """Explicit assumptions for one provisional numerical calculation."""

    epsilon_r: float
    temperature_K: float = 65.0
    electrons_per_local_momentum: float = 1.0
    inplane_shells: int = 4
    inplane_spacing_Ainv: float = 0.040
    z_shells: int = 3
    z_spacing_Ainv: float = 0.050
    self_cell_policy: SelfCellPolicy = "equal_volume_sphere"
    precision: float = 1.0e-9
    max_iter: int = 500
    mixing: float = 0.20
    seed_delta_ev: float = 1.0e-3
    max_dense_memory_gb: float = 1.0
    eigensolver_workers: int = 1
    screening_provenance: str = "explicit user/model input; absent from Monney 2008"
    regulator_provenance: str = "C3-closed local hexagonal-prism cutoff; absent from Monney 2008"

    def __post_init__(self) -> None:
        finite = (
            self.epsilon_r,
            self.temperature_K,
            self.electrons_per_local_momentum,
            self.inplane_spacing_Ainv,
            self.z_spacing_Ainv,
            self.precision,
            self.mixing,
            self.seed_delta_ev,
            self.max_dense_memory_gb,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("MonneyHFConfig scalar inputs must be finite")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive for Fermi occupations")
        if not 0.0 <= self.electrons_per_local_momentum <= 4.0:
            raise ValueError("electrons_per_local_momentum must lie in [0,4]")
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
        if not 0.0 < self.mixing <= 1.0:
            raise ValueError("mixing must be in (0,1]")
        if self.seed_delta_ev < 0.0:
            raise ValueError("seed_delta_ev must be nonnegative")
        if self.max_dense_memory_gb <= 0.0:
            raise ValueError("max_dense_memory_gb must be positive")
        if type(self.eigensolver_workers) is not int or self.eigensolver_workers <= 0:
            raise TypeError("eigensolver_workers must be an exact positive integer")
        if not self.screening_provenance or not self.regulator_provenance:
            raise ValueError("assumption provenance strings must be nonempty")

    @property
    def kbt_ev(self) -> float:
        return float(BOLTZMANN_EV_PER_K * self.temperature_K)


@dataclass
class MonneyHFState:
    h0: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float
    precision: float
    diagnostics: dict[str, float]
    mesh: MonneyLocalMesh
    coulomb: MonneyCoulombKernel
    reference_projector_stored: ComplexArray
    params: MonneyTiSe2Parameters
    config: MonneyHFConfig

    @property
    def nk(self) -> int:
        return self.mesh.nk


@dataclass(frozen=True)
class MonneyLinearizedGapResult:
    largest_eigenvalues: FloatArray
    critical_epsilon_r: FloatArray
    c3_relative_spread: float
    chemical_potential_ev: float
    susceptibilities_per_ev: FloatArray


@dataclass(frozen=True)
class MonneyHFResult:
    run: HartreeFockRun
    interaction_h_ev: ComplexArray
    total_hamiltonian_ev: ComplexArray
    mixed_projector_stored: ComplexArray
    raw_update_projector_stored: ComplexArray
    raw_update_density_delta_stored: ComplexArray
    energies_ev: FloatArray
    occupations: FloatArray
    chemical_potential_ev: float
    delta_ev: ComplexArray
    delta_gamma_ev: ComplexArray
    max_delta_ev: float
    c3_gamma_spread_ev: float
    raw_particle_residual: float
    mixed_particle_residual: float
    final_raw_norm: float
    reference_relative_free_energy_ev_A3: float
    hermiticity_residual_ev: float


def build_monney_local_mesh(
    *,
    inplane_shells: int,
    inplane_spacing_Ainv: float,
    z_shells: int,
    z_spacing_Ainv: float,
) -> MonneyLocalMesh:
    """Build an equal-weight C3/C6-closed local hexagonal-prism mesh."""

    if type(inplane_shells) is not int or inplane_shells < 0:
        raise TypeError("inplane_shells must be an exact nonnegative integer")
    if type(z_shells) is not int or z_shells < 0:
        raise TypeError("z_shells must be an exact nonnegative integer")
    g = inplane_shells
    gz = z_shells
    dk = float(inplane_spacing_Ainv)
    dz = float(z_spacing_Ainv)
    if dk <= 0.0 or dz <= 0.0:
        raise ValueError("mesh spacings must be positive")

    u1 = dk * np.array([1.0, 0.0])
    u2 = dk * np.array([0.5, np.sqrt(3.0) / 2.0])
    labels: list[tuple[int, int, int]] = []
    points: list[tuple[float, float, float]] = []
    for iz, im, inn in product(range(-gz, gz + 1), range(-g, g + 1), range(-g, g + 1)):
        if max(abs(im), abs(inn), abs(im + inn)) > g:
            continue
        xy = im * u1 + inn * u2
        labels.append((im, inn, iz))
        points.append((float(xy[0]), float(xy[1]), float(iz * dz)))

    order = np.lexsort(
        (
            np.asarray([value[1] for value in labels]),
            np.asarray([value[0] for value in labels]),
            np.asarray([value[2] for value in labels]),
        )
    )
    points_array = np.asarray(points, dtype=np.float64)[order]
    labels_array = np.asarray(labels, dtype=np.int64)[order]
    cell_volume = float(np.sqrt(3.0) * 0.5 * dk**2 * dz)
    quadrature_weight = cell_volume / (2.0 * np.pi) ** 3
    weights = np.full(points_array.shape[0], quadrature_weight, dtype=np.float64)
    return MonneyLocalMesh(
        points_Ainv=points_array,
        weights_Ainv3=weights,
        labels=labels_array,
        inplane_shells=g,
        inplane_spacing_Ainv=dk,
        z_shells=gz,
        z_spacing_Ainv=dz,
        cell_volume_Ainv3=cell_volume,
    )


def precompute_monney_coulomb_kernel(
    mesh: MonneyLocalMesh,
    *,
    epsilon_r: float,
    self_cell_policy: SelfCellPolicy = "equal_volume_sphere",
    max_dense_memory_gb: float = 1.0,
) -> MonneyCoulombKernel:
    """Precompute ``w_k' V(p-p')`` with an integrated q=0 cell.

    The diagonal uses a sphere with the same volume as one momentum cell:

    ``int_cell d^3q/(2pi)^3 4pi e^2/(epsilon q^2)
       ~= 2 e^2 r_cell/(pi epsilon)``.

    This is a declared regulator approximation, not a value supplied by the
    paper.  No q-floor is used.
    """

    eps = float(epsilon_r)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon_r must be finite and positive")
    if self_cell_policy != "equal_volume_sphere":
        raise ValueError(f"unsupported self_cell_policy {self_cell_policy!r}")
    byte_cap = int(float(max_dense_memory_gb) * 1.0e9)
    itemsize = np.dtype(np.float64).itemsize
    persistent_bytes = mesh.nk * mesh.nk * itemsize
    # Conservative row-block budget: three coordinate differences, q^2,
    # one Boolean mask rounded up to a float, and one ufunc workspace.
    bytes_per_work_row = 6 * mesh.nk * itemsize
    if persistent_bytes + bytes_per_work_row > byte_cap:
        raise MemoryError(
            "dense Coulomb precompute cannot fit one work row within the declared "
            f"{max_dense_memory_gb} GB peak-memory cap"
        )
    work_rows = max(
        1,
        min(mesh.nk, (byte_cap - persistent_bytes) // bytes_per_work_row),
    )

    matrix = np.empty((mesh.nk, mesh.nk), dtype=np.float64)
    prefactor = 4.0 * np.pi * COULOMB_EV_ANGSTROM / eps
    for start in range(0, mesh.nk, work_rows):
        stop = min(mesh.nk, start + work_rows)
        difference = (
            mesh.points_Ainv[start:stop, None, :]
            - mesh.points_Ainv[None, :, :]
        )
        q2 = np.einsum("ijk,ijk->ij", difference, difference, optimize=True)
        block = matrix[start:stop]
        nonzero = q2 > 0.0
        weighted_source = np.broadcast_to(mesh.weights_Ainv3[None, :], q2.shape)
        np.divide(weighted_source, q2, out=block, where=nonzero)
        block *= prefactor

    cell_radius = float((3.0 * mesh.cell_volume_Ainv3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    self_cell_ev = float(2.0 * COULOMB_EV_ANGSTROM * cell_radius / (np.pi * eps))
    np.fill_diagonal(matrix, self_cell_ev)
    return MonneyCoulombKernel(
        weighted_matrix_ev=matrix,
        epsilon_r=eps,
        self_cell_policy=self_cell_policy,
        self_cell_ev=self_cell_ev,
    )


def _stored_projector_update(
    hamiltonian_abk: ComplexArray,
    *,
    mesh: MonneyLocalMesh,
    config: MonneyHFConfig,
) -> tuple[ComplexArray, FloatArray, float, FloatArray, float, float]:
    hamiltonian = np.asarray(hamiltonian_abk, dtype=np.complex128)
    if hamiltonian.shape != (4, 4, mesh.nk):
        raise ValueError(f"hamiltonian must have shape {(4,4,mesh.nk)}")
    occupation = global_fermi_occupations(
        np.moveaxis(hamiltonian, 2, 0),
        target_particle_number=float(config.electrons_per_local_momentum),
        kbt_ev=config.kbt_ev,
        k_weights=mesh.normalized_weights,
        particle_tolerance=1.0e-12,
        eigensolver_workers=int(config.eigensolver_workers),
    )
    ket_abk = np.moveaxis(occupation.density_ket, 0, 2)
    stored = conventional_projector_to_stored(ket_abk)
    return (
        np.asarray(stored, dtype=np.complex128),
        np.asarray(occupation.energies, dtype=np.float64),
        float(occupation.chemical_potential),
        np.asarray(occupation.occupations, dtype=np.float64),
        float(occupation.particle_residual),
        float(occupation.entropy_dimensionless),
    )


def build_monney_hf_state(
    config: MonneyHFConfig,
    *,
    params: MonneyTiSe2Parameters | None = None,
) -> MonneyHFState:
    model = MonneyTiSe2Parameters() if params is None else params
    mesh = build_monney_local_mesh(
        inplane_shells=config.inplane_shells,
        inplane_spacing_Ainv=config.inplane_spacing_Ainv,
        z_shells=config.z_shells,
        z_spacing_Ainv=config.z_spacing_Ainv,
    )
    h0 = monney_bare_hamiltonian(mesh.points_Ainv, params=model)
    coulomb = precompute_monney_coulomb_kernel(
        mesh,
        epsilon_r=config.epsilon_r,
        self_cell_policy=config.self_cell_policy,
        max_dense_memory_gb=config.max_dense_memory_gb,
    )
    reference, energies, mu, _occupations, residual, _entropy = _stored_projector_update(
        h0,
        mesh=mesh,
        config=config,
    )
    return MonneyHFState(
        h0=h0,
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=energies.copy(),
        mu=mu,
        precision=float(config.precision),
        diagnostics={"normal_particle_residual": residual},
        mesh=mesh,
        coulomb=coulomb,
        reference_projector_stored=reference,
        params=model,
        config=config,
    )


def monney_interaction_action(
    density_delta_stored: np.ndarray,
    *,
    coulomb: MonneyCoulombKernel,
) -> ComplexArray:
    """Return the paper's anomalous interband Fock Hamiltonian.

    For stored ``S_ab=<c_a^dagger c_b>``, Eq. (5) is
    ``Delta_i = K @ S[i,0]`` and the Hamiltonian has
    ``Sigma[0,i]=-Delta_i``, ``Sigma[i,0]=-Delta_i*``.
    """

    density = np.asarray(density_delta_stored, dtype=np.complex128)
    nk = coulomb.weighted_matrix_ev.shape[0]
    if density.shape != (4, 4, nk):
        raise ValueError(f"density_delta_stored must have shape {(4,4,nk)}")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_delta_stored must be finite")
    hermitian_residual = float(
        np.max(np.abs(density - density.conj().swapaxes(0, 1)))
    )
    if hermitian_residual > 1.0e-11:
        raise ValueError(
            "density_delta_stored must be Hermitian; residual="
            f"{hermitian_residual:.3e}"
        )

    sigma = np.zeros_like(density)
    for valley in range(1, 4):
        delta = coulomb.weighted_matrix_ev @ density[valley, 0, :]
        sigma[0, valley, :] = -delta
        sigma[valley, 0, :] = -delta.conj()
    return sigma


def _entropy_phi_density(projector_stored: ComplexArray, mesh: MonneyLocalMesh) -> float:
    hermitian_residual = float(
        np.max(
            np.abs(
                projector_stored - projector_stored.conj().swapaxes(0, 1)
            )
        )
    )
    if hermitian_residual > 1.0e-11:
        raise ValueError(
            "projector_stored must be Hermitian before entropy evaluation; "
            f"residual={hermitian_residual:.3e}"
        )
    values = np.empty((4, mesh.nk), dtype=np.float64)
    for ik in range(mesh.nk):
        values[:, ik] = np.linalg.eigvalsh(projector_stored[:, :, ik])
    tolerance = 2.0e-10
    if np.min(values) < -tolerance or np.max(values) > 1.0 + tolerance:
        raise ValueError("mixed one-body density has eigenvalues outside [0,1]")
    clipped = np.clip(values, 0.0, 1.0)
    phi = xlogy(clipped, clipped) + xlogy(1.0 - clipped, 1.0 - clipped)
    return float(np.einsum("bk,k->", phi, mesh.weights_Ainv3, optimize=True))


def monney_reference_relative_free_energy_density(
    interaction_h_ev: np.ndarray,
    h0_ev: np.ndarray,
    density_delta_stored: np.ndarray,
    *,
    state: MonneyHFState,
) -> float:
    """Return F-F_normal in eV/Angstrom^3 per spin copy."""

    sigma = np.asarray(interaction_h_ev, dtype=np.complex128)
    h0 = np.asarray(h0_ev, dtype=np.complex128)
    density = np.asarray(density_delta_stored, dtype=np.complex128)
    if sigma.shape != h0.shape or density.shape != h0.shape:
        raise ValueError("interaction_h_ev, h0_ev, and density must have matching shapes")
    weights = state.mesh.weights_Ainv3
    one_body = np.einsum("abk,abk,k->", h0, density, weights, optimize=True)
    interaction = 0.5 * np.einsum(
        "abk,abk,k->", sigma, density, weights, optimize=True
    )
    projector = state.reference_projector_stored + density
    entropy_phi = _entropy_phi_density(projector, state.mesh)
    reference_phi = _entropy_phi_density(state.reference_projector_stored, state.mesh)
    value = one_body + interaction + state.config.kbt_ev * (entropy_phi - reference_phi)
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise ValueError(f"reference-relative free energy is not real: {value!r}")
    return float(value.real)


def linearized_monney_gap(
    state: MonneyHFState,
) -> MonneyLinearizedGapResult:
    """Diagonalize the normal-state linearized gap kernel for all three valleys."""

    estimated_peak_bytes = 8 * state.nk * state.nk * np.dtype(np.float64).itemsize
    if estimated_peak_bytes > state.config.max_dense_memory_gb * 1.0e9:
        raise MemoryError(
            "linearized dense eigensolve exceeds the declared conservative "
            "peak-memory cap"
        )
    bare = np.real(np.diagonal(state.h0, axis1=0, axis2=1).T)
    # np.diagonal(...).T gives (4,nk) for h0 shape (4,4,nk).
    if bare.shape != (4, state.nk):
        raise RuntimeError(f"unexpected bare-energy shape {bare.shape}")
    mu = float(state.mu)
    kbt = state.config.kbt_ev
    fermi = expit((mu - bare) / kbt)
    susceptibilities = np.empty((3, state.nk), dtype=np.float64)
    largest = np.empty(3, dtype=np.float64)

    ev = bare[0]
    fv = fermi[0]
    for valley in range(3):
        ec = bare[valley + 1]
        fc = fermi[valley + 1]
        difference = ec - ev
        chi = np.empty_like(difference)
        regular = np.abs(difference) > 1.0e-10
        chi[regular] = (fv[regular] - fc[regular]) / difference[regular]
        midpoint = 0.5 * (ev[~regular] + ec[~regular])
        fmid = expit((mu - midpoint) / kbt)
        chi[~regular] = fmid * (1.0 - fmid) / kbt
        if np.min(chi) < -1.0e-10:
            raise ValueError("linearized interband susceptibility must be nonnegative")
        chi = np.maximum(chi, 0.0)
        susceptibilities[valley] = chi
        root = np.sqrt(chi)
        symmetric_kernel = (
            root[:, None] * state.coulomb.weighted_matrix_ev * root[None, :]
        )
        largest[valley] = float(np.linalg.eigvalsh(symmetric_kernel)[-1])

    mean = float(np.mean(largest))
    spread = float(np.max(np.abs(largest - mean)) / max(1.0e-15, abs(mean)))
    critical_epsilon = state.config.epsilon_r * largest
    return MonneyLinearizedGapResult(
        largest_eigenvalues=largest,
        critical_epsilon_r=critical_epsilon,
        c3_relative_spread=spread,
        chemical_potential_ev=mu,
        susceptibilities_per_ev=susceptibilities,
    )


def _seed_hamiltonian(
    state: MonneyHFState,
    *,
    mode: SeedMode,
    seed: int,
) -> ComplexArray:
    source = np.zeros_like(state.h0)
    amplitude = float(state.config.seed_delta_ev)
    if mode == "normal" or amplitude == 0.0:
        return source
    if mode == "symmetric":
        delta = np.full(3, amplitude, dtype=np.complex128)
    elif mode == "single_q1":
        delta = np.asarray([amplitude, 0.0, 0.0], dtype=np.complex128)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        delta = rng.normal(size=3) + 1j * rng.normal(size=3)
        norm = float(np.linalg.norm(delta))
        delta = amplitude * delta / norm if norm > 0.0 else np.full(3, amplitude)
    else:
        raise ValueError(f"unsupported seed mode {mode!r}")
    for valley, value in enumerate(delta, start=1):
        source[0, valley, :] = -value
        source[valley, 0, :] = -value.conjugate()
    return source


def run_monney_hf(
    state: MonneyHFState,
    *,
    init_mode: SeedMode = "symmetric",
    seed: int = 0,
) -> MonneyHFResult:
    """Run one provisional branch through the generic HF problem engine."""

    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    latest_observables: dict[str, np.ndarray | float] = {}

    def density_update(hamiltonian: ComplexArray) -> DensityUpdateResult:
        projector, energies, mu, occupations, residual, entropy = _stored_projector_update(
            hamiltonian,
            mesh=state.mesh,
            config=state.config,
        )
        density = projector - state.reference_projector_stored
        observables: dict[str, np.ndarray | float] = {
            "projector_stored": projector,
            "occupations": occupations,
            "particle_residual": residual,
            "entropy_dimensionless": entropy,
        }
        latest_observables.clear()
        latest_observables.update(observables)
        return DensityUpdateResult(
            density=density,
            energies=energies,
            mu=mu,
            observables=observables,
        )

    def initializer(target: MonneyHFState, *, init_mode: str, seed: int) -> None:
        source = _seed_hamiltonian(
            target,
            mode=init_mode,  # type: ignore[arg-type]
            seed=seed,
        )
        update = density_update(target.h0 + source)
        target.density[:, :, :] = update.density
        target.energies[:, :] = update.energies
        target.mu = float(update.mu)

    def interaction_builder(density: ComplexArray) -> ComplexArray:
        return monney_interaction_action(density, coulomb=state.coulomb)

    def energy_functional(
        interaction_h: ComplexArray,
        h0: ComplexArray,
        density: ComplexArray,
    ) -> float:
        return monney_reference_relative_free_energy_density(
            interaction_h,
            h0,
            density,
            state=state,
        )

    def convergence_metric(updated: np.ndarray, previous: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(updated) - np.asarray(previous))))

    def final_acceptance(
        target: MonneyHFState,
        update: DensityUpdateResult,
        final_norm: float,
    ) -> FinalAcceptanceResult:
        hermitian = float(
            np.max(
                np.abs(
                    target.hamiltonian
                    - target.hamiltonian.conj().swapaxes(0, 1)
                )
            )
        )
        residual = float(update.observables["particle_residual"])
        accepted = (
            hermitian <= 1.0e-11
            and abs(residual) <= 1.0e-10
            and final_norm <= target.precision
        )
        reason = "accepted" if accepted else "physical_final_gate"
        return FinalAcceptanceResult(
            accepted=bool(accepted),
            reason=reason,
            diagnostics={
                "final_hermiticity_residual_ev": hermitian,
                "final_particle_residual": abs(residual),
            },
        )

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_update,
            energy_functional=energy_functional,
            oda_parameterizer=lambda _state, _delta: float(state.config.mixing),
            final_acceptance=final_acceptance,
            convergence_metric=convergence_metric,
            convergence_rule="raw",
        ),
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(state.config.max_iter),
        oda_stall_threshold=1.0e-15,
        max_oda_lambda=None,
    )

    if not run.converged:
        raise RuntimeError(
            "Monney HF did not pass the generic SCF/final-acceptance gates: "
            f"exit_reason={run.exit_reason}, final_raw_norm="
            f"{state.diagnostics.get('final_raw_norm')}"
        )

    interaction_h = interaction_builder(state.density)
    total_h = state.h0 + interaction_h
    final = density_update(total_h)
    # Preserve both sides of the final fixed-point residual.  The mixed state
    # is the density used to construct ``total_h`` and evaluate the functional;
    # the raw update is the Fermi projector obtained from that Hamiltonian.
    mixed_projector = np.asarray(
        state.reference_projector_stored + state.density,
        dtype=np.complex128,
    )
    raw_update_projector = np.asarray(
        final.observables["projector_stored"], dtype=np.complex128
    )
    raw_update_density = np.asarray(final.density, dtype=np.complex128)
    occupations = np.asarray(final.observables["occupations"], dtype=np.float64)
    mixed_number = float(
        np.einsum(
            "aak,k->",
            mixed_projector,
            state.mesh.normalized_weights,
            optimize=True,
        ).real
    )
    mixed_particle_residual = (
        mixed_number - state.config.electrons_per_local_momentum
    )
    final_raw_norm = float(state.diagnostics["final_raw_norm"])
    delta = -interaction_h[0, 1:4, :]
    delta_gamma = delta[:, state.mesh.gamma_index]
    spread = float(np.max(np.abs(delta_gamma - np.mean(delta_gamma))))
    hermitian = float(np.max(np.abs(total_h - total_h.conj().swapaxes(0, 1))))
    free_energy = energy_functional(interaction_h, state.h0, state.density)
    return MonneyHFResult(
        run=run,
        interaction_h_ev=interaction_h,
        total_hamiltonian_ev=total_h,
        mixed_projector_stored=mixed_projector,
        raw_update_projector_stored=raw_update_projector,
        raw_update_density_delta_stored=raw_update_density,
        energies_ev=np.asarray(final.energies, dtype=np.float64),
        occupations=occupations,
        chemical_potential_ev=float(final.mu),
        delta_ev=np.asarray(delta, dtype=np.complex128),
        delta_gamma_ev=np.asarray(delta_gamma, dtype=np.complex128),
        max_delta_ev=float(np.max(np.abs(delta))),
        c3_gamma_spread_ev=spread,
        raw_particle_residual=float(final.observables["particle_residual"]),
        mixed_particle_residual=float(mixed_particle_residual),
        final_raw_norm=final_raw_norm,
        reference_relative_free_energy_ev_A3=free_energy,
        hermiticity_residual_ev=hermitian,
    )


__all__ = [
    "BOLTZMANN_EV_PER_K",
    "COULOMB_EV_ANGSTROM",
    "MonneyCoulombKernel",
    "MonneyHFConfig",
    "MonneyHFResult",
    "MonneyHFState",
    "MonneyLinearizedGapResult",
    "MonneyLocalMesh",
    "build_monney_hf_state",
    "build_monney_local_mesh",
    "linearized_monney_gap",
    "monney_interaction_action",
    "monney_reference_relative_free_energy_density",
    "precompute_monney_coulomb_kernel",
    "run_monney_hf",
]
