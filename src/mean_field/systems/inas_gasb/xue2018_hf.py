"""Q=0 four-band Hartree-Fock adapter for Xue--MacDonald 2018.

This module connects the paper-specific BHZ basis and source interaction to the
reusable HF problem/SCF engine. It is intended first for regulator-qualified
Fig. 2 anchor calculations, not for pixel fitting of the published figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mean_field.core.hf.engine import DensityUpdateResult, HartreeFockRun
from mean_field.core.hf.problem import HartreeFockKernel, HartreeFockProblem, run_hartree_fock_problem

from .xue2018 import xue2018_standard_parameters
from .zeng2022 import (
    Zeng2022Parameters,
    ZengSlabBasis,
    build_zeng2022_folded_h0,
    zeng2022_reference_density,
)
from .zeng2022_hf import (
    Q0CoulombKernel,
    Q0ToeplitzCoulombKernel,
    UniformKappaMesh,
    precompute_q0_coulomb_kernel,
    precompute_q0_toeplitz_coulomb_kernel,
    q0_interaction_from_precomputed_kernel,
    q0_interaction_from_toeplitz_kernel,
    uniform_midpoint_kappa_mesh,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
SeedMode = Literal["normal", "trsb_nematic", "trs_nematic", "qah", "random"]


@dataclass
class Xue2018HFState:
    h0: ComplexArray
    reference_density: ComplexArray
    mesh: UniformKappaMesh
    basis: ZengSlabBasis
    params: Zeng2022Parameters
    q0_kernel: Q0CoulombKernel | Q0ToeplitzCoulombKernel
    q0_kernel_backend: Literal["dense", "toeplitz_fft"]
    self_cell_policy: Literal["integrated", "omitted_diagnostic"]
    mesh_policy: Literal["midpoint_cells", "inclusive_nodes_uniform_weight_diagnostic"]
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float = 0.0
    precision: float = 1.0e-8
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return self.mesh.nk


@dataclass(frozen=True)
class Xue2018HFResult:
    run: HartreeFockRun
    interaction_h: ComplexArray
    total_hamiltonian: ComplexArray
    raw_projector: ComplexArray
    energies: FloatArray
    chemical_potential_ry: float
    global_gap_ry: float
    occupied_rank_per_k: NDArray[np.int64]
    phi1_gamma_ry: complex
    phi1_gamma_abs_ry: float
    trs_nematic_gamma_ry: complex
    reference_relative_energy_ry_ab2: float


def xue2018_square_mesh(
    *,
    kmax_ab_inv: float,
    points_per_axis: int,
    policy: Literal[
        "midpoint_cells", "inclusive_nodes_uniform_weight_diagnostic"
    ] = "midpoint_cells",
) -> UniformKappaMesh:
    """Build a symmetric odd square mesh with an exact saved Gamma point."""

    if points_per_axis % 2 != 1:
        raise ValueError("Xue 2018 Gamma diagnostics require an odd points_per_axis")
    kmax = float(kmax_ab_inv)
    if not np.isfinite(kmax) or kmax <= 0.0:
        raise ValueError("kmax_ab_inv must be finite and positive")
    if policy == "midpoint_cells":
        return uniform_midpoint_kappa_mesh(
            kx_bounds_ab_inv=(-kmax, kmax),
            ky_bounds_ab_inv=(-kmax, kmax),
            nkx=int(points_per_axis),
            nky=int(points_per_axis),
        )
    if policy != "inclusive_nodes_uniform_weight_diagnostic":
        raise ValueError(f"unsupported mesh policy {policy!r}")
    axis = np.linspace(-kmax, kmax, int(points_per_axis), dtype=np.float64)
    spacing = float(axis[1] - axis[0])
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    weights = np.full(points.shape[0], spacing**2 / (2.0 * np.pi) ** 2)
    mesh = UniformKappaMesh(
        points_ab_inv=points,
        weights_ab2=weights,
        cell_widths_ab_inv=(spacing, spacing),
        shape=(int(points_per_axis), int(points_per_axis)),
    )
    mesh.validate()
    return mesh


def xue2018_global_neutral_projector(
    hamiltonian: ComplexArray,
    *,
    reference_density: ComplexArray,
) -> DensityUpdateResult:
    """Diagonalize each k and globally fill exactly two states per k on average."""

    h = np.asarray(hamiltonian, dtype=np.complex128)
    reference = np.asarray(reference_density, dtype=np.complex128)
    if h.shape != reference.shape or h.ndim != 3 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian and reference_density must match (dim,dim,nk)")
    dim, _, nk = h.shape
    eigenvalues = np.empty((dim, nk), dtype=np.float64)
    eigenvectors = np.empty_like(h)
    for ik in range(nk):
        values, vectors = np.linalg.eigh(h[:, :, ik])
        eigenvalues[:, ik] = values
        eigenvectors[:, :, ik] = vectors
    occupied_count = 2 * nk
    flat_order = np.argsort(eigenvalues, axis=None, kind="stable")
    occupied_flat = flat_order[:occupied_count]
    occupation = np.zeros((dim, nk), dtype=np.float64)
    occupation.ravel()[occupied_flat] = 1.0
    projector = np.einsum(
        "aik,ik,bik->abk", eigenvectors, occupation, eigenvectors.conj(), optimize=True
    )
    density_delta = projector - reference
    sorted_values = np.sort(eigenvalues, axis=None)
    highest_occupied = sorted_values[occupied_count - 1]
    lowest_empty = sorted_values[occupied_count]
    mu = 0.5 * (highest_occupied + lowest_empty)
    ranks = np.sum(occupation, axis=0).astype(np.int64)
    return DensityUpdateResult(
        density=density_delta,
        energies=eigenvalues,
        mu=float(mu),
        observables={
            "raw_projector": projector,
            "occupied_rank_per_k": ranks,
            "global_gap_ry": float(lowest_empty - highest_occupied),
        },
    )


def xue2018_seed_hamiltonian(
    basis: ZengSlabBasis,
    nk: int,
    *,
    mode: SeedMode,
    amplitude_ry: float,
    seed: int,
) -> ComplexArray:
    """Return a Hermitian source field for a documented candidate branch."""

    if basis.slab_indices != (0,):
        raise ValueError("Xue 2018 Q=0 seeds require the one-slab basis")
    amplitude = float(amplitude_ry)
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude_ry must be finite and nonnegative")
    source = np.zeros((basis.dimension, basis.dimension, int(nk)), dtype=np.complex128)
    cup = basis.index("c", "up", 0)
    vup = basis.index("v", "up", 0)
    cdown = basis.index("c", "down", 0)
    vdown = basis.index("v", "down", 0)

    if mode == "normal":
        return source
    if mode in {"trsb_nematic", "trs_nematic"}:
        second_sign = 1.0 if mode == "trsb_nematic" else -1.0
        source[cup, vdown, :] = amplitude
        source[vdown, cup, :] = amplitude
        source[vup, cdown, :] = second_sign * amplitude
        source[cdown, vup, :] = second_sign * amplitude
    elif mode == "qah":
        source[cup, cup, :] = amplitude
        source[vup, vup, :] = -amplitude
        source[cdown, cdown, :] = -amplitude
        source[vdown, vdown, :] = amplitude
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        raw = rng.normal(size=(basis.dimension, basis.dimension))
        raw = raw + 1j * rng.normal(size=raw.shape)
        hermitian = raw + raw.conj().T
        norm = np.linalg.norm(hermitian)
        if norm > 0.0:
            hermitian *= amplitude / norm
        source[:, :, :] = hermitian[:, :, None]
    else:
        raise ValueError(f"unsupported seed mode {mode!r}")
    return source


def xue2018_reference_relative_energy_density(
    interaction_h: ComplexArray,
    h0: ComplexArray,
    density_delta: ComplexArray,
    *,
    mesh: UniformKappaMesh,
) -> float:
    """Return ``Tr(h0 D)+0.5 Tr(Sigma[D]D)`` in ``Ry*/a_B*^2``."""

    sigma = np.asarray(interaction_h, dtype=np.complex128)
    bare = np.asarray(h0, dtype=np.complex128)
    density = np.asarray(density_delta, dtype=np.complex128)
    if sigma.shape != bare.shape or sigma.shape != density.shape:
        raise ValueError("interaction_h, h0, and density_delta shapes must match")
    weights = np.asarray(mesh.weights_ab2)
    value = np.einsum("abk,bak,k->", bare, density, weights, optimize=True)
    value += 0.5 * np.einsum("abk,bak,k->", sigma, density, weights, optimize=True)
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise ValueError(f"reference-relative energy is not real: {value!r}")
    return float(value.real)


def build_xue2018_hf_state(
    *,
    eg_ry: float,
    hybridization_ab_ry: float,
    kmax_ab_inv: float,
    points_per_axis: int,
    precision: float = 1.0e-8,
    self_cell_policy: Literal["integrated", "omitted_diagnostic"] = "integrated",
    mesh_policy: Literal[
        "midpoint_cells", "inclusive_nodes_uniform_weight_diagnostic"
    ] = "midpoint_cells",
    q0_kernel_backend: Literal["dense", "toeplitz_fft"] = "dense",
) -> Xue2018HFState:
    mesh = xue2018_square_mesh(
        kmax_ab_inv=kmax_ab_inv,
        points_per_axis=points_per_axis,
        policy=mesh_policy,
    )
    basis = ZengSlabBasis((0,))
    params = xue2018_standard_parameters(
        eg_ry=eg_ry,
        hybridization_ab_ry=hybridization_ab_ry,
    )
    h0 = build_zeng2022_folded_h0(mesh.points_ab_inv, basis, params)
    reference = zeng2022_reference_density(basis, mesh.nk)
    if self_cell_policy not in {"integrated", "omitted_diagnostic"}:
        raise ValueError(f"unsupported self_cell_policy {self_cell_policy!r}")
    if q0_kernel_backend == "dense":
        q0_kernel: Q0CoulombKernel | Q0ToeplitzCoulombKernel = (
            precompute_q0_coulomb_kernel(mesh, d_over_ab=params.d_over_ab)
        )
        if self_cell_policy == "omitted_diagnostic":
            intra = np.asarray(q0_kernel.intra_ry_ab2).copy()
            inter = np.asarray(q0_kernel.inter_ry_ab2).copy()
            np.fill_diagonal(intra, 0.0)
            np.fill_diagonal(inter, 0.0)
            q0_kernel = Q0CoulombKernel(intra_ry_ab2=intra, inter_ry_ab2=inter)
    elif q0_kernel_backend == "toeplitz_fft":
        q0_kernel = precompute_q0_toeplitz_coulomb_kernel(
            mesh,
            d_over_ab=params.d_over_ab,
            omit_self_cell_diagnostic=self_cell_policy == "omitted_diagnostic",
        )
    else:
        raise ValueError(f"unsupported q0_kernel_backend {q0_kernel_backend!r}")
    return Xue2018HFState(
        h0=h0,
        reference_density=reference,
        mesh=mesh,
        basis=basis,
        params=params,
        q0_kernel=q0_kernel,
        q0_kernel_backend=q0_kernel_backend,
        self_cell_policy=self_cell_policy,
        mesh_policy=mesh_policy,
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=np.zeros((basis.dimension, mesh.nk), dtype=np.float64),
        precision=float(precision),
    )


def _gamma_index(mesh: UniformKappaMesh) -> int:
    norms = np.linalg.norm(np.asarray(mesh.points_ab_inv), axis=1)
    index = int(np.argmin(norms))
    if norms[index] > 1.0e-14:
        raise ValueError("mesh has no exact Gamma point")
    return index


def _xue2018_interaction_action(
    state: Xue2018HFState, density: ComplexArray
) -> ComplexArray:
    if state.q0_kernel_backend == "dense":
        if not isinstance(state.q0_kernel, Q0CoulombKernel):
            raise TypeError("dense backend requires Q0CoulombKernel")
        return q0_interaction_from_precomputed_kernel(
            density,
            basis=state.basis,
            mesh=state.mesh,
            params=state.params,
            kernel=state.q0_kernel,
        )
    if not isinstance(state.q0_kernel, Q0ToeplitzCoulombKernel):
        raise TypeError("toeplitz_fft backend requires Q0ToeplitzCoulombKernel")
    return q0_interaction_from_toeplitz_kernel(
        density,
        basis=state.basis,
        mesh=state.mesh,
        params=state.params,
        kernel=state.q0_kernel,
    )


def xue2018_interaction_action(
    state: Xue2018HFState,
    density_delta: ComplexArray,
) -> ComplexArray:
    """Return the bound Xue Hartree--Fock self-energy action on ``D``."""

    return _xue2018_interaction_action(state, density_delta)


def _weighted_oda_parameter(state: Xue2018HFState, delta_density: ComplexArray) -> float:
    delta_h = _xue2018_interaction_action(state, delta_density)
    interaction_h = state.hamiltonian - state.h0
    weights = np.asarray(state.mesh.weights_ab2)
    a = np.einsum("abk,bak,k->", delta_h, delta_density, weights, optimize=True).real
    b = np.einsum("abk,bak,k->", state.h0, delta_density, weights, optimize=True).real
    b += 0.5 * np.einsum(
        "abk,bak,k->", interaction_h, delta_density, weights, optimize=True
    ).real
    b += 0.5 * np.einsum("abk,bak,k->", delta_h, state.density, weights, optimize=True).real
    if abs(a) < 1.0e-15:
        return 1.0 if b < 0.0 else 0.0
    stationary = -b / a
    if a > 0.0:
        return float(np.clip(stationary, 0.0, 1.0))
    # Along a concave chord the minimum is at an endpoint.  Comparing
    # E(1)-E(0)=b+a/2 selects lambda=1 exactly when the stationary maximum
    # lies at or left of the midpoint (the generic core ODA convention).
    return 1.0 if stationary <= 0.5 else 0.0


def run_xue2018_hf(
    state: Xue2018HFState,
    *,
    init_mode: SeedMode,
    seed: int = 0,
    seed_amplitude_ry: float = 0.2,
    max_iter: int = 300,
    max_oda_lambda: float | None = 0.5,
    oda_stall_threshold: float = 1.0e-12,
    initial_density_delta: ComplexArray | None = None,
) -> Xue2018HFResult:
    """Run one candidate Q=0 HF branch and return exact-grid diagnostics."""

    def initializer(target: Xue2018HFState, *, init_mode: str, seed: int) -> None:
        if initial_density_delta is not None:
            initial = np.asarray(initial_density_delta, dtype=np.complex128)
            if initial.shape != target.density.shape or not np.all(np.isfinite(initial)):
                raise ValueError("initial_density_delta must be finite and match state density shape")
            target.density[:, :, :] = initial
            return
        source = xue2018_seed_hamiltonian(
            target.basis,
            target.nk,
            mode=init_mode,  # type: ignore[arg-type]
            amplitude_ry=seed_amplitude_ry,
            seed=seed,
        )
        update = xue2018_global_neutral_projector(
            target.h0 + source,
            reference_density=target.reference_density,
        )
        target.density[:, :, :] = update.density
        target.energies[:, :] = update.energies
        target.mu = float(update.mu)

    def interaction_builder(density: ComplexArray) -> ComplexArray:
        return _xue2018_interaction_action(state, density)

    def density_builder(hamiltonian: ComplexArray) -> DensityUpdateResult:
        return xue2018_global_neutral_projector(
            hamiltonian,
            reference_density=state.reference_density,
        )

    def energy_functional(
        interaction_h: ComplexArray,
        h0: ComplexArray,
        density: ComplexArray,
    ) -> float:
        return xue2018_reference_relative_energy_density(
            interaction_h,
            h0,
            density,
            mesh=state.mesh,
        )

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            oda_parameterizer=_weighted_oda_parameter,
            convergence_rule="raw",
        ),
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(max_iter),
        oda_stall_threshold=float(oda_stall_threshold),
        max_oda_lambda=max_oda_lambda,
    )
    interaction_h = interaction_builder(state.density)
    total_h = state.h0 + interaction_h
    final = density_builder(total_h)
    raw_projector = np.asarray(final.observables["raw_projector"], dtype=np.complex128)
    ranks = np.asarray(final.observables["occupied_rank_per_k"], dtype=np.int64)
    gap = float(final.observables["global_gap_ry"])
    gamma = _gamma_index(state.mesh)
    cup = state.basis.index("c", "up", 0)
    vup = state.basis.index("v", "up", 0)
    cdown = state.basis.index("c", "down", 0)
    vdown = state.basis.index("v", "down", 0)
    first = total_h[cup, vdown, gamma]
    second = total_h[vup, cdown, gamma]
    phi1 = first + second
    trs_order = first - second
    energy = energy_functional(interaction_h, state.h0, state.density)
    return Xue2018HFResult(
        run=run,
        interaction_h=interaction_h,
        total_hamiltonian=total_h,
        raw_projector=raw_projector,
        energies=final.energies,
        chemical_potential_ry=float(final.mu),
        global_gap_ry=gap,
        occupied_rank_per_k=ranks,
        phi1_gamma_ry=complex(phi1),
        phi1_gamma_abs_ry=float(abs(phi1)),
        trs_nematic_gamma_ry=complex(trs_order),
        reference_relative_energy_ry_ab2=energy,
    )


__all__ = [
    "Xue2018HFResult",
    "Xue2018HFState",
    "build_xue2018_hf_state",
    "run_xue2018_hf",
    "xue2018_global_neutral_projector",
    "xue2018_interaction_action",
    "xue2018_reference_relative_energy_density",
    "xue2018_seed_hamiltonian",
    "xue2018_square_mesh",
]
