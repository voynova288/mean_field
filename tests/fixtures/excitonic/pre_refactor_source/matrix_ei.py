"""Normal-ordered matrix EI scaffold in an E1/H1 basis.

The generic HF engine iterates ``D(k)=P(k)-P0(k)``. ``P0`` is always built
from the same :class:`~.projected_model.Kane4Bundle` and temperature, using
either a declared active-space constraint or a supplied fixed chemical
potential. Supported certified interactions are projected exchange alone and
projected exchange plus a differential periodic-Poisson Hartree response.
Both are explicitly normal ordered relative to ``P0``. The reference is
fail-closed and explicit: either the noninteracting Fermi state (historical
incremental diagnostic) or the ordinary-electron E1-empty/H1-filled vacuum
needed to retain carrier Fock terms in an electron--hole model.

The periodic differential Hartree closes a controlled frozen-potential
Phase-1 functional, but it is not the unknown device fixed-gate boundary
condition. Results therefore remain model diagnostics rather than physical
fixed-gate phase energies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStateProtocol,
    HartreeFockStepResult,
    run_hartree_fock_iterations,
)

from .angular_fock import PolarHarmonicProjectedFockOperator
from .axial_fock import (
    AxialAveragedProjectedFockOperator,
    AxialProjectedFockOperator,
)
from .carriers import KaneCarrierProjectors, charge_neutral_fermi_density
from .conventions import E1H1BasisSpec
from .hartree import ProjectedHartreeFockOperator, reference_subtracted_charge_density
from .projected_model import Kane4Bundle, ProjectedFockOperator, excitonic_fock_singular_values


KB_MEV_PER_K = 0.08617333262
ComplexArray = np.ndarray
FloatArray = np.ndarray
ReferencePolicy = Literal["normal_ordered_exchange_only", "normal_ordered_hartree_fock"]
SearchMode = Literal["normal_reference", "seeded_ei"]
ConstraintPolicy = Literal[
    "active_half_filling",
    "microscopic_charge_neutrality",
    "kane_poisson_fixed_mu",
]
NormalOrderingReferencePolicy = Literal[
    "noninteracting_fermi_state",
    "electron_hole_vacuum",
]


@dataclass(frozen=True)
class MatrixEIConfig:
    temperature_K: float = 0.1
    target_occupation_per_k: float = 2.0
    mixing: float = 0.2
    precision: float = 1e-8
    max_iter: int = 500
    reference_policy: ReferencePolicy = "normal_ordered_exchange_only"
    search_mode: SearchMode = "normal_reference"
    constraint_policy: ConstraintPolicy = "active_half_filling"
    fixed_mu_mev: float | None = None
    normal_ordering_reference_policy: NormalOrderingReferencePolicy = (
        "noninteracting_fermi_state"
    )

    def __post_init__(self) -> None:
        if self.temperature_K <= 0.0:
            raise ValueError("the initial weighted solver requires temperature_K > 0")
        if abs(self.target_occupation_per_k - 2.0) > 1e-12:
            raise ValueError("the current CNP matrix-EI scaffold requires target_occupation_per_k=2")
        if not 1e-3 <= self.mixing <= 1.0:
            raise ValueError("mixing must lie in [1e-3, 1]")
        if self.precision <= 0.0 or self.max_iter < 1:
            raise ValueError("precision and max_iter must be positive")
        if self.reference_policy not in {"normal_ordered_exchange_only", "normal_ordered_hartree_fock"}:
            raise ValueError("unsupported reference policy")
        if self.search_mode not in {"normal_reference", "seeded_ei"}:
            raise ValueError("unsupported matrix-EI search mode")
        if self.constraint_policy not in {
            "active_half_filling",
            "microscopic_charge_neutrality",
            "kane_poisson_fixed_mu",
        }:
            raise ValueError("unsupported matrix-EI constraint policy")
        if self.constraint_policy == "kane_poisson_fixed_mu":
            if self.fixed_mu_mev is None or not np.isfinite(self.fixed_mu_mev):
                raise ValueError(
                    "kane_poisson_fixed_mu requires a finite fixed_mu_mev"
                )
        elif self.fixed_mu_mev is not None:
            raise ValueError(
                "fixed_mu_mev is only valid with constraint_policy="
                "'kane_poisson_fixed_mu'"
            )
        if self.normal_ordering_reference_policy not in {
            "noninteracting_fermi_state",
            "electron_hole_vacuum",
        }:
            raise ValueError("unsupported normal-ordering reference policy")


@dataclass
class MatrixEIState:
    h0: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    mu: float
    precision: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class MatrixEIResult:
    classification: str
    config: MatrixEIConfig
    bundle_fingerprint: str
    interaction_fingerprint: str
    reference_mu_mev: float
    reference_density: ComplexArray
    noninteracting_density: ComplexArray
    density_delta: ComplexArray
    total_density: ComplexArray
    sigma_hartree_mev: ComplexArray
    sigma_fock_mev: ComplexArray
    interaction_h_mev: ComplexArray
    hamiltonian_mev: ComplexArray
    energies_mev: FloatArray
    mu_mev: float
    one_body_internal_energy_density_mev_nm2: float
    hartree_internal_energy_density_mev_nm2: float
    fock_internal_energy_density_mev_nm2: float
    total_internal_energy_density_mev_nm2: float
    entropy_difference_mev_per_K_nm2: float
    canonical_free_energy_difference_mev_nm2: float
    grand_potential_difference_mev_nm2: float | None
    excitonic_singular_values_mev: FloatArray | None
    run: HartreeFockRun


def _fermi(energy_minus_mu_mev: FloatArray, temperature_K: float) -> FloatArray:
    x = np.asarray(energy_minus_mu_mev, dtype=float) / (KB_MEV_PER_K * float(temperature_K))
    out = np.empty_like(x)
    positive = x >= 0.0
    exp_minus = np.exp(-np.clip(x[positive], 0.0, 745.0))
    out[positive] = exp_minus / (1.0 + exp_minus)
    exp_plus = np.exp(np.clip(x[~positive], -745.0, 0.0))
    out[~positive] = 1.0 / (1.0 + exp_plus)
    return out


def weighted_fermi_density(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    *,
    temperature_K: float,
    target_occupation_per_k: float,
) -> DensityUpdateResult:
    """Diagonalize a matrix Hamiltonian and enforce weighted total filling."""

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    if h.ndim != 3 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian_mev must have shape (n, n, nk)")
    n, _, nk = h.shape
    if wk.shape != (nk,) or np.any(wk <= 0.0):
        raise ValueError("k_weights_nm2 must be positive with shape (nk,)")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if not 0.0 <= target_occupation_per_k <= float(n):
        raise ValueError("target occupation is outside the active-space dimension")
    hermiticity_error = float(np.max(np.abs(h - np.swapaxes(h.conj(), 0, 1))))
    if hermiticity_error > 1e-9:
        raise ValueError(f"Hamiltonian is not Hermitian: error={hermiticity_error:.3e}")

    energies = np.empty((n, nk), dtype=float)
    vectors = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])

    target = float(target_occupation_per_k) * float(np.sum(wk))
    thermal_margin = max(10.0, 80.0 * KB_MEV_PER_K * float(temperature_K))
    lower = float(np.min(energies) - thermal_margin)
    upper = float(np.max(energies) + thermal_margin)

    def number_residual(mu: float) -> float:
        occupation = _fermi(energies - float(mu), temperature_K)
        return float(np.einsum("k,nk->", wk, occupation, optimize=True) - target)

    if target <= 0.0:
        mu = lower
    elif target >= float(n) * float(np.sum(wk)):
        mu = upper
    else:
        mu = float(brentq(number_residual, lower, upper, xtol=1e-13, rtol=1e-14))
    occupation = _fermi(energies - mu, temperature_K)
    density = np.empty_like(h)
    for ik in range(nk):
        density[:, :, ik] = (vectors[:, :, ik] * occupation[:, ik][None, :]) @ vectors[:, :, ik].conj().T
    achieved = float(np.einsum("k,aak->", wk, density, optimize=True).real / np.sum(wk))
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=mu,
        observables={
            "target_occupation_per_k": float(target_occupation_per_k),
            "achieved_occupation_per_k": achieved,
            "number_residual_nm2": number_residual(mu),
            "hamiltonian_hermiticity_error_mev": hermiticity_error,
        },
    )


def fixed_mu_fermi_density(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    *,
    temperature_K: float,
    mu_mev: float,
) -> DensityUpdateResult:
    """Diagonalize ``H(k)`` and evaluate occupations at one immutable ``mu``.

    Unlike :func:`weighted_fermi_density`, this function performs no number,
    neutrality, or active-half root. It is the grand-canonical occupation map
    needed when a preceding Kane--Poisson calculation has already fixed the
    ordinary-electron chemical potential in the same energy gauge. This is
    not the pair-density Lagrange multiplier also denoted ``mu`` in the
    Supplementary-Note-2 scalar electron-hole BCS equations.
    """

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    if h.ndim != 3 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian_mev must have shape (n, n, nk)")
    n, _, nk = h.shape
    if wk.shape != (nk,) or np.any(wk <= 0.0) or not np.all(np.isfinite(wk)):
        raise ValueError("k_weights_nm2 must be finite and positive with shape (nk,)")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if not np.isfinite(mu_mev):
        raise ValueError("mu_mev must be finite")
    hermiticity_error = float(np.max(np.abs(h - np.swapaxes(h.conj(), 0, 1))))
    if hermiticity_error > 1e-9:
        raise ValueError(f"Hamiltonian is not Hermitian: error={hermiticity_error:.3e}")

    energies = np.empty((n, nk), dtype=float)
    vectors = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])
    occupation = _fermi(energies - float(mu_mev), temperature_K)
    density = np.empty_like(h)
    for ik in range(nk):
        density[:, :, ik] = (
            vectors[:, :, ik] * occupation[:, ik][None, :]
        ) @ vectors[:, :, ik].conj().T
    active_number = float(np.einsum("k,nk->", wk, occupation, optimize=True))
    achieved = active_number / float(np.sum(wk))
    return DensityUpdateResult(
        density=density,
        energies=energies,
        mu=float(mu_mev),
        observables={
            "achieved_occupation_per_k": achieved,
            "active_number_density_nm2": active_number,
            "fixed_mu_mev": float(mu_mev),
            "hamiltonian_hermiticity_error_mev": hermiticity_error,
        },
    )


def fermionic_entropy_density_mev_per_K_nm2(
    density: ComplexArray,
    k_weights_nm2: FloatArray,
) -> float:
    """Return ``-k_B int_k Tr[P ln P + (1-P) ln(1-P)]``."""

    projector = np.asarray(density, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    if projector.ndim != 3 or projector.shape[0] != projector.shape[1]:
        raise ValueError("density must have shape (n, n, nk)")
    if wk.shape != (projector.shape[2],):
        raise ValueError("k weights do not match density")
    entropy = 0.0
    for ik in range(projector.shape[2]):
        values = np.linalg.eigvalsh(projector[:, :, ik])
        if float(np.min(values)) < -1e-8 or float(np.max(values)) > 1.0 + 1e-8:
            raise ValueError("density eigenvalues lie outside [0, 1]")
        clipped = np.clip(values, 1e-15, 1.0 - 1e-15)
        entropy -= wk[ik] * KB_MEV_PER_K * float(
            np.sum(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        )
    return entropy


def relative_internal_energy_components_mev_nm2(
    sigma_fock_mev: ComplexArray,
    h0_mev: ComplexArray,
    density_delta: ComplexArray,
    k_weights_nm2: FloatArray,
) -> tuple[float, float, float]:
    """Return one-body, exchange, and total reference-relative internal energy."""

    sigma = np.asarray(sigma_fock_mev, dtype=np.complex128)
    h0 = np.asarray(h0_mev, dtype=np.complex128)
    density = np.asarray(density_delta, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    if not (sigma.shape == h0.shape == density.shape) or density.ndim != 3:
        raise ValueError("sigma, h0, and density must have matching (n, n, nk) shapes")
    if wk.shape != (density.shape[2],):
        raise ValueError("k weights do not match the density mesh")
    one_body = np.einsum("k,abk,bak->", wk, h0, density, optimize=True)
    exchange = 0.5 * np.einsum("k,abk,bak->", wk, sigma, density, optimize=True)
    for name, value in (("one-body", one_body), ("exchange", exchange)):
        if abs(value.imag) > 1e-8 * max(1.0, abs(value.real)):
            raise ValueError(f"{name} energy is not real: {value}")
    return float(one_body.real), float(exchange.real), float((one_body + exchange).real)


def relative_hf_energy_density_mev_nm2(
    interaction_h_mev: ComplexArray,
    h0_mev: ComplexArray,
    density_delta: ComplexArray,
    k_weights_nm2: FloatArray,
) -> float:
    """Compatibility helper returning the reference-relative internal energy."""

    return relative_internal_energy_components_mev_nm2(
        interaction_h_mev,
        h0_mev,
        density_delta,
        k_weights_nm2,
    )[2]


def _validate_reference_density(
    reference: ComplexArray,
    h0_shape: tuple[int, ...],
    weights: FloatArray,
    target: float | None,
) -> None:
    p0 = np.asarray(reference, dtype=np.complex128)
    if p0.shape != h0_shape:
        raise ValueError("reference density shape does not match h0")
    hermiticity_error = float(np.max(np.abs(p0 - np.swapaxes(p0.conj(), 0, 1))))
    if hermiticity_error > 1e-9:
        raise ValueError("reference density is not Hermitian")
    eigenvalues = np.concatenate([np.linalg.eigvalsh(p0[:, :, ik]) for ik in range(p0.shape[2])])
    if float(np.min(eigenvalues)) < -1e-8 or float(np.max(eigenvalues)) > 1.0 + 1e-8:
        raise ValueError("reference density eigenvalues lie outside [0, 1]")
    if target is not None:
        achieved = float(np.einsum("k,aak->", weights, p0, optimize=True).real / np.sum(weights))
        if abs(achieved - target) > 1e-9:
            raise ValueError("reference density has the wrong active-space filling")


def solve_reference_subtracted_matrix_ei(
    bundle: Kane4Bundle,
    interaction_operator: (
        ProjectedFockOperator
        | AxialProjectedFockOperator
        | AxialAveragedProjectedFockOperator
        | PolarHarmonicProjectedFockOperator
        | ProjectedHartreeFockOperator
    ),
    *,
    config: MatrixEIConfig | None = None,
    seed_hamiltonian_mev: ComplexArray | None = None,
    initial_density_delta: ComplexArray | None = None,
    initial_density_is_postprocessed: bool = False,
    density_symmetrizer: Callable[[ComplexArray], ComplexArray] | None = None,
    symmetry_constraint_label: str | None = None,
    step_callback: (
        Callable[[HartreeFockStateProtocol, HartreeFockStepResult], None] | None
    ) = None,
    final_state_callback: (
        Callable[[HartreeFockStateProtocol, DensityUpdateResult], None] | None
    ) = None,
) -> MatrixEIResult:
    """Run a source-bound normal-ordered matrix-EI fixed point."""

    cfg = MatrixEIConfig() if config is None else config
    bundle.validate()
    bundle_fingerprint = bundle.fingerprint()
    if isinstance(interaction_operator, ProjectedHartreeFockOperator):
        required_policy = "normal_ordered_hartree_fock"
        model_label = "normal-ordered Hartree-Fock"
    elif isinstance(interaction_operator, AxialAveragedProjectedFockOperator):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered co-rotating-m0 exchange diagnostic"
    elif isinstance(interaction_operator, PolarHarmonicProjectedFockOperator):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered selected-harmonic exchange diagnostic"
    elif isinstance(
        interaction_operator,
        (ProjectedFockOperator, AxialProjectedFockOperator),
    ):
        required_policy = "normal_ordered_exchange_only"
        model_label = "normal-ordered exchange-only"
    else:
        raise TypeError("matrix-EI solver requires a certified projected interaction operator")
    if cfg.reference_policy != required_policy:
        raise ValueError(
            f"reference_policy={cfg.reference_policy!r} does not match interaction {required_policy!r}"
        )
    interaction_operator.validate_against_bundle(bundle)
    interaction_fingerprint = str(interaction_operator.fingerprint())
    if not interaction_fingerprint:
        raise ValueError("interaction operator fingerprint is required")

    h0 = np.asarray(bundle.h0_mev, dtype=np.complex128)
    wk = np.asarray(bundle.weights_nm2, dtype=float)
    # The interaction operators already bind the exact positive quadrature
    # weights.  Radial annular meshes are intentionally nonuniform, so the
    # SCF convergence norm must use the same measure rather than requiring
    # equal-area cells.
    if initial_density_is_postprocessed and initial_density_delta is None:
        raise ValueError(
            "initial_density_is_postprocessed requires initial_density_delta"
        )
    if initial_density_is_postprocessed and density_symmetrizer is None:
        raise ValueError(
            "initial_density_is_postprocessed requires a density_symmetrizer"
        )
    if cfg.search_mode == "normal_reference" and (
        seed_hamiltonian_mev is not None or initial_density_delta is not None
    ):
        raise ValueError("normal_reference mode does not accept an EI seed")
    if cfg.search_mode == "seeded_ei" and (
        (seed_hamiltonian_mev is None) == (initial_density_delta is None)
    ):
        raise ValueError(
            "seeded_ei mode requires exactly one of seed_hamiltonian_mev "
            "or initial_density_delta"
        )

    carrier_projectors = (
        KaneCarrierProjectors.from_bundle(bundle)
        if cfg.constraint_policy == "microscopic_charge_neutrality"
        else None
    )

    def constrained_density(total_hamiltonian: ComplexArray) -> DensityUpdateResult:
        if cfg.constraint_policy == "kane_poisson_fixed_mu":
            assert cfg.fixed_mu_mev is not None
            return fixed_mu_fermi_density(
                total_hamiltonian,
                wk,
                temperature_K=cfg.temperature_K,
                mu_mev=cfg.fixed_mu_mev,
            )
        if carrier_projectors is None:
            return weighted_fermi_density(
                total_hamiltonian,
                wk,
                temperature_K=cfg.temperature_K,
                target_occupation_per_k=cfg.target_occupation_per_k,
            )
        neutral = charge_neutral_fermi_density(
            total_hamiltonian,
            wk,
            carrier_projectors,
            temperature_K=cfg.temperature_K,
        )
        return DensityUpdateResult(
            density=neutral.density,
            energies=neutral.energies_mev,
            mu=neutral.mu_mev,
            observables={
                "electron_density_nm2": neutral.electron_density_nm2,
                "hole_density_nm2": neutral.hole_density_nm2,
                "charge_imbalance_nm2": neutral.charge_imbalance_nm2,
            },
        )

    normal = constrained_density(h0)
    noninteracting_density = np.asarray(normal.density, dtype=np.complex128)
    if cfg.normal_ordering_reference_policy == "noninteracting_fermi_state":
        p0 = noninteracting_density.copy()
    else:
        p0 = np.repeat(
            bundle.basis.h1_electron_projector[:, :, None],
            bundle.nk,
            axis=2,
        ).astype(np.complex128, copy=False)
    reference_target = (
        cfg.target_occupation_per_k
        if (
            cfg.constraint_policy == "active_half_filling"
            or cfg.normal_ordering_reference_policy == "electron_hole_vacuum"
        )
        else None
    )
    _validate_reference_density(p0, h0.shape, wk, reference_target)
    if carrier_projectors is not None:
        n_e0, n_h0 = carrier_projectors.densities_nm2(noninteracting_density, wk)
        if abs(n_e0 - n_h0) > 1e-9:
            raise ValueError("microscopic noninteracting density is not charge neutral")

    if initial_density_delta is not None:
        initial_delta = np.asarray(initial_density_delta, dtype=np.complex128).copy()
        if initial_delta.shape != h0.shape:
            raise ValueError("initial_density_delta must match h0")
        if not np.all(np.isfinite(initial_delta)):
            raise ValueError("initial_density_delta must be finite")
    elif seed_hamiltonian_mev is None:
        initial_delta = noninteracting_density - p0
    else:
        seed = np.asarray(seed_hamiltonian_mev, dtype=np.complex128)
        if seed.shape != h0.shape:
            raise ValueError("seed_hamiltonian_mev must match h0")
        seeded = constrained_density(h0 + seed)
        initial_delta = seeded.density - p0
    if density_symmetrizer is not None:
        postprocessed_initial = np.asarray(
            density_symmetrizer(initial_delta), dtype=np.complex128
        )
        if initial_density_is_postprocessed:
            postprocessing_error = float(
                np.max(np.abs(postprocessed_initial - initial_delta))
            )
            if postprocessing_error > 1e-9:
                raise ValueError(
                    "declared postprocessed initial density violates the supplied "
                    f"symmetry projector: error={postprocessing_error:.3e}"
                )
        else:
            initial_delta = postprocessed_initial
    check_initial_error = float(
        np.max(np.abs(initial_delta - np.swapaxes(initial_delta.conj(), 0, 1)))
    )
    if check_initial_error > 1e-8:
        raise ValueError("initial density delta is not Hermitian")

    state = MatrixEIState(
        h0=h0.copy(),
        density=initial_delta.copy(),
        hamiltonian=h0.copy(),
        energies=np.asarray(normal.energies, dtype=float).copy(),
        mu=float(normal.mu),
        precision=float(cfg.precision),
    )

    def density_builder(total_hamiltonian: ComplexArray) -> DensityUpdateResult:
        update = constrained_density(total_hamiltonian)
        density_delta = np.asarray(update.density) - p0
        if density_symmetrizer is not None:
            density_delta = np.asarray(density_symmetrizer(density_delta), dtype=np.complex128)
        return DensityUpdateResult(
            density=density_delta,
            energies=update.energies,
            mu=update.mu,
            observables=update.observables,
        )

    def energy_functional(sigma: ComplexArray, bare_h: ComplexArray, density: ComplexArray) -> float:
        return relative_hf_energy_density_mev_nm2(sigma, bare_h, density, wk)

    def check_hermitian(values: ComplexArray) -> None:
        error = float(np.max(np.abs(values - np.swapaxes(values.conj(), 0, 1))))
        if error > 1e-8:
            raise ValueError(f"SCF matrix lost Hermiticity: error={error:.3e}")

    def postprocess_density(values: ComplexArray) -> None:
        if density_symmetrizer is not None:
            values[:, :, :] = np.asarray(density_symmetrizer(values), dtype=np.complex128)
        check_hermitian(values)

    def weighted_frobenius_norm(values: ComplexArray) -> float:
        array = np.asarray(values, dtype=np.complex128)
        norm_squared = np.einsum(
            "k,abk,abk->",
            wk,
            array.conj(),
            array,
            optimize=True,
        )
        if abs(norm_squared.imag) > 1e-12:
            raise ValueError("weighted SCF norm is not real")
        return float(np.sqrt(max(float(norm_squared.real), 0.0)))

    reference_norm = max(weighted_frobenius_norm(p0), 1e-15)

    def reference_scaled_convergence(updated: ComplexArray, previous: ComplexArray) -> float:
        return weighted_frobenius_norm(
            np.asarray(updated) - np.asarray(previous)
        ) / reference_norm

    run = run_hartree_fock_iterations(
        state,
        init_mode=cfg.search_mode,
        seed=0,
        interaction_builder=interaction_operator,
        density_builder=density_builder,
        energy_functional=energy_functional,
        oda_parameterizer=lambda _state, _delta: float(cfg.mixing),
        hamiltonian_postprocessor=check_hermitian,
        density_postprocessor=postprocess_density,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_metric=reference_scaled_convergence,
        convergence_rule="raw",
        max_iter=int(cfg.max_iter),
        oda_stall_threshold=0.0,
    )
    if isinstance(interaction_operator, ProjectedHartreeFockOperator):
        sigma_hartree, sigma_fock = interaction_operator.components(state.density)
    else:
        sigma_hartree = np.zeros_like(h0)
        sigma_fock = np.asarray(interaction_operator(state.density), dtype=np.complex128)
    sigma = sigma_hartree + sigma_fock
    total_density = p0 + state.density
    charge_delta = reference_subtracted_charge_density(bundle, state.density)
    one_body = float(np.einsum("k,abk,bak->", wk, h0, state.density, optimize=True).real)
    hartree = float(
        0.5 * np.einsum("k,abk,bak->", wk, sigma_hartree, state.density, optimize=True).real
    )
    exchange = float(
        0.5 * np.einsum("k,abk,bak->", wk, sigma_fock, state.density, optimize=True).real
    )
    internal = one_body + hartree + exchange
    entropy_reference = fermionic_entropy_density_mev_per_K_nm2(p0, wk)
    entropy_final = fermionic_entropy_density_mev_per_K_nm2(total_density, wk)
    entropy_difference = entropy_final - entropy_reference
    free_energy = internal - cfg.temperature_K * entropy_difference
    active_number_change = float(
        np.einsum("k,aak->", wk, state.density, optimize=True).real
    )
    if abs(active_number_change - charge_delta.active_number_change_nm2) > 1e-9:
        raise ValueError("active-number change disagrees with integrated charge vertices")
    grand_potential = (
        free_energy - float(cfg.fixed_mu_mev) * active_number_change
        if cfg.constraint_policy == "kane_poisson_fixed_mu"
        else None
    )
    reference_occupation = float(
        np.einsum("k,aak->", wk, p0, optimize=True).real / np.sum(wk)
    )
    final_occupation = float(
        np.einsum("k,aak->", wk, total_density, optimize=True).real / np.sum(wk)
    )
    state.diagnostics.update(
        {
            "one_body_internal_energy_density_mev_nm2": one_body,
            "hartree_internal_energy_density_mev_nm2": hartree,
            "fock_internal_energy_density_mev_nm2": exchange,
            "total_internal_energy_density_mev_nm2": internal,
            "entropy_difference_mev_per_K_nm2": entropy_difference,
            "canonical_free_energy_difference_mev_nm2": free_energy,
            "active_number_change_nm2": active_number_change,
            "reference_occupation_per_k": reference_occupation,
            "final_occupation_per_k": final_occupation,
            "integrated_charge_profile_nm2": charge_delta.integrated_profile_nm2,
            "charge_sum_rule_error_nm2": charge_delta.sum_rule_error_nm2,
        }
    )
    singular_values = (
        excitonic_fock_singular_values(sigma_fock, bundle.basis)
        if run.converged
        else None
    )
    if grand_potential is not None:
        state.diagnostics["grand_potential_difference_mev_nm2"] = grand_potential
        state.diagnostics["fixed_mu_mev"] = float(cfg.fixed_mu_mev)
    constraint_suffix = "" if symmetry_constraint_label is None else f" [{symmetry_constraint_label}]"
    if cfg.constraint_policy == "kane_poisson_fixed_mu":
        constraint_suffix += (
            f" [fixed Kane-Poisson ordinary-electron mu="
            f"{cfg.fixed_mu_mev:.12g} meV]"
        )
    if cfg.normal_ordering_reference_policy == "electron_hole_vacuum":
        constraint_suffix += " [E1-empty/H1-filled normal ordering]"
    classification = (
        f"converged {model_label} matrix-EI scaffold{constraint_suffix}"
        if run.converged
        else f"nonconverged matrix-EI scaffold{constraint_suffix}: {run.exit_reason}"
    )
    return MatrixEIResult(
        classification=classification,
        config=cfg,
        bundle_fingerprint=bundle_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        reference_mu_mev=float(normal.mu),
        reference_density=p0,
        noninteracting_density=noninteracting_density,
        density_delta=state.density.copy(),
        total_density=total_density,
        sigma_hartree_mev=np.asarray(sigma_hartree, dtype=np.complex128),
        sigma_fock_mev=np.asarray(sigma_fock, dtype=np.complex128),
        interaction_h_mev=np.asarray(sigma, dtype=np.complex128),
        hamiltonian_mev=state.hamiltonian.copy(),
        energies_mev=state.energies.copy(),
        mu_mev=float(state.mu),
        one_body_internal_energy_density_mev_nm2=one_body,
        hartree_internal_energy_density_mev_nm2=hartree,
        fock_internal_energy_density_mev_nm2=exchange,
        total_internal_energy_density_mev_nm2=internal,
        entropy_difference_mev_per_K_nm2=entropy_difference,
        canonical_free_energy_difference_mev_nm2=free_energy,
        grand_potential_difference_mev_nm2=grand_potential,
        excitonic_singular_values_mev=singular_values,
        run=run,
    )
