"""Fail-closed normal-reference and chemical-potential checks for Kane HF.

The routines accept an arbitrary selected band window whose wavefunctions retain
all eight microscopic Kane orbital components.  They distinguish two ensembles:

* canonical/diagnostic neutrality: solve one electronic chemical potential;
* fixed-gate reservoir: accept a supplied reservoir chemical potential and only
  evaluate the resulting charge (the gate control must be solved elsewhere).

A numerically precise root in a finite band window is never labelled a physical
absolute chemical potential.  The typed contract and its blockers are evidence
for downstream production gates, not self-authorization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from .kane_source import CANONICAL_KANE_ORBITALS
from .kane4_bundle import Kane4Bundle

ComplexArray = np.ndarray
FloatArray = np.ndarray
ElectrostaticEnsemble = Literal[
    "diagnostic",
    "canonical_neutral_slab",
    "fixed_gate_reservoir",
]
KB_MEV_PER_K = 0.08617333262


@dataclass(frozen=True)
class KaneCarrierProjectors:
    """Microscopic Gamma6-electron and valence-hole projectors for a Kane4 bundle."""

    electron_orbital: ComplexArray
    valence_orbital: ComplexArray
    bundle_fingerprint: str

    @classmethod
    def from_diabatic_e1_h1(cls, bundle: Kane4Bundle) -> "KaneCarrierProjectors":
        bundle.validate()
        electron = np.repeat(
            bundle.basis.electron_projector[:, :, None], bundle.nk, axis=2
        )
        valence = np.repeat(
            bundle.basis.h1_electron_projector[:, :, None], bundle.nk, axis=2
        )
        result = cls(electron, valence, bundle.fingerprint())
        result.validate(bundle)
        return result

    @classmethod
    def from_bundle(cls, bundle: Kane4Bundle) -> "KaneCarrierProjectors":
        bundle.validate()
        phi = np.asarray(bundle.micro_wavefunctions, dtype=np.complex128)
        wz = np.asarray(bundle.z_weights_nm, dtype=float)
        electron = np.einsum(
            "kzma,z,kzmb->abk",
            phi[:, :, :2, :].conj(),
            wz,
            phi[:, :, :2, :],
            optimize=True,
        )
        identity = np.eye(bundle.basis.dimension)[:, :, None]
        valence = identity - electron
        result = cls(electron, valence, bundle.fingerprint())
        result.validate(bundle)
        return result

    def validate(self, bundle: Kane4Bundle | None = None, *, atol: float = 1e-8) -> None:
        electron = np.asarray(self.electron_orbital, dtype=np.complex128)
        valence = np.asarray(self.valence_orbital, dtype=np.complex128)
        if (
            electron.shape != valence.shape
            or electron.ndim != 3
            or electron.shape[0] != electron.shape[1]
        ):
            raise ValueError("carrier projectors must have matching (n,n,nk) shapes")
        if not all(np.all(np.isfinite(array)) for array in (electron, valence)):
            raise ValueError("carrier projectors must be finite")
        identity = np.eye(electron.shape[0])[:, :, None]
        completeness = float(np.max(np.abs(electron + valence - identity)))
        hermiticity = max(
            float(np.max(np.abs(electron - np.swapaxes(electron.conj(), 0, 1)))),
            float(np.max(np.abs(valence - np.swapaxes(valence.conj(), 0, 1)))),
        )
        if completeness > atol or hermiticity > atol:
            raise ValueError("carrier projectors violate completeness or Hermiticity")
        for ik in range(electron.shape[2]):
            eig_e = np.linalg.eigvalsh(electron[:, :, ik])
            eig_h = np.linalg.eigvalsh(valence[:, :, ik])
            if min(float(np.min(eig_e)), float(np.min(eig_h))) < -atol:
                raise ValueError("carrier projector is not positive semidefinite")
        if bundle is not None and (
            self.bundle_fingerprint != bundle.fingerprint()
            or electron.shape[2] != bundle.nk
        ):
            raise ValueError("carrier projectors do not match the Kane4 bundle")

    def densities_nm2(
        self, density: ComplexArray, k_weights_nm2: FloatArray
    ) -> tuple[float, float]:
        projector = np.asarray(density, dtype=np.complex128)
        weights = np.asarray(k_weights_nm2, dtype=float)
        if (
            projector.shape != self.electron_orbital.shape
            or weights.shape != (projector.shape[2],)
        ):
            raise ValueError("density or k weights do not match carrier projectors")
        identity_minus = np.eye(projector.shape[0])[:, :, None] - projector
        n_e = np.einsum(
            "k,abk,bak->", weights, self.electron_orbital, projector, optimize=True
        )
        n_h = np.einsum(
            "k,abk,bak->", weights, self.valence_orbital, identity_minus, optimize=True
        )
        if max(abs(n_e.imag), abs(n_h.imag)) > 1e-9:
            raise ValueError("microscopic carrier densities are not real")
        return float(n_e.real), float(n_h.real)


@dataclass(frozen=True)
class NeutralDensityResult:
    density: ComplexArray
    energies_mev: FloatArray
    mu_mev: float
    electron_density_nm2: float
    hole_density_nm2: float

    @property
    def charge_imbalance_nm2(self) -> float:
        return self.electron_density_nm2 - self.hole_density_nm2


@dataclass(frozen=True)
class ChemicalPotentialRootTolerances:
    xtol_mev: float = 1e-12
    rtol: float = 1e-13
    residual_atol_nm2: float = 1e-13
    residual_rtol: float = 1e-11
    covariance_energy_atol_mev: float = 1e-12
    covariance_density_matrix_atol: float = 1e-12
    covariance_carrier_atol_nm2: float = 1e-13
    covariance_rtol: float = 1e-11
    minimum_covariance_shift_mev: float = 1e-3

    def __post_init__(self) -> None:
        values = (
            self.xtol_mev,
            self.rtol,
            self.residual_atol_nm2,
            self.residual_rtol,
            self.covariance_energy_atol_mev,
            self.covariance_density_matrix_atol,
            self.covariance_carrier_atol_nm2,
            self.covariance_rtol,
            self.minimum_covariance_shift_mev,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("chemical-potential tolerances must be finite and positive")
        if self.rtol < 4.0 * np.finfo(float).eps:
            raise ValueError("Brent relative tolerance is below the float64 minimum")
        if self.residual_rtol >= 1.0 or self.covariance_rtol >= 1.0:
            raise ValueError("relative tolerances must be smaller than one")


@dataclass(frozen=True)
class KaneNormalReferenceContract:
    """Typed source metadata; booleans are declared gates, not authorization."""

    source_fingerprint: str
    selected_band_indices: tuple[int, ...]
    parent_hilbert_dimension: int
    calculation_split_mev: float
    energy_zero_label: str
    electrostatic_ensemble: ElectrostaticEnsemble
    kane_orbital_labels: tuple[str, ...] = CANONICAL_KANE_ORBITALS
    fixed_gate_source_complete: bool = False
    band_window_converged: bool = False
    momentum_window_converged: bool = False
    z_mesh_converged: bool = False
    poisson_residual_mev: float | None = None
    poisson_residual_tolerance_mev: float = 1e-5
    radial_only: bool = True

    def __post_init__(self) -> None:
        digest = str(self.source_fingerprint)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source_fingerprint must be a lowercase SHA-256 hex digest")
        indices = tuple(int(value) for value in self.selected_band_indices)
        if not indices or len(indices) != len(set(indices)) or any(value == 0 for value in indices):
            raise ValueError("selected_band_indices must be unique nonzero band labels")
        if int(self.parent_hilbert_dimension) < len(indices):
            raise ValueError("parent_hilbert_dimension is smaller than the selected band window")
        if tuple(self.kane_orbital_labels) != CANONICAL_KANE_ORBITALS:
            raise ValueError("Kane orbital labels/order do not match Supplementary Note 6")
        if not np.isfinite(self.calculation_split_mev) or self.calculation_split_mev < 0.0:
            raise ValueError("calculation_split_mev must be finite and nonnegative")
        if not str(self.energy_zero_label).strip():
            raise ValueError("energy_zero_label must be explicit")
        if self.electrostatic_ensemble not in {
            "diagnostic",
            "canonical_neutral_slab",
            "fixed_gate_reservoir",
        }:
            raise ValueError("unsupported electrostatic ensemble")
        if self.poisson_residual_mev is not None and (
            not np.isfinite(self.poisson_residual_mev) or self.poisson_residual_mev < 0.0
        ):
            raise ValueError("poisson_residual_mev must be finite and nonnegative")
        if (
            not np.isfinite(self.poisson_residual_tolerance_mev)
            or self.poisson_residual_tolerance_mev <= 0.0
        ):
            raise ValueError("poisson_residual_tolerance_mev must be positive")

    @property
    def selected_band_count(self) -> int:
        return len(self.selected_band_indices)

    @property
    def selected_fraction_of_parent(self) -> float:
        return self.selected_band_count / float(self.parent_hilbert_dimension)

    def credibility_blockers(self) -> tuple[str, ...]:
        """Return declared blockers; an external runner must verify every gate."""

        blockers: list[str] = []
        if self.calculation_split_mev != 0.0:
            blockers.append("calculation_split_is_nonzero")
        if self.electrostatic_ensemble == "diagnostic":
            blockers.append("electrostatic_ensemble_is_diagnostic")
        if self.electrostatic_ensemble == "fixed_gate_reservoir" and not self.fixed_gate_source_complete:
            blockers.append("fixed_gate_source_is_incomplete")
        if not self.band_window_converged:
            blockers.append("band_window_is_not_converged")
        if not self.momentum_window_converged:
            blockers.append("momentum_window_is_not_converged")
        if not self.z_mesh_converged:
            blockers.append("z_mesh_is_not_converged")
        if self.radial_only:
            blockers.append("source_is_radial_only")
        if self.poisson_residual_mev is None:
            blockers.append("poisson_residual_is_missing")
        elif self.poisson_residual_mev > self.poisson_residual_tolerance_mev:
            blockers.append("poisson_residual_exceeds_tolerance")
        return tuple(blockers)


@dataclass(frozen=True)
class KaneChargeNeutralReference:
    """Window-restricted numerical neutrality root in a canonical ensemble."""

    contract: KaneNormalReferenceContract
    tolerances: ChemicalPotentialRootTolerances
    temperature_K: float
    density: ComplexArray
    energies_mev: FloatArray
    mu_mev: float
    electron_density_nm2: float
    hole_density_nm2: float
    target_metric_occupation_nm2: float
    achieved_metric_occupation_nm2: float
    number_residual_nm2: float
    neutrality_residual_nm2: float
    chemical_potential_bracket_mev: tuple[float, float]
    bracket_residuals_nm2: tuple[float, float]
    root_iterations: int
    root_function_calls: int
    root_converged: bool
    frame_orthonormality_error: float
    orbital_projector_completeness_error: float
    hamiltonian_hermiticity_error_mev: float

    @property
    def numerical_residual_tolerance_nm2(self) -> float:
        scale = max(
            abs(self.target_metric_occupation_nm2),
            abs(self.electron_density_nm2),
            abs(self.hole_density_nm2),
            np.finfo(float).tiny,
        )
        return self.tolerances.residual_atol_nm2 + self.tolerances.residual_rtol * scale

    @property
    def numerically_certified(self) -> bool:
        tolerance = self.numerical_residual_tolerance_nm2
        return (
            self.root_converged
            and abs(self.number_residual_nm2) <= tolerance
            and abs(self.neutrality_residual_nm2) <= tolerance
            and self.frame_orthonormality_error <= 1e-8
            and self.orbital_projector_completeness_error <= 1e-8
            and self.hamiltonian_hermiticity_error_mev <= 1e-9
        )

    @property
    def claim_scope(self) -> str:
        return "window_restricted_numerical_charge_neutrality_root"


@dataclass(frozen=True)
class KaneFixedMuState:
    """State evaluated at a supplied reservoir chemical potential."""

    contract: KaneNormalReferenceContract
    temperature_K: float
    density: ComplexArray
    energies_mev: FloatArray
    reservoir_mu_mev: float
    electron_density_nm2: float
    hole_density_nm2: float
    neutrality_residual_nm2: float
    frame_orthonormality_error: float
    orbital_projector_completeness_error: float
    hamiltonian_hermiticity_error_mev: float

    @property
    def claim_scope(self) -> str:
        return "fixed_reservoir_mu_charge_evaluation_not_cnp_root"


@dataclass(frozen=True)
class ChemicalPotentialShiftCovariance:
    baseline_reference_fingerprint: str
    shifted_source_fingerprint: str
    tolerances: ChemicalPotentialRootTolerances
    energy_shift_mev: float
    baseline_mu_mev: float
    shifted_mu_mev: float
    baseline_electron_density_nm2: float
    shifted_electron_density_nm2: float
    baseline_hole_density_nm2: float
    shifted_hole_density_nm2: float
    mu_shift_error_mev: float
    energy_shift_error_mev: float
    density_max_error: float
    electron_density_error_nm2: float
    hole_density_error_nm2: float

    @property
    def passed(self) -> bool:
        energy_scale = max(
            abs(self.energy_shift_mev),
            abs(self.baseline_mu_mev),
            abs(self.shifted_mu_mev),
            np.finfo(float).tiny,
        )
        energy_bound = (
            self.tolerances.covariance_energy_atol_mev
            + self.tolerances.covariance_rtol * energy_scale
        )
        density_matrix_bound = (
            self.tolerances.covariance_density_matrix_atol
            + self.tolerances.covariance_rtol
        )
        electron_scale = max(
            abs(self.baseline_electron_density_nm2),
            abs(self.shifted_electron_density_nm2),
            np.finfo(float).tiny,
        )
        hole_scale = max(
            abs(self.baseline_hole_density_nm2),
            abs(self.shifted_hole_density_nm2),
            np.finfo(float).tiny,
        )
        electron_bound = (
            self.tolerances.covariance_carrier_atol_nm2
            + self.tolerances.covariance_rtol * electron_scale
        )
        hole_bound = (
            self.tolerances.covariance_carrier_atol_nm2
            + self.tolerances.covariance_rtol * hole_scale
        )
        return (
            self.mu_shift_error_mev <= energy_bound
            and self.energy_shift_error_mev <= energy_bound
            and self.density_max_error <= density_matrix_bound
            and self.electron_density_error_nm2 <= electron_bound
            and self.hole_density_error_nm2 <= hole_bound
        )


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))


def kane_window_source_fingerprint(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
) -> str:
    """Hash the exact array payload consumed by the normal-reference solver."""

    digest = hashlib.sha256()
    for values in (
        np.asarray(hamiltonian_mev, dtype=np.complex128),
        np.asarray(k_weights_nm2, dtype=np.float64),
        np.asarray(micro_wavefunctions, dtype=np.complex128),
        np.asarray(z_weights_nm, dtype=np.float64),
    ):
        _hash_array(digest, values)
    return digest.hexdigest()


def _validate_kane_window(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
    contract: KaneNormalReferenceContract,
) -> tuple[ComplexArray, FloatArray, ComplexArray, FloatArray, float, float]:
    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    phi = np.asarray(micro_wavefunctions, dtype=np.complex128)
    wz = np.asarray(z_weights_nm, dtype=float)
    n = contract.selected_band_count
    if h.ndim != 3 or h.shape[:2] != (n, n):
        raise ValueError("hamiltonian_mev must have shape (nband, nband, nk)")
    nk = h.shape[2]
    if wk.shape != (nk,) or np.any(wk <= 0.0) or not np.all(np.isfinite(wk)):
        raise ValueError("k_weights_nm2 must be finite and positive with shape (nk,)")
    if phi.ndim != 4 or phi.shape[0] != nk or phi.shape[2:] != (8, n):
        raise ValueError("micro_wavefunctions must have shape (nk, nz, 8, nband)")
    if wz.shape != (phi.shape[1],) or np.any(wz <= 0.0) or not np.all(np.isfinite(wz)):
        raise ValueError("z_weights_nm must be finite and positive with shape (nz,)")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(phi)):
        raise ValueError("Kane window arrays must be finite")
    expected_fingerprint = kane_window_source_fingerprint(h, wk, phi, wz)
    if contract.source_fingerprint != expected_fingerprint:
        raise ValueError("source_fingerprint does not bind the supplied Kane arrays")
    hermiticity = float(np.max(np.abs(h - np.swapaxes(h.conj(), 0, 1))))
    gram = np.einsum("kzma,z,kzmb->kab", phi.conj(), wz, phi, optimize=True)
    orthonormality = float(np.max(np.abs(gram - np.eye(n)[None, :, :])))
    if hermiticity > 1e-9:
        raise ValueError(f"Kane-window Hamiltonian is not Hermitian: {hermiticity:.3e} meV")
    if orthonormality > 1e-8:
        raise ValueError(f"Kane-window frame is not orthonormal: {orthonormality:.3e}")
    return h, wk, phi, wz, hermiticity, orthonormality


def _orbital_projectors(
    phi: ComplexArray,
    z_weights_nm: FloatArray,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, float]:
    electron = np.einsum(
        "kzma,z,kzmb->abk",
        phi[:, :, :2, :].conj(),
        z_weights_nm,
        phi[:, :, :2, :],
        optimize=True,
    )
    valence = np.einsum(
        "kzma,z,kzmb->abk",
        phi[:, :, 2:, :].conj(),
        z_weights_nm,
        phi[:, :, 2:, :],
        optimize=True,
    )
    metric = electron + valence
    identity = np.eye(phi.shape[3], dtype=np.complex128)[:, :, None]
    completeness = float(np.max(np.abs(metric - identity)))
    return electron, valence, metric, completeness


def _fermi(energy_minus_mu_mev: FloatArray, temperature_K: float) -> FloatArray:
    x = np.asarray(energy_minus_mu_mev, dtype=float) / (KB_MEV_PER_K * float(temperature_K))
    out = np.empty_like(x)
    positive = x >= 0.0
    exp_minus = np.exp(-np.clip(x[positive], 0.0, 745.0))
    out[positive] = exp_minus / (1.0 + exp_minus)
    exp_plus = np.exp(np.clip(x[~positive], -745.0, 0.0))
    out[~positive] = 1.0 / (1.0 + exp_plus)
    return out


def _diagonalize(h: ComplexArray) -> tuple[FloatArray, ComplexArray]:
    n, _, nk = h.shape
    energies = np.empty((n, nk), dtype=float)
    vectors = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])
    return energies, vectors


def _density_at_mu(
    energies: FloatArray,
    vectors: ComplexArray,
    mu_mev: float,
    temperature_K: float,
) -> ComplexArray:
    occupation = _fermi(energies - float(mu_mev), temperature_K)
    density = np.empty_like(vectors)
    for ik in range(vectors.shape[2]):
        density[:, :, ik] = (
            vectors[:, :, ik] * occupation[:, ik][None, :]
        ) @ vectors[:, :, ik].conj().T
    return density


def charge_neutral_fermi_density(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    carrier_projectors: KaneCarrierProjectors,
    *,
    temperature_K: float,
) -> NeutralDensityResult:
    """Solve microscopic Gamma6-electron/valence-hole charge neutrality."""

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    weights = np.asarray(k_weights_nm2, dtype=float)
    carrier_projectors.validate()
    if (
        h.shape != carrier_projectors.electron_orbital.shape
        or weights.shape != (h.shape[2],)
    ):
        raise ValueError("Hamiltonian, weights, and carrier projectors do not match")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    energies, vectors = _diagonalize(h)

    def evaluate(mu_mev: float) -> tuple[ComplexArray, float, float]:
        density = _density_at_mu(energies, vectors, mu_mev, temperature_K)
        n_e, n_h = carrier_projectors.densities_nm2(density, weights)
        return density, n_e, n_h

    def neutrality(mu_mev: float) -> float:
        _density, n_e, n_h = evaluate(mu_mev)
        return n_e - n_h

    margin = max(10.0, 80.0 * KB_MEV_PER_K * temperature_K)
    lower = float(np.min(energies) - margin)
    upper = float(np.max(energies) + margin)
    residual_lower = neutrality(lower)
    residual_upper = neutrality(upper)
    if not residual_lower < 0.0 < residual_upper:
        raise ValueError(
            "active Kane4 window does not bracket microscopic charge neutrality: "
            f"({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
        )
    mu = float(
        brentq(
            neutrality,
            lower,
            upper,
            xtol=1e-13,
            rtol=1e-14,
        )
    )
    density, n_e, n_h = evaluate(mu)
    return NeutralDensityResult(density, energies, mu, n_e, n_h)


def _carrier_densities(
    density: ComplexArray,
    k_weights_nm2: FloatArray,
    electron: ComplexArray,
    valence: ComplexArray,
) -> tuple[float, float]:
    n_e = np.einsum("k,abk,bak->", k_weights_nm2, electron, density, optimize=True)
    identity_minus_density = np.eye(density.shape[0], dtype=np.complex128)[:, :, None] - density
    n_h = np.einsum(
        "k,abk,bak->",
        k_weights_nm2,
        valence,
        identity_minus_density,
        optimize=True,
    )
    if max(abs(n_e.imag), abs(n_h.imag)) > 1e-9:
        raise ValueError("Kane orbital-transfer carrier densities are not real")
    return float(n_e.real), float(n_h.real)


def _window_components(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
    contract: KaneNormalReferenceContract,
) -> tuple[
    ComplexArray,
    FloatArray,
    ComplexArray,
    FloatArray,
    float,
    float,
    ComplexArray,
    ComplexArray,
    ComplexArray,
    float,
    FloatArray,
    ComplexArray,
]:
    h, wk, phi, wz, hermiticity, orthonormality = _validate_kane_window(
        hamiltonian_mev,
        k_weights_nm2,
        micro_wavefunctions,
        z_weights_nm,
        contract,
    )
    electron, valence, metric, completeness = _orbital_projectors(phi, wz)
    if completeness > 1e-8:
        raise ValueError(f"Kane orbital projectors are incomplete: error={completeness:.3e}")
    energies, vectors = _diagonalize(h)
    return (
        h,
        wk,
        phi,
        wz,
        hermiticity,
        orthonormality,
        electron,
        valence,
        metric,
        completeness,
        energies,
        vectors,
    )


def build_charge_neutral_kane_reference(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
    *,
    contract: KaneNormalReferenceContract,
    temperature_K: float,
    tolerances: ChemicalPotentialRootTolerances | None = None,
) -> KaneChargeNeutralReference:
    """Solve Note-6 orbital-transfer neutrality in a canonical band window."""

    if contract.electrostatic_ensemble == "fixed_gate_reservoir":
        raise ValueError(
            "fixed_gate_reservoir has a supplied mu; use evaluate_kane_state_at_fixed_mu"
        )
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    tol = ChemicalPotentialRootTolerances() if tolerances is None else tolerances
    (
        _h,
        wk,
        _phi,
        _wz,
        hermiticity,
        orthonormality,
        electron,
        valence,
        metric,
        completeness,
        energies,
        vectors,
    ) = _window_components(
        hamiltonian_mev,
        k_weights_nm2,
        micro_wavefunctions,
        z_weights_nm,
        contract,
    )
    target_metric = float(np.einsum("k,aak->", wk, valence, optimize=True).real)

    def evaluate(mu_mev: float) -> tuple[float, ComplexArray, float, float, float]:
        density = _density_at_mu(energies, vectors, mu_mev, float(temperature_K))
        n_e, n_h = _carrier_densities(density, wk, electron, valence)
        achieved_metric = float(np.einsum("k,abk,bak->", wk, metric, density, optimize=True).real)
        number_residual = achieved_metric - target_metric
        neutrality_residual = n_e - n_h
        identity_tolerance = tol.residual_atol_nm2 + tol.residual_rtol * max(
            abs(target_metric), abs(n_e), abs(n_h), np.finfo(float).tiny
        )
        if abs(number_residual - neutrality_residual) > identity_tolerance:
            raise ValueError("microscopic neutrality and metric-number identities disagree")
        return neutrality_residual, density, n_e, n_h, achieved_metric

    thermal_margin = max(10.0, 80.0 * KB_MEV_PER_K * float(temperature_K))
    lower = float(np.min(energies) - thermal_margin)
    upper = float(np.max(energies) + thermal_margin)
    residual_lower = evaluate(lower)[0]
    residual_upper = evaluate(upper)[0]
    if not residual_lower < 0.0 < residual_upper:
        raise ValueError(
            "selected Kane window does not bracket charge neutrality: "
            f"({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
        )
    mu, root = brentq(
        lambda value: evaluate(float(value))[0],
        lower,
        upper,
        xtol=tol.xtol_mev,
        rtol=tol.rtol,
        full_output=True,
        disp=False,
    )
    neutrality, density, n_e, n_h, achieved_metric = evaluate(float(mu))
    number_residual = achieved_metric - target_metric
    result = KaneChargeNeutralReference(
        contract=contract,
        tolerances=tol,
        temperature_K=float(temperature_K),
        density=density,
        energies_mev=energies,
        mu_mev=float(mu),
        electron_density_nm2=n_e,
        hole_density_nm2=n_h,
        target_metric_occupation_nm2=target_metric,
        achieved_metric_occupation_nm2=achieved_metric,
        number_residual_nm2=number_residual,
        neutrality_residual_nm2=neutrality,
        chemical_potential_bracket_mev=(lower, upper),
        bracket_residuals_nm2=(residual_lower, residual_upper),
        root_iterations=int(root.iterations),
        root_function_calls=int(root.function_calls),
        root_converged=bool(root.converged),
        frame_orthonormality_error=orthonormality,
        orbital_projector_completeness_error=completeness,
        hamiltonian_hermiticity_error_mev=hermiticity,
    )
    if not result.numerically_certified:
        raise ValueError("charge-neutral Kane reference failed numerical certification")
    return result


def evaluate_kane_state_at_fixed_mu(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
    *,
    contract: KaneNormalReferenceContract,
    temperature_K: float,
    reservoir_mu_mev: float,
) -> KaneFixedMuState:
    """Evaluate, but do not root-solve, a fixed-reservoir Kane state."""

    if contract.electrostatic_ensemble != "fixed_gate_reservoir":
        raise ValueError("fixed-mu evaluation requires electrostatic_ensemble='fixed_gate_reservoir'")
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    if not np.isfinite(reservoir_mu_mev):
        raise ValueError("reservoir_mu_mev must be finite")
    (
        _h,
        wk,
        _phi,
        _wz,
        hermiticity,
        orthonormality,
        electron,
        valence,
        _metric,
        completeness,
        energies,
        vectors,
    ) = _window_components(
        hamiltonian_mev,
        k_weights_nm2,
        micro_wavefunctions,
        z_weights_nm,
        contract,
    )
    density = _density_at_mu(energies, vectors, float(reservoir_mu_mev), float(temperature_K))
    n_e, n_h = _carrier_densities(density, wk, electron, valence)
    return KaneFixedMuState(
        contract=contract,
        temperature_K=float(temperature_K),
        density=density,
        energies_mev=energies,
        reservoir_mu_mev=float(reservoir_mu_mev),
        electron_density_nm2=n_e,
        hole_density_nm2=n_h,
        neutrality_residual_nm2=n_e - n_h,
        frame_orthonormality_error=orthonormality,
        orbital_projector_completeness_error=completeness,
        hamiltonian_hermiticity_error_mev=hermiticity,
    )


def chemical_potential_shift_covariance(
    hamiltonian_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    micro_wavefunctions: ComplexArray,
    z_weights_nm: FloatArray,
    *,
    contract: KaneNormalReferenceContract,
    temperature_K: float,
    energy_shift_mev: float,
    tolerances: ChemicalPotentialRootTolerances | None = None,
) -> ChemicalPotentialShiftCovariance:
    """Verify that an energy-zero shift moves mu but leaves the state invariant."""

    tol = ChemicalPotentialRootTolerances() if tolerances is None else tolerances
    shift = float(energy_shift_mev)
    if not np.isfinite(shift) or abs(shift) < tol.minimum_covariance_shift_mev:
        raise ValueError("energy_shift_mev is nonfinite or too small for a nonvacuous check")
    baseline = build_charge_neutral_kane_reference(
        hamiltonian_mev,
        k_weights_nm2,
        micro_wavefunctions,
        z_weights_nm,
        contract=contract,
        temperature_K=temperature_K,
        tolerances=tol,
    )
    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    identity = np.eye(h.shape[0], dtype=np.complex128)[:, :, None]
    shifted_h = h + shift * identity
    shifted_contract = replace(
        contract,
        source_fingerprint=kane_window_source_fingerprint(
            shifted_h,
            k_weights_nm2,
            micro_wavefunctions,
            z_weights_nm,
        ),
    )
    shifted = build_charge_neutral_kane_reference(
        shifted_h,
        k_weights_nm2,
        micro_wavefunctions,
        z_weights_nm,
        contract=shifted_contract,
        temperature_K=temperature_K,
        tolerances=tol,
    )
    diagnostics = ChemicalPotentialShiftCovariance(
        baseline_reference_fingerprint=normal_reference_fingerprint(baseline),
        shifted_source_fingerprint=shifted_contract.source_fingerprint,
        tolerances=tol,
        energy_shift_mev=shift,
        baseline_mu_mev=baseline.mu_mev,
        shifted_mu_mev=shifted.mu_mev,
        baseline_electron_density_nm2=baseline.electron_density_nm2,
        shifted_electron_density_nm2=shifted.electron_density_nm2,
        baseline_hole_density_nm2=baseline.hole_density_nm2,
        shifted_hole_density_nm2=shifted.hole_density_nm2,
        mu_shift_error_mev=abs((shifted.mu_mev - baseline.mu_mev) - shift),
        energy_shift_error_mev=float(
            np.max(np.abs((shifted.energies_mev - baseline.energies_mev) - shift))
        ),
        density_max_error=float(np.max(np.abs(shifted.density - baseline.density))),
        electron_density_error_nm2=abs(
            shifted.electron_density_nm2 - baseline.electron_density_nm2
        ),
        hole_density_error_nm2=abs(shifted.hole_density_nm2 - baseline.hole_density_nm2),
    )
    if not diagnostics.passed:
        raise ValueError("chemical-potential energy-zero covariance failed")
    return diagnostics


def normal_reference_fingerprint(reference: KaneChargeNeutralReference) -> str:
    """Hash the exact reference arrays, contract, root settings, and diagnostics."""

    digest = hashlib.sha256()
    contract_json = json.dumps(asdict(reference.contract), sort_keys=True, separators=(",", ":"))
    tolerances_json = json.dumps(asdict(reference.tolerances), sort_keys=True, separators=(",", ":"))
    digest.update(contract_json.encode())
    digest.update(tolerances_json.encode())
    for scalar in (
        reference.temperature_K,
        reference.mu_mev,
        reference.electron_density_nm2,
        reference.hole_density_nm2,
        reference.target_metric_occupation_nm2,
        reference.achieved_metric_occupation_nm2,
        reference.number_residual_nm2,
        reference.neutrality_residual_nm2,
        *reference.chemical_potential_bracket_mev,
        *reference.bracket_residuals_nm2,
        reference.frame_orthonormality_error,
        reference.orbital_projector_completeness_error,
        reference.hamiltonian_hermiticity_error_mev,
    ):
        digest.update(np.float64(scalar).tobytes())
    digest.update(np.int64(reference.root_iterations).tobytes())
    digest.update(np.int64(reference.root_function_calls).tobytes())
    digest.update(bytes([int(reference.root_converged)]))
    for values in (reference.density, reference.energies_mev):
        _hash_array(digest, np.asarray(values))
    return digest.hexdigest()
