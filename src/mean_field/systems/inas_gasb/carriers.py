"""Microscopic Gamma6/valence carrier accounting and density closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .projected_model import Kane4Bundle


KB_MEV_PER_K = 0.08617333262


@dataclass(frozen=True)
class KaneCarrierProjectors:
    electron_orbital: np.ndarray
    valence_orbital: np.ndarray
    bundle_fingerprint: str

    @classmethod
    def from_diabatic_e1_h1(cls, bundle: Kane4Bundle) -> "KaneCarrierProjectors":
        """Use fixed-reference E1 electrons and H1 vacancies as free carriers."""

        bundle.validate()
        electron = np.repeat(bundle.basis.electron_projector[:, :, None], bundle.nk, axis=2)
        valence = np.repeat(bundle.basis.h1_electron_projector[:, :, None], bundle.nk, axis=2)
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
        if electron.shape != valence.shape or electron.ndim != 3 or electron.shape[0] != electron.shape[1]:
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
        if bundle is not None:
            if self.bundle_fingerprint != bundle.fingerprint() or electron.shape[2] != bundle.nk:
                raise ValueError("carrier projectors do not match the Kane4 bundle")

    def densities_nm2(self, density: np.ndarray, k_weights_nm2: np.ndarray) -> tuple[float, float]:
        projector = np.asarray(density, dtype=np.complex128)
        weights = np.asarray(k_weights_nm2, dtype=float)
        if projector.shape != self.electron_orbital.shape or weights.shape != (projector.shape[2],):
            raise ValueError("density or k weights do not match carrier projectors")
        identity_minus = np.eye(projector.shape[0])[:, :, None] - projector
        n_e = np.einsum(
            "k,abk,bak->",
            weights,
            self.electron_orbital,
            projector,
            optimize=True,
        )
        n_h = np.einsum(
            "k,abk,bak->",
            weights,
            self.valence_orbital,
            identity_minus,
            optimize=True,
        )
        if max(abs(n_e.imag), abs(n_h.imag)) > 1e-9:
            raise ValueError("microscopic carrier densities are not real")
        return float(n_e.real), float(n_h.real)


@dataclass(frozen=True)
class NeutralDensityResult:
    density: np.ndarray
    energies_mev: np.ndarray
    mu_mev: float
    electron_density_nm2: float
    hole_density_nm2: float

    @property
    def charge_imbalance_nm2(self) -> float:
        return self.electron_density_nm2 - self.hole_density_nm2


def _fermi(energy_minus_mu_mev: np.ndarray, temperature_K: float) -> np.ndarray:
    x = np.asarray(energy_minus_mu_mev, dtype=float) / (KB_MEV_PER_K * float(temperature_K))
    out = np.empty_like(x)
    positive = x >= 0.0
    exp_minus = np.exp(-np.clip(x[positive], 0.0, 745.0))
    out[positive] = exp_minus / (1.0 + exp_minus)
    exp_plus = np.exp(np.clip(x[~positive], -745.0, 0.0))
    out[~positive] = 1.0 / (1.0 + exp_plus)
    return out


def charge_neutral_fermi_density(
    hamiltonian_mev: np.ndarray,
    k_weights_nm2: np.ndarray,
    carrier_projectors: KaneCarrierProjectors,
    *,
    temperature_K: float,
) -> NeutralDensityResult:
    """Solve the microscopic ``n_Gamma6 = n_valence_holes`` constraint."""

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    weights = np.asarray(k_weights_nm2, dtype=float)
    carrier_projectors.validate()
    if h.shape != carrier_projectors.electron_orbital.shape or weights.shape != (h.shape[2],):
        raise ValueError("Hamiltonian, weights, and carrier projectors do not match")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    n, _, nk = h.shape
    energies = np.empty((n, nk), dtype=float)
    vectors = np.empty((n, n, nk), dtype=np.complex128)
    for ik in range(nk):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])

    def density_at_mu(mu: float) -> np.ndarray:
        occupation = _fermi(energies - float(mu), temperature_K)
        density = np.empty_like(h)
        for ik in range(nk):
            density[:, :, ik] = (
                vectors[:, :, ik] * occupation[:, ik][None, :]
            ) @ vectors[:, :, ik].conj().T
        return density

    margin = max(10.0, 80.0 * KB_MEV_PER_K * temperature_K)
    lower = float(np.min(energies) - margin)
    upper = float(np.max(energies) + margin)

    def neutrality(mu: float) -> float:
        density = density_at_mu(mu)
        n_e, n_h = carrier_projectors.densities_nm2(density, weights)
        return n_e - n_h

    residual_lower = neutrality(lower)
    residual_upper = neutrality(upper)
    if not residual_lower < 0.0 < residual_upper:
        raise ValueError(
            "active Kane4 window does not bracket microscopic charge neutrality: "
            f"({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
        )
    mu = float(brentq(neutrality, lower, upper, xtol=1e-13, rtol=1e-14))
    density = density_at_mu(mu)
    n_e, n_h = carrier_projectors.densities_nm2(density, weights)
    return NeutralDensityResult(density, energies, mu, n_e, n_h)


def energy_sorted_pocket_density(
    hamiltonian_mev: np.ndarray,
    k_weights_nm2: np.ndarray,
    *,
    temperature_K: float,
) -> NeutralDensityResult:
    """Count occupied upper-pair states and vacancies in the lower pair."""

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    weights = np.asarray(k_weights_nm2, dtype=float)
    if h.ndim != 3 or h.shape[0] != h.shape[1] or h.shape[0] != 4:
        raise ValueError("energy-pocket closure requires a (4,4,nk) Hamiltonian")
    if weights.shape != (h.shape[2],) or temperature_K <= 0.0:
        raise ValueError("invalid pocket-density weights or temperature")
    energies = np.empty((4, h.shape[2]), dtype=float)
    vectors = np.empty_like(h)
    for ik in range(h.shape[2]):
        energies[:, ik], vectors[:, :, ik] = np.linalg.eigh(h[:, :, ik])
    target = 2.0 * float(np.sum(weights))

    def occupation(mu: float) -> np.ndarray:
        return _fermi(energies - float(mu), temperature_K)

    def total_residual(mu: float) -> float:
        return float(np.einsum("k,nk->", weights, occupation(mu), optimize=True) - target)

    margin = max(10.0, 80.0 * KB_MEV_PER_K * temperature_K)
    mu = float(
        brentq(
            total_residual,
            float(np.min(energies) - margin),
            float(np.max(energies) + margin),
            xtol=1e-13,
            rtol=1e-14,
        )
    )
    occupations = occupation(mu)
    density = np.empty_like(h)
    for ik in range(h.shape[2]):
        density[:, :, ik] = (
            vectors[:, :, ik] * occupations[:, ik][None, :]
        ) @ vectors[:, :, ik].conj().T
    n_e = float(np.einsum("k,nk->", weights, occupations[2:], optimize=True))
    n_h = float(np.einsum("k,nk->", weights, 1.0 - occupations[:2], optimize=True))
    return NeutralDensityResult(density, energies, mu, n_e, n_h)


@dataclass(frozen=True)
class DensityCalibratedDetuning:
    delta_tau_z_mev: float
    target_density_nm2: float
    achieved_electron_density_nm2: float
    achieved_hole_density_nm2: float
    mu_mev: float
    detuned_bundle: Kane4Bundle


def apply_relative_detuning(bundle: Kane4Bundle, delta_tau_z_mev: float) -> Kane4Bundle:
    """Apply the declared frozen-envelope one-body term ``delta*tau_z/2``."""

    detuned_h0 = np.asarray(bundle.h0_mev, dtype=np.complex128).copy()
    detuned_h0 += 0.5 * float(delta_tau_z_mev) * bundle.basis.tau_z[:, :, None]
    provenance = {
        **bundle.provenance,
        "frozen_envelope_relative_detuning": {
            "delta_tau_z_mev": float(delta_tau_z_mev),
            "term": "delta_tau_z_mev * tau_z / 2",
        },
    }
    return Kane4Bundle(
        k_cart_nm_inv=np.asarray(bundle.k_cart_nm_inv, dtype=float),
        weights_nm2=np.asarray(bundle.weights_nm2, dtype=float),
        z_nm=np.asarray(bundle.z_nm, dtype=float),
        z_weights_nm=np.asarray(bundle.z_weights_nm, dtype=float),
        h0_mev=detuned_h0,
        micro_wavefunctions=np.asarray(bundle.micro_wavefunctions, dtype=np.complex128),
        dhdk_mev_nm=None if bundle.dhdk_mev_nm is None else np.asarray(bundle.dhdk_mev_nm, dtype=np.complex128),
        time_reversal_unitary=(
            None
            if bundle.time_reversal_unitary is None
            else np.asarray(bundle.time_reversal_unitary, dtype=np.complex128)
        ),
        basis=bundle.basis,
        provenance=provenance,
    )


def calibrate_relative_detuning_to_pocket_density(
    bundle: Kane4Bundle,
    *,
    target_density_cm2: float,
    temperature_K: float,
    bracket_mev: tuple[float, float],
) -> DensityCalibratedDetuning:
    """Calibrate ``delta*tau_z/2`` from a declared cutoff-regulated energy-pair count."""

    target_nm2 = float(target_density_cm2) * 1e-14
    if target_nm2 <= 0.0:
        raise ValueError("target density must be positive")

    def evaluate(delta: float) -> tuple[float, NeutralDensityResult, Kane4Bundle]:
        detuned = apply_relative_detuning(bundle, delta)
        pocket = energy_sorted_pocket_density(
            detuned.h0_mev,
            detuned.weights_nm2,
            temperature_K=temperature_K,
        )
        density = 0.5 * (pocket.electron_density_nm2 + pocket.hole_density_nm2)
        return density - target_nm2, pocket, detuned

    lower, upper = map(float, bracket_mev)
    residual_lower = evaluate(lower)[0]
    residual_upper = evaluate(upper)[0]
    if residual_lower * residual_upper >= 0.0:
        raise ValueError(
            "pocket-density detuning bracket does not enclose target: "
            f"({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
        )
    delta = float(brentq(lambda value: evaluate(value)[0], lower, upper, xtol=1e-11, rtol=1e-12))
    _residual, pocket, detuned = evaluate(delta)
    provenance = {
        **detuned.provenance,
        "frozen_envelope_relative_detuning": {
            **detuned.provenance["frozen_envelope_relative_detuning"],
            "closure": "occupied upper energy pair equals lower-pair vacancies and target",
            "carrier_definition": "energy_sorted_pockets",
            "target_density_cm2": float(target_density_cm2),
            "temperature_K": float(temperature_K),
            "detuning_bracket_mev": [lower, upper],
        },
    }
    detuned = Kane4Bundle(**{**detuned.__dict__, "provenance": provenance})
    return DensityCalibratedDetuning(
        delta_tau_z_mev=delta,
        target_density_nm2=target_nm2,
        achieved_electron_density_nm2=pocket.electron_density_nm2,
        achieved_hole_density_nm2=pocket.hole_density_nm2,
        mu_mev=pocket.mu_mev,
        detuned_bundle=detuned,
    )


def calibrate_relative_detuning_to_density(
    bundle: Kane4Bundle,
    *,
    target_density_cm2: float,
    temperature_K: float,
    bracket_mev: tuple[float, float] = (-50.0, 50.0),
    carrier_definition: str = "microscopic_gamma6_valence",
) -> DensityCalibratedDetuning:
    """Determine frozen-envelope detuning from an independently supplied CNP density."""

    if target_density_cm2 <= 0.0:
        raise ValueError("target density must be positive")
    target_nm2 = float(target_density_cm2) * 1e-14
    if carrier_definition == "microscopic_gamma6_valence":
        carrier = KaneCarrierProjectors.from_bundle(bundle)
    elif carrier_definition == "diabatic_e1_h1":
        carrier = KaneCarrierProjectors.from_diabatic_e1_h1(bundle)
    else:
        raise ValueError("unsupported carrier_definition")

    def evaluate(delta: float) -> tuple[float, NeutralDensityResult, Kane4Bundle]:
        detuned = apply_relative_detuning(bundle, delta)
        neutral = charge_neutral_fermi_density(
            detuned.h0_mev,
            detuned.weights_nm2,
            carrier,
            temperature_K=temperature_K,
        )
        density = 0.5 * (neutral.electron_density_nm2 + neutral.hole_density_nm2)
        return density - target_nm2, neutral, detuned

    lower, upper = map(float, bracket_mev)
    residual_lower = evaluate(lower)[0]
    residual_upper = evaluate(upper)[0]
    if residual_lower * residual_upper >= 0.0:
        raise ValueError(
            "detuning bracket does not enclose the target density: "
            f"residuals=({residual_lower:.3e}, {residual_upper:.3e}) nm^-2"
        )
    delta = float(brentq(lambda value: evaluate(value)[0], lower, upper, xtol=1e-11, rtol=1e-12))
    _residual, neutral, detuned = evaluate(delta)
    final_provenance = {
        **detuned.provenance,
        "frozen_envelope_relative_detuning": {
            **detuned.provenance["frozen_envelope_relative_detuning"],
            "closure": "declared electron density equals declared hole density and target",
            "carrier_definition": carrier_definition,
            "target_density_cm2": float(target_density_cm2),
            "temperature_K": float(temperature_K),
        },
    }
    detuned = Kane4Bundle(**{**detuned.__dict__, "provenance": final_provenance})
    return DensityCalibratedDetuning(
        delta_tau_z_mev=delta,
        target_density_nm2=target_nm2,
        achieved_electron_density_nm2=neutral.electron_density_nm2,
        achieved_hole_density_nm2=neutral.hole_density_nm2,
        mu_mev=neutral.mu_mev,
        detuned_bundle=detuned,
    )
