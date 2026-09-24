"""Ordinary-electron Hartree--Fock--BCS conventions for InAs/GaSb.

The production representation is number-conserving and ket-oriented::

    c = (E1 electrons, H1 electrons)
    P_ab(k) = <c_b(k)^dagger c_a(k)>
    P_ref = diag(0_E1, 1_H1)
    D = P - P_ref

``paper_literal_legacy`` remains isolated under ``src/ei_mf`` and is not
accepted or re-exported by this matrix-HF production module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .conventions import E1H1BasisSpec, active_electron_hole_areal_densities

ComplexArray = np.ndarray
FloatArray = np.ndarray
PhysicsMode = Literal["derived_ordinary_electron_hf_bcs"]

DERIVED_ORDINARY_ELECTRON_HF_BCS: PhysicsMode = (
    "derived_ordinary_electron_hf_bcs"
)
DERIVED_PHYSICS_STATUS = "derived_reference_subtracted_ordinary_electron_hf"
KB_MEV_PER_K = 0.08617333262


@dataclass(frozen=True)
class OrdinaryElectronConventionLock:
    """Typed lock for the maintained E1/H1 ordinary-electron convention."""

    physics_mode: PhysicsMode = DERIVED_ORDINARY_ELECTRON_HF_BCS
    basis_representation: str = "ordinary_electron"
    density_orientation: str = "P_ab=<c_b^dagger c_a>"
    coherence_definition: str = "chi_EH=P_EH=<H1^dagger E1>"
    normal_ordering: str = "D=P-P_ref"
    du_gap_definition: str = "Delta_Du_EH=-2*SigmaF_EH"
    active_column_order: str = "E1_pair_then_H1_pair"
    hole_interpretation: str = "H1_hole_is_vacancy_of_ordinary_electron_H1"
    kramers_pair_factor: int = 2
    additional_degeneracy_factor: int = 1
    energy_convention: str = "ordinary_electron_energies_no_hole_sign_flip"

    def __post_init__(self) -> None:
        if self.physics_mode != DERIVED_ORDINARY_ELECTRON_HF_BCS:
            raise ValueError(
                "the matrix-HF production convention accepts only "
                "derived_ordinary_electron_hf_bcs; use the isolated ei_mf "
                "historical interface for paper_literal_legacy"
            )
        if self.basis_representation != "ordinary_electron":
            raise ValueError("production basis must be ordinary_electron")
        if self.density_orientation != "P_ab=<c_b^dagger c_a>":
            raise ValueError("unsupported density-matrix orientation")
        if self.coherence_definition != "chi_EH=P_EH=<H1^dagger E1>":
            raise ValueError("unsupported coherence definition")
        if self.normal_ordering != "D=P-P_ref":
            raise ValueError("unsupported normal-ordering convention")
        if self.du_gap_definition != "Delta_Du_EH=-2*SigmaF_EH":
            raise ValueError("unsupported Du gap convention")
        if self.active_column_order != "E1_pair_then_H1_pair":
            raise ValueError("unsupported active-column order")
        if self.hole_interpretation != (
            "H1_hole_is_vacancy_of_ordinary_electron_H1"
        ):
            raise ValueError("unsupported H1-hole interpretation")
        if self.kramers_pair_factor != 2 or self.additional_degeneracy_factor != 1:
            raise ValueError("unsupported ordinary-electron degeneracy convention")
        if self.energy_convention != (
            "ordinary_electron_energies_no_hole_sign_flip"
        ):
            raise ValueError("unsupported ordinary-electron energy convention")


@dataclass(frozen=True)
class TwoBandOrdinaryElectronState:
    """Analytic two-band oracle in the Du doubled-gap normalization."""

    hamiltonian: ComplexArray
    eigenvalue_plus: FloatArray
    eigenvalue_minus: FloatArray
    f_plus: FloatArray
    f_minus: FloatArray
    density: ComplexArray
    density_delta: ComplexArray
    coherence: ComplexArray
    electron_occupation: FloatArray
    hole_occupation: FloatArray
    pair_energy: FloatArray


@dataclass(frozen=True)
class OrdinaryElectronOrderDiagnostics:
    """Keep coherence, Fock order, Du gap, and total EH mixing distinct."""

    p_ref: ComplexArray
    density: ComplexArray
    density_delta: ComplexArray
    sigma_hartree_mev: ComplexArray
    sigma_fock_mev: ComplexArray
    chi_eh: ComplexArray
    phi_eh_mev: ComplexArray
    delta_du_eh_mev: ComplexArray
    h0_eh_mev: ComplexArray
    hmf_eh_mev: ComplexArray
    electron_density_nm2: float
    hole_density_nm2: float
    delta_du_singular_values_mev: FloatArray


def ordinary_electron_reference_density(
    nk: int,
    basis: E1H1BasisSpec | None = None,
) -> ComplexArray:
    """Return ``diag(0_E1, 1_H1)`` on every momentum point."""

    if int(nk) != nk or nk < 1:
        raise ValueError("nk must be a positive integer")
    spec = E1H1BasisSpec() if basis is None else basis
    return np.repeat(
        spec.h1_electron_projector[:, :, None], int(nk), axis=2
    ).astype(np.complex128, copy=False)


def validate_cartesian_2d_weights(
    k_cart_nm_inv: FloatArray,
    weights_nm2: FloatArray,
    *,
    sampled_k_area_nm_minus2: float,
    atol: float = 1.0e-13,
) -> None:
    """Validate an explicit 2D quadrature for ``d^2k/(2*pi)^2``."""

    k = np.asarray(k_cart_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] != 2 or weights.shape != (k.shape[0],):
        raise ValueError("k_cart_nm_inv and weights_nm2 must have shapes (nk,2) and (nk,)")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(weights)):
        raise ValueError("2D momentum quadrature must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("all 2D momentum weights must be positive")
    area = float(sampled_k_area_nm_minus2)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("sampled_k_area_nm_minus2 must be finite and positive")
    expected = area / (2.0 * np.pi) ** 2
    if not np.isclose(np.sum(weights), expected, rtol=0.0, atol=float(atol)):
        raise ValueError("2D weights do not close to sampled area/(2*pi)^2")


def validate_explicit_2d_momentum_mesh(
    k_cart_nm_inv: FloatArray,
    weights_nm2: FloatArray,
    *,
    sampled_k_area_nm_minus2: float | None = None,
) -> float:
    """Validate a full-rank 2D mesh and return its sampled Cartesian area.

    Without an explicit area receipt, only a complete uniformly spaced
    Cartesian product is accepted.  An irreducible wedge must supply its
    expanded sampled area and already-expanded positive weights.
    """

    k = np.asarray(k_cart_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] != 2 or weights.shape != (k.shape[0],):
        raise ValueError("explicit 2D momentum data require shapes (nk,2) and (nk,)")
    centered = k - np.mean(k, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=1.0e-13) < 2:
        raise ValueError("production HF requires a genuinely two-dimensional momentum mesh")
    area = sampled_k_area_nm_minus2
    if area is None:
        x = np.unique(k[:, 0])
        y = np.unique(k[:, 1])
        if x.size < 2 or y.size < 2 or x.size * y.size != k.shape[0]:
            raise ValueError(
                "non-Cartesian or irreducible 2D meshes require an explicit sampled area receipt"
            )
        expected_points = {(float(kx), float(ky)) for kx in x for ky in y}
        actual_points = {(float(kx), float(ky)) for kx, ky in k}
        if actual_points != expected_points:
            raise ValueError("momentum points do not form a complete Cartesian product")
        dx = np.diff(x)
        dy = np.diff(y)
        if not np.allclose(dx, dx[0], rtol=0.0, atol=1.0e-13) or not np.allclose(
            dy, dy[0], rtol=0.0, atol=1.0e-13
        ):
            raise ValueError("Cartesian production mesh must be uniformly spaced")
        area = float(x.size * dx[0] * y.size * dy[0])
    validate_cartesian_2d_weights(
        k,
        weights,
        sampled_k_area_nm_minus2=float(area),
    )
    return float(area)


def validate_hartree_selection(
    *,
    projected_hartree_enabled: bool,
    scalar_hartree_enabled: bool,
) -> None:
    """Reject direct-Hartree double counting before entering SCF."""

    if projected_hartree_enabled and scalar_hartree_enabled:
        raise ValueError(
            "projected Hartree/Poisson and scalar V(0)n Hartree cannot both be enabled"
        )


def _fermi(energy_mev: FloatArray, temperature_K: float) -> FloatArray:
    if temperature_K <= 0.0 or not np.isfinite(temperature_K):
        raise ValueError("temperature_K must be finite and positive")
    x = np.asarray(energy_mev, dtype=np.float64) / (
        KB_MEV_PER_K * float(temperature_K)
    )
    out = np.empty_like(x)
    positive = x >= 0.0
    exp_minus = np.exp(-np.clip(x[positive], 0.0, 745.0))
    out[positive] = exp_minus / (1.0 + exp_minus)
    exp_plus = np.exp(np.clip(x[~positive], -745.0, 0.0))
    out[~positive] = 1.0 / (1.0 + exp_plus)
    return out


def matrix_fermi_density(
    hamiltonian_mev: ComplexArray,
    *,
    electron_mu_mev: float,
    temperature_K: float,
) -> tuple[ComplexArray, FloatArray]:
    """Return ket-oriented ``P=f_T(H-mu I)`` and ascending eigenvalues."""

    h = np.asarray(hamiltonian_mev, dtype=np.complex128)
    squeeze = h.ndim == 2
    if squeeze:
        h = h[:, :, None]
    if h.ndim != 3 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian_mev must have shape (n,n) or (n,n,nk)")
    if not np.isfinite(electron_mu_mev):
        raise ValueError("electron_mu_mev must be finite")
    error = float(np.max(np.abs(h - np.swapaxes(h.conj(), 0, 1))))
    if error > 1.0e-12:
        raise ValueError(f"hamiltonian is not Hermitian: error={error:.3e}")
    n, _, nk = h.shape
    density = np.empty_like(h)
    eigenvalues = np.empty((n, nk), dtype=np.float64)
    for ik in range(nk):
        values, vectors = np.linalg.eigh(h[:, :, ik])
        occupations = _fermi(values - float(electron_mu_mev), temperature_K)
        density[:, :, ik] = (
            vectors * occupations[None, :]
        ) @ vectors.conj().T
        eigenvalues[:, ik] = values
    if squeeze:
        return density[:, :, 0], eigenvalues[:, 0]
    return density, eigenvalues


def two_band_hf_bcs_hamiltonian(
    xi_mev: FloatArray,
    eta_mev: FloatArray,
    delta_du_mev: ComplexArray,
) -> ComplexArray:
    """Build the number-conserving two-band HF matrix in Du normalization."""

    xi, eta, delta = np.broadcast_arrays(
        np.asarray(xi_mev, dtype=np.float64),
        np.asarray(eta_mev, dtype=np.float64),
        np.asarray(delta_du_mev, dtype=np.complex128),
    )
    out = np.empty((2, 2) + xi.shape, dtype=np.complex128)
    out[0, 0] = 0.5 * (eta + xi)
    out[1, 1] = 0.5 * (eta - xi)
    out[0, 1] = -0.5 * delta
    out[1, 0] = -0.5 * delta.conj()
    return out


def two_band_ordinary_electron_state(
    xi_mev: FloatArray,
    eta_mev: FloatArray,
    delta_du_mev: ComplexArray,
    *,
    temperature_K: float,
    energy_floor_mev: float = 1.0e-14,
) -> TwoBandOrdinaryElectronState:
    """Evaluate the analytic density and occupations for the two-band oracle."""

    xi, eta, delta = np.broadcast_arrays(
        np.asarray(xi_mev, dtype=np.float64),
        np.asarray(eta_mev, dtype=np.float64),
        np.asarray(delta_du_mev, dtype=np.complex128),
    )
    pair_energy = np.sqrt(xi * xi + np.abs(delta) ** 2)
    safe = np.maximum(pair_energy, float(energy_floor_mev))
    eigenvalue_plus = 0.5 * (eta + pair_energy)
    eigenvalue_minus = 0.5 * (eta - pair_energy)
    f_plus = _fermi(eigenvalue_plus, temperature_K)
    f_minus = _fermi(eigenvalue_minus, temperature_K)
    blocking = f_minus - f_plus
    coherence = delta * blocking / (2.0 * safe)
    p_aa = 0.5 * (f_plus + f_minus - xi * blocking / safe)
    p_bb = 0.5 * (f_plus + f_minus + xi * blocking / safe)
    density = np.empty((2, 2) + xi.shape, dtype=np.complex128)
    density[0, 0] = p_aa
    density[0, 1] = coherence
    density[1, 0] = coherence.conj()
    density[1, 1] = p_bb
    p_ref = np.zeros_like(density)
    p_ref[1, 1] = 1.0
    return TwoBandOrdinaryElectronState(
        hamiltonian=two_band_hf_bcs_hamiltonian(xi, eta, delta),
        eigenvalue_plus=eigenvalue_plus,
        eigenvalue_minus=eigenvalue_minus,
        f_plus=f_plus,
        f_minus=f_minus,
        density=density,
        density_delta=density - p_ref,
        coherence=coherence,
        electron_occupation=p_aa,
        hole_occupation=1.0 - p_bb,
        pair_energy=pair_energy,
    )


def derived_xi_bracket(
    xi_mev: FloatArray,
    pair_energy_mev: FloatArray,
    g_minus: FloatArray,
    g_plus: FloatArray,
) -> FloatArray:
    """Reference-subtracted HF bracket in the paper's positive-excitation notation."""

    return 1.0 - np.asarray(xi_mev) / np.asarray(pair_energy_mev) * (
        1.0 - np.asarray(g_minus) - np.asarray(g_plus)
    )


def derived_eta_fock_factor_from_electron_occupations(
    f_minus: FloatArray,
    f_plus: FloatArray,
) -> FloatArray:
    """Return ``D_aa+D_bb=f_-+f_+-1`` for equal diagonal kernels."""

    return np.asarray(f_minus) + np.asarray(f_plus) - 1.0


def du_normalized_gap_from_fock(sigma_fock_eh_mev: ComplexArray) -> ComplexArray:
    """Return ``Delta_Du=-2 SigmaF_EH`` without conflating it with coherence."""

    return -2.0 * np.asarray(sigma_fock_eh_mev, dtype=np.complex128)


def ordinary_electron_order_diagnostics(
    *,
    density: ComplexArray,
    density_delta: ComplexArray,
    sigma_hartree_mev: ComplexArray,
    sigma_fock_mev: ComplexArray,
    h0_mev: ComplexArray,
    hmf_mev: ComplexArray,
    k_weights_nm2: FloatArray,
    basis: E1H1BasisSpec | None = None,
) -> OrdinaryElectronOrderDiagnostics:
    """Construct the required post-SCF ordinary-electron diagnostics."""

    spec = E1H1BasisSpec() if basis is None else basis
    p = np.asarray(density, dtype=np.complex128)
    d = np.asarray(density_delta, dtype=np.complex128)
    sigma_h = np.asarray(sigma_hartree_mev, dtype=np.complex128)
    sigma_f = np.asarray(sigma_fock_mev, dtype=np.complex128)
    h0 = np.asarray(h0_mev, dtype=np.complex128)
    hmf = np.asarray(hmf_mev, dtype=np.complex128)
    if p.ndim != 3:
        raise ValueError("all diagnostic matrices must have matching (4,4,nk) shapes")
    expected = (spec.dimension, spec.dimension, p.shape[2])
    if any(x.shape != expected for x in (p, d, sigma_h, sigma_f, h0, hmf)):
        raise ValueError("all diagnostic matrices must have matching (4,4,nk) shapes")
    p_ref = ordinary_electron_reference_density(p.shape[2], spec)
    if not np.allclose(d, p - p_ref, rtol=0.0, atol=1.0e-11):
        raise ValueError("density_delta is not P-P_ref in the ordinary-electron convention")
    e = np.asarray(spec.electron_indices)
    h = np.asarray(spec.hole_indices)
    chi = d[np.ix_(e, h, np.arange(p.shape[2]))]
    fock_eh = sigma_f[np.ix_(e, h, np.arange(p.shape[2]))]
    phi = -fock_eh
    delta_du = -2.0 * fock_eh
    h0_eh = h0[np.ix_(e, h, np.arange(p.shape[2]))]
    hmf_eh = hmf[np.ix_(e, h, np.arange(p.shape[2]))]
    singular = np.empty((p.shape[2], min(len(e), len(h))), dtype=np.float64)
    for ik in range(p.shape[2]):
        singular[ik] = np.linalg.svd(delta_du[:, :, ik], compute_uv=False)
    n_e, n_h = active_electron_hole_areal_densities(
        p, np.asarray(k_weights_nm2, dtype=np.float64), spec
    )
    return OrdinaryElectronOrderDiagnostics(
        p_ref=p_ref,
        density=p.copy(),
        density_delta=d.copy(),
        sigma_hartree_mev=sigma_h.copy(),
        sigma_fock_mev=sigma_f.copy(),
        chi_eh=chi,
        phi_eh_mev=phi,
        delta_du_eh_mev=delta_du,
        h0_eh_mev=h0_eh,
        hmf_eh_mev=hmf_eh,
        electron_density_nm2=n_e,
        hole_density_nm2=n_h,
        delta_du_singular_values_mev=singular,
    )
