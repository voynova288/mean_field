from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from analysis.response_derivative_gauge import (
    berry_connection_generalized_derivative_pair as _gauge_generalized_derivative_pair,
    berry_connection_pair as _gauge_berry_connection_pair,
)

from .constants import KB_EV_PER_K, SHIFT_CURRENT_PREFAC_UA_NM_PER_V2

Axis = int
Component = tuple[Axis, Axis, Axis]  # (output current a, optical b, optical c), x=0, y=1

AXIS_LABELS = {"x": 0, "y": 1, 0: 0, 1: 1}


def axis_index(axis: str | int) -> int:
    try:
        value = AXIS_LABELS[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported axis {axis!r}; expected x/y or 0/1") from exc
    return int(value)


def parse_component(text: str) -> Component:
    """Parse labels such as 'x;yy' or 'y;xx'."""

    if ";" not in text:
        raise ValueError(f"Component must look like 'x;yy', got {text!r}")
    left, right = text.split(";", 1)
    if len(left) != 1 or len(right) != 2:
        raise ValueError(f"Component must look like 'x;yy', got {text!r}")
    return (axis_index(left), axis_index(right[0]), axis_index(right[1]))


def component_label(component: Component) -> str:
    labels = ("x", "y")
    a, b, c = component
    return f"{labels[a]};{labels[b]}{labels[c]}"


def lorentzian_delta(photon_energies_ev: np.ndarray, transition_energy_ev: float, eta_ev: float) -> np.ndarray:
    """Lorentzian replacement for delta(E_gamma - E_mn), normalized in eV^{-1}."""

    photon_energies_ev = np.asarray(photon_energies_ev, dtype=float)
    eta = float(eta_ev)
    if eta <= 0.0:
        raise ValueError(f"eta_ev must be positive, got {eta_ev}")
    diff = photon_energies_ev - float(transition_energy_ev)
    return (eta / np.pi) / (diff * diff + eta * eta)


def fermi_occupation(energies_ev: np.ndarray, *, mu_ev: float = 0.0, temperature_k: float = 0.0) -> np.ndarray:
    """Fermi occupation for energies in eV."""

    energies = np.asarray(energies_ev, dtype=float)
    if float(temperature_k) <= 0.0:
        return (energies < float(mu_ev)).astype(float)
    beta_arg = (energies - float(mu_ev)) / (KB_EV_PER_K * float(temperature_k))
    out = np.empty_like(beta_arg, dtype=float)
    out[beta_arg > 40.0] = 0.0
    out[beta_arg < -40.0] = 1.0
    mask = (beta_arg >= -40.0) & (beta_arg <= 40.0)
    out[mask] = 1.0 / (np.exp(beta_arg[mask]) + 1.0)
    return out


def velocity_matrices(evecs: np.ndarray, dhdk: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Return D[a,n,m] = <u_n|partial_{k_a} H|u_m> in eV*nm.

    ``evecs`` must store eigenvectors as columns.  The returned first axis is
    (x, y).  This function intentionally works with dH/dk instead of velocity;
    hbar cancels from the gauge-free formulas used below.
    """

    vectors = np.asarray(evecs, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError(f"evecs must be a 2D matrix with eigenvectors as columns, got {vectors.shape}")
    if len(dhdk) != 2:
        raise ValueError("dhdk must contain (dH/dkx, dH/dky)")
    out = np.empty((2, vectors.shape[1], vectors.shape[1]), dtype=np.complex128)
    udag = vectors.conjugate().T
    for axis, deriv in enumerate(dhdk):
        deriv = np.asarray(deriv, dtype=np.complex128)
        if deriv.shape != (vectors.shape[0], vectors.shape[0]):
            raise ValueError(f"dhdk[{axis}] has shape {deriv.shape}, expected {(vectors.shape[0], vectors.shape[0])}")
        out[axis] = udag @ deriv @ vectors
    return out


def second_derivative_matrices(
    evecs: np.ndarray,
    d2hdk: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | np.ndarray,
) -> np.ndarray:
    """Return W[a,b,n,m] = <u_n|partial_{k_a}partial_{k_b} H|u_m>.

    Mao Eq. (4) is written for the continuum Dirac Hamiltonian used for hTTG,
    where this tensor is zero.  Tight-binding validation models, such as the
    Appendix-A gapped graphene benchmark, have nonzero second derivatives and
    need the extra ``-i W/(E_n-E_m)`` term in the generalized derivative.
    """

    vectors = np.asarray(evecs, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError(f"evecs must be a 2D matrix with eigenvectors as columns, got {vectors.shape}")
    raw = np.asarray(d2hdk, dtype=np.complex128)
    expected = (2, 2, vectors.shape[0], vectors.shape[0])
    if raw.shape != expected:
        raise ValueError(f"d2hdk has shape {raw.shape}, expected {expected}")
    out = np.empty((2, 2, vectors.shape[1], vectors.shape[1]), dtype=np.complex128)
    udag = vectors.conjugate().T
    for axis_a in range(2):
        for axis_b in range(2):
            out[axis_a, axis_b] = udag @ raw[axis_a, axis_b] @ vectors
    return out


def berry_connection_from_D(D: np.ndarray, energies_ev: np.ndarray, *, denominator_cutoff_ev: float) -> np.ndarray:
    """Gauge-free interband Berry connection r_nm^a = -i D_nm^a/(E_n-E_m)."""

    energies = np.asarray(energies_ev, dtype=float)
    D = np.asarray(D, dtype=np.complex128)
    nb = energies.size
    if D.shape != (2, nb, nb):
        raise ValueError(f"D has shape {D.shape}, expected {(2, nb, nb)}")
    r = np.zeros_like(D)
    cutoff = float(denominator_cutoff_ev)
    for n in range(nb):
        for m in range(nb):
            if n == m:
                continue
            denom = energies[n] - energies[m]
            if abs(denom) <= cutoff:
                continue
            r[:, n, m] = -1.0j * D[:, n, m] / denom
    return r


@dataclass(frozen=True)
class GeneralizedDerivativeResult:
    values: np.ndarray  # values[deriv_a, conn_b, n, m] = r_{nm;a}^b
    skipped_small_denominators: int


def generalized_derivative_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> GeneralizedDerivativeResult:
    """Gauge-free generalized derivative in the energy-difference convention.

    Implements the work-document rewrite of Mao et al. Eq. (4):

        r_{nm;a}^b = i/(E_n-E_m) sum_{l!=n,m}
            [D^b_{nl} D^a_{lm}/(E_l-E_m) - D^a_{nl} D^b_{lm}/(E_n-E_l)]
          + i/(E_n-E_m)^2 [D^a_{nm} Delta D^b_{nm} + D^b_{nm} Delta D^a_{nm}]
          - i W^{ab}_{nm}/(E_n-E_m).

    The last term is absent from Mao Eq. (4) because the hTTG Dirac blocks are
    linear in k.  It is required for nonlinear tight-binding Hamiltonians.

    The first axis of the output is the derivative/current direction ``a``;
    the second is the Berry-connection direction ``b``.
    """

    energies = np.asarray(energies_ev, dtype=float)
    D = np.asarray(D, dtype=np.complex128)
    nb = energies.size
    if D.shape != (2, nb, nb):
        raise ValueError(f"D has shape {D.shape}, expected {(2, nb, nb)}")
    W_array = None if W is None else np.asarray(W, dtype=np.complex128)
    if W_array is not None and W_array.shape != (2, 2, nb, nb):
        raise ValueError(f"W has shape {W_array.shape}, expected {(2, 2, nb, nb)}")

    cutoff = float(denominator_cutoff_ev)
    values = np.zeros((2, 2, nb, nb), dtype=np.complex128)
    skipped = 0
    for n in range(nb):
        for m in range(nb):
            if n == m:
                continue
            e_nm = energies[n] - energies[m]
            if abs(e_nm) <= cutoff:
                skipped += 1
                continue
            delta_D = D[:, n, n] - D[:, m, m]
            for deriv_a in range(2):
                for conn_b in range(2):
                    total = 0.0 + 0.0j
                    for ell in range(nb):
                        if ell == n or ell == m:
                            continue
                        e_lm = energies[ell] - energies[m]
                        e_nl = energies[n] - energies[ell]
                        if abs(e_lm) <= cutoff or abs(e_nl) <= cutoff:
                            skipped += 1
                            continue
                        total += (
                            D[conn_b, n, ell] * D[deriv_a, ell, m] / e_lm
                            - D[deriv_a, n, ell] * D[conn_b, ell, m] / e_nl
                        )
                    total *= 1.0j / e_nm
                    total += (
                        1.0j
                        / (e_nm * e_nm)
                        * (D[deriv_a, n, m] * delta_D[conn_b] + D[conn_b, n, m] * delta_D[deriv_a])
                    )
                    if W_array is not None:
                        total += -1.0j * W_array[deriv_a, conn_b, n, m] / e_nm
                    values[deriv_a, conn_b, n, m] = total
    return GeneralizedDerivativeResult(values=values, skipped_small_denominators=skipped)


@dataclass(frozen=True)
class PairGeneralizedDerivativeResult:
    values: np.ndarray  # values[deriv_a, conn_b] = r_{nm;a}^b for one selected pair
    skipped_small_denominators: int


def generalized_derivative_pair_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    n: int,
    m: int,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> PairGeneralizedDerivativeResult:
    """Vectorized gauge-free generalized derivative for one band pair.

    This is the production-oriented version of ``generalized_derivative_from_D``
    when only selected transitions are needed.  It still sums over every
    intermediate band ``ell != n,m`` but avoids constructing the full
    ``(2,2,N,N)`` tensor.
    """

    result = _gauge_generalized_derivative_pair(
        D,
        energies_ev,
        int(n),
        int(m),
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=W,
    )
    return PairGeneralizedDerivativeResult(
        values=result.values,
        skipped_small_denominators=result.skipped_small_denominators,
    )


def berry_connection_pair_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    n: int,
    m: int,
    *,
    denominator_cutoff_ev: float,
) -> np.ndarray:
    """Return r_nm^a for one selected pair as a two-component vector."""

    return _gauge_berry_connection_pair(
        D,
        energies_ev,
        int(n),
        int(m),
        denominator_cutoff=float(denominator_cutoff_ev),
    )


def component_transition_weight_from_D(
    D: np.ndarray,
    energies_ev: np.ndarray,
    occupations: np.ndarray,
    n: int,
    m: int,
    component: Component,
    *,
    denominator_cutoff_ev: float,
    W: np.ndarray | None = None,
) -> tuple[complex, int]:
    """Return a selected-pair transition weight and skipped-denominator count."""

    a, b, c = component
    r_mn = berry_connection_pair_from_D(
        D,
        energies_ev,
        m,
        n,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    gd_nm = generalized_derivative_pair_from_D(
        D,
        energies_ev,
        n,
        m,
        denominator_cutoff_ev=denominator_cutoff_ev,
        W=W,
    )
    fnm = float(np.asarray(occupations, dtype=float)[int(n)] - np.asarray(occupations, dtype=float)[int(m)])
    weight = fnm * (r_mn[b] * gd_nm.values[a, c] + r_mn[c] * gd_nm.values[a, b])
    return complex(weight), int(gd_nm.skipped_small_denominators)


@dataclass(frozen=True)
class ResponseTensors:
    energies_ev: np.ndarray
    occupations: np.ndarray
    D: np.ndarray
    r: np.ndarray
    r_covariant: np.ndarray
    skipped_small_denominators: int


def precompute_response_tensors(
    energies_ev: np.ndarray,
    evecs: np.ndarray,
    dhdk: tuple[np.ndarray, np.ndarray],
    *,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
    d2hdk: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | np.ndarray | None = None,
) -> ResponseTensors:
    energies = np.asarray(energies_ev, dtype=float)
    D = velocity_matrices(evecs, dhdk)
    W = None if d2hdk is None else second_derivative_matrices(evecs, d2hdk)
    r = berry_connection_from_D(D, energies, denominator_cutoff_ev=denominator_cutoff_ev)
    gd = generalized_derivative_from_D(D, energies, denominator_cutoff_ev=denominator_cutoff_ev, W=W)
    occupations = fermi_occupation(energies, mu_ev=mu_ev, temperature_k=temperature_k)
    return ResponseTensors(
        energies_ev=energies,
        occupations=occupations,
        D=D,
        r=r,
        r_covariant=gd.values,
        skipped_small_denominators=gd.skipped_small_denominators,
    )


def component_transition_weight(tensors: ResponseTensors, n: int, m: int, component: Component) -> complex:
    """Return f_nm (r_mn^b r_nm;a^c + r_mn^c r_nm;a^b) for one band pair."""

    a, b, c = component
    fnm = float(tensors.occupations[n] - tensors.occupations[m])
    return fnm * (
        tensors.r[b, m, n] * tensors.r_covariant[a, c, n, m]
        + tensors.r[c, m, n] * tensors.r_covariant[a, b, n, m]
    )


def positive_transition_terms(
    tensors: ResponseTensors,
    component: Component,
    *,
    min_transition_ev: float = 0.0,
    min_abs_occupation_diff: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect positive-energy interband transition energies and component weights."""

    energies = tensors.energies_ev
    nb = energies.size
    transitions: list[float] = []
    weights: list[complex] = []
    for n in range(nb):
        for m in range(nb):
            if n == m:
                continue
            transition_ev = float(energies[m] - energies[n])
            if transition_ev <= float(min_transition_ev):
                continue
            if abs(float(tensors.occupations[n] - tensors.occupations[m])) < min_abs_occupation_diff:
                continue
            weight = component_transition_weight(tensors, n, m, component)
            if np.isfinite(weight.real) and np.isfinite(weight.imag):
                transitions.append(transition_ev)
                weights.append(complex(weight))
    return np.asarray(transitions, dtype=float), np.asarray(weights, dtype=np.complex128)


def add_transitions_to_integral(
    integral: np.ndarray,
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
) -> None:
    """Accumulate one k-point's transitions into the BZ integral in-place."""

    if transition_energies_ev.size == 0:
        return
    factor = float(k_weight_nm_inv_sq) / (2.0 * np.pi) ** 2
    for transition_ev, weight in zip(transition_energies_ev, transition_weights, strict=True):
        integral += factor * weight * lorentzian_delta(photon_energies_ev, float(transition_ev), eta_ev)


def sigma_from_integral(integral_nm_per_ev: np.ndarray) -> np.ndarray:
    """Convert the accumulated complex integral to sigma in microampere*nm/V^2."""

    return np.real(-1.0j * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 * np.asarray(integral_nm_per_ev))


def spectra_from_transition_table(
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
) -> dict[str, np.ndarray]:
    """Build spectra for several named components from transition arrays.

    ``transition_table[name]`` stores ``(transition_energies, weights)`` for an
    already chosen set of k points.  ``k_weight_nm_inv_sq`` is applied to each
    entry; for pre-aggregated tables pass the desired total weight explicitly.
    """

    spectra: dict[str, np.ndarray] = {}
    for name, (transition_energies, weights) in transition_table.items():
        integral = np.zeros_like(photon_energies_ev, dtype=np.complex128)
        add_transitions_to_integral(
            integral,
            photon_energies_ev,
            np.asarray(transition_energies, dtype=float),
            np.asarray(weights, dtype=np.complex128),
            k_weight_nm_inv_sq=k_weight_nm_inv_sq,
            eta_ev=eta_ev,
        )
        spectra[name] = sigma_from_integral(integral)
    return spectra
