"""Coulomb kernels for the isotropic electron-hole bilayer EI model."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline
from scipy.special import ellipkm1

from .grids import q_floor_from_grid
from .params import EIParams
from .units import COULOMB_MEV_NM
from .wavefunction_form_factor import (
    KaneLayerProfiles,
    KaneMomentumLayerProfiles,
    _integration_weights,
)

Array = NDArray[np.float64]


def v_intralayer(q_nm_inv: Array, eps_r: float) -> Array:
    """2D Fourier intralayer Coulomb potential in meV nm^2."""

    q = np.asarray(q_nm_inv, dtype=np.float64)
    return 2.0 * np.pi * COULOMB_MEV_NM / float(eps_r) / q


def v_interlayer(q_nm_inv: Array, eps_r: float, d_nm: float) -> Array:
    """Positive e-h interlayer attraction kernel magnitude in meV nm^2."""

    q = np.asarray(q_nm_inv, dtype=np.float64)
    return v_intralayer(q, eps_r) * np.exp(-q * float(d_nm))


def _apply_screening(vq: Array, q_eff: Array, params: EIParams) -> Array:
    if params.screening_model == "none" or params.q_tf_nm_inv <= 0:
        return vq
    if params.screening_model == "tf":
        return vq / (1.0 + params.q_tf_nm_inv / q_eff)
    raise ValueError(f"unsupported screening model: {params.screening_model!r}")


def angular_averaged_kernel(
    k_nm_inv: Array,
    params: EIParams,
    *,
    kind: str = "interlayer",
    block_rows: int = 16,
    progress: bool = False,
    form_factor_profiles: KaneLayerProfiles | None = None,
) -> Array:
    """Build K(k,k')=(1/2pi) int dphi V(|k-k'|).

    ``kind`` is ``"interlayer"`` for K_ab or ``"intralayer"`` for K_aa.
    The kernel is positive; the e-h attraction sign belongs to the Hamiltonian,
    not to this gap-equation kernel.
    """

    params.validate()
    if kind not in {"interlayer", "intralayer"}:
        raise ValueError("kind must be 'interlayer' or 'intralayer'")
    k = np.asarray(k_nm_inv, dtype=np.float64)
    if k.ndim != 1 or k.size < 4:
        raise ValueError("k_nm_inv must be a one-dimensional grid")
    q_floor = q_floor_from_grid(k, params.q_floor_nm_inv)
    nphi = int(params.nphi)
    cos_phi = np.cos(2.0 * np.pi * np.arange(nphi, dtype=np.float64) / nphi)
    out = np.empty((k.size, k.size), dtype=np.float64)
    if form_factor_profiles is not None:
        q_table = np.linspace(0.0, max(2.0 * float(np.max(k)), q_floor), 4097)
        Faa_table, _Fbb_table, Fab_table = form_factor_profiles.form_factors(q_table)
    else:
        q_table = Faa_table = Fab_table = None
    block_rows = max(1, int(block_rows))
    kj = k[None, :, None]
    cos = cos_phi[None, None, :]
    for start in range(0, k.size, block_rows):
        stop = min(start + block_rows, k.size)
        ki = k[start:stop, None, None]
        q2 = ki * ki + kj * kj - 2.0 * ki * kj * cos
        q = np.sqrt(np.maximum(q2, 0.0))
        q_eff = np.maximum(q, q_floor)
        if form_factor_profiles is None:
            if kind == "interlayer":
                vq = v_interlayer(q_eff, params.eps_r, params.d_eh_nm)
            else:
                vq = v_intralayer(q_eff, params.eps_r)
        else:
            assert q_table is not None and Faa_table is not None and Fab_table is not None
            table = Fab_table if kind == "interlayer" else Faa_table
            form_factor = np.interp(q_eff, q_table, table)
            vq = v_intralayer(q_eff, params.eps_r) * form_factor
        vq = _apply_screening(vq, q_eff, params)
        out[start:stop, :] = np.mean(vq, axis=2)
        if progress:
            print(f"[kernel:{kind}] rows {start}:{stop} / {k.size}", file=sys.stderr, flush=True)
    # Enforce the exact symmetry of the angular-average definition at roundoff level.
    out = 0.5 * (out + out.T)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite Coulomb kernel entries")
    if np.any(out < 0.0):
        raise FloatingPointError("negative Coulomb kernel entries")
    return out


def _cell_integrated_intralayer_dimensionless(
    k_nm_inv: Array,
    weights_nm2: Array,
    *,
    diagonal_order: int,
) -> Array:
    """Return the unscreened point-layer ``∫dφ/q`` kernel.

    Off-diagonal entries use the exact complete-elliptic-integral angular
    average.  Each logarithmically singular diagonal entry is integrated over
    its radial source cell before division by that cell's ``k dk/(2π)``
    weight.  This is the physical-grid counterpart of the source-corrected
    Zhu/du-thesis diagonal-cell treatment in :mod:`ei_mf.zhu1995`.
    """

    k = np.asarray(k_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    area = np.concatenate(([0.0], np.cumsum(weights)))
    edges = np.sqrt(4.0 * np.pi * area)
    if np.any(k <= edges[:-1]) or np.any(k >= edges[1:]):
        raise ValueError("k points must lie inside the annular cells implied by weights")

    ki = k[:, None]
    kj = k[None, :]
    denom = ki + kj
    complementary = ((ki - kj) / denom) ** 2
    kernel = 4.0 * ellipkm1(complementary) / denom
    np.fill_diagonal(kernel, np.nan)

    x, wx = leggauss(int(diagonal_order))

    def segment(k0: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        kp = 0.5 * (hi - lo) * x + 0.5 * (lo + hi)
        wk = 0.5 * (hi - lo) * wx
        pair_denom = k0 + kp
        pair_complementary = ((k0 - kp) / pair_denom) ** 2
        angular = 4.0 * ellipkm1(pair_complementary) / pair_denom
        return float(np.dot(wk, kp * angular / (2.0 * np.pi)))

    for index, (k0, lo, hi, weight) in enumerate(
        zip(k, edges[:-1], edges[1:], weights)
    ):
        cell_integral = segment(float(k0), float(lo), float(k0))
        cell_integral += segment(float(k0), float(k0), float(hi))
        kernel[index, index] = cell_integral / float(weight)
    return kernel


def angular_averaged_kernel_cell_integrated(
    k_nm_inv: Array,
    weights_nm2: Array,
    params: EIParams,
    *,
    kind: str = "interlayer",
    nphi_regular: int | None = None,
    diagonal_order: int = 256,
    regular_cell_order: int = 128,
    form_factor_profiles: KaneLayerProfiles | None = None,
) -> Array:
    """Build the source-model Coulomb kernel with exact singular-cell handling.

    This implementation is restricted deliberately to the unscreened paper
    model. It removes the coupled ``q_floor``/angular-grid regulator from the
    logarithmic ``1/q`` diagonal. For point layers, the interlayer finite
    correction is ``(exp(-q d)-1)/q``. Optional source-derived Kane profiles
    instead use ``(F_xy(q)-1)/q``; in both cases the regular correction is
    angularly integrated with Gauss-Legendre quadrature and cell-integrated on
    the diagonal.
    """

    params.validate()
    if kind not in {"interlayer", "intralayer"}:
        raise ValueError("kind must be 'interlayer' or 'intralayer'")
    if params.screening_model != "none" or params.q_tf_nm_inv != 0.0:
        raise ValueError("cell-integrated paper-model kernel requires unscreened Coulomb")
    if diagonal_order < 32 or regular_cell_order < 16:
        raise ValueError("cell-integration quadrature orders are too small")
    nphi = int(params.nphi if nphi_regular is None else nphi_regular)
    if nphi < 32:
        raise ValueError("nphi_regular must be at least 32")

    k = np.asarray(k_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if not (
        k.ndim == weights.ndim == 1
        and k.size == weights.size
        and k.size >= 4
        and np.all(np.diff(k) > 0.0)
        and np.all(weights > 0.0)
    ):
        raise ValueError("k and weights must be matching positive radial-grid vectors")

    prefactor = COULOMB_MEV_NM / float(params.eps_r)
    intralayer = prefactor * _cell_integrated_intralayer_dimensionless(
        k, weights, diagonal_order=diagonal_order
    )
    if form_factor_profiles is None and (
        kind == "intralayer" or params.d_eh_nm == 0.0
    ):
        return intralayer.copy()

    q_table: Array | None = None
    form_factor_table: Array | None = None
    form_factor_slope0: float | None = None
    if form_factor_profiles is not None:
        radial_edge_max = float(np.sqrt(4.0 * np.pi * np.sum(weights)))
        q_table = np.linspace(0.0, max(2.0 * radial_edge_max, 1e-6), 4097)
        Faa_table, _Fbb_table, Fab_table = form_factor_profiles.form_factors(q_table)
        form_factor_table = Fab_table if kind == "interlayer" else Faa_table
        if not np.isclose(form_factor_table[0], 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("finite-width form factor must satisfy F(0)=1")
        form_factor_slope0 = float(
            (form_factor_table[1] - form_factor_table[0])
            / (q_table[1] - q_table[0])
        )

    area = np.concatenate(([0.0], np.cumsum(weights)))
    edges = np.sqrt(4.0 * np.pi * area)
    xphi, wphi = leggauss(nphi)
    phi = np.pi * (xphi + 1.0)
    angular_weights = np.pi * wphi
    distance = float(params.d_eh_nm)

    def regular_correction(k0: float, kp: Array) -> Array:
        radial = np.asarray(kp, dtype=np.float64)
        q = np.sqrt(
            np.maximum(
                k0 * k0
                + radial[:, None] ** 2
                - 2.0 * k0 * radial[:, None] * np.cos(phi),
                0.0,
            )
        )
        regular = np.empty_like(q)
        mask = q > 1e-13
        if form_factor_table is None:
            regular[mask] = np.expm1(-distance * q[mask]) / q[mask]
            regular[~mask] = -distance
        else:
            assert q_table is not None and form_factor_slope0 is not None
            form_factor = np.interp(q, q_table, form_factor_table)
            regular[mask] = (form_factor[mask] - 1.0) / q[mask]
            regular[~mask] = form_factor_slope0
        return regular @ angular_weights

    correction = np.empty((k.size, k.size), dtype=np.float64)
    for index, k0 in enumerate(k):
        correction[index] = regular_correction(float(k0), k)
    correction = 0.5 * (correction + correction.T)

    xcell, wcell = leggauss(int(regular_cell_order))
    for index, (k0, lo, hi, weight) in enumerate(
        zip(k, edges[:-1], edges[1:], weights)
    ):
        cell_integral = 0.0
        for left, right in ((float(lo), float(k0)), (float(k0), float(hi))):
            if right <= left:
                continue
            kp = 0.5 * (right - left) * xcell + 0.5 * (left + right)
            wk = 0.5 * (right - left) * wcell
            radial_measure = kp / (2.0 * np.pi)
            cell_integral += float(
                np.dot(wk, radial_measure * regular_correction(float(k0), kp))
            )
        correction[index, index] = cell_integral / float(weight)

    result = intralayer + prefactor * correction
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("invalid cell-integrated Coulomb kernel")
    return result


@dataclass(frozen=True)
class RadialScreeningSpec:
    """Typed forensic screening hypothesis for a 2D point-layer kernel.

    ``single_metal_gate`` uses the image-charge factor
    ``1-exp(-2 q D)`` with gate distance ``D``. ``symmetric_dual_metal_gate``
    uses ``tanh(q D)`` for a layer at the midplane between two gates separated
    by ``2D``. ``thomas_fermi`` replaces ``1/q`` by ``1/(q+q_tf)``. These are
    explicit hypothesis branches and are not attributed to the Du authors.
    """

    model: str
    gate_distance_nm: float | None = None
    q_tf_nm_inv: float | None = None

    def __post_init__(self) -> None:
        allowed = {
            "single_metal_gate",
            "symmetric_dual_metal_gate",
            "thomas_fermi",
        }
        if self.model not in allowed:
            raise ValueError(f"unsupported radial screening model: {self.model!r}")
        if self.model in {"single_metal_gate", "symmetric_dual_metal_gate"}:
            if (
                self.gate_distance_nm is None
                or not np.isfinite(self.gate_distance_nm)
                or self.gate_distance_nm <= 0.0
            ):
                raise ValueError("metal-gate screening requires a positive gate distance")
            if self.q_tf_nm_inv is not None:
                raise ValueError("metal-gate screening does not accept q_tf_nm_inv")
        else:
            if (
                self.q_tf_nm_inv is None
                or not np.isfinite(self.q_tf_nm_inv)
                or self.q_tf_nm_inv <= 0.0
            ):
                raise ValueError("Thomas-Fermi screening requires positive q_tf_nm_inv")
            if self.gate_distance_nm is not None:
                raise ValueError("Thomas-Fermi screening does not accept a gate distance")

    def inverse_momentum_nm(self, q_nm_inv: Array) -> Array:
        """Return the finite screened replacement for ``1/q`` in nm."""

        q = np.asarray(q_nm_inv, dtype=np.float64)
        if np.any(q < 0.0) or not np.all(np.isfinite(q)):
            raise ValueError("q must be finite and nonnegative")
        if self.model == "thomas_fermi":
            assert self.q_tf_nm_inv is not None
            return 1.0 / (q + float(self.q_tf_nm_inv))
        assert self.gate_distance_nm is not None
        distance = float(self.gate_distance_nm)
        result = np.empty_like(q)
        nonzero = q > 1e-13
        if self.model == "single_metal_gate":
            result[nonzero] = -np.expm1(-2.0 * distance * q[nonzero]) / q[nonzero]
            result[~nonzero] = 2.0 * distance
        else:
            result[nonzero] = np.tanh(distance * q[nonzero]) / q[nonzero]
            result[~nonzero] = distance
        return result



def angular_averaged_screened_kernel_cell_integrated(
    k_nm_inv: Array,
    weights_nm2: Array,
    params: EIParams,
    screening: RadialScreeningSpec,
    *,
    kind: str = "interlayer",
    nphi: int = 240,
    radial_cell_order: int = 128,
) -> Array:
    """Build a finite screened point-layer kernel on annular radial cells.

    Screening removes the ``1/q`` logarithmic singularity. Off-diagonal cells
    use their radial nodes; every diagonal source cell is integrated with the
    exact ``k' dk'/(2*pi)`` measure. The interlayer channel additionally uses
    the published point-layer factor ``exp(-q d)``.
    """

    params.validate()
    if kind not in {"interlayer", "intralayer"}:
        raise ValueError("kind must be 'interlayer' or 'intralayer'")
    if params.screening_model != "none" or params.q_tf_nm_inv != 0.0:
        raise ValueError("screening must be supplied only through RadialScreeningSpec")
    if int(nphi) != nphi or nphi < 32 or radial_cell_order < 16:
        raise ValueError("screened-kernel quadrature orders are too small")
    k = np.asarray(k_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if not (
        k.ndim == weights.ndim == 1
        and k.size == weights.size
        and k.size >= 4
        and np.all(np.diff(k) > 0.0)
        and np.all(weights > 0.0)
    ):
        raise ValueError("k and weights must be matching positive radial-grid vectors")
    area = np.concatenate(([0.0], np.cumsum(weights)))
    edges = np.sqrt(4.0 * np.pi * area)
    if np.any(k <= edges[:-1]) or np.any(k >= edges[1:]):
        raise ValueError("k points must lie inside the annular cells implied by weights")

    xphi, wphi = leggauss(int(nphi))
    phi = np.pi * (xphi + 1.0)
    angular_weights = np.pi * wphi
    cosine = np.cos(phi)
    interlayer_distance = float(params.d_eh_nm)
    if screening.model == "thomas_fermi":
        assert screening.q_tf_nm_inv is not None
        screening_momentum_scale = float(screening.q_tf_nm_inv)
    elif screening.model == "single_metal_gate":
        assert screening.gate_distance_nm is not None
        screening_momentum_scale = 1.0 / (2.0 * float(screening.gate_distance_nm))
    else:
        assert screening.gate_distance_nm is not None
        screening_momentum_scale = 1.0 / float(screening.gate_distance_nm)
    angular_resolution = 2.0 * float(edges[-1]) * np.sin(0.5 * float(phi[0]))
    radial_resolution = float(np.max(np.diff(edges))) / float(radial_cell_order)
    required_scale = 8.0 * max(angular_resolution, radial_resolution)
    if screening_momentum_scale < required_scale:
        raise ValueError(
            "screening crossover is under-resolved by the requested quadrature: "
            f"scale={screening_momentum_scale:.3e} nm^-1, "
            f"required>={required_scale:.3e} nm^-1"
        )

    def angular_integral(k0: float, kp: Array) -> Array:
        radial = np.asarray(kp, dtype=np.float64)
        q = np.sqrt(
            np.maximum(
                k0 * k0
                + radial[:, None] ** 2
                - 2.0 * k0 * radial[:, None] * cosine[None, :],
                0.0,
            )
        )
        values = screening.inverse_momentum_nm(q)
        if kind == "interlayer" and interlayer_distance != 0.0:
            values *= np.exp(-interlayer_distance * q)
        return values @ angular_weights

    dimensionless = np.empty((k.size, k.size), dtype=np.float64)
    for index, k0 in enumerate(k):
        dimensionless[index] = angular_integral(float(k0), k)
    dimensionless = 0.5 * (dimensionless + dimensionless.T)

    xcell, wcell = leggauss(int(radial_cell_order))
    for index, (k0, lo, hi, weight) in enumerate(
        zip(k, edges[:-1], edges[1:], weights)
    ):
        kp = 0.5 * (hi - lo) * xcell + 0.5 * (lo + hi)
        wk = 0.5 * (hi - lo) * wcell
        radial_measure = kp / (2.0 * np.pi)
        cell_integral = float(
            np.dot(wk, radial_measure * angular_integral(float(k0), kp))
        )
        dimensionless[index, index] = cell_integral / float(weight)

    result = COULOMB_MEV_NM / float(params.eps_r) * dimensionless
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("invalid screened cell-integrated Coulomb kernel")
    return result


@dataclass(frozen=True)
class KDependentDensityKernels:
    """Hermitian scalar kernels from momentum-resolved endpoint densities."""

    K_eh_mev_nm2: Array
    K_ee_mev_nm2: Array
    K_hh_mev_nm2: Array
    K_aa_balanced_mev_nm2: Array
    source_regular_correction_eh: Array
    source_regular_correction_ee: Array
    source_regular_correction_hh: Array
    profile_mode: str
    frozen_k_nm_inv: float | None
    q_samples: int
    nphi_regular: int

    def validate(self) -> None:
        shape = self.K_eh_mev_nm2.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("k-dependent kernels must be square")
        for kernel in (
            self.K_eh_mev_nm2,
            self.K_ee_mev_nm2,
            self.K_hh_mev_nm2,
            self.K_aa_balanced_mev_nm2,
        ):
            if kernel.shape != shape or not np.all(np.isfinite(kernel)):
                raise ValueError("invalid k-dependent kernel")
            if np.max(np.abs(kernel - kernel.T)) > 2e-12 * np.max(np.abs(kernel)):
                raise ValueError("k-dependent kernel is not symmetric")
            if np.any(kernel < 0.0):
                raise ValueError("k-dependent Coulomb kernel is negative")
        if not np.allclose(
            self.K_aa_balanced_mev_nm2,
            0.5 * (self.K_ee_mev_nm2 + self.K_hh_mev_nm2),
            rtol=0.0,
            atol=2e-12 * np.max(np.abs(self.K_aa_balanced_mev_nm2)),
        ):
            raise ValueError("balanced same-species kernel mismatch")


def _angular_regular_correction_from_form_factor_table(
    source_k_nm_inv: Array,
    q_table_nm_inv: Array,
    form_factor_table: Array,
    *,
    nphi_regular: int,
) -> Array:
    """Return ``∫dφ (F(k,k',q)-1)/q`` on the source k grid."""

    source_k = np.asarray(source_k_nm_inv, dtype=np.float64)
    q_table = np.asarray(q_table_nm_inv, dtype=np.float64)
    table = np.asarray(form_factor_table, dtype=np.float64)
    if table.shape != (source_k.size, source_k.size, q_table.size):
        raise ValueError("form-factor table shape mismatch")
    if not np.allclose(table[:, :, 0], 1.0, rtol=0.0, atol=2e-8):
        raise ValueError("momentum-dependent form factors must satisfy F(k,k',0)=1")
    xphi, wphi = leggauss(int(nphi_regular))
    phi = np.pi * (xphi + 1.0)
    angular_weights = np.pi * wphi
    cosine = np.cos(phi)
    correction = np.empty((source_k.size, source_k.size), dtype=np.float64)
    slope0 = (table[:, :, 1] - table[:, :, 0]) / (q_table[1] - q_table[0])
    for i, ki in enumerate(source_k):
        for j, kj in enumerate(source_k):
            q = np.sqrt(np.maximum(ki * ki + kj * kj - 2.0 * ki * kj * cosine, 0.0))
            form_factor = np.interp(q, q_table, table[i, j])
            regular = np.empty_like(q)
            mask = q > 1e-13
            regular[mask] = (form_factor[mask] - 1.0) / q[mask]
            regular[~mask] = slope0[i, j]
            correction[i, j] = float(np.dot(angular_weights, regular))
    return 0.5 * (correction + correction.T)


def angular_averaged_k_dependent_density_kernels(
    k_nm_inv: Array,
    weights_nm2: Array,
    params: EIParams,
    profiles: KaneMomentumLayerProfiles,
    *,
    frozen_k_nm_inv: float | None = None,
    nphi_regular: int = 240,
    q_samples: int = 1025,
    diagonal_order: int = 256,
    regular_cell_order: int = 96,
) -> KDependentDensityKernels:
    """Build a typed Hermitian endpoint-density approximation.

    The microscopic projected pair vertex uses transition densities between k
    and k'. This deliberately narrower scalar closure instead uses normalized
    E1/H1 pair-projector endpoint densities. The e-h channel is Hermitianized
    before angular integration, and the single Note-2 same-species kernel is
    the balanced average ``(K_ee+K_hh)/2``. This is a diagnostic closure, not an
    attribution to the Du authors.
    """

    params.validate()
    profiles.validate()
    if params.screening_model != "none" or params.q_tf_nm_inv != 0.0:
        raise ValueError("k-dependent density kernels require unscreened Coulomb")
    if nphi_regular < 32 or q_samples < 257:
        raise ValueError("insufficient k-dependent form-factor quadrature")
    k = np.asarray(k_nm_inv, dtype=np.float64)
    weights = np.asarray(weights_nm2, dtype=np.float64)
    if k.ndim != 1 or weights.shape != k.shape or np.any(np.diff(k) <= 0.0):
        raise ValueError("invalid solver radial grid")
    if k.min() < profiles.k_nm_inv[0] or k.max() > profiles.k_nm_inv[-1]:
        raise ValueError("solver k grid extends outside momentum-profile source")

    source_k = np.asarray(profiles.k_nm_inv, dtype=np.float64)
    if frozen_k_nm_inv is None:
        electron = np.asarray(profiles.electron_density_nm_inv, dtype=np.float64)
        hole = np.asarray(profiles.hole_density_nm_inv, dtype=np.float64)
        profile_mode = "k_dependent_pair_projector_density"
    else:
        frozen = profiles.frozen_profile(float(frozen_k_nm_inv))
        electron = np.repeat(frozen.electron_density_nm_inv[None, :], source_k.size, axis=0)
        hole = np.repeat(frozen.hole_density_nm_inv[None, :], source_k.size, axis=0)
        profile_mode = "frozen_pair_projector_density"

    z_weights = _integration_weights(profiles.z_nm)
    probability_e = electron * z_weights[None, :]
    probability_h = hole * z_weights[None, :]
    distance = np.abs(profiles.z_nm[:, None] - profiles.z_nm[None, :])
    radial_edge_max = float(np.sqrt(4.0 * np.pi * np.sum(weights)))
    q_max = max(2.0 * radial_edge_max, 2.0 * float(source_k[-1]), 1e-6)
    q_table = np.linspace(0.0, q_max, int(q_samples), dtype=np.float64)
    ns = source_k.size
    F_ee = np.empty((ns, ns, q_table.size), dtype=np.float64)
    F_hh = np.empty_like(F_ee)
    F_eh_directed = np.empty_like(F_ee)
    for iq, q_value in enumerate(q_table):
        green = np.exp(-q_value * distance)
        F_ee[:, :, iq] = probability_e @ green @ probability_e.T
        F_hh[:, :, iq] = probability_h @ green @ probability_h.T
        F_eh_directed[:, :, iq] = probability_e @ green @ probability_h.T
    F_eh = 0.5 * (F_eh_directed + np.swapaxes(F_eh_directed, 0, 1))

    correction_ee_source = _angular_regular_correction_from_form_factor_table(
        source_k, q_table, F_ee, nphi_regular=nphi_regular
    )
    correction_hh_source = _angular_regular_correction_from_form_factor_table(
        source_k, q_table, F_hh, nphi_regular=nphi_regular
    )
    correction_eh_source = _angular_regular_correction_from_form_factor_table(
        source_k, q_table, F_eh, nphi_regular=nphi_regular
    )

    def interpolate_regular(source: Array) -> Array:
        spline = RectBivariateSpline(source_k, source_k, source, kx=3, ky=3, s=0.0)
        result = np.asarray(spline(k, k), dtype=np.float64)
        result = 0.5 * (result + result.T)
        area = np.concatenate(([0.0], np.cumsum(weights)))
        edges = np.sqrt(4.0 * np.pi * area)
        xcell, wcell = leggauss(int(regular_cell_order))
        for index, (k0, lo, hi, weight) in enumerate(zip(k, edges[:-1], edges[1:], weights)):
            kp = 0.5 * (hi - lo) * xcell + 0.5 * (lo + hi)
            wk = 0.5 * (hi - lo) * wcell
            values = spline.ev(np.full_like(kp, k0), kp)
            result[index, index] = float(
                np.dot(wk, kp * values / (2.0 * np.pi)) / weight
            )
        return 0.5 * (result + result.T)

    correction_ee = interpolate_regular(correction_ee_source)
    correction_hh = interpolate_regular(correction_hh_source)
    correction_eh = interpolate_regular(correction_eh_source)
    singular = _cell_integrated_intralayer_dimensionless(
        k, weights, diagonal_order=diagonal_order
    )
    prefactor = COULOMB_MEV_NM / float(params.eps_r)
    K_ee = prefactor * (singular + correction_ee)
    K_hh = prefactor * (singular + correction_hh)
    K_eh = prefactor * (singular + correction_eh)
    result = KDependentDensityKernels(
        K_eh,
        K_ee,
        K_hh,
        0.5 * (K_ee + K_hh),
        correction_eh_source,
        correction_ee_source,
        correction_hh_source,
        profile_mode,
        None if frozen_k_nm_inv is None else float(frozen_k_nm_inv),
        int(q_samples),
        int(nphi_regular),
    )
    result.validate()
    return result


def save_kernel_npz(
    path: Path,
    k: Array,
    kernel: Array,
    params: EIParams,
    *,
    kind: str,
    form_factor_model: str = "point_layers",
    form_factor_profile_source: str | None = None,
    singularity_policy: str = "q_floor",
    diagonal_order: int | None = None,
    regular_cell_order: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        k_nm_inv=np.asarray(k, dtype=np.float64),
        kernel_mev_nm2=np.asarray(kernel, dtype=np.float64),
        kind=np.array(kind),
        eps_r=np.array(params.eps_r),
        d_eh_nm=np.array(params.d_eh_nm),
        nphi=np.array(params.nphi),
        q_floor_nm_inv=np.array(q_floor_from_grid(np.asarray(k, dtype=np.float64), params.q_floor_nm_inv)),
        screening_model=np.array(params.screening_model),
        q_tf_nm_inv=np.array(params.q_tf_nm_inv),
        form_factor_model=np.array(form_factor_model),
        form_factor_profile_source=np.array(form_factor_profile_source or ""),
        singularity_policy=np.array(singularity_policy),
        diagonal_order=np.array(-1 if diagonal_order is None else int(diagonal_order)),
        regular_cell_order=np.array(
            -1 if regular_cell_order is None else int(regular_cell_order)
        ),
    )


def load_kernel_npz(path: Path) -> tuple[Array, Array]:
    data = np.load(path)
    return np.asarray(data["k_nm_inv"], dtype=np.float64), np.asarray(data["kernel_mev_nm2"], dtype=np.float64)
