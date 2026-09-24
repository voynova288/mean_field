"""Xue-like four-band excitonic Hartree--Fock pilot for monolayer TPT.

This adapter deliberately implements an *effective*, eigenvalue-only model.
It uses exact saved eigenvalues of archived-HR bands 59--62 in the qualified
Gamma patch, but unit band-form factors and energy-sector layer labels.  It is
therefore not a material-faithful projection of the archived 80-Wannier model.
Matching Wannier metadata, smooth projectors, projected layer operators, and
Coulomb form factors are required before that stronger claim is possible.

The interaction follows Xue and MacDonald, PRL 120, 186802 (2018), Eqs. (2),
(5), and (7), in physical units::

    V_intra(q) = 2*pi*C/(epsilon*q)
    V_inter(q) = V_intra(q) * exp(-q*d)
    D = P - P_ref
    Sigma_F,ab(k) = - integral V_ab(k-k') D_ba(k') d^2k'/(2*pi)^2

where ``C=e^2/(4*pi*epsilon_0)``.  The q=0 Hartree common mode is removed only
under explicit reference-relative neutrality.  The surviving capacitor term
is ``+/- 2*pi*C*d*n_exc/epsilon`` in the electron/valence proxy sectors.
The integrable q=0 Fock cell is averaged over the actual rectangular k cell;
no q-floor is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from mean_field.core.hf.density import ket_projector_to_stored_orientation
from mean_field.core.hf.engine import DensityUpdateResult, HartreeFockRun
from mean_field.core.hf.occupations import occupied_state_mask
from mean_field.core.hf.problem import HartreeFockKernel, HartreeFockProblem, run_hartree_fock_problem

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
SeedMode = Literal[
    "normal",
    "exciton_spin_conserving",
    "exciton_spin_conserving_orbital_opposite",
    "exciton_spin_flip",
    "random",
]

COULOMB_EV_ANGSTROM = 14.3996454784255
TPT_ARCHIVED_HR_SHA256 = "46aa589893d1d24670fce5872aeef6b0b18f75a488b6c0fcec4da9ccaaf5a54d"
TPT_SAVED_GRID_SHA256 = "cbbf989c6bfb63c6fcb65fe323107ebc7b310a4f98f2c23924eca33666c1b22b"
TPT_PROJECTOR_AUDIT_SHA256 = "94d3cd74d911ba80dcf7f570f445be792d020d76514e74d397837b709155eddb"
TPT_ACTIVE_BANDS_1BASED = (59, 60, 61, 62)
TPT_VALENCE_BANDS_1BASED = (59, 60)
TPT_CONDUCTION_BANDS_1BASED = (61, 62)
TPT_GAMMA_PATCH_LIMITS_ANGSTROM_INV = (0.15, 0.05)
TPT_MODEL_AUTHORITY = "xue_like_eigenvalue_only_effective_model"


class TPTClosedOccupationBoundaryError(ValueError):
    """Scientific rejection: strict global filling cuts a closed shell."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TPTGammaPatch:
    """Exact archived-HR eigenvalues on a uniform open Gamma patch."""

    k_cartesian_angstrom_inv: FloatArray
    weights_angstrom_minus2: FloatArray
    cell_widths_angstrom_inv: tuple[float, float]
    mesh_shape: tuple[int, int]
    active_energies_ev: FloatArray
    source_path: Path
    source_sha256: str
    source_hr_sha256: str | None = None
    projector_audit_sha256: str | None = None
    active_bands_1based: tuple[int, int, int, int] = TPT_ACTIVE_BANDS_1BASED
    quadrature_policy: str = "uniform_node_centered_open_cells"
    projector_authority: str = "unqualified"
    reciprocal_periods_angstrom_inv: tuple[float, float] | None = None

    @property
    def nk(self) -> int:
        return int(self.k_cartesian_angstrom_inv.shape[0])

    def validate(self) -> None:
        points = np.asarray(self.k_cartesian_angstrom_inv, dtype=np.float64)
        weights = np.asarray(self.weights_angstrom_minus2, dtype=np.float64)
        energies = np.asarray(self.active_energies_ev, dtype=np.float64)
        nx, ny = self.mesh_shape
        if points.shape != (nx * ny, 2):
            raise ValueError(f"k points shape {points.shape} does not match mesh {self.mesh_shape}")
        if weights.shape != (nx * ny,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("patch weights must be finite, positive, and have shape (nk,)")
        if energies.shape != (4, nx * ny) or not np.all(np.isfinite(energies)):
            raise ValueError("active energies must be finite and have shape (4,nk)")
        if not np.allclose(weights, weights[0], rtol=1.0e-13, atol=1.0e-16):
            raise ValueError("the effective-model pilot requires uniform node-centered weights")
        dx, dy = self.cell_widths_angstrom_inv
        if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
            raise ValueError("declared cell widths must be finite and positive")
        expected_weight = dx * dy / (2.0 * np.pi) ** 2
        if not np.isclose(weights[0], expected_weight, rtol=1.0e-12, atol=1.0e-16):
            raise ValueError("patch weight is inconsistent with the declared rectangular cell")
        grid = points.reshape(nx, ny, 2)
        if not np.allclose(grid[:, :, 0], grid[:, :1, 0], atol=2.0e-13):
            raise ValueError("k patch is not an x-major rectangular product")
        if not np.allclose(grid[:, :, 1], grid[:1, :, 1], atol=2.0e-13):
            raise ValueError("k patch is not an x-major rectangular product")
        x_axis = grid[:, 0, 0]
        y_axis = grid[0, :, 1]
        if np.any(np.diff(x_axis) <= 0.0) or np.any(np.diff(y_axis) <= 0.0):
            raise ValueError("k patch axes must be strictly increasing")
        if not np.allclose(np.diff(x_axis), dx, rtol=1.0e-11, atol=2.0e-13):
            raise ValueError("declared delta_kx does not match coordinate spacing")
        if not np.allclose(np.diff(y_axis), dy, rtol=1.0e-11, atol=2.0e-13):
            raise ValueError("declared delta_ky does not match coordinate spacing")
        if np.min(np.linalg.norm(points, axis=1)) > 1.0e-13:
            raise ValueError("k patch has no exact Gamma point")
        if self.reciprocal_periods_angstrom_inv is not None:
            periods = np.asarray(self.reciprocal_periods_angstrom_inv, dtype=np.float64)
            if periods.shape != (2,) or not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
                raise ValueError("reciprocal periods must be two finite positive values")
            if not np.allclose(
                periods,
                np.asarray([nx * dx, ny * dy]),
                rtol=1.0e-11,
                atol=2.0e-12,
            ):
                raise ValueError("reciprocal periods are inconsistent with the periodic mesh")


def load_tpt_gamma_patch(
    source_path: str | Path,
    *,
    kx_abs_max_angstrom_inv: float = TPT_GAMMA_PATCH_LIMITS_ANGSTROM_INV[0],
    ky_abs_max_angstrom_inv: float = TPT_GAMMA_PATCH_LIMITS_ANGSTROM_INV[1],
    projector_audit_path: str | Path | None = None,
) -> TPTGammaPatch:
    """Load the exact saved grid and extract the qualified 59--62 Gamma patch.

    The selected archived nodes are assigned equal node-centered rectangular
    cells.  Consequently the represented open integration rectangle extends
    by half a cell beyond the extreme saved nodes.  This is an explicit UV
    regulator, not a full-Brillouin-zone quadrature.
    """

    path = Path(source_path).expanduser().resolve()
    actual_hash = _sha256(path)
    if actual_hash.lower() != TPT_SAVED_GRID_SHA256:
        raise ValueError(
            f"saved-grid SHA-256 mismatch: expected {TPT_SAVED_GRID_SHA256}, got {actual_hash}"
        )
    audit_path = (
        Path(projector_audit_path).expanduser().resolve()
        if projector_audit_path is not None
        else path.parent.parent / "projector_window_audit_v1" / "REPORT.json"
    )
    audit_hash = _sha256(audit_path)
    if audit_hash.lower() != TPT_PROJECTOR_AUDIT_SHA256:
        raise ValueError(
            f"projector-audit SHA-256 mismatch: expected {TPT_PROJECTOR_AUDIT_SHA256}, got {audit_hash}"
        )
    audit = json.loads(audit_path.read_text())
    if str(audit.get("source_hr_sha256", "")).lower() != TPT_ARCHIVED_HR_SHA256:
        raise ValueError("projector audit is not bound to the canonical archived HR")
    qualified = audit.get("window_results", {}).get("bands_59_62", {})
    if qualified.get("band_indices_1_based") != [59, 62]:
        raise ValueError("projector audit does not qualify bands 59--62")
    if not np.isfinite(kx_abs_max_angstrom_inv) or float(kx_abs_max_angstrom_inv) <= 0.0:
        raise ValueError("kx_abs_max_angstrom_inv must be finite and positive")
    if not np.isfinite(ky_abs_max_angstrom_inv) or float(ky_abs_max_angstrom_inv) <= 0.0:
        raise ValueError("ky_abs_max_angstrom_inv must be finite and positive")

    with np.load(path) as source:
        required = {"grid_k_fractional", "grid_energies_ev", "lattice_angstrom"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"saved-grid archive is missing keys {sorted(missing)}")
        k_fractional = np.asarray(source["grid_k_fractional"], dtype=np.float64)
        energies = np.asarray(source["grid_energies_ev"], dtype=np.float64)
        lattice = np.asarray(source["lattice_angstrom"], dtype=np.float64)

    if k_fractional.ndim != 2 or k_fractional.shape[1] != 3:
        raise ValueError("grid_k_fractional must have shape (nk,3)")
    if energies.shape != (k_fractional.shape[0], 80):
        raise ValueError(f"expected exact 80-band energies, got {energies.shape}")
    if lattice.shape != (3, 3):
        raise ValueError("lattice_angstrom must have shape (3,3)")
    if k_fractional.shape[0] != 51 * 51:
        raise ValueError(f"expected the audited 51x51 grid, got {k_fractional.shape[0]} points")

    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    k_cartesian = k_fractional @ reciprocal
    if np.max(np.abs(k_cartesian[:, 2])) > 1.0e-12:
        raise ValueError("saved TPT grid is not two-dimensional")
    mask = (
        (np.abs(k_cartesian[:, 0]) <= float(kx_abs_max_angstrom_inv) + 1.0e-12)
        & (np.abs(k_cartesian[:, 1]) <= float(ky_abs_max_angstrom_inv) + 1.0e-12)
    )
    selected_points = k_cartesian[mask, :2]
    selected_energies = energies[mask, 58:62]
    order = np.lexsort((selected_points[:, 1], selected_points[:, 0]))
    selected_points = selected_points[order]
    selected_energies = selected_energies[order]

    x_values = np.unique(np.round(selected_points[:, 0], decimals=13))
    y_values = np.unique(np.round(selected_points[:, 1], decimals=13))
    if x_values.size < 3 or y_values.size < 3:
        raise ValueError("Gamma patch must contain at least three points per axis")
    if selected_points.shape[0] != x_values.size * y_values.size:
        raise ValueError("selected Gamma patch is not a complete rectangular product")
    dx_values = np.diff(x_values)
    dy_values = np.diff(y_values)
    dx = float(np.mean(dx_values))
    dy = float(np.mean(dy_values))
    if not np.allclose(dx_values, dx, rtol=1.0e-11, atol=1.0e-13):
        raise ValueError("selected kx nodes are not uniformly spaced")
    if not np.allclose(dy_values, dy, rtol=1.0e-11, atol=1.0e-13):
        raise ValueError("selected ky nodes are not uniformly spaced")
    weights = np.full(selected_points.shape[0], dx * dy / (2.0 * np.pi) ** 2)
    patch = TPTGammaPatch(
        k_cartesian_angstrom_inv=selected_points,
        weights_angstrom_minus2=weights,
        cell_widths_angstrom_inv=(dx, dy),
        mesh_shape=(int(x_values.size), int(y_values.size)),
        active_energies_ev=selected_energies.T.copy(),
        source_path=path,
        source_sha256=actual_hash,
        source_hr_sha256=TPT_ARCHIVED_HR_SHA256,
        projector_audit_sha256=audit_hash,
        projector_authority="raw_energy_indices_qualified_only_in_gamma_patch",
    )
    patch.validate()
    return patch


def load_tpt_full_bz_energy_rank_diagnostic(
    source_path: str | Path,
    *,
    projector_audit_path: str | Path | None = None,
) -> TPTGammaPatch:
    """Load raw energy ranks 59--62 on the periodic 51x51 full BZ.

    This loader exists only for the explicitly requested global energy-rank
    diagnostic. The rank-4 bundle failed the archived global projector audit;
    therefore this object must not be described as a smooth material projector.
    """

    path = Path(source_path).expanduser().resolve()
    actual_hash = _sha256(path)
    if actual_hash.lower() != TPT_SAVED_GRID_SHA256:
        raise ValueError(
            f"saved-grid SHA-256 mismatch: expected {TPT_SAVED_GRID_SHA256}, got {actual_hash}"
        )
    audit_path = (
        Path(projector_audit_path).expanduser().resolve()
        if projector_audit_path is not None
        else path.parent.parent / "projector_window_audit_v1" / "REPORT.json"
    )
    audit_hash = _sha256(audit_path)
    if audit_hash.lower() != TPT_PROJECTOR_AUDIT_SHA256:
        raise ValueError(
            f"projector-audit SHA-256 mismatch: expected {TPT_PROJECTOR_AUDIT_SHA256}, got {audit_hash}"
        )
    audit = json.loads(audit_path.read_text())
    rank4 = audit.get("window_results", {}).get("bands_59_62", {})
    global_min = rank4.get("neighbor_link_min_singular_full_bz", {}).get("min")
    if global_min is None or float(global_min) >= 1.0e-2:
        raise ValueError("expected the raw global 59--62 projector audit to be explicitly unqualified")

    with np.load(path) as source:
        k_fractional = np.asarray(source["grid_k_fractional"], dtype=np.float64)
        energies = np.asarray(source["grid_energies_ev"], dtype=np.float64)
        lattice = np.asarray(source["lattice_angstrom"], dtype=np.float64)
    if k_fractional.shape != (51 * 51, 3) or energies.shape != (51 * 51, 80):
        raise ValueError("full-BZ diagnostic requires the exact 51x51 archived grid and 80 bands")
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    if np.max(np.abs(reciprocal[:2, 2])) > 1.0e-13 or np.max(np.abs(reciprocal[2, :2])) > 1.0e-13:
        raise ValueError("periodic energy-rank diagnostic currently requires the orthorhombic TPT cell")
    k_cartesian = k_fractional @ reciprocal
    order = np.lexsort((k_cartesian[:, 1], k_cartesian[:, 0]))
    points = k_cartesian[order, :2]
    active = energies[order, 58:62].T.copy()
    x_values = np.unique(np.round(points[:, 0], decimals=13))
    y_values = np.unique(np.round(points[:, 1], decimals=13))
    if x_values.size != 51 or y_values.size != 51:
        raise ValueError("full-BZ diagnostic grid is not 51x51")
    dx = float(np.mean(np.diff(x_values)))
    dy = float(np.mean(np.diff(y_values)))
    periods = (
        float(np.linalg.norm(reciprocal[0, :2])),
        float(np.linalg.norm(reciprocal[1, :2])),
    )
    weights = np.full(points.shape[0], dx * dy / (2.0 * np.pi) ** 2)
    patch = TPTGammaPatch(
        k_cartesian_angstrom_inv=points,
        weights_angstrom_minus2=weights,
        cell_widths_angstrom_inv=(dx, dy),
        mesh_shape=(51, 51),
        active_energies_ev=active,
        source_path=path,
        source_sha256=actual_hash,
        source_hr_sha256=TPT_ARCHIVED_HR_SHA256,
        projector_audit_sha256=audit_hash,
        quadrature_policy="uniform_periodic_full_bz_raw_energy_ranks",
        projector_authority="user_requested_unsewn_global_energy_ranks_not_material_projector",
        reciprocal_periods_angstrom_inv=periods,
    )
    patch.validate()
    return patch


@dataclass(frozen=True)
class TPTXueLikeParameters:
    dielectric_constant: float
    electron_hole_separation_angstrom: float
    model_authority: str = TPT_MODEL_AUTHORITY
    explicit_spin_policy: str = "duplicate_provisionally_scalar_no_soc_parent"
    layer_proxy_policy: str = "bands_59_60_valence_proxy__bands_61_62_electron_proxy"
    form_factor_policy: str = "unit_band_form_factors"

    def validate(self) -> None:
        epsilon = float(self.dielectric_constant)
        distance = float(self.electron_hole_separation_angstrom)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("dielectric_constant must be finite and positive")
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("electron_hole_separation_angstrom must be finite and nonnegative")
        if self.model_authority != TPT_MODEL_AUTHORITY:
            raise ValueError("unsupported TPT model authority")


@dataclass(frozen=True)
class TPTXueCoulombKernel:
    intra_ev_angstrom2: FloatArray
    inter_ev_angstrom2: FloatArray
    self_cell_intra_ev_angstrom2: float
    self_cell_inter_ev_angstrom2: float

    def validate(self, nk: int) -> None:
        intra = np.asarray(self.intra_ev_angstrom2, dtype=np.float64)
        inter = np.asarray(self.inter_ev_angstrom2, dtype=np.float64)
        if intra.shape != (int(nk), int(nk)) or inter.shape != intra.shape:
            raise ValueError("Coulomb kernels must both have shape (nk,nk)")
        if not np.all(np.isfinite(intra)) or not np.all(np.isfinite(inter)):
            raise ValueError("Coulomb kernels must be finite")
        if np.any(intra <= 0.0) or np.any(inter < 0.0):
            raise ValueError("Coulomb kernels must be nonnegative and intralayer positive")
        if np.max(np.abs(intra - intra.T)) > 1.0e-12:
            raise ValueError("intralayer kernel must be symmetric")
        if np.max(np.abs(inter - inter.T)) > 1.0e-12:
            raise ValueError("interlayer kernel must be symmetric")
        if np.any(inter > intra * (1.0 + 1.0e-13)):
            raise ValueError("interlayer kernel cannot exceed intralayer kernel")


def rectangular_coulomb_self_cell_average_ev_angstrom2(
    *,
    delta_kx_angstrom_inv: float,
    delta_ky_angstrom_inv: float,
    dielectric_constant: float,
    electron_hole_separation_angstrom: float,
    quadrature_order: int = 96,
) -> tuple[float, float]:
    """Average the physical Xue kernels over a centered rectangular cell."""

    dx = float(delta_kx_angstrom_inv)
    dy = float(delta_ky_angstrom_inv)
    epsilon = float(dielectric_constant)
    distance = float(electron_hole_separation_angstrom)
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("cell widths must be finite and positive")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("dielectric_constant must be finite and positive")
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("electron_hole_separation_angstrom must be finite and nonnegative")
    if int(quadrature_order) != quadrature_order or int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be an integer >= 8")

    a = 0.5 * dx
    b = 0.5 * dy
    area = dx * dy
    prefactor = 2.0 * np.pi * COULOMB_EV_ANGSTROM / epsilon
    integral_intra = 4.0 * (a * np.arcsinh(b / a) + b * np.arcsinh(a / b))
    intra_average = prefactor * integral_intra / area
    if distance == 0.0:
        return float(intra_average), float(intra_average)

    nodes, weights = leggauss(int(quadrature_order))
    theta_switch = float(np.arctan2(b, a))

    def integrate_segment(lo: float, hi: float, boundary: str) -> float:
        theta = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        radial_max = a / np.cos(theta) if boundary == "x" else b / np.sin(theta)
        radial_integral = -np.expm1(-distance * radial_max) / distance
        return float(0.5 * (hi - lo) * np.dot(weights, radial_integral))

    quadrant = integrate_segment(0.0, theta_switch, "x")
    quadrant += integrate_segment(theta_switch, 0.5 * np.pi, "y")
    inter_average = prefactor * (4.0 * quadrant) / area
    return float(intra_average), float(inter_average)


def precompute_tpt_xue_coulomb_kernel(
    patch: TPTGammaPatch,
    params: TPTXueLikeParameters,
    *,
    self_cell_quadrature_order: int = 96,
) -> TPTXueCoulombKernel:
    patch.validate()
    params.validate()
    points = np.asarray(patch.k_cartesian_angstrom_inv)
    delta = points[None, :, :] - points[:, None, :]
    if patch.reciprocal_periods_angstrom_inv is not None:
        periods = np.asarray(patch.reciprocal_periods_angstrom_inv)
        delta -= np.rint(delta / periods[None, None, :]) * periods[None, None, :]
    q = np.linalg.norm(delta, axis=2)
    diagonal = np.eye(patch.nk, dtype=bool)
    if np.any((q <= 1.0e-14) & ~diagonal):
        raise ValueError("TPT Gamma patch contains duplicate momentum points")
    prefactor = 2.0 * np.pi * COULOMB_EV_ANGSTROM / params.dielectric_constant
    intra = np.empty((patch.nk, patch.nk), dtype=np.float64)
    inter = np.empty_like(intra)
    off_diagonal = ~diagonal
    intra[off_diagonal] = prefactor / q[off_diagonal]
    inter[off_diagonal] = intra[off_diagonal] * np.exp(
        -q[off_diagonal] * params.electron_hole_separation_angstrom
    )
    self_intra, self_inter = rectangular_coulomb_self_cell_average_ev_angstrom2(
        delta_kx_angstrom_inv=patch.cell_widths_angstrom_inv[0],
        delta_ky_angstrom_inv=patch.cell_widths_angstrom_inv[1],
        dielectric_constant=params.dielectric_constant,
        electron_hole_separation_angstrom=params.electron_hole_separation_angstrom,
        quadrature_order=self_cell_quadrature_order,
    )
    intra[diagonal] = self_intra
    inter[diagonal] = self_inter
    kernel = TPTXueCoulombKernel(
        intra_ev_angstrom2=intra,
        inter_ev_angstrom2=inter,
        self_cell_intra_ev_angstrom2=self_intra,
        self_cell_inter_ev_angstrom2=self_inter,
    )
    kernel.validate(patch.nk)
    return kernel


_TPT_BASIS_LABELS = tuple(
    (spin, band)
    for spin in ("up", "down")
    for band in TPT_ACTIVE_BANDS_1BASED
)
_TPT_ELECTRON_PROXY_MASK = np.asarray(
    [band in TPT_CONDUCTION_BANDS_1BASED for _spin, band in _TPT_BASIS_LABELS],
    dtype=bool,
)
_TPT_SAME_SECTOR_MATRIX = (
    _TPT_ELECTRON_PROXY_MASK[:, None] == _TPT_ELECTRON_PROXY_MASK[None, :]
)


def tpt_basis_labels() -> tuple[tuple[str, int], ...]:
    """Return basis labels ``(spin, archived_band_1based)`` for the 8D pilot."""

    return _TPT_BASIS_LABELS


def _is_electron_proxy(index: int) -> bool:
    return bool(_TPT_ELECTRON_PROXY_MASK[int(index)])


def tpt_xue_reference_density(nk: int) -> ComplexArray:
    reference = np.zeros((8, 8, int(nk)), dtype=np.complex128)
    for index, (_spin, band) in enumerate(tpt_basis_labels()):
        if band in TPT_VALENCE_BANDS_1BASED:
            reference[index, index, :] = 1.0
    return reference


def build_tpt_xue_h0(patch: TPTGammaPatch) -> ComplexArray:
    patch.validate()
    h0 = np.zeros((8, 8, patch.nk), dtype=np.complex128)
    for spin_index in range(2):
        for active_index in range(4):
            index = 4 * spin_index + active_index
            h0[index, index, :] = patch.active_energies_ev[active_index]
    return h0


def tpt_xue_global_neutral_projector(
    hamiltonian: ComplexArray,
    *,
    reference_density: ComplexArray,
    boundary_degeneracy_tolerance_ev: float = 1.0e-12,
) -> DensityUpdateResult:
    """Globally fill exactly four states per k and return stored ``D=P-P_ref``."""

    h = np.asarray(hamiltonian, dtype=np.complex128)
    reference = np.asarray(reference_density, dtype=np.complex128)
    if h.shape != reference.shape or h.ndim != 3 or h.shape[:2] != (8, 8):
        raise ValueError("hamiltonian and reference_density must match shape (8,8,nk)")
    nk = h.shape[2]
    eigenvalues = np.empty((8, nk), dtype=np.float64)
    eigenvectors = np.empty_like(h)
    for ik in range(nk):
        values, vectors = np.linalg.eigh(h[:, :, ik])
        eigenvalues[:, ik] = values
        eigenvectors[:, :, ik] = vectors
    total_occupied = 4 * nk
    sorted_values = np.sort(eigenvalues.ravel(order="F"))
    highest_occupied = float(sorted_values[total_occupied - 1])
    lowest_empty = float(sorted_values[total_occupied])
    boundary_gap = lowest_empty - highest_occupied
    if boundary_gap <= float(boundary_degeneracy_tolerance_ev):
        raise TPTClosedOccupationBoundaryError(
            "strict zero-temperature global filling cuts a closed-gap/degenerate shell: "
            f"boundary_gap_ev={boundary_gap:.16e}"
        )
    occupation = occupied_state_mask(eigenvalues, total_occupied).astype(np.float64)
    projector_ket = np.einsum(
        "aik,ik,bik->abk", eigenvectors, occupation, eigenvectors.conj(), optimize=True
    )
    projector_stored = ket_projector_to_stored_orientation(projector_ket)
    density_delta = projector_stored - reference
    ranks = np.sum(occupation, axis=0).astype(np.int64)
    return DensityUpdateResult(
        density=density_delta,
        energies=eigenvalues,
        mu=0.5 * (highest_occupied + lowest_empty),
        observables={
            "raw_projector_stored": projector_stored,
            "occupied_rank_per_k": ranks,
            "global_gap_ev": float(boundary_gap),
        },
    )


def tpt_xue_hartree(
    density_delta_stored: ComplexArray,
    *,
    patch: TPTGammaPatch,
    params: TPTXueLikeParameters,
    neutrality_tolerance_angstrom_minus2: float = 1.0e-10,
) -> ComplexArray:
    """Return the neutral q=0 capacitor Hartree field in eV."""

    patch.validate()
    params.validate()
    density = np.asarray(density_delta_stored, dtype=np.complex128)
    if density.shape != (8, 8, patch.nk):
        raise ValueError("density_delta_stored must have shape (8,8,nk)")
    weights = np.asarray(patch.weights_angstrom_minus2)
    electron_indices = [index for index in range(8) if _is_electron_proxy(index)]
    valence_indices = [index for index in range(8) if not _is_electron_proxy(index)]
    n_electron = np.einsum(
        "ak,k->", density[electron_indices, electron_indices, :], weights, optimize=True
    )
    n_valence = np.einsum(
        "ak,k->", density[valence_indices, valence_indices, :], weights, optimize=True
    )
    if abs(n_electron.imag) > 1.0e-12 or abs(n_valence.imag) > 1.0e-12:
        raise ValueError("sector densities must be real")
    total = float(n_electron.real + n_valence.real)
    if abs(total) > float(neutrality_tolerance_angstrom_minus2):
        raise ValueError(
            "uniform Hartree q=0 requires reference-relative neutrality: "
            f"n_e+n_v={total:.16e} A^-2"
        )
    n_exciton = float(n_electron.real)
    potential = (
        2.0
        * np.pi
        * COULOMB_EV_ANGSTROM
        * params.electron_hole_separation_angstrom
        * n_exciton
        / params.dielectric_constant
    )
    result = np.zeros_like(density)
    for index in electron_indices:
        result[index, index, :] = potential
    for index in valence_indices:
        result[index, index, :] = -potential
    return result


def tpt_xue_fock(
    density_delta_stored: ComplexArray,
    *,
    patch: TPTGammaPatch,
    kernel: TPTXueCoulombKernel,
) -> ComplexArray:
    """Apply Xue exchange to core-stored ``D_ab=<c_a^dagger c_b>``.

    The operator coefficient ``Sigma_ab`` contracts ``D_ba``.  This transpose
    is essential for complex coherences and for consistency with the generic
    core's stored-projector energy convention.
    """

    patch.validate()
    density = np.asarray(density_delta_stored, dtype=np.complex128)
    if density.shape != (8, 8, patch.nk):
        raise ValueError("density_delta_stored must have shape (8,8,nk)")
    kernel.validate(patch.nk)
    weights = np.asarray(patch.weights_angstrom_minus2)
    weighted_transpose = np.swapaxes(density, 0, 1) * weights[None, None, :]
    flat = weighted_transpose.reshape(64, patch.nk)
    intra = (flat @ kernel.intra_ev_angstrom2.T).reshape(8, 8, patch.nk)
    inter = (flat @ kernel.inter_ev_angstrom2.T).reshape(8, 8, patch.nk)
    return -np.where(_TPT_SAME_SECTOR_MATRIX[:, :, None], intra, inter)


def tpt_xue_fock_at_momenta(
    density_delta_stored: ComplexArray,
    *,
    source_patch: TPTGammaPatch,
    params: TPTXueLikeParameters,
    source_kernel: TPTXueCoulombKernel,
    target_k_cartesian_angstrom_inv: FloatArray,
) -> ComplexArray:
    """Contract a converged source-grid density onto explicit target momenta."""

    source_patch.validate()
    params.validate()
    source_kernel.validate(source_patch.nk)
    density = np.asarray(density_delta_stored, dtype=np.complex128)
    targets = np.asarray(target_k_cartesian_angstrom_inv, dtype=np.float64)
    if density.shape != (8, 8, source_patch.nk):
        raise ValueError("density_delta_stored must have shape (8,8,nk_source)")
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.all(np.isfinite(targets)):
        raise ValueError("target momenta must be a finite array with shape (ntarget,2)")
    source = np.asarray(source_patch.k_cartesian_angstrom_inv)
    delta = source[None, :, :] - targets[:, None, :]
    if source_patch.reciprocal_periods_angstrom_inv is not None:
        periods = np.asarray(source_patch.reciprocal_periods_angstrom_inv)
        delta -= np.rint(delta / periods[None, None, :]) * periods[None, None, :]
    q = np.linalg.norm(delta, axis=2)
    zero = q <= 1.0e-14
    if np.any(np.sum(zero, axis=1) > 1):
        raise ValueError("a target momentum matches more than one source node")
    prefactor = 2.0 * np.pi * COULOMB_EV_ANGSTROM / params.dielectric_constant
    intra = np.empty_like(q)
    inter = np.empty_like(q)
    nonzero = ~zero
    intra[nonzero] = prefactor / q[nonzero]
    inter[nonzero] = intra[nonzero] * np.exp(
        -q[nonzero] * params.electron_hole_separation_angstrom
    )
    intra[zero] = source_kernel.self_cell_intra_ev_angstrom2
    inter[zero] = source_kernel.self_cell_inter_ev_angstrom2
    weighted_transpose = np.swapaxes(density, 0, 1) * np.asarray(
        source_patch.weights_angstrom_minus2
    )[None, None, :]
    flat = weighted_transpose.reshape(64, source_patch.nk)
    intra_action = (flat @ intra.T).reshape(8, 8, targets.shape[0])
    inter_action = (flat @ inter.T).reshape(8, 8, targets.shape[0])
    return -np.where(_TPT_SAME_SECTOR_MATRIX[:, :, None], intra_action, inter_action)


def tpt_xue_interaction(
    density_delta_stored: ComplexArray,
    *,
    patch: TPTGammaPatch,
    params: TPTXueLikeParameters,
    kernel: TPTXueCoulombKernel,
) -> ComplexArray:
    return tpt_xue_hartree(density_delta_stored, patch=patch, params=params) + tpt_xue_fock(
        density_delta_stored, patch=patch, kernel=kernel
    )


def tpt_xue_interaction_at_momenta(
    density_delta_stored: ComplexArray,
    *,
    source_patch: TPTGammaPatch,
    params: TPTXueLikeParameters,
    source_kernel: TPTXueCoulombKernel,
    target_k_cartesian_angstrom_inv: FloatArray,
) -> ComplexArray:
    """Return the frozen-density Hartree--Fock interaction on target momenta."""

    targets = np.asarray(target_k_cartesian_angstrom_inv, dtype=np.float64)
    fock = tpt_xue_fock_at_momenta(
        density_delta_stored,
        source_patch=source_patch,
        params=params,
        source_kernel=source_kernel,
        target_k_cartesian_angstrom_inv=targets,
    )
    source_hartree = tpt_xue_hartree(
        density_delta_stored, patch=source_patch, params=params
    )
    return fock + np.repeat(source_hartree[:, :, :1], targets.shape[0], axis=2)


def tpt_xue_absolute_density_residual(
    updated_density: ComplexArray,
    previous_density: ComplexArray,
) -> float:
    """Return the RMS-per-k Frobenius residual for reference-relative ``D``.

    A relative norm divided by ``||D_raw||`` is singular when a seeded branch
    cleanly collapses to the normal reference ``D=0``.  The density matrix is
    dimensionless, so this absolute per-k residual has a fixed physical scale.
    """

    updated = np.asarray(updated_density, dtype=np.complex128)
    previous = np.asarray(previous_density, dtype=np.complex128)
    if updated.shape != previous.shape or updated.ndim != 3:
        raise ValueError("density residual arrays must have matching shape (nt,nt,nk)")
    if not np.all(np.isfinite(updated)) or not np.all(np.isfinite(previous)):
        raise ValueError("density residual arrays must be finite")
    return float(np.linalg.norm(updated - previous) / np.sqrt(updated.shape[2]))


def tpt_xue_reference_relative_energy_density(
    interaction_h: ComplexArray,
    h0: ComplexArray,
    density_delta_stored: ComplexArray,
    *,
    patch: TPTGammaPatch,
) -> float:
    """Return ``E[D]/area`` in eV/Angstrom^2 relative to ``P_ref``."""

    patch.validate()
    sigma = np.asarray(interaction_h, dtype=np.complex128)
    bare = np.asarray(h0, dtype=np.complex128)
    density = np.asarray(density_delta_stored, dtype=np.complex128)
    if sigma.shape != bare.shape or sigma.shape != density.shape:
        raise ValueError("interaction_h, h0, and density must have matching shapes")
    weights = np.asarray(patch.weights_angstrom_minus2)
    value = np.einsum("abk,abk,k->", bare, density, weights, optimize=True)
    value += 0.5 * np.einsum("abk,abk,k->", sigma, density, weights, optimize=True)
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise ValueError(f"reference-relative energy density is not real: {value!r}")
    return float(value.real)


def tpt_xue_seed_hamiltonian(
    patch: TPTGammaPatch,
    *,
    mode: SeedMode,
    amplitude_ev: float,
    seed: int,
) -> ComplexArray:
    amplitude = float(amplitude_ev)
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude_ev must be finite and nonnegative")
    source = np.zeros((8, 8, patch.nk), dtype=np.complex128)
    if mode == "normal":
        return source
    gamma_width = 0.06
    envelope = np.exp(
        -np.sum(np.asarray(patch.k_cartesian_angstrom_inv) ** 2, axis=1) / (2.0 * gamma_width**2)
    )

    def pair(left: int, right: int, value: complex) -> None:
        source[left, right, :] += value * envelope
        source[right, left, :] += np.conjugate(value) * envelope

    if mode in {
        "exciton_spin_conserving",
        "exciton_spin_conserving_orbital_opposite",
    }:
        orbital_signs = (1.0, -1.0) if mode.endswith("orbital_opposite") else (1.0, 1.0)
        for spin_index in range(2):
            base = 4 * spin_index
            pair(base + 0, base + 2, amplitude * orbital_signs[0])
            pair(base + 1, base + 3, amplitude * orbital_signs[1])
    elif mode == "exciton_spin_flip":
        pair(0, 6, amplitude)
        pair(1, 7, amplitude)
        pair(4, 2, amplitude)
        pair(5, 3, amplitude)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        hermitian = raw + raw.conj().T
        norm = float(np.linalg.norm(hermitian))
        if norm > 0.0:
            hermitian *= amplitude / norm
        source[:, :, :] = hermitian[:, :, None] * envelope[None, None, :]
    else:
        raise ValueError(f"unsupported seed mode {mode!r}")
    return source


@dataclass
class TPTXueLikeHFState:
    patch: TPTGammaPatch
    params: TPTXueLikeParameters
    kernel: TPTXueCoulombKernel
    h0: ComplexArray
    reference_density: ComplexArray
    density: ComplexArray
    hamiltonian: ComplexArray
    energies: FloatArray
    seed_amplitude_ev: float = 0.02
    mu: float = 0.0
    precision: float = 1.0e-8
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return self.patch.nk


@dataclass(frozen=True)
class TPTXueLikeHFResult:
    run: HartreeFockRun
    density_delta_stored: ComplexArray
    raw_density_delta_stored: ComplexArray
    final_raw_residual: float
    interaction_h_ev: ComplexArray
    total_hamiltonian_ev: ComplexArray
    raw_projector_stored: ComplexArray
    energies_ev: FloatArray
    chemical_potential_ev: float
    global_gap_ev: float
    occupied_rank_per_k: NDArray[np.int64]
    exciton_density_angstrom_minus2: float
    excitonic_fock_gamma_frobenius_ev: float
    reference_relative_energy_ev_angstrom_minus2: float


def build_tpt_xue_like_hf_state(
    patch: TPTGammaPatch,
    *,
    dielectric_constant: float,
    electron_hole_separation_angstrom: float,
    accept_provisional_scalar_spin_duplication: bool,
    accept_energy_sector_layer_proxy: bool,
    precision: float = 1.0e-8,
    seed_amplitude_ev: float = 0.02,
) -> TPTXueLikeHFState:
    """Build the deliberately scoped effective-model state.

    The two explicit acknowledgement flags prevent this pilot from silently
    being promoted to a material-faithful projected-HR calculation.
    """

    if not accept_provisional_scalar_spin_duplication:
        raise ValueError("explicit spin duplication is provisional and must be acknowledged")
    if not accept_energy_sector_layer_proxy:
        raise ValueError("energy-sector layer labels are a model proxy and must be acknowledged")
    patch.validate()
    if not np.isfinite(precision) or float(precision) <= 0.0:
        raise ValueError("precision must be finite and positive")
    if not np.isfinite(seed_amplitude_ev) or float(seed_amplitude_ev) < 0.0:
        raise ValueError("seed_amplitude_ev must be finite and nonnegative")
    params = TPTXueLikeParameters(
        dielectric_constant=float(dielectric_constant),
        electron_hole_separation_angstrom=float(electron_hole_separation_angstrom),
    )
    params.validate()
    kernel = precompute_tpt_xue_coulomb_kernel(patch, params)
    h0 = build_tpt_xue_h0(patch)
    reference = tpt_xue_reference_density(patch.nk)
    return TPTXueLikeHFState(
        patch=patch,
        params=params,
        kernel=kernel,
        h0=h0,
        reference_density=reference,
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=np.zeros((8, patch.nk), dtype=np.float64),
        seed_amplitude_ev=float(seed_amplitude_ev),
        precision=float(precision),
    )


def _gamma_index(patch: TPTGammaPatch) -> int:
    norms = np.linalg.norm(np.asarray(patch.k_cartesian_angstrom_inv), axis=1)
    index = int(np.argmin(norms))
    if norms[index] > 1.0e-13:
        raise ValueError("patch has no exact Gamma point")
    return index


def run_tpt_xue_like_hf(
    state: TPTXueLikeHFState,
    *,
    init_mode: SeedMode,
    seed: int = 0,
    max_iter: int = 300,
    max_oda_lambda: float | None = None,
    oda_stall_threshold: float = 1.0e-12,
    initial_density_delta: ComplexArray | None = None,
    force_full_step: bool = False,
    allow_unconverged_diagnostic: bool = False,
) -> TPTXueLikeHFResult:
    """Run one candidate branch through the reusable generic SCF engine."""

    if force_full_step and max_oda_lambda is not None:
        raise ValueError("force_full_step cannot be combined with max_oda_lambda")
    if max_oda_lambda is not None and not np.isclose(float(max_oda_lambda), 1.0):
        raise ValueError(
            "clipping an unconstrained ODA minimizer is not a constrained line search; "
            "use max_oda_lambda=None (or 1.0)"
        )

    def interaction_builder(density: ComplexArray) -> ComplexArray:
        return tpt_xue_interaction(
            density, patch=state.patch, params=state.params, kernel=state.kernel
        )

    def density_builder(hamiltonian: ComplexArray) -> DensityUpdateResult:
        return tpt_xue_global_neutral_projector(
            hamiltonian, reference_density=state.reference_density
        )

    def initializer(target: TPTXueLikeHFState, *, init_mode: str, seed: int) -> None:
        if initial_density_delta is not None:
            supplied = np.asarray(initial_density_delta, dtype=np.complex128)
            if supplied.shape != target.density.shape:
                raise ValueError("initial_density_delta shape mismatch")
            if not np.all(np.isfinite(supplied)):
                raise ValueError("initial_density_delta must be finite")
            if np.max(np.abs(supplied - np.swapaxes(supplied.conj(), 0, 1))) > 1.0e-12:
                raise ValueError("initial_density_delta must be Hermitian in stored orientation")
            weights = np.asarray(target.patch.weights_angstrom_minus2)
            total = np.einsum("aak,k->", supplied, weights, optimize=True)
            if abs(total) > 1.0e-10:
                raise ValueError("initial_density_delta must be reference-relative neutral")
            projector = supplied + target.reference_density
            for ik in range(target.nk):
                occupations = np.linalg.eigvalsh(projector[:, :, ik])
                if occupations[0] < -1.0e-10 or occupations[-1] > 1.0 + 1.0e-10:
                    raise ValueError("initial density does not define a physical ensemble projector")
            target.density[:, :, :] = supplied
        else:
            source = tpt_xue_seed_hamiltonian(
                target.patch,
                mode=init_mode,  # type: ignore[arg-type]
                amplitude_ev=target.seed_amplitude_ev,
                seed=seed,
            )
            update = density_builder(target.h0 + source)
            target.density[:, :, :] = update.density
        target.hamiltonian[:, :, :] = target.h0

    def energy_functional(
        interaction_h: ComplexArray, h0: ComplexArray, density: ComplexArray
    ) -> float:
        return tpt_xue_reference_relative_energy_density(
            interaction_h, h0, density, patch=state.patch
        )

    problem = HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            oda_parameterizer=(lambda _state, _delta: 1.0) if force_full_step else None,
            oda_delta_interaction_builder=None if force_full_step else interaction_builder,
            convergence_metric=tpt_xue_absolute_density_residual,
            convergence_rule="raw",
        ),
    )
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=seed,
        max_iter=max_iter,
        oda_stall_threshold=oda_stall_threshold,
        max_oda_lambda=max_oda_lambda,
    )
    if not run.converged and not allow_unconverged_diagnostic:
        raise RuntimeError(
            f"TPT Xue-like HF did not converge: exit_reason={run.exit_reason}, "
            f"final_raw_residual={state.diagnostics.get('final_raw_norm', float('nan')):.6e}"
        )
    interaction_h = interaction_builder(state.density)
    total_hamiltonian = state.h0 + interaction_h
    final_update = density_builder(total_hamiltonian)
    raw_projector = np.asarray(final_update.observables["raw_projector_stored"])
    ranks = np.asarray(final_update.observables["occupied_rank_per_k"], dtype=np.int64)
    weights = np.asarray(state.patch.weights_angstrom_minus2)
    electron_indices = [index for index in range(8) if _is_electron_proxy(index)]
    n_exciton = float(
        np.einsum(
            "ak,k->",
            state.density[electron_indices, electron_indices, :].real,
            weights,
            optimize=True,
        )
    )
    gamma = _gamma_index(state.patch)
    valence_indices = [index for index in range(8) if not _is_electron_proxy(index)]
    excitonic_block = interaction_h[np.ix_(valence_indices, electron_indices, [gamma])][:, :, 0]
    energy_density = tpt_xue_reference_relative_energy_density(
        interaction_h, state.h0, state.density, patch=state.patch
    )
    return TPTXueLikeHFResult(
        run=run,
        density_delta_stored=state.density.copy(),
        raw_density_delta_stored=np.asarray(final_update.density).copy(),
        final_raw_residual=float(state.diagnostics["final_raw_norm"]),
        interaction_h_ev=interaction_h,
        total_hamiltonian_ev=total_hamiltonian,
        raw_projector_stored=raw_projector,
        energies_ev=final_update.energies,
        chemical_potential_ev=float(final_update.mu),
        global_gap_ev=float(final_update.observables["global_gap_ev"]),
        occupied_rank_per_k=ranks,
        exciton_density_angstrom_minus2=n_exciton,
        excitonic_fock_gamma_frobenius_ev=float(np.linalg.norm(excitonic_block)),
        reference_relative_energy_ev_angstrom_minus2=energy_density,
    )


__all__ = [
    "COULOMB_EV_ANGSTROM",
    "SeedMode",
    "TPTClosedOccupationBoundaryError",
    "TPTGammaPatch",
    "TPTXueCoulombKernel",
    "TPTXueLikeHFResult",
    "TPTXueLikeHFState",
    "TPTXueLikeParameters",
    "TPT_ACTIVE_BANDS_1BASED",
    "TPT_ARCHIVED_HR_SHA256",
    "TPT_GAMMA_PATCH_LIMITS_ANGSTROM_INV",
    "TPT_MODEL_AUTHORITY",
    "TPT_PROJECTOR_AUDIT_SHA256",
    "TPT_SAVED_GRID_SHA256",
    "build_tpt_xue_h0",
    "build_tpt_xue_like_hf_state",
    "load_tpt_full_bz_energy_rank_diagnostic",
    "load_tpt_gamma_patch",
    "precompute_tpt_xue_coulomb_kernel",
    "rectangular_coulomb_self_cell_average_ev_angstrom2",
    "run_tpt_xue_like_hf",
    "tpt_basis_labels",
    "tpt_xue_absolute_density_residual",
    "tpt_xue_fock",
    "tpt_xue_fock_at_momenta",
    "tpt_xue_global_neutral_projector",
    "tpt_xue_hartree",
    "tpt_xue_interaction",
    "tpt_xue_interaction_at_momenta",
    "tpt_xue_reference_density",
    "tpt_xue_reference_relative_energy_density",
    "tpt_xue_seed_hamiltonian",
]
