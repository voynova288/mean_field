r"""Phenomenological SOC-8 :math:`k\cdot p` Keldysh Hartree--Fock adapter.

This module connects the source-sealed TPT SOC-160 Gamma-frame eight-band model
to the generic reference-subtracted excitonic HF solver.  Its physical scope is
intentionally narrow:

* ``H8(k)`` is the exported zero-fit Lowdin quadratic model;
* the density vertex is the leading long-wavelength monopole ``Lambda=I`` in
  the fixed Gamma frame, not a reconstructed microscopic Wannier/PAW vertex;
* the interaction is the paper convention
  ``v(q)=2*pi*e^2/[kappa*q*(1+alpha_2D*q)]``;
* the macroscopic Hartree term is removed for a neutral, translation-invariant
  reference, and only reference-subtracted Fock exchange is retained;
* the open rectangular momentum patch is a continuum regulator, not a periodic
  Brillouin zone.

The paper's printed normalization area ``S`` is absent from ``v(q)`` because
``S^{-1} sum_p -> integral d^2p/(2*pi)^2``.  Mesh weights therefore have units
``Angstrom^-2`` and the generic energy functional returns ``eV/Angstrom^2``.
This adapter does not promote the finite-node energy validation of the parent
SOC-8 artifact to wavefunction, form-factor, interaction, or material authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.integrate import quad

from mean_field.core.hf.excitonic import (
    ElectronHoleSubspaces,
    FixedOccupation,
    LinearSelfEnergyFunctional,
    ReferenceSubtractedHFConfig,
    ReferenceSubtractedHFResult,
    make_fermi_density_builder,
    run_reference_subtracted_hf,
)

Array = np.ndarray
SeedChannel = Literal[
    "edge_scalar_real",
    "edge_scalar_imaginary",
    "edge_kramers_cross",
    "edge_random_complex",
]

KB_EV_PER_K = 8.617333262e-5
COULOMB_EV_ANGSTROM = 14.3996454784255
TPT_PUBLISHED_ALPHA_2D_ANGSTROM = 17.854
TPT_SOC8_MODEL_SHA256 = "eac6f1ab299dad52c80f32ebc1fd6cc968bf0ffacc0e2d963a71fae4645c8bd5"
TPT_SOC8_SCHEMA = "tpt-soc160-gamma8-fermi-pair-model-v1"
TPT_SOC8_PAIR_ORDER = np.asarray(
    [[117, 118], [123, 124], [119, 120], [121, 122]], dtype=np.int64
)
TPT_SOC8_ACTIVE_RANKS = np.arange(117, 125, dtype=np.int64)
TPT_SOC8_TARGET_RANKS = np.arange(119, 123, dtype=np.int64)


def _maximum_hermiticity_error(values: Array) -> float:
    matrices = np.asarray(values, dtype=np.complex128)
    return float(np.max(np.abs(matrices - np.swapaxes(matrices.conj(), -1, -2))))


def _maximum_time_reversal_error(values: Array, theta: Array, *, odd: bool) -> float:
    matrices = np.asarray(values, dtype=np.complex128)
    transformed = np.einsum(
        "ai,...ij,bj->...ab", theta, matrices.conj(), theta.conj(), optimize=True
    )
    target = -transformed if odd else transformed
    return float(np.max(np.abs(matrices - target)))


@dataclass(frozen=True)
class TPTSoc8Model:
    """Validated fixed-Gamma-frame SOC-8 tensor bundle."""

    source_path: Path
    source_sha256: str
    fermi_energy_eV: float
    lattice_angstrom: Array
    theta: Array
    h_gamma_eV: Array
    pi_eV_angstrom: Array
    lowdin_w_eV_angstrom2: Array

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = TPT_SOC8_MODEL_SHA256,
    ) -> "TPTSoc8Model":
        source = Path(path).resolve()
        digest = sha256(source.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"SOC-8 model hash mismatch: expected {expected_sha256}, got {digest}"
            )
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "schema",
                "active_ranks_1based",
                "target_ranks_1based",
                "tensor_pair_order_ranks_1based",
                "fermi_energy_eV",
                "lattice_angstrom",
                "canonical_theta",
                "H_Gamma_eV",
                "Pi_eV_angstrom",
                "Lowdin_W_eV_angstrom2",
            }
            if not required.issubset(archive.files):
                raise ValueError(f"SOC-8 archive is missing {sorted(required-set(archive.files))}")
            if str(archive["schema"].item()) != TPT_SOC8_SCHEMA:
                raise ValueError("unexpected SOC-8 schema")
            if not np.array_equal(archive["active_ranks_1based"], TPT_SOC8_ACTIVE_RANKS):
                raise ValueError("unexpected SOC-8 active ranks")
            if not np.array_equal(archive["target_ranks_1based"], TPT_SOC8_TARGET_RANKS):
                raise ValueError("unexpected SOC-8 target ranks")
            if not np.array_equal(
                archive["tensor_pair_order_ranks_1based"], TPT_SOC8_PAIR_ORDER
            ):
                raise ValueError("unexpected SOC-8 tensor pair order")
            fermi = float(archive["fermi_energy_eV"].item())
            lattice = np.asarray(archive["lattice_angstrom"], dtype=float).copy()
            theta = np.asarray(archive["canonical_theta"], dtype=np.complex128).copy()
            h_gamma = np.asarray(archive["H_Gamma_eV"], dtype=np.complex128).copy()
            pi = np.asarray(archive["Pi_eV_angstrom"], dtype=np.complex128).copy()
            lowdin_w = np.asarray(
                archive["Lowdin_W_eV_angstrom2"], dtype=np.complex128
            ).copy()

        expected_shapes = {
            "lattice": (3, 3),
            "theta": (8, 8),
            "h_gamma": (8, 8),
            "pi": (2, 8, 8),
            "lowdin_w": (2, 2, 8, 8),
        }
        arrays = {
            "lattice": lattice,
            "theta": theta,
            "h_gamma": h_gamma,
            "pi": pi,
            "lowdin_w": lowdin_w,
        }
        for name, values in arrays.items():
            if values.shape != expected_shapes[name] or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid {name} tensor")
        if not np.isfinite(fermi):
            raise ValueError("fermi energy must be finite")
        if _maximum_hermiticity_error(h_gamma) > 1.0e-10:
            raise ValueError("H_Gamma is not Hermitian")
        if _maximum_hermiticity_error(pi) > 1.0e-10:
            raise ValueError("Pi is not Hermitian")
        if _maximum_hermiticity_error(lowdin_w) > 1.0e-10:
            raise ValueError("Lowdin W is not Hermitian")
        if float(np.max(np.abs(lowdin_w - np.swapaxes(lowdin_w, 0, 1)))) > 1.0e-12:
            raise ValueError("Lowdin W is not symmetric in coordinate indices")
        identity = np.eye(8)
        if float(np.max(np.abs(theta.conj().T @ theta - identity))) > 1.0e-12:
            raise ValueError("canonical time reversal is not unitary")
        if float(np.max(np.abs(theta @ theta.conj() + identity))) > 1.0e-12:
            raise ValueError("canonical time reversal does not square to -1")
        if _maximum_time_reversal_error(h_gamma, theta, odd=False) > 1.0e-10:
            raise ValueError("H_Gamma violates the stored time-reversal frame")
        if _maximum_time_reversal_error(pi, theta, odd=True) > 2.0e-10:
            raise ValueError("Pi violates the stored time-reversal frame")
        if _maximum_time_reversal_error(lowdin_w, theta, odd=False) > 1.0e-9:
            raise ValueError("Lowdin W violates the stored time-reversal frame")
        return cls(
            source_path=source,
            source_sha256=digest,
            fermi_energy_eV=fermi,
            lattice_angstrom=lattice,
            theta=theta,
            h_gamma_eV=h_gamma,
            pi_eV_angstrom=pi,
            lowdin_w_eV_angstrom2=lowdin_w,
        )

    @property
    def in_plane_cell_area_angstrom2(self) -> float:
        return float(np.linalg.norm(np.cross(self.lattice_angstrom[0], self.lattice_angstrom[1])))

    def hamiltonian(self, k_cart_inv_angstrom: Array, *, relative_to_fermi: bool = True) -> Array:
        """Evaluate ``H8(k)`` and return shape ``(8, 8, nk)``."""

        k = np.asarray(k_cart_inv_angstrom, dtype=float)
        if k.ndim != 2 or k.shape[1] != 2 or not np.all(np.isfinite(k)):
            raise ValueError("k_cart_inv_angstrom must have finite shape (nk, 2)")
        values = np.repeat(self.h_gamma_eV[:, :, None], k.shape[0], axis=2)
        values += np.einsum("ki,iab->abk", k, self.pi_eV_angstrom, optimize=True)
        values += 0.5 * np.einsum(
            "ki,kj,ijab->abk", k, k, self.lowdin_w_eV_angstrom2, optimize=True
        )
        if relative_to_fermi:
            values -= self.fermi_energy_eV * np.eye(8)[:, :, None]
        values = 0.5 * (values + np.swapaxes(values.conj(), 0, 1))
        return values


@dataclass(frozen=True)
class RectangularMomentumMesh:
    """Uniform midpoint-cell open rectangular momentum mesh."""

    kx_inv_angstrom: Array
    ky_inv_angstrom: Array
    k_cart_inv_angstrom: Array
    weights_inv_angstrom2: Array
    nx: int
    ny: int
    dkx_inv_angstrom: float
    dky_inv_angstrom: float
    kx_limit_inv_angstrom: float
    ky_limit_inv_angstrom: float

    @classmethod
    def centered_midpoint(
        cls,
        *,
        kx_limit_inv_angstrom: float,
        ky_limit_inv_angstrom: float,
        nx: int,
        ny: int,
    ) -> "RectangularMomentumMesh":
        if int(nx) != nx or int(ny) != ny or nx < 3 or ny < 3:
            raise ValueError("nx and ny must be integers at least three")
        if nx % 2 != 1 or ny % 2 != 1:
            raise ValueError("nx and ny must be odd so the midpoint mesh contains Gamma")
        kx_limit = float(kx_limit_inv_angstrom)
        ky_limit = float(ky_limit_inv_angstrom)
        if not np.isfinite(kx_limit) or not np.isfinite(ky_limit) or min(kx_limit, ky_limit) <= 0.0:
            raise ValueError("momentum limits must be finite and positive")
        dkx = 2.0 * kx_limit / int(nx)
        dky = 2.0 * ky_limit / int(ny)
        kx = -kx_limit + (np.arange(int(nx)) + 0.5) * dkx
        ky = -ky_limit + (np.arange(int(ny)) + 0.5) * dky
        grid_x, grid_y = np.meshgrid(kx, ky, indexing="ij")
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        weight = dkx * dky / (2.0 * np.pi) ** 2
        weights = np.full(points.shape[0], weight, dtype=float)
        return cls(
            kx_inv_angstrom=kx,
            ky_inv_angstrom=ky,
            k_cart_inv_angstrom=points,
            weights_inv_angstrom2=weights,
            nx=int(nx),
            ny=int(ny),
            dkx_inv_angstrom=dkx,
            dky_inv_angstrom=dky,
            kx_limit_inv_angstrom=kx_limit,
            ky_limit_inv_angstrom=ky_limit,
        )



def keldysh_kernel_eV_angstrom2(
    q_inv_angstrom: Array | float,
    *,
    alpha_2d_angstrom: float = TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    kappa: float = 1.0,
    coulomb_eV_angstrom: float = COULOMB_EV_ANGSTROM,
) -> Array:
    """Return the nonzero-q continuum Keldysh kernel."""

    q = np.asarray(q_inv_angstrom, dtype=float)
    alpha = float(alpha_2d_angstrom)
    dielectric = float(kappa)
    coulomb = float(coulomb_eV_angstrom)
    if alpha <= 0.0 or dielectric <= 0.0 or coulomb <= 0.0:
        raise ValueError("alpha_2d, kappa, and Coulomb prefactor must be positive")
    if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("q must be finite and strictly positive")
    return 2.0 * np.pi * coulomb / (dielectric * q * (1.0 + alpha * q))



def rectangular_keldysh_self_cell_average_eV_angstrom2(
    dkx_inv_angstrom: float,
    dky_inv_angstrom: float,
    *,
    alpha_2d_angstrom: float = TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    kappa: float = 1.0,
    coulomb_eV_angstrom: float = COULOMB_EV_ANGSTROM,
    integration_tolerance: float = 1.0e-12,
) -> float:
    """Average the singular kernel over one centered rectangular mesh cell.

    The radial integral is analytic; only two smooth angular integrals remain.
    No q floor, clipping, or fitted replacement is used.
    """

    dx = float(dkx_inv_angstrom)
    dy = float(dky_inv_angstrom)
    alpha = float(alpha_2d_angstrom)
    dielectric = float(kappa)
    coulomb = float(coulomb_eV_angstrom)
    if min(dx, dy, alpha, dielectric, coulomb, integration_tolerance) <= 0.0:
        raise ValueError("cell widths and Keldysh parameters must be positive")
    a = 0.5 * dx
    b = 0.5 * dy
    split = float(np.arctan2(b, a))

    def x_limited(theta: float) -> float:
        return float(np.log1p(alpha * a / np.cos(theta)))

    def y_limited(theta: float) -> float:
        return float(np.log1p(alpha * b / np.sin(theta)))

    first = quad(x_limited, 0.0, split, epsabs=integration_tolerance, epsrel=integration_tolerance)[0]
    second = quad(
        y_limited,
        split,
        0.5 * np.pi,
        epsabs=integration_tolerance,
        epsrel=integration_tolerance,
    )[0]
    integral = 8.0 * np.pi * coulomb * (first + second) / (dielectric * alpha)
    return float(integral / (dx * dy))


@dataclass(frozen=True)
class OpenPatchKeldyshFock:
    """Open-boundary identity-vertex Keldysh Fock superoperator."""

    mesh: RectangularMomentumMesh
    alpha_2d_angstrom: float = TPT_PUBLISHED_ALPHA_2D_ANGSTROM
    kappa: float = 1.0
    coulomb_eV_angstrom: float = COULOMB_EV_ANGSTROM

    def __post_init__(self) -> None:
        if min(self.alpha_2d_angstrom, self.kappa, self.coulomb_eV_angstrom) <= 0.0:
            raise ValueError("Keldysh parameters must be positive")

    @cached_property
    def lag_kernel_eV_angstrom2(self) -> Array:
        mx = np.arange(-(self.mesh.nx - 1), self.mesh.nx, dtype=float)
        my = np.arange(-(self.mesh.ny - 1), self.mesh.ny, dtype=float)
        lag_x, lag_y = np.meshgrid(
            mx * self.mesh.dkx_inv_angstrom,
            my * self.mesh.dky_inv_angstrom,
            indexing="ij",
        )
        q = np.hypot(lag_x, lag_y)
        kernel = np.empty_like(q)
        nonzero = q > 0.0
        kernel[nonzero] = keldysh_kernel_eV_angstrom2(
            q[nonzero],
            alpha_2d_angstrom=self.alpha_2d_angstrom,
            kappa=self.kappa,
            coulomb_eV_angstrom=self.coulomb_eV_angstrom,
        )
        kernel[~nonzero] = rectangular_keldysh_self_cell_average_eV_angstrom2(
            self.mesh.dkx_inv_angstrom,
            self.mesh.dky_inv_angstrom,
            alpha_2d_angstrom=self.alpha_2d_angstrom,
            kappa=self.kappa,
            coulomb_eV_angstrom=self.coulomb_eV_angstrom,
        )
        return kernel

    @cached_property
    def fingerprint(self) -> str:
        metadata = np.asarray(
            [
                self.mesh.nx,
                self.mesh.ny,
                self.mesh.dkx_inv_angstrom,
                self.mesh.dky_inv_angstrom,
                self.alpha_2d_angstrom,
                self.kappa,
                self.coulomb_eV_angstrom,
            ],
            dtype=np.float64,
        )
        digest = sha256()
        digest.update(metadata.tobytes())
        digest.update(np.asarray(self.lag_kernel_eV_angstrom2, dtype=np.float64).tobytes())
        digest.update(b"open-patch-identity-vertex-reference-subtracted-fock-v1")
        return digest.hexdigest()

    @cached_property
    def _convolution_shape(self) -> tuple[int, int]:
        return (3 * self.mesh.nx - 2, 3 * self.mesh.ny - 2)

    @cached_property
    def _fft_shape(self) -> tuple[int, int]:
        return tuple(next_fast_len(size) for size in self._convolution_shape)

    @cached_property
    def _lag_kernel_fft(self) -> Array:
        return fftn(self.lag_kernel_eV_angstrom2, s=self._fft_shape)

    def self_energy(self, density_delta: Array) -> Array:
        density = np.asarray(density_delta, dtype=np.complex128)
        nk = self.mesh.nx * self.mesh.ny
        if density.ndim != 3 or density.shape[0] != density.shape[1] or density.shape[2] != nk:
            raise ValueError("density_delta must have shape (n, n, nx*ny)")
        if not np.all(np.isfinite(density)):
            raise ValueError("density_delta must be finite")
        weight = float(self.mesh.weights_inv_angstrom2[0])
        start_x = self.mesh.nx - 1
        start_y = self.mesh.ny - 1
        fields = density.reshape(
            density.shape[0], density.shape[1], self.mesh.nx, self.mesh.ny
        )
        # All matrix elements are independent convolutions with one scalar
        # kernel.  Batch them into one FFT plan rather than launching n^2
        # Python/SciPy FFT calls per SCF iteration.
        transformed = fftn(fields, s=self._fft_shape, axes=(-2, -1))
        full = ifftn(
            transformed * self._lag_kernel_fft[None, None, :, :],
            s=self._fft_shape,
            axes=(-2, -1),
        )
        full = full[
            :,
            :,
            : self._convolution_shape[0],
            : self._convolution_shape[1],
        ]
        result = (
            -weight
            * full[
                :,
                :,
                start_x : start_x + self.mesh.nx,
                start_y : start_y + self.mesh.ny,
            ].reshape(density.shape)
        )
        return 0.5 * (result + np.swapaxes(result.conj(), 0, 1))

    def direct_self_energy(self, density_delta: Array) -> Array:
        """Small-mesh O(Nk^2) oracle for tests and preflight only."""

        density = np.asarray(density_delta, dtype=np.complex128)
        nk = self.mesh.nx * self.mesh.ny
        if density.ndim != 3 or density.shape[0] != density.shape[1] or density.shape[2] != nk:
            raise ValueError("density_delta must have shape (n, n, nx*ny)")
        kernel = self.lag_kernel_eV_angstrom2
        weight = float(self.mesh.weights_inv_angstrom2[0])
        result = np.zeros_like(density)
        for ix in range(self.mesh.nx):
            for iy in range(self.mesh.ny):
                target = ix * self.mesh.ny + iy
                for px in range(self.mesh.nx):
                    for py in range(self.mesh.ny):
                        source = px * self.mesh.ny + py
                        result[:, :, target] -= (
                            weight
                            * kernel[ix - px + self.mesh.nx - 1, iy - py + self.mesh.ny - 1]
                            * density[:, :, source]
                        )
        return 0.5 * (result + np.swapaxes(result.conj(), 0, 1))

    def certified_functional(self, dimension: int = 8) -> LinearSelfEnergyFunctional:
        if int(dimension) != dimension or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        rng = np.random.default_rng(20260928)
        probes: list[Array] = []
        for _ in range(2):
            raw = rng.normal(size=(dimension, dimension, self.mesh.nx * self.mesh.ny))
            raw = raw + 1j * rng.normal(size=raw.shape)
            probes.append(0.5 * (raw + np.swapaxes(raw.conj(), 0, 1)))
        return LinearSelfEnergyFunctional.from_probes(
            self.self_energy,
            probes[0],
            probes[1],
            self.mesh.weights_inv_angstrom2,
            validation_label="open-patch identity-vertex Keldysh Fock",
            label="fock",
            absolute_tolerance=2.0e-12,
            relative_tolerance=2.0e-10,
            operator_fingerprint=self.fingerprint,
        )


@dataclass(frozen=True)
class TPTSoc8HFConfig:
    """Numerical and physical contract for one SOC-8 diagnostic branch."""

    temperature_K: float = 1.0
    occupation_per_k: float = 4.0
    mixing: float = 0.15
    precision: float = 1.0e-9
    max_iter: int = 1200
    seed_channel: SeedChannel | None = None
    seed_amplitude_eV: float = 0.001
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if abs(self.occupation_per_k - 4.0) > 1.0e-12:
            raise ValueError("the SOC-8 neutral adapter requires weighted mean occupation four")
        if not 1.0e-3 <= self.mixing <= 1.0:
            raise ValueError("mixing must lie in [1e-3, 1]")
        if self.precision <= 0.0 or self.max_iter < 1:
            raise ValueError("precision and max_iter must be positive")
        if self.seed_channel is not None and self.seed_amplitude_eV <= 0.0:
            raise ValueError("seed_amplitude_eV must be positive for a seeded branch")



def _gamma_frame_bridge(channel: SeedChannel, random_seed: int) -> Array:
    bridge = np.zeros((8, 8), dtype=np.complex128)
    if channel == "edge_scalar_real":
        bridge[4, 6] = bridge[5, 7] = 1.0
    elif channel == "edge_scalar_imaginary":
        bridge[4, 6] = bridge[5, 7] = -1.0j
    elif channel == "edge_kramers_cross":
        bridge[4, 7] = 1.0
        bridge[5, 6] = -1.0
    elif channel == "edge_random_complex":
        rng = np.random.default_rng(int(random_seed))
        bridge[4:6, 6:8] = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    else:
        raise ValueError(f"unsupported seed channel: {channel}")
    bridge += bridge.conj().T
    return bridge



def edge_projected_seed_hamiltonian(
    h0_eV: Array,
    *,
    channel: SeedChannel,
    amplitude_eV: float,
    random_seed: int = 0,
) -> tuple[Array, dict[str, float]]:
    """Construct a per-k normalized rank-2 valence/conduction bridge source."""

    h0 = np.asarray(h0_eV, dtype=np.complex128)
    if h0.shape[:2] != (8, 8) or h0.ndim != 3:
        raise ValueError("h0_eV must have shape (8, 8, nk)")
    operator = _gamma_frame_bridge(channel, random_seed)
    operator_norm = float(np.linalg.norm(operator))
    seed = np.empty_like(h0)
    raw_norms = np.empty(h0.shape[2], dtype=float)
    normalized_errors = np.empty(h0.shape[2], dtype=float)
    for ik in range(h0.shape[2]):
        _, vectors = np.linalg.eigh(h0[:, :, ik])
        valence = vectors[:, 2:4] @ vectors[:, 2:4].conj().T
        conduction = vectors[:, 4:6] @ vectors[:, 4:6].conj().T
        raw = conduction @ operator @ valence + valence @ operator @ conduction
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm <= 1.0e-12 * max(operator_norm, 1.0):
            raise ValueError("projected seed bridge is numerically forbidden")
        seed[:, :, ik] = float(amplitude_eV) * raw / raw_norm
        raw_norms[ik] = raw_norm
        normalized_errors[ik] = abs(float(np.linalg.norm(seed[:, :, ik])) - float(amplitude_eV))
    return seed, {
        "minimum_raw_bridge_norm": float(np.min(raw_norms)),
        "maximum_raw_bridge_norm": float(np.max(raw_norms)),
        "minimum_to_maximum_raw_bridge_ratio": float(np.min(raw_norms) / np.max(raw_norms)),
        "maximum_normalized_seed_norm_error_eV": float(np.max(normalized_errors)),
    }



def solve_tpt_soc8_keldysh_hf(
    model: TPTSoc8Model,
    mesh: RectangularMomentumMesh,
    *,
    interaction: OpenPatchKeldyshFock | None = None,
    config: TPTSoc8HFConfig | None = None,
) -> tuple[ReferenceSubtractedHFResult, dict[str, float | str]]:
    """Run one normal or source-removed seeded diagnostic branch."""

    cfg = TPTSoc8HFConfig() if config is None else config
    fock = OpenPatchKeldyshFock(mesh) if interaction is None else interaction
    if fock.mesh is not mesh:
        same_mesh = bool(
            fock.mesh.nx == mesh.nx
            and fock.mesh.ny == mesh.ny
            and fock.mesh.dkx_inv_angstrom == mesh.dkx_inv_angstrom
            and fock.mesh.dky_inv_angstrom == mesh.dky_inv_angstrom
            and fock.mesh.kx_limit_inv_angstrom == mesh.kx_limit_inv_angstrom
            and fock.mesh.ky_limit_inv_angstrom == mesh.ky_limit_inv_angstrom
            and np.array_equal(
                fock.mesh.k_cart_inv_angstrom, mesh.k_cart_inv_angstrom
            )
            and np.array_equal(
                fock.mesh.weights_inv_angstrom2, mesh.weights_inv_angstrom2
            )
        )
        if not same_mesh:
            raise ValueError("interaction mesh disagrees with Hamiltonian mesh")
    h0 = model.hamiltonian(mesh.k_cart_inv_angstrom, relative_to_fermi=True)
    thermal_energy = KB_EV_PER_K * cfg.temperature_K
    density_builder = make_fermi_density_builder(
        mesh.weights_inv_angstrom2,
        thermal_energy=thermal_energy,
        ensemble=FixedOccupation(cfg.occupation_per_k),
    )
    normal = density_builder(h0)
    reference = np.asarray(normal.density, dtype=np.complex128)
    seed = None
    seed_diagnostics: dict[str, float] = {}
    if cfg.seed_channel is not None:
        seed, seed_diagnostics = edge_projected_seed_hamiltonian(
            h0,
            channel=cfg.seed_channel,
            amplitude_eV=cfg.seed_amplitude_eV,
            random_seed=cfg.random_seed,
        )
    result = run_reference_subtracted_hf(
        h0,
        mesh.weights_inv_angstrom2,
        reference,
        absolute_density_builder=density_builder,
        interaction=fock.certified_functional(dimension=8),
        config=ReferenceSubtractedHFConfig(
            thermal_energy=thermal_energy,
            mixing=cfg.mixing,
            precision=cfg.precision,
            max_iter=cfg.max_iter,
            search_mode="normal_reference" if seed is None else "seeded_ei",
        ),
        normal_density_update=normal,
        electron_hole_subspaces=ElectronHoleSubspaces(
            electron_indices=(6, 7), hole_indices=(4, 5)
        ),
        seed_hamiltonian=seed,
    )
    occupied_max = float(np.max(result.energies[3]))
    empty_min = float(np.min(result.energies[4]))
    diagnostics: dict[str, float | str] = {
        "model_sha256": model.source_sha256,
        "interaction_fingerprint": fock.fingerprint,
        "vertex_policy": "identity_monopole_in_fixed_Gamma_frame",
        "hartree_policy": "neutral_reference_macroscopic_q0_removed_no_local_field_hartree",
        "momentum_boundary": "open_rectangular_midpoint_cells",
        "indirect_gap_eV": empty_min - occupied_max,
        "fixed_gamma_frame_coherence_max": float(
            0.0
            if result.self_energy_coherence_singular_values is None
            else np.max(result.self_energy_coherence_singular_values)
        ),
        "energy_density_eV_per_angstrom2": result.energy.free_energy,
        "energy_per_primitive_cell_eV": (
            result.energy.free_energy * model.in_plane_cell_area_angstrom2
        ),
        **seed_diagnostics,
    }
    return result, diagnostics


__all__ = [
    "COULOMB_EV_ANGSTROM",
    "KB_EV_PER_K",
    "OpenPatchKeldyshFock",
    "RectangularMomentumMesh",
    "TPTSoc8HFConfig",
    "TPTSoc8Model",
    "TPT_PUBLISHED_ALPHA_2D_ANGSTROM",
    "TPT_SOC8_MODEL_SHA256",
    "edge_projected_seed_hamiltonian",
    "keldysh_kernel_eV_angstrom2",
    "rectangular_keldysh_self_cell_average_eV_angstrom2",
    "solve_tpt_soc8_keldysh_hf",
]
