"""Finite-grid co-rotating angular-harmonic Kane4 Fock actions.

This module implements only the uniform-dielectric (G0) linear Fock block for
one co-rotating Fourier mode.  It deliberately contains no SCF, Jacobian, or
time-reversal projection.  With

    D_tilde_j(theta_l) = sum_m exp(+1j*m*theta_l) D_{j,m},

the finite-angle-grid action is

    K_m[i,j,a,d,b,c] = (1/nphi) sum_l exp(+1j*m*theta_l) C_l[i,j,a,d,b,c],
    Sigma_{i,m} = -sum_{j,b,c} w_j K_m[i,j,a,d,b,c] D_{j,m;b,c}.

A single nonzero ``D_m`` is generally not Hermitian.  Physical Hermiticity is
a relation between the paired modes, ``D_-m = D_m^dagger``; consequently this
API accepts arbitrary finite complex mode matrices.

For non-axial target frames, modes need not decouple. The reduced diagnostic
functions ``direct_full_2d_co_rotating_fock_modes`` and
``direct_full_2d_co_rotating_fock_mode_contributions`` retain the exact target
vertices and expose the coupled action ``Sigma_m=sum_q K_mq[D_q]``. They are
oracles for validating the axial diagonal approximation, not an SCF backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .axial import AxialRotationSpec
from .axial_fock import (
    ANGLE_ORIGIN_RAD,
    SELF_CELL_POLICY,
    AxialAveragedProjectedFockOperator,
    _rotation_fingerprint,
    _validated_harmonic_mode,
    _validated_nphi,
    co_rotating_mode_alias,
    precompute_axial_harmonic_exchange_tensor_mev_nm2,
)
from .projected_model import (
    Kane4Bundle,
    ProjectedFockOperator,
    fock_energy_density_mev_nm2,
)

ComplexArray = np.ndarray
FloatArray = np.ndarray
FOURIER_CONVENTION = (
    "D_tilde(theta_l)=sum_m exp(+1j*m*theta_l) D_m; "
    "K_m=(1/nphi) sum_l exp(+1j*m*theta_l) C_l"
)
HARMONIC_SELF_CELL_DESCRIPTION = (
    "co-rotating finite-angle harmonic with uniform dielectric and circular "
    "2D self cell built from radial_weight/nphi"
)


def _validate_polar_bundle_layout(
    bundle: Kane4Bundle,
    *,
    nr: int,
    nphi: int,
    atol: float = 2e-10,
) -> None:
    bundle.validate()
    if int(nr) < 1 or int(nphi) < 2 or bundle.nk != int(nr) * int(nphi):
        raise ValueError("polar bundle dimensions are incompatible with nr*nphi")
    k = np.asarray(bundle.k_cart_nm_inv, dtype=float).reshape(int(nr), int(nphi), 2)
    radii = np.linalg.norm(k[:, 0, :], axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("polar bundle radii must be strictly positive")
    angles = 2.0 * np.pi * np.arange(int(nphi), dtype=float) / float(nphi)
    expected = radii[:, None, None] * np.stack(
        (np.cos(angles), np.sin(angles)), axis=1
    )[None, :, :]
    if not np.allclose(k, expected, rtol=0.0, atol=atol):
        raise ValueError("polar bundle k ordering is not radius-major uniform-angle")
    weights = np.asarray(bundle.weights_nm2, dtype=float).reshape(int(nr), int(nphi))
    if not np.allclose(weights, weights[:, :1], rtol=0.0, atol=atol):
        raise ValueError("polar bundle weights are not angle-uniform at fixed radius")


def _active_rotation(spec: AxialRotationSpec, angle: float) -> np.ndarray:
    return np.diag(
        np.exp(
            1j
            * int(spec.exponent_sign)
            * np.asarray(spec.active_jz, dtype=float)
            * float(angle)
        )
    )


def _co_rotating_modes_on_polar_grid(
    matrices: ComplexArray,
    *,
    nr: int,
    nphi: int,
    spec: AxialRotationSpec,
) -> ComplexArray:
    values = np.asarray(matrices, dtype=np.complex128)
    nactive = len(spec.active_jz)
    if values.shape != (nactive, nactive, int(nr) * int(nphi)):
        raise ValueError("polar matrices have an incompatible shape")
    polar = values.reshape(nactive, nactive, int(nr), int(nphi))
    co_rotating = np.empty_like(polar)
    for iphi in range(int(nphi)):
        angle = 2.0 * np.pi * iphi / float(nphi)
        unitary = _active_rotation(spec, angle)
        co_rotating[:, :, :, iphi] = np.einsum(
            "ab,bcj,cd->adj",
            unitary.conj().T,
            polar[:, :, :, iphi],
            unitary,
            optimize=True,
        )
    return np.fft.fft(co_rotating, axis=3) / float(nphi)


def _reconstruct_co_rotating_modes_on_polar_grid(
    modes: ComplexArray,
    *,
    nr: int,
    nphi: int,
    spec: AxialRotationSpec,
) -> ComplexArray:
    values = np.asarray(modes, dtype=np.complex128)
    nactive = len(spec.active_jz)
    expected = (nactive, nactive, int(nr), int(nphi))
    if values.shape != expected:
        raise ValueError("co-rotating mode array has an incompatible shape")
    co_rotating = np.fft.ifft(values, axis=3) * float(nphi)
    polar = np.empty_like(co_rotating)
    for iphi in range(int(nphi)):
        angle = 2.0 * np.pi * iphi / float(nphi)
        unitary = _active_rotation(spec, angle)
        polar[:, :, :, iphi] = np.einsum(
            "ab,bcj,cd->adj",
            unitary,
            co_rotating[:, :, :, iphi],
            unitary.conj().T,
            optimize=True,
        )
    return polar.reshape(nactive, nactive, int(nr) * int(nphi))


def _raw_projected_fock_action(
    operator: ProjectedFockOperator,
    density: ComplexArray,
) -> ComplexArray:
    """Apply a certified full-2D Fock action without modewise Hermiticity checks."""

    values = np.asarray(density, dtype=np.complex128)
    nk = operator.local_vertices.shape[0]
    nactive = operator.local_vertices.shape[3]
    if values.shape != (nactive, nactive, nk) or not np.all(np.isfinite(values)):
        raise ValueError("raw full-2D Fock density has an incompatible shape or values")
    if operator.exchange_tensor_mev_nm2 is not None:
        return -np.einsum(
            "kpadbc,bcp,p->adk",
            operator.exchange_tensor_mev_nm2,
            values,
            operator.k_weights_nm2,
            optimize=True,
        )
    sigma = np.zeros_like(values)
    for ik in range(nk):
        for ip in range(nk):
            term = np.einsum(
                "xab,bc,ycd,xy,x,y->ad",
                operator.local_vertices[ik, ip],
                values[:, :, ip],
                operator.local_vertices[ip, ik],
                operator.green_mev_nm2[ik, ip],
                operator.z_weights_nm,
                operator.z_weights_nm,
                optimize=True,
            )
            sigma[:, :, ik] -= operator.k_weights_nm2[ip] * term
    return sigma


def direct_full_2d_co_rotating_fock_modes(
    operator: ProjectedFockOperator,
    target_bundle: Kane4Bundle,
    density_modes: ComplexArray,
    *,
    nr: int,
    nphi: int,
    spec: AxialRotationSpec | None = None,
) -> ComplexArray:
    """Apply the exact target-frame full-2D Fock action and return all modes.

    This is a reduced-oracle path. Unlike the axial harmonic approximation it
    retains the target bundle's actual off-axis microscopic vertices, so an
    output mode can receive contributions from every input mode.
    """

    rotation = AxialRotationSpec() if spec is None else spec
    _validate_polar_bundle_layout(target_bundle, nr=nr, nphi=nphi)
    operator.validate_against_bundle(target_bundle)
    density = _reconstruct_co_rotating_modes_on_polar_grid(
        density_modes, nr=nr, nphi=nphi, spec=rotation
    )
    density_error = float(
        np.max(np.abs(density - np.swapaxes(density.conj(), 0, 1)))
    )
    if density_error > 1e-9:
        raise ValueError(
            "the complete co-rotating mode family does not reconstruct a "
            f"Hermitian density: error={density_error:.3e}"
        )
    sigma = _raw_projected_fock_action(operator, density)
    sigma_error = float(np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))))
    if not np.all(np.isfinite(sigma)) or sigma_error > 1e-8:
        raise ValueError(
            "direct full-2D Fock oracle produced an invalid physical action: "
            f"Hermiticity error={sigma_error:.3e}"
        )
    return _co_rotating_modes_on_polar_grid(
        sigma, nr=nr, nphi=nphi, spec=rotation
    )


def direct_full_2d_co_rotating_fock_mode_contributions(
    operator: ProjectedFockOperator,
    target_bundle: Kane4Bundle,
    density_modes: ComplexArray,
    *,
    nr: int,
    nphi: int,
    spec: AxialRotationSpec | None = None,
) -> ComplexArray:
    """Return action-level ``K[m,q][D_q]`` contributions for the exact frame.

    The output has shape ``(n,n,nr,nphi_out,nphi_in)`` and obeys
    ``Sigma_m = sum_q output[..., m, q]``. Individual ``q`` contributions need
    not be Hermitian; Hermiticity applies only after the physical mode family
    is summed and reconstructed.
    """

    rotation = AxialRotationSpec() if spec is None else spec
    _validate_polar_bundle_layout(target_bundle, nr=nr, nphi=nphi)
    operator.validate_against_bundle(target_bundle)
    values = np.asarray(density_modes, dtype=np.complex128)
    nactive = len(rotation.active_jz)
    expected = (nactive, nactive, int(nr), int(nphi))
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError("co-rotating density modes have an incompatible shape or values")
    physical_density = _reconstruct_co_rotating_modes_on_polar_grid(
        values, nr=nr, nphi=nphi, spec=rotation
    )
    physical_error = float(
        np.max(
            np.abs(
                physical_density
                - np.swapaxes(physical_density.conj(), 0, 1)
            )
        )
    )
    if physical_error > 1e-9:
        raise ValueError(
            "the complete mode family is not physical: "
            f"Hermiticity error={physical_error:.3e}"
        )
    contributions = np.empty(
        (nactive, nactive, int(nr), int(nphi), int(nphi)),
        dtype=np.complex128,
    )
    isolated = np.zeros_like(values)
    for input_mode in range(int(nphi)):
        isolated.fill(0.0)
        isolated[:, :, :, input_mode] = values[:, :, :, input_mode]
        density = _reconstruct_co_rotating_modes_on_polar_grid(
            isolated, nr=nr, nphi=nphi, spec=rotation
        )
        sigma = _raw_projected_fock_action(operator, density)
        contributions[:, :, :, :, input_mode] = (
            _co_rotating_modes_on_polar_grid(
                sigma, nr=nr, nphi=nphi, spec=rotation
            )
        )
    if not np.all(np.isfinite(contributions)):
        raise ValueError("exact-frame mode-coupling contributions are non-finite")
    direct = direct_full_2d_co_rotating_fock_modes(
        operator,
        target_bundle,
        values,
        nr=nr,
        nphi=nphi,
        spec=rotation,
    )
    summed = np.sum(contributions, axis=4)
    linearity_error = float(np.max(np.abs(summed - direct)))
    if linearity_error > 2e-10:
        raise ValueError(
            "exact-frame mode decomposition fails linearity: "
            f"error={linearity_error:.3e}"
        )
    return contributions


@dataclass(frozen=True)
class _CoRotatingHarmonicFockAttestation:
    bundle_fingerprint: str
    action_sha256: str
    requested_mode: int
    mode_mod_nphi: int
    nphi: int
    rotation_fingerprint: str
    angle_origin_rad: float
    self_cell_policy: str
    fourier_convention: str


def _harmonic_action_sha256(
    tensor: np.ndarray,
    weights: np.ndarray,
    *,
    bundle_fingerprint: str,
    mode_mod_nphi: int,
    nphi: int,
    epsilon_r: float,
    coulomb_mev_nm: float,
    rotation_fingerprint: str,
    angle_origin_rad: float,
    self_cell_policy: str,
    self_cell_description: str,
    fourier_convention: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"inas-gasb-co-rotating-g0-fock-v1")
    digest.update(bundle_fingerprint.encode())
    digest.update(str(int(mode_mod_nphi)).encode())
    digest.update(str(int(nphi)).encode())
    digest.update(np.float64(epsilon_r).tobytes())
    digest.update(np.float64(coulomb_mev_nm).tobytes())
    digest.update(rotation_fingerprint.encode())
    digest.update(np.float64(angle_origin_rad).tobytes())
    digest.update(self_cell_policy.encode())
    digest.update(self_cell_description.encode())
    digest.update(fourier_convention.encode())
    for array in (tensor, weights):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class CoRotatingHarmonicFockOperator:
    """Immutable, builder-attested G0 action for one finite-grid mode.

    ``mode`` records the requested integer representative and
    ``mode_mod_nphi`` records the actual finite-grid channel.  Thus, for
    example, ``mode=-1`` and ``mode=nphi-1`` are explicit aliases and have the
    same action fingerprint.  No Hermiticity condition is imposed on the
    input to :meth:`__call__`.
    """

    exchange_tensor_mev_nm2: ComplexArray
    k_weights_nm2: FloatArray
    bundle_fingerprint: str
    mode: int
    mode_mod_nphi: int
    nphi: int
    epsilon_r: float
    coulomb_mev_nm: float
    self_cell_description: str
    micro_jz: tuple[float, ...]
    active_jz: tuple[float, ...]
    exponent_sign: int
    angle_origin_rad: float
    self_cell_policy: str
    fourier_convention: str
    _builder_attestation: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bundle_fingerprint:
            raise ValueError("harmonic Fock bundle fingerprint is required")
        requested_mode = _validated_harmonic_mode(self.mode)
        nphi_value = _validated_nphi(self.nphi)
        mode_mod_nphi = co_rotating_mode_alias(requested_mode, nphi_value)
        supplied_alias = _validated_harmonic_mode(self.mode_mod_nphi)
        if supplied_alias != mode_mod_nphi:
            raise ValueError("harmonic Fock mode alias does not equal mode modulo nphi")
        if not np.isfinite(self.epsilon_r) or not np.isfinite(self.coulomb_mev_nm):
            raise ValueError("harmonic Fock scalar parameters must be finite")
        if self.epsilon_r <= 0.0 or self.coulomb_mev_nm <= 0.0:
            raise ValueError("harmonic Fock scalar parameters must be positive")
        if self.self_cell_description != HARMONIC_SELF_CELL_DESCRIPTION:
            raise ValueError("harmonic Fock self-cell description is not builder-issued")
        if float(self.angle_origin_rad) != ANGLE_ORIGIN_RAD:
            raise ValueError("harmonic Fock angle origin does not match the builder policy")
        if self.self_cell_policy != SELF_CELL_POLICY:
            raise ValueError("harmonic Fock self-cell policy does not match the builder policy")
        if self.fourier_convention != FOURIER_CONVENTION:
            raise ValueError("harmonic Fock Fourier convention does not match the builder policy")

        rotation = AxialRotationSpec(
            micro_jz=np.asarray(self.micro_jz, dtype=float),
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        tensor = np.array(
            self.exchange_tensor_mev_nm2, dtype=np.complex128, copy=True
        )
        weights = np.array(self.k_weights_nm2, dtype=float, copy=True)
        if tensor.ndim != 6 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(
                "harmonic exchange tensor must have shape (nr,nr,n,n,n,n)"
            )
        nr = tensor.shape[0]
        nactive = tensor.shape[2]
        if tensor.shape != (
            nr,
            nr,
            nactive,
            nactive,
            nactive,
            nactive,
        ) or weights.shape != (nr,):
            raise ValueError("harmonic Fock tensor and radial weights are incompatible")
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(weights)):
            raise ValueError("harmonic Fock arrays must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("harmonic Fock radial weights must be positive")

        # K_m[i,j,a,d,b,c] = conj(K_m[j,i,b,c,a,d]) makes each mode
        # self-adjoint in the radial-weighted Frobenius inner product.
        self_adjoint_reverse = tensor.conj().transpose(1, 0, 4, 5, 2, 3)
        self_adjoint_error = float(
            np.max(np.abs(tensor - self_adjoint_reverse))
        )
        if self_adjoint_error > 1e-8:
            raise ValueError(
                "harmonic Fock tensor violates weighted self-adjointness: "
                f"{self_adjoint_error:.3e}"
            )

        expected_sha = _harmonic_action_sha256(
            tensor,
            weights,
            bundle_fingerprint=self.bundle_fingerprint,
            mode_mod_nphi=mode_mod_nphi,
            nphi=nphi_value,
            epsilon_r=float(self.epsilon_r),
            coulomb_mev_nm=float(self.coulomb_mev_nm),
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=float(self.angle_origin_rad),
            self_cell_policy=self.self_cell_policy,
            self_cell_description=self.self_cell_description,
            fourier_convention=self.fourier_convention,
        )
        attestation = self._builder_attestation
        if not isinstance(attestation, _CoRotatingHarmonicFockAttestation):
            raise ValueError(
                "harmonic Fock action was not issued by the certified builder"
            )
        if (
            attestation.bundle_fingerprint != self.bundle_fingerprint
            or attestation.action_sha256 != expected_sha
            or attestation.requested_mode != requested_mode
            or attestation.mode_mod_nphi != mode_mod_nphi
            or attestation.nphi != nphi_value
            or attestation.rotation_fingerprint != rotation_fingerprint
            or attestation.angle_origin_rad != self.angle_origin_rad
            or attestation.self_cell_policy != self.self_cell_policy
            or attestation.fourier_convention != self.fourier_convention
        ):
            raise ValueError("harmonic Fock action lacks a matching builder attestation")

        tensor.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "mode", requested_mode)
        object.__setattr__(self, "mode_mod_nphi", mode_mod_nphi)
        object.__setattr__(self, "nphi", nphi_value)
        object.__setattr__(self, "exchange_tensor_mev_nm2", tensor)
        object.__setattr__(self, "k_weights_nm2", weights)

    @classmethod
    def from_precomputed_uniform_dielectric(
        cls,
        radial_bundle: Kane4Bundle,
        exchange_tensor_mev_nm2: ComplexArray,
        *,
        mode: int,
        nphi: int,
        epsilon_r: float,
        coulomb_mev_nm: float = 1439.96448,
        spec: AxialRotationSpec | None = None,
    ) -> "CoRotatingHarmonicFockOperator":
        """Rehydrate one source-bound harmonic tensor with fresh attestation."""

        rotation = AxialRotationSpec() if spec is None else spec
        radial_bundle.validate()
        requested_mode = _validated_harmonic_mode(mode)
        nphi_value = _validated_nphi(nphi)
        mode_mod_nphi = co_rotating_mode_alias(requested_mode, nphi_value)
        tensor = np.asarray(exchange_tensor_mev_nm2, dtype=np.complex128)
        weights = np.asarray(radial_bundle.weights_nm2, dtype=float)
        bundle_fingerprint = radial_bundle.fingerprint()
        rotation_fingerprint = _rotation_fingerprint(rotation)
        action_sha = _harmonic_action_sha256(
            tensor,
            weights,
            bundle_fingerprint=bundle_fingerprint,
            mode_mod_nphi=mode_mod_nphi,
            nphi=nphi_value,
            epsilon_r=float(epsilon_r),
            coulomb_mev_nm=float(coulomb_mev_nm),
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
            self_cell_description=HARMONIC_SELF_CELL_DESCRIPTION,
            fourier_convention=FOURIER_CONVENTION,
        )
        attestation = _CoRotatingHarmonicFockAttestation(
            bundle_fingerprint=bundle_fingerprint,
            action_sha256=action_sha,
            requested_mode=requested_mode,
            mode_mod_nphi=mode_mod_nphi,
            nphi=nphi_value,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
            fourier_convention=FOURIER_CONVENTION,
        )
        return cls(
            exchange_tensor_mev_nm2=tensor,
            k_weights_nm2=weights,
            bundle_fingerprint=bundle_fingerprint,
            mode=requested_mode,
            mode_mod_nphi=mode_mod_nphi,
            nphi=nphi_value,
            epsilon_r=float(epsilon_r),
            coulomb_mev_nm=float(coulomb_mev_nm),
            self_cell_description=HARMONIC_SELF_CELL_DESCRIPTION,
            micro_jz=tuple(np.asarray(rotation.micro_jz, dtype=float)),
            active_jz=tuple(np.asarray(rotation.active_jz, dtype=float)),
            exponent_sign=int(rotation.exponent_sign),
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
            fourier_convention=FOURIER_CONVENTION,
            _builder_attestation=attestation,
        )

    @classmethod
    def from_bundle_uniform_dielectric(
        cls,
        radial_bundle: Kane4Bundle,
        *,
        mode: int,
        nphi: int,
        epsilon_r: float,
        coulomb_mev_nm: float = 1439.96448,
        spec: AxialRotationSpec | None = None,
    ) -> "CoRotatingHarmonicFockOperator":
        """Build one source-certified finite-grid G0 harmonic block."""

        rotation = AxialRotationSpec() if spec is None else spec
        requested_mode = _validated_harmonic_mode(mode)
        nphi_value = _validated_nphi(nphi)
        tensor = precompute_axial_harmonic_exchange_tensor_mev_nm2(
            radial_bundle,
            mode=requested_mode,
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            coulomb_mev_nm=coulomb_mev_nm,
            spec=rotation,
        )
        return cls.from_precomputed_uniform_dielectric(
            radial_bundle,
            tensor,
            mode=requested_mode,
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            coulomb_mev_nm=coulomb_mev_nm,
            spec=rotation,
        )

    @property
    def conjugate_mode_mod_nphi(self) -> int:
        """Return the finite-grid channel representing ``-mode``."""

        return (-self.mode_mod_nphi) % self.nphi

    @property
    def is_self_conjugate_mode(self) -> bool:
        """Whether this is the zero or even-grid Nyquist channel."""

        return self.mode_mod_nphi == self.conjugate_mode_mod_nphi

    def validate_against_bundle(
        self, radial_bundle: Kane4Bundle, *, atol: float = 1e-10
    ) -> None:
        """Reject a different radial source or a modified attested action."""

        radial_bundle.validate()
        if self.bundle_fingerprint != radial_bundle.fingerprint():
            raise ValueError(
                "harmonic Fock operator and radial bundle fingerprints differ"
            )
        if not bool(radial_bundle.provenance.get("radial_only", False)):
            raise ValueError("harmonic Fock operator requires a radial-only bundle")
        if not np.allclose(
            self.k_weights_nm2,
            radial_bundle.weights_nm2,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError("harmonic Fock weights do not match the radial bundle")

        rotation = AxialRotationSpec(
            micro_jz=np.asarray(self.micro_jz, dtype=float),
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        current_sha = _harmonic_action_sha256(
            self.exchange_tensor_mev_nm2,
            self.k_weights_nm2,
            bundle_fingerprint=self.bundle_fingerprint,
            mode_mod_nphi=self.mode_mod_nphi,
            nphi=self.nphi,
            epsilon_r=self.epsilon_r,
            coulomb_mev_nm=self.coulomb_mev_nm,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=self.angle_origin_rad,
            self_cell_policy=self.self_cell_policy,
            self_cell_description=self.self_cell_description,
            fourier_convention=self.fourier_convention,
        )
        attestation = self._builder_attestation
        if (
            not isinstance(attestation, _CoRotatingHarmonicFockAttestation)
            or current_sha != attestation.action_sha256
            or attestation.bundle_fingerprint != self.bundle_fingerprint
            or attestation.requested_mode != self.mode
            or attestation.mode_mod_nphi != self.mode_mod_nphi
            or attestation.nphi != self.nphi
            or attestation.rotation_fingerprint != rotation_fingerprint
            or attestation.angle_origin_rad != self.angle_origin_rad
            or attestation.self_cell_policy != self.self_cell_policy
            or attestation.fourier_convention != self.fourier_convention
        ):
            raise ValueError("harmonic Fock action attestation is no longer valid")

    def __call__(self, density_mode: ComplexArray) -> ComplexArray:
        """Apply this mode block to an arbitrary finite complex ``D_m``."""

        density = np.asarray(density_mode, dtype=np.complex128)
        nr = self.exchange_tensor_mev_nm2.shape[0]
        nactive = self.exchange_tensor_mev_nm2.shape[2]
        if density.shape != (nactive, nactive, nr):
            raise ValueError("co-rotating density mode has an incompatible shape")
        if not np.all(np.isfinite(density)):
            raise ValueError("co-rotating density mode contains non-finite values")
        sigma = -np.einsum(
            "ijadbc,bcj,j->adi",
            self.exchange_tensor_mev_nm2,
            density,
            self.k_weights_nm2,
            optimize=True,
        )
        if not np.all(np.isfinite(sigma)):
            raise ValueError("co-rotating harmonic Fock action is non-finite")
        return sigma

    def fingerprint(self) -> str:
        """Return the concrete action hash (shared by modulo aliases)."""

        attestation = self._builder_attestation
        if not isinstance(attestation, _CoRotatingHarmonicFockAttestation):
            raise ValueError("harmonic Fock action has no builder attestation")
        return attestation.action_sha256

    def action_storage_fingerprint(self) -> str:
        return self.fingerprint()

    def is_zero_action(self) -> bool:
        return float(np.max(np.abs(self.exchange_tensor_mev_nm2))) == 0.0


@dataclass(frozen=True)
class _PolarHarmonicFockAttestation:
    target_bundle_fingerprint: str
    radial_bundle_fingerprint: str
    action_sha256: str
    nr: int
    target_nphi: int
    interaction_nphi: int
    target_mode_aliases: tuple[int, ...]
    operator_target_mode_aliases: tuple[int, ...]
    harmonic_action_fingerprints: tuple[str, ...]
    rotation_fingerprint: str


@dataclass(frozen=True)
class PolarHarmonicProjectedFockOperator:
    """Reconstruct selected co-rotating Fock harmonics on a polar bundle.

    This closes the supplied finite angular Fourier channels of the axial
    radial interaction.  It may be paired with a non-axial one-body bundle as
    a diagnostic, but it does not replace the exact off-axis wavefunctions in
    the interaction vertices.
    """

    harmonic_operators: tuple[CoRotatingHarmonicFockOperator, ...] = field(
        repr=False, compare=False
    )
    radial_bundle_fingerprint: str
    target_bundle_fingerprint: str
    nr: int
    target_nphi: int
    interaction_nphi: int
    target_mode_aliases: tuple[int, ...]
    operator_target_mode_aliases: tuple[int, ...]
    active_jz: tuple[float, ...]
    exponent_sign: int
    interaction_scope: str
    _builder_attestation: object = field(repr=False, compare=False)

    @staticmethod
    def _action_sha256(
        *,
        target_bundle_fingerprint: str,
        radial_bundle_fingerprint: str,
        nr: int,
        target_nphi: int,
        interaction_nphi: int,
        target_mode_aliases: tuple[int, ...],
        operator_target_mode_aliases: tuple[int, ...],
        harmonic_action_fingerprints: tuple[str, ...],
        rotation_fingerprint: str,
        interaction_scope: str,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"inas-gasb-polar-harmonic-fock-v1")
        digest.update(target_bundle_fingerprint.encode())
        digest.update(radial_bundle_fingerprint.encode())
        digest.update(str(int(nr)).encode())
        digest.update(str(int(target_nphi)).encode())
        digest.update(str(int(interaction_nphi)).encode())
        digest.update(str(tuple(target_mode_aliases)).encode())
        digest.update(str(tuple(operator_target_mode_aliases)).encode())
        for fingerprint in harmonic_action_fingerprints:
            digest.update(fingerprint.encode())
        digest.update(rotation_fingerprint.encode())
        digest.update(interaction_scope.encode())
        return digest.hexdigest()

    @classmethod
    def from_harmonic_operators(
        cls,
        radial_bundle: Kane4Bundle,
        target_bundle: Kane4Bundle,
        harmonic_operators: tuple[CoRotatingHarmonicFockOperator, ...],
        *,
        target_nphi: int,
        spec: AxialRotationSpec | None = None,
    ) -> "PolarHarmonicProjectedFockOperator":
        rotation = AxialRotationSpec() if spec is None else spec
        target_nphi_value = _validated_nphi(target_nphi)
        operators = tuple(harmonic_operators)
        if not operators:
            raise ValueError("at least one harmonic Fock operator is required")
        for harmonic in operators:
            harmonic.validate_against_bundle(radial_bundle)
        interaction_nphi = operators[0].nphi
        if any(harmonic.nphi != interaction_nphi for harmonic in operators):
            raise ValueError("harmonic operators use different interaction nphi")
        operator_aliases = tuple(
            int(harmonic.mode) % target_nphi_value for harmonic in operators
        )
        grouped: dict[int, list[CoRotatingHarmonicFockOperator]] = {}
        for alias, harmonic in zip(operator_aliases, operators):
            grouped.setdefault(alias, []).append(harmonic)
        target_nyquist = target_nphi_value // 2
        for alias, group in grouped.items():
            expected_interaction_aliases = {
                target_nyquist % interaction_nphi,
                (-target_nyquist) % interaction_nphi,
            }
            observed_interaction_aliases = {
                harmonic.mode_mod_nphi for harmonic in group
            }
            requires_nyquist_pair = (
                alias == target_nyquist
                and len(expected_interaction_aliases) == 2
            )
            if requires_nyquist_pair:
                if (
                    len(group) != 2
                    or observed_interaction_aliases
                    != expected_interaction_aliases
                ):
                    raise ValueError(
                        "target Nyquist requires its +/- interaction-mode pair"
                    )
            elif len(group) != 1:
                raise ValueError(
                    "only a +/- target-Nyquist pair may alias on the target grid"
                )
        ordered = sorted(
            zip(operator_aliases, operators),
            key=lambda item: (item[0], item[1].mode),
        )
        operator_aliases = tuple(item[0] for item in ordered)
        operators = tuple(item[1] for item in ordered)
        target_aliases = tuple(sorted(set(operator_aliases)))
        alias_set = set(target_aliases)
        if 0 not in alias_set or {
            (-alias) % target_nphi_value for alias in alias_set
        } != alias_set:
            raise ValueError("target harmonic family must include zero and conjugate closure")
        AxialAveragedProjectedFockOperator._validate_polar_mesh(
            radial_bundle,
            target_bundle,
            nphi=target_nphi_value,
        )
        scope = (
            "selected co-rotating axial-radial exchange harmonics on a polar "
            "target; not exact off-axis interaction vertices"
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        fingerprints = tuple(operator.fingerprint() for operator in operators)
        action_sha = cls._action_sha256(
            target_bundle_fingerprint=target_bundle.fingerprint(),
            radial_bundle_fingerprint=radial_bundle.fingerprint(),
            nr=radial_bundle.nk,
            target_nphi=target_nphi_value,
            interaction_nphi=interaction_nphi,
            target_mode_aliases=target_aliases,
            operator_target_mode_aliases=operator_aliases,
            harmonic_action_fingerprints=fingerprints,
            rotation_fingerprint=rotation_fingerprint,
            interaction_scope=scope,
        )
        attestation = _PolarHarmonicFockAttestation(
            target_bundle_fingerprint=target_bundle.fingerprint(),
            radial_bundle_fingerprint=radial_bundle.fingerprint(),
            action_sha256=action_sha,
            nr=radial_bundle.nk,
            target_nphi=target_nphi_value,
            interaction_nphi=interaction_nphi,
            target_mode_aliases=target_aliases,
            operator_target_mode_aliases=operator_aliases,
            harmonic_action_fingerprints=fingerprints,
            rotation_fingerprint=rotation_fingerprint,
        )
        return cls(
            harmonic_operators=operators,
            radial_bundle_fingerprint=radial_bundle.fingerprint(),
            target_bundle_fingerprint=target_bundle.fingerprint(),
            nr=radial_bundle.nk,
            target_nphi=target_nphi_value,
            interaction_nphi=interaction_nphi,
            target_mode_aliases=target_aliases,
            operator_target_mode_aliases=operator_aliases,
            active_jz=tuple(np.asarray(rotation.active_jz, dtype=float)),
            exponent_sign=int(rotation.exponent_sign),
            interaction_scope=scope,
            _builder_attestation=attestation,
        )

    def co_rotating_modes(self, matrices: ComplexArray) -> ComplexArray:
        rotation = AxialRotationSpec(
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        return _co_rotating_modes_on_polar_grid(
            matrices,
            nr=self.nr,
            nphi=self.target_nphi,
            spec=rotation,
        )

    def reconstruct_modes(self, modes: ComplexArray) -> ComplexArray:
        rotation = AxialRotationSpec(
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        return _reconstruct_co_rotating_modes_on_polar_grid(
            modes,
            nr=self.nr,
            nphi=self.target_nphi,
            spec=rotation,
        )

    def validate_against_bundle(
        self, target_bundle: Kane4Bundle, *, atol: float = 1e-10
    ) -> None:
        target_bundle.validate()
        if self.target_bundle_fingerprint != target_bundle.fingerprint():
            raise ValueError("polar harmonic Fock and target bundle fingerprints differ")
        if any(
            operator.bundle_fingerprint != self.radial_bundle_fingerprint
            for operator in self.harmonic_operators
        ):
            raise ValueError("polar harmonic Fock radial source fingerprint changed")
        rotation = AxialRotationSpec(
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        fingerprints = tuple(
            operator.fingerprint() for operator in self.harmonic_operators
        )
        expected = self._action_sha256(
            target_bundle_fingerprint=self.target_bundle_fingerprint,
            radial_bundle_fingerprint=self.radial_bundle_fingerprint,
            nr=self.nr,
            target_nphi=self.target_nphi,
            interaction_nphi=self.interaction_nphi,
            target_mode_aliases=self.target_mode_aliases,
            operator_target_mode_aliases=self.operator_target_mode_aliases,
            harmonic_action_fingerprints=fingerprints,
            rotation_fingerprint=rotation_fingerprint,
            interaction_scope=self.interaction_scope,
        )
        attestation = self._builder_attestation
        if (
            not isinstance(attestation, _PolarHarmonicFockAttestation)
            or attestation.action_sha256 != expected
            or attestation.target_bundle_fingerprint
            != self.target_bundle_fingerprint
            or attestation.radial_bundle_fingerprint
            != self.radial_bundle_fingerprint
            or attestation.nr != self.nr
            or attestation.target_nphi != self.target_nphi
            or attestation.interaction_nphi != self.interaction_nphi
            or attestation.target_mode_aliases != self.target_mode_aliases
            or attestation.operator_target_mode_aliases
            != self.operator_target_mode_aliases
            or attestation.harmonic_action_fingerprints != fingerprints
            or attestation.rotation_fingerprint != rotation_fingerprint
        ):
            raise ValueError("polar harmonic Fock attestation is invalid")

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        density_modes = self.co_rotating_modes(density_delta)
        sigma_modes = np.zeros_like(density_modes)
        for alias in self.target_mode_aliases:
            group = [
                operator
                for operator_alias, operator in zip(
                    self.operator_target_mode_aliases,
                    self.harmonic_operators,
                )
                if operator_alias == alias
            ]
            sigma_modes[:, :, :, alias] = sum(
                (operator(density_modes[:, :, :, alias]) for operator in group),
                start=np.zeros_like(density_modes[:, :, :, alias]),
            ) / float(len(group))
        sigma = self.reconstruct_modes(sigma_modes)
        hermiticity_error = float(
            np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1)))
        )
        if not np.all(np.isfinite(sigma)) or hermiticity_error > 1e-8:
            raise ValueError(
                "polar harmonic Fock action is invalid: "
                f"Hermiticity error={hermiticity_error:.3e}"
            )
        return sigma

    def energy_density_mev_nm2(self, density_delta: ComplexArray) -> float:
        sigma = self(density_delta)
        radial_weights = self.harmonic_operators[0].k_weights_nm2
        weights = np.repeat(radial_weights / self.target_nphi, self.target_nphi)
        return fock_energy_density_mev_nm2(density_delta, sigma, weights)

    def fingerprint(self) -> str:
        attestation = self._builder_attestation
        if not isinstance(attestation, _PolarHarmonicFockAttestation):
            raise ValueError("polar harmonic Fock has no builder attestation")
        return attestation.action_sha256

    def action_storage_fingerprint(self) -> str:
        return self.fingerprint()

    def is_zero_action(self) -> bool:
        return all(operator.is_zero_action() for operator in self.harmonic_operators)
