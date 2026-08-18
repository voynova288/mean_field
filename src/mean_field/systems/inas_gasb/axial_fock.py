"""Axial reduction of the projected Kane4 Fock operator.

The target momentum is kept at ``phi=0`` on each positive radial shell.  Source
angles are integrated explicitly, including the 2D mesh self cell.  The active
source-leg rotations cancel analytically against ``D(phi)=U(phi)D(0)U(phi)^dagger``;
the remaining microscopic rotation is retained in the projected density vertex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import operator

import numpy as np

from .axial import AxialRotationSpec, axial_radial_time_reversal_residuals
from .projected_model import (
    Kane4Bundle,
    apply_e1h1_coherence_exchange_superoperator,
    e1h1_coherence_exchange_tensors_mev_nm2,
    fock_energy_density_mev_nm2,
    hermitian_density_from_e1h1_coherence,
)

ComplexArray = np.ndarray
FloatArray = np.ndarray
SELF_CELL_POLICY = "equal-area circular 2D cell from radial_weight/nphi"
ANGLE_ORIGIN_RAD = 0.0


@dataclass(frozen=True)
class _AxialFockAttestation:
    bundle_fingerprint: str
    action_sha256: str
    nphi: int
    rotation_fingerprint: str
    angle_origin_rad: float
    self_cell_policy: str


@dataclass(frozen=True)
class _AxialAveragedFockAttestation:
    target_bundle_fingerprint: str
    radial_action_fingerprint: str
    action_sha256: str
    nr: int
    nphi: int
    rotation_fingerprint: str


@dataclass(frozen=True)
class _E1H1CoherenceFockAttestation:
    parent_action_fingerprint: str
    bundle_fingerprint: str
    action_sha256: str


def _validated_nphi(nphi: int) -> int:
    value = int(nphi)
    if value != nphi or value < 2 or value % 2:
        raise ValueError("nphi must be an even integer >=2")
    return value


def _validate_canonical_axial_e1h1_basis(bundle: Kane4Bundle) -> None:
    basis = bundle.basis
    if (
        basis.labels != ("E1+", "E1-", "H1+", "H1-")
        or basis.electron_indices != (0, 1)
        or basis.hole_indices != (2, 3)
    ):
        raise ValueError(
            "axial E1/H1 Fock requires canonical labels and E1[:2]/H1[2:] partition"
        )


def _validated_radial_time_reversal_source(
    bundle: Kane4Bundle,
    rotation: AxialRotationSpec,
) -> dict[str, float]:
    """Use strict defaults or tighter source-recorded finite TR tolerances."""

    residuals = axial_radial_time_reversal_residuals(bundle, spec=rotation)
    h0_tolerance = 1e-10
    frame_tolerance = 1e-10
    verification = bundle.provenance.get("finite_k_time_reversal_verification")
    if bundle.provenance.get("finite_k_time_reversal_sewing_verified") is True:
        if not isinstance(verification, dict):
            raise ValueError("verified radial TR provenance lacks its tolerance receipt")
        h0_tolerance = float(verification.get("h0_tolerance_mev", np.nan))
        frame_tolerance = float(
            verification.get("frame_tolerance_nm_minus_half", np.nan)
        )
        if (
            not np.isfinite(h0_tolerance)
            or not np.isfinite(frame_tolerance)
            or h0_tolerance < 0.0
            or frame_tolerance < 0.0
            or h0_tolerance > 1e-8
            or frame_tolerance > 1e-8
        ):
            raise ValueError("radial TR provenance tolerances are invalid or too loose")
    if (
        residuals["h0_radial_time_reversal_error_mev"] > h0_tolerance
        or residuals["h0_radial_projector_error_mev"] > h0_tolerance
        or residuals["frame_radial_time_reversal_error_nm_minus_half"]
        > frame_tolerance
    ):
        raise ValueError(
            "radial Kane4 source fails axial time-reversal sewing under its "
            "recorded numerical precision"
        )
    return residuals


def _validated_harmonic_mode(mode: int) -> int:
    if isinstance(mode, (bool, np.bool_)):
        raise ValueError("co-rotating harmonic mode must be an integer")
    try:
        return int(operator.index(mode))
    except TypeError as error:
        raise ValueError("co-rotating harmonic mode must be an integer") from error


def co_rotating_mode_alias(mode: int, nphi: int) -> int:
    """Return the finite-angle-grid residue that represents ``mode``.

    On the uniform grid ``theta_l=2*pi*l/nphi``, modes that differ by an
    integer multiple of ``nphi`` have exactly the same discrete action.
    """

    nphi_value = _validated_nphi(nphi)
    return _validated_harmonic_mode(mode) % nphi_value


def _rotation_fingerprint(spec: AxialRotationSpec) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(spec.micro_jz, dtype=np.float64).view(np.uint8))
    digest.update(np.ascontiguousarray(spec.active_jz, dtype=np.float64).view(np.uint8))
    digest.update(str(int(spec.exponent_sign)).encode())
    return digest.hexdigest()


def _action_sha256(
    tensor: np.ndarray,
    weights: np.ndarray,
    *,
    bundle_fingerprint: str,
    nphi: int,
    epsilon_r: float,
    coulomb_mev_nm: float,
    rotation_fingerprint: str,
    angle_origin_rad: float,
    self_cell_policy: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(bundle_fingerprint.encode())
    digest.update(str(int(nphi)).encode())
    digest.update(np.float64(epsilon_r).tobytes())
    digest.update(np.float64(coulomb_mev_nm).tobytes())
    digest.update(rotation_fingerprint.encode())
    digest.update(np.float64(angle_origin_rad).tobytes())
    digest.update(self_cell_policy.encode())
    for array in (tensor, weights):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _uniform_green_for_pair_mev_nm2(
    q_nm_inv: float,
    z_nm: np.ndarray,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float,
    self_cell_weight_nm2: float | None = None,
) -> np.ndarray:
    distance = np.abs(z_nm[:, None] - z_nm[None, :])
    prefactor = 2.0 * np.pi * float(coulomb_mev_nm) / float(epsilon_r)
    if self_cell_weight_nm2 is None:
        if not q_nm_inv > 0.0:
            raise ValueError("off-diagonal axial Green pair requires q>0")
        return prefactor / q_nm_inv * np.exp(-q_nm_inv * distance)
    if self_cell_weight_nm2 <= 0.0:
        raise ValueError("axial self-cell weight must be positive")
    cell_area = (2.0 * np.pi) ** 2 * float(self_cell_weight_nm2)
    q_cell = np.sqrt(cell_area / np.pi)
    averaged = np.empty_like(distance)
    zero_distance = distance < 1e-14
    averaged[zero_distance] = 2.0 / q_cell
    nonzero = ~zero_distance
    averaged[nonzero] = (
        2.0
        * (-np.expm1(-q_cell * distance[nonzero]))
        / (q_cell**2 * distance[nonzero])
    )
    return prefactor * averaged


def precompute_axial_harmonic_exchange_tensors_mev_nm2(
    radial_bundle: Kane4Bundle,
    *,
    modes: tuple[int, ...],
    nphi: int,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
    spec: AxialRotationSpec | None = None,
) -> dict[int, ComplexArray]:
    """Build several co-rotating harmonic blocks in one angular pass.

    The Fourier convention is
    ``D_tilde(theta)=sum_m exp(+1j*m*theta) D_m`` and therefore
    ``K_m=(1/nphi) sum_l exp(+1j*m*theta_l) C_l``.  Requested modes must be
    distinct modulo the interaction quadrature ``nphi``.  Sharing the costly
    microscopic ``C_l`` contraction makes a conjugate/C4 mode family no more
    ambiguous than separate builds while avoiding repeated physics work.
    """

    rotation = AxialRotationSpec() if spec is None else spec
    radial_bundle.validate()
    _validate_canonical_axial_e1h1_basis(radial_bundle)
    nphi_value = _validated_nphi(nphi)
    requested_modes = tuple(_validated_harmonic_mode(mode) for mode in modes)
    if not requested_modes:
        raise ValueError("at least one harmonic mode is required")
    aliases = tuple(co_rotating_mode_alias(mode, nphi_value) for mode in requested_modes)
    if len(set(aliases)) != len(aliases):
        raise ValueError("harmonic modes must be distinct modulo nphi")
    phase_modes = tuple(
        alias if alias <= nphi_value // 2 else alias - nphi_value
        for alias in aliases
    )
    if epsilon_r <= 0.0 or coulomb_mev_nm <= 0.0:
        raise ValueError("dielectric constant and Coulomb constant must be positive")
    if not bool(radial_bundle.provenance.get("radial_only", False)):
        raise ValueError("axial-reduced Fock requires a declared radial-only bundle")
    k = np.asarray(radial_bundle.k_cart_nm_inv, dtype=float)
    if np.any(np.abs(k[:, 1]) > 1e-12) or np.any(k[:, 0] <= 0.0):
        raise ValueError("radial bundle must contain strictly positive kx and ky=0")
    if np.any(np.diff(k[:, 0]) <= 0.0):
        raise ValueError("radial momentum nodes must be strictly increasing")
    _validated_radial_time_reversal_source(radial_bundle, rotation)

    phi = np.asarray(radial_bundle.micro_wavefunctions, dtype=np.complex128)
    z = np.asarray(radial_bundle.z_nm, dtype=float)
    wz = np.asarray(radial_bundle.z_weights_nm, dtype=float)
    wk = np.asarray(radial_bundle.weights_nm2, dtype=float)
    nr, _nz, _nmicro, nactive = phi.shape
    angles = (
        ANGLE_ORIGIN_RAD
        + 2.0 * np.pi * np.arange(nphi_value, dtype=float) / float(nphi_value)
    )
    microscopic_phases = np.exp(
        1j
        * int(rotation.exponent_sign)
        * angles[:, None]
        * np.asarray(rotation.micro_jz, dtype=float)[None, :]
    )
    shape = (nr, nr, nactive, nactive, nactive, nactive)
    tensors = {
        mode: np.zeros(shape, dtype=np.complex128) for mode in requested_modes
    }
    for ik in range(nr):
        radius_k = float(k[ik, 0])
        phi_k_conj = phi[ik].conj()
        for ip in range(nr):
            radius_p = float(k[ip, 0])
            b_vertices = np.einsum(
                "zma,lm,zmb->lzab",
                phi_k_conj,
                microscopic_phases,
                phi[ip],
                optimize=True,
            )
            for iangle, angle in enumerate(angles):
                if ik == ip and iangle == 0:
                    green = _uniform_green_for_pair_mev_nm2(
                        0.0,
                        z,
                        epsilon_r=epsilon_r,
                        coulomb_mev_nm=coulomb_mev_nm,
                        self_cell_weight_nm2=wk[ip] / float(nphi_value),
                    )
                else:
                    q_squared = (
                        (radius_k - radius_p) ** 2
                        + 4.0
                        * radius_k
                        * radius_p
                        * np.sin(0.5 * angle) ** 2
                    )
                    q = float(np.sqrt(max(q_squared, 0.0)))
                    green = _uniform_green_for_pair_mev_nm2(
                        q,
                        z,
                        epsilon_r=epsilon_r,
                        coulomb_mev_nm=coulomb_mev_nm,
                    )
                vertex = b_vertices[iangle]
                contraction = np.einsum(
                    "xab,ydc,xy,x,y->adbc",
                    vertex,
                    vertex.conj(),
                    green,
                    wz,
                    wz,
                    optimize=True,
                )
                for mode, alias, phase_mode in zip(
                    requested_modes, aliases, phase_modes
                ):
                    if alias == 0:
                        tensors[mode][ik, ip] += contraction / float(nphi_value)
                    else:
                        harmonic_phase = np.exp(
                            1j * float(phase_mode) * float(angle)
                        )
                        tensors[mode][ik, ip] += (
                            harmonic_phase * contraction / float(nphi_value)
                        )
    for mode, alias in zip(requested_modes, aliases):
        if not np.all(np.isfinite(tensors[mode])):
            if alias == 0:
                raise ValueError(
                    "axial-reduced exchange tensor contains non-finite values"
                )
            raise ValueError(
                f"axial harmonic exchange tensor for mode {mode} is non-finite"
            )
    return tensors


def precompute_axial_harmonic_exchange_tensor_mev_nm2(
    radial_bundle: Kane4Bundle,
    *,
    mode: int,
    nphi: int,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
    spec: AxialRotationSpec | None = None,
) -> ComplexArray:
    """Return one finite-grid co-rotating harmonic exchange block."""

    requested_mode = _validated_harmonic_mode(mode)
    return precompute_axial_harmonic_exchange_tensors_mev_nm2(
        radial_bundle,
        modes=(requested_mode,),
        nphi=nphi,
        epsilon_r=epsilon_r,
        coulomb_mev_nm=coulomb_mev_nm,
        spec=spec,
    )[requested_mode]


def precompute_axial_reduced_exchange_tensor_mev_nm2(
    radial_bundle: Kane4Bundle,
    *,
    nphi: int,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
    spec: AxialRotationSpec | None = None,
) -> ComplexArray:
    """Return the established ``m=0`` axial exchange tensor.

    This compatibility entry point deliberately delegates to the generalized
    builder with ``mode=0``; that branch retains the original operation order.
    """

    return precompute_axial_harmonic_exchange_tensor_mev_nm2(
        radial_bundle,
        mode=0,
        nphi=nphi,
        epsilon_r=epsilon_r,
        coulomb_mev_nm=coulomb_mev_nm,
        spec=spec,
    )


@dataclass(frozen=True)
class AxialProjectedFockOperator:
    """Static, builder-attested Fock action on positive radial Kane4 nodes."""

    exchange_tensor_mev_nm2: ComplexArray
    k_weights_nm2: FloatArray
    bundle_fingerprint: str
    nphi: int
    epsilon_r: float
    coulomb_mev_nm: float
    self_cell_description: str
    micro_jz: tuple[float, ...]
    active_jz: tuple[float, ...]
    exponent_sign: int
    angle_origin_rad: float
    self_cell_policy: str
    _builder_attestation: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bundle_fingerprint or not self.self_cell_description:
            raise ValueError("axial Fock bundle fingerprint and self-cell description are required")
        nphi_value = _validated_nphi(self.nphi)
        rotation = AxialRotationSpec(
            micro_jz=np.asarray(self.micro_jz, dtype=float),
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        if float(self.angle_origin_rad) != ANGLE_ORIGIN_RAD:
            raise ValueError("axial Fock angle origin does not match the builder policy")
        if self.self_cell_policy != SELF_CELL_POLICY:
            raise ValueError("axial Fock self-cell policy does not match the builder policy")
        tensor = np.array(
            self.exchange_tensor_mev_nm2, dtype=np.complex128, copy=True
        )
        weights = np.array(self.k_weights_nm2, dtype=float, copy=True)
        if tensor.ndim != 6 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("axial exchange tensor must have shape (nr,nr,n,n,n,n)")
        nr = tensor.shape[0]
        n = tensor.shape[2]
        if n != 4:
            raise ValueError("axial E1/H1 Fock requires four active states")
        if tensor.shape != (nr, nr, n, n, n, n) or weights.shape != (nr,):
            raise ValueError("axial Fock tensor and radial weights are incompatible")
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(weights)):
            raise ValueError("axial Fock arrays must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("axial Fock radial weights must be positive")
        hermitian_reverse = tensor.conj().transpose(0, 1, 3, 2, 5, 4)
        hermiticity_error = float(np.max(np.abs(tensor - hermitian_reverse)))
        reciprocal_reverse = tensor.transpose(1, 0, 5, 4, 3, 2)
        reciprocity_error = float(np.max(np.abs(tensor - reciprocal_reverse)))
        if hermiticity_error > 1e-8:
            raise ValueError(
                f"axial Fock tensor violates Hermiticity: {hermiticity_error:.3e}"
            )
        if reciprocity_error > 1e-8:
            raise ValueError(
                f"axial Fock tensor violates reciprocity: {reciprocity_error:.3e}"
            )
        expected_sha = _action_sha256(
            tensor,
            weights,
            bundle_fingerprint=self.bundle_fingerprint,
            nphi=nphi_value,
            epsilon_r=self.epsilon_r,
            coulomb_mev_nm=self.coulomb_mev_nm,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=self.angle_origin_rad,
            self_cell_policy=self.self_cell_policy,
        )
        attestation = self._builder_attestation
        if not isinstance(attestation, _AxialFockAttestation):
            raise ValueError("axial Fock action was not issued by the certified builder")
        if (
            attestation.bundle_fingerprint != self.bundle_fingerprint
            or attestation.action_sha256 != expected_sha
            or attestation.nphi != nphi_value
            or attestation.rotation_fingerprint != rotation_fingerprint
            or attestation.angle_origin_rad != self.angle_origin_rad
            or attestation.self_cell_policy != self.self_cell_policy
        ):
            raise ValueError("axial Fock action lacks a matching builder attestation")
        tensor.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "exchange_tensor_mev_nm2", tensor)
        object.__setattr__(self, "k_weights_nm2", weights)

    @classmethod
    def from_precomputed_uniform_dielectric(
        cls,
        radial_bundle: Kane4Bundle,
        exchange_tensor_mev_nm2: ComplexArray,
        *,
        nphi: int,
        epsilon_r: float,
        expected_action_fingerprint: str,
        coulomb_mev_nm: float = 1439.96448,
        spec: AxialRotationSpec | None = None,
    ) -> "AxialProjectedFockOperator":
        """Rehydrate a source-bound tensor previously issued by this builder.

        The constructor recomputes the action fingerprint and all structural
        checks.  Persistence metadata must retain that fingerprint so callers
        can compare it before accepting a cached tensor.
        """

        rotation = AxialRotationSpec() if spec is None else spec
        radial_bundle.validate()
        _validate_canonical_axial_e1h1_basis(radial_bundle)
        _validated_radial_time_reversal_source(radial_bundle, rotation)
        nphi_value = _validated_nphi(nphi)
        if epsilon_r <= 0.0 or coulomb_mev_nm <= 0.0:
            raise ValueError("dielectric constant and Coulomb constant must be positive")
        tensor = np.asarray(exchange_tensor_mev_nm2, dtype=np.complex128)
        weights = np.asarray(radial_bundle.weights_nm2, dtype=float)
        bundle_fingerprint = radial_bundle.fingerprint()
        rotation_fingerprint = _rotation_fingerprint(rotation)
        action_sha = _action_sha256(
            tensor,
            weights,
            bundle_fingerprint=bundle_fingerprint,
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            coulomb_mev_nm=coulomb_mev_nm,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
        )
        if (
            not isinstance(expected_action_fingerprint, str)
            or len(expected_action_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_action_fingerprint
            )
        ):
            raise ValueError(
                "expected axial action fingerprint must be a lowercase SHA-256 digest"
            )
        if action_sha != expected_action_fingerprint:
            raise ValueError(
                "precomputed axial tensor does not match its independently pinned "
                "action fingerprint"
            )
        attestation = _AxialFockAttestation(
            bundle_fingerprint=bundle_fingerprint,
            action_sha256=action_sha,
            nphi=nphi_value,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
        )
        return cls(
            exchange_tensor_mev_nm2=tensor,
            k_weights_nm2=weights,
            bundle_fingerprint=bundle_fingerprint,
            nphi=nphi_value,
            epsilon_r=float(epsilon_r),
            coulomb_mev_nm=float(coulomb_mev_nm),
            self_cell_description=(
                "axial angle average with uniform dielectric and circular 2D "
                "self cell built from radial_weight/nphi"
            ),
            micro_jz=tuple(np.asarray(rotation.micro_jz, dtype=float)),
            active_jz=tuple(np.asarray(rotation.active_jz, dtype=float)),
            exponent_sign=int(rotation.exponent_sign),
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
            _builder_attestation=attestation,
        )

    @classmethod
    def from_bundle_uniform_dielectric(
        cls,
        radial_bundle: Kane4Bundle,
        *,
        nphi: int,
        epsilon_r: float,
        coulomb_mev_nm: float = 1439.96448,
        spec: AxialRotationSpec | None = None,
    ) -> "AxialProjectedFockOperator":
        rotation = AxialRotationSpec() if spec is None else spec
        nphi_value = _validated_nphi(nphi)
        tensor = precompute_axial_reduced_exchange_tensor_mev_nm2(
            radial_bundle,
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            coulomb_mev_nm=coulomb_mev_nm,
            spec=rotation,
        )
        expected_action_fingerprint = _action_sha256(
            tensor,
            np.asarray(radial_bundle.weights_nm2, dtype=float),
            bundle_fingerprint=radial_bundle.fingerprint(),
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            coulomb_mev_nm=coulomb_mev_nm,
            rotation_fingerprint=_rotation_fingerprint(rotation),
            angle_origin_rad=ANGLE_ORIGIN_RAD,
            self_cell_policy=SELF_CELL_POLICY,
        )
        return cls.from_precomputed_uniform_dielectric(
            radial_bundle,
            tensor,
            nphi=nphi_value,
            epsilon_r=epsilon_r,
            expected_action_fingerprint=expected_action_fingerprint,
            coulomb_mev_nm=coulomb_mev_nm,
            spec=rotation,
        )

    def validate_against_bundle(
        self, radial_bundle: Kane4Bundle, *, atol: float = 1e-10
    ) -> None:
        radial_bundle.validate()
        _validate_canonical_axial_e1h1_basis(radial_bundle)
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("axial bundle validation tolerance must be finite and nonnegative")
        if self.bundle_fingerprint != radial_bundle.fingerprint():
            raise ValueError("axial Fock operator and radial bundle fingerprints differ")
        if not bool(radial_bundle.provenance.get("radial_only", False)):
            raise ValueError("axial Fock operator requires a radial-only bundle")
        if not np.allclose(
            self.k_weights_nm2,
            radial_bundle.weights_nm2,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError("axial Fock weights do not match the radial bundle")
        self.validate_attestation()

    def validate_attestation(self) -> None:
        """Recompute the certified action hash from current immutable arrays."""

        rotation = AxialRotationSpec(
            micro_jz=np.asarray(self.micro_jz, dtype=float),
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        current_sha = _action_sha256(
            self.exchange_tensor_mev_nm2,
            self.k_weights_nm2,
            bundle_fingerprint=self.bundle_fingerprint,
            nphi=self.nphi,
            epsilon_r=self.epsilon_r,
            coulomb_mev_nm=self.coulomb_mev_nm,
            rotation_fingerprint=rotation_fingerprint,
            angle_origin_rad=self.angle_origin_rad,
            self_cell_policy=self.self_cell_policy,
        )
        attestation = self._builder_attestation
        if (
            not isinstance(attestation, _AxialFockAttestation)
            or current_sha != attestation.action_sha256
            or attestation.bundle_fingerprint != self.bundle_fingerprint
            or attestation.rotation_fingerprint != rotation_fingerprint
            or attestation.nphi != self.nphi
            or attestation.angle_origin_rad != self.angle_origin_rad
            or attestation.self_cell_policy != self.self_cell_policy
        ):
            raise ValueError("axial Fock parent action attestation is stale or invalid")

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        density = np.asarray(density_delta, dtype=np.complex128)
        nr = self.exchange_tensor_mev_nm2.shape[0]
        n = self.exchange_tensor_mev_nm2.shape[2]
        if density.shape != (n, n, nr):
            raise ValueError("radial density has an incompatible shape")
        if not np.all(np.isfinite(density)):
            raise ValueError("radial density contains non-finite values")
        density_error = float(
            np.max(np.abs(density - np.swapaxes(density.conj(), 0, 1)))
        )
        if density_error > 1e-9:
            raise ValueError(
                f"radial density must be Hermitian: error={density_error:.3e}"
            )
        sigma = -np.einsum(
            "ijadbc,bcj,j->adi",
            self.exchange_tensor_mev_nm2,
            density,
            self.k_weights_nm2,
            optimize=True,
        )
        sigma_error = float(
            np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1)))
        )
        if not np.all(np.isfinite(sigma)) or sigma_error > 1e-8:
            raise ValueError(
                f"axial Fock action is invalid: Hermiticity error={sigma_error:.3e}"
            )
        return sigma

    def energy_density_mev_nm2(self, density_delta: ComplexArray) -> float:
        sigma = self(density_delta)
        return fock_energy_density_mev_nm2(
            density_delta, sigma, self.k_weights_nm2
        )

    def fingerprint(self) -> str:
        self.validate_attestation()
        attestation = self._builder_attestation
        assert isinstance(attestation, _AxialFockAttestation)
        return attestation.action_sha256

    def action_storage_fingerprint(self) -> str:
        return self.fingerprint()

    def is_zero_action(self) -> bool:
        return float(np.max(np.abs(self.exchange_tensor_mev_nm2))) == 0.0


@dataclass(frozen=True)
class AxialE1H1CoherenceFockSuperoperator:
    """Source-bound real-linear Fock action on ``X=D_EH``.

    The two stored kernels retain both the directed ``D_EH`` contribution and
    the generally nonzero ``D_HE=X^dagger`` contribution.  The latter acts on
    ``X*``.  This is an exact coherence-sector view of one certified axial Fock
    operator, not a fitted scalar pairing kernel.
    """

    direct_tensor_mev_nm2: ComplexArray
    conjugate_tensor_mev_nm2: ComplexArray
    k_weights_nm2: FloatArray
    parent_action_fingerprint: str
    bundle_fingerprint: str
    electron_indices: tuple[int, int]
    hole_indices: tuple[int, int]
    _builder_attestation: object = field(repr=False, compare=False)

    @staticmethod
    def _action_sha256(
        direct: np.ndarray,
        conjugate: np.ndarray,
        weights: np.ndarray,
        *,
        parent_action_fingerprint: str,
        bundle_fingerprint: str,
        electron_indices: tuple[int, int],
        hole_indices: tuple[int, int],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"axial-e1h1-coherence-fock-superoperator-v1")
        digest.update(parent_action_fingerprint.encode())
        digest.update(bundle_fingerprint.encode())
        digest.update(repr((electron_indices, hole_indices)).encode())
        for array in (direct, conjugate, weights):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def __post_init__(self) -> None:
        if not self.parent_action_fingerprint or not self.bundle_fingerprint:
            raise ValueError("coherence superoperator requires parent and bundle fingerprints")
        if self.electron_indices != (0, 1) or self.hole_indices != (2, 3):
            raise ValueError("coherence superoperator requires canonical E1[:2]/H1[2:] indices")
        direct = np.array(self.direct_tensor_mev_nm2, dtype=np.complex128, copy=True)
        conjugate = np.array(
            self.conjugate_tensor_mev_nm2, dtype=np.complex128, copy=True
        )
        weights = np.array(self.k_weights_nm2, dtype=float, copy=True)
        if direct.ndim != 6 or direct.shape != conjugate.shape:
            raise ValueError("coherence tensors must have matching rank-six shapes")
        nr = direct.shape[0]
        if direct.shape != (nr, nr, 2, 2, 2, 2) or weights.shape != (nr,):
            raise ValueError("coherence tensors or radial weights have incompatible shapes")
        if not all(np.all(np.isfinite(value)) for value in (direct, conjugate, weights)):
            raise ValueError("coherence superoperator arrays must be finite")
        if np.any(weights <= 0.0):
            raise ValueError("coherence superoperator weights must be positive")
        action_sha = self._action_sha256(
            direct,
            conjugate,
            weights,
            parent_action_fingerprint=self.parent_action_fingerprint,
            bundle_fingerprint=self.bundle_fingerprint,
            electron_indices=self.electron_indices,
            hole_indices=self.hole_indices,
        )
        attestation = self._builder_attestation
        if not isinstance(attestation, _E1H1CoherenceFockAttestation):
            raise ValueError("coherence superoperator was not issued by its certified builder")
        if (
            attestation.parent_action_fingerprint != self.parent_action_fingerprint
            or attestation.bundle_fingerprint != self.bundle_fingerprint
            or attestation.action_sha256 != action_sha
        ):
            raise ValueError("coherence superoperator builder attestation does not match")
        direct.setflags(write=False)
        conjugate.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "direct_tensor_mev_nm2", direct)
        object.__setattr__(self, "conjugate_tensor_mev_nm2", conjugate)
        object.__setattr__(self, "k_weights_nm2", weights)

    @classmethod
    def from_axial_fock_operator(
        cls,
        parent: AxialProjectedFockOperator,
    ) -> "AxialE1H1CoherenceFockSuperoperator":
        """Derive the exact E1--H1 coherence view from one certified parent."""

        if not isinstance(parent, AxialProjectedFockOperator):
            raise TypeError("coherence superoperator requires an AxialProjectedFockOperator")
        parent.validate_attestation()
        direct, conjugate = e1h1_coherence_exchange_tensors_mev_nm2(
            parent.exchange_tensor_mev_nm2
        )
        parent_fingerprint = parent.fingerprint()
        bundle_fingerprint = parent.bundle_fingerprint
        electron_indices = (0, 1)
        hole_indices = (2, 3)
        action_sha = cls._action_sha256(
            direct,
            conjugate,
            parent.k_weights_nm2,
            parent_action_fingerprint=parent_fingerprint,
            bundle_fingerprint=bundle_fingerprint,
            electron_indices=electron_indices,
            hole_indices=hole_indices,
        )
        attestation = _E1H1CoherenceFockAttestation(
            parent_action_fingerprint=parent_fingerprint,
            bundle_fingerprint=bundle_fingerprint,
            action_sha256=action_sha,
        )
        return cls(
            direct_tensor_mev_nm2=direct,
            conjugate_tensor_mev_nm2=conjugate,
            k_weights_nm2=parent.k_weights_nm2,
            parent_action_fingerprint=parent_fingerprint,
            bundle_fingerprint=bundle_fingerprint,
            electron_indices=electron_indices,
            hole_indices=hole_indices,
            _builder_attestation=attestation,
        )

    def __call__(self, coherence_eh: ComplexArray) -> ComplexArray:
        return apply_e1h1_coherence_exchange_superoperator(
            coherence_eh,
            self.direct_tensor_mev_nm2,
            self.conjugate_tensor_mev_nm2,
            self.k_weights_nm2,
        )

    def validate_against_parent(
        self,
        parent: AxialProjectedFockOperator,
        *,
        atol: float = 1e-12,
    ) -> None:
        """Require exact lineage and numerical agreement with the full action."""

        if not isinstance(parent, AxialProjectedFockOperator):
            raise TypeError("coherence validation requires an AxialProjectedFockOperator")
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("coherence-parent validation tolerance must be finite and nonnegative")
        parent.validate_attestation()
        if self.parent_action_fingerprint != parent.fingerprint():
            raise ValueError("coherence superoperator and parent action fingerprints differ")
        if self.bundle_fingerprint != parent.bundle_fingerprint:
            raise ValueError("coherence superoperator and parent bundle fingerprints differ")
        direct, conjugate = e1h1_coherence_exchange_tensors_mev_nm2(
            parent.exchange_tensor_mev_nm2
        )
        if not np.allclose(self.direct_tensor_mev_nm2, direct, rtol=0.0, atol=atol):
            raise ValueError("direct coherence tensor does not match the parent")
        if not np.allclose(
            self.conjugate_tensor_mev_nm2,
            conjugate,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError("conjugate coherence tensor does not match the parent")
        if not np.allclose(
            self.k_weights_nm2,
            parent.k_weights_nm2,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError("coherence weights do not match the parent")

    def direct_impulse_error(self, parent: AxialProjectedFockOperator) -> float:
        """Compare real and imaginary E1--H1 impulses to the full Fock action."""

        self.validate_against_parent(parent)
        nr = self.k_weights_nm2.size
        maximum_error = 0.0
        radial = tuple(range(nr))
        for source in radial:
            for electron in range(2):
                for hole in range(2):
                    for amplitude in (1.0 + 0.0j, 0.0 + 1.0j):
                        coherence = np.zeros((2, 2, nr), dtype=np.complex128)
                        coherence[electron, hole, source] = amplitude
                        density = hermitian_density_from_e1h1_coherence(coherence)
                        full = parent(density)
                        expected = full[np.ix_((0, 1), (2, 3), radial)]
                        actual = self(coherence)
                        maximum_error = max(
                            maximum_error,
                            float(np.max(np.abs(actual - expected))),
                        )
        return maximum_error

    def fingerprint(self) -> str:
        attestation = self._builder_attestation
        if not isinstance(attestation, _E1H1CoherenceFockAttestation):
            raise ValueError("coherence superoperator has no builder attestation")
        return attestation.action_sha256

    def conjugate_to_direct_norm_ratio(self) -> float:
        direct_norm = float(np.linalg.norm(self.direct_tensor_mev_nm2))
        conjugate_norm = float(np.linalg.norm(self.conjugate_tensor_mev_nm2))
        if direct_norm == 0.0:
            return np.inf if conjugate_norm > 0.0 else 0.0
        return conjugate_norm / direct_norm

@dataclass(frozen=True)
class AxialAveragedProjectedFockOperator:
    """Apply only the co-rotating ``m=0`` interaction to a polar 2D bundle.

    The target bundle may contain an exact non-axial one-body Hamiltonian and
    exact off-axis frames.  The interaction remains the explicitly narrower
    axial-radial source model: the density is co-rotated, angle averaged,
    passed through one certified radial Fock operator, and rotated back.  This
    is a controlled diagnostic closure, not an exact off-axis Coulomb action.
    """

    radial_operator: AxialProjectedFockOperator = field(repr=False, compare=False)
    radial_bundle_fingerprint: str
    target_bundle_fingerprint: str
    nr: int
    nphi: int
    active_jz: tuple[float, ...]
    exponent_sign: int
    interaction_scope: str
    _builder_attestation: object = field(repr=False, compare=False)

    @staticmethod
    def _action_sha256(
        *,
        target_bundle_fingerprint: str,
        radial_action_fingerprint: str,
        nr: int,
        nphi: int,
        rotation_fingerprint: str,
        interaction_scope: str,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"axial-averaged-polar-fock-v1")
        digest.update(target_bundle_fingerprint.encode())
        digest.update(radial_action_fingerprint.encode())
        digest.update(str(int(nr)).encode())
        digest.update(str(int(nphi)).encode())
        digest.update(rotation_fingerprint.encode())
        digest.update(interaction_scope.encode())
        return digest.hexdigest()

    @staticmethod
    def _validate_polar_mesh(
        radial_bundle: Kane4Bundle,
        target_bundle: Kane4Bundle,
        *,
        nphi: int,
        atol: float = 1e-11,
    ) -> None:
        radial_bundle.validate()
        target_bundle.validate()
        nphi_value = _validated_nphi(nphi)
        nr = radial_bundle.nk
        if target_bundle.nk != nr * nphi_value:
            raise ValueError("target polar bundle size is not nr*nphi")
        k = np.asarray(target_bundle.k_cart_nm_inv, dtype=float).reshape(
            nr, nphi_value, 2
        )
        radii = np.linalg.norm(k, axis=2)
        radial_radii = np.asarray(radial_bundle.k_cart_nm_inv[:, 0], dtype=float)
        if np.any(np.abs(radial_bundle.k_cart_nm_inv[:, 1]) > atol):
            raise ValueError("source Fock bundle must lie on the positive radial ray")
        if not np.allclose(
            radii, radial_radii[:, None], rtol=0.0, atol=atol
        ):
            raise ValueError("target polar radii do not match the radial bundle")
        expected_angles = 2.0 * np.pi * np.arange(nphi_value) / nphi_value
        actual_angles = np.mod(np.arctan2(k[0, :, 1], k[0, :, 0]), 2.0 * np.pi)
        if not np.allclose(actual_angles, expected_angles, rtol=0.0, atol=atol):
            raise ValueError("target polar angles are not the canonical uniform grid")
        target_weights = np.asarray(target_bundle.weights_nm2, dtype=float).reshape(
            nr, nphi_value
        )
        if not np.allclose(
            np.sum(target_weights, axis=1),
            radial_bundle.weights_nm2,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("target polar weights do not sum to radial weights")
        if not np.allclose(
            target_weights,
            radial_bundle.weights_nm2[:, None] / nphi_value,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError("target polar weights are not uniform within each shell")

    @classmethod
    def from_radial_operator(
        cls,
        radial_bundle: Kane4Bundle,
        target_bundle: Kane4Bundle,
        radial_operator: AxialProjectedFockOperator,
        *,
        nphi: int,
        spec: AxialRotationSpec | None = None,
    ) -> "AxialAveragedProjectedFockOperator":
        rotation = AxialRotationSpec() if spec is None else spec
        nphi_value = _validated_nphi(nphi)
        radial_operator.validate_against_bundle(radial_bundle)
        cls._validate_polar_mesh(
            radial_bundle, target_bundle, nphi=nphi_value
        )
        if bool(target_bundle.provenance.get("radial_only", True)):
            raise ValueError("axial-averaged polar Fock requires a 2D target bundle")
        scope = (
            "co-rotating m=0 axial-radial exchange applied to a polar target; "
            "not exact off-axis interaction"
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        action_sha = cls._action_sha256(
            target_bundle_fingerprint=target_bundle.fingerprint(),
            radial_action_fingerprint=radial_operator.fingerprint(),
            nr=radial_bundle.nk,
            nphi=nphi_value,
            rotation_fingerprint=rotation_fingerprint,
            interaction_scope=scope,
        )
        attestation = _AxialAveragedFockAttestation(
            target_bundle_fingerprint=target_bundle.fingerprint(),
            radial_action_fingerprint=radial_operator.fingerprint(),
            action_sha256=action_sha,
            nr=radial_bundle.nk,
            nphi=nphi_value,
            rotation_fingerprint=rotation_fingerprint,
        )
        return cls(
            radial_operator=radial_operator,
            radial_bundle_fingerprint=radial_bundle.fingerprint(),
            target_bundle_fingerprint=target_bundle.fingerprint(),
            nr=radial_bundle.nk,
            nphi=nphi_value,
            active_jz=tuple(np.asarray(rotation.active_jz, dtype=float)),
            exponent_sign=int(rotation.exponent_sign),
            interaction_scope=scope,
            _builder_attestation=attestation,
        )

    def _rotation(self, angle: float) -> np.ndarray:
        return np.diag(
            np.exp(
                1j
                * int(self.exponent_sign)
                * np.asarray(self.active_jz, dtype=float)
                * float(angle)
            )
        )

    def co_rotating_average(self, matrices: ComplexArray) -> ComplexArray:
        values = np.asarray(matrices, dtype=np.complex128)
        n = len(self.active_jz)
        if values.shape != (n, n, self.nr * self.nphi):
            raise ValueError("polar matrices have an incompatible shape")
        polar = values.reshape(n, n, self.nr, self.nphi)
        average = np.zeros((n, n, self.nr), dtype=np.complex128)
        for iphi in range(self.nphi):
            angle = 2.0 * np.pi * iphi / self.nphi
            unitary = self._rotation(angle)
            average += np.einsum(
                "ab,bcj,cd->adj",
                unitary.conj().T,
                polar[:, :, :, iphi],
                unitary,
                optimize=True,
            )
        return average / float(self.nphi)

    def expand_radial_matrices(self, matrices: ComplexArray) -> ComplexArray:
        radial = np.asarray(matrices, dtype=np.complex128)
        n = len(self.active_jz)
        if radial.shape != (n, n, self.nr):
            raise ValueError("radial matrices have an incompatible shape")
        polar = np.empty((n, n, self.nr, self.nphi), dtype=np.complex128)
        for iphi in range(self.nphi):
            angle = 2.0 * np.pi * iphi / self.nphi
            unitary = self._rotation(angle)
            polar[:, :, :, iphi] = np.einsum(
                "ab,bcj,cd->adj",
                unitary,
                radial,
                unitary.conj().T,
                optimize=True,
            )
        return polar.reshape(n, n, self.nr * self.nphi)

    def validate_against_bundle(
        self, target_bundle: Kane4Bundle, *, atol: float = 1e-10
    ) -> None:
        target_bundle.validate()
        if self.radial_operator.bundle_fingerprint != self.radial_bundle_fingerprint:
            raise ValueError("axial-averaged Fock radial source fingerprint changed")
        if self.target_bundle_fingerprint != target_bundle.fingerprint():
            raise ValueError(
                "axial-averaged Fock and target bundle fingerprints differ"
            )
        rotation = AxialRotationSpec(
            active_jz=np.asarray(self.active_jz, dtype=float),
            exponent_sign=int(self.exponent_sign),
        )
        rotation_fingerprint = _rotation_fingerprint(rotation)
        attestation = self._builder_attestation
        expected = self._action_sha256(
            target_bundle_fingerprint=self.target_bundle_fingerprint,
            radial_action_fingerprint=self.radial_operator.fingerprint(),
            nr=self.nr,
            nphi=self.nphi,
            rotation_fingerprint=rotation_fingerprint,
            interaction_scope=self.interaction_scope,
        )
        if (
            not isinstance(attestation, _AxialAveragedFockAttestation)
            or attestation.action_sha256 != expected
            or attestation.target_bundle_fingerprint
            != self.target_bundle_fingerprint
            or attestation.radial_action_fingerprint
            != self.radial_operator.fingerprint()
            or attestation.nr != self.nr
            or attestation.nphi != self.nphi
            or attestation.rotation_fingerprint != rotation_fingerprint
        ):
            raise ValueError("axial-averaged Fock attestation is invalid")

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        radial_density = self.co_rotating_average(density_delta)
        radial_sigma = self.radial_operator(radial_density)
        return self.expand_radial_matrices(radial_sigma)

    def energy_density_mev_nm2(self, density_delta: ComplexArray) -> float:
        return self.radial_operator.energy_density_mev_nm2(
            self.co_rotating_average(density_delta)
        )

    def fingerprint(self) -> str:
        attestation = self._builder_attestation
        if not isinstance(attestation, _AxialAveragedFockAttestation):
            raise ValueError("axial-averaged Fock has no builder attestation")
        return attestation.action_sha256

    def action_storage_fingerprint(self) -> str:
        return self.fingerprint()

    def is_zero_action(self) -> bool:
        return self.radial_operator.is_zero_action()
