"""Axial-symmetry expansion of a radial Kane4 bundle to a 2D polar mesh."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import KANE8_JZ, kane8_time_reversal_unitary
from .projected_model import Kane4Bundle


@dataclass(frozen=True)
class AxialRotationSpec:
    """Angular-momentum convention verified for kdotpy ``axial=True``."""

    micro_jz: np.ndarray = field(default_factory=lambda: KANE8_JZ.copy())
    active_jz: np.ndarray = field(
        default_factory=lambda: np.asarray((0.5, -0.5, 1.5, -1.5), dtype=float)
    )
    exponent_sign: int = -1

    def __post_init__(self) -> None:
        micro = np.asarray(self.micro_jz, dtype=float)
        active = np.asarray(self.active_jz, dtype=float)
        if micro.shape != (8,) or active.shape != (4,):
            raise ValueError("axial micro/active Jz arrays must have shapes (8,) and (4,)")
        if self.exponent_sign not in {-1, 1}:
            raise ValueError("axial exponent_sign must be +/-1")


def _unitary_from_jz(jz: np.ndarray, phi: float, sign: int) -> np.ndarray:
    return np.diag(np.exp(1j * int(sign) * np.asarray(jz, dtype=float) * float(phi)))


def expand_axial_radial_bundle(
    radial_bundle: Kane4Bundle,
    *,
    nphi: int,
    spec: AxialRotationSpec | None = None,
) -> Kane4Bundle:
    """Expand positive radial nodes into a uniform-angle 2D bundle.

    Radial weights are divided equally among angular nodes.  If the radial
    annuli have equal physical area, the resulting 2D bundle has equal weights
    and is accepted by the current matrix-EI SCF adapter.
    """

    rotation = AxialRotationSpec() if spec is None else spec
    radial_bundle.validate()
    nphi_value = int(nphi)
    if nphi_value != nphi or nphi_value < 2 or nphi_value % 2:
        raise ValueError("nphi must be an even integer >= 2 for exact opposite-k sewing")
    k_radial = np.asarray(radial_bundle.k_cart_nm_inv, dtype=float)
    if np.any(np.abs(k_radial[:, 1]) > 1e-12) or np.any(k_radial[:, 0] <= 0.0):
        raise ValueError("radial bundle must contain strictly positive kx nodes on ky=0")
    if np.any(np.diff(k_radial[:, 0]) <= 0.0):
        raise ValueError("radial k nodes must be strictly increasing")
    if radial_bundle.micro_wavefunctions.shape[2] != 8:
        raise ValueError("axial Kane expansion requires the full eight-orbital microscopic basis")
    if radial_bundle.time_reversal_unitary is None:
        raise ValueError("axial Kane expansion requires a verified active time-reversal unitary")

    phi_values = 2.0 * np.pi * np.arange(nphi_value, dtype=float) / float(nphi_value)
    k_out = []
    weights_out = []
    h_out = []
    wavefunctions_out = []
    current_out = []
    has_current = radial_bundle.dhdk_mev_nm is not None
    for ir, radius in enumerate(k_radial[:, 0]):
        radial_phi = radial_bundle.micro_wavefunctions[ir]
        radial_h = radial_bundle.h0_mev[:, :, ir]
        radial_current = None if not has_current else radial_bundle.dhdk_mev_nm[:, :, :, ir]
        for angle in phi_values:
            micro_u = _unitary_from_jz(rotation.micro_jz, angle, rotation.exponent_sign)
            active_u = _unitary_from_jz(rotation.active_jz, angle, rotation.exponent_sign)
            frame = np.einsum(
                "mn,zna,ab->zmb",
                micro_u,
                radial_phi,
                active_u.conj().T,
                optimize=True,
            )
            h_rotated = active_u @ radial_h @ active_u.conj().T
            k_out.append((radius * np.cos(angle), radius * np.sin(angle)))
            weights_out.append(radial_bundle.weights_nm2[ir] / float(nphi_value))
            h_out.append(h_rotated)
            wavefunctions_out.append(frame)
            if radial_current is not None:
                internal_x = active_u @ radial_current[0] @ active_u.conj().T
                internal_y = active_u @ radial_current[1] @ active_u.conj().T
                cosine = np.cos(angle)
                sine = np.sin(angle)
                current_out.append(
                    np.stack(
                        [
                            cosine * internal_x - sine * internal_y,
                            sine * internal_x + cosine * internal_y,
                        ],
                        axis=0,
                    )
                )

    provenance = {
        **radial_bundle.provenance,
        "axial_expansion": {
            "nphi": nphi_value,
            "angles_rad": phi_values.tolist(),
            "micro_jz": np.asarray(rotation.micro_jz, dtype=float).tolist(),
            "active_jz": np.asarray(rotation.active_jz, dtype=float).tolist(),
            "rotation": f"exp({rotation.exponent_sign:+d} i Jz phi)",
            "full_hamiltonian_covariance_oracle": "phase1_axial_rotation_audit_20260713",
        },
        "radial_only": False,
        "nphi": nphi_value,
        "finite_k_time_reversal_sewing_verified": False,
    }
    return Kane4Bundle(
        k_cart_nm_inv=np.asarray(k_out, dtype=float),
        weights_nm2=np.asarray(weights_out, dtype=float),
        z_nm=np.asarray(radial_bundle.z_nm, dtype=float),
        z_weights_nm=np.asarray(radial_bundle.z_weights_nm, dtype=float),
        h0_mev=np.stack(h_out, axis=-1),
        micro_wavefunctions=np.stack(wavefunctions_out, axis=0),
        dhdk_mev_nm=None if not has_current else np.stack(current_out, axis=-1),
        time_reversal_unitary=np.asarray(radial_bundle.time_reversal_unitary, dtype=np.complex128),
        basis=radial_bundle.basis,
        provenance=provenance,
    )


def expand_axial_radial_matrices(
    radial_matrices: np.ndarray,
    target_bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> np.ndarray:
    """Rotate k-dependent active matrices from positive radial nodes to a 2D mesh."""

    rotation = AxialRotationSpec() if spec is None else spec
    values = np.asarray(radial_matrices, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("radial_matrices must have shape (n, n, nr)")
    if values.shape[:2] != target_bundle.h0_mev.shape[:2]:
        raise ValueError("radial matrices and target active dimensions do not match")
    k = np.asarray(target_bundle.k_cart_nm_inv, dtype=float)
    radii = np.linalg.norm(k, axis=1)
    unique_radii = np.unique(np.round(radii, 12))
    if unique_radii.size != values.shape[2]:
        raise ValueError("target bundle radial shells do not match radial matrices")
    output = np.empty_like(target_bundle.h0_mev)
    for ik, (kx, ky) in enumerate(k):
        radial_index = int(np.argmin(np.abs(unique_radii - np.round(radii[ik], 12))))
        angle = float(np.arctan2(ky, kx))
        unitary = _unitary_from_jz(
            rotation.active_jz, angle, rotation.exponent_sign
        )
        output[:, :, ik] = (
            unitary @ values[:, :, radial_index] @ unitary.conj().T
        )
    return output


def axial_radial_time_reversal_unitary(
    radial_bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> np.ndarray:
    """Return ``S=U(pi)^dagger T`` for positive-k radial TR covariance."""

    rotation = AxialRotationSpec() if spec is None else spec
    if radial_bundle.time_reversal_unitary is None:
        raise ValueError("radial bundle requires an active time-reversal unitary")
    u_pi = _unitary_from_jz(
        rotation.active_jz, np.pi, rotation.exponent_sign
    )
    sewing = u_pi.conj().T @ np.asarray(
        radial_bundle.time_reversal_unitary, dtype=np.complex128
    )
    involution_error = float(
        np.max(np.abs(sewing @ sewing.conj() - np.eye(sewing.shape[0])))
    )
    if involution_error > 1e-10:
        raise ValueError(
            f"radial time-reversal sewing is not an involution: {involution_error:.3e}"
        )
    return sewing


def project_axial_radial_time_reversal_matrices(
    matrices: np.ndarray,
    radial_bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> np.ndarray:
    """Project positive-k radial matrices onto axial spinful-TR covariance."""

    values = np.asarray(matrices, dtype=np.complex128)
    if values.shape != radial_bundle.h0_mev.shape:
        raise ValueError("radial matrices must match the radial bundle h0 shape")
    sewing = axial_radial_time_reversal_unitary(radial_bundle, spec=spec)
    transformed = np.einsum(
        "ab,bck,dc->adk",
        sewing,
        values.conj(),
        sewing.conj(),
        optimize=True,
    )
    projected = 0.5 * (values + transformed)
    return 0.5 * (projected + np.swapaxes(projected.conj(), 0, 1))


def axial_radial_time_reversal_error(
    matrices: np.ndarray,
    radial_bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> float:
    """Maximum error in ``M=S M* S^dagger`` on positive radial nodes."""

    values = np.asarray(matrices, dtype=np.complex128)
    if values.shape != radial_bundle.h0_mev.shape:
        raise ValueError("radial matrices must match the radial bundle h0 shape")
    sewing = axial_radial_time_reversal_unitary(radial_bundle, spec=spec)
    transformed = np.einsum(
        "ab,bck,dc->adk",
        sewing,
        values.conj(),
        sewing.conj(),
        optimize=True,
    )
    return float(np.max(np.abs(values - transformed)))


def axial_radial_time_reversal_residuals(
    radial_bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> dict[str, float]:
    """Audit radial H0 and Kane8 frames against axial+TR sewing at ``phi=pi``."""

    rotation = AxialRotationSpec() if spec is None else spec
    radial_bundle.validate()
    k = np.asarray(radial_bundle.k_cart_nm_inv, dtype=float)
    if np.any(np.abs(k[:, 1]) > 1e-12) or np.any(k[:, 0] <= 0.0):
        raise ValueError("radial TR audit requires strictly positive kx and ky=0")
    if radial_bundle.time_reversal_unitary is None:
        raise ValueError("radial TR audit requires an active time-reversal unitary")
    if radial_bundle.micro_wavefunctions.shape[2] != 8:
        raise ValueError("radial TR audit requires Kane8 microscopic frames")
    active_tr = np.asarray(
        radial_bundle.time_reversal_unitary, dtype=np.complex128
    )
    micro_tr = kane8_time_reversal_unitary()
    u_pi = _unitary_from_jz(
        rotation.active_jz, np.pi, rotation.exponent_sign
    )
    r_pi = _unitary_from_jz(
        rotation.micro_jz, np.pi, rotation.exponent_sign
    )
    h_errors = []
    frame_errors = []
    for ik in range(radial_bundle.nk):
        h_pi_axial = u_pi @ radial_bundle.h0_mev[:, :, ik] @ u_pi.conj().T
        h_pi_tr = active_tr @ radial_bundle.h0_mev[:, :, ik].conj() @ active_tr.conj().T
        h_errors.append(float(np.max(np.abs(h_pi_axial - h_pi_tr))))
        theta_frame = np.einsum(
            "mn,zna->zma",
            micro_tr,
            radial_bundle.micro_wavefunctions[ik].conj(),
            optimize=True,
        )
        axial_pi_frame = np.einsum(
            "mn,zna,ab->zmb",
            r_pi,
            radial_bundle.micro_wavefunctions[ik],
            u_pi.conj().T @ active_tr,
            optimize=True,
        )
        frame_errors.append(
            float(np.max(np.abs(theta_frame - axial_pi_frame)))
        )
    return {
        "h0_radial_time_reversal_error_mev": max(h_errors, default=0.0),
        "frame_radial_time_reversal_error_nm_minus_half": max(
            frame_errors, default=0.0
        ),
        "h0_radial_projector_error_mev": axial_radial_time_reversal_error(
            radial_bundle.h0_mev, radial_bundle, spec=rotation
        ),
    }


def axial_eh_seed_hamiltonian(
    bundle: Kane4Bundle,
    radial_eh_channel: np.ndarray,
    *,
    amplitude_mev: float,
    spec: AxialRotationSpec | None = None,
) -> np.ndarray:
    """Rotate one radial E1--H1 seed channel over an axial 2D mesh."""

    rotation = AxialRotationSpec() if spec is None else spec
    channel = np.asarray(radial_eh_channel, dtype=np.complex128)
    if channel.shape != (2, 2):
        raise ValueError("radial_eh_channel must be 2x2")
    field = np.zeros_like(bundle.h0_mev, dtype=np.complex128)
    for ik, (kx, ky) in enumerate(np.asarray(bundle.k_cart_nm_inv, dtype=float)):
        angle = float(np.arctan2(ky, kx))
        electron_u = _unitary_from_jz(rotation.active_jz[:2], angle, rotation.exponent_sign)
        hole_u = _unitary_from_jz(rotation.active_jz[2:], angle, rotation.exponent_sign)
        block = float(amplitude_mev) * electron_u @ channel @ hole_u.conj().T
        field[:2, 2:, ik] = block
        field[2:, :2, ik] = block.conj().T
    return field


def project_axial_covariant_matrices(
    matrices: np.ndarray,
    bundle: Kane4Bundle,
    *,
    spec: AxialRotationSpec | None = None,
) -> np.ndarray:
    """Orthogonally project k-local matrices onto axial covariance."""

    rotation = AxialRotationSpec() if spec is None else spec
    values = np.asarray(matrices, dtype=np.complex128)
    if values.shape != bundle.h0_mev.shape:
        raise ValueError("matrices must match bundle h0 shape")
    k = np.asarray(bundle.k_cart_nm_inv, dtype=float)
    radii = np.linalg.norm(k, axis=1)
    rounded = np.round(radii, 12)
    output = np.empty_like(values)
    for radius in np.unique(rounded):
        indices = np.flatnonzero(rounded == radius)
        radial_average = np.zeros(values.shape[:2], dtype=np.complex128)
        unitaries = []
        for ik in indices:
            angle = float(np.arctan2(k[ik, 1], k[ik, 0]))
            unitary = _unitary_from_jz(rotation.active_jz, angle, rotation.exponent_sign)
            unitaries.append(unitary)
            radial_average += unitary.conj().T @ values[:, :, ik] @ unitary
        radial_average /= float(indices.size)
        for ik, unitary in zip(indices, unitaries):
            output[:, :, ik] = unitary @ radial_average @ unitary.conj().T
    return output


def project_time_reversal_matrices(matrices: np.ndarray, bundle: Kane4Bundle) -> np.ndarray:
    """Project k-local matrices onto spinful time-reversal covariance."""

    values = np.asarray(matrices, dtype=np.complex128)
    if values.shape != bundle.h0_mev.shape or bundle.time_reversal_unitary is None:
        raise ValueError("matrices and active time-reversal unitary are required")
    k = np.asarray(bundle.k_cart_nm_inv, dtype=float)
    unitary = np.asarray(bundle.time_reversal_unitary, dtype=np.complex128)
    output = np.empty_like(values)
    for ik, vector in enumerate(k):
        opposite = int(np.argmin(np.linalg.norm(k + vector[None, :], axis=1)))
        if np.linalg.norm(k[opposite] + vector) > 1e-10:
            raise ValueError("mesh lacks exact opposite-k partners")
        transformed = unitary @ values[:, :, opposite].conj() @ unitary.conj().T
        output[:, :, ik] = 0.5 * (values[:, :, ik] + transformed)
    return output


def project_axial_time_reversal_matrices(matrices: np.ndarray, bundle: Kane4Bundle) -> np.ndarray:
    """Project onto the intersection of axial and time-reversal covariance."""

    axial = project_axial_covariant_matrices(matrices, bundle)
    time_reversal = project_time_reversal_matrices(axial, bundle)
    return project_axial_covariant_matrices(time_reversal, bundle)


def matrix_time_reversal_error(matrices: np.ndarray, bundle: Kane4Bundle) -> float:
    """Maximum active-space time-reversal covariance error."""

    values = np.asarray(matrices, dtype=np.complex128)
    if values.shape != bundle.h0_mev.shape or bundle.time_reversal_unitary is None:
        raise ValueError("matrices and active time-reversal unitary are required")
    k = np.asarray(bundle.k_cart_nm_inv, dtype=float)
    unitary = np.asarray(bundle.time_reversal_unitary, dtype=np.complex128)
    errors = []
    for ik, vector in enumerate(k):
        opposite = int(np.argmin(np.linalg.norm(k + vector[None, :], axis=1)))
        expected = unitary @ values[:, :, ik].conj() @ unitary.conj().T
        errors.append(float(np.max(np.abs(values[:, :, opposite] - expected))))
    return max(errors, default=0.0)


def axial_time_reversal_residuals(bundle: Kane4Bundle) -> dict[str, float]:
    """Check h, microscopic frames, and current on exact opposite-k pairs."""

    bundle.validate()
    if bundle.time_reversal_unitary is None or bundle.micro_wavefunctions.shape[2] != 8:
        raise ValueError("time-reversal audit requires Kane8 frames and active U_T")
    k = np.asarray(bundle.k_cart_nm_inv, dtype=float)
    active_tr = np.asarray(bundle.time_reversal_unitary, dtype=np.complex128)
    micro_tr = kane8_time_reversal_unitary()
    h_errors = []
    frame_errors = []
    current_errors = []
    for ik, vector in enumerate(k):
        distances = np.linalg.norm(k + vector[None, :], axis=1)
        opposite = int(np.argmin(distances))
        if distances[opposite] > 1e-10:
            raise ValueError(f"mesh lacks an exact opposite point for index {ik}")
        expected_h = active_tr @ bundle.h0_mev[:, :, ik].conj() @ active_tr.conj().T
        h_errors.append(float(np.max(np.abs(bundle.h0_mev[:, :, opposite] - expected_h))))
        theta_frame = np.einsum(
            "mn,zna->zma",
            micro_tr,
            bundle.micro_wavefunctions[ik].conj(),
            optimize=True,
        )
        expected_frame = bundle.micro_wavefunctions[opposite] @ active_tr
        frame_errors.append(float(np.max(np.abs(theta_frame - expected_frame))))
        if bundle.dhdk_mev_nm is not None:
            for axis in range(2):
                expected_current = -active_tr @ bundle.dhdk_mev_nm[axis, :, :, ik].conj() @ active_tr.conj().T
                current_errors.append(
                    float(np.max(np.abs(bundle.dhdk_mev_nm[axis, :, :, opposite] - expected_current)))
                )
    return {
        "h0_time_reversal_error_mev": max(h_errors, default=0.0),
        "frame_time_reversal_error_nm_minus_half": max(frame_errors, default=0.0),
        "dhdk_time_reversal_error_mev_nm": max(current_errors, default=0.0),
    }
