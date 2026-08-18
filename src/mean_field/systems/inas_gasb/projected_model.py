"""Projected Kane4 wavefunctions, density vertices, and Fock contractions.

All arrays use the ordinary electron-band E1/H1 representation.  The local
projected density vertex is

    Lambda[k,p,z] = Phi[k,z]^dagger Phi[p,z],

and the reference-subtracted exchange self-energy is

    Sigma_F[k] = -sum_p w_p int dz dz' G[k,p,z,z']
                 Lambda[k,p,z] D[p] Lambda[p,k,z'].

The module deliberately does not choose a Coulomb self-cell prescription or a
screening model.  Those are physical inputs and must be supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import E1H1BasisSpec


ComplexArray = np.ndarray
FloatArray = np.ndarray


@dataclass(frozen=True)
class LiftedKaneFrame:
    """Active eigenstates lifted into a fixed microscopic reference frame."""

    micro_wavefunctions: ComplexArray
    h0_mev: ComplexArray
    overlap: ComplexArray
    polar_isometry: ComplexArray
    principal_cos2: FloatArray
    spectrum_error_mev: float

    @property
    def min_completeness(self) -> float:
        return float(np.min(self.principal_cos2))


@dataclass(frozen=True)
class Kane4Bundle:
    """One-source low-energy bundle for matrix EI and optical calculations.

    Shapes
    ------
    ``k_cart_nm_inv``: ``(nk, 2)``
    ``weights_nm2``: ``(nk,)``, implementing ``d^2k/(2*pi)^2``
    ``z_nm``, ``z_weights_nm``: ``(nz,)``
    ``h0_mev``: ``(4, 4, nk)``
    ``micro_wavefunctions``: ``(nk, nz, nmicro, 4)`` in ``nm^-1/2``
    ``dhdk_mev_nm``: optional ``(2, 4, 4, nk)`` current numerator
    ``time_reversal_unitary``: optional ``(4, 4)`` unitary part of ``Theta=U_T K``
    """

    k_cart_nm_inv: FloatArray
    weights_nm2: FloatArray
    z_nm: FloatArray
    z_weights_nm: FloatArray
    h0_mev: ComplexArray
    micro_wavefunctions: ComplexArray
    dhdk_mev_nm: ComplexArray | None = None
    time_reversal_unitary: ComplexArray | None = None
    basis: E1H1BasisSpec = field(default_factory=E1H1BasisSpec)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(np.asarray(self.k_cart_nm_inv).shape[0])

    @property
    def nz(self) -> int:
        return int(np.asarray(self.z_nm).size)

    def validate(self, *, atol: float = 1e-8) -> dict[str, float | None]:
        k = np.asarray(self.k_cart_nm_inv, dtype=float)
        wk = np.asarray(self.weights_nm2, dtype=float)
        z = np.asarray(self.z_nm, dtype=float)
        wz = np.asarray(self.z_weights_nm, dtype=float)
        h0 = np.asarray(self.h0_mev, dtype=np.complex128)
        phi = np.asarray(self.micro_wavefunctions, dtype=np.complex128)
        if k.ndim != 2 or k.shape[1] != 2 or k.shape[0] == 0:
            raise ValueError("k_cart_nm_inv must have shape (nk, 2)")
        nk = k.shape[0]
        if wk.shape != (nk,) or np.any(wk <= 0.0):
            raise ValueError("weights_nm2 must be positive with shape (nk,)")
        if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0):
            raise ValueError("z_nm must be strictly increasing")
        if wz.shape != z.shape or np.any(wz <= 0.0):
            raise ValueError("z_weights_nm must be positive and match z_nm")
        if h0.shape != (self.basis.dimension, self.basis.dimension, nk):
            raise ValueError("h0_mev must have shape (4, 4, nk)")
        if phi.ndim != 4 or phi.shape[:2] != (nk, z.size) or phi.shape[-1] != self.basis.dimension:
            raise ValueError("micro_wavefunctions must have shape (nk, nz, nmicro, 4)")
        if not all(np.all(np.isfinite(a)) for a in (k, wk, z, wz, h0, phi)):
            raise ValueError("Kane4 bundle contains non-finite values")
        h_error = float(np.max(np.abs(h0 - np.swapaxes(h0.conj(), 0, 1))))
        gram = np.einsum("kzma,z,kzmb->kab", phi.conj(), wz, phi, optimize=True)
        ortho_error = float(np.max(np.abs(gram - np.eye(self.basis.dimension)[None, :, :])))
        if h_error > atol:
            raise ValueError(f"h0 is not Hermitian: error={h_error:.3e}")
        if ortho_error > atol:
            raise ValueError(f"micro wavefunctions are not orthonormal: error={ortho_error:.3e}")
        current_error: float | None = None
        if self.dhdk_mev_nm is not None:
            derivative = np.asarray(self.dhdk_mev_nm, dtype=np.complex128)
            if derivative.shape != (2, self.basis.dimension, self.basis.dimension, nk):
                raise ValueError("dhdk_mev_nm must have shape (2, 4, 4, nk)")
            if not np.all(np.isfinite(derivative)):
                raise ValueError("dhdk_mev_nm contains non-finite values")
            current_error = float(np.max(np.abs(derivative - np.swapaxes(derivative.conj(), 1, 2))))
            if current_error > atol:
                raise ValueError(f"dhdk is not Hermitian: error={current_error:.3e}")
        tr_unitarity_error: float | None = None
        tr_square_error: float | None = None
        if self.time_reversal_unitary is not None:
            tr = np.asarray(self.time_reversal_unitary, dtype=np.complex128)
            if tr.shape != (self.basis.dimension, self.basis.dimension):
                raise ValueError("time_reversal_unitary must have shape (4, 4)")
            if not np.all(np.isfinite(tr)):
                raise ValueError("time_reversal_unitary contains non-finite values")
            tr_unitarity_error = float(np.max(np.abs(tr.conj().T @ tr - np.eye(self.basis.dimension))))
            tr_square_error = float(np.max(np.abs(tr @ tr.conj() + np.eye(self.basis.dimension))))
            if tr_unitarity_error > atol or tr_square_error > atol:
                raise ValueError(
                    "invalid spinful time-reversal representation: "
                    f"unitarity={tr_unitarity_error:.3e}, square={tr_square_error:.3e}"
                )
        return {
            "h0_hermiticity_error_mev": h_error,
            "wavefunction_orthonormality_error": ortho_error,
            "dhdk_hermiticity_error_mev_nm": current_error,
            "time_reversal_unitarity_error": tr_unitarity_error,
            "time_reversal_square_error": tr_square_error,
        }

    def fingerprint(self) -> str:
        """Content hash used to prevent mixing Hamiltonian and vertex sources."""

        digest = hashlib.sha256()
        canonical_arrays = (
            np.asarray(self.k_cart_nm_inv, dtype=np.float64),
            np.asarray(self.weights_nm2, dtype=np.float64),
            np.asarray(self.z_nm, dtype=np.float64),
            np.asarray(self.z_weights_nm, dtype=np.float64),
            np.asarray(self.h0_mev, dtype=np.complex128),
            np.asarray(self.micro_wavefunctions, dtype=np.complex128),
            None if self.dhdk_mev_nm is None else np.asarray(self.dhdk_mev_nm, dtype=np.complex128),
            None if self.time_reversal_unitary is None else np.asarray(self.time_reversal_unitary, dtype=np.complex128),
        )
        for array in canonical_arrays:
            if array is None:
                digest.update(b"none")
                continue
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.view(np.uint8))
        digest.update(json.dumps(self.provenance, sort_keys=True, default=str).encode())
        digest.update(repr(self.basis).encode())
        return digest.hexdigest()


def remove_normal_e1_h1_hybridization(bundle: Kane4Bundle) -> Kane4Bundle:
    """Return the declared diabatic BCS reduction with bare E--H mixing removed.

    This transform preserves the E1 and H1 diagonal blocks, microscopic frames,
    quadrature, and all source coordinates.  It removes only the normal-state
    E1--H1 blocks of ``h0`` (and ``dhdk`` when present).  The result is a typed
    two-sector BCS diagnostic, not the full projected Kane Hamiltonian.
    """

    bundle.validate()
    if "normal_e1_h1_hybridization_removal" in bundle.provenance:
        raise ValueError("normal E1-H1 hybridization was already removed")
    h0 = np.asarray(bundle.h0_mev, dtype=np.complex128).copy()
    removed_hybridization = h0[:2, 2:].copy()
    h0[:2, 2:] = 0.0
    h0[2:, :2] = 0.0
    derivative = None
    removed_derivative_maximum = None
    if bundle.dhdk_mev_nm is not None:
        derivative = np.asarray(bundle.dhdk_mev_nm, dtype=np.complex128).copy()
        removed_derivative_maximum = float(
            np.max(np.abs(derivative[:, :2, 2:]))
        )
        derivative[:, :2, 2:] = 0.0
        derivative[:, 2:, :2] = 0.0
    provenance = {
        **bundle.provenance,
        "normal_e1_h1_hybridization_removal": {
            "policy": "zero E1-H1 blocks in the fixed diabatic basis",
            "parent_bundle_fingerprint": bundle.fingerprint(),
            "maximum_removed_h0_block_mev": float(
                np.max(np.abs(removed_hybridization))
            ),
            "maximum_removed_dhdk_block_mev_nm": removed_derivative_maximum,
            "scope": (
                "normal-hybridization-removed BCS diagnostic; not the full "
                "projected Kane Hamiltonian"
            ),
        },
    }
    reduced = replace(
        bundle,
        h0_mev=h0,
        dhdk_mev_nm=derivative,
        provenance=provenance,
    )
    reduced.validate()
    return reduced


def save_kane4_bundle(
    bundle: Kane4Bundle,
    npz_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Save a validated bundle without Python pickles and return metadata."""

    diagnostics = bundle.validate()
    destination = Path(npz_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    derivative = (
        np.empty((0,), dtype=np.complex128)
        if bundle.dhdk_mev_nm is None
        else np.asarray(bundle.dhdk_mev_nm, dtype=np.complex128)
    )
    time_reversal = (
        np.empty((0,), dtype=np.complex128)
        if bundle.time_reversal_unitary is None
        else np.asarray(bundle.time_reversal_unitary, dtype=np.complex128)
    )
    np.savez_compressed(
        destination,
        k_cart_nm_inv=np.asarray(bundle.k_cart_nm_inv, dtype=float),
        weights_nm2=np.asarray(bundle.weights_nm2, dtype=float),
        z_nm=np.asarray(bundle.z_nm, dtype=float),
        z_weights_nm=np.asarray(bundle.z_weights_nm, dtype=float),
        h0_mev=np.asarray(bundle.h0_mev, dtype=np.complex128),
        micro_wavefunctions=np.asarray(bundle.micro_wavefunctions, dtype=np.complex128),
        dhdk_mev_nm=derivative,
        time_reversal_unitary=time_reversal,
        labels=np.asarray(bundle.basis.labels),
        electron_indices=np.asarray(bundle.basis.electron_indices, dtype=int),
        hole_indices=np.asarray(bundle.basis.hole_indices, dtype=int),
        provenance_json=np.asarray(json.dumps(bundle.provenance, sort_keys=True, default=str)),
    )
    metadata: dict[str, Any] = {
        "classification": "Kane4Bundle",
        "schema_version": 1,
        "bundle_fingerprint": bundle.fingerprint(),
        "npz_path": str(destination),
        "shapes": {
            "k_cart_nm_inv": list(np.asarray(bundle.k_cart_nm_inv).shape),
            "h0_mev": list(np.asarray(bundle.h0_mev).shape),
            "micro_wavefunctions": list(np.asarray(bundle.micro_wavefunctions).shape),
            "dhdk_mev_nm": None if bundle.dhdk_mev_nm is None else list(np.asarray(bundle.dhdk_mev_nm).shape),
            "time_reversal_unitary": None if bundle.time_reversal_unitary is None else list(np.asarray(bundle.time_reversal_unitary).shape),
        },
        "basis": {
            "labels": list(bundle.basis.labels),
            "electron_indices": list(bundle.basis.electron_indices),
            "hole_indices": list(bundle.basis.hole_indices),
        },
        "diagnostics": diagnostics,
        "provenance": bundle.provenance,
    }
    if metadata_path is not None:
        meta_destination = Path(metadata_path)
        meta_destination.parent.mkdir(parents=True, exist_ok=True)
        meta_destination.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return metadata


def load_kane4_bundle(npz_path: str | Path) -> Kane4Bundle:
    """Load and validate a bundle written by :func:`save_kane4_bundle`."""

    with np.load(Path(npz_path), allow_pickle=False) as data:
        derivative = np.asarray(data["dhdk_mev_nm"], dtype=np.complex128)
        time_reversal = np.asarray(data["time_reversal_unitary"], dtype=np.complex128)
        basis = E1H1BasisSpec(
            labels=tuple(str(value) for value in data["labels"].tolist()),
            electron_indices=tuple(int(value) for value in data["electron_indices"].tolist()),
            hole_indices=tuple(int(value) for value in data["hole_indices"].tolist()),
        )
        bundle = Kane4Bundle(
            k_cart_nm_inv=np.asarray(data["k_cart_nm_inv"], dtype=float),
            weights_nm2=np.asarray(data["weights_nm2"], dtype=float),
            z_nm=np.asarray(data["z_nm"], dtype=float),
            z_weights_nm=np.asarray(data["z_weights_nm"], dtype=float),
            h0_mev=np.asarray(data["h0_mev"], dtype=np.complex128),
            micro_wavefunctions=np.asarray(data["micro_wavefunctions"], dtype=np.complex128),
            dhdk_mev_nm=None if derivative.size == 0 else derivative,
            time_reversal_unitary=None if time_reversal.size == 0 else time_reversal,
            basis=basis,
            provenance=json.loads(str(data["provenance_json"].item())),
        )
    bundle.validate()
    return bundle


def lift_active_frame_to_reference(
    reference_wavefunctions: ComplexArray,
    active_wavefunctions: ComplexArray,
    active_energies_mev: FloatArray,
    z_weights_nm: FloatArray,
    *,
    rank_tol: float = 1e-10,
    orthonormality_tol: float = 1e-8,
) -> LiftedKaneFrame:
    """Polar/Löwdin lift of one microscopic active quartet to a fixed frame."""

    reference = np.asarray(reference_wavefunctions, dtype=np.complex128)
    active = np.asarray(active_wavefunctions, dtype=np.complex128)
    energies = np.asarray(active_energies_mev, dtype=float)
    wz = np.asarray(z_weights_nm, dtype=float)
    if reference.ndim != 3 or active.shape != reference.shape:
        raise ValueError("reference and active wavefunctions must have shape (nz, nmicro, nactive)")
    nz, _nmicro, n = reference.shape
    if energies.shape != (n,) or wz.shape != (nz,):
        raise ValueError("energies or z weights do not match the wavefunction frame")
    reference_gram = np.einsum("zma,z,zmb->ab", reference.conj(), wz, reference, optimize=True)
    active_gram = np.einsum("zma,z,zmb->ab", active.conj(), wz, active, optimize=True)
    if float(np.max(np.abs(reference_gram - np.eye(n)))) > orthonormality_tol:
        raise ValueError("reference microscopic frame is not orthonormal")
    if float(np.max(np.abs(active_gram - np.eye(n)))) > orthonormality_tol:
        raise ValueError("active microscopic frame is not orthonormal")
    overlap = np.einsum("zma,z,zmb->ab", reference.conj(), wz, active, optimize=True)
    gram_raw = overlap.conj().T @ overlap
    gram = 0.5 * (gram_raw + gram_raw.conj().T)
    cos2, vectors = np.linalg.eigh(gram)
    if float(np.min(cos2)) <= rank_tol:
        raise ValueError(
            "active/reference overlap is rank deficient: "
            f"minimum principal cos^2={float(np.min(cos2)):.3e}"
        )
    inverse_sqrt = (vectors * (1.0 / np.sqrt(cos2))[None, :]) @ vectors.conj().T
    isometry = overlap @ inverse_sqrt
    lifted = np.einsum("zma,ba->zmb", active, isometry.conj(), optimize=True)
    h0 = (isometry * energies[None, :]) @ isometry.conj().T
    spectrum_error = float(np.max(np.abs(np.linalg.eigvalsh(h0) - np.sort(energies))))
    return LiftedKaneFrame(
        micro_wavefunctions=lifted,
        h0_mev=h0,
        overlap=overlap,
        polar_isometry=isometry,
        principal_cos2=np.asarray(cos2, dtype=float),
        spectrum_error_mev=spectrum_error,
    )


def local_density_vertices(micro_wavefunctions: ComplexArray) -> ComplexArray:
    """Return ``Lambda[k,p,z,a,b] = Phi[k,z]^dagger Phi[p,z]``."""

    phi = np.asarray(micro_wavefunctions, dtype=np.complex128)
    if phi.ndim != 4 or phi.shape[-1] == 0:
        raise ValueError("micro_wavefunctions must have shape (nk, nz, nmicro, nactive)")
    return np.einsum("kzma,pzmb->kpzab", phi.conj(), phi, optimize=True)


def integrated_density_vertices(local_vertices: ComplexArray, z_weights_nm: FloatArray) -> ComplexArray:
    """Integrate local density vertices over z."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    wz = np.asarray(z_weights_nm, dtype=float)
    if local.ndim != 5 or wz.shape != (local.shape[2],):
        raise ValueError("local vertices and z weights have incompatible shapes")
    return np.einsum("kpzab,z->kpab", local, wz, optimize=True)


def density_vertex_reciprocity_error(local_vertices: ComplexArray) -> float:
    """Return max error in ``Lambda[k,p]^dagger = Lambda[p,k]``."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    if local.ndim != 5 or local.shape[0] != local.shape[1] or local.shape[3] != local.shape[4]:
        raise ValueError("local_vertices must have shape (nk, nk, nz, n, n)")
    reverse = np.swapaxes(np.swapaxes(local.conj(), 0, 1), 3, 4)
    return float(np.max(np.abs(local - reverse)))


def uniform_dielectric_green_mev_nm2(
    q_effective_nm_inv: FloatArray,
    z_nm: FloatArray,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
) -> FloatArray:
    """Uniform-dielectric mixed-representation Coulomb Green function.

    ``q_effective_nm_inv`` must be strictly positive.  A production self-cell
    average is a mesh property and must be supplied by the caller rather than
    hidden behind an arbitrary q floor.
    """

    q = np.asarray(q_effective_nm_inv, dtype=float)
    z = np.asarray(z_nm, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or np.any(q <= 0.0):
        raise ValueError("q_effective_nm_inv must be a positive square matrix")
    q_symmetry_error = float(np.max(np.abs(q - q.T)))
    if q_symmetry_error > 1e-12:
        raise ValueError(f"q_effective_nm_inv must be symmetric: error={q_symmetry_error:.3e}")
    if z.ndim != 1 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z_nm must be strictly increasing")
    if epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive")
    distance = np.abs(z[:, None] - z[None, :])
    prefactor = 2.0 * np.pi * float(coulomb_mev_nm) / float(epsilon_r)
    return prefactor / q[:, :, None, None] * np.exp(-q[:, :, None, None] * distance)


def uniform_dielectric_green_on_mesh_mev_nm2(
    k_cart_nm_inv: FloatArray,
    k_weights_nm2: FloatArray,
    z_nm: FloatArray,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
) -> FloatArray:
    """Coulomb Green function with a mesh-derived circular self-cell average.

    Off-diagonal momentum pairs use the exact uniform-dielectric kernel.  For
    ``k=p``, each quadrature cell of physical k-space area
    ``A_k=(2*pi)^2*w_k`` is replaced by an equal-area disk of radius
    ``q_c=sqrt(A_k/pi)`` and ``exp(-q*|z-z'|)/q`` is averaged analytically.
    This is a convergent cell quadrature, not an arbitrary q floor.
    """

    k = np.asarray(k_cart_nm_inv, dtype=float)
    wk = np.asarray(k_weights_nm2, dtype=float)
    z = np.asarray(z_nm, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or wk.shape != (k.shape[0],):
        raise ValueError("k mesh must have shape (nk, 2) with matching weights")
    if not all(np.all(np.isfinite(array)) for array in (k, wk, z)):
        raise ValueError("Coulomb mesh inputs must be finite")
    if np.any(wk <= 0.0) or epsilon_r <= 0.0:
        raise ValueError("Coulomb mesh weights and epsilon_r must be positive")
    difference = k[:, None, :] - k[None, :, :]
    q = np.linalg.norm(difference, axis=2)
    distance = np.abs(z[:, None] - z[None, :])
    prefactor = 2.0 * np.pi * float(coulomb_mev_nm) / float(epsilon_r)
    nk = k.shape[0]
    green = np.empty((nk, nk, z.size, z.size), dtype=float)
    off_diagonal = ~np.eye(nk, dtype=bool)
    q_off = q[off_diagonal]
    green[off_diagonal] = (
        prefactor
        / q_off[:, None, None]
        * np.exp(-q_off[:, None, None] * distance[None, :, :])
    )
    for ik in range(nk):
        cell_area = (2.0 * np.pi) ** 2 * wk[ik]
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
        green[ik, ik] = prefactor * averaged
    return green


def projected_fock_self_energy(
    density_delta: ComplexArray,
    local_vertices: ComplexArray,
    green_mev_nm2: FloatArray,
    k_weights_nm2: FloatArray,
    z_weights_nm: FloatArray,
) -> ComplexArray:
    """Apply the reference-subtracted projected Fock operator."""

    density = np.asarray(density_delta, dtype=np.complex128)
    local = np.asarray(local_vertices, dtype=np.complex128)
    green = np.asarray(green_mev_nm2, dtype=float)
    wk = np.asarray(k_weights_nm2, dtype=float)
    wz = np.asarray(z_weights_nm, dtype=float)
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError("density_delta must have shape (n, n, nk)")
    if not np.all(np.isfinite(density)):
        raise ValueError("density_delta contains non-finite values")
    density_error = float(np.max(np.abs(density - np.swapaxes(density.conj(), 0, 1))))
    if density_error > 1e-9:
        raise ValueError(f"density_delta must be Hermitian: error={density_error:.3e}")
    n, _, nk = density.shape
    if local.shape[:2] != (nk, nk) or local.shape[3:] != (n, n):
        raise ValueError("local_vertices have incompatible active/k dimensions")
    nz = local.shape[2]
    if green.shape != (nk, nk, nz, nz):
        raise ValueError("green_mev_nm2 must have shape (nk, nk, nz, nz)")
    if not np.all(np.isfinite(green)):
        raise ValueError("green_mev_nm2 contains non-finite values")
    vertex_error = density_vertex_reciprocity_error(local)
    if vertex_error > 1e-9:
        raise ValueError(f"density vertices violate reciprocity: error={vertex_error:.3e}")
    green_reverse = np.swapaxes(np.swapaxes(green, 0, 1), 2, 3)
    green_error = float(np.max(np.abs(green - green_reverse)))
    if green_error > 1e-9:
        raise ValueError(f"Coulomb Green function violates reciprocity: error={green_error:.3e}")
    if wk.shape != (nk,) or wz.shape != (nz,):
        raise ValueError("quadrature weights have incompatible shapes")
    if not all(np.all(np.isfinite(array)) for array in (local, wk, wz)):
        raise ValueError("vertices and quadrature weights must be finite")
    if np.any(wk <= 0.0) or np.any(wz <= 0.0):
        raise ValueError("quadrature weights must be positive")
    sigma = np.zeros_like(density)
    for ik in range(nk):
        for ip in range(nk):
            term = np.einsum(
                "xab,bc,ycd,xy,x,y->ad",
                local[ik, ip],
                density[:, :, ip],
                local[ip, ik],
                green[ik, ip],
                wz,
                wz,
                optimize=True,
            )
            sigma[:, :, ik] -= wk[ip] * term
    if not np.all(np.isfinite(sigma)):
        raise ValueError("projected Fock self-energy contains non-finite values")
    sigma_error = float(np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))))
    if sigma_error > 1e-8:
        raise ValueError(f"projected Fock self-energy is not Hermitian: error={sigma_error:.3e}")
    return sigma


def hermitian_density_from_e1h1_coherence(
    coherence_eh: ComplexArray,
    basis: E1H1BasisSpec | None = None,
) -> ComplexArray:
    """Embed ``X=D_EH`` and ``D_HE=X^dagger`` into a Hermitian density field."""

    spec = E1H1BasisSpec() if basis is None else basis
    coherence = np.asarray(coherence_eh, dtype=np.complex128)
    ne = len(spec.electron_indices)
    nh = len(spec.hole_indices)
    if coherence.ndim != 3 or coherence.shape[:2] != (ne, nh):
        raise ValueError("coherence_eh must have shape (n_e, n_h, nk)")
    if not np.all(np.isfinite(coherence)):
        raise ValueError("coherence_eh contains non-finite values")
    nk = coherence.shape[2]
    density = np.zeros((spec.dimension, spec.dimension, nk), dtype=np.complex128)
    density[np.ix_(spec.electron_indices, spec.hole_indices, range(nk))] = coherence
    density[np.ix_(spec.hole_indices, spec.electron_indices, range(nk))] = (
        np.swapaxes(coherence.conj(), 0, 1)
    )
    return density


def e1h1_coherence_exchange_tensors_mev_nm2(
    exchange_tensor_mev_nm2: ComplexArray,
    basis: E1H1BasisSpec | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Restrict a full exchange tensor to the Hermitian E1--H1 coherence sector.

    For ``X[p]=D_EH[p]``, the E1--H1 Fock block is

    ``Sigma_EH[k] = -sum_p w[p] (A[k,p] X[p] + B[k,p] X[p]^*)``.

    ``A`` comes from the directed ``D_EH`` block.  ``B`` comes from its
    Hermitian partner ``D_HE=X^dagger`` and is therefore an antilinear term;
    dropping it is not generally allowed for complex microscopic vertices.
    """

    spec = E1H1BasisSpec() if basis is None else basis
    tensor = np.asarray(exchange_tensor_mev_nm2, dtype=np.complex128)
    if tensor.ndim != 6 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("exchange tensor must have shape (nk,nk,n,n,n,n)")
    nk = tensor.shape[0]
    n = spec.dimension
    if tensor.shape != (nk, nk, n, n, n, n):
        raise ValueError("exchange tensor active dimension does not match E1/H1 basis")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("exchange tensor contains non-finite values")
    e = tuple(spec.electron_indices)
    h = tuple(spec.hole_indices)
    radial = tuple(range(nk))
    direct = tensor[np.ix_(radial, radial, e, h, e, h)]
    he_input = tensor[np.ix_(radial, radial, e, h, h, e)]
    conjugate = np.swapaxes(he_input, 4, 5)
    return np.asarray(direct, dtype=np.complex128), np.asarray(
        conjugate, dtype=np.complex128
    )


def apply_e1h1_coherence_exchange_superoperator(
    coherence_eh: ComplexArray,
    direct_tensor_mev_nm2: ComplexArray,
    conjugate_tensor_mev_nm2: ComplexArray,
    k_weights_nm2: FloatArray,
) -> ComplexArray:
    """Apply the real-linear microscopic E1--H1 Fock superoperator."""

    coherence = np.asarray(coherence_eh, dtype=np.complex128)
    direct = np.asarray(direct_tensor_mev_nm2, dtype=np.complex128)
    conjugate = np.asarray(conjugate_tensor_mev_nm2, dtype=np.complex128)
    weights = np.asarray(k_weights_nm2, dtype=float)
    if coherence.ndim != 3:
        raise ValueError("coherence_eh must have shape (n_e,n_h,nk)")
    ne, nh, nk = coherence.shape
    expected = (nk, nk, ne, nh, ne, nh)
    if direct.shape != expected or conjugate.shape != expected:
        raise ValueError("coherence exchange tensors have incompatible shapes")
    if weights.shape != (nk,) or np.any(weights <= 0.0):
        raise ValueError("coherence k weights must be positive with shape (nk,)")
    if not all(np.all(np.isfinite(value)) for value in (coherence, direct, conjugate, weights)):
        raise ValueError("coherence superoperator inputs must be finite")
    output = -np.einsum(
        "ijadbc,bcj,j->adi", direct, coherence, weights, optimize=True
    )
    output -= np.einsum(
        "ijadbc,bcj,j->adi", conjugate, coherence.conj(), weights, optimize=True
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("E1-H1 coherence Fock output is non-finite")
    return output


def precompute_projected_exchange_tensor_mev_nm2(
    local_vertices: ComplexArray,
    green_mev_nm2: FloatArray,
    z_weights_nm: FloatArray,
) -> ComplexArray:
    """Precontract z coordinates into ``K[k,p,a,d,b,c]``."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    green = np.asarray(green_mev_nm2, dtype=float)
    wz = np.asarray(z_weights_nm, dtype=float)
    if local.ndim != 5 or local.shape[0] != local.shape[1] or local.shape[3] != local.shape[4]:
        raise ValueError("local_vertices must have shape (nk, nk, nz, n, n)")
    nk, _, nz, n, _ = local.shape
    if green.shape != (nk, nk, nz, nz) or wz.shape != (nz,):
        raise ValueError("Green function or z weights do not match local vertices")
    if not all(np.all(np.isfinite(array)) for array in (local, green, wz)):
        raise ValueError("exchange-tensor inputs must be finite")
    tensor = np.empty((nk, nk, n, n, n, n), dtype=np.complex128)
    for ik in range(nk):
        for ip in range(nk):
            tensor[ik, ip] = np.einsum(
                "xab,ycd,xy,x,y->adbc",
                local[ik, ip],
                local[ip, ik],
                green[ik, ip],
                wz,
                wz,
                optimize=True,
            )
    return tensor


@dataclass(frozen=True)
class ProjectedFockOperator:
    """Validated linear, self-adjoint, reference-subtracted Fock operator."""

    local_vertices: ComplexArray
    green_mev_nm2: FloatArray
    k_weights_nm2: FloatArray
    z_weights_nm: FloatArray
    bundle_fingerprint: str
    self_cell_description: str
    exchange_tensor_mev_nm2: ComplexArray | None = None
    electrostatics_fingerprint: str | None = None
    electrostatics_token: object | None = None
    _action_arrays_owned: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.bundle_fingerprint:
            raise ValueError("bundle_fingerprint is required")
        if not self.self_cell_description:
            raise ValueError("self_cell_description is required")
        if self.electrostatics_fingerprint is not None and (
            len(self.electrostatics_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.electrostatics_fingerprint)
        ):
            raise ValueError("electrostatics_fingerprint must be a lowercase SHA-256 hex digest")
        if self.electrostatics_token is not None and self.electrostatics_fingerprint is None:
            raise ValueError("an electrostatics token requires an electrostatics fingerprint")
        local_input = np.asarray(self.local_vertices, dtype=np.complex128)
        green_input = np.asarray(self.green_mev_nm2, dtype=float)
        if self._action_arrays_owned and local_input.flags.owndata:
            local = local_input
        else:
            local = np.array(local_input, dtype=np.complex128, copy=True)
        if self._action_arrays_owned and green_input.flags.owndata:
            green = green_input
        else:
            green = np.array(green_input, dtype=float, copy=True)
        wk = np.array(self.k_weights_nm2, dtype=float, copy=True)
        wz = np.array(self.z_weights_nm, dtype=float, copy=True)
        for name, array in (
            ("local_vertices", local),
            ("green_mev_nm2", green),
            ("k_weights_nm2", wk),
            ("z_weights_nm", wz),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if local.ndim != 5 or local.shape[0] != local.shape[1] or local.shape[3] != local.shape[4]:
            raise ValueError("local_vertices must have shape (nk, nk, nz, n, n)")
        nk, _, nz, _n, _ = local.shape
        if green.shape != (nk, nk, nz, nz) or wk.shape != (nk,) or wz.shape != (nz,):
            raise ValueError("Fock operator arrays have incompatible shapes")
        if not all(np.all(np.isfinite(array)) for array in (local, green, wk, wz)):
            raise ValueError("Fock operator arrays must be finite")
        if np.any(wk <= 0.0) or np.any(wz <= 0.0):
            raise ValueError("Fock quadrature weights must be positive")
        if density_vertex_reciprocity_error(local) > 1e-9:
            raise ValueError("Fock density vertices violate reciprocity")
        if self.exchange_tensor_mev_nm2 is not None:
            tensor_input = np.asarray(self.exchange_tensor_mev_nm2, dtype=np.complex128)
            if self._action_arrays_owned and tensor_input.flags.owndata:
                tensor = tensor_input
            else:
                tensor = np.array(tensor_input, dtype=np.complex128, copy=True)
            if tensor.shape != (nk, nk, _n, _n, _n, _n):
                raise ValueError("exchange_tensor_mev_nm2 has an incompatible shape")
            if not np.all(np.isfinite(tensor)):
                raise ValueError("exchange_tensor_mev_nm2 must be finite")
            tensor.setflags(write=False)
            object.__setattr__(self, "exchange_tensor_mev_nm2", tensor)
        reverse = np.swapaxes(np.swapaxes(green, 0, 1), 2, 3)
        if not np.all(np.isfinite(green)) or float(np.max(np.abs(green - reverse))) > 1e-9:
            raise ValueError("Fock Green function must be finite and reciprocal")

    @classmethod
    def from_bundle(
        cls,
        bundle: Kane4Bundle,
        green_mev_nm2: FloatArray,
        *,
        self_cell_description: str,
        precompute_exchange_tensor: bool = False,
        electrostatics_fingerprint: str | None = None,
        _electrostatics_token: object | None = None,
        _take_green_ownership: bool = False,
    ) -> "ProjectedFockOperator":
        """Construct a source-certified operator from one Kane4 bundle."""

        bundle.validate()
        local = local_density_vertices(bundle.micro_wavefunctions)
        green_input = np.asarray(green_mev_nm2, dtype=float)
        green = (
            green_input
            if _take_green_ownership and green_input.flags.owndata
            else np.array(green_input, dtype=float, copy=True)
        )
        tensor = (
            precompute_projected_exchange_tensor_mev_nm2(local, green, bundle.z_weights_nm)
            if precompute_exchange_tensor
            else None
        )
        return cls(
            local_vertices=local,
            green_mev_nm2=green,
            k_weights_nm2=np.asarray(bundle.weights_nm2, dtype=float),
            z_weights_nm=np.asarray(bundle.z_weights_nm, dtype=float),
            bundle_fingerprint=bundle.fingerprint(),
            self_cell_description=self_cell_description,
            exchange_tensor_mev_nm2=tensor,
            electrostatics_fingerprint=electrostatics_fingerprint,
            electrostatics_token=_electrostatics_token,
            _action_arrays_owned=True,
        )

    def validate_against_bundle(self, bundle: Kane4Bundle, *, atol: float = 1e-10) -> None:
        """Reject forged labels or vertices/weights from a different source."""

        bundle.validate()
        if self.bundle_fingerprint != bundle.fingerprint():
            raise ValueError("Fock operator and Kane4 bundle fingerprints do not match")
        if not np.allclose(self.k_weights_nm2, bundle.weights_nm2, rtol=0.0, atol=atol):
            raise ValueError("Fock k weights do not come from the supplied bundle")
        if not np.allclose(self.z_weights_nm, bundle.z_weights_nm, rtol=0.0, atol=atol):
            raise ValueError("Fock z weights do not come from the supplied bundle")
        expected = local_density_vertices(bundle.micro_wavefunctions)
        if expected.shape != np.asarray(self.local_vertices).shape or not np.allclose(
            self.local_vertices,
            expected,
            rtol=1e-10,
            atol=atol,
        ):
            raise ValueError("Fock density vertices do not come from the supplied bundle")
        if self.exchange_tensor_mev_nm2 is not None:
            expected_tensor = precompute_projected_exchange_tensor_mev_nm2(
                expected,
                self.green_mev_nm2,
                self.z_weights_nm,
            )
            if not np.allclose(
                self.exchange_tensor_mev_nm2,
                expected_tensor,
                rtol=1e-10,
                atol=atol,
            ):
                raise ValueError("compressed exchange tensor does not match bundle vertices and Green function")

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        if self.exchange_tensor_mev_nm2 is None:
            return projected_fock_self_energy(
                density_delta,
                self.local_vertices,
                self.green_mev_nm2,
                self.k_weights_nm2,
                self.z_weights_nm,
            )
        density = np.asarray(density_delta, dtype=np.complex128)
        if density.shape != (self.local_vertices.shape[3], self.local_vertices.shape[4], self.local_vertices.shape[0]):
            raise ValueError("density_delta has an incompatible shape")
        if not np.all(np.isfinite(density)):
            raise ValueError("density_delta contains non-finite values")
        sigma = -np.einsum(
            "kpadbc,bcp,p->adk",
            self.exchange_tensor_mev_nm2,
            density,
            self.k_weights_nm2,
            optimize=True,
        )
        error = float(np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))))
        if not np.all(np.isfinite(sigma)) or error > 1e-8:
            raise ValueError(f"compressed Fock action is invalid: Hermiticity error={error:.3e}")
        return sigma

    def energy_density_mev_nm2(self, density_delta: ComplexArray) -> float:
        sigma = self(density_delta)
        return fock_energy_density_mev_nm2(density_delta, sigma, self.k_weights_nm2)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.bundle_fingerprint.encode())
        digest.update(self.self_cell_description.encode())
        digest.update((self.electrostatics_fingerprint or "unspecified").encode())
        for array in (
            np.asarray(self.local_vertices, dtype=np.complex128),
            np.asarray(self.green_mev_nm2, dtype=np.float64),
            np.asarray(self.k_weights_nm2, dtype=np.float64),
            np.asarray(self.z_weights_nm, dtype=np.float64),
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def action_storage_fingerprint(self) -> str:
        """Hash the concrete direct/compressed arrays used by ``__call__``."""

        digest = hashlib.sha256()
        digest.update(self.fingerprint().encode())
        if self.exchange_tensor_mev_nm2 is None:
            digest.update(b"direct-fock-action")
        else:
            digest.update(b"compressed-fock-action")
            tensor = np.ascontiguousarray(
                self.exchange_tensor_mev_nm2, dtype=np.complex128
            )
            digest.update(str(tensor.shape).encode())
            digest.update(tensor.view(np.uint8))
        return digest.hexdigest()


def fock_energy_density_mev_nm2(
    density_delta: ComplexArray,
    sigma_fock_mev: ComplexArray,
    k_weights_nm2: FloatArray,
) -> float:
    """Return ``1/2 int_k Tr[Sigma_F(k) D(k)]``."""

    density = np.asarray(density_delta, dtype=np.complex128)
    sigma = np.asarray(sigma_fock_mev, dtype=np.complex128)
    wk = np.asarray(k_weights_nm2, dtype=float)
    if density.shape != sigma.shape or density.ndim != 3 or wk.shape != (density.shape[2],):
        raise ValueError("density, self-energy, and weights have incompatible shapes")
    value = 0.5 * np.einsum("k,abk,bak->", wk, sigma, density, optimize=True)
    if abs(value.imag) > 1e-8 * max(1.0, abs(value.real)):
        raise ValueError(f"Fock energy is not real: {value}")
    return float(value.real)


def gauge_transform_local_vertices(local_vertices: ComplexArray, gauge: ComplexArray) -> ComplexArray:
    """Transform vertices under independent active-basis rotations at each k."""

    local = np.asarray(local_vertices, dtype=np.complex128)
    rotations = np.asarray(gauge, dtype=np.complex128)
    nk, nk2, _nz, n, n2 = local.shape
    if nk != nk2 or n != n2 or rotations.shape != (nk, n, n):
        raise ValueError("vertex and gauge shapes are incompatible")
    return np.einsum("kai,kpzab,pbj->kpzij", rotations.conj(), local, rotations, optimize=True)


def gauge_transform_k_matrices(matrices: ComplexArray, gauge: ComplexArray) -> ComplexArray:
    """Transform ``M(k) -> G(k)^dagger M(k) G(k)``."""

    values = np.asarray(matrices, dtype=np.complex128)
    rotations = np.asarray(gauge, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("matrices must have shape (n, n, nk)")
    n, _, nk = values.shape
    if rotations.shape != (nk, n, n):
        raise ValueError("gauge must have shape (nk, n, n)")
    return np.einsum("kai,abk,kbj->ijk", rotations.conj(), values, rotations, optimize=True)


def excitonic_fock_singular_values(
    sigma_fock_mev: ComplexArray,
    basis: E1H1BasisSpec | None = None,
) -> FloatArray:
    """Gauge-invariant singular values of the interaction-induced E1--H1 block."""

    spec = E1H1BasisSpec() if basis is None else basis
    sigma = np.asarray(sigma_fock_mev, dtype=np.complex128)
    if sigma.shape[:2] != (spec.dimension, spec.dimension) or sigma.ndim != 3:
        raise ValueError("sigma_fock_mev must have shape (4, 4, nk)")
    values = []
    for ik in range(sigma.shape[2]):
        block = sigma[np.ix_(spec.electron_indices, spec.hole_indices, [ik])][:, :, 0]
        values.append(np.linalg.svd(block, compute_uv=False))
    return np.asarray(values, dtype=float)
