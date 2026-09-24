"""Source-bound Kane4 bundle contracts, storage, and frame lifting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.core.io import (
    file_object_sha256,
    open_stable_binary_file,
    publish_staged_file_noreplace,
    unique_staging_path,
    write_json_artifact,
    write_npz_artifact,
)

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
    arrays = {
        "k_cart_nm_inv": np.asarray(bundle.k_cart_nm_inv, dtype=float),
        "weights_nm2": np.asarray(bundle.weights_nm2, dtype=float),
        "z_nm": np.asarray(bundle.z_nm, dtype=float),
        "z_weights_nm": np.asarray(bundle.z_weights_nm, dtype=float),
        "h0_mev": np.asarray(bundle.h0_mev, dtype=np.complex128),
        "micro_wavefunctions": np.asarray(
            bundle.micro_wavefunctions, dtype=np.complex128
        ),
        "dhdk_mev_nm": derivative,
        "time_reversal_unitary": time_reversal,
        "labels": np.asarray(bundle.basis.labels),
        "electron_indices": np.asarray(bundle.basis.electron_indices, dtype=int),
        "hole_indices": np.asarray(bundle.basis.hole_indices, dtype=int),
        "provenance_json": np.asarray(
            json.dumps(bundle.provenance, sort_keys=True, allow_nan=False)
        ),
    }
    meta_destination = None if metadata_path is None else Path(metadata_path)
    if meta_destination is not None:
        if meta_destination.resolve() == destination.resolve():
            raise ValueError("Kane4Bundle NPZ and metadata paths must be distinct")
        if meta_destination.parent.resolve() != destination.parent.resolve():
            raise ValueError("Kane4Bundle NPZ and metadata must share one directory")
    if destination.exists() or (
        meta_destination is not None and meta_destination.exists()
    ):
        raise FileExistsError("Kane4Bundle outputs must not already exist")
    staged_npz = unique_staging_path(destination)
    staged_metadata = (
        None if meta_destination is None else unique_staging_path(meta_destination)
    )
    write_npz_artifact(arrays, staged_npz, compressed=True)
    reloaded = load_kane4_bundle(staged_npz)
    if reloaded.fingerprint() != bundle.fingerprint():
        staged_npz.unlink(missing_ok=True)
        raise ValueError("Kane4Bundle typed reload changed the bundle fingerprint")
    with open_stable_binary_file(staged_npz) as handle:
        staged_npz_sha256 = file_object_sha256(handle)
    metadata: dict[str, Any] = {
        "classification": "Kane4Bundle",
        "schema_version": 2,
        "bundle_fingerprint": bundle.fingerprint(),
        "npz_path": str(destination),
        "npz_sha256": staged_npz_sha256,
        "array_manifest": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "nbytes": int(value.nbytes),
            }
            for name, value in arrays.items()
        },
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
    try:
        if meta_destination is not None and staged_metadata is not None:
            write_json_artifact(metadata, staged_metadata)
        publish_staged_file_noreplace(staged_npz, destination)
        if meta_destination is not None and staged_metadata is not None:
            publish_staged_file_noreplace(staged_metadata, meta_destination)
    except Exception:
        if staged_metadata is not None:
            staged_metadata.unlink(missing_ok=True)
        staged_npz.unlink(missing_ok=True)
        # Never delete a public member after publication; a lone NPZ has no
        # typed metadata authority and is safer than rollback across writers.
        raise
    return metadata


def _json_pairs_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate Kane4Bundle provenance key {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"nonstandard Kane4Bundle provenance constant {value!r}")


def load_kane4_bundle(npz_path: str | Path) -> Kane4Bundle:
    """Load and validate a bundle written by :func:`save_kane4_bundle`."""

    with open_stable_binary_file(npz_path) as handle:
        source_sha256 = file_object_sha256(handle)
        with np.load(handle, allow_pickle=False) as data:
            expected_keys = {
                "k_cart_nm_inv",
                "weights_nm2",
                "z_nm",
                "z_weights_nm",
                "h0_mev",
                "micro_wavefunctions",
                "dhdk_mev_nm",
                "time_reversal_unitary",
                "labels",
                "electron_indices",
                "hole_indices",
                "provenance_json",
            }
            if set(data.files) != expected_keys:
                raise ValueError("Kane4Bundle archive keys changed")
            derivative = np.asarray(data["dhdk_mev_nm"], dtype=np.complex128)
            time_reversal = np.asarray(
                data["time_reversal_unitary"],
                dtype=np.complex128,
            )
            basis = E1H1BasisSpec(
                labels=tuple(str(value) for value in data["labels"].tolist()),
                electron_indices=tuple(
                    int(value) for value in data["electron_indices"].tolist()
                ),
                hole_indices=tuple(
                    int(value) for value in data["hole_indices"].tolist()
                ),
            )
            bundle = Kane4Bundle(
                k_cart_nm_inv=np.asarray(data["k_cart_nm_inv"], dtype=float),
                weights_nm2=np.asarray(data["weights_nm2"], dtype=float),
                z_nm=np.asarray(data["z_nm"], dtype=float),
                z_weights_nm=np.asarray(data["z_weights_nm"], dtype=float),
                h0_mev=np.asarray(data["h0_mev"], dtype=np.complex128),
                micro_wavefunctions=np.asarray(
                    data["micro_wavefunctions"],
                    dtype=np.complex128,
                ),
                dhdk_mev_nm=None if derivative.size == 0 else derivative,
                time_reversal_unitary=(
                    None if time_reversal.size == 0 else time_reversal
                ),
                basis=basis,
                provenance=json.loads(
                    str(data["provenance_json"].item()),
                    object_pairs_hook=lambda pairs: _json_pairs_without_duplicates(
                        pairs
                    ),
                    parse_constant=lambda value: _reject_json_constant(value),
                ),
            )
        if file_object_sha256(handle) != source_sha256:
            raise RuntimeError("Kane4Bundle archive changed during load")
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

__all__ = [
    "Kane4Bundle",
    "LiftedKaneFrame",
    "lift_active_frame_to_reference",
    "load_kane4_bundle",
    "remove_normal_e1_h1_hybridization",
    "save_kane4_bundle",
]
