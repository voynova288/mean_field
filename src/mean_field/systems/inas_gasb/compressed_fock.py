"""Memory-controlled compressed projected Fock actions on explicit 2D meshes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import matmul_toeplitz

from .kane4_bundle import Kane4Bundle
from .projected_fock import fock_energy_density_mev_nm2

ComplexArray = np.ndarray
FloatArray = np.ndarray


def precompute_uniform_toeplitz_exchange_tensor_mev_nm2(
    bundle: Kane4Bundle,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
    workers: int = 1,
    output_npy: str | Path | None = None,
) -> ComplexArray:
    """Stream the uniform-dielectric exchange tensor without dense ``G``.

    This is exact for the open-z kernel used by
    ``uniform_dielectric_green_on_mesh_mev_nm2`` on a uniform z grid.  The
    exponential kernel is Toeplitz, so each z contraction uses an FFT
    Toeplitz product.  The function never materializes full
    ``(nk,nk,nz,nz)`` Green or ``(nk,nk,nz,n,n)`` vertex arrays.
    """

    bundle.validate()
    if not np.isfinite(epsilon_r) or epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be finite and positive")
    if not np.isfinite(coulomb_mev_nm) or coulomb_mev_nm <= 0.0:
        raise ValueError("coulomb_mev_nm must be finite and positive")
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    phi = np.asarray(bundle.micro_wavefunctions, dtype=np.complex128)
    k_cart = np.asarray(bundle.k_cart_nm_inv, dtype=np.float64)
    wk = np.asarray(bundle.weights_nm2, dtype=np.float64)
    z = np.asarray(bundle.z_nm, dtype=np.float64)
    wz = np.asarray(bundle.z_weights_nm, dtype=np.float64)
    dz = np.diff(z)
    if not np.allclose(dz, dz[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Toeplitz exchange requires a uniform z grid")
    if not np.allclose(wz, wz[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Toeplitz exchange requires uniform z quadrature weights")
    nk, nz, _nmicro, n = phi.shape
    shape = (nk, nk, n, n, n, n)
    if output_npy is None:
        tensor: ComplexArray = np.empty(shape, dtype=np.complex128)
    else:
        tensor = np.lib.format.open_memmap(
            Path(output_npy), mode="w+", dtype=np.complex128, shape=shape
        )
    distance = np.arange(nz, dtype=np.float64) * float(dz[0])
    prefactor = 2.0 * np.pi * float(coulomb_mev_nm) / float(epsilon_r)

    def kernel_column(ik: int, ip: int) -> FloatArray:
        if ik == ip:
            cell_area = (2.0 * np.pi) ** 2 * wk[ik]
            q_cell = np.sqrt(cell_area / np.pi)
            column = np.empty(nz, dtype=np.float64)
            column[0] = prefactor * 2.0 / q_cell
            nonzero_distance = distance[1:]
            column[1:] = prefactor * (
                2.0
                * (-np.expm1(-q_cell * nonzero_distance))
                / (q_cell**2 * nonzero_distance)
            )
            return column
        q = float(np.linalg.norm(k_cart[ik] - k_cart[ip]))
        if q <= 0.0:
            raise ValueError("distinct momentum points have zero separation")
        return prefactor / q * np.exp(-q * distance)

    def compute_row(ik: int) -> tuple[int, ComplexArray]:
        row = np.empty((nk, n, n, n, n), dtype=np.complex128)
        phi_k = phi[ik]
        for ip in range(nk):
            local_kp = np.einsum(
                "zma,zmb->zab", phi_k.conj(), phi[ip], optimize=True
            )
            local_pk = local_kp.conj().swapaxes(1, 2)
            source = wz[:, None] * local_pk.reshape(nz, n * n)
            column = kernel_column(ik, ip)
            convolved = matmul_toeplitz(
                (column, column), source, check_finite=False
            ).reshape(nz, n, n)
            raw_abcd = np.einsum(
                "zab,zcd,z->abcd",
                local_kp,
                convolved,
                wz,
                optimize=True,
            )
            row[ip] = raw_abcd.transpose(0, 3, 1, 2)
        return ik, row

    if int(workers) == 1:
        iterator = map(compute_row, range(nk))
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=int(workers))
        iterator = executor.map(compute_row, range(nk))
    try:
        for ik, row in iterator:
            tensor[ik] = row
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if isinstance(tensor, np.memmap):
        tensor.flush()
    if not np.all(np.isfinite(tensor)):
        raise ValueError("streamed exchange tensor contains non-finite values")
    return tensor


def dense_uniform_pair_exchange_tensor_mev_nm2(
    bundle: Kane4Bundle,
    ik: int,
    ip: int,
    *,
    epsilon_r: float,
    coulomb_mev_nm: float = 1439.96448,
) -> ComplexArray:
    """Independent dense-z oracle for one uniform-dielectric momentum pair."""

    bundle.validate()
    nk = bundle.nk
    if not (0 <= int(ik) < nk and 0 <= int(ip) < nk):
        raise IndexError("momentum-pair index is outside the bundle")
    phi = np.asarray(bundle.micro_wavefunctions, dtype=np.complex128)
    z = np.asarray(bundle.z_nm, dtype=np.float64)
    wz = np.asarray(bundle.z_weights_nm, dtype=np.float64)
    wk = np.asarray(bundle.weights_nm2, dtype=np.float64)
    k = np.asarray(bundle.k_cart_nm_inv, dtype=np.float64)
    local_kp = np.einsum(
        "zma,zmb->zab", phi[int(ik)].conj(), phi[int(ip)], optimize=True
    )
    local_pk = local_kp.conj().swapaxes(1, 2)
    distance = np.abs(z[:, None] - z[None, :])
    prefactor = 2.0 * np.pi * float(coulomb_mev_nm) / float(epsilon_r)
    if int(ik) == int(ip):
        q_cell = np.sqrt((2.0 * np.pi) ** 2 * wk[int(ik)] / np.pi)
        green = np.empty_like(distance)
        zero = distance < 1.0e-14
        green[zero] = prefactor * 2.0 / q_cell
        green[~zero] = prefactor * (
            2.0
            * (-np.expm1(-q_cell * distance[~zero]))
            / (q_cell**2 * distance[~zero])
        )
    else:
        q = float(np.linalg.norm(k[int(ik)] - k[int(ip)]))
        green = prefactor / q * np.exp(-q * distance)
    return np.einsum(
        "xab,ycd,xy,x,y->adbc",
        local_kp,
        local_pk,
        green,
        wz,
        wz,
        optimize=True,
    )


@dataclass(frozen=True)
class CompressedProjectedFockOperator:
    """Tensor-only projected Fock action bound to one exact Kane4 bundle."""

    exchange_tensor_mev_nm2: ComplexArray
    k_weights_nm2: FloatArray
    bundle_fingerprint: str
    self_cell_description: str
    electrostatics_fingerprint: str | None = None

    def __post_init__(self) -> None:
        tensor = np.asarray(self.exchange_tensor_mev_nm2, dtype=np.complex128)
        weights = np.asarray(self.k_weights_nm2, dtype=np.float64)
        if tensor.ndim != 6 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("exchange tensor must have shape (nk,nk,n,n,n,n)")
        nk, _nk, n0, n1, n2, n3 = tensor.shape
        if len({n0, n1, n2, n3}) != 1 or weights.shape != (nk,):
            raise ValueError("compressed Fock tensor/weights have incompatible shapes")
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("compressed Fock weights must be finite and positive")
        if not np.all(np.isfinite(tensor)):
            raise ValueError("compressed Fock tensor must be finite")
        if not self.bundle_fingerprint or not self.self_cell_description:
            raise ValueError("bundle fingerprint and self-cell description are required")
        tensor.setflags(write=False)
        weights = np.array(weights, copy=True)
        weights.setflags(write=False)
        object.__setattr__(self, "exchange_tensor_mev_nm2", tensor)
        object.__setattr__(self, "k_weights_nm2", weights)

    @classmethod
    def from_bundle_and_tensor(
        cls,
        bundle: Kane4Bundle,
        exchange_tensor_mev_nm2: ComplexArray,
        *,
        self_cell_description: str,
        electrostatics_fingerprint: str | None = None,
    ) -> "CompressedProjectedFockOperator":
        bundle.validate()
        return cls(
            exchange_tensor_mev_nm2=exchange_tensor_mev_nm2,
            k_weights_nm2=bundle.weights_nm2,
            bundle_fingerprint=bundle.fingerprint(),
            self_cell_description=self_cell_description,
            electrostatics_fingerprint=electrostatics_fingerprint,
        )

    def validate_against_bundle(self, bundle: Kane4Bundle, *, atol: float = 1e-12) -> None:
        bundle.validate()
        if self.bundle_fingerprint != bundle.fingerprint():
            raise ValueError("compressed Fock operator and bundle fingerprints differ")
        if not np.allclose(self.k_weights_nm2, bundle.weights_nm2, rtol=0.0, atol=atol):
            raise ValueError("compressed Fock weights do not match the bundle")
        n = bundle.basis.dimension
        if self.exchange_tensor_mev_nm2.shape != (bundle.nk, bundle.nk, n, n, n, n):
            raise ValueError("compressed Fock tensor does not match the bundle shape")

    def __call__(self, density_delta: ComplexArray) -> ComplexArray:
        density = np.asarray(density_delta, dtype=np.complex128)
        n = self.exchange_tensor_mev_nm2.shape[2]
        nk = self.exchange_tensor_mev_nm2.shape[0]
        if density.shape != (n, n, nk) or not np.all(np.isfinite(density)):
            raise ValueError("density_delta has an incompatible or non-finite shape")
        sigma = -np.einsum(
            "kpadbc,bcp,p->adk",
            self.exchange_tensor_mev_nm2,
            density,
            self.k_weights_nm2,
            optimize=True,
        )
        error = float(np.max(np.abs(sigma - np.swapaxes(sigma.conj(), 0, 1))))
        if not np.all(np.isfinite(sigma)) or error > 1.0e-8:
            raise ValueError(
                f"compressed Fock action is invalid: Hermiticity error={error:.3e}"
            )
        return sigma

    def energy_density_mev_nm2(self, density_delta: ComplexArray) -> float:
        return fock_energy_density_mev_nm2(
            density_delta, self(density_delta), self.k_weights_nm2
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.bundle_fingerprint.encode())
        digest.update(self.self_cell_description.encode())
        digest.update((self.electrostatics_fingerprint or "unspecified").encode())
        for array in (self.exchange_tensor_mev_nm2, self.k_weights_nm2):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def action_storage_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.fingerprint().encode())
        digest.update(b"compressed-only-fock-action")
        return digest.hexdigest()
