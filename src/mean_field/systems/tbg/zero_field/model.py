from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.linalg import eigh

from ....core.lattice import LatticeGrid
from ..params import TBGParameters


@dataclass(frozen=True)
class BMSolution:
    params: TBGParameters
    lattice_kvec: np.ndarray
    lg: int
    nlocal: int
    n_eta: int
    n_spin: int
    nb: int
    hamiltonian: np.ndarray
    sigma_z: np.ndarray
    uk: np.ndarray
    spectrum: np.ndarray
    gvec: np.ndarray

    @property
    def nk(self) -> int:
        return int(self.lattice_kvec.size)

    @property
    def nt(self) -> int:
        return self.n_eta * self.n_spin * self.nb

    def flattened_energies(self) -> np.ndarray:
        data = np.zeros((self.nt, self.nk), dtype=float)
        row = 0
        for ib in range(self.nb):
            for ieta in range(self.n_eta):
                for ispin in range(self.n_spin):
                    data[row, :] = self.spectrum[ib, ieta, :]
                    row += 1
        return data

    def with_reference_uk(self, uk: np.ndarray) -> "BMSolution":
        uk = np.asarray(uk, dtype=np.complex128)
        if uk.shape != self.uk.shape:
            raise ValueError(f"Expected uk shape {self.uk.shape}, got {uk.shape}")
        sigma_z = build_sigma_z_from_uk(uk, lg=self.lg, n_spin=self.n_spin)
        return replace(self, uk=uk.copy(), sigma_z=sigma_z)


def dirac(k: complex, zeta: int, theta0: float = 0.0) -> np.ndarray:
    phase = np.exp(-1j * zeta * (np.angle(k) - theta0))
    scale = zeta * abs(k)
    return scale * np.asarray([[0.0, phase], [np.conj(phase), 0.0]], dtype=np.complex128)


def build_b0_uniform_lattice(params: TBGParameters, lk: int) -> LatticeGrid:
    frac = np.arange(0, lk + 1, dtype=float) / float(lk)
    kvec = np.ravel(frac[:, None] * params.g1 + frac[None, :] * params.g2, order="F")
    return LatticeGrid(
        k1=frac.copy(),
        k2=frac.copy(),
        kvec=np.asarray(kvec, dtype=np.complex128),
        nk=int(kvec.size),
        lk=int(lk),
        flag_inv=True,
    )


def _generate_gvec(params: TBGParameters, lg: int) -> np.ndarray:
    coords = np.arange(-(lg // 2), lg // 2 + 1, dtype=int)
    return np.ravel(coords[:, None] * params.g1 + coords[None, :] * params.g2, order="F").astype(np.complex128)


def _generate_t12(params: TBGParameters, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(lg * lg).reshape(lg, lg, order="F")
    idx_nn1 = np.roll(idx, shift=(-zeta, zeta), axis=(0, 1))
    idx_nn2 = np.roll(idx, shift=(0, zeta), axis=(0, 1))
    idx_nn12 = np.roll(idx, shift=(-zeta, 0), axis=(0, 1))
    idx_nn1_flat = np.ravel(idx_nn1, order="F")
    idx_nn2_flat = np.ravel(idx_nn2, order="F")
    idx_nn12_flat = np.ravel(idx_nn12, order="F")

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    for ig in range(lg * lg):
        left = 4 * ig
        right1 = 4 * int(idx_nn1_flat[ig])
        right2 = 4 * int(idx_nn2_flat[ig])
        right0 = 4 * int(idx_nn12_flat[ig])

        t12[left + 2 : left + 4, right1 : right1 + 2] = t2
        t12[right1 : right1 + 2, left + 2 : left + 4] = t2
        t12[left + 2 : left + 4, right2 : right2 + 2] = t1
        t12[right2 : right2 + 2, left + 2 : left + 4] = t1
        t12[left + 2 : left + 4, right0 : right0 + 2] = t0
        t12[right0 : right0 + 2, left + 2 : left + 4] = t0

    return t12


def _construct_diagonal_block(params: TBGParameters, gvec: np.ndarray, lg: int, k: complex, zeta: int, sigma_rotation: bool) -> np.ndarray:
    dim = 4 * lg * lg
    h = np.zeros((dim, dim), dtype=np.complex128)
    sigma0 = np.eye(2, dtype=np.complex128)
    rotation = -params.dtheta_rad / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    div_u = float((params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0)

    for ig in range(lg * lg):
        qc = gvec[ig]
        if zeta == 1:
            kb = k - params.kb_point + qc
            kt = k - params.kt + qc
        elif zeta == -1:
            kb = k - params.kt + qc
            kt = k - params.kb_point + qc
        else:
            raise ValueError(f"Unexpected valley label: {zeta}")
        if sigma_rotation:
            k1 = (np.eye(2) + rotation - params.strain_matrix * params.alpha) @ np.asarray([kb.real, kb.imag], dtype=float)
            k2 = (np.eye(2) - rotation + params.strain_matrix * (1.0 - params.alpha)) @ np.asarray([kt.real, kt.imag], dtype=float)
        else:
            k1 = np.asarray([kb.real, kb.imag], dtype=float)
            k2 = np.asarray([kt.real, kt.imag], dtype=float)

        left = 4 * ig
        h[left : left + 2, left : left + 2] = params.vf * dirac(complex(k1[0], k1[1]), zeta, 0.0) - (params.deformation_potential * div_u) * sigma0
        h[left + 2 : left + 4, left + 2 : left + 4] = params.vf * dirac(complex(k2[0], k2[1]), zeta, 0.0) + (params.deformation_potential * div_u) * sigma0

    return h


def _c2t_operator(lg: int) -> np.ndarray:
    s0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    s1 = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    ig = np.eye(lg * lg, dtype=np.complex128)
    return np.kron(ig, np.kron(s0, s1))


def _sigma_z_operator(lg: int) -> np.ndarray:
    s0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ig = np.eye(lg * lg, dtype=np.complex128)
    return np.kron(ig, np.kron(s0, sz))


def build_sigma_z_from_uk(uk: np.ndarray, *, lg: int, n_spin: int = 2) -> np.ndarray:
    uk = np.asarray(uk, dtype=np.complex128)
    if uk.ndim != 4:
        raise ValueError(f"Expected uk to have rank 4, got shape {uk.shape}")
    dim, nb, n_eta, nk = uk.shape
    sigma_z = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)
    sigma_z_local = _sigma_z_operator(lg)
    if sigma_z_local.shape != (dim, dim):
        raise ValueError(f"Expected local sigma_z shape {(dim, dim)}, got {sigma_z_local.shape}")

    for ik in range(nk):
        for ieta, zeta in enumerate((1, -1)):
            mat = uk[:, :, ieta, ik].conj().T @ sigma_z_local @ uk[:, :, ieta, ik] * zeta
            for ispin in range(n_spin):
                base = 2 * ieta + ispin
                rows = slice(base, n_spin * n_eta * nb, n_spin * n_eta)
                cols = slice(base, n_spin * n_eta * nb, n_spin * n_eta)
                sigma_z[rows, cols, ik] = mat
    return sigma_z


def solve_bm_model(
    params: TBGParameters,
    lattice_kvec: np.ndarray,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    calculate_chern_operator: bool = True,
) -> BMSolution:
    n_eta, n_spin, nb, nlocal = 2, 2, 2, 4
    nk = int(lattice_kvec.size)
    dim = nlocal * lg * lg
    gvec = _generate_gvec(params, lg)

    hamiltonian = np.zeros((dim, dim, n_eta, nk), dtype=np.complex128)
    spectrum = np.zeros((nb, n_eta, nk), dtype=float)
    uk = np.zeros((dim, nb, n_eta, nk), dtype=np.complex128)
    sigma_z = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)

    c2t = _c2t_operator(lg)
    sigma_z_local = _sigma_z_operator(lg)
    tunnel = {1: _generate_t12(params, lg, 1), -1: _generate_t12(params, lg, -1)}

    start = dim // 2 - 1
    stop = start + nb - 1

    for ieta, zeta in enumerate((1, -1)):
        valley_tunnel = tunnel[zeta]
        for ik, kval in enumerate(lattice_kvec):
            h0 = _construct_diagonal_block(params, gvec, lg, complex(kval), zeta, sigma_rotation)
            h = h0 + valley_tunnel - params.chemical_potential * np.eye(dim, dtype=np.complex128)
            hamiltonian[:, :, ieta, ik] = h
            evals, evecs = eigh(h, subset_by_index=[start, stop], driver="evr")
            evecs = evecs + c2t @ np.conj(evecs)
            norms = np.linalg.norm(evecs, axis=0)
            evecs = evecs / norms[None, :]
            spectrum[:, ieta, ik] = evals
            uk[:, :, ieta, ik] = evecs

    if calculate_chern_operator:
        sigma_z[:, :, :] = build_sigma_z_from_uk(uk, lg=lg, n_spin=n_spin)

    return BMSolution(
        params=params,
        lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
        lg=lg,
        nlocal=nlocal,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        hamiltonian=hamiltonian,
        sigma_z=sigma_z,
        uk=uk,
        spectrum=spectrum,
        gvec=gvec,
    )
