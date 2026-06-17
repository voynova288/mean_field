from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.linalg import eigh

from ....core.bands import GridBandsResult, PathBandsResult
from ....core.hf import ComponentGroup
from ....core.lattice import KPath, LatticeGrid
from ..params import TBGParameters
from .path import build_b0_benchmark_kpath, build_fig6_kpath, build_gamma_m_k_gamma_kprime_kpath


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
    periodic_g_grid: bool = True

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


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _resolve_bm_valley_index(valley: int) -> int:
    value = int(valley)
    if value == 1:
        return 0
    if value == -1:
        return 1
    raise ValueError(f"TBG zero-field BM valley must be +1 or -1, got {valley}")


def _resolve_bm_band_count(n_bands: int | None) -> int:
    if n_bands is None:
        return 2
    count = int(n_bands)
    if count != 2:
        raise NotImplementedError("TBG zero-field BM public adapter currently exposes only the central two bands")
    return count


@dataclass(frozen=True)
class TBGZeroFieldBMModel:
    """Narrow public adapter for zero-field BM single-particle bands."""

    params: TBGParameters
    theta_deg: float
    lg: int = 9
    sigma_rotation: bool = True
    periodic_g_grid: bool = True

    @classmethod
    def from_config(
        cls,
        theta_deg: float,
        *,
        lg: int = 9,
        params: TBGParameters | None = None,
        sigma_rotation: bool = True,
        periodic_g_grid: bool = True,
    ) -> "TBGZeroFieldBMModel":
        resolved_params = params if params is not None else TBGParameters.from_degrees(theta_deg)
        return cls(
            params=resolved_params,
            theta_deg=float(theta_deg),
            lg=int(lg),
            sigma_rotation=bool(sigma_rotation),
            periodic_g_grid=bool(periodic_g_grid),
        )

    @property
    def matrix_dim(self) -> int:
        return int(4 * self.lg * self.lg)

    def lattice_summary(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "lg": int(self.lg),
            "g1_nm_inv": _complex_pair(self.params.g1),
            "g2_nm_inv": _complex_pair(self.params.g2),
            "a1_nm": _complex_pair(self.params.a1),
            "a2_nm": _complex_pair(self.params.a2),
            "kt_nm_inv": _complex_pair(self.params.kt),
            "kb_nm_inv": _complex_pair(self.params.kb_point),
            "model_name": "zero_field_bm",
            "sigma_rotation": bool(self.sigma_rotation),
            "periodic_g_grid": bool(self.periodic_g_grid),
        }

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        return (
            ComponentGroup("layer_bottom", np.asarray([0, 1], dtype=int)),
            ComponentGroup("layer_top", np.asarray([2, 3], dtype=int)),
        )

    def standard_kpath(self, *, points_per_segment: int = 120, path_kind: str = "fig6") -> KPath:
        kind = str(path_kind).strip().lower().replace("-", "_")
        if kind in {"fig6", "m_k_gamma_m"}:
            return build_fig6_kpath(self.params, int(points_per_segment))
        if kind in {"b0_benchmark", "benchmark"}:
            return build_b0_benchmark_kpath(self.params, int(points_per_segment))
        if kind in {"gamma_m_k_gamma_kprime", "gamma_m_k_gamma_kp"}:
            return build_gamma_m_k_gamma_kprime_kpath(self.params, int(points_per_segment))
        raise ValueError(f"Unsupported TBG zero-field BM path_kind={path_kind!r}")

    def _solve(self, kvec: np.ndarray) -> BMSolution:
        return solve_bm_model(
            self.params,
            np.asarray(kvec, dtype=np.complex128).reshape(-1),
            lg=int(self.lg),
            sigma_rotation=bool(self.sigma_rotation),
            calculate_chern_operator=False,
            periodic_g_grid=bool(self.periodic_g_grid),
        )

    def build_hamiltonian(self, k_tilde: complex, *, valley: int = 1) -> np.ndarray:
        solution = self._solve(np.asarray([complex(k_tilde)], dtype=np.complex128))
        return np.asarray(solution.hamiltonian[:, :, _resolve_bm_valley_index(valley), 0], dtype=np.complex128)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _resolve_bm_band_count(n_bands)
        solution = self._solve(np.asarray([complex(k_tilde)], dtype=np.complex128))
        valley_index = _resolve_bm_valley_index(valley)
        return (
            np.asarray(solution.spectrum[:, valley_index, 0], dtype=float),
            np.asarray(solution.uk[:, :, valley_index, 0], dtype=np.complex128),
        )

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        _resolve_bm_band_count(n_bands)
        solution = self._solve(np.asarray(path.kvec, dtype=np.complex128))
        valley_index = _resolve_bm_valley_index(valley)
        energies = np.asarray(solution.spectrum[:, valley_index, :], dtype=float).T
        eigenvectors = None
        if return_eigenvectors:
            eigenvectors = np.transpose(np.asarray(solution.uk[:, :, valley_index, :], dtype=np.complex128), (2, 0, 1))
        return PathBandsResult(
            path=path,
            energies=energies,
            eigenvectors=eigenvectors,
            band_indices=(self.matrix_dim // 2 - 1, self.matrix_dim // 2),
            metadata={"system": "tbg", "model": "zero_field_bm", "valley": int(valley), "lg": int(self.lg)},
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        _resolve_bm_band_count(n_bands)
        mesh = int(mesh_size)
        if mesh <= 0:
            raise ValueError(f"mesh_size must be positive, got {mesh_size}")
        if endpoint:
            frac_1d = np.linspace(0.0, 1.0, mesh, dtype=float)
        else:
            frac_1d = (np.arange(mesh, dtype=float) + np.asarray(frac_shift, dtype=float)[0]) / float(mesh)
        frac_y = frac_1d if endpoint else (np.arange(mesh, dtype=float) + np.asarray(frac_shift, dtype=float)[1]) / float(mesh)
        f1, f2 = np.meshgrid(frac_1d, frac_y, indexing="ij")
        kvec = f1 * self.params.g1 + f2 * self.params.g2
        solution = self._solve(np.asarray(kvec, dtype=np.complex128).reshape(-1))
        valley_index = _resolve_bm_valley_index(valley)
        energies = np.asarray(solution.spectrum[:, valley_index, :], dtype=float).T.reshape(mesh, mesh, 2)
        eigenvectors = None
        if return_eigenvectors:
            eigenvectors = np.transpose(np.asarray(solution.uk[:, :, valley_index, :], dtype=np.complex128), (2, 0, 1)).reshape(
                mesh,
                mesh,
                self.matrix_dim,
                2,
            )
        return GridBandsResult(
            k_grid_frac=np.stack([f1, f2], axis=-1),
            kvec=np.asarray(kvec, dtype=np.complex128),
            energies=energies,
            eigenvectors=eigenvectors,
            band_indices=(self.matrix_dim // 2 - 1, self.matrix_dim // 2),
            metadata={"system": "tbg", "model": "zero_field_bm", "valley": int(valley), "lg": int(self.lg)},
        )


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


def _generate_t12_zero_fill(params: TBGParameters, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    def flat(ix: int, iy: int) -> int:
        return int(ix) + int(lg) * int(iy)

    def in_bounds(ix: int, iy: int) -> bool:
        return 0 <= int(ix) < int(lg) and 0 <= int(iy) < int(lg)

    for iy in range(lg):
        for ix in range(lg):
            here = flat(ix, iy)
            left = 4 * here
            neighbors = (
                (ix + zeta, iy - zeta, t2),
                (ix, iy - zeta, t1),
                (ix + zeta, iy, t0),
            )
            for nx, ny, tunnel in neighbors:
                if not in_bounds(nx, ny):
                    continue
                right = 4 * flat(nx, ny)
                t12[left + 2 : left + 4, right : right + 2] = tunnel
                t12[right : right + 2, left + 2 : left + 4] = tunnel

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
    periodic_g_grid: bool = True,
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
    tunnel_builder = _generate_t12 if periodic_g_grid else _generate_t12_zero_fill
    tunnel = {1: tunnel_builder(params, lg, 1), -1: tunnel_builder(params, lg, -1)}

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
        periodic_g_grid=bool(periodic_g_grid),
    )
