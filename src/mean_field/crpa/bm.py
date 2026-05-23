from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from ..systems.tbg.params import TBGParameters
from ..systems.tbg.zero_field.model import (
    _construct_diagonal_block,
    _generate_gvec,
    _generate_t12,
)


@dataclass(frozen=True)
class AllBandBMSolution:
    """BM eigensystem retaining remote bands for cRPA.

    Energies are stored in meV and eigenvectors use the same plane-wave basis
    ordering as ``mean_field.systems.tbg.zero_field.BMSolution``:
    ``uk[basis, band, valley, k]``.
    """

    params: TBGParameters
    lattice_kvec: np.ndarray
    lg: int
    nlocal: int
    n_eta: int
    nb: int
    spectrum: np.ndarray
    uk: np.ndarray
    gvec: np.ndarray
    band_start: int
    band_stop: int
    sigma_rotation: bool = True
    periodic_g_grid: bool = True
    k_grid_kind: str = "uniform_crpa"

    @property
    def nk(self) -> int:
        return int(self.lattice_kvec.size)

    @property
    def basis_dimension(self) -> int:
        return int(self.uk.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.lg), int(self.lg))


def _params_init_dict(params: TBGParameters) -> dict[str, float | str]:
    return {
        "dtheta_rad": float(params.dtheta_rad),
        "convention": str(params.convention),
        "vf": float(params.vf),
        "chemical_potential": float(params.chemical_potential),
        "w0": float(params.w0),
        "w1": float(params.w1),
        "delta": float(params.delta),
        "strain": float(params.strain),
        "strain_angle_rad": float(params.strain_angle_rad),
        "poisson": float(params.poisson),
        "beta_g": float(params.beta_g),
        "alpha": float(params.alpha),
        "deformation_potential": float(params.deformation_potential),
    }


def write_all_band_bm_solution(
    solution: AllBandBMSolution,
    output_path: Path | str,
    *,
    compressed: bool = False,
) -> Path:
    path = Path(output_path)
    metadata = {
        "params": _params_init_dict(solution.params),
        "lg": int(solution.lg),
        "nlocal": int(solution.nlocal),
        "n_eta": int(solution.n_eta),
        "nb": int(solution.nb),
        "nk": int(solution.nk),
        "band_start": int(solution.band_start),
        "band_stop": int(solution.band_stop),
        "sigma_rotation": bool(solution.sigma_rotation),
        "periodic_g_grid": bool(solution.periodic_g_grid),
        "k_grid_kind": str(solution.k_grid_kind),
    }
    if not path.suffix or path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
        np.save(path / "lattice_kvec.npy", np.asarray(solution.lattice_kvec, dtype=np.complex128))
        np.save(path / "spectrum.npy", np.asarray(solution.spectrum, dtype=float))
        np.save(path / "uk.npy", np.asarray(solution.uk, dtype=np.complex128))
        np.save(path / "gvec.npy", np.asarray(solution.gvec, dtype=np.complex128))
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        lattice_kvec=np.asarray(solution.lattice_kvec, dtype=np.complex128),
        spectrum=np.asarray(solution.spectrum, dtype=float),
        uk=np.asarray(solution.uk, dtype=np.complex128),
        gvec=np.asarray(solution.gvec, dtype=np.complex128),
    )
    return path


def read_all_band_bm_solution(path: Path | str) -> AllBandBMSolution:
    path = Path(path)
    if path.is_dir():
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        params = TBGParameters(**metadata["params"])
        lattice_kvec = np.load(path / "lattice_kvec.npy")
        spectrum = np.load(path / "spectrum.npy")
        uk = np.load(path / "uk.npy")
        gvec = np.load(path / "gvec.npy")
        return AllBandBMSolution(
            params=params,
            lattice_kvec=np.asarray(lattice_kvec, dtype=np.complex128),
            lg=int(metadata["lg"]),
            nlocal=int(metadata["nlocal"]),
            n_eta=int(metadata["n_eta"]),
            nb=int(metadata["nb"]),
            spectrum=np.asarray(spectrum, dtype=float),
            uk=np.asarray(uk, dtype=np.complex128),
            gvec=np.asarray(gvec, dtype=np.complex128),
            band_start=int(metadata["band_start"]),
            band_stop=int(metadata["band_stop"]),
            sigma_rotation=bool(metadata.get("sigma_rotation", True)),
            periodic_g_grid=bool(metadata.get("periodic_g_grid", False)),
            k_grid_kind=str(metadata.get("k_grid_kind", "uniform_crpa")),
        )

    with np.load(path) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        params = TBGParameters(**metadata["params"])
        lattice_kvec = np.asarray(data["lattice_kvec"], dtype=np.complex128)
        spectrum = np.asarray(data["spectrum"], dtype=float)
        uk = np.asarray(data["uk"], dtype=np.complex128)
        gvec = np.asarray(data["gvec"], dtype=np.complex128)
    return AllBandBMSolution(
        params=params,
        lattice_kvec=lattice_kvec,
        lg=int(metadata["lg"]),
        nlocal=int(metadata["nlocal"]),
        n_eta=int(metadata["n_eta"]),
        nb=int(metadata["nb"]),
        spectrum=spectrum,
        uk=uk,
        gvec=gvec,
        band_start=int(metadata["band_start"]),
        band_stop=int(metadata["band_stop"]),
        sigma_rotation=bool(metadata.get("sigma_rotation", True)),
        periodic_g_grid=bool(metadata.get("periodic_g_grid", False)),
        k_grid_kind=str(metadata.get("k_grid_kind", "uniform_crpa")),
    )


def _band_subset(dim: int, bands_per_valley: int | None) -> tuple[int, int] | None:
    if bands_per_valley is None or int(bands_per_valley) >= dim:
        return None
    bands = int(bands_per_valley)
    if bands <= 0:
        raise ValueError(f"bands_per_valley must be positive, got {bands_per_valley}")
    center = dim // 2
    start = max(0, center - bands // 2)
    stop = start + bands - 1
    if stop >= dim:
        stop = dim - 1
        start = stop - bands + 1
    return (start, stop)


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


def solve_all_band_bm_model(
    params: TBGParameters,
    lattice_kvec: np.ndarray,
    *,
    lg: int = 9,
    bands_per_valley: int | None = None,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = True,
    check_finite: bool = False,
) -> AllBandBMSolution:
    """Solve the BM model for all or a centered window of bands.

    The existing HF solver intentionally keeps only the two flat bands. cRPA
    needs remote bands, so this routine reuses the same BM Hamiltonian builder
    but stores a larger band window and leaves the raw ``eigh`` gauge intact.
    """

    lg = int(lg)
    if lg <= 0 or lg % 2 == 0:
        raise ValueError(f"lg must be a positive odd integer, got {lg}")
    n_eta, nlocal = 2, 4
    dim = nlocal * lg * lg
    subset = _band_subset(dim, bands_per_valley)
    if subset is None:
        band_start, band_stop = 0, dim - 1
        nb = dim
    else:
        band_start, band_stop = subset
        nb = band_stop - band_start + 1

    kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    nk = int(kvec.size)
    gvec = _generate_gvec(params, lg)
    spectrum = np.zeros((nb, n_eta, nk), dtype=float)
    uk = np.zeros((dim, nb, n_eta, nk), dtype=np.complex128)
    tunnel_builder = _generate_t12 if periodic_g_grid else _generate_t12_zero_fill
    tunnel = {1: tunnel_builder(params, lg, 1), -1: tunnel_builder(params, lg, -1)}

    for ieta, zeta in enumerate((1, -1)):
        valley_tunnel = tunnel[zeta]
        for ik, kval in enumerate(kvec):
            h0 = _construct_diagonal_block(params, gvec, lg, complex(kval), zeta, sigma_rotation)
            h = h0 + valley_tunnel - params.chemical_potential * np.eye(dim, dtype=np.complex128)
            if subset is None:
                evals, evecs = eigh(h, driver="evr", check_finite=check_finite)
            else:
                evals, evecs = eigh(
                    h,
                    subset_by_index=[band_start, band_stop],
                    driver="evr",
                    check_finite=check_finite,
                )
            spectrum[:, ieta, ik] = np.asarray(evals, dtype=float)
            uk[:, :, ieta, ik] = np.asarray(evecs, dtype=np.complex128)

    return AllBandBMSolution(
        params=params,
        lattice_kvec=kvec,
        lg=lg,
        nlocal=nlocal,
        n_eta=n_eta,
        nb=nb,
        spectrum=spectrum,
        uk=uk,
        gvec=gvec,
        band_start=band_start,
        band_stop=band_stop,
        sigma_rotation=bool(sigma_rotation),
        periodic_g_grid=bool(periodic_g_grid),
        k_grid_kind="uniform_crpa",
    )
