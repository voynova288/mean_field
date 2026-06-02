from __future__ import annotations

"""Fukui-Hatsugai-Suzuki Chern calculation with moire-BZ boundary sewing.

The plane-wave basis used by the continuum model is not literally periodic at
k -> k + G_Mi.  At the torus boundary one must compare the eigenvector at the
neighboring k point after shifting the plane-wave index G -> G + G_Mi.  This
module implements that sewing operation explicitly.

The returned Chern number is in the "raw" fractional-coordinate orientation
by default, i.e. the orientation of (G_M1, G_M2).  Use orientation="physical"
if you want the sign multiplied by sign(G_M1 x G_M2).
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.linalg import eigh

from analysis.topology import WavefunctionIndex, compute_lattice_topology

from .hamiltonian import build_hamiltonian
from .lattice import _complex_key
from .model import TMBGModel

Orientation = Literal["raw", "physical"]
BandSelector = Literal["valence", "conduction", "pair"]


@dataclass(frozen=True)
class SewnGridResult:
    model: TMBGModel
    mesh: int
    valley: int
    global_i0: int
    global_i1: int
    iv: int
    ic: int
    k_grid_frac: np.ndarray  # shape: (mesh, mesh, 2)
    evals: np.ndarray  # shape: (mesh, mesh, n_sub)
    evecs: np.ndarray  # shape: (mesh, mesh, dim, n_sub)

    @property
    def lattice(self):
        return self.model.lattice

    @property
    def valence_local_index(self) -> int:
        return int(self.iv - self.global_i0)

    @property
    def conduction_local_index(self) -> int:
        return int(self.ic - self.global_i0)


def translation_srcmap(lattice, reciprocal_shift: complex) -> np.ndarray:
    """Map target G index to source G index for G -> G + reciprocal_shift."""
    mapping = {_complex_key(complex(g)): idx for idx, g in enumerate(lattice.g_vectors)}
    src = np.full(lattice.n_g, -1, dtype=int)
    for target_index, g in enumerate(lattice.g_vectors):
        source_index = mapping.get(_complex_key(complex(g + reciprocal_shift)))
        if source_index is not None:
            src[target_index] = source_index
    return src


def transform_vec_by_g_shift(vec: np.ndarray, srcmap: np.ndarray) -> np.ndarray:
    """Apply G-index sewing to one eigenvector or a selected subspace."""
    vec = np.asarray(vec)
    out = np.zeros_like(vec)
    if vec.ndim == 1:
        for target_g, source_g in enumerate(srcmap):
            if source_g >= 0:
                out[6 * target_g : 6 * (target_g + 1)] = vec[6 * source_g : 6 * (source_g + 1)]
    elif vec.ndim == 2:
        for target_g, source_g in enumerate(srcmap):
            if source_g >= 0:
                out[6 * target_g : 6 * (target_g + 1), :] = vec[6 * source_g : 6 * (source_g + 1), :]
    else:
        raise ValueError(f"Expected vector/subspace with ndim 1 or 2, got shape {vec.shape}")
    return out


def _unit_complex(z: complex, *, atol: float = 1.0e-14) -> complex:
    z = complex(z)
    magnitude = abs(z)
    if magnitude <= atol:
        raise ValueError("near-zero overlap link; the selected band/subspace is not isolated on this mesh")
    return z / magnitude


def _link(left: np.ndarray, right: np.ndarray, *, atol: float = 1.0e-14) -> tuple[complex, float]:
    overlap = left.conjugate().T @ right
    if overlap.shape == (1, 1):
        z = complex(overlap[0, 0])
        return _unit_complex(z, atol=atol), abs(z)
    z = complex(np.linalg.det(overlap))
    return _unit_complex(z, atol=atol), abs(z)


def compute_sewn_grid(
    model: TMBGModel,
    *,
    mesh: int,
    valley: int = 1,
    subset_pad: int = 1,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> SewnGridResult:
    """Diagonalize a small band window on a uniform moire-BZ torus grid."""
    mesh = int(mesh)
    if mesh < 3:
        raise ValueError("mesh must be at least 3")
    shift_1 = float(frac_shift[0])
    shift_2 = float(frac_shift[1])
    dim = model.lattice.matrix_dim
    iv = dim // 2 - 1
    ic = dim // 2
    i0 = max(0, iv - int(subset_pad))
    i1 = min(dim - 1, ic + int(subset_pad))
    nsub = i1 - i0 + 1
    k_grid_frac = np.empty((mesh, mesh, 2), dtype=float)
    evals = np.empty((mesh, mesh, nsub), dtype=float)
    evecs = np.empty((mesh, mesh, dim, nsub), dtype=np.complex128)

    for i in range(mesh):
        f1 = float(np.mod(i / mesh + shift_1, 1.0))
        for j in range(mesh):
            f2 = float(np.mod(j / mesh + shift_2, 1.0))
            k_grid_frac[i, j, :] = (f1, f2)
            k = f1 * model.lattice.g_m1 + f2 * model.lattice.g_m2
            hmat = build_hamiltonian(k, model.lattice, model.params, valley=valley)
            w, v = eigh(
                hmat,
                subset_by_index=[i0, i1],
                driver="evr",
                check_finite=False,
            )
            evals[i, j, :] = w
            evecs[i, j, :, :] = v

    return SewnGridResult(
        model=model,
        mesh=mesh,
        valley=int(valley),
        global_i0=int(i0),
        global_i1=int(i1),
        iv=int(iv),
        ic=int(ic),
        k_grid_frac=k_grid_frac,
        evals=evals,
        evecs=evecs,
    )


def select_subspace(grid: SewnGridResult, selector: BandSelector | Iterable[int]) -> np.ndarray:
    if selector == "valence":
        indices = [grid.valence_local_index]
    elif selector == "conduction":
        indices = [grid.conduction_local_index]
    elif selector == "pair":
        indices = [grid.valence_local_index, grid.conduction_local_index]
    else:
        indices = [int(i) - grid.global_i0 for i in selector]
    if min(indices) < 0 or max(indices) >= grid.evecs.shape[-1]:
        raise ValueError(
            f"selected local indices {indices} outside stored band window "
            f"[{grid.global_i0}, {grid.global_i1}]"
        )
    return grid.evecs[:, :, :, indices]


def fhs_chern_sewn(
    grid: SewnGridResult,
    selector: BandSelector | Iterable[int],
    *,
    orientation: Orientation = "raw",
    return_berry: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Compute an FHS Chern number with reciprocal-lattice boundary sewing."""
    if orientation not in {"raw", "physical"}:
        raise ValueError("orientation must be 'raw' or 'physical'")

    selected = select_subspace(grid, selector)
    lat = grid.lattice
    src_g1 = translation_srcmap(lat, lat.g_m1)
    src_g2 = translation_srcmap(lat, lat.g_m2)
    selector_label = str(selector) if isinstance(selector, str) else ",".join(str(index) for index in selector)

    geometry = compute_lattice_topology(
        selected,
        index=WavefunctionIndex(
            indices=tuple(range(selected.shape[-1])),
            role="sewn_band_selector",
            labels=(selector_label,),
            system="tmbg",
            valley=int(grid.valley),
            metadata={"global_band_window": (int(grid.global_i0), int(grid.global_i1))},
        ),
        k_grid_frac=grid.k_grid_frac,
        sewing_transforms=(
            lambda vectors: transform_vec_by_g_shift(vectors, src_g1),
            lambda vectors: transform_vec_by_g_shift(vectors, src_g2),
        ),
        link_method="determinant",
    )

    chern = float(geometry.chern_number)
    if orientation == "physical":
        cross = lat.g_m1.real * lat.g_m2.imag - lat.g_m1.imag * lat.g_m2.real
        chern *= float(np.sign(cross))

    if return_berry:
        return chern, float(geometry.min_link_magnitude), geometry.berry_curvature
    return chern, float(geometry.min_link_magnitude)


def central_band_cherns(
    model: TMBGModel,
    *,
    mesh: int,
    valley: int = 1,
    orientation: Orientation = "raw",
    subset_pad: int = 1,
) -> dict[str, float | int]:
    grid = compute_sewn_grid(model, mesh=mesh, valley=valley, subset_pad=subset_pad)
    cv, mlv = fhs_chern_sewn(grid, "valence", orientation=orientation)
    cc, mlc = fhs_chern_sewn(grid, "conduction", orientation=orientation)
    cp, mlp = fhs_chern_sewn(grid, "pair", orientation=orientation)
    locv = grid.valence_local_index
    locc = grid.conduction_local_index
    return {
        "theta_deg": float(model.theta_deg),
        "n_shells": int(model.n_shells),
        "mesh": int(mesh),
        "valley": int(valley),
        "orientation": orientation,
        "C_valence": float(cv),
        "C_conduction": float(cc),
        "C_pair": float(cp),
        "C_valence_round": int(np.rint(cv)),
        "C_conduction_round": int(np.rint(cc)),
        "C_pair_round": int(np.rint(cp)),
        "min_link_valence": float(mlv),
        "min_link_conduction": float(mlc),
        "min_link_pair": float(mlp),
        "central_gap_min_meV": float(np.min(grid.evals[:, :, locc] - grid.evals[:, :, locv]) * 1000.0),
    }
