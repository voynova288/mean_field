from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LatticeGrid:
    k1: np.ndarray
    k2: np.ndarray
    kvec: np.ndarray
    nk: int
    lk: int
    flag_inv: bool = True


@dataclass(frozen=True)
class KPathNode:
    label: str
    index: int
    k_dist: float
    kvec: complex

    @property
    def kx(self) -> float:
        return float(self.kvec.real)

    @property
    def ky(self) -> float:
        return float(self.kvec.imag)


@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    kdist: np.ndarray
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

    @property
    def nodes(self) -> tuple[KPathNode, ...]:
        return tuple(
            KPathNode(
                label=label,
                index=index,
                k_dist=float(self.kdist[index - 1]),
                kvec=complex(self.kvec[index - 1]),
            )
            for label, index in zip(self.labels, self.node_indices, strict=True)
        )


def build_uniform_lattice(g1: complex, g2: complex, lk: int) -> LatticeGrid:
    frac = np.arange(lk + 1, dtype=float) / float(lk)
    kvec = np.ravel(frac[:, None] * g1 + frac[None, :] * g2, order="F")
    return LatticeGrid(
        k1=frac.copy(),
        k2=frac.copy(),
        kvec=np.asarray(kvec, dtype=np.complex128),
        nk=int(kvec.size),
        lk=int(lk),
        flag_inv=True,
    )


def cumulative_distance(kvec: Iterable[complex]) -> np.ndarray:
    values = np.asarray(list(kvec), dtype=np.complex128)
    if values.size == 0:
        return np.asarray([], dtype=float)
    diffs = np.abs(np.diff(values))
    return np.concatenate([np.asarray([0.0]), np.cumsum(diffs)])


def almost_equal_complex(a: complex, b: complex, *, atol: float = 1e-12) -> bool:
    return math.isclose(a.real, b.real, abs_tol=atol) and math.isclose(a.imag, b.imag, abs_tol=atol)
